import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# PyTorch para extração de features (igual ao notebook)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurações do modelo (igual ao notebook)
IMG_SIZE = 416

# Transformação de imagem (igual ao notebook)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# FEATURE EXTRACTOR (ResNet18 - igual ao notebook)
# ============================================================

class FeatureExtractor(nn.Module):
    """Extrai features de múltiplas camadas do ResNet18."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Modo de avaliação
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = {}
        x = self.layer0(x)
        features['layer0'] = x
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features


# ============================================================
# DETECTORES (igual ao notebook)
# ============================================================

class PatchCoreDetector:
    """PatchCore: Memory bank com KNN."""
    def __init__(self):
        self.memory_bank = None
        self.threshold = None
        self.n_neighbors = 9
        self.layer_ids = ['layer2', 'layer3', 'layer4']
        self.dim_per_layer = [128, 256, 512]
        self.knn = None
    
    def _extract_patch_embeddings(self, images, feature_extractor):
        """Extrai embeddings de patches."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        embeddings_list = []
        for layer_id in self.layer_ids:
            f = features[layer_id]
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            embeddings_list.append(f)
        
        embeddings = torch.cat(embeddings_list, dim=1)
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        return embeddings.cpu().numpy()
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        embeddings = self._extract_patch_embeddings(image_tensor, feature_extractor)
        distances, _ = self.knn.kneighbors(embeddings)
        max_distances = np.max(distances, axis=1)
        score = np.max(max_distances)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.memory_bank = state['memory_bank']
        self.layer_ids = state['layer_ids']
        self.dim_per_layer = state['dim_per_layer']
        self.threshold = state['threshold']
        self.n_neighbors = state['n_neighbors']
        
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.knn.fit(self.memory_bank)


class PaDiMDetector:
    """PaDiM: Patch Distribution Modeling."""
    def __init__(self):
        self.pca = None
        self.mean_embeddings = {}
        self.inv_cov_matrices = {}
        self.patch_sizes = None
        self.threshold = None
        self.layer_ids = ['layer1', 'layer2', 'layer3']
        self.dim_per_layer = [64, 128, 256]
    
    def _extract_patch_embeddings(self, images, feature_extractor):
        """Extrai embeddings de patches."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        embeddings_list = []
        for layer_id in self.layer_ids:
            f = features[layer_id]
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            embeddings_list.append(f)
        
        embeddings = torch.cat(embeddings_list, dim=1)
        _, C, H, W = embeddings.shape
        self.patch_sizes = (H, W)
        
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = embeddings.reshape(-1, C)
        
        return embeddings.cpu().numpy()
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        embeddings = self._extract_patch_embeddings(image_tensor, feature_extractor)
        embeddings_pca = self.pca.transform(embeddings)
        
        H, W = self.patch_sizes
        n_patches = H * W
        n_components = embeddings_pca.shape[1]
        
        embeddings_pca = embeddings_pca.reshape(1, n_patches, n_components)
        
        patch_scores = []
        for pos in range(n_patches):
            emb = embeddings_pca[0, pos]
            diff = emb - self.mean_embeddings[pos]
            score = np.sqrt(np.dot(diff, np.dot(self.inv_cov_matrices[pos], diff)))
            patch_scores.append(score)
        
        score = max(patch_scores)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.pca = state['pca']
        self.mean_embeddings = state['mean_embeddings']
        self.inv_cov_matrices = state['inv_cov_matrices']
        self.patch_sizes = state['patch_sizes']
        self.threshold = state['threshold']
        self.layer_ids = state['layer_ids']
        self.dim_per_layer = state['dim_per_layer']


class SPADEDetector:
    """SPADE: Semantic Pyramid Anomaly Detection."""
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.pyramid_levels = [0, 1, 2, 3]
        self.threshold = None
    
    def _extract_pyramid_features(self, images, feature_extractor):
        """Extrai features em múltiplas escalas."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        pyramid = {}
        for level in self.pyramid_levels:
            f = features[f'layer{level}']
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            pyramid[level] = f
        
        return pyramid
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        pyramid = self._extract_pyramid_features(image_tensor, feature_extractor)
        
        level_scores = []
        for level in self.pyramid_levels:
            f = pyramid[level][0]  # Primeira (única) imagem
            f_patches = f.permute(1, 2, 0).reshape(-1, f.shape[0]).cpu().numpy()
            
            level_mean = self.means[level]
            level_std = self.stds[level]
            
            diff = f_patches - level_mean
            normalized_diff = diff / (level_std + 1e-8)
            patch_scores = np.linalg.norm(normalized_diff, axis=1)
            
            level_scores.append(np.max(patch_scores))
        
        score = np.max(level_scores)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.means = state['means']
        self.stds = state['stds']
        self.pyramid_levels = state['pyramid_levels']
        self.threshold = state['threshold']



# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def preprocess_image(uploaded_file):
    """Pré-processa a imagem para o formato esperado pelos modelos."""
    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="EASESPOT - Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminhos das imagens
LOGO_PATH = Path(__file__).parent / "logo.png"
BG_PATH = Path(__file__).parent / "fundo.png"

# Estilo CSS customizado
def set_custom_style():
    bg_base64 = ""
    if BG_PATH.exists():
        bg_base64 = get_base64_image(BG_PATH)
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(10, 25, 47, 0.7);
            z-index: -1;
        }}
        
        .header-container {{
            background: linear-gradient(135deg, rgba(20, 40, 80, 0.95) 0%, rgba(10, 25, 50, 0.95) 100%);
            padding: 20px 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(100, 150, 220, 0.3);
        }}
        
        .logo-img {{ max-height: 80px; margin-right: 20px; }}
        .header-text {{ color: #ffffff; font-size: 2rem; font-weight: 600; }}
        
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a1929 0%, #132f4c 100%);
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #1e4976 0%, #2d5a8a 100%);
            color: white;
            border: 1px solid rgba(100, 150, 220, 0.4);
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(30, 73, 118, 0.4);
        }}
        
        .result-card {{
            background: linear-gradient(135deg, rgba(30, 50, 90, 0.9) 0%, rgba(20, 40, 70, 0.9) 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 15px 0;
            border: 1px solid rgba(100, 150, 220, 0.3);
        }}
        
        .result-normal {{ border-left: 5px solid #4caf50; }}
        .result-anomaly {{ border-left: 5px solid #f44336; }}
        
        h1, h2, h3 {{ color: #ffffff !important; }}
        
        [data-testid="stMetricValue"] {{ color: #60a5fa !important; }}
    </style>
    """, unsafe_allow_html=True)


set_custom_style()

# Header
if LOGO_PATH.exists():
    logo_base64 = get_base64_image(LOGO_PATH)
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="EASESPOT Logo">
        <span class="header-text">Anomaly Detection System</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <span class="header-text">🔍 EASESPOT - Anomaly Detection System</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CARREGAR MODELOS
# ============================================================

@st.cache_resource
def load_feature_extractor():
    """Carrega o feature extractor (ResNet18)."""
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    return extractor


@st.cache_resource
def load_models():
    """Carrega todos os modelos pré-treinados."""
    models = {}
    model_dir = Path(__file__).parent
    
    # PatchCore
    patchcore_path = model_dir / "patchcore_model.pkl"
    if patchcore_path.exists():
        try:
            detector = PatchCoreDetector()
            detector.load(patchcore_path)
            models['PatchCore (97% Precision)'] = detector
            st.sidebar.success("✅ PatchCore carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro PatchCore: {e}")
    
    # PaDiM
    padim_path = model_dir / "padim_model.pkl"
    if padim_path.exists():
        try:
            detector = PaDiMDetector()
            detector.load(padim_path)
            models['PaDiM'] = detector
            st.sidebar.success("✅ PaDiM carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro PaDiM: {e}")
    
    # SPADE
    spade_path = model_dir / "spade_model.pkl"
    if spade_path.exists():
        try:
            detector = SPADEDetector()
            detector.load(spade_path)
            models['SPADE'] = detector
            st.sidebar.success("✅ SPADE carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro SPADE: {e}")
    
    return models


# Carregar recursos
feature_extractor = load_feature_extractor()
models = load_models()

# ============================================================
# INTERFACE
# ============================================================

with st.sidebar:
    if LOGO_PATH.exists():
        logo_base64 = get_base64_image(LOGO_PATH)
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(100, 150, 220, 0.3); margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width: 180px;">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Selecione o Modelo")
    
    if models:
        model_name = st.selectbox(
            "Modelo para detecção",
            list(models.keys()),
            index=0
        )
    else:
        st.error("⚠️ Nenhum modelo encontrado!")
        model_name = None
    
    st.markdown("---")
    st.markdown("### 📊 Informações")
    st.info(f"🖥️ Dispositivo: {device}")
    st.info(f"📐 Tamanho da imagem: {IMG_SIZE}x{IMG_SIZE}")


# Área principal
st.markdown("""
<div style="background: rgba(30, 50, 90, 0.7); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
    <p style="color: #b0c4de; font-size: 1.1rem; margin: 0;">
        🎯 <strong>Sistema de detecção de anomalias em imagens têxteis.</strong><br>
        Usando modelos treinados com alta precisão para detectar defeitos.
    </p>
</div>
""", unsafe_allow_html=True)

# Upload de imagem
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload de imagem para análise",
        type=["png", "jpg", "jpeg"]
    )
    
    if model_name:
        detect_btn = st.button("🔍 Analisar Imagem", use_container_width=True)
    else:
        detect_btn = False

with col2:
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="📷 Imagem enviada", use_column_width=True)
        uploaded_file.seek(0)

# Detecção
if detect_btn and uploaded_file and model_name:
    with st.spinner("🔄 Analisando imagem..."):
        try:
            # Pré-processar imagem
            img_tensor, original_img = preprocess_image(uploaded_file)
            
            # Selecionar modelo
            detector = models[model_name]
            
            # Fazer predição
            is_anomaly, score = detector.predict_single(img_tensor, feature_extractor)
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultado da Análise")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modelo", model_name.split(' (')[0])
            with col2:
                st.metric("Score", f"{score:.4f}")
            with col3:
                st.metric("Threshold", f"{detector.threshold:.4f}")
            
            # Card de resultado
            if is_anomaly:
                st.markdown("""
                <div class="result-card result-anomaly">
                    <h3 style="color: #f44336; margin-top: 0;">⚠️ ANOMALIA DETECTADA!</h3>
                    <p style="color: #b0c4de;">Esta imagem apresenta características fora do padrão normal.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-normal">
                    <h3 style="color: #4caf50; margin-top: 0;">✅ Imagem Normal</h3>
                    <p style="color: #b0c4de;">Esta imagem está dentro do padrão normal esperado.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detalhes expandíveis
            with st.expander("📋 Detalhes da Análise"):
                st.write(f"**Modelo utilizado:** {model_name}")
                st.write(f"**Score de anomalia:** {score:.6f}")
                st.write(f"**Threshold do modelo:** {detector.threshold:.6f}")
                st.write(f"**Diferença (Score - Threshold):** {score - detector.threshold:.6f}")
                st.write(f"**Classificação:** {'ANOMALIA' if is_anomaly else 'NORMAL'}")
                
        except Exception as e:
            st.error(f"❌ Erro ao analisar imagem: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #6b7c93;">
    <p>EASESPOT © 2026 - Sistema de Detecção de Anomalias Têxteis</p>
    <p style="font-size: 0.9rem;">Powered by PyTorch, ResNet18 & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
