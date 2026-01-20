# EASESPOT - Anomaly Detection System

![EASESPOT Logo](logo.png)

Sistema de detecção de anomalias em imagens têxteis usando Machine Learning.

## 🎯 Funcionalidades

- **Detecção de Anomalias**: Identifica defeitos em tecidos automaticamente
- **Múltiplos Modelos**: PatchCore (97% precisão), PaDiM e SPADE
- **Interface Moderna**: Design premium com fundo azul e branding EASESPOT
- **Alta Precisão**: Modelos treinados com datasets industriais

## 🚀 Como Usar

### Executar Localmente

```bash
# Clonar repositório
git clone https://github.com/SEU_USUARIO/easespot-anomaly-detection.git
cd easespot-anomaly-detection

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar app
streamlit run app.py
```

### Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Selecione o branch `main` e arquivo `app.py`
5. Clique em "Deploy"

## 📁 Estrutura do Projeto

```
├── app.py                 # Aplicação Streamlit principal
├── requirements.txt       # Dependências Python
├── logo.png              # Logo EASESPOT
├── fundo.png             # Imagem de fundo
├── padim_model.pkl       # Modelo PaDiM treinado
├── patchcore_model.pkl   # Modelo PatchCore treinado
├── spade_model.pkl       # Modelo SPADE treinado
└── README.md             # Este arquivo
```

## 🤖 Modelos Disponíveis

| Modelo | Precisão | Descrição |
|--------|----------|-----------|
| **PatchCore** | 97% | Memory bank com KNN para detecção |
| **PaDiM** | ~90% | Modelagem de distribuição por patches |
| **SPADE** | ~85% | Pirâmide semântica multi-escala |

## 📊 Como Funciona

1. **Upload**: Faça upload de uma imagem têxtil
2. **Seleção**: Escolha o modelo na sidebar
3. **Análise**: Clique em "Analisar Imagem"
4. **Resultado**: Veja se é Normal ou Anomalia

## 🔧 Tecnologias

- **Python 3.9+**
- **Streamlit** - Interface web
- **PyTorch** - Deep Learning
- **ResNet18** - Feature extraction
- **scikit-learn** - Machine Learning

## 📄 Licença

© 2026 EASESPOT - Todos os direitos reservados.
