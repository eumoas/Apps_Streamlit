import streamlit as st
import pandas as pd
import requests
import datetime

@st.cache_data(ttl="1day")
def get_selic():
    """Busca o histórico da taxa Selic na API do Banco Central."""
    url = "https://www.bcb.gov.br/api/servico/sitebcb/historicotaxasjuros"
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # Lança um erro para respostas com código de status ruim (4xx ou 5xx)
        df = pd.DataFrame(resp.json()["conteudo"])
        
        # Converte as colunas de data, especificando o formato dd/mm/aaaa
        df["DataInicioVigencia"] = pd.to_datetime(df["DataInicioVigencia"], dayfirst=True).dt.date
        df["DataFimVigencia"] = pd.to_datetime(df["DataFimVigencia"], dayfirst=True, errors='coerce').dt.date
        
        # Preenche datas de fim de vigência nulas com a data de hoje
        df["DataFimVigencia"] = df["DataFimVigencia"].fillna(datetime.date.today())
        
        # CORREÇÃO: Converte a coluna para string ANTES de usar o acessor .str
        df['MetaSelic'] = df['MetaSelic'].astype(str).str.replace(',', '.').astype(float)
        
        # --- INÍCIO DA MUDANÇA ---
        # REMOÇÃO: A API não fornece a coluna 'TaxaSelic', então a linha que a processava foi removida.
        # --- FIM DA MUDANÇA ---
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar dados da Selic: {e}")
        return pd.DataFrame() # Retorna um DataFrame vazio em caso de erro


def calc_general_stats(df: pd.DataFrame):
    """Calcula estatísticas gerais sobre a evolução do patrimônio."""
    df_data = df.groupby(by="Data")[["Valor"]].sum()
    df_data["lag_1"] = df_data["Valor"].shift(1)
    df_data["Diferença Mensal Abs."] = df_data["Valor"] - df_data["lag_1"]
    df_data["Média 6M Diferença Mensal Abs."] = df_data["Diferença Mensal Abs."].rolling(6).mean()
    df_data["Média 12M Diferença Mensal Abs."] = df_data["Diferença Mensal Abs."].rolling(12).mean()
    df_data["Média 24M Diferença Mensal Abs."] = df_data["Diferença Mensal Abs."].rolling(24).mean()
    df_data["Diferença Mensal Rel."] = df_data["Valor"] / df_data["lag_1"] - 1
    df_data["Evolução 6M Total"] = df_data["Valor"].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0])
    df_data["Evolução 12M Total"] = df_data["Valor"].rolling(12).apply(lambda x: x.iloc[-1] - x.iloc[0])
    df_data["Evolução 24M Total"] = df_data["Valor"].rolling(24).apply(lambda x: x.iloc[-1] - x.iloc[0])
    df_data["Evolução 6M Relativa"] = df_data["Valor"].rolling(6).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    df_data["Evolução 12M Relativa"] = df_data["Valor"].rolling(12).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    df_data["Evolução 24M Relativa"] = df_data["Valor"].rolling(24).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)

    df_data = df_data.drop("lag_1", axis=1)

    return df_data


def main_metas(df_stats):
    """Cria a interface para configuração e cálculo de metas financeiras."""
    col1, col2 = st.columns(2)

    data_inicio_meta = col1.date_input("Início da Meta", max_value=df_stats.index.max())
    
    # Garante que a data selecionada existe no índice
    if data_inicio_meta not in df_stats.index:
        # Pega a data mais próxima anterior à data selecionada
        datas_anteriores = df_stats.index[df_stats.index <= data_inicio_meta]
        if not datas_anteriores.empty:
            data_filtrada = datas_anteriores[-1]
        else:
            st.warning("Não há dados históricos antes da data de início da meta selecionada.")
            st.stop()
    else:
        data_filtrada = data_inicio_meta
        
    custos_fixos = col1.number_input("Custos Fixos", min_value=0.0, format="%.2f")
    salario_bruto = col2.number_input("Salário Bruto", min_value=0.0, format="%.2f")
    salario_liq = col2.number_input("Salário Líquido", min_value=0.0, format="%.2f")

    valor_inicio = df_stats.loc[data_filtrada]["Valor"]
    col1.markdown(f"**Patrimônio no Início da Meta**: R$ {valor_inicio:,.2f}")

    selic_gov = get_selic()
    # Verifica se o dataframe da Selic não está vazio
    if not selic_gov.empty:
        filter_selic_date = (selic_gov["DataInicioVigencia"] <= data_inicio_meta) & (selic_gov["DataFimVigencia"] >= data_inicio_meta)
        selic_filtrada = selic_gov[filter_selic_date]
        selic_default = selic_filtrada["MetaSelic"].iloc[0] if not selic_filtrada.empty else 10.0
    else:
        selic_default = 10.0 # Valor padrão caso a API falhe

    selic = st.number_input("Taxa Selic Anual (%)", min_value=0.0, value=selic_default, format="%.2f")
    selic_ano = selic / 100
    selic_mes = (1 + selic_ano) ** (1/12) - 1

    rendimento_ano = valor_inicio * selic_ano
    rendimento_mes = valor_inicio * selic_mes

    col1_pot, col2_pot = st.columns(2)
    mensal = salario_liq - custos_fixos + rendimento_mes
    anual = 12 * (salario_liq - custos_fixos) + rendimento_ano

    with col1_pot.container(border=True):
        st.markdown(f"**Potencial Arrecadação Mês**: R$ {mensal:,.2f}")
        st.caption(f"Salário Líq.: {salario_liq:,.2f} - Custos: {custos_fixos:,.2f} + Rendimento: {rendimento_mes:,.2f}")
    
    with col2_pot.container(border=True):
        st.markdown(f"**Potencial Arrecadação Ano**: R$ {anual:,.2f}")
        st.caption(f"12 * ({salario_liq:,.2f} - {custos_fixos:,.2f}) + Rendimento Anual: {rendimento_ano:,.2f}")

    with st.container(border=True):
        col1_meta, col2_meta = st.columns(2)
        with col1_meta:
            meta_estipulada = st.number_input("Meta de Acúmulo Anual", min_value=0.0, format="%.2f", value=anual)

        with col2_meta:
            patrimonio_final = meta_estipulada + valor_inicio
            st.markdown(f"**Patrimônio Estimado (Final)**: R$ {patrimonio_final:,.2f}")

    return data_inicio_meta, valor_inicio, meta_estipulada, patrimonio_final


# --- Configuração da Página ---
st.set_page_config(page_title="Finanças Pessoais", page_icon="💰", layout="wide")

st.title("💰 App de Análise Financeira")
st.markdown("Bem-vindo! Faça o upload do seu extrato financeiro para começar a análise.")

# Widget de upload de dados
file_upload = st.file_uploader(label="Faça upload do seu arquivo CSV", type=['csv'])

if file_upload:
    
    df = pd.read_csv(file_upload)
    
    if 'Valor' in df.columns:
        df['Valor'] = df['Valor'].astype(str)
        df['Valor'] = df['Valor'].str.replace('R$', '', regex=False).str.strip()
        df['Valor'] = df['Valor'].str.replace('.', '', regex=False)
        df['Valor'] = df['Valor'].str.replace(',', '.', regex=False)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
        df.dropna(subset=['Valor'], inplace=True)
    else:
        st.error("O arquivo CSV precisa ter uma coluna chamada 'Valor'.")
        st.stop()
    
    if 'Data' in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors='coerce').dt.date
        df.dropna(subset=['Data'], inplace=True)
    else:
        st.error("O arquivo CSV precisa ter uma coluna chamada 'Data'.")
        st.stop()
    
    st.success("Arquivo carregado e processado com sucesso!")

    # Exibição dos dados no App
    with st.expander("Visualizar Dados Brutos"):
        columns_fmt = {"Valor": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f")}
        st.dataframe(df, hide_index=True, column_config=columns_fmt)

    # Visão por Instituição
    with st.expander("Análise por Instituições"):
        # Garante que a coluna Instituição existe e preenche valores nulos para evitar erros
        if 'Instituição' not in df.columns:
            df['Instituição'] = 'Não especificada'
        df['Instituição'] = df['Instituição'].fillna('Não especificada')

        df_instituicao = df.pivot_table(index="Data", columns="Instituição", values="Valor", aggfunc="sum")

        tab_data, tab_history, tab_share = st.tabs(["Dados Agrupados", "Histórico de Patrimônio", "Distribuição por Data"])

        with tab_data:
            st.dataframe(df_instituicao.style.format("R$ {:,.2f}", na_rep="-"))
        
        with tab_history:
            st.line_chart(df_instituicao)

        with tab_share:
            date = st.selectbox("Selecione uma data para ver a distribuição:", options=sorted(df_instituicao.index, reverse=True), format_func=lambda d: d.strftime('%d/%m/%Y'))
            st.bar_chart(df_instituicao.loc[date])

    # Expander de estatísticas gerais
    with st.expander("Estatísticas Gerais de Evolução"):
        df_stats = calc_general_stats(df)
        
        columns_config = {
            "Valor": st.column_config.NumberColumn("Valor (R$)", format='R$ %.2f'),
            "Diferença Mensal Abs.": st.column_config.NumberColumn("Diferença Mensal (R$)", format='R$ %.2f'),
            "Média 6M Diferença Mensal Abs.": st.column_config.NumberColumn("Média 6M (R$)", format='R$ %.2f'),
            "Média 12M Diferença Mensal Abs.": st.column_config.NumberColumn("Média 12M (R$)", format='R$ %.2f'),
            "Média 24M Diferença Mensal Abs.": st.column_config.NumberColumn("Média 24M (R$)", format='R$ %.2f'),
            "Evolução 6M Total": st.column_config.NumberColumn("Evolução 6M (R$)", format='R$ %.2f'),
            "Evolução 12M Total": st.column_config.NumberColumn("Evolução 12M (R$)", format='R$ %.2f'),
            "Evolução 24M Total": st.column_config.NumberColumn("Evolução 24M (R$)", format='R$ %.2f'),
            "Diferença Mensal Rel.": st.column_config.ProgressColumn("Diferença Mensal (%)", format='%.2f%%'),
            "Evolução 6M Relativa": st.column_config.ProgressColumn("Evolução 6M (%)", format='%.2f%%'),
            "Evolução 12M Relativa": st.column_config.ProgressColumn("Evolução 12M (%)", format='%.2f%%'),
            "Evolução 24M Relativa": st.column_config.ProgressColumn("Evolução 24M (%)", format='%.2f%%'),
        }

        tab_stats, tab_abs, tab_rel = st.tabs(tabs=["Dados Detalhados", "Histórico de Evolução Absoluta (R$)", "Histórico de Crescimento Relativo (%)"])

        with tab_stats:
            st.dataframe(df_stats.sort_index(ascending=False), column_config=columns_config)

        with tab_abs:
            abs_cols = ["Diferença Mensal Abs.", "Média 6M Diferença Mensal Abs.", "Média 12M Diferença Mensal Abs.", "Média 24M Diferença Mensal Abs." ]
            st.line_chart(df_stats[abs_cols])

        with tab_rel:
            rel_cols = ["Diferença Mensal Rel.", "Evolução 6M Relativa", "Evolução 12M Relativa", "Evolução 24M Relativa"]
            st.line_chart(data=df_stats[rel_cols])

    # Expander de Metas
    with st.expander("Planejamento de Metas"):
        tab_main, tab_data_meta, tab_graph = st.tabs(tabs=["Configuração da Meta", "Acompanhamento Mensal", "Gráficos de Atingimento"])

        with tab_main:
            data_inicio_meta, valor_inicio, meta_estipulada, patrimonio_final = main_metas(df_stats)

        with tab_data_meta:
            meses = pd.DataFrame({
                "Data Referência": pd.to_datetime([(data_inicio_meta + pd.DateOffset(months=i)) for i in range(1, 13)]),
                "Meta Mensal": [valor_inicio + round(meta_estipulada / 12, 2) * i for i in range(1, 13)],
            })
            
            meses["Data Referência"] = meses["Data Referência"].dt.strftime("%Y-%m")
            df_patrimonio = df_stats.reset_index()[["Data", "Valor"]]
            df_patrimonio["Data Referência"] = pd.to_datetime(df_patrimonio["Data"]).dt.strftime("%Y-%m")
            
            df_patrimonio_mensal = df_patrimonio.groupby("Data Referência")['Valor'].last().reset_index()

            meses = meses.merge(df_patrimonio_mensal, how='left', on="Data Referência")
            
            meses = meses[['Data Referência', "Meta Mensal", "Valor"]]
            meses["Atingimento (%)"] = (meses["Valor"] / meses["Meta Mensal"]) * 100
            meses["Atingimento Ano"] = (meses["Valor"] / patrimonio_final) * 100
            meses["Atingimento Esperado"] = (meses["Meta Mensal"] / patrimonio_final) * 100
            meses = meses.set_index("Data Referência")

            columns_config_meses = {
                "Meta Mensal": st.column_config.NumberColumn("Meta Mensal (R$)", format='R$ %.2f'),
                "Valor": st.column_config.NumberColumn("Valor Atingido (R$)", format='R$ %.2f'),
                "Atingimento (%)": st.column_config.ProgressColumn("Atingimento da Meta Mensal (%)", format='%.2f%%', min_value=0, max_value=150),
                "Atingimento Ano": st.column_config.ProgressColumn("Atingimento da Meta Anual (%)", format='%.2f%%', min_value=0, max_value=100),
                "Atingimento Esperado": st.column_config.LineChartColumn("Curva Esperada de Atingimento", y_min=0, y_max=100),
            }
            st.dataframe(meses, column_config=columns_config_meses)

        with tab_graph:
            st.line_chart(meses[["Atingimento Ano", "Atingimento Esperado"]])
else:
    st.info("Aguardando o upload de um arquivo CSV para iniciar a análise.")
