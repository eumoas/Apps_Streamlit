import streamlit as st

import pandas as pd 
st.set_page_config(page_title = "Finanças", page_icon= "👜")
st.text("Olá, mundo!")

st.markdown(""" 
# Boas vindas!
            
## Nosso APP de Controle Financeiro!
            
            ## Curta a organização do nosso sistema para realizar seus sonhos!
           
          """)
#Widget de captura de upload de dados 
file_upload = st.file_uploader(label= "Faça upload dos dados aqui", type=['csv'])

# Verifica se há algum arquivo para upload
if file_upload:
    #Leitura dos dados
    df = pd.read_csv(file_upload)
   # Converter a coluna Data para o formato de data
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y").dt.date
    
    # Tratar a coluna Valor: remover possíveis caracteres e converter para numérico
    df["Valor"] = df["Valor"].replace(r'[^\d,]', '', regex=True)  # Remove tudo exceto números e vírgula
    df["Valor"] = df["Valor"].str.replace(',', '.')  # Substitui vírgula por ponto
    df["Valor"] = pd.to_numeric(df["Valor"], errors='coerce')  # Converte para float, transformando erros em NaN
    
    # Tratar valores nulos, se necessário
    df["Valor"] = df["Valor"].fillna(0)  # Preenche NaN com 0 (ajuste conforme necessário)
    
    #Exibição dos dados no App
    exp1 = st.expander("Dados Brutos")
    columns_fmt = {"Valor": st.column_config.NumberColumn("Valor", format="R$ %f")}
    exp1.dataframe(df, hide_index=True, column_config=columns_fmt)

    #Visão Instituição
    exp2 = st.expander("Instituições")
    df_instituicao = df.pivot_table(index="Data", columns="Instituição", values= "Valor")
  
   #Abas para diferentes visualizações 
    tab_data, tab_history, tab_share = exp2.tabs(["Dados", "Histórico", "Distribuição"])
   
   #Exibe dataframe
    with tab_data:
        st.dataframe(df_instituicao)

        #Exibe histórico
    with tab_history:
        st.line_chart(df_instituicao)

#Exibe distribuição
    with tab_share:

        date = st.selectbox("Filtro Data", options=df_instituicao.index)

    # obtem a última data de dados
        last_dt = df_instituicao.sort_index().iloc[-1]
        st.bar_chart(last_dt)
    