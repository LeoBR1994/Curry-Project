#Carregando as bibliotecas 
#=================================

import folium
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html
from streamlit_folium import folium_static
from PIL import Image #manipulação de imagem

st.set_page_config (page_title = 'Delivery vision',
                    page_icon = '🚀',
                    layout = 'wide')
#Funções
#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma tabela
    
            
def avg_rating_delivery(df1):

    df_aux = (df1.loc[:,['Delivery_person_Ratings','Delivery_person_ID']].
              groupby('Delivery_person_ID').
              mean().
              reset_index())

    df_aux = (df_aux.
              rename(columns={'Delivery_person_ID': 'Person ID',
                              'Delivery_person_Ratings': 'Person Ratings'}))

    fig = px.line(df_aux, y='Person Ratings', title= ' ')
    fig.update_layout(title_font=dict(size=20), xaxis_title="Delivery", yaxis_title="Mean")

    return fig
        
    
#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma tabela
    
def avg_rating_traffic(df1):
    df1.rename(columns={'Road_traffic_density': 'Road traffic density'}, inplace=True)
    df_aux = df1.groupby('Road traffic density')['Delivery_person_Ratings'].agg(['mean', 'std'])
    df_aux.reset_index(inplace=True)
    df_aux.rename(columns={'mean': 'Mean', 'std': 'Standard deviation'}, inplace=True)

    avg_distance = df_aux.sort_values(by='Standard deviation', ascending=True)
    pull_values = [0] * len(avg_distance)
    pull_values[0] = 0.08

    df_aux.rename(columns={'Road traffic density': 'Traffic Density'}, inplace=True)

    fig = go.Figure(data=[go.Pie(labels=avg_distance.index,
                                 values=avg_distance['Standard deviation'], pull=pull_values)])
    fig.update_layout(width=800, height=500)
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1.2, xanchor="left", x=0.1))

    return df_aux
        
#=================================
    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma tabela
    
def avg_rating_condition(df1):

    df_aux = (df1.loc[:,['Weatherconditions','Delivery_person_Ratings']]
                 .groupby('Weatherconditions')
                 .agg({'Delivery_person_Ratings': ['mean','std','min','max']}))

    df_aux.columns = ['Mean Rating','Standard deviation Rating','Min Rating','Max Rating']
    df_aux01 = df_aux.reset_index(inplace=True)
    df_aux01 = df_aux.reset_index(drop=True)

    return df_aux01

#=================================
    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma tabela
    
def top_delivers(df1, top_asc):
    df_aux = (df1.loc[:,['Delivery_person_ID','City','Time_taken(min)']]
                 .groupby(['City','Delivery_person_ID'])
                 .min()
                 .sort_values(['City','Time_taken(min)'], ascending= top_asc)
                 .reset_index())
    
    df2 = (df_aux.rename(columns=
                               {'Delivery_person_ID':
                                'Person ID',
                                'Time_taken(min)':
                                'Time taken(min)'}
                              ))

    a = df2[df2['City']=='Metropolitian'].head(10)
    b = df2[df2['City']=='Urban'].head(10)
    c = df2[df2['City']=='Semi-Urban'].head(10)
    df3 = pd.concat([a,b,c], ignore_index=True)

    return df3.head(10)

#=================================

    #1. Extrair o texto após o primeiro espaço de cada linha.
    #2. Retorna a segunda posição

def extrair_texto(texto):
    partes = texto.split(' ')
    if len(partes) > 1:
        return partes[1]
    else:
        return texto

#=================================

    #Tipos de limpeza:
    
    #1. Renovação dos dados NaN
    #2. Mudançã do tipo da coluna de dados
    #3. Renovação dos espaçõs das variáveis de texto
    #4. Formatação da coluna de datas
    #5. Limpeza da coluna de tempo ( remoção do texto da variável numérica)
    
    #Input: DataFrame
    #Output: DataFrame

def clean_code(df):

    #Clean column Time Taken (min)
    df['Time_taken(min)']= df['Time_taken(min)'].apply(lambda x: x.split('(min)')[1]) 
    df['Time_taken(min)']= df['Time_taken(min)'].astype(int)

    # removing NA
    linhas_selecionadas = (df['Delivery_person_Age'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    linhas_selecionadas = (df['Road_traffic_density'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    linhas_selecionadas = (df['City'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    linhas_selecionadas = (df['Festival'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    linhas_selecionadas = (df['multiple_deliveries'] != 'NaN ')
    df = df.loc[linhas_selecionadas,:].copy()
    
    # Converting types
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings']. astype(float)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format = '%d-%m-%Y')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)

    #Removing strings' 
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()
    
    return df
        
#=================================INICIO DA ESTRUTURA LÓGICA=================================#
#Importação
#=================================
df = pd.read_csv('dataset/train.csv')

#=================================
#Transformação dos dados
#=================================
df1 = clean_code(df)
#df = df.loc[df['Order_Date'] <= date_slider, :]
#df = df.loc[df['Road_traffic_density'].isin(traffic_options, :)] 
#df = df.loc[df['Weatherconditions'].isin(weatherconditions_options, :)]
#=================================

#Cabeçalho central
st.header('Marketplace - Delivery Person', divider='red')
#=================================
#Fundo


#=================================
#Barra Lateral
#=================================

#Imagem (CARREGANDO IMAGEM)

image = Image.open('curr_company.png')
st.sidebar.image( image, width = 250)
st.sidebar.write('<span style="font-size: small;">:blue[Comunidade DS]</span>', unsafe_allow_html=True)

#=================================
#Barra de opções
st.sidebar.markdown("""___""")
st.sidebar.markdown('# Curry\'s Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""___""")

#Filtro
st.sidebar.markdown('## Select a deadline ')

#Transformando a data miníma e máxima em {variável} de acordo com o tamanho do arquivo
#=================================

#Datas automáticas

min_date_str = df1['Order_Date'].min()  
min_date = pd.to_datetime(min_date_str) + pd.Timedelta(days=1) 
max_date = df1['Order_Date'].max()

#=================================
date_slider = st.sidebar.slider('Set the date', 
                                value=pd.to_datetime(max_date).to_pydatetime(),
                                min_value=min_date.to_pydatetime(),
                                max_value=pd.to_datetime(max_date).to_pydatetime(),
                                format='DD/MM/YYYY')

st.sidebar.markdown("""___""")

#=================================
#Cidade

bycity = df1['City'].unique()
bycity_options = st.sidebar.multiselect(
                       'Which city do you want to analyze?',
                       bycity,
                       default= bycity)

st.sidebar.markdown("""___""")

#=================================
#Densidade automática

density = df1['Road_traffic_density'].unique()
traffic_options = st.sidebar.multiselect(
                       'What are the traffic conditions?',
                       density,
                       default= density)

st.sidebar.markdown("""___""")

#=================================
# Clima

df1['Weatherconditions'] = df1['Weatherconditions'].apply(extrair_texto) # aplicando a função extrair_texto

weatherconditions = df1['Weatherconditions'].unique()
weatherconditions_options = st.sidebar.multiselect(
                       'What are the weather conditions?',
                       weatherconditions,
                       default= weatherconditions)

st.sidebar.markdown("""___""")
st.sidebar.markdown('### Powered by Leonardo Rosa')

#=================================

#Filtros

linhas_selecionadas = df1['Order_Date'] <= date_slider
df1 = df1.loc[linhas_selecionadas,:]
linhas_selecionadas01 = df1['City'].isin(bycity_options)
df1 = df1.loc[linhas_selecionadas01,:]
linhas_selecionadas02 = df1['Road_traffic_density'].isin(traffic_options)
df1 = df1.loc[linhas_selecionadas02,:]
linhas_selecionadas03 = df1['Weatherconditions'].isin(weatherconditions_options)
df1 = df1.loc[linhas_selecionadas03,:]

#st.dataframe(df1)
st.sidebar.markdown("""___""")

#=================================
#Criar abas (BLOCOS)
#=================================

tab1, tab2, tab3 = st.tabs (['Management Vision', ' ', ' '])

with tab1:
    #4 Cards com os resultados
    with st.container(): # linha do gráfico
        
        st.title('Overall Metrics')
        st.subheader('Delivery info') #, divider = 'gray')
        
        col1,col2,col3,col4 = st.columns (4, gap = 'large')
        
        with col1:
            #st.subheader('Maior idade') # colunas do gráfico
            maior_idade = df1.loc[:, 'Delivery_person_Age'].max()
            col1.metric('Max age', maior_idade)

        with col2:
            #st.subheader('Menor idade')
            menor_idade = df1.loc[:, 'Delivery_person_Age'].min()
            col2.metric('Min age', menor_idade)
            
        with col3:
            #st.subheader('Melhor condição do veiculo')
            melhor_condicao =df1.loc[:, 'Vehicle_condition'].max()
            col3.metric('Vehicle in good condition', melhor_condicao)
            
        with col4:
            #st.subheader('Pior condições do veiculo')
            pior_condicao =df1.loc[:, 'Vehicle_condition'].min()
            col4.metric('Vehicle in poor condition', pior_condicao)
            
    with st.container():
        # 1 coluna com 2 linhas
        st.markdown("""___""")
        st.title('Assessments')
        
        #Colocar dentro de função e subir para melhorar o código!
        
    with st.container():

        st.markdown('### Average for Delivery')
        fig = avg_rating_delivery(df1)
        st.plotly_chart(fig, use_container_width =True)

    with st.container():
        
        st.markdown('### Mean and Standard deviation for traffic density')
        df3 = avg_rating_traffic(df1)
        st.table(df3.reset_index(drop=True))

    with st.container():

        st.markdown("""___""")
        st.markdown('### Average ratings and standard deviation for weather conditions')
        df3 = avg_rating_condition(df1)
        st.table(df3)

    with st.container():
        # 2 colunas
        st.markdown("""___""")
        st.markdown('### Delivery Speed')
        
        col1,col2 = st.columns(2)
        
        with col1:
            st.markdown('### Top fastest couriers')
            df3 = top_delivers(df1, top_asc = True)
            st.table(df3)
                  
        with col2:
            
            st.markdown('### Top slowest couriers')
            df3 = top_delivers(df1, top_asc = False)
            st.table(df3.reset_index(drop=True))
