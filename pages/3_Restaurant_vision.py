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

import haversine
from haversine import haversine

st.set_page_config (page_title = 'Restaurant vision',
                    page_icon = '🍽️',
                    layout = 'wide')
#Funções
#=================================

def avg_order_city(df1):

    cols = ['Time_taken(min)', 'City', 'Type_of_order']
    df_aux = (df1.loc[:, cols].
                  groupby(['City', 'Type_of_order']).
                  agg({'Time_taken(min)': ['mean', 'std']}))

    df_aux.columns = ['Média', 'Desvio padrão']
    df_aux01 = df_aux.reset_index()

    fig = px.sunburst(df_aux01,
    path=['City', 'Type_of_order'],
    values='Média',
    color='Desvio padrão',
    color_continuous_scale='RdBu',
    color_continuous_midpoint=np.average(df_aux01['Desvio padrão']))

    return fig, df_aux01

#=================================
        
def avg_time_on_traffic(df1):

    cols = ['Time_taken(min)','City','Road_traffic_density']
    df_aux = (df1.loc[:,cols].
                 groupby(['City','Road_traffic_density']).
                 agg({'Time_taken(min)':['mean','std']}))

    df_aux.columns = ['Média','Desvio padrão']
    df_aux01 = df_aux.reset_index()

    fig = px.sunburst(df_aux01,
    path=['City', 'Road_traffic_density'],
    values='Média',
    color='Desvio padrão',
    color_continuous_scale=['lightgray', 'blue'],
    color_continuous_midpoint=np.average(df_aux01['Desvio padrão']))

    return fig
        
#=================================

def avg_std_time_graph(df1):

# Gráfico de colunas com boxplot
    cols = ['Time_taken(min)','City']
    df_aux = (df1.loc[:,cols].groupby('City')
                     .agg({'Time_taken(min)':['mean','std']}))


    df_aux.columns = ['Média','Desvio padrão']
    df_aux01 = df_aux.reset_index()


    fig = go.Figure()
    fig.add_trace( go.Bar(name='Control',
                     x=df_aux01['City'],
                     y = df_aux01['Média'],
                     error_y = dict(type='data', array=df_aux01['Desvio padrão'])))

    fig.update_layout(barmode='group')

    return fig

#=================================

def avg_std_time_delivery(df1, festival, op):  #col 3, col4, col5, e col6
    """
    Esta função calcula o tempo médio e o desvio padrão do tempo de entrega.
    Parâmetros:
        Input:
            - df: Dataframe com os dados necessários para o cálculo.
            - op: Tipo de operação que precisa ser calculado.
                'Avg time': Calcula o tempo médio.
                'Avg std' : Calcula o desvio padrão do tempo.
    """

    cols = ['Time_taken(min)','Festival']
    df_aux = (df1.loc[:,cols]
                 .groupby('Festival')
                 .agg({'Time_taken(min)':['mean','std','sum']}))

    df_aux.columns = ['Avg time','Avg std','Delivery time']
    df_aux = df_aux.reset_index()

    linhas_selecionadas = df_aux['Festival'] == festival
    df_aux = np.round(df_aux.loc[linhas_selecionadas, op],2)

    return df_aux

#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma tabela
    
def distance(df1, fig):  # col2
    if fig == False:

        cols =(['Restaurant_latitude',
                'Restaurant_longitude',
                'Delivery_location_latitude',
                'Delivery_location_longitude'])

        df1['distance'] = df1.loc[:,cols].apply(lambda x:
                  haversine(
                      (x['Restaurant_latitude'],x['Restaurant_longitude']),
                      (x['Delivery_location_latitude'],x['Delivery_location_longitude'])),
                  axis=1)

        avg_distance = np.round(df1['distance'].mean(), 2)

        return  avg_distance
    
    else: 
        
        cols = (['Restaurant_latitude',
                         'Restaurant_longitude',
                         'Delivery_location_latitude',
                         'Delivery_location_longitude'])
                
        df1['distance'] = df1.loc[:,cols].apply(lambda x:
                      haversine(
                          (x['Restaurant_latitude'],x['Restaurant_longitude']),
                          (x['Delivery_location_latitude'],x['Delivery_location_longitude'])),
                      axis=1)
        
        avg_distance = df1.loc[:,['City','distance']].groupby('City').mean().reset_index()
        
        #Tornando o destaque automnático
        avg_distance = avg_distance.sort_values(by='distance', ascending=True)
        pull_values = [0] * len(avg_distance)
        pull_values[0] = 0.08

        fig = (go.Figure(
        data=[go.Pie(labels=avg_distance['City'],
        values = avg_distance['distance'], pull= pull_values)]))
        
        return fig

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
df = pd.read_csv('dataset\\train.csv')

#=================================
#Transformação dos dados
#=================================
df1 = clean_code(df)
#df = df.loc[df['Order_Date'] <= date_slider, :]
#df = df.loc[df['Road_traffic_density'].isin(traffic_options, :)] 
#df = df.loc[df['Weatherconditions'].isin(weatherconditions_options, :)]
#=================================

#Cabeçalho central
st.header('Marketplace - Restaurant ', divider='red')
#=================================

#=================================
#Barra Lateral
#=================================

#Imagem (CARREGANDO IMAGEM)

image = Image.open('curr_company.png')
st.sidebar.image( image, width = 250)

st.sidebar.write('<span style="font-size: medium;">Leonardo Rosa</span> <span style="font-size: small;">:blue[ | Data Scientist]</span>', unsafe_allow_html=True)

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
st.sidebar.markdown('### Powered by Comunidade DS')

#=================================

#Filtros

linhas_selecionadas = df1['Order_Date'] <= date_slider
df1 = df1.loc[linhas_selecionadas,:]
linhas_selecionadas01 = df1['City'].isin(bycity_options)
df1 = df1.loc[linhas_selecionadas01,:]
linhas_selecionadas02 = df1['Road_traffic_density'].isin(traffic_options)
df1 = df1.loc[linhas_selecionadas01,:]
linhas_selecionadas03 = df1['Weatherconditions'].isin(weatherconditions_options)
df1 = df1.loc[linhas_selecionadas02,:]

#st.dataframe(df1)
st.sidebar.markdown("""___""")

#====================================================================
#VISÃO RESTAURANTES
#====================================================================

tab1, tab2, tab3 = st.tabs (['Management Vision', ' ', ' '])

with tab1:
    with st.container(): # linha do gráfico
        st.markdown("""___""")
        st.title('Overall Metrics')

        
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        
        with col1:
            
            entregadores_unicos = len(df1.loc[:,'Delivery_person_ID'].unique())
            col1.metric('Person ID',entregadores_unicos)
            
        with col2:
            
            avg_distance = distance(df1, fig = False)
            col2.metric('Avg Distance ', avg_distance)
            
        with col3:
            
            df_aux = avg_std_time_delivery(df1,'Yes','Avg time')
            col3.metric('Avg Time', df_aux)
            
            
        with col4:
            
            df_aux = avg_std_time_delivery(df1, 'Yes','Avg std')
            col4.metric('Avg Std', df_aux)
            
            
        with col5:
            
            df_aux = avg_std_time_delivery(df1, 'No','Avg time')
            col5.metric('Avg Time n/Festival', df_aux)

       
        with col6:
            
            df_aux = avg_std_time_delivery(df1, 'No','Avg std')
            col6.metric('Avg Std n/Festival', df_aux)
                     
            
    with st.container():
        
        st.markdown("""___""")
        st.title('Average delivery time by city')
        fig = avg_std_time_graph(df1)
        st.plotly_chart(fig)

#================================================
        
    with st.container():
        
        st.markdown("""___""")
        st.title('Time distribution') ## Pizza 
        fig = distance(df1, fig = True)
        st.plotly_chart(fig)
        
      
    with st.container():
        
        st.markdown("""___""")
        st.title('Average orders per city')

        fig, df_aux01 = avg_order_city(df1) # passando duas saídas para o streamlit
        st.plotly_chart(fig)
        st.table(df_aux01)
        
        
    with st.container():
        
        st.markdown("""___""")
        st.title('Distance distribution')
        fig = avg_time_on_traffic(df1)
        st.plotly_chart(fig)

