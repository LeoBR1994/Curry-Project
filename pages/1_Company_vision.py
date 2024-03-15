#Carregando as bibliotecas
#=================================

import folium
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from folium.features import DivIcon
from folium import plugins
from streamlit.components.v1 import html
from streamlit_folium import folium_static
from PIL import Image #manipulação de imagem
from datetime import datetime

st.set_page_config (page_title = 'Company vision',
                    page_icon = '🎯',
                    layout = 'wide')
# Funções
#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura
    
    #Mapa 
    
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 6371 km is the radius of the Earth
    km = 6371 * c
    return km

def country_maps(df1):
    df_aux = (df1.loc[:,['City','Road_traffic_density',
                         'Delivery_location_latitude',
                         'Delivery_location_longitude',
                         'Delivery_person_ID']].
              groupby(['City','Road_traffic_density']).
              agg({'Delivery_location_latitude':'median',
                   'Delivery_location_longitude':'median',
                   'Delivery_person_ID':'count'}).
              reset_index())

    map = folium.Map()

    for index, location_info in df_aux.iterrows():
        description = f"City: {location_info['City']}<br><br>" \
                      f"Road Traffic Density: {location_info['Road_traffic_density']}<br><br>" \
                      f"Delivery Persons: {location_info['Delivery_person_ID']}"
        folium.Marker([location_info['Delivery_location_latitude'],
                       location_info['Delivery_location_longitude']],
                      popup=description).add_to(map)
        folium.Circle(
            radius=1250,  
            location=[location_info['Delivery_location_latitude'], location_info['Delivery_location_longitude']],
            color='lightgray', 
            fill=True,
            fill_color='lightblue',
            fill_opacity=0.4,
        ).add_to(map)

    # Add a plugin for cluster markers
    marker_cluster = plugins.MarkerCluster(maxClusterRadius=50, disableClusteringAtZoom=15).add_to(map)
    
    for index, location_info in df_aux.iterrows():
        description = f"City: {location_info['City']}<br><br>" \
                      f"Road Traffic Density: {location_info['Road_traffic_density']}<br><br>" \
                      f"Delivery Persons: {location_info['Delivery_person_ID']}"
        popup = folium.Popup(description, max_width=300)
        folium.Marker([location_info['Delivery_location_latitude'],
                       location_info['Delivery_location_longitude']],
                      popup=popup).add_to(marker_cluster)

    # Add a plugin for fullscreen
    plugins.Fullscreen().add_to(map)

    folium_static(map, width=1024, height=600)
    return None

#=================================    

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura
    
def Order_share_by_week(df1):
    # Exercicio 5 (visao empresa)
    df1 = df1.rename(columns={'week_of_year': 'week of year', 'order_by_deliver': 'order by deliver'}) 
    df_aux01 = df1.loc[:,['ID','week of year']].groupby('week of year').count().reset_index()
    df_aux02 = (df1.loc[:,['ID',
                           'Delivery_person_ID',
                           'week of year']].
                    groupby('week of year').
                    nunique().
                    reset_index())
    df_aux02 = df_aux02.rename(columns={'Delivery_person_ID': 'Delivery Persons'})
    df_aux = pd.merge(df_aux01, df_aux02, how='inner')
    df_aux['order by deliver'] = df_aux['ID'] / df_aux['Delivery Persons']
    
    fig = px.line(df_aux, x='week of year', y='order by deliver')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False))
    fig.update_traces(line=dict(color = 'lightgreen', shape='spline'))

    return fig
    
#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura

def Order_by_week(df1):

    df1['week_of_year']= df1['Order_Date'].dt.strftime('%U')
    df1 = df1.rename(columns={'week_of_year': 'week of year'})
    df_aux = df1.loc[:,['ID','week of year']].groupby('week of year').count().reset_index()
    
    
    fig = px.line( df_aux, x='week of year', y = 'ID')
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False))
    
    fig.update_traces(line=dict(color='lightgreen', shape='spline'))

    return fig

#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura
    
def traffic_order_city(df1):
    # Exercicio 4 (visao empresa)
    # Gráfico de Bolhas
    df_aux = (df1.loc[:,['ID',
                         'City',
                         'Road_traffic_density']].
                          groupby(['City',
                                   'Road_traffic_density']).
                                    count().
                                    reset_index())
    
    df_aux.rename(columns={'Road_traffic_density': 'Road traffic density'}, inplace=True)
    
    fig = px.scatter(df_aux, 
                     x='Road traffic density', 
                     y='City', 
                     size='ID', 
                     color='City'
                    )

    # Multiplicando o tamanho por um fator para torná-lo maior
    fig.update_traces(marker=dict(size=df_aux['ID']*2.5))

    return fig

#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura
    
def traffic_order_share( df1):
    #Exercicio 3 (visao empresa)
    #Gráfico de Pizza

    df_aux = df1.loc[:,['ID','Road_traffic_density']].groupby('Road_traffic_density').count().reset_index()
    df_aux['entregas_perc']= df_aux['ID']/ df_aux['ID'].sum()

    fig = px.pie( df_aux, values='entregas_perc', names= 'Road_traffic_density')

    return fig

#=================================

    #1. Recebe um Dataframe
    #2. Guarda um Dataframe
    #3. Gera uma figura
    
def order_metric(df1):
    #Orders by Day        
    cols = ['ID','Order_Date']
    df_aux = df1.loc[:,cols].groupby('Order_Date').count().reset_index()

    #Gráfico de linhas no streamlit
    fig =px.bar(df_aux, x='Order_Date', y='ID')

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
st.header('Marketplace - Customer Vision', divider='red')
#=================================
#Fundo


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
st.sidebar.markdown('### Powered by Leonardo DS')

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

#=================================
#Criar abas (BLOCOS)
#=================================

tab1, tab2, tab3 = st.tabs (['Management Vision', 'Tactical Vision', 'Geographic View'])
#================================= 
with tab1:
#=================================
    #Visão gerencial
    with st.container():
     #Order Metric
    
        fig = order_metric(df1)
        st.markdown('## Orders by Day')
        st.subheader('Number of orders per day')
        st.plotly_chart(fig, use_container_width = True)
        
#Fim do primeiro bloco
#=================================

#Visão distribuição 

    with st.container():

        fig = traffic_order_share(df1)
        st.subheader('Distribution of requests by type of traffic')
        st.plotly_chart(fig, use_container_width = True)

    with st.container():
        
        fig = traffic_order_city(df1)
        st.subheader('Volume of orders by city and type of traffic')
        st.plotly_chart(fig, use_container_width = True)
            
#=================================   
with tab2:
#================================= 
    with st.container():
        #Exercicio 2 (visao empresa) # linha do gráfico
        fig = Order_by_week(df1)
        st.markdown('Quantity ordered per week')
        st.plotly_chart(fig, use_container_width = True)
        
    with st.container():
        
        fig= Order_share_by_week(df1)
        st.markdown('Number of orders completed by delivery person per week')
        st.plotly_chart(fig, use_container_width = True)

        
#=================================
with tab3:
#================================= 
    #Exercicio 6 (visao empresa)
    st.markdown('## Country Maps')
    st.subheader('Downtown location of each city by type of traffic')
    
    fig = country_maps(df1)
    
    
#=================================
