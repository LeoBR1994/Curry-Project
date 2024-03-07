import streamlit as st
from PIL import Image

st.set_page_config(
       page_title = 'Home',
       page_icon='🎲')




image = Image.open('curr_company.png')
st.sidebar.image( image, width = 250)

st.sidebar.write('<span style="font-size: medium;">Leonardo Rosa</span> <span style="font-size: small;">:blue[ | Data Scientist]</span>', unsafe_allow_html=True)


st.sidebar.markdown("""___""")
st.sidebar.markdown('# Curry\'s Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""___""")
st.sidebar.markdown('### Powered by Comunidade DS')

st.write('# Curry Company Growth Dashboard' )
st.markdown(
    
    """
        Growth Dashboard was built to track the growth metrics of Delivery Drivers and Restaurants.
        
        - Company Vision:

            - Management Vision: General behavior metrics.
            - Tactical View: Weekly growth indicators.
            - Geographic Vision: Geolocation insights.
    """)

st.markdown(
    """
        - Delivery Vision:

            - Monitoring weekly growth indicators
    """)

st.markdown(
    """
        - Vision Restaurants:

            - Weekly restaurant growth indicators
    """)

st.markdown(


