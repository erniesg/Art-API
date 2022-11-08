import streamlit as st
import base64

# Custom imports
from multipage import MultiPage
from Pages import Color, Home, Object, Predict # import your pages here

# Create an instance of the app
st.set_page_config(layout="wide")

app = MultiPage()

def image_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
image_local('creambackground.png')


# Title of the main page
#st.title("Data Storyteller Application")
Title= '<p style="font-family:Avenir;color:rgb(180, 23, 30); font-size: 55px;">Art API. </p>'
st.markdown(Title, unsafe_allow_html=True)



# Add all your applications (pages) here
app.add_page("Home", Home.app)
#app.add_page("Predict", Predict.app)
app.add_page("Object", Object.app)
app.add_page("Color", Color.app)

#app.add_page("Color_Try",color_try.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
app.run()
#app_run = st.sidebar.button('Run')
