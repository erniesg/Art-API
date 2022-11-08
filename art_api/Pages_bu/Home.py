import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import base64


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


def app():


    #st.title('Art Analyzer')
    #Title= '<p style="font-family:Futura; color:Grey; font-size: 55px;">Welcome to Art Analyzer!</p>'
    #st.markdown(Title, unsafe_allow_html=True)

    Hello_Title= '<p style="font-family:Avenir; color:Black; font-size: 46px;">Hello, Welcome to Art Analyzer!</p>'
    st.markdown(Hello_Title, unsafe_allow_html=True)
    st.write("""
     This app predicts artwork based on the objects and colors selected.
     """)



    # elif choice == "Object":

    #  # Object
    #     st.title("Object Detection: ")
    #     st.write("Select an object below")

    #     options = st.multiselect(
    #         'Select object',
    #         ['Boat','Plane','Human','Cat','Dog', 'Cow', 'Elephant'])



    # elif choice == "Color":

    #  # Object
    #     st.title("Color Detection: ")
    #     st.write("Select a color below")

    #     options = st.multiselect(
    #         'Select colors',
    #         ['Red','Orange','Yellow','Green','Blue', 'Indigo', 'Violet'])


    #     return options


# if __name__ == '__main__':
#     main()
