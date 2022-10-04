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

    #Hello_Title= '<p style="font-family:Avenir; color:Brown; font-size: 40px;">Hello, Welcome to Art Analyzer!</p>'
    Welcome_Title= '<p style="font-family:Avenir; color:rgb(180, 23, 26); font-size: 20px;">Welcome!</p>'
    st.markdown(Welcome_Title, unsafe_allow_html=True)


    Hello_Title= '<p style="font-family:Avenir; color:rgb(180, 23, 26); font-size: 20px;">Beyond its literal tech-connectivity meaning, encapsulating a hope of facilitating greater accessibility to Art for all.<br><br>Through this project, with the permission of the National Gallery of Singapore (NGS) we aim to enhance the search capabilities of its Collection to include more intuitive searches - by object and colour.<br><br>Enjoy.</p>'
    # Hello_Title= '<p style="font-family:Avenir; color:rgb(180, 23, 30); font-size: 20px;">This project aims to improve art galleries search function and reduce manual labelling by using Machine Learning to predict artwork based on user\'s objects and/or colors search input.</p>'

    st.markdown(Hello_Title, unsafe_allow_html=True)

    # st.write("""
    #  This app predicts artwork based on the objects and colors selected.
    #  """)



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
