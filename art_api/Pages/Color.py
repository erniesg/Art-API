# Import necessary libraries
from enum import auto
import json
import joblib
import os
import streamlit as st
from itertools import cycle

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import urllib
from skimage import io
import pdb
#*** added
from PIL import Image
import webcolors
#***
import seaborn as sns
import pandas as pd
import base64
#from Pages import utils
#*** For prototype data extraction from excel - [maybe used for meta-data extraction]
# import openpyxl
# from openpyxl_image_loader import SheetImageLoader
#
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def RGB2HEX(options):
    return "#{:02x}{:02x}{:02x}".format(int(options[0]), int(options[1]), int(options[2]))

# Part 2: Image Search by color
# - Define search parameters - 7 colors of rainbow
# - User will pick search color, as input to be searched against the pre-generated DF of # image colors
# -[Search uses delta_CIE76 color difference to return colors within a specified threshold]


#*** RAINBOW COLOR DICTIONARY REFERENCE
# COLORS = {
#     'VIOLET': [148, 0, 211], #Hex #9400D3
#     'INDIGO': [75, 0, 130], #Hex #4B0082
#     'BLUE': [0, 0, 255],  # Hex #0000FF THRESHOLD [70?]
#     'GREEN': [0, 128, 0], # Hex #008000, THRESHOLD [65]
#     'YELLOW': [255, 255, 0], # Hex #FFFF00
#     'ORANGE': [255, 127, 0], # Hex #FF7F00
#     'RED': [255, 0, 0] # Hex #FF0000
# }

COLORS = {
    'VIOLET': [148, 0, 211], #9400D3, T65 **OK
    'INDIGO': [75, 0, 130], #4B0082, T67 **OK
    'BLUE': [0, 0, 255], #0000FF, T108 **OK
    'GREEN': [0, 128, 0], #008000, T63 **OK
    'YELLOW': [255, 255, 0], #FFFF00, T78 **OK
    'ORANGE': [255, 127, 0], #FF7F00, T63 **OK
    'RED': [255, 0, 0], #FF0000, T78 **OK
}
THOLDS = {
    'VIOLET': [65], #9400D3, T65 **OK
    'INDIGO': [67], #4B0082, T67 **OK
    'BLUE': [126.1], #0000FF, T108 **OK
    'GREEN': [63], #008000, T63 **OK
    'YELLOW': [78], #FFFF00, T78 **OK
    'ORANGE': [63], #FF7F00, T63 **OK
    'RED': [78], #FF0000, T78 **OK
}


#*** WORKING function to convert hex to RGB: webcolors
# then RGB to lab
# >>> hex_to_rgb('#000080')
# (0, 0, 128)
# >>> rgb_to_hex((255, 255, 255))
# '#FFFFFF'
#  pip install webcolors
# import webcolors
def hex2lab(column_hexcolor):
    column_RGBcolor = webcolors.hex_to_rgb(column_hexcolor)
    print(column_RGBcolor)
    # column_LABcolor = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
    column_LABcolor = rgb2lab(np.uint8(np.asarray([[column_RGBcolor]])))
    print(column_LABcolor)
    return column_LABcolor

#*** WORKING *** TEST 3 (NO PRINTS)
#* incorporate column loop direct - reduce loops
# importing Image class from PIL package
#from PIL import Image

def match_image_by_color(color, threshold, rows_to_chk, columns_to_chk): #num_pic_col
    img_colors_df = pd.read_csv('../raw_data/df_10K_copy.csv')
    #img_colors_df = app()
    #* Selected_color_ is user's chosen color for search
    selected_color_name = color
    #st.write(f'selected_color_name = {selected_color_name}')
    selected_color_lab = rgb2lab(np.uint8(np.asarray([[color]]))) #* input color by user
    #print(f'selected_color_lab= {selected_color_lab}\n')
    #iterate over rows in img_color_df:
    filtered = []
    caption = []

    for row in range(rows_to_chk):
        #st.write(f'\nRow_num = {row}')
        row_img_shown = False  #denotes whether row image shown b4. reset to false for each new row.
        for column in range(columns_to_chk):
            print(f'\nRow, Column_num = {row},{column}')
            column_hexcolor = img_colors_df.loc[row][column+1]  #*use column+1 to skip ID column
            #print(f'Column_hexcolor = {column_hexcolor}')
            select_image_id = False
            if str(column_hexcolor).startswith('#'):
                column_LABcolor = hex2lab(column_hexcolor) #calling hex2lab
                #st.write(f'Column_LABcolor = {column_LABcolor}')
                diff = deltaE_cie76(selected_color_lab, column_LABcolor)

                #st.write(f'Delta Diff= {diff}')
                if (diff < threshold):
                    select_image_id = True
                    img_colors_df.loc[row]['id']
                    #st.write(f'\nrow_img_shown = {row_img_shown}')
                    #st.write(f'\nimg_colors_df_row = {img_colors_df.loc[row]}')
                    image_id = img_colors_df.loc[row]['id']

                    
                    print(f'\nimage_id = {image_id}')
                    if row_img_shown == False:
                        #Show the image
                        image_path = f'../raw_data/aws10k/{image_id}.jpg'
                        # creating a object
                        # im = Image.open(image_path)

                        try:

                            im = Image.open(image_path)

                            cap = image_path.split('/')[-1].split('.')[0]

                            caption.append(f'Accession Number: {cap}') #<br><br>Diff: {diff} <br><br>(Artist) <br><br>(Artname) <br><br>(Year)

                            filtered.append(im)


                            row_img_shown = True #once row img shown at least once b4 no need to show again.



                            #st.image(im, caption= f'{image_id}.jpg', use_column_width=True)

                        except:
                            pass

    # your images here
    # your caption here
    cols = cycle(st.columns(3)) # st.columns here since it is out of beta at the time I'm writing this

    for  filteredImage, cap in zip(filtered, caption):
        next(cols).image(filteredImage, width=250, caption=cap, use_column_width=auto)



                        #for i in filtered:
                                #cols = st.columns(2)
                                #cols[0].image(im, caption= f'{image_id}.jpg',use_column_width=True)
                                #cols[1].image(im, caption= f'{image_id}.jpg',use_column_width=True)
                                # cols[2].image(im, caption= f'{image_id}.jpg')







                        #display(im)

                        # row_img_shown = True #once row img shown at least once b4 no need to show again.
                #print(f'select_image = {select_image_id}')
                #return select_image_id

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
    """This application helps in running machine learning models without having to write explicit code
    by the user. It runs some basic models and let's the user select the color variables.
        """

        # Load the data
        # if 'main_data.csv' not in os.listdir('data'):
        #     st.markdown("Please upload data through `Upload Data` page!")

    img_colors_df = pd.read_csv('../raw_data/df_10K_copy.csv')

    #adding a color selectbox

    # def choose_color(options):

    color = st.multiselect(
        'Select colors',
        ['RED','ORANGE','YELLOW','GREEN','BLUE', 'INDIGO', 'VIOLET'])
    #st.sidebar.button('Run')

    st.write(f"Color to be predicted:{color}")

        #choice = st.sidebar._selectbox("Menu", menu)
    #st.write(f"Color to be predicted:{color} again")
    if st.button("Go"):
         match_image_by_color(COLORS[color[0]], THOLDS[color[0]], 4380, 6)




    # st.button("Ok", on_click=match_image_by_color(COLORS['GREEN'], 63, 4953, 6))

    # if color == "RED":


         ### INFO
     #st.title("Hello, Welcome to Art Analyzer!")
     #Hello_Title= '<p style="font-family:Avenir; color:Dark Grey; font-size: 46px;">Hello, Welcome to Art Analyzer!</p>'




    # if options == "Red":
    #     match_image_by_color(COLORS['RED'], 60, 10, 6)

    # if st.button("Ok"):
