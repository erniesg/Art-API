# Import necessary libraries
from enum import auto
from fileinput import filename
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
import glob
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



OBJECT = {
    'aeroplane':'aeroplane',
    'bird':'bird',
    'boat':'boat',
    'chair':'chair',
    'cow':'cow',
    'table':'diningtable',
    'dog':'dog',
    'horse':'horse',
    'sheep':'sheep',
    'train':'train',
    'people': 'figure',
    'building': 'building'

}


def match_image_by_obj(object, rows_to_chk, columns_to_chk): #num_pic_col
    img_obj_df = pd.read_csv(('../raw_data/12_classes_df_pred_0.5.csv'), index_col=0)
    #img_colors_df = app()
    #* Selected_color_ is user's chosen color for search
    selected_obj_name = object
    #st.write(f'selected_object_name = {selected_obj_name}')


    filtered = []
    caption = []

    for row in range(rows_to_chk):
        #st.write(f'\nRow_num = {row}')
        row_img_shown = False  #denotes whether row image shown b4. reset to false for each new row.
        for column in range(columns_to_chk):
            print(f'\nRow, Column_num = {row},{column}')
            #f = lambda x: True if x==1 else False
            column_obj= img_obj_df.loc[row][column+1]

            select_image_id = False

            a = img_obj_df.loc[img_obj_df[f'{object}'] == 1, 'filename']
            df = a.to_frame(name='filename')
            #st.write(df['filename'])




            #a.reset_index(inplace=True, drop=True)
    #print(df.head(20))
    for i in df['filename']:
        image_path = f'../raw_data/aws10k/{i}'
        #st.write(image_path)

        #     st.write(image_path)
            #glob(image_path)
        #for j in a:
        #Image.open(image_path )
            # cap = image_path.split('/')[-1].split('.')[0]
            # filtered.append(im)

    # cols = cycle(st.columns(3)) # st.columns here since it is out of beta at the time I'm writing this

    # for  filteredImage, cap in zip(filtered, caption):
    #     next(cols).image(filteredImage, width=250, caption=cap, use_column_width=auto)
                # try:
                #     im = Image.open(image_path)

                #     cap = image_path.split('/')[-1].split('.')[0]

                #     caption.append(f'Accession Number: {cap} \n(Artist) \n(Artname)\n(Year)')

                #     filtered.append(im)

                # except:
                #         pass

                #st.write(image_path)



            #st.write(df['filename'])

                #str(column_obj).strip()== '1':
                # a = img_obj_df.loc[img_obj_df[f'{object}'] == 1, 'filename']
                # st.write(a)
            #np.where(img_obj_df[f'{object}'] == '1')



                    #st.write('loop passed')

                #object.fillna(method='ffill')



    #if img_obj_df.apply(lambda x: (x.column == 1), axis=1):

        #column_LABcolor = hex2lab(column_hexcolor) #calling hex2lab
        #st.write(f'Column_LABcolor = {column_LABcolor}')
        #diff = deltaE_cie76(selected_color_lab, column_LABcolor)

        #st.write(f'Delta Diff= {diff}')
        #if (diff < threshold):
                #select_image_id = True
        #st.write('starts with 1')
            #st.write(f'\nrow_img_shown = {row_img_shown}')
            #st.write(f'\nimg_colors_df_row = {img_colors_df.loc[row]}')
                #image_id = img_obj_df.loc[row]['filename']




        print(f'\nimage_id = {i}')
        # if row_img_shown == False:
        #     #Show the image
        #     image_path = f'/Users/plst/code/supersuzie/ArtWebsite/aws10K/{i}'
        #     # creating a object
        #     # im = Image.open(image_path)

        try:

            im = Image.open(image_path)

            cap = image_path.split('/')[-1].split('.')[0]

            caption.append(f'Accession Number: {cap}') #<br><br>(Artist) <br><br>(Artname) <br><br>(Year)

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


    img_obj_df = pd.read_csv(('../raw_data/12_classes_df_pred_0.5.csv'),index_col=0)

    #adding a color selectbox

    # def choose_color(options):

    object = st.multiselect(
        'Select Object',
        ['aeroplane','bird','boat','chair','cow','table','dog','horse','sheep','train', 'people', 'building']) #figure, building
    #st.sidebar.button('Run')
    st.write(f"Object to be predicted:{object}")

        #choice = st.sidebar._selectbox("Menu", menu)
    #st.write(f"Color to be predicted:{color} again")
    if st.button("Go"):
         match_image_by_obj(OBJECT[object[0]], 4900, 10) #THOLDS[color[0]],4




    # st.button("Ok", on_click=match_image_by_color(COLORS['GREEN'], 63, 4953, 6))

    # if color == "RED":


         ### INFO
     #st.title("Hello, Welcome to Art Analyzer!")
     #Hello_Title= '<p style="font-family:Avenir; color:Dark Grey; font-size: 46px;">Hello, Welcome to Art Analyzer!</p>'




    # if options == "Red":
    #     match_image_by_color(COLORS['RED'], 60, 10, 6)

    # if st.button("Ok"):
