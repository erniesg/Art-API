# Import necessary libraries
from art_api import config
from enum import auto
#from fileinput import filename
import json
import os
import streamlit as st
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import os
import urllib
from skimage import io
import pdb
from PIL import Image, ImageOps
import seaborn as sns
import pandas as pd
import tensorflow as tf
from io import BytesIO
from art_api import colour
from tensorflow.python.ops.numpy_ops import np_config
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

np_config.enable_numpy_behavior()

def load_image():
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
    #img = Image.open(uploaded_file).convert('RGB')
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image = Image.open(image_data)
        img_array = np.array(image)
        return image, img_array
        #return Image.open(BytesIO(image_data))
        #image = np.asarray(image_data)
        # image = Image.open(image_data)
        # img_resize = image.resize((256, 256), Image.ANTIALIAS)
        # print(f"image is of shape {img_resize.shape}")
        # img_array = np.asarray(img_resize)
        # print(f"image array is of shape {img_array.shape}")
        # return img_resize, img_array
    else:
        return None
#        img_array = np.array([np.array(Image.open(image_data).convert('RGB'))])
#        print(f"Shape of uploaded image is {img_array.shape}")

def optimal_cluster(X, y):
    X, y = X, y

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(3,12))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    elbow = visualizer.elbow_value_

    return elbow

def predict(image):
    #print(image.shape)
    #image = image.resize((256, 256), Image.ANTIALIAS)
    #print(image.shape)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colour.get_colors(image, 12, True)
    return hex_colors, X, y

def add_to_collection(img_file_buffer, hex_colors):
    data = {}
    data['id'] = img_file_buffer.name.rstrip(".jpg")
    for i in range(len(hex_colors)):
        data[str(i)] = hex_colors[i]
    len(hex_colors)
    for i in hex_colors:
        print(i)
    data
    df = pd.read_csv('../raw_data/df_10K_copy.csv')
    df = df.copy()
    data_df = pd.DataFrame([data])
    new_df = pd.concat([df, data_df])
    #new_df = new_df.drop(columns=['Unnamed: 0'])
    new_df.to_csv('../raw_data/df_10K_copy.csv', index=False)
    st.dataframe(data=new_df)
    return new_df

def app():
    img_file_buffer = st.file_uploader('Upload an image')
    # if img_file_buffer is not None:
    #     image = Image.open(img_file_buffer)
    #     img_array = np.array(image)
    #     st.text(type(image))
    #     st.text(type(img_array))
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            image = image.save(f"../raw_data/aws10k/{img_file_buffer.name}")

            #img_array = np.array(image) # if you want to pass it to OpenCV
            #image_data = img_file_buffer.getvalue()
            #st.image(image, caption="Image", use_column_width=True)
            #img = tf.image.resize(img_array, size=(224,224))
            #img = tf.expand_dims(img, axis=0)
            #Image.open(BytesIO(image_data))
            #img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
            #modified_image = img_array.reshape((img_array.shape[0]*img_array.shape[1], 3))
            #image = cv2.imread(img_array)
            
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # return image
            col1, col2 = st.columns(2)

            with col1:
                st.header("Image to process")
                st.image(f"../raw_data/aws10k/{img_file_buffer.name}", caption="Extracting colours for image", use_column_width=True)


            image = cv2.imread(f"../raw_data/aws10k/{img_file_buffer.name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hex_colors, X, y = colour.get_colors(image, 12, True)
            elbow = optimal_cluster(X, y)
            hex_colors, X, y = colour.get_colors(image, elbow, True)

            fig1, ax1 = plt.subplots(figsize=(8,6))
            ax1.pie(list(y), labels=hex_colors, colors = hex_colors)
            with col2:
                st.header("Colours extracted")
                st.pyplot(fig1)
                st.text(hex_colors)
                st.text(y)
                add_to_collection(img_file_buffer, hex_colors)
                st.text(f"Image {img_file_buffer.name} and its colours have been added to collection.")
                # add = st.button('Add to collection')
                # st.write(add)
                # if add:
                #     #add_to_collection(img_file_buffer, hex_colors)
                #     st.text("Image and its colours have been added to collection.")