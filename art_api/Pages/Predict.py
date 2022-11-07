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
import webcolors
import seaborn as sns
import pandas as pd
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
from tensorflow.keras.applications.resnet import preprocess_input as preproc_resnet
from tensorflow.keras import layers
from art_api import colour
from tensorflow.python.ops.numpy_ops import np_config
import plotly.graph_objects as go

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

def predict(image):
    #print(image.shape)
    #image = image.resize((256, 256), Image.ANTIALIAS)
    #print(image.shape)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colour.get_colors(image, 12, True)
    return hex_colors, X, y

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
            image = image.save("img.jpg")

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
            st.image("img.jpg", caption="Extracting colours for image", use_column_width=True)
            image = cv2.imread("img.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hex_colors, X, y = colour.get_colors(image, 5, True)
            st.text(hex_colors)
            st.text(y)
            fig1, ax1 = plt.subplots()
            ax1.pie(list(y), labels=hex_colors, colors = hex_colors)
            # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            #plt.pie(list(y), labels = hex_colors, colors = hex_colors)
            st.pyplot(fig1)
            #plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

            #The plot
            #colour.get_colors(colour.get_image(img), 12, True)
        # if img_file_buffer is not None:
        #     image_data = img_file_buffer.getvalue()
        #     st.text(type(type(image_data)))
        #     #st.text(type(image_data.shape))
        #     image = Image.open(img_file_buffer)
        #     img_array = np.array(image)
        #     st.text(type(image))
        #     #st.text(image.shape)
        #     st.text(type(img_array))
        #     st.text(img_array.shape)
        #     predict(image_data)
        # print(hex_colors)