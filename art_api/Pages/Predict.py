# Import necessary libraries
from art_api import config
from enum import auto
from fileinput import filename
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

def load_image():
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
    #img = Image.open(uploaded_file).convert('RGB')
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(BytesIO(image_data))
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

def pretrain_model():
    model = load_model("../raw_data/models/resnet_12_classes")
    print("Model loaded")
    categories = config.CLASSES

    return model, categories

def predict(model, categories, image):
    print(image.shape)
    image = image.resize((256, 256), Image.ANTIALIAS)
    print(image.shape)
    input = preproc_resnet(image)
    pred = model.predict(input)
    return pred

def app():
    model, categories = pretrain_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)
    


    # with torch.no_grad():
    #     output = model(input_batch)

    # probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     st.write(categories[top5_catid[i]], top5_prob[i].item())