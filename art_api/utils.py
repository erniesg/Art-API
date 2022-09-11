import pandas as pd
from urllib.request import Request, urlretrieve
from pathlib import Path  
import ssl
import os
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from art_api import config

def init():
    imgs = []
    df = pd.read_csv(f"gs://{config.BUCKET_NAME}/{config.BUCKET_TRAIN_DATA_PATH}/{config.BUCKET_TRAIN_DATA_FILE}")
    return imgs, df

def resize_rescale(df):
    '''download images and resize to 256x256 if not already in local disk'''
#    df = pd.read_csv(config.PATH_FILE)
    
    for index, row in df.iterrows():
        img_file = str(row["filename"])
        image = Image.open(os.path.join(config.PATH_YOURPAINTINGS, img_file))
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.save(os.path.join(config.PATH_YOURPAINTINGS_SM, img_file))
        #imgs.append(np.array(image/255))
    return imgs, df

def load_data(df):
    '''generates X and y'''
    '''read from disk or read from cloud <--- to be implemented'''
    for index, row in df.iterrows():
        img_file = str(row["filename"])
        image = Image.open(os.path.join(config.PATH_YOURPAINTINGS_SM, img_file))   
        imgs.append(np.array(image))
        X = np.array(imgs)
        X.shape
        y = df.drop(columns=['index', 'Image URL', 'Web page URL', 'Subset', 'Labels', 'filename', 'labels'])
        y.shape
    return X, y