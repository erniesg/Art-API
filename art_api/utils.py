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
#    df = pd.read_csv
    df = pd.read_csv(f"gs://{config.BUCKET_NAME}/{config.BUCKET_TRAIN_DATA_PATH}/{config.BUCKET_TRAIN_DATA_FILE}")
    return imgs, df

def resize(dir, dir_sm):
    '''Resize image to 256x256 (to be implemented: if not already in local disk)
    Args: make sure you specify the relative file path,
    e.g. resize("../raw_data/test", "../raw_data/test_sm")
        dir: source directory
        dir_sm: destination directory
    Returns:
        String output describing number of images resized
    '''
    counter = 0
    for file in os.listdir(dir):
        image = Image.open(os.path.join(dir, file))
        image = image.resize((256, 256), Image.ANTIALIAS)
        if not os.path.exists(dir_sm):
            print(f"Directory does not exist. Creating {dir_sm}.")
            os.makedirs(dir_sm)
        image.save(os.path.join(dir_sm, file))
        counter +=1
    return f"{counter} images resized and rescaled to {dir_sm}"

def load_data():
    '''generates X and y'''
    '''read from disk or read from cloud <--- to be implemented'''
    imgs, df = init()
    for index, row in df.iterrows():
        img_file = str(row["filename"])
        image = Image.open(os.path.join(config.PATH_YOURPAINTINGS_SM, img_file))   
        imgs.append(np.array(image))
        X = np.array(imgs)
        X.shape
        y = df.drop(columns=['index', 'Image URL', 'Web page URL', 'Subset', 'Labels', 'filename', 'labels'])
        y.shape
    return X, y

def load_google():
    '''Loads images scraped from google and creates a list per class
    '''
    fname = []
    for file in os.listdir(config.PATH_GOOGLE_SM):
        fname.append(file)
        
    lst_aeroplane, lst_bird, lst_boat, lst_chair, lst_cow, lst_diningtable, lst_dog, lst_horse, lst_sheep, lst_train =  [[s for s in fname if cls in s] for cls in config.CLASSES]
    print(f"list of aeroplane contains {len(lst_aeroplane)} images")
    print(f" list of bird contains {len(lst_bird)} images")
    print(f" list of boat contains {len(lst_boat)} images")
    print(f" list of chair contains {len(lst_chair)} images")
    print(f" list of cow contains {len(lst_cow)} images")
    print(f" list of diningtable contains {len(lst_diningtable)} images")
    print(f" list of dog contains {len(lst_dog)} images")
    print(f" list of horse contains {len(lst_horse)} images")
    print(f" list of sheep contains {len(lst_sheep)} images")
    print(f" list of train contains {len(lst_train)} images")
    return lst_aeroplane, lst_bird, lst_boat, lst_chair, lst_cow, lst_diningtable, lst_dog, lst_horse, lst_sheep, lst_train

def get_classes_df():
    undersample_num = y.sum().min()
    print(f"Minimum number of samples to take is {undersample_num}.")
    classes = ['aeroplane', 'bird', 'boat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'sheep', 'train']
    df_aeroplane, df_bird, df_boat, df_chair, df_cow, df_diningtable, df_dog, df_horse, df_sheep, df_train = [y_sub[y_sub[cls] == 1] for cls in classes]
    return df_aeroplane, df_bird, df_boat, df_chair, df_cow, df_diningtable, df_dog, df_horse, df_sheep, df_train

def add_rows():
    imgs, df = init()
    lst_aeroplane, lst_bird, lst_boat, lst_chair, lst_cow, lst_diningtable, lst_dog, lst_horse, lst_sheep, lst_train = load_google()
    df_big = df.copy()
    dict_aeroplane = dict.fromkeys(lst_aeroplane, 1)
    dict_bird = dict.fromkeys(lst_bird, 1)
    dict_boat = dict.fromkeys(lst_boat, 1)
    dict_chair = dict.fromkeys(lst_chair, 1)
    dict_cow = dict.fromkeys(lst_cow, 1)
    dict_diningtable = dict.fromkeys(lst_diningtable, 1)
    dict_dog = dict.fromkeys(lst_dog, 1)
    dict_horse = dict.fromkeys(lst_horse, 1)
    dict_sheep = dict.fromkeys(lst_sheep, 1)
    dict_train = dict.fromkeys(lst_train, 1)
    
    add_aeroplane = pd.DataFrame.from_dict(dict_aeroplane, orient='index')
    add_aeroplane = add_aeroplane.reset_index()
    add_aeroplane.columns=['filename', 'aeroplane']
    
    add_bird = pd.DataFrame.from_dict(dict_bird, orient='index')
    add_bird = add_bird.reset_index()
    add_bird.columns=['filename', 'bird']
    
    add_boat = pd.DataFrame.from_dict(dict_boat, orient='index')
    add_boat = add_boat.reset_index()
    add_boat.columns=['filename', 'boat']
    
    add_chair = pd.DataFrame.from_dict(dict_chair, orient='index')
    add_chair = add_chair.reset_index()
    add_chair.columns=['filename', 'chair']

    add_cow = pd.DataFrame.from_dict(dict_cow, orient='index')
    add_cow = add_cow.reset_index()
    add_cow.columns=['filename', 'cow']
    
    add_diningtable = pd.DataFrame.from_dict(dict_diningtable, orient='index')
    add_diningtable = add_diningtable.reset_index()
    add_diningtable.columns=['filename', 'diningtable']

    add_dog = pd.DataFrame.from_dict(dict_dog, orient='index')
    add_dog = add_dog.reset_index()
    add_dog.columns=['filename', 'dog']

    add_horse = pd.DataFrame.from_dict(dict_horse, orient='index')
    add_horse = add_horse.reset_index()
    add_horse.columns=['filename', 'horse']
    
    add_sheep = pd.DataFrame.from_dict(dict_sheep, orient='index')
    add_sheep = add_sheep.reset_index()
    add_sheep.columns=['filename', 'sheep']
    
    add_train = pd.DataFrame.from_dict(dict_train, orient='index')
    add_train = add_train.reset_index()
    add_train.columns=['filename', 'train']
        
    df_big = pd.concat([df_big, add_aeroplane, add_bird, add_boat, add_chair, add_cow, add_diningtable, add_dog, add_horse, add_sheep, add_train], ignore_index=True)
    
    return df_big

def undersample(n, cls_df):
    '''This function will undersample all other classes based on the minority class
    Args:
    n = number of samples to obtain
    X = class df, containing images from only 1 class
    y = target df
    Returns:
    '''
    cls_df = cls_df.sample(n)
    cls_X = X[cls_df.index]
    print(cls_df.index)

    return cls_df, cls_X