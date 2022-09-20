# -*- coding: utf-8 -*-
'''Data Loader'''

from art_api import config
import pandas as pd
import os
import numpy as np

class DataLoader:
    '''Data Loader class to load dataframes from local or cloud'''

    @staticmethod
    def load_local(local_path):
        '''Loads dataset from local path'''
        df = pd.read_csv(local_path)
        return df

    @staticmethod
    def load_cloud():
        '''Loads dataset from cloud bucket'''
        df = pd.read_csv(f"gs://{config.BUCKET_NAME}/{config.BUCKET_TRAIN_DATA_PATH}/{config.BUCKET_TRAIN_DATA_FILE}")
        return df
    
    '''Methods to add to dataframe with more images of existing class or create new class (to be implemented)'''
    
    def add_img(dir):
        '''Adds images from dir
        Args: make sure you specify the relative file path
        e.g. add_img("../raw_data/test_sm")
            dir: directory to add images from
        
        Returns:
            df_add: df with new images added to existing classes
        '''
        
        df = DataLoader.load_cloud()
        df['path'] = config.PATH_YOURPAINTINGS_SM
        
        fname = []
        for file in os.listdir(dir):
            fname.append(file)
        
        df_file = pd.DataFrame(fname, columns = ['filename'])
        df_file['path'] = dir
        df_add = pd.concat([df, df_file], ignore_index=True)
            
        for index, row in df_add.iterrows():
            if "aeroplane" in row["filename"]:
                df_add.at[index, "aeroplane"] = 1
            if "bird" in row["filename"]:
                df_add.at[index, "bird"] = 1
            if "boat" in row["filename"]:
                df_add.at[index, "boat"] = 1
            if "chair" in row["filename"]:
                df_add.at[index, "chair"] = 1
            if "cow" in row["filename"]:
                df_add.at[index, "cow"] = 1
            if "diningtable" in row["filename"]:
                df_add.at[index, "diningtable"] = 1
            if "dog" in row["filename"]:
                df_add.at[index, "dog"] = 1        
            if "horse" in row["filename"]:
                df_add.at[index, "horse"] = 1        
            if "sheep" in row["filename"]:
                df_add.at[index, "sheep"] = 1        
            if "train" in row["filename"]:
                df_add.at[index, "train"] = 1     

        df_add = df_add.drop(columns=["index", "Image URL", "Web page URL", "Subset", "Labels", "labels"])
        df_add = df_add.replace(np.nan,0)
        df_add = df_add.astype({"aeroplane":"int", "bird":"int", "boat":"int", "chair":"int",
                                "cow":"int", "diningtable":"int", "dog":"int", "horse":"int",
                                "sheep":"int", "train":"int"})

        return df_add
    
    '''Methods to load images from local or cloud below'''
    
    