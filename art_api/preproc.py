"""Preprocessing script to perform the following steps:
1. df -> sample
2. train-test-split
3. Rescale and augment training data <--- input as model layer
4. Convert to array and obtain X and y for training
"""

from art_api import config
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def sample(df):
    '''Pass in any dataframe, and be able to sample based on defined classes [can be pre-defined list] and number of samples
    Args:
    df - pass in the dataframe to be sampled
    
    Returns:
    df_sample - randomly select df w/ min_sample_num which is obtained from 
    '''
    min_sample_num = df.select_dtypes(include="number").sum().min()
    
    df_sample = pd.DataFrame()
    
    for cls in config.CLASSES:
        df_cls = df.query(f"{cls} == 1").sample(n=min_sample_num)
        df_sample = pd.concat([df_sample, df_cls])
        print(f"{min_sample_num} sampled per {cls}")
    print("\ndropping duplicates based on filename\n")    
    df_sample.drop_duplicates(subset=["filename"], inplace=True)
    print(f"After sampling, number of positive labels per class as follows: {df_sample.select_dtypes(include='number').sum()}, number of records in df_sample = {len(df_sample)}")
    
    return df_sample

def load_data(df):
    '''Perform train-test-split on data and load as array, building X and y for modelling'''
    
    