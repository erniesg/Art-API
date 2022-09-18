# -*- coding: utf-8 -*-
"""Data Loader"""

from art_api import config
import pandas as pd

class DataLoader:
    """Data Loader class to load images from local or cloud"""

    @staticmethod
    def load_local(local_path):
        """Loads dataset from local path"""
        df = pd.read_csv(local_path)
        return df

    @staticmethod
    def load_cloud():
        """Loads dataset from cloud bucket"""
        df = pd.read_csv(f"gs://{config.BUCKET_NAME}/{config.BUCKET_TRAIN_DATA_PATH}/{config.BUCKET_TRAIN_DATA_FILE}")
        return df