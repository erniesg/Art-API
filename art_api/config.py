from pathlib import Path
import wandb
from wandb.keras import WandbCallback

BUCKET_NAME = 'art-api'
#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
BUCKET_TRAIN_DATA_PATH = 'data/raw_data'
BUCKET_TRAIN_DATA_FILE = 'df_yourpaintings.csv'
PATH = Path("../raw_data")
DF_PATH = Path("../raw_data/df_yourpaintings.csv")
PATH_MODELS = Path("../models")
PATH_YOURPAINTINGS = PATH/"yourpaintings"
PATH_YOURPAINTINGS_SM = PATH/"yourpaintings_sm"
PATH_GOOGLE = PATH/"google"
PATH_GOOGLE_SM = PATH/"google_sm"
MODEL_NAME = 'art_api'
MODEL_VERSION = 'baseline'
PATH_FILE = Path(PATH_YOURPAINTINGS/"df_yourpaintings.csv")  
PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = (256, 256)
INTERPOLATION = "bilinear"
CLASSES = ["aeroplane", "bird", "boat", "chair", "cow", "diningtable", "dog", "horse", "sheep", "train"]
PATH_BING = PATH/"bing"
PATH_USERS = "../raw_data/users"
IM_SIZE = 256

'''Training parameters below'''
wandb.config = {
  "LEARNING_RATE": 0.0001,
  "N_EPOCHS": 50,
  "BATCH_SIZE": 32,
  "DROPOUT_RATE": 0.0,
  "IM_SIZE": 256,
  "REGULARIZATION_RATE": 0.0,
  "N_FILTERS": 6,
  "KERNEL_SIZE": 3,
  "N_STRIDES": 1,
  "POOL_SIZE": 2,
  "N_DENSE_1": 100,
  "N_DENSE_2": 10,
}

'''
#AUTO = tf.data.AUTOTUNE
#BATCH_SIZE = 64
#EPOCHS = 5

IM_SIZE = 256
DROPOUT_RATE = CONFIGURATION['DROPOUT_RATE']
REGULARIZATION_RATE = CONFIGURATION['REGULARIZATION_RATE']
N_FILTERS = CONFIGURATION['N_FILTERS']
KERNEL_SIZE = CONFIGURATION['KERNEL_SIZE']
POOL_SIZE = CONFIGURATION['POOL_SIZE']
N_STRIDES = CONFIGURATION['N_STRIDES']
LEARNING_RATE = 1e-4

wandb.config = {
  "LEARNING_RATE": 0.001,
  "N_EPOCHS": 5,
  "BATCH_SIZE": 128,
  "DROPOUT_RATE": 0.0,
  "IM_SIZE": 224,
  "REGULARIZATION_RATE": 0.0,
  "N_FILTERS": 6,
  "KERNEL_SIZE": 3,
  "N_STRIDES": 1,
  "POOL_SIZE": 2,
  "N_DENSE_1": 100,
  "N_DENSE_2": 10,
}
'''
CONFIGURATION = wandb.config
import os

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/erniesg/secrets/art-api-361815-5a390ae75860.json'
