from pathlib import Path  

BUCKET_NAME = 'art-api'
#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
BUCKET_TRAIN_DATA_PATH = 'data/raw_data'
BUCKET_TRAIN_DATA_FILE = 'df_yourpaintings.csv'
PATH = Path("../raw_data")
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

#AUTO = tf.data.AUTOTUNE
#BATCH_SIZE = 64
#EPOCHS = 5

import os

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/erniesg/secrets/art-api-361815-5a390ae75860.json'
