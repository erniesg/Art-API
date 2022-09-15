from pathlib import Path  

BUCKET_NAME = 'art-api'
#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
BUCKET_TRAIN_DATA_PATH = 'data/raw_data'
BUCKET_TRAIN_DATA_FILE = 'df_yourpaintings.csv'
PATH = Path("../raw_data")
PATH_YOURPAINTINGS = PATH/"yourpaintings"
PATH_YOURPAINTINGS_SM = PATH/"yourpaintings_sm"
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
os.environ['API_PASSWORD'] = '5268e5e33b2e615dae8f035b9de98a36756960f693d80baa022c54d757cdf947'
