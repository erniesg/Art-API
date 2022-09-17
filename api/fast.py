from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
#from TaxiFareModel.trainer import Trainer
from datetime import datetime
import pytz
import pandas as pd
import joblib
import os
import pandas as pd
from art_api import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/post")
def post(id, input):
    
    params = dict(
        id=id,
        input=input)
        
    X_user = pd.DataFrame(data=params, index=[0])
    #X_user.to_csv(os.path.join(config.PATH_USERS, "users.csv"))
    return X_user

@app.get("/predict")
def predict(pickup_datetime,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count):
    
    params = dict(
        key='2013-07-06 17:18:00.000000119',
        pickup_datetime=pickup_datetime,
        pickup_longitude=pickup_longitude,
        pickup_latitude=pickup_latitude,
        dropoff_longitude=dropoff_longitude,
        dropoff_latitude=dropoff_latitude,
        passenger_count=passenger_count
        )
    
    pickup_datetime = datetime.strptime(params['pickup_datetime'], "%Y-%m-%d %H:%M:%S")

    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    params['pickup_datetime'] = formatted_pickup_datetime
    
    X_pred = pd.DataFrame(data=params, index=[0])
    PATH_TO_LOCAL_MODEL = 'model.joblib'
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    y_pred = pipeline.predict(X_pred)
    y_pred = y_pred.tolist()
    prediction = {"fare": y_pred}

    return prediction

