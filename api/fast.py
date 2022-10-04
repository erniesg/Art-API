from fastapi import FastAPI, Form, status, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
#from TaxiFareModel.trainer import Trainer
from datetime import datetime
import pytz
import pandas as pd
import joblib
import os
import pandas as pd
from art_api import config
from art_api.database import Base, engine, SessionLocal
from pydantic import BaseModel
from sqlalchemy.orm import Session
import art_api.models
import art_api.schemas
from typing import List

# Create the database
Base.metadata.create_all(engine)

app = FastAPI()

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return "Welcome to Art API"

@app.post("/tag", status_code=status.HTTP_201_CREATED)
def create_tag(tags: art_api.schemas.TagCreate, session: Session = Depends(get_session)):
    # create a new database session
    #session = Session(bind=engine, expire_on_commit=False)
    session = SessionLocal()

    # create an instance of the  database model
    artapidb = art_api.models.Tags(user_tags = tags.user_tags)

    # add it to the session and commit it
    session.add(artapidb)
    session.commit()
    session.refresh(artapidb) 
    
    # # grab the id given to the object from the database
    # id = artapidb.id

    # close the session
    #session.close()

    # return the id
    return artapidb

@app.get("/tag", response_model = List[art_api.schemas.TagRequest])
def read_tag_list(session: Session = Depends(get_session)):
    # create a new database session
    #session = Session(bind=engine, expire_on_commit=False)
    session = SessionLocal()

    # get all todo items
    tags_list = session.query(art_api.models.Tags).all()

    # close the session
    #session.close()

    return tags_list

@app.get("/tag/{id}", response_model=art_api.schemas.TagRequest)
def read_tag(id: int, session: Session = Depends(get_session)):
    # create a new database session
    #session = Session(bind=engine, expire_on_commit=False)
    session = SessionLocal()

    # get the item with the given id
    tags = session.query(art_api.models.Tags).get(id)

    # close the session
    #session.close()

    # check if item with given id exists. If not, raise exception and return 404 not found response
    if not tags:
        raise HTTPException(status_code=404, detail=f"item with id {id} not found")

    return tags

@app.put("/tag/{id}")
def update_tag(id: int, user_tags: str, session: Session = Depends(get_session)):

    # create a new database session
    #session = Session(bind=engine, expire_on_commit=False)
    session = SessionLocal()    

    # get the item with the given id
    tags = session.query(art_api.models.Tags).get(id)
    
    # update item with the given task (if an item with the given id was found)
    if tags:
        tags.user_tags = user_tags
        session.commit()

    # close the session
    #session.close()

    # check if item with given id exists. If not, raise exception and return 404 not found response
    if not tags:
        raise HTTPException(status_code=404, detail=f"tag item with id {id} not found")

    return tags

@app.delete("/tag/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tag(id: int, session: Session = Depends(get_session)):

    # create a new database session
    #session = Session(bind=engine, expire_on_commit=False)
    session = SessionLocal()
    # get the todo item with the given id
    tags = session.query(art_api.models.Tags).get(id)

    # if todo item with given id exists, delete it from the database. Otherwise raise 404 error
    if tags:
        session.delete(tags)
        session.commit()
        #session.close()
    else:
        raise HTTPException(status_code=404, detail=f"tag item with id {id} not found")

    return None

# @app.get("/tag")
# def read_tag_list():
#     return "read tag list"

# @app.get("/")
# def root():
#     return {"message": "Hello World"}

# @app.post("/post")
# def post(id, input):
    
#     params = dict(
#         id=id,
#         input=input)
        
#     X_user = pd.DataFrame(data=params, index=[0])
#     #X_user.to_csv(os.path.join(config.PATH_USERS, "users.csv"))
#     return X_user

# @app.get("/predict")
# def predict(pickup_datetime,
#             pickup_longitude,
#             pickup_latitude,
#             dropoff_longitude,
#             dropoff_latitude,
#             passenger_count):
    
#     params = dict(
#         key='2013-07-06 17:18:00.000000119',
#         pickup_datetime=pickup_datetime,
#         pickup_longitude=pickup_longitude,
#         pickup_latitude=pickup_latitude,
#         dropoff_longitude=dropoff_longitude,
#         dropoff_latitude=dropoff_latitude,
#         passenger_count=passenger_count
#         )
    
#     pickup_datetime = datetime.strptime(params['pickup_datetime'], "%Y-%m-%d %H:%M:%S")

#     eastern = pytz.timezone("US/Eastern")
#     localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    
#     utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

#     formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

#     params['pickup_datetime'] = formatted_pickup_datetime
    
#     X_pred = pd.DataFrame(data=params, index=[0])
#     PATH_TO_LOCAL_MODEL = 'model.joblib'
#     pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
#     y_pred = pipeline.predict(X_pred)
#     y_pred = y_pred.tolist()
#     prediction = {"fare": y_pred}

#     return prediction

