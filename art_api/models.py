from sqlalchemy import Column, Integer, String
from art_api.database import Base
    
class Tags(Base):
    __tablename__ = 'user_tags'
    id = Column(Integer, primary_key=True)
    user_tags =  Column(String(256))    