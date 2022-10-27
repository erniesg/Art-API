from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a sqlite engine instance
engine = create_engine("sqlite:///../raw_data/artapidb.db")

# Create SessionLocal class from sessionmaker factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Create a DeclarativeMeta instance
Base = declarative_base()

    # c_aeroplane = Column(Integer)
    # c_bird = Column(Integer)
    # c_boat = Column(Integer)
    # c_chair = Column(Integer)
    # c_cow = Column(Integer)
    # c_diningtable = Column(Integer)
    # c_dog = Column(Integer)
    # c_horse = Column(Integer)
    # c_sheep = Column(Integer)
    # c_train = Column(Integer)
