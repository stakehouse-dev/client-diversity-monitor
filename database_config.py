from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from apiconfig import APIConfig


session_options = {"autoflush": True}
config = APIConfig()


Base = declarative_base()

Base = declarative_base()
engine = create_engine(
    "mysql+pymysql://" + config.etl_db("connection_string"),
    convert_unicode=True,
    pool_size=int(config.etl_db("connection_pool_size")),
    pool_recycle=3600,
)
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    **session_options
)
