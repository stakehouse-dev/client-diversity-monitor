'''ETL model'''

from database_config import Base
from sqlalchemy import Column, String, DECIMAL


class ValidatorLSD(Base):
    '''
    This is a SQLAlchemy model class for a table named "Validator_Indexes" 
    with columns for bls_key, epoch, and indexes.
    '''
    __tablename__ = "Validator_Indexes"

    bls_key = Column(String(255), primary_key=True, nullable=False, index=True)
    epoch = Column(DECIMAL(10, 0), primary_key=True, nullable=False, index=True)
    indexes = Column(DECIMAL(10, 0), nullable=False, index=True)
