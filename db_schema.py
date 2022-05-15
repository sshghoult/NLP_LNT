from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, \
    UniqueConstraint


Base = declarative_base()


class Ngram(Base):
    __tablename__ = 'ngram'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word1 = Column(String)
    word2 = Column(String)
    pos_mask = Column(String)

