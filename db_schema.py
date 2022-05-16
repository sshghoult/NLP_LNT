from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, \
    UniqueConstraint, Float


Base = declarative_base()


class Ngram(Base):
    __tablename__ = 'ngram'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word1 = Column(String, index=True)
    word2 = Column(String, index=True)
    pos_mask = Column(String, index=True)
    count = Column(Integer, default=1)
    chi_square = Column(Float)
    p_value = Column(Float)
    unique_bigrams = UniqueConstraint(word1, word2, name='unique_bigrams')

