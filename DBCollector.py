import typing
import pymorphy2
import abc
import db_schema
from entities import *
from sqlalchemy import select, insert, delete, update, create_engine
from sqlalchemy.orm import Session
import config
import psycopg2
from psycopg2 import DatabaseError


class AbstractDBCollector(abc.ABC):


    @abc.abstractmethod
    def add_ngram(self, ngram: Ngram):
        pass

    @abc.abstractmethod
    def observed_count(self, words: Ngram,
                       mask: typing.Sequence[bool]) -> int:
        """Mask marks if: word_i must be in position i (True), can not be in position i (False) or
        it is not important and should be ignored in query (None)"""
        pass

    @abc.abstractmethod
    def all_ngrams(self) -> typing.Generator:
        # should return a generator that loads ngrams in chunks per requirement, i guess
        pass

    @abc.abstractmethod
    def ngrams_count(self):
        pass


class DBCollector(AbstractDBCollector):
    model = db_schema.Ngram

    def __init__(self):
        self.counter = 0
        self.engine = create_engine(config.DB_CONNECTION_STRING)

    def execute(self, query):
        session = Session(self.engine)
        with session.begin():
            try:
                result = session.execute(query)
            except DatabaseError as err:
                session.rollback()
                raise err
            else:
                session.commit()
                return result


    def add_ngram(self, ngram: Ngram):
        query = insert(self.model).values(word1=str(ngram[0]), word2=str(ngram[1]), pos_mask=ngram.pos_mask)
        self.execute(query)
        self.counter += 1

    def observed_count(self, words: Ngram, mask: typing.Sequence[bool]):
        query = select(self.model)

        for prop, word, flag in zip((self.model.word1, self.model.word2), words, mask):
            if flag is True:
                query = query.where(prop == word.token)
            if flag is False:
                query = query.where(prop != word.token)

        answer = self.execute(query)
        return len(answer.fetchall())


    def all_ngrams(self) -> typing.Generator:
        def generator(result, batchsize: int):
            batch = result.fetchmany(batchsize)
            while batch:
                batch = [Ngram((WordToken(k[0].word1), WordToken(k[0].word2))) for k in batch]
                # print(*rows, sep='\n')
                while batch:
                    yield batch.pop()
                batch = result.fetchmany(batchsize)

        # TODO: return singhle ngrams, not batches

        query = select(self.model)
        res = self.execute(query)
        return generator(res, 1000)

    def ngrams_count(self):
        return self.counter


def migrate():
    engine = create_engine(config.DB_CONNECTION_STRING)
    db_schema.Base.metadata.create_all(engine)


migrate()