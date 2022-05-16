import typing
import pymorphy2
import abc
import db_schema
from entities import *
from sqlalchemy import select, insert, delete, update, create_engine, func, desc, asc
from sqlalchemy.dialects.postgresql import insert as ps_insert
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
        query = ps_insert(self.model).values(word1=str(ngram[0]), word2=str(ngram[1]), pos_mask=ngram.pos_mask)
        query = query.on_conflict_do_update(constraint=self.model.unique_bigrams.name, set_=dict(count=self.model.count + 1))
        self.execute(query)
        self.counter += 1

    def update_ngram(self, ngram: Ngram, stats: typing.Tuple):
        query = update(self.model).values(chi_square=stats[0], p_value=stats[1]).where(
            self.model.word1 == str(ngram[0]), self.model.word2 == str(ngram[1]))
        self.execute(query)

    def get_statistics(self):
        def generator(result, batchsize: int):
            batch = result.fetchmany(batchsize)
            while batch:
                batch = [(Ngram((WordToken(k[0]), WordToken(k[1]))), k[2]) for k in batch]
                # print(*rows, sep='\n')

                counter = 0
                while counter < len(batch):
                    yield batch[counter]
                    counter += 1
                batch = result.fetchmany(batchsize)

        query = select(self.model.word1, self.model.word2, self.model.chi_square).order_by(desc(self.model.chi_square))
        res = self.execute(query)
        return generator(res, 1000)

    def observed_count(self, words: Ngram, mask: typing.Sequence[bool]):
        query = select(func.sum(self.model.count))

        for prop, word, flag in zip((self.model.word1, self.model.word2), words, mask):
            if flag is True:
                query = query.where(prop == word.token)
            if flag is False:
                query = query.where(prop != word.token)
        answer = self.execute(query).fetchall()
        answer = answer[0][0]
        if answer is None:
            answer = 0

        # print(answer)
        return answer

    def all_ngrams(self, offset=None, limit=None) -> typing.Generator:
        def generator(result, batchsize: int):
            batch = result.fetchmany(batchsize)
            while batch:
                batch = [Ngram((WordToken(k[0].word1), WordToken(k[0].word2))) for k in batch]
                # print(*rows, sep='\n')
                while batch:
                    yield batch.pop()
                batch = result.fetchmany(batchsize)

        query = select(self.model).order_by(asc(self.model.id))

        if offset is not None:
            query = query.offset(offset)

        if limit is not None:
            query = query.limit(limit)

        res = self.execute(query)
        return generator(res, 1000)

    def ngrams_count(self):
        return self.counter


def migrate():
    engine = create_engine(config.DB_CONNECTION_STRING)
    db_schema.Base.metadata.create_all(engine)


migrate()
