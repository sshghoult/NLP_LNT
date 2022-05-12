from spacy.lang.ru import Russian
import spacy
import pymorphy2
import string
import os
import abc
import typing
import entities
import numpy as np
import scipy

import nltk

# nltk.download("stopwords")
from nltk.corpus import stopwords

from typing import List, TextIO

from collections import deque


class BaseLazyTextPreprocessor(object):
    """Ленивый препроцессор для очистки текста перед работой с моделями"""

    def __init__(self, fp: TextIO, nlp, stopwords):
        self.src = fp
        self.nlp = nlp
        self.stopwords = stopwords
        self.stoppunct = set(string.punctuation).union({'...', '?!'})
        self.endpunct = {'.', '!', '?'}
        self.special_symbols = {'\n', '\r'}

    def preprocess(self, chunk: str, use_stopwords=True):
        doc = self.nlp(chunk)

        check_against = [self.stoppunct, [' '], self.special_symbols]
        if use_stopwords:
            check_against.append(self.stopwords)

        cleaned = []
        for token in doc:
            dump_flag = False
            for forbidden in check_against:
                if any(k in forbidden for k in token.lemma_):
                    dump_flag = True
                    break
            if not dump_flag:
                cleaned.append(token.lemma_)

        return cleaned


class CollocationLazyTextProcessor(BaseLazyTextPreprocessor):
    def __init__(self, fp: TextIO, nlp, stopwords):
        super().__init__(fp, nlp, stopwords)
        self.clean_buffer = deque()
        self.eof_flag = False

    def __next__(self):
        if len(self.clean_buffer) == 0:
            if self.eof_flag:
                raise StopIteration
            self.__load_buffer()
        next_ = self.clean_buffer.pop()
        return next_

    def __iter__(self):
        return self

    def __load_buffer(self, size=5):
        # вычитать предложение, отправить его в препроцесс, положить в буффер
        # eof reaction might be a problem later if connected to a pipe or socket
        for _ in range(size):
            subbuffer = []
            ptr = ''
            while ptr not in self.endpunct:
                ptr = self.src.read(1)
                if ptr == '':
                    self.eof_flag = True
                    break
                subbuffer.append(ptr)
            if self.eof_flag:
                break
            cleaned = self.preprocess(''.join(subbuffer), use_stopwords=False)
            if cleaned:
                self.clean_buffer.appendleft(cleaned)


class SourceAggregatorContinuous(object):
    """Объект с интерфейсом TextIO, незаметно для следующих звеньев объединяющий несколько
    непрерывных потоков данных в один последовательно"""

    def __init__(self, sources: List[os.PathLike]):
        self.src = sources
        self.cur = None


class CollocationStatisticsConsumerDB(object):
    def __init__(self, window_size, container: entities.AbstractDBCollector):
        self.window_border = window_size // 2
        self.container = container

    def process(self, generators: typing.List[typing.Iterable]):
        for flow in generators:
            # theoretically, each flow and each chunk might be processed in parallel, summarizing
            # the counters afterwards
            for chunk in flow:

                parsed_words = [entities.WordToken(wi) for wi in chunk]

                for w in range(len(chunk) - 1):
                    self.register_pair(parsed_words[w], parsed_words[w + 1])

    def register_pair(self, a, b):
        ngram = entities.Ngram((a, b))

        self.container.add_ngram(ngram)

    def ngram_chi_square(self, ngram: entities.Ngram[entities.WordToken, entities.WordToken]) -> \
            typing.Tuple[float, float]:

        observed = np.empty([2, 2])
        expected = np.empty([2, 2])

        # TODO: can be optimized, parts of equations are reoccurring

        observed[0][0] = self.container.observed_count(ngram, [True, True])
        expected[0][0] = self.container.observed_count(ngram, [True, None]) * self.container.observed_count(ngram, [None, True]) / \
                         self.container.ngrams_count()

        observed[0][1] = self.container.observed_count(ngram, [False, True])
        expected[0][1] = self.container.observed_count(ngram, [False, None]) * self.container.observed_count(ngram, [None, True]) / \
                         self.container.ngrams_count()


        observed[1][0] = self.container.observed_count(ngram, [True, False])
        expected[1][0] = self.container.observed_count(ngram, [True, None]) * self.container.observed_count(ngram, [None, False]) / \
                         self.container.ngrams_count()

        observed[1][1] = self.container.observed_count(ngram, [False, False])
        expected[1][1] = self.container.observed_count(ngram, [False, None]) * self.container.observed_count(ngram, [None, False]) / \
                         self.container.ngrams_count()

        return scipy.stats.chisquare(observed.flatten(), expected.flatten())

    def compute_statistics(self):
        chi_square = {}
        ngrams_generator = self.container.all_ngrams()

        for i in ngrams_generator:
            chi_square[i] = self.ngram_chi_square(i)

        return chi_square


# source = 'prestuplenie-i-nakazanie.txt'
source = 'notebooks/tmp.txt'
# source = 'pinshort.txt'
# source = ''

#
# coll = entities.InMemoryCategorizedCollector()
# consumer = CollocationStatisticsConsumer(0, coll)
#
# nlp = spacy.load('ru_core_news_sm')
#
# with open(source, 'r', encoding='utf-8') as fp:
#     LTP = CollocationLazyTextProcessor(fp, nlp, stopwords.words("russian"))
#     consumer.process([LTP])
#     consumer.compute_statistics()
#
# print(*coll.dict.items(), sep='\n\n', end='\n\n\n\n')
# print(*consumer.statistics.items(), sep='\n\n')
