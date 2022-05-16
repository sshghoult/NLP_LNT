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
import DBCollector
import math
import nltk
import multiprocessing

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
    def __init__(self):
        # self.window_border = window_size // 2
        self.container = DBCollector.DBCollector()

    def process(self, generators: typing.List[typing.Iterable]):
        counter = 0
        for flow in generators:
            # theoretically, each flow and each chunk might be processed in parallel, summarizing
            # the counters afterwards
            for chunk in flow:

                parsed_words = [entities.WordToken(wi) for wi in chunk]

                for w in range(len(chunk) - 1):
                    self.register_pair(parsed_words[w], parsed_words[w + 1])
                    counter += 1
                    print(f"{counter} bigrams processed")

    def register_pair(self, a, b):
        ngram = entities.Ngram((a, b))

        self.container.add_ngram(ngram)

    def ngram_chi_square(self, ngram: entities.Ngram) -> \
            typing.Tuple[float, float]:

        observed = np.empty([2, 2])
        expected = np.empty([2, 2])

        observed[0][0] = self.container.observed_count(ngram, [True, True])
        observed[0][1] = self.container.observed_count(ngram, [False, True])
        observed[1][0] = self.container.observed_count(ngram, [True, False])
        observed[1][1] = self.container.observed_count(ngram, [False, False])

        total_ngrams = self.container.ngrams_count()

        obs_w1_true = self.container.observed_count(ngram, [True, None])

        obs_w1_false = self.container.observed_count(ngram, [False, None])

        obs_w2_true = self.container.observed_count(ngram, [None, True])

        obs_w2_false = self.container.observed_count(ngram, [None, False])

        expected[0][0] = obs_w1_true * obs_w2_true / total_ngrams

        expected[0][1] = obs_w1_false * obs_w2_true / total_ngrams

        expected[1][0] = obs_w1_true * obs_w2_false / total_ngrams

        expected[1][1] = obs_w1_false * obs_w2_false / total_ngrams

        # print(observed)
        # print(expected)

        observed /= sum(observed)
        expected /= sum(expected)

        return scipy.stats.chisquare(observed.flatten(), expected.flatten())

    def compute_statistics(self, n_processes: int):
        def compute_stats_for_partition(k):
            ngrams_generator = self.container.all_ngrams(offset=partition_size * k, limit=partition_size)
            counter = 0
            for i in ngrams_generator:
                print(i)
                stats = self.ngram_chi_square(i)
                self.container.update_ngram(i, stats)

                counter += 1
                # print(f"{counter} statistics calculated")
            print('\n\n')

        partition_size = math.ceil(self.container.counter / n_processes)

        for k in range(n_processes):
            # multiprocessing.Process(target=compute_stats_for_partition, args=(k, )).start()
            compute_stats_for_partition(k)

    def get_statistics(self):
        return self.container.get_statistics()



if __name__ == '__main__':
    # source = 'prestuplenie-i-nakazanie.txt'
    source = 'notebooks/tmp.txt'
    # source = 'pinshort.txt'
    # source = ''


    consumer = CollocationStatisticsConsumerDB()

    nlp = spacy.load('ru_core_news_sm')

    with open(source, 'r', encoding='utf-8') as fp:
        LTP = CollocationLazyTextProcessor(fp, nlp, stopwords.words("russian"))
        consumer.process([LTP])

    consumer.compute_statistics(n_processes=3)
    chi_stat = list(consumer.get_statistics())

    # print(*chi_stat[:10], sep='\n')
