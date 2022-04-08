from spacy.lang.ru import Russian
import spacy
import pymorphy2
import string

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
                if token.lemma_ in forbidden:
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


nlp = spacy.load('ru_core_news_sm')

with open('notebooks/tmp.txt', 'r', encoding='utf-8') as fp:
    LTP = CollocationLazyTextProcessor(fp, nlp, stopwords.words("russian"))
    for k in LTP:
        print(k)
