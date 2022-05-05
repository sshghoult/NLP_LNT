import typing
import pymorphy2
import abc
from collections import Counter, deque


class MorphAnalyzerSingleton(pymorphy2.MorphAnalyzer):
    _instances = []

    def __new__(cls, *args, **kwargs):
        if cls._instances:
            return cls._instances[0]
        else:
            ins = super().__new__(cls, *args, **kwargs)
            cls._instances.append(ins)
            return ins


class WordToken:
    """Wrapper for single semantically-complete token.
    Input - str, init - inferred info on token"""
    morph_analyzer = MorphAnalyzerSingleton()

    def __init__(self, token: str):
        self.token: str = token
        self.parse: pymorphy2.analyzer.Parse = self.morph_analyzer.parse(self.token)[0]


    def hash_key(self):
        return self.token

    def __hash__(self):
        return hash(self.hash_key())

    def __eq__(self, other):
        return self.hash_key() == other.hash_key()

    def __repr__(self):
        return str(self.token)

    def __str__(self):
        return self.token


class NGRam:
    """Container for extracted windows of tokens"""

    def __init__(self, n_gram: typing.Iterable[WordToken]):
        self.seq = tuple(n_gram)
        self.pos_mask = tuple(wt.parse.tag.POS for wt in self.seq)

    def hash_key(self):
        return self.seq

    def __hash__(self):
        return hash(self.hash_key())

    def __eq__(self, other):
        return self.hash_key() == other.hash_key()

    def __iter__(self):
        return iter(self.seq)

    def __repr__(self):
        return str(self.seq)

    def __str__(self):
        return str(self.seq)


class AbstractCollector(abc.ABC):
    """Interface of different Collectors"""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get(self, key) -> typing.Iterator:
        pass

    @abc.abstractmethod
    def add(self, key, value):
        pass


class InMemoryCollector(AbstractCollector):
    """Binds instances to its occurrences in other entities in order to achieve O(1) access to
    relevant subset of data to calculate statistics. Only usable for small amount of data since it
    utilizes process's memory"""

    def __init__(self):
        self.dict: typing.Dict[object, typing.Deque[object]] = dict()

    def get(self, key):
        return self.dict[key]

    def add(self, key, value):
        if key not in self.dict:
            self.dict[key] = deque((value,))
        else:
            self.dict[key].append(value)
