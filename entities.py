import typing
import pymorphy2
import abc
from collections import Counter, deque


class SmartCounter(Counter):
    def __init__(self):
        super().__init__()
        # this actually violates Liskov's principle and shouldn't be used
        self.total = 0


class SimpleCounter:
    def __init__(self):
        self.dict = dict()
        self.total = 0

    def keys(self):
        return self.dict.keys()

    def hit(self, key):
        if key not in self.dict:
            self.dict[key] = 0
        self.dict[key] += 1

        self.total += 1

    def __getitem__(self, key):
        return self.dict[key]


# TODO: right custom counter from scratch that allow only dict-like updates to avoid SOLID violation
# TODO: use this counter as a structure inside the Container for access to total for O(1)


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


class Ngram:
    """Container for extracted windows of tokens, currently only bi-grams"""

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

    def __getitem__(self, index):
        return self.seq[index]

    def __repr__(self):
        return str(self.seq)

    def __str__(self):
        return str(self.seq)


class AbstractCategorizedCollector(abc.ABC):
    """Interface of Collectors"""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get(self, category, key):
        pass

    @abc.abstractmethod
    def add(self, category, key, value):
        pass

    @abc.abstractmethod
    def category_total(self, category):
        pass



class InMemoryCategorizedCollector(AbstractCategorizedCollector):
    """Binds instances to its occurrences in other entities in order to achieve O(1) access to
    relevant subset of data to calculate statistics. Only usable for small amount of data since it
    utilizes process's memory"""

    # TODO: introduce categories

    def __init__(self):
        super().__init__()
        self.dict: typing.Dict[object, typing.Dict[object, SimpleCounter]] = dict()
        self.categorized_totals = SimpleCounter()
        # category: {key: [related objects]}

    def get(self, category, key) -> SimpleCounter:
        return self.dict[category][key]

    def add(self, category, key, value):
        if category not in self.dict:
            self.dict[category] = dict()
            self.categorized_totals.hit(category)

        if key not in self.dict[category]:
            self.dict[category][key] = SimpleCounter()
        self.dict[category][key].hit(value)

    def category_total(self, category):
        return self.categorized_totals[category]


class AbstractDBCollector(abc.ABC):

    @abc.abstractmethod
    def add_ngram(self, ngram: Ngram):
        pass

    @abc.abstractmethod
    def observed_count(self, words: Ngram[WordToken, WordToken],
                     mask: typing.Sequence[bool, None]):
        """Mask marks if word_i must be in position i (True), can not be in position i (False) or
        whether it is not important and should be ignored in query (None)"""
        pass

    @abc.abstractmethod
    def all_ngrams(self) -> typing.Generator[Ngram]:
        # should return a generator that loads ngrams in chunks per requirement, i guess
        pass

    @abc.abstractmethod
    def ngrams_count(self):
        pass






