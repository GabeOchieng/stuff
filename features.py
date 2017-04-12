from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import numpy as np
from scipy import sparse
import array
from collections import deque
from itertools import compress
import nltk
from gensim.models.word2vec import Word2Vec

tokenizer = nltk.TreebankWordTokenizer()


def iterize(return_type=list):
    def decorator(f):
        def f_iter(*args, **kwargs):
            if args and isinstance(args[0], list):
                return return_type([f_iter(arg, *args[1:], **kwargs) for arg in args[0]])
            else:
                return f(*args, **kwargs)
        return f_iter
    return decorator


@iterize()
def tokenize(doc):
    return tokenizer.tokenize(doc.lower())


class Dictionary:
    def __init__(self, docs=None, stopwords=None, num_words=None, archive=False):

        self.archive = archive  # archive is necessary for doing stuff like skip-gram sets
        self.word2id = {'UNK': 0}
        self.id2word = ['UNK']
        self.doc_ids = None
        self.word_ids = None
        self.word2docfreq = {'UNK': 1}
        self.word2freq = {'UNK': 1}
        self.num_docs = 0
        self.num_words = num_words
        self.pca = None
        if stopwords is None:
            self.stopwords = set()
        elif stopwords == 'english':
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        else:
            self.stopwords = set(stopwords)

        if docs is not None:
            self.fit(docs)
        self.index = 0

    def fit(self, docs, archive=None):
        if archive is not None:
            self.archive = archive
        self.word2docfreq['UNK'] = len(docs)
        assert docs, "argument to Dictionary.fit must not be empty"
        assert isinstance(docs[0], list) and docs[0] and isinstance(docs[0][0], str),\
            "argument to Dictionary.fit must be a list of nonempty lists of tokens"
        for doc in docs:
            self.num_docs += 1
            doc_words = set()
            for word in doc:
                if word not in self.stopwords:
                    self.word2freq[word] = self.word2freq.get(word, 0) + 1
                    doc_words.add(word)
            for word in doc_words:
                self.word2docfreq[word] = self.word2docfreq.get(word, 0) + 1
        unk_freq = self.word2freq['UNK']
        del self.word2freq['UNK']
        self.id2word = sorted(self.word2freq.keys(), key=lambda x: -self.word2freq[x])
        self.id2word.insert(0, 'UNK')
        self.word2freq['UNK'] = unk_freq
        self.word2id = dict([(word, i) for i, word in enumerate(self.id2word)])
        #self.num_words = len(self.id2word)
        if self.archive:
            self.doc_ids = list()
            self.word_ids = list()
            for i, doc in enumerate(docs):
                for word in doc:
                    if word in self.word2id:
                        self.doc_ids.append(i)
                        self.word_ids.append(self.word2id[word])
        self.prune()
        return self

    def prune(self, num_words=None, min_freq=None):
        if num_words is None:
            if self.num_words is None:
                self.num_words = len(self.id2word)
                num_words = self.num_words
            else:
                num_words = self.num_words
        if min_freq is not None:
            for i in range(self.num_words - 1, -1, -1):
                if self.word2freq[self.id2word[i]] >= min_freq:
                    num_words = min(num_words, i)
                    break
        current_num_words = len(self.id2word)
        if num_words < current_num_words:
            delete_ids = set(range(current_num_words)[num_words:])
            if self.archive:
                for i, word in enumerate(self.word_ids):
                    if word in delete_ids:
                        self.word_ids[i] = 0
            for i in delete_ids:
                word = self.id2word[i]
                self.id2word[i] = None
                self.word2freq['UNK'] += self.word2freq[word]
                del self.word2freq[word]
                del self.word2id[word]
                del self.word2docfreq[word]
            self.id2word = [w for w in self.id2word if w is not None]
            self.id2word = self.id2word[:num_words]
            self.num_words = num_words

        assert len(self.id2word) == len(self.word2id) == len(self.word2docfreq) == self.num_words == num_words
        return self

    def generate_batch(self, batch_size, window_size, mode='pv'):
        assert batch_size % window_size == 0, 'batch_size must be a multiple of window_size'
        if mode == 'pv':
            batch = np.zeros((batch_size, window_size + 1), dtype=np.int32)
            labels = np.zeros((batch_size, 1), dtype=np.int32)
            span = window_size + 1
            buffer_words = deque()
            buffer_docs = deque()
        for i in range(span):
            buffer_words.append(self.word_ids[self.index])
            buffer_docs.append(self.doc_ids[self.index])
            self.index += 1
            self.index %= len(self.word_ids)

        mask = [1] * span
        mask[-1] = 0
        for i in range(batch_size):
            if len(set(buffer_docs)) == 1:
                doc_id = buffer_docs[0]
                batch[i, :] = list(compress(buffer_words, mask)) + [doc_id]
                labels[i, 0] = buffer_words[-1]
            buffer_words.append(self.word_ids[self.index])
            buffer_docs.append(self.doc_ids[self.index])
            self.index += 1
            self.index %= len(self.word_ids)
        return batch, labels


class BagOfWords(Dictionary, TransformerMixin):
    sparse = True

    def transform(self, doc):
        return self._process(self._raw_transform(doc))

    def _raw_transform(self, doc, parallel=True):
        if doc and isinstance(doc[0], list):
            X = self._stack([self._raw_transform(d) for d in doc])
        else:
            doc = [self.word2id[word] if word in self.word2id else 0 for word in doc]
            X = self._transform(doc)
        return X

    def _process(self, X, norm=True, pca=True):
        if norm:
            X = normalize(X.astype(np.float), copy=False)
        if pca and hasattr(self.pca, 'components_'):
            X = self.pca.transform(X)
        return X

    def _transform(self, doc):
        # This is borrowed from sklearn's CountVectorizer
        j_indices = []
        indptr = array.array("i")
        values = array.array("i")
        indptr.append(0)
        feature_counter = dict()
        for word in doc:
            feature_counter[word] = feature_counter.get(word, 0) + 1

        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)

        vec = sparse.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, self._get_raw_d()),
                                dtype=np.int32)
        return vec

    def _stack(self, list_of_vecs):
        """
        Does the appropriate stacking method.
        (i.e. scipy.sparse.vstack if sparse or np.vstack if dense)
        """
        if self.sparse:
            return sparse.vstack(list_of_vecs)
        else:
            return np.vstack(list_of_vecs)

    def _get_raw_d(self):
        if getattr(self.pca, 'components_', False) is not False:
            if getattr(self, 'word_matrix', False) is False:
                return self.pca.components_.shape[1]
        return self.num_words

    def _fit_pca(self, docs):
        if self.pca is not None and self.pca.get_params()['n_components'] < min(self.num_words, self.num_docs):
            X = self.transform(docs)
            self.pca.fit(self.transform(docs))
        else:
            self.pca = None


class Tfidf(BagOfWords):
    def __init__(self, docs=None, stopwords=None, pca_d=None, size=None, archive=False):
        self.pca = None
        self.pca_d = pca_d
        self.idf = None
        super(Tfidf, self).__init__(docs, stopwords, size, archive)

    def fit(self, docs, archive=None):

        if self.pca_d is None:
            self.pca = None
        else:
            if self.sparse:
                self.pca = TruncatedSVD(self.pca_d)
            else:
                self.pca = PCA(self.pca_d)
        super(Tfidf, self).fit(docs, archive)
        self.idf = np.zeros(self.num_words)
        for i, word in enumerate(self.id2word):
            self.idf[i] = (1 + np.log(self.num_docs / max(self.word2docfreq[word], 1)))
        self.idf = sparse.diags(self.idf)
        self._fit_pca(docs)
        return self

    def prune(self, num_words=None, min_freq=None):
        super(Tfidf, self).prune(num_words, min_freq)
        if self.idf is not None:
            self.idf = sparse.diags(self.idf.data[0, :self._get_raw_d()])
        return self

    def _transform(self, doc):
        vec = super(Tfidf, self)._transform(doc)
        vec *= self.idf
        return vec


class Text2Vec(Tfidf):
    def __init__(self, docs=None, stopwords=None, pca_d=None, size=None, archive=False, word2vec=None, d=32):
        self.word_matrix = None
        self.word2vec = word2vec
        if self.word2vec is None:
            self.d = d
        else:
            self.d = self.word2vec.layer1_size
        if pca_d is not None and pca_d >= self.d:
            pca_d = None
        super(Text2Vec, self).__init__(docs, stopwords, pca_d, size, archive)
        self.sparse = False

    def fit(self, docs, archive=None):

        if self.word2vec is None:
            self.word2vec = Word2Vec(docs, size=self.d, iter=1, min_count=1)

        super(Text2Vec, self).fit(docs, archive)
        self.word_matrix = np.zeros((self.num_words, self.d))
        for i, word in enumerate(self.id2word):
            if word in self.word2vec:
                self.word_matrix[i, :] = self.word2vec[word]
        del self.word2vec
        return self

    def prune(self, num_words=None, min_freq=None):
        super(Text2Vec, self).prune(num_words, min_freq)
        if self.word_matrix is not None:
            self.word_matrix = self.word_matrix[:self._get_raw_d(), :]
        return self

    def _transform(self, doc):
        if self.word_matrix is not None:
            vec = np.ravel(super(Text2Vec, self)._transform(doc) * self.word_matrix)
        else:
            vec = np.zeros(self.d)
            for word in doc:
                if self.id2word[word] in self.word2vec:
                    vec += self.word2vec[self.id2word[word]] * self.idf.data[0][word]
        return vec


def _test():
    TEST_CORPUS = ['this is a test',
                   'This should show the bag of words and text2vec representations of these sentences',
                   'It is just a bunch of random sentences',
                   'I like candy']
    TEST_CORPUS = tokenize(TEST_CORPUS)
    print("tokenizing works...")
    d = Dictionary()
    d.fit(TEST_CORPUS)
    d.prune(5)
    assert d.num_words == 5
    print("pruning dictionary works...")
    print("dictionary fit works...")
    d = BagOfWords()
    d.fit(TEST_CORPUS)
    d.transform(TEST_CORPUS)
    print("dictionary transform works...")
    d.prune(5)
    print("pruning dictionary works...")
    X = d.transform(TEST_CORPUS)
    assert X.shape == (4, 5)
    print("BagOfWords works...")

    TEST_CORPUS = ['this is a test',
                   'This should show the bag of words and text2vec representations of these sentences',
                   'It is just a bunch of random sentences',
                   'I like candy']
    TEST_CORPUS = tokenize(TEST_CORPUS)
    d = Tfidf()
    d.fit(TEST_CORPUS)
    d.transform(TEST_CORPUS)
    print("Tfidf transform works...")
    d.prune(5)
    print("pruning Tfidf works...")
    X = d.transform(TEST_CORPUS)
    assert X.shape == (4, 5)
    X = Tfidf(pca_d=3).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("Tfidf with pca works...")
    X = Tfidf().fit_transform(TEST_CORPUS)
    assert X.shape[0] == 4
    X = Tfidf(pca_d=3).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("Tfidf works...")

    t = Text2Vec()
    t.fit(TEST_CORPUS)
    print("Text2Vec fit works...")
    X = t.transform(TEST_CORPUS)
    assert X.shape[0] == 4
    print("Text2Vec transform works...")
    X = Text2Vec(pca_d=3).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("Text2Vec with pca works...")
    X = Text2Vec().fit_transform(TEST_CORPUS)
    assert X.shape[0] == 4
    X = Text2Vec(pca_d=3).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("Text2Vec works...\n")


if __name__ == '__main__':
    _test()
