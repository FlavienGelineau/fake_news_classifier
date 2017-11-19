# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import string
import unicodedata

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_str(sentence, stem=True):
    english_stopwords = set(
        [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    if stem:
        stemmer = SnowballStemmer('english')
        return list((filter(lambda x: x.lower() not in english_stopwords and
                                      x.lower() not in punctuation,
                            [stemmer.stem(t.lower())
                             for t in word_tokenize(sentence)
                             if t.isalpha()])))

    return list((filter(lambda x: x.lower() not in english_stopwords and
                                  x.lower() not in punctuation,
                        [t.lower() for t in word_tokenize(sentence)
                         if t.isalpha()])))


def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features. """

    def __init__(self):
        super(FeatureExtractor, self).__init__(
            input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None, lowercase=True,
            preprocessor=None, tokenizer=None, analyzer='word',
            stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1), max_df=1.0, min_df=1,
            max_features=None, vocabulary=None, binary=False,
            dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
            sublinear_tf=False)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self._feat = np.array([' '.join(
            clean_str(strip_accents_unicode(dd)))
            for dd in X_df.statement])
        self.sources_seen = list(set(source for source in X_df.source))
        self.researched_by_seen = list(set(researched_by for researched_by in X_df.researched_by))
        self.state_seen = list(set(state for state in X_df.state))

        super(FeatureExtractor, self).fit(self._feat)
        return self

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(X_df)

    def transform(self, X_df):
        return self.transform_source(X_df)

    def transform_statement(self, X_df):
        X = np.array([' '.join(clean_str(strip_accents_unicode(dd)))
                      for dd in X_df.statement])
        check_is_fitted(self, '_feat', 'The tfidf vector is not fitted')
        X = super(FeatureExtractor, self).transform(X)
        return X.todense()

    def get_number_for_elt(self, source, data_source):
        try:
            return self.data_source.index(source)
        except:
            return -1

    def transform_source(self, X_df):
        print(X_df)
        X_source = [
            [self.get_number_for_elt(source, self.sources_seen)] for source in
                      X_df.source]
        X_researched_by = [
            [self.get_number_for_elt(researched_by, self.researched_by_seen)] for researched_by in
                    X_df.researched_by]
        X_state = [
            [self.get_number_for_elt(state, self.state_seen)] for state in
                    X_df.state]

        fusion = [source+researched_by+state for source, researched_by, state in zip(X_source, X_researched_by, X_state)]
        return np.array(fusion)

