# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Flatten
from keras.layers import LSTM
from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class Classifier_lstm():
    def __init__(self):
        model = Sequential()
        model.add(Embedding(700, output_dim=1))
        model.add(LSTM(600))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(200, activation='relu'))

        model.add(Dense(6, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        optimizer = Adam(lr=10**-3, decay=0.01)
        self.clf = model

    def fit(self, X, y):
        y = to_categorical(y, 6)
        self.clf.fit(X, y, batch_size=20, epochs=7, verbose = 1)

    def predict(self, X):
        return self.clf.predict_classes(X)

    def predict_proba(self, X):
        return self.clf.predict(X)