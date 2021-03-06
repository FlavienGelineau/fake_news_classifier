# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from datetime import timedelta

import submissions.starting_kit.feature_extractor as feat_extr
import submissions.starting_kit.classifier as classifier

problem_title = 'Fake news: classify statements of public figures'
_target_column_name = 'truth'
_prediction_label_names = [0, 1, 2, 3, 4, 5]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

soft_score_matrix = np.array([
    [1, 0.8, 0, 0, 0, 0],
    [0.4, 1, 0.4, 0, 0, 0],
    [0, 0.4, 1, 0.4, 0, 0],
    [0, 0, 0.4, 1, 0.4, 0],
    [0, 0, 0, 0.4, 1, 0.4],
    [0, 0, 0, 0, 0.8, 1],
])

true_false_score_matrix = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

score_types = [
    rw.score_types.SoftAccuracy(
        name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.SoftAccuracy(
        name='tfacc', score_matrix=true_false_score_matrix, precision=3),
]


def get_cv(X, y):
    """Slice folds by equal date intervals."""
    date = pd.to_datetime(X['date'])
    n_days = (date.max() - date.min()).days
    n_splits = 8
    fold_length = n_days // n_splits
    fold_dates = [date.min() + timedelta(days=i * fold_length)
                  for i in range(n_splits + 1)]
    for i in range(n_splits):
        test_is = (date >= fold_dates[i]) & (date < fold_dates[i + 1])
        train_is = ~test_is
        yield np.arange(len(date))[train_is], np.arange(len(date))[test_is]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep='\t')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df, y_array
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


X_train_unprocessed, Y_train = get_train_data()
X_test_unprocessed, Y_test = get_test_data()
ext = feat_extr.FeatureExtractor()
X_train_processed = ext.fit_transform(X_train_unprocessed)
X_test_processed = ext.fit_transform(X_test_unprocessed)

model = classifier.Classifier()

model.fit(X_train_processed, Y_train)
print(X_train_processed.shape, X_test_processed.shape)
Y_pred = model.predict(X_train_processed)

