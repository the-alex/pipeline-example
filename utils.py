"""
A collection of utility functions to help ease the repetition in notebooks and
model scripts.
"""
import pandas as pd
import numpy as np
import constants as c

np.random.seed(c.RANDOM_STATE)

def fetch_data():
    data = pd.read_csv(c.DATA_PATH + c.TRAIN_FILENAME)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return data


def run_classifiers(clfs, X, y, cv):
    """Simple experiment which returns the result of a cross validation
    strategy, provided as an argument."""
    reports = dict()

    for name, clf in clfs:
        reports[name] = list()

        for train_inds, test_inds in cv.split(X):
            X_train = X[train_inds]
            X_test = X[test_inds]
            y_train, y_test = y[train_inds], y[test_inds]

            clf.fit(X_train, y_train)

            y_hat = clf.predict(X_test)
            y_hat_probs = clf.predict_proba(X_test)

            report[name].append(make_report(y_hat, y_hat_probs, y_test))

    return reports

