"""
A collection of utility functions to help ease the repetition in notebooks and
model scripts.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import make_pipeline
from constants import *


def fetch_data():
    data = pd.read_csv(DATA_PATH + TRAIN_FILENAME)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    data.dropna(inplace=True)
    return data


def transform(X, transformers=dict()):
    ## Numerical Features
    ## Datetime Features
    ## Text Features

    ## Categorical Features
    X = pd.get_dummies(X, columns=['embarked'])
    
    ## Boolean Features
    X['sex'] = X.sex == 'male'

    return X


def make_report(y_pred, y_probs, y_true):
    report = dict(
        accuracy=accuracy_score(y_true, y_pred),
        auroc=roc_auc_score(y_true, y_probs),
        f1_score=f1_score(y_true, y_pred),
    )
    return report


def pretty_print_report(report):
    return """
Accuracy: {accuracy}
AUROC   : {auroc}
F1      : {f1_score}
""".format(**report)
