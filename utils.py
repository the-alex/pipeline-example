"""
A collection of utility functions to help ease the repetition in notebooks and
model scripts.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def fetch_data():
    X, y = make_classification(n_samples=2000, n_features=10)
    return pd.DataFrame(X), pd.DataFrame(y)


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
