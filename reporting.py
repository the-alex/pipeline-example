#!/usr/bin/env python
"""
A module for reporting experimental results.
"""
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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
