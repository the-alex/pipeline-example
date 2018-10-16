"""
A collection of utility functions to help ease the repetition in notebooks and
model scripts.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def fetch_data():
    X, y = make_classification(n_samples=2000, n_features=10)
    return pd.DataFrame(X), pd.DataFrame(y)
