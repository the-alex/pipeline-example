#!/usr/bin/env python

DATA_PATH = './data/'
TRAIN_FILENAME = 'train.csv'

TARGET_LABEL = 'survived'

# These columns have no business in MY models.
DROP_COLS = [
    'passengerid',
    'parch',
    'ticket',
    'name',
    'sibsp',
    'cabin',
]

FEATURES = ['sex', 'age', 'fare', 'embarked_C', 'embarked_Q', 'embarked_S']
