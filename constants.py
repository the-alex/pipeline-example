#!/usr/bin/env python

RANDOM_STATE = 42

DATA_PATH = './data/'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

TARGET_LABEL = 'survived'

# These columns have no business in MY models.
DROP_COLS = [
    'passengerid',
    'ticket',
    'name',
    'cabin',
]

CACHE_DIR = './cache/'
