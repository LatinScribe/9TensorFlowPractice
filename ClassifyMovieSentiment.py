"""
Author: Henry "TJ" Chen
Last modified: July 13, 2023

This will demonstrate text classification. It trains a binary classifier to
perform sentiment analysis on an IMDB dataset.

Classifies movie reviews as positive or negative based on text of review
(hence binary)
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import random
from DatasetDownloader import get_dataset

from tensorflow.keras import layers
from tensorflow.keras import losses

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

print('TensorFlow version:', tf.__version__)

# Uncomment to manually download and extract dataset from the web
# NOT reocmmended to do this each time since it is significantly slower
# print('Downloading dataset, this may take a while, please wait...')
# DATASET_PATH, TRAIN_PATH = get_dataset(URL)

if os.path.isdir('aclImdb') and os.path.isdir('aclImdb/train') and os.path.isdir('aclImdb/test'):
    print("Dataset detected as downloaded, if issues persist, please manually redownload")
    DATASET_PATH = 'aclImdb'
    TRAIN_PATH = 'aclImdb/test'
else:
    print('No dataset detected')
    print('Downloading dataset, this may take a while, please wait...')
    DATASET_PATH, TRAIN_PATH = get_dataset(URL)

# test a random sample text file to open
print('The following is a random sample movie review from the training dataset:')

# choose a random file
rand_first_num = str(random.randint(0, 12499))
rand_secd_num = str(random.randint(7, 10))

sample_file = os.path.join(TRAIN_PATH, 'pos/' + rand_first_num + '_' + rand_secd_num + '.txt')
while not os.path.isfile(sample_file):
    rand_first_num = str(random.randint(0, 12499))
    rand_secd_num = str(random.randint(7, 10))

    sample_file = os.path.join(TRAIN_PATH, 'pos/' + rand_first_num + '_' + rand_secd_num + '.txt')

# open and print the chosen file
with open(sample_file) as f:
    print(f.read())
