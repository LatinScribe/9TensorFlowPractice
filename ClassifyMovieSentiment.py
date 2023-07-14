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

from tensorflow.keras import layers
from tensorflow.keras import losses

DATASET_PATH = 'Imdb_dataset/aclImdb'
TRAIN_PATH = 'Imdb_dataset/aclImdb/train'

print('TensorFlow version:', tf.__version__)

# Uncomment to download and extract dataset from the web upon first use!!
# NOT reocmmended to do this each time since it is significantly slower
# print('Downloading dataset, please wait...')
# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#
# dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
# DATASET_PATH = os.path.join(os.path.dirname(dataset), 'aclImdb')
# TRAIN_PATH = os.path.join(dataset_dir, 'train')

# test a random sample text file to open
print('The following is a random sample move review from the training dataset:')

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
