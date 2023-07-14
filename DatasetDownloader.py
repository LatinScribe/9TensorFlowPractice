"""
Author: Henry "TJ" Chen
Last Modified: July 13, 2023

This file contains the functions nesscary for downloading and extracting the given dataset
"""

import tensorflow as tf
import os


def get_dataset(url: str) -> tuple[str, str]:
    """Downloads and extracts the dataset from the given source URL"""
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
    dataset_path = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_path = os.path.join(dataset_path, 'train')
    return (dataset_path, train_path)
