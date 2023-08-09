"""
Author: Henry "TJ" Chen
Last Modified: July 13, 2023

This file contains the functions nesscary for downloading and extracting the given dataset
"""

import tensorflow as tf
import os


def get_dataset_movie(url: str) -> tuple[str, str]:
    """Downloads and extracts the dataset from the given source URL"""
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
    dataset_path = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_path = os.path.join(dataset_path, 'train')
    return (dataset_path, train_path)


def get_dataset(url: str, file_name: str, save_dir: str) -> tuple[str, str, str]:
    """Downloads and extracts the dataset from the given source URL
    Saves the file with the given file name

    Returns the directory path of dataset, train_folder, and test_folder
    """
    dataset = tf.keras.utils.get_file(file_name, url, untar=True, cache_dir=save_dir, cache_subdir='')
    dataset_path = os.path.dirname(dataset)
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    return (dataset_path, train_path, test_path)
