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

from keras import layers
from keras import losses

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

print('TensorFlow version:', tf.__version__)

# Uncomment to manually download and extract dataset from the web
# NOT reocmmended to do this each time since it is significantly slower
# print('Downloading dataset, this may take a while, please wait...')
# DATASET_PATH, TRAIN_PATH = get_dataset(URL)

if os.path.isdir('aclImdb') and os.path.isdir('aclImdb/train') and os.path.isdir('aclImdb/test'):
    print('Dataset detected as downloaded, if issues persist, please manually redownload')
    DATASET_PATH = 'aclImdb'
    TRAIN_PATH = 'aclImdb/train'
    print('Dataset downloaded sucessfully')
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

# remove unnesscary folders
remove_dir = os.path.join(TRAIN_PATH, 'unsup')
if os.path.isdir(remove_dir):
    print('Unesscary dataset folders detected, removing folders...')
    shutil.rmtree(remove_dir)
    print('Removal sucessfully completed')

# Need to create a validation dataset, create one using 80:20 split
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

# see how the training dataset is now structured
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)


# created a custom standardiser that removes HTML as well
def custom_standardization(input_data):
    """Standize the inputted data by converting all text to lower case
    Also tokenizes the text and strips HTML tags
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)


def seperate_first(x, _):
    """takes in x and y, returns only x"""
    return x


# makes a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(seperate_first)
vectorize_layer.adapt(train_text)


# function to see how vectorization of layer works
def vectorize_text(text, label):
    """Returns vectorized text"""
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))


# check what each token corresponds to
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# apply the TextVectorization layer on all components of dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# configuring dataset for best performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model now
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.summary()

# Loss function and optimizer
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
              )

# train the model
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# check the accuracy
loss, accuracy = model.evaluate(test_ds)

print('Loss: ', loss)
print('Accuracy: ', accuracy)

# we can do better - History object contains data of training
history_dict = history.history

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# plot the losses
# 'bo' stands for blue dot
plt.figure('Training and validation loss')
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for 'solid blue line'
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# plot the accuracy
plt.figure('Training and validation accuracy')
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# export the model
# create a new model using the weights already trained
# for export where the model can process raw strings by
# including TextVectorization layer inside model
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

# Test it with 'raw_test_ds', which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# get prediction on new data
# example reviews
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

# get prediction on new reviews
print(export_model.predict(examples))
