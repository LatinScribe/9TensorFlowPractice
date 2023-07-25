"""
Author: Henry "TJ" Chen
Last modified: July 24, 2023

This will demonstrate text classification. It trains a multi-class clasifier
to predict the tag of a stack overflow question

Classifies stack overflow questions based on whether they are
Python, Java, CSharp, or JavaScript
"""

import os
import re
import shutil
import string
import tensorflow as tf
import random
from DatasetDownloader import get_dataset
from Plotting import plot_loss, plot_accuracy

from keras import layers
from keras import losses

URL = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
TAGS = ['Python', 'Java', 'Csharp', 'JavaScript']

print('TensorFlow version:', tf.__version__)

# Uncomment to manually download and extract dataset from the web
# NOT reocmmended to do this each time since it is significantly slower
# print('Downloading dataset, this may take a while, please wait...')
# DATASET_PATH, TRAIN_PATH, TEST_PATH = get_dataset(URL, 'stack_overflow_16k')
# print('Dataset downloaded sucessfully')
# print('Dataset-path: ', DATASET_PATH)
# print('Train-path: ', TRAIN_PATH)
# print('Test-path: ', TEST_PATH)

if os.path.isdir('train') and os.path.isdir('test'):
    print('\nDataset detected as downloaded, if issues persist, please manually redownload')
    TRAIN_PATH = 'train'
    TEST_PATH = 'test'
else:
    print('\nNo dataset detected')
    print('Downloading dataset, this may take a while, please wait...')
    DATASET_PATH, TRAIN_PATH, TEST_PATH = get_dataset(URL, 'stack_overflow_16k')
    print('Dataset downloaded sucessfully')
    print('Dataset-path: ', DATASET_PATH)
    print('Train-path: ', TRAIN_PATH)
    print('Test-path: ', TEST_PATH)

# test a random sample text file to open
print('\nThe following is a random sample stackoverflow post from the training dataset:')

# choose a random file
rand_first_num = str(random.randint(0, 1999))
rand_type = random.choice(TAGS)
rand_tag = rand_type.lower()

sample_file = os.path.join(TRAIN_PATH, rand_tag + '\\' + rand_tag + rand_first_num + '.txt')

while not os.path.isfile(sample_file):
    rand_first_num = str(random.randint(0, 1999))
    rand_type = random.choice(TAGS)
    rand_tag = rand_type.lower()

    sample_file = os.path.join(TRAIN_PATH, rand_tag + '\\' + rand_first_num + '.txt')

# open and print the chosen file
with open(sample_file) as f:
    print(f.read())

# Need to create a validation dataset, create one using 80:20 split
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    TRAIN_PATH,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

# see how the training dataset is now structured
print('\nThe following shows how our input data has been structured')
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    TRAIN_PATH,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    TEST_PATH,
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
print('\nThe following shows a movie review post processing')
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
    layers.Dense(4)
])

model.summary()

# Loss function and optimizer
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy']
              )

# train the model
epochs = 10

print('\nTraining model, please wait...')
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

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

print('\nGenerating plots of training history')
# plot the losses
plot_loss(loss, val_loss, epochs)

# plot the accuracy
plot_accuracy(acc, val_acc, epochs)

# export the model
# create a new model using the weights already trained
# for export where the model can process raw strings by
# including TextVectorization layer inside model
print('\nCreating export model...')
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# Test it with 'raw_test_ds', which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print('\nExport model accuracy on raw strings: ', accuracy)

# get prediction on new data
# example reviews
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]
print('\nSample reviews: ', examples)

# get prediction on new reviews
print('\nAccuracy of trained model on sample reviews: ', export_model.predict(examples))
