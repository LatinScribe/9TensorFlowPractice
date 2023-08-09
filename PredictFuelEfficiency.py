"""
Author: Henry "TJ" Chen
Last modified: July 26, 2023

Neural network which predicts the fuel economy of 70s and 80s
automobiles based on attributes such as cylinder count, displacemenent,
weight, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make Numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from keras import layers

print(tf.__version__)

# get the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# get rid of unkown values
dataset.isna().sum()
dataset = dataset.dropna()

# switch original catigorical data to numeric
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# split data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# plotting the data to look for dependencies
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

train_dataset.describe().transpose()

# seperate the features frpm the labels (we want to predict label)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# show just mean and std
train_dataset.describe().transpose()[['mean', 'std']]

# since values are everywhere, we need to normalize!
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

# Single variable regression (use horsepower to predict mpg)
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

# output the visualisation of the training history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(given_history):
    """Plot the training history"""
    plt.plot(given_history.history['loss'], label='loss')
    plt.plot(given_history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


plot_loss(history)

# Collect test set results
test_results = {'horsepower_model': horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)}

# visualise prediction of model as a function of the input
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
    """Plots the prediction as a function of the input"""
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()


plot_horsepower(x, y)

# linear regression with multiple variables
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

# linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# plot the new multivalraible model results, note that the error is much lower
plot_loss(history)

# collect the results on the test set
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

# Regression with a deep neural network
# model is expanded to include some "hidden" non-linear layers

def build_and_compile_model(norm):
    """Build and compile the DNN model"""
    model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# DNN model with a single input
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

dnn_horsepower_model.summary()

# train the model
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

# model does slightly better than the original single input model
plot_loss(history)

# plot the preidction vs data
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

# notice how the hidden layer provides nonlinearity which is an advantage in this case
plot_horsepower(x, y)

# collect the test results
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)

# DNN with multiple inputs
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# train
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

# finally collect the test results
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# Time to evaluate the performance of each model!!
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

# You can see that the DNN multivariable model was the most effective!!

# making predictions with the model
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# from the plot, you can see that the model is fairly accurate

# check error distribution
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

# time to save the model for later use!
dnn_model.save('dnn_model')

# reloading the model gives identical output
reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
