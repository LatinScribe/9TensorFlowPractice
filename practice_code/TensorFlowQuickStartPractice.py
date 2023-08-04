"""
Author: Henry "TJ" Chen
Last modified: July 12, 2023

Testing out TensorFlow!

This will load a prebuilt dataset, build a neural network machine learning model that
classifies images. It will train the neural network and evaluate the accuracy of the model
"""

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build a sequencial model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print('These are a vector of logits or log-odds scores')
print(predictions)

print('These are the convereted probabilities')
print(tf.nn.softmax(predictions).numpy())

# define a loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print('Probability of loss - expect ~= 2.3')
print(loss_fn(y_train[:1], predictions).numpy())

# before training, need to configure and compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=5)

print("WhooHoo! The image classifier should now be trained to ~98% accuracy on this dataset")
