"""
Author: Henry "TJ" Chen
Last modified: July 13, 2023

Build a neural network to classify images of clothing

To try it out, simply run this file!
"""
import keras.losses
import tensorflow as tf
import numpy as np
import random

from Plotting import show_img, show_many_img, plot_item, plot_many, plot_single


print('TensorFlow version:', tf.__version__)

# practice on the Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist

# create a numpy array from the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# clothing types to sort by
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# show the first image in the set
show_img(0, train_images)

# need to turn the color values to range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# check the first 25 images to make sure it works
show_many_img(25, train_images, class_names, train_labels)

# set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# configure the settings for compiling - what we are testing for
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model!
print('Training model, please wait...')
model.fit(train_images, train_labels, epochs=10)

# now to check the accuracy, compare the model to the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# convert the linear output to probabilities
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# save the prediction data for testing
predictions = probability_model.predict(test_images)


# check confidence on the first item
print(predictions[0])
print(np.argmax(predictions[0]))

# example, plot the first item
plot_item(0, predictions, test_labels, test_images, class_names)


# example, plot the first 15 item
plot_many(5, 3, predictions, test_labels, test_images, class_names)

# Finally, let's actually use the trained model!
# Pick an image from the test dataset
random_num = random.randint(0, 9999)
img = test_images[random_num]

# print(img.shape)
show_img(random_num, test_images)

# since keras models are optimized on batches, we create a batch with this one item
img = (np.expand_dims(img, 0))
# print(img.shape)

# now predict the correct label for this image:
predictions_single = probability_model.predict(img)
print(predictions_single)

# plot the predition
plot_single(random_num, predictions_single, test_labels, class_names)

# get the actual prediction
prediction_num = np.argmax(predictions_single[0])
print('The prediction is: ', prediction_num)
