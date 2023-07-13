"""
Author: Henry "TJ" Chen
Last modified: July 12, 2023

Build a neural network to classify images of clothing
"""
import keras.losses
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# practice on the Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist

# create a numpy array from the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# clothing types to sort by
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show the first image in the set
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# need to turn the color values to range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# check the first 25 images to make sure it works
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

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
# print(predictions[0])
# print(np.argmax(predictions[0]))

# graph the confidence on the given item
def plot_image(i, predictions_array, true_label, img):
    """This function shows the image for item i"""
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    """This function plots the confidence in each category for item i"""
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_item(i: int):
    """display both plots for item i"""
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


# example, plot the first item
plot_item(0)
