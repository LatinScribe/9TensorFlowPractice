"""
Author: Henry "TJ" Chen
Last modified: July 13, 2023

Functions for plotting images and graphs for the classifying
clothing neural network
"""
import matplotlib.pyplot as plt
import numpy as np


def show_img(i: int, train_images):
    """Displays the image of the i-th item in the training dataset"""
    plt.figure()
    plt.imshow(train_images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_many_img(x: int, train_images, class_names, train_labels):
    """Display the first x number of images in the training dataset"""
    plt.figure(figsize=(10, 10))
    for i in range(x):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


def plot_image(i, predictions_array, true_label, img, class_names):
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


def plot_item(i: int, predictions, test_labels, test_images, class_names):
    """Display both plots for item i"""
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


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


def plot_many(num_rows: int, num_cols: int, predictions, test_labels, test_images, class_names):
    """Plot the first (num_rows * num_col) number of images, their predicted labels, and the true labels.
    Color the correct predictions in blue and incorrect predictions in red.
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def plot_single(predictions_single, test_labels, class_names):
    """Given a batch of only one item, plot the prediciton"""
    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()
