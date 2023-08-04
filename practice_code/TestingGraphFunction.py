"""
Author: Henry "TJ" Chen

Last Modified: July 27, 2023

File for messing around
"""

import tensorflow as tf


# using tf.function to seperate pure-TensorFlow code from python
@tf.function
def my_func(x):
    """Example function"""
    print('Tracing.\n')
    return tf.reduce_sum(x)


# quick demo of a tf.Module Object
class MyModule(tf.Module):
  def __init__(self, value):
    self.weight = tf.Variable(value)

  @tf.function
  def multiply(self, x):
    return x * self.weight


if __name__ == '__main__':
    # The first time you run the tf.function, although it executes
    # in Python, it captures a complete, optimized graph representing the TensorFlow computations
    # done within the function.
    x = tf.constant([1, 2, 3])
    my_func(x)

    # On subsequent calls TensorFlow only executes the optimized graph, skipping any non-TensorFlow steps.
    # Below, note that my_func doesn't print tracing since print is a Python function, not a TensorFlow function.
    x = tf.constant([10, 9, 8])
    my_func(x)

    # A graph may not be reusable for inputs with a different signature (shape and dtype), so a new graph
    # is generated instead:
    x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
    my_func(x)

    # use the module
    mod = MyModule(3)
    mod.multiply(tf.constant([1, 2, 3]))

    # save the module
    save_path = './demo_saved'
    tf.saved_model.save(mod, save_path)

    # load the saved module
    reload = tf.saved_model.load(save_path)
    reload.multiply(tf.constant([1, 2, 3]))
