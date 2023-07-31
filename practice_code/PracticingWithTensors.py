"""
Author: Henry "TJ" Chen
Last Modified: July 7, 2023

File for practicing with tensors
"""
import tensorflow as tf

# this is an int32 tensor by default
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# Make a rank 1 tensor with float
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# Matrix - rank 2 tensor has two axes
# You can set the dtype at creation
rank_2_tensor = tf.constant([[1, 2],
                            [3, 4],
                            [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]], ])

print(rank_3_tensor)

# basic operations on tensors
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2], dtype=tf.int32)`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
