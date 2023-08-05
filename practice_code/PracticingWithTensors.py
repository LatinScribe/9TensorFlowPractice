"""
Author: Henry "TJ" Chen
Last Modified: July 7, 2023

File for practicing with tensors

Following the Google Guide
"""
import tensorflow as tf
import numpy as np

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

print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.math.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

#  ---------   Tensor shapes ---------

rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())


# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())

# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# rank 3
print(rank_3_tensor[:, :, 4])

t2 = tf.constant([[0, 1, 2, 3, 4],
                  [5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]])

print(t2[:-1, 1:3])

t3 = tf.constant([[[1, 3, 5, 7],
                   [9, 11, 13, 15]],
                  [[17, 19, 21, 23],
                   [25, 27, 29, 31]]
                  ])

print(tf.slice(t3,
               begin=[1, 1, 0],
               size=[1, 1, 2]))

t5 = np.reshape(np.arange(18), [2, 3, 3])

print(tf.gather_nd(t5,
                   indices=[[0, 0, 0], [1, 2, 1]]))

# Return a list of two matrices

print(tf.gather_nd(t5,
                   indices=[[[0, 0], [0, 2]], [[1, 0], [1, 2]]]))

# Return one matrix

print(tf.gather_nd(t5,
                   indices=[[0, 0], [0, 2], [1, 0], [1, 2]]))

#  ------ inserting data into tensors -----

t6 = tf.constant([10])
indices = tf.constant([[1], [3], [5], [7], [9]])
data = tf.constant([2, 4, 6, 8, 10])

print(tf.scatter_nd(indices=indices,
                    updates=data,
                    shape=t6))

# Gather values from one tensor by specifying indices

new_indices = tf.constant([[0, 2], [2, 1], [3, 3]])
t7 = tf.gather_nd(t2, indices=new_indices)

# note, this is for a tensor with no pre-existing values
# Add these values into a new tensor

t8 = tf.scatter_nd(indices=new_indices, updates=t7, shape=tf.constant([4, 5]))

print(t8)

# similar to
t9 = tf.SparseTensor(indices=[[0, 2], [2, 1], [3, 3]],
                     values=[2, 11, 18],
                     dense_shape=[4, 5])

print(t9)

# Convert the sparse tensor into a dense tensor

t10 = tf.sparse.to_dense(t9)

print(t10)

t11 = tf.constant([[2, 7, 0],
                   [9, 0, 1],
                   [0, 3, 8]])

# inserting into a tensor with pre-existing values
# Convert the tensor into a magic square by inserting numbers at appropriate indices

t12 = tf.tensor_scatter_nd_add(t11,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[6, 5, 4])

print(t12)

# subtraction
# Convert the tensor into an identity matrix

t13 = tf.tensor_scatter_nd_sub(t11,
                               indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]],
                               updates=[1, 7, 9, -1, 1, 3, 7])

print(t13)

t14 = tf.constant([[-2, -7, 0],
                   [-9, 0, 1],
                   [0, -3, -8]])

# copy only the min value
t15 = tf.tensor_scatter_nd_min(t14,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[-6, -5, -4])

print(t15)

# copy only the max value
t16 = tf.tensor_scatter_nd_max(t14,
                               indices=[[0, 2], [1, 1], [2, 0]],
                               updates=[6, 5, 4])

print(t16)

t16 = tf.tensor_scatter_nd_max(t14,
                               indices=[[0, 2], [1, 1], [2, 0], [1, 2]],
                               updates=[6, 5, 4, 0])
print(t16)

t16 = tf.tensor_scatter_nd_max(t14,
                               indices=[[0, 2], [1, 1], [2, 0], [1, 2]],
                               updates=[6, 5, 4, 2])
print(t16)
