"""Practicing With Variables in Tensorflow

This file contains practice code

Based off guide by Google: https://www.tensorflow.org
"""
import tensorflow as tf

# Uncomment to see where your variables get placed
# tf.debugging.set_log_device_placement(True)

my_tensor = tf.constant([[1., 2.], [3., 4.]])
my_variable = tf.Variable(my_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())

print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.math.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1, 4]))

# rassigning a variable

a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable:
try:
    a.assign([1.0, 2.0, 3.0])
except Exception as e:
    print(f"{type(e).__name__}: {e}")

# Creating new variables from existing variables duplicates the backing tensors.
# Two variables will not share the same memory.

a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]


# Create a and b; they will have the same name but will be backed by
# different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)

# Although variables are important for differentiation, some variables will not need to be differentiated.
# You can turn off gradients for a variable by setting trainable to false at creation.
# An example of a variable that would not need gradients is a training step counter.
step_counter = tf.Variable(1, trainable=False)

# Placing variables and tensors
with tf.device('CPU:0'):

    # Create some tensors
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print(c)

# multiple workers but only want one copy of the variables
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
    # Element-wise multiply
    k = a * b

print(k)
