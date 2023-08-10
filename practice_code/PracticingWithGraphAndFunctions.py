"""Practicing with Graphs and Functions in Tensorflow

Author: Henry "TJ" Chen
Last Modified: August 10, 2023

Based off Google guide: https://www.tensorflow.org/
"""
import tensorflow as tf
import timeit
# from datetime import datetime

# You create and run a graph in TensorFlow by using tf.function, either as a direct call or as a decorator.
# Takes a regular function as input and returns a function


# Define a Python function.
def a_regular_function(x, y, b):
    """Example regular python function"""
    x = tf.matmul(x, y)
    x = x + b
    return x


# `a_function_that_uses_a_graph` is a TensorFlow `Function`.
a_function_that_uses_a_graph = tf.function(a_regular_function)

# Make some tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# Call a `Function` like a Python function.
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert(orig_value == tf_function_value)


# tf.function applies to a function and all other functions it calls:
def inner_function(x, y, b):
    """Example inner function"""
    x = tf.matmul(x, y)
    x = x + b
    return x


# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
    """Example outer function"""
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)

    return inner_function(x, y, b)


# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
outer_function(tf.constant([[1.0, 2.0]])).numpy()


# converting python functions to graphs
def simple_relu(x):
    """Example function"""
    if tf.greater(x, 0):
        return x
    else:
        return 0


# `tf_simple_relu` is a TensorFlow `Function` that wraps `simple_relu`.
tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())

# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))

# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())


# Polymorphism: one Function, many graphs
# Each time you invoke a Function with a set of arguments that can't be handled by any of its existing graphs
# (such as arguments with new dtypes or incompatible shapes),
# Function creates a new tf.Graph specialized to those new arguments.
@tf.function
def my_relu(x):
    """Example"""
    return tf.maximum(0., x)


# `my_relu` creates new graphs as it observes more signatures.
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3., -3.])))

# These two calls do *not* create new graphs.
print(my_relu(tf.constant(-2.5)))  # Signature matches `tf.constant(5.5)`.
print(my_relu(tf.constant([-1., 1.])))  # Signature matches `tf.constant([3., -3.])`

# There are three `ConcreteFunction`s (one for each graph) in `my_relu`.
# The `ConcreteFunction` also knows the return type and shape!
print(my_relu.pretty_printed_concrete_signatures())


# ------ Using tf.function properly ---------
@tf.function
def get_MSE(y_true, y_pred):
    """Returns Mean Squared Error"""
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)


y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)

get_MSE(y_true, y_pred)

# To verify that your Function's graph is doing the same computation as its equivalent Python function,
# you can make it execute eagerly with tf.config.run_functions_eagerly(True).
# This is a switch that turns off Function's ability to create and run graphs, instead of executing the code normally.
tf.config.run_functions_eagerly(True)
get_MSE(y_true, y_pred)
# Don't forget to set it back when you are done.
tf.config.run_functions_eagerly(False)


# But with the print statements, things aren't so simple!
@tf.function
def get_MSE(y_true, y_pred):
    """Returns Mean Squared Error"""
    print("Calculating MSE!")
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)


# only prints once!
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)

# to compare
# Now, globally set everything to run eagerly to force eager execution.
tf.config.run_functions_eagerly(True)

# Observe what is printed below.
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)

tf.config.run_functions_eagerly(False)

# the print statement is executed when Function runs the original code in order to create the graph in a process known
# as "tracing"
# Tracing captures the TensorFlow operations into a graph, and print is not captured in the graph.
# That graph is then executed for all three calls without ever running the Python code again.


# Non strick execution
# In the following example, the "unnecessary" operation tf.gather is skipped during graph execution,
# so the runtime error InvalidArgumentError is not raised as it would be in eager execution.
# Do not rely on an error being raised while executing a graph.

def unused_return_eager(x):
    """Example"""
    # Get index 1 will fail when `len(x) == 1`
    tf.gather(x, [1])  # unused
    return x


try:
    print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
    # All operations are run during eager execution so an error is raised.
    print(f'{type(e).__name__}: {e}')


@tf.function
def unused_return_graph(x):
    """Example"""
    # Get index 1 will fail when `len(x) == 1`
    tf.gather(x, [1])  # unused
    return x


# Only needed operations are run during graph execution. The error is not raised.
print(unused_return_graph(tf.constant([0.0])))

# Experience the speed-up
x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)


def power(x, y):
    """Example"""
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(y):
        result = tf.matmul(x, result)
    return result


print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000), "seconds")

power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000), "seconds")

# when is a function tracing?


# To figure out when your Function is tracing, add a print statement to its code.
# As a rule of thumb, Function will execute the print statement every time it traces.
@tf.function
def a_function_with_python_side_effect(x):
    """Example"""
    print("Tracing!")  # An eager-only side effect.
    return x * x + tf.constant(2)


# This is traced the first time.
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect.
print(a_function_with_python_side_effect(tf.constant(3)))

# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter.
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))

# New Python arguments always trigger the creation of a new graph, hence the extra tracing.
