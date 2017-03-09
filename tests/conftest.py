"""Test fixtures."""
import pytest
import numpy as np
import tensorflow as tf

import aboleth as ab


@pytest.fixture
def make_data():
    """Make some simple data."""
    N = 100
    x1 = np.linspace(-10, 10, N)
    x2 = np.linspace(-1, 1, N)**2
    x = np.vstack((x1, x2)).T
    w = np.array([[0.5], [2.0]])
    Y = np.dot(x, w) + np.random.randn(N, 1)
    X = tf.tile(tf.expand_dims(x, 0), [3, 1, 1])
    return x, Y, X


@pytest.fixture
def make_graph():

    x, Y, X = make_data()

    like = ab.normal(variance=1.)
    layers = [ab.dense_map(output_dim=1)]
    N = len(x)

    X_ = tf.placeholder(tf.float32, x.shape)
    Y_ = tf.placeholder(tf.float32, Y.shape)
    N_ = tf.placeholder(tf.float32)

    return x, Y, N, X_, Y_, N_, like, layers