"""Test fixtures."""
import pytest
import numpy as np
import tensorflow as tf

import aboleth as ab

from scipy.special import expit


SEED = 666
RAND = np.random.RandomState(SEED)


@pytest.fixture
def random():
    return RAND


@pytest.fixture
def make_data():
    """Make some simple data."""
    N = 100
    x1 = np.linspace(-10, 10, N)
    x2 = np.linspace(-1, 1, N)**2
    x = np.vstack((x1, x2)).T
    w = np.array([[0.5], [2.0]])
    Y = np.dot(x, w) + RAND.randn(N, 1)
    X = tf.tile(tf.expand_dims(x, 0), [3, 1, 1])
    return x, Y, X


@pytest.fixture
def make_image_data():
    """Make some simple data."""
    N = 100
    M = 3
    # N 28x28 RGB float images
    x = expit(RAND.randn(N, 28, 28, 3)).astype(np.float32)
    w = np.linspace(-2.5, 2.5, 28*28*3)
    Y = np.dot(x.reshape(-1, 28*28*3), w) + RAND.randn(N, 1)
    X = tf.tile(tf.expand_dims(x, 0), [M, 1, 1, 1, 1])
    return x, Y, X


@pytest.fixture
def make_missing_data():
    """Make some simple data."""
    N = 10
    D = 5
    x = np.ones((N, D)) * np.linspace(1, D, D)
    mask = np.zeros((N, D)).astype(bool)
    mask[N-5:] = True
    x[mask] = 666.
    X = tf.tile(tf.expand_dims(x, 0), [3, 1, 1])
    X = tf.cast(X, tf.float32)
    w = np.linspace(1, 5, 5)[np.newaxis, :]
    y = np.sum(x * w, axis=1)
    return x, mask, X, y


@pytest.fixture
def make_categories():
    """Make some simple categorical data."""
    N = 100
    K = 20
    x = RAND.randint(0, K, size=N)[:, np.newaxis]
    x = x.astype(np.int32)
    return x, K


@pytest.fixture
def make_missing_categories():
    """Make some simple categorical data."""
    S = 3
    N = 100
    K1 = 20
    K2 = 3
    x1 = RAND.randint(0, K1, size=(S, N))[:, :, np.newaxis]
    x2 = RAND.randint(0, K2, size=(S, N))[:, :, np.newaxis]
    x = np.concatenate((x1, x2), axis=2)
    mask = np.random.choice([True, False], size=x.shape[1:])
    x = x.astype(np.int32)
    return x, mask, (K1, K2)


@pytest.fixture
def make_graph():
    """Make the requirements for making a simple tf graph."""
    x, Y, X = make_data()

    layers = ab.stack(
        ab.InputLayer(name='X', n_samples=10),
        lambda X: (X[:, :, 0:1], 0.0)   # Mock a sampling layer
    )
    N = len(x)

    X_ = tf.placeholder(tf.float32, x.shape)
    Y_ = tf.placeholder(tf.float32, Y.shape)
    N_ = tf.placeholder(tf.float32)

    return x, Y, N, X_, Y_, N_, layers
