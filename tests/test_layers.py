"""Test the layers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab


def test_eye(make_data):
    """Test identity layer."""
    x, _, X = make_data
    eye = ab.eye()

    F, KL = eye(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.all(X.eval() == F.eval())
    assert KL == 0


def test_activation(make_data):
    """Test nonlinear activation layer."""
    x, _, X = make_data
    act = ab.activation(tf.tanh)

    tc = tf.test.TestCase()
    with tc.test_session():
        F, KL = act(X)

        assert np.all(np.tanh(X.eval()) == F.eval())
        assert KL == 0


def test_fork(make_data):
    """Test forking layers."""
    x, _, X = make_data
    fork = ab.fork(replicas=2)

    (X1, X2), KL = fork(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.all(X.eval() == X1.eval()) and np.all(X.eval() == X2.eval())
        assert KL == 0


def test_lmap(make_data):
    """Test layer map functions to multiple input layers."""
    x, _, X = make_data
    app = ab.lmap(ab.activation(tf.tanh), ab.eye())

    tc = tf.test.TestCase()
    with tc.test_session():
        (Xtan, Xeye), KL = app([X, X])

        assert np.all(np.tanh(X.eval()) == Xtan.eval())
        assert np.all(X.eval() == Xeye.eval())
        assert KL == 0


def test_add(make_data):
    """Test adding of forked input layers."""
    x, _, X = make_data
    add = ab.add()

    tc = tf.test.TestCase()
    with tc.test_session():
        Xadd, KL = add([X, X, X])
        assert np.all(3 * X.eval() == Xadd.eval())
        assert KL == 0


def test_cat(make_data):
    """Test concatenating of forked input layers."""
    x, _, X = make_data
    cat = ab.cat()

    N, D = x.shape
    tc = tf.test.TestCase()
    with tc.test_session():
        Xcat, KL = cat([X, X, X])
        assert Xcat.eval().shape == (3, N, 3 * D)
        assert KL == 0


@pytest.mark.parametrize('kernels', [
    (ab.RBF, {}),
    (ab.Matern, {'p': 1}),
    (ab.Matern, {'p': 2})
])
def test_kernels(kernels, make_data):
    """Test random kernels approximations."""
    d, D = 10, 100
    kern, p = kernels
    k = kern(**p)

    # Check dim
    P = k.weights(input_dim=d, n_features=D)
    assert P.shape == (d, D)

    # Check behaving properly with k(x, x) ~ 1.0
    x, _, _ = make_data
    x_ = tf.placeholder(tf.float32, x.shape)
    X_ = tf.tile(tf.expand_dims(x_, 0), [3, 1, 1])
    N = x.shape[0]

    rff = ab.randomFourier(D, kernel=k)
    Phi, KL = rff(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        P = Phi.eval(feed_dict={x_: x})
        for i in range(P.shape[0]):
            assert np.allclose((P[i]**2).sum(axis=1), np.ones(N))
