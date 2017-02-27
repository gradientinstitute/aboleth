"""Test the layers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab


def test_eye(make_data):
    """Test identity layer."""
    X, _ = make_data
    eye = ab.eye()

    F, KL = eye(X)

    assert np.all(X == F)
    assert KL == 0


def test_activation(make_data):
    """Test nonlinear activation layer."""
    X, _ = make_data
    act = ab.activation(tf.tanh)

    tc = tf.test.TestCase()
    with tc.test_session():
        F, KL = act(X)

        assert np.all(np.tanh(X) == F.eval())
        assert KL == 0


def test_fork(make_data):
    """Test forking layers."""
    X, _ = make_data
    fork = ab.fork(replicas=2)

    (X1, X2), KL = fork(X)

    assert np.all(X == X1) and np.all(X == X2)
    assert KL == 0


def test_apply(make_data):
    """Test apply functions to multiple input layers."""
    X, _ = make_data
    app = ab.apply(ab.activation(tf.tanh), ab.eye())

    tc = tf.test.TestCase()
    with tc.test_session():
        (Xtan, Xeye), KL = app([X, X])

        assert np.all(np.tanh(X) == Xtan.eval())
        assert np.all(X == Xeye)
        assert KL == 0


def test_add(make_data):
    """Test adding of forked input layers."""
    X, _ = make_data
    add = ab.add()

    tc = tf.test.TestCase()
    with tc.test_session():
        Xadd, KL = add([X, X, X])
        assert np.all(3 * X == Xadd.eval())
        assert KL == 0


def test_cat(make_data):
    """Test concatenating of forked input layers."""
    X, _ = make_data
    cat = ab.cat()

    N, D = X.shape
    tc = tf.test.TestCase()
    with tc.test_session():
        Xcat, KL = cat([X, X, X])
        assert Xcat.eval().shape == (N, 3 * D)
        assert np.all(Xcat.eval() == np.hstack([X, X, X]))
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
    X, _ = make_data
    X_ = tf.placeholder(tf.float32, X.shape)
    N = X.shape[0]

    rff = ab.randomFourier(D, kernel=k)
    Phi, KL = rff(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi = Phi.eval(feed_dict={X_: X})
        assert np.allclose((phi**2).sum(axis=1), np.ones(N))
