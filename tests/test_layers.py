"""Test the layers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab


def test_activation(make_data):
    """Test nonlinear activation layer."""
    x, _, X = make_data
    act = ab.activation(tf.tanh)

    tc = tf.test.TestCase()
    with tc.test_session():
        F, KL = act(X)

        assert np.all(np.tanh(X.eval()) == F.eval())
        assert KL == 0


def test_fork_cat(make_data):
    """Test forking layers with concatenation join."""
    x, _, X = make_data
    l1 = [ab.activation(), ab.activation()]
    l2 = [ab.activation()]
    fork = ab.fork('cat', l1, l2)

    F, KL = fork(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape[0:2] + (2 * orig.shape[2],)
        assert np.all(forked == np.dstack((orig, orig)))
        assert KL == 0


def test_fork_add(make_data):
    """Test forking layers with add join."""
    x, _, X = make_data
    l1 = [ab.activation(), ab.activation()]
    l2 = [ab.activation()]
    fork = ab.fork('add', l1, l2)

    F, KL = fork(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape
        assert np.all(forked == 2 * orig)
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

    x, _, _ = make_data
    x_ = tf.placeholder(tf.float32, x.shape)
    X_ = tf.tile(tf.expand_dims(x_, 0), [3, 1, 1])
    N = x.shape[0]

    Phi, KL = ab.randomFourier(D, kernel=k)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        P = Phi.eval(feed_dict={x_: x})
        for i in range(P.shape[0]):
            p = P[i]
            assert p.shape == (N, 2 * D)
            # Check behaving properly with k(x, x) ~ 1.0
            assert np.allclose((p**2).sum(axis=1), np.ones(N))
        assert KL == 0


@pytest.mark.parametrize('dense', [ab.dense_map, ab.dense_var])
def test_dense_outputs(dense, make_data):
    """Make sure the dense layers output expected dimensions."""
    x, _, _ = make_data
    D = 20

    x_ = tf.placeholder(tf.float32, x.shape)
    X_ = tf.tile(tf.expand_dims(x_, 0), [3, 1, 1])
    N = x.shape[0]

    Phi, KL = dense(output_dim=D)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        assert P.shape == (3, N, D)
        assert P.dtype == np.float32
        assert np.isscalar(KL.eval(feed_dict={x_: x}))
