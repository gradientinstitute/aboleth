"""Test the layers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab

from aboleth.distributions import norm_prior, gaus_posterior

D = 10
DIM = (2, 10)
EDIM = (5, 10)


def test_sample(make_data):
    """Test the sample tiling layer."""
    x, _, X = make_data
    s = ab.sample(3)

    F, KL = s(x)
    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        X_array = X.eval()
        assert KL == 0.0
        assert np.array_equal(f, X_array)
        for i in range(3):
            assert np.array_equal(f[i], x)


def test_activation(make_data):
    """Test nonlinear activation layer."""
    x, _, X = make_data
    act = ab.activation(tf.tanh)

    tc = tf.test.TestCase()
    with tc.test_session():
        F, KL = act(X)

        assert np.all(np.tanh(X.eval()) == F.eval())
        assert KL == 0


def test_dropout(make_data):
    """Test dropout layer."""
    x, _, X = make_data
    drop = ab.dropout(0.5)

    F, KL = drop(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        prop_zero = np.sum(f == 0) / np.prod(f.shape)

        assert f.shape == X.eval().shape
        assert (prop_zero > 0.4) and (prop_zero < 0.6)
        assert KL == 0


@pytest.mark.parametrize('kernels', [
    (ab.RBF, {}),
    (ab.Matern, {'p': 1}),
    (ab.Matern, {'p': 2})
])
def test_kernels(kernels, make_data):
    """Test random kernels approximations."""
    d, D = 10, 100
    S = 3
    kern, p = kernels
    k = kern(**p)

    # Check dim
    P = k.weights(input_dim=d, n_features=D)
    assert P.shape == (d, D)

    x, _, _ = make_data
    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = ab.random_fourier(D, kernel=k)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        P = Phi.eval(feed_dict={x_: x})
        for i in range(P.shape[0]):
            p = P[i]
            assert p.shape == (N, 2 * D)
            # Check behaving properly with k(x, x) ~ 1.0
            assert np.allclose((p**2).sum(axis=1), np.ones(N))
        assert KL == 0


def test_arc_cosine(make_data):
    """Test the random Arc Cosine kernel."""
    S = 3
    x, _, _ = make_data
    x_, X_ = _make_placeholders(x, S)

    F, KL = ab.random_arccosine(n_features=10)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval(feed_dict={x_: x})

        assert f.shape == (3, x.shape[0], 10)
        assert KL == 0


def test_dense_embeddings(make_categories):
    """Test the embedding layer."""
    x, K = make_categories
    N = len(x)
    S = 3
    x_, X_ = _make_placeholders(x, S, tf.int32)
    output, KL = ab.embed_var(output_dim=D, n_categories=K)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        kl = KL.eval()

        assert np.isscalar(kl)
        assert kl > 0

        Phi = output.eval(feed_dict={x_: x})

        assert Phi.shape == (S, N, D)


@pytest.mark.parametrize('dense', [ab.dense_map, ab.dense_var])
def test_dense_outputs(dense, make_data):
    """Make sure the dense layers output expected dimensions."""
    x, _, _ = make_data
    S = 3

    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = dense(output_dim=D)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        assert P.shape == (S, N, D)
        assert P.dtype == np.float32
        assert np.isscalar(KL.eval(feed_dict={x_: x}))


@pytest.mark.parametrize('dists', [
    {'prior_W': norm_prior(DIM, 1.), 'prior_b': norm_prior((D,), 1.)},
    {'post_W': norm_prior(DIM, 1.), 'post_b': norm_prior((D,), 1.)},
    {'prior_W': norm_prior(DIM, 1.), 'post_W': norm_prior(DIM, 1.)},
    {'prior_W': norm_prior(DIM, 1.), 'post_W': gaus_posterior(DIM, 1.)},
    {'prior_W': gaus_posterior(DIM, 1.), 'post_W': gaus_posterior(DIM, 1.)},
])
def test_dense_distribution(dists, make_data):
    x, _, _ = make_data
    S = 3

    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = ab.dense_var(output_dim=D, **dists)(X_)
    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        assert P.shape == (S, N, D)
        assert KL.eval() >= 0.


@pytest.mark.parametrize('dists', [
    {'prior_W': norm_prior(EDIM, 1.), 'post_W': norm_prior(EDIM, 1.)},
    {'prior_W': norm_prior(EDIM, 1.), 'post_W': gaus_posterior(EDIM, 1.)},
    {'prior_W': gaus_posterior(EDIM, 1.), 'post_W': gaus_posterior(EDIM, 1.)},
])
def test_embeddings_distribution(dists, make_categories):
    """Test the embedding layer."""
    x, K = make_categories
    N = len(x)
    S = 3
    x_, X_ = _make_placeholders(x, S, tf.int32)
    output, KL = ab.embed_var(output_dim=D, n_categories=K, **dists)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        Phi = output.eval(feed_dict={x_: x})
        assert Phi.shape == (S, N, D)
        assert KL.eval() >= 0.


def _make_placeholders(x, S, xtype=tf.float32):
    x_ = tf.placeholder(xtype, x.shape)
    X_ = tf.tile(tf.expand_dims(x_, 0), [S, 1, 1])
    return x_, X_
