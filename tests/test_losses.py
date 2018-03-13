"""Test the layers module."""

import pytest
import numpy as np
import tensorflow as tf

import aboleth as ab


def test_elbo_likelihood(make_graph):
    """Test for expected output dimensions from loss ."""
    x, y, N, X_, Y_, N_, layers = make_graph
    nn, kl = layers(X=X_)
    log_like = tf.distributions.Normal(nn, scale=1.).log_prob(Y_)

    loss = ab.elbo(log_like, kl, N)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        L = loss.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(L)


def test_map_likelihood(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, _, _, _, _, layers = make_graph
    Y = y.astype(np.float32)
    nn, reg = layers(X=x.astype(np.float32))
    log_like = tf.distributions.Normal(nn, scale=1.).log_prob(Y)

    loss = ab.max_posterior(log_like, reg)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        L = loss.eval()
        assert np.isscalar(L)


def test_categorical_likelihood(make_data):
    """Test aboleth with a tf.distributions.Categorical likelihood.

    Since it is a bit of an odd half-multivariate case.
    """
    x, y, _, = make_data
    N, K = x.shape

    # Make two classes (K = 2)
    Y = np.zeros(len(y), dtype=np.int32)
    Y[y[:, 0] > 0] = 1

    layers = ab.stack(
        ab.InputLayer(name='X', n_samples=10),
        lambda X: (X, 0.0)   # Mock a sampling layer, with 2-class output
    )

    nn, reg = layers(X=x.astype(np.float32))
    like = tf.distributions.Categorical(logits=nn)
    log_like = like.log_prob(Y)

    ELBO = ab.elbo(log_like, reg, N)
    MAP = ab.max_posterior(log_like, reg)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        assert like.probs.eval().shape == (10, N, K)
        assert like.prob(Y).eval().shape == (10, N)

        L = ELBO.eval()
        assert np.isscalar(L)

        L = MAP.eval()
        assert np.isscalar(L)
