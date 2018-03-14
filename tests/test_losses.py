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
    x, y, N, X_, Y_, N_, layers = make_graph
    nn, reg = layers(X=X_)
    log_like = tf.distributions.Normal(nn, scale=1.).log_prob(Y_)

    loss = ab.max_posterior(log_like, reg)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        L = loss.eval(feed_dict={X_: x, Y_: y})
        assert np.isscalar(L)


@pytest.mark.parametrize('likelihood', [(tf.distributions.Categorical, 2),
                                        (tf.distributions.Bernoulli, 1)])
def test_categorical_likelihood(make_data, likelihood):
    """Test aboleth with discrete likelihoods.

    Since these are kind of corner cases...
    """
    x, y, _, = make_data
    like, K = likelihood
    N, _ = x.shape

    # Make two classes (K = 2)
    Y = np.zeros(len(y), dtype=np.int32)
    Y[y[:, 0] > 0] = 1

    if K == 1:
        Y = Y[:, np.newaxis]

    X_ = tf.placeholder(tf.float32, x.shape)
    Y_ = tf.placeholder(tf.int32, Y.shape)
    n_samples_ = tf.placeholder(tf.int32)

    layers = ab.stack(
        ab.InputLayer(name='X', n_samples=n_samples_),
        ab.DenseMAP(output_dim=K)
    )

    nn, reg = layers(X=X_)
    like = like(logits=nn)
    log_like = like.log_prob(Y_)
    prob = like.prob(Y_)

    ELBO = ab.elbo(log_like, reg, N)
    MAP = ab.max_posterior(log_like, reg)

    fd = {X_: x, Y_: Y, n_samples_: 10}
    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        assert like.probs.eval(feed_dict=fd).shape == (10, N, K)
        assert prob.eval(feed_dict=fd).shape == (10,) + Y.shape

        L = ELBO.eval(feed_dict=fd)

        L = MAP.eval(feed_dict=fd)
        assert np.isscalar(L)
