"""Test the layers module."""

import numpy as np
import tensorflow as tf

import aboleth as ab
from aboleth.losses import _sum_likelihood


def test_elbo_likelihood(make_graph):
    """Test for expected output dimensions from loss ."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)

    # Now test with likelihood weights
    loss = ab.elbo(Net, Y_, N, kl, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        L = loss.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(L)


def test_map_likelihood(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, _, _, _, _, like, layers = make_graph
    Net, reg = layers(X=x.astype(np.float32))

    # Now test with likelihood weights
    loss = ab.max_posterior(Net, y.astype(np.float32),  reg, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        L = loss.eval()
        assert np.isscalar(L)


def test_sum_likelihood():
    """Test we can do weighted sums of likelihoods."""
    like = ab.likelihoods.Bernoulli()
    N = 5
    Net = np.ones(N) * .5
    Y = np.ones(N)

    def weight_fn(Y):
        return Y * np.arange(N)

    unweighted = _sum_likelihood(Y, Net, like, None)
    value = _sum_likelihood(Y, Net, like, np.arange(N))
    call = _sum_likelihood(Y, Net, like, weight_fn)

    tc = tf.test.TestCase()
    with tc.test_session():
        sumll = unweighted.eval()
        assert np.allclose(sumll, np.log(0.5) * N)

        sumll = value.eval()
        assert np.allclose(sumll, np.sum(np.log(0.5) * np.arange(N)))

        sumll = call.eval()
        assert np.allclose(sumll, np.sum(np.log(0.5) * np.arange(N)))
