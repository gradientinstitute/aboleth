"""Test the layers module."""
import numpy as np
import tensorflow as tf

import aboleth as ab


def test_deepnet_outputs(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)
    loss = ab.elbo(Net, Y_, N, kl, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Net.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        l = loss.eval(feed_dict={X_: x, Y_: y, N_: float(N)})

        assert P.shape == (10, N, 1)
        assert np.isscalar(l)


def test_deepnet_likelihood_weights(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)

    # Now test with likelihood weights
    lw = np.ones_like(y)
    lw_ = tf.placeholder(tf.float32, (len(y), 1))
    lossw = ab.elbo(Net, Y_, N, kl, like, like_weights=lw_)
    lossf = ab.elbo(Net, Y_, N, kl, like, like_weights=lambda y: 1.)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Net.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        l = lossw.eval(feed_dict={X_: x, Y_: y, N_: float(N), lw_: lw})
        assert P.shape == (10, N, 1)
        assert np.isscalar(l)

        l = lossf.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(l)


def test_elbo_output(make_graph):
    """Test for scalar output from elbo."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)
    kl = 0.0
    elbo = ab.elbo(Net, Y_, N, kl, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        obj = elbo.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(obj)


def test_logprob_output(make_graph):
    """Test for correct output dimensions from logprob."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)

    logp = ab.log_prob(Y_, like, Net)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        ll = logp.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert ll.shape == (10,) + y.shape
