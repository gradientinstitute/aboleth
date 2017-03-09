"""Test the layers module."""
import numpy as np
import tensorflow as tf

import aboleth as ab


def test_deepnet_outputs(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Phi, loss = ab.deepnet(X_, Y_, N_, layers, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Phi.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        l = loss.eval(feed_dict={X_: x, Y_: y, N_: float(N)})

        assert P.shape == (10, N, 1)
        assert np.isscalar(l)


def test_elbo_output(make_graph):
    """Test for scalar output from elbo."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Phi, loss = ab.deepnet(X_, Y_, N_, layers, like)
    KL = 0
    elbo = ab.elbo(Phi, Y_, N_, KL, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        obj = elbo.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(obj)


def test_logprob_output(make_graph):
    """Test for correct output dimensions from logprob."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Phi, loss = ab.deepnet(X_, Y_, N_, layers, like)
    logp = ab.log_prob(Y_, like, Phi)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        ll = logp.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert ll.shape == y.shape
