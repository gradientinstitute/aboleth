"""Test the layers module."""
import numpy as np
import tensorflow as tf

import aboleth as ab


def test_deepnet_outputs(make_graph):
    """Test for expected output dimensions from deepnet."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, loss = ab.deepnet(X_, Y_, N_, layers, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Net.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        l = loss.eval(feed_dict={X_: x, Y_: y, N_: float(N)})

        assert P.shape == (10, N, 1)
        assert np.isscalar(l)


def test_featurenet_outputs(make_graph):
    """Test for expected output dimensions from featurenet."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    features = [
        (X_, [ab.dense_map(output_dim=10), ab.activation(tf.tanh)]),
        (X_, [ab.dense_map(output_dim=10), ab.activation(tf.tanh)])
    ]

    Net, loss = ab.featurenet(features, Y_, N_, layers, like)

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

    # Now test with likelihood weights
    lw = np.ones_like(y)
    lw_ = tf.placeholder(tf.float32, (len(y), 1))
    Netw, lossw = ab.deepnet(X_, Y_, N_, layers, like, like_weights=lw_)
    Netf, lossf = ab.deepnet(X_, Y_, N_, layers, like,
                             like_weights=lambda y: 1.)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Netw.eval(feed_dict={X_: x, Y_: y, N_: float(N), lw_: lw})
        l = lossw.eval(feed_dict={X_: x, Y_: y, N_: float(N), lw_: lw})

        assert P.shape == (10, N, 1)
        assert np.isscalar(l)

        P = Netf.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        l = lossf.eval(feed_dict={X_: x, Y_: y, N_: float(N)})

        assert P.shape == (10, N, 1)
        assert np.isscalar(l)


def test_elbo_output(make_graph):
    """Test for scalar output from elbo."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, loss = ab.deepnet(X_, Y_, N_, layers, like)
    KL = 0
    elbo = ab.elbo(Net, Y_, N_, KL, like)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        obj = elbo.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert np.isscalar(obj)


def test_logprob_output(make_graph):
    """Test for correct output dimensions from logprob."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, loss = ab.deepnet(X_, Y_, N_, layers, like)
    logp = ab.log_prob(Y_, like, Net)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        ll = logp.eval(feed_dict={X_: x, Y_: y, N_: float(N)})
        assert ll.shape == y.shape
