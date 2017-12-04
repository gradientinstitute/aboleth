"""Tests for the prediction utilities."""
import numpy as np
import tensorflow as tf

import aboleth as ab
from aboleth.layers import _sample_W


def test_sample_mean():
    X = np.arange(10, dtype=float).reshape((2, 5))
    true = X.mean(axis=0)
    mean = ab.sample_mean(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.all(true == mean.eval())


def test_sample_quantiles():
    X = np.arange(100, dtype=float).reshape((10, 10))

    # NOTE: numpy takes lower nearest, tensorflow takes higher nearest when
    # equidistant
    true = np.percentile(X, q=[10, 51, 90], axis=0, interpolation='nearest')
    pers = ab.sample_percentiles(X, per=[10, 51, 90])

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.all(true == pers.eval())


def test_sample_model():
    g = tf.Graph()
    with g.as_default():
        dist = tf.distributions.Normal(loc=0., scale=1.)
        n_samples = 10
        W = _sample_W(dist, n_samples)
        tc = tf.test.TestCase()
        with tc.test_session() as sess:
            sample_dict = ab.sample_model()
            res = sess.run(W, feed_dict=sample_dict)
        assert list(sample_dict.keys()) == [W]
        assert np.all(res == sample_dict[W])


def test_sample_model_nodefaults():
    g = tf.Graph()
    with g.as_default():
        dist = tf.distributions.Normal(loc=0., scale=1.)
        n_samples = tf.placeholder(shape=tuple(), dtype=tf.int32)
        W = _sample_W(dist, n_samples)
        feed_dict = {n_samples: 10}
        tc = tf.test.TestCase()
        with tc.test_session() as sess:
            graph = tf.get_default_graph()
            sample_dict = ab.sample_model(graph, sess, feed_dict)
            feed_dict.update(sample_dict)
            res = sess.run(W, feed_dict=feed_dict)
        assert list(sample_dict.keys()) == [W]
        assert np.all(res == sample_dict[W])
