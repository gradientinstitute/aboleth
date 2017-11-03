"""Tests for the prediction utilities."""
import numpy as np
import tensorflow as tf

import aboleth as ab


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
