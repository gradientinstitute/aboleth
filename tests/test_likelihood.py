"""Test the likelihood module."""
import pytest
import numpy as np
import tensorflow as tf
import scipy.stats as ss
from scipy.special import expit

import aboleth.likelihood as lk


@pytest.mark.parametrize('likelihood', [
    (lk.normal(variance=1),
     ss.norm.rvs,
     lambda x, f: ss.norm.logpdf(x, loc=f)),

    (lk.binomial(n=10.),
     lambda f, size: ss.binom.rvs(n=10, p=f, size=size),
     lambda x, f: ss.binom.logpmf(x, n=10, p=f)),

    (lk.bernoulli(),
     ss.bernoulli.rvs,
     lambda x, f: ss.bernoulli.logpmf(x, p=f)),
])
def test_log_likelihoods(likelihood):

    alike, rvs, logprob = likelihood

    f = expit(np.random.randn(100).astype(np.float32))
    x = rvs(f, size=100).astype(np.float32)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(logprob(x, f), alike(x, f).eval())
