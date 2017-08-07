"""Test the likelihood module."""
import pytest
import numpy as np
import tensorflow as tf
import scipy.stats as ss
from scipy.special import expit

from aboleth.likelihoods import Normal, Bernoulli, Binomial, Categorical


@pytest.mark.parametrize('likelihood', [
    (Normal(variance=1),
     ss.norm.rvs,
     lambda x, f: ss.norm.logpdf(x, loc=f)),

    (Binomial(n=10.),
     lambda f, size: ss.binom.rvs(n=10, p=f, size=size),
     lambda x, f: ss.binom.logpmf(x, n=10, p=f)),

    (Bernoulli(),
     ss.bernoulli.rvs,
     lambda x, f: ss.bernoulli.logpmf(x, p=f)),

])
def test_log_likelihoods(likelihood, random):

    alike, rvs, logprob = likelihood

    f = expit(random.randn(100).astype(np.float32))
    x = rvs(f, size=100).astype(np.float32)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(logprob(x, f), alike(x, f).eval())


@pytest.mark.parametrize('likelihood', [
    (Categorical(),
     lambda f, size: ss.multinomial.rvs(n=1, p=f, size=size),
     lambda x, f: ss.multinomial.logpmf(x, n=1, p=f)),
])
def test_log_likelihoods_multitask(likelihood, random):

    alike, rvs, logprob = likelihood

    f = expit(random.randn(100, 5))
    f /= np.sum(f, axis=-1).reshape(-1, 1)  # normalize
    f = f.astype(np.float32)

    # apply_along_axis as scipy rvs doesn't support broadcasting
    x = np.apply_along_axis(lambda f: rvs(f, size=1), arr=f, axis=1)
    x = x.astype(np.float32)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(logprob(x, f), alike(x, f).eval())
