"""Test distributions.py functionality."""
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_solve
from scipy.stats import wishart
from tensorflow.contrib.distributions import MultivariateNormalTriL

from aboleth.distributions import kl_sum, _chollogdet
from .conftest import SEED


def test_kl_normal_normal():
    """Test Normal/Normal KL."""
    dim = (5, 10)
    mu = np.zeros(dim, dtype=np.float32)
    std = 1.0

    q = tf.distributions.Normal(mu, std)

    # Test 0 KL
    p = tf.distributions.Normal(mu, std)
    KL0 = kl_sum(q, p)

    # Test diff var
    std1 = 2.0
    p = tf.distributions.Normal(mu, std1)
    KL1 = kl_sum(q, p)
    rKL1 = 0.5 * ((std / std1)**2 - 1 + np.log((std1 / std)**2)) * np.prod(dim)

    # Test diff mu
    mu1 = np.ones(dim, dtype=np.float32)
    p = tf.distributions.Normal(mu1, std)
    KL2 = kl_sum(q, p)
    rKL2 = 0.5 * (np.sum((mu1 - mu)**2) / std**2)

    tc = tf.test.TestCase()
    with tc.test_session():
        kl0 = KL0.eval()
        assert np.isscalar(kl0)
        assert kl0 == 0.
        assert np.allclose(KL1.eval(), rKL1)
        assert np.allclose(KL2.eval(), rKL2)


def test_kl_gaussian_normal(random):
    """Test Gaussian/Normal KL."""
    dim = (5, 10)
    Dim = (5, 10, 10)

    mu0 = random.randn(*dim).astype(np.float32)
    L0 = random_chol(Dim)
    q = MultivariateNormalTriL(mu0, L0)

    mu1 = random.randn(*dim).astype(np.float32)
    std1 = 1.0
    L1 = [(std1 * np.eye(dim[1])).astype(np.float32) for _ in range(dim[0])]
    p = tf.distributions.Normal(mu1, std1)

    KL = kl_sum(q, p)
    KLr = KLdiv(mu0, L0, mu1, L1)

    tc = tf.test.TestCase()
    with tc.test_session():
        kl = KL.eval()
        assert np.isscalar(kl)
        assert np.allclose(kl, KLr)


def test_kl_gaussian_gaussian(random):
    """Test Gaussian/Gaussian KL."""
    dim = (5, 10)
    Dim = (5, 10, 10)

    mu0 = random.randn(*dim).astype(np.float32)
    L0 = random_chol(Dim)
    q = MultivariateNormalTriL(mu0, L0)

    mu1 = random.randn(*dim).astype(np.float32)
    L1 = random_chol(Dim)
    p = MultivariateNormalTriL(mu1, L1)

    KL = kl_sum(q, p)
    KLr = KLdiv(mu0, L0, mu1, L1)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(KL.eval(), KLr)


def test_chollogdet():
    """Test log det with cholesky matrices."""
    Dim = (5, 10, 10)
    L = random_chol(Dim)
    rlogdet = np.sum([logdet(l) for l in L])
    tlogdet = _chollogdet(L)

    L[0, 0, 0] = 1e-17  # Near zero to test numerics
    L[1, 3, 3] = -1.
    L[4, 5, 5] = -20.

    nlogdet = _chollogdet(L)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(tlogdet.eval(), rlogdet)
        assert not np.isnan(nlogdet.eval())


def random_chol(dim):
    """Generate random pos def matrices."""
    D = dim[1]
    n = dim[0]
    np.random.seed(SEED)
    C = wishart.rvs(df=D, scale=10 * np.eye(D), size=n)
    np.random.seed(None)
    L = np.array([np.linalg.cholesky(c).astype(np.float32) for c in C])
    return L


def KLdiv(mu0, Lcov0, mu1, Lcov1):
    """Numpy KL calculation."""
    tr, dist, ldet = 0., 0., 0.
    D, n = mu0.shape
    for m0, m1, L0, L1 in zip(mu0, mu1, Lcov0, Lcov1):
        tr += np.trace(cho_solve((L1, True), L0.dot(L0.T)))
        md = m1 - m0
        dist += md.dot(cho_solve((L1, True), md))
        ldet += logdet(L1) - logdet(L0)

    KL = 0.5 * (tr + dist + ldet - D * n)
    return KL


def logdet(L):
    """Log Determinant from Cholesky."""
    return 2. * np.log(L.diagonal()).sum()
