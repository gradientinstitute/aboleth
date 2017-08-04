"""Test distributions.py functionality."""
import pytest
import numpy as np
import tensorflow as tf
from scipy.linalg import cho_solve
from scipy.stats import wishart

from aboleth.distributions import (Normal, Gaussian, kl_qp, _chollogdet)
from .conftest import SEED


def test_kl_normal_normal():
    """Test Normal/Normal KL."""
    dim = (10, 5)
    mu = np.zeros(dim)
    var = 1.0

    q = Normal(mu, var)

    # Test 0 KL
    p = Normal(mu, var)
    KL0 = kl_qp(q, p)

    # Test diff var
    var1 = 2.0
    p = Normal(mu, var1)
    KL1 = kl_qp(q, p)
    rKL1 = 0.5 * (var / var1 - 1 + np.log(var1 / var)) * np.prod(dim)

    # Test diff mu
    mu1 = np.ones(dim)
    p = Normal(mu1, var)
    KL2 = kl_qp(q, p)
    rKL2 = 0.5 * (np.sum((mu1 - mu)**2) / var)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert KL0.eval() == 0.
        assert np.allclose(KL1.eval(), rKL1)
        assert np.allclose(KL2.eval(), rKL2)


def test_kl_gaussian_normal(random):
    """Test Gaussian/Normal KL."""
    dim = (10, 5)
    Dim = (5, 10, 10)

    mu0 = random.randn(*dim).astype(np.float32)
    L0 = random_chol(Dim)
    q = Gaussian(mu0, L0)

    mu1 = random.randn(*dim).astype(np.float32)
    var1 = 1.0
    L1 = [(np.sqrt(var1) * np.eye(dim[0])).astype(np.float32)
          for _ in range(dim[1])]
    p = Normal(mu1, var1)

    KL = kl_qp(q, p)
    KLr = KLdiv(mu0, L0, mu1, L1)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(KL.eval(), KLr)


def test_kl_gaussian_gaussian(random):
    """Test Gaussian/Gaussian KL."""
    dim = (10, 5)
    Dim = (5, 10, 10)

    mu0 = random.randn(*dim).astype(np.float32)
    L0 = random_chol(Dim)
    q = Gaussian(mu0, L0)

    mu1 = random.randn(*dim).astype(np.float32)
    L1 = random_chol(Dim)
    p = Gaussian(mu1, L1)

    KL = kl_qp(q, p)
    KLr = KLdiv(mu0, L0, mu1, L1)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(KL.eval(), KLr)


def test_kl_qp():
    """Test the validity of the results coming back from kl_qp."""
    dim = (10, 5)
    Dim = (5, 10, 10)

    mu = np.zeros(dim).astype(np.float32)
    var = 1.0
    L = random_chol(Dim)

    qn = Normal(mu, var)
    qg = Gaussian(mu, L)
    p = Normal(mu, var)
    kl_nn = kl_qp(qn, p)
    kl_gn = kl_qp(qg, p)

    tc = tf.test.TestCase()
    with tc.test_session():
        nn = kl_nn.eval()
        assert nn >= 0
        assert np.isscalar(nn)

        gn = kl_gn.eval()
        assert gn >= 0
        assert np.isscalar(gn)

    # This is not implemented and should error
    with pytest.raises(NotImplementedError):
        kl_qp(p, qg)


def test_chollogdet():
    """Test log det with cholesky matrices."""
    Dim = (5, 10, 10)
    L = random_chol(Dim)
    rlogdet = np.sum([logdet(l) for l in L])
    tlogdet = _chollogdet(L)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(tlogdet.eval(), rlogdet)


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
    for m0, m1, L0, L1 in zip(mu0.T, mu1.T, Lcov0, Lcov1):
        tr += np.trace(cho_solve((L1, True), L0.dot(L0.T)))
        md = m1 - m0
        dist += md.dot(cho_solve((L1, True), md))
        ldet += logdet(L1) - logdet(L0)

    KL = 0.5 * (tr + dist + ldet - D * n)
    return KL


def logdet(L):
    """Log Determinant from Cholesky."""
    return 2. * np.log(L.diagonal()).sum()
