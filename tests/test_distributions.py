"""Test distributions.py functionality."""
import numpy as np
import tensorflow as tf
from scipy.linalg import solve
from scipy.stats import wishart

from aboleth.distributions import (Normal, Gaussian, kl_normal_normal,
                                   kl_gaussian_normal)


def test_kl_normal_normal():
    """Test Normal/Normal KL."""
    dim = (10, 5)
    mu = np.zeros(dim)
    var = 1.0

    q = Normal(mu, var)

    # Test 0 KL
    p = Normal(mu, var)
    KL0 = kl_normal_normal(q, p)

    # Test diff var
    var1 = 2.0
    p = Normal(mu, var1)
    KL1 = kl_normal_normal(q, p)
    rKL1 = 0.5 * (var / var1 - 1 + np.log(var1 / var)) * np.prod(dim)

    # Test diff mu
    mu1 = np.ones(dim)
    p = Normal(mu1, var)
    KL2 = kl_normal_normal(q, p)
    rKL2 = 0.5 * (np.sum((mu1 - mu)**2) / var)

    tc = tf.test.TestCase()
    with tc.test_session():
        assert KL0.eval() == 0.
        assert np.allclose(KL1.eval(), rKL1)
        assert np.allclose(KL2.eval(), rKL2)


def test_kl_gaussian_normal():
    """Test Gaussian/Normal KL."""
    dim = (10, 5)
    Dim = (5, 10, 10)

    mu0 = np.random.randn(*dim).astype(np.float32)
    L0, C0 = random_chol(Dim)
    q = Gaussian(mu0, L0)

    mu1 = np.random.randn(*dim).astype(np.float32)
    var1 = 1.0
    C1 = [var1 * np.eye(dim[0]) for _ in range(dim[1])]
    p = Normal(mu1, var1)

    KL = kl_gaussian_normal(q, p)
    KLr = KLdiv(mu0, C0, mu1, C1)

    # import IPython; IPython.embed()

    tc = tf.test.TestCase()
    with tc.test_session():
        assert np.allclose(KL.eval(), KLr)


def random_chol(dim):
    """Generate random pos def matrices."""
    D = dim[1]
    n = dim[0]
    C = wishart.rvs(df=D, scale=np.eye(D), size=n).astype(np.float32)
    L = np.array([np.linalg.cholesky(c) for c in C])
    return L, C


def KLdiv(mu0, Cov0, mu1, Cov1):
    """Numpy KL calculation."""
    KL = 0
    D, _ = mu0.shape
    for m0, m1, C0, C1 in zip(mu0.T, mu1.T, Cov0, Cov1):
        KL += 0.5 * (np.trace(solve(C1, C0, sym_pos=True))
                     + (m1 - m0).dot(solve(C1, (m1 - m0), sym_pos=True))
                     - D
                     + np.linalg.slogdet(C1)[1] - np.linalg.slogdet(C0)[1])
    return KL
