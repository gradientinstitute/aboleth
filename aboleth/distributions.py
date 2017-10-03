"""Helper functions for model parameter distributions."""
import numpy as np
import tensorflow as tf

from tensorflow.contrib.distributions import MultivariateNormalTriL

from aboleth.util import pos
from aboleth.random import seedgen


#
# Streamlined interfaces for initialising the priors and posteriors
#

def norm_prior(dim, std):
    """Initialise a prior (zero mean, isotropic) Normal distribution.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    std : float
        the prior standard deviation of this distribution.

    Returns
    -------
    P : tf.distributions.Normal
        the initialised prior Normal object.

    Note
    ----
    This will make a tf.Variable on the variance of the prior that is
    initialised with ``std``.

    """
    mu = tf.zeros(dim)
    std = pos(tf.Variable(std, name="W_mu_p"))
    P = tf.distributions.Normal(loc=mu, scale=std)
    return P


def norm_posterior(dim, std0):
    """Initialise a posterior (diagonal) Normal distribution.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    std0 : float
        the initial (unoptimized) standard deviation of this distribution.

    Returns
    -------
    Q : tf.distributions.Normal
        the initialised posterior Normal object.

    Note
    ----
    This will make tf.Variables on the randomly initialised mean and standard
    deviation of the posterior. The initialisation of the mean is from a Normal
    with zero mean, and ``std0`` standard deviation, and the initialisation of
    the standard deviation is from a gamma distribution with an alpha of
    ``std0`` and a beta of 1.

    """
    mu_0 = tf.random_normal(dim, stddev=std0, seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")

    std_0 = tf.random_gamma(alpha=std0, shape=dim, seed=next(seedgen))
    std = pos(tf.Variable(std_0, name="W_std_q"))

    Q = tf.distributions.Normal(loc=mu, scale=std)
    return Q


def gaus_posterior(dim, std0):
    """Initialise a posterior Gaussian distribution with a diagonal covariance.

    Even though this is initialised with a diagonal covariance, a full
    covariance will be learned, using a lower triangular Cholesky
    parameterisation.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    std0 : float
        the initial (unoptimized) diagonal standard deviation of this
        distribution.

    Returns
    -------
    Q : tf.contrib.distributions.MultivariateNormalTriL
        the initialised posterior Gaussian object.

    Note
    ----
    This will make tf.Variables on the randomly initialised mean and covariance
    of the posterior. The initialisation of the mean is from a Normal with zero
    mean, and ``std0`` standard deviation, and the initialisation of the (lower
    triangular of the) covariance is from a gamma distribution with an alpha of
    ``std0`` and a beta of 1.

    """
    O, I = dim

    # Optimize only values in lower triangular
    u, v = np.tril_indices(I)
    indices = (u * I + v)[:, np.newaxis]
    l0 = np.tile(np.eye(I), [O, 1, 1])[:, u, v].T
    l0 = l0 * tf.random_gamma(alpha=std0, shape=l0.shape, seed=next(seedgen))
    l = tf.Variable(l0, name="W_cov_q")
    Lt = tf.transpose(tf.scatter_nd(indices, l, shape=(I * I, O)))
    L = tf.reshape(Lt, (O, I, I))

    mu_0 = tf.random_normal((O, I), stddev=std0, seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")
    Q = MultivariateNormalTriL(mu, L)
    return Q


#
# KL divergence calculations
#

def kl_sum(q, p):
    r"""Compute the total KL between (potentially) many distributions.

    I.e. :math:`\sum_i \text{KL}[q_i || p_i]`

    Parameters
    ----------
    q : tf.distributions.Distribution
        A tensorflow Distribution object
    p : tf.distributions.Distribution
        A tensorflow Distribution object

    Returns
    -------
    kl : Tensor
        the result of the sum of the KL divergences of the ``q`` and ``p``
        distibutions.
    """
    kl = tf.reduce_sum(tf.distributions.kl_divergence(q, p))
    return kl


@tf.distributions.RegisterKL(MultivariateNormalTriL, tf.distributions.Normal)
def _kl_gaussian_normal(q, p, name=None):
    """Gaussian-Normal Kullback Leibler divergence calculation.

    Parameters
    ----------
    q : tf.contrib.distributions.MultivariateNormalTriL
        the approximating 'q' distribution(s).
    p : tf.distributions.Normal
        the prior 'p' distribution(s), ``p.scale`` should be a *scalar* value!
    name : str
        name to give the resulting KL divergence Tensor

    Returns
    -------
    KL : Tensor
        the result of KL[q||p].

    """
    assert len(p.scale.shape) == 0, "This KL divergence is only implemented " \
        "for Normal distributions that share a scale parameter for p"
    D = tf.to_float(q.event_shape_tensor())
    n = tf.to_float(q.batch_shape_tensor())
    p_var = p.scale**2
    L = q.scale.to_dense()
    tr = tf.reduce_sum(L * L) / p_var
    # tr = tf.reduce_sum(tf.trace(q.covariance())) / p_var  # Above is faster
    dist = tf.reduce_sum((p.mean() - q.mean())**2) / p_var
    # logdet = n * D * tf.log(p_var) \
    #     - 2 * tf.reduce_sum(q.scale.log_abs_determinant())  # Numerical issue
    logdet = n * D * tf.log(p_var) - _chollogdet(L)
    KL = 0.5 * (tr + dist + logdet - n * D)
    if name:
        KL = tf.identity(KL, name=name)
    return KL


#
# Private module stuff
#

def _chollogdet(L):
    """Log det of a cholesky, where L is (..., D, D)."""
    l = pos(tf.matrix_diag_part(L))  # keep > 0, and no vanashing gradient
    logdet = 2. * tf.reduce_sum(tf.log(l))
    return logdet
