"""Model parameter distributions."""
import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

from aboleth.util import pos
from aboleth.random import seedgen


#
# Generic prior and posterior classes
#

class ParameterDistribution:
    """Abstract base class for parameter distribution objects."""

    def __init__(self):
        """Construct a ParameterDistibution object."""
        raise NotImplementedError('Abstract base class only.')

    def sample(self):
        """Draw a random sample from the distribution."""
        raise NotImplementedError('Abstract base class only.')


class Normal(ParameterDistribution):
    """
    Normal (IID) prior/posterior.

    Parameters
    ----------
    mu : Tensor
        mean, shape (d_in, d_out)
    var : Tensor
        variance, shape (d_in, d_out)

    """

    def __init__(self, mu=0., var=1.):
        """Construct a Normal distribution object."""
        self.mu = mu
        self.var = var
        self.sigma = tf.sqrt(var)
        self.d = mu.shape

    def sample(self, e=None):
        """Draw a random sample from this object.

        Parameters
        ----------
        e : ndarray, Tensor, optional
            the random standard-Normal samples to transform to yeild samples
            from this distrubution. These must be of shape (d_in, ...). If
            this is none, these are generated in this method.

        Returns
        -------
        x : Tensor
            a sample of shape (d_in, d_out), or ``e.shape`` if provided

        """
        # Reparameterisation trick
        if e is None:
            e = tf.random_normal(self.d, seed=next(seedgen))
        x = self.mu + e * self.sigma

        return x


class Gaussian(ParameterDistribution):
    """
    Gaussian prior/posterior.

    Parameters
    ----------
    mu : Tensor
        mean, shape (d_in, d_out)
    L : Tensor
        Cholesky of the covariance matrix, shape (d_out, d_in, d_in)
    eps : ndarray, Tensor, optional
        random draw from a unit normal if you want to "fix" the sampling, this
        should be of shape (d_in, d_out). If this is ``None`` then a new random
        draw is used for every call to sample().

    """

    def __init__(self, mu, L):
        """Construct a Normal distribution object."""
        self.mu = mu
        self.L = L  # O x I x I
        self.d = mu.shape

    def sample(self, e=None):
        """Draw a random sample from this object.

        Parameters
        ----------
        e : ndarray, Tensor, optional
            the random standard-Normal samples to transform to yeild samples
            from this distrubution. These must be of shape (d_in, ...). If
            this is none, these are generated in this method.

        Returns
        -------
        x : Tensor
            a sample of shape (d_in, d_out), or ``e.shape`` if provided

        """
        # Reparameterisation trick
        mu = self.transform_w(self.mu)
        if e is None:
            e = tf.random_normal(mu.shape, seed=next(seedgen))
        else:
            e = self.transform_w(e)
        x = self.itransform_w(mu + tf.matmul(self.L, e))

        return x

    @staticmethod
    def transform_w(w):
        """Transform a weight matrix, (d_in, d_out) -> (d_out, d_in, 1)."""
        wt = tf.expand_dims(tf.transpose(w), 2)  # O x I x 1
        return wt

    @staticmethod
    def itransform_w(wt):
        """Un-transform a weight matrix, (d_out, d_in, 1) -> (d_in, d_out)."""
        w = tf.transpose(wt[:, :, 0])
        return w


#
# Streamlined interfaces for initialising the priors and posteriors
#

def norm_prior(dim, var):
    """Initialise a prior (zero mean, diagonal) Normal distribution.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    var : float
        the prior variance of this distribution.

    Returns
    -------
    P : Normal
        the initialised prior Normal object.

    Note
    ----
    This will make a tf.Variable on the variance of the prior that is
    initialised with ``var``.

    """
    mu = tf.zeros(dim)
    var = pos(tf.Variable(var, name="W_mu_p"))
    P = Normal(mu, var)
    return P


def norm_posterior(dim, var0):
    """Initialise a posterior (diagonal) Normal distribution.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    var0 : float
        the initial (unoptimized) variance of this distribution.

    Returns
    -------
    Q : Normal
        the initialised posterior Normal object.

    Note
    ----
    This will make tf.Variables on the randomly initialised mean and variance
    of the posterior. The initialisation of the mean is from a Normal with zero
    mean, and ``var0`` variance, and the initialisation of the variance is from
    a gamma distribution with an alpha of ``var0`` and a beta of 1.

    """
    mu_0 = tf.random_normal(dim, stddev=tf.sqrt(var0), seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")

    var_0 = tf.random_gamma(alpha=var0, shape=dim, seed=next(seedgen))
    var = pos(tf.Variable(var_0, name="W_var_q"))

    Q = Normal(mu, var)
    return Q


def gaus_posterior(dim, var0):
    """Initialise a posterior Gaussian distribution with a diagonal covariance.

    Even though this is initialised with a diagonal covariance, a full
    covariance will be learned, using a lower triangular Cholesky
    parameterisation.

    Parameters
    ----------
    dim : tuple or list
        the dimension of this distribution.
    var0 : float
        the initial (unoptimized) diagonal variance of this distribution.

    Returns
    -------
    Q : Gaussian
        the initialised posterior Gaussian object.

    Note
    ----
    This will make tf.Variables on the randomly initialised mean and covariance
    of the posterior. The initialisation of the mean is from a Normal with zero
    mean, and ``var0`` variance, and the initialisation of the variance is from
    a gamma distribution with an alpha of ``var0`` and a beta of 1.

    """
    I, O = dim
    sig0 = np.sqrt(var0)

    # Optimize only values in lower triangular
    u, v = np.tril_indices(I)
    indices = (u * I + v)[:, np.newaxis]
    l0 = np.tile(np.eye(I), [O, 1, 1])[:, u, v].T
    l0 = l0 * tf.random_gamma(alpha=var0, shape=l0.shape, seed=next(seedgen))
    l = tf.Variable(l0, name="W_cov_q")
    Lt = tf.transpose(tf.scatter_nd(indices, l, shape=(I * I, O)))
    L = tf.reshape(Lt, (O, I, I))

    mu_0 = tf.random_normal((I, O), stddev=sig0, seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")
    Q = Gaussian(mu, L)
    return Q


#
# KL divergence calculations
#


@dispatch(Normal, Normal)
def kl_qp(q, p):
    """Normal-Normal Kullback Leibler divergence calculation.

    Parameters
    ----------
    q : Normal
        the approximating 'q' distribution.
    p : Normal
        the prior 'p' distribution.

    Returns
    -------
    KL : Tensor
        the result of KL[q||p].

    """
    KL = 0.5 * (tf.log(p.var) - tf.log(q.var) + q.var / p.var - 1. +
                (q.mu - p.mu)**2 / p.var)
    KL = tf.reduce_sum(KL)
    return KL


@dispatch(Gaussian, Normal)  # noqa
def kl_qp(q, p):
    """Gaussian-Normal Kullback Leibler divergence calculation.

    Parameters
    ----------
    q : Gaussian
        the approximating 'q' distribution.
    p : Normal
        the prior 'p' distribution.

    Returns
    -------
    KL : Tensor
        the result of KL[q||p].

    """
    D, n = tf.to_float(q.d[0]), tf.to_float(q.d[1])
    tr = tf.reduce_sum(q.L * q.L) / p.var
    dist = tf.reduce_sum((p.mu - q.mu)**2) / p.var
    logdet = n * D * tf.log(p.var) - _chollogdet(q.L)
    KL = 0.5 * (tr + dist + logdet - n * D)
    return KL


@dispatch(Gaussian, Gaussian)  # noqa
def kl_qp(q, p):
    """Gaussian-Gaussian Kullback Leibler divergence calculation.

    Parameters
    ----------
    q : Gaussian
        the approximating 'q' distribution.
    p : Gaussian
        the prior 'p' distribution.

    Returns
    -------
    KL : Tensor
        the result of KL[q||p].

    """
    D, n = tf.to_float(q.d[0]), tf.to_float(q.d[1])
    qCipC = tf.cholesky_solve(p.L, tf.matmul(q.L, q.L, transpose_b=True))
    tr = tf.reduce_sum(tf.trace(qCipC))
    md = q.transform_w(p.mu - q.mu)
    dist = tf.reduce_sum(md * tf.cholesky_solve(p.L, md))
    logdet = _chollogdet(p.L) - _chollogdet(q.L)
    KL = 0.5 * (tr + dist + logdet - n * D)
    return KL


#
# Private module stuff
#

def _chollogdet(L):
    """Log det of a cholesky, where L is (..., D, D)."""
    l = tf.maximum(tf.matrix_diag_part(L), 1e-15)  # Make sure we don't go to 0
    logdet = 2. * tf.reduce_sum(tf.log(l))
    return logdet
