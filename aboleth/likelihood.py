"""Output likelihoods."""
import numpy as np
import tensorflow as tf

from aboleth.util import pos


def normal(variance):
    """Normal log-likelihood.

    Parameters
    ----------
    variance : float, Tensor
        the variance of the Normal likelihood, this can be made a tf.Variable
        if you want to learn this.

    Returns
    -------
    loglike : callable
        build the log likelihood graph of this distribution
    """
    def loglike(x, f):
        ll = -0.5 * (tf.log(2 * variance * np.pi) + (x - f)**2 / variance)
        return ll
    return loglike


def bernoulli():
    """Bernoulli log-likelihood.

    Returns
    -------
    loglike : callable
        build the log likelihood graph of this distribution
    """
    def loglike(x, f):
        ll = x * tf.log(pos(f)) + (1 - x) * tf.log(pos(1 - f))
        return ll
    return loglike


def categorical():
    """Categorical, or Generalized Bernoulli log-likelihood.

    Returns
    -------
    loglike : callable
        build the log likelihood graph of this distribution
    """
    def loglike(x, f):
        # sum along last axis, which is assumed to be the `tasks` axis
        ll = tf.reduce_sum(x * tf.log(pos(f)), axis=-1)
        return ll

    return loglike


def binomial(n):
    """Binomial log-likelihood.

    Parameters
    ----------
    n : float, ndarray, Tensor
        the number of trials of this binomial distribution

    Returns
    -------
    loglike : callable
        build the log likelihood graph of this distribution
    """
    def loglike(x, f):
        bincoef = tf.lgamma(n + 1) - tf.lgamma(x + 1) - tf.lgamma(n - x + 1)
        ll = bincoef + x * tf.log(pos(f)) + (n - x) * tf.log(pos(1 - f))
        return ll
    return loglike
