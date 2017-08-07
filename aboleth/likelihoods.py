"""Target/Output likelihoods."""
import numpy as np
import tensorflow as tf

from aboleth.util import pos


class Likelihood:
    """Abstract base class for likelihood objects."""

    def __call__(self, y, f):
        """Build the log likelihood.

        See: _loglike.

        """
        ll = self._loglike(y, f)
        return ll

    def _loglike(self, y, f):
        """Build the log likelihood.

        Parameters
        ----------
        y : Tensor
            the target variable of shape (N, tasks)
        f : Tensor
            the latent function output from the network of shape (N, tasks)

        """
        raise NotImplementedError('Abstract base class only.')


class Normal(Likelihood):
    """Normal log-likelihood.

    Parameters
    ----------
    variance : float, Tensor
        the variance of the Normal likelihood, this can be made a tf.Variable
        if you want to learn this.

    """

    def __init__(self, variance):
        """Construct an instance of a Normal likelihood."""
        self.variance = variance

    def _loglike(self, y, f):
        """Build the log likelihood.

        Parameters
        ----------
        y : Tensor
            the target variable of shape (N, tasks)
        f : Tensor
            the latent function output from the network of shape (N, tasks)

        """
        ll = -0.5 * (tf.log(2 * self.variance * np.pi) +
                     (y - f)**2 / self.variance)
        return ll


class Bernoulli(Likelihood):
    """Bernoulli log-likelihood."""

    def _loglike(self, y, f):
        """Build the log likelihood.

        Parameters
        ----------
        y : Tensor
            the target variable of shape (N, tasks)
        f : Tensor
            the latent function output from the network of shape (N, tasks)

        """
        ll = y * tf.log(pos(f)) + (1 - y) * tf.log(pos(1 - f))
        return ll


class Categorical(Likelihood):
    """Categorical, or Generalized Bernoulli log-likelihood."""

    def _loglike(self, y, f):
        """Build the log likelihood.

        Parameters
        ----------
        y : Tensor
            the target variable of shape (N, tasks)
        f : Tensor
            the latent function output from the network of shape (N, tasks)

        """
        # sum along last axis, which is assumed to be the `tasks` axis
        ll = tf.reduce_sum(y * tf.log(pos(f)), axis=-1)
        return ll


class Binomial(Likelihood):
    """Binomial log-likelihood.

    Parameters
    ----------
    n : float, ndarray, Tensor
        the number of trials of this binomial distribution

    """

    def __init__(self, n):
        """Construct an instance of a Binomial likelihood."""
        self.n = n

    def _loglike(self, y, f):
        """Build the log likelihood.

        Parameters
        ----------
        y : Tensor
            the target variable of shape (N, tasks)
        f : Tensor
            the latent function output from the network of shape (N, tasks)

        """
        bincoef = tf.lgamma(self.n + 1) - tf.lgamma(y + 1) \
            - tf.lgamma(self.n - y + 1)
        ll = bincoef + y * tf.log(pos(f)) + (self.n - y) * tf.log(pos(1 - f))
        return ll
