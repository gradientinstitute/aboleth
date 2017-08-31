"""Network loss functions."""
import tensorflow as tf


def elbo(Net, Y, N, KL, likelihood, like_weights=None):
    """Build the evidence lower bound loss for a neural net.

    Parameters
    ----------
    Net : ndarray, Tensor
        the neural net featues of shape (n_samples, N, tasks).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    likelihood : Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the :ref:`likelihoods` module.
    like_weights : callable, ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be an array of shape (N, 1) or can be called as
        ``like_weights(Y)`` and should return a (N, 1) array.

    Returns
    -------
    nelbo : Tensor
        the loss function of the Bayesian neural net (negative ELBO).

    """
    rank = len(Net.shape)
    assert rank > 2, "We need a Tensor of at least rank 3 for Bayesian models!"

    B = N / tf.to_float(tf.shape(Net)[1])  # Batch amplification factor
    n_samples = tf.to_float(Net.shape[0])  # averaging over samples

    # Just mean over samps for expected log-likelihood
    ELL = _sum_likelihood(Y, Net, likelihood, like_weights) / n_samples

    # negative ELBO is batch weighted ELL and KL
    nELBO = - B * ELL + KL

    return nELBO


def max_posterior(Net, Y, regulariser, likelihood, like_weights=None,
                  first_axis_is_obs=True):
    """Build maximum a-posteriori (MAP) loss for a neural net.

    Parameters
    ----------
    Net : ndarray, Tensor
        the neural net featues of shape (N, tasks) or (n_samples, N, tasks).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    likelihood : Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the :ref:`likelihoods` module.
    like_weights : callable, ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be an array of shape (N, 1) or can be called as
        ``like_weights(Y)`` and should return a (N, 1) array.
    first_axis_is_obs : bool
        indicates if the first axis indexes the observations/data or not. This
        will be True if ``Net`` is of shape (N, tasks) or False if ``Net`` is
        of shape (n_samples, N, tasks).

    Returns
    -------
    map : Tensor
        the loss function of the MAP neural net.

    """
    # Get the batch size to average the likelihood over
    M = tf.to_float(tf.shape(Net)[0 if first_axis_is_obs else 1])

    # Average likelihood for batch
    AVLL = _sum_likelihood(Y, Net, likelihood, like_weights) / M

    # MAP objective
    MAP = - AVLL + regulariser

    return MAP


#
# Private module utilities
#

def _sum_likelihood(Y, Net, likelihood, like_weights):
    """Sum the log-likelihood of the Y's under the model."""
    like = likelihood(Y, Net)
    if callable(like_weights):
        like *= like_weights(Y)
    elif like_weights is not None:
        like *= like_weights

    sumlike = tf.reduce_sum(like)

    return sumlike
