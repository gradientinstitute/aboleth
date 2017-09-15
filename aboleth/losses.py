"""Network loss functions."""
import tensorflow as tf


def elbo(likelihood, Y, N, KL, like_weights=None):
    """Build the evidence lower bound loss for a neural net.

    Parameters
    ----------
    likelihood : tf.distributions.Distribution
        the likelihood object that takes neural network(s) as an input. The
        ``batch_shape`` of this object should be (n_samples, N, ...), where
        ``n_samples`` is the number of likelihood samples (defined by
        ab.InputLayer) and ``N`` is the number of observations (can be ``?`` if
        you are using a placeholder and mini-batching).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    like_weights : callable, ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be an array of shape (N,) or can be called as
        ``like_weights(Y)`` and should return a (N,) array.

    Returns
    -------
    nelbo : Tensor
        the loss function of the Bayesian neural net (negative ELBO).

    """
    rank = len(likelihood.batch_shape)
    assert rank >= 2, "likelihood should be at least a rank 2 tensor! The " \
        "first dimension the network samples, the second the observations."

    # Batch amplification factor
    B = N / tf.to_float(likelihood.batch_shape_tensor()[1])

    # averaging over samples
    n_samples = tf.to_float(likelihood.batch_shape_tensor()[0])

    # Just mean over samps for expected log-likelihood
    ELL = _sum_likelihood(likelihood, Y, like_weights) / n_samples

    # negative ELBO is batch weighted ELL and KL
    nELBO = - B * ELL + KL

    return nELBO


def max_posterior(likelihood, Y, regulariser, like_weights=None,
                  first_axis_is_obs=True):
    """Build maximum a-posteriori (MAP) loss for a neural net.

    Parameters
    ----------
    likelihood : tf.distributions.Distribution
        the likelihood object that takes neural network(s) as an input. The
        ``batch_shape`` of this object should be (n_samples, N, ...), where
        ``n_samples`` is the number of likelihood samples (defined by
        ab.InputLayer) and ``N`` is the number of observations (can be ``?`` if
        you are using a placeholder and mini-batching).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    like_weights : callable, ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be an array of shape (N,) or can be called as
        ``like_weights(Y)`` and should return a (N,) array.
    first_axis_is_obs : bool
        indicates if the first axis indexes the observations/data or not. This
        will be True if the likelihood outputs a ``batch_shape`` of (N, tasks)
        or False if ``batch_shape`` is (n_samples, N, tasks).

    Returns
    -------
    map : Tensor
        the loss function of the MAP neural net.

    """
    # Get the batch size to average the likelihood over
    obs_ax = 0 if first_axis_is_obs else 1
    M = tf.to_float(likelihood.batch_shape_tensor()[obs_ax])

    # Average likelihood for batch
    AVLL = _sum_likelihood(likelihood, Y, like_weights) / M

    # MAP objective
    MAP = - AVLL + regulariser

    return MAP


#
# Private module utilities
#

def _sum_likelihood(likelihood, Y, like_weights):
    """Sum the log-likelihood of the Y's under the model."""
    log_prob = likelihood.log_prob(Y)
    if callable(like_weights):
        log_prob *= like_weights(Y)
    elif like_weights is not None:
        log_prob *= like_weights

    sumlike = tf.reduce_sum(log_prob)

    return sumlike
