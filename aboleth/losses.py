"""Network loss functions."""
import tensorflow as tf


def elbo(likelihood, Y, N, KL, like_weights=None):
    r"""Build the evidence lower bound loss for a neural net.

    Parameters
    ----------
    likelihood : tf.distributions.Distribution
        the likelihood object that takes neural network(s) as an input. The
        ``batch_shape`` of this object should be ``(n_samples, N, ...)``, where
        ``n_samples`` is the number of likelihood samples (defined by
        ab.InputLayer) and ``N`` is the number of observations (can be ``?`` if
        you are using a placeholder and mini-batching).
    Y : ndarray, Tensor
        the targets of shape ``(N, tasks)``.
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    KL : float, Tensor
        the Kullback Leibler divergence between the posterior and prior
        parameters of the model (:math:`\text{KL}[q\|p]`).
    like_weights : ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be a tensor/array of shape ``(N,)`` (or a shape that
        prevents broadcasting).

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


def max_posterior(likelihood, Y, regulariser, like_weights=None):
    r"""Build maximum a-posteriori (MAP) loss for a neural net.

    Parameters
    ----------
    likelihood : tf.distributions.Distribution
        the likelihood object that takes neural network(s) as an input. The
        ``batch_shape`` of this object should be ``(n_samples, N, ...)``, where
        ``n_samples`` is the number of likelihood samples (defined by
        ab.InputLayer) and ``N`` is the number of observations (can be ``?`` if
        you are using a placeholder and mini-batching).
    Y : ndarray, Tensor
        the targets of shape ``(N, tasks)``.
    regulariser : float, Tensor
        the regulariser on the parameters of the model to penalise model
        complexity.
    like_weights : ndarray, Tensor
        weights to apply to each observation in the expected log likelihood.
        This should be a tensor/array of shape ``(N,)`` (or a shape that
        prevents broadcasting).

    Returns
    -------
    map : Tensor
        the loss function of the MAP neural net.

    """
    # Get the batch size to average the likelihood over
    M = tf.to_float(likelihood.batch_shape_tensor()[1])

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
    if not (likelihood.batch_shape[2:] == Y.shape[1:]):
        raise ValueError("Incompatible target and likelihood shapes.")

    log_prob = likelihood.log_prob(Y)
    log_prob_dims = log_prob.shape

    if like_weights is not None:
        log_prob *= like_weights
        assert log_prob.shape == log_prob_dims  # Guard against broadcasting

    sumlike = tf.reduce_sum(log_prob)

    return sumlike
