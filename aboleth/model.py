"""Network construction and evaluation."""
import tensorflow as tf

# from aboleth.layer import compose_layers


#
# Graph Building -- Models and Optimisation
#

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
        neural net, see the ``likelihood`` module.
    like_weights : callable, ndarray, Tensor
        weights to apply to each sample in the expected log likelihood. This
        should be an array of shape (samples, 1) or can be called as
        ``like_weights(Y)`` and should return a (samples, 1) array.

    Returns
    -------
    elbo : Tensor
        the loss function of the Bayesian neural net.
    """
    B = N / tf.to_float(tf.shape(Net)[1])  # Batch amplification factor
    n_samples = tf.to_float(tf.shape(Net)[0])

    # Just mean over samps for expected log-likelihood
    if like_weights is None:
        ELL = tf.reduce_sum(likelihood(Y, Net)) / n_samples
    elif callable(like_weights):
        ELL = tf.reduce_sum(likelihood(Y, Net) * like_weights(Y)) / n_samples
    else:
        ELL = tf.reduce_sum(likelihood(Y, Net) * like_weights) / n_samples

    l = - B * ELL + KL
    return l

#
# Graph Building -- Prediction and evaluation
#


def log_prob(Y, likelihood, Net):
    """Build the log probability density of the model for each observation.

    Parameters
    ----------
    Y: ndarray, Tensor
        the targets of shape (N, tasks).
    likelihood: Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the ``likelihood`` module.
    Net: Tensor
        the neural net featues of shape (n_samples, N, tasks).

    Returns
    -------
    logp : Tensor
        ``n_samples`` of the log probability of each ``Y`` under the model.
        This is of shape (n_samples, N).
    """
    log_prob = likelihood(Y, Net)
    return log_prob
