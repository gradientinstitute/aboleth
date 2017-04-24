"""Neural Net Construction."""
import tensorflow as tf

from aboleth.layer import compose_layers

#
# Graph Building -- Models and Optimisation
#


def deepnet(X, Y, N, layers, likelihood, n_samples=10, like_weights=None):
    """Make a supervised Bayesian deep network.

    Parameters
    ----------
    X: ndarray, Tensor
        the covariates of shape (samples, dimensions)
    Y: ndarray, Tensor
        the targets of shape (samples, tasks)
    N: int, Tensor
        the total size of the dataset (i.e. samples)
    layers: sequence
        a list (or sequence) of layers defining the neural net. See also the
        ``layers`` module
    n_samples: int
        the number of samples to use for evaluating the expected log-likelihood
        in the objective function. This replicates the whole network for each
        sample
    like_weights: callable, ndarray, Tensor
        weights to apply to each sample in the expected log likelihood - the
        result of this should sum up to N. This should be an array of shape
        (samples, 1) or can be called as ``like_weights(Y)`` and should return
        a (samples, 1) array
    """
    Phi = tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1])
    Phi, KL = compose_layers(layers, Phi)
    loss = elbo(Phi, Y, N, KL, likelihood, like_weights)
    return Phi, loss


def elbo(Phi, Y, N, KL, likelihood, like_weights=None):
    """Build the evidence lower bound loss."""
    B = N / tf.to_float(tf.shape(Phi)[1])  # Batch amplification factor
    n_samples = tf.to_float(tf.shape(Phi)[0])

    # Just mean over samps for expected log-likelihood
    if like_weights is None:
        ELL = tf.reduce_sum(likelihood(Y, Phi)) / n_samples
    elif callable(like_weights):
        ELL = tf.reduce_sum(likelihood(Y, Phi) * like_weights(Y)) / n_samples
    else:
        ELL = tf.reduce_sum(likelihood(Y, Phi) * like_weights) / n_samples

    l = - B * ELL + KL
    return l


#
# Graph Building -- Prediction and evaluation
#


def log_prob(Y, likelihood, Phi):
    """Build the log probability density of the model for each observation.

    NOTE: This uses ``n_samples`` (from ``deepnet``) of the posterior to build
        up the log probability for each sample.
    """
    log_prob = tf.reduce_mean(likelihood(Y, Phi), axis=0)
    return log_prob


def average_log_prob(Y, likelihood, Phi):
    """Build the mean log probability of the model over the observations.

    NOTE: This only returns one posterior sample of this log probability.
    """
    lp = tf.reduce_mean(likelihood(Y, Phi[0]))
    return lp
