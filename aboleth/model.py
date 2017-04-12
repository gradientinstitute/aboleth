"""Neural Net Construction."""
import tensorflow as tf

from aboleth.layer import compose_layers

#
# Graph Building -- Models and Optimisation
#


def deepnet(X, Y, N, layers, likelihood, n_samples=10, bias_fn=None):
    """Make a supervised Bayesian deep network."""
    Phi = tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1])
    Phi, KL = compose_layers(layers, Phi)
    loss = elbo(Phi, Y, N, KL, likelihood, bias_fn)
    return Phi, loss


def elbo(Phi, Y, N, KL, likelihood, bias_fn=None):
    """Build the evidence lower bound loss."""
    B = N / tf.to_float(tf.shape(Phi)[1])  # Batch amplification factor
    n_samples = tf.to_float(tf.shape(Phi)[0])

    # Just mean over samps for expected log-likelihood
    if not bias_fn:
        ELL = tf.reduce_sum(likelihood(Y, Phi)) / n_samples
    else:
        ELL = tf.reduce_sum(likelihood(Y, Phi) * bias_fn(Y)) / n_samples

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
