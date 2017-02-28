"""Neural Net Construction."""
import tensorflow as tf


#
# Graph Building
#

def bayesmodel(X, Y, N, layers, likelihood):
    """Make a supervised Bayesian model.

    Note: This simply combines calls to ``deepnet`` and ``elbo``.
    """
    Phi, KL = deepnet(X, layers)
    loss = elbo(Phi, Y, N, KL, likelihood)
    return Phi, loss


def deepnet(X, layers):
    """Build a neural net."""
    Phi = X
    KL = 0.
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl
    return Phi, KL


def elbo(F, Y, N, KL, likelihood):
    """Build the evidence lower bound loss."""
    B = N / tf.to_float(tf.shape(F)[0])  # Batch amplification factor
    ELL = tf.reduce_sum(likelihood(Y, F))  # YOSO, you only sample once
    l = - B * ELL + KL
    return l


def log_prob(Y, likelihood, Phi):
    """Build the log probability density of the model."""
    log_prob = likelihood(Y, Phi)
    return log_prob
