"""Neural Net Construction."""
import tensorflow as tf


#
# Graph Building
#

def deepnet(X, Y, N, layers, likelihood, n_samples=10):
    """Make a supervised Bayesian deep network.

    Note: This simply combines calls to ``deepnet`` and ``elbo``.
    """
    Phi = [X for k in range(n_samples)]
    KL = 0.
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl
    loss = elbo(Phi, Y, N, KL, likelihood)
    return Phi, loss


def elbo(F, Y, N, KL, likelihood):
    """Build the evidence lower bound loss."""
    B = N / tf.to_float(tf.shape(F)[0])  # Batch amplification factor
    ELL = tf.reduce_sum([likelihood(Y, phi) for phi in F]) / len(F)
    l = - B * ELL + KL
    return l


def log_prob(Y, likelihood, Phi):
    """Build the log probability density of the model."""
    log_prob = tf.reduce_mean([likelihood(Y, p) for p in Phi], axis=0)
    return log_prob
