"""Neural Net Construction."""
import tensorflow as tf


#
# Graph Building
#

def deepnet(X, Y, N, layers, likelihood, n_samples=10, bias_fn=None):
    """Make a supervised Bayesian deep network."""
    Phi = tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1])
    KL = 0.
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl
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


def log_prob(Y, likelihood, Phi):
    """Build the log probability density of the model using ``n_samples``."""
    log_prob = tf.reduce_mean(likelihood(Y, Phi), axis=0)
    return log_prob


def predict_nlp(Y, likelihood, Phi):
    """Build the mean negative log probability for one sample."""
    nlp = - tf.reduce_mean(likelihood(Y, Phi[0]))
    return nlp
