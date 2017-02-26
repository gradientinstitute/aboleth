"""Neural Net Construction."""
import tensorflow as tf


def bayesmodel(X, Y, N, layers, likelihood, n_samples=10):
    """Make a supervised Bayesian model.

    Note: This simply combines calls to ``deepnet`` and ``elbo``.
    """
    Phi, KL = deepnet(X, layers)
    loss = elbo(Phi, Y, N, KL, likelihood, n_samples)
    return Phi, loss


def deepnet(X, layers):
    """Build a neural net."""
    Phi = X
    KL = 0.
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl
    return Phi, KL


def elbo(F, Y, N, KL, likelihood, n_samples=10):
    """Evaluate the evidence lower bound."""
    # Expected log-likelihood with MC integration (reparameterization trick)
    ELL = 0.
    for _ in range(n_samples):
        ll = likelihood(Y, F)
        ELL += tf.reduce_sum(ll) / n_samples

    B = N / tf.to_float(tf.shape(F)[0])
    l = - B * ELL + KL
    return l


def density(Phi, Y, likelihood, n_samples=10):
    """
    Something about how this is going to work.
    """
    samples = [likelihood(Y, Phi) for _ in range(n_samples)]
    density = tf.reduce_mean(samples, axis=0)
    return density
