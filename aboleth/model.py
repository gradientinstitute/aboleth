"""Neural Net Construction."""
import tensorflow as tf


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
    ELL = ell(F, Y, likelihood, n_samples)
    B = N / tf.to_float(tf.shape(F)[0])
    l = - B * ELL + KL
    return l


def ell(F, Y, likelihood, n_samples):
    """Expected log likelihood, sample the log likelihood."""
    ELL = 0.
    for _ in range(n_samples):
        ll = likelihood(Y, F)
        ELL += tf.reduce_sum(ll)
    ELL = ELL / n_samples
    return ELL
