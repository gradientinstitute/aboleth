import tensorflow as tf

def deepnet(X, layers):

    Phi = X
    KL = 0.0
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl

    return Phi, KL

def elbo(f, Y, N, KL, likelihood, n_samples):
    ELL = ell(f, Y, likelihood, n_samples)
    B = N / tf.to_float(tf.shape(f)[0])
    l = - B * ELL + KL
    return l

def ell(f, Y, likelihood, n_samples):
    ELL = 0.
    for _ in range(n_samples):
        ll = likelihood(Y, f)
        ELL += tf.reduce_sum(ll)
    ELL = ELL / n_samples
    return ELL

def density(Phi, Y, likelihood, n_samples):
    """
    Something about how this is going to work.
    """
    samples = [likelihood(Y, Phi) for _ in range(n_samples)]
    density = tf.reduce_mean(samples, axis=0)
    return density
