"""Model parameter distributions."""
import numpy as np
import tensorflow as tf

from aboleth.util import pos
from aboleth.random import seedgen


#
# Generic prior and posterior classes
#

class Normal:
    """
    Normal (IID) prior/posterior.

    Parameters
    ----------
    mu : Tensor
        mean, shape [d_i, d_o]
    var : Tensor
        variance, shape [d_i, d_o]
    """

    def __init__(self, mu=0., var=1.):
        self.mu = mu
        self.var = var
        self.sigma = tf.sqrt(var)
        self.D = tf.shape(mu)

    def sample(self):
        # Reparameterisation trick
        e = tf.random_normal(self.D, seed=next(seedgen))
        x = self.mu + e * self.sigma
        return x

    def KL(self, p):
        """KL between self and univariate prior, p."""
        KL = 0.5 * (tf.log(p.var) - tf.log(self.var) + self.var / p.var - 1. +
                    (self.mu - p.mu)**2 / p.var)
        return KL


class Gaussian:
    """
    Gaussian prior/posterior.

    Parameters
    ----------
    mu : Tensor
        mean, shape [d_i, d_o]
    L : Tensor
        Cholesky, shape [d_o, d_i, d_i]
    """

    def __init__(self, mu, L):
        self.mu = tf.expand_dims(tf.transpose(mu), 2)  # O x I x 1
        self.L = L  # O x I x I
        self.d = tf.shape(mu)
        self.D = tf.shape(self.mu)

    def sample(self):
        # Reparameterisation trick
        e = tf.random_normal(self.D, seed=next(seedgen))
        x = tf.reshape(self.mu + tf.matmul(self.L, e), self.d)
        return x

    def KL(self, p):
        """KL between self and univariate prior, p."""
        D = tf.to_float(self.D)
        tr = tf.reduce_sum(self.L * self.L) / p.var
        dist = tf.nn.l2_loss(p.mu - tf.reshape(self.mu, self.d)) / p.var
        logdet = D * tf.log(p.var) - _chollogdet(self.L)
        KL = 0.5 * (tr + dist + logdet - D)
        return KL


#
# Streamlined interfaces for initialising the priors and posteriors
#

def norm_prior(dim, var):
    mu = tf.zeros(dim)
    var = pos(tf.Variable(var, name="W_mu_p"))
    P = Normal(mu, var)
    return P


def norm_posterior(dim, var0):
    mu_0 = tf.random_normal(dim, stddev=np.sqrt(var0), seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")

    var_0 = tf.random_gamma(alpha=var0, shape=dim, seed=next(seedgen))
    var = pos(tf.Variable(var_0, name="W_var_q"))

    Q = Normal(mu, var)
    return Q


def gaus_posterior(dim, var0):
    I, O = dim
    sig0 = np.sqrt(var0)

    # Optimize only values in lower triangular
    u, v = np.tril_indices(I)
    indices = (u * I + v)[:, np.newaxis]
    l_0 = np.tile(np.eye(I), [O, 1, 1])[:, u, v].T
    l_0 = l_0 * tf.random_gamma(alpha=sig0, shape=l_0.shape, seed=next(seedgen))
    l = tf.Variable(l_0, name="W_cov_q")
    L = tf.scatter_nd(indices, l, shape=(I * I, O))
    L = tf.transpose(L)
    L = tf.reshape(L, (O, I, I))

    mu_0 = tf.random_normal((I, O), stddev=sig0, seed=next(seedgen))
    mu = tf.Variable(mu_0, name="W_mu_q")
    Q = Gaussian(mu, L)
    return Q


#
# Private module stuff
#

def _chollogdet(L):
    """L is [..., D, D]."""
    l = pos(tf.matrix_diag_part(L))  # Make sure we don't go to zero
    logdet = 2. * tf.reduce_sum(tf.log(l))
    return logdet
