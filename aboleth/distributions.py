"""Model parameter distributions."""
import tensorflow as tf

from aboleth.util import pos


#
# Generic prior and posterior classes
#

class Normal:
    """
    Normal (IID) prior/posterior.

    Parameters:
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
        e = tf.random_normal(self.D)
        x = self.mu + e * self.sigma
        return x

    def KL(self, p):
        KL = 0.5 * (tf.log(p.var) - tf.log(self.var) + self.var / p.var - 1. +
                    (self.mu - p.mu)**2 / p.var)
        return KL


class Gaussian:
    """
    Gaussian prior/posterior.

    Parameters:
        mu : Tensor
            mean, shape [d_i, d_o]
        L : Tensor
            Cholesky, shape [d_o, d_i, d_i]
    """

    def __init__(self, mu, L):
        self.mu = tf.expand_dims(tf.transpose(mu), 1)  # O x 1 x I
        self.L = L  # O x I x I
        self.d = tf.shape(mu)
        self.D = tf.shape(self.mu)

    def sample(self):
        e = tf.random_normal(self.D)
        x = tf.reshape(self.mu + tf.matmul(e, self.L), self.d)
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

class NormPrior(Normal):

    def __init__(self, dim, var, learn_var):
        mu = tf.zeros(dim)
        var = pos(tf.Variable(var)) if learn_var else var
        super().__init__(mu, var)


class NormPosterior(Normal):

    def __init__(self, dim, prior_var):
        mu = tf.Variable(tf.sqrt(prior_var) * tf.random_normal(dim))
        var = pos(tf.Variable(prior_var * tf.random_normal(dim)))
        super().__init__(mu, var)


class GausPosterior(Gaussian):

    def __init__(self, dim, prior_var):
        I, O = dim
        mu, L = [], []
        for o in range(O):
            Le = tf.eye(I) * tf.sqrt(prior_var)  # TODO make random diagonal
            e = tf.random_normal((I, 1))
            mu.append(tf.Variable(tf.matmul(Le, e)))
            L.append(tf.matrix_band_part(tf.Variable(Le), -1, 0))

        mu = tf.concat(mu, axis=1)
        L = tf.stack(L)

        super().__init__(mu=mu, L=L)


#
# Private module stuff
#

def _chollogdet(L):
    """L is [..., D, D]."""
    l = tf.map_fn(tf.diag_part, L)
    logdet = 2. * tf.reduce_sum(tf.log(l))
    return logdet
