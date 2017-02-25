"""Neural Net Layer tools."""
import numpy as np
import tensorflow as tf

from aboleth.util import pos


#
# Layers
#

def eye():
    """Indentity Layer."""
    def build_eye(X):
        KL = 0.
        return X, KL

    return build_eye


def activation(h=lambda X: X):
    """Activation function layer."""
    def build_activation(X):
        Phi = h(X)
        KL = 0.
        return Phi, KL
    return build_activation


def cat(*layers):
    """Concatenate multiple layers/activations."""
    def build_cat(X):

        Phis, KLs = zip(*[p(X) for p in layers])
        Phi = tf.concat(Phis, axis=1)
        KL = sum(KLs)

        return Phi, KL

    return build_cat


def add(*layers):
    """Add multiple layers/activations."""
    def build_add(X):

        Phis, KLs = zip(*(p(X) for p in layers))
        Phi = sum(Phis)
        KL = sum(KLs)

        return Phi, KL

    return build_add


def dense(output_dim, reg=1., learn_prior=True):
    """Dense (fully connected) linear layer, Bayesian style."""
    def build_dense(X):
        input_dim = int(X.get_shape()[1])
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        # Layer priors
        pW = __Weights(
            mu=tf.zeros(Wdim),
            var=pos(tf.Variable(reg)) * tf.ones(Wdim)
            if learn_prior else reg * tf.ones(Wdim)
        )
        pb = __Weights(
            mu=tf.zeros(bdim),
            var=pos(tf.Variable(reg)) * tf.ones(bdim)
            if learn_prior else reg * tf.ones(bdim)
        )

        # Layer Posteriors
        qW = __Weights(
            mu=tf.Variable(reg * tf.random_normal(Wdim)),
            var=pos(tf.Variable(reg * tf.random_normal(Wdim)))
        )
        qb = __Weights(
            mu=tf.Variable(reg * tf.random_normal(bdim)),
            var=pos(tf.Variable(reg * tf.random_normal(bdim)))
        )

        Phi = tf.matmul(X, qW.sample()) + qb.sample()
        KL = (tf.reduce_sum(qW.KL(pW)) +
              tf.reduce_sum(qb.KL(pb)))
        return Phi, KL
    return build_dense


def randomFourier(n_features, kernel=None):
    """Random fourier feature layer."""
    kernel = kernel if kernel else RBF()

    def build_randomRBF(X):
        input_dim = int(X.get_shape()[1])
        P = kernel.weights(input_dim, n_features)
        XP = tf.matmul(X, P)
        real = tf.cos(XP)
        imag = tf.sin(XP)
        Phi = tf.concat([real, imag], axis=1) / np.sqrt(n_features)
        KL = 0.0
        return Phi, KL

    return build_randomRBF


#
# Random Fourier Kernels
#

class RBF:
    """RBF kernel approximation."""

    def __init__(self, lenscale=1.0):
        self.lenscale = lenscale

    def weights(self, input_dim, n_features):
        P = np.random.randn(input_dim, n_features).astype(np.float32)
        return P / self.lenscale


class Matern(RBF):
    """Matern kernel approximation."""
    def __init__(self, p=1, lenscale=1.0):
        super().__init__(lenscale)
        self.p = p

    def weights(self, input_dim, n_features):
        # p is the matern number (v = p + .5) and the two is a transformation
        # of variables between Rasmussen 2006 p84 and the CF of a Multivariate
        # Student t (see wikipedia). Also see "A Note on the Characteristic
        # Function of Multivariate t Distribution":
        #   http://ocean.kisti.re.kr/downfile/volume/kss/GCGHC8/2014/v21n1/
        #   GCGHC8_2014_v21n1_81.pdf
        # To sample from a m.v. t we use the formula
        # from wikipedia, x = y * np.sqrt(df / u) where y ~ norm(0, I),
        # u ~ chi2(df), then x ~ mvt(0, I, df)
        df = 2 * (self.p + 0.5)
        y = np.random.randn(input_dim, n_features)
        u = np.random.chisquare(df, size=(n_features,))
        P = y * np.sqrt(df / u)
        P = P.astype(np.float32)
        return P / self.lenscale


#
# Private module stuff
#

class __Weights:

    def __init__(self, mu=0., var=1.):
        self.mu = mu
        self.var = var
        self.sigma = tf.sqrt(var)

    def sample(self):
        # Reparameterisation trick
        e = tf.random_normal(self.mu.get_shape())
        x = self.mu + e * self.sigma
        return x

    def KL(self, p):
        KL = 0.5 * (tf.log(p.var) - tf.log(self.var) + self.var / p.var - 1. +
                    (self.mu - p.mu)**2 / p.var)
        return KL
