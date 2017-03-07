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
        Phi = tf.map_fn(h, X)
        KL = 0.
        return Phi, KL
    return build_activation


def fork(replicas=2):
    """Fork an input into multiple, unmodified, outputs."""
    def build_fork(X):
        KL = 0.
        return [X for _ in range(replicas)], KL

    return build_fork


def lmap(*layers):
    """Map multiple layers to multiple inputs (after forking)."""
    def build_lmap(Xs):
        if len(Xs) != len(layers):
            raise ValueError("Number of layers and inputs not the same!")
        Phis, KLs = zip(*map(lambda p, X: p(X), layers, Xs))
        KL = sum(KLs)
        return Phis, KL

    return build_lmap


def cat():
    """Join multiple inputs by concatenation."""
    def build_cat(Xs):
        Phi = tf.concat(Xs, axis=2)
        KL = 0.
        return Phi, KL

    return build_cat


def add():
    """Join multiple inputs by addition."""
    def build_add(Xs):
        Phi = tf.add_n(Xs)
        KL = 0.
        return Phi, KL

    return build_add


def dense_var(output_dim, reg=1., learn_prior=True, full=False):
    """Dense (fully connected) linear layer, with variational inference."""
    def build_dense(X):
        """X is a rank 3 tensor, [n_samples, N, D]."""
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        # Layer priors
        pW = _NormPrior(dim=Wdim, var=reg, learn_var=learn_prior)
        pb = _NormPrior(dim=bdim, var=reg, learn_var=learn_prior)

        # Layer Posterior samples
        qW = (_GausPosterior(dim=Wdim, prior_var=reg) if full else
              _NormPosterior(dim=Wdim, prior_var=reg))
        qb = _NormPosterior(dim=bdim, prior_var=reg)  # TODO: keep independent?

        # Linear layer
        Wsamples = _sample(qW, n_samples)
        bsamples = tf.expand_dims(_sample(qb, n_samples), 1)
        Phi = tf.matmul(X, Wsamples) + bsamples

        # Regularizers
        KL = tf.reduce_sum(qW.KL(pW)) + tf.reduce_sum(qb.KL(pb))

        return Phi, KL

    return build_dense


def dense_map(output_dim, l1_reg=1., l2_reg=1.):
    """Dense (fully connected) linear layer, with MAP inference."""

    def build_dense_map(X):
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        W = tf.Variable(tf.random_normal(Wdim))
        b = tf.Variable(tf.random_normal(bdim))

        # Linear layer, don't want to copy Variable, so map
        Phi = tf.map_fn(lambda x: tf.matmul(x, W), X)

        # Regularizers
        l1, l2 = 0, 0
        if l2_reg > 0:
            l2 = l2_reg * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
        if l1_reg > 0:
            l1 = l1_reg * (_l1_loss(W) + _l1_loss(b))
        pen = l1 + l2

        return Phi, pen

    return build_dense_map


def randomFourier(n_features, kernel=None):
    """Random fourier feature layer."""
    kernel = kernel if kernel else RBF()

    def build_randomFF(X):
        n_samples, input_dim = _get_dims(X)

        # Random weights, copy faster than map here
        P = kernel.weights(input_dim, n_features)
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        real = tf.cos(XP)
        imag = tf.sin(XP)
        Phi = tf.concat([real, imag], axis=2) / np.sqrt(n_features)
        KL = 0.0
        return Phi, KL

    return build_randomFF


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

    def __init__(self, lenscale=1.0, p=1):
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

class _Normal:

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


class _Gaussian:
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


class _NormPrior(_Normal):

    def __init__(self, dim, var, learn_var):
        mu = tf.zeros(dim)
        var = pos(tf.Variable(var)) if learn_var else var
        super().__init__(mu, var)


class _NormPosterior(_Normal):

    def __init__(self, dim, prior_var):
        mu = tf.Variable(tf.sqrt(prior_var) * tf.random_normal(dim))
        var = pos(tf.Variable(prior_var * tf.random_normal(dim)))
        super().__init__(mu, var)


class _GausPosterior(_Gaussian):

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


def _l1_loss(X):
    l1 = tf.reduce_sum(tf.abs(X))
    return l1


def _chollogdet(L):
    """L is [..., D, D]."""
    l = tf.map_fn(tf.diag_part, L)
    logdet = 2. * tf.reduce_sum(tf.log(l))
    return logdet


def _get_dims(X):
        n_samples, input_dim = X.shape[0], X.shape[2]
        return int(n_samples), int(input_dim)


def _sample(dist, n_samples):
    samples = tf.stack([dist.sample() for _ in range(n_samples)])
    return samples
