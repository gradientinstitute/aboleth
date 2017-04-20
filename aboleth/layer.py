"""Neural Net Layer tools."""
import numpy as np
import tensorflow as tf

from aboleth.distributions import norm_prior, norm_posterior, gaus_posterior


#
# Layer Composition
#

def compose_layers(layers, Phi):
    """Compose a list of layers into a network."""
    KL = 0.
    for l in layers:
        Phi, kl = l(Phi)
        KL += kl

    return Phi, KL


#
# Layers
#

def activation(h=lambda X: X):
    """Activation function layer."""
    def build_activation(X):
        Phi = h(X)
        KL = 0.
        return Phi, KL

    return build_activation


def fork(join='cat', *layers):
    """Fork into multiple layer-pipelines, then join the outputs."""
    joinops = {
        'cat': lambda Xs: tf.concat(Xs, axis=2),
        'add': lambda Xs: tf.add_n(Xs),
    }

    if not callable(join):
        join = joinops[join]

    def build_fork(X):
        Phis, KLs = zip(*map(lambda l: compose_layers(l, X), layers))
        KL = sum(KLs)
        Phi = join(Phis)
        return Phi, KL

    return build_fork


def dropout(keep_prob, seed=None):
    """Dropout layer, Bernoulli probability of not setting an input to zero."""
    def build_dropout(X):
        noise_shape = None  # equivalent to different samples from posterior
        Phi = tf.nn.dropout(X, keep_prob, noise_shape, seed)
        KL = 0.
        return Phi, KL

    return build_dropout


def dense_var(output_dim, reg=1., learn_prior=True, full=False, seed=None,
              bias=True):
    """Dense (fully connected) linear layer, with variational inference."""
    def build_dense(X):
        """X is a rank 3 tensor, [n_samples, N, D]."""
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        # Layer weights
        pW = norm_prior(dim=Wdim, var=reg, learn_var=learn_prior)
        qW = (gaus_posterior(dim=Wdim, var0=reg, seed=seed) if full else
              norm_posterior(dim=Wdim, var0=reg, seed=seed))
        Wsamples = _sample(qW, n_samples)

        # Linear layer
        Phi = tf.matmul(X, Wsamples)

        # Regularizers
        KL = tf.reduce_sum(qW.KL(pW))

        # Optional bias
        if bias:
            if bias is True:
                qb = norm_posterior(dim=bdim, var0=reg, seed=seed)
                pb = norm_prior(dim=bdim, var=reg, learn_var=learn_prior)
                bsamples = tf.expand_dims(_sample(qb, n_samples), 1)
                Phi += bsamples
                KL += tf.reduce_sum(qb.KL(pb))
            else:  # Bias is set value
                Phi += bias

        return Phi, KL

    return build_dense


def dense_map(output_dim, l1_reg=1., l2_reg=1., seed=None, bias=True):
    """Dense (fully connected) linear layer, with MAP inference."""
    def build_dense_map(X):
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = output_dim
        rand = np.random.RandomState(seed)

        W = tf.Variable(rand.randn(*Wdim).astype(np.float32))

        # Linear layer, don't want to copy Variable, (W) so map over X
        Phi = tf.map_fn(lambda x: tf.matmul(x, W), X)

        # Regularizers
        pen = l2_reg * tf.nn.l2_loss(W) + l1_reg * _l1_loss(W)

        # Optional Bias
        if bias:
            if bias is True:
                b = tf.Variable(rand.randn(bdim).astype(np.float32))
                Phi += b
                pen += l2_reg * tf.nn.l2_loss(b) + l1_reg * _l1_loss(b)
            else:
                Phi += bias

        return Phi, pen

    return build_dense_map


def randomFourier(n_features, kernel=None, seed=None):
    """Random fourier feature layer."""
    kernel = kernel if kernel else RBF()

    def build_randomFF(X):
        n_samples, input_dim = _get_dims(X)

        # Random weights, copy faster than map here
        P = kernel.weights(input_dim, n_features, seed)
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

    def weights(self, input_dim, n_features, seed=None):
        rand = np.random.RandomState(seed)
        P = rand.randn(input_dim, n_features).astype(np.float32)
        return P / self.lenscale


class Matern(RBF):
    """Matern kernel approximation."""

    def __init__(self, lenscale=1.0, p=1):
        super().__init__(lenscale)
        self.p = p

    def weights(self, input_dim, n_features, seed=None):
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
        rand = np.random.RandomState(seed)
        y = rand.randn(input_dim, n_features)
        u = rand.chisquare(df, size=(n_features,))
        P = y * np.sqrt(df / u)
        P = P.astype(np.float32)
        return P / self.lenscale


#
# Private module stuff
#


def _l1_loss(X):
    l1 = tf.reduce_sum(tf.abs(X))
    return l1


def _get_dims(X):
        n_samples, input_dim = X.shape[0], X.shape[2]
        return int(n_samples), int(input_dim)


def _sample(dist, n_samples):
    samples = tf.stack([dist.sample() for _ in range(n_samples)])
    return samples
