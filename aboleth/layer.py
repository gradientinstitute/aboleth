"""Network layers and utilities."""
import numpy as np
import tensorflow as tf

from aboleth.distributions import norm_prior, norm_posterior, gaus_posterior


#
# Layer Composition
#

def compose_layers(Net, layers):
    """Compose a list of layers into a network.

    Parameters
    ----------
    Net : ndarray, Tensor
        the neural net featues of shape (n_samples, N, output_dimensions).
    layers : sequence
        a list (or sequence) of layers defining the neural net.

    Returns
    -------
    Net : Tensor
        the neural net Tensor with ``layer`` applied.
    KL : float, Tensor
        the Kullback-Leibler divergence regularizer of the model parameters (or
        just the weight Regularizers).
    """
    KL = 0.
    for l in layers:
        Net, kl = l(Net)
        KL += kl

    return Net, KL


#
# Layers
#

def activation(h=lambda X: X):
    """Activation function layer.

    Parameters
    ----------
    h : callable
        the *element-wise* activation function.

    Returns
    -------
    build_activation : callable
        a function that builds the activation layer.
    """
    def build_activation(X):
        Net = h(X)
        KL = 0.
        return Net, KL

    return build_activation


def fork(join='cat', *layers):
    """Fork into multiple layer-pipelines, then join the outputs.

    Parameters
    ----------
    join : str, callable
        the operation used to join the forked layers, this can be 'cat' to
        concatenate, 'add' to add them (they must have the same shape) or a
        callable.
    *layers : args
        layers-sequences to fork the input into.

    Returns
    -------
    build_fork : callable
        a function that builds the fork layer.
    """
    if not callable(join):
        if join == 'add':
            def join(P):
                return tf.add_n(P)
        elif join == 'cat':
            def join(P):
                return tf.concat(P, axis=-1)
        else:
            raise ValueError("join must be a callable, 'cat' or 'add'")

    def build_fork(X):
        Nets, KLs = zip(*map(lambda l: compose_layers(X, l), layers))
        KL = sum(KLs)
        Net = join(Nets)
        return Net, KL

    return build_fork


def dropout(keep_prob, seed=None):
    """Dropout layer, Bernoulli probability of not setting an input to zero.

    This is just a thin wrapper around `tf.dropout
    <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

    Parameters
    ----------
    keep_prob : float, Tensor
        the probability of keeping an input. See `tf.dropout
        <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_.
    seed : int
        used to create random seeds.

    Returns
    -------
    build_dropout : callable
        a function that builds the dropout layer.
    """
    def build_dropout(X):
        noise_shape = None  # equivalent to different samples from posterior
        Net = tf.nn.dropout(X, keep_prob, noise_shape, seed)
        KL = 0.
        return Net, KL

    return build_dropout


def dense_var(output_dim, reg=1., full=False, use_bias=True, seed=None):
    """Dense (fully connected) linear layer, with variational inference."""
    def build_dense(X):
        # X is a rank 3 tensor, [n_samples, N, D]
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        # Layer weights
        pW = norm_prior(dim=Wdim, var=reg)
        qW = (gaus_posterior(dim=Wdim, var0=reg, seed=seed) if full else
              norm_posterior(dim=Wdim, var0=reg, seed=seed))
        Wsamples = _sample(qW, n_samples)

        # Linear layer
        Net = tf.matmul(X, Wsamples)

        # Regularizers
        KL = tf.reduce_sum(qW.KL(pW))

        # Optional bias
        if use_bias is True:
            qb = norm_posterior(dim=bdim, var0=reg, seed=seed)
            pb = norm_prior(dim=bdim, var=reg)
            bsamples = tf.expand_dims(_sample(qb, n_samples), 1)
            Net += bsamples
            KL += tf.reduce_sum(qb.KL(pb))

        return Net, KL

    return build_dense


def embedding_var(output_dim, n_categories, reg=1., full=False, seed=None):
    """Dense (fully connected) embedding layer, with variational inference."""
    if n_categories < 2:
        raise ValueError("There must be more than 2 categories for embedding!")

    def build_embedding(X):
        # X is a rank 3 tensor, [n_samples, N, 1]
        if X.shape[2] > 1:
            raise ValueError("X must be a *column* of indices!")

        Wdim = (n_categories, output_dim)
        n_samples = X.shape[0]

        # Layer weights
        pW = norm_prior(dim=Wdim, var=reg)
        qW = (gaus_posterior(dim=Wdim, var0=reg, seed=seed) if full else
              norm_posterior(dim=Wdim, var0=reg, seed=seed))
        Wsamples = tf.transpose(_sample(qW, n_samples), [1, 2, 0])

        # Embedding layer -- gather only works on the first dim hence transpose
        embedding = tf.gather(Wsamples, X[0, :, 0])  # X ind is just replicated
        Net = tf.transpose(embedding, [2, 0, 1])  # reshape after index 1st dim

        # Regularizers
        KL = tf.reduce_sum(qW.KL(pW))

        return Net, KL

    return build_embedding


def dense_map(output_dim, l1_reg=1., l2_reg=1., use_bias=True, seed=None):
    """Dense (fully connected) linear layer, with MAP inference."""
    def build_dense_map(X):
        # X is a rank 3 tensor, [n_samples, N, D]
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)

        W = tf.Variable(tf.random_normal(shape=Wdim, seed=seed))

        # We don't want to copy tf.Variable W so map over X
        Net = tf.map_fn(lambda x: tf.matmul(x, W), X)

        # Regularizers
        penalty = l2_reg * tf.nn.l2_loss(W) + l1_reg * _l1_loss(W)

        # Optional Bias
        if use_bias is True:
            b = tf.Variable(tf.zeros(output_dim))
            Net += b
            penalty += l2_reg * tf.nn.l2_loss(b) + l1_reg * _l1_loss(b)

        return Net, penalty

    return build_dense_map


def random_fourier(n_features, kernel=None, seed=None):
    """Random fourier feature layer."""
    kernel = kernel if kernel else RBF()

    def build_random_ff(X):
        n_samples, input_dim = _get_dims(X)

        # Random weights, copy faster than map here
        P = kernel.weights(input_dim, n_features, seed)
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        real = tf.cos(XP)
        imag = tf.sin(XP)
        Net = tf.concat([real, imag], axis=-1) / np.sqrt(n_features)
        KL = 0.

        return Net, KL

    return build_random_ff


def random_arccosine(n_features, lenscale=1.0, p=1, seed=None):
    """Random Arc-Cosine kernel layer."""
    if p < 0 or not isinstance(p, int):
        raise ValueError("p must be a positive integer!")
    elif p == 0:
        def pfunc(x):
            return tf.sign(x)
    elif p == 1:
        def pfunc(x):
            return x
    else:
        def pfunc(x):
            return tf.pow(x, p)

    def build_random_ac(X):
        n_samples, input_dim = _get_dims(X)

        # Random weights
        rand = np.random.RandomState(seed)
        P = rand.randn(input_dim, n_features).astype(np.float32) / lenscale
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        Net = np.sqrt(2. / n_features) * tf.nn.relu(pfunc(XP))
        KL = 0.

        return Net, KL

    return build_random_ac


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
