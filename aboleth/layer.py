"""Network layers and utilities."""
import numpy as np
import tensorflow as tf

from aboleth.random import seedgen
from aboleth.distributions import (norm_prior, norm_posterior, gaus_posterior,
                                   kl_qp)


#
# Sampling layer
#

def sample(n):
    """Create a Sampling layer.

    This layer takes a 2D tensor of shape (k,d) and tiles it along a new
    first axis creating a (n,k,d) tensor. Used to propagate samples through
    a variational deep net.

    Parameters
    ----------
    n : int > 0
        The number of samples.

    Returns
    -------
    samplefunc : callable
        A function implements the tiling.

    """
    def samplefunc(X):
        Xs = tf.tile(tf.expand_dims(X, 0), [n, 1, 1])  # (n, N, D)
        return Xs, 0.0
    return samplefunc


#
# Activation Layers
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


def dropout(keep_prob):
    """Dropout layer, Bernoulli probability of not setting an input to zero.

    This is just a thin wrapper around `tf.dropout
    <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

    Parameters
    ----------
    keep_prob : float, Tensor
        the probability of keeping an input. See `tf.dropout
        <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_.

    Returns
    -------
    build_dropout : callable
        a function that builds the dropout layer.
    """
    def build_dropout(X):
        noise_shape = None  # equivalent to different samples from posterior
        Net = tf.nn.dropout(X, keep_prob, noise_shape, seed=next(seedgen))
        KL = 0.
        return Net, KL

    return build_dropout


def random_fourier(n_features, kernel=None):
    """Random fourier feature layer.

    NOTE: This should be followed by a dense layer to properly implement a
        kernel approximation.

    Parameters
    ----------
    n_features : int
        the number of unique random features, the actual output dimension of
        this layer will be ``2 * n_features``.
    kernel : object
        the object that yeilds the random samples from the fourier spectrum of
        a particular kernel to approximate. See ``RBF`` and ``Matern`` etc.

    Returns
    -------
    build_random_ff : callable
        a function that builds the random fourier feature layer.
    """
    kernel = kernel if kernel else RBF()

    def build_random_ff(X):
        n_samples, input_dim = _get_dims(X)

        # Random weights, copy faster than map here
        P = kernel.weights(input_dim, n_features)
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        real = tf.cos(XP)
        imag = tf.sin(XP)
        Net = tf.concat([real, imag], axis=-1) / np.sqrt(n_features)
        KL = 0.

        return Net, KL

    return build_random_ff


def random_arccosine(n_features, lenscale=1.0, p=1):
    """Random arc-cosine kernel layer.

    NOTE: This should be followed by a dense layer to properly implement a
        kernel approximation.

    Parameters
    ----------
    n_features : int
        the number of unique random features, the actual output dimension of
        this layer will be ``2 * n_features``.
    lenscale : float, ndarray, Tensor
        the lenght scales of the ar-cosine kernel, this can be a scalar for
        an isotropic kernel, or a vector for an automatic relevance detection
        (ARD) kernel.
    p : int
        The order of the arc-cosine kernel, this must be an integer greater
        than zero. 0 will lead to sigmoid-like kernels, 1 will lead to
        relu-like kernels, 2 quadratic-relu kernels etc.

    Returns
    -------
    build_random_ac: callable
        a function that builds the random arc-cosine feature layer.

    See Also
    --------
    [1] Cho, Youngmin, and Lawrence K. Saul. "Analysis and extension of
        arc-cosine kernels for large margin classification." arXiv preprint
        arXiv:1112.3712 (2011).
    [2] Cutajar, Kurt, Edwin V. Bonilla, Pietro Michiardi, and Maurizio
        Filippone. "Accelerating Deep Gaussian Processes Inference with
        Arc-Cosine Kernels." Bayesian Deep Learning Workshop, Advances in
        Neural Information Processing Systems, NIPS 2016, Barcelona
    """
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
        rand = np.random.RandomState(next(seedgen))
        P = rand.randn(input_dim, n_features).astype(np.float32) / lenscale
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        Net = np.sqrt(2. / n_features) * tf.nn.relu(pfunc(XP))
        KL = 0.

        return Net, KL

    return build_random_ac


#
# Weight layers
#

def dense_var(output_dim, reg=1., full=False, use_bias=True, prior_W=None,
              prior_b=None, post_W=None, post_b=None):
    """Dense (fully connected) linear layer, with variational inference.

    Parameters
    ----------
    output_dim : int
        the dimension of the output of this layer
    reg : float
        the initial value of the weight prior, w ~ N(0, reg * I), this is
        optimized (a la maximum likelihood type II).
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.
    prior_W: {Normal, Gaussian}, optional
        This is the prior distribution object to use on the layer weights. It
        must have parameters compatible with ``(input_dim, output_dim)`` shaped
        weights. This ignores the ``reg`` parameter.
    prior_b: {Normal, Gaussian}, optional
        This is the prior distribution object to use on the layer intercept. It
        must have parameters compatible with ``(output_dim,)`` shaped weights.
        This ignores the ``reg`` and ``use_bias`` parameters.
    post_W: {Normal, Gaussian}, optional
        This is the posterior distribution object to use on the layer weights.
        It must have parameters compatible with ``(input_dim, output_dim)``
        shaped weights. This ignores the ``full`` parameter. See
        ``distributions.gaus_posterior``.
    post_b: {Normal, Gaussian}, optional
        This is the posterior distribution object to use on the layer
        intercept. It must have parameters compatible with ``(output_dim,)``
        shaped weights. This ignores the ``use_bias`` parameters.
        See ``distributions.norm_posterior``.

    Returns
    -------
    build_dense : callable
        a function that builds the dense variational layer.
    """
    def build_dense(X):
        # X is a rank 3 tensor, [n_samples, N, D]
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)
        bdim = (output_dim,)

        # Layer weights
        pW, qW = _make_bayesian_weights(prior_W, post_W, Wdim, reg, full)
        Wsamples = _sample(qW, n_samples)

        # Linear layer
        Net = tf.matmul(X, Wsamples)

        # Regularizers
        KL = kl_qp(qW, pW)

        # Optional bias
        if use_bias is True or prior_b or post_b:
            pb, qb = _make_bayesian_weights(prior_b, post_b, bdim, reg, False)
            bsamples = tf.expand_dims(_sample(qb, n_samples), 1)
            Net += bsamples
            KL += kl_qp(qb, pb)

        return Net, KL

    return build_dense


def embed_var(n_categories, output_dim, reg=1., full=False, prior_W=None,
              post_W=None):
    """Dense (fully connected) embedding layer, with variational inference.

    This layer works directly on shape (N, 1) inputs of category *indices*
    rather than one-hot representations, for efficiency.

    Parameters
    ----------
    n_categories: int
        the number of categories in the input variable
    output_dim : int
        the dimension of the output (embedding) of this layer
    reg : float
        the initial value of the weight prior, w ~ N(0, reg * I), this is
        optimized (a la maximum likelihood type II)
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.
    prior_W: {Normal, Gaussian}, optional
        This is the prior distribution object to use on the layer weights. It
        must have parameters compatible with ``(input_dim, output_dim)`` shaped
        weights. This ignores the ``reg`` parameter.
    post_W: {Normal, Gaussian}, optional
        This is the posterior distribution object to use on the layer weights.
        It must have parameters compatible with ``(input_dim, output_dim)``
        shaped weights. This ignores the ``full`` parameter. See
        ``distributions.gaus_posterior``.

    Returns
    -------
    build_embedding : callable
        a function that builds the embedding variational layer.
    """
    if n_categories < 2:
        raise ValueError("There must be more than 2 categories for embedding!")

    def build_embedding(X):
        # X is a rank 3 tensor, [n_samples, N, 1]
        if X.shape[2] > 1:
            print("embedding X: {}".format(X))
            raise ValueError("X must be a *column* of indices!")

        Wdim = (n_categories, output_dim)
        n_samples = X.shape[0]

        # Layer weights
        pW, qW = _make_bayesian_weights(prior_W, post_W, Wdim, reg, full)
        Wsamples = tf.transpose(_sample(qW, n_samples), [1, 2, 0])

        # Embedding layer -- gather only works on the first dim hence transpose
        embedding = tf.gather(Wsamples, X[0, :, 0])  # X ind is just replicated
        Net = tf.transpose(embedding, [2, 0, 1])  # reshape after index 1st dim

        # Regularizers
        KL = kl_qp(qW, pW)

        return Net, KL

    return build_embedding


def dense_map(output_dim, l1_reg=1., l2_reg=1., use_bias=True):
    """Dense (fully connected) linear layer, with MAP inference.

    Parameters
    ----------
    output_dim : int
        the dimension of the output of this layer
    l1_reg : float
        the value of the l1 weight regularizer, reg * ||w||_1
    l2_reg : float
        the value of the l2 weight regularizer, reg * 0.5 * ||w||^2_2
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.

    Returns
    -------
    build_dense_map : callable
        a function that builds the dense MAP layer.
    """
    def build_dense_map(X):
        # X is a rank 3 tensor, [n_samples, N, D]
        n_samples, input_dim = _get_dims(X)
        Wdim = (input_dim, output_dim)

        W = tf.Variable(tf.random_normal(shape=Wdim, seed=next(seedgen)),
                        name="W_map")

        # We don't want to copy tf.Variable W so map over X
        Net = tf.map_fn(lambda x: tf.matmul(x, W), X)

        # Regularizers
        penalty = l2_reg * tf.nn.l2_loss(W) + l1_reg * _l1_loss(W)

        # Optional Bias
        if use_bias is True:
            b = tf.Variable(tf.zeros(output_dim), name="b_map")
            Net += b
            penalty += l2_reg * tf.nn.l2_loss(b) + l1_reg * _l1_loss(b)

        return Net, penalty

    return build_dense_map


#
# Random Fourier Kernels
#

class RBF:
    """Radial basis kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor
        the lenght scales of the radial basis kernel, this can be a scalar for
        an isotropic kernel, or a vector for an automatic relevance detection
        (ARD) kernel.
    """

    def __init__(self, lenscale=1.0):
        """Constuct an RBF kernel object."""
        self.lenscale = lenscale

    def weights(self, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        """
        rand = np.random.RandomState(next(seedgen))
        P = rand.randn(input_dim, n_features).astype(np.float32)
        return P / self.lenscale


class Matern(RBF):
    """Matern kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor
        the lenght scales of the Matern kernel, this can be a scalar for an
        isotropic kernel, or a vector for an automatic relevance detection
        (ARD) kernel.
    """

    def __init__(self, lenscale=1.0, p=1):
        """Constuct a Matern kernel object."""
        super().__init__(lenscale)
        self.p = p

    def weights(self, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        """
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
        rand = np.random.RandomState(next(seedgen))
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


def _is_dim(X, dims):
    shape = tuple([int(d) for d in X.get_shape()])
    return shape == dims


def _sample(dist, n_samples):
    samples = tf.stack([dist.sample() for _ in range(n_samples)])
    return samples


def _make_bayesian_weights(prior_W, post_W, Wdim, reg, full):
    # Check/make prior
    if prior_W:
        if not _is_dim(prior_W.mu, Wdim):
            raise ValueError("Incompatible dimension in prior distribution!")
    else:
        prior_W = norm_prior(dim=Wdim, var=reg)

    # Check/make posterior
    if post_W:
        if not _is_dim(post_W.mu, Wdim):
            raise ValueError("Incompatible dimension in posterior"
                             " distribution!")
    else:
        post_W = (gaus_posterior(dim=Wdim, var0=reg) if full else
                  norm_posterior(dim=Wdim, var0=reg))

    return prior_W, post_W
