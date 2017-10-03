"""Network layers and utilities."""
import numpy as np
import tensorflow as tf

from aboleth.kernels import RBF, RBFVariational
from aboleth.random import seedgen
from aboleth.distributions import (norm_prior, norm_posterior, gaus_posterior,
                                   kl_sum)
from aboleth.baselayers import Layer, MultiLayer


#
# Layer type base classes
#

class InputLayer(MultiLayer):
    r"""Create an input layer.

    This layer defines input kwargs so that a user may easily provide the right
    inputs to a complex set of layers. It takes a 2D tensor of shape (N, D).
    If n_samples is specified, the input is tiled along a new first axis
    creating a (n_samples, N, D) tensor for propogating samples through a
    variational deep net.

    Parameters
    ----------
    name : string
        The name of the input. Used as the agument for input into the net.
    n_samples : int > 0
        The number of samples.

    """

    def __init__(self, name, n_samples=None):
        """Construct an instance of InputLayer."""
        self.name = name
        self.n_samples = n_samples

    def _build(self, **kwargs):
        """Build the tiling input layer."""
        X = kwargs[self.name]
        if self.n_samples is not None:
            # (n_samples, N, D)
            Xs = tf.tile(tf.expand_dims(X, 0), [self.n_samples, 1, 1])
        else:
            Xs = tf.convert_to_tensor(X)
        return Xs, 0.0


class SampleLayer(Layer):
    r"""Sample Layer base class.

    This is the base class for layers that build upon stochastic (variational)
    nets. These expect *rank >= 3* input Tensors, where the first dimension
    indexes the random samples of the stochastic net.
    """

    def __call__(self, X):
        r"""Construct the subgraph for this layer.

        Parameters
        ----------
        X : Tensor
            the input to this layer

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : float, Tensor
            the regularizer/Kullback Leibler 'cost' of the parameters in this
            layer.

        """
        rank = len(X.shape)
        assert rank > 2, "SampleLayers require rank > 2 input Tensors, with" \
            " the first axis being the random samples of the net."""
        Net, KL = self._build(X)
        return Net, KL

    @staticmethod
    def _get_X_dims(X):
        r"""Get the dimensions of the rank >= 3 input tensor, X."""
        n_samples, _, *input_shape = X.shape.as_list()
        return n_samples, input_shape


class SampleLayer3(SampleLayer):
    r"""Special case of SampleLayer restricted to *rank == 3* input Tensors."""

    def __call__(self, X):
        r"""Construct the subgraph for this layer.

        Parameters
        ----------
        X : Tensor
            the input to this layer

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : float, Tensor
            the regularizer/Kullback Leibler 'cost' of the parameters in this
            layer.

        """
        rank = len(X.shape)
        assert rank == 3
        Net, KL = super(SampleLayer3, self).__call__(X)
        return Net, KL

    @staticmethod
    def _get_X_dims(X):
        """Get the dimensions of the rank 3 input tensor, X."""
        n_samples, _, input_dim = X.shape.as_list()
        return n_samples, input_dim


#
# Activation Layers
#

class Activation(Layer):
    """Activation function layer.

    Parameters
    ----------
    h : callable
        the *element-wise* activation function.

    """

    def __init__(self, h=lambda X: X):
        """Create an instance of an Activation layer."""
        self.h = h

    def _build(self, X):
        """Build the graph of this layer."""
        Net = self.h(X)
        KL = 0.
        return Net, KL


class DropOut(Layer):
    r"""Dropout layer, Bernoulli probability of not setting an input to zero.

    This is just a thin wrapper around `tf.dropout
    <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

    Parameters
    ----------
    keep_prob : float, Tensor
        the probability of keeping an input. See `tf.dropout
        <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_.

    """

    def __init__(self, keep_prob):
        """Create an instance of a Dropout layer."""
        self.keep_prob = keep_prob

    def _build(self, X):
        """Build the graph of this layer."""
        noise_shape = None  # equivalent to different samples from posterior
        Net = tf.nn.dropout(X, self.keep_prob, noise_shape, seed=next(seedgen))
        KL = 0.
        return Net, KL


class MaxPool2D(Layer):
    r"""Max pooling layer for 2D inputs (e.g. images).

    This is just a thin wrapper around `tf.nn.max_pool
    <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool>`_

    Parameters
    ----------
    pool_size : tuple or list of 2 ints
        width and height of the pooling window.
    strides : tuple or list of 2 ints
        the strides of the pooling operation along the height and width.
    padding : str
        One of 'SAME' or 'VALID'. Defaults to 'SAME'. The type of padding

    """

    def __init__(self, pool_size, strides, padding='SAME'):
        """Initialize instance of a MaxPool2D layer."""
        self.ksize = [1] + list(pool_size) + [1]
        self.strides = [1] + list(strides) + [1]
        self.padding = padding

    def _build(self, X):
        """Build the graph of this layer."""
        Net = tf.map_fn(lambda inputs: tf.nn.max_pool(inputs,
                                                      ksize=self.ksize,
                                                      strides=self.strides,
                                                      padding=self.padding), X)
        KL = 0.
        return Net, KL


class Reshape(Layer):
    """Reshape layer.

    Reshape and output an tensor to a specified shape.

    Parameters
    ----------
    targe_shape : tuple of ints
        Does not include the samples or batch axes.

    """

    def __init__(self, target_shape):
        """Initialize instance of a Reshape layer."""
        self.target_shape = target_shape

    def _build(self, X):
        """Build the graph of this layer."""
        new_shape = X.shape[:2].concatenate(tf.TensorShape(self.target_shape))
        Net = tf.reshape(X, new_shape)
        KL = 0.
        return Net, KL


#
# Kernel Approximation Layers
#

class RandomFourier(SampleLayer3):
    r"""Random Fourier feature (RFF) kernel approximation layer.

    NOTE: This should be followed by a dense layer to properly implement a
        kernel approximation.

    Parameters
    ----------
    n_features : int
        the number of unique random features, the actual output dimension of
        this layer will be ``2 * n_features``.
    kernel : kernels.ShiftInvariant
        the kernel object that yeilds the random samples from the fourier
        spectrum of a particular kernel to approximate. See the :ref:`kernels`
        module.

    """

    def __init__(self, n_features, kernel):
        """Construct and instance of a RandomFourier object."""
        self.n_features = n_features
        self.kernel = kernel

    def _build(self, X):
        """Build the graph of this layer."""
        # Random weights
        n_samples, input_dim = self._get_X_dims(X)
        P, KL = self.kernel.weights(input_dim, self.n_features)
        Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])

        # Random features
        XP = tf.matmul(X, Ps)
        Net = self._transformation(XP)
        return Net, KL

    def _transformation(self, XP):
        """Build the kernel feature space transformation."""
        real = tf.cos(XP)
        imag = tf.sin(XP)
        Net = tf.concat([real, imag], axis=-1) / np.sqrt(self.n_features)
        return Net


class RandomArcCosine(RandomFourier):
    r"""Random arc-cosine kernel layer.

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
        than, or eual to zero. 0 will lead to sigmoid-like kernels, 1 will lead
        to relu-like kernels, 2 quadratic-relu kernels etc.
    variational : bool
        use variational features instead of random features, (i.e. VAR-FIXED in
        [2]).
    lenscale_posterior : float, ndarray, optional
        the *initial* value for the posterior length scale. This is only used
        if ``variational==True``. This can be a scalar or vector (different
        initial value per input dimension). If this is left as None, it will be
        set to ``sqrt(1 / input_dim)`` (this is similar to the 'auto' setting
        for a scikit learn SVM with a RBF kernel).

    See Also
    --------
    [1] Cho, Youngmin, and Lawrence K. Saul.
        "Analysis and extension of arc-cosine kernels for large margin
        classification." arXiv preprint arXiv:1112.3712 (2011).
    [2] Cutajar, K. Bonilla, E. Michiardi, P. Filippone, M.
        Random Feature Expansions for Deep Gaussian Processes. In ICML, 2017.

    """

    def __init__(self, n_features, lenscale=1.0, p=1, variational=False,
                 lenscale_posterior=None):
        """Create an instance of an arc cosine kernel layer."""
        # Setup random weights
        if variational:
            kern = RBFVariational(lenscale=lenscale,
                                  lenscale_posterior=lenscale_posterior)
        else:
            kern = RBF(lenscale=lenscale)
        super().__init__(n_features=n_features, kernel=kern)

        # Kernel order
        assert isinstance(p, int) and p >= 0
        if p == 0:
            self.pfunc = tf.sign
        elif p == 1:
            self.pfunc = lambda x: x
        else:
            self.pfunc = lambda x: tf.pow(x, p)

    def _transformation(self, XP):
        """Build the kernel feature space transformation."""
        Net = np.sqrt(2. / self.n_features) * tf.nn.relu(self.pfunc(XP))
        return Net


#
# Weight layers
#

class DenseVariational(SampleLayer3):
    r"""A dense (fully connected) linear layer, with variational inference.

    This implements a dense linear layer,

    .. math::
        f(\mathbf{X}) = \mathbf{X} \mathbf{W} + \mathbf{b}

    where prior, :math:`p(\cdot)`, and approximate posterior, :math:`q(\cdot)`
    distributions are placed on the weights and *also* the biases. Here
    :math:`\mathbf{X} \in \mathbb{R}^{N \times D_{in}}`, :math:`\mathbf{W} \in
    \mathbb{R}^{D_{in} \times D_{out}}` and :math:`\mathbf{b} \in
    \mathbb{R}^{D_{out}}`. By default, the same Normal prior is placed on each
    of the layer weights and biases,

    .. math::
        w_{ij} \sim \mathcal{N}(0, \sigma^2), \quad
        b_{j} \sim \mathcal{N}(0, \sigma^2),

    and a different Normal posterior is learned for each of the layer weights
    and biases,

    .. math::
        w_{ij} \sim \mathcal{N}(m_{ij}, c_{ij}), \quad
        b_{j} \sim \mathcal{N}(l_{j}, o_{j}).

    We also have the option of placing full-covariance Gaussian posteriors on
    the input dimension of the weights,

    .. math::
        \mathbf{w}_{j} \sim \mathcal{N}(\mathbf{m}_{j}, \mathbf{C}_{j}),

    where :math:`\mathbf{m}_j \in \mathbb{R}^{D_{in}}` and
    :math:`\mathbf{C}_j \in \mathbb{R}^{D_{in} \times D_{in}}`.

    This layer will use variational inference to learn *all* of the non-zero
    *prior* and posterior parameters.

    Whenever this layer is called, it will return the result,

    .. math::
        f^{(s)}(\mathbf{X}) = \mathbf{X} \mathbf{W}^{(s)} + \mathbf{b}^{(s)}

    with samples from the posteriors, :math:`\mathbf{W}^{(s)} \sim
    q(\mathbf{W})` and :math:`\mathbf{b}^{(s)} \sim q(\mathbf{b})`. The number
    of samples, *s*, can be controlled by using the ``n_samples`` argument in
    an ``InputLayer`` used to feed the first layer of a model, or by tiling
    :math:`\mathbf{X}` on the first dimension. This layer also returns the
    result of :math:`\text{KL}[q\|p]` for all parameters.

    Parameters
    ----------
    output_dim : int
        the dimension of the output of this layer
    std : float
        the initial value of the weight prior standard deviation
        (:math:`\sigma` above), this is optimized a la maximum likelihood type
        II.
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.
    prior_W : tf.distributions.Distribution, optional
        This is the prior distribution object to use on the layer weights. It
        must have parameters compatible with (input_dim, output_dim) shaped
        weights. This ignores the ``std`` parameter.
    prior_b : tf.distributions.Distribution, optional
        This is the prior distribution object to use on the layer intercept. It
        must have parameters compatible with (output_dim,) shaped weights.
        This ignores the ``std`` and ``use_bias`` parameters.
    post_W : tf.distributions.Distribution, optional
        It must have parameters compatible with (input_dim, output_dim) shaped
        weights. This ignores the ``full`` parameter. See also
        ``distributions.gaus_posterior``.
    post_b : tf.distributions.Distributions, optional
        This is the posterior distribution object to use on the layer
        intercept. It must have parameters compatible with (output_dim,) shaped
        weights. This ignores the ``use_bias`` parameters.  See also
        ``distributions.norm_posterior``.

    """

    def __init__(self, output_dim, std=1., full=False, use_bias=True,
                 prior_W=None, prior_b=None, post_W=None, post_b=None):
        """Create and instance of a variational dense layer."""
        self.output_dim = output_dim
        self.std = std
        self.full = full
        self.use_bias = use_bias
        self.pW = prior_W
        self.pb = prior_b
        self.qW = post_W
        self.qb = post_b

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, input_dim = self._get_X_dims(X)
        W_shape, b_shape = self._weight_shapes(input_dim)

        # Layer weights
        self.pW = self._make_prior(self.pW, W_shape)
        self.qW = self._make_posterior(self.qW, W_shape)

        # Regularizers
        KL = kl_sum(self.qW, self.pW)

        # Linear layer
        Wsamples = self._sample_W(self.qW, n_samples)
        Net = tf.matmul(X, Wsamples)

        # Optional bias
        if self.use_bias is True or self.prior_b or self.post_b:
            # Layer intercepts
            self.pb = self._make_prior(self.pb, b_shape)
            self.qb = self._make_posterior(self.qb, b_shape)

            # Regularizers
            KL += kl_sum(self.qb, self.pb)

            # Linear layer
            bsamples = tf.expand_dims(self._sample_W(self.qb, n_samples), 1)
            Net += bsamples

        return Net, KL

    def _make_prior(self, prior_W, weight_shape):
        """Check/make prior."""
        if prior_W is None:
            prior_W = norm_prior(weight_shape, std=self.std)

        assert _is_dim(prior_W, weight_shape), \
            "Prior inconsistent dimension!"

        return prior_W

    def _make_posterior(self, post_W, weight_shape):
        """Check/make posterior."""
        if post_W is None:
            # We don't want a full-covariance on an intercept, check input_dim
            if self.full and len(weight_shape) > 1:
                post_W = gaus_posterior(weight_shape, std0=self.std)
            else:
                post_W = norm_posterior(weight_shape, std0=self.std)

        assert _is_dim(post_W, weight_shape), \
            "Posterior inconsistent dimension!"

        return post_W

    def _weight_shapes(self, input_dim):
        """Generate weight and bias weight shape tuples."""
        weight_shape = (self.output_dim, input_dim)
        bias_shape = (self.output_dim,)

        return weight_shape, bias_shape

    @staticmethod
    def _sample_W(dist, n_samples):
        samples = tf.stack([tf.transpose(dist.sample(seed=next(seedgen)))
                            for _ in range(n_samples)])
        return samples


class EmbedVariational(DenseVariational):
    r"""Dense (fully connected) embedding layer, with variational inference.

    This layer works directly on shape (N, 1) inputs of *K* category *indices*
    rather than one-hot representations, for efficiency, and is a dense linear
    layer,

    .. math::
        f(\mathbf{X}) = \mathbf{X} \mathbf{W},

    where prior, :math:`p(\cdot)`, and approximate posterior, :math:`q(\cdot)`
    distributions are placed on the weights. Here :math:`\mathbf{X} \in
    \mathbb{N}_2^{N \times K}` and :math:`\mathbf{W} \in \mathbb{R}^{K \times
    D_{out}}`. Though in code we represent :math:`\mathbf{X}` as a vector of
    indices in :math:`\mathbb{N}_K^{N \times 1}`. By default, the same Normal
    prior is placed on each of the layer weights,

    .. math::
        w_{ij} \sim \mathcal{N}(0, \sigma^2),

    and a different Normal posterior is learned for each of the layer weights,

    .. math::
        w_{ij} \sim \mathcal{N}(m_{ij}, c_{ij}).

    We also have the option of placing full-covariance Gaussian posteriors on
    the input dimension of the weights,

    .. math::
        \mathbf{w}_{j} \sim \mathcal{N}(\mathbf{m}_{j}, \mathbf{C}_{j}),

    where :math:`\mathbf{m}_j \in \mathbb{R}^{K}` and
    :math:`\mathbf{C}_j \in \mathbb{R}^{K \times K}`.

    This layer will use variational inference to learn *all* of the non-zero
    *prior* and posterior parameters.

    Whenever this layer is called, it will return the result,

    .. math::
        f^{(s)}(\mathbf{X}) = \mathbf{X} \mathbf{W}^{(s)}

    with samples from the posterior, :math:`\mathbf{W}^{(s)} \sim
    q(\mathbf{W})`. The number of samples, *s*, can be controlled by using the
    ``n_samples`` argument in an ``InputLayer`` used to feed the first layer of
    a model, or by tiling :math:`\mathbf{X}` on the first dimension. This layer
    also returns the result of :math:`\text{KL}[q\|p]` for all parameters.

    Parameters
    ----------
    output_dim : int
        the dimension of the output (embedding) of this layer
    n_categories : int
        the number of categories in the input variable
    std : float
        the initial value of the weight prior standard deviation
        (:math:`\sigma` above), this is optimized a la maximum likelihood type
        II.
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.
    prior_W : tf.distributions.Distribution, optional
        This is the prior distribution object to use on the layer weights. It
        must have parameters compatible with (input_dim, output_dim) shaped
        weights. This ignores the ``std`` parameter.
    post_W : tf.distributions.Distribution, optional
        This is the posterior distribution object to use on the layer weights.
        It must have parameters compatible with (input_dim, output_dim) shaped
        weights. This ignores the ``full`` parameter. See also
        ``distributions.gaus_posterior``.

    """

    def __init__(self, output_dim, n_categories, std=1., full=False,
                 prior_W=None, post_W=None):
        """Create and instance of a variational dense embedding layer."""
        assert n_categories >= 2, "Need 2 or more categories for embedding!"
        self.output_dim = output_dim
        self.n_categories = n_categories
        self.std = std
        self.full = full
        self.pW = prior_W
        self.qW = post_W

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, input_dim = self._get_X_dims(X)
        W_shape, _ = self._weight_shapes(self.n_categories)

        assert input_dim == 1, "X must be a *column* of indices!"

        # Layer weights
        self.pW = self._make_prior(self.pW, W_shape)
        self.qW = self._make_posterior(self.qW, W_shape)

        # Index into the relevant weights rather than using sparse matmul
        Wsamples = self._sample_W(self.qW, n_samples)
        Net = tf.gather(Wsamples, X[0, :, 0], axis=1)

        # Regularizers
        KL = kl_sum(self.qW, self.pW)

        return Net, KL


class DenseMAP(SampleLayer):
    r"""Dense (fully connected) linear layer, with MAP inference.

    This implements a linear layer, and when called returns

    .. math::
        f(\mathbf{X}) = \mathbf{X} \mathbf{W} + \mathbf{b}

    where :math:`\mathbf{X} \in \mathbb{R}^{N \times D_{in}}`,
    :math:`\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}` and
    :math:`\mathbf{b} \in \mathbb{R}^{D_{out}}`. This layer uses maximum
    *a-posteriori* inference to learn the weights and biases, and so also
    returns complexity penalities (l1 or l2) for the weights and biases.

    Parameters
    ----------
    output_dim : int
        the dimension of the output of this layer
    l1_reg : float
        the value of the l1 weight regularizer,
        :math:`\text{l1_reg} \times \|\mathbf{W}\|_1`
    l2_reg : float
        the value of the l2 weight regularizer,
        :math:`\frac{1}{2} \text{l2_reg} \times \|\mathbf{W}\|^2_2`
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.

    """

    def __init__(self, output_dim, l1_reg=1., l2_reg=1., use_bias=True):
        """Create and instance of a dense layer with MAP regularizers."""
        self.output_dim = output_dim
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.use_bias = use_bias

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, input_shape = self._get_X_dims(X)

        Wdim = tuple(input_shape) + (self.output_dim,)

        W = tf.Variable(tf.random_normal(shape=Wdim, seed=next(seedgen)),
                        name="W_map")

        # We don't want to copy tf.Variable W so map over X
        Net = tf.map_fn(lambda x: tf.matmul(x, W), X)

        # Regularizers
        penalty = self.l2 * tf.nn.l2_loss(W) + self.l1 * _l1_loss(W)

        # Optional Bias
        if self.use_bias is True:
            b = tf.Variable(tf.random_normal(shape=(1, self.output_dim),
                                             seed=next(seedgen)), name="b_map")
            Net += b
            penalty += self.l2 * tf.nn.l2_loss(b) + self.l1 * _l1_loss(b)

        return Net, penalty


#
# Private module stuff
#

def _l1_loss(X):
    r"""Calculate the L1 loss of X, :math:`\|X\|_1`."""
    l1 = tf.reduce_sum(tf.abs(X))
    return l1


def _is_dim(distribution, dims):
    r"""Check if ``X``'s dimension is the same as the tuple ``dims``."""
    shape = tuple(distribution.loc.shape)
    return shape == dims
