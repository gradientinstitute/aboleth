"""Network layers and utilities."""
import numpy as np
import tensorflow as tf

from aboleth.kernels import RBF, RBFVariational
from aboleth.random import seedgen
from aboleth.distributions import (norm_prior, norm_posterior, gaus_posterior,
                                   kl_sum)
from aboleth.baselayers import Layer, MultiLayer
from aboleth.util import summary_histogram
from aboleth.initialisers import initialise_weights, initialise_stds


#
# Layer type base classes
#


class InputLayer(MultiLayer):
    r"""Create an input layer.

    This layer defines input kwargs so that a user may easily provide the right
    inputs to a complex set of layers. It takes a tensor of shape ``(N, ...)``.
    The input is tiled along a new first axis creating a ``(n_samples, N,
    ...)`` tensor for propagating samples through a variational deep net.

    Parameters
    ----------
    name : string
        The name of the input. Used as the argument for input into the net.
    n_samples : int, Tensor
        The number of samples to propagate through the network. We recommend
        making this a ``tf.placeholder`` so you can vary it as required.

    Note
    ----
    We recommend making ``n_samples`` a ``tf.placeholder`` so it can be varied
    between training and prediction!

    """

    def __init__(self, name, n_samples=1):
        """Construct an instance of InputLayer."""
        self.name = name
        self.n_samples = n_samples

    def _build(self, **kwargs):
        """Build the tiling input layer."""
        X = kwargs[self.name]
        # tile like (n_samples, ...)
        Xs = _tile2samples(self.n_samples, X)
        return Xs, 0.0


#
#  Sample Layer Abstract base classes
#

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
        n_samples = tf.to_int32(tf.shape(X)[0])
        input_shape = X.shape[2:].as_list()
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


#
# Activation and Transformation Layers
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
    independent: bool
        Use independently sampled droput for each observation if ``True``. This
        may dramatically increase convergence, but will no longer only sample
        the latent function.
    observation_axis : int
        The axis that indexes the observations (``N``). This will assume the
        obserations are on the *second* axis, i.e. ``(n_samples, N, ...)``.
        This is so we can repeat the dropout pattern over observations, which
        has the effect of dropping out weights consistently, thereby sampling
        the "latent function" of the layer. This is only active if
        ``independent`` is set to ``False``.
    alpha : bool
        Use alpha dropout (tf.contrib.nn.alpha_dropout) that maintains the self
        normalising property of SNNs.

    Note
    ----
    If a more complex noise shape, or some other modification to dropout is
    required, you can use an Activation layer. E.g.
    ``ab.Activation(lambda x: tf.nn.dropout(x, **your_args))``.

    """

    def __init__(self, keep_prob, independent=True, observation_axis=1,
                 alpha=False):
        """Create an instance of a Dropout layer."""
        self.keep_prob = keep_prob
        self.obsax = observation_axis
        self.independent = independent
        self.dropout = tf.contrib.nn.alpha_dropout if alpha else tf.nn.dropout

    def _build(self, X):
        """Build the graph of this layer."""
        # Set noise shape to equivalent to different samples from posterior
        # i.e. share the samples along the data-observations axis
        noise_shape = None
        if not self.independent:
            noise_shape = tf.concat([tf.shape(X)[:self.obsax], [1],
                                     tf.shape(X)[(self.obsax + 1):]], axis=0)
        Net = self.dropout(X, self.keep_prob, noise_shape, seed=next(seedgen))
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


class Flatten(Layer):
    """Flattening layer.

    Reshape and output a tensor to be always rank 3 (keeps first dimension
    which is samples, and second dimension which is observations).

    I.e. if ``X.shape`` is ``(3, 100, 5, 5, 3)`` this flatten the last
    dimensions to ``(3, 100, 75)``.

    """

    def _build(self, X):
        """Build the graph of this layer."""
        flat_dim = np.product(X.shape[2:])
        new_shape = tf.concat([tf.shape(X)[0:2], [flat_dim]], 0)
        Net = tf.reshape(X, new_shape)
        KL = 0.
        return Net, KL


#
# Kernel Approximation Layers
#

class RandomFourier(SampleLayer3):
    r"""Random Fourier feature (RFF) kernel approximation layer.

    Parameters
    ----------
    n_features : int
        the number of unique random features, the actual output dimension of
        this layer will be ``2 * n_features``.
    kernel : kernels.ShiftInvariant
        the kernel object that yeilds the random samples from the fourier
        spectrum of a particular kernel to approximate. See the :ref:`kernels`
        module.

    Note
    ----
    This should be followed by a dense layer to properly implement a kernel
    approximation.

    """

    def __init__(self, n_features, kernel):
        """Construct and instance of a RandomFourier object."""
        self.n_features = n_features
        self.kernel = kernel

    def _build(self, X):
        """Build the graph of this layer."""
        # Random weights
        n_samples, (input_dim,) = self._get_X_dims(X)
        dtype = X.dtype.as_numpy_dtype
        P, KL = self.kernel.weights(input_dim, self.n_features, dtype)
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

    Parameters
    ----------
    n_features : int
        the number of unique random features, the actual output dimension of
        this layer will be ``2 * n_features``.
    lenscale : float, ndarray, optional
        The length scales of the arc-cosine kernel. This can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim,) for an
        automatic relevance detection (ARD) kernel. If not provided, it will
        be set to ``sqrt(1 / input_dim)`` (this is similar to the 'auto'
        setting for a scikit learn SVM with a RBF kernel).
        If learn_lenscale is True, lenscale will be its initial value.
    p : int
        The order of the arc-cosine kernel, this must be an integer greater
        than, or eual to zero. 0 will lead to sigmoid-like kernels, 1 will lead
        to relu-like kernels, 2 quadratic-relu kernels etc.
    variational : bool
        use variational features instead of random features, (i.e. VAR-FIXED in
        [2]).
    learn_lenscale : bool
        Whether to learn the length scale. If True, the lenscale value provided
        is used for initialisation.

    Note
    ----
    This should be followed by a dense layer to properly implement a kernel
    approximation.

    See Also
    --------
    [1] Cho, Youngmin, and Lawrence K. Saul.
        "Analysis and extension of arc-cosine kernels for large margin
        classification." arXiv preprint arXiv:1112.3712 (2011).
    [2] Cutajar, K. Bonilla, E. Michiardi, P. Filippone, M.
        Random Feature Expansions for Deep Gaussian Processes. In ICML, 2017.

    """

    def __init__(self, n_features, lenscale=None, p=1, variational=False,
                 learn_lenscale=False):
        """Create an instance of an arc cosine kernel layer."""
        # Setup random weights
        if variational:
            kern = RBFVariational(lenscale=lenscale,
                                  learn_lenscale=learn_lenscale)
        else:
            kern = RBF(lenscale=lenscale, learn_lenscale=learn_lenscale)
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
# Variational Weight layers
#

class Conv2DVariational(SampleLayer):
    r"""A 2D convolution layer, with variational inference.

    (Does not currently support full covariance weights.)

    Parameters
    ----------
    filters : int
        the dimension of the output of this layer (i.e. the number of filters
        in the convolution).
    kernel_size : int, tuple or list
        width and height of the 2D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
    strides : int, tuple or list
        the strides of the convolution along the height and width. Can be a
        single integer to specify the same value for all spatial dimensions
    padding : str
        One of 'SAME' or 'VALID'. Defaults to 'SAME'. The type of padding
        algorithm to use.
    prior_std : str, float
        the value of the weight prior standard deviation
        (:math:`\sigma` above). The user can also provide a string to specify
        an initialisation function. Defaults to 'glorot'. If a string,
        must be one of 'glorot' or 'autonorm'.
    learn_prior: bool, optional
        Whether to learn the prior standard deviation.
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.

    """

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME',
                 prior_std='glorot', learn_prior=False, use_bias=True):
        """Create and instance of a variational Conv2D layer."""
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = [1] + list(strides) + [1]
        self.padding = padding
        self.use_bias = use_bias
        self.prior_std0 = prior_std
        self.learn_prior = learn_prior

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, (height, width, channels) = self._get_X_dims(X)
        W_shp, b_shp = self._weight_shapes(channels)

        # get effective IO shapes, DAN's fault if this is wrong
        receptive_field = np.product(W_shp[:-2])
        n_inputs = receptive_field * channels
        n_outputs = receptive_field * self.filters

        self.pstd, self.qstd = initialise_stds(n_inputs, n_outputs,
                                               self.prior_std0,
                                               self.learn_prior, "conv2d")
        # Layer weights
        self.pW = _make_prior(self.pstd, W_shp)
        self.qW = _make_posterior(self.qstd, W_shp, False, "conv")

        # Regularizers
        KL = kl_sum(self.qW, self.pW)

        # Linear layer
        Wsamples = _sample_W(self.qW, n_samples, False)
        Net = tf.map_fn(
            lambda args: tf.nn.conv2d(*args,
                                      padding=self.padding,
                                      strides=self.strides),
            elems=(X, Wsamples), dtype=tf.float32)

        # Optional bias
        if self.use_bias:
            # Layer intercepts
            self.pb = _make_prior(self.pstd, b_shp)
            self.qb = _make_posterior(self.qstd, b_shp, False, "conv_bias")

            # Regularizers
            KL += kl_sum(self.qb, self.pb)

            # Linear layer
            bsamples = tf.reshape(_sample_W(self.qb, n_samples, False),
                                  [n_samples, 1, 1, 1, self.filters])
            Net += bsamples

        return Net, KL

    def _weight_shapes(self, channels):
        """Generate weight and bias weight shape tuples."""
        weight_shape = self.kernel_size + (channels, self.filters)
        bias_shape = (self.filters,)

        return weight_shape, bias_shape


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

    This layer will use variational inference to learn the posterior
    parameters, and optionally the ``prior_std`` parameter can be  learned
    if ``learn_prior`` is set to True. The given value is then used to
    initialize.

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
    prior_std : str, float
        the value of the weight prior standard deviation
        (:math:`\sigma` above). The user can also provide a string to specify
        an initialisation function. Defaults to 'glorot'. If a string,
        must be one of 'glorot' or 'autonorm'.
    learn_prior : bool, optional
        Whether to learn the prior
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.

    """

    def __init__(self, output_dim, prior_std=1., learn_prior=False, full=False,
                 use_bias=True):
        """Create and instance of a variational dense layer."""
        self.output_dim = output_dim
        self.full = full
        self.use_bias = use_bias
        self.prior_std0 = prior_std
        self.learn_prior = learn_prior

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, (input_dim,) = self._get_X_dims(X)
        W_shp, b_shp = self._weight_shapes(input_dim)

        self.pstd, self.qstd = initialise_stds(input_dim, self.output_dim,
                                               self.prior_std0,
                                               self.learn_prior, "dense")

        # Layer weights
        self.pW = _make_prior(self.pstd, W_shp)
        self.qW = _make_posterior(self.qstd, W_shp, self.full, "dense")

        # Regularizers
        KL = kl_sum(self.qW, self.pW)

        # Linear layer
        Wsamples = _sample_W(self.qW, n_samples)
        Net = tf.matmul(X, Wsamples)

        # Optional bias
        if self.use_bias:
            # Layer intercepts
            self.pb = _make_prior(self.pstd, b_shp)
            self.qb = _make_posterior(self.qstd, b_shp, False, "dense_bias")

            # Regularizers
            KL += kl_sum(self.qb, self.pb)

            # Linear layer
            bsamples = tf.expand_dims(_sample_W(self.qb, n_samples), 1)
            Net += bsamples

        return Net, KL

    def _weight_shapes(self, input_dim):
        """Generate weight and bias weight shape tuples."""
        weight_shape = (self.output_dim, input_dim)
        bias_shape = (self.output_dim,)

        return weight_shape, bias_shape


class EmbedVariational(DenseVariational):
    r"""Dense (fully connected) embedding layer, with variational inference.

    This layer works directly on inputs of *K* category *indices* rather than
    one-hot representations, for efficiency. Note, this only works on a single
    column, see the ``PerFeature`` layer to embed multiple columns. Eg.


    .. code::

        cat_layers = [EmbedVar(10, k) for k in x_categories]

        net = (
            ab.InputLayer(name="X", n_samples=n_samples_) >>
            ab.PerFeature(*cat_layers) >>
            ab.Activation(tf.nn.selu) >>
            ...
        )

    This layer is a effectively a ``DenseVariational`` layer,

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

    This layer will use variational inference to learn the posterior
    parameters, and optionally the ``prior_std`` parameter can be learned
    if ``learn_prior`` is set to True. The ``prior_std`` value given will
    be used for initialization.

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
    prior_std : str, float
        the value of the weight prior standard deviation
        (:math:`\sigma` above). The user can also provide a string to specify
        an initialisation function. Defaults to 'glorot'. If a string,
        must be one of 'glorot' or 'autonorm'.
    learn_prior : bool, optional
        Whether to learn the prior
    full : bool
        If true, use a full covariance Gaussian posterior for *each* of the
        output weight columns, otherwise use an independent (diagonal) Normal
        posterior.

    """

    def __init__(self, output_dim, n_categories, prior_std=1.,
                 learn_prior=False, full=False):
        """Create and instance of a variational dense embedding layer."""
        assert n_categories >= 2, "Need 2 or more categories for embedding!"
        self.output_dim = output_dim
        self.n_categories = n_categories
        self.full = full
        self.prior_std0 = prior_std
        self.learn_prior = learn_prior

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, (input_dim,) = self._get_X_dims(X)
        W_shape, _ = self._weight_shapes(self.n_categories)
        n_batch = tf.shape(X)[1]

        self.pstd, self.qstd = initialise_stds(input_dim, self.output_dim,
                                               self.prior_std0,
                                               self.learn_prior, "embed")

        # Layer weights
        self.pW = _make_prior(self.pstd, W_shape)
        self.qW = _make_posterior(self.qstd, W_shape, self.full, "embed")

        # Index into the relevant weights rather than using sparse matmul
        Wsamples = _sample_W(self.qW, n_samples)
        features = tf.map_fn(lambda wx: tf.gather(*wx, axis=0), (Wsamples, X),
                             dtype=Wsamples.dtype)

        # Now concatenate the resulting features on the last axis
        f_dims = int(np.prod(features.shape[2:]))  # need this for placeholders
        Net = tf.reshape(features, [n_samples, n_batch, f_dims])

        # Regularizers
        KL = kl_sum(self.qW, self.pW)

        return Net, KL


#
# Noise Contrastive Layers
#

class NCPContinuousPerturb(SampleLayer):
    r"""Noise Constrastive Prior continous variable perturbation layer.

    This layer doubles the number of samples going through the model, and adds
    a random normal perturbation to the second set of samples. This implements
    Equation 3 in "Reliable Uncertainty Estimates in Deep Neural Networks using
    Noise Contrastive Priors" https://arxiv.org/abs/1807.09289.

    This should be the *first* layer in a network after an input layer, and
    needs to be used in conjuction with ``DenseNCP``. For example:

    .. code::

        net = (
            ab.InputLayer(name="X", n_samples=n_samples_) >>
            ab.NCPContinuousPerturb() >>
            ab.Dense(output_dim=32) >>
            ab.Activation(tf.nn.selu) >>
            ...
            ab.Dense(output_dim=8) >>
            ab.Activation(tf.nn.selu) >>
            ab.DenseNCP(output_dim=1)
        )

    Parameters
    ----------
    input_noise : float, tf.Tensor, tf.Variable
        The standard deviation of the random perturbation to add to the inputs.

    """

    def __init__(self, input_noise=1.):
        """Instantiate a NCPContinuousPerturb layer."""
        self.input_noise = input_noise

    def _build(self, X):
        # calculate the perturbation
        loc = tf.constant(0.)
        noise_dist = tf.distributions.Normal(loc, self.input_noise)
        noise = noise_dist.sample(tf.shape(X))

        X_pert = tf.concat([X, X + noise], axis=0)
        return X_pert, 0.0


class NCPCategoricalPerturb(SampleLayer):
    r"""Noise Constrastive Prior categorical variable perturbation layer.

    This layer doubles the number of samples going through the model, and
    randomly flips the categories in the second set of samples. This implements
    (the categorical version of) Equation 3 in "Reliable Uncertainty Estimates
    in Deep Neural Networks using Noise Contrastive Priors"
    https://arxiv.org/abs/1807.09289.

    The choice to randomly flip a category is drawn from a Bernoulli
    distribution per sample (with probability ``flip_prob``), then the new
    category is randomly chosen with probability ``1 / n_categories``.

    This should be the *first* layer in a network after an input layer, and
    needs to be used in conjuction with ``DenseNCP``. Also, like the embedding
    layers, this only applies to *one column of categorical inputs*, so we
    advise you use it with the ``PerFeature`` layer. For example:

    .. code::

        cat_layers = [
            (NCPCategoricalPerturb(k) >> Embed(10, k))
            for k in x_categories
        ]

        net = (
            ab.InputLayer(name="X", n_samples=n_samples_) >>
            ab.PerFeature(*cat_layers) >>
            ab.Activation(tf.nn.selu) >>
            ab.Dense(output_dim=32) >>
            ab.Activation(tf.nn.selu) >>
            ...
            ab.Dense(output_dim=8) >>
            ab.Activation(tf.nn.selu) >>
            ab.DenseNCP(output_dim=1)
        )

    Parameters
    ----------
    input_noise : float, tf.Tensor, tf.Variable
        The standard deviation of the random perturbation to add to the inputs.

    """

    def __init__(self, n_categories, flip_prob=0.1):
        """Instantiate a NCPCategoricalPerturb layer."""
        self.n_categories = n_categories
        self.flip_prob = flip_prob

    def _build(self, X):
        dim = tf.shape(X)

        # Binary decision to flip category
        mask_dist = tf.distributions.Bernoulli(probs=self.flip_prob)
        mask = mask_dist.sample(dim)

        # Uniform categorical to choose which category to flip to
        p = tf.ones(self.n_categories) / self.n_categories
        flip_dist = tf.distributions.Categorical(probs=p)
        flips = flip_dist.sample(dim)

        # Flip and concatenate
        X_flips = (mask * X) + ((1 - mask) * flips)
        X_pert = tf.concat([X, X_flips], axis=0)
        return X_pert, 0.


class DenseNCP(DenseVariational):
    r"""A DenseVariational layer with Noise Constrastive Prior.

    This is basically just a ``DenseVariational`` layer, but with an added
    Kullback Leibler penalty on the latent function, as derived in Equation (6)
    in "Reliable Uncertainty Estimates in Deep Neural Networks using Noise
    Contrastive Priors" https://arxiv.org/abs/1807.09289.

    This should be the *last* layer in a network, and needs to be used in
    conjuction with ``NCPContinuousPerturb`` and/or ``NCPCategoricalPerturb``
    layers (after an input layer). For example:

    .. code::

        net = (
            ab.InputLayer(name="X", n_samples=n_samples_) >>
            ab.NCPContinuousPerturb() >>
            ab.Dense(output_dim=32) >>
            ab.Activation(tf.nn.selu) >>
            ...
            ab.Dense(output_dim=8) >>
            ab.Activation(tf.nn.selu) >>
            ab.DenseNCP(output_dim=1)
        )

    As you can see from this example, we have only made the last layer
    probabilistic/Bayesian (``DenseNCP``), and have left the rest of the
    network maximum likelihood/MAP. This is also how the original authors of
    the algorithm have implemented it. While this layer also works with
    ``DenseVariational`` layers (etc.) this is not how is has been originally
    implemented, and the contribution of uncertainty from these layers to the
    latent function will not be accounted for in this layer. This is because
    the nonlinear activations between layers make evaluating this density
    intractable, unless we had something like normalising flows.

    Parameters
    ----------
    output_dim : int
        the dimension of the output of this layer
    prior_std : str, float
        the value of the weight prior standard deviation
        (:math:`\sigma` above). The user can also provide a string to specify
        an initialisation function. Defaults to 'glorot'. If a string,
        must be one of 'glorot' or 'autonorm'.
    learn_prior : bool, optional
        Whether to learn the prior on the weights.
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.
    latent_mean : float
        The prior mean over the latent function(s) on the output of this layer.
        This specifies what value the latent function should take away from the
        support of the training data.
    latent_std : float
        The prior standard deviation over the latent function(s) on the output
        of this layer. This controls the strength of the regularisation away
        from the latent mean.

    Note
    ----
    This implementation is inspired by:
    https://github.com/brain-research/ncp/blob/master/ncp/models/bbb_ncp.py

    """

    def __init__(self, output_dim, prior_std=1., learn_prior=False,
                 use_bias=True, latent_mean=0., latent_std=1.):
        """Instantiate a DenseNCP layer."""
        super().__init__(
            output_dim=output_dim,
            prior_std=prior_std,
            learn_prior=learn_prior,
            full=False,
            use_bias=use_bias
        )
        self.f_prior = tf.distributions.Normal(latent_mean, latent_std)

    def _build(self, X):
        # Extract perturbed predictions
        n_samples = tf.shape(X)[0] // 2
        X_orig, X_pert = X[:n_samples], X[n_samples:]

        # Build Dense Layer
        F, KL = super()._build(X_orig)

        # Build a latent function density
        qWmean = _tile2samples(n_samples, tf.transpose(self.qW.mean()))
        qWvar = _tile2samples(n_samples, tf.transpose(self.qW.variance()))
        f_loc = tf.matmul(X_pert, qWmean)
        if self.use_bias:
            f_loc += self.qb.mean()
        f_scale = tf.sqrt(tf.matmul(X_pert ** 2, qWvar))
        f_post = tf.distributions.Normal(f_loc, f_scale)

        # Calculate NCP loss
        KL += kl_sum(f_post, self.f_prior) / tf.to_float(n_samples)

        return F, KL


#
# Maximum likelihood/MAP Weight layers
#

class Conv2D(SampleLayer):
    r"""A 2D convolution layer.

    This layer uses maximum likelihood or maximum *a-posteriori* inference to
    learn the convolutional kernels and biases, and so also returns complexity
    penalities (l1 or l2) for the weights and biases.

    Parameters
    ----------
    filters : int
        the dimension of the output of this layer (i.e. the number of filters
        in the convolution).
    kernel_size : int, tuple or list
        width and height of the 2D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
    strides : int, tuple or list
        the strides of the convolution along the height and width. Can be a
        single integer to specify the same value for all spatial dimensions
    padding : str
        One of 'SAME' or 'VALID'. Defaults to 'SAME'. The type of padding
        algorithm to use.
    l1_reg : float
        the value of the l1 weight regularizer,
        :math:`\text{l1_reg} \times \|\mathbf{W}\|_1`
    l2_reg : float
        the value of the l2 weight regularizer,
        :math:`\frac{1}{2} \text{l2_reg} \times \|\mathbf{W}\|^2_2`
    use_bias : bool
        If true, also learn a bias weight, e.g. a constant offset weight.
    init_fn : str, callable
        The function to use to initialise the weights. The default is
        'glorot_trunc', the truncated normal glorot function. If supplied,
        the callable takes a shape (input_dim, output_dim) as an argument
        and returns the weight matrix.

    """

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME',
                 l1_reg=0., l2_reg=0., use_bias=True, init_fn='glorot_trunc'):
        """Create and instance of a variational Conv2D layer."""
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = [1] + list(strides) + [1]
        self.padding = padding
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.use_bias = use_bias
        self.init_fn = init_fn

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, (height, width, channels) = self._get_X_dims(X)
        W_shape, b_shape = self._weight_shapes(channels)

        W_init = initialise_weights(W_shape, self.init_fn)
        W = tf.Variable(W_init, name="W_map")
        summary_histogram(W)

        Net = tf.map_fn(
            lambda x: tf.nn.conv2d(x, W,
                                   padding=self.padding,
                                   strides=self.strides), X)
        # Regularizers
        penalty = self.l2 * tf.nn.l2_loss(W) + self.l1 * _l1_loss(W)

        # Optional Bias
        if self.use_bias:
            b_init = initialise_weights(b_shape, self.init_fn)
            b = tf.Variable(b_init, name="b_map")
            summary_histogram(b)

            Net = tf.nn.bias_add(Net, b)
            penalty += self.l2 * tf.nn.l2_loss(b) + self.l1 * _l1_loss(b)

        return Net, penalty

    def _weight_shapes(self, channels):
        """Generate weight and bias weight shape tuples."""
        weight_shape = self.kernel_size + (channels, self.filters)
        bias_shape = (self.filters,)

        return weight_shape, bias_shape


class Dense(SampleLayer):
    r"""Dense (fully connected) linear layer.

    This implements a linear layer, and when called returns

    .. math::
        f(\mathbf{X}) = \mathbf{X} \mathbf{W} + \mathbf{b}

    where :math:`\mathbf{X} \in \mathbb{R}^{N \times D_{in}}`,
    :math:`\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}` and
    :math:`\mathbf{b} \in \mathbb{R}^{D_{out}}`. This layer uses maximum
    likelihood or maximum *a-posteriori* inference to learn the weights and
    biases, and so also returns complexity penalities (l1 or l2) for the
    weights and biases.

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
    init_fn : str, callable
        The function to use to initialise the weights. The default is
        'glorot', the uniform glorot function. If supplied,
        the callable takes a shape (input_dim, output_dim) as an argument
        and returns the weight matrix.

    """

    def __init__(self, output_dim, l1_reg=0., l2_reg=0., use_bias=True,
                 init_fn='glorot'):
        """Create and instance of a dense layer with regularizers."""
        self.output_dim = output_dim
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.use_bias = use_bias
        self.init_fn = init_fn

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, input_shape = self._get_X_dims(X)
        Wdim = input_shape + [self.output_dim]

        W_init = initialise_weights(Wdim, self.init_fn)
        W = tf.Variable(W_init, name="W_map")
        summary_histogram(W)

        # Tiling W is much faster than mapping (tf.map_fn) the matmul
        Net = tf.matmul(X, _tile2samples(n_samples, W))

        # Regularizers
        penalty = self.l2 * tf.nn.l2_loss(W) + self.l1 * _l1_loss(W)

        # Optional Bias
        if self.use_bias is True:
            b_init = initialise_weights((1, self.output_dim), self.init_fn)
            b = tf.Variable(b_init, name="b_map")
            summary_histogram(b)

            Net += b
            penalty += self.l2 * tf.nn.l2_loss(b) + self.l1 * _l1_loss(b)

        return Net, penalty


class Embed(SampleLayer3):
    r"""Dense (fully connected) embedding layer.

    This layer works directly on inputs of *K* category *indices* rather than
    one-hot representations, for efficiency. Note, this only works on a single
    column, see the ``PerFeature`` layer to embed multiple columns. E.g.

    .. code::

        cat_layers = [Embed(10, k) for k in x_categories]

        net = (
            ab.InputLayer(name="X", n_samples=n_samples_) >>
            ab.PerFeature(*cat_layers) >>
            ab.Activation(tf.nn.selu) >>
            ...
        )

    It is a dense linear layer,

    .. math::
        f(\mathbf{X}) = \mathbf{X} \mathbf{W}

    Here :math:`\mathbf{X} \in \mathbb{N}_2^{N \times K}` and :math:`\mathbf{W}
    \in \mathbb{R}^{K \times D_{out}}`. Though in code we represent
    :math:`\mathbf{X}` as a vector of indices in :math:`\mathbb{N}_K^{N \times
    1}`. This layer uses maximum likelihood or maximum *a-posteriori* inference
    to learn the weights and so also returns complexity penalities (l1 or l2)
    for the weights.

    Parameters
    ----------
    output_dim : int
        the dimension of the output (embedding) of this layer
    n_categories : int
        the number of categories in the input variable
    l1_reg : float
        the value of the l1 weight regularizer,
        :math:`\text{l1_reg} \times \|\mathbf{W}\|_1`
    l2_reg : float
        the value of the l2 weight regularizer,
        :math:`\frac{1}{2} \text{l2_reg} \times \|\mathbf{W}\|^2_2`
    init_fn : str, callable
        The function to use to initialise the weights. The default is
        'glorot', the uniform glorot function. If supplied,
        the callable takes a shape (input_dim, output_dim) as an argument
        and returns the weight matrix.


    """

    def __init__(self, output_dim, n_categories, l1_reg=0., l2_reg=0.,
                 init_fn='glorot'):
        """Create and instance of an embedding layer."""
        assert n_categories >= 2, "Need 2 or more categories for embedding!"
        self.output_dim = output_dim
        self.n_categories = n_categories
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.init_fn = init_fn

    def _build(self, X):
        """Build the graph of this layer."""
        n_samples, (input_dim,) = self._get_X_dims(X)
        Wdim = (self.n_categories, self.output_dim)
        n_batch = tf.shape(X)[1]

        W_init = initialise_weights(Wdim, self.init_fn)
        W = tf.Variable(W_init, name="W_map")
        summary_histogram(W)

        # Index into the relevant weights rather than using sparse matmul
        features = tf.gather(W, X, axis=0)
        f_dims = int(np.prod(features.shape[2:]))  # need this for placeholders
        Net = tf.reshape(features, [n_samples, n_batch, f_dims])

        # Regularizers
        penalty = self.l2 * tf.nn.l2_loss(W) + self.l1 * _l1_loss(W)

        return Net, penalty


#
# Private module stuff
#

def _tile2samples(n_samples, tensor):
    """Tile a tensor along axis 0 to match the number of samples."""
    new_shape = [n_samples] + ([1] * len(tensor.shape))
    tiled = tf.tile(tf.expand_dims(tensor, 0), new_shape)
    return tiled


def _l1_loss(X):
    r"""Calculate the L1 loss of X, :math:`\|X\|_1`."""
    l1 = tf.reduce_sum(tf.abs(X))
    return l1


def _is_dim(distribution, dims):
    r"""Check if ``X``'s dimension is the same as the tuple ``dims``."""
    shape = tuple(distribution.loc.shape)
    return shape == dims


def _sample_W(dist, n_samples, transpose=True):
    """Get samples of the weight distributions for the re-param trick."""
    Wsamples = dist.sample(seed=next(seedgen), sample_shape=n_samples)
    tf.add_to_collection("SampleTensors", Wsamples)
    rank = len(Wsamples.shape)
    if rank > 2 and transpose:
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        Wsamples = tf.transpose(Wsamples, perm)
    return Wsamples


def _make_prior(std, weight_shape):
    """Check/make prior weight distributions."""
    prior_W = norm_prior(weight_shape, std=std)
    assert _is_dim(prior_W, weight_shape), \
        "Prior inconsistent dimension!"
    return prior_W


def _make_posterior(std, weight_shape, full, suffix=None):
    """Check/make posterior."""
    # We don't want a full-covariance on an intercept, check input_dim
    if full and len(weight_shape) > 1:
        post_W = gaus_posterior(weight_shape, std0=std, suffix=suffix)
    else:
        post_W = norm_posterior(weight_shape, std0=std, suffix=suffix)
    assert _is_dim(post_W, weight_shape), \
        "Posterior inconsistent dimension!"
    return post_W
