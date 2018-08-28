"""Random kernel classes for use with the RandomKernel layers."""
import numpy as np
import tensorflow as tf

from aboleth.distributions import norm_posterior, kl_sum
from aboleth.util import pos_variable, summary_histogram, summary_scalar


#
# Random Fourier Kernels
#

class ShiftInvariant:
    """Abstract base class for shift invariant kernel approximations.

    Parameters
    ----------
    lenscale : float, ndarray, optional
        The length scales of the shift invariant kernel. This can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim,) for an
        automatic relevance detection (ARD) kernel. If learn_lenscale is
        True, lenscale will be its initial value.
    learn_lenscale : bool, optional
        Whether to learn the length scale. If True, the lenscale
        value provided (or its default) is used for initialisation.
    seed : int, optional
        The seed for the internal random number generator. Setting a fixed
        seed ensures that remaking the tensorflow graph results in the
        same weights.

    """

    def __init__(self, lenscale=None, learn_lenscale=False, seed=0):
        """Constuct a shift invariant kernel object."""
        self.given_lenscale = lenscale
        self.learn_lenscale = learn_lenscale
        self.lenscale = None
        self._random_state = np.random.RandomState(seed)

    def weights(self, input_dim, n_features, dtype=np.float32):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.
        dtype : np.dtype
            the dtype of the features to draw, this should match the
            observations.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel.

        """
        raise NotImplementedError("Abstract base class for shift invariant"
                                  " kernels!")


class RBF(ShiftInvariant):
    """Radial basis kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, optional
        The length scales of the RBF kernel. This can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim,) for an
        automatic relevance detection (ARD) kernel. If not provided, it will
        be set to ``sqrt(1 / input_dim)`` (this is similar to the 'auto'
        setting for a scikit learn SVM with a RBF kernel).
        If learn_lenscale is True, lenscale will be its initial value.
    learn_lenscale : bool, optional
        Whether to learn the length scale. If True, the lenscale
        value provided (or its default) is used for initialisation.
    seed : int, optional
        The seed for the internal random number generator. Setting a fixed
        seed ensures that remaking the tensorflow graph results in the
        same weights.

    """

    def weights(self, input_dim, n_features, dtype=np.float32):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.
        dtype : np.dtype
            the dtype of the features to draw, this should match the
            observations.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel (0.0).

        """
        self.lenscale, _ = _init_lenscale(self.given_lenscale,
                                          self.learn_lenscale,
                                          input_dim)
        e = self._random_state.randn(input_dim, n_features).astype(dtype)
        P = e / tf.expand_dims(self.lenscale, axis=-1)
        return P, 0.


class RBFVariational(ShiftInvariant):
    """Variational Radial basis kernel approximation.

    This kernel is similar to the RBF kernel, however we learn an independant
    Gaussian posterior distribution over the kernel weights to sample from.

    Parameters
    ----------
    lenscale : float, ndarray, optional
        The length scales of the RBF kernel. This can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim,) for an
        automatic relevance detection (ARD) kernel. If not provided, it will
        be set to ``sqrt(1 / input_dim)`` (this is similar to the 'auto'
        setting for a scikit learn SVM with a RBF kernel).
        If learn_lenscale is True, lenscale will be the initial value of the
        prior precision of the Fourier weight distribution.
    learn_lenscale : bool, optional
        Whether to learn the (prior) length scale. If True, the lenscale
        value provided (or its default) is used for initialisation.
    seed : int, optional
        The seed for the internal random number generator. Setting a fixed
        seed ensures that remaking the tensorflow graph results in the
        same weights.

    """

    def __init__(self, lenscale=None, learn_lenscale=False, seed=0):
        """Constuct an instance of the RBFVariational kernel."""
        self.lenscale_post = None
        super().__init__(lenscale, learn_lenscale, seed)

    def weights(self, input_dim, n_features, dtype=np.float32):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.
        dtype : np.dtype
            the dtype of the features to draw, this should match the
            observations.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel.

        """
        self.lenscale, self.lenscale_post = _init_lenscale(self.given_lenscale,
                                                           self.learn_lenscale,
                                                           input_dim)
        dim = (input_dim, n_features)

        # Setup the prior, lenscale may be a variable, so dont use prior_normal
        pP_scale = self.__len2std(self.lenscale, n_features)
        pP = tf.distributions.Normal(
            loc=tf.zeros(dim),
            scale=pP_scale)
        # Initialise the posterior
        qP_scale = 1.0 / self.lenscale_post
        if qP_scale.ndim > 0:
            qP_scale = np.repeat(qP_scale[:, np.newaxis], n_features, axis=1)
        qP = norm_posterior(dim=dim, std0=qP_scale, suffix="kernel")

        KL = kl_sum(qP, pP)

        # We implement the VAR-FIXED method here from Cutajar et. al 2017, so
        # we pre-generate and fix the standard normal samples
        e = self._random_state.randn(*dim).astype(dtype)
        P = qP.mean() + qP.stddev() * e

        return P, KL

    @staticmethod
    def __len2std(lenscale, n_features):
        std = tf.tile(1.0 / tf.expand_dims(lenscale, axis=-1), (1, n_features))
        return std


class Matern(ShiftInvariant):
    """Matern kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, optional
        The length scales of the Matern kernel. This can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim,) for an
        automatic relevance detection (ARD) kernel. If not provided, it will
        be set to ``sqrt(1 / input_dim)`` (this is similar to the 'auto'
        setting for a scikit learn SVM with a RBF kernel).
        If learn_lenscale is True, lenscale will be its initial value.
    learn_lenscale : bool, optional
        Whether to learn the length scale. If True, the lenscale
        value provided (or its default) is used for initialisation.
    p : int
        a zero or positive integer specifying the number of the Matern kernel,
        e.g. ``p == 0`` results int a Matern 1/2 kernel, ``p == 1``  results in
        the Matern 3/2 kernel etc.
    seed : int, optional
        The seed for the internal random number generator. Setting a fixed
        seed ensures that remaking the tensorflow graph results in the
        same weights.

    """

    def __init__(self, lenscale=None, learn_lenscale=False, p=1, seed=0):
        """Constuct a Matern kernel object."""
        super().__init__(lenscale, learn_lenscale, seed)
        assert isinstance(p, int) and p >= 0
        self.p = p

    def weights(self, input_dim, n_features, dtype=np.float32):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.
        dtype : np.dtype
            the dtype of the features to draw, this should match the
            observations.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel (0.0).

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
        self.lenscale, _ = _init_lenscale(self.given_lenscale,
                                          self.learn_lenscale, input_dim)
        df = 2 * (self.p + 0.5)
        y = self._random_state.randn(input_dim, n_features)
        u = self._random_state.chisquare(df, size=(n_features,))
        P = (y * np.sqrt(df / u)).astype(dtype) / \
            tf.expand_dims(self.lenscale, axis=-1)
        return P, 0.


def _init_lenscale(given_lenscale, learn_lenscale, input_dim):
    """Provide the lenscale variable and its initial value."""
    given_lenscale = (np.sqrt(1.0 / input_dim) if given_lenscale is None
                      else np.array(given_lenscale).squeeze()).astype(
                          np.float32)

    if learn_lenscale:
        lenscale = pos_variable(given_lenscale, name="kernel_lenscale")
        if np.size(given_lenscale) == 1:
            summary_scalar(lenscale)
        else:
            summary_histogram(lenscale)
    else:
        lenscale = given_lenscale

    lenscale_vec = tf.ones(input_dim, dtype=tf.float32) * lenscale
    init_lenscale = given_lenscale * np.ones(input_dim, dtype=np.float32)
    return lenscale_vec, init_lenscale
