"""Random kernel classes for use with the RandomKernel layers."""
import numpy as np
import tensorflow as tf

from aboleth.random import seedgen
from aboleth.distributions import norm_posterior, kl_sum


#
# Random Fourier Kernels
#

class ShiftInvariant:
    """Abstract base class for shift invariant kernel approximations.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor, Variable
        the length scales of the shift invariant kernel, this can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim, 1) for an
        automatic relevance detection (ARD) kernel. If you wish to learn this
        parameter, make it a Variable (or ``ab.pos(tf.Variable(...))`` to keep
        it positively constrained).

    """

    def __init__(self, lenscale=1.0):
        """Constuct a shift invariant kernel object."""
        self.lenscale = lenscale

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
    lenscale : float, ndarray, Tensor, Variable
        the length scales of the shift invariant kernel, this can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim, 1) for an
        automatic relevance detection (ARD) kernel. If you wish to learn this
        parameter, make it a Variable (or ``ab.pos(tf.Variable(...))`` to keep
        it positively constrained).

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
        rand = np.random.RandomState(next(seedgen))
        e = rand.randn(input_dim, n_features).astype(dtype)
        P = e / self.lenscale
        return P, 0.


class RBFVariational(ShiftInvariant):
    """Variational Radial basis kernel approximation.

    This kernel is similar to the RBF kernel, however we learn an independant
    Gaussian posterior distribution over the kernel weights to sample from.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor, Variable
        the length scales of the shift invariant kernel, this can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim, 1) for an
        automatic relevance detection (ARD) kernel. If you wish to learn this
        parameter, make it a Variable (or ``ab.pos(tf.Variable(...))`` to keep
        it positively constrained).
    lenscale_posterior : float, ndarray, optional
        the *initial* value for the posterior length scale, this can be a
        scalar or vector (different initial value per input dimension). If this
        is left as None, it will be set to ``sqrt(1 / input_dim)`` (this is
        similar to the 'auto' setting for a scikit learn SVM with a RBF
        kernel).

    """

    def __init__(self, lenscale=1.0, lenscale_posterior=None):
        """Constuct an instance of the RBFVariational kernel."""
        super().__init__(lenscale)
        self.lenscale_post = lenscale_posterior

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
        dim = (input_dim, n_features)

        # Setup the prior, lenscale may be a variable, so dont use prior_normal
        pP = tf.distributions.Normal(
            loc=tf.zeros(dim),
            scale=self.__len2std(self.lenscale)
        )

        # Initialise the posterior
        if self.lenscale_post is None:
            self.lenscale_post = np.sqrt(1 / input_dim)
        qP = norm_posterior(dim=dim, std0=self.__len2std(self.lenscale_post),
                            suffix="kernel")

        KL = kl_sum(qP, pP)

        # We implement the VAR-FIXED method here from Cutajar et. al 2017, so
        # we pre-generate and fix the standard normal samples
        rand = np.random.RandomState(next(seedgen))
        e = rand.randn(*dim).astype(dtype)
        P = qP.mean() + qP.stddev() * e

        return P, KL

    @staticmethod
    def __len2std(lenscale):
        std = tf.to_float(1. / lenscale)
        return std


class Matern(ShiftInvariant):
    """Matern kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor, Variable
        the length scales of the shift invariant kernel, this can be a scalar
        for an isotropic kernel, or a vector of shape (input_dim, 1) for an
        automatic relevance detection (ARD) kernel. If you wish to learn this
        parameter, make it a Variable (or ``ab.pos(tf.Variable(...))`` to keep
        it positively constrained).
    p : int
        a zero or positive integer specifying the number of the Matern kernel,
        e.g. ``p == 0`` results int a Matern 1/2 kernel, ``p == 1``  results in
        the Matern 3/2 kernel etc.

    """

    def __init__(self, lenscale=1.0, p=1):
        """Constuct a Matern kernel object."""
        super().__init__(lenscale)
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
        df = 2 * (self.p + 0.5)
        rand = np.random.RandomState(next(seedgen))
        y = rand.randn(input_dim, n_features)
        u = rand.chisquare(df, size=(n_features,))
        P = (y * np.sqrt(df / u)).astype(dtype) / self.lenscale
        return P, 0.
