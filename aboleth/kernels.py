"""Random kernel classes for use with the RandomKernel layers."""
import numpy as np
import tensorflow as tf

from aboleth.random import seedgen
from aboleth.distributions import norm_prior, norm_posterior, kl_qp


#
# Random Fourier Kernels
#

class ShiftInvariant:
    """Abstract base class for shift invariant kernel approximations.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor
        the lenght scales of the shift invariant kernel, this can be a scalar
        for an isotropic kernel, or a vector for an automatic relevance
        detection (ARD) kernel.

    """

    def __init__(self, lenscale=1.0):
        """Constuct a shift invariant kernel object."""
        self.lenscale = lenscale

    def weights(self, n_samples, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        n_samples : int
            the number of random samples for stochastic variational bayes.
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(n_samples, input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel.

        """
        raise NotImplementedError("Abstract base class for shift invariant"
                                  " kernels!")


class RBF(ShiftInvariant):
    """Radial basis kernel approximation.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor
        the lenght scales of the radial basis kernel, this can be a scalar for
        an isotropic kernel, or a vector for an automatic relevance detection
        (ARD) kernel.

    """

    def weights(self, n_samples, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        n_samples : int
            the number of random samples for stochastic variational bayes.
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(n_samples, input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel (0.0).

        """
        rand = np.random.RandomState(next(seedgen))
        # FIXME should we be using MORE random draws from the priors instead of
        # tiling?
        P = rand.randn(input_dim, n_features).astype(np.float32)
        # P = rand.randn(n_samples, input_dim, n_features).astype(np.float32)
        P = P / self.lenscale
        Ps = _tile_weights(n_samples, P)
        return Ps, 0.
        # return P, 0.


class RBFVariational(ShiftInvariant):
    """Variational Radial basis kernel approximation.

    This kernel is similar to the RBF kernel, however we learn an independant
    Gaussian posterior distribution over the kernel weights to sample from.

    Parameters
    ----------
    lenscale : float, ndarray, Tensor
        the lenght scales of the radial basis kernel, this can be a scalar for
        an isotropic kernel, or a vector for an automatic relevance detection
        (ARD) kernel.

    """

    def weights(self, n_samples, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        n_samples : int
            the number of random samples for stochastic variational bayes.
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(n_samples, input_dim, n_features)``.
        KL : Tensor, float
            the KL penalty associated with the parameters in this kernel.

        """
        var = 1. / self.lenscale**2
        dim = (input_dim, n_features)
        pP = norm_prior(dim=dim, var=var)
        # FIXME var0 should be the 1 / lenscale**2!
        # FIXME why aren't we using lenscale as std?
        # FIXME should we be using MORE random draws from the priors in the
        #   other kernel objects?
        qP = norm_posterior(dim=dim, var0=1.)

        KL = kl_qp(qP, pP)
        Ps = tf.stack([qP.sample() for _ in range(n_samples)])

        return Ps, KL


class Matern(ShiftInvariant):
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

    def weights(self, n_samples, input_dim, n_features):
        """Generate the random fourier weights for this kernel.

        Parameters
        ----------
        n_samples : int
            the number of random samples for stochastic variational bayes.
        input_dim : int
            the input dimension to this layer.
        n_features : int
            the number of unique random features, the actual output dimension
            of this layer will be ``2 * n_features``.

        Returns
        -------
        P : ndarray
            the random weights of the fourier features of shape
            ``(n_samples, input_dim, n_features)``.
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
        # FIXME should we be using MORE random draws from the priors instead of
        # tiling?
        y = rand.randn(input_dim, n_features)
        u = rand.chisquare(df, size=(n_features,))
        P = (y * np.sqrt(df / u)).astype(np.float32) / self.lenscale
        Ps = _tile_weights(n_samples, P)
        return Ps, 0.


#
# Private module utilities
#

# FIXME should we be using MORE random draws from the priors instead of tiling?
def _tile_weights(n_samples, P):
    Ps = tf.tile(tf.expand_dims(P, 0), [n_samples, 1, 1])
    return Ps
