"""Random kernel classes for use with the RandomKernel layers."""
import numpy as np

from aboleth.random import seedgen


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
