import tensorflow as tf
import numpy as np

from layers import Activation


class RandomRBF(Activation):

    def __init__(self, input_dim, output_dim, name=None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        self.P = self._weights().astype(np.float32)
        self._D = tf.to_float(self.output_dim)

    def __call__(self, X):
        XP = tf.matmul(X, self.P)
        real = tf.cos(XP)
        imag = tf.sin(XP)
        return tf.concat([real, imag], axis=1) / tf.sqrt(self._D)

    def _weights(self):
        P = np.random.randn(self.input_dim, self.output_dim)
        return P


class RandomMatern32(RandomRBF):
    p = 1.

    def _weights(self):

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
        y = np.random.randn(self.input_dim, self.output_dim)
        u = np.random.chisquare(df, size=(self.output_dim,))
        return y * np.sqrt(df / u)


class RandomMatern52(RandomMatern32):
    p = 2.
