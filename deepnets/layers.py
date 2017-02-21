import tensorflow as tf

from deepnets.utils import pos
from deepnets.likelihoods import Normal


class Layer():

    def __init__(self, input_dim, output_dim, name=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def __call__(self, X):
        raise NotImplementedError("This is just an abstract base class")

    def build(self):
        return self

    def KL(self):
        return 0.

    def __str__(self):
        return self.__class__.__name__ if self.name is None else self.name

    def __repr__(self):
        reprstr = "{}(input_dim={}, output_dim={}, name={})" \
            .format(self.__class__.__name__, self.input_dim, self.output_dim,
                    self.name)
        return reprstr


class Dense(Layer):

    def __init__(self, input_dim, output_dim, reg=1., learn_prior=True,
                 name=None):

        super().__init__(input_dim, output_dim, name)
        self.reg = reg
        self.learn_prior = learn_prior

    def build(self):

        Wdim = (self.input_dim, self.output_dim)
        bdim = (self.output_dim,)

        # Layer priors
        self.pW = _Weights(
            mu=tf.zeros(Wdim),
            var=pos(tf.Variable(self.reg)) * tf.ones(Wdim)
            if self.learn_prior else self.reg * tf.ones(Wdim)
        )
        self.pb = _Weights(
            mu=tf.zeros(bdim),
            var=pos(tf.Variable(self.reg)) * tf.ones(bdim)
            if self.learn_prior else self.reg * tf.ones(bdim)
        )

        # Layer Posteriors
        self.qW = _Weights(
            mu=tf.Variable(self.reg * tf.random_normal(Wdim)),
            var=pos(tf.Variable(self.reg * tf.random_normal(Wdim)))
        )
        self.qb = _Weights(
            mu=tf.Variable(self.reg * tf.random_normal(bdim)),
            var=pos(tf.Variable(self.reg * tf.random_normal(bdim)))
        )

        return self

    def __call__(self, X):
        XWb = tf.matmul(X, self.qW.sample()) + self.qb.sample()
        return XWb

    def KL(self):
        KL = tf.reduce_sum(self.qW.KL(self.pW)) + \
            tf.reduce_sum(self.qb.KL(self.pb))
        return KL


class Activation(Layer):

    def __init__(self, func=lambda X: X, input_dim=None, output_dim=None,
                 name=None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        self.func = func

    def __call__(self, X):
        return self.func(X)


#
# Private Module Classes
#

class _Weights(Normal):

    # def __init__(self, mu=0., var=1.):
    #     self.mu = mu
    #     self.var = var
    #     self.sigma = tf.sqrt(var)

    def sample(self):
        # Reparameterisation trick
        e = tf.random_normal(self.mu.get_shape())
        x = self.mu + e * self.sigma
        return x

    def KL(self, p):
        KL = 0.5 * (tf.log(p.var) - tf.log(self.var) + self.var / p.var - 1. +
                    (self.mu - p.mu)**2 / p.var)
        return KL
