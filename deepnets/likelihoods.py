import numpy as np
import tensorflow as tf

from utils import pos

# TODO: Make these functions, then kwargs them into the loss function


class Normal():

    def __init__(self, mu=0., var=1.):
        self.mu = mu
        self.var = var
        self.sigma = tf.sqrt(var)

    def log_prob(self, x, mu=None, var=None):
        mu = self.mu if mu is None else mu
        var = self.var if var is None else var
        ll = -0.5 * (tf.log(2. * var * np.pi) + (x - mu)**2 / var)
        return ll


class Bernoulli():

    def __init__(self, p=.5):
        self.p = p

    def log_prob(self, x, p=None):
        p = self.p if p is None else p
        ll = x * tf.log(pos(p)) + (1 - x) * tf.log(pos(1 - p))
        return ll
