import numpy as np
import tensorflow as tf

from deepnets.utils import pos

# TODO: Make these functions, then kwargs them into the loss function


def normal(variance):
    def l(x, f):
        ll = -0.5 * (tf.log(2. * variance * np.pi) + (x - f)**2 / variance)
        return ll
    return l

def bernoulli():
    def b(x, f):
        ll = x * tf.log(pos(f)) + (1 - x) * tf.log(pos(1 - f))
        return ll
    return b
