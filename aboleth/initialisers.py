"""Functions for initialising weights or distributions."""
import numpy as np
import tensorflow as tf

from aboleth.random import seedgen


INIT_DICT = {"glorot": tf.glorot_uniform_initializer(seed=next(seedgen)),
             "glorot_trunc": tf.glorot_normal_initializer(seed=next(seedgen))}


def initialise_weights(shape, init_fn):
    """
    Draw random initial weights using the specified function or method

    Parameters
    ----------
    shape : tuple, list
        The shape of the weight matrix to initialise. Typically this is
        3D ie of size (samples, input_size, output_size).
    init_fn : str, callable
        The function to use to initialise the weights. The default is
        'glorot_trunc', the truncated normal glorot function. If supplied,
        the callable takes a shape (input_dim, output_dim) as an argument
        and returns the weight matrix.

    """
    if isinstance(init_fn, str):
        fn = INIT_DICT[init_fn]
    else:
        fn = init_fn
    W = fn(shape)
    return W
