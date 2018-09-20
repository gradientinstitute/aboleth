"""Functions for initialising weights or distributions."""
import numpy as np
import tensorflow as tf

from aboleth.random import seedgen
from aboleth.util import pos_variable, summary_scalar


def _glorot_std(n_in, n_out):
    """
    Compute the standard deviation for initialising weights.

    See Glorot and Bengio, AISTATS2010.
    """
    std = 1. / np.sqrt(3 * (n_in + n_out))
    return std


def _autonorm_std(n_in, n_out):
    """
    Compute the auto-normalizing NN initialisation.

    To be used with SELU nonlinearities.  See Klambaur et. al. 2017
    (https://arxiv.org/pdf/1706.02515.pdf)
    """
    std = 1. / np.sqrt(n_in)
    return std


_INIT_DICT = {"glorot": tf.glorot_uniform_initializer(seed=next(seedgen)),
              "glorot_trunc": tf.glorot_normal_initializer(seed=next(seedgen)),
              "autonorm": tf.variance_scaling_initializer(
                  scale=1.0,
                  mode="fan_in",
                  distribution="untruncated_normal",
                  seed=next(seedgen))}

_PRIOR_DICT = {"glorot": _glorot_std,
               "autonorm": _autonorm_std}


def initialise_weights(shape, init_fn):
    """
    Draw random initial weights using the specified function or method.

    Parameters
    ----------
    shape : tuple, list
        The shape of the weight matrix ``W``. This uses the same convention as
        tnesorflow for weight shapes (see their initializers in
        tensorflow.python.ops.init_ops).
    init_fn : str, callable
        The function to use to initialise the weights. The default is
        'glorot_trunc', the truncated normal glorot function. If supplied,
        the callable takes a shape (input_dim, output_dim) as an argument
        and returns the weight matrix.

    """
    if isinstance(init_fn, str):
        fn = _INIT_DICT[init_fn]
    else:
        fn = init_fn
    W = fn(shape)
    return W


def initialise_stds(n_in, n_out, init_val, learn_prior, suffix):
    """
    Initialise the prior standard devation and initial poststerior.

    Parameters
    ----------
    n_in : int
        The total number of input units in the layer.
    n_out : int
        The total number of output units in the layer.
    init_val : str, float
        If a string, must be one of "glorot" or "autonorm", which will use
        these methods to initialise a value. Otherwise, will use the provided
        float to initialise.
    learn_prior : bool
        Whether to learn the prior or not. If true, will make the prior
        a variable.
    suffix : str
        A string used to name the variable so Tensorboard can track it.

    Returns
    -------
    std : tf.Variable, np.array
        The standard deviation value/variable
    std0 :
        The initial value of the standard deviation

    """
    if isinstance(init_val, str):
        fn = _PRIOR_DICT[init_val]
        std0 = fn(n_in, n_out)
    else:
        std0 = init_val
    std0 = np.array(std0).astype(np.float32)

    if learn_prior:
        std = pos_variable(std0, name="prior_std_{}".format(suffix))
        summary_scalar(std)
    else:
        std = std0
    return std, std0
