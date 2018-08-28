"""Package helper utilities."""
import tensorflow as tf
import numpy as np

from aboleth.random import endless_permutations


def pos_variable(initial_value, name=None, **kwargs):
    """Make a tf.Variable that will remain positive.

    Parameters
    ----------
    initial_value : float, np.array, tf.Tensor
        the initial value of the Variable.
    name : string
        the name to give the returned tensor.
    kwargs : dict
        optional arguments to give the created ``tf.Variable``.

    Returns
    -------
    var : tf.Tensor
        a tf.Variable within a Tensor that will remain positive through
        training.

    """
    var0 = tf.Variable(_inverse_softplus(initial_value), **kwargs)
    var = tf.nn.softplus(var0, name=name)
    return var


def batch(feed_dict, batch_size, n_iter=10000, N_=None):
    r"""Create random batches for Stochastic gradients.

    Feed dict data generator for SGD that will yeild random batches for a
    a defined number of iterations, which can be infinite. This generator makes
    consecutive passes through the data, drawing without replacement on each
    pass.

    Parameters
    ----------
    feed_dict : dict of ndarrays
        The data with ``{tf.placeholder: data}`` entries. This assumes all
        items have the *same* length!
    batch_size : int
        number of data points in each batch.
    n_iter : int, optional
        The number of iterations
    N_ : tf.placeholder (int), optional
        Place holder for the size of the dataset. This will be fed to an
        algorithm.

    Yields
    ------
    dict:
        with each element an array length ``batch_size``, i.e. a subset of
        data, and an element for ``N_``. Use this as your feed-dict when
        evaluating a loss, training, etc.

    """
    N = __data_len(feed_dict)
    perms = endless_permutations(N)

    i = 0
    while i < n_iter:
        i += 1
        ind = np.array([next(perms) for _ in range(batch_size)])
        batch_dict = {k: v[ind] for k, v in feed_dict.items()}
        if N_ is not None:
            batch_dict[N_] = N
        yield batch_dict


def batch_prediction(feed_dict, batch_size):
    r"""Split the data in a feed_dict into contiguous batches for prediction.

    Parameters
    ----------
    feed_dict : dict of ndarrays
        The data with ``{tf.placeholder: data}`` entries. This assumes all
        items have the *same* length!
    batch_size : int
        number of data points in each batch.

    Yields
    ------
    ndarray :
        an array of shape approximately (``batch_size``,) of indices into the
        original data for the current batch
    dict :
        with each element an array length ``batch_size``, i.e. a subset of
        data. Use this as your feed-dict when evaluating a model, prediction,
        etc.

    Note
    ----
    The exact size of the batch may not be ``batch_size``, but the nearest size
    that splits the size of the data most evenly.

    """
    N = __data_len(feed_dict)
    n_batches = max(np.round(N / batch_size), 1)
    batch_inds = np.array_split(np.arange(N, dtype=int), n_batches)

    for ind in batch_inds:
        batch_dict = {k: v[ind] for k, v in feed_dict.items()}
        yield ind, batch_dict


def summary_histogram(values):
    """Add a summary histogram to TensorBoard.

    This will add a summary histogram with name ``variable.name``.

    Parameters
    ----------
    values : tf.Variable, tf.Tensor
        the Tensor to add to the summaries.

    """
    name = values.name.replace(':', '_')
    tf.summary.histogram(name=name, values=values)


def summary_scalar(values):
    """Add a summary scalar to TensorBoard.

    This will add a summary scalar with name ``variable.name``.

    Parameters
    ----------
    values : tf.Variable, tf.Tensor
        the Tensor to add to the summaries.

    """
    name = values.name.replace(':', '_')
    tf.summary.scalar(name=name, tensor=values)


def __data_len(feed_dict):
    N = feed_dict[list(feed_dict.keys())[0]].shape[0]
    return N


def _inverse_softplus(x):
    r"""Inverse softplus function for initialising values.

    This is useful for when we want to constrain a value to be positive using a
    softplus function, but we wish to specify an exact value for
    initialisation.

    Examples
    --------
    Say we wish a variable to be positive, and have an initial value of 1.,
    >>> var = tf.nn.softplus(tf.Variable(1.0))
    >>> with tf.Session() as sess:
    ...     sess.run(tf.global_variables_initializer())
    ...     print(var.eval())
    1.3132616

    If we use this function,
    >>> var = tf.nn.softplus(tf.Variable(_inverse_softplus(1.0)))
    >>> with tf.Session() as sess:
    ...     sess.run(tf.global_variables_initializer())
    ...     print(np.allclose(var.eval(), 1.0))
    True

    """
    x_prime = tf.log(tf.exp(x) - 1.)
    return x_prime
