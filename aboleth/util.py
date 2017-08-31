"""Package helper utilities."""
import tensorflow as tf
import numpy as np

from aboleth.random import endless_permutations


def pos(X, minval=1e-15):
    r"""Constrain a ``tf.Variable`` to be positive only.

    At the moment this is implemented as:

        :math:`\max(|\mathbf{X}|, \text{minval})`

    This is fast and does not result in vanishing gradients, but will lead to
    non-smooth gradients and more local minima. In practice we haven't noticed
    this being a problem.

    Parameters
    ----------
    X : Tensor
        any Tensor in which all elements will be made positive.
    minval : float
        the minimum "positive" value the resulting tensor will have.

    Returns
    -------
    X : Tensor
        a tensor the same shape as the input ``X`` but positively constrained.

    Examples
    --------
    >>> X = tf.constant(np.array([1.0, -1.0, 0.0]))
    >>> Xp = pos(X)
    >>> with tf.Session():
    ...     xp = Xp.eval()
    >>> xp
    array([  1.00000000e+00,   1.00000000e+00,   1.00000000e-15])

    """
    # Other alternatives could be:
    # Xp = tf.exp(X)  # Medium speed, but gradients tend to explode
    # Xp = tf.nn.softplus(X)  # Slow but well behaved!
    Xp = tf.maximum(tf.abs(X), minval)  # Faster, but more local optima
    return Xp


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


def predict_samples(predictor, feed_dict, n_groups=1, session=None):
    r"""Help to get samples from a predictor.

    Parameters
    ----------
    predictor : Tensor
        a tensor that outputs a shape (n_samples, N, tasks) where
        ``n_samples`` are the random samples from the predictor (e.g. the
        output of ``Net``), ``N`` is the size of the query dataset, and
        ``tasks`` the number of prediction tasks.
    feed_dict : dict
        The data with ``{tf.placeholder: data}`` entries.
    n_groups : int
        The number of times to evaluate the ``predictor`` and concatenate the
        samples.
    session : Session
        the session to be used to evaluate the predictor.

    Returns
    -------
    pred : ndarray
        prediction samples of shape (n_samples * n_groups, N, tasks).

    Note
    ----
    This has to be called in an *active* tensorflow session!

    """
    pred = [predictor.eval(feed_dict=feed_dict, session=session)
            for _ in range(n_groups)]
    pred = np.concatenate(pred, axis=0)
    return pred


def predict_expected(predictor, feed_dict, n_groups=1, session=None):
    r"""Help to get the expected value from a predictor.

    Parameters
    ----------
    predictor : Tensor
        a tensor that outputs a shape (n_samples, N, tasks) where
        ``n_samples`` are the random samples from the predictor (e.g. the
        output of ``Net``), ``N`` is the size of the query dataset, and
        ``tasks`` the number of prediction tasks.
    feed_dict : dict
        The data with ``{tf.placeholder: data}`` entries.
    n_groups : int
        The number of times to evaluate the ``predictor`` and concatenate the
        samples.
    session : Session
        the session to be used to evaluate the predictor.

    Returns
    -------
    pred : ndarray
        expected value of the prediction with shape (N, tasks). ``n_samples *
        n_groups`` samples go into evaluating this expectation.

    Note
    ----
    This has to be called in an *active* tensorflow session!

    """
    pred = 0
    for _ in range(n_groups):
        pred += predictor.eval(feed_dict=feed_dict, session=session) / n_groups

    pred = pred.mean(axis=0)
    return pred


def __data_len(feed_dict):
    N = feed_dict[list(feed_dict.keys())[0]].shape[0]
    return N
