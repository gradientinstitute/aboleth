"""Convenience functions for building prediction graphs."""
import tensorflow as tf
from tensorflow.contrib.distributions import percentile


def sample_mean(predictor):
    """
    Get the mean of the samples of a predictor.

    Parameter
    ---------
    predictor : Tensor
        A tensor of samples, where the first dimension indexes the samples.

    Returns
    -------
    expec : Tensor
        A tensor that contains the mean of the predicted samples.

    """
    expec = tf.reduce_mean(predictor, axis=0)
    return expec


def sample_percentiles(predictor, per=[10, 90], interpolation='nearest'):
    """
    Get the percentiles of the samples of a predictor.

    Parameter
    ---------
    predictor : Tensor
        A tensor of samples, where the first dimension indexes the samples.
    per : list
        A list of the percentiles to calculate from the samples. These must be
        in [0, 100].
    interpolation : string
        The type of interpolation method to use, see
        tf.contrib.distributions.percentile for details.

    Returns
    -------
    percen: Tensor
        A tensor whose first dimension indexes the percentiles, computed along
        the first axis of the input.

    """
    for p in per:
        assert 0 <= p <= 100

    pers = [percentile(predictor, p, interpolation=interpolation, axis=0)
            for p in per]

    percen = tf.stack(pers)
    return percen
