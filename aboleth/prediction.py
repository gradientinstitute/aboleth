"""Convenience functions for building prediction graphs."""
import tensorflow as tf
from tensorflow.contrib.distributions import percentile


def sample_mean(predictor, name=None):
    """
    Get the mean of the samples of a predictor.

    Parameter
    ---------
    predictor : Tensor
        A tensor of samples, where the first dimension indexes the samples.
    name : str
        name to give this operation

    Returns
    -------
    expec : Tensor
        A tensor that contains the mean of the predicted samples.

    """
    expec = tf.reduce_mean(predictor, axis=0, name=name)
    return expec


def sample_percentiles(predictor, per=[10, 90], interpolation='nearest',
                       name=None):
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
    name : str
        name to give this operation

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

    percen = tf.stack(pers, name=name)
    return percen


def sample_model(graph=None, sess=None, feed_dict=None):
    """
    Sample the model parameters.

    This function returns a feed_dict containing values for the sample tensors
    in the model. It means that multiple calls to eval() will not change the
    model parameters as long as the output of this function is used as a
    feed_dict.

    Parameters
    ----------
    graph : tf.Graph
        The current graph. If none provided use the default.
    sess : tf.Session
        The session to use for evaluating the tensors. If none provided
        will use the default.
    feed_dict : dict
        An optional feed_dict to pass the session.

    Returns
    -------
    collection : dict
        A feed_dict to use when evaluating the model.

    """
    if not graph:
        graph = tf.get_default_graph()
    if not sess:
        sess = tf.get_default_session()

    params = graph.get_collection('SampleTensors')
    param_values = sess.run(params, feed_dict=feed_dict)
    sample_feed_dict = dict(zip(params, param_values))
    return sample_feed_dict
