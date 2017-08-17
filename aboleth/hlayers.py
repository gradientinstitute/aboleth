"""Higher-order neural network layers (made from other layers)."""
import tensorflow as tf

from aboleth.baselayers import Layer, MultiLayer


class Concat(MultiLayer):
    r"""Concatenates the output of multiple layers.

    Parameters
    ----------
    layers : [MultiLayer]
        The layers to concatenate.

    """

    def __init__(self, *layers):
        """Initialise with the individual layers."""
        self.layers = layers

    def _build(self, **kwargs):
        """Build the concatenation."""
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss


class Sum(MultiLayer):
    r"""Sums multiple layers by adding their outputs.

    Parameters
    ----------
    layers : [MultiLayer]
        The layers to add.

    """

    def __init__(self, *layers):
        """Initialise with the individual layers."""
        self.layers = layers

    def _build(self, **kwargs):
        """Build the summation layer."""
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.add_n(tensors)
        loss = tf.add_n(losses)
        return result, loss


class PerFeature(Layer):
    r"""Concatenate multiple layers with sliced inputs.

    Each layer will recieve a slice along the last axis of the input to the new
    function. In other words, ``PerFeature(l1, l2)(X)``
    will call ``l1(X[..., 0]) and l2(X[..., 1])`` then concatenate their
    outputs into a single tensor. This is mostly useful for simplifying
    embedding multiple categorical inputs that are stored columnwise
    in the same 2D tensor.

    This function assumes the tensor being provided is 3D.

    Parameters
    ----------
    layers : [Layer]
        The layers to concatenate.

    """

    def __init__(self, *layers):
        """Initialise with the individual layers."""
        self.layers = layers

    def _build(self, X):
        """Build slice concatenation operation. ``X`` is a rank 3 Tensor."""
        rank = len(X.shape)
        assert rank == 3

        tensors, losses = zip(*[l(X[..., i:i + 1])
                                for i, l in enumerate(self.layers)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss
