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
    slices : [slice]
        The slices into X to give to each layer, this has to be the same length
        as layers. If this is None, it will give *columns* of X to each layer,
        the number of columns is determined by the number of layers.

    """

    def __init__(self, *layers, slices=None):
        """Initialise with the individual layers."""
        self.layers = layers
        self.slices = slices if slices is not None \
            else [slice(i, i + 1) for i in range(len(layers))]

        if len(self.layers) != len(self.slices):
            raise ValueError("This requires one slice per layer.")

    def _build(self, X):
        """Build slice concatenation operation. ``X`` is a rank 3 Tensor."""
        rank = len(X.shape)
        assert rank == 3

        tensors, losses = zip(*[l(X[..., s])
                                for s, l in zip(self.slices, self.layers)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss
