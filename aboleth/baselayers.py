"""Base Classes for Layers."""

from functools import reduce

import tensorflow as tf


class Layer:
    """Layer base class.

    This is an identity layer, and is primarily meant to be subclassed to
    construct more intersting layers.
    """

    def __call__(self, X):
        r"""Construct the subgraph for this layer.

        Parameters
        ----------
        X : Tensor
            the input to this layer

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : float, Tensor
            the regularizer/Kullback Leibler 'cost' of the parameters in this
            layer.

        """
        Net, KL = self._build(X)
        return Net, KL

    def _build(self, X):
        """Implement graph construction. Should be over-ridden."""
        return X, 0.0

    def __rshift__(self, other):
        """Implement layer composition, other(self(x))."""
        return LayerComposite(self, other)


class MultiLayer:
    """Base class for layers that take multiple inputs as kwargs.

    This is an Abstract class as there is no canonical identity for this
    layer (because it must do some kind of reduction).

    """

    def __call__(self, **kwargs):
        r"""Construct the subgraph for this layer.

        Parameters
        ----------
        **kwargs :
            the inputs to this layer (Tensors)

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : float, Tensor
            the regularizer/Kullback Leibler 'cost' of the parameters in this
            layer.

        """
        Net, KL = self._build(**kwargs)
        return Net, KL

    def _build(self, **kwargs):
        """Implement graph construction. Should be over-ridden."""
        raise NotImplementedError("Base class for MultiLayers only!")

    def __rshift__(self, other):
        """Implement multi-layer composition, other(self(x))."""
        return MultiLayerComposite(self, other)


class MultiLayerComposite(MultiLayer):
    """Composition of MultiLayers.

    Parameters
    ----------
    *layers :
        the layers to compose. First layer must be of type Multilayer,
        subsequent layers must be of type Layer.

    """

    def __init__(self, *layers):
        """Construct a new object."""
        self.stack = reduce(_stack2, layers)  # foldl

    def _build(self, **kwargs):
        """Stack the layers using function composition."""
        Net, KL = self.stack(**kwargs)
        return Net, KL


class LayerComposite(Layer):
    """Composition of Layers.

    Parameters
    ----------
    *layers :
        the layers to compose. All must be of type Layer.

    """

    def __init__(self, *layers):
        """Construct a new object."""
        self.stack = reduce(_stack2, layers)  # foldl

    def _build(self, X):
        """Stack the layers using function composition."""
        Net, KL = self.stack(X)
        return Net, KL


def stack(l, *layers):
    """Stack multiple Layers.

    This is a convenience function that acts as an alternative to the
    rshift operator implemented for Layers and Multilayers. It is syntatically
    more compact for stacking large numbers of layers or lists of layers.

    The type of stacking (Layer or Multilayer) is dispatched on the first
    argument.

    Parameters
    ----------
    l : Layer or MultiLayer
        The first layer to stack. The type of this layer determines the type
        of the output; MultiLayerComposite or LayerComposite.

    *layers :
        list of additional layers to stack. Must all be of type Layer,
        because function composition only works with the first function having
        multiple arguments.

    Returns
    -------
    result : MultiLayerComposite or LayerComposite
        A single layer that is the composition of the input layers.

    """
    if isinstance(l, MultiLayer):
        result = MultiLayerComposite(l, *layers)
    else:
        result = LayerComposite(l, *layers)
    return result


def _stack2(layer1, layer2):
    """Stack 2 functions, by composing w.r.t tensor, adding w.r.t losses."""
    def stackfunc(*args, **kwargs):
        result1, loss1 = layer1(*args, **kwargs)
        result, loss2 = layer2(result1)
        loss = tf.add(loss1, loss2)
        return result, loss
    return stackfunc
