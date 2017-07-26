"""Operations for composing layers."""

import tensorflow as tf
from functools import reduce


def stack(*layers):
    """Stack multiple layers together (function composition).

    When called stack(f, g) stack returns h(.) = g(f(.)), ie the functions
    will be evaluated on the input in order from left to right in the call.

    Parameters
    ----------
    layers : [callable]
        The layers to compose. The first layer can have multiple inputs.

    Returns
    -------
    stackfunc : callable
        A layer function of the stacked input layers.

    """
    stackfunc = reduce(_stack2, layers)
    return stackfunc


def concat(*layers):
    """Concatenate multiple layers by concatenating their outputs.

    Note that it is expected that the functions will take different arguments.
    The output function will take all arguments in the order they were
    provided. That is, concat(f(X), g(Y)) -> h(X, Y).

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Supports layers with multiple arguments.

    Returns
    -------
    concatfunc : callable
        A layer function of the concatenated input layers, that will have as
        many inputs as the set of original functions.

    """
    def concatfunc(*Xl):
        tensors, losses = zip(*[l(X) for l, X in zip(layers, Xl)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss
    return concatfunc


def slicecat(*layers):
    """Concatenate multiple layers with sliced inputs.

    Each layer will recieve a slice along the last axis of the input to the
    new function. In other words, slicecat(l1, l2)(X) will call
    l1(X[..., 0]) and l2(X[..., 1]) then concatenate their outputs into a
    single tensor. This is mostly useful for simplifying embedding multiple
    categorical inputs that are stored columnwise in the same 2D tensor.

    Note that this function assumes the tensor being provided is 3D.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Supports layers with a single argument only.

    Returns
    -------
    slicefunc : callable
        A layer function of the concatenated input layers, that will take
        an single 3D tensor as input of size (-1, -1, n) where n is the number
        of input layers, and output a tensor of size (-1,-1,n) where each of
        the kth < n  slice has been processed by the corresponding  kth layer
        provided in input.

    """
    def slicefunc(X):
        tensors, losses = zip(*[l(X[:, :, i:i + 1])
                                for i, l in enumerate(layers)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss
    return slicefunc


def add(*layers):
    """Concatenate multiple layers by adding their outputs.

    Note that it is expected that the functions will take different arguments.
    The output function will take all arguments in the order they were
    provided. That is, concat(f(X), g(Y)) -> h(X, Y). The outputs will be added
    along the last dimension.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Supports layers with multiple arguments.

    Returns
    -------
    concatfunc : callable
        A layer function of the concatenated input layers, that will have as
        many inputs as the set of original functions.

    """
    def addfunc(*Xl):
        tensors, losses = zip(*[l(X) for l, X in zip(layers, Xl)])
        result = tf.add_n(tensors)
        loss = tf.add_n(losses)
        return result, loss
    return addfunc


# Private utility functions


def _stack2(layer1, layer2):
    """Stack 2 functions, by composing w.r.t tensor, adding w.r.t losses."""
    def stackfunc(*Xl):
        result1, loss1 = layer1(*Xl)
        result, loss2 = layer2(result1)
        loss = tf.add(loss1, loss2)
        return result, loss
    return stackfunc
