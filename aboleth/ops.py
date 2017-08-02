"""Operations for composing layers."""

from functools import reduce

import tensorflow as tf


#
# Input Layer Operation Classes
#

class InputLayerOp:
    """Base class for ``InputLayer`` operations.

    ``InputLayerOps`` take in ``InputLayer``s as arguments (in the constructor)
    and return Tensor tuples like layers, e.g. (Net, KL).

    The callable layers must all take (only) **kwargs. They can pull from this
    dictionary as required. It is intended to work on ``InputLayer``s, or
    functions composed with ``InputLayer``s.

    Parameters
    ----------
    layers : [callable]
        The ``InputLayer``s to compose. The first layer can have multiple
        inputs.

    """

    def __init__(self, *layers):
        """Construct and instance of an InputLayerOp."""
        # NOTE: we could check for layers of type InputLayer here (mypy)
        self.layers = layers

    def __call__(self, **kwargs):
        """Build the operation, kwargs are fed to this operation's layers.

        See: operation.

        """
        # NOTE: we could check for expected kwargs here according to
        #   self.layers... or leave it to "duck typing"
        op = self.operation(**kwargs)
        return op

    def operation(self, **kwargs):
        """Build the operation, kwargs are fed to this operation's layers.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        raise NotImplementedError("Abstract base class for InputLayerOps")


class Stack(InputLayerOp):
    """Stack multiple layers together (function composition).

    When called stack(f, g) stack returns h(.) = g(f(.)), ie the functions
    will be evaluated on the input in order from left to right in the call.
    Parameters
    ----------
    layers : [callable]
        The layers to compose. The first layer can have multiple inputs.

    """

    def __init__(self, *layers):
        """Construct and instance of the Stack layer operation."""
        super().__init__(*layers)
        self.compose = reduce(_stack2, self.layers)

    def operation(self, **kwargs):
        """Build the operation, kwargs are fed to this operation's layers.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        stack = self.compose(**kwargs)
        return stack


class Concat(InputLayerOp):
    """Join multiple layers by concatenating their outputs.

    The callable layers must all take (only) **kwargs. They can pull from this
    dictionary as required. It is intended to work on ``InputLayer``s, or
    functions composed with ``InputLayer``s.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Must be f(**kwargs).
    """

    def operation(self, **kwargs):
        """Build the operation, kwargs are fed to this operation's layers.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss


class Add(InputLayerOp):
    """Join multiple layers by adding their outputs.

    Similar to concatenate, the layers must take (only) **kwargs. The outputs
    of the functions will be added element-wise. It is intended to work on
    ``InputLayer``s, or functions composed with ``InputLayer``s.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Must be of form f(**kwargs).

    """

    def operation(self, **kwargs):
        """Build the operation, kwargs are fed to this operation's layers.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.add_n(tensors)
        loss = tf.add_n(losses)
        return result, loss


def MeanImpute(InputLayerOp):
    """Impute the missing values using the stochastic mean of their column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer.  Returns a layer that returns a tensor in which the masked
    values have been imputed as the column means calculated from the batch.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form f(**kwargs).

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form f(**kwargs).
    """

    def __init__(self, datalayer, masklayer):
        super().__init__(datalayer, masklayer)
        self.datalayer = datalayer
        self.masklayer = masklayer

    def operation(self, **kwargs):
        X_ND, loss1 = self.datalayer(**kwargs)
        M, loss2 = self.masklayer(**kwargs)

        # Identify indices of the missing datapoints
        missing_ind = tf.where(M)
        real_val_mask = tf.cast(tf.logical_not(M), tf.float32)

        def mean_impute_2D(X_2D):
            # Fill zeros in for missing data initially
            data_zeroed_missing_tf = X_2D * real_val_mask

            # Sum the real values in each column
            col_tot = tf.reduce_sum(data_zeroed_missing_tf, 0)

            # Divide column totals by the number of non-nan values
            num_values_col = tf.reduce_sum(real_val_mask, 0)
            num_values_col = tf.maximum(num_values_col,
                                        tf.ones(tf.shape(num_values_col)))
            col_nan_means = tf.div(col_tot, num_values_col)

            # Make an vector of the impute values for each missing point
            imputed_vals = tf.gather(col_nan_means, missing_ind[:, 1])

            # Fill the imputed values into the data tensor of zeros
            shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
            missing_imputed = tf.scatter_nd(missing_ind, imputed_vals,
                                            shape)

            X_with_impute = data_zeroed_missing_tf + missing_imputed

            return X_with_impute

        Net = tf.map_fn(mean_impute_2D, X_ND)

        loss = tf.add(loss1, loss2)
        return Net, loss


#
# Layer Operation Classes
#

class LayerOp:
    """Base class for ``Layer`` operations.

    ``LayerOps`` take in ``Layer``s as arguments (in the constructor)
    and return Tensor tuples like layers, e.g. (Net, KL).

    The callable layers must all take (only) Tensors (X). It is intended to
    work on ``Layer``s, or functions composed with ``Layer``s.

    Parameters
    ----------
    layers : [callable]
        The ``Layer``s to compose. The first layer can have multiple inputs.

    """

    def __init__(self, *layers):
        """Construct and instance of an InputLayerOp."""
        # NOTE: we could check for layers of type Layer here (mypy)
        self.layers = layers

    def __call__(self, X):
        """Build the operation, X is fed to this operation's layers.

        See: operation.

        """
        # NOTE: we could check for expected X here according to
        #   self.layers... or leave it to "duck typing"
        op = self.operation(X)
        return op

    def operation(self, X):
        """Build the operation, X is fed to this operation's layers.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        raise NotImplementedError("Abstract base class for LayerOps")


class SliceCat(LayerOp):
    """Concatenate multiple layers with sliced inputs.

    Each layer will recieve a slice along the last axis of the input to the
    new function. In other words, slicecat(l1, l2)(X) will call
    l1(X[..., 0]) and l2(X[..., 1]) then concatenate their outputs into a
    single tensor. This is mostly useful for simplifying embedding multiple
    categorical inputs that are stored columnwise in the same 2D tensor.

    This function assumes the tensor being provided is 3D.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Supports layers with a single argument only.
    """

    def operation(self, X):
        """Build the operation, X is fed to this operation's layers.

        This is layer function of the concatenated input layers, that will take
        an single 3D tensor as input of size (-1, -1, n) where n is the number
        of input layers, and output a tensor of size (-1,-1, n) where each of
        the kth < n  slice has been processed by the corresponding  kth layer
        provided in input.

        Returns
        -------
        Net : Tensor
            the output of this layer
        KL : {float, Tensor}
            the regularizer/Kullback Liebler 'cost' of the parameters in this
            layer.

        """
        tensors, losses = zip(*[l(X[..., i:i + 1])
                                for i, l in enumerate(self.layers)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss


#
# Private utility functions
#
def _stack2(layer1, layer2):
    """Stack 2 functions, by composing w.r.t tensor, adding w.r.t losses."""
    def stackfunc(**kwargs):
        result1, loss1 = layer1(**kwargs)
        result, loss2 = layer2(result1)
        loss = tf.add(loss1, loss2)
        return result, loss
    return stackfunc
