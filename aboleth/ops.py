"""Operations on neural network layers."""

from functools import reduce

import tensorflow as tf

from aboleth.distributions import Normal
from aboleth.random import seedgen


class LayerOp:
    r"""Base class for an operation on Layers that return new layers.

    The layer functions can take a Tensor or ``**kwargs``. In the latter case
    they can pull from this dictionary as required. This is intended to work on
    any layers.Layer or layers.MultiLayer type (See :ref:`layers`).

    Parameters
    ----------
    layers : [callable]
        The layers to operate on, these layers must return a return signature
        ``f(Tensor or **kwargs) -> (Tensor, Tensor or float)`` like
        layers.Layers.

    """

    def __init__(self, *layers):
        """Template constructor for LayerOps."""
        self.layers = layers

    def __call__(self, X=None, **kwargs):
        """Call the graph building method (with extra checking).

        See: _build

        """
        # Make this work for Layer and MultiLayer types
        if X is not None:
            kwargs.update({'X': X})
        Net, KL = self._build(**kwargs)
        return Net, KL

    def _build(self, **kwargs):
        """Build the LayerOp graph, this can take ``**kwargs`` or a Tensor."""
        raise NotImplementedError("Abstract base class for layer operations!")


class Stack(LayerOp):
    r"""Stack multiple layers together (function composition).

    When called ``stack(f, g)`` stack returns ``h(.) = g(f(.))``, i.e. the
    functions will be evaluated on the input in order from left to right in the
    call.

    Parameters
    ----------
    layers : [callable]
        The layers to compose.

    """

    def __init__(self, *layers):
        """Create an instance of a Stack operation."""
        super().__init__(*layers)
        self.stack = reduce(_stack2, layers)  # foldl

    def _build(self, **kwargs):
        """Build the stack, this can take ``**kwargs`` or a Tensor."""
        Net, KL = self.stack(**kwargs)
        return Net, KL


class Concat(LayerOp):
    r"""Concatenate multiple layers by concatenating their outputs.

    The functions can take a Tensor or ``**kwargs``. In the latter case they
    can pull from this dictionary as required. This is intended to work on any
    layers.Layer or layers.MultiLayer type (See :ref:`layers`).

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate.

    """

    def _build(self, **kwargs):
        """Build the concatenation, this can take ``**kwargs`` or a Tensor."""
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss


class Add(LayerOp):
    r"""Concatenate multiple layers by adding their outputs.

    The functions can take a  Tensor or ``**kwargs``. In the latter case they
    can pull from this dictionary as required. The outputs of the functions
    will be added element-wise. This is intended to work on any layers.Layer or
    layers.MultiLayer type (See :ref:`layers`).

    Parameters
    ----------
    layers : [callable]
        The layers to add.

    """

    def _build(self, **kwargs):
        """Build the add operation, this can take ``**kwargs`` or a Tensor."""
        tensors, losses = zip(*map(lambda l: l(**kwargs), self.layers))
        result = tf.add_n(tensors)
        loss = tf.add_n(losses)
        return result, loss


class SliceCat(LayerOp):
    r"""Concatenate multiple layers with sliced inputs.

    Each layer will recieve a slice along the last axis of the input to the new
    function. In other words, ``slicecat(l1, l2)(X)`` will call ``l1(X[..., 0])
    and l2(X[..., 1])`` then concatenate their outputs into a single tensor.
    This is mostly useful for simplifying embedding multiple categorical inputs
    that are stored columnwise in the same 2D tensor.

    This function assumes the tensor being provided is 3D.

    Parameters
    ----------
    layers : [callable]
        The layers to concatenate. Supports layers with a single argument only.

    """

    def _build(self, X):
        """Build slice concatenation operation. ``X`` is a rank 3 Tensor."""
        rank = len(X.shape)
        assert rank == 3

        tensors, losses = zip(*[l(X[..., i:i + 1])
                                for i, l in enumerate(self.layers)])
        result = tf.concat(tensors, axis=-1)
        loss = tf.add_n(losses)
        return result, loss


class ImputeOp(LayerOp):
    r"""Abstract Base Impute operation. These specialise LayerOps.

    They expect a data InputLayer and a mask InputLayer. They return layers in
    which the masked values have been imputed.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form ``f(**kwargs)``.

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form ``f(**kwargs)``.

    """

    def __init__(self, datalayer, masklayer):
        """Construct and instance of an ImputeOp operation."""
        self.datalayer = datalayer
        self.masklayer = masklayer

    def _build(self, **kwargs):
        """Build an impute operation graph, this needs ``**kwargs`` input."""
        X_ND, loss1 = self.datalayer(**kwargs)
        M, loss2 = self.masklayer(**kwargs)

        rank = len(X_ND.shape)
        assert rank == 3

        # Identify indices of the missing datapoints
        self.missing_ind = tf.where(M)
        self.real_val_mask = tf.cast(tf.logical_not(M), tf.float32)

        # Map over the samples layer
        Net = tf.map_fn(self._impute2D, X_ND)

        loss = tf.add(loss1, loss2)
        return Net, loss

    def _impute2D(self, X_2D):
        r"""Impute a rank 2 tensor.

        This function is mapped over the rank 3 data tensors, additionally it
        has access to two properties:
        - ``self.missing_ind`` a row and column index of the missing data
        - ``self.real_val_mask`` a tf.float32 mask of the non missing values

        Parameters
        ----------
        X_2D : Tensor
            a rank 2 Tensor with missing data

        Returns
        -------
        X_imputed : Tensor
            a rank 2 Tensor with imputed data

        """
        raise NotImplementedError("Abstract base class for imputation ops!")
        X_imputed = None  # You imputation implementation
        return X_imputed


class MeanImpute(ImputeOp):
    r"""Impute the missing values using the stochastic mean of their column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer. Returns a layer that returns a tensor in which the masked
    values have been imputed as the column means calculated from the batch.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form ``f(**kwargs)``.

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form ``f(**kwargs)``.

    """

    def _impute2D(self, X_2D):
        r"""Mean impute a rank 2 tensor.

        Parameters
        ----------
        X_2D : Tensor
            a rank 2 Tensor with missing data

        Returns
        -------
        X_imputed : Tensor
            a rank 2 Tensor with imputed data

        """
        # Fill zeros in for missing data initially
        data_zeroed_missing_tf = X_2D * self.real_val_mask

        # Sum the real values in each column
        col_tot = tf.reduce_sum(data_zeroed_missing_tf, 0)

        # Divide column totals by the number of non-nan values
        num_values_col = tf.reduce_sum(self.real_val_mask, 0)
        num_values_col = tf.maximum(num_values_col,
                                    tf.ones(tf.shape(num_values_col)))
        col_nan_means = tf.div(col_tot, num_values_col)

        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(col_nan_means, self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing_tf + missing_imputed

        return X_with_impute


class FixedNormalImpute(ImputeOp):
    """Impute the missing values using the marginal gaussians over each column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer. Returns a layer that returns a tensor in which the masked
    values have been imputed as the column means calculated from the batch.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form ``f(**kwargs)``.

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form ``f(**kwargs)``.

    mu_array : array-like
        A list of the global mean values of each dat column

    var_array : array-like
        A list of the global variance of each data column

    """

    def __init__(self, datalayer, masklayer, mu_array, var_array):
        """Construct and instance of a RandomGaussImpute operation."""
        super().__init__(datalayer, masklayer)
        self.normal_array = [Normal(m, v) for m, v in zip(mu_array, var_array)]

    def _impute2D(self, X_2D):
        r"""Randomly impute a rank 2 tensor.

        Parameters
        ----------
        X_2D : Tensor
            a rank 2 Tensor with missing data

        Returns
        -------
        X_imputed : Tensor
            a rank 2 Tensor with imputed data

        """
        # Fill zeros in for missing data initially
        data_zeroed_missing_tf = X_2D * self.real_val_mask

        # Divide column totals by the number of non-nan values
        col_draws = [n.sample() for n in self.normal_array]
        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(col_draws, self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing_tf + missing_imputed

        return X_with_impute


class VarScalarImpute(ImputeOp):
    """Impute the missing values using learnt scalar for each column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer.  Returns a layer that returns a tensor in which the masked
    values have been imputed with a learnt different scalar per colum.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form f(**kwargs).

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form f(**kwargs).

    """

    def __init__(self, datalayer, masklayer):
        r"""Construct and instance of a VarScalarImpute operation."""
        super().__init__(datalayer, masklayer)

    def _build(self, **kwargs):
        r"""Build an impute operation graph, this needs ``**kwargs`` input."""
        X_ND, loss1 = self.datalayer(**kwargs)
        M, loss2 = self.masklayer(**kwargs)

        rank = len(X_ND.shape)
        assert rank == 3

        # Identify indices of the missing datapoints
        self.missing_ind = tf.where(M)
        self.real_val_mask = tf.cast(tf.logical_not(M), tf.float32)

        # Initialise the impute variables
        datadim = int(X_ND.shape[2])
        impute_scalars = tf.Variable(tf.random_normal(shape=(1, datadim),
                                                      seed=next(seedgen)),
                                     name="impute_scalars")

        # Map over the samples layer
        Net = tf.map_fn(lambda x: self._impute2D(x, impute_scalars), X_ND)

        loss = tf.add(loss1, loss2)
        return Net, loss

    def _impute2D(self, X_2D, scalars):
        r"""Randomly impute a rank 2 tensor.

        Parameters
        ----------
        X_2D : Tensor
            a rank 2 Tensor with missing data
        scalars : Tensor 1 x D
            these values are filled into the missing elements (per column)

        Returns
        -------
        X_imputed : Tensor
            a rank 2 Tensor with imputed data

        """
        # Fill zeros in for missing data initially
        data_zeroed_missing_tf = X_2D * self.real_val_mask

        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(scalars[0, :], self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing_tf + missing_imputed

        return X_with_impute


class VarNormalImpute(ImputeOp):
    r"""Impute the missing values with draws from learnt normal distributions.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form f(**kwargs).

    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form f(**kwargs).

    """

    def __init__(self, datalayer, masklayer):
        r"""Construct and instance of a VarScalarImpute operation."""
        super().__init__(datalayer, masklayer)

    def _build(self, **kwargs):
        r"""Build an impute operation graph, this needs ``**kwargs`` input."""
        X_ND, loss1 = self.datalayer(**kwargs)
        M, loss2 = self.masklayer(**kwargs)

        rank = len(X_ND.shape)
        assert rank == 3

        # Identify indices of the missing datapoints
        self.missing_ind = tf.where(M)
        self.real_val_mask = tf.cast(tf.logical_not(M), tf.float32)

        # Initialise the impute variables
        datadim = int(X_ND.shape[2])
        impute_means = tf.Variable(tf.random_normal(shape=(1, datadim),
                                                    seed=next(seedgen)),
                                   name="impute_scalars")

        impute_variances = tf.abs(tf.Variable(tf.random_normal(
            shape=(1, datadim), seed=next(seedgen)), name="impute_scalars"))

        self.normal = Normal(impute_means, impute_variances)
        # Map over the samples layer
        Net = tf.map_fn(lambda x: self._impute2D(x), X_ND)

        loss = tf.add(loss1, loss2)
        return Net, loss

    def _impute2D(self, X_2D):
        r"""Impute a rank 2 tensor with draws from normal distributions.

        Parameters
        ----------
        X_2D : Tensor
            a rank 2 Tensor with missing data

        Returns
        -------
        X_imputed : Tensor
            a rank 2 Tensor with imputed data

        """
        # Fill zeros in for missing data initially
        data_zeroed_missing_tf = X_2D * self.real_val_mask

        # Divide column totals by the number of non-nan values
        col_draws = tf.transpose(self.normal.sample())
        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(col_draws, self.missing_ind[:, 1])[:, 0]

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing_tf + missing_imputed

        return X_with_impute


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
