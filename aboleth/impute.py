"""Layers that impute missing data."""
import tensorflow as tf
from aboleth.baselayers import MultiLayer
from aboleth.distributions import Normal


class ImputeOp(MultiLayer):
    r"""Abstract Base Impute operation. These specialise MultiLayers.

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
        """Build an impute operation graph."""
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
        r"""Mean impute a rank 2 tensor."""
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


class RandomGaussImpute(ImputeOp):
    r"""Impute the missing values using marginal gaussians over each column.

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
        r"""Randomly impute a rank 2 tensor."""
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
