"""Layers that impute missing data."""
import tensorflow as tf

from aboleth.baselayers import MultiLayer
from aboleth.random import seedgen
from aboleth.util import pos


class MaskInputLayer(MultiLayer):
    r"""Create an input layer for a binary mask tensor.

    This layer defines input kwargs so that a user may easily provide the right
    binary mask inputs to a complex set of layers to enable imputation.

    Parameters
    ----------
    name : string
        The name of the input. Used as the agument for input into the net.
    """

    def __init__(self, name):
        """Construct an instance of MaskInputLayer."""
        self.name = name

    def _build(self, **kwargs):
        """Build the mask input layer."""
        Mask = kwargs[self.name]
        assert tf.as_dtype(Mask.dtype).is_bool
        M = tf.convert_to_tensor(Mask)
        return M, 0.0


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

        self._check_rank_type(X_ND, M)
        self._set_mask(M, X_ND.dtype)

        # Extra build/initialisation here
        self._initialise_variables(X_ND)

        # Map over the samples layer
        Net = tf.map_fn(self._impute2D, X_ND)

        loss = tf.add(loss1, loss2)
        return Net, loss

    def _impute2D(self, X_2D):
        r"""Impute a rank 2 tensor.

        This function is mapped over the rank 3 data tensors, additionally it
        has access to two properties:
        - ``self.missing_ind`` a row and column index of the missing data
        - ``self.real_val_mask`` a mask of the non missing values

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

    def _check_rank_type(self, X, M):
        """Check the rank of the input tensors."""
        data_rank = len(X.shape)
        mask_rank = len(M.shape)
        assert data_rank == 3
        assert mask_rank == 2
        assert tf.as_dtype(M.dtype).is_bool

    def _set_mask(self, M, dtype):
        """Create Tensor Masks."""
        # Identify indices of the missing datapoints
        self.missing_ind = tf.where(M)
        self.real_val_mask = tf.cast(tf.logical_not(M), dtype)

    def _initialise_variables(self, X):
        """Optional extra build stage."""
        pass


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


class FixedNormalImpute(ImputeOp):
    r"""Impute the missing values using marginal Gaussians over each column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer. Creates a layer that returns a tensor in which the masked
    values have been imputed as random draws from the marginal Gaussians.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be of form ``f(**kwargs)``.
    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be of form ``f(**kwargs)``.
    mu_array : array-like
        A list of the global mean values of each dat column
    std_array : array-like
        A list of the global standard deviation of each data column

    """

    def __init__(self, datalayer, masklayer, mu_array, std_array):
        """Construct and instance of a RandomGaussImpute operation."""
        super().__init__(datalayer, masklayer)
        self.normal_array = [
            tf.distributions.Normal(m, s) for m, s in zip(mu_array, std_array)
        ]

    def _impute2D(self, X_2D):
        r"""Randomly impute a rank 2 tensor."""
        # Fill zeros in for missing data initially
        data_zeroed_missing_tf = X_2D * self.real_val_mask

        # Divide column totals by the number of non-nan values
        col_draws = [n.sample(seed=next(seedgen)) for n in self.normal_array]
        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(col_draws, self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing_tf), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing_tf + missing_imputed

        return X_with_impute


class LearnedScalarImpute(ImputeOp):
    r"""Impute the missing values using learnt scalar for each column.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer. Creates a layer that returns a tensor in which the masked
    values have been imputed with a learned scalar value per colum.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be an InputLayer.
    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be an InputLayer.

    """

    def __init__(self, datalayer, masklayer):
        r"""Construct and instance of a VarScalarImpute operation."""
        super().__init__(datalayer, masklayer)

    def _initialise_variables(self, X):
        """Initialise the impute variables."""
        datadim = int(X.shape[2])
        self.impute_scalars = tf.Variable(
            tf.random_normal(shape=(1, datadim), seed=next(seedgen)),
            name="impute_scalars"
        )

    def _impute2D(self, X_2D):
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
        data_zeroed_missing = X_2D * self.real_val_mask

        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(self.impute_scalars[0, :],
                                 self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing + missing_imputed

        return X_with_impute


class LearnedNormalImpute(ImputeOp):
    r"""Impute the missing values with draws from learned normal distributions.

    Takes two layers, one the returns a data tensor and the other returns a
    mask layer. This creates a layer that will learn marginal Gaussian
    parameters per column, and infill missing values using draws from these
    Gaussians.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be an InputLayer.
    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be an InputLayer.
    """

    def __init__(self, datalayer, masklayer):
        r"""Construct and instance of a VarScalarImpute operation."""
        super().__init__(datalayer, masklayer)

    def _initialise_variables(self, X):
        """Initialise the impute variables."""
        datadim = int(X.shape[2])
        impute_means = tf.Variable(
            tf.random_normal(shape=(1, datadim), seed=next(seedgen)),
            name="impute_scalars"
        )
        impute_stddev = tf.Variable(
            tf.random_gamma(alpha=1., shape=(1, datadim), seed=next(seedgen)),
            name="impute_scalars"
        )
        self.normal = tf.distributions.Normal(
            impute_means,
            tf.sqrt(pos(impute_stddev))
        )

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
        data_zeroed_missing = X_2D * self.real_val_mask

        # Divide column totals by the number of non-nan values
        col_draws = tf.transpose(self.normal.sample(seed=next(seedgen)))
        # Make an vector of the impute values for each missing point
        imputed_vals = tf.gather(col_draws, self.missing_ind[:, 1])[:, 0]

        # Fill the imputed values into the data tensor of zeros
        shape = tf.cast(tf.shape(data_zeroed_missing), dtype=tf.int64)
        missing_imputed = tf.scatter_nd(self.missing_ind, imputed_vals, shape)

        X_with_impute = data_zeroed_missing + missing_imputed

        return X_with_impute
