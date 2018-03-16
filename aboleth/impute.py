"""Layers that impute missing data."""
import tensorflow as tf

from aboleth.baselayers import MultiLayer
from aboleth.random import seedgen
from aboleth.util import pos, summary_histogram


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
        X_imputed = None  # Your imputation implementation
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
        """Extra build stage."""
        pass


class ImputeColumnWise(ImputeOp):
    r"""Abstract class for imputing column-wise from a vector or scalar.

    This implements ``_impute2D`` and this calls the ``_impute_columns`` method
    that returns a vector or scalar to impute X column-wise (as opposed to
    element-wise). You need to supply the ``_impute_columns`` method.
    """

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
        X_2D_zero = X_2D * self.real_val_mask

        # Make a vector of the impute values for each missing point
        column_vals = self._impute_columns(X_2D_zero)
        imputed_vals = tf.gather(column_vals, self.missing_ind[:, 1])

        # Fill the imputed values into the data tensor of zeros
        X_new = tf.scatter_nd(self.missing_ind, imputed_vals,
                              shape=tf.shape(X_2D, out_type=tf.int64))
        X_imp = X_2D_zero + X_new
        return X_imp

    def _impute_columns(self, X_2D_zero):
        """Generate a vector to subtitute missing data in X from.

        Parameters
        ----------
        X_2D_zero : Tensor
            a rank 2 Tensor with missing data zero-ed

        Returns
        -------
        column_vals : float, array, Tensor
            a scalar or rank 1 Tensor to impute X column-wise with

        """
        raise NotImplementedError("Abstract base class for imputation ops!")
        column_vals = None  # Your imputation implementation
        return column_vals


class MeanImpute(ImputeColumnWise):
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

    def _impute_columns(self, X_2D_zero):
        """Generate a vector of means from X batches."""
        # Sum the real values in each column
        col_tot = tf.reduce_sum(X_2D_zero, 0)

        # Divide column totals by the number of non-nan values
        num_values_col = tf.reduce_sum(self.real_val_mask, 0)
        num_values_col = tf.maximum(num_values_col,
                                    tf.ones(tf.shape(num_values_col)))
        col_nan_means = tf.div(col_tot, num_values_col)
        return col_nan_means


class LearnedScalarImpute(ImputeColumnWise):
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

    def _initialise_variables(self, X):
        """Initialise the impute variables."""
        datadim = int(X.shape[2])
        self.impute_scalars = tf.Variable(
            tf.random_normal(shape=(datadim,), seed=next(seedgen)),
            name="impute_scalars"
        )
        summary_histogram(self.impute_scalars)

    def _impute_columns(self, X_2D_zero):
        """Return the learned scalars for imputation."""
        return self.impute_scalars


class LearnedNormalImpute(ImputeColumnWise):
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

    def _initialise_variables(self, X):
        """Initialise the impute variables."""
        datadim = int(X.shape[2])
        impute_means = tf.Variable(
            tf.random_normal(shape=(datadim,), seed=next(seedgen)),
            name="impute_means"
        )
        impute_var = tf.Variable(
            tf.random_gamma(alpha=1., shape=(datadim,), seed=next(seedgen)),
            name="impute_vars"
        )

        summary_histogram(impute_means)
        summary_histogram(impute_var)

        self.normal = tf.distributions.Normal(
            impute_means,
            tf.sqrt(pos(impute_var))
        )

    def _impute_columns(self, X_2D_zero):
        """Return random draws from an iid Normal for imputation."""
        col_draws = self.normal.sample(seed=next(seedgen))
        return col_draws


class FixedNormalImpute(LearnedNormalImpute):
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
    loc : array-like
        A list of the global mean values of each data column
    scale : array-like
        A list of the global standard deviation of each data column

    """

    def __init__(self, datalayer, masklayer, loc, scale):
        """Construct and instance of a RandomGaussImpute operation."""
        super().__init__(datalayer, masklayer)
        self.loc = loc
        self.scale = scale

    def _initialise_variables(self, X):
        self.normal = tf.distributions.Normal(self.loc, self.scale)


class ExtraCategoryImpute(ImputeColumnWise):
    r"""Impute missing values from categorical data with an extra category.

    Given categorical data, a missing mask and a number of categories for
    each feature (last dimension), this will assign missing values as
    an extra category equal to the number of categories. e.g. for 2
    categories (0 and 1) missing data will be assigned 2.

    Parameters
    ----------
    datalayer : callable
        A layer that returns a data tensor. Must be an InputLayer.
    masklayer : callable
        A layer that returns a boolean mask tensor where True values are
        masked. Must be an InputLayer.
    ncategory_list : list
        A list that provides the total number of categories for each
        feature (last dimension) of the input. Length of the list must be
        equal to the size of the last dimension of X.

    """

    def __init__(self, datalayer, masklayer, ncategory_list):
        """Initialise the object."""
        self._ncats = tf.constant(ncategory_list)
        super().__init__(datalayer, masklayer)

    def _impute_columns(self, X_2D_zero):
        """Return random draws from an iid Normal for imputation."""
        return self._ncats
