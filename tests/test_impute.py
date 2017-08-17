"""Tests for imputation layers."""

import numpy as np
import tensorflow as tf

import aboleth as ab


def test_mean_impute(make_missing_data):
    """Test the impute_mean."""
    _, m, X = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    impute = ab.MeanImpute(data_layer, mask_layer)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        assert list(imputed_data[-5:]) == [1., 2., 3., 4., 5.]
        assert KL.eval() == 0.0


def test_random_gaussian_impute(make_missing_data):
    """Test the impute_mean."""
    ab.set_hyperseed(100)
    _, m, X = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    mean_array = 2 * np.ones(D).astype(np.float32)
    var_array = 0.001 * np.ones(D).astype(np.float32)
    impute = ab.RandomGaussImpute(data_layer, mask_layer, mean_array,
                                  var_array)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        correct = [1.9842881, 1.97161114,  1.93794906,  2.02734923, 2.02340364]
        assert np.isclose(list(imputed_data[-5:]), correct).all()
        assert KL.eval() == 0.0
