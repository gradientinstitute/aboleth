"""Tests for imputation layers."""

import numpy as np
import tensorflow as tf

import aboleth as ab


def test_mean_impute(make_missing_data):
    """Test the impute_mean."""
    _, m, X, _ = make_missing_data

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


def test_fixed_gaussian_impute(make_missing_data):
    """Test the impute_mean."""
    ab.set_hyperseed(100)
    _, m, X, _ = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    mean_array = 2 * np.ones(D).astype(np.float32)
    var_array = 0.001 * np.ones(D).astype(np.float32)
    impute = ab.FixedNormalImpute(data_layer, mask_layer, mean_array,
                                  var_array)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        correct = [1.98, 1.97,  1.93,  2.02, 2.02]
        assert np.isclose(list(imputed_data[-5:]), correct, atol=0.1).all()
        assert KL.eval() == 0.0


def test_leanred_scalar_impute(make_missing_data):
    """Test the impute that learns a scalar value to impute for each col."""
    ab.set_hyperseed(100)
    _, m, X, _ = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    impute = ab.LearnedScalarImpute(data_layer, mask_layer)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        X_imputed = F.eval()
        assert KL.eval() == 0.0  # Might want to change this in the future
        assert(X_imputed.shape == X.shape)


def test_learned_normal_impute(make_missing_data):
    """Test the learned normal impute function."""
    ab.set_hyperseed(100)
    _, m, X, _ = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    impute = ab.LearnedNormalImpute(data_layer, mask_layer)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        X_imputed = F.eval()
        assert KL.eval() == 0.0  # Might want to change this in the future
        assert(X_imputed.shape == X.shape)
