"""Tests for imputation layers."""

import numpy as np
import tensorflow as tf

import aboleth as ab


def test_mask_input(make_missing_data):
    """Test the mask input layer."""
    _, m, _, _ = make_missing_data
    s = ab.MaskInputLayer(name='myname')

    F, KL = s(myname=m)
    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        assert KL == 0.0
        assert np.array_equal(f, m)


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
        assert np.allclose(imputed_data[-5:], [1., 2., 3., 4., 5.])
        assert KL.eval() == 0.0


def test_learned_scalar_impute(make_missing_data):
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


def test_fixed_normal_impute(make_missing_data):
    """Test the fixed normal random imputation."""
    ab.set_hyperseed(100)
    _, m, X, _ = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    mean_array = 2 * np.ones(D).astype(np.float32)
    std_array = np.sqrt(0.001) * np.ones(D).astype(np.float32)
    impute = ab.FixedNormalImpute(data_layer, mask_layer, mean_array,
                                  std_array)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        correct = np.array([1.94, 1.97,  1.93,  2.03, 2.02])
        assert np.allclose(imputed_data[-5:], correct, atol=0.1)
        assert KL.eval() == 0.0


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


def test_extra_category_impute(make_missing_categories):
    """Test the impute that learns a scalar value to impute for each col."""
    ab.set_hyperseed(100)
    X, m, ncats = make_missing_categories
    X_true = np.copy(X)
    X_true[:, m[:, 0], 0] = ncats[0]
    X_true[:, m[:, 1], 1] = ncats[1]

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    impute = ab.ExtraCategoryImpute(data_layer, mask_layer, ncats)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        X_imputed = F.eval()
        assert np.all(X_imputed == X_true)
