"""Test the kernels module."""
import pytest
import numpy as np
import tensorflow as tf

import aboleth as ab

kernel_list = [
    (ab.RBF, {}),
    (ab.RBFVariational, {}),
    (ab.Matern, {'p': 1}),
    (ab.Matern, {'p': 2})
]


@pytest.mark.parametrize('kernels', kernel_list)
def test_shift_invariant_kernels(kernels):
    """Test random kernels approximations."""
    d, D = 10, 100
    kern, p = kernels
    k = kern(**p)

    # Check dim
    P, KL = k.weights(input_dim=d, n_features=D)
    assert P.shape == (d, D)


@pytest.mark.parametrize('kernels', kernel_list)
@pytest.mark.parametrize('lenscales', [
    1.0,
    np.array([1.0]),
    np.ones((10, 1), dtype=np.float32),
    tf.constant(1.0),
    tf.ones((10, 1))
])
def test_ARD_lenscales(kernels, lenscales):
    """Test random kernels with multi-dim lenscales."""
    d, D = 10, 100
    kern, p = kernels
    k = kern(lenscale=lenscales, **p)

    # Check dim
    P, KL = k.weights(input_dim=d, n_features=D)
    assert P.shape == (d, D)
