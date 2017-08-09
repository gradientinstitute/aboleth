"""Test the kernels module."""
import pytest

import aboleth as ab


@pytest.mark.parametrize('kernels', [
    (ab.RBF, {}),
    (ab.Matern, {'p': 1}),
    (ab.Matern, {'p': 2})
])
def test_shift_invariant_kernels(kernels, make_data):
    """Test random kernels approximations."""
    d, D = 10, 100
    kern, p = kernels
    k = kern(D, **p)

    # Check dim
    P = k.weights(input_dim=d, n_features=D)
    assert P.shape == (d, D)
