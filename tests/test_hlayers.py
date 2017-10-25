"""Test the hlayers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab


def test_concat(make_data):
    """Test concatenation layer."""
    x, _, X = make_data

    # This replicates the input layer behaviour
    f = ab.InputLayer('X', n_samples=3)
    g = ab.InputLayer('Y', n_samples=3)

    catlayer = ab.Concat(f, g)

    F, KL = catlayer(X=x, Y=x)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape[0:2] + (2 * orig.shape[2],)
        assert np.all(forked == np.dstack((orig, orig)))
        assert KL.eval() == 0.0


@pytest.mark.parametrize('reps', [1, 3, 10])
def test_perfeature(make_data, reps):
    """Test per-feature application of layers."""
    x, _, _ = make_data
    x = np.repeat(x, reps, axis=-1)
    X = x[np.newaxis, ...]

    def make_idxlayer(i):
        def idlayer(X):
            return X + i, float(i)
        return idlayer

    if reps > 1:
        begin = range(0, 2 * reps, 2)
        end = range(2, 2 * reps + 2, 2)
        slices = [slice(b, e) for b, e in zip(begin, end)]
        layers = [make_idxlayer(i) for i in range(reps)]
    else:
        slices = None
        layers = [make_idxlayer(2), make_idxlayer(3)]

    catlayer = ab.PerFeature(*layers, slices=slices)
    F, KL = catlayer(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        catted = F.eval()
        assert catted.shape == X.shape
        if reps == 1:
            assert np.allclose(catted[:, :, 0], X[:, :, 0] + 2)
            assert np.allclose(catted[:, :, 1], X[:, :, 1] + 3)
            assert KL.eval() == 5.0
        else:
            for i, s in enumerate(slices):
                assert np.allclose(catted[:, :, s], X[:, :, s] + i)
                KL.eval() == np.sum(range(reps))


def test_sum(make_data):
    """Test the summation layer."""
    x, _, X = make_data

    # This replicates the input layer behaviour
    def f(**kwargs):
        return kwargs['X'], 0.0

    def g(**kwargs):
        return kwargs['Y'], 0.0

    addlayer = ab.Sum(f, g)

    F, KL = addlayer(X=X, Y=X)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape
        assert np.all(forked == 2 * orig)
        assert KL.eval() == 0.0
