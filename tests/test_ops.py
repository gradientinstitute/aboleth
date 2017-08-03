"""Test the ops module."""
import numpy as np
import numpy.ma as ma
import tensorflow as tf
import aboleth as ab


def test_stack2():
    """Test base implementation of stack."""
    def f(X):
        return "f({})".format(X), 10.0

    def g(X):
        return "g({})".format(X), 20.0

    h = ab.ops._stack2(f, g)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = h(X="x")
        assert phi == "g(f(x))"
        assert loss.eval() == 30.0


def test_stack2_multi():
    """Test base implementation of stack."""
    def f(X, Y):
        return "f({}, {})".format(X, Y), 10.0

    def g(X):
        return "g({})".format(X), 20.0

    h = ab.ops._stack2(f, g)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = h(X="x", Y="y")
        assert phi == "g(f(x, y))"
        assert loss.eval() == 30.0


def test_stack(mocker):
    """Test stack another way with mocking."""
    mocked_reduce = mocker.patch('aboleth.ops.reduce')
    f = mocker.MagicMock()
    g = mocker.MagicMock()
    h = mocker.MagicMock()
    ab.stack(f, g, h)
    mocked_reduce.assert_called_once_with(ab.ops._stack2, (f, g, h))


def test_stack_real():
    """Test implementation of stack."""
    def f(X, Y):
        return "f({}, {})".format(X, Y), 10.0

    def g(X):
        return "g({})".format(X), 20.0

    def h(X):
        return "h({})".format(X), 5.0

    h = ab.stack(f, g, h)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = h(X="x", Y="y")
        assert phi == "h(g(f(x, y)))"
        assert loss.eval() == 35.0


def test_concat(make_data):
    """Test concatenation op."""
    x, _, X = make_data

    # This replicates the input layer behaviour
    def f(**kwargs):
        return kwargs['X'], 0.0

    def g(**kwargs):
        return kwargs['Y'], 0.0

    catlayer = ab.concat(f, g)

    F, KL = catlayer(X=X, Y=X)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape[0:2] + (2 * orig.shape[2],)
        assert np.all(forked == np.dstack((orig, orig)))
        assert KL.eval() == 0.0


def test_slicecat(make_data):
    """Test concatenation  of slices op."""
    x, _, X = make_data

    def make_idxlayer(i):
        def idlayer(X):
            return X + i, float(i)
        return idlayer

    catlayer = ab.slicecat(make_idxlayer(2), make_idxlayer(3))
    F, KL = catlayer(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        catted = F.eval()
        orig = X.eval()
        assert catted.shape == orig.shape
        assert np.allclose(catted[:, :, 0], orig[:, :, 0] + 2)
        assert np.allclose(catted[:, :, 1], orig[:, :, 1] + 3)
        assert KL.eval() == 5.0


def test_add(make_data):
    """Test the add join."""
    x, _, X = make_data

    # This replicates the input layer behaviour
    def f(**kwargs):
        return kwargs['X'], 0.0

    def g(**kwargs):
        return kwargs['Y'], 0.0

    addlayer = ab.add(f, g)

    F, KL = addlayer(X=X, Y=X)

    tc = tf.test.TestCase()
    with tc.test_session():
        forked = F.eval()
        orig = X.eval()
        assert forked.shape == orig.shape
        assert np.all(forked == 2 * orig)
        assert KL.eval() == 0.0


def test_mean_impute(make_missing_data):
    """Test the impute_mean."""
    _, m, X = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    impute = ab.mean_impute(data_layer, mask_layer)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        assert list(imputed_data[-5:]) == [1., 2., 3., 4., 5.]
        assert KL.eval() == 0.0


def test_gaussian_impute(make_missing_data):
    """Test the impute_mean."""
    RSEED=100
    ab.set_hyperseed(RSEED)
    _, m, X = make_missing_data

    # This replicates the input layer behaviour
    def data_layer(**kwargs):
        return kwargs['X'], 0.0

    def mask_layer(**kwargs):
        return kwargs['M'], 0.0

    n, N, D = X.shape
    mean_array = 2 * np.ones(D).astype(np.float32)
    var_array = 0.001 * np.ones(D).astype(np.float32)
    impute = ab.gaussian_impute(data_layer, mask_layer, mean_array, var_array)

    F, KL = impute(X=X, M=m)

    tc = tf.test.TestCase()
    with tc.test_session():
        X_imputed = F.eval()
        imputed_data = X_imputed[1, m]
        correct = [1.9842881, 1.97161114,  1.93794906,  2.02734923, 2.02340364]
        assert np.isclose(list(imputed_data[-5:]), correct).all()
        assert KL.eval() == 0.0
