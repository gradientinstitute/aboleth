"""Test the layers module."""
import pytest
import numpy as np
import tensorflow as tf
import aboleth as ab

from aboleth.distributions import norm_prior, gaus_posterior
from aboleth.layers import SampleLayer


D = 10
DIM = (2, 10)
EDIM = (5, 10)


def test_net_outputs(make_graph):
    """Test for expected output dimensions from a net."""
    x, y, N, X_, Y_, N_, like, layers = make_graph
    Net, kl = layers(X=X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()

        P = Net.eval(feed_dict={X_: x, Y_: y, N_: float(N)})

        assert P.shape == (10, N, 1)


def test_input(make_data):
    """Test the input layer."""
    x, _, X = make_data
    s = ab.InputLayer(name='myname')

    F, KL = s(myname=x)
    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        assert KL == 0.0
        assert np.array_equal(f, x)


def test_input_sample(make_data):
    """Test the input and tiling layer."""
    x, _, X = make_data
    s = ab.InputLayer(name='myname', n_samples=3)

    F, KL = s(myname=x)
    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        X_array = X.eval()
        assert KL == 0.0
        assert np.array_equal(f, X_array)
        for i in range(3):
            assert np.array_equal(f[i], x)


def test_activation(make_data):
    """Test nonlinear activation layer."""
    x, _, X = make_data
    act = ab.Activation(tf.tanh)

    tc = tf.test.TestCase()
    with tc.test_session():
        F, KL = act(X)

        assert np.all(np.tanh(X.eval()) == F.eval())
        assert KL == 0


def test_dropout(make_data):
    """Test dropout layer."""
    x, _, X = make_data
    drop = ab.DropOut(0.5)

    F, KL = drop(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()
        prop_zero = np.sum(f == 0) / np.prod(f.shape)

        assert f.shape == X.eval().shape
        assert (prop_zero > 0.4) and (prop_zero < 0.6)
        assert KL == 0


def test_max_pooling2d(make_image_data):
    """Test dropout layer."""
    x, _, X = make_image_data

    # downsample by 2x
    max_pool = ab.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))

    F, KL = max_pool(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()

        # test equivalence of first window across batches
        assert np.all(np.max(x[:, :2, :2, :], axis=(1, 2)) == f[0, :, 0, 0, :])

        # n_samples and batch size remain unchanged
        assert f.shape[:2] == X.eval().shape[:2]

        # downsampled by 2x
        assert 2 * f.shape[2] == X.eval().shape[2]
        assert 2 * f.shape[3] == X.eval().shape[3]

        # number of channels remain unchanged
        assert f.shape[-1] == X.eval().shape[-1]

        assert KL == 0


def test_reshape(make_image_data):
    """Test dropout layer."""
    x, _, X = make_image_data

    # reshape. useful for feeding output of conv layer into dense layer
    reshape = ab.Reshape(target_shape=(28*28*3,))

    F, KL = reshape(X)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval()

        assert f.shape[-1] == np.prod(X.eval().shape[2:])
        assert np.all(f == np.reshape(X.eval(), (5, 100, 28*28*3)))

        assert KL == 0


def test_arc_cosine(make_data):
    """Test the random Arc Cosine kernel."""
    S = 3
    x, _, _ = make_data
    x_, X_ = _make_placeholders(x, S)

    F, KL = ab.RandomArcCosine(n_features=10)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        f = F.eval(feed_dict={x_: x})

        assert f.shape == (3, x.shape[0], 10)
        assert KL == 0


def test_dense_embeddings(make_categories):
    """Test the embedding layer."""
    x, K = make_categories
    N = len(x)
    S = 3
    x_, X_ = _make_placeholders(x, S, tf.int32)
    output, KL = ab.EmbedVariational(output_dim=D, n_categories=K)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        kl = KL.eval()

        assert np.isscalar(kl)
        assert kl > 0

        Phi = output.eval(feed_dict={x_: x})

        assert Phi.shape == (S, N, D)


@pytest.mark.parametrize('dense', [ab.DenseMAP, ab.DenseVariational])
def test_dense_outputs(dense, make_data):
    """Make sure the dense layers output expected dimensions."""
    x, _, _ = make_data
    S = 3

    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = dense(output_dim=D)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        assert P.shape == (S, N, D)
        assert P.dtype == np.float32
        assert np.isscalar(KL.eval(feed_dict={x_: x}))


@pytest.mark.parametrize('layer_args', [
    (SampleLayer, ()),
    (ab.DenseMAP, (D,)),
    (ab.DenseVariational, (D,)),
    (ab.EmbedVariational, (2, D)),
    (ab.RandomFourier, (2, ab.RBF())),
    (ab.RandomArcCosine, (2,)),
])
def test_sample_layer_input_exception(layer_args, make_data):
    """Make sure sample layers fail when the don't get a rank 3 tensor."""
    x, _, _ = make_data
    layer, args = layer_args
    with pytest.raises(AssertionError):
        layer(*args)(x)


@pytest.mark.parametrize('kernels', [
    (ab.RBF, {}),
    (ab.RBFVariational, {}),
    (ab.Matern, {'p': 1}),
    (ab.Matern, {'p': 2})
])
def test_fourier_features(kernels, make_data):
    """Test random fourier kernels approximations."""
    D = 100
    S = 3
    kern, p = kernels
    k = kern(D, **p)

    x, _, _ = make_data
    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = ab.RandomFourier(D, k)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        for i in range(P.shape[0]):
            p = P[i]
            assert p.shape == (N, 2 * D)
            # Check behaving properly with k(x, x) ~ 1.0
            assert np.allclose((p**2).sum(axis=1), np.ones(N))

        # Make sure we get a valid KL
        kl = KL.eval() if isinstance(KL, tf.Tensor) else KL
        assert kl >= 0


@pytest.mark.parametrize('dists', [
    {'prior_W': norm_prior(DIM, 1.), 'prior_b': norm_prior((D,), 1.)},
    {'post_W': norm_prior(DIM, 1.), 'post_b': norm_prior((D,), 1.)},
    {'prior_W': norm_prior(DIM, 1.), 'post_W': norm_prior(DIM, 1.)},
    {'prior_W': norm_prior(DIM, 1.), 'post_W': gaus_posterior(DIM, 1.)},
    {'prior_W': gaus_posterior(DIM, 1.), 'post_W': gaus_posterior(DIM, 1.)},
])
def test_dense_distribution(dists, make_data):
    """Test initialising dense variational layers with distributions."""
    x, _, _ = make_data
    S = 3

    x_, X_ = _make_placeholders(x, S)
    N = x.shape[0]

    Phi, KL = ab.DenseVariational(output_dim=D, **dists)(X_)
    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        P = Phi.eval(feed_dict={x_: x})
        assert P.shape == (S, N, D)
        assert KL.eval() >= 0.


@pytest.mark.parametrize('dists', [
    {'prior_W': norm_prior(EDIM, 1.), 'post_W': norm_prior(EDIM, 1.)},
    {'prior_W': norm_prior(EDIM, 1.), 'post_W': gaus_posterior(EDIM, 1.)},
    {'prior_W': gaus_posterior(EDIM, 1.), 'post_W': gaus_posterior(EDIM, 1.)},
])
def test_embeddings_distribution(dists, make_categories):
    """Test initialising embedding variational layers with distributions."""
    x, K = make_categories
    N = len(x)
    S = 3
    x_, X_ = _make_placeholders(x, S, tf.int32)
    output, KL = ab.EmbedVariational(output_dim=D, n_categories=K, **dists)(X_)

    tc = tf.test.TestCase()
    with tc.test_session():
        tf.global_variables_initializer().run()
        Phi = output.eval(feed_dict={x_: x})
        assert Phi.shape == (S, N, D)
        assert KL.eval() >= 0.


def _make_placeholders(x, S, xtype=tf.float32):
    x_ = tf.placeholder(xtype, x.shape)
    X_ = tf.tile(tf.expand_dims(x_, 0), [S, 1, 1])
    return x_, X_
