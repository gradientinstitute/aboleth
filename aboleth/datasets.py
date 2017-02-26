"""Dataset generation and loading utilitites."""
import os

import numpy as np
from scipy.io import loadmat
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets.base import Bunch
from six.moves import urllib


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')


def gp_draws(ntrain, ntest, kern=RBF(length_scale=0.5), noise=0.1, scale=1.,
             xmin=-10, xmax=10):
    """Generate a random (noisy) draw from a Gaussian Process."""
    Xtrain = np.random.rand(ntrain)[:, np.newaxis] * (xmin - xmax) - xmin
    Xtest = np.linspace(xmin, xmax, ntest)[:, np.newaxis]
    Xcat = np.vstack((Xtrain, Xtest))

    K = kern(Xcat, Xcat)
    U, S, V = np.linalg.svd(K)
    L = U.dot(np.diag(np.sqrt(S))).dot(V)
    f = np.random.randn(ntrain + ntest).dot(L)

    Ytrain = f[0:ntrain] + np.random.randn(ntrain) * noise
    ftest = f[ntrain:]

    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain[:, np.newaxis].astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    ftest = ftest[:, np.newaxis].astype(np.float32)
    return Xtrain, Ytrain, Xtest, ftest


def fetch_gpml_sarcos_data(transpose_data=True):
    """
    Fetch the SARCOS dataset from the internet and parse appropriately into
    python arrays

    >>> gpml_sarcos = fetch_gpml_sarcos_data()

    >>> gpml_sarcos.train.data.shape
    (44484, 21)

    >>> gpml_sarcos.train.targets.shape
    (44484,)

    >>> gpml_sarcos.train.targets.round(2) # doctest: +ELLIPSIS
    array([ 50.29,  44.1 ,  37.35, ...,  22.7 ,  17.13,   6.52])

    >>> gpml_sarcos.test.data.shape
    (4449, 21)

    >>> gpml_sarcos.test.targets.shape
    (4449,)
    """
    train_src_url = "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat"
    test_src_url = ("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test"
                    ".mat")

    train_filename = os.path.join(DEFAULT_DATA_PATH, 'sarcos_inv.mat')
    test_filename = os.path.join(DEFAULT_DATA_PATH, 'sarcos_inv_test.mat')

    if not os.path.exists(train_filename):
        urllib.request.urlretrieve(train_src_url, train_filename)

    if not os.path.exists(test_filename):
        urllib.request.urlretrieve(test_src_url, test_filename)

    train_data = loadmat(train_filename).get('sarcos_inv')
    test_data = loadmat(test_filename).get('sarcos_inv_test')

    train_bunch = Bunch(data=train_data[:, :21],
                        targets=train_data[:, 21])

    test_bunch = Bunch(data=test_data[:, :21],
                       targets=test_data[:, 21])

    return Bunch(train=train_bunch, test=test_bunch)