"""Dataset generation and loading utilitites."""
import os

import numpy as np
from scipy.io import loadmat
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets.base import Bunch
from six.moves import urllib

from aboleth.random import seedgen

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')


def gp_draws(ntrain, ntest, kern=RBF(length_scale=0.5), noise=0.1, xmin=-10,
             xmax=10):
    r"""Generate a random (noisy) draw from a Gaussian Process.

    Parameters
    ----------
    ntrain : int
        number of training points to generate
    ntest : int
        number of testing points to generate
    kern : scikit.gaussian_process.kernels
        kernel to generate data from
    noise : float
        Gaussian noise (standard deviation) to add to GP draws
    xmin : float
        minimum extent of inputs, X
    xmax : float
        maximum extent of inputs, X

    Returns
    -------
    Xtrain : ndarray
        of shape (ntrain, 1) of training inputs
    Ytrain : ndarray
        of shape (ntrain, 1) of training targets
    Xtest : ndarray
        of shape (ntrain, 1) of testing inputs
    Ytest : ndarray
        of shape (ntrain, 1) of testing targets
    """
    randgen = np.random.RandomState(next(seedgen))

    Xtrain = randgen.rand(ntrain)[:, np.newaxis] * (xmin - xmax) - xmin
    Xtest = np.linspace(xmin, xmax, ntest)[:, np.newaxis]
    Xcat = np.vstack((Xtrain, Xtest))

    K = kern(Xcat, Xcat)
    U, S, V = np.linalg.svd(K)
    L = U.dot(np.diag(np.sqrt(S))).dot(V)
    f = randgen.randn(ntrain + ntest).dot(L)

    Ytrain = f[0:ntrain] + randgen.randn(ntrain) * noise
    ftest = f[ntrain:]

    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain[:, np.newaxis].astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    ftest = ftest[:, np.newaxis].astype(np.float32)
    return Xtrain, Ytrain, Xtest, ftest


def fetch_gpml_sarcos_data():
    r"""Fetch the SARCOS dataset.

    Fetch the SARCOS dataset from the internet and parse appropriately into
    python arrays

    Returns
    -------
    data : sklearn.datasets.Bunch
        Bunch object that contrains the dataset

    Examples
    --------
    >>> gpml_sarcos = fetch_gpml_sarcos_data()

    >>> gpml_sarcos.train.data.shape
    (44484, 21)

    >>> gpml_sarcos.train.targets.shape
    (44484,)

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
