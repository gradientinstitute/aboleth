"""Package helper utilities."""
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF
import numpy as np


def pos(X, minval=1e-10):
    # return tf.exp(X)  # Medium speed, but gradients tend to explode
    # return tf.nn.softplus(X)  # Slow but well behaved!
    return tf.maximum(tf.abs(X), minval)  # Faster, but more local optima


def gp_draws(ntrain, ntest, kern=RBF(length_scale=0.5), noise=0.1, scale=1.,
             xmin=-10, xmax=10):
    """
    Generate a random (noisy) draw from a Gaussian Process.
    """

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


def batch(data_dict, N_, batch_size, n_iter=10000, random_state=None):

    N = data_dict[list(data_dict.keys())[0]].shape[0]
    perms = endless_permutations(N, random_state)

    i = 0
    while i < n_iter:
        i += 1
        ind = np.array([next(perms) for _ in range(batch_size)])
        batch_dict = {k: v[ind] for k, v in data_dict.items()}
        batch_dict[N_] = N
        yield batch_dict


def endless_permutations(N, random_state=None):
    """
    Generate an endless sequence of random integers from permutations of the
    set [0, ..., N).
    If we call this N times, we will sweep through the entire set without
    replacement, on the (N+1)th call a new permutation will be created, etc.
    Parameters
    ----------
    N: int
        the length of the set
    random_state: int or RandomState, optional
        random seed

    Yields
    ------
    int:
        a random int from the set [0, ..., N)
    """
    if isinstance(random_state, np.random.RandomState):
        generator = random_state
    else:
        generator = np.random.RandomState(random_state)

    while True:
        batch_inds = generator.permutation(N)
        for b in batch_inds:
            yield b
