from itertools import chain
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

import edward as ed
from edward.models import Normal


def learnedFF(WX):
    n = tf.to_float(tf.shape(WX)[1])
    real = tf.cos(WX)
    imag = tf.sin(WX)
    return tf.concat([real, imag], axis=1) / tf.sqrt(n)


# Settings
N = 200
Ns = 200
kernel = RBF(length_scale=1.)
noise = 0.1

# Setup the network
layer_sizes = [50, 50, 50, 1]
activations = [learnedFF, learnedFF, learnedFF, None]

# Optimization
NITER = 30000
N_SAMP = 20


def gen_gausprocess(ntrain, ntest, kern=RBF(length_scale=1.), noise=1.,
                    scale=1., xmin=-10, xmax=10):
    """
    Generate a random (noisy) draw from a Gaussian Process with a RBF kernel.
    """

    # Xtrain = np.linspace(xmin, xmax, ntrain)[:, np.newaxis]
    Xtrain = np.random.rand(ntrain)[:, np.newaxis] * (xmin - xmax) - xmin
    Xtest = np.linspace(xmin, xmax, ntest)[:, np.newaxis]
    Xcat = np.vstack((Xtrain, Xtest))

    K = kern(Xcat, Xcat)
    U, S, V = np.linalg.svd(K)
    L = U.dot(np.diag(np.sqrt(S))).dot(V)
    f = np.random.randn(ntrain + ntest).dot(L)

    ytrain = f[0:ntrain] + np.random.randn(ntrain) * noise
    ftest = f[ntrain:]

    return Xtrain, ytrain, Xtest, ftest


def neural_network(x, W, b, activations):
    h = x
    for W_l, b_l, a_l in zip(W, b, activations):
        WX = tf.matmul(h, W_l) + b_l
        h = a_l(WX) if a_l is not None else WX
    return tf.reshape(h, [-1])


def main():

    np.random.seed(10)

    # Get data
    Xr, yr, Xs, ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr, yr, Xs, ys = np.float32(Xr), np.float32(yr), np.float32(Xs), \
        np.float32(ys)

    # Adjust input layer sizes depending on activation
    dims_in = [1] + [(2 * l if a == learnedFF else l)
                     for a, l in zip(activations[:-1], layer_sizes[:-1])]

    # Configure Network Prior and posterior
    W, qW, b, qb = [], [], [], []
    for d_in, d_out in zip(dims_in, layer_sizes):

        # Prior
        W.append(Normal(
            mu=tf.zeros((d_in, d_out)),
            sigma=tf.ones((d_in, d_out))
        ))
        b.append(Normal(mu=tf.zeros(d_out), sigma=tf.ones(d_out)))

        # Posterior
        qW.append(Normal(
            mu=tf.Variable(tf.random_normal((d_in, d_out))),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal((d_in, d_out))))
        ))
        qb.append(Normal(
            mu=tf.Variable(tf.random_normal((d_out,))),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal((d_out,))))
        ))

    # Model
    y = Normal(mu=neural_network(Xr, W, b, activations),
               sigma=0.1 * tf.ones(N))

    # Setup Variational Inference
    # inference = ed.KLqp(
    inference = ed.ReparameterizationKLKLqp(
        dict(zip(chain(W, b), chain(qW, qb))),
        data={y: yr})

    # Sample functions from variational model to visualize fits.
    # x = tf.constant(Xs)
    X_plt = np.linspace(-20, 20, Ns, dtype=np.float32)[:, np.newaxis]
    x = tf.constant(X_plt)
    mus = []
    for s in range(N_SAMP):
        qW_samp = [q.sample() for q in qW]
        qb_samp = [q.sample() for q in qb]
        mus.append(neural_network(x, qW_samp, qb_samp, activations))

    mus = tf.stack(mus)

    # Start session
    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    # Run inference
    inference.run(n_iter=NITER, n_samples=5)

    # Predict
    Ey = mus.eval()

    pl.figure()
    pl.plot(Xr.flatten(), yr, 'bx')
    pl.plot(Xs.flatten(), ys, 'k')
    # pl.plot(Xs.flatten(), Ey.T, 'r', alpha=0.2)
    pl.plot(X_plt.flatten(), Ey.T, 'r', alpha=0.2)
    pl.show()


if __name__ == "__main__":
    main()
