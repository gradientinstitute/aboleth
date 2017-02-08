from itertools import chain
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

import edward as ed
from edward.models import Normal


def randomFF(WX):
    n = tf.to_float(tf.shape(WX)[1])
    real = tf.cos(WX)
    imag = tf.sin(WX)
    return tf.concat([real, imag], axis=1) / tf.sqrt(n)


# Settings
# N = 200
Ns = 200
noise = 0.1

# Setup the network
layer_sizes = [100, 100, 1]
activations = [randomFF, randomFF, None]

# Optimization
NITER = 20000
N_SAMP = 20


def neural_network(x, W, b, activations):
    h = x
    for W_l, b_l, a_l in zip(W, b, activations):
        WX = tf.matmul(h, W_l) + b_l
        h = a_l(WX) if a_l is not None else WX
    return tf.reshape(h, [-1])


def main():

    # Get data
    # Xr, yr, Xs, ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr = np.load('motorcycle_X.npy')
    yr = np.load('motorcycle_y.npy').flatten()
    yr -= yr.mean()
    yr /= yr.std()
    Xs = np.linspace(Xr.min() - 10, Xr.max() + 10, Ns)[:, np.newaxis]

    Xr, yr, Xs = np.float32(Xr), np.float32(yr), np.float32(Xs)
    N = Xr.shape[0]

    # Adjust input layer sizes depending on activation
    dims_in = [1] + [(2 * l if a == randomFF else l)
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
    x = tf.constant(Xs)
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
    pl.plot(Xs.flatten(), Ey.T, 'r', alpha=0.2)
    pl.show()


if __name__ == "__main__":
    main()
