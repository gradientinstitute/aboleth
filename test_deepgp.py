from functools import partial

import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

from deepgp import DeepGP, Normal, gen_batch

# Settings
pl.style.use("ggplot")
N = 200
Ns = 400
kernel = Matern(length_scale=1.)
noise = 0.1
var = noise**2

# Setup the network
no_features = 50
layer_sizes = [10]

# Optimization
NITER = 40000
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU


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


def main():

    np.random.seed(10)

    # Get data
    Xr, yr, Xs, ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr, yr, Xs, ys = np.float32(Xr), np.float32(yr), np.float32(Xs), \
        np.float32(ys)

    _, D = Xr.shape

    # Create NN
    layers = [D] + layer_sizes + [1]
    like = partial(Normal.log_pdf, var=var)
    dgp = DeepGP(N, like, no_features, layers)

    X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
    y_ = tf.placeholder(dtype=tf.float32, shape=(None,))

    loss = dgp.loss(X_, y_)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    # Launch the graph.
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    # Fit the network.
    batches = gen_batch({X_: Xr, y_: yr}, batch_size=10, n_iter=NITER)
    loss_val = []
    for i, data in enumerate(batches):
        sess.run(train, feed_dict=data)
        if i % 100 == 0:
            loss_val.append(sess.run(loss, feed_dict=data))
            print("Iteration {}, loss = {}".format(i, loss_val[-1]))

    # Predict
    Ey = sess.run(dgp.predict(Xs))
    Eymean = Ey.mean(axis=1)

    # Plot
    pl.figure()
    pl.plot(Xr.flatten(), yr, 'bx')
    pl.plot(Xs.flatten(), ys, 'k')
    pl.plot(Xs.flatten(), Ey, 'r', alpha=0.2)
    pl.plot(Xs.flatten(), Eymean, 'r--')

    pl.figure()
    pl.plot(range(len(loss_val)), loss_val, 'r')
    pl.xlabel("Iteration")
    pl.ylabel("-ve ELBO")
    pl.show()

    # Close the Session when we're done.
    sess.close()


if __name__ == "__main__":
    main()
