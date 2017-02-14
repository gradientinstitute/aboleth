
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

from deepgp import DeepGP

# Settings
N = 200
Ns = 400
kernel = Matern(length_scale=0.5)
noise = 0.1

# Setup the network
no_features = 200
layer_sizes = [10]

# Optimization
NITER = 1000


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
    # yr = np.abs(yr)
    # ys = np.abs(ys)
    Xr, yr, Xs, ys = np.float32(Xr), np.float32(yr), np.float32(Xs), \
        np.float32(ys)

    dgp = DeepGP(no_features, layer_sizes)
    loss = dgp.fit(Xr, yr)

    # Launch the graph.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Fit the network.
    for step in range(NITER):
        sess.run(train)
        loss_val = sess.run(loss)
        if step % 100 == 0:
            print("Iteration {}, loss = {}".format(step, loss_val))

    # Predict
    Ey = sess.run(dgp.predict(Xs))

    # Plot
    pl.figure()
    pl.plot(Xr.flatten(), yr, 'bx')
    pl.plot(Xs.flatten(), ys, 'k')
    pl.plot(Xq.flatten(), Ey.flatten(), 'r')
    pl.show()

    # Close the Session when we're done.
    sess.close()


if __name__ == "__main__":
    main()
