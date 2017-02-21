import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

from deepnets import BayesNN, Normal, gen_batch, pos, Dense, RandomRBF


# Settings
pl.style.use("ggplot")
N = 2000
Ns = 400
kernel = RBF(length_scale=.5)
noise = 0.1
var = 1.

# Setup the network
no_features = 50

# Optimization
NITER = 20000
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU


def gen_gausprocess(ntrain, ntest, kern=RBF(length_scale=1.), noise=1.,
                    scale=1., xmin=-10, xmax=10):
    """
    Generate a random (noisy) draw from a Gaussian Process.
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

    return Xtrain, ytrain[:, np.newaxis], Xtest, ftest[:, np.newaxis]


def main():

    np.random.seed(10)

    # Get data
    Xr, yr, Xs, ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr, yr, Xs, ys = np.float32(Xr), np.float32(yr), np.float32(Xs), \
        np.float32(ys)

    _, D = Xr.shape

    # Create NN
    like = Normal(var=tf.Variable(pos(var)))
    dgp = BayesNN(N=N, likelihood=like)
    dgp.add(RandomRBF(1, 50))
    dgp.add(Dense(100, 5))
    dgp.add(RandomRBF(5, 50))
    dgp.add(Dense(100, 1))

    X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

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
    Xq = np.linspace(-20, 20, Ns).astype(np.float32)[:, np.newaxis]
    Ey = np.hstack(sess.run(dgp.predict(Xq)))
    # Ey = sess.run(dgp.predict(Xs))
    Eymean = Ey.mean(axis=1)

    print("noise = {}".format(np.sqrt(sess.run(like.var * 1))))
    # for W, b in zip(dgp.qW, dgp.qb):
    #     print(sess.run(1. * W.sigma))
    #     print(sess.run(1. * b.sigma))

    # Plot
    pl.figure()
    pl.plot(Xr.flatten(), yr.flatten(), 'b.', alpha=0.2)
    pl.plot(Xs.flatten(), ys.flatten(), 'k')
    # pl.plot(Xs.flatten(), Ey, 'r', alpha=0.2)
    # pl.plot(Xs.flatten(), Eymean, 'r--')
    pl.plot(Xq.flatten(), Ey, 'r', alpha=0.2)
    pl.plot(Xq.flatten(), Eymean, 'r--')

    pl.figure()
    pl.plot(range(len(loss_val)), loss_val, 'r')
    pl.xlabel("Iteration ($\\times 100$)")
    pl.ylabel("-ve ELBO")
    pl.show()

    # Close the Session when we're done.
    sess.close()


if __name__ == "__main__":
    main()
