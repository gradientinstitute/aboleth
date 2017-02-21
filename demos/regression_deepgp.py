import numpy as np
import bokeh.plotting as bk
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

from deepnets import BayesNN, normal, gen_batch, pos, Dense, RandomRBF


# Settings
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

    Ytrain = f[0:ntrain] + np.random.randn(ntrain) * noise
    ftest = f[ntrain:]

    return Xtrain, Ytrain[:, np.newaxis], Xtest, ftest[:, np.newaxis]


def main():

    np.random.seed(10)

    # Get data
    Xr, Yr, Xs, Ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr, Yr, Xs, Ys = np.float32(Xr), np.float32(Yr), np.float32(Xs), \
        np.float32(Ys)

    _, D = Xr.shape

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    # Create NN
    like = normal(variance=pos(tf.Variable(var)))
    dgp = BayesNN(N=N, likelihood=like)
    dgp.add(RandomRBF(input_dim=1, n_features=50))
    dgp.add(Dense(output_dim=5))
    dgp.add(RandomRBF(n_features=50))
    dgp.add(Dense(output_dim=1))

    loss = dgp.loss(X_, Y_)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    # Launch the graph.
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    # Fit the network.
    batches = gen_batch({X_: Xr, Y_: Yr}, batch_size=10, n_iter=NITER)
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

    # Plot
    f = bk.figure(tools='pan,box_zoom,reset', sizing_mode='stretch_both')
    f.circle(Xr.flatten(), Yr.flatten(), fill_color='blue', alpha=0.2,
            legend='Training')
    f.line(Xs.flatten(), Ys.flatten(), line_color='black', legend='Truth')
    for y in Ey.T:
        f.line(Xq.flatten(), y, line_color='red', alpha=0.2, legend='Samples')
    f.line(Xq.flatten(), Eymean.flatten(), line_color='green', legend='Mean')
    bk.show(f)

    # Close the Session when we're done.
    sess.close()


if __name__ == "__main__":
    main()
