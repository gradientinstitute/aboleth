# Pure tensorflow neural net implementation
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern


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
layer_sizes = [1000, 200, 1]
activations = [learnedFF, learnedFF, None]

# Optimization
NITER = 5000


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


def regularizer(W, b, lambdaW=0.1, lambdab=0.1):

    reg = 0
    for W_l, b_l in zip(W, b):

        reg += lambdaW * tf.nn.l2_loss(W_l) + lambdab * tf.nn.l2_loss(b_l)

    return reg


def main():

    np.random.seed(10)

    # Get data
    Xr, yr, Xs, ys = gen_gausprocess(N, Ns, kern=kernel, noise=noise)
    Xr, yr, Xs, ys = np.float32(Xr), np.float32(yr), np.float32(Xs), \
        np.float32(ys)

    # Adjust input layer sizes depending on activation
    dims_in = [1] + [(2 * l if a == learnedFF else l)
                     for a, l in zip(activations[:-1], layer_sizes[:-1])]

    # Configure Network
    W, b = [], []
    for d_in, d_out in zip(dims_in, layer_sizes):
        W.append(tf.Variable(tf.random_normal((d_in, d_out))))
        b.append(tf.Variable(tf.random_normal((d_out,))))

    # Model and training
    y = neural_network(Xr, W, b, activations)
    loss = tf.nn.l2_loss(y - yr) + regularizer(W, b)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    # Model prediction
    Xq = np.linspace(-20, 20, Ns, dtype=np.float32)[:, np.newaxis]
    # Xq = tf.constant(Xq)
    # Xq = tf.constant(Xs)
    yq = neural_network(Xq, W, b, activations)

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    # Fit the network.
    for step in range(NITER):
        sess.run(train)
        if step % 100 == 0:
            print("Iteration {}".format(step))

    # Predict
    Ey = sess.run(yq)

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
