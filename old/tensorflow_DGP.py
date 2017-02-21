# Pure tensorflow Deep GP MAP approximation
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, Matern

# Settings
N = 200
Ns = 400
kernel = Matern(length_scale=0.5)
noise = 0.1

# Setup the network
no_features = 200
layer_sizes = [10]

# Optimization
NITER = 20000


class RandomFF():

    def __init__(self, input_dim, n_bases):
        self.D = np.float32(n_bases)
        self.d = input_dim
        self.P = np.random.randn(input_dim, n_bases).astype(np.float32)

    def transform(self, F):
        FP = tf.matmul(F, self.P)
        real = tf.cos(FP)
        imag = tf.sin(FP)
        return tf.concat([real, imag], axis=1) / tf.sqrt(self.D)


def neural_network(x, W, b, Phi):
    F = x
    for W_l, b_l, phi in zip(W, b, Phi):
        P = phi.transform(F)
        F = tf.matmul(P, W_l) + b_l
    return tf.reshape(F, [-1])


def regularizer(W, b, lambdaW=0.1, lambdab=0.1):
    reg = 0
    for W_l, b_l in zip(W, b):
        reg += lambdaW * tf.nn.l2_loss(W_l) + lambdab * tf.nn.l2_loss(b_l)

    return reg


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

    # Adjust input layer sizes depending on activation
    dims_in = [1] + layer_sizes
    dims_out = layer_sizes + [1]

    # Configure Network
    W, b, Phi = [], [], []
    for d_in, d_out in zip(dims_in, dims_out):
        Phi.append(RandomFF(d_in, no_features))
        W.append(tf.Variable(tf.random_normal((2 * no_features, d_out))))
        b.append(tf.Variable(tf.random_normal((d_out,))))

    # Model and training
    y = neural_network(Xr, W, b, Phi)
    loss = tf.nn.l2_loss(y - yr) + regularizer(W, b)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    # Model prediction
    Xq = np.linspace(-20, 20, Ns, dtype=np.float32)[:, np.newaxis]
    # Xq = tf.constant(Xs)
    yq = neural_network(Xq, W, b, Phi)

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    # Fit the network.
    for step in range(NITER):
        sess.run(train)
        loss_val = sess.run(loss)
        if step % 100 == 0:
            print("Iteration {}, loss = {}".format(step, loss_val))

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
