# Pure tensorflow Deep GP MAP approximation
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
import edward as ed
from sklearn.gaussian_process.kernels import RBF, Matern
from edward.models import Normal

# Settings
N = 200
Ns = 400
kernel = RBF(length_scale=1)
noise = 0.1

# Setup the network
no_features = 100
layer_sizes = [10, 10]

# Optimization
NITER = 5000
NPREDSAMP = 20
NSAMP = 5
WPRIOR = 1


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
    if len(W) == 0:
        return F
    for W_l, b_l, phi in zip(W, b, Phi):
        P = phi.transform(F)
        F = tf.matmul(P, W_l) + b_l
    return F


def standardlinearmodel(X, W, b, basis):
    P = basis.transform(X)
    F = tf.matmul(P, W) + b
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
    dims_in = [1] + layer_sizes[:-1]
    dims_out = layer_sizes

    # Configure Network
    W, b, Phi = [], [], []
    for d_in, d_out in zip(dims_in, dims_out):
        Phi.append(RandomFF(d_in, no_features))
        W.append(tf.Variable(tf.random_normal((2 * no_features, d_out))))
        b.append(tf.Variable(tf.random_normal((d_out,))))

    if len(layer_sizes) == 0:
        layer_sizes.append(1)

    # Final Layer
    Phi_out = RandomFF(layer_sizes[-1], no_features)

    # Prior
    W_out = Normal(
        mu=tf.zeros((2 * no_features, 1)),
        sigma=tf.ones((2 * no_features, 1)) * WPRIOR
    )
    b_out = Normal(mu=tf.zeros(1), sigma=tf.ones(1) * WPRIOR)

    # Posterior
    qW_out = Normal(
        mu=tf.Variable(tf.random_normal((2 * no_features, 1))),
        sigma=tf.nn.softplus(
            tf.Variable(tf.random_normal((2 * no_features, 1)))
        )
    )
    qb_out = Normal(
        mu=tf.Variable(tf.random_normal((1,))),
        sigma=tf.nn.softplus(tf.Variable(tf.random_normal((1,))))
    )

    # Model and training
    y = Normal(
        mu=standardlinearmodel(
            neural_network(Xr, W, b, Phi),
            W_out,
            b_out,
            Phi_out
        ),
        sigma=noise * tf.ones(N)
    )

    # Setup Variational Inference
    inference = ed.KLqp(
        {W_out: qW_out, b_out: qb_out},
        data={y: yr})

    # Model prediction
    Xq = np.linspace(-20, 20, Ns, dtype=np.float32)[:, np.newaxis]
    # Xq = tf.constant(Xs)
    mus = []
    for s in range(NPREDSAMP):
        mus.append(standardlinearmodel(
            neural_network(Xq, W, b, Phi),
            qW_out.sample(),
            qb_out.sample(),
            Phi_out
        ))

    mus = tf.transpose(tf.stack(mus))

    # Start session
    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    # Run inference
    inference.run(n_iter=NITER)#, n_samples=NSAMP)

    # Predict
    Eys = mus.eval()
    Ey = Eys.mean(axis=1)

    # Plot
    pl.figure()
    pl.plot(Xr.flatten(), yr, 'bx')
    pl.plot(Xs.flatten(), ys, 'k')
    pl.plot(Xq.flatten(), Eys, 'r', alpha=0.1)
    pl.plot(Xq.flatten(), Ey, 'r--')
    pl.show()

    # Close the Session when we're done.
    sess.close()


if __name__ == "__main__":
    main()
