# requirements:
# - numpy
# - scipy
# - Tensorflow
# - Keras
# - matplotlib
# - scikit learn

import numpy as np
import matplotlib.pyplot as pl
import keras.backend as K
import tensorflow as tf

from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, Matern
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda


# Settings
N = 100
Ns = 200
kernel = Matern(length_scale=1.)
noise = 0.1


def randomFF(WX):
    n = tf.to_float(K.shape(WX)[1])
    real = K.cos(WX)
    imag = K.sin(WX)
    return K.concatenate([real, imag], axis=1) / tf.sqrt(n)


def randomFF_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)


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

    # Make model
    model = Sequential()

    model.add(Dense(20, input_dim=1, init='normal', activation='linear'))
    model.add(Lambda(randomFF, output_shape=randomFF_output_shape))
    model.add(Dense(40, init='normal', activation='linear'))
    model.add(Lambda(randomFF, output_shape=randomFF_output_shape))
    # model.add(Dense(40, init='normal', activation="relu"))
    model.add(Dense(80, init='normal', activation='linear'))
    model.add(Lambda(randomFF, output_shape=randomFF_output_shape))

    # model.add(Dense(10, input_dim=1, init='normal', activation='linear'))
    # model.add(Dense(200, input_dim=1, init='normal', activation="relu"))
    # model.add(Dropout(0.1))
    # model.add(Dense(200, init='normal', activation="relu"))
    # model.add(Dropout(0.1))
    # model.add(Dense(50, init='normal', activation="relu"))
    # model.add(Dropout(0.1))

    model.add(Dense(1, init='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Learn model
    model.fit(Xr, yr, nb_epoch=3000, batch_size=10)

    # Test model
    loss_and_metrics = model.evaluate(Xs, ys, batch_size=10)
    print("")
    print(loss_and_metrics)
    Ey = model.predict(Xs, batch_size=10)

    pl.figure()
    pl.plot(Xr.flatten(), yr, 'bx')
    pl.plot(Xs.flatten(), ys, 'k')
    pl.plot(Xs.flatten(), Ey, 'r')
    pl.show()


if __name__ == "__main__":
    main()
