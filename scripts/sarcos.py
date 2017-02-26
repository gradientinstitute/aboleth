"""Sarcos regression demo."""
import click
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import aboleth as ab
from aboleth.datasets import fetch_gpml_sarcos_data


VARIANCE = 10.0
KERN = ab.RBF(lenscale=ab.pos(tf.Variable(10. * tf.ones((21, 1)))))
LAYERS = [
    ab.randomFourier(n_features=100, kernel=KERN),
    ab.dense_map(output_dim=10, l2_reg=0),
    ab.randomFourier(n_features=50),
    ab.dense_var(output_dim=1)
]
BATCH_SIZE = 50
NITERATIONS = 100000
NPREDICTSAMPLES = 100

CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU


@click.command()
def main():

    data = fetch_gpml_sarcos_data()
    Xr, Yr = data.train.data, data.train.targets[:, np.newaxis]
    Xs, Ys = data.test.data, data.test.targets[:, np.newaxis]
    N, D = Xr.shape

    # Scale and centre the data
    ss = StandardScaler()
    Xr = ss.fit_transform(Xr)
    Xs = ss.transform(Xs)
    ym = Yr.mean()
    Yr -= ym
    Ys -= ym

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        N_ = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("Likelihood"):
        var = ab.pos(tf.Variable(VARIANCE))
        lkhood = ab.normal(variance=var)

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.bayesmodel(X_, Y_, N_, LAYERS, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)

    with tf.Session(config=CONFIG):
        tf.global_variables_initializer().run()
        batches = ab.batch({X_: Xr, Y_: Yr}, N_, batch_size=BATCH_SIZE,
                           n_iter=NITERATIONS)
        for i, d in enumerate(batches):
            train.run(feed_dict=d)
            if i % 1000 == 0:
                l = loss.eval(feed_dict=d)
                print("Iteration {}, loss = {}".format(i, l))

        # Prediction
        Ey = np.hstack([Phi.eval(feed_dict={X_: Xs})
                        for _ in range(NPREDICTSAMPLES)])
        sigma2 = (1. * var).eval()

    # Score
    Eymean = Ey.mean(axis=1)
    Eyvar = Ey.var(axis=1) + sigma2
    r2 = r2_score(Ys.flatten(), Eymean)
    snlp = msll(Ys.flatten(), Eymean, Eyvar, Yr.flatten())
    smse = 1 - r2

    print("------------")
    print("r-square: {:.4g}, smse: {:.4g}, msll: {:.4g}."
          .format(r2, smse, snlp))


def msll(Y_true, Y_pred, V_pred, Y_train):
    mt, st = Y_train.mean(), Y_train.std()
    ll = norm.logpdf(Y_true, loc=Y_pred, scale=np.sqrt(V_pred))
    rand_ll = norm.logpdf(Y_true, loc=mt, scale=st)
    msll = - (ll - rand_ll).mean()
    return msll
