#! /usr/bin/env python3
import tensorflow as tf
import numpy as np
import numpy.ma as ma

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import aboleth as ab
from aboleth.likelihoods import Bernoulli


FOLDS = 5
RSEED = 100
FRAC_MISSING = 0.2
ab.set_hyperseed(RSEED)

# Optimization
NITER = 20000
BSIZE = 10
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
LSAMPLES = 10
PSAMPLES = 5  # This will give LSAMPLES * PSAMPLES predictions
REG = 0.1

# Network structure
datanet = ab.InputLayer(name='X_nan', n_samples=LSAMPLES)
masknet = ab.InputLayer(name='M')

net = ab.Stack(
    ab.MeanImpute(datanet, masknet),
    ab.DropOut(0.95),
    ab.DenseMAP(output_dim=64, l1_reg=0., l2_reg=REG),
    ab.Activation(h=tf.nn.relu),
    ab.DropOut(0.5),
    ab.DenseMAP(output_dim=64, l1_reg=0., l2_reg=REG),
    ab.Activation(h=tf.nn.relu),
    ab.DropOut(0.5),
    ab.DenseMAP(output_dim=1, l1_reg=0., l2_reg=REG),
    ab.Activation(h=tf.nn.sigmoid)
)


def main():

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)[:, np.newaxis]
    X = StandardScaler().fit_transform(X).astype(np.float32)
    N, D = X.shape

    # Add random missingness
    missing_mask = np.random.rand(N, D) < FRAC_MISSING
    X_corrupted = X.copy()
    X_corrupted[missing_mask] = 666.  # Might not be needed
    masked_data = ma.asarray(X_corrupted)
    masked_data[missing_mask] = ma.masked

    # Benchmark classifier
    bcl = RandomForestClassifier(random_state=RSEED)

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        N_ = tf.placeholder(dtype=tf.float32)
        M_ = tf.placeholder(dtype=np.bool, shape=(None, D))

    with tf.name_scope("Likelihood"):
        lkhood = Bernoulli()

    with tf.name_scope("Deepnet"):
        Phi, kl = net(X_nan=X_, M=M_)
        loss = ab.elbo(Phi, Y_, N_, kl, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=RSEED)

    # Launch the graph.
    acc, acc_o, ll, ll_o = [], [], [], []
    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG):

        for k, (r_ind, s_ind) in enumerate(kfold.split(X)):
            init.run()

            Xr, Yr, Mr = [masked_data.data[r_ind], y[r_ind],
                          masked_data.mask[r_ind]]

            Xs, Ys, Ms = [masked_data.data[s_ind], y[s_ind],
                          masked_data.mask[s_ind]]

            batches = ab.batch(
                {X_: Xr, Y_: Yr, M_: Mr},
                batch_size=BSIZE,
                n_iter=NITER,
                N_=N_)
            for i, data in enumerate(batches):
                train.run(feed_dict=data)
                if i % 1000 == 0:
                    loss_val = loss.eval(feed_dict=data)
                    print("Iteration {}, loss = {}".format(i, loss_val))

            # Predict
            Ey = ab.predict_expected(Phi, {X_: Xs, M_: Ms}, PSAMPLES)

            print("Fold {}:".format(k))
            Ep = np.hstack((1. - Ey, Ey))

            print_k_result(Ys, Ep, ll, acc, "BNN")

            bcl.fit(Xr, Yr.flatten())
            Ep_o = bcl.predict_proba(Xs)
            print_k_result(Ys, Ep_o, ll_o, acc_o, "RF")
            print("-----")

        print_final_result(acc, ll, "BNN")
        print_final_result(acc_o, ll_o, "RF")


def print_k_result(ys, Ep, ll, acc, name):
    acc.append(accuracy_score(ys, Ep.argmax(axis=1)))
    ll.append(log_loss(ys, Ep))
    print("{}: accuracy = {:.4g}, log-loss = {:.4g}"
          .format(name, acc[-1], ll[-1]))


def print_final_result(acc, ll, name):
    print("{} final: accuracy = {:.4g} ({:.4g}), log-loss = {:.4g} ({:.4g})"
          .format(name, np.mean(acc), np.std(acc), np.mean(ll), np.std(ll)))


if __name__ == "__main__":
    main()
