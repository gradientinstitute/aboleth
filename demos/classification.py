#! /usr/bin/env python3
import tensorflow as tf
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import aboleth as ab


FOLDS = 5
RSEED = 100

# Optimization
NITER = 20000
BSIZE = 10
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU
PSAMPLES = 20

# Network structure
layers = [
    ab.dense_var(output_dim=20, full=True),
    ab.activation(h=tf.nn.relu),
    ab.dense_var(output_dim=1, reg=0.1, full=True),
    ab.activation(h=tf.nn.sigmoid)
]


def main():

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)[:, np.newaxis]
    X = StandardScaler().fit_transform(X).astype(np.float32)
    N, D = X.shape

    # Benchmark classifier
    bcl = RandomForestClassifier()

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        N_ = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("Likelihood"):
        lkhood = ab.bernoulli()

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.deepnet(X_, Y_, N_, layers, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=RSEED)

    # Launch the graph.
    acc, acc_o, ll, ll_o = [], [], [], []
    init = tf.global_variables_initializer()
    for k, (r_ind, s_ind) in enumerate(kfold.split(X)):
        Xr, Yr = X[r_ind], y[r_ind]
        Xs, Ys = X[s_ind], y[s_ind]

        with tf.Session(config=CONFIG) as sess:
            sess.run(init)

            # Fit the network.
            batches = ab.batch(
                {X_: Xr, Y_: Yr},
                N_,
                batch_size=BSIZE,
                n_iter=NITER,
                random_state=RSEED
            )
            for i, data in enumerate(batches):
                sess.run(train, feed_dict=data)
                if i % 1000 == 0:
                    loss_val = loss.eval(feed_dict=data)
                    print("Iteration {}, loss = {}".format(i, loss_val))

            # Predict
            Eys = [Phi[0].eval(feed_dict={X_: Xs}) for _ in range(PSAMPLES)]
            Ey = np.hstack(Eys).mean(axis=1)

        print("Fold {}:".format(k))
        Ep = np.vstack((1 - Ey, Ey)).T
        print_k_result(Ys, Ep, ll, acc, "DGP")

        bcl.fit(Xr, Yr.flatten())
        Ep_o = bcl.predict_proba(Xs)
        print_k_result(Ys, Ep_o, ll_o, acc_o, "RF")
        print("-----")

    print_final_result(acc, ll, "DGP")
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
