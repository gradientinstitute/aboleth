#! /usr/bin/env python3
import tensorflow as tf
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import deepnets as dn


FOLDS = 5
RSEED = 100

# Optimization
NITER = 30000
BSIZE = 10
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU


def main():

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)[:, np.newaxis]
    X = StandardScaler().fit_transform(X).astype(np.float32)
    N, D = X.shape

    # Benchmark classifier
    bcl = RandomForestClassifier()

    # Create NN
    like = dn.Bernoulli()
    dgp = dn.BayesNN(N=np.round(N * (FOLDS - 1) / FOLDS), likelihood=like)
    dgp.add(dn.Dense(input_dim=D, output_dim=20))
    dgp.add(dn.Activation(func=tf.nn.relu, output_dim=20))
    # dgp.add(dn.Dense(output_dim=10))
    # dgp.add(dn.Activation(func=tf.nn.relu, output_dim=10))
    dgp.add(dn.Dense(output_dim=1))
    dgp.add(dn.Activation(func=tf.nn.sigmoid, output_dim=1))

    X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    loss = dgp.loss(X_, y_)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=RSEED)

    # Launch the graph.
    acc, acc_o, ll, ll_o = [], [], [], []
    init = tf.global_variables_initializer()
    for k, (r_ind, s_ind) in enumerate(kfold.split(X)):
        Xr, yr = X[r_ind], y[r_ind]
        Xs, ys = X[s_ind], y[s_ind]

        with tf.Session(config=CONFIG) as sess:
            sess.run(init)

            # Fit the network.
            batches = dn.gen_batch(
                {X_: Xr, y_: yr},
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
            Ey = np.hstack(sess.run(dgp.predict(Xs))).mean(axis=1)

        print("Fold {}:".format(k))
        Ep = np.vstack((1 - Ey, Ey)).T
        print_k_result(ys, Ep, ll, acc, "DGP")

        bcl.fit(Xr, yr.flatten())
        Ep_o = bcl.predict_proba(Xs)
        print_k_result(ys, Ep_o, ll_o, acc_o, "RF")
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
