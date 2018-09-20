#! /usr/bin/env python3
"""This script demonstrates an alternative way of making a Bayesian Neural Net.

This is based on Yarin Gal's work on interpreting dropout networks as a special
case of Bayesian neural nets, see http://mlg.eng.cam.ac.uk/yarin/blog_2248.html
"""
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
ab.set_hyperseed(RSEED)

# Optimization
NITER = 20000  # Training iterations per fold
BSIZE = 10  # mini-batch size
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
LSAMPLES = 1  # We're only using 1 dropout "sample" for learning to be more
# like a MAP network
PSAMPLES = 50  # Number of samples for prediction
REG = 0.001  # weight regularizer

# Network structure
n_samples_ = tf.placeholder_with_default(LSAMPLES, [])
net = ab.stack(
    ab.InputLayer(name='X', n_samples=n_samples_),
    ab.DropOut(0.95, alpha=True),
    ab.Dense(output_dim=128, l2_reg=REG, init_fn="autonorm"),
    ab.Activation(h=tf.nn.selu),
    ab.DropOut(0.9, alpha=True),
    ab.Dense(output_dim=64, l2_reg=REG, init_fn="autonorm"),
    ab.Activation(h=tf.nn.selu),
    ab.DropOut(0.9, alpha=True),
    ab.Dense(output_dim=32, l2_reg=REG, init_fn="autonorm"),
    ab.Activation(h=tf.nn.selu),
    ab.DropOut(0.9, alpha=True),
    ab.Dense(output_dim=1, l2_reg=REG, init_fn="autonorm"),
)


def main():
    """Run the demo."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int32)[:, np.newaxis]
    X = StandardScaler().fit_transform(X).astype(np.float32)
    N, D = X.shape

    # Benchmark classifier
    bcl = RandomForestClassifier(random_state=RSEED)

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    with tf.name_scope("Deepnet"):
        nn, reg = net(X=X_)
        lkhood = tf.distributions.Bernoulli(logits=nn)
        loss = ab.max_posterior(lkhood.log_prob(Y_), reg)
        prob = ab.sample_mean(lkhood.probs)

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

            Xr, Yr = X[r_ind], y[r_ind]
            Xs, Ys = X[s_ind], y[s_ind]

            batches = ab.batch(
                {X_: Xr, Y_: Yr},
                batch_size=BSIZE,
                n_iter=NITER)
            for i, data in enumerate(batches):
                train.run(feed_dict=data)
                if i % 1000 == 0:
                    loss_val = loss.eval(feed_dict=data)
                    print("Iteration {}, loss = {}".format(i, loss_val))

            # Predict, NOTE: we use the mean of the likelihood to get the
            # probabilies
            ps = prob.eval(feed_dict={X_: Xs, n_samples_: PSAMPLES})

            print("Fold {}:".format(k))
            Ep = np.hstack((1. - ps, ps))

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
