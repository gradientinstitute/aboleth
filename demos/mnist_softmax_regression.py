#! /usr/bin/env python3
import tensorflow as tf
import numpy as np

import aboleth as ab

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from sklearn.metrics import accuracy_score, log_loss


RSEED = 100
ab.set_hyperseed(RSEED)

# Optimization
NITER = 20000
BSIZE = 50
CONFIG = tf.ConfigProto(device_count={'GPU': 2})  # Use GPU ?
LSAMPLES = 5
PSAMPLES = 5  # This will give LSAMPLES * PSAMPLES predictions
REG = 0.1

# Network architecture
net = ab.stack(
    ab.InputLayer(name='X', n_samples=LSAMPLES),  # LSAMPLES, BATCH_SIZE, 28*28
    ab.Reshape(target_shape=(28, 28, 1)),  # LSAMPLES, BATCH_SIZE, 28, 28, 1

    ab.Conv2DVariational(filters=32,
                         kernel_size=(5, 5),
                         std=REG),  # LSAMPLES, BATCH_SIZE, 28, 28, 32
    ab.Activation(h=tf.nn.relu),
    ab.MaxPool2D(pool_size=(2, 2),
                 strides=(2, 2)),  # LSAMPLES, BATCH_SIZE, 14, 14, 32

    ab.Conv2DVariational(filters=64,
                         kernel_size=(5, 5),
                         std=REG),  # LSAMPLES, BATCH_SIZE, 14, 14, 64
    ab.Activation(h=tf.nn.relu),
    ab.MaxPool2D(pool_size=(2, 2),
                 strides=(2, 2)),  # LSAMPLES, BATCH_SIZE, 7, 7, 64

    ab.Reshape(target_shape=(7*7*64,)),  # LSAMPLES, BATCH_SIZE, 7*7*64

    ab.DenseVariational(output_dim=1024,
                        std=REG),  # LSAMPLES, BATCH_SIZE, 1024
    ab.Activation(h=tf.nn.relu),
    ab.DropOut(0.5),

    ab.DenseVariational(output_dim=10,
                        std=REG),  # LSAMPLES, BATCH_SIZE, 10
)


def main():
    """Run the demo."""
    mnist = mnist_data.read_data_sets('./mnist_demo', one_hot=True)
    X = mnist.train.images
    y = mnist.train.labels
    N, D = X.shape

    Xs = mnist.test.images
    ys = mnist.test.labels

    # Data
    with tf.name_scope("inputs"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))

    with tf.name_scope("model"):
        nn_logits, nn_reg = net(X=X_)
        llh = tf.distributions.Categorical(logits=nn_logits)
        loss = ab.elbo(llh, Y_, N, nn_reg)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train = optimizer.minimize(loss)

    # Launch the graph.
    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG):

        init.run()

        batches = ab.batch(
            {X_: X, Y_: y},
            batch_size=BSIZE,
            n_iter=NITER,
            N_=N_)
        for i, data in enumerate(batches):
            train.run(feed_dict=data)
            if not i % 10:
                loss_val = loss.eval(feed_dict=data)
                print("Iteration {}, loss = {}".format(i, loss_val))

        # Predict
        ps = ab.predict_expected(llh.probs, {X_: Xs}, PSAMPLES)

        Ep = np.hstack((1. - ps, ps))

        print("BayesianConvNet: accuracy = {:.4g}, log-loss = {:.4g}"
              .format(accuracy_score(ys, Ep.argmax(axis=1)),
                      log_loss(ys, Ep)))


if __name__ == "__main__":
    main()
