#! /usr/bin/env python3
"""Sarcos regression demo with TensorBoard."""
import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import aboleth as ab
from aboleth.datasets import fetch_gpml_sarcos_data


# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

NSAMPLES = 10  # Number of random samples to get from an Aboleth net
NFEATURES = 500  # Number of random features/bases to use in the approximation
NOISE = 3.0  # Initial estimate of the observation noise

# Random Fourier Features, this is setting up an anisotropic length scale, or
# one length scale per dimension
LENSCALE = 5 * np.ones((21, 1), dtype=np.float32)
KERNEL = ab.RBF(LENSCALE, learn_lenscale=True)

# Variational Fourier Features -- length-scale setting here is the "prior"
# LENSCALE = 10.
# KERNEL = ab.RBFVariational(lenscale=LENSCALE, lenscale_posterior=LENSCALE)

# Build the approximate GP
n_samples_ = tf.placeholder_with_default(NSAMPLES, [])
net = ab.stack(
    ab.InputLayer(name='X', n_samples=n_samples_),
    ab.RandomFourier(n_features=NFEATURES, kernel=KERNEL),
    ab.DenseVariational(output_dim=1, full=True)
)

# Learning and prediction settings
BATCH_SIZE = 200  # number of observations per mini batch
NEPOCHS = 200  # Number of times to iterate though the dataset
NPREDICTSAMPLES = 20  # Number of prediction samples

CONFIG = tf.ConfigProto(device_count={'GPU': 1})  # Use GPU ?


def main():
    """Run the demo."""
    data = fetch_gpml_sarcos_data()
    Xr = data.train.data.astype(np.float32)
    Yr = data.train.targets.astype(np.float32)[:, np.newaxis]
    Xs = data.test.data.astype(np.float32)
    Ys = data.test.targets.astype(np.float32)[:, np.newaxis]
    N, D = Xr.shape

    print("Iterations: {}".format(int(round(N * NEPOCHS / BATCH_SIZE))))

    # Scale and centre the data, as per the original experiment
    ss = StandardScaler()
    Xr = ss.fit_transform(Xr)
    Xs = ss.transform(Xs)
    ym = Yr.mean()
    Yr -= ym
    Ys -= ym

    # Training batches
    data_tr = tf.data.Dataset.from_tensor_slices({'X': Xr, 'Y': Yr}) \
        .shuffle(buffer_size=1000) \
        .batch(BATCH_SIZE)

    # Testing iterators
    data_ts = tf.data.Dataset.from_tensors({'X': Xs, 'Y': Ys}).repeat()

    with tf.name_scope("DataIterators"):
        iterator = tf.data.Iterator.from_structure(data_tr.output_types,
                                                   data_tr.output_shapes)
        data = iterator.get_next()
        training_init = iterator.make_initializer(data_tr)
        testing_init = iterator.make_initializer(data_ts)

    with tf.name_scope("Deepnet"):
        phi, kl = net(X=data['X'])
        std = ab.pos_variable(NOISE, name="noise")
        ll_f = tf.distributions.Normal(loc=phi, scale=std)
        ll = ll_f.log_prob(data['Y'])
        loss = ab.elbo(ll, kl, N)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope("Test"):
        Ey = ab.sample_mean(phi)

    # Logging
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
    )

    with tf.train.MonitoredTrainingSession(
            config=CONFIG,
            scaffold=tf.train.Scaffold(local_init_op=training_init),
            checkpoint_dir="./sarcos/",
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[log]
    ) as sess:
        for i in range(NEPOCHS):

            # Train for one epoch
            sess.run(training_init)
            try:
                while not sess.should_stop():
                    _, g = sess.run([train, global_step])
            except tf.errors.OutOfRangeError:
                pass

            # Init testing and assess and log R-square score on test set
            sess.run(testing_init)
            Eymean = sess.run(Ey, feed_dict={n_samples_: NPREDICTSAMPLES})
            r2 = r2_score(Ys, Eymean)
            print("Training epoch {}, r-square = {}".format(i, r2))
            rsquare_summary(r2, sess, g)

    print("------------")
    print("r-square: {:.4f}, smse: {:.4f}".format(r2, 1 - r2))


def rsquare_summary(r2, session, step=None):
    """Get a summary writer for R-square."""
    summary_writer = session._hooks[1]._summary_writer
    sum_val = tf.Summary.Value(tag='r-square', simple_value=r2)
    score_sum = tf.Summary(value=[sum_val])
    summary_writer.add_summary(score_sum, step)


if __name__ == "__main__":
    main()
