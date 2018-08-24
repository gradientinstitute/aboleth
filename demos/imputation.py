#! /usr/bin/env python3
"""This script demonstrates Aboleth's imputation layers."""
import logging

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

import aboleth as ab

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

RSEED = 666
ab.set_hyperseed(RSEED)
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?

FRAC_TEST = 0.1  # Fraction of data to use for hold-out testing
FRAC_MISSING = 0.2  # Fraction of data that is missing
MISSING_VAL = -666  # Value to indicate missingness
NCLASSES = 7  # Number of target classes

# Imputation method CHANGE THESE
# METHOD = None
# METHOD = "LearnedNormalImpute"
# METHOD = "FixedNormalImpute"
# METHOD = "FixedScalarImpute"
# METHOD = "LearnedScalarImpute"
METHOD = "MeanImpute"

# Optimization
NEPOCHS = 5  # Number of times to see the data in training
BSIZE = 100  # Mini batch size
LSAMPLES = 3  # Number of samples for training
PSAMPLES = 50  # Number of predictions samples


def main():
    """Run the imputation demo."""
    # Fetch data, one-hot targets and standardise data
    data = fetch_covtype()
    Xo = data.data[:, :10]
    Xc = data.data[:, 10:]
    Y = (data.target - 1)
    Xo[:, :10] = StandardScaler().fit_transform(Xo[:, :10])

    # Network construction
    n_samples_ = tf.placeholder_with_default(LSAMPLES, [])
    data_input = ab.InputLayer(name='Xo', n_samples=n_samples_)  # Data input

    # Run this with imputation
    if METHOD is not None:
        print("Imputation method {}.".format(METHOD))

        # Fake some missing data
        rnd = np.random.RandomState(RSEED)
        mask = rnd.rand(*Xo.shape) < FRAC_MISSING
        Xo[mask] = MISSING_VAL

        # Use Aboleth to imputate
        mask_input = ab.MaskInputLayer(name='M')  # Missing data mask input
        if METHOD == "LearnedNormalImpute":
            input_layer = ab.LearnedNormalImpute(data_input, mask_input)
        elif METHOD == "LearnedScalarImpute":
            input_layer = ab.LearnedScalarImpute(data_input, mask_input)
        elif METHOD == "FixedNormalImpute":
            xm = np.ma.array(Xo, mask=mask)
            mean = np.ma.mean(xm, axis=0).data.astype(np.float32)
            std = np.ma.std(xm, axis=0).data.astype(np.float32)
            input_layer = ab.FixedNormalImpute(data_input, mask_input, mean,
                                               std)
        elif METHOD == "FixedScalarImpute":
            xm = np.ma.array(Xo, mask=mask)
            mean = np.ma.mean(xm, axis=0).data.astype(np.float32)
            input_layer = ab.FixedScalarImpute(data_input, mask_input, mean)
        elif METHOD == "MeanImpute":
            input_layer = ab.MeanImpute(data_input, mask_input)

        else:
            raise ValueError("Invalid method!")

    # Run this without imputation
    else:
        print("No missing data")
        input_layer = data_input
        mask = np.zeros_like(Xo)

    cat_layers = (
        ab.InputLayer(name='Xc', n_samples=n_samples_) >>
        ab.DenseVariational(output_dim=8)
    )

    con_layers = (
        input_layer >>
        ab.DenseVariational(output_dim=8)
    )

    net = (
        ab.Concat(cat_layers, con_layers) >>
        ab.Activation(tf.nn.selu) >>
        ab.DenseVariational(output_dim=NCLASSES)
    )

    # Split the training and testing data
    Xo_tr, Xo_ts, Xc_tr, Xc_ts, Y_tr, Y_ts, M_tr, M_ts = train_test_split(
        Xo.astype(np.float32),
        Xc.astype(np.float32),
        Y.astype(np.int32),
        mask,
        test_size=FRAC_TEST,
        random_state=RSEED
    )
    N_tr, Do = Xo_tr.shape
    _, Dc = Xc_tr.shape

    # Data
    with tf.name_scope("Input"):
        Xob, Xcb, Yb, Mb = batch_training(Xo_tr, Xc_tr, Y_tr, M_tr,
                                          n_epochs=NEPOCHS, batch_size=BSIZE)
        Xo_ = tf.placeholder_with_default(Xob, shape=(None, Do))
        Xc_ = tf.placeholder_with_default(Xcb, shape=(None, Dc))
        # Y_ has to be this dimension for compatability with Categorical
        Y_ = tf.placeholder_with_default(Yb, shape=(None,))
        if METHOD is not None:
            M_ = tf.placeholder_with_default(Mb, shape=(None, Do))

    with tf.name_scope("Deepnet"):
        if METHOD is not None:
            nn, kl = net(Xo=Xo_, Xc=Xc_, M=M_)
        else:
            nn, kl = net(Xo=Xo_, Xc=Xc_)

        lkhood = tf.distributions.Categorical(logits=nn)
        loss = ab.elbo(lkhood.log_prob(Y_), kl, N_tr)
        prob = ab.sample_mean(lkhood.probs)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging learning progress
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
    )

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=CONFIG,
            save_summaries_steps=None,
            save_checkpoint_secs=None,
            hooks=[log]
    ) as sess:
        try:
            while not sess.should_stop():
                sess.run(train)
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass

        # Prediction
        feed_dict = {Xo_: Xo_ts, Xc_: Xc_ts, Y_: [0], n_samples_: PSAMPLES}
        if METHOD is not None:
            feed_dict[M_] = M_ts

        p = sess.run(prob, feed_dict=feed_dict)

    # Get mean of samples for prediction, and max probability assignments
    Ey = p.argmax(axis=1)

    # Score results
    acc = accuracy_score(Y_ts, Ey)
    ll = log_loss(Y_ts, p)
    conf = confusion_matrix(Y_ts, Ey)
    print("Final scores: {}".format(METHOD))
    print("\tAccuracy = {}\n\tLog loss = {}\n\tConfusion =\n{}".
          format(acc, ll, conf))


def batch_training(Xo, Xc, Y, M, batch_size, n_epochs):
    """Batch training queue convenience function."""
    fd = {'Xo': Xo, 'Xc': Xc, 'Y': Y, 'M': M}
    data_tr = tf.data.Dataset.from_tensor_slices(fd) \
        .shuffle(buffer_size=1000, seed=RSEED) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['Xo'], data['Xc'], data['Y'], data['M']


if __name__ == "__main__":
    main()
