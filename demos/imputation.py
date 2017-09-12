#! /usr/bin/env python3
"""This script demonstrates Aboleth's imputation layers."""
import logging

import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer

import aboleth as ab

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Make missing data?
USE_ABOLETH = True  # Use Aboleth to learn an imputation?

RSEED = 666
ab.set_hyperseed(RSEED)

FRAC_TEST = 0.1  # Fraction of data to use for hold-out testing
FRAC_MISSING = 0.2  # Fraction of data that is missing
MISSING_VAL = -666  # Value to indicate missingness

# Optimization
NEPOCHS = 2  # Number of times to see the data in training
BSIZE = 10  # Mini batch size
CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?
LSAMPLES = 5  # Number of samples the mode returns
PSAMPLES = 10  # This will give LSAMPLES * PSAMPLES predictions

NCLASSES = 7  # Number of target classes
NFEATURES = 100  # Number of random features to use

# Network construction
data_input = ab.InputLayer(name='X', n_samples=LSAMPLES)  # Data input
mask_input = ab.InputLayer(name='M')  # Missing data mask input

lenscale = ab.pos(tf.Variable(np.ones((54, 1), dtype=np.float32)))

layers = (
    ab.RandomArcCosine(n_features=NFEATURES, lenscale=lenscale) >>
    ab.DenseVariational(output_dim=NCLASSES)
)


def main():
    """Run the imputation demo."""
    # Fetch data, one-hot targets and standardise data
    data = fetch_covtype()
    X = data.data
    # Y = OneHotEncoder(sparse=False).fit_transform(data.target[:, np.newaxis])
    Y = (data.target - 1)[:, np.newaxis]
    X = StandardScaler().fit_transform(X)

    # Now fake some missing data with a mask
    rnd = np.random.RandomState(RSEED)
    mask = rnd.rand(*X.shape) < FRAC_MISSING
    X[mask] = MISSING_VAL

    # Use Aboleth to learn imputation statistics
    if USE_ABOLETH:
        net = ab.LearnedNormalImpute(data_input, mask_input) >> layers

    # Or just mean impute
    else:
        net = data_input >> layers
        imp = Imputer(missing_values=MISSING_VAL, strategy='mean')
        X = imp.fit_transform(X)

    # Split the training and testing data
    X_tr, X_ts, Y_tr, Y_ts, M_tr, M_ts = train_test_split(
        X.astype(np.float32),
        Y.astype(np.int32),
        mask,
        test_size=FRAC_TEST,
        random_state=RSEED
    )
    N_tr, D = X_tr.shape

    # Data
    with tf.name_scope("Input"):
        Xb, Yb, Mb = batch_training(X_tr, Y_tr, M_tr, n_epochs=NEPOCHS,
                                    batch_size=BSIZE)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        # Y_ = tf.placeholder_with_default(Yb, shape=(None, NCLASSES))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))
        M_ = tf.placeholder_with_default(Mb, shape=(None, D))

    with tf.name_scope("Deepnet"):
        # Conditionally assign a placeholder for masks if USE_ABOLETH
        nn, kl = net(X=X_, M=M_) if USE_ABOLETH else net(X=X_)
        lkhood = tf.distributions.Categorical(logits=nn)
        import IPython; IPython.embed(); exit()
        loss = ab.elbo(lkhood, Y_, N_tr, kl)

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
        feed_dict = {X_: X_ts, Y_: [[None]]}
        if USE_ABOLETH:
            feed_dict[M_] = M_ts

        Ep = ab.predict_samples(lkhood.mean(), feed_dict=feed_dict,
                                n_groups=PSAMPLES, session=sess)

    # Get mean of samples for prediction, and max probability assignments
    p = Ep.mean(axis=0)
    Ey = get_labels(p)

    # Score results
    acc = accuracy_score(Y_ts, Ey)
    ll = log_loss(Y_ts, p)
    conf = confusion_matrix(Y_ts.argmax(axis=1), Ey.argmax(axis=1))
    print("Final scores:")
    print("\tAccuracy = {}\n\tLog loss = {}\n\tConfusion =\n{}".
          format(acc, ll, conf))


def batch_training(X, Y, M, batch_size, n_epochs):
    """Batch training queue convenience function."""
    data_tr = Dataset.from_tensor_slices({'X': X, 'Y': Y, 'M': M}) \
        .shuffle(buffer_size=1000, seed=RSEED) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['X'], data['Y'], data['M']


def get_labels(p):
    """Get hard assignment labels from probabilities"""
    N = len(p)
    Ey = np.zeros_like(p)
    Ey[np.arange(N), p.argmax(axis=1)] = 1
    return Ey


if __name__ == "__main__":
    main()
