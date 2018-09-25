#! /usr/bin/env python3
"""Sarcos regression demo with TensorBoard and Custom Estimators."""
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

NSAMPLES = 1  # Number of random samples to get from an Aboleth net
NFEATURES = 500  # Number of random features/bases to use in the approximation
NOISE = 1.0  # Initial estimate of the observation noise

# Random Fourier Features, this is setting up an anisotropic length scale, or
# one length scale per dimension
LENSCALE = np.ones((21, 1), dtype=np.float32)

# Learning and prediction settings
BATCH_SIZE = 200  # number of observations per mini batch
NEPOCHS = 200  # Number of times to iterate though the dataset
EPOCHS_PER_EVAL = 10  # Number of epochs between evals
NPREDICTSAMPLES = 50  # Number of prediction samples


def get_data():
    data = fetch_gpml_sarcos_data()
    Xr = data.train.data.astype(np.float32)
    Yr = data.train.targets.astype(np.float32)[:, np.newaxis]
    Xs = data.test.data.astype(np.float32)
    Ys = data.test.targets.astype(np.float32)[:, np.newaxis]
    N, D = Xr.shape

    # Scale and centre the data, as per the original experiment
    ss = StandardScaler()
    Xr = ss.fit_transform(Xr)
    Xs = ss.transform(Xs)
    ym = Yr.mean()
    Yr -= ym
    Ys -= ym

    # Make dictionary of the column features
    Xr_dict = {"col{}".format(i): Xr[:, i:i+1] for i in range(Xr.shape[1])}
    Xs_dict = {"col{}".format(i): Xs[:, i:i+1] for i in range(Xs.shape[1])}

    return Xr_dict, Yr, Xs_dict, Ys

def train_input_fn(Xr, Yr):
    def f():
        data_tr = tf.data.Dataset.from_tensor_slices((Xr, Yr)) \
            .repeat() \
            .shuffle(buffer_size=10000) \
            .batch(BATCH_SIZE)
        return data_tr
    return f

def test_input_fn(Xs, Ys):
    def f():
        data_ts = tf.data.Dataset.from_tensor_slices((Xs, Ys)) \
            .batch(BATCH_SIZE)
        return data_ts
    return f

def predict_input_fn(Xs):
    def f():
        data_pr = tf.data.Dataset.from_tensor_slices(Xs) \
            .batch(BATCH_SIZE)
        return data_pr
    return f

def r2_metric(labels, predictions):
    SST, update_op1 = tf.metrics.mean_squared_error(
        labels, tf.reduce_mean(labels, axis=0))
    SSE, update_op2 = tf.metrics.mean_squared_error(labels, predictions)
    return tf.subtract(1.0, tf.div(SSE, SST)), tf.group(update_op1, update_op2)

def my_model(features, labels, mode, params):

    N = params["N"]
    n_samples = NSAMPLES if mode == tf.estimator.ModeKeys.TRAIN \
        else NPREDICTSAMPLES

    X = tf.feature_column.input_layer(features, params['feature_columns'])

    kernel = ab.RBF(LENSCALE, learn_lenscale=True)
    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.RandomFourier(n_features=NFEATURES, kernel=kernel) >>
        ab.DenseVariational(output_dim=1, full=True, prior_std=1.0,
                            learn_prior=True)
    )

    phi, kl = net(X=X)
    std = ab.pos_variable(NOISE, name="noise")
    ll_f = tf.distributions.Normal(loc=phi, scale=std)
    predict_mean = ab.sample_mean(phi)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': predict_mean,
            'samples': phi
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ll = ll_f.log_prob(labels)
    loss = ab.elbo(ll, kl, N)
    tf.summary.scalar('loss', loss)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=predict_mean,
                                        name='mse_op')
    r2 = r2_metric(labels, predict_mean)
    metrics = {'mse': mse,
               'r2': r2}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main():
    """Run the demo."""
    # Training batches
    Xr, Yr, Xs, Ys = get_data()
    N = Yr.shape[0]

    my_feature_columns = []
    for key in Xr.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # # Build 2 hidden layer DNN with 10, 10 units respectively.
    estimator = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir="./sarcos/",
        params={"N": N,
                "feature_columns": my_feature_columns})

    input_fn = train_input_fn(Xr, Yr)
    eval_fn = test_input_fn(Xs, Ys)
    predict_fn = predict_input_fn(Xs)

    steps = EPOCHS_PER_EVAL * (N // BATCH_SIZE)

    for i in range(NEPOCHS // EPOCHS_PER_EVAL):
        # Train the Model.
        estimator.train(input_fn=input_fn, steps=steps)
        # Evaluate the model.
        eval_result = estimator.evaluate(input_fn=eval_fn)

        # Use the predict interface to do alternative R2 calculation
        d = estimator.predict(input_fn=predict_fn,
                              yield_single_examples=False)
        Yt = []
        print("\n\nEval Result: {}".format(eval_result))
        for di in d:
            Yt.append(di["predictions"])
        Yt = np.concatenate(Yt, axis=0)
        print("R2: {}\n\n".format(r2_score(Ys, Yt)))


if __name__ == "__main__":
    main()
