#! /usr/bin/env python3
"""This tutorial shows how to make a variety of regressors with Aboleth."""
import logging

import numpy as np
import bokeh.plotting as bk
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from sklearn.metrics import r2_score

import aboleth as ab

# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set up a consistent random seed in Aboleth so we get repeatable, but random
# results
RSEED = 666
ab.set_hyperseed(RSEED)

# Data settings
N = 100  # Number of training points to generate
Ns = 400  # Number of testing points to generate
true_noise = 0.05  # Add noise to the GP draws, to make things a little harder

# Model settings
n_samples = 5  # Number of random samples to get from an Aboleth net
n_pred_samples = 20  # This will give n_samples by n_pred_samples predictions
n_epochs = 4000  # how many times to see the data for training
batch_size = 10  # mini batch size for stochastric gradients
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU? 0 is no

model = "deep_gaussian_process"


# Models for regression
def linear(X, Y):
    """Linear regression with l2 regularization."""
    reg = .01  # Weight prior
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=1) >>
        ab.DenseMAP(output_dim=1, l2_reg=reg, l1_reg=0.)
    )

    phi, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=noise)
    loss = ab.max_posterior(lkhood, Y, reg)

    return phi, loss


def bayesian_linear(X, Y):
    """Bayesian Linear Regression."""
    reg = .01  # Initial weight prior std. dev, this is optimised later
    noise = tf.Variable(.5)  # Likelihood st. dev. initialisation, and learning

    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.DenseVariational(output_dim=1, std=reg, full=True)
    )

    phi, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, N, kl)

    return phi, loss


def nnet(X, Y):
    """Neural net with regularization."""
    reg = .01  # Weight prior
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=1) >>
        ab.DenseMAP(output_dim=40, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DenseMAP(output_dim=20, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DenseMAP(output_dim=10, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DenseMAP(output_dim=1, l2_reg=reg, l1_reg=0.)
    )

    phi, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=noise)
    loss = ab.max_posterior(lkhood, Y, reg)
    return phi, loss


def nnet_dropout(X, Y):
    """Neural net with dropout."""
    reg = 0.01  # Weight prior
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.DenseMAP(output_dim=40, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DropOut(keep_prob=0.9) >>
        ab.DenseMAP(output_dim=20, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DropOut(keep_prob=0.95) >>
        ab.DenseMAP(output_dim=10, l2_reg=reg, l1_reg=0.) >>
        ab.Activation(tf.tanh) >>
        ab.DenseMAP(output_dim=1, l2_reg=reg, l1_reg=0.)
    )

    phi, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=noise)
    loss = ab.max_posterior(lkhood, Y, reg)
    return phi, loss


def nnet_bayesian(X, Y):
    """Bayesian neural net."""
    reg = 0.1  # Weight prior
    noise = tf.Variable(0.01)  # Likelihood st. dev. initialisation

    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.DenseVariational(output_dim=20, std=reg) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseVariational(output_dim=7, std=reg) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseVariational(output_dim=5, std=reg) >>
        ab.Activation(tf.tanh) >>
        ab.DenseVariational(output_dim=1, std=reg)
    )

    phi, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, N, kl)
    return phi, loss


def svr(X, Y):
    """Support vector regressor."""
    reg = 0.1
    eps = 0.01
    lenscale = 1.

    kern = ab.RBF(lenscale=lenscale)  # keep the length scale positive
    net = (
        ab.InputLayer(name="X", n_samples=1) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        ab.DenseMAP(output_dim=1, l2_reg=reg, l1_reg=0.)
    )

    phi, reg = net(X=X)
    loss = tf.reduce_mean(tf.nn.relu(tf.abs(Y - phi - eps))) + reg
    return phi, loss


def gaussian_process(X, Y):
    """Gaussian Process Regression."""
    reg = 0.1  # Initial weight prior std. dev, this is optimised later
    noise = tf.Variable(.5)  # Likelihood st. dev. initialisation, and learning
    lenscale = tf.Variable(1.)  # learn the length scale
    kern = ab.RBF(lenscale=ab.pos(lenscale))  # keep the length scale positive

    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        ab.DenseVariational(output_dim=1, std=reg, full=True)
    )

    phi, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, N, kl)

    return phi, loss


def deep_gaussian_process(X, Y):
    """Deep Gaussian Process Regression."""
    reg = 0.1  # Initial weight prior std. dev, this is optimised later
    noise = tf.Variable(.01)  # Likelihood st. dev. initialisation
    lenscale = tf.Variable(1.)  # learn the length scale

    net = (
        ab.InputLayer(name="X", n_samples=n_samples) >>
        ab.RandomFourier(n_features=50, kernel=ab.RBF(ab.pos(lenscale))) >>
        ab.DenseVariational(output_dim=5, std=reg, full=True) >>
        ab.RandomFourier(n_features=10, kernel=ab.RBF(1.)) >>
        ab.DenseVariational(output_dim=1, std=reg, full=True)
    )

    phi, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, N, kl)

    return phi, loss


# Allow us to easily select models
model_dict = {
    "linear": linear,
    "bayesian_linear": bayesian_linear,
    "nnet": nnet,
    "nnet_dropout": nnet_dropout,
    "nnet_bayesian": nnet_bayesian,
    "svr": svr,
    "gaussian_process": gaussian_process,
    "deep_gaussian_process": deep_gaussian_process
}

# A list of models that have predictive distributions we can draw from
probabilistic = [
    "bayesian_linear",
    "nnet_dropout",
    "nnet_bayesian",
    "gaussian_process",
    "deep_gaussian_process",
]


def main():
    """Run the demo."""
    n_iters = int(round(n_epochs * N / batch_size))
    print("Iterations = {}".format(n_iters))

    # Latent function
    def f(X):
        Y = np.sin(X) / X
        return Y

    # Get training and testing data
    train_bounds = [-10, 10]
    pred_bounds = [-14, 14]
    rnd = np.random.RandomState(RSEED)
    Xr = rnd.rand(N, 1) * (train_bounds[1] - train_bounds[0]) + train_bounds[0]
    Xr = Xr.astype(np.float32)
    Yr = f(Xr) + rnd.randn(N, 1).astype(np.float32) * true_noise
    Xs = np.linspace(*pred_bounds, Ns, dtype=np.float32)[:, np.newaxis]
    Ys = f(Xs)
    test_bounds = np.logical_and(Xs[:, 0] > -10, Xs[:, 0] < 10)

    _, D = Xr.shape

    # Name the "data" parts of the graph
    with tf.name_scope("Input"):
        # This function will make a TensorFlow queue for shuffling and batching
        # the data, and will run through n_epochs of the data.
        Xb, Yb = batch_training(Xr, Yr, n_epochs=n_epochs,
                                batch_size=batch_size)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))

    # This is where we build the actual GP model
    with tf.name_scope("Model"):
        phi, loss = model_dict[model](X_, Y_)

    # Set up the trainig graph
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
            config=config,
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

        # Prediction, the [[None]] is to stop the default placeholder queue
        if model in probabilistic:
            Ey = ab.predict_samples(phi, feed_dict={X_: Xs, Y_: [[None]]},
                                    n_groups=n_pred_samples, session=sess)
            Eymean = Ey.mean(axis=0)  # Average samples
        else:
            Eymean = sess.run(phi, feed_dict={X_: Xs, Y_: [[None]]})

    # Score
    r2 = r2_score(Ys.flatten()[test_bounds], Eymean.flatten()[test_bounds])
    print("Score: {:.4f}".format(r2))

    # Plot
    f = bk.figure(plot_width=1000, plot_height=600, x_axis_label="x",
                  y_axis_label="y",
                  title="{}, R-square = {:.4f}".format(model, r2))
    if model in probabilistic:
        for y in Ey:
            f.line(Xs.flatten(), y.flatten(), line_color='green',
                   legend='sample predictions', alpha=0.1)
    f.circle(Xr.flatten(), Yr.flatten(), fill_color='blue',
             legend='training points')
    f.line(Xs.flatten(), Ys.flatten(), line_color='blue', line_width=3,
           legend='Truth')
    f.line(Xs.flatten(), Eymean.flatten(), line_color='green',
           legend='mean prediction', line_width=3)
    f.line([train_bounds[0], train_bounds[0]], [-.5, 1], line_color='gray',
           line_dash='dashed', line_width=3, legend="training domain")
    f.line([train_bounds[1], train_bounds[1]], [-.5, 1], line_color='gray',
           line_dash='dashed', line_width=3)
    bk.show(f)


def batch_training(X, Y, batch_size, n_epochs):
    """Batch training queue convenience function."""
    data_tr = Dataset.from_tensor_slices({'X': X, 'Y': Y}) \
        .shuffle(buffer_size=1000, seed=RSEED) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['X'], data['Y']


if __name__ == "__main__":
    main()
