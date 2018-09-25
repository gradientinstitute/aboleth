#! /usr/bin/env python3
"""This tutorial shows how to make a variety of regressors with Aboleth."""
import logging

import numpy as np
import bokeh.plotting as bk
import tensorflow as tf
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
n_samples = 1  # Number of random samples to get from an Aboleth net
p_samples = 100  # Number of prediction samples
n_epochs = 4000  # how many times to see the data for training
batch_size = 10  # mini batch size for stochastric gradients
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU? 0 is no
n_samples_ = tf.placeholder_with_default(n_samples, [])

model = "nnet_ncp"


# Models for regression
def linear(X, Y):
    """Linear regression with l2 regularization."""
    lambda_ = 1e-4  # Weight regularizer
    noise = 1.  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X") >>
        ab.Dense(output_dim=1, l2_reg=lambda_)
    )

    Xw, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=Xw, scale=noise).log_prob(Y)
    loss = ab.max_posterior(lkhood, reg)
    # loss = 0.5 * tf.reduce_mean((Y - Xw)**2) + reg

    return Xw, loss


def bayesian_linear(X, Y):
    """Bayesian Linear Regression."""
    noise = ab.pos_variable(1.0)

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.DenseVariational(output_dim=1, full=True)
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.elbo(lkhood, kl, N)

    return f, loss


def nnet(X, Y):
    """Neural net with regularization."""
    lambda_ = 1e-4  # Weight regularizer
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=1) >>
        ab.Dense(output_dim=40, l2_reg=lambda_) >>
        ab.Activation(tf.tanh) >>
        ab.Dense(output_dim=20, l2_reg=lambda_) >>
        ab.Activation(tf.tanh) >>
        ab.Dense(output_dim=10, l2_reg=lambda_) >>
        ab.Activation(tf.tanh) >>
        ab.Dense(output_dim=1, l2_reg=lambda_)
    )

    f, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.max_posterior(lkhood, reg)
    return f, loss


def nnet_dropout(X, Y):
    """Neural net with dropout."""
    lambda_ = 1e-3  # Weight prior
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.Dense(output_dim=32, l2_reg=lambda_) >>
        ab.Activation(tf.nn.selu) >>
        ab.DropOut(keep_prob=0.9, independent=True) >>
        ab.Dense(output_dim=16, l2_reg=lambda_) >>
        ab.Activation(tf.nn.selu) >>
        ab.DropOut(keep_prob=0.95, independent=True) >>
        ab.Dense(output_dim=8, l2_reg=lambda_) >>
        ab.Activation(tf.nn.selu) >>
        ab.Dense(output_dim=1, l2_reg=lambda_)
    )

    f, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.max_posterior(lkhood, reg)
    return f, loss


def nnet_snn(X, Y):
    """Self normalising neural net."""
    noise = .5  # Likelihood st. dev.

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.Dense(output_dim=64, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.DropOut(keep_prob=0.9, independent=False, alpha=True) >>
        ab.Dense(output_dim=32, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.DropOut(keep_prob=0.9, independent=False, alpha=True) >>
        ab.Dense(output_dim=32, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.DropOut(keep_prob=0.9, independent=False, alpha=True) >>
        ab.Dense(output_dim=16, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.Dense(output_dim=1, init_fn="autonorm")
    )

    f, reg = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.max_posterior(lkhood, reg)
    return f, loss


def nnet_bayesian(X, Y):
    """Bayesian neural net."""
    noise = 0.01

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.DenseVariational(output_dim=5) >>
        ab.Activation(tf.nn.selu) >>
        ab.DenseVariational(output_dim=4) >>
        ab.Activation(tf.nn.selu) >>
        ab.DenseVariational(output_dim=3) >>
        ab.Activation(tf.nn.selu) >>
        ab.DenseVariational(output_dim=1)
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.elbo(lkhood, kl, N)
    return f, loss


def nnet_ncp(X, Y):
    """Noise contrastive prior network."""
    perturb_noise = 6.  # approximately std(X)
    noise = ab.pos_variable(0.1)

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.NCPContinuousPerturb(input_noise=perturb_noise) >>
        ab.Dense(output_dim=64, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.Dense(output_dim=32, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.Dense(output_dim=32, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.Dense(output_dim=16, init_fn="autonorm") >>
        ab.Activation(tf.nn.selu) >>
        ab.DenseNCP(output_dim_apply=1, prior_std="autonorm")
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.elbo(lkhood, kl, N)
    return f, loss


def svr(X, Y):
    """Support vector regressor, kind of..."""
    lambda_ = 1e-4
    eps = 0.01
    lenscale = 1.

    # Specify which kernel to approximate with the random Fourier features
    kern = ab.RBF(lenscale=lenscale)

    net = (
        # ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.InputLayer(name="X", n_samples=1) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        # ab.DropOut(keep_prob=0.9, independent=True) >>
        ab.Dense(output_dim=1, l2_reg=lambda_)
    )

    f, reg = net(X=X)
    loss = tf.reduce_mean(tf.nn.relu(tf.abs(Y - f) - eps)) + reg
    return f, loss


def gaussian_process(X, Y):
    """Gaussian Process Regression."""
    noise = ab.pos_variable(.5)
    kern = ab.RBF(learn_lenscale=False)  # learn lengthscale

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.RandomFourier(n_features=50, kernel=kern) >>
        ab.DenseVariational(output_dim=1, full=True, learn_prior=True)
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.elbo(lkhood, kl, N)

    return f, loss


def deep_gaussian_process(X, Y):
    """Deep Gaussian Process Regression."""
    noise = ab.pos_variable(.1)

    net = (
        ab.InputLayer(name="X", n_samples=n_samples_) >>
        ab.RandomFourier(n_features=20, kernel=ab.RBF(learn_lenscale=True)) >>
        ab.DenseVariational(output_dim=5, full=False) >>
        ab.RandomFourier(n_features=10, kernel=ab.RBF(1., seed=1)) >>
        ab.DenseVariational(output_dim=1, full=False, learn_prior=True)
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=noise).log_prob(Y)
    loss = ab.elbo(lkhood, kl, N)

    return f, loss


# Allow us to easily select models
model_dict = {
    "linear": linear,
    "bayesian_linear": bayesian_linear,
    "nnet": nnet,
    "nnet_dropout": nnet_dropout,
    "nnet_bayesian": nnet_bayesian,
    "svr": svr,
    "gaussian_process": gaussian_process,
    "deep_gaussian_process": deep_gaussian_process,
    "nnet_ncp": nnet_ncp,
    "nnet_snn": nnet_snn
}

# A list of models that have predictive distributions we can draw from
probabilistic = [
    "bayesian_linear",
    "nnet_dropout",
    "nnet_bayesian",
    "bayesian_svr",
    "gaussian_process",
    "deep_gaussian_process",
    "nnet_ncp",
    "nnet_snn"
    # "svr"
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
        f, loss = model_dict[model](X_, Y_)

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
            Ey = sess.run(f, feed_dict={X_: Xs, Y_: [[None]],
                                        n_samples_: p_samples})
            Eymean = Ey.mean(axis=0)  # Average samples
        else:
            Eymean = sess.run(f, feed_dict={X_: Xs, Y_: [[None]]})

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
    data_tr = tf.data.Dataset.from_tensor_slices({'X': X, 'Y': Y}) \
        .shuffle(buffer_size=1000, seed=RSEED) \
        .repeat(n_epochs) \
        .batch(batch_size)
    data = data_tr.make_one_shot_iterator().get_next()
    return data['X'], data['Y']


if __name__ == "__main__":
    main()
