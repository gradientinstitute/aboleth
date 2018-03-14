#! /usr/bin/env python3
"""This is a demonstration of how to use Keras with Aboleth."""
import logging

import numpy as np
import tensorflow as tf

import bokeh.plotting as bk
import bokeh.palettes as bp

# from sklearn.gaussian_process.kernels import Matern as kern
from sklearn.gaussian_process.kernels import RBF as kern

import aboleth as ab
from aboleth.datasets import gp_draws
from aboleth.layers import SampleLayer


# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set up a consistent random seed in Aboleth so we get repeatable, but random
# results
RSEED = 666
ab.set_hyperseed(RSEED)

# Data settings
N = 1000  # Number of training points to generate
Ns = 400  # Number of testing points to generate
kernel = kern(length_scale=0.5)  # Kernel to use for making a random GP draw
true_noise = 0.1  # Add noise to the GP draws, to make things a little harder

# Model settings
n_samples = 5  # Number of random samples to get from an Aboleth net
p_samples = 50  # Number of samples for prediction
n_epochs = 500  # how many times to see the data for training
batch_size = 10  # mini batch size for stochastric gradients
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU? 0 is no

# Model initialisation
noise = tf.Variable(1.)  # Likelihood st. dev. initialisation, and learning


class WrapperLayer(SampleLayer):

    def __init__(self, layer, *args, **kwargs):
        self.layer = layer(*args, **kwargs)

    def _build(self, X):
        """Build the graph of this layer."""
        Net = self.layer(X)
        # aggregate layer regularization terms
        KL = tf.reduce_sum(self.layer.losses)

        return Net, KL


n_samples_ = tf.placeholder(tf.int32)

l1_l2_reg = tf.keras.regularizers.l1_l2(l1=0., l2=0.)
net = (
   ab.InputLayer(name="X", n_samples=n_samples_) >>
   WrapperLayer(tf.keras.layers.Dense, units=64, activation='tanh',
                kernel_regularizer=l1_l2_reg, bias_regularizer=l1_l2_reg) >>
   ab.DropOut(keep_prob=.9) >>
   WrapperLayer(tf.keras.layers.Dense, units=32, activation='tanh',
                kernel_regularizer=l1_l2_reg, bias_regularizer=l1_l2_reg) >>
   ab.DropOut(keep_prob=.9) >>
   WrapperLayer(tf.keras.layers.Dense, units=1, kernel_regularizer=l1_l2_reg,
                bias_regularizer=l1_l2_reg)
)


def main():
    """Run the demo."""
    n_iters = int(round(n_epochs * N / batch_size))
    print("Iterations = {}".format(n_iters))

    # Get training and testing data
    Xr, Yr, Xs, Ys = gp_draws(N, Ns, kern=kernel, noise=true_noise)

    # Prediction points
    Xq = np.linspace(-20, 20, Ns).astype(np.float32)[:, np.newaxis]
    Yq = np.linspace(-4, 4, Ns).astype(np.float32)[:, np.newaxis]

    # Set up the probability image query points
    Xi, Yi = np.meshgrid(Xq, Yq)
    Xi = Xi.astype(np.float32).reshape(-1, 1)
    Yi = Yi.astype(np.float32).reshape(-1, 1)

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
    with tf.name_scope("Deepnet"):
        phi, reg = net(X=X_)
        ll = tf.distributions.Normal(loc=phi, scale=ab.pos(noise)).log_prob(Y_)
        loss = ab.max_posterior(ll, reg)

    # Set up the trainig graph
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # This is used for building the predictive density image
    with tf.name_scope("Predict"):
        logprob = tf.reduce_mean(ll, axis=0)

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
                sess.run(train, feed_dict={
                            n_samples_: n_samples,
                            # tf.keras.backend.learning_phase(): 1
                         })
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass

        # Prediction, the [[None]] is to stop the default placeholder queue
        # we keep learning phase flag on even in prediction phase to draw
        # samples from predictive distribution
        Ey = sess.run(phi,
                      feed_dict={X_: Xq, Y_: [[None]], n_samples_: p_samples})

        logPY = sess.run(logprob,
                         feed_dict={Y_: Yi, X_: Xi, n_samples_: p_samples})

    Eymean = Ey.mean(axis=0)  # Average samples to get mean predicted funtion
    Py = np.exp(logPY.reshape(Ns, Ns))  # Turn log-prob into prob

    # Plot
    im_min = np.amin(Py)
    im_size = np.amax(Py) - im_min
    img = (Py - im_min) / im_size
    f = bk.figure(tools='pan,box_zoom,reset', sizing_mode='stretch_both')
    f.image(image=[img], x=-20., y=-4., dw=40., dh=8,
            palette=bp.Plasma256)
    f.circle(Xr.flatten(), Yr.flatten(), fill_color='blue', legend='Training')
    f.line(Xs.flatten(), Ys.flatten(), line_color='blue', legend='Truth')
    for y in Ey:
        f.line(Xq.flatten(), y.flatten(), line_color='red', legend='Samples',
               alpha=0.2)
    f.line(Xq.flatten(), Eymean.flatten(), line_color='green', legend='Mean')
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
