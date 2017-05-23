"""Demo using aboleth for regression."""
import logging

import numpy as np
import bokeh.plotting as bk
import bokeh.palettes as bp
import tensorflow as tf
from sklearn.gaussian_process.kernels import Matern as kern
# from sklearn.gaussian_process.kernels import RBF as kern

import aboleth as ab
from aboleth.datasets import gp_draws

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Data settings
N = 2000
Ns = 400
kernel = kern(length_scale=1.)
true_noise = 0.1

# Model settings
n_samples = 5
n_pred_samples = 10  # This will give n_samples by n_pred_samples predictions
n_epochs = 100
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?

variance = tf.Variable(1.)
reg = 1.

lenscale1 = tf.Variable(1.)
layers = [
    # ab.random_arccosine(n_features=100, lenscale=ab.pos(lenscale1)),
    ab.random_fourier(n_features=50, kernel=ab.RBF(ab.pos(lenscale1))),
    ab.dense_var(output_dim=1, reg=reg, full=True)
]


def main():

    np.random.seed(100)
    n_iters = int(round(n_epochs * N / batch_size))
    print("Iterations = {}".format(n_iters))

    # Get training and testing data
    Xr, Yr, Xs, Ys = gp_draws(N, Ns, kern=kernel, noise=true_noise)

    # Prediction points
    Xq = np.linspace(-20, 20, Ns).astype(np.float32)[:, np.newaxis]
    Yq = np.linspace(-4, 4, Ns).astype(np.float32)[:, np.newaxis]

    # Image
    Xi, Yi = np.meshgrid(Xq, Yq)
    Xi = Xi.astype(np.float32).reshape(-1, 1)
    Yi = Yi.astype(np.float32).reshape(-1, 1)

    _, D = Xr.shape

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(tf.float32, shape=(None, D))
        Y_ = tf.placeholder(tf.float32, shape=(None, 1))

    with tf.name_scope("Likelihood"):
        lkhood = ab.normal(variance=ab.pos(variance))

    with tf.name_scope("Deepnet"):
        Net, loss = ab.deepnet(X_, Y_, N, layers, lkhood, n_samples)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)
        logprob = ab.log_prob(Y_, lkhood, Net)

    # Logging
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
    )

    with tf.train.MonitoredTrainingSession(
            config=config,
            save_summaries_steps=None,
            save_checkpoint_secs=None,
            hooks=[log]
    ) as sess:
        for d in ab.batch({X_: Xr, Y_: Yr}, batch_size, n_iters):
            if sess.should_stop():
                break
            sess.run(train, feed_dict=d)

        # Prediction
        Ey = ab.predict_samples(Net, feed_dict={X_: Xq, Y_: np.zeros_like(Yq)},
                                n_groups=n_pred_samples, session=sess)
        logPY = ab.predict_expected(logprob, feed_dict={Y_: Yi, X_: Xi},
                                    n_groups=n_pred_samples, session=sess)

    Eymean = Ey.mean(axis=0)
    Py = np.exp(logPY.reshape(Ns, Ns))

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


if __name__ == "__main__":
    main()
