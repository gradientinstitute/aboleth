"""Demo using aboleth for regression."""
import numpy as np
import bokeh.plotting as bk
import bokeh.palettes as bp
import tensorflow as tf
from sklearn.gaussian_process.kernels import Matern as kern
# from sklearn.gaussian_process.kernels import RBF as kern

import aboleth as ab
from aboleth.datasets import gp_draws


# Data settings
N = 1000
Ns = 400
kernel = kern(length_scale=1.)
true_noise = 0.1

# Model settings
n_samples = 10
n_pred_samples = 100
n_iterations = 30000
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU

# lenscale = tf.Variable(1.)
lenscale = 1.
variance = tf.Variable(1.)
# variance = 0.01

layers = [
    ab.randomFourier(n_features=20, kernel=ab.RBF(ab.pos(lenscale))),
    ab.dense_var(output_dim=5, reg=0.1, full=True),
    ab.randomFourier(n_features=10, kernel=ab.RBF(ab.pos(lenscale))),
    ab.dense_var(output_dim=1, reg=0.1, full=True)
]


def main():

    np.random.seed(100)

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
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        N_ = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("Likelihood"):
        lkhood = ab.normal(variance=ab.pos(variance))

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.deepnet(X_, Y_, N_, layers, lkhood, n_samples)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)

    with tf.name_scope("LogProb"):
        logprob = ab.log_prob(Y_, lkhood, Phi)

    with tf.Session(config=config):
        tf.global_variables_initializer().run()
        batches = ab.batch({X_: Xr, Y_: Yr}, N_, batch_size=batch_size,

                           n_iter=n_iterations)
        for i, d in enumerate(batches):
            train.run(feed_dict=d)
            if i % 100 == 0:
                l = loss.eval(feed_dict=d)
                print("Iteration {}, loss = {}".format(i, l))

        # Prediction
        Ey = [Phi[0].eval(feed_dict={X_: Xq}) for _ in range(n_pred_samples)]
        Eymean = sum(Ey) / n_pred_samples
        logPY = logprob.eval(feed_dict={Y_: Yi, X_: Xi})

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
