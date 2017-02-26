"""Demo using aboleth for regression."""
import click
import numpy as np
import bokeh.plotting as bk
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF as skl_RBF

import aboleth as ab
from aboleth.datasets import gp_draws


# Data settings
N = 200
Ns = 400
kernel = skl_RBF(length_scale=0.5)
true_noise = 0.1

# Model settings
variance = 1.
n_loss_samples = 10
n_predict_samples = 10
n_density_samples = 100
n_iterations = 10000
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU

layers = [
    ab.randomFourier(n_features=50, kernel=ab.RBF()),
    ab.dense_map(output_dim=5),
    ab.randomFourier(n_features=50, kernel=ab.RBF()),
    ab.dense_var(output_dim=1, reg=.1)
]


@click.command()
def main():

    np.random.seed(10)

    # Get training and testing data
    Xr, Yr, Xs, Ys = gp_draws(N, Ns, kern=kernel, noise=true_noise)

    # Prediction points
    Xq = np.linspace(-20, 20, Ns).astype(np.float32)[:, np.newaxis]
    Yq = np.linspace(-5, 5, Ns).astype(np.float32)[:, np.newaxis]

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
        lkhood = ab.normal(variance=ab.pos(
            tf.Variable(variance)))

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.bayesmodel(X_, Y_, N_, layers, lkhood)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)

    with tf.name_scope("Density"):
        density = ab.density(Phi,  Y_, lkhood, n_density_samples)

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
        Ey = np.hstack([Phi.eval(feed_dict={X_: Xq})
                        for _ in range(n_predict_samples)])
        Eymean = Ey.mean(axis=1)

        Py = density.eval(feed_dict={Y_: Yi, X_: Xi})
        Py = np.exp(Py.reshape(Ns, Ns))

    # Plot
    im_min = np.amin(Py)
    im_size = np.amax(Py) - im_min
    img = (Py - im_min) / im_size
    f = bk.figure(tools='pan,box_zoom,reset', sizing_mode='stretch_both')
    f.image(image=[img], x=-20., y=-5., dw=40., dh=10, palette="Inferno256",
            alpha=0.2)
    f.circle(Xr.flatten(), Yr.flatten(), fill_color='blue', legend='Training')
    f.line(Xs.flatten(), Ys.flatten(), line_color='blue', legend='Truth')
    for y in Ey.T:
        f.line(Xq.flatten(), y, line_color='black', legend='Samples')
    f.line(Xq.flatten(), Eymean.flatten(), line_color='green', legend='Mean')
    bk.show(f)
