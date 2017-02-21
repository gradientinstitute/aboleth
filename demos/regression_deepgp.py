import numpy as np
import bokeh.plotting as bk
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF as skl_RBF

from deepnets import util, likelihood, model
from deepnets.layer import randomFourier, dense, RBF

# Data settings
N = 2000
Ns = 400
kernel = skl_RBF(length_scale=.5)
true_noise = 0.1

# Model settings
variance = 1.
n_loss_samples = 10
n_predict_samples = 10
n_iterations = 20000
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU

# Network structure
layers = [randomFourier(n_features=50, kernel=RBF()),
          dense(output_dim=5),
          randomFourier(n_features=50, kernel=RBF()),
          dense(output_dim=1)]


def main():

    np.random.seed(10)

    # Get training and testing data
    Xr, Yr, Xs, Ys = util.gp_draws(N, Ns, kern=kernel, noise=true_noise)

    # Prediction points
    Xq = np.linspace(-20, 20, Ns).astype(np.float32)[:, np.newaxis]

    _, D = Xr.shape

    # Data
    with tf.name_scope("Input"):
        X_ = tf.placeholder(dtype=tf.float32, shape=(None, D))
        Y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        N_ = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("Likelihood"):
        lkhood = likelihood.normal(variance=util.pos(
            tf.Variable(variance)))

    with tf.name_scope("Deepnet"):
        Phi, KL = model.deepnet(X_, layers)

    with tf.name_scope("Loss"):
        loss = model.loss(Phi, Y_, N_, KL, lkhood, n_loss_samples)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        batches = util.batch({X_: Xr, Y_: Yr}, N_, batch_size=batch_size,
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

    # Plot
    f = bk.figure(tools='pan,box_zoom,reset', sizing_mode='stretch_both')
    f.circle(Xr.flatten(), Yr.flatten(), fill_color='blue', alpha=0.2,
            legend='Training')
    f.line(Xs.flatten(), Ys.flatten(), line_color='black', legend='Truth')
    for y in Ey.T:
        f.line(Xq.flatten(), y, line_color='red', alpha=0.2, legend='Samples')
    f.line(Xq.flatten(), Eymean.flatten(), line_color='green', legend='Mean')
    bk.show(f)


if __name__ == "__main__":
    main()
