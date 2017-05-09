"""Demo using aboleth for regression."""
from time import time

import numpy as np
import bokeh.plotting as bk
import bokeh.palettes as bp
import tensorflow as tf
from sklearn.gaussian_process.kernels import Matern as kern

# from sklearn.gaussian_process.kernels import RBF as kern

import aboleth as ab
from aboleth.datasets import gp_draws


# Data settings
N = 2000
Ns = 400
kernel = kern(length_scale=1.)
true_noise = 0.1

# Model settings
n_samples = 5
n_pred_samples = 100
n_epochs = 300
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?

variance = tf.Variable(1.)
reg = 1.

lenscale1 = tf.Variable(1.)
# lenscale1 = 1.
# lenscale2 = tf.Variable(1.)
lenscale2 = 1.
layers = [
    # ab.random_arccosine(n_features=100, lenscale=ab.pos(lenscale1)),
    ab.random_fourier(n_features=50, kernel=ab.RBF(ab.pos(lenscale1))),
    # ab.dense_var(output_dim=5, reg=reg, full=True),
    ab.dense_var(output_dim=1, reg=reg, full=True)
]
# layers = [
#     ab.dense_map(output_dim=200, l1_reg=0, l2_reg=reg),
#     ab.activation(tf.nn.relu),
#     ab.dropout(0.9),
#     ab.dense_map(output_dim=200, l1_reg=0, l2_reg=reg),
#     ab.activation(tf.nn.relu),
#     ab.dropout(0.9),
#     # ab.dense_map(output_dim=200, l1_reg=0, l2_reg=reg),
#     # ab.activation(tf.nn.relu),
#     # ab.dropout(0.9),
#     ab.dense_map(output_dim=1, l1_reg=0, l2_reg=reg),
# ]


def main():

    np.random.seed(100)
    print("Iterations = {}".format(int(round(n_epochs * N / batch_size))))

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

        Xb, Yb = batch_training(Xr, Yr, n_epochs=n_epochs,
                                batch_size=batch_size)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))

    with tf.name_scope("Likelihood"):
        lkhood = ab.normal(variance=ab.pos(variance))

    with tf.name_scope("Deepnet"):
        Net, loss = ab.deepnet(X_, Y_, N, layers, lkhood, n_samples)

    with tf.name_scope("Predict"):
        pred = ab.predict(Net)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        logprob = ab.log_prob(Y_, lkhood, Net)

    # saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config):
        init_op.run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            time_inc = time()
            while not coord.should_stop():
                train.run()
                if step % 500 == 0:
                    delta = step / (time() - time_inc)
                    l = loss.eval()
                    print("Iteration {}, loss = {}, speed = {}"
                          .format(step, l, delta))
                step += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads)

        # Prediction
        Ey = [pred.eval(feed_dict={X_: Xq}) for _ in range(n_pred_samples)]
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


def batch_training(X, Y, batch_size, n_epochs, num_threads=4):
    samples = tf.train.slice_input_producer([X, Y], num_epochs=n_epochs,
                                            shuffle=True, capacity=100)
    X_batch, Y_batch = tf.train.batch(samples, batch_size=batch_size,
                                      num_threads=num_threads, capacity=100)
    return X_batch, Y_batch


if __name__ == "__main__":
    main()
