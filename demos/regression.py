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
N = 2000
Ns = 400
kernel = kern(length_scale=.5)
true_noise = 0.1

# Model settings
n_samples = 10
n_pred_samples = 100
# n_iterations = 30000
n_epochs = 30
batch_size = 10
config = tf.ConfigProto(device_count={'GPU': 0})  # Use CPU

lenscale1 = tf.Variable(1.)
# lenscale1 = 1.
# lenscale2 = tf.Variable(1.)
lenscale2 = 1.
variance = tf.Variable(1.)
# variance = 0.01

layers = [
    ab.randomFourier(n_features=20, kernel=ab.RBF(ab.pos(lenscale1))),
    ab.dense_var(output_dim=5, reg=0.1, full=True),
    ab.randomFourier(n_features=20, kernel=ab.RBF(ab.pos(lenscale2))),
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

        X_in = tf.constant(Xr)
        Y_in = tf.constant(Yr)
        N_train = tf.constant(Xr.shape[0])
        X_element, Y_element = tf.train.slice_input_producer(
            [X_in, Y_in], num_epochs=n_epochs, shuffle=True)
        X, Y = tf.train.batch([X_element, Y_element], batch_size=batch_size)

    with tf.name_scope("Likelihood"):
        lkhood = ab.normal(variance=ab.pos(variance))

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.deepnet(X, Y, N, layers, lkhood, n_samples)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        logprob = ab.log_prob(Y, lkhood, Phi)

    saver = tf.train.Saver()
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with tf.Session(config=config) as sess:
        init_op.run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                train.run()
                if step % 100 == 0:
                    l = loss.eval()
                    print("Iteration {}, loss = {}".format(step, l))

                # Save a checkpoint periodically.
                if (step + 1) % 1000 == 0:
                    print('Saving')
                    saver.save(sess, ".", global_step=step)
                step += 1

        except tf.errors.OutOfRangeError:
            print('Training Complete. Saving final model')
            saver.save(sess, ".", global_step=step)
        finally:
            coord.request_stop()

        coord.join(threads)

        # # Prediction
        # Ey = [Phi[0].eval(feed_dict={X_: Xq}) for _ in range(n_pred_samples)]
        # Eymean = sum(Ey) / n_pred_samples
        # logPY = logprob.eval(feed_dict={Y_: Yi, X_: Xi})

    # Py = np.exp(logPY.reshape(Ns, Ns))

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
