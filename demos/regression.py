"""Demo using aboleth for regression."""
import numpy as np
import bokeh.plotting as bk
import bokeh.palettes as bp
import tensorflow as tf
from sklearn.gaussian_process.kernels import Matern as kern
from tensorflow.python.client import timeline

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
n_epochs = 10
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
    ab.dense_var(output_dim=10, reg=0.1, full=True),
    ab.randomFourier(n_features=20, kernel=ab.RBF(ab.pos(lenscale2))),
    ab.dense_var(output_dim=1, reg=0.1, full=True)
]


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
        Phi, loss = ab.deepnet(X_, Y_, N, layers, lkhood, n_samples)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        logprob = ab.log_prob(Y_, lkhood, Phi)

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
                if step % 100 == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    sess.run(train, options=run_options,
                             run_metadata=run_metadata)
                    l = loss.eval()
                    print("Iteration {}, loss = {}".format(step, l))
                else:
                    train.run()

                # Save a checkpoint periodically.
                if (step + 1) % 1000 == 0:
                    print('Saving')
                    saver.save(sess, "regression", global_step=step)
                step += 1

        except tf.errors.OutOfRangeError:
            print('Training Complete. Saving final model')
            saver.save(sess, "regression_final", global_step=step)
        finally:
            coord.request_stop()

        coord.join(threads)

        # Prediction
        Ey = [Phi[0].eval(feed_dict={X_: Xq}) for _ in range(n_pred_samples)]
        Eymean = sum(Ey) / n_pred_samples
        logPY = logprob.eval(feed_dict={Y_: Yi, X_: Xi})

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_cpu.json', 'w') as f:
            f.write(ctf)

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
