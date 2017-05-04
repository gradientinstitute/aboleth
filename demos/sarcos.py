"""Sarcos regression demo."""
from time import time

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import aboleth as ab
from aboleth.datasets import fetch_gpml_sarcos_data


VARIANCE = 10.0
# KERN = ab.RBF(
#     lenscale=tf.exp(tf.Variable(2. * np.ones((21, 1), dtype=np.float32)))
# )
# LAYERS = [
#     ab.randomFourier(n_features=1000, kernel=KERN),
#     ab.dense_var(output_dim=1, full=True)
# ]
# LAYERS = [
#     ab.dense_var(output_dim=100, full=True),
#     ab.activation(tf.tanh),
#     ab.dense_var(output_dim=100, full=True),
#     ab.activation(tf.tanh),
#     ab.dense_var(output_dim=1, full=True)
# ]
LAYERS = [
    ab.dense_var(output_dim=200, full=True),
    ab.activation(lambda x: tf.concat([tf.cos(x), tf.sin(x)], axis=2)),
    # ab.dense_var(output_dim=10, full=True),
    # ab.activation(lambda x: tf.concat([tf.cos(x), tf.sin(x)], axis=2)),
    ab.dense_var(output_dim=1, full=True)
]
NSAMPLES = 10
BATCH_SIZE = 10
NEPOCHS = 10
NPREDICTSAMPLES = 100

CONFIG = tf.ConfigProto(device_count={'GPU': 1})  # Use GPU ?


def main():

    data = fetch_gpml_sarcos_data()
    Xr = data.train.data.astype(np.float32)
    Yr = data.train.targets.astype(np.float32)[:, np.newaxis]
    Xs = data.test.data.astype(np.float32)
    Ys = data.test.targets.astype(np.float32)[:, np.newaxis]
    N, D = Xr.shape

    print("Iterations: {}".format(int(round(N * NEPOCHS / BATCH_SIZE))))

    # Scale and centre the data
    ss = StandardScaler()
    Xr = ss.fit_transform(Xr)
    Xs = ss.transform(Xs)
    ym = Yr.mean()
    Yr -= ym
    Ys -= ym

    # Data
    with tf.name_scope("Input"):
        Xb, Yb = batch_training(Xr, Yr, n_epochs=NEPOCHS,
                                batch_size=BATCH_SIZE)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))

    with tf.name_scope("Likelihood"):
        var = ab.pos(tf.Variable(VARIANCE))
        lkhood = ab.normal(variance=var)

    with tf.name_scope("Deepnet"):
        Phi, loss = ab.deepnet(X_, Y_, N, LAYERS, lkhood, n_samples=NSAMPLES)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train = optimizer.minimize(loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sv = tf.train.Supervisor(logdir="./sarcos/",
                             global_step=global_step,
                             save_summaries_secs=20,
                             save_model_secs=60)

    with sv.managed_session(config=CONFIG) as sess:

        try:
            local_step = 0
            time_inc = time()
            while not sv.should_stop():
                sess.run(train)
                if local_step % 1000 == 0:
                    delta = local_step / (time() - time_inc)
                    l = sess.run(loss)
                    print("Iteration {}, loss = {}, speed = {}, Global step {}"
                          .format(local_step, l, delta, sv.global_step.eval(sess)))
                local_step += 1
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass
        finally:
            # Prediction
            # Ey = np.hstack([Phi[0].eval(feed_dict={X_: Xs})
                            # for _ in range(NPREDICTSAMPLES)])
            sigma2 = (1. * var).eval()

    # Score
    Eymean = Ey.mean(axis=1)
    Eyvar = Ey.var(axis=1) + sigma2
    r2 = r2_score(Ys.flatten(), Eymean)
    snlp = msll(Ys.flatten(), Eymean, Eyvar, Yr.flatten())
    smse = 1 - r2

    print("------------")
    print("r-square: {:.4f}, smse: {:.4f}, msll: {:.4f}."
          .format(r2, smse, snlp))


def msll(Y_true, Y_pred, V_pred, Y_train):
    mt, st = Y_train.mean(), Y_train.std()
    ll = norm.logpdf(Y_true, loc=Y_pred, scale=np.sqrt(V_pred))
    rand_ll = norm.logpdf(Y_true, loc=mt, scale=st)
    msll = - (ll - rand_ll).mean()
    return msll


def batch_training(X, Y, batch_size, n_epochs, num_threads=4):
    samples = tf.train.slice_input_producer([X, Y], num_epochs=n_epochs,
                                            shuffle=True, capacity=100)
    X_batch, Y_batch = tf.train.batch(samples, batch_size=batch_size,
                                      num_threads=num_threads, capacity=100)
    return X_batch, Y_batch


if __name__ == "__main__":
    main()
