#! /usr/bin/env python3
import tensorflow as tf
import numpy as np

import aboleth as ab

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from sklearn.metrics import accuracy_score, log_loss


rseed = 100
ab.set_hyperseed(rseed)

# Optimization
n_epochs = 100
batch_size = 50
config = tf.ConfigProto(device_count={'GPU': 2})  # Use GPU ?
l_samples = 5
p_samples = 5  # This will give LSAMPLES * PSAMPLES predictions
reg = 0.1


# Network architecture
net = ab.stack(
    ab.InputLayer(name='X', n_samples=l_samples),  # LSAMPLES,BATCH_SIZE,28*28
    ab.Reshape(target_shape=(28, 28, 1)),  # LSAMPLES, BATCH_SIZE, 28, 28, 1

    ab.Conv2DMAP(filters=32,
                 kernel_size=(5, 5)),  # LSAMPLES, BATCH_SIZE, 28, 28, 32
    ab.Activation(h=tf.nn.relu),
    ab.MaxPool2D(pool_size=(2, 2),
                 strides=(2, 2)),  # LSAMPLES, BATCH_SIZE, 14, 14, 32

    ab.Conv2DMAP(filters=64,
                 kernel_size=(5, 5)),  # LSAMPLES, BATCH_SIZE, 14, 14, 64
    ab.Activation(h=tf.nn.relu),
    ab.MaxPool2D(pool_size=(2, 2),
                 strides=(2, 2)),  # LSAMPLES, BATCH_SIZE, 7, 7, 64

    ab.Reshape(target_shape=(7*7*64,)),  # LSAMPLES, BATCH_SIZE, 7*7*64

    ab.DenseVariational(output_dim=1024,
                        std=reg),  # LSAMPLES, BATCH_SIZE, 1024
    ab.Activation(h=tf.nn.relu),
    ab.DropOut(0.5),

    ab.DenseVariational(output_dim=10,
                        std=reg),  # LSAMPLES, BATCH_SIZE, 10
)


def main():

    # Dataset

    mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(
        './mnist_demo', reshape=True)

    N, D = mnist_data.train.images.shape

    X, Y = tf.data.Dataset.from_tensor_slices(
        (np.asarray(mnist_data.train.images, dtype=np.float32),
         np.asarray(mnist_data.train.labels, dtype=np.int64))
    ).repeat(n_epochs).shuffle(N).batch(batch_size) \
     .make_one_shot_iterator().get_next()

    # Model specification

    with tf.name_scope("model"):
        logits, reg = net(X=X)
        llh = tf.distributions.Categorical(logits=logits)
        loss = ab.elbo(llh, Y, N, reg)

    # Training graph building

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train = optimizer.minimize(loss)

    with tf.train.MonitoredTrainingSession(
        config=config
    ) as sess:

        while not sess.should_stop():

            sess.run(train)


if __name__ == "__main__":
    main()
