import tensorflow as tf
import numpy as np

import aboleth as ab


tf.logging.set_verbosity(tf.logging.INFO)

RSEED = 666
ab.set_hyperseed(RSEED)

# Optimization
N_EPOCHS = 50
BATCH_SIZE = 100
CONFIG = tf.ConfigProto(device_count={'GPU': 2})  # Use GPU ?

# Network config
N_SAMPLES = 5
LATENT_D = 2
N_HIDDEN = 500


def main():

    # Dataset
    mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(
        './mnist_demo', reshape=True)

    N, D = mnist_data.train.images.shape

    # Make batches
    X = tf.data.Dataset.from_tensor_slices(
        np.asarray(mnist_data.train.images, dtype=np.float32),
    ).repeat(N_EPOCHS).shuffle(N).batch(BATCH_SIZE) \
     .make_one_shot_iterator().get_next()

    # Place-holder for sampling
    n_samples_ = tf.placeholder_with_default(N_SAMPLES, shape=())

    # Make the Encoder
    h_enc = (
        ab.InputLayer(name='X', n_samples=n_samples_) >>
        ab.DenseMAP(output_dim=N_HIDDEN) >>
        ab.Activation(tf.tanh)
    )
    u_enc = h_enc >> ab.DenseMAP(output_dim=LATENT_D)
    s_enc = h_enc >> ab.DenseMAP(output_dim=LATENT_D)

    # Make the decoder
    h_dec = u_enc >> ab.DenseMAP(output_dim=N_HIDDEN) >> ab.Activation(tf.tanh)
    u_dec = h_dec >> ab.DenseMAP(output_dim=D) >> ab.Activation(tf.nn.sigmoid)
    s_dec = h_dec >> ab.DenseMAP(output_dim=D)

    # Make the prior

    # Make the posterior

    # Make the likelihood

