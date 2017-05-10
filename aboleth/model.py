"""Network construction and evaluation."""
import tensorflow as tf

from aboleth.layer import compose_layers


#
# Graph Building -- Models and Optimisation
#


def deepnet(X, Y, N, layers, likelihood, n_samples=10, like_weights=None):
    """Make a supervised Bayesian deep net.

    Parameters
    ----------
    X : ndarray, Tensor
        the covariates of shape (N, dimensions).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    layers : sequence
        a list (or sequence) of layers defining the neural net. See also the
        ``layers`` module.
    likelihood : Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the ``likelihood`` module.
    n_samples : int
        the number of samples to use for evaluating the expected log-likelihood
        in the objective function. This replicates the whole net for each
        sample.
    like_weights : callable, ndarray, Tensor
        weights to apply to each sample in the expected log likelihood. This
        should be an array of shape (samples, 1) or can be called as
        ``like_weights(Y)`` and should return a (samples, 1) array.

    Returns
    -------
    Net : Tensor
        the neural net Tensor that can be used for prediction. This may be only
        using *one* sample from the posterior over the model parameters.
    loss : Tensor
        the loss function use to train the model.
    """
    Net, KL = _tile_compose(X, layers, n_samples)
    loss = elbo(Net, Y, N, KL, likelihood, like_weights)
    samplenet = tf.identity(Net[0], name="Net")
    return samplenet, loss


def featurenet(features, Y, N, layers, likelihood, n_samples=10,
               like_weights=None):
    """Make a supervised Bayesian deep net with multiple input nets.

    Parameters
    ----------
    features : list of (ndarray or Tensor, list) tuples
        list of (``X``, ``layers``) pairs so the net can have different input
        features of different type, e.g. categorical and continuous. The output
        of these layers are concatenated before feeding into the rest of the
        net.
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    layers : sequence
        a list (or sequence) of layers defining the neural net. See also the
        ``layers`` module. This is applies after the ``features`` layers.
    likelihood : Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the ``likelihood`` module.
    n_samples : int
        the number of samples to use for evaluating the expected log-likelihood
        in the objective function. This replicates the whole net for each
        sample.
    like_weights : callable, ndarray, Tensor
        weights to apply to each sample in the expected log likelihood. This
        should be an array of shape (samples, 1) or can be called as
        ``like_weights(Y)`` and should return a (samples, 1) array.

    Returns
    -------
    Net : Tensor
        the neural net Tensor that can be used for prediction. This may be only
        using *one* sample from the posterior over the model parameters.
    loss : Tensor
        the loss function use to train the model.
    """
    # Constuct all input nets and concatenate outputs
    Net, KLs = zip(*map(lambda f: _tile_compose(*f, n_samples), features))
    Net = tf.concat(Net, axis=-1)

    # Now construct the rest of the net and add all penalty terms
    Net, KL = compose_layers(Net, layers)
    KL += sum(KLs)
    loss = elbo(Net, Y, N, KL, likelihood, like_weights)
    samplenet = tf.identity(Net[0], name="Net")
    return samplenet, loss


def elbo(Net, Y, N, KL, likelihood, like_weights=None):
    """Build the evidence lower bound loss for a neural net.

    Parameters
    ----------
    Net : ndarray, Tensor
        the neural net featues of shape (n_samples, N, output_dimensions).
    Y : ndarray, Tensor
        the targets of shape (N, tasks).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).
    likelihood : Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the ``likelihood`` module.
    like_weights : callable, ndarray, Tensor
        weights to apply to each sample in the expected log likelihood. This
        should be an array of shape (samples, 1) or can be called as
        ``like_weights(Y)`` and should return a (samples, 1) array.

    Returns
    -------
    elbo : Tensor
        the loss function of the Bayesian neural net.
    """
    B = N / tf.to_float(tf.shape(Net)[1])  # Batch amplification factor
    n_samples = tf.to_float(tf.shape(Net)[0])

    # Just mean over samps for expected log-likelihood
    if like_weights is None:
        ELL = tf.reduce_sum(likelihood(Y, Net)) / n_samples
    elif callable(like_weights):
        ELL = tf.reduce_sum(likelihood(Y, Net) * like_weights(Y)) / n_samples
    else:
        ELL = tf.reduce_sum(likelihood(Y, Net) * like_weights) / n_samples

    l = - B * ELL + KL
    return tf.identity(l, name="loss")


#
# Graph Building -- Prediction and evaluation
#


def log_prob(Y, likelihood, Net):
    """Build the log probability density of the model for each observation.

    Parameters
    ----------
    Y: ndarray, Tensor
        the targets of shape (N, tasks).
    likelihood: Tensor
        the likelihood model to use on the output of the last layer of the
        neural net, see the ``likelihood`` module.
    Net: Tensor
        the neural net featues of shape (N, output_dimensions).

    Returns
    -------
    logp : Tensor
        a *sample* of the log probability of each ``Y`` under the model. This
        is of shape (N,).
    """
    log_prob = tf.identity(likelihood(Y, Net), name="log_prob")
    return log_prob


#
# Private module utils
#

def _tile_compose(X, layers, n_samples):
    """Tile X into seperate samples, then compose layers for each sample."""
    Net = tf.tile(tf.expand_dims(X, 0), [n_samples, 1, 1])  # (n_samples, N, D)
    Net, KL = compose_layers(Net, layers)
    return Net, KL
