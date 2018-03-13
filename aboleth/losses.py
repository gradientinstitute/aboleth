"""Network loss functions."""
import tensorflow as tf


def elbo(log_likelihood, KL, N):
    r"""Build the evidence lower bound (ELBO) loss for a neural net.

    Parameters
    ----------
    log_likelihood : Tensor
        the log-likelihood Tensor that takes neural network(s) and targets as
        an input. We recommend using a ``tf.distributions`` object's
        ``log_prob()`` method to obtain this tensor.  The shape of this Tensor
        should be ``(n_samples, N, ...)``, where ``n_samples`` is the number of
        log-likelihood samples (defined by ab.InputLayer) and ``N`` is the
        number of observations (can be ``?`` if you are using a placeholder and
        mini-batching). These likelihoods can also be weighted to, for example,
        adjust for class imbalance etc. This weighting is left up to the user.
    KL : float, Tensor
        the Kullback Leibler divergence between the posterior and prior
        parameters of the model (:math:`\text{KL}[q\|p]`).
    N : int, Tensor
        the total size of the dataset (i.e. number of observations).

    Returns
    -------
    nelbo : Tensor
        the loss function of the Bayesian neural net (negative ELBO).

    Example
    -------
    This is how we would typically generate a likelihood for this objective,

    .. code-block:: python

        noise = tf.Variable(1.0)
        likelihood = tf.distributions.Normal(loc=NN, scale=ab.pos(noise))
        log_likelihood = likelihood.log_prob(Y)

    where ``NN`` is our neural network, and ``Y`` are our targets.

    """
    # Batch amplification factor
    B = N / tf.to_float(tf.shape(log_likelihood)[1])

    # averaging over samples
    n_samples = tf.to_float(tf.shape(log_likelihood)[0])

    # Just mean over samps for expected log-likelihood
    ELL = tf.reduce_sum(log_likelihood) / n_samples

    # negative ELBO is batch weighted ELL and KL
    nELBO = - B * ELL + KL

    return nELBO


def max_posterior(log_likelihood, regulariser):
    r"""Build maximum a-posteriori (MAP) loss for a neural net.

    Parameters
    ----------
    log_likelihood : Tensor
        the log-likelihood Tensor that takes neural network(s) and targets as
        an input. We recommend using a ``tf.distributions`` object's
        ``log_prob()`` method to obtain this tensor.  The shape of this Tensor
        should be ``(n_samples, N, ...)``, where ``n_samples`` is the number of
        log-likelihood samples (defined by ab.InputLayer) and ``N`` is the
        number of observations (can be ``?`` if you are using a placeholder and
        mini-batching). These likelihoods can also be weighted to, for example,
        adjust for class imbalance etc. This weighting is left up to the user.
    regulariser : float, Tensor
        the regulariser on the parameters of the model to penalise model
        complexity.

    Returns
    -------
    map : Tensor
        the loss function of the MAP neural net.

    Example
    -------
    This is how we would typically generate a likelihood for this objective,

    .. code-block:: python

        noise = tf.Variable(1.0)
        likelihood = tf.distributions.Normal(loc=NN, scale=ab.pos(noise))
        log_likelihood = likelihood.log_prob(Y)

    where ``NN`` is our neural network, and ``Y`` are our targets.

    """
    # Average likelihood for batch
    AVLL = tf.reduce_mean(log_likelihood)

    # MAP objective
    MAP = - AVLL + regulariser

    return MAP
