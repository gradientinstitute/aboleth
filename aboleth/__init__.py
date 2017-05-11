"""Package init."""
from .version import __version__
from .model import deepnet, featurenet, elbo, log_prob
from .layer import (activation, fork, dropout, dense_var, dense_map,
                    embedding_var, random_fourier, random_arccosine, Matern,
                    RBF)
from .likelihood import normal, bernoulli, binomial
from .util import (batch, pos, predict_expected, predict_samples,
                   batch_prediction)

__all__ = [
    '__version__',
    'deepnet',
    'featurenet',
    'elbo',
    'log_prob',
    'activation',
    'fork',
    'dropout',
    'dense_var',
    'dense_map',
    'embedding_var',
    'random_fourier',
    'random_arccosine',
    'RBF',
    'Matern',
    'normal',
    'bernoulli',
    'binomial',
    'batch',
    'pos',
    'predict_expected',
    'predict_samples',
    'batch_prediction'
]
