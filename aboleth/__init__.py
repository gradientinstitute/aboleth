"""Package init."""
from .version import __version__
from .model import (deepnet, featurenet, elbo, log_prob, average_log_prob,
                    predict)
from .layer import (activation, fork, dropout, dense_var, dense_map,
                    embedding_var, random_fourier, random_arccosine, Matern,
                    RBF)
from .likelihood import normal, bernoulli, binomial
from .util import batch, pos

__all__ = [
    '__version__',
    'deepnet',
    'featurenet',
    'elbo',
    'log_prob',
    'average_log_prob',
    'predict',
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
    'pos'
]
