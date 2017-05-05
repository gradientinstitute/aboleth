"""Package init."""
from .model import deepnet, featurenet, elbo, log_prob, average_log_prob
from .layer import (activation, fork, dropout, dense_var, dense_map,
                    embedding_var, randomFourier, randomArcCosine, Matern, RBF)
from .likelihood import normal, bernoulli, binomial
from .util import batch, pos

__all__ = [
    'deepnet',
    'featurenet',
    'elbo',
    'log_prob',
    'average_log_prob',
    'activation',
    'fork',
    'dropout',
    'dense_var',
    'dense_map',
    'randomFourier',
    'randomArcCosine',
    'RBF',
    'Matern',
    'normal',
    'bernoulli',
    'binomial',
    'batch',
    'pos'
]
