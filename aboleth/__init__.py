"""Package init."""
from .model import deepnet, elbo, log_prob, predict_nlp
from .layer import (eye, activation, fork, lmap, cat, add, dense_var,
                    dense_map, randomFourier, Matern, RBF)
from .likelihood import normal, bernoulli, binomial
from .util import batch, pos

__all__ = [
    'deepnet',
    'elbo',
    'log_prob',
    'predict_nlp',
    'eye',
    'activation',
    'fork',
    'lmap',
    'cat',
    'add',
    'dense_var',
    'dense_map',
    'randomFourier',
    'RBF',
    'Matern',
    'normal',
    'bernoulli',
    'binomial',
    'batch',
    'pos'
]
