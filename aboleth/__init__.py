"""Package init."""
from .model import deepnet, elbo, bayesmodel
from .layer import activation, eye, fork, apply, cat, add, dense_var, \
    dense_map, randomFourier, Matern, RBF
from .likelihood import normal, bernoulli, binomial
from .util import batch, pos

__all__ = [
    'deepnet',
    'elbo',
    'bayesmodel',
    'activation',
    'eye',
    'cat',
    'add',
    'fork',
    'apply',
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
