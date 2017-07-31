"""Package init."""
from .version import __version__
from .model import elbo, log_prob
from .layer import (activation, dropout, dense_var, dense_map, input,
                    embed_var, random_fourier, random_arccosine, Matern, RBF)
from .ops import stack, concat, slicecat, add
from .likelihood import normal, bernoulli, binomial
from .distributions import (Normal, Gaussian, norm_prior, norm_posterior,
                            gaus_posterior)
from .util import (batch, pos, predict_expected, predict_samples,
                   batch_prediction)
from .random import set_hyperseed

__all__ = (
    '__version__',
    'deepnet',
    'featurenet',
    'elbo',
    'log_prob',
    'activation',
    'dropout',
    'dense_var',
    'dense_map',
    'embed_var',
    'random_fourier',
    'random_arccosine',
    'RBF',
    'Matern',
    'normal',
    'bernoulli',
    'binomial',
    'Normal',
    'Gaussian',
    'norm_prior',
    'norm_posterior',
    'gaus_posterior',
    'batch',
    'pos',
    'predict_expected',
    'predict_samples',
    'batch_prediction',
    'set_hyperseed',
    'input',
    'stack',
    'concat',
    'slicecat',
    'add'
)
