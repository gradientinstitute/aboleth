"""Package init."""
from .version import __version__
from .model import elbo, log_prob
from .layer import (Activation, DropOut, MaxPool2D, Reshape, DenseVariational,
                    DenseMAP, InputLayer, EmbedVariational, RandomFourier,
                    RandomArcCosine)
from .ops import stack, concat, slicecat, add, mean_impute, gaussian_impute
from .kernels import RBF, Matern
from .likelihood import normal, bernoulli, binomial
from .distributions import (Normal, Gaussian, norm_prior, norm_posterior,
                            gaus_posterior)
from .util import (batch, pos, predict_expected, predict_samples,
                   batch_prediction)
from .random import set_hyperseed

__all__ = (
    '__version__',
    'elbo',
    'log_prob',
    'Activation',
    'DropOut',
    'MaxPool2D',
    'Reshape',
    'DenseVariational',
    'DenseMAP',
    'EmbedVariational',
    'RandomFourier',
    'RandomArcCosine',
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
    'InputLayer',
    'stack',
    'concat',
    'slicecat',
    'add',
    'mean_impute',
    'gaussian_impute',
    'RBF',
    'Matern'
)
