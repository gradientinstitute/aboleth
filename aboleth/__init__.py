"""Package init."""
from .version import __version__
from .losses import elbo
from .baselayers import stack
from .layers import (Activation, DropOut, MaxPool2D, Reshape, DenseVariational,
                     DenseMAP, InputLayer, EmbedVariational, RandomFourier,
                     RandomArcCosine)
from .hlayers import Concat, Sum, PerFeature
from .impute import (MeanImpute, FixedNormalImpute, VarScalarImpute,
                     VarNormalImpute)
from .kernels import RBF, Matern, RBFVariational
from .distributions import (norm_prior, norm_posterior, gaus_posterior)
from .util import (batch, pos, predict_expected, predict_samples,
                   batch_prediction)
from .random import set_hyperseed

__all__ = (
    '__version__',
    'elbo',
    'Activation',
    'DropOut',
    'MaxPool2D',
    'Reshape',
    'DenseVariational',
    'DenseMAP',
    'EmbedVariational',
    'RandomFourier',
    'RandomArcCosine',
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
    'Sum',
    'Concat',
    'PerFeature',
    'MeanImpute',
    'FixedNormalImpute',
    'VarScalarImpute',
    'VarNormalImpute',
    'RBF',
    'RBFVariational',
    'Matern'
)
