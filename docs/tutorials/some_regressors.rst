.. _tut_regress:

A regression masterclass with Aboleth
=====================================

In this tutorial we will show you how to build a variety of linear and non
linear regressors with the building blocks in Aboleth - and demonstrate how
easy it is once you have the basics down!

We'll start off with with some linear regressors, then we'll extend these 
models to various types of neural networks. We'll also talk about how we can
approximate other types of non linear regressors with Aboleth, such as support
vector regressors and Gaussian processes.

We fit all of the regressors to 100 noisy samples of the non-linear function:

.. math::

    y_i = \frac{\sin(x_i)}{x_i} + \epsilon_i,

where we draw :math:`\epsilon_i \sim \mathcal{N}(0, 0.05)`. The aim here is for
the regressors to reconstruct the latent function:

.. math::
    f = \frac{\sin(x)}{x},

from these noisy observations. This is what this data set looks like:

.. figure:: regression_figs/data.png

    The dataset used for fitting the regressors. There are 100 noisy training
    points (blue dots) that the algorithms get to see, and 1000 noise free
    points (blue line) that the algorithm has to predict.

We use :math:`R^2`, AKA the `coefficient of determination
<https://en.wikipedia.org/wiki/Coefficient_of_determination>`_ to evaluate how
good the estimate of the latent functions is. An :math:`R^2` of 1.0 is a
perfect fit, and 0 means no better than a Normal distribution fit only to the
targets, :math:`y_i`.

Note in the figure above that we have only generated training data for
:math:`x` from -10 to 10, but we evaluate the algorithms from -14 to 14! This
is because we want to see how well the algorithms extrapolate away from the
data, which is a very hard problem. We don't evaluate the :math:`R^2` in this
extrapolation region (it makes it harder to differentiate the performance of
the algorithms in the bounds of the training data), however, it is interesting
to see how the algorithms represent their uncertainty (or don't) in this
region.


Linear regression
-----------------

.. figure:: regression_figs/linear.png

    Simple linear regression, R-square :math:`\approx 0`.

.. figure:: regression_figs/ridge_linear.png

    Ridge linear regression, R-square :math:`\approx 0`.

.. figure:: regression_figs/bayesian_linear.png

    Bayesian linear regression, R-square :math:`\approx 0`.


Neural Networks
---------------

.. figure:: regression_figs/nnet.png

    Neural network with l2 regularization, R-square :math:`0.9903`.


.. figure:: regression_figs/nnet_dropout.png

    Neural network with dropout, R-square :math:`0.9865`.


.. figure:: regression_figs/nnet_bayesian.png

    Bayesian Neural network, R-square :math:`0.9668`.


.. figure:: regression_figs/nnet_bayesian_1000.png

    Bayesian Neural network with 1000 training points, R-square :math:`0.9983`.


Support Vector Regression
-------------------------

.. figure:: regression_figs/svr.png

    Support vector regression, R-square :math:`0.9948`.


.. figure:: regression_figs/svr_dropout.png

    Support vector regression with dropout, R-square :math:`0.9957`.


Gaussian process
----------------

.. figure:: regression_figs/gpr.png

    Gaussian process regression, RBF kernel, R-square = 0.9974.


.. figure:: regression_figs/gpr_varrbf.png

    Gaussian process regression, variational RBF kernel, R-square = 0.9941.

.. figure:: regression_figs/robust_gpr.png

    Robust Gaussian process, RBF kernel, R-square = 0.9984.

.. figure:: regression_figs/deep_gpr.png

    Deep Gaussian process regression, RBF kernel, R-square = 0.9939.


You can find the code used to generate this tutorial in the `demos
<https://github.com/data61/aboleth/blob/develop/demos/>`_ folder in Aboleth.
