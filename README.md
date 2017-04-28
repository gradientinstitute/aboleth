Aboleth
=======

[![CircleCI](https://circleci.com/gh/determinant-io/aboleth/tree/develop.svg?style=svg&circle-token=f02db635cf3a7e998e17273c91f13ffae7dbf088)](https://circleci.com/gh/determinant-io/aboleth/tree/develop)

<img src="http://fc03.deviantart.net/fs71/i/2010/162/e/3/Aboleth__Sunken_Empires_by_butterfrog.jpg"
 alt="aboleth by Butterfrog" width=300>

A bare-bones tensorflow framework for supervised Bayesian deep learning with
the stochastic gradient variational Bayes (SGVB, Kingma and Welling 2014).


Dependencies
------------

Minimal:
- numpy
- scipy
- tensorflow

Demos:
- bokeh
- scikit-learn


Installation
------------

At the command line via pip in the project directory:

    $ pip install .

To install in develop mode with packages required for development:

    $ pip install -e .[dev]


Examples
--------

See the [demos](https://github.com/determinant-io/aboleth/tree/develop/demos)
folder for some examples of creating and training algorithms with Aboleth.


References
----------

Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In ICLR, 2014.
