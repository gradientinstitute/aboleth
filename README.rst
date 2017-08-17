=======
Aboleth
=======

.. image:: https://circleci.com/gh/determinant-io/aboleth/tree/develop.svg?style=svg&circle-token=f02db635cf3a7e998e17273c91f13ffae7dbf088
    :target: https://circleci.com/gh/determinant-io/aboleth/tree/develop
    :alt: circleCI

.. image:: http://fc03.deviantart.net/fs71/i/2010/162/e/3/Aboleth__Sunken_Empires_by_butterfrog.jpg
    :width: 50%
    :alt: aboleth
    :align: center


A bare-bones TensorFlow framework for supervised Bayesian deep learning and
Gaussian process [1]_ approximation with stochastic gradient variational Bayes
[2]_.


Installation
------------

For a minimal install, at the command line via pip in the project directory::

    $ pip install .

To install additional dependencies required by the `demos <https://github.com/determinant-io/aboleth/tree/develop/demos>`_::

    $ pip install .[demos]

To install in develop mode with packages required for development::

    $ pip install -e .[dev]


Examples
--------

See the `demos <https://github.com/determinant-io/aboleth/tree/develop/demos>`_
folder for some examples of creating and training algorithms with Aboleth.


References
----------

.. [1] Cutajar, K. Bonilla, E. Michiardi, P. Filippone, M. Random Feature 
       Expansions for Deep Gaussian Processes. In ICML, 2017.
.. [2] Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In ICLR,
       2014.
