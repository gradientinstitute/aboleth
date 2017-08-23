=======
Aboleth
=======

.. |copy| unicode:: 0xA9

.. image:: https://circleci.com/gh/determinant-io/aboleth/tree/develop.svg?style=svg&circle-token=f02db635cf3a7e998e17273c91f13ffae7dbf088
    :target: https://circleci.com/gh/determinant-io/aboleth/tree/develop
    :alt: circleCI

.. figure:: http://fc03.deviantart.net/fs71/i/2010/162/e/3/Aboleth__Sunken_Empires_by_butterfrog.jpg
    :width: 50%
    :alt: (c) 2010-2017 butterfrog
    :align: center

    Aboleth |copy| 2010-2017 butterfrog.


A bare-bones TensorFlow framework for *Bayesian* deep learning and Gaussian
process approximation [1]_ with stochastic gradient variational Bayes [2]_.

Features
--------

Some of the features of Aboleth:

- Bayesian fully-connected, embedding and convolutional layers using SGVB [2]_
  for inference.
- Random Fourier and arc-cosine features for approximate GPs. Optional
  variational optimisation of these feature weights as per [1]_.
- Imputation layers with parameters that are learned as part of a model.
- Very flexible construction of networks, e.g. multiple inputs, ResNets etc.
- Optional maximum-likelihood type II inference for model parameters such as
  weight priors/regularizers and regression observation noise.


Why?
----

The purpose of Aboleth is to provide a set of high performance and light weight
components for building Bayesian neural nets and approximate (deep) Gaussian
process computational graphs. We aim for *minimal* abstraction over pure
TensorFlow, so you can still assign parts of the computational graph to
different hardware, use your own data feeds/queues, and manage your own
sessions etc.

Here is an example of building a simple BNN classifier with one hidden layer
and Normal prior/posterior distributions on the network weights:

.. code-block:: python

    import tensorflow as tf
    import aboleth as ab

    # Define the network, ">>" implements function composition,
    # the InputLayer gives a kwarg for this network, and
    # allows us to specify the number of samples for stochastic
    # gradient variational Bayes.
    layers = (
        ab.InputLayer(name="X", n_samples=5) >>
        ab.DenseVariational(output_dim=100) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseVariational(output_dim=1) >>
        ab.Activation(tf.nn.sigmoid) >>
    )

    X_ = tf.placeholder(tf.float, shape=(None, D))
    Y_ = tf.placeholder(tf.float, shape=(None, 1))

    # Define the likelihood model
    likelihood = ab.likelihoods.Bernoulli()

    # Build the network, net, and the parameter regularisation, kl
    net, kl = net(X=X_)

    # Build the final loss function to use with TensorFlow train
    loss = ab.elbo(net, Y_, N, kl, likelihood)

    # Now your TensorFlow training code here!
    ...

At the moment the focus of Aboleth is on supervised tasks, however this is
subject to change in subsequent releases if there is interest in this
capability.


Examples
--------

See the `demos <https://github.com/determinant-io/aboleth/tree/develop/demos>`_
folder for more examples of creating and training algorithms with Aboleth.


Installation
------------

For a minimal install, at the command line via pip in the project directory::

    $ pip install .

To install additional dependencies required by the `demos <https://github.com/determinant-io/aboleth/tree/develop/demos>`_::

    $ pip install .[demos]

To install in develop mode with packages required for development::

    $ pip install -e .[dev]


References
----------

.. [1] Cutajar, K. Bonilla, E. Michiardi, P. Filippone, M. Random Feature 
       Expansions for Deep Gaussian Processes. In ICML, 2017.
.. [2] Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In ICLR,
       2014.
