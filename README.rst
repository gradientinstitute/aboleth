=======
Aboleth
=======

.. |copy| unicode:: 0xA9

.. image:: https://circleci.com/gh/data61/aboleth/tree/develop.svg?style=svg&circle-token=f02db635cf3a7e998e17273c91f13ffae7dbf088
    :target: https://circleci.com/gh/data61/aboleth/tree/develop
    :alt: circleCI

.. image:: https://readthedocs.org/projects/aboleth/badge/?version=stable
    :target: http://aboleth.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

A bare-bones `TensorFlow <https://www.tensorflow.org/>`_ framework for
*Bayesian* deep learning and Gaussian process approximation [1]_ with
stochastic gradient variational Bayes inference [2]_.


Features
--------

Some of the features of Aboleth:

- Bayesian fully-connected, embedding and convolutional layers using SGVB [2]_
  for inference.
- Random Fourier and arc-cosine features for approximate Gaussian processes.
  Optional variational optimisation of these feature weights as per [1]_.
- Imputation layers with parameters that are learned as part of a model.
- Very flexible construction of networks, e.g. multiple inputs, ResNets etc.
- Compatible and interoperable with other neural net frameworks such as `Keras
  <https://keras.io/>`_ (see the `demos
  <https://github.com/data61/aboleth/tree/develop/demos>`_ for more
  information).


Why?
----

The purpose of Aboleth is to provide a set of high performance and light weight
components for building Bayesian neural nets and approximate (deep) Gaussian
process computational graphs. We aim for *minimal* abstraction over pure
TensorFlow, so you can still assign parts of the computational graph to
different hardware, use your own data feeds/queues, and manage your own
sessions etc.

Here is an example of building a simple Bayesian neural net classifier with one
hidden layer and Normal prior/posterior distributions on the network weights:

.. code-block:: python

    import tensorflow as tf
    import aboleth as ab

    # Define the network, ">>" implements function composition,
    # the InputLayer gives a kwarg for this network, and
    # allows us to specify the number of samples for stochastic
    # gradient variational Bayes.
    net = (
        ab.InputLayer(name="X", n_samples=5) >>
        ab.DenseVariational(output_dim=100) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseVariational(output_dim=1) >>
    )

    X_ = tf.placeholder(tf.float, shape=(None, D))
    Y_ = tf.placeholder(tf.float, shape=(None, 1))

    # Build the network, nn, and the parameter regularization, kl
    nn, kl = net(X=X_)

    # Define the likelihood model
    likelihood = tf.distributions.Bernoulli(logits=nn).log_prob(Y_)

    # Build the final loss function to use with TensorFlow train
    loss = ab.elbo(likelihood, kl, N)

    # Now your TensorFlow training code here!
    ...

At the moment the focus of Aboleth is on supervised tasks, however this is
subject to change in subsequent releases if there is interest in this
capability.


Installation
------------

**NOTE**: Aboleth is a *Python 3* library only. Some of the functionality 
within it depends on features only found in python 3. Sorry.    

To get up and running quickly you can use pip and get the Aboleth package from
`PyPI <https://pypi.python.org/pypi>`_::

    $ pip install aboleth

For the best performance on your architecture, we recommend installing
`TensorFlow from sources
<https://www.tensorflow.org/install/install_sources>`_.

Or, to install additional dependencies required by the `demos
<https://github.com/data61/aboleth/tree/develop/demos>`_::

    $ pip install aboleth[demos]

To install in develop mode with packages required for development we recommend
you clone the repository from GitHub::

    $ git clone git@github.com:data61/aboleth.git

Then in the directory that you cloned into, issue the following::

    $ pip install -e .[dev]


Getting Started
---------------

See the `quick start guide
<http://aboleth.readthedocs.io/en/latest/quickstart.html>`_ to get started, and
for more in depth guide, have a look at our `tutorials
<http://aboleth.readthedocs.io/en/latest/tutorials/tutorials.html>`_.
Also see the `demos
<https://github.com/data61/aboleth/tree/develop/demos>`_ folder for more
examples of creating and training algorithms with Aboleth.

The full project documentation can be found on `readthedocs
<http://aboleth.readthedocs.io>`_.


References
----------

.. [1] Cutajar, K. Bonilla, E. Michiardi, P. Filippone, M. Random Feature 
       Expansions for Deep Gaussian Processes. In ICML, 2017.
.. [2] Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In ICLR,
       2014.


License
-------

Copyright 2017 CSIRO (Data61)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
