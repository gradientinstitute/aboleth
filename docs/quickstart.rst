Quick Start Guide
=================

In Aboleth we use function composition, that we have implemented with the
right-shift operator, ``>>``, to compose neural network models. These models
are classes that when called return a TensorFlow computational graph
(tf.Tensor). We can best demonstrate this with a few examples.

.. _log_clas:

Logistic Classification
-----------------------

For our first example, lets make a simple logistic classifier with :math:`L_2`
regularisation on the model weights:

.. code-block:: python

    import tensorflow as tf
    import aboleth as ab

    layers = (
        ab.InputLayer(name="X", n_samples=1) >>
        ab.DenseMap(output_dim=1, l1_reg=0, l2_reg=.05) >>
        ab.Activation(tf.nn.sigmoid)
    )

At this stage ``layers`` is a callable class (of type
``ab.baselayers.MultiLayer``), and no computational graph has been built.
``ab.InputLayer`` allows us to name our inputs so we can refer to them later
when we call our class ``layers``. This is useful when we have multiple inputs
into our model, for examples, if we want to deal with continuous and
categorical features separately (see :ref:`multi_in`). Well worry about what
``n_samples`` does in :ref:`bayes_log_clas`.

We need to define a likelihood model for our classifier, which is typically a
Bernoulli distribution (which corresponds to a log-loss):

.. code-block:: python
        
    likelihood = ab.likelihoods.Bernoulli()

TODO:
    - call layers
    - make loss
    - train

To make the computational graph, we call ``layers``: TODO


.. _bayes_log_clas:

Bayesian Logistic Classification
--------------------------------
TODO


.. _ml_type_2:

Maximum Likelihood Type II
--------------------------
TODO


.. _multi_in:

Multiple Inputs
---------------
TODO


.. _gp:

Approximate Gaussian Process
----------------------------
TODO
