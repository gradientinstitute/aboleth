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
        ab.InputLayer(name="X") >>
        ab.DenseMap(output_dim=1, l1_reg=0, l2_reg=.05) >>
        ab.Activation(tf.nn.sigmoid)
    )

At this stage ``layers`` is a callable class
(``ab.baselayers.MultiLayerComposite``), and no computational graph has been
built.  ``ab.InputLayer`` allows us to name our inputs so we can refer to them
later when we call our class ``layers``. This is useful when we have multiple
inputs into our model, for examples, if we want to deal with continuous and
categorical features separately (see :ref:`multi_in`).

So now we defined the structure of the predictive model, if we wish we can
create it's computational graph,

.. code-block:: python

    net, reg = layers(X=X_)

Where the key word argument ``X`` was defined in the ``InputLayer`` and ``X_``
is a placeholder (``tf.placeholder``) or the actual predictive data we want to
build into our model. ``net`` is the resulting computational graph of our
predictive model/network, and ``reg`` are the regularisation terms associated
with the model parameters (layer weights in this case).

If we wanted, we could evaluate ``net`` right now in a TensorFlow session,
however, none of the weights have been fit to the data. In order to fit the
weights, we need to define a loss function. So firstly we need to define a
likelihood model for our classifier, here we choose a Bernoulli distribution
for our binary classifier (which corresponds to a log-loss):

.. code-block:: python
        
    likelihood = ab.likelihoods.Bernoulli()

Now we have enough to build the loss function we will use to optimize the
model weights:

.. code-block:: python
        
    loss = ab.max_posterior(net, Y_, reg, likelihood)

This is a maximum a-posteriori loss function, which can be though of as a 
maximum likelihood objective with a penalty on the magnitude of the weights
(controlled by ``l2_reg``). Now we have enough to use the ``tf.train`` module
to learn the weights of our model:

.. code-block:: python

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for _ in range(1000):
            sess.run(train, feed_dict={X_: X, Y_: Y})

This will run 1000 iterations of stochastic gradient optimization (using the
Adam learning rate algorithm) where the model sees all of the data every
iteration. We can also run this on mini-batches, see ``ab.batch`` for a simple
batch generator, or TensorFlow's `train` module for a more comprehensive set of
utilities.

Now that we have learned our classifier's weights, we will probably want to use
it to predict class labels for unseen data. This can be very easily achieved by
just evaluating our model on the unseen predictive data (still in the
Tensorflow session from above):

.. code-block:: python

    ...
        probabilities = net.eval(feed_dict={X_: X_query})

And that is it!

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
