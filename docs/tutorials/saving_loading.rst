.. _tut_saveload:

Saving and Loading Aboleth Models
=================================

In this tutorial we will cover the basics of how to save and load models
constructed with Aboleth. We don't provide any inherent saving and loading code
in this library, and rely directly on TensorFlow functionality.


Naming the Graph
----------------

Even though the whole graph you create is saved and automatically named, it
helps when loading to know the exact name of the part of the graph you want to
evaluate. So to begin, we will create a very simple Bayesian linear regressor
with place holders for data. Let's start with the place holders,

.. code::

    with tf.name_scope("Placeholders"):
        n_samples_ = tf.placeholder_with_default(NSAMPLES, shape=[],
                                                 name="samples")
        X_ = tf.placeholder_with_default(X_train, shape=(None, D),
                                         name="X")
        Y_ = tf.placeholder_with_default(Y_train, shape=(None, 1),
                                         name="Y")


We have used a ``name_scope`` here for easy reference later. Also, we'll assume
variables in all-caps have been defined elsewhere. Now let's make our simple
network (just a linear layer),

.. code::

    net = ab.stack(
        ab.InputLayer(name='X', n_samples=n_samples_),
        ab.DenseVariational(output_dim=1, full=True)
    )

And now lets build and name our graph and associate names with the parts of it
we will to evaluate later,

.. code::

    with tf.name_scope("Model"):
        f, kl = net(X=X_)
        likelihood = tf.distributions.Normal(loc=f, scale=ab.pos(NOISE))
        loss = ab.elbo(likelihood, Y_, N, kl)

    with tf.name_scope("Predict"):
        tf.identity(f, name="f")
        ab.sample_mean(f, name="Ey")

Now note how we have used ``tf.identity`` here to name the latent function,
``f``, again this is so we can easily load it later for drawing samples from
our network. We also don't need any variables to assign these operations to
(unless we want to use them before saving), we just need to build them into the
graph.


Saving the Graph
----------------

At this point we recommend reading the Tensorflow tutorial on `saving and
restoring <https://www.tensorflow.org/programmers_guide/saved_model>`_. We
typically use a ``tf.MonitoredTrainingSession`` as it handles all of the model
saving and check-pointing etc. You can see how we do this in the
:ref:`sarcos_reg` demo, but we have also copied the code below for convenience,

.. code::

    # Training graph with step counter
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
    )

    # Training loop
    with tf.train.MonitoredTrainingSession(
            config=CONFIG,
            checkpoint_dir="./",
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[log]
    ) as sess:
        for i in range(NEPOCHS):

            # your training code here
            ...

This code will also make it easy to use TensorBoard to monitor your training,
simply point it at the ``checkpoint_dir`` and run it like,

.. code:: 

    $ tensorboard --logdir=<checkpoint_dir>

Once you are satisfied that your model has converged, you can just kill the
python process. If you think it could do with a bit more "baking", then just
simply re-run the training script and the ``MonitoredTrainingSession`` will
ensure you resume learning where you left off! 


Loading Specific Parts of the Graph for Prediction
--------------------------------------------------

Typically we only want to evaluate particular parts of the graph (that is, the
ones we named previously). In this section we'll go through how to load the
last checkpoint saved by the ``MonitoredTrainingSession``, and to get hold of
the tensors that we named. We then use these tensors to predict on new query
data!

.. code::

    # Get latest checkpoint
    model = tf.train.latest_checkpoint(CHECKPOINT_DIR)

    # Make a graph and a session we will populate with our saved graph
    graph = tf.Graph()
    with graph.as_default():

        sess = tf.Session()
        with sess.as_default():

            # Restore graph
            saver = tf.train.import_meta_graph("{}.meta".format(model))
            saver.restore(sess, model_file)

            # Restore place holders
            X_ = graph.get_operation_by_name("Placeholders/X").outputs[0]
            Y_ = graph.get_operation_by_name("Placeholders/Y").outputs[0]
            n_samples_ = graph.\
                get_operation_by_name("Placeholders/samples").outputs[0]

            feed_dict = {X_: X_test, n_samples_: PREDICTSAMPLES}

            f = graph.get_operation_by_name("Predict/f").outputs[0]
            Ey = graph.get_operation_by_name("Predict/Ey").outputs[0]

            f_samples, y_pred = sess.run([f, Ey], feed_dict=feed_dict) 


The most complicated part of the above code is remembering all of the
boiler-plate to insert the saved graph into a new session, and then do get our
place holders and prediction tensors. Once we have done this though, evaluating
the operations we need for prediction is handled in the usual way. We have also
assumed in this demo that you want to use more samples for prediction
(``PREDICTSAMPLES``) than for training (``NSAMPLES``), so we have made this
also a place holder.

That's it!
