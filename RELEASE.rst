Release 0.7.0
=============

- Update to TensorFlow r1.4.
- Tutorials in the documentation on:

  1. Interfacing with Keras
  2. Saving/loading models
  3. How to build a variety of regressors with Aboleth

- New prediction module with some convenience functions, including freezing the 
  weight samples during prediction.
- Bayesian convolutional layers with accompanying demo.
- Allow the number of samples drawn from a model to be varied by using 
  placeholders.
- Generalise the feature embedding layers to work on matrix inputs (instead of
  just column vectors).
- Numerous numerical and useability fixes.

Release 0.6.5
=============

Hotfix: Test batch shape of likelihoods to see if they are compatible with
models. Without this test the likelihoods may be broadcast, and result in poor
performance.

Release 0.6.4
=============

Hotfix: Make a ab.MaskInputLayer for binary mask inputs when we don't want to
tile the inputs.

Release 0.6.3
=============

- Make ab.InputLayer always make at least 1 sample of the networks for
  consistency and simplicity.
- This also makes the quick start guide examples work.

Release 0.6.2
=============

Hotfix: Fix the dropout noise shape so we get samples of the latent function of
the layer (rather than the observations). Also some doco tweaks.

Release 0.6.1
=============

Hotfix: Fix regression whereby setting the random seed was not working with the
new distribution objects from TensorFlow (tf.distributions).


Release 0.6.0
=============

Some moderate changes to the API from:

- Using TensorFlow's tf.distributions to replace Aboleth's likelihoods
- Using TensorFlow's tf.distributions to replace Aboleth's distributions


Release 0.5.0
=============

Initial release of Aboleth
