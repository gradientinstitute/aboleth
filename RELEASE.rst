Release 0.9.0
=============

This release focuses on better initialisation for the weights, and improves the
performance of feed-forward neural nets.

- Self-normalising neural net initialisation and dropout options.
- Noise contrastive prior layers for better uncertainty estimation away from
  training data.
- TensorFlow Custom estimator interface demonstrated in the SARCOS demos.
- Simplifies interfaces for learning priors etc in the variational and kernel
  layers.
- Remove "MAP" nomenclature from the non-variational layers, as these layers
  have no regularisation by default now.
- Simplifies imputation layers interfaces.


Release 0.8.0
=============

Refactor the user interface for more clarity and flexibility. Also a lot of
code maintenance and TensorBoard integration, specifically:

- Compatibility checked with TensorFlow up to r1.6.
- Convert the likelihoods to tensors away from distributions.
- Clarify what is being optimised in the layers (do not optimise priors by
  default)
- Clean up the imputation module
- Make all Variables constructed within the layers view-able trough TensorBoard

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
- Numerous numerical and usability fixes.

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
