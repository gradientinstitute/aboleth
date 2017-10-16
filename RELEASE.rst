Release 0.6.3
=============
Make ab.InputLayer always make at least 1 sample of the networks for
consistency and simplicity. This also makes the quick start guide examples
work.

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
