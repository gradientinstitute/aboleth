from collections import OrderedDict

import tensorflow as tf

from layers import Layer


class BayesNN():
    # NOTE: More layer/parameters will mean you WILL need more data! The KL
    # penalties stack up pretty fast!

    def __init__(
            self,
            N,
            likelihood,
            n_samples=10,
    ):
        self.N = tf.to_float(N)
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.layers = OrderedDict()
        self.nlayers = 0

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("layer {} is not compatible!".format(layer))

        # TODO: parse input and output dimensions to make sure they match, OR,
        # Fill in appropriate dimensions (input) if not!
        # TODO: make the following better
        if self.nlayers > 0:
            prevl = next(reversed(self.layers.values()))
            l = layer.build(input_dim=prevl.output_dim)
        else:
            l = layer.build()

        self.layers["{}_{}".format(layer, self.nlayers)] = l
        self.nlayers += 1

    def loss(self, X, y):
        # Mini-batch discount factor
        B = self.N / tf.to_float(tf.shape(X)[0])
        loss = - B * self._ELL(X, y) + self._KL()
        return loss

    def predict(self, X, n_samples=20):
        Eys = [self._evaluate_NN(X) for _ in range(n_samples)]
        return Eys

    def _evaluate_NN(self, X):
        if self.nlayers == 0:
            raise ValueError("Please add layers to this network first!")

        F = X
        for l in self.layers.values():
            F = l(F)
        return F

    def _KL(self):
        KL = 0.
        for l in self.layers.values():
            KL += l.KL()
        return KL

    def _ELL(self, X, y):
        ELL = 0.
        for _ in range(self.n_samples):
            #TODO: make these key value, i.e. log_prob(Y=, H=...)
            ll = self.likelihood.log_prob(y, self._evaluate_NN(X))
            ELL += tf.reduce_sum(ll)
        return ELL / self.n_samples
