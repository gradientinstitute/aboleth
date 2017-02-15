import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal


class DeepGP():

    def __init__(
            self,
            n_features=100,
            layer_sizes=[],
            var=1.0,
            n_samples=5
    ):
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.var = var
        self.n_samples = n_samples

    def fit(self, X, y):

        self._make_NN(X, y)
        loss = - self._ELL(X, y) + self._KL()
        return loss

    def predict(self, X, n_samples=20):
        Eys = []
        for _ in range(n_samples):
            W_samp = [q.sample() for q in self.qW]
            b_samp = [q.sample() for q in self.qb]
            Eys.append(self._evaluate_NN(X, W_samp, b_samp))
        return tf.transpose(tf.stack(Eys))

    def _make_NN(self, X, y):

        self.Xd = X.shape[1]
        self.yd = y.shape[1] if np.ndim(y) > 1 else 1

        # Adjust input layer sizes depending on activation
        dims_in = [self.Xd] + self.layer_sizes
        dims_out = self.layer_sizes + [self.yd]
        fout = 2 * self.n_features

        # Initialize weight priors and approximate posteriors
        self.pW, self.qW, self.pb, self.qb, self.Phi = [], [], [], [], []
        for din, dout in zip(dims_in, dims_out):
            self.Phi.append(RandomFF(din, self.n_features))

            # Priors
            self.pW.append(Gaussian(
                mu=tf.zeros((fout, dout)),
                var=tf.ones((fout, dout))
            ))
            self.pb.append(Gaussian(
                mu=tf.zeros((dout,)),
                var=tf.ones((dout,))
            ))

            # Posteriors
            self.qW.append(Gaussian(
                mu=tf.Variable(tf.random_normal((fout, dout))),
                var=tf.nn.softplus(tf.Variable(tf.random_normal((fout, dout))))
            ))
            self.qb.append(Gaussian(
                mu=tf.Variable(tf.random_normal((dout,))),
                var=tf.nn.softplus(tf.Variable(tf.random_normal((dout,))))
            ))

        # TODO: Initialize this properly! Or better yet, make this class not
        # just a regressor, but likelihood agnostic!
        # self.var = tf.nn.softplus(tf.Variable(self.var))

    def _evaluate_NN(self, X, W, b):
        F = X
        for W_l, b_l, phi in zip(W, b, self.Phi):
            P = phi.transform(F)
            F = tf.matmul(P, W_l) + b_l
        return tf.reshape(F, [-1])

    def _likelihood(self, X, y, W, b):
        f = self._evaluate_NN(X, W, b)
        ll = normal_logpdf(y, f, self.var)
        return tf.reduce_sum(ll)

    def _KL(self):
        KL = 0
        for qW, pW, qb, pb in zip(self.qW, self.pW, self.qb, self.pb):
            KL += normal_KLqp(qW.mu, pW.mu, qW.var, pW.var)
            KL += normal_KLqp(qb.mu, pb.mu, qb.var, pb.var)
        return KL

    def _ELL(self, X, y):

        ELL = 0
        for _ in range(self.n_samples):
            W, b = [], []
            for qW, qb in zip(self.qW, self.qb):
                E = np.random.randn(*qW.shape())
                e = np.random.randn(*qb.shape())
                W.append(qW.mu + E * qW.std)
                b.append(qb.mu + e * qb.std)

            ELL += self._likelihood(X, y, W, b)

        return ELL / self.n_samples


# TODO can we just use Normal? Or even sub-class normal to provide var?
class Gaussian():

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        self.std = tf.sqrt(var)

    def sample(self):
        norm = Normal(self.mu, self.std)
        return norm.sample()

    def shape(self):
        return self.mu.get_shape()


class RandomFF():

    def __init__(self, input_dim, n_features):
        self.D = np.float32(n_features)
        self.d = input_dim
        # TODO use tensorflow constants here?
        self.P = np.random.randn(input_dim, n_features).astype(np.float32)

    def transform(self, F):
        FP = tf.matmul(F, self.P)
        real = tf.cos(FP)
        imag = tf.sin(FP)
        return tf.concat([real, imag], axis=1) / tf.sqrt(self.D)


def normal_KLqp(mu_p, mu_q, var_p, var_q):
    var_qp = var_q / var_p
    KL = (mu_q - mu_p)**2 / (2 * var_p) + 0.5 * (var_qp - 1 - tf.log(var_qp))
    return tf.reduce_sum(KL)


def normal_logpdf(x, mu, var):
    norm = Normal(mu, tf.sqrt(var))
    return norm.log_pdf(x)
