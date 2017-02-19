import numpy as np
import tensorflow as tf


# TODO: Use Keras to build up the NN, then we can use other layers apart from
#   just random features? We could also pass in the NN instead of layer_sizes
#   into the initialiser? Then we just have to loop through the layers and
#   place the appropriate priors and posteriors the weights.
# TODO: This sucks for learning the variance parameters of the Gaussian
#   likelihood (i.e. the var input to the init). Maybe if we lock this off
#   initially while learning weights, then "unlock" it later? once we have
#   reasonble weights?
class DeepGP():

    def __init__(
            self,
            N,
            loglikelihood,
            n_features=100,
            layer_sizes=[],
            reg=1.,
            n_samples=10,
            learn_prior=False,
    ):
        self.N = tf.to_float(N)
        self.likelihood = loglikelihood
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.reg = reg
        self.n_samples = n_samples
        self.learn_prior = learn_prior
        self._make_NN()

    def loss(self, X, y):
        # Mini-batch discount factor
        B = self.N / tf.to_float(tf.shape(X)[0])
        loss = - B * self._ELL(X, y) + self._KL()
        return loss

    def predict(self, X, n_samples=20):
        Eys = [self._evaluate_NN(X, *self._sample_q())
               for _ in range(n_samples)]
        return tf.transpose(tf.stack(Eys))

    def _make_NN(self):

        # Adjust input layer sizes dependig on activation
        dims_in = self.layer_sizes[:-1]
        dims_out = self.layer_sizes[1:]
        fout = 2 * self.n_features

        # Initialize weight priors and approximate posteriors
        self.pW, self.qW, self.pb, self.qb, self.Phi = [], [], [], [], []
        for di, do in zip(dims_in, dims_out):
            self.Phi.append(RandomFF(di, self.n_features))

            # Priors
            self.pW.append(Normal(
                mu=tf.zeros((fout, do)),
                var=stay_pos(tf.Variable(self.reg)) * tf.ones((fout, do))
                if self.learn_prior else self.reg * tf.ones((fout, do))
            ))
            self.pb.append(Normal(
                mu=tf.zeros((do,)),
                var=stay_pos(tf.Variable(self.reg)) * tf.ones((do,))
                if self.learn_prior else self.reg * tf.ones((do,))
            ))

            # Posteriors
            self.qW.append(Normal(
                mu=tf.Variable(tf.random_normal((fout, do))),
                var=stay_pos(tf.Variable(tf.random_normal((fout, do))))
            ))
            self.qb.append(Normal(
                mu=tf.Variable(tf.random_normal((do,))),
                var=stay_pos(tf.Variable(tf.random_normal((do,))))
            ))

    def _evaluate_NN(self, X, W, b):
        F = X
        for W_l, b_l, phi in zip(W, b, self.Phi):
            P = phi.transform(F)
            F = tf.matmul(P, W_l) + b_l
        Ey = tf.squeeze(F) if self.layer_sizes[-1] == 1 else F
        return Ey

    def _KL(self):
        KL = 0.
        for qW, pW, qb, pb in zip(self.qW, self.pW, self.qb, self.pb):
            KL += tf.reduce_sum(qW.KLqp(pW)) + tf.reduce_sum(qb.KLqp(pb))
        return KL

    def _ELL(self, X, y):
        ELL = 0.
        for _ in range(self.n_samples):
            f = self._evaluate_NN(X, *self._sample_q())
            ELL += tf.reduce_sum(self.likelihood(y, f))
        return ELL / self.n_samples

    def _sample_q(self):
        W_samp = [q.sample() for q in self.qW]
        b_samp = [q.sample() for q in self.qb]
        return W_samp, b_samp


# TensorFlow contribs stats (tf.contrib.distribution.norm) seem to be effed,
# also we can control the reparameterisation trick in here
class Normal():

    def __init__(self, mu=0., var=1.):
        self.mu = mu
        self.var = var
        self.sigma = tf.sqrt(var)

    def sample(self):
        # Reparameterisation trick
        e = tf.random_normal(self.shape())
        x = self.mu + e * self.sigma
        return x

    def shape(self):
        return self.mu.get_shape()

    def log_pdf(self, x, mu=None, var=None):
        mu = self.mu if mu is None else mu
        var = self.var if var is None else var
        lp = -0.5 * (tf.log(2 * var * np.pi) + (x - mu)**2 / var)
        return lp

    def KLqp(self, p):
        KL = 0.5 * (tf.log(p.var) - tf.log(self.var) + self.var / p.var - 1 +
                    (self.mu - p.mu)**2 / p.var)
        return KL


class RandomFF():

    def __init__(self, input_dim, n_features):
        self.D = np.float32(n_features)
        self.d = input_dim
        self.P = np.random.randn(input_dim, n_features).astype(np.float32)

    def transform(self, F):
        FP = tf.matmul(F, self.P)
        real = tf.cos(FP)
        imag = tf.sin(FP)
        return tf.concat([real, imag], axis=1) / tf.sqrt(self.D)


def stay_pos(X, minval=1e-10):
    # return tf.exp(X)
    # return tf.nn.softplus(X)
    return tf.maximum(tf.abs(X), minval)


def endless_permutations(N, random_state=None):
    """
    Generate an endless sequence of random integers from permutations of the
    set [0, ..., N).
    If we call this N times, we will sweep through the entire set without
    replacement, on the (N+1)th call a new permutation will be created, etc.
    Parameters
    ----------
    N: int
        the length of the set
    random_state: int or RandomState, optional
        random seed

    Yields
    ------
    int:
        a random int from the set [0, ..., N)
    """
    if isinstance(random_state, np.random.RandomState):
        generator = random_state
    else:
        generator = np.random.RandomState(random_state)

    while True:
        batch_inds = generator.permutation(N)
        for b in batch_inds:
            yield b


def gen_batch(data_dict, batch_size, n_iter=10000, random_state=None):

    N = len(data_dict[list(data_dict.keys())[0]])
    perms = endless_permutations(N, random_state)

    i = 0
    while i < n_iter:
        i += 1
        ind = np.array([next(perms) for _ in range(batch_size)])
        batch_dict = {k: v[ind] for k, v in data_dict.items()}
        yield batch_dict
