import tensorflow as tf
from numpy import np


def pos(X, minval=1e-10):
    # return tf.exp(X)  # Medium speed, but gradients tend to explode
    # return tf.nn.softplus(X)  # Slow but well behaved!
    return tf.maximum(tf.abs(X), minval)  # Faster, but more local optima


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
