"""Random generators and state."""
import numpy as np


class SeedGenerator:
    r"""Make new random seeds deterministically from a base random seed."""

    def __init__(self):
        """Construct a SeedGenerator object."""
        self.state = np.random.RandomState()

    def set_hyperseed(self, hs):
        """Set the random seed state in this object.

        Parameters
        ----------
        hs : None, int, array_like
            seed the random state of this object, see numpy.random.RandomState
            for valid inputs.
        """
        self.state.seed(hs)

    def next(self):
        """Generate a random int using this object's base state.

        Returns
        -------
        result : int
            an integer that can be used to seed other random states
            deterministically.
        """
        result = self.state.randint(0, 2**32)
        return result

    def __next__(self):
        """Next generator."""
        return self.next()


# Dont judge me -- most RNGs are global vars
seedgen = SeedGenerator()


def set_hyperseed(hs):
    r"""Set the global hyperseed from which to generate all other seeds.

    Parameters
    ----------
    hs : None, int, array_like
        seed the random state of the global hyperseed, see
        numpy.random.RandomState for valid inputs.
    """
    seedgen.set_hyperseed(hs)


def endless_permutations(N):
    r"""
    Generate an endless sequence of permutations of the set [0, ..., N).

    If we call this N times, we will sweep through the entire set without
    replacement, on the (N+1)th call a new permutation will be created, etc.

    Parameters
    ----------
    N: int
        the length of the set

    Yields
    ------
    int :
        yeilds a random int from the set [0, ..., N)

    Examples
    -------
    >>> perm = endless_permutations(5)
    >>> type(perm)
    <class 'generator'>
    >>> p = next(perm)
    >>> p < 5
    True
    >>> p2 = next(perm)
    >>> p2 != p
    True
    """
    generator = np.random.RandomState(next(seedgen))

    while True:
        batch_inds = generator.permutation(N)
        for b in batch_inds:
            yield b
