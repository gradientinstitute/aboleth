"""Random State."""
import numpy as np


class SeedGenerator:
    """Make new random seeds deterministically from a base random seed."""

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
    """Set the global hyperseed from which to generate all other seeds.

    Parameters
    ----------
    hs : None, int, array_like
        seed the random state of the global hyperseed, see
        numpy.random.RandomState for valid inputs.
    """
    seedgen.set_hyperseed(hs)
