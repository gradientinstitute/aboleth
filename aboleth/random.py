"""Random State."""
import numpy as np
import tensorflow as tf


class SeedGenerator:
    def __init__(self):
        self.state = np.random.RandomState()

    def set_hyperseed(self, hs):
        self.state.seed(hs)

    def next(self):
        result = self.state.randint(0, 2**32)
        return result

    def __next__(self):
        return self.next()

# Dont judge me -- most RNGs are global vars
seedgen = SeedGenerator()


def set_hyperseed(hs):
    seedgen.set_hyperseed(hs)

