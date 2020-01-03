from functools import reduce
from operator import add
import numpy as np


class Patches:
    """ Methods for analyzing patch patterns. """

    def __init__(self, data):
        self.data = data

    @property
    def keys(self):
        return self.data.keys()

    def apply(self, func, join):
        """ Apply to recombinant patches. """
        return reduce(join, [func(self.data[x]) for x in self.keys])

    @property
    def num_patches(self):
        """ Number of patches. """
        return self.apply(lambda x: x['number'], join=add)

    @property
    def sizes(self):
        return self.apply(lambda x: x['sizes'], join=add)

    @property
    def mean_patch_size(self):
        return np.mean(self.sizes)

    @property
    def median_patch_size(self):
        return np.median(self.sizes)

    @property
    def size_variation(self):
        sizes = self.sizes
        return np.std(sizes) / np.mean(sizes)
