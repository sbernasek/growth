from functools import reduce
from operator import add
import numpy as np


class Clones:

    def __init__(self, data):
        self.data = data

    @property
    def size(self):
        return self.apply(lambda x: sum(x['sizes']), add)

    @property
    def num_genotypes(self):
        return len(self.data)

    @property
    def sizes(self):
        return self.apply(lambda x: x['sizes'], join=add)

    @property
    def num_clones(self):
        return self.apply(lambda x: x['number'], join=add)

    @property
    def size_variation(self):
        sizes = self.sizes
        return np.std(sizes) / np.mean(sizes)

    def apply(self, func, join):
        return reduce(join, [func(self.data[x]) for x in [0, 2]])
