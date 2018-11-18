from os.path import join
import numpy as np
from functools import reduce
from operator import add


class Cell:

    def __init__(self, xy=None, chromosomes=None, lineage=''):

        # set generation
        self.lineage = lineage

        # set chromosomes
        if chromosomes is None:
            chromosomes = np.array([0, 1])
        self.chromosomes = chromosomes

        # set position
        if xy is None:
            xy = np.zeros(2, dtype=float)
        self.xy = xy

    @property
    def generation(self):
        return len(self.lineage)

    @property
    def genotype(self):
        return self.chromosomes.sum()

    @property
    def phenotype(self):
        return np.random.normal(loc=self.genotype, scale=1.)

    def copy(self):
        """ Returns copy of cell. """
        return self.__class__(self.xy, self.chromosomes, self.lineage)

    def set_xy(self, xy):
        self.xy = xy

    def recombine(self, recombination=0.):

        # duplicate chromosomes
        chromosomes = np.tile(self.chromosomes, 2)

        # recombination
        if np.random.random() <= recombination:
            chromosomes.sort()

        return chromosomes

    def divide(self, recombination=0., reference_population=1000):

        # set average spacing between cells
        spacing = np.sqrt(2/reference_population) / 1e5

        # perform recombination
        chromosomes = self.recombine(recombination=recombination)

        # determine child positions
        jitter = np.random.normal(scale=spacing, size=(2, 2))
        xy_a, xy_b = self.xy+jitter[0], self.xy+jitter[1]

        # instantiate children
        daughter_a = self.__class__(xy_a, chromosomes[:2], self.lineage+'0')
        daughter_b = self.__class__(xy_b, chromosomes[2:], self.lineage+'1')

        return [daughter_a, daughter_b]

    def grow(self, max_generation=3, **kwargs):
        """
        Recursive growth.
        """

        # stopping criterion
        if self.generation >= max_generation:
            return [self]

        # divide
        else:
            children = self.divide(**kwargs)
            recurse = lambda x: x.grow(max_generation=max_generation, **kwargs)
            return reduce(add, map(recurse, children))
