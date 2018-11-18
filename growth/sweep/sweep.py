import numpy as np
from .batch import Batch


class Sweep(Batch):

    def __init__(self, density=11, batch_size=10):
        self.density = density
        parameters = np.array(list(zip(*[grid.ravel() for grid in self.grid])))
        parameters = np.repeat(parameters, repeats=batch_size, axis=0)
        super().__init__(parameters)

    @property
    def division(self):
        """ Division rate values.  """
        return np.linspace(0.1, 1., self.density)

    @property
    def recombination(self):
        """ Recombination rate values.  """
        return np.linspace(0.0, 1., self.density)

    @property
    def grid(self):
        """ Meshgrid of division and recombination rates. """
        return np.meshgrid(self.division, self.recombination, indexing='xy')
