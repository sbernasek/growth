from os.path import isdir
from os import mkdir
import numpy as np
from .batch import Batch
from .simulation import GrowthSimulation


class Sweep(Batch):

    def __init__(self,
                 density=11,
                 batch_size=10,
                 division_rate=0.1,
                 population=2**12):
        self.density = density
        self.population = population
        self.division_rate = division_rate
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
        return np.logspace(-(self.density-1), 0, num=self.density, base=2)

    @property
    def recombinant_fraction(self):
        """ Fraction of growth period subject to recombination. """
        return np.logspace(-(self.density-1), 0, num=self.density, base=2)

    @property
    def grid(self):
        """ Meshgrid of division and recombination rates. """
        return np.meshgrid(self.recombinant_fraction, self.recombination, indexing='xy')

    def build_simulation(self, parameters, simulation_path, **kwargs):
        """
        Builds and saves a simulation instance for a set of parameters.

        Args:

            parameters (iterable) - parameter sets

            simulation_path (str) - simulation path

            kwargs: keyword arguments for GrowthSimulation

        """

        # parse parameters
        recombinant_fraction, recombination_rate  = parameters
        recombinant_population = int(recombinant_fraction * self.population)

        # instantiate simulation
        simulation = GrowthSimulation(
            division=self.division_rate,
            recombination=recombination_rate,
            recombinant_population=recombinant_population,
            final_population=self.population,
            **kwargs)

        # create simulation directory
        if not isdir(simulation_path):
            mkdir(simulation_path)

        # save simulation
        simulation.save(simulation_path)

