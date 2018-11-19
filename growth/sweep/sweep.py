from os.path import isdir
from os import mkdir
import numpy as np
from .batch import Batch
from .simulation import GrowthSimulation


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
    def recombination_duration(self):
        """ Fraction of growth period subject to recombination. """
        return np.linspace(0.0, 1., self.density)

    @property
    def grid(self):
        """ Meshgrid of division and recombination rates. """
        return np.meshgrid(self.recombination_duration, self.recombination, indexing='xy')

    @classmethod
    def build_simulation(cls, parameters, simulation_path, **kwargs):
        """
        Builds and saves a simulation instance for a set of parameters.

        Args:

            parameters (iterable) - parameter sets

            simulation_path (str) - simulation path

            kwargs: keyword arguments for GrowthSimulation

        """

        # parse parameters
        recombination_duration, recombination = parameters

        # instantiate simulation
        final_population = 2**12
        recombinant_population = recombination_duration*final_population

        simulation = GrowthSimulation(
            division=1.0,
            recombination=recombination,
            recombinant_population=recombinant_population,
            final_population=final_population,
            **kwargs)

        # create simulation directory
        if not isdir(simulation_path):
            mkdir(simulation_path)

        # save simulation
        simulation.save(simulation_path)

