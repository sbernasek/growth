from os.path import isdir
from os import mkdir
import numpy as np
import pandas as pd
from .batch import Batch
from .simulation import GrowthSimulation


class Sweep(Batch):

    def __init__(self,
                 density=12,
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
        return np.logspace(-2, 0, num=self.density, base=2)

    @property
    def recombinant_fraction(self):
        """ Fraction of growth period subject to recombination. """
        magnitude = int(np.log2(self.population))
        return np.logspace(-magnitude, 0, num=self.density, base=2)

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

    def aggregate(self):
        """ Aggregate results from all sweeps. """

        row_size = self.density*self.batch_size
        col_size = self.batch_size

        records = []
        for index in range(self.N):
            row_id = index // row_size
            col_id = (index % row_size) // col_size
            batch_id = (index % row_size) % col_size

            # load simulation
            simulation = self[index]

            record = {
                'row': row_id,
                'column': col_id,
                'replicate': batch_id,
                'division_rate': simulation.division,
                'recombination_rate': simulation.recombination,
                'recombinant_population': simulation.recombinant_population,
                'recombinant_fraction': simulation.recombinant_population / simulation.final_population,
                'population': simulation.size,
                'transclone_edges': simulation.heterogeneity,
                'percent_heterozygous': simulation.percent_heterozygous,
                'num_clones': simulation.clones.num_clones,
                'clone_size_variation': simulation.clones.size_variation}

            records.append(record)

        # compile dataframe
        data = pd.concat(records)

        return data

