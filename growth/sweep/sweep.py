from os.path import isdir, exists, join
from os import mkdir
import numpy as np
import pandas as pd
from .batch import Batch
from .simulation import GrowthSimulation


class Sweep(Batch):

    def __init__(self,
                 density=10,
                 batch_size=10,
                 division_rate=0.1,
                 recombination_duration=200,
                 population=1000):
        self.density = density
        self.population = population
        self.division_rate = division_rate
        self.recombination_duration = recombination_duration

        parameters = np.array(list(zip(*[grid.ravel() for grid in self.grid])))
        parameters = np.repeat(parameters, repeats=batch_size, axis=0)
        super().__init__(parameters)

    @classmethod
    def load(cls, path):
        sweep = super().load(path)
        results_path = join(path, 'data.hdf')
        if exists(results_path):
            sweep.results = pd.read_hdf(results_path, 'results')
        return sweep

    @staticmethod
    def load(path):
        """ Load batch from target <path>. """
        with open(join(path, 'batch.pkl'), 'rb') as file:
            batch = pickle.load(file)
        batch.path = path
        return batch

    @property
    def division(self):
        """ Division rate values.  """
        return np.linspace(0.1, 1., self.density)

    @property
    def recombination(self):
        """ Recombination rate values.  """
        return np.logspace(-2, 0, num=self.density, base=10)

    @property
    def recombination_start(self):
        """ Population size at which recombination begins. """
        ubound = (self.population-self.recombination_duration)
        return np.linspace(0, ubound, num=self.density)

    @property
    def grid(self):
        """ Meshgrid of division and recombination rates. """
        return np.meshgrid(self.recombination_start, self.recombination, indexing='xy')

    def build_simulation(self, parameters, simulation_path, **kwargs):
        """
        Builds and saves a simulation instance for a set of parameters.

        Args:

            parameters (iterable) - parameter sets

            simulation_path (str) - simulation path

            kwargs: keyword arguments for GrowthSimulation

        """

        # parse parameters
        recombination_start, recombination_rate  = parameters

        # instantiate simulation
        simulation = GrowthSimulation(
            division=self.division_rate,
            recombination=recombination_rate,
            recombination_start=recombination_start,
            recombination_duration=self.recombination_duration,
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
                'recombination_start': simulation.recombination_start,
                'recombination_duration': simulation.recombination_duration,
                'population': simulation.size,
                'transclone_edges': simulation.heterogeneity,
                'percent_heterozygous': simulation.percent_heterozygous,
                'num_clones': simulation.clones.num_clones,
                'clone_size_variation': simulation.clones.size_variation}

            records.append(record)

        # save results
        self.results = pd.DataFrame(records)
        self.results.to_hdf(join(self.path, 'data.hdf'), key='results')
