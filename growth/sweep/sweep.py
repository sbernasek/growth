from os.path import isdir, exists, join
from os import mkdir
import numpy as np
import pandas as pd
import pickle
from ..visualization.sweep import SweepVisualization
from .jobs import Job
from .simulation import GrowthSimulation
from .analysis import SweepResults


class SweepProperties:
    """
    Properties for parameter sweep.
    """

    @property
    def shape(self):
        """ Parameter sweep dimensions. """
        return self.grid[0].shape

    @property
    def division(self):
        """ Division rate values.  """
        return np.linspace(0.1, 1., self.density)

    @property
    def recombination(self):
        """ Recombination rate values.  """
        return np.array([0.25], dtype=np.float64)
        #return np.logspace(-5, 0, num=self.density, base=2)

    @property
    def recombination_start(self):
        """ Population size at which recombination begins. """
        return np.arange(0,(self.population-self.recombination_duration)+1, .5)

    @property
    def grid(self):
        """ Meshgrid of division and recombination rates. """
        return np.meshgrid(self.recombination_start, self.recombination, indexing='xy')


class Sweep(Job, SweepProperties, SweepVisualization):
    """
    Class for performing a parameter sweep.
    """

    def __init__(self,
                 density=5,
                 replicates=10,
                 division_rate=0.1,
                 recombination_duration=4,
                 population=11):
        self.density = density
        self.population = population
        self.division_rate = division_rate
        self.recombination_duration = recombination_duration

        parameters = np.array(list(zip(*[grid.ravel() for grid in self.grid])))
        parameters = np.repeat(parameters, repeats=replicates, axis=0)
        super().__init__(parameters, batch_size=replicates)

    @property
    def batches(self):
        """ 2D array of Batch objects. """
        return np.array(super().batches).reshape(self.shape)

    @classmethod
    def load(cls, path):
        sweep = super().load(path)
        results_path = join(path, 'data.hdf')
        if exists(results_path):
            sweep._results = pd.read_hdf(results_path, 'results')
        return sweep

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

        # compile results from all batches
        data = []
        for row_id, row in enumerate(self.batches):
            for column_id, batch in enumerate(row):
                batch_data = batch.results
                batch_data['row_id'] = row_id
                batch_data['column_id'] = column_id
                data.append(batch_data)
        data = pd.concat(data)

        # add mean clone size and start time attributes
        data['mean_clone_size'] = data.population / data.num_clones
        data['start_time'] = data.recombination_start

        # save results
        self._results = data
        self._results.to_hdf(join(self.path, 'data.hdf'), key='results')

    @property
    def results(self):
        """ Returns simulation results object. """
        return SweepResults(self._results, self.shape)
