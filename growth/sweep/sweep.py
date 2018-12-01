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
    def rates(self):
        """ Recombination rate values.  """
        return np.linspace(self.min_rate, self.max_rate, self.num_rates)

    @property
    def starts(self):
        """ Generations at which recombination begins. """
        return np.linspace(self.first_start, self.last_start, self.num_periods)

    @property
    def grid(self):
        """ Meshgrid of recombination start points and recombination rates. """
        return np.meshgrid(self.starts, self.rates, indexing='xy')


class Sweep(Job, SweepProperties, SweepVisualization):
    """
    Class for performing a parameter sweep.
    """

    def __init__(self,

                 # argument defining growth rate
                 division_rate=0.2,

                 # arguments defining recombination period
                 duration=4,
                 first_start=0,
                 last_start=None,
                 num_periods=None,

                 # arguments defining recombination rate
                 min_rate=0.25,
                 max_rate=1.0,
                 num_rates=1,

                 # arguments defining simulation size
                 min_population=11,
                 num_replicates=10):

        # set division rate
        self.division_rate = division_rate

        # set population size
        self.min_population = min_population
        self.num_replicates = num_replicates

        # define recombination periods
        self.duration = duration
        self.first_start = first_start
        if last_start is None or last_start == -1:
            last_start = min_population - last_start
        self.last_start = last_start
        if num_periods is None or num_periods == -1:
            num_periods = 1 + (last_start - first_start)
        self.num_periods = num_periods

        # define recombination rates
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.num_rates = num_rates

        # construct parameter array
        parameters = np.array(list(zip(*[grid.ravel() for grid in self.grid])))
        parameters = np.repeat(parameters, repeats=replicates, axis=0)

        # instantiate job
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
            recombination_duration=self.duration,
            final_population=self.min_population,
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

        # save results
        self._results = data
        self._results.to_hdf(join(self.path, 'data.hdf'), key='results')

    @property
    def results(self):
        """ Returns simulation results object. """
        return SweepResults(self._results, self.shape)
