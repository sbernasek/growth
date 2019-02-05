import numpy as np
import pandas as pd
from os.path import join
from ..visualization.batch import BatchVisualization
from .simulation import GrowthSimulation


class Batch(BatchVisualization):
    """
    Class for managing batch of simulations.
    """

    def __init__(self, paths, root='.'):
        self.paths = paths
        self.root = root

    @property
    def size(self):
        """ Batch size. """
        return len(self.paths)

    @property
    def results(self):
        """ Returns results of each simulation as a pandas dataframe. """
        df = pd.DataFrame([simulation.results for simulation in self])
        df['replicate_id'] = np.arange(self.size)
        return df

    def __getitem__(self, index):
        """ Returns simulation instance. """
        return self.load_simulation(index)

    def __iter__(self):
        """ Iterate over serialized simulations. """
        self.count = 0
        return self

    def __next__(self):
        """ Returns next simulation instance. """
        if self.count < self.size:
            simulation = self.load_simulation(self.count)
            self.count += 1
            return simulation
        else:
            raise StopIteration

    def load_simulation(self, index):
        """ Load simulation. """
        return GrowthSimulation.load(join(self.root, self.paths[index]))

    def measure(self, ambiguity=0.1, replicates=1):
        measurements = []
        for growth_id, simulation in enumerate(self):
            for fluorescence_id in range(replicates):
                data = simulation.measure(ambiguity)
                data['growth_replicate'] = growth_id
                data['fluorescence_replicate'] = fluorescence_id
                measurements.append(data)
        return pd.concat(measurements)
