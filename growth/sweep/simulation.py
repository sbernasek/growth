from os.path import join, isdir
from os import mkdir

from ..cells.cultures import Culture


class GrowthSimulation(Culture):

    def __init__(self,
                 division=0.1,
                 recombination=0.1,
                 recombinant_population=2**8,
                 final_population=2**11,
                 **kwargs):

        super().__init__(**kwargs)
        self.division = division
        self.recombination = recombination
        self.recombinant_population = recombinant_population
        self.final_population = final_population

    def save(self, path, save_history=True):
        """ Save pickled object to <path/simulation.pkl>. """

        # create simulation directory
        if not isdir(path):
            mkdir(path)
        super().save(join(path, 'simulation.pkl'), save_history=save_history)

    @classmethod
    def load(cls, path):
        """ Load pickled instance from <path/simulation.pkl>. """
        return super().load(join(path, 'simulation.pkl'))

    def run(self):
        """
        Run growth simulation.
        """

        # growth with recombination
        self.grow(min_population=self.recombinant_population,
                 division=self.division,
                 recombination=self.recombination,
                 reference_population=self.final_population)

        # growth without recombination
        self.grow(min_population=self.final_population,
                 division=self.division,
                 recombination=0.,
                 reference_population=self.final_population)

    def branch(self, t=None):
        """ Returns copy of culture at generation <t> including history. """
        sim = super().branch(t)
        sim.division = self.division
        sim.recombination = self.recombination
        sim.recombinant_population = self.recombinant_population
        sim.final_population = self.final_population
        return sim

    def freeze(self, t):
        """ Returns snapshot of culture at generation <t>. """
        sim = super().freeze(t)
        sim.division = self.division
        sim.recombination = self.recombination
        sim.recombinant_population = self.recombinant_population
        sim.final_population = self.final_population
        return sim
