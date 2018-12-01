from os.path import join, isdir
from os import mkdir
from functools import reduce
from operator import add

from ..cells.cultures import Culture
from ..cells.cells import Cell


class GrowthSimulation(Culture):

    def __init__(self,
                 division_rate=0.1,
                 recombination_rate=0.1,
                 recombination_start=0,
                 recombination_duration=4,
                 min_population=11,
                 **kwargs):

        # define seed
        seed = [Cell()]
        for generation in range(3):
            rate = recombination_rate
            is_before = generation < recombination_start
            is_after = generation >= recombination_start+recombination_duration
            if is_before or is_after:
                rate *= 0.
            seed = reduce(add, [cell.divide(rate) for cell in seed])

        # instantiate culture
        super().__init__(starter=seed,
                         reference_population=2**min_population,
                         **kwargs)

        # store additional properties
        self.division_rate = division_rate
        self.recombination_rate = recombination_rate
        self.recombination_start = recombination_start
        self.recombination_duration = recombination_duration
        self.min_population = min_population

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
        """ Run growth simulation. """

        # define population windows
        pop0 = int(2**self.recombination_start)
        pop1 = int(2**(self.recombination_start+self.recombination_duration))
        pop2 = int(2**self.min_population)
        if pop1 > pop2:
            pop1 = pop2

        # growth before recombination_rate
        self.grow(min_population=pop0,
                 division_rate=self.division_rate,
                 recombination_rate=0.)

        # growth with recombination_rate
        self.grow(min_population=pop1,
                 division_rate=self.division_rate,
                 recombination_rate=self.recombination_rate)

        # growth after recombination_rate
        self.grow(min_population=pop2,
                 division_rate=self.division_rate,
                 recombination_rate=0.)

    def branch(self, t=None):
        """ Returns copy of culture at generation <t> including history. """
        sim = super().branch(t)
        sim.division_rate = self.division_rate
        sim.recombination_rate = self.recombination_rate
        sim.recombination_start = self.recombination_start
        sim.recombination_duration = self.recombination_duration
        sim.min_population = self.min_population
        return sim

    def freeze(self, t):
        """ Returns snapshot of culture at generation <t>. """
        sim = super().freeze(t)
        sim.division_rate = self.division_rate
        sim.recombination_rate = self.recombination_rate
        sim.recombination_start = self.recombination_start
        sim.recombination_duration = self.recombination_duration
        sim.min_population = self.min_population
        return sim

    @property
    def results(self):
        """ Returns simulation results in dictionary format. """
        return {
            'division_rate': self.division_rate,
            'recombination_rate': self.recombination_rate,
            'recombination_start': self.recombination_start,
            'recombination_duration': self.recombination_duration,
            'population': self.size,
            'transclone_edges': self.heterogeneity,
            'percent_heterozygous': self.percent_heterozygous,
            'num_clones': self.clones.num_clones,
            'clone_size_variation': self.clones.size_variation}
