from os.path import join, isdir
from os import mkdir
from functools import reduce
from operator import add

from ..cells.cultures import Culture


class GrowthSimulation(Culture):

    def __init__(self,
                 division=0.1,
                 recombination=0.1,
                 recombination_start=0,
                 recombination_duration=8,
                 final_population=11,
                 **kwargs):

        # define seed
        seed = [Cell(**kwargs)]
        for generation in range(2):
            if recombination_start == generation:
                seed = reduce(add, [c.grow(recombination=recombination) for c in seed])
            else:
                seed = reduce(add, [c.grow(recombination=0.) for c in seed])

        super().__init__(starter=seed, **kwargs)
        self.division = division
        self.recombination = recombination
        self.recombination_start = recombination_start
        self.recombination_duration = recombination_duration
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
        """ Run growth simulation. """

        # define population windows
        pop0 = int(2**self.recombination_start)
        pop1 = int(2**(self.recombination_start+self.recombination_duration))
        pop2 = int(2**self.final_population)
        if pop1 > pop2:
            pop1 = pop2

        # growth before recombination
        self.grow(min_population=pop0,
                 division=self.division,
                 recombination=0.,
                 reference_population=pop2)

        # growth with recombination
        self.grow(min_population=pop1,
                 division=self.division,
                 recombination=self.recombination,
                 reference_population=pop2)

        # growth after recombination
        self.grow(min_population=pop2,
                 division=self.division,
                 recombination=0.,
                 reference_population=pop2)

    def branch(self, t=None):
        """ Returns copy of culture at generation <t> including history. """
        sim = super().branch(t)
        sim.division = self.division
        sim.recombination = self.recombination
        sim.recombination_start = self.recombination_start
        sim.recombination_duration = self.recombination_duration
        sim.final_population = self.final_population
        return sim

    def freeze(self, t):
        """ Returns snapshot of culture at generation <t>. """
        sim = super().freeze(t)
        sim.division = self.division
        sim.recombination = self.recombination
        sim.recombination_start = self.recombination_start
        sim.recombination_duration = self.recombination_duration
        sim.final_population = self.final_population
        return sim
