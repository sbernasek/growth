from os.path import join, isdir
from os import mkdir

from ..cells.cultures import Culture


class GrowthSimulation(Culture):

    def __init__(self,
                 division=0.1,
                 recombination=0.1,
                 recombination_start=0,
                 recombination_duration=100,
                 final_population=1000,
                 **kwargs):

        # define initial recombination rate
        if recombination_start <= 0:
            initial_recombination = recombination
        else:
            initial_recombination = 0.

        super().__init__(recombination=initial_recombination, **kwargs)
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
        pop0 = self.recombination_start
        pop1 = pop0 + self.recombination_duration
        pop2 = self.final_population

        # growth before recombination
        self.grow(min_population=pop0,
                 division=self.division,
                 recombination=0.,
                 reference_population=self.final_population)

        # growth with recombination
        self.grow(min_population=pop1,
                 division=self.division,
                 recombination=self.recombination,
                 reference_population=self.final_population)

        # growth after recombination
        self.grow(min_population=pop2,
                 division=self.division,
                 recombination=0.,
                 reference_population=self.final_population)

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
