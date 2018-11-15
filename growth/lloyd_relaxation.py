import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
lloyd_path = '/Users/Sebi/Documents/grad_school/research/lloyd_relaxation/'
if lloyd_path not in sys.path:
    sys.path.insert(0, lloyd_path)
from modules.lloyd_relaxation import LloydRelaxation



class LloydGrowth(LloydRelaxation):

    def __init__(self,
                 xycoords,
                 division_rate=0.1,
                 max_pop=100,
                 boundary_type=None,
                 **kwargs):

        self.division_rate = division_rate
        self.max_pop = max_pop
        self.boundary_type = boundary_type
        super().__init__(xycoords, boundary_type=boundary_type, **kwargs)

        # update instantiation parameters
        init_params = dict(division_rate=division_rate,
                           max_pop=max_pop,
                           boundary_type=boundary_type)
        self.init_params.update(init_params)

    @property
    def population(self):
        return self.xycoords.shape[-1]

    def iterate(self, xy_0, bias=None):
        """
        Execute single iteration of LLoyd relaxation. Coordinates are merged with boundary, voronoi regions are
        constructed and filtered such that only those inside the bounding box remain. The centroids of retained regions
        are returned.

        Args:

            xy_0 (initial xy coordinates)

            bias (np.ndarray[float])

        Returns:
        centroids (np array) - centroids of filtered voronoi cells, ordered by position in xycoords
        """

        divide = None

        if self.population < self.max_pop:

            # double cell population and set new boundary
            divide = self.divide()

            # evaluate population increase
            population_increase = self.population / divide.size

            # set new boundary (diameter grows with square root of population)
            if population_increase > 1:
                dilation = np.sqrt(population_increase)
                self.set_boundary(self.boundary_type, dilation)

        # evaluate new centroids
        centroids = super().iterate()
        assert centroids.shape[-1] >= xy_0.shape[-1], 'Point outside boundary.'

        # apply bias
        if bias is not None:

            if divide is not None:
                repeats = np.ones(xy_0.shape[-1], dtype=int) + divide
                xy_0 = np.repeat(xy_0, repeats=repeats, axis=1)

            delta = centroids - xy_0
            step = (bias*(delta.T)).T
            xy = xy_0 + step

        else:
            xy = centroids

        # compute change and update coordinates
        self.xycoords = xy

    def run(self, bias=None, max_iters=100, tol=1e-3, record=False, reset=True):
        """
        Run Lloyd's relaxation.

        Args:
        bias (np array) - bias applied to relaxation in each direction
        max_iters (int) - maximum iterations allowed
        tol (float) - minimum percent change in coordinates for convergence
        record (bool) - if True, append each voronoi cell to history
        """

        if reset:
            self.reset()

        for i in range(max_iters):

            # store existing state
            xy_0 = deepcopy(self.xycoords)

            # run iteration
            self.iterate(xy_0, bias=bias)

            # save current voronoi cell
            if record:
                self.record()

            # check for convergence
            grew = (self.population > xy_0.shape[-1])
            if self.population >= self.max_pop and not grew:
                delta = np.abs(self.xycoords - xy_0) / np.abs(xy_0)

                if delta.sum() < tol:
                   print('Relaxation converged in {:d} iterations.'.format(i))
                   break

    def divide(self, noise=1e-5):

        # determine which cells divide
        divide = np.random.random(size=self.population) <= self.division_rate

        # apply cell division
        if divide.sum() > 0:
            repeats = np.ones(self.population, dtype=int) + divide
            self.xycoords = np.repeat(self.xycoords, repeats=repeats, axis=1)

            # add position jitter
            jitter = np.random.normal(0, noise, size=self.xycoords.shape)
            self.xycoords += jitter

        return divide

    def record(self):
        """ Record voronoi state. """
        record = dict(xy=deepcopy(self.xycoords))
        self.history.append(record)

    def plot_before_and_after(self, **kwargs):
        """ Plot voronoi regions before and after relaxation. """

        fig, axes = plt.subplots(ncols=2, figsize=(6.5, 3))

        before, after = self.history[0]['xy'], self.history[-1]['xy']
        axes[0].scatter(*before, **kwargs)
        axes[1].scatter(*after, **kwargs)

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect(1)


class LloydRecombination(LloydGrowth):

    def __init__(self,
                 xycoords,
                 recombination_rate=0.1,
                 chromosomes=None,
                 **kwargs):

        if chromosomes is None:
            dim = xycoords.shape[-1]
            chromosomes = np.vstack((np.ones(dim), np.zeros(dim))).astype(int)
        self.chromosomes = chromosomes

        self.recombination_rate = recombination_rate

        super().__init__(xycoords, **kwargs)

        # update instantiation parameters
        init_params = dict(chromosomes=chromosomes,
                           recombination_rate=recombination_rate)
        self.init_params.update(init_params)

    @property
    def genotypes(self):
        return self.chromosomes.sum(axis=0)

    @property
    def phenotypes(self):
        return np.random.normal(self.genotypes, scale=1)

    def divide(self, noise=0.01):

        # determine which cells divide
        divide = np.random.random(size=self.population) <= self.division_rate

        # apply cell division
        if divide.sum() > 0:
            repeats = np.ones(self.population, dtype=int) + divide
            self.xycoords = np.repeat(self.xycoords, repeats=repeats, axis=1)

            # add position jitter
            jitter = np.random.normal(0, noise, size=self.xycoords.shape)
            self.xycoords += jitter

            # perform recombination
            self.recombine(repeats)

        return divide

    def recombine(self, repeats):
        """
        Perform genetic recombination.
        """
        self.chromosomes = np.repeat(self.chromosomes, repeats=repeats, axis=1)
        indices = np.hstack((0, np.cumsum(repeats)[:-1]))
        divided = (repeats - 1).astype(bool)
        recombined = np.random.random(divided.sum()) < self.recombination_rate
        flipped = indices[divided][recombined]

        # flip chromosomes
        self.chromosomes[1, flipped], self.chromosomes[0, flipped+1] = self.chromosomes[0, flipped+1], self.chromosomes[1, flipped]

    def record(self):
        """ Record voronoi state. """
        record = dict(xy=deepcopy(self.xycoords),
                      genotypes=deepcopy(self.genotypes))
        self.history.append(record)

