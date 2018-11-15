
import numpy as np
from functools import reduce
from operator import add
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize


class Cell:

    def __init__(self, xy=None, chromosomes=None, lineage=''):

        # set generation
        self.lineage = lineage

        # set chromosomes
        if chromosomes is None:
            chromosomes = np.array([0, 1])
        self.chromosomes = chromosomes

        # set position
        if xy is None:
            xy = np.zeros(2, dtype=float)
        self.xy = xy

    @property
    def generation(self):
        return len(self.lineage)

    @property
    def genotype(self):
        return self.chromosomes.sum()

    @property
    def phenotype(self):
        return np.random.normal(loc=self.genotype, scale=1.)

    def set_xy(self, xy):
        self.xy = xy

    def recombine(self, recombination=0.):

        # duplicate chromosomes
        chromosomes = np.tile(self.chromosomes, 2)

        # recombination
        if np.random.random() <= recombination:
            chromosomes.sort()
            #np.random.shuffle(chromosomes)

        return chromosomes

    def divide(self, recombination=0., tol=1e-5):

        # perform recombination
        chromosomes = self.recombine(recombination=recombination)

        # determine child positions
        jitter = np.random.normal(scale=tol, size=(2, 2))
        xy_a, xy_b = self.xy+jitter[0], self.xy+jitter[1]

        # instantiate children
        daughter_a = self.__class__(xy_a, chromosomes[:2], self.lineage+'0')
        daughter_b = self.__class__(xy_b, chromosomes[2:], self.lineage+'1')

        return [daughter_a, daughter_b]

    def grow(self, max_generation=3, **kwargs):
        """
        Recursive growth.
        """

        # stopping criterion
        if self.generation >= max_generation:
            return [self]

        # divide
        else:
            children = self.divide(**kwargs)
            recurse = lambda x: x.grow(max_generation=max_generation, **kwargs)
            return reduce(add, map(recurse, children))


class Population:

    def __init__(self, cells):
        self.cells = cells

    def __add__(self, b):
        return self.__class__(self.cells + b.cells)

    @property
    def size(self):
        return len(self.cells)

    @property
    def genotypes(self):
        return np.array([cell.genotype for cell in self.cells])

    @property
    def phenotypes(self):
        return np.array([cell.phenotype for cell in self.cells])

    @property
    def xy_dict(self):
        return {i: cell.xy for i, cell in enumerate(self.cells)}

    @property
    def xy(self):
        return np.vstack([cell.xy for cell in self.cells])

    @property
    def generations(self):
        return [cell.generation for cell in self.cells]

    @property
    def lineages(self):
        return [cell.lineage for cell in self.cells]

    @classmethod
    def build_edges(cls, lineage):
        edges = [(lineage[:-1], lineage)]
        if len(lineage) > 1:
            edges += cls.build_edges(lineage[:-1])
        return edges

    @staticmethod
    def build_graph(xy):
        G = nx.Graph()
        G.add_edges_from(Triangulation(*xy.T).edges)
        return G

    @property
    def dendrogram(self):
        return reduce(add, map(self.build_edges, self.lineages))

    def move(self, k=None, iterations=500, center=None):

        # fix centerpoint
        #if center is None:
        #    center = np.zeros(2, dtype=float)

        if k is None:
            k = np.sqrt(self.size)

        # build graph
        xy = self.xy
        G = self.build_graph(xy)

        # run graph relaxation
        # xy_dict = nx.fruchterman_reingold_layout(
        #     G,
        #     k=k,
        #     pos=dict(enumerate(xy)),
        #     iterations=iterations,
        #     center=center)

        xy_dict = nx.kamada_kawai_layout(
            G,
            pos=dict(enumerate(xy)),
            center=center)


        # update cell positions
        _ = [cell.set_xy(xy_dict[i]) for i, cell in enumerate(self.cells)]

    def divide(self, division=0.1, recombination=0.1):

        # select cells for division
        divided = np.random.random(size=self.size) < division

        # replace parent with children
        for index in divided.nonzero()[0]:
            parent = self.cells.pop(index)
            children = parent.divide(recombination=recombination)
            self.cells.extend(children)

    def step(self, division=0.1, recombination=0.1, iterations=500):
        self.divide(division, recombination)
        self.move(iterations=iterations)

    def grow(self, max_population=10, **kwargs):
        while self.size < max_population:
            self.step(**kwargs)

    def plot(self, ax=None, s=50, cmap=plt.cm.viridis):

        # set normalization
        norm = Normalize(0, 2)

        # create and format figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect(1)

        # scatter points
        ax.scatter(*self.xy.T, s=s, lw=0, c=cmap(norm(self.genotypes)))







