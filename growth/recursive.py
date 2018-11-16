from copy import deepcopy
import numpy as np
from functools import reduce
from operator import add
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize

from .fluorescence import Fluorescence


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

    def divide(self, recombination=0., reference_population=1000):

        # set average spacing between cells
        spacing = np.sqrt(2/reference_population) / 1e5

        # perform recombination
        chromosomes = self.recombine(recombination=recombination)

        # determine child positions
        jitter = np.random.normal(scale=spacing, size=(2, 2))
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

    def __init__(self, cells, fluorescence=None):

        # set fluorescence model
        if fluorescence is None:
            fluorescence = Fluorescence()
        self.fluorescence = fluorescence

        self.cells = cells
        self.history = []
        self.move()
        self.record()

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
        return self.fluorescence(self.genotypes)

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

    @property
    def triangulation(self):
        return LocalTriangulation(*self.xy.T)

    @classmethod
    def build_edges(cls, lineage):
        edges = [(lineage[:-1], lineage)]
        if len(lineage) > 1:
            edges += cls.build_edges(lineage[:-1])
        return edges

    @staticmethod
    def build_graph(xy):
        G = nx.Graph()
        G.add_edges_from(LocalTriangulation(*xy.T).edges)
        return G

    @property
    def dendrogram(self):
        return reduce(add, map(self.build_edges, self.lineages))

    def move(self, center=None, reference=1000):
        """
        Update cell positions.

        Args:

            center (np.ndarray[float]) - center position

            reference (int) - number of cells in unit circle

        """

        # fix centerpoint
        if center is None:
           center = np.zeros(2, dtype=float)

        # determine scaling (colony radius)
        radius = np.sqrt(self.size/reference)

        # build graph
        xy = self.xy
        G = self.build_graph(xy)

        # run relaxation
        xy_dict = nx.kamada_kawai_layout(
            G,
            pos=dict(enumerate(xy)),
            center=center,
            scale=radius)

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

    def update(self, division=0.1, recombination=0.1, record=True, **kwargs):
        self.divide(division, recombination)
        self.move(**kwargs)
        if record:
            self.record()

    def grow(self, max_population=10, max_iters=None, **kwargs):
        i = 0
        while self.size < max_population:
            self.update(**kwargs)
            if max_iters is not None:
                i += 1
                if i >= max_iters:
                    break

    def record(self):
        """ Record voronoi state. """
        record = dict(xy=deepcopy(self.xy),
                      genotypes=deepcopy(self.genotypes))
        self.history.append(record)

    def plot(self,
             ax=None,
             colorby='genotype',
             tri=False,
             s=30,
             cmap=plt.cm.viridis):

        # set normalization
        norm = Normalize(0, 2)

        # evaluate marker colors
        if colorby == 'genotype':
            c = cmap(norm(self.genotypes))
        elif colorby == 'phenotype':
            c = cmap(norm(self.phenotypes))
        elif colorby == 'lineage':
            c = None

        # create and format figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect(1)
            ax.axis('off')

        # add triangulation
        if tri:
            ax.triplot(self.triangulation, 'r-', lw=1, alpha=1, zorder=0)

        # scatter points
        ax.scatter(*self.xy.T, s=s, lw=0, c=c)


class LocalTriangulation(Triangulation):
    """
    Triangulation with edge distance filter.

    """

    def __init__(self, *args, q=100, **kwargs):
        super().__init__(*args, **kwargs)
        #self.filter_triangles(q)

    # @property
    # def edges(self):
    #     edges = super().edges

    #     # get extra edges to reconnect floating nodes
    #     disconnected = self.disconnected
    #     if len(disconnected) > 0:
    #         reconnecting = lambda x: len(disconnected.intersection(x)) > 0
    #         extra = np.array(list(filter(reconnecting, self.filter_edges())))
    #         edges = np.vstack((edges, extra))

    #     return edges

    @property
    def edges(self):
        return self.filter_edges(max_length=0.1)

    @property
    def disconnected(self):
        """ Disconnected nodes. """
        all_nodes = np.unique(self.triangles)
        included_nodes = np.unique(self.triangles[self.mask])
        return set(all_nodes).difference(included_nodes)

    def _evaluate_edge_lengths(self):
        """ Returns max edge length per triangle. """
        merge = lambda x: np.hstack((x, x.sum(axis=1).reshape(-1, 1)))
        dx = np.diff(self.x[self.triangles], axis=1)
        dy = np.diff(self.y[self.triangles], axis=1)
        return np.sqrt((merge(dx)**2) + (merge(dy)**2))

    def _evaluate_max_edge_lengths(self):
        """ Returns max edge length per triangle. """
        return self._evaluate_edge_lengths().max(axis=1)

    def filter_triangles(self, q=95):
        """
        Mask edges with lengths exceeding specified quantile.

        Args:

            q (float) - length quantile, 0 to 100

        """
        max_lengths = self._evaluate_max_edge_lengths()
        mask = max_lengths > np.percentile(max_lengths, q=q)
        self.set_mask(mask)

    def filter_edges(self, max_length=0.1):

        # build accepted edge mask
        mask = self._evaluate_edge_lengths() <= max_length

        # define filter function
        node = lambda x, y: self.triangles[mask[:, x], y]

        # filter edges
        edges = []
        for i in range(3):
            edges += list(zip(node(i, i), node(i, (i+1)%3)))

        return edges
