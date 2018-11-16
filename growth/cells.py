from copy import deepcopy
import numpy as np
from functools import reduce
from operator import add
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .triangulation import LocalTriangulation
from .fluorescence import Fluorescence
from .phylogeny import Phylogeny
from .animation import Animation


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

    def copy(self):
        """ Returns copy of cell. """
        return self.__class__(self.xy, self.chromosomes, self.lineage)

    def set_xy(self, xy):
        self.xy = xy

    def recombine(self, recombination=0.):

        # duplicate chromosomes
        chromosomes = np.tile(self.chromosomes, 2)

        # recombination
        if np.random.random() <= recombination:
            chromosomes.sort()

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


class CultureProperties:

    @property
    def cells(self):
        """ Current generation of cells. """
        return self.history[-1]

    @property
    def parents(self):
        """ Parent cells of current generation. """
        return self.history[-2]

    @property
    def size(self):
        """ Culture size. """
        return len(self.cells)

    @property
    def generation(self):
        return len(self.history) - 1

    @property
    def genotypes(self):
        """ Cell genotypes. """
        return np.array([cell.genotype for cell in self.cells])

    @property
    def phenotypes(self):
        """ Sampled cell phenotypes. """
        return self.fluorescence(self.genotypes)

    @property
    def xy_dict(self):
        """ Cell position dictionary keyed by cell index. """
        return {i: cell.xy for i, cell in enumerate(self.cells)}

    @property
    def xy(self):
        """ Cell positions. """
        return np.vstack([cell.xy for cell in self.cells])

    @property
    def triangulation(self):
        """ Delaunay triangulation with edge-length filtering. """
        return LocalTriangulation(*self.xy.T)

    @property
    def xy_graph(self):
        #return nx.Graph(LocalTriangulation(*xy.T).edges)
        return nx.Graph(self.triangulation.edges)

    @property
    def generations(self):
        """ List of generation numbers. """
        return [cell.generation for cell in self.cells]

    @property
    def lineages(self):
        """ List of cell lineages. """
        return [cell.lineage for cell in self.cells]

    @property
    def dendrogram_edges(self):
        """ Dendrogram edge list. """
        return reduce(add, map(self.predecessor_search, self.lineages))

    @property
    def phylogeny(self):
        """ Phylogeny. """
        return Phylogeny(self.dendrogram_edges)

    @classmethod
    def predecessor_search(cls, lineage):
        """ Returns all predecessors of <lineage>. """
        edges = [(lineage[:-1], lineage)]
        if len(lineage) > 1:
            edges += cls.predecessor_search(lineage[:-1])
        return edges

    @property
    def diversification(self):
        """ Phylogenetic distance from earliest common ancestor. """
        spread = self.size / 2.
        indices = np.argsort(self.lineages)
        return ((indices - spread) / spread)  * self.scaling


class CultureVisualization:

    def animate(self, interval=500, **kwargs):
        """ Returns animation of culture growth. """
        freeze = np.vectorize(self.freeze)
        frames = freeze(np.arange(self.generation+1))
        animation = Animation(frames)
        video = animation.get_video(interval=interval, **kwargs)
        return video

    def plot(self,
             ax=None,
             colorby='genotype',
             tri=False,
             s=30,
             cmap=plt.cm.viridis):
        """
        Scatter cells in space.

        """

        # set normalization


        # evaluate marker colors
        if colorby == 'genotype':
            norm = Normalize(0, 2)
            c = cmap(norm(self.genotypes))
        elif colorby == 'phenotype':
            norm = Normalize(0, 1)
            c = cmap(norm(self.phenotypes))
        elif colorby == 'lineage':
            norm = Normalize(-1, 1)
            c = cmap(norm(self.diversification))

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


class Culture(CultureProperties, CultureVisualization):

    def __init__(self, starter=None, fluorescence=None, scaling=1):

        # seed with four heterozygous cells
        if starter is None:
            starter = self.inoculate()
        self.history = [starter]

        # set fluorescence model
        if fluorescence is None:
            fluorescence = Fluorescence()
        self.fluorescence = fluorescence

        # set population size scaling
        self.scaling = scaling

    def __add__(self, b):
        return self.__class__(self.cells + b.cells)

    def branch(self, t=None):
        """ Returns copy of culture at generation <t> including history. """

        culture = Culture()

        if t is None:
            culture.history = self.history[:]
            culture.fluorescence = deepcopy(self.fluorescence)

        else:
            culture.history = self.history[:t+1]
            culture.fluorescence = deepcopy(self.fluorescence)

        return culture

    def freeze(self, t):
        """ Returns snapshot of culture at generation <t>. """
        cells = self.history[t]
        culture = Culture(scaling=float(len(cells))/self.size)
        culture.history = [cells]
        culture.fluorescence = deepcopy(self.fluorescence)
        return culture

    @staticmethod
    def inoculate(N=2):
        """ Inoculate with <N> generations of heterozygous cells. """
        return Cell().grow(max_generation=N)

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

        # run relaxation
        xy_dict = nx.kamada_kawai_layout(
            self.xy_graph,
            pos=dict(enumerate(self.xy)),
            center=center,
            scale=radius)

        # update cell positions
        _ = [cell.set_xy(xy_dict[i]) for i, cell in enumerate(self.cells)]

    def divide(self, division=0.1, recombination=0.1):

        # select cells for division
        divided = np.random.random(len(self.parents)) < division

        # create next generation
        for index, parent in enumerate(self.parents):

            # if cell divided, pass children to next generation
            if divided[index]:
                children = parent.divide(recombination=recombination)
                self.cells.extend(children)

            # otherwise, pass cell to next generation
            else:
                self.cells.append(parent.copy())

    def update(self, division=0.1, recombination=0.1, **kwargs):
        self.history.append([])
        self.divide(division, recombination)
        self.move(**kwargs)

    def grow(self, max_population=10, max_iters=None, **kwargs):
        i = 0
        while self.size < max_population:
            self.update(**kwargs)
            if max_iters is not None:
                i += 1
                if i >= max_iters:
                    break
