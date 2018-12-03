from copy import deepcopy
import pickle
import numpy as np
from functools import reduce
from operator import add
import networkx as nx
import pandas as pd

from .clones import Clones
from .phylogeny import Phylogeny
from ..spatial.triangulation import LocalTriangulation
from ..measure.fluorescence import Fluorescence
from ..visualization.culture import CultureVisualization
from .cells import Cell


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
    def radius(self):
        """ Approximate colony radius. """
        return (self.size/self.reference_population/np.pi)**0.5

    @property
    def circumference(self):
        """ Approximate colony circumference. """
        return 2 * np.pi * self.radius

    @property
    def cell_radius(self):
        """ Mean half-distance between adjacent cells. """
        return ((1/self.reference_population)**0.5)

    @property
    def generation(self):
        return len(self.history) - 1

    @property
    def genotypes(self):
        """ Cell genotypes. """
        return np.array([cell.genotype for cell in self.cells])

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
        """ Graph of locally adjacent cells. """
        return nx.Graph(self.triangulation.edges.tolist())

    @property
    def weighted_xy_graph(self):
        """ Graph of adjacent cells, weighted by genetic similarity. """

        # compile edge weights
        weighting = 0.1
        edge_genotypes = self.genotypes[self.triangulation.edge_list]
        weighted = (np.diff(edge_genotypes, axis=1).ravel()!=0).astype(float)
        weights = np.ones(weighted.size, dtype=np.float64) + weighted*weighting

        # compile weighted edge list
        edge_list = self.triangulation.edges.tolist()
        edges = [edge+[{'weight': w}] for edge, w in zip(edge_list, weights)]

        return nx.Graph(edges)

    @property
    def labeled_graph(self):
        """ Graph of locally adjacent cells including cell genotypes. """
        G = self.xy_graph
        _ = [G.add_nodes_from(self.select(x), genotype=x) for x in range(3)]
        return G

    @property
    def heterogeneity(self):
        """ Returns fraction of edges that connect differing genotypes. """
        edges = np.array(self.xy_graph.edges)
        num_edges = np.not_equal(*self.genotypes[edges].T).sum()
        return num_edges / self.size

    @property
    def percent_heterozygous(self):
        """ Fraction of population with heterozygous chromosomes. """
        return (self.genotypes==1).sum() / self.size

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

    def select(self, genotype):
        """ Returns indices of cells with <genotype>.  """
        return (self.genotypes==genotype).nonzero()[0]

    def parse_clones(self, genotype):
        """ Returns properties for clones of specified <genotype>.  """
        clones = self.xy_graph.subgraph(self.select(genotype))
        return {
            'number': nx.connected.number_connected_components(clones),
            'sizes': [len(c) for c in nx.connected_components(clones)],
            'nodes': [np.array(c) for c in nx.connected_components(clones)]}

    @property
    def clones(self):
        """ Clones instance. """
        data = {genotype: self.parse_clones(genotype) for genotype in [0, 2]}
        return Clones(data)


class CultureMeasurements:
    """ Methods for generating fluorescence measurements. """

    def measure(self, scale=10):
        """ Returns clones-compatible dataframe. """
        fluorescence = Fluorescence.from_scale(scale)
        df = pd.DataFrame(self.xy, columns=['centroid_x', 'centroid_y'])
        df['ground'] = self.genotypes
        df['fluorescence'] = fluorescence(self.genotypes)
        return df


class Culture(CultureProperties, CultureVisualization, CultureMeasurements):

    def __init__(self,
                 starter=None,
                 scaling=1,
                 reference_population=1000,
                 **kwargs):
        """
        Args:

            reference_population (int) - number of cells in unit circle

        """

        # seed with four heterozygous cells
        if starter is None:
            starter = self.inoculate(**kwargs)
        self.history = [starter]

        # set population size scaling
        self.scaling = scaling
        self.reference_population = reference_population

    def __add__(self, b):
        return self.__class__(self.cells + b.cells)

    def save(self, filepath, save_history=True):
        """ Save pickled object to <path>. """

        # get object to be saved
        if save_history:
            obj = self
        else:
            obj = self.freeze(-1)

        # save object
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file, protocol=-1)

    @staticmethod
    def load(filepath):
        """ Load pickled instance from <path>. """
        with open(filepath, 'rb') as file:
            instance = pickle.load(file)
        return instance

    def branch(self, t=None):
        """ Returns copy of culture at generation <t> including history. """

        culture = self.__class__(
            scaling=self.scaling,
            reference_population=self.reference_population)

        # assign history to culture
        if t is None:
            culture.history = self.history[:]
        else:
            culture.history = self.history[:t+1]

        return culture

    def freeze(self, t):
        """ Returns snapshot of culture at generation <t>. """
        cells = self.history[t]
        culture = self.__class__(scaling=float(len(cells))/self.size)
        culture.history = [cells]
        return culture

    @staticmethod
    def inoculate(N=2, **kwargs):
        """ Inoculate with <N> generations of heterozygous cells. """
        return Cell().grow(max_generation=N, **kwargs)

    def move(self, center=None, weight='weight'):
        """
        Update cell positions.

        Args:

            center (np.ndarray[float]) - center position

        """

        # fix centerpoint
        if center is None:
           center = np.zeros(2, dtype=float)

        # determine scaling (colony radius)
        radius = np.sqrt(self.size/self.reference_population)

        # build graph
        if weight is not None:
            graph = self.weighted_xy_graph
        else:
            graph = self.xy_graph

        # run relaxation
        xy_dict = nx.kamada_kawai_layout(
            graph,
            pos=dict(enumerate(self.xy)),
            center=center,
            scale=radius,
            weight=weight)

        # update cell positions
        _ = [cell.set_xy(xy_dict[i]) for i, cell in enumerate(self.cells)]

    def divide(self, division_rate=0.1, recombination_rate=0.1):

        # select cells for division
        divided = np.random.random(len(self.parents)) < division_rate

        # create next generation
        for index, parent in enumerate(self.parents):

            # if cell divided, pass children to next generation
            if divided[index]:
                children = parent.divide(recombination_rate=recombination_rate)
                self.cells.extend(children)

            # otherwise, pass cell to next generation
            else:
                self.cells.append(parent.copy())

    def update(self,
               division_rate=0.1,
               recombination_rate=0.1,
               **kwargs):
        self.history.append([])
        self.divide(division_rate, recombination_rate)
        self.move(**kwargs)

    def grow(self,
             min_population=10,
             max_iters=None,
             **kwargs):
        i = 0
        while self.size < min_population:
            self.update(**kwargs)
            if max_iters is not None:
                i += 1
                if i >= max_iters:
                    break
