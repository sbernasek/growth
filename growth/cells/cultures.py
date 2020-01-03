from copy import deepcopy
import pickle
import numpy as np
from functools import reduce
from collections import Counter
from operator import add
import networkx as nx
import pandas as pd

from .patches import Patches
from .phylogeny import Phylogeny
from ..spatial.triangulation import LocalTriangulation
from ..spatial.points import Points
from ..measure import MeasurementGenerator
from ..microscopy import SyntheticMicroscopy
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
    def num_recombinant_cells(self):
        """ Number of recombinant cells. """
        return (self.genotypes != 1).sum()

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

    def parse_patches(self, genotype):
        """ Returns properties for patches of specified <genotype>.  """
        patches = self.xy_graph.subgraph(self.select(genotype))
        return {
            'number': nx.connected.number_connected_components(patches),
            'sizes': [len(c) for c in nx.connected_components(patches)],
            'nodes': [np.array(c) for c in nx.connected_components(patches)]}

    def get_patches(self, genotypes=(0, 2)):
        """ Patches instance. """
        data = {g: self.parse_patches(g) for g in genotypes}
        return Patches(data)


class CultureMeasurements:
    """ Methods for generating synthetic measurements. """

    def measure(self, ambiguity=0.1, rho=0.0, **kwargs):
        """
        Returns dataframe of synthetic measurements.

        Args:

            ambiguity (float) - fluorescence ambiguity coefficient

            rho (float) - expression capacity correlation coefficient

            kwargs: keyword arguments for measurement generator

        """
        return MeasurementGenerator(self,
            ambiguity=ambiguity,
            rho=rho,
            **kwargs).data

    def generate_microscopy(self, ambiguity, rho, bleedthrough,
                            measurement_kwargs={},
                            microscopy_kwargs={}):
        """
        Generate synthetic microscopy data.

        Args:

            ambiguity (float) - clonal marker ambiguity coefficient

            rho (float) - expression capacity correlation coefficient

            bleedthrough (float) - bleedthrough coefficient

            measurement_kwargs (dict) - keyword arguments for measurement generations

            microscopy_kwargs (dict) - keyword arguments for synthetic microscopy

        Returns:

            image (SyntheticMicroscopy) - synthetic microscopy data

        """
        data = self.measure(ambiguity, rho, **measurement_kwargs)
        return SyntheticMicroscopy(data, bleedthrough, **microscopy_kwargs)


class CloneCounting:
    """
    Methods for counting the number of coherent clones.
    """

    @staticmethod
    def get_sibling(lineage):
        """ Returns lineage of sibling. """
        if lineage[-1] == '0':
            return lineage[:-1] + '1'
        else:
            return lineage[:-1] + '0'

    @property
    def num_coherent_clones(self):
        """ Number of independent clonal lineages. """

        num_clones = 0

        # initialize genotype dictionary
        genotypes = {}
        for cell in self.cells:
            genotypes[cell.lineage] = cell.genotype

        # traverse phylogenetic tree from bottom to top
        gen = max(self.generations)
        while gen > 0:
            lineages = [k for k in genotypes.keys() if len(k) == gen]
            for lineage in lineages:
                genotype = genotypes[lineage]
                if genotypes[self.get_sibling(lineage)] == genotype:
                    genotypes[lineage[:-1]] = genotype
                else:
                    genotypes[lineage[:-1]] = 1
                    num_clones += 1
            gen -= 1

        return num_clones

    @property
    def genotype_dict(self):
        """ Dictionary mapping lineage to genotype. """

        # initialize genotype dictionary
        genotypes = {}
        for cell in self.cells:
            genotypes[cell.lineage] = cell.genotype

        # traverse phylogenetic tree from bottom to top
        gen = max(self.generations)
        while gen > 0:
            lineages = [k for k in genotypes.keys() if len(k) == gen]
            for lineage in lineages:
                genotype = genotypes[lineage]
                if genotypes[self.get_sibling(lineage)] == genotype:
                    genotypes[lineage[:-1]] = genotype
                else:
                    genotypes[lineage[:-1]] = 1
            gen -= 1

        return genotypes

    @property
    def dendrogram_edges(self):
        """ List of phylogenetic tree edges. """

        lineages = self.lineages

        # traverse phylogenetic tree from bottom to top
        edges = []

        while len(lineages) > 0:
            child = lineages.pop()
            if child == '':
                continue

            parent = child[:-1]
            edges.append((parent, child))
            lineages.append(parent)

        return list(set(edges))

    @property
    def dendrogram(self):
        """ Phylogenetic tree. """
        return nx.Graph(self.dendrogram_edges)

    def get_clones(self):
        """ Returns list of recombinant clones. """

        genotypes = self.genotype_dict
        G = self.dendrogram
        G_0 = G.subgraph([k for k,v in genotypes.items() if v == 0])
        G_2 = G.subgraph([k for k,v in genotypes.items() if v == 2])

        clones = []
        clones += [c for c in nx.connected_components(G_0)]
        clones += [c for c in nx.connected_components(G_2)]

        leaf_nodes = set(self.lineages)

        return [clone.intersection(leaf_nodes) for clone in clones]

    def get_patches_list(self):
        """ Returns list of patches. """

        G_0 = self.xy_graph.subgraph(self.select(0))
        G_2 = self.xy_graph.subgraph(self.select(2))

        clones = []
        clones += [c for c in nx.connected_components(G_0)]
        clones += [c for c in nx.connected_components(G_2)]

        return clones

    def get_clone_sizes(self, factor=None):
        """
        Returns number of cells per coherent clone, with optional scaling factor used to exclude border region.
        """

        if factor == 1.0 or factor is None:
            return [len(c) for c in self.get_clones()]

        # compile scaling mask
        mask = Points(self.xy).get_scale_mask(factor)
        included = set(np.array(self.lineages)[mask])

        # filter clone size list
        sizes = [len(c.intersection(included)) for c in self.get_clones()]

        return [s for s in sizes if s > 0]

    @property
    def mean_clone_size(self):
        """ Mean number of cells per coherent clone. """
        return self.num_recombinant_cells / self.num_coherent_clones

    @property
    def num_patches(self):
        """ Number of distinct recombinant patches. """
        return self.get_patches((0,2)).num_patches

    @property
    def mean_patch_size(self):
        """ Mean number of cells per distinct recombinant patch. """
        return self.get_patches((0,2)).mean_patch_size

    @property
    def clone_sizes_per_patch(self):
        """ Clone sizes per distinct recombinant patch. """

        to_clone = {}
        for clone_id, members in enumerate(self.get_clones()):
            for member in members:
                to_clone[member] = clone_id

        patches_list = self.get_patches_list()

        clone_sizes = []
        for patches in patches_list:
            clone_ids = [to_clone[self.lineages[patch]] for patch in patches]
            clone_sizes += list(Counter(clone_ids).values())

        return clone_sizes


class Culture(CultureProperties,
              CultureVisualization,
              CultureMeasurements,
              CloneCounting):

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

    def filter_edges(self, factor=1.0):
        """
        Returns culture with edge cells excluded.

        Args:

            factor (float) - scale factor above which cells are excluded

        """

        # compile scaling mask
        mask = Points(self.xy).get_scale_mask(factor)
        cells = np.array(self.cells)[mask]

        # instantiate
        child = Culture(starter=cells, scaling=self.scaling, reference_population=self.reference_population)

        return child

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
