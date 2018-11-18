import numpy as np
from matplotlib.tri import Triangulation


class LocalTriangulation(Triangulation):
    """
    Triangulation with edge distance filter.

    """

    def __init__(self, *args, max_length=0.1, **kwargs):

        # call matplotlib.tri.Triangulation instantiation
        super().__init__(*args, **kwargs)

        # compile edges
        edge_list = self.compile_edge_list()
        edge_lengths = self.evaluate_edge_lengths(edge_list, self.x, self.y)

        # set max_length attribute
        self.max_length = max_length

        # sort edges
        sort_indices = np.argsort(edge_lengths)
        self.edge_list = edge_list[sort_indices]
        self.edge_lengths = edge_lengths[sort_indices]

    @property
    def nodes(self):
        """ All nodes. """
        return np.unique(self.triangles)

    @property
    def edges(self):
        """ Filtered edges. """
        return self.filter_edges(self.nodes, self.edge_list, self.edge_lengths, max_length=self.max_length)

    def compile_edge_list(self):
        """ Returns list of (node_from, node_to) tuples. """
        edges = []
        for i in range(3):
            edges += list(zip(self.triangles[:, i], self.triangles[:, (i+1)%3]))
        return np.array(edges)

    @staticmethod
    def evaluate_edge_lengths(edge_list, x, y):
        """ Returns 1D array of edge lengths. """
        dx, dy = x[edge_list], y[edge_list]
        return np.sqrt(np.diff(dx, axis=1)**2 + np.diff(dy, axis=1)**2).reshape(-1)

    @staticmethod
    def find_disconnected_nodes(nodes, edges):
        """ Returns boolean array of nodes not included in edges. """
        return nodes[~np.isin(nodes, np.unique(edges))]

    @staticmethod
    def find_first_edge(edges, node):
        """ Returns index of first edge containing <node>. """
        return (edges==node).any(axis=1).nonzero()[0][0]

    @classmethod
    def filter_edges(cls, nodes, edges, lengths, max_length=0.1):
        """ Returns all edges less than <max_length>, with at least one edge containing each node. """

        mask = (lengths <= max_length)
        rejected, accepted = edges[~mask], edges[mask]

        # find disconnected nodes
        disconnected = cls.find_disconnected_nodes(nodes, accepted)

        # add shortest edge for each disconnected node
        if disconnected.size > 0:
            f = np.vectorize(lambda node: cls.find_first_edge(rejected, node))
            connecting = rejected[f(disconnected)]
            accepted = np.vstack((accepted, connecting))

        return accepted




# class LocalTriangulation(Triangulation):
#     """
#     Triangulation with edge distance filter.

#     """

#     def __init__(self, *args, q=100, **kwargs):
#         super().__init__(*args, **kwargs)
#         #self.filter_triangles(q)

#     @property
#     def edges(self):
#         return self.filter_edges(max_length=0.1)

#     @property
#     def disconnected(self):
#         """ Disconnected nodes. """
#         all_nodes = np.unique(self.triangles)
#         included_nodes = np.unique(self.triangles[self.mask])
#         return set(all_nodes).difference(included_nodes)

#     def _evaluate_edge_lengths(self):
#         """ Returns max edge length per triangle. """
#         merge = lambda x: np.hstack((x, x.sum(axis=1).reshape(-1, 1)))
#         dx = np.diff(self.x[self.triangles], axis=1)
#         dy = np.diff(self.y[self.triangles], axis=1)
#         return np.sqrt((merge(dx)**2) + (merge(dy)**2))

#     def _evaluate_max_edge_lengths(self):
#         """ Returns max edge length per triangle. """
#         return self._evaluate_edge_lengths().max(axis=1)

#     def filter_triangles(self, q=95):
#         """
#         Mask edges with lengths exceeding specified quantile.

#         Args:

#             q (float) - length quantile, 0 to 100

#         """
#         max_lengths = self._evaluate_max_edge_lengths()
#         mask = max_lengths > np.percentile(max_lengths, q=q)
#         self.set_mask(mask)

#     def filter_edges(self, max_length=0.1):

#         # build accepted edge mask
#         mask = self._evaluate_edge_lengths() <= max_length

#         # define filter function
#         node = lambda x, y: self.triangles[mask[:, x], y]

#         # filter edges
#         edges = []
#         for i in range(3):
#             edges += list(zip(node(i, i), node(i, (i+1)%3)))

#         return edges
