import numpy as np
from matplotlib.tri import Triangulation


class LocalTriangulation(Triangulation):
    """
    Triangulation with edge distance filter.

    """

    def __init__(self, *args, q=100, **kwargs):
        super().__init__(*args, **kwargs)
        #self.filter_triangles(q)

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
