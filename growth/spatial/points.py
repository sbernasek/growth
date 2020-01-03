import numpy as np
from scipy.spatial import Delaunay, ConvexHull


class Points:
    """
    Methods operating on a set of <xy> points.

    Properties:

        xy (np.ndarray[float]) - N x 2

        hull (ConvexHull)

    """

    def __init__(self, xy):
        self.xy = xy
        self.hull = ConvexHull(xy)

    @property
    def vertex_indices(self):
        """ Indixes of convex hull vertices. """
        return self.hull.vertices

    @property
    def vertices(self):
        """ Convex hull vertices. """
        return self.hull.points[self.vertex_indices]

    @property
    def centroid(self):
        """ Area centroid. """
        return self.compute_area_centroid(self.xy)

    @property
    def hull_centroid(self):
        """ Area centroid. """
        return self.compute_area_centroid(self.vertices)

    @staticmethod
    def compute_area_centroid(points):
        """
        Compute area centroid of a set of <N> points.

        Args:

            points (np.ndarray) - N x 2

        Returns:

            centroid (np.ndarray) - 2 x 1

        """

        def moment(points):

            # Polygon's signed area, centroid's x and y
            A, C_x, C_y = 0, 0, 0
            for i in range(0, len(points) - 1):
                s = (points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1])
                A += s
                C_x += (points[i, 0] + points[i + 1, 0]) * s
                C_y += (points[i, 1] + points[i + 1, 1]) * s
            A *= 0.5
            C_x *= (1.0 / (6.0 * A))
            C_y *= (1.0 / (6.0 * A))
            return np.array([[C_x, C_y]])

        shift = points.mean(axis=0)
        return shift + moment(points-shift)

    @classmethod
    def _scale(cls, points, factor=1.0):
        """
        Scale <N> points about centroid by <factor>.

        Args:

            points (np.ndarray) - N x 2

            factor (float) - scale factor

        Returns:

            scaled (np.ndarray) - scaled points, N x 2

        """

        centroid = cls.compute_area_centroid(points).flatten()
        return np.apply_along_axis(lambda x: centroid + factor*(x-centroid), axis=1, arr=points)

    def scale(self, factor=1.0):
        """ Returns scaled points. """
        xy = self._scale(self.xy, factor)
        return Points(xy)

    def get_scale_mask(self, factor):
        """ Returns boolean mask of points outside scaled region. """
        scaled_vertices = self._scale(self.vertices, factor)
        return self._outside_hull(self.xy, scaled_vertices)

    def scale_filter(self, factor):
        """ Returns points outside scaled region. """
        mask = self.get_scale_mask(factor)
        return Points(self.xy[mask])

    @staticmethod
    def _outside_hull(points, hull):
        """
        Returns boolean mask denoting which of <N> points lie outside the convex hull defined by <M> vertices.

        Args:

            points (np.ndarray) - N x 2

            hull (np.ndarray) - M x 2

        Returns:

            outside (np.ndarray[bool]) - length N

        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(points)>=0

    def scatter(self, ax=None, **kwargs):
        """ Scatter points on <ax>. """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.axis('off')
        ax.scatter(*self.xy.T, **kwargs)
        return ax
