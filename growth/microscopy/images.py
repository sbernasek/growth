import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from ..measure import ConditionedLognormalSampler


class ScalarImage:
    """
    Class containing a scalar image.
    """

    def __init__(self, height=1000, width=1000):
        """ Instantiate scalar image with shape (<height>, <width>). """
        self.height = height
        self.width = width
        self.initialize()

    @property
    def shape(self):
        """ Image shape. """
        return self.im.shape[-2:]

    @property
    def pixels(self):
        """ Returns image pixels. """
        return self.im.ravel()

    @property
    def max(self):
        """ Maximum pixel intensity. """
        return self.im.max()

    def percentile(self, q):
        """ 98th percentile of pixel intensities. """
        return np.percentile(self.im.ravel(), q=q)

    def initialize(self):
        """ Initialize blank image. """
        self.im = np.zeros((self.height, self.width), dtype=np.float64)

    def fill(self, mu=0.1, sigma=0.1):
        """
        Fill image background with values sampled from a lognormal distribution.

        Args:

            mu (float) - mean of underlying normal distribution

            sigma (float) - std dev of underlying normal distribution

        """
        pixels = np.exp(np.random.normal(np.log(mu), sigma, size=self.shape))
        self.im[:, :] = pixels

    @staticmethod
    def _render(im, vmin=0, vmax=None, cmap=plt.cm.Greys, size=5, ax=None):
        """
        Render image.

        Args:

            im (np.ndarray[float]) - image

            vmin, vmax (int) - colormap bounds

            cmap (matplotlib.ColorMap)

            size (int) - image panel size, in inches

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.invert_yaxis()
        ax.axis('off')

    def render(self, **kwargs):
        """ Render image. """
        self._render(self.im.T, **kwargs)

    def center_xycoords(self, xy, shrinkage=0.9):
        """ Project zero-centered coordinates to center of image. """
        center_x, center_y = self.shape[0]/2,  self.shape[1]/2
        centered_xy = deepcopy(xy)
        centered_xy[:, 0] = ((xy[:, 0]*center_x*shrinkage) + center_x)
        centered_xy[:, 1] = ((xy[:, 1]*center_y*shrinkage) + center_y)
        return centered_xy.astype(np.uint16)


class DependentScalarImage(ScalarImage):
    """
    Class defines a scalar image whose pixel intensities are sampled with some dependence upon another scalar image.
    """

    def __init__(self, pixels, mean, sigma):
        """ Instantiate a dependent scalar image. """
        super().__init__(*pixels.shape)
        x = np.log(pixels.ravel())
        self.sampler = ConditionedLognormalSampler(x, np.log(mean), sigma)

    def fill(self, rho=0.0):
        """ Generate randomly sampled pixel values. """
        pixels = self.sampler.sample(rho=rho)
        self.im[:, :] = pixels.reshape(self.shape)
