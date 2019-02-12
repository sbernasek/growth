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

    @property
    def im_normalized(self):
        """ Image normalized by the maximum value. """
        return self.im/self.max

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

            cmap (matplotlib.ColorMap or str) - if value is 'r', 'g', or 'b', use RGB colorscheme

            size (int) - image panel size, in inches

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(size, size))

        if vmax is None:
            vmax = im.max()

        # render image
        if type(cmap) == str:
            assert cmap in 'rgb', 'Color not recognized.'
            im_rgb = np.zeros(im.shape+(3,), dtype=np.float64)
            im_rgb[:,:,'rgb'.index(cmap)] = (im-vmin)/(vmax-vmin)
            im_rgb[im_rgb>1.] = 1.
            ax.imshow(im_rgb)
        else:
            ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)

        # invert axis and remove ticks
        ax.invert_yaxis()
        ax.axis('off')

    def render(self, **kwargs):
        """ Render image. """
        self._render(self.im.T, **kwargs)

    def render_blank(self, **kwargs):
        """ Render image. """
        self._render(np.zeros(self.shape, dtype=int), **kwargs)

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
