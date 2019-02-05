import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import measurements
from skimage.morphology import disk

from .images import ScalarImage, DependentScalarImage
from .nucleus import Nucleus


class SyntheticMicroscopy(ScalarImage):
    """
    Class allows for construction of a synthetic microscope image given a set of synthetic measurements.
    """

    def __init__(self, data,
                 bleedthrough=0.0,
                 bg_level=0.5,
                 bg_noise=0.3,
                 radius=6,
                 height=1000,
                 width=1000):
        """
        Instantiate synthetic image from a set of synthetic measurements.

        Args:

            data (pd.DataFrame) - synthetic measurements for each nucleus

            bleedthrough (float) - bleedthrough coefficient, implemented as the pseudo correlation coefficient between the clonal marker and the control protein channel

            bg_level (float) - background level (mean of log-transformed level)

            bg_noise (float) - background noise (std dev of log-transformed level)

            radius (int) - nuclear radius, in pixels

            height, width (int) - image dimensions, in pixels

        """

        self.data = data

        # instantiate a 3-layer image
        self.depth = 3
        super().__init__(height=height, width=width)

        # set bleedthrough coefficient, background level, and background noise
        self.rho = bleedthrough
        self.bg_level = bg_level
        self.bg_noise = bg_noise
        self.radius = radius

        # define nuclear positions and clonal marker dosages
        centroids = data[['centroid_x', 'centroid_y']].values
        self.centroids = self.center_xycoords(centroids)
        self.dosages = data['true_dosage'].values

    def __getitem__(self, idx):
        """ Returns ScalarImage indexed by <idx>. """
        return self.im[idx]

    @property
    def num_nuclei(self):
        """ Number of nuclei. """
        return self.dosages.size

    def initialize(self):
        """ Initialize blank image. """
        self.im = np.zeros((self.depth, self.height, self.width), dtype=np.float64)

    @property
    def segmentation(self):
        """ Segment label mask. """
        mask = np.ones((self.shape), dtype=np.int64) * -1
        mask[tuple(self.centroids.T)] = np.arange(self.num_nuclei)
        return maximum_filter(mask, footprint=disk(self.radius))

    @property
    def foreground(self):
        """ Foreground mask. """
        mask = np.zeros((self.shape), dtype=bool)
        mask[tuple(self.centroids.T)] = True
        return binary_dilation(mask, structure=disk(self.radius))

    @property
    def foreground_pixels(self):
        """ Return all pixels from foreground. """
        return self.im[np.stack((self.foreground,)*3)]

    def extract_foreground(self, channel):
        """ Return all pixels from foreground of specified <channel>. """
        return self.im[channel][self.foreground]

    def extract_background(self, channel):
        """ Return all pixels from background of specified <channel>. """
        return self.im[channel][~self.foreground]

    def measure(self, channel):
        """ Measure level in specified synthetic image channel. """
        labels = self.segmentation
        index = np.arange(self.num_nuclei)
        return measurements.mean(self.im[channel], labels=labels, index=index)

    def fill(self, channel, mu=0.1, sigma=0.1):
        """ Fill background of specified channel with values sampled from a lognormal distribution. """
        pixels = np.exp(np.random.normal(np.log(mu), sigma, size=self.shape))
        self.im[channel, :, :] = pixels

    def draw_nuclei(self, channel, means, stds):
        """
        Add individal nuclei to specified channel of the image.

        Args:

            channel (int) - image channel in which nuclei are drawn

            means (np.ndarray[float]) - mean fluorescence level for each nucleus

            stds (np.ndarray[float]) - std dev of log-transformed pixel values within each nucleus

        """
        im = self.im[channel]
        for i in range(self.num_nuclei):
            xy = self.centroids[i]
            nucleus = Nucleus(xy, means[i], stds[i], radius=self.radius)
            nucleus.draw(im, replace=True)

    def draw_nuclear_stain(self):
        """ Draw synthetic nuclear stain. """

        # add background noise
        self.fill(0, self.bg_level, self.bg_noise)

        # draw nuclei
        means = self.data['nuclear_stain'].values
        stds = self.data['nuclear_stain_std'].values
        self.draw_nuclei(channel=0, means=means, stds=stds)

    def draw_clonal_marker(self):
        """ Draw synthetic clonal marker. """

        # add background noise
        self.fill(1, self.bg_level, self.bg_noise)

        # draw nuclei
        means = self.data['clonal_marker'].values
        stds = self.data['clonal_marker_std'].values
        self.draw_nuclei(channel=1, means=means, stds=stds)

    def draw_control(self):
        """ Draw synthetic control protein. """

        # draw nuclei
        means = self.data['control'].values
        stds = self.data['control_std'].values
        self.draw_nuclei(channel=2, means=means, stds=stds)

        # add bleedthrough
        self.add_bleedthrough(1, 2, rho=self.rho)

    def add_correlated_fluorescence(self, src, dst, rho=1.):
        """
        Add fluorescence bleedthrough from <src> to <dst>.

        Args:

            src (int) - source channel

            dst (int) - destination channel

            rho (float) - approximate correlation coefficient

        """
        bleed = DependentScalarImage(self.im[src])
        bleed.fill(mu=self.bg_level, sigma=self.bg_noise, rho=rho)
        self.im[dst] = self.im[dst] + bleed.im

    def add_bleedthrough(self, src, dst, rho=1.):
        """
        Add fluorescence bleedthrough from <src> to <dst>.

        Args:

            src (int) - source channel

            dst (int) - destination channel

            rho (float) - approximate correlation coefficient

        """

        # sample background pixels
        mu, sigma = np.log(self.bg_level), self.bg_noise
        background = np.exp(np.random.normal(mu, sigma, size=self.shape))

        # evaluate bleed
        bleed = self.im[src]
        # src_pixels = np.log(self.im[src])
        # src_zscore = (src_pixels - src_pixels.mean()) / src_pixels.std()
        # bleed = np.exp(mu + (src_zscore * sigma))

        self.im[dst] = self.im[dst] + (self.rho*bleed+(1-self.rho)*background)

    def fill(self, channel, mu=0.1, sigma=0.1):
        """ Fill background of specified channel with values sampled from a lognormal distribution. """
        pixels = np.exp(np.random.normal(np.log(mu), sigma, size=self.shape))
        self.im[channel, :, :] = pixels

    def draw(self):
        """ Draw all images. """
        self.draw_nuclear_stain()
        self.draw_clonal_marker()
        self.draw_control()

    def render(self, size=8, label=True, vmax=None, **kwargs):
        """ Render all image channels. """

        # if no color scale is specified, use the 95th percentile of foreground
        if vmax is None:
            vmax = np.percentile(self.foreground_pixels, q=95)

        figsize = (3*size+.25, size)
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=figsize)
        self._render(self.im[0].T, ax=ax0, vmax=vmax, **kwargs)
        self._render(self.im[1].T, ax=ax1, vmax=vmax, **kwargs)
        self._render(self.im[2].T, ax=ax2, vmax=vmax, **kwargs)

        if label:
            ax0.set_title('Nuclear Marker')
            ax1.set_title('Clonal Marker')
            ax2.set_title('Bleedthrough Control')
