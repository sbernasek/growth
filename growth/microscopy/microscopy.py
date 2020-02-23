import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import measurements

from .images import ScalarImage, DependentScalarImage
from .nucleus import Nucleus, NucleusLabel, disk


class SyntheticImage(ScalarImage):
    """
    Class allows for construction of a synthetic microscope image of an individual reporter given a set of synthetic measurements.


    Attributes:

        data (pd.DataFrame) - position and mean expression level of each cell

        centroids (np.ndarray[int]) - nuclear positions in image

        radius (int) - nuclear radius, in pixels

        bg_level (float) - background level (mean of log-transformed level)

        bg_noise (float) - background noise (std dev of log-transformed level)

        height, width (int) - image dimensions, in pixels

    """

    def __init__(self, data,
                 bg_level=0.5,
                 bg_noise=0.3,
                 radius=6,
                 height=1000,
                 width=1000):
        """
        Instantiate synthetic image from a set of synthetic measurements.

        Args:

            data (pd.DataFrame) - position and mean expression level of each cell

            bg_level (float) - background level (mean of log-transformed level)

            bg_noise (float) - background noise (std dev of log-transformed level)

            radius (int) - nuclear radius, in pixels

            height, width (int) - image dimensions, in pixels

        """

        # instantiate a scalar image
        super().__init__(height=height, width=width)

        # store data
        self.data = data

        # set background level and background noise
        self.bg_level = bg_level
        self.bg_noise = bg_noise

        # define nuclear positions
        self.centroids = self.center_xycoords(data[['x', 'y']].values)
        self.data['centroid_x'] = self.centroids[:, 0]
        self.data['centroid_y'] = self.centroids[:, 1]

        # set nuclear radius
        self.radius = radius

    @staticmethod
    def from_channel(data, im, **kwargs):
        """ Instantiate from existing image. """
        image = SyntheticImage(data, **kwargs)
        if len(im.shape) == 2:
            im = im.reshape(1, *im.shape)
        image.im = im
        return image

    @property
    def num_channels(self):
        """ Number of fluorescence channels. """
        return self.im.shape[0]

    @property
    def num_nuclei(self):
        """ Number of nuclei. """
        return len(self.data)

    @property
    def segmentation(self):
        """ Segment label mask. """
        mask = np.ones((self.shape), dtype=np.int64) * -1
        mask[tuple(self.centroids.T)] = np.arange(self.num_nuclei)
        return maximum_filter(mask, footprint=disk(self.radius))

    @property
    def foreground_mask(self):
        """ Foreground mask. """
        mask = np.zeros((self.shape), dtype=bool)
        mask[tuple(self.centroids.T)] = True
        return binary_dilation(mask, structure=disk(self.radius))

    @property
    def foreground_pixels(self):
        """ Return all pixels from foreground. """
        return self.extract_foreground_pixels(0)

    @property
    def background_pixels(self):
        """ Return all pixels from background. """
        return self.extract_background_pixels(0)

    def extract_foreground_pixels(self, channel=0):
        """ Returns all pixels from foreground of <channel>. """
        return self.im[channel][self.foreground_mask]

    def extract_background_pixels(self, channel=0):
        """ Returns all pixels from background of <channel>. """
        return self.im[channel][~self.foreground_mask]

    @staticmethod
    def sample_radii(n, mu=250):
        """
        Randomly generate radii for <n> nuclei by sampling their areas from a poisson distribution. The location is specified by a parameter defining the mean pixel area.

        Args:

            n (int) - number of nuclei

            mu (int) - mean cell area (in pixels)

        Returns:

            radii (np.ndarray[int]) - nuclear radii, in pixels

        """
        areas = np.random.poisson(lam=mu, size=n)
        return np.round(np.sqrt(areas/np.pi)).astype(int)

    def initialize(self):
        """ Initialize blank image. """
        shape = (1, self.height, self.width)
        self.im = np.zeros(shape, dtype=np.float64)

    def _measure(self, channel):
        """ Returns measured <channel> level in for each contour. """
        labels = self.segmentation
        index = np.arange(self.num_nuclei)
        return measurements.mean(self.im[channel], labels=labels, index=index)

    def measure(self):
        """ Returns measured fluorescence level in each contour. """
        return self._measure(0)

    def _fill(self, channel, mu=0.1, sigma=0.1):
        """
        Fill background of specified channel with values sampled from a lognormal distribution.
        """
        pixels = np.exp(np.random.normal(np.log(mu), sigma, size=self.shape))
        self.im[channel, :, :] = pixels

    def fill(self, mu, sigma):
        """
        Fill image with values sampled from a lognormal distribution whose underlying normal distribution is parameterized by <mu> and <sigma>.
        """
        self._fill(0, mu, sigma)

    def _build_segmentation(self):
        """
        Draw nuclear labels on a segmentation mask.

        Returns:

            im (2D np.ndarray[np.int64]) - segmentation mask

        """
        mask = np.ones((self.shape), dtype=np.int64) * -1
        for i in range(self.num_nuclei):
            xy = self.centroids[i]
            nucleus = NucleusLabel(xy, label=i, radius=self.radius)
            nucleus.draw(mask)
        return mask

    def _draw_nuclei(self, im, means, stds, replace=True):
        """
        Draw individal nuclei on specified channel of the image.

        Args:

            im (2D np.ndarray) - image in which nuclei are drawn

            means (np.ndarray[float]) - mean fluorescence level for each nucleus

            stds (np.ndarray[float]) - std dev of log-transformed pixel values within each nucleus

            replace (bool) - if True, replace existing pixels

        """
        for i in range(self.num_nuclei):
            xy = self.centroids[i]
            nucleus = Nucleus(xy, means[i], stds[i], radius=self.radius)
            nucleus.draw(im, replace=replace)

    def draw_nuclei(self, means, stds):
        """
        Draw individal nuclei on specified channel of the image.

        Args:
            means (np.ndarray[float]) - mean fluorescence level for each nucleus

            stds (np.ndarray[float]) - std dev of log-transformed pixel values within each nucleus

        """
        self._draw_nuclei(self.im[0], means, stds)

    def render(self, ax=None, size=2, vmax=None, **kwargs):
        """
        Render image.

        Args:

            ax (matplotlib.AxesSubplot) - if None, create figure

            size (int) - figure size

            vmax (float) - upper bound for color scale

            kwargs: keyword argument for rendering

        Returns:

            fig (matplotlib.figure)

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(size, size))
        else:
            fig = plt.gcf()

        # if no color scale is specified, use the 95th percentile of foreground
        if vmax is None:
            vmax = np.percentile(self.foreground_pixels, q=95)

        self._render(self.im[0].T, ax=ax, vmax=vmax, **kwargs)

        return fig

    def render_mask(self, attribute, **kwargs):
        """
        Render image masked by specified nucleus <attribute>.
        """

        indices, values = self.data.index.values, self.data[attribute]
        segment_to_label = dict(zip(indices, values))
        segment_to_label[-1] = -1
        segment_to_label = np.vectorize(segment_to_label.get)

        label_mask = segment_to_label(self.segmentation)
        label_mask = np.ma.MaskedArray(label_mask, label_mask==-1)

        return self._render(label_mask.T, **kwargs)


class SyntheticMicroscopy(SyntheticImage):
    """
    Class allows for construction of a synthetic microscope image given a set of synthetic measurements.
    """

    def __init__(self, data,
                 bleedthrough=0.0,
                 bg_level=0.2,
                 bg_noise=0.3,
                 radius=6,
                 height=1000,
                 width=1000):
        """
        Instantiate synthetic image from a set of synthetic measurements.

        Args:

            data (pd.DataFrame) - position and mean expression level of each cell

            bleedthrough (float) - bleedthrough coefficient, implemented as the pseudo correlation coefficient between the clonal marker and the control protein channel

            bg_level (float) - background level (mean of log-transformed level)

            bg_noise (float) - background noise (std dev of log-transformed level)

            radius (int) - nuclear radius, in pixels

            height, width (int) - image dimensions, in pixels

        """

        # instantiate image
        super().__init__(data=data,
                         bg_level=bg_level,
                         bg_noise=bg_noise,
                         radius=radius,
                         height=height,
                         width=width)

        # set bleedthrough coefficient
        self.bleedthrough = bleedthrough

        # draw image channels
        self.draw_nuclear_stain()
        self.draw_clonal_marker()
        self.draw_control()

    def __getitem__(self, channel):
        """ Returns SyntheticImage of <channel>. """
        args = (self.data, self.im[channel])
        kwargs = dict(
            bg_level=self.bg_level,
            bg_noise=self.bg_noise,
            radius=self.radius,
            height=self.height,
            width=self.width)
        return super().from_channel(*args, **kwargs)

    @property
    def foreground_pixels(self):
        """ Return all pixels from foreground. """
        return self.im[np.stack((self.foreground_mask,) * 3)]

    @property
    def background_pixels(self):
        """ Return all pixels from background. """
        return self.im[~np.stack((self.foreground_mask,) * 3)]

    @property
    def max(self):
        """ Maximum fluorescence level in each channel. """
        return self.im.max(axis=(1, 2))

    @property
    def im_normalized(self):
        """ Image normalized by the maximum value in each channel. """
        return self.im/self.max.reshape(-1,1,1)

    @property
    def rgb_im(self):
        """ Image in RGB format. """
        #im = np.swapaxes(np.swapaxes(self.im_normalized, 0, 1), 1, 2)
        im = np.swapaxes(self.im_normalized, 0, 2)
        return im[:, :, [1,2,0]]

    def initialize(self):
        """ Initialize blank image. """
        self.im = np.zeros((3, self.height, self.width), dtype=np.float64)

    def measure(self, channel):
        """ Returns measured level of <channel> in each contour. """
        return self._measure(channel)

    def fill(self, channel, mu=0.1, sigma=0.1):
        """
        Fill image with values sampled from a lognormal distribution whose underlying normal distribution is parameterized by <mu> and <sigma>. """
        self._fill(channel, mu, sigma)

    def draw_nuclei(self, channel, means, stds):
        """
        Draw individal nuclei on specified channel of the image.

        Args:

            channel (int) - image channel in which nuclei are drawn

            means (np.ndarray[float]) - mean fluorescence level for each nucleus

            stds (np.ndarray[float]) - std dev of log-transformed pixel values within each nucleus

        """
        self._draw_nuclei(self.im[channel], means, stds)

    def add_correlated_fluorescence(self, src, dst, rho=1.):
        """
        Add fluorescence bleedthrough from <src> to <dst>.

        Args:

            src (int) - source channel

            dst (int) - destination channel

            rho (float) - approximate correlation coefficient

        """
        bleed = DependentScalarImage(self.im[src],self.bg_level, self.bg_noise)
        bleed.fill(rho=rho)
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

        self.im[dst] = self.im[dst] + (rho*bleed+(1-rho) * background)

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
        self.add_bleedthrough(1, 2, rho=self.bleedthrough)

    def render(self, size=5, ax=None, **kwargs):
        """
        Render all image channels.

        Args:


            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            size (int) - figure panel size

            kwargs: keyword argument for rendering


        Returns:

            fig (matplotlib.figure)

        """

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(size, size))

        # render image
        ax.imshow(self.rgb_im, **kwargs)
        ax.invert_yaxis()
        ax.axis('off')

        return fig

    def render_panels(self, size=5, label=True, vmax=None, **kwargs):
        """
        Render all image channels.

        Args:

            size (int) - figure panel size

            label (bool) - if True, label panels

            vmax (float) - upper bound for color scale

            kwargs: keyword argument for rendering


        Returns:

            fig (matplotlib.figure)

        """

        # if no color scale is specified, use the 95th percentile of foreground
        if vmax is None:
            vmax = np.percentile(self.foreground_pixels, q=95)

        figsize = (3*size+.25, size)
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=figsize)
        super()._render(self.im[0].T, ax=ax0, vmax=vmax, **kwargs)
        super()._render(self.im[1].T, ax=ax1, vmax=vmax, **kwargs)
        super()._render(self.im[2].T, ax=ax2, vmax=vmax, **kwargs)

        if label:
            ax0.set_title('Nuclear Marker')
            ax1.set_title('Clonal Marker')
            ax2.set_title('Bleedthrough Control')

        return fig
