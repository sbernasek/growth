import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class Fluorescence:
    """
    Model for generating synthetic fluorescence intensities. Intensities are drawn from a log-normal distribution.

    Attributes:

        loc (tuple) - location parameters

        scale (tuple) - scale parameters

    """

    def __init__(self, mu=(0, 3, 4), sigma=(1, .3, .3), density=100000):
        self.set_loc(sigma)
        self.set_scale(mu)
        self.support = np.linspace(0, self.saturation, num=density)

    @classmethod
    def from_scale(cls, scale=20):
        """ Instantiate fluorescence model from <scale>. """
        loc = (1, scale, 2*scale)
        sigma = (1., 1./scale, 1./scale)
        return cls(mu=np.log(loc), sigma=sigma)

    def __call__(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        return self.sample(genotypes)

    @property
    def saturation(self):
        """ Upper bound on support. """
        return st.lognorm(self.loc[2], scale=np.exp(self.scale[2])).ppf(0.999)

    @property
    def pdf(self):
        """ Probability density. """
        pdf = np.zeros(self.support.size, dtype=np.float64)
        for i in range(3):
            rv = self.freeze_univariate(i)
            pdf += rv.pdf(self.support)
        return pdf / 3

    @property
    def error(self):
        """ Rounding error due to sampling. """
        return 1 - np.trapz(self.pdf[1:], x=self.support[1:])

    def set_scale(self, scale):
        """ Set scale parameters. """
        if type(scale) in (int, float):
            scale = (scale,) * 3
        self.scale = scale

    def set_loc(self, loc):
        """ Set loc parameters. """
        if type(loc) in (int, float):
            loc = (loc,) * 3
        self.loc = loc

    def freeze_univariate(self, i):
        """ Returns frozen model for <i>th univariate distribution. """
        return st.lognorm(self.loc[i], scale=np.exp(self.scale[i]))

    def sample(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        otypes = [np.float64]
        scale = np.vectorize(dict(enumerate(self.scale)).get, otypes=otypes)
        loc = np.vectorize(dict(enumerate(self.loc)).get, otypes=otypes)
        size = genotypes.size
        sample = np.random.lognormal(
            mean=scale(genotypes),
            sigma=loc(genotypes),
            size=size)
        return sample

    def show_pdf(self, ax=None, xlim=(0, 1)):
        """ Plot probability density function. """

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')

        # plot component distributions
        pdf = np.zeros(self.support.size, dtype=np.float64)
        for i in range(3):
            rv = self.freeze_univariate(i)
            component = rv.pdf(self.support) / 3
            pdf += component
            ax.plot(self.support, component)

        # plot pdf
        ax.plot(self.support, pdf, '-k')

        # format axes
        ylim = (0, np.percentile(pdf[1:], q=99))
        ax.set_ylim(*ylim)
