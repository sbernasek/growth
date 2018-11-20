import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class Fluorescence:

    def __init__(self, shape=(0.5, 5, 15), scale=1, loc=0, density=100000):
        self.set_shape(shape)
        self.set_scale(scale)
        self.set_loc(loc)
        self.support = np.linspace(0, self.saturation, num=density)

    @property
    def saturation(self):
        """ Upper bound on support. """
        return st.gamma(self.shape[2], scale=self.scale[2]).ppf(0.999)

    @property
    def pdf(self):
        """ Probability density. """
        pdf = np.zeros(self.support.size, dtype=np.float64)
        for i in range(3):
            rv = st.gamma(self.shape[i], loc=self.loc[i], scale=self.scale[i])
            pdf += rv.pdf(self.support)
        return pdf / 3

    @property
    def error(self):
        """ Rounding error due to sampling. """
        return 1 - np.trapz(self.pdf[1:], x=self.support[1:])

    def __call__(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        return self.sample(genotypes)

    def set_shape(self, shape):
        """ Set shape parameters. """
        if type(shape) in (int, float):
            shape = (shape,)*3
        otypes = [np.float64]
        self.shape = shape

    def set_scale(self, scale):
        """ Set scale parameters. """
        if type(scale) in (int, float):
            scale = (scale,)*3
        self.scale = scale

    def set_loc(self, loc):
        """ Set loc parameters. """
        if type(loc) in (int, float):
            loc = (loc,)*3
        self.loc = loc

    def sample(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        otypes = [np.float64]
        shape = np.vectorize(dict(enumerate(self.shape)).get, otypes=otypes)
        scale = np.vectorize(dict(enumerate(self.scale)).get, otypes=otypes)
        loc = np.vectorize(dict(enumerate(self.loc)).get, otypes=otypes)
        size = genotypes.size
        sample = np.random.gamma(shape(genotypes),
                                 scale=scale(genotypes),
                                 size=size) + loc(genotypes)
        return sample

    def show_pdf(self, xlim=(0, 1)):
        """ Plot probability density function. """

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')

        # plot component distributions
        pdf = np.zeros(self.support.size, dtype=np.float64)
        for i in range(3):
            rv = st.gamma(self.shape[i], loc=self.loc[i], scale=self.scale[i])
            component = rv.pdf(self.support) / 3
            pdf += component
            ax.plot(self.support, component)

        # plot pdf
        ax.plot(self.support, pdf, '-k')

        # format axes
        ylim = (0, np.percentile(pdf[1:], q=99))
        ax.set_ylim(*ylim)
