import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class Fluorescence:

    def __init__(self, shape=(1, 5, 10), scale=1):
        self.set_shape(shape)
        self.set_scale(scale)

    @property
    def saturation(self):
        return st.gamma(self.shape(2), scale=self.scale(2)).ppf(0.999)

    def __call__(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        return self.sample(genotypes)

    def set_shape(self, shape):
        """ Set shape parameters. """
        if type(shape) in (int, float):
            shape = (shape,)*3
        self.shape = np.vectorize(dict(enumerate(shape)).get)

    def set_scale(self, scale):
        """ Set scale parameters. """
        if type(scale) in (int, float):
            scale = (scale,)*3
        self.scale = np.vectorize(dict(enumerate(scale)).get)

    def sample(self, genotypes):
        """ Draw luminescence samples for <genotypes>. """
        shapes = self.shape(genotypes)
        scales = self.scale(genotypes)
        size = genotypes.size
        sample = np.random.gamma(shapes, scales, size=size)
        return sample / self.saturation

    def show_pdf(self, xlim=(0, 1)):
        """ Plot probability density function. """

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')

        x = np.linspace(0, 1, num=100)
        for i in range(3):
            rv = st.gamma(self.shape(i), loc=0., scale=self.scale(i))
            ax.plot(x, rv.pdf(x*self.saturation))
