import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class LognormalSampler:
    """
    Model for generating samples from a log-normal distribution.

    Attributes:

        mu (np.ndarray[float]) - mean of the underlying normal distribution

        sigma (np.ndarray[float]) - std dev of the underlying normal distribution

        support (np.ndarray[float]) - support vector for distributions

    """

    def __init__(self, mu, sigma, density=100000):
        """
        Instantiate model for sampling from a lognormal distribution.

        Args:

            mu (float) - mean of the underlying normal distribution

            sigma (float) - std dev of the underlying normal distribution

            density (int) - number of datapoints in the support vector

        """

        self.mu = mu
        self.sigma = sigma

    def __call__(self, N):
        """ Draw <N> samples. """
        return self.sample(N)

    @property
    def support(self):
        """ Distribution support vector. """
        return np.linspace(0, self.saturation, num=density)

    @property
    def saturation(self):
        """ Upper bound on support. """
        return st.lognorm(self.sigma, scale=np.exp(self.mu)).ppf(0.999)

    @property
    def pdf(self):
        """ Probability density. """
        pdf = np.zeros(self.support.size, dtype=np.float64)
        pdf += self.freeze().pdf(self.support)
        return pdf

    @property
    def error(self):
        """ Rounding error due to sampling. """
        return 1 - np.trapz(self.pdf[1:], x=self.support[1:])

    def freeze(self):
        """
        Returns frozen model for the distribution.

        * Note that scipy.stats uses a standardized form in which the "shape" parameter corresponds to the scale attribute and the "scale" parameter corresponds to the exponentiated location.
        """
        return st.lognorm(self.sigma, loc=0, scale=np.exp(self.mu))

    def sample(self, N):
        """ Draw <N> samples from the distribution. """
        return np.random.lognormal(self.mu, self.sigma, size=N)

    def show_pdf(self, ax=None, xlim=(0, 1), norm=True):
        """ Plot probability density function. """

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')

        # plot pdf
        ax.plot(self.support, self.pdf, '-k')

        # format axes
        ylim = (0, np.percentile(self.pdf[1:], q=99))
        ax.set_ylim(*ylim)


class MultiLognormalSampler:
    """
    Model for sampling gene dosage-dependent fluorescence levels. Intensities are drawn from three independent lognormal distributions based on the gene dosage of each cell. The separation between the distributions is determined by the 'ambiguity' parameter.

    Attributes:

        mu (np.ndarray[float]) - mean of the underlying normal distribution

        sigma (np.ndarray[float]) - std dev of the underlying normal distribution

        ambiguity (float) - fluorescence ambiguity coefficient, > 0

        loc (np.ndarray[float]) - location parameters

        scale np.ndarray[float]) - scale parameters

        support (np.ndarray[float]) - support vector for all distributions

    """

    def __init__(self, ambiguity=0.1, mu=None, density=100000):
        """
        Instantiate model for generating synthetic fluorescence intensities. Intensities are sampled from one of multiple log-normal distributions. The location and scale parameters defining each distribution are stored as vectors of coefficients. The distiribution used to generate each sample is defined by a vector of distribution indices passed to the __call__ method.

        Args:

            ambiguity (float) - fluorescence ambiguity coefficient, value must be greater than zero and is equivalent to the std dev of each underyling normal distribution

            mu (np.ndarray[float]) - mean of the underyling normal distribution

            sigma (np.ndarray[float]) - std dev of the underyling normal distribution

            density (int) - number of datapoints in the support vector

        """

        # determine location parameters
        if mu is None:
            self.mu = np.logspace(-1, 1, base=2, num=3)
        else:
            self.mu = np.array(mu)
        loc = np.log(self.mu)
        self.set_loc(loc)

        # determine scale parameters
        self.sigma = np.ones(3) * ambiguity
        self.set_scale(self.sigma)

        # determine support vector
        self.support = np.linspace(0, self.saturation, num=density)

    def __call__(self, indices):
        """ Draw luminescence samples for distribution <indices>. """
        return self.sample(indices)

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
        """
        Returns frozen model for <i>th univariate distribution.

        * Note that scipy.stats uses a standardized form in which the "shape" parameter corresponds to the scale attribute and the "scale" parameter corresponds to the exponentiated location.
        """
        return st.lognorm(self.scale[i], loc=0, scale=np.exp(self.loc[i]))

    def sample(self, indices):
        """ Draw luminescence samples for distribution <indices>. """
        otypes = [np.float64]
        scale = np.vectorize(dict(enumerate(self.scale)).get, otypes=otypes)
        loc = np.vectorize(dict(enumerate(self.loc)).get, otypes=otypes)
        size = indices.size
        sample = np.random.lognormal(
            mean=loc(indices),
            sigma=scale(indices),
            size=size)
        return sample

    def show_pdf(self, ax=None, xlim=(0, 1), norm=True):
        """ Plot probability density function. """

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')

        # plot component distributions
        pdf = np.zeros(self.support.size, dtype=np.float64)
        for i in range(3):
            rv = self.freeze_univariate(i)
            component = rv.pdf(self.support)
            if norm:
                component /= 3

            pdf += component
            ax.plot(self.support, component)

        # plot pdf
        ax.plot(self.support, pdf, '-k')

        # format axes
        ylim = (0, np.percentile(pdf[1:], q=99))
        ax.set_ylim(*ylim)
