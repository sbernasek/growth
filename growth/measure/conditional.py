import numpy as np

from .sampling import LognormalSampler, MultiLognormalSampler


class ConditionedLognormalSampler(LognormalSampler):
    """
    Class for sampling Y conditioned on a jointly-distributed sample X.

    Attributes:

        x (np.ndarray[float]) - jointly-distributed sample

        mu_x (float) - mean of jointly-distributed sample

        sigma_x (float) - std dev of jointly-distributed sample

    Inherited attributes:

        mu (np.ndarray[float]) - mean of the underlying normal distribution

        sigma (np.ndarray[float]) - std dev of the underlying normal distribution

        support (np.ndarray[float]) - support vector for distributions

    """
    def __init__(self, x, mu=None, sigma=None, **kwargs):
        """
        Args:

            x (np.ndarray[float]) - jointly-distributed sample

            mu (float) - mean of the log-transformed conditioned sample

            sigma (float) - std dev of the log-transformed conditioned sample

            kwargs: keyword arguments for LognormalSampler

        """

        self.x = x
        self.mu_x = x.mean()
        self.sigma_x = x.std()

        # instantiate lognormal sampler
        if mu is None:
            mu = self.mu_x
        if sigma is None:
            sigma = self.sigma_x
        super().__init__(mu, sigma, **kwargs)

    def __call__(self, rho):
        """
        Draw samples for the conditioned variable.

        Args:

            rho (float) - correlation coefficient

        """
        return self.sample(rho)

    def sample(self, rho=0.0):
        """
        Generate sample for y conditioned upon x.

        Args:

            rho (float) - correlation coefficient

        Returns:

            y (np.ndarray[float]) - conditioned sample

        """

        mu = self.mu + (rho*self.sigma/self.sigma_x)*(self.x - self.mu_x)
        sigma = np.sqrt((1-rho**2)*(self.sigma**2))
        return np.exp(mu + sigma * np.random.normal(size=self.x.size))


class ConditionedMultiLognormalSampler(MultiLognormalSampler):
    """
    Model for sampling gene dosage-dependent fluorescence levels, conditioned upon a jointly-distributed sample. Intensities are drawn from three independent lognormal distributions based on the gene dosage of each cell. The separation between the distributions is determined by the 'ambiguity' parameter.

    Samples are drawn from each distribution conditioned upon a jointly-distributed sample.

    Attributes:

        x (np.ndarray[float]) - jointly-distributed sample

        mu_x (float) - mean of jointly-distributed sample

        sigma_x (float) - std dev of jointly-distributed sample

    Inherited attributes:

        mu (np.ndarray[float]) - means of the log-transformed conditioned sample

        sigma (np.ndarray[float]) - std devs of the log-transformed conditioned sample

        ambiguity (float) - fluorescence ambiguity coefficient, > 0

        loc (np.ndarray[float]) - location parameters

        scale np.ndarray[float]) - scale parameters

        support (np.ndarray[float]) - support vector for all distributions

    """

    def __init__(self, x, ambiguity=0.1, **kwargs):
        """
        Instantiate model for generating synthetic fluorescence intensities. Intensities are sampled from one of multiple log-normal distributions, each conditioned upon a jointly-distributed sample.

        The location and scale parameters defining each distribution are stored as vectors of coefficients. The distribution used to generate each sample is defined by a vector of distribution indices passed to the __call__ method.

        Args:

            x (np.ndarray[float]) - jointly-distributed sample

            ambiguity (float) - fluorescence ambiguity coefficient, value must be greater than zero and is equivalent to the std dev of each underyling normal distribution

            kwargs: keyword arguments for MultiLognormalSampler

        """
        self.x = x
        self.mu_x = x.mean()
        self.sigma_x = x.std()
        super().__init__(ambiguity, **kwargs)

    def __call__(self, indices, rho=0.0):
        """ Draw luminescence samples for distribution <indices>. """
        assert (len(indices) == self.x.size), 'Wrong number of indices.'
        return self.sample(indices, rho=rho)

    def sample(self, indices, rho=0.0):
        """
        Draw samples for distribution <indices>.

        Args:

            indices (array like) - distribution indices

            rho (float) - correlation coefficient

        Returns:

            sample (np.ndarray[float]) - conditioned sample

        """
        otypes = [np.float64]
        scale = np.vectorize(dict(enumerate(self.scale)).get, otypes=otypes)
        loc = np.vectorize(dict(enumerate(self.loc)).get, otypes=otypes)
        size = indices.size

        # evaluate mu/sigma for log-transformed conditioned sample
        mu_y = loc(indices)
        sigma_y = scale(indices)

        # draw samples
        mu = mu_y + (rho*sigma_y/self.sigma_x)*(self.x - self.mu_x)
        sigma = np.sqrt((1-rho**2)*(sigma_y**2))
        sample = np.exp(mu + sigma * np.random.normal(size=self.x.size))

        return sample
