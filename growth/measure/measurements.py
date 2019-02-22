import numpy as np
import pandas as pd

from .sampling import LognormalSampler, MultiLognormalSampler
from .conditional import ConditionedLognormalSampler
from .conditional import ConditionedMultiLognormalSampler


class MeasurementGenerator:
    """
    Class for generating fluorescence measurements from a synthetic culture.
    """

    def __init__(self, culture,
                ambiguity=0.,
                rho=0.,
                expression_capacity_sigma=0.5,
                nuclear_stain_level=2.,
                control_level=0.5,
                nuclear_stain_sigma=0.3,
                control_sigma=0.3,
                clonal_marker_mu=None,
                measurement_noise=0.):

        # instantiate measurement dataframe with cell positions
        self.data = pd.DataFrame(culture.xy, columns=['x','y'])
        self.data['true_dosage'] = culture.genotypes

        # store parameters
        self.ambiguity = ambiguity
        self.rho = rho
        self.expression_capacity_sigma = expression_capacity_sigma
        self.nuclear_stain_mu = np.log(nuclear_stain_level)
        self.control_mu = np.log(control_level)
        self.nuclear_stain_sigma = nuclear_stain_sigma
        self.control_sigma = control_sigma
        self.clonal_marker_mu = clonal_marker_mu
        self.measurement_noise = measurement_noise

        # generate measurements
        self.generate_measurements()

    @property
    def N(self):
        """ Number of samples. """
        return len(self.data)

    def generate_measurements(self):
        """ Generate fluorescence measurements for each nucleus. """

        kwargs = dict(scale=self.expression_capacity_sigma, size=self.N)
        self.data['expression_capacity'] = np.random.normal(**kwargs)

        # measure nuclear stain
        args = (self.nuclear_stain_mu, self.nuclear_stain_sigma, self.rho)
        levels = self.conditioned_measurement(*args)
        self.data['nuclear_stain'] = levels
        self.data['nuclear_stain_std'] = levels * self.measurement_noise

        # measure clonal marker
        args = (self.ambiguity, self.clonal_marker_mu, self.rho)
        levels = self.conditioned_clonal_measurement(*args)
        self.data['clonal_marker'] = levels
        self.data['clonal_marker_std'] = levels * self.measurement_noise

        # measure control species
        args = (self.control_mu, self.control_sigma, self.rho)
        levels = self.conditioned_measurement(*args)
        self.data['control'] = levels
        self.data['control_std'] = levels * self.measurement_noise

    def measurement(self, mu, sigma):
        """
        Sample fluorescence levels.

        Args:

            mu (float) - mean of underlying normal distribution

            sigma (float) - std dev of underlying normal distribution

        """
        sampler = LognormalSampler(mu, sigma)
        return sampler(self.N)

    def conditioned_measurement(self, mu, sigma, rho):
        """
        Sample fluorescence levels conditioned on expression capacity.

        Args:

            mu (float) - mean of underlying normal distribution

            sigma (float) - std dev of underlying normal distribution

            rho (float) - correlation coefficient with expression capacity

        """
        x = self.data.expression_capacity.values
        sampler = ConditionedLognormalSampler(x, mu, sigma)
        return sampler(rho=rho)

    def clonal_measurement(self, ambiguity, mu):
        """
        Sample fluorescence levels for a clonal dosage-dependent signal.

        Args:

            ambiguity (float) - fluorescence ambiguity coefficient, value must be greater than zero and is equivalent to the std dev of each underyling normal distribution

            mu (np.ndarray[float]) - mean of the underyling normal distribution

        """
        sampler = MultiLognormalSampler(ambiguity, mu=mu)
        return sampler(self.data.true_dosage.values)

    def conditioned_clonal_measurement(self, ambiguity, mu, rho):
        """
        Sample fluorescence levels for a clonal dosage-dependent signal.

        Args:

            ambiguity (float) - fluorescence ambiguity coefficient, value must be greater than zero and is equivalent to the std dev of each underyling normal distribution

            mu (np.ndarray[float]) - mean of the underyling normal distribution

        """
        x = self.data.expression_capacity.values
        sampler = ConditionedMultiLognormalSampler(x, ambiguity, mu=mu)
        return sampler(self.data.true_dosage.values, rho=rho)
