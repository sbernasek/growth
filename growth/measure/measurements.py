import numpy as np
import pandas as pd

from .fluorescence import FluorescenceSampler
from .fluorescence import DosageDependentFluorescenceSampler


class MeasurementGenerator:
    """
    Class for generating fluorescence measurements from a synthetic culture.
    """

    def __init__(self, culture,
                ambiguity=0.,
                nuclear_stain_level=2.,
                control_level=0.5,
                nuclear_stain_sigma=0.3,
                control_sigma=0.3,
                clonal_marker_mu=None,
                measurement_noise=0.3):

        # instantiate measurement dataframe with cell positions
        self.df = pd.DataFrame(culture.xy, columns=['centroid_x','centroid_y'])
        self.df['true_dosage'] = culture.genotypes

        # store parameters
        self.ambiguity = ambiguity
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
        return len(self.df)

    def generate_measurements(self):
        """ Generate fluorescence measurements for each nucleus. """

        # measure nuclear stain
        args = (self.N, self.nuclear_stain_mu, self.nuclear_stain_sigma)
        levels = self.constant_measurement(*args)
        self.df['nuclear_stain'] = levels
        self.df['nuclear_stain_std'] = levels * self.measurement_noise

        # measure clonal marker
        args = (self.df, self.ambiguity, self.clonal_marker_mu)
        levels = self.dosage_dependent_measurement(*args)
        self.df['clonal_marker'] = levels
        self.df['clonal_marker_std'] = levels * self.measurement_noise

        # measure control species
        args = (self.N, self.control_mu, self.control_sigma)
        levels = self.constant_measurement(*args)
        self.df['control'] = levels
        self.df['control_std'] = levels * self.measurement_noise

    @staticmethod
    def constant_measurement(N, mu, sigma):
        """ Sample fluorescence levels for a constant signal. """
        sampler = FluorescenceSampler(mu, sigma)
        return sampler(N)

    @staticmethod
    def dosage_dependent_measurement(df, ambiguity, mu):
        """ Sample fluorescence levels for a dosage-dependent signal. """
        sampler = DosageDependentFluorescenceSampler(ambiguity, mu=mu)
        return sampler(df.true_dosage.values)
