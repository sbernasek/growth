from unittest import TestCase
from pandas import DataFrame
from growth import Culture


class TestGrowth(TestCase):
    """
    Tests for basic growth simulation.
    """

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Culture. """
        cls.culture = Culture(reference_population=100)

    def test00_seed(self):
        """ Check that culture is valid.  """
        self.assertTrue(isinstance(self.culture, Culture))

    def test01_grow(self):
        """ Run a growth simulation.  """
        N = 100
        self.culture.grow(N, division_rate=0.1, recombination_rate=0.1)
        self.assertTrue(self.culture.size > N)

    def test02_measure(self):
        """ Generate synthetic measurements.  """
        measurements = self.culture.measure(ambiguity=0.1)
        self.assertTrue(isinstance(measurements, DataFrame))
