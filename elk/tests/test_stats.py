import numpy as np
import elk.stats as stats
import unittest
from scipy.stats import skewnorm


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_MAD(self):
        """check that MAD is larger for something with a higher SD"""

        n_vals = 10000
        x = np.random.normal(0, 1, n_vals)
        y = np.random.normal(0, 10, n_vals)

        self.assertTrue(stats.get_MAD(x) < stats.get_MAD(y))

    def test_skewness(self):
        """check that skewness is working properly"""

        n_vals = 10000
        x = skewnorm(a = 0).rvs(size=n_vals)
        y = skewnorm(a = 3).rvs(size=n_vals)

        self.assertTrue(stats.get_skewness(x) < stats.get_skewness(y))
