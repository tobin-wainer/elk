import numpy as np
import elk.stats as stats
import unittest
from scipy.stats import skewnorm

N_VALS = 10000

def random_walk(start=0, steps=N_VALS, step_size=0.01, walk_prob=0.05):
    walk = np.empty(steps)
    walk[0] = start
    choices = np.random.rand(steps)
    for i in range(1, steps):
        if choices[i] > 1 - walk_prob:
            walk[i] = walk[i - 1] + step_size
        elif choices[i] < walk_prob:
            walk[i] = walk[i - 1] - step_size
        else:
            walk[i] = walk[i - 1]
    return walk


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_MAD(self):
        """check that MAD is larger for something with a higher SD"""
        x = random_walk(walk_prob=0.01)
        y = random_walk(walk_prob=0.2)

        self.assertTrue(stats.get_MAD(x) < stats.get_MAD(y))

    def test_skewness(self):
        """check that skewness is working properly"""
        x = skewnorm(a = 0).rvs(size=N_VALS)
        y = skewnorm(a = 30).rvs(size=N_VALS)

        self.assertTrue(stats.get_skewness(x) < stats.get_skewness(y))

    def test_von_neumann(self):
        """check that VNR works sensibly"""
        x = random_walk(walk_prob=0.001)
        y = random_walk(walk_prob=0.3)

        self.assertTrue(stats.von_neumann_ratio(x) / np.var(x) > stats.von_neumann_ratio(y) / np.var(y))

    def test_j_stetson_far_times(self):
        """Check J stetson works for spread out times"""
        t = np.arange(N_VALS)
        mag = random_walk(start=16, walk_prob=0.01)
        big_var_mag = random_walk(start=16, walk_prob=0.2)
        mag_err = np.repeat(0.01, len(mag))
        
        self.assertTrue(stats.J_stetson(t, mag, mag_err) < stats.J_stetson(t, big_var_mag, mag_err))

    def test_j_stetson_close_times(self):
        """Check J stetson works for close packed times"""
        t = np.linspace(0, 0.021, N_VALS)
        mag = np.random.normal(loc=16, scale=0.1, size=N_VALS)
        big_var_mag = np.random.normal(loc=16, scale=1, size=N_VALS)
        mag_err = np.repeat(0.01, len(mag))
        
        self.assertTrue(stats.J_stetson(t, mag, mag_err) < stats.J_stetson(t, big_var_mag, mag_err))
        
    def test_periodogram(self):
        """test that we can recover the correct period"""
        f_range = np.linspace(0.05, 0.95, 1000)
        period = np.random.choice(f_range)

        t = np.sort(N_VALS * np.random.rand(N_VALS))
        y = np.sin(2 * np.pi * t * period) + 0.1 * np.random.standard_normal(N_VALS)

        test = stats.periodogram(t, y, np.repeat(0.01 / N_VALS, len(y)),
                                 frequencies=f_range, freq_thresh=0.5, n_bootstrap=10)
        self.assertTrue(np.abs(period - test[-1]["freq_at_max_power"]) < 1e-2)