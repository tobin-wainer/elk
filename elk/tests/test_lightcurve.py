import numpy as np
import elk.lightcurve as lightcurve
import unittest

def random_lc(n_times=10000, n_lc=1000):
    times = np.linspace(50, 150, n_times)
    amplitudes = np.random.uniform(0, 10, size=n_lc)
    phases = np.random.uniform(0, 2 * np.pi, size=n_lc)
    periods = np.random.uniform(0, 10, size=n_lc)

    curves = np.array([amplitudes[i] * np.sin(2 * np.pi / periods[i] * (times - phases[i]))
                       for i in range(n_lc)])
    flux = curves.sum(axis=0)
    flux -= (min(flux) - 1)
    return times, flux


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_basic_creation(self):
        """check that we can create a basic light curve class"""
        t, f = random_lc()
        s = np.random.randint(1, 20)
        lc = lightcurve.BasicLightcurve(time=t, flux=f, flux_err=np.repeat(0.01, len(f)), sector=s)

        self.assertTrue(lc.sector == s)
        self.assertTrue(all(lc.corrected_lc.flux == f))

    def test_flux_normalisation(self):
        """check flux is being normalised well"""
        t, f = random_lc()
        s = np.random.randint(1, 20)
        lc = lightcurve.BasicLightcurve(time=t, flux=f, flux_err=np.repeat(0.01, len(f)), sector=s)

        self.assertTrue(round(np.mean(lc.normalized_flux)) == 1.0)
