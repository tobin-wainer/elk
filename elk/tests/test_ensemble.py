import numpy as np
from elk.ensemble import EnsembleLC, from_fits
import unittest
import os
import shutil


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_main_functionality(self):
        """This test is annoying because you need to do the full analysis before the rest can be tested.
        This cluster is smaller than most on the processing side and so we get test in about 2 minutes and
        then do stuff with this data. But this unfortunately makes this test quite long (in code and time)"""

        # start an output folder
        os.mkdir("ensemble_test_output")

        # NGC 6304 has a short lightcurve in its first sector so is useful for testing
        # this cutout size is chosen to be just larger than the radius, but small enough to speed things up
        c = EnsembleLC(identifier='NGC 6304', location='258.635, -29.462', radius=0.13, cutout_size=50,
                       verbose=True, just_one_lc=True, ignore_scattered_light=True, auto_confirm=True,
                       output_path="ensemble_test_output")
        
        # compute all of the lightcurves
        c.create_output_table()

        # check the light curve has actually been computed
        self.assertTrue(c.lcs[0] is not None)

        new_c = from_fits("ensemble_test_output/Corrected_LCs/NGC 6304output_table.fits")
        self.assertTrue(all(new_c.lcs[0].corrected_lc.flux == c.lcs[0].corrected_lc.flux))

        self.assertTrue(new_c.output_path is not None)

        shutil.rmtree("ensemble_test_output")