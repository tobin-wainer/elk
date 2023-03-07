import numpy as np


class TESScutLightcurve():
    def __init__(self, lk_search_result=None, tpfs=None, cutout_size=99):

        assert lk_search_result is not None or tpfs is not None, "Must supply either a search result or tpfs"

        self.lk_search_results = lk_search_result
        self._tpfs = tpfs
        self._use_tpfs = None
        self.cutout_size = cutout_size

    @property
    def tpfs(self):
        if self._tpfs is None:
            self._tpfs = self.lk_search_results.download(cutout_size=(self.cutout_size, self.cutout_size))
        return self._tpfs

    @property
    def basic_lc(self):
        if self._basic_lc is None:
            self._basic_lc = self.tpfs.to_lightcurve()
        return self._basic_lc

    @property
    def use_tpfs(self):
        if self._use_tpfs is None:
            self._use_tpfs = self.tpfs[(self.basic_lc.quality == 0) & (self.basic_lc.flux_err > 0)]
        return self._use_tpfs

    def near_edge(self):
        # Only selecting time steps that are good, or have quality == 0
        t1 = self.tpfs[np.where(self.tpfs.to_lightcurve().quality == 0)]

        min_not_nan = ~np.isnan(np.min(t1[0].flux.value))
        # Also making sure the Sector isn't the one with the Systematic
        not_sector_one = self.tpfs.sector != 1
        min_flux_greater_one = np.min(t1[0].flux.value) > 1
        return ~(min_not_nan & not_sector_one & min_flux_greater_one)