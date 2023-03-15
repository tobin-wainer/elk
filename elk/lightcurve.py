import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import lightkurve as lk
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import flux_to_mag, flux_err_to_mag_err
import elk.plot as elkplot
import elk.stats as elkstats


__all__ = ["SimpleCorrectedLightcurve", "TESSCutLightcurve"]


TESS_RESOLUTION = 21 * u.arcsec / u.pixel


class SimpleCorrectedLightcurve():
    def __init__(self, time=None, flux=None, flux_err=None, sector=None, fits_path=None, hdu_index=None):
        has_data = time is not None and flux is not None and flux_err is not None and sector is not None
        has_file = fits_path is not None and hdu_index is not None
        assert has_data or has_file, "Must either provide data or a fits file and HDU index"

        if has_file:
            with fits.open(fits_path) as hdul:
                hdu = hdul[hdu_index]
                self.corrected_lc = lk.LightCurve(time=hdu.data["time"] * u.day,
                                                  flux=hdu.data["flux"] * u.electron / u.s,
                                                  flux_err=hdu.data["flux_err"] * u.electron / u.s)
                self.sector = hdu.header["sector"]
        else:
            self.corrected_lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            self.sector = sector

        self.hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name='time', format='D', array=self.corrected_lc.time.value),
             fits.Column(name='flux', format='D', array=self.corrected_lc.flux.value),
             fits.Column(name='flux_err', format='D', array=self.corrected_lc.flux_err.value),
             fits.Column(name='mag', format='D', array=flux_to_mag(self.corrected_lc.flux.value)),
             fits.Column(name='mag_err', format='D',
                         array=flux_err_to_mag_err(self.corrected_lc.flux.value,
                                                   self.corrected_lc.flux_err.value))]
        )
        self.hdu.header.set('sector', self.sector)

        self.stats = {}
        self.periodogram_frequencies = None
        self.ac_time = None

    @property
    def normalized_flux(self):
        return self.corrected_lc.flux.value / np.median(self.corrected_lc.flux.value)

    @property
    def rms(self):
        self.stats["rms"] = np.sqrt(np.mean(self.normalized_flux**2))
        return self.stats["rms"]

    @property
    def std(self):
        self.stats["std"] = np.std(self.normalized_flux)
        return self.stats["std"]

    @property
    def MAD(self):
        self.stats["MAD"] = elkstats.get_MAD(self.normalized_flux)
        return self.stats["MAD"]

    @property
    def skewness(self):
        self.stats["skewness"] = elkstats.get_skewness(self.corrected_lc.flux.value)
        return self.stats["skewness"]

    @property
    def von_neumann_ratio(self):
        self.stats["von_neumann_ratio"] = elkstats.von_neumann_ratio(self.normalized_flux)
        return self.stats["von_neumann_ratio"]

    def J_stetson(self, **kwargs):
        mag_err = flux_err_to_mag_err(self.corrected_lc.flux.value, self.corrected_lc.flux_err.value)
        self.stats["J_Stetson"] = elkstats.J_stetson(time=self.corrected_lc.time.value,
                                                     mag=flux_to_mag(self.corrected_lc.flux.value),
                                                     mag_err=mag_err, **kwargs)
        return self.stats["J_Stetson"]

    def to_periodogram(self, frequencies, **kwargs):
        self.periodogram, self.periodogram_percentiles,\
            lsp_stats = elkstats.periodogram(self.corrected_lc.time.value, self.corrected_lc.flux.value,
                                             self.corrected_lc.flux_err.value, frequencies=frequencies,
                                             **kwargs)
        self.periodogram_frequencies = frequencies
        self.stats.update(lsp_stats)
        return self.periodogram, self.periodogram_percentiles, lsp_stats

    def to_acf(self, **kwargs):
        r = elkstats.autocorr(time=self.corrected_lc.time.value, flux=self.corrected_lc.flux.value, **kwargs)
        self.ac_time, self.acf, self.acf_percentiles, acf_stats = r[:4]
        self.stats.update(acf_stats)
        return r

    def get_stats_using_defaults(self):
        (self.rms, self.std, self.MAD,
            self.von_neumann_ratio, self.J_stetson(),
            self.to_periodogram(frequencies=np.arange(0.04, 11, 0.01)), self.to_acf())
        return self.stats

    def get_stats_table(self, name, run_all=False):
        if run_all:
            self.get_stats_using_defaults()
        table_dict = {"name": [name]}
        for k, v in self.stats.items():
            table_dict[k] = [v]
        return Table(table_dict)

    def plot(self, title="auto", **kwargs):
        title = f'Lightcurve for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_lightcurve(self.corrected_lc.time.value, self.corrected_lc.flux.value,
                                       title=title, **kwargs)

    def plot_periodogram(self, frequencies=None, title="auto", **kwargs):
        if self.periodogram_frequencies is None and frequencies is None:
            raise ValueError(("Must either provide an array of frequencies or have already calculated the "
                              "periodogram using `.to_periodogram()`"))
        elif self.periodogram_frequencies is None:
            self.to_periodogram(frequencies=frequencies)

        title = f'Periodogram for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_periodogram(frequencies=self.periodogram_frequencies, power=self.periodogram,
                                        power_percentiles=self.periodogram_percentiles,
                                        peak_freqs=self.stats["peak_freqs"][:self.stats["n_peaks"]],
                                        title=title, **kwargs)

    def plot_acf(self, title="auto", **kwargs):
        if self.ac_time is None:
            self.to_acf()

        title = f'Autocorrelation function for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_acf(time=self.ac_time, acf=self.acf, acf_percentiles=self.acf_percentiles,
                                title=title, **kwargs)

    def analysis_plot(self, name=None, run_all=False, show=True):
        if run_all:
            self.get_stats_using_defaults()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        plt.suptitle(f'{name} (Sector {self.sector})', fontsize="xx-large")

        self.plot(fig=fig, ax=axes[0], show=False, title="Lightcurve")
        self.plot_periodogram(fig=fig, ax=axes[1], show=False, title="Periodogram")
        self.plot_acf(fig=fig, ax=axes[2], show=show, title="Autocorrelation function")

        return fig, axes


class TESSCutLightcurve(SimpleCorrectedLightcurve):
    def __init__(self, radius, lk_search_result=None, tpfs=None,
                 cutout_size=99, percentile=80, n_pca=6, progress_bar=False):

        assert lk_search_result is not None or tpfs is not None, "Must supply either a search result or tpfs"

        # convert radius to degrees if it has units
        if hasattr(radius, 'unit'):
            radius = radius.to(u.deg).value

        self.radius = radius
        self.lk_search_results = lk_search_result
        self._tpfs = tpfs
        self.cutout_size = cutout_size
        self.percentile = percentile
        self.n_pca = n_pca
        self.progress_bar = progress_bar

        self._quality_tpfs = None
        self._basic_lc = None
        self._quality_lc = None
        self._uncorrected_lc = None

    @property
    def tpfs(self):
        if self._tpfs is None:
            self._tpfs = self.lk_search_results.download(cutout_size=(self.cutout_size, self.cutout_size))
        return self._tpfs

    @property
    def sector(self):
        return self.tpfs.sector

    @property
    def quality_tpfs(self):
        if self._quality_tpfs is None:
            self._quality_tpfs = self.tpfs[(self.basic_lc.quality == 0) & (self.basic_lc.flux_err > 0)]
        return self._quality_tpfs

    @property
    def basic_lc(self):
        if self._basic_lc is None:
            self._basic_lc = self.tpfs.to_lightcurve()
        return self._basic_lc

    @property
    def quality_lc(self):
        if self._quality_lc is None:
            self._quality_lc = self.quality_tpfs.to_lightcurve()
        return self._quality_lc

    @property
    def uncorrected_lc(self):
        if self._uncorrected_lc is None:
            self.star_mask = self.circle_aperture()
            self._uncorrected_lc = self.quality_tpfs.to_lightcurve(aperture_mask=self.star_mask)
        return self._uncorrected_lc

    def near_edge(self):
        min_flux = np.min(self.quality_tpfs[0].flux.value)
        min_not_nan = ~np.isnan(min_flux)
        # Also making sure the Sector isn't the one with the Systematic
        not_sector_one = self.tpfs.sector != 1
        min_flux_greater_one = min_flux > 1
        return ~(min_not_nan & not_sector_one & min_flux_greater_one)

    def circle_aperture(self):
        radius_in_pixels = (self.radius * u.deg / TESS_RESOLUTION).to(u.pixel).value
        pix, _ = np.meshgrid(np.arange(self.cutout_size), np.arange(self.cutout_size))
        return (pix - self.cutout_size // 2)**2 + (pix - self.cutout_size // 2)**2 < radius_in_pixels**2

    def correct_lc(self):
        # Time average of the pixels in the TPF:
        max_frame = self.quality_tpfs.flux.value.max(axis=0)

        # This renormalizes any columns which are bright because of straps on the detector
        max_frame -= np.median(max_frame, axis=0)

        # This aperture is any "faint" pixels:
        bkg_aper = max_frame < np.percentile(max_frame, self.percentile)

        # The average light curve of the faint pixels is a good estimate of the scattered light
        scattered_light = self.quality_tpfs.flux.value[:, bkg_aper].mean(axis=1)

        # We can use our background aperture to create pixel time series and then take Principal
        # Components of the data using Singular Value Decomposition. This gives us the "top" trends
        # that are present in the background data
        pca_dm1 = lk.DesignMatrix(self.quality_tpfs.flux.value[:, bkg_aper], name='PCA').pca(self.n_pca)

        # Here we are going to set the priors for the PCA to be located around the
        # flux values of the uncorrected LC
        pca_dm1.prior_mu = np.array([np.median(self.uncorrected_lc.flux.value) for _ in range(self.n_pca)])
        pca_dm1.prior_sigma = np.array([(np.percentile(self.uncorrected_lc.flux.value, 84)
                                         - np.percentile(self.uncorrected_lc.flux.value, 16))
                                        for _ in range(self.n_pca)])

        # The TESS mission pipeline provides co-trending basis vectors (CBVs) which capture common trends
        # in the dataset. We can use these to de-trend out pixel level data. The mission provides
        # MultiScale CBVs, which are at different time scales. In this case, we don't want to use the long
        # scale CBVs, because this may fit out real astrophysical variability. Instead we will use the
        # medium and short time scale CBVs.
        cbvs_1 = lk.correctors.cbvcorrector.load_tess_cbvs(sector=self.quality_tpfs.sector,
                                                           camera=self.quality_tpfs.camera,
                                                           ccd=self.quality_tpfs.ccd,
                                                           cbv_type='MultiScale',
                                                           band=2).interpolate(self.quality_lc)
        cbvs_2 = lk.correctors.cbvcorrector.load_tess_cbvs(sector=self.quality_tpfs.sector,
                                                           camera=self.quality_tpfs.camera,
                                                           ccd=self.quality_tpfs.ccd,
                                                           cbv_type='MultiScale',
                                                           band=3).interpolate(self.quality_lc)

        cbv_dm1 = cbvs_1.to_designmatrix(cbv_indices=np.arange(1, 8))
        cbv_dm2 = cbvs_2.to_designmatrix(cbv_indices=np.arange(1, 8))

        # This combines the different timescale CBVs into a single `designmatrix` object
        cbv_dm_use = lk.DesignMatrixCollection([cbv_dm1, cbv_dm2]).to_designmatrix()

        # We can make a simple basis-spline (b-spline) model for astrophysical variability. This will be a
        # flexible, smooth model. The number of knots is important, we want to only correct for very long
        # term variability that looks like systematics, so here we have 5 knots, the smaller the better
        spline_dm1 = lk.designmatrix.create_spline_matrix(self.quality_tpfs.time.value, n_knots=5)

        # Here we create our design matrix
        self.dm = lk.DesignMatrixCollection([pca_dm1, cbv_dm_use, spline_dm1])

        full_model, systematics_model, self.full_model_normalized = np.ones((3, *self.quality_tpfs.shape))
        if self.progress_bar:
            for i, j in tqdm([(i, j) for i in range(self.cutout_size) for j in range(self.cutout_size)]):
                systematics_model[:, i, j],\
                    full_model[:, i, j], self.full_model_normalized[:, i, j] = self.correct_pixel(i, j)
        else:
            for i, j in [(i, j) for i in range(self.cutout_size) for j in range(self.cutout_size)]:
                systematics_model[:, i, j],\
                    full_model[:, i, j], self.full_model_normalized[:, i, j] = self.correct_pixel(i, j)

        # Calculate Lightcurves
        # NOTE - we are also calculating a lightcurve which does not include the spline model,
        # this is the systematics_model_corrected_lightcurve1
        # TODO: actually use these other lightcurves
        # scattered_light_model_corrected_lightcurve=(self.quality_tpfs - scattered_light[:, None, None]).to_lightcurve(aperture_mask=self.star_mask)
        # systematics_model_corrected_lightcurve=(self.quality_tpfs - systematics_model).to_lightcurve(aperture_mask=self.star_mask)
        self.corrected_lc = (self.quality_tpfs - full_model).to_lightcurve(aperture_mask=self.star_mask)

        self.hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name='time', format='D', array=self.corrected_lc.time.value),
             fits.Column(name='flux', format='D', array=self.corrected_lc.flux.value),
             fits.Column(name='flux_err', format='D', array=self.corrected_lc.flux_err.value),
             fits.Column(name='mag', format='D', array=flux_to_mag(self.corrected_lc.flux.value)),
             fits.Column(name='mag_err', format='D',
                         array=flux_err_to_mag_err(self.corrected_lc.flux.value,
                                                   self.corrected_lc.flux_err.value))]
        )
        self.hdu.header.set('sector', self.sector)

    def correct_pixel(self, i, j):
        # create a lightcurve for just this pixel
        pixel_lightcurve = lk.LightCurve(time=self.quality_tpfs.time.value,
                                         flux=self.quality_tpfs.flux.value[:, i, j],
                                         flux_err=self.quality_tpfs.flux_err.value[:, i, j])

        # Adding a test to make sure there are No Flux_err's <= 0
        # TODO: is this necessary? (no quality_tpfs should make it through right?)
        pixel_lightcurve = pixel_lightcurve[pixel_lightcurve.flux_err > 0]
        r1 = lk.RegressionCorrector(pixel_lightcurve)

        # correct the pixel lightcurve by our design matrix
        r1.correct(self.dm)

        # extract just the systematics components
        systematics_model = (r1.diagnostic_lightcurves['PCA'].flux.value
                             + r1.diagnostic_lightcurves['CBVs'].flux.value)

        # normalise the model for scattered light test
        full_model_normalized = systematics_model + r1.diagnostic_lightcurves['spline'].flux.value

        # add all the components and adjust by the mean to avoid being centered around 0
        full_model = full_model_normalized - r1.diagnostic_lightcurves['spline'].flux.value.mean()

        return systematics_model, full_model, full_model_normalized
