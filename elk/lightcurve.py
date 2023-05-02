import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, vstack, Column
from astropy.timeseries import LombScargle
import lightkurve as lk
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import imageio
from astroquery.simbad import Simbad
from IPython.display import HTML
import astropy.coordinates as coord

from .utils import flux_to_mag, flux_err_to_mag_err
import elk.plot as elkplot
import elk.stats as elkstats


__all__ = ["BasicLightcurve", "TESSCutLightcurve"]


TESS_RESOLUTION = 21 * u.arcsec / u.pixel


class BasicLightcurve():
    def __init__(self, time=None, flux=None, flux_err=None, sector=None, fits_path=None, hdu_index=None,
                 periodogram_freqs=np.arange(0.04, 11, 0.01)):
        """A basic lightcurve class for calculating statistics of corrected lightcurves and plotting them.

        This class can either be instantiated manually using time, flux and flux_err values, or loaded in from
        a fits file that has previously been created.

        Parameters
        ----------
        time : :class:`~numpy.ndarray`, optional
            Time of observations, by default None
        flux : :class:`~numpy.ndarray`, optional
            Flux from each observations, by default None
        flux_err : :class:`~numpy.ndarray`, optional
            Errors on the flux, by default None
        sector : `int`, optional
            TESS sector in which observations were taken, by default None
        fits_path : `str`, optional
            Path to a fits file containing the lightcurve, by default None
        hdu_index : `int`, optional
            Index of the HDU containing the lightcurve, by default None
        periodogram_freqs : :class:`numpy.ndarray`, optional
            Frequencies at which to evaluate any periodograms, by default np.arange(0.04, 11, 0.01)
        """
        # check that either data or a file has been given
        has_data = time is not None and flux is not None and flux_err is not None and sector is not None
        has_file = fits_path is not None and hdu_index is not None
        assert has_data or has_file, "Must either provide data or a fits file and HDU index"

        # if the user has given us a file then load it into the class
        if has_file:
            with fits.open(fits_path) as hdul:
                hdu = hdul[hdu_index]
                self.corrected_lc = lk.LightCurve(time=hdu.data["time"] * u.day,
                                                  flux=hdu.data["flux"] * u.electron / u.s,
                                                  flux_err=hdu.data["flux_err"] * u.electron / u.s)
                self.sector = hdu.header["sector"]
        # otherwise create a lightcurve directly
        else:
            self.corrected_lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            self.sector = sector

        # set up an HDU for saving lightcurves
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

        # prep some class variables for the stats helper functions
        self.stats = {}
        self.periodogram_freqs = periodogram_freqs
        self.periodogram_peaks = None
        self.ac_time = None

    def __repr__(self):
        return f"<{self.__class__.__name__}: TESS Sector {self.sector}, {len(self.corrected_lc)} timesteps>"

    @property
    def normalized_flux(self):
        """The normalised flux of the lightcurve"""
        return self.corrected_lc.flux.value / np.median(self.corrected_lc.flux.value)

    @property
    def rms(self):
        """The root-mean-squared normalised flux of the lightcurve"""
        self.stats["rms"] = np.sqrt(np.mean(self.normalized_flux**2))
        return self.stats["rms"]

    @property
    def std(self):
        """The standard deviation of the normalised flux of the lightcurve"""
        self.stats["std"] = np.std(self.normalized_flux)
        return self.stats["std"]

    @property
    def MAD(self):
        """The median absolute deviation of the normalised flux of the lightcurve"""
        self.stats["MAD"] = elkstats.get_MAD(self.normalized_flux)
        return self.stats["MAD"]
    
    @property
    def sigmaG(self):
        """The sigmaG of the normalised flux of the lightcurve"""
        self.stats["sigmaG"] = elkstats.get_sigmaG(self.normalized_flux)
        return self.stats["sigmaG"]

    @property
    def skewness(self):
        """The skewness of the flux of the lightcurve"""
        self.stats["skewness"] = elkstats.get_skewness(self.corrected_lc.flux.value)
        return self.stats["skewness"]

    @property
    def von_neumann_ratio(self):
        """The Von Neumann ratio for the normalised flux of the lightcurve. See
        :class:`elk.stats.von_neumann_ratio` for more information"""
        self.stats["von_neumann_ratio"] = elkstats.von_neumann_ratio(self.normalized_flux)
        return self.stats["von_neumann_ratio"]

    def J_stetson(self, **kwargs):
        """Calculate the J Stetson statistic for the lightcurve

        Keyword arguments are passed directly to :class:`elk.stats.J_stetson`

        Returns
        -------
        J_Stetson
            The J Stetson statistic
        """
        mag_err = flux_err_to_mag_err(self.corrected_lc.flux.value, self.corrected_lc.flux_err.value)
        self.stats["J_Stetson"] = elkstats.J_stetson(time=self.corrected_lc.time.value,
                                                     mag=flux_to_mag(self.corrected_lc.flux.value),
                                                     mag_err=mag_err, **kwargs)
        return self.stats["J_Stetson"]

    def to_periodogram(self, frequencies=None, **kwargs):
        """Construct a periodogram using the lightcurve

        Parameters
        ----------
        frequencies : :class:`numpy.ndarray`, optional
            Frequencies at which to evaluate the periodogram, by default self.periodogram_freqs
        **kwargs : `various`
            Keyword arguments to pass to :class:`elk.stats.periodogram`

        Returns
        -------
        Same as :class:`elk.stats.periodogram`
        """
        # use default frequencies
        if frequencies is None:
            frequencies = self.periodogram_freqs

        self.periodogram, self.periodogram_percentiles,\
            lsp_stats = elkstats.periodogram(self.corrected_lc.time.value, self.corrected_lc.flux.value,
                                             self.corrected_lc.flux_err.value, frequencies=frequencies,
                                             **kwargs)
        self.periodogram_freqs = frequencies
        self.periodogram_peaks = lsp_stats["peak_freqs"]
        self.stats.update(lsp_stats)
        return self.periodogram, self.periodogram_percentiles, lsp_stats

    def to_acf(self, **kwargs):
        """Calculate the autocorrelation function for the lightcurve

        Keyword arguments are passed directly to :class:`elk.stats.autocorr`

        Returns
        -------
        Same as :class:`elk.stats.periodogram`
        """
        r = elkstats.autocorr(time=self.corrected_lc.time.value, flux=self.corrected_lc.flux.value, **kwargs)
        self.ac_time, self.acf, self.acf_percentiles, acf_stats = r[:4]
        self.stats.update(acf_stats)
        return r

    def get_stats_using_defaults(self):
        """Generate all statistics for the lightcurve **using default settings**

        NOTE: This may not be optimal for your use case, be sure to consider each setting! By default we use
        a list of frequencies generated with ``np.arange(0.04, 11, 0.01)`` for the periodogram.

        Returns
        -------
        stats : `dict`
            A dictionary of the various statistics (also stored in ``self.stats``)
        """
        (self.rms, self.std, self.MAD, self.sigmaG, self.skewness,
            self.von_neumann_ratio, self.J_stetson(),
            self.to_periodogram(frequencies=np.arange(0.04, 11, 0.01)), self.to_acf())
        return self.stats

    def get_stats_table(self, name, run_all=False):
        """Create an Astropy Table of the statistics (for aggregation with other lightcurves)

        Parameters
        ----------
        name : `str`
            Identifier for this lightcurve (e.g. name of cluster)
        run_all : `bool`, optional
            Whether to run all statistics **using default settings**
            (see ``.get_stats_using_defaults()``), by default False

        Returns
        -------
        stats_table : :class:`~astropy.table.Table`
            Table of statistics
        """
        if run_all:
            self.get_stats_using_defaults()
        table_dict = {"name": [name]}
        for k, v in self.stats.items():
            table_dict[k] = [v]
        return Table(table_dict)

    def plot(self, title="auto", fold_period=None, **kwargs):
        """Plot the lightcurve

        Parameters
        ----------
        title : `str`, optional
            Title for the plot, by default "auto" (resulting in "Lightcurve for Sector ...")
        fold_period : `float`, optional
            Period on which to fold the light curve, if None then no folding is done, by default None
        **kwargs: `various`
            Keyword arguments passed to :class:`elk.plot.plot_lightcurve`

        Returns
        -------
        fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
            Figure and axis on which the lightcurve has been plotted
        """
        title = f'Lightcurve for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_lightcurve(self.corrected_lc.time.value, self.corrected_lc.flux.value,
                                       title=title, fold_period=fold_period, **kwargs)

    def plot_periodogram(self, frequencies=None, title="auto", **kwargs):
        """Plot the periodogram for the lightcurve

        Parameters
        ----------
        frequencies : :class:`~numpy.ndarray`
            Frequencies at which periodogram evaluated, if None then periodogram must have already been
            generated using ``.to_periodogram()``
        title : `str`, optional
            Title for the plot, by default "auto" (resulting in "Periodogram for Sector ...")
        **kwargs: `various`
            Keyword arguments passed to :class:`elk.plot.plot_periodogram`

        Returns
        -------
        fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
            Figure and axis on which the periodogram has been plotted
        """
        # get the periodogram if it hasn't already been run
        if "peak_freqs" not in self.stats:
            self.to_periodogram(frequencies=frequencies)

        title = f'Periodogram for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_periodogram(frequencies=self.periodogram_freqs, power=self.periodogram,
                                        power_percentiles=self.periodogram_percentiles,
                                        peak_freqs=self.stats["peak_freqs"][:self.stats["n_peaks"]],
                                        fap=self.stats["FAP"], title=title, **kwargs)

    def plot_acf(self, title="auto", **kwargs):
        """Plot the autocorrelation function

        Parameters
        ----------
        title : `str`, optional
            Title for the plot, by default "auto" (resulting in "Autocorrelation function for Sector ...")
        **kwargs: `various`
            Keyword arguments passed to :class:`elk.plot.plot_acf`

        Returns
        -------
        fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
            Figure and axis on which the autocorrelation function has been plotted
        """
        if self.ac_time is None:
            self.to_acf()

        title = f'Autocorrelation function for Sector {self.sector}' if title == "auto" else title
        return elkplot.plot_acf(time=self.ac_time, acf=self.acf, acf_percentiles=self.acf_percentiles,
                                title=title, **kwargs)

    def analysis_plot(self, name=None, run_all=False, show=True):
        """Plot a 3-panel plot of the lightcurve for quick analysis

        Plot includes the lightcurve, periodogram and autocorrelation function.

        Parameters
        ----------
        name : `str`
            Identifier for this lightcurve (e.g. name of cluster)
        run_all : `bool`, optional
            Whether to run all statistics **using default settings** (see ``.get_stats_using_defaults()``),
            by default False
        show : `bool`, optional
            Whether to immediately show the plot, by default True

        Returns
        -------
        fig, axes : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
            Figure and axes on which the analysis has been plotted
        """
        # run everything if the user wants to
        if run_all:
            self.get_stats_using_defaults()

        fig, axes = plt.subplots(1, 3, figsize=(30, 5))

        name = "Lightcurve Analysis Plot" if name is None else name
        plt.suptitle(f'{name} (Sector {self.sector})', fontsize="xx-large")

        self.plot(fig=fig, ax=axes[0], show=False, title="Lightcurve")
        self.plot_periodogram(fig=fig, ax=axes[1], show=False, title="Periodogram")
        self.plot_acf(fig=fig, ax=axes[2], show=show, title="Autocorrelation function")

        return fig, axes


class TESSCutLightcurve(BasicLightcurve):
    def __init__(self, radius, lk_search_result=None, tpfs=None,
                 cutout_size=99, percentile=80, n_pca=6, spline_knots=5,
                 periodogram_freqs=np.arange(0.04, 11, 0.01), save_pixel_periodograms=True,
                 progress_bar=False):
        """A lightcurve constructed from a TESSCut search with various correction functionalities

        Parameters
        ----------
        radius : `float`
            Radius of the cluster in degrees.
        lk_search_result : :class:`lightkurve.SearchResult`, optional
            Search result from a LightKurve tesscut call, by default None
        tpfs : :class:`lightkurve.TessTargetPixelFile`, optional
            Target pixel files, by default None
        cutout_size : `int`, optional
            Cutout size for the TESSCut call, by default 99
        percentile : `int`, optional
            Which percentile to use in the upper limit calculation, by default 80
        n_pca : `int`, optional
            Number of principle components to use in the DesignMatrix, by default 6
        spline_knots: `int`, optional
            Number of knots to include in the spline corrector, by default 5. If None, spline corrector
            will not be used
        periodogram_freqs : :class:`numpy.ndarray`, optional
            Frequencies at which to evaluate any periodograms, by default np.arange(0.04, 11, 0.01)
        progress_bar : `bool`, optional
            Whether to show a progress bar of pixel correction, by default False
        """

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
        self.spline_knots = spline_knots
        self.save_pixel_periodograms = save_pixel_periodograms
        self.progress_bar = progress_bar

        self.periodogram_freqs = periodogram_freqs
        self.pixel_periodograms = None if not self.save_pixel_periodograms else [None for _ in
                                                                                 range(self.cutout_size**2)]

        # defaults for the cached variables
        self._quality_tpfs = None
        self._basic_lc = None
        self._quality_lc = None
        self._uncorrected_lc = None

        # prep some class variables for the stats helper functions
        self.stats = {}
        self.ac_time = None

    @property
    def tpfs(self):
        """All target pixel files for the lightcurve"""
        if self._tpfs is None:
            self._tpfs = self.lk_search_results.download(cutout_size=(self.cutout_size, self.cutout_size))
        return self._tpfs

    @property
    def sector(self):
        """TESS sector in which observations were taken"""
        return self.tpfs.sector

    @property
    def quality_tpfs(self):
        """Target pixel files that have a quality flag of 0 and a positive flux_err"""
        if self._quality_tpfs is None:
            self._quality_tpfs = self.tpfs[(self.basic_lc.quality == 0) & (self.basic_lc.flux_err > 0)]
        return self._quality_tpfs

    @property
    def basic_lc(self):
        """Lightcurve constructed using **all** target pixel files"""
        if self._basic_lc is None:
            self._basic_lc = self.tpfs.to_lightcurve()
        return self._basic_lc

    @property
    def quality_lc(self):
        """Lightcurve constructed using only quality target pixel files"""
        if self._quality_lc is None:
            self._quality_lc = self.quality_tpfs.to_lightcurve()
        return self._quality_lc

    @property
    def uncorrected_lc(self):
        """Lightcurve constructed using only quality target pixel files and a circle aperture mask"""
        if self._uncorrected_lc is None:
            self.star_mask = self.circle_aperture()
            self._uncorrected_lc = self.quality_tpfs.to_lightcurve(aperture_mask=self.star_mask)
        return self._uncorrected_lc

    def fails_quality_test(self):
        """Test whether this lightcurve has (1) any quality TPFs, (2) that observations are not within ~0.28
        degrees of the edge of the detector (such that NaN values would appear in the cutout) and (3) that it
        isn't part of TESS Sector 1, which has an unremovable systematic.

        Returns
        -------
        flag : `bool`
            Flag of whether the test was passed
        """
        # check that there is at least one good quality TPF
        if not np.any((self.basic_lc.quality == 0) & (self.basic_lc.flux_err > 0)):
            return True

        min_flux = np.min(self.quality_tpfs[0].flux.value)
        min_not_nan = ~np.isnan(min_flux)
        # Also making sure the Sector isn't the one with the Systematic
        not_sector_one = self.tpfs.sector != 1
        min_flux_greater_one = min_flux > 1
        return ~(min_not_nan & not_sector_one & min_flux_greater_one)

    def circle_aperture(self):
        """Generate a circular aperture mask based on the radius and cutout_size of this lightcurve

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Aperture mask
        """
        # convert the radius to pixels based on the TESS resolution
        radius_in_pixels = (self.radius * u.deg / TESS_RESOLUTION).to(u.pixel).value

        # mask the grid of pixels based on this radius
        pix_x, pix_y = np.meshgrid(np.arange(self.cutout_size), np.arange(self.cutout_size))
        return (pix_x - self.cutout_size // 2)**2 + (pix_y - self.cutout_size // 2)**2 < radius_in_pixels**2

    def correct_lc(self):
        """Correct the lightcurve using the method described in Wainer+2023"""
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
        pca_dm = lk.DesignMatrix(self.quality_tpfs.flux.value[:, bkg_aper], name='PCA').pca(self.n_pca)

        # Here we are going to set the priors for the PCA to be located around the
        # flux values of the uncorrected LC
        pca_dm.prior_mu = np.array([np.median(self.uncorrected_lc.flux.value) for _ in range(self.n_pca)])
        pca_dm.prior_sigma = np.array([(np.percentile(self.uncorrected_lc.flux.value, 84)
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
                                                           band=2).interpolate(self.quality_lc,
                                                                               extrapolate=True)
        cbvs_2 = lk.correctors.cbvcorrector.load_tess_cbvs(sector=self.quality_tpfs.sector,
                                                           camera=self.quality_tpfs.camera,
                                                           ccd=self.quality_tpfs.ccd,
                                                           cbv_type='MultiScale',
                                                           band=3).interpolate(self.quality_lc,
                                                                               extrapolate=True)

        cbv_dm1 = cbvs_1.to_designmatrix(cbv_indices=np.arange(1, 8))
        cbv_dm2 = cbvs_2.to_designmatrix(cbv_indices=np.arange(1, 8))

        # This combines the different timescale CBVs into a single `designmatrix` object
        cbv_dm_use = lk.DesignMatrixCollection([cbv_dm1, cbv_dm2]).to_designmatrix()

        # We can make a simple basis-spline (b-spline) model for astrophysical variability. This will be a
        # flexible, smooth model. The number of knots is important, we want to only correct for very long
        # term variability that looks like systematics, so here we have 5 knots, the smaller the better
        spline_dm1 = lk.designmatrix.create_spline_matrix(self.quality_tpfs.time.value, 
                                                            n_knots=self.spline_knots)
        self.dm = lk.DesignMatrixCollection([pca_dm, cbv_dm_use, spline_dm1])

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
        self.corrected_lc = (self.quality_tpfs - full_model).to_lightcurve(aperture_mask=self.star_mask)
            
        self.systematics_corrected_lc = (self.quality_tpfs - systematics_model).to_lightcurve(aperture_mask=self.star_mask)

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
        """Correct an individual pixel of the lightcurve

        Parameters
        ----------
        i, j : `int`
            Indices for the pixel

        Returns
        -------
        systematics_model : :class:`numpy.ndarray`
            A model for the systematics in the pixel
        full_model : :class:`numpy.ndarray`
            The full model for the pixel lightcurve
        full_model_normalized : :class:`numpy.ndarray`
            The normalised model for the pixel lightcurve
        """
        # create a lightcurve for just this pixel
        pixel_lightcurve = lk.LightCurve(time=self.quality_tpfs.time.value,
                                         flux=self.quality_tpfs.flux.value[:, i, j],
                                         flux_err=self.quality_tpfs.flux_err.value[:, i, j])

        # create a regression corrector based on the design matrix
        r1 = lk.RegressionCorrector(pixel_lightcurve)

        # correct the pixel lightcurve by our design matrix
        corrected_lc = r1.correct(self.dm)

        if self.save_pixel_periodograms:
            lsp = LombScargle(t=corrected_lc["time"], y=corrected_lc["flux"], dy=corrected_lc["flux_err"])
            self.pixel_periodograms[i * self.cutout_size + j] = lsp.power(self.periodogram_freqs / u.day)

        # extract just the systematics components
        systematics_model = (r1.diagnostic_lightcurves['PCA'].flux.value
                             + r1.diagnostic_lightcurves['CBVs'].flux.value)

        # normalise the model for scattered light test
        full_model_normalized = systematics_model + r1.diagnostic_lightcurves['spline'].flux.value

        # add all the components and adjust by the mean to avoid being centered around 0
        full_model = full_model_normalized - r1.diagnostic_lightcurves['spline'].flux.value.mean()
        
        del r1

        return systematics_model, full_model, full_model_normalized

    def diagnose_lc_periodogram(self, output_path, freq_bins='auto', identifier='', query_simbad=True):
        """Create gif showing pixels that contribute power to periodogram for different frequency ranges

        The GIF has 3 panels, the first shows the overall TPFs and aperture, the second shows the maximum
        power in each pixel for this frequency bin (and is annotated with the frequency/range used for this
        frame) and the last panel shows the overall ensemble light curve periodogram with the frame's
        frequency range highlighted.

        NOTE: `self.correct_lc` must have already been run and `self.save_pixel_periodograms` must be true.

        Parameters
        ----------
        output_path : `str`
            Path to a folder in which to output the gif and frames
        freq_bins : `str` or `int` or :class:`~numpy.ndarray`, optional
            Frequency bins to use for the GIF. Either 'auto' to create a frame for each peak in the
            periodogram, an integer to use log-spaced bins in the periodogram range or an array of bin edges,
            by default 'auto'
        identifier : `str`, optional
            An identifier for this target to put in the title (e.g. the cluster name), by default ''
        """
        # ensure the necessary data is available to run this
        assert self.save_pixel_periodograms, ("Pixel periodograms not available. Set "
                                              "`self.save_pixel_periodograms=True` and re-run "
                                              "`self.correct_lc`")

        # mask the pixel powers to only be for pixels in the aperture
        assert hasattr(self, "star_mask"), "No aperture mask found - have you run `self.correct_lc`?"
        aperture_powers = np.asarray(self.pixel_periodograms)[self.star_mask.flatten()]
        assert len(aperture_powers) > 0, "No pixel periodograms found - did you run `self.correct_lc`?"

        # ensure output folder and subfolder exists
        assert os.path.exists(output_path), f"No folder found at output_path: {output_path}"
        if not os.path.exists(os.path.join(output_path, 'diagnostics')):
            os.mkdir(os.path.join(output_path, 'diagnostics'))

        if query_simbad:
            simbad_results = None

        # convert input into bin bounds
        if isinstance(freq_bins, str):
            assert freq_bins == "auto", "`freq_bins` can only be a str if it is equal to 'auto'"
            self.to_periodogram()
            bounds = list(zip(self.stats["peak_left_edge"], self.stats["peak_freqs"],
                              self.stats["peak_right_edge"]))
        else:
            if isinstance(freq_bins, int):
                freq_bins = np.logspace(min(self.periodogram_freqs), max(self.periodogram_freqs), freq_bins)
            bounds = list(zip(freq_bins[:-1], 0.5 *(freq_bins[:-1] + freq_bins[1:]), freq_bins[1:]))

        # create a separate frame for each frequency bin
        i = 0
        for lower, center, upper in bounds:
            if lower > max(self.periodogram_freqs):
                continue
            # start a three panel figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            # plot the entire target pixel file in the first axis
            self.quality_tpfs.plot(frame=len(self.quality_tpfs) // 2, ax=axes[0])

            # create a circle around the aperture
            x_range = axes[0].get_xlim()[1] - axes[0].get_xlim()[0]
            y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            circle = plt.Circle(xy=(axes[0].get_xlim()[0] + x_range / 2, axes[0].get_ylim()[0] + y_range / 2),
                                radius=(self.radius * u.deg / TESS_RESOLUTION).to(u.pixel).value,
                                edgecolor="red", facecolor="none", linewidth=2)
            axes[0].add_artist(circle)
            axes[0].set_title(f'{identifier} ({self.quality_tpfs[0].ra}, {self.quality_tpfs[0].dec})')

            # get the indices of the aperture pixels and plot a marker on each pixel that matches
            pixel_inds = np.argwhere(self.star_mask)
            axes[0].scatter(axes[0].get_xlim()[0] + pixel_inds[:, 1] + 0.5,
                            axes[0].get_ylim()[0] + pixel_inds[:, 0] + 0.5,
                            c='r', s=1, alpha=0.5)

            # create a mask for the frequency range
            frequency_mask = (self.periodogram_freqs >= lower) & (self.periodogram_freqs < upper)

            # get power in each pixel that is within the aperture and for the given frequency range
            pixel_powers = aperture_powers[:, frequency_mask]

            # get the max power in each pixel
            pixel_max_power = np.zeros([self.cutout_size, self.cutout_size], dtype='float64')
            pixel_max_power[self.star_mask] = np.max(pixel_powers, axis=1)

            if query_simbad:
                query_pixels = np.argwhere(pixel_max_power > 0.9 * np.max(pixel_max_power))

                world_values = np.array(self.quality_tpfs.wcs.pixel_to_world_values(np.fliplr(query_pixels)))
                ra, dec = world_values[:, 0], world_values[:, 1]

                # create custom simbad that includes variable star columns
                var_Simbad = Simbad()
                var_Simbad.add_votable_fields('v*', 'otype', 'flux(V)')

                # query SIMBAD for a region that fully encloses the pixel
                pixel_radius = (TESS_RESOLUTION * (1 * u.pixel)).to(u.deg) * np.sqrt(2)
                query_result = var_Simbad.query_region(coord.SkyCoord(ra=ra, dec=dec,
                                                       unit=(u.deg, u.deg), frame='icrs'),
                                                       radius=pixel_radius)

                query_result=query_result['MAIN_ID', 'RA', 'DEC', 'V__vartyp', 'V__Vmax', 'V__R_Vmax', 
                                          'V__magtyp', 'V__UpVmin', 'V__Vmin', 'V__R_Vmin', 'V__UpPeriod', 
                                          'V__period', 'V__R_period', 'OTYPE', 'FLUX_V']
                
                if query_result is not None:
                    # only selecting the columns we want
                    query_result = query_result['MAIN_ID', 'RA', 'DEC', 'V__vartyp', 'V__Vmax', 'V__R_Vmax',
                                                'V__magtyp', 'V__UpVmin', 'V__Vmin', 'V__R_Vmin',
                                                'V__UpPeriod', 'V__period', 'V__R_period', 'OTYPE', 'FLUX_V']
                    
                    # add columns indicating which peak these are associated with
                    query_result.add_column(Column([round(center, 3)]), name='peak_freq')
                    query_result.add_column(Column([round(lower, 3)]), name='peak_lower')
                    query_result.add_column(Column([round(upper, 3)]), name='peak_upper')

                    if simbad_results is None:
                        # if this is the first query then just save
                        simbad_results = query_result
                    else:
                        # otherwise concatenate with previous queries
                        simbad_results = vstack([simbad_results, query_result],
                                                join_type="exact", metadata_conflicts="silent")
                                                

            # plot the max power in each pixel in the same range as the right panel
            im = axes[1].imshow(pixel_max_power, extent=list(axes[0].get_xlim()) + list(axes[0].get_ylim()),
                                origin='lower', cmap='Greys', vmax=max(self.periodogram))

            cbar = fig.colorbar(im, ax=axes[1])
            cbar.set_label('LS periodogram power')
            axes[1].set_title('Maximum Lomb Scargle Power')

            axes[1].annotate((f'Frequency: {(lower + upper) / 2:1.2f} 1/day\n'
                              f'Range: [{lower:1.2f}, {upper:1.2f}] 1/day'), xy=(0.5, 0.95),
                             xycoords="axes fraction", ha="center", va="top")

            circle = plt.Circle(xy=(axes[0].get_xlim()[0] + x_range / 2, axes[0].get_ylim()[0] + y_range / 2),
                                radius=(self.radius * u.deg / TESS_RESOLUTION).to(u.pixel).value,
                                edgecolor="red", facecolor="none", linestyle="dotted")
            axes[1].add_artist(circle)

            # plot the LS periodogram for the ensemble cluster LC
            fig, axes[2] = self.plot_periodogram(self.periodogram_freqs, fig=fig, ax=axes[2], show=False,
                                                 title="Ensemble Light Curve Periodogram")
            axes[2].axvspan(lower, upper, color="lightgrey", zorder=-1)

            # save and close the figure and move on to the next
            fig.savefig(os.path.join(output_path, 'diagnostics', f'{identifier}_gif_plot_frame_{i}.png'),
                        bbox_inches="tight")
            plt.close(fig)
            i += 1

        simbad_results["MAIN_ID"] = simbad_results["MAIN_ID"].astype('str')
        simbad_results["OTYPE"] = simbad_results["OTYPE"].astype('str')

        # save the simbad results
        simbad_results.write(os.path.join(output_path, 'diagnostics',
                                          f'{identifier}_simbad_results.hdf5'),
                             path='data', serialize_meta=True, overwrite=True)


        # convert individual frames to a GIF
        gif_path = os.path.join(output_path, 'diagnostics', f'{identifier}_pixel_power_gif.gif')
        with imageio.get_writer(gif_path, mode='I', fps=2.5) as writer:
            for i in range(len(bounds)):
                writer.append_data(imageio.imread(os.path.join(output_path, 'diagnostics',
                                                               f'{identifier}_gif_plot_frame_{i}.png')))
                
        #get GIF back
        gif = HTML(f'<img src="{gif_path}">')

        return gif, simbad_results 
