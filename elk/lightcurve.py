import numpy as np
import astropy.units as u
from astropy.table import Table
import lightkurve as lk
from tqdm import tqdm

TESS_RESOLUTION = 21 * u.arcsec / u.pixel


class TESScutLightcurve():
    def __init__(self, radius, lk_search_result=None, tpfs=None, cutout_size=99, percentile=80, n_pca=6):

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

        self._use_tpfs = None
        self._basic_lc = None
        self._uncorrected_lc = None

    @property
    def tpfs(self):
        if self._tpfs is None:
            self._tpfs = self.lk_search_results.download(cutout_size=(self.cutout_size, self.cutout_size))
        return self._tpfs

    @property
    def use_tpfs(self):
        if self._use_tpfs is None:
            self._use_tpfs = self.tpfs[(self.basic_lc.quality == 0) & (self.basic_lc.flux_err > 0)]
        return self._use_tpfs

    @property
    def basic_lc(self):
        if self._basic_lc is None:
            self._basic_lc = self.tpfs.to_lightcurve()
        return self._basic_lc

    @property
    def uncorrected_lc(self):
        if self._uncorrected_lc is None:
            # TODO: why are the data and bkg the same?
            self.star_mask, _ = self.circle_aperture(self.use_tpfs[0].flux.value, self.use_tpfs[0].flux.value)
            self._uncorrected_lc = self.use_tpfs.to_lightcurve(aperture_mask=self.star_mask)
        return self._uncorrected_lc

    def near_edge(self):
        min_flux = np.min(self.use_tpfs[0].flux.value)
        min_not_nan = ~np.isnan(min_flux)
        # Also making sure the Sector isn't the one with the Systematic
        not_sector_one = self.tpfs.sector != 1
        min_flux_greater_one = min_flux > 1
        return ~(min_not_nan & not_sector_one & min_flux_greater_one)

    def circle_aperture(self, data, bkg):
        radius_in_pixels = (self.radius * u.deg / TESS_RESOLUTION).to(u.pixel).value
        data_mask = np.zeros_like(data)
        x_len = np.shape(data_mask)[1]
        y_len = np.shape(data_mask)[2]
        # centers
        cen_x = x_len//2
        cen_y = y_len//2
        bkg_mask = np.zeros_like(bkg)
        bkg_cutoff = np.nanpercentile(bkg, self.percentile)
        for i in range(x_len):
            for j in range(y_len):
                if (i - cen_x)**2 + (j - cen_y)**2 < (radius_in_pixels)**2:   # star mask condition
                    data_mask[0, i, j] = 1

        # TODO: not a fan of variable overwrites
        x_len = np.shape(bkg_mask)[1]
        y_len = np.shape(bkg_mask)[2]
        cen_x = x_len//2
        cen_y = y_len//2
        for i in range(x_len):
            for j in range(y_len):
                if np.logical_and((i - cen_x)**2+(j - cen_y)**2 > (radius_in_pixels)**2, bkg[0, i, j] < bkg_cutoff):  # sky mask condition
                    bkg_mask[0, i, j] = 1

        star_mask = data_mask == 1
        sky_mask = bkg_mask == 1
        return star_mask[0], sky_mask[0]

    def correct_lc(self):
        # Time average of the pixels in the TPF:
        max_frame = self.use_tpfs.flux.value.max(axis=0)

        # This renormalizes any columns which are bright because of straps on the detector
        max_frame -= np.median(max_frame, axis=0)

        # This aperture is any "faint" pixels:
        bkg_aper = max_frame < np.percentile(max_frame, self.percentile)

        # The average light curve of the faint pixels is a good estimate of the scattered light
        scattered_light = self.use_tpfs.flux.value[:, bkg_aper].mean(axis=1)

        # We can use our background aperture to create pixel time series and then take Principal
        # Components of the data using Singular Value Decomposition. This gives us the "top" trends
        # that are present in the background data
        pca_dm1 = lk.DesignMatrix(self.use_tpfs.flux.value[:, bkg_aper], name='PCA').pca(self.n_pca)

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
        # TODO: reduce these .to_lightcurve() calls
        cbvs_1 = lk.correctors.cbvcorrector.load_tess_cbvs(sector=self.use_tpfs.sector,
                                                           camera=self.use_tpfs.camera,
                                                           ccd=self.use_tpfs.ccd,
                                                           cbv_type='MultiScale',
                                                           band=2).interpolate(self.use_tpfs.to_lightcurve())
        cbvs_2 = lk.correctors.cbvcorrector.load_tess_cbvs(sector=self.use_tpfs.sector,
                                                           camera=self.use_tpfs.camera,
                                                           ccd=self.use_tpfs.ccd,
                                                           cbv_type='MultiScale',
                                                           band=3).interpolate(self.use_tpfs.to_lightcurve())

        cbv_dm1 = cbvs_1.to_designmatrix(cbv_indices=np.arange(1, 8))
        cbv_dm2 = cbvs_2.to_designmatrix(cbv_indices=np.arange(1, 8))

        # This combines the different timescale CBVs into a single `designmatrix` object
        cbv_dm_use = lk.DesignMatrixCollection([cbv_dm1, cbv_dm2]).to_designmatrix()

        # We can make a simple basis-spline (b-spline) model for astrophysical variability. This will be a
        # flexible, smooth model. The number of knots is important, we want to only correct for very long
        # term variabilty that looks like systematics, so here we have 5 knots, the smaller the better
        spline_dm1 = lk.designmatrix.create_spline_matrix(self.use_tpfs.time.value, n_knots=5)

        # Here we create our design matrix
        dm1 = lk.DesignMatrixCollection([pca_dm1, cbv_dm_use, spline_dm1])

        full_model, systematics_model, full_model_Normalized = np.ones((3, *self.use_tpfs.shape))
        for idx in tqdm(range(self.use_tpfs.shape[1])):
            for jdx in range(self.use_tpfs.shape[2]):
                pixel_lightcurve = lk.LightCurve(time=self.use_tpfs.time.value,
                                                 flux=self.use_tpfs.flux.value[:, idx, jdx],
                                                 flux_err=self.use_tpfs.flux_err.value[:, idx, jdx])

                # Adding a test to make sure there are No Flux_err's <= 0
                pixel_lightcurve = pixel_lightcurve[pixel_lightcurve.flux_err > 0]

                r1 = lk.RegressionCorrector(pixel_lightcurve)

                # Correct the pixel light curve by our design matrix
                r1.correct(dm1)

                # Extract just the systematics components
                systematics_model[:, idx, jdx] = (r1.diagnostic_lightcurves['PCA'].flux.value +
                                                  r1.diagnostic_lightcurves['CBVs'].flux.value)
                # Add all the components
                full_model[:, idx, jdx] = (r1.diagnostic_lightcurves['PCA'].flux.value +
                                           r1.diagnostic_lightcurves['CBVs'].flux.value +
                                           r1.diagnostic_lightcurves['spline'].flux.value)

                # Making so the model isn't centered around 0
                full_model[:, idx, jdx] -= r1.diagnostic_lightcurves['spline'].flux.value.mean()

                # Making Normalized Model For the Test of Scattered Light
                full_model_Normalized[:, idx, jdx] = (r1.diagnostic_lightcurves['PCA'].flux.value +
                                                      r1.diagnostic_lightcurves['CBVs'].flux.value +
                                                      r1.diagnostic_lightcurves['spline'].flux.value)

        self.full_model_normalized = full_model_Normalized

        # Calculate Lightcurves
        # NOTE - we are also calculating a lightcurve which does not include the spline model,
        # this is the systematics_model_corrected_lightcurve1
        # TODO: actually use these other lightcurves
        # scattered_light_model_corrected_lightcurve=(self.use_tpfs - scattered_light[:, None, None]).to_lightcurve(aperture_mask=self.star_mask)
        # systematics_model_corrected_lightcurve=(self.use_tpfs - systematics_model).to_lightcurve(aperture_mask=self.star_mask)
        full_corrected_lightcurve = (self.use_tpfs - full_model).to_lightcurve(aperture_mask=self.star_mask)

        self.full_corrected_lightcurve_table = Table([full_corrected_lightcurve.time.value,
                                                      full_corrected_lightcurve.flux.value,
                                                      full_corrected_lightcurve.flux_err.value],
                                                     names=('time', 'flux', 'flux_err'))