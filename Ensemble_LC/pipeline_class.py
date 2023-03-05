import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import scipy
from astropy.table import Table
from astropy.table import Column
from tqdm import tqdm

import os.path
import gc


class ClusterPipeline:
    def __init__(self, radius, cluster_age, output_path="./", cluster_name=None, location=None,
                 upper_limit_method=1, percentile=80, cutout_size=99, scattered_light_frequency=5,
                 debug=False):
        """Class for generating lightcurves from TESS cutouts of a specific cluster

        Parameters
        ----------
        radius : `float`
            Radius of the cluster in #TODO What units?
        cluster_age : `float`
            Age of the cluster in #TODO What units?
        output_path : `str`, optional
            Path to a folder in which to save outputs - must have subfolders Corrected_LCs/ and Figures/LCs/,
            by default "./"
        cluster_name : `str`, optional
            Name of the cluster, by default None
        location : `str`, optional
            Location of the cluster #TODO What format here?, by default None
        upper_limit_method : `int`, optional
            Which upper limit method to use #TODO more details, by default 1
        percentile : `int`, optional
            Which percentile to use in the upper limit calculation, by default 80
        cutout_size : `int`, optional
            How large to make the cutout, by default 99
        scattered_light_frequency : `int`, optional
            Frequency at which to check for scattered light, by default 5
        debug : `bool`, optional
            #TODO DELETE THIS, by default False
        """

        assert cluster_name is not None or location is not None,\
            "Must provide at least one of `cluster_name` and `location`"

        self.output_path = output_path
        self.radius = radius
        self.cluster_age = cluster_age
        self.callable = cluster_name if cluster_name is not None else location
        self.cluster_name = cluster_name
        self.location = location
        self.upper_limit_method = upper_limit_method
        self.percentile = percentile
        self.cutout_size = cutout_size
        self.scattered_light_frequency = scattered_light_frequency
        self.debug = debug

    def __repr__(self):
        return f"<{self.__class__.__name__} - {self.callable}>"

    def previously_downloaded(self):
        """Check whether the files have previously been downloaded for this cluster

        Returns
        -------
        exists : `bool`
            Whether the file exists
        """
        path = os.path.join(self.output_path, 'Corrected_LCs/', str(self.callable) + 'output_table.fits')
        return os.path.exists(path)

    def has_tess_data(self):
        """Check whether TESS has data on the cluster

        Returns
        -------
        has_data : `bool`
            Whether these is at least one observation in TESS
        """
        # search for the cluster in TESS using lightkurve
        search = lk.search_tesscut(self.callable)
        print(f'{self.callable} has {len(search)} observations')
        return len(search) > 0

    def downloadable(self, ind):
        # Using a Try statement to see if we can download the cluster data If we cannot
        try:
            self.tpfs = lk.search_tesscut(self.callable)[ind].download(cutout_size=(self.cutout_size,
                                                                                    self.cutout_size))
        except:
            print("No Download")
            self.tpfs = None

    def near_edge(self):
        # Only selecting time steps that are good, or have quality == 0
        t1 = self.tpfs[np.where(self.tpfs.to_lightcurve().quality == 0)]

        min_not_nan = ~np.isnan(np.min(t1[0].flux.value))
        # Also making sure the Sector isn't the one with the Systematic
        not_sector_one = self.tpfs.sector != 1
        min_flux_greater_one = np.min(t1[0].flux.value) > 1
        return ~(min_not_nan & not_sector_one & min_flux_greater_one)

    def get_upper_limit(self, dataDistribution):
        # TODO: Several of these methods don't work
        if self.upper_limit_method == 1:
            return np.nanpercentile(dataDistribution, self.percentile)

        elif self.upper_limit_method == 2:
            hist = np.histogram(dataDistribution, bins=BINS, range=(0, 3000))  # Bin the data
            return hist[1][np.argmax(hist[0])]  # Return the flux corresponding to the most populated bin

        elif self.upper_limit_method == 3:
            pass

        elif self.upper_limit_method == 4:
            numMaxima = countMaxima(tpfs[i][frame].flux.reshape((self.cutout_size, self.cutout_size)))
            numPixels = np.count_nonzero(~np.isnan(tpfs[i][frame].flux))
            return np.nanpercentile(dataDistribution, 100 - numMaxima / numPixels * 100)
        else:
            return 150

    def circle_aperture(self, data, bkg):
        radius_in_pixels = degs_to_pixels(self.radius)
        data_mask = np.zeros_like(data)
        x_len = np.shape(data_mask)[1]
        y_len = np.shape(data_mask)[2]
        # centers
        cen_x = x_len//2
        cen_y = y_len//2
        bkg_mask = np.zeros_like(bkg)
        bkg_cutoff = self.get_upper_limit(bkg)
        for i in range(x_len):
            for j in range(y_len):
                if (i-cen_x)**2 + (j-cen_y)**2 < (radius_in_pixels)**2:   # star mask condition
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
        return star_mask, sky_mask

    def scattered_light(self, use_tpfs, full_model_Normalized):
        if self.debug:
            return False
        # regular grid covering the domain of the data
        X, Y = np.meshgrid(np.arange(0, self.cutout_size, 1), np.arange(0, self.cutout_size, 1))
        XX = X.flatten()
        YY = Y.flatten()

        # Define the steps for which we test for scattered light
        time_steps = np.arange(0, len(use_tpfs), self.scattered_light_frequency)
        coefficients_array = np.zeros((len(time_steps), 3))
        data_flux_values = (use_tpfs - full_model_Normalized).flux.value

        for i in range(len(time_steps)):
            data = data_flux_values[time_steps[i]]
            # best-fit linear plane
            A = np.c_[XX, YY, np.ones(XX.shape)]
            C, _, _, _ = scipy.linalg.lstsq(A, data.flatten())    # coefficients
            coefficients_array[i] = C

            # Deleting defined items we don't need any more to save memory
            del A
            del C
            del data

        X_cos = coefficients_array[:, 0]
        Y_cos = coefficients_array[:, 1]
        Z_cos = coefficients_array[:, 2]

        mxc = max(abs(X_cos))
        myc = max(abs(Y_cos))
        mzc = max(abs(Z_cos))

        #Deleting defined items we don't need any more to save memory
        del X_cos
        del Y_cos
        del Z_cos
        del coefficients_array
        gc.collect() #This is a command which will delete stray arguments to save memory

        print(mzc, mxc, myc)

        return (mzc > 2.5) | ((mxc > 0.02) & (myc > 0.02))

    def get_lcs(self):
        """Get lightcurves for each of the observations of the cluster

        Returns
        -------
        good_obs : `int`
            Number of good observations
        sectors_available : `int`
            How many sectors of data are available
        which_sectors_good : :class:`~numpy.ndarray`
            Which sectors are good
        failed_download : `int`
            How many observations failed to download
        near_edge_or_Sector_1 : `int`
            # TODO
        scattered_light : `int`
            # TODO
        lc_Lens : :class:`~numpy.ndarray`
            The length of each lightcurve
        """
        # Knowing how many observations we have to work with
        search = lk.search_tesscut(self.callable)
        self.sectors_available = len(search)

        # We are also going to document how many observations failed each one of our quality tests
        self.n_failed_download = 0
        self.n_near_edge = 0
        self.n_scattered_light = 0
        self.n_good_obs = 0
        self.which_sectors_good = []
        self.lc_lens = []

        # start iterating through the observations
        for current_try_sector in range(self.sectors_available):
            print(f"Starting Quality Tests for Observation: {current_try_sector}")

            # First is the Download Test
            self.downloadable(current_try_sector)
            if (self.tpfs is None) & (current_try_sector + 1 < self.sectors_available):
                print('Failed Download')
                self.n_failed_download += 1
                continue
            elif (self.tpfs is None) & (current_try_sector + 1 == self.sectors_available):
                print('Failed Download')
                self.n_failed_download += 1
                return

            # Now Edge Test
            near_edge = self.near_edge()
            if near_edge & (current_try_sector + 1 < self.sectors_available):
                print('Failed Near Edge Test')
                self.n_near_edge += 1
                continue
            if near_edge & (current_try_sector + 1 == self.sectors_available):
                print('Failed Near Edge Test')
                self.n_near_edge += 1
                return
            else:
                use_tpfs = self.tpfs[np.where(self.tpfs.to_lightcurve().quality == 0)]

            # Getting Rid of where the flux err < 0
            use_tpfs = use_tpfs[use_tpfs.to_lightcurve().flux_err > 0]

            print(use_tpfs.shape)

            # Define the aperture for our Cluster based on previous Vijith functions
            star_mask1 = np.empty([len(use_tpfs), self.cutout_size, self.cutout_size], dtype='bool')
            sky_mask1 = np.empty([len(use_tpfs), self.cutout_size, self.cutout_size], dtype='bool')

            star_mask1[0], sky_mask1[0] = self.circle_aperture(use_tpfs[0].flux.value, use_tpfs[0].flux.value)

            keep_mask = star_mask1[0]

            # Now we will begin to correct the lightcurve
            uncorrected_lc = use_tpfs.to_lightcurve(aperture_mask=keep_mask)

            # Time average of the pixels in the TPF:
            max_frame = use_tpfs.flux.value.max(axis=0)

            # This renormalizes any columns which are bright because of straps on the detector
            max_frame -= np.median(max_frame, axis=0)

            # This aperture is any "faint" pixels:
            bkg_aper = max_frame < np.percentile(max_frame, self.percentile)

            # The average light curve of the faint pixels is a good estimate of the scattered light
            scattered_light = use_tpfs.flux.value[:, bkg_aper].mean(axis=1)

            # We can use our background aperture to create pixel time series and then take Principal
            # Components of the data using Singular Value Decomposition. This gives us the "top" trends
            # that are present in the background data. I have picked 6 based on previous studies showing that
            # is an arbitrarily optimal number of components
            pca_dm1 = lk.DesignMatrix(use_tpfs.flux.value[:, bkg_aper], name='PCA').pca(6)

            # Here we are going to set the priors for the PCA to be located around the
            # flux values of the uncorrected LC
            pca_dm1.prior_mu = np.array([np.median(uncorrected_lc.flux.value) for _ in range(6)])
            pca_dm1.prior_sigma = np.array([(np.percentile(uncorrected_lc.flux.value, 84)
                                             - np.percentile(uncorrected_lc.flux.value, 16))
                                            for _ in range(6)])

            #The TESS mission pipeline provides co-trending basis vectors (CBVs) which capture common trends
            # in the dataset. We can use these to de-trend out pixel level data. The mission provides
            # MultiScale CBVs, which are at different time scales. In this case, we don't want to use the long
            # scale CBVs, because this may fit out real astrophysical variability. Instead we will use the
            # medium and short time scale CBVs.
            # TODO: this function seems to be deprecated
            cbvs_1 = lk.correctors.cbvcorrector.download_tess_cbvs(sector=use_tpfs.sector,
                                                                   camera=use_tpfs.camera,
                                                                   ccd=use_tpfs.ccd,
                                                                   cbv_type='MultiScale',
                                                                   band=2).interpolate(use_tpfs.to_lightcurve())
            cbvs_2 = lk.correctors.cbvcorrector.download_tess_cbvs(sector=use_tpfs.sector,
                                                                   camera=use_tpfs.camera,
                                                                   ccd=use_tpfs.ccd,
                                                                   cbv_type='MultiScale',
                                                                   band=3).interpolate(use_tpfs.to_lightcurve())

            cbv_dm1 = cbvs_1.to_designmatrix(cbv_indices=np.arange(1, 8))
            cbv_dm2 = cbvs_2.to_designmatrix(cbv_indices=np.arange(1, 8))

            # This combines the different timescale CBVs into a single `designmatrix` object
            cbv_dm_use = lk.DesignMatrixCollection([cbv_dm1, cbv_dm2]).to_designmatrix()

            # We can make a simple basis-spline (b-spline) model for astrophysical variability. This will be a
            # flexible, smooth model. The number of knots is important, we want to only correct for very long
            # term variabilty that looks like systematics, so here we have 5 knots, the smaller the better
            spline_dm1 = lk.designmatrix.create_spline_matrix(use_tpfs.time.value, n_knots=5)

            # Here we create our design matrix
            dm1 = lk.DesignMatrixCollection([pca_dm1, cbv_dm_use, spline_dm1])

            full_model, systematics_model, full_model_Normalized = np.ones((3, *use_tpfs.shape))
            for idx in tqdm(range(use_tpfs.shape[1])):
                for jdx in range(use_tpfs.shape[2]):
                    pixel_lightcurve = lk.LightCurve(time=use_tpfs.time.value,
                                                     flux=use_tpfs.flux.value[:, idx, jdx],
                                                     flux_err=use_tpfs.flux_err.value[:, idx, jdx])

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

            # Calculate Lightcurves
            # NOTE - we are also calculating a lightcurve which does not include the spline model,
            # this is the systematics_model_corrected_lightcurve1
            scattered_light_model_correected_lightcurve=(use_tpfs - scattered_light[:, None, None]).to_lightcurve(aperture_mask=keep_mask)
            systematics_model_corrected_lightcurve=(use_tpfs - systematics_model).to_lightcurve(aperture_mask=keep_mask)
            full_corrected_lightcurve = (use_tpfs - full_model).to_lightcurve(aperture_mask=keep_mask)

            full_corrected_lightcurve_table = Table([full_corrected_lightcurve.time.value,
                                                     full_corrected_lightcurve.flux.value,
                                                     full_corrected_lightcurve.flux_err.value],
                                                    names=('time', 'flux', 'flux_err'))

            scattered_light_test = self.scattered_light(use_tpfs, full_model_Normalized)
            # TODO: Conditions might be off here?
            if scattered_light_test:
                print("Failed Scattered Light Test")
                self.n_scattered_light += 1
                continue
            if scattered_light_test & (current_try_sector + 1 < self.sectors_available):
                print("Failed Scattered Light Test")
                self.n_scattered_light += 1
                return
            else:
                print(current_try_sector, "Passed Quality Tests")
                self.n_good_obs += 1
                self.which_sectors_good.append(current_try_sector)
                # This Else Statement means that the Lightcurve is good and has passed our quality checks

                # Writing out the data, so I never have to Download and Correct again, but only if there is data
                full_corrected_lightcurve_table.add_column(Column(flux_to_mag(full_corrected_lightcurve_table['flux'])), name='mag')
                full_corrected_lightcurve_table.add_column(Column(flux_err_to_mag_err(full_corrected_lightcurve_table['flux'], full_corrected_lightcurve_table['flux_err'])), name='mag_err')

                full_corrected_lightcurve_table.write(os.path.join(self.output_path,
                                                                   "Corrected_LCs",
                                                                   self.callable + ".fits"),
                                                      format='fits', append=True)

                # Now I am going to save a plot of the light curve to go visually inspect later
                range_ = max(full_corrected_lightcurve_table['flux']) - min(full_corrected_lightcurve_table['flux'])
                fig = plt.figure()
                plt.title(f'Observation: {current_try_sector}')
                plt.plot(full_corrected_lightcurve_table['time'], full_corrected_lightcurve_table['flux'],
                         color='k', linewidth=.5)
                plt.xlabel('Delta Time [Days]')
                plt.ylabel('Flux [e/s]')
                plt.text(full_corrected_lightcurve_table['time'][0],
                         (max(full_corrected_lightcurve_table['flux'])-(range_*0.05)),
                         self.callable, fontsize=14)
                plt.subplots_adjust(right=1.4, top=1)

                path = os.path.join(self.output_path, "Figures", "LCs",
                                    f'{self.callable}_Full_Corrected_LC_Observation_{current_try_sector}.png')
                plt.savefig(path, format='png')
                plt.close(fig)

                self.lc_lens.append(len(full_corrected_lightcurve_table))

    def generate_lightcurves(self):
        """Generate lightcurve files for the cluster and save them in `self.output_path`

        Returns
        -------
        output_table : :class:`~astropy.table.Table`
            The full lightcurves output table that was saved
        """
        LC_PATH = os.path.join(self.output_path, 'Corrected_LCs/',
                               str(self.callable) + 'output_table.fits')
        if self.has_tess_data():
            # Test to see if I have already downloaded and corrected this cluster, If I have, read in the data
            if self.previously_downloaded():
                output_table = Table.read(LC_PATH, hdu=1)
                return output_table

            else:
                # This else statement refers to the Cluster Not Previously Being Downloaded
                # So Calling function to download and correct data
                self.get_lcs()

                # Making the Output Table
                name___ = [self.cluster_name]
                HTD = [True]
                OB_use = [self.n_good_obs]
                OB_av = [self.sectors_available]
                OB_good = [str(self.which_sectors_good)]
                OB_fd = [self.n_failed_download]
                OB_ne = [self.n_near_edge]
                OB_sl = [self.n_scattered_light]
                OB_lens = [str((self.lc_lens))]

                output_table = Table([name___, [self.location], [self.radius], [self.cluster_age], HTD, OB_av,
                                      OB_use, OB_good, OB_fd, OB_ne, OB_sl, OB_lens],
                                     names=('Name', 'Location', 'Radius [deg]', 'Log Age', 'Has_TESS_Data',
                                            'Obs_Available', 'Num_Good_Obs', 'Which_Obs_Good',
                                            'Obs_DL_Failed', 'Obs_Near_Edge_S1', 'Obs_Scattered_Light',
                                            'Light_Curve_Lengths'))

                # Writing out the data, so I never have to Download and Correct again
                output_table.write(LC_PATH)

                if self.n_good_obs != 0:
                    # now I'm going to read in the lightcurves and attach them to the output table to have all data in one place
                    for i in range(output_table['Num_Good_Obs'][0]):
                        light_curve_table = Table.read(LC_PATH.replace("output_table", ""), hdu=i + 1)
                        light_curve_table.write(LC_PATH, append=True)

                    return output_table

        else:
            # This Means that there is no TESS coverage for the Cluster (Easiest to check)
            name___= [self.cluster_name]       
            HTD=[False]  
            OB_use= np.ma.array([0], mask=[1])
            OB_good= np.ma.array([0], mask=[1])
            OB_av= np.ma.array([0], mask=[1])
            OB_fd= np.ma.array([0], mask=[1])
            OB_ne= np.ma.array([0], mask=[1])
            OB_sl= np.ma.array([0], mask=[1])  
            OB_lens= np.ma.array([0], mask=[1])

            output_table = Table([name___, [self.location], [self.radius], [self.cluster_age], HTD, OB_av,
                                OB_use, [str(OB_good)], OB_fd, OB_ne, OB_sl, [str(OB_lens)]],
                                names=('Name', 'Location', 'Radius [deg]', 'Log Age', 'Has_TESS_Data',
                                        'Obs_Available', 'Num_Good_Obs', 'Which_Obs_Good', 'Obs_DL_Failed',
                                        'Obs_Near_Edge_S1', 'Obs_Scattered_Light', 'Light_Curve_Lengths'))
            output_table.write(LC_PATH, overwrite=True)
            return output_table

    def access_lightcurve(self, sector):
        path = os.path.join(self.output_path, 'Corrected_LCs', self.callable + 'output_table.fits')
        if not os.path.exists(path):
            print("The Lightcurve has not been downloaded and corrected. Please run 'Generate_Lightcurves()' function for this cluster.")
            return None, None
        output_table = Table.read(path, hdu=1)

        # Get the Light Curve
        if output_table['Num_Good_Obs'] == 1:
            light_curve_table = Table.read(path, hdu=2)
        else:
            light_curve_table = Table.read(path, hdu=(int(sector)+2))

        # Now I am going to save a plot of the light curve to go visually inspect later
        range_ = max(light_curve_table['flux']) - min(light_curve_table['flux'])
        fig = plt.figure()
        plt.title(f'Observation: {sector}')
        plt.plot(light_curve_table['time'], light_curve_table['flux'], color='k', linewidth=.5)
        plt.xlabel('Delta Time [Days]')
        plt.ylabel('Flux [e/s]')
        plt.text(light_curve_table['time'][0], (max(light_curve_table['flux'])-(range_*0.05)),
                 self.cluster_name, fontsize=14)
        plt.subplots_adjust(right=1.4, top=1)

        plt.show()
        plt.close(fig)

        return fig, light_curve_table


def degs_to_pixels(degs):
    # convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)
    return degs*60*60/21


def pixels_to_degs(pixels):
    # convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)
    return pixels*21/(60*60)


def flux_to_mag(flux):
    m1 = 10
    f1 = 15000
    mag = 2.5 * np.log10(f1 / flux) + m1
    return mag


def flux_err_to_mag_err(flux, flux_err):
    d_mag_d_flux = -2.5 / (flux * np.log(10))
    m_err_squared = abs(d_mag_d_flux)**2 * flux_err**2
    return np.sqrt(m_err_squared)
