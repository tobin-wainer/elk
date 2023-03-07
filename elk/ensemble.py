import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import scipy
from astropy.table import Table, Column, join, vstack
import astropy.units as u

import os.path
import gc

from .lightcurve import TESScutLightcurve


class EnsembleLC:
    def __init__(self, radius, cluster_age, output_path="./", cluster_name=None, location=None,
                 percentile=80, cutout_size=99, scattered_light_frequency=5, n_pca=6,
                 debug=False):
        """Class for generating lightcurves from TESS cutouts

        Parameters
        ----------
        radius : `float`
            Radius of the cluster. If a `float` is given then unit is assumed to be degrees. Otherwise, I'll
            convert your unit to what I need.
        cluster_age : `float`
            Age of the cluster. If a `float` is given then unit is assumed to be dex. Otherwise, I'll
            convert your unit to what I need.
        output_path : `str`, optional
            Path to a folder in which to save outputs - must have subfolders Corrected_LCs/ and Figures/LCs/,
            by default "./"
        cluster_name : `str`, optional
            Name of the cluster, by default None
        location : `str`, optional
            Location of the cluster #TODO What format here?, by default None
        percentile : `int`, optional
            Which percentile to use in the upper limit calculation, by default 80
        cutout_size : `int`, optional
            How large to make the cutout, by default 99
        scattered_light_frequency : `int`, optional
            Frequency at which to check for scattered light, by default 5
        n_pca : `int`, optional
            Number of principle components to use in the DesignMatrix, by default 6
        debug : `bool`, optional
            #TODO DELETE THIS, by default False
        """

        # make sure that some sort of identifier has been provided
        assert cluster_name is not None or location is not None,\
            "Must provide at least one of `cluster_name` and `location`"

        # convert radius to degrees if it has units
        if hasattr(radius, 'unit'):
            radius = radius.to(u.deg).value

        # convert cluster age to dex if it has units
        if hasattr(cluster_age, 'unit'):
            if cluster_age.unit == u.dex:
                cluster_age = cluster_age.value
            else:
                cluster_age = np.log10(cluster_age.to(u.yr).value)

        # check main output folder
        if not os.path.exists(output_path):
            print(f"WARNING: There is no output folder at the path that you supplied ({output_path})")
            create_it = input(("  Would you like me to create it for you? "
                               "(If not then no files will be saved) [Y/n]"))
            if create_it == "" or create_it.lower() == "y":
                # create the folder
                os.mkdir(output_path)
            else:
                output_path = None

        # check subfolders
        self.save = {"lcs": False, "figures": False}
        if output_path is not None:
            for subpath, key in zip(["Corrected_LCs", os.path.join("Figures", "LCs")], ["lcs", "figures"]):
                path = os.path.join(output_path, subpath)
                if not os.path.exists(path):
                    print(f"WARNING: The necessary subfolder at ({path}) does not exist")
                    create_it = input(("  Would you like me to create it for you? "
                                       "(If not then these files will not be saved) [Y/n]"))
                    if create_it == "" or create_it.lower() == "y":
                        # create the folder
                        os.makedirs(path)
                        self.save[key] = True
                else:
                    self.save[key] = True

        self.output_path = output_path
        self.radius = radius
        self.cluster_age = cluster_age
        self.callable = cluster_name if cluster_name is not None else location
        self.cluster_name = cluster_name
        self.location = location
        self.percentile = percentile
        self.cutout_size = cutout_size
        self.scattered_light_frequency = scattered_light_frequency
        self.n_pca = n_pca
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
        self.tess_search_results = lk.search_tesscut(self.callable)
        self.sectors_available = len(self.tess_search_results)
        print(f'{self.callable} has {self.sectors_available} observations')
        return self.sectors_available > 0

    def downloadable(self, ind):
        # use a Try statement to see if we can download the cluster data
        try:
            tpfs = self.tess_search_results[ind].download(cutout_size=(self.cutout_size, self.cutout_size))
        except lk.search.SearchError:
            print("No Download")
            tpfs = None
        return tpfs

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
            tpfs = self.downloadable(current_try_sector)
            if (tpfs is None) & (current_try_sector + 1 < self.sectors_available):
                print('Failed Download')
                self.n_failed_download += 1
                continue
            elif (tpfs is None) & (current_try_sector + 1 == self.sectors_available):
                print('Failed Download')
                self.n_failed_download += 1
                return

            lc = TESScutLightcurve(tpfs=tpfs, radius=self.radius, cutout_size=self.cutout_size,
                                   percentile=self.percentile, n_pca=self.n_pca)

            # Now Edge Test
            near_edge = lc.near_edge()
            if near_edge & (current_try_sector + 1 < self.sectors_available):
                print('Failed Near Edge Test')
                self.n_near_edge += 1
                continue
            if near_edge & (current_try_sector + 1 == self.sectors_available):
                print('Failed Near Edge Test')
                self.n_near_edge += 1
                return

            print(lc.use_tpfs.shape)

            lc.correct_lc()

            scattered_light_test = self.scattered_light(lc.use_tpfs, lc.full_model_normalized)
            if scattered_light_test & (current_try_sector + 1 < self.sectors_available):
                print("Failed Scattered Light Test")
                self.n_scattered_light += 1
                continue
            if scattered_light_test & (current_try_sector + 1 == self.sectors_available):
                print("Failed Scattered Light Test")
                self.n_scattered_light += 1
                return
            else:
                print(current_try_sector, "Passed Quality Tests")
                self.n_good_obs += 1
                self.which_sectors_good.append(current_try_sector)
                # This Else Statement means that the Lightcurve is good and has passed our quality checks

                # Writing out the data, so I never have to Download and Correct again, but only if there is data
                lc.full_corrected_lightcurve_table.add_column(Column(flux_to_mag(lc.full_corrected_lightcurve_table['flux'])), name='mag')
                lc.full_corrected_lightcurve_table.add_column(Column(flux_err_to_mag_err(lc.full_corrected_lightcurve_table['flux'], lc.full_corrected_lightcurve_table['flux_err'])), name='mag_err')

                if self.output_path is not None and self.save["lcs"]:
                    lc.full_corrected_lightcurve_table.write(os.path.join(self.output_path,
                                                                          "Corrected_LCs",
                                                                          self.callable + ".fits"),
                                                             format='fits', append=True)

                # Now I am going to save a plot of the light curve to go visually inspect later
                range_ = max(lc.full_corrected_lightcurve_table['flux']) - min(lc.full_corrected_lightcurve_table['flux'])
                fig = plt.figure()
                plt.title(f'Observation: {current_try_sector}')
                plt.plot(lc.full_corrected_lightcurve_table['time'], lc.full_corrected_lightcurve_table['flux'],
                         color='k', linewidth=.5)
                plt.xlabel('Delta Time [Days]')
                plt.ylabel('Flux [e/s]')
                plt.text(lc.full_corrected_lightcurve_table['time'][0],
                         (max(lc.full_corrected_lightcurve_table['flux'])-(range_*0.05)),
                         self.callable, fontsize=14)

                if self.output_path is not None and self.save["figures"]:
                    path = os.path.join(self.output_path, "Figures", "LCs",
                                        f'{self.callable}_Full_Corrected_LC_Observation_{current_try_sector}.png')
                    plt.savefig(path, format='png')
                plt.close(fig)

                self.lc_lens.append(len(lc.full_corrected_lightcurve_table))

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
                if self.output_path is not None and self.save["lcs"]:
                    output_table.write(LC_PATH)

                if self.n_good_obs != 0 and self.output_path is not None and self.save["lcs"]:
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

            if self.output_path is not None and self.save["lcs"]:
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

        plt.show()
        plt.close(fig)

        return fig, light_curve_table


def flux_to_mag(flux):
    m1 = 10
    f1 = 15000
    mag = 2.5 * np.log10(f1 / flux) + m1
    return mag


def flux_err_to_mag_err(flux, flux_err):
    d_mag_d_flux = -2.5 / (flux * np.log(10))
    m_err_squared = abs(d_mag_d_flux)**2 * flux_err**2
    return np.sqrt(m_err_squared)
