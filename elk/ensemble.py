import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import scipy
from astropy.table import Table
from astropy.io import fits
import astropy.units as u

import os.path
import gc

from .lightcurve import BasicLightcurve, TESSCutLightcurve

__all__ = ["EnsembleLC", "from_fits"]


class EnsembleLC:
    def __init__(self, radius, cluster_age, output_path="./", cluster_name=None, location=None,
                 percentile=80, cutout_size=99, scattered_light_frequency=5, n_pca=6, verbose=False,
                 just_one_lc=False, no_lk_cache=False, ignore_previous_downloads=False, debug=False):
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
        verbose : `bool`, optional
            Whether to print out information and progress bars, by default False
        just_one_lc : `bool`, optional
            Whether to return after the first lightcurve that passes the quality tests, by default False
        no_lk_cache : `bool`, optional
            Whether to skip using the LightKurve cache and scrub downloads instead (can be useful for runs
            on a computing cluster with limited memory space), by default False
        ignore_previous_downloads : `bool`, optional
            Whether to ignore previously downloaded and corrected lightcurves            
        debug : `bool`, optional
            #TODO DELETE THIS, by default False
        """

        # make sure that some sort of identifier has been provided
        assert cluster_name is not None or location is not None,\
            "Must provide at least one of `cluster_name` and `location`"
        self.callable = cluster_name if cluster_name is not None else location

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
        if output_path is not None and not os.path.exists(output_path):
            print(f"WARNING: There is no output folder at the path that you supplied ({output_path})")
            create_it = input(("  Would you like me to create it for you? "
                               "(If not then no files will be saved) [Y/n]"))
            if create_it == "" or create_it.lower() == "y":
                # create the folder
                os.mkdir(output_path)
            else:
                output_path = None

        # if we wan't to avoid the lk cache we shall need our own dummy
        if no_lk_cache and not os.path.exists(os.path.join(output_path, 'cache')):
            os.mkdir(os.path.join(output_path, 'cache'))
            if not os.path.exists(os.path.join(output_path, 'cache', self.callable)):
                os.mkdir(os.path.join(output_path, 'cache', self.callable))
                os.mkdir(os.path.join(output_path, 'cache', self.callable, 'tesscut'))

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

        self.lcs = []
        self.output_path = output_path
        self.radius = radius
        self.cluster_age = cluster_age
        self.cluster_name = cluster_name
        self.location = location
        self.percentile = percentile
        self.cutout_size = cutout_size
        self.scattered_light_frequency = scattered_light_frequency
        self.n_pca = n_pca
        self.verbose = verbose
        self.just_one_lc = just_one_lc
        self.no_lk_cache = no_lk_cache
        self.debug = debug

        if self.output_path is not None and self.previously_downloaded() and not ignore_previous_downloads:
            if self.verbose:
                print(("Found previously corrected data for this target, loading it! "
                       "(Set `ignore_previous_downloads=True` to ignore data)"))
            self = from_fits(os.path.join(self.output_path, "Corrected_LCs",
                                          self.callable + "output_table.fits"), existing_class=self)

        # We are also going to document how many observations failed each one of our quality tests
        self.n_failed_download = 0
        self.n_near_edge = 0
        self.n_scattered_light = 0
        self.n_good_obs = 0

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
        if self.verbose:
            print(f'{self.callable} has {self.sectors_available} observations')
        return self.sectors_available > 0

    def downloadable(self, ind):
        # use a Try statement to see if we can download the cluster data
        try:
            download_dir = os.path.join(self.output_path,
                                        'cache', self.callable) if self.no_lk_cache else None
            tpfs = self.tess_search_results[ind].download(cutout_size=(self.cutout_size, self.cutout_size),
                                                          download_dir=download_dir)
        except lk.search.SearchError:
            tpfs = None
        return tpfs

    def clear_cache(self):
        """Clear the folder containing manually cached lightkurve files"""
        for file in os.listdir(os.path.join(self.output_path, 'cache', self.callable, 'tesscut')):
            if file.endswith(".fits"):
                os.remove(os.path.join(self.output_path, 'cache', self.callable, 'tesscut', file))

    def scattered_light(self, quality_tpfs, full_model_Normalized):
        if self.debug:
            return False
        # regular grid covering the domain of the data
        X, Y = np.meshgrid(np.arange(0, self.cutout_size, 1), np.arange(0, self.cutout_size, 1))
        XX = X.flatten()
        YY = Y.flatten()

        # Define the steps for which we test for scattered light
        time_steps = np.arange(0, len(quality_tpfs), self.scattered_light_frequency)
        coefficients_array = np.zeros((len(time_steps), 3))
        data_flux_values = (quality_tpfs - full_model_Normalized).flux.value

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
            The number of sectors where the target is located too close to the edge of a TESS detector, 
            where we cannot accurately perform uniform background subtraction; or the custer was observed in
            Sector 1, which has known systematics
        scattered_light : `int`
            The number of sectors with significant scattered light after the correction process. 'Significant'
            is arbitrarily defined by a by-eye calibration, and the threshold values can be changed if needed
        lc_Lens : :class:`~numpy.ndarray`
            The length of each lightcurve
        """
        self.lcs = [None for _ in range(self.sectors_available)]

        # start iterating through the observations
        for sector_ind in range(self.sectors_available):
            if self.verbose:
                print(f"Starting Quality Tests for Observation: {sector_ind}")

            # if we are avoiding caching then delete every fits file in the cache folder
            if self.no_lk_cache:
                self.clear_cache()

            # First is the Download Test
            tpfs = self.downloadable(sector_ind)
            if tpfs is None:
                if self.verbose:
                    print('  Failed Download')
                self.n_failed_download += 1
                continue

            # check whether this lightcurve has already been corrected
            lc_path = os.path.join(self.output_path, "Corrected_LCs",
                                   self.callable + f"_lc_{tpfs.sector}.fits")
            if os.path.exists(lc_path):
                if self.verbose:
                    print("  Found a pre-corrected lightcurve for this sector, loading it!")
                # if yes then load the lightcurve in, add to good obs and move onto next sector
                self.lcs[sector_ind] = BasicLightcurve(fits_path=lc_path, hdu_index=1)
                self.n_good_obs += 1
                continue

            lc = TESSCutLightcurve(tpfs=tpfs, radius=self.radius, cutout_size=self.cutout_size,
                                   percentile=self.percentile, n_pca=self.n_pca, progress_bar=self.verbose)

            # Now Edge Test
            near_edge = lc.near_edge()
            if near_edge:
                if self.verbose:
                    print('  Failed Near Edge Test')
                self.n_near_edge += 1
                continue

            lc.correct_lc()

            scattered_light_test = self.scattered_light(lc.quality_tpfs, lc.full_model_normalized)
            if scattered_light_test:
                if self.verbose:
                    print("  Failed Scattered Light Test")
                self.n_scattered_light += 1
                continue
            else:
                # This Else Statement means that the Lightcurve is good and has passed our quality checks
                if self.verbose:
                    print("  Passed Quality Tests")
                self.n_good_obs += 1
                self.lcs[sector_ind] = lc

                # save the lightcurve for later in case of crashes
                if self.output_path is not None:
                    empty_primary = fits.PrimaryHDU()
                    hdul = fits.HDUList([empty_primary, lc.hdu])
                    hdul.writeto(os.path.join(self.output_path, "Corrected_LCs",
                                              self.callable + f"_lc_{lc.sector}.fits"), overwrite=True)

                # save a plot of the light curve to visually inspect later
                if self.output_path is not None and self.save["figures"]:
                    _, ax = lc.plot(show=False)
                    ax.annotate(self.callable, xy=(0.98, 0.98), xycoords="axes fraction",
                                ha="right", va="top", fontsize="large")
                    path = os.path.join(self.output_path, "Figures", "LCs",
                                        f'{self.callable}_Full_Corrected_LC_Observation_{sector_ind}.png')
                    plt.savefig(path, format='png', bbox_inches="tight")

                if self.just_one_lc:
                    if self.verbose:
                        print(("Found a lightcurve that passed quality tests - exiting since "
                               "`self.just_one_lc=True`"))
                    break

        if self.no_lk_cache:
            self.clear_cache()

    def lightcurves_summary_file(self):
        """Generate lightcurve output files for the cluster and save them in `self.output_path`

        Returns
        -------
        output_table : :class:`~astropy.table.Table`
            The full lightcurves output table that was saved
        """
        # if data is available and the lightcurves have not yet been calculated
        if self.has_tess_data() and self.lcs == []:
            # download and correct lightcurves
            self.get_lcs()

            # clear out the cache after we're done making lightcurves
            if self.no_lk_cache:
                self.clear_cache()

        # write out the full file
        hdr = fits.Header()
        hdr['name'] = self.cluster_name
        hdr['location'] = self.location
        hdr["radius"] = (self.radius, "Radius in degrees")
        hdr['log_age'] = (self.cluster_age, "Log Age in dex")
        hdr["has_data"] = (self.sectors_available > 0, "Whether there was TESS data available")
        hdr["n_obs"] = (self.sectors_available, "How many sectors of observations exist")
        hdr["n_good"] = (self.n_good_obs, "Number of good observations")
        hdr["n_dlfail"] = (self.n_failed_download, "Number of failed downloads")
        hdr["n_edge"] = (self.n_near_edge, "Number of obs near edge")
        hdr["n_scatt"] = (self.n_scattered_light, "Number of obs with scattered light")
        empty_primary = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([empty_primary] + [lc.hdu for lc in self.lcs if lc is not None])
        if self.output_path is not None:
            hdul.writeto(os.path.join(self.output_path, 'Corrected_LCs/',
                         str(self.callable) + 'output_table.fits'), overwrite=True)

        # return a simple summary table that can be stacked with other clusters
        return Table({'name': [self.cluster_name], 'location': [self.location], 'radius': [self.radius],
                      'log_age': [self.cluster_age], 'has_data': [self.sectors_available > 0],
                      'n_obs': [self.sectors_available], 'n_good_obs': [self.n_good_obs],
                      'n_failed_download': [self.n_failed_download], 'n_near_edge': [self.n_near_edge],
                      'n_scatter_light': [self.n_scattered_light],
                      'lc_lens': [[len(lc.corrected_lc) for lc in self.lcs if lc is not None]],
                      'which_sectors_good': [[lc.sector for lc in self.lcs if lc is not None]]})


def from_fits(filepath, existing_class=None, **kwargs):
    # if an existing class is not provided then create a new blank one
    if existing_class is None:
        new_ecl = EnsembleLC(cluster_name="", radius=None, cluster_age=None, output_path=None, **kwargs)
    else:
        new_ecl = existing_class

    # open up the fits file and load in the information
    with fits.open(filepath) as hdul:
        details = hdul[0]
        new_ecl.cluster_name = details.header["name"]
        new_ecl.location = details.header["location"]
        new_ecl.callable = new_ecl.cluster_name if new_ecl.cluster_name is not None else new_ecl.location
        new_ecl.radius = details.header["radius"]
        new_ecl.cluster_age = details.header["log_age"]
        new_ecl.sectors_available = details.header["n_obs"]
        new_ecl.n_good_obs = details.header["n_good"]
        new_ecl.n_failed_download = details.header["n_dlfail"]
        new_ecl.n_near_edge = details.header["n_edge"]
        new_ecl.n_scattered_light = details.header["n_scatt"]

        new_ecl.lcs = [BasicLightcurve(fits_path=filepath, hdu_index=hdu_ind)
                       for hdu_ind in range(1, len(hdul))]

    return new_ecl
