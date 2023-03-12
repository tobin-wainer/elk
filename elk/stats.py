import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels 
import os

from scipy.signal import find_peaks, argrelextrema
import scipy.linalg

from astropy.table import Table, Column
from astropy.timeseries import LombScargle


Path_to_Save_to = '/Users/Tobin/Dropbox/TESS_project/Variability_Statistics/Test_Pipeline_Module/Variability_Metrics/'
Path_to_Read_in_LCs = '/Users/Tobin/Dropbox/TESS_project/Variability_Statistics/Test_Pipeline_Module/Corrected_LCs/'


def Have_Lightcurve(Cluster_name):
    if os.path.exists(Path_to_Read_in_LCs+str(Cluster_name)+'.fits'): 
        return True
    else:
        return False


def Read_in_Lightcurve(Cluster_name):
    return Table.read(Path_to_Read_in_LCs+str(Cluster_name)+'.fits')


def get_rms(flux):
    return np.sqrt(np.mean(np.square(flux)))


def get_std(flux):
    # Standard Deviation
    return np.std(flux)


def get_MAD(flux):
    return np.median(np.abs(flux - np.median(flux)))


def get_range(flux, bottom_percentile, upper_percentile):
    # Generalized to get the range for whatever percentile the heart desires
    flux_l = list(flux)
    flux_l.sort()

    bottom_ind = int(len(flux_l)*bottom_percentile)
    upper_ind = int(len(flux_l)*upper_percentile)

    new = flux_l[bottom_ind:upper_ind]
    r = new[-1] - new[0]
    return r


def skewness(flux):
    return scipy.stats.moment(flux, moment=3)


def von_neumann_ratio(flux):
    # https://www.jstor.org/stable/2235951
    nu = np.sum(np.ediff1d(flux)**2) / np.var(flux)
    return 1 / nu


def J_stetson(time, mag, mag_err):
    # get the difference in time between observations
    delta_times = np.concatenate((np.ediff1d(time), [1]))

    # calculate the delta for each observation
    n = len(time)
    delta = (np.sqrt(n / (n - 1)) * ((mag - np.mean(mag)) / mag_err))

    # start the P and weights
    P = np.empty(len(delta_times))
    w = np.ones(len(delta_times))

    # mask for which observations are part of a pair (0.021 day=~30min)
    # TODO: This should probably be an input and use astropy.units
    pairs = delta_times < 0.021
    inds = np.argwhere(pairs).flatten()

    # calculate P and w
    P[inds] = delta[inds] * delta[inds]
    P[~pairs] = delta[~pairs]**2 - 1
    w[~pairs] = 0.25

    # combine into the J stetson stat
    J = np.sum(w * np.sign(P) * np.sqrt(np.abs(P))) / np.sum(w)
    return J


def periodogram(time, flux, flux_err, frequencies, n_bootstrap=1000, max_peaks=25, freq_thresh=5):
    flux_samples = np.transpose([np.repeat(flux[i], n_bootstrap)
                                 + np.random.normal(loc=0,
                                                    scale=flux_err[i],
                                                    size=n_bootstrap) for i in range(len(flux))])
    powers = np.asarray([LombScargle(time, flux_samples[i], dy=flux_err).power(frequencies)
                         for i in range(len(flux_samples))])

    percentiles = np.percentile(powers, [50, 2.27, 15.89, 84.1, 97.725], axis=0)
    med, _, one_below, _, _ = percentiles

    peak_threshold = np.mean(med) + 2 * np.std(med)
    peak_inds, peak_info = find_peaks(one_below, height=peak_threshold)

    # 5 Days is around the middle of our time scale, and represents a threshold of how different stars vary (McQuillan et al. 2012)
    # So we want to calculate how much of our power is coming from time scales shorter/longer than 5 days
    low_freq = frequencies > freq_thresh
    ratio_of_power_at_high_v_low_freq = np.mean(med[~low_freq]) / np.mean(med[low_freq])

    # if no limit on the peaks then save all of them, otherwise truncate to ensure identical list length
    n_peaks = len(peak_inds)
    max_peaks = n_peaks if max_peaks is None else max_peaks

    peak_freqs, power_at_peaks = np.zeros(max_peaks), np.zeros(max_peaks)

    peak_freqs[:min(n_peaks, max_peaks)] = frequencies[peak_inds][:max_peaks]
    power_at_peaks[:min(n_peaks, max_peaks)] = peak_info['peak_heights'][:max_peaks]

    max_power = max(med)
    freq_at_max_power = frequencies[med == max_power][0]

    lsp_stats = {
        "max_power": max_power,
        "freq_at_max_power": freq_at_max_power,
        "peak_freqs": peak_freqs,
        "power_at_peaks": power_at_peaks,
        "n_peaks": n_peaks,
        "ratio_of_power_at_high_v_low_freq": ratio_of_power_at_high_v_low_freq
    }

    return med, percentiles[1:], lsp_stats


def longest_contiguous_chunk(times, largest_gap_allowed=0.25):
    """Create a mask for the largest contiguous chunk of observations

    Parameters
    ----------
    times : :class:`~numpy.ndarray`
        Array of times at which observations were taken
    largest_gap_allowed : `float`, optional
        Largest gap in the data which is allowed in a chunk (in days), by default 0.25

    Returns
    -------
    chunk_mask : :class:`~numpy.ndarray`
        Boolean mask on observations in the largest chunk
    """
    # find the difference in time between observations
    delta_times = np.ediff1d(times)

    # identify inds at which the change is larger than the maximum allowed and append start/end
    gap_inds = np.argwhere(delta_times > largest_gap_allowed).flatten()
    gap_inds = np.concatenate(([-1], gap_inds, [len(times) - 1]))

    # find the largest length of time between gaps
    max_ind = np.argmax(np.ediff1d(gap_inds))

    # create a mask for that longest chunk
    all_inds = np.arange(len(times))
    chunk_mask = (all_inds > gap_inds[max_ind]) & (all_inds <= gap_inds[max_ind + 1])
    return chunk_mask


def autocorr(light_curve, Cluster_name, print_figs, save_figs):
    # Because the gap in the middle of the observation can drastically effect the ACF,
    # we only calculate the first half
    fh_cor_lc = light_curve[:(len(light_curve)//2)]

    acf = statsmodels.tsa.stattools.acf(fh_cor_lc['flux'], nlags=len(fh_cor_lc)-1, alpha=.33)

    err = acf[1]
    ac = acf[0]
    acf_p16 = [err[i][0] for i in range(len(err))]
    acf_p84 = [err[i][1] for i in range(len(err))]

    deltatimes = []
    for i in range(len(fh_cor_lc)-1):
        deltatimes.append(fh_cor_lc['time'][i+1]-fh_cor_lc['time'][i])

    plot_times = np.cumsum(deltatimes)

    fig = plt.figure()
    plt.title("ACF"+str(Cluster_name))
    plt.plot(plot_times, ac[:-1])
    plt.fill_between(plot_times, acf_p16[:-1], acf_p84[:-1], color='grey',
                     label=r'$1 \sigma$ Confidence', alpha=.5)
    plt.xlabel('Delta Time [Days]')
    plt.text(1, .9, str(Cluster_name), fontsize=16)
    path = "Figures/"
    which_fig = "_ACF_firsthalf"
    out = ".png"

    if save_figs:
        plt.savefig(Path_to_Save_to + path + str(Cluster_name) + which_fig + out, format='png')
    if print_figs:
        plt.show()

    plt.close(fig)

    first_min_ts = plot_times[argrelextrema(ac, np.less)[0][0]]
    ac_ai = ac[np.where(plot_times > first_min_ts)]
    max_ac = max(ac_ai)
    pa_mac = plot_times[np.where(ac == max_ac)]
    ac_rms = get_rms(ac)

    return fig, (round(max_ac, 5), round(pa_mac[0], 4), round(ac_rms, 5))


def Get_Variable_Stats_Table(Cluster_name, print_figs=True, save_figs=True):
    # TODO: This should all get moved into the LightCurve Class
    # Test to see if I have already downloaded and corrected this cluster, If I have, read in the data
    if Have_Lightcurve(Cluster_name):
        data = Table.read(Path_to_Read_in_LCs + str(Cluster_name) + '.fits')

        normalized_flux = np.array(data['flux'])/np.median(data['flux'])

        rms = [np.log10(get_rms(normalized_flux))]
        std = [np.log10(get_std(normalized_flux))]
        MAD = [np.log10(get_MAD(normalized_flux))]
        range_5_95 = [np.log10(get_range(normalized_flux, .05, .95))]
        range_1_99 = [np.log10(get_range(normalized_flux, .01, .99))]

        j_stat = [get_J_Stetson_Stat(data)]
        sness = [np.log10(abs(skewness(data['flux'])))]
        vnr = [Von_Neumann_Ratio(normalized_flux)]

        max_ac = []
        period_amac = []
        ac_rms = []
        max_LS_power = []
        freq_amlp = []
        LS_peak_freq = []
        LS_peak_power = []
        LS_n_peaks = []
        LS_ratio = []

        ac_fig, ac_arr = autocorr(data, Cluster_name, print_figs, save_figs)
        LS_fig, LS_arr = get_LSP(data, Cluster_name, print_figs, save_figs)

        max_ac.append(ac_arr[0])
        period_amac.append(ac_arr[1])
        ac_rms.append(ac_arr[2])
        max_LS_power.append(LS_arr[0])
        freq_amlp.append(LS_arr[1])
        LS_peak_freq.append(LS_arr[2])
        LS_peak_power.append(LS_arr[3])
        LS_n_peaks.append(LS_arr[4])
        LS_ratio.append(LS_arr[5])

        name___ = [Cluster_name]


        Simple_Stats_Table = Table((name___, rms, std, range_5_95, range_1_99, MAD, j_stat, vnr, sness,
                                    max_ac, period_amac, ac_rms, max_LS_power, freq_amlp, LS_peak_freq, LS_peak_power,
                                    LS_n_peaks, LS_ratio), 
                                    names=('Cluster', 'RMS', 'STD', '5-95_Range', '1-99_Range',
                                           'MAD', 'Stetson_J_stat', 'Von_Neumann_Ratio', 'Skewness', 'Max_AutoCorr', 'AutoCorr_Period_atM', 'AutoCorr_rms',
                                           'Max_LS_Power', 'LS_Freq_atM', 'LS_Peak_Freq','LS_Peak_Power', 'N_LS_Peaks',
                                           'LS_Ratio'))

        # Writing out the table
        Simple_Stats_Table.write(Path_to_Save_to +'Stats_Tables/' + str(Cluster_name) + 'stats_table.fits', overwrite=True)

        return Simple_Stats_Table

    else:
        # This statement means that the lightcuve does not exist in our path
        name___ = [Cluster_name]

        std = np.ma.array([0], mask=[1])
        rms = np.ma.array([0], mask=[1])
        range_5_95 = np.ma.array([0], mask=[1])
        range_1_99 = np.ma.array([0], mask=[1])
        MAD = np.ma.array([0], mask=[1])
        j_stat = np.ma.array([0], mask=[1])
        vnr = np.ma.array([0], mask=[1])
        sness = np.ma.array([0], mask=[1])
        max_ac = np.ma.array([0], mask=[1])
        period_amac = np.ma.array([0], mask=[1])
        ac_rms = np.ma.array([0], mask=[1])
        max_LS_power = np.ma.array([0], mask=[1])
        freq_amlp = np.ma.array([0], mask=[1])
        LS_peak_freq = [np.zeros(25)]
        LS_peak_power = [np.zeros(25)]
        LS_n_peaks = np.ma.array([0], mask=[1])
        LS_ratio = np.ma.array([0], mask=[1])

        Simple_Stats_Table = Table((name___, rms, std, range_5_95, range_1_99, MAD, j_stat, vnr, sness,
                                    max_ac, period_amac, ac_rms, max_LS_power, freq_amlp, LS_peak_freq, LS_peak_power, 
                                    LS_n_peaks, LS_ratio), 
                                    names=('Cluster', 'RMS', 'STD', '5-95_Range', '1-99_Range', 
                                            'MAD', 'Stetson_J_stat', 'Von_Neumann_Ratio', 'Skewness', 'Max_AutoCorr', 'AutoCorr_Period_atM','AutoCorr_rms', 
                                            'Max_LS_Power', 'LS_Freq_atM', 'LS_Peak_Freq','LS_Peak_Power', 'N_LS_Peaks', 
                                            'LS_Ratio')) 

        # Writing out the table
        Simple_Stats_Table.write(Path_to_Save_to+'Stats_Tables/'+str(Cluster_name)+'stats_table.fits', overwrite=True) 

        return Simple_Stats_Table
