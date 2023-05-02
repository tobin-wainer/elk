import numpy as np
import scipy
from statsmodels.tsa.stattools import acf as calc_acf

from scipy.signal import find_peaks, peak_widths, argrelextrema
import scipy.linalg

from astropy.table import Table
from astropy.timeseries import LombScargle

from .plot import plot_acf

__all__ = ["get_MAD", "get_sigmaG", "get_range", "get_skewness", "von_neumann_ratio", "J_stetson",
           "periodogram", "longest_contiguous_chunk", "autocorr"]


def get_MAD(flux):
    """Get the median absolute deviation of an array of fluxes

    Parameters
    ----------
    flux : :class:`~numpy.ndarray`
        Array of fluxes

    Returns
    -------
    mad : `float`
        Median absolute deviation
    """
    return np.median(np.abs(flux - np.median(flux)))

def get_sigmaG(flux):
    """Compute a rank-based estimate of the standard deviation

    Follows https://www.astroml.org/modules/generated/astroML.stats.sigmaG.html

    Parameters
    ----------
    flux : :class:`~numpy.ndarray`
        Array of fluxes

    Returns
    -------
    sigmaG : `float`
        sigmaG (a robust estimate of the standard deviation sigma)
    """
    # This factor is ~ 1 / (2 sqrt(2) erf^-1(0.5))
    sigmaG_factor = 0.74130110925280102
    percs = np.percentile(flux, [25, 75])
    return sigmaG_factor * (percs[1] - percs[0])

def get_range(flux, bottom_percentile, upper_percentile):
    # TODO: Ask Tobin but it seems like this will only work for an array of fluxes of length 100?
    # Generalized to get the range for whatever percentile the heart desires
    flux_l = list(flux)
    flux_l.sort()

    bottom_ind = int(len(flux_l)*bottom_percentile)
    upper_ind = int(len(flux_l)*upper_percentile)

    new = flux_l[bottom_ind:upper_ind]
    r = new[-1] - new[0]
    return r


def get_skewness(flux):
    """Calculate the skewness of an array of fluxes

    Parameters
    ----------
    flux : :class:`~numpy.ndarray`
        Array of fluxes

    Returns
    -------
    skewness : `float`
        Skewness
    """
    return scipy.stats.moment(flux, moment=3)


def von_neumann_ratio(flux):
    """Calculate the (inverted) Von Neumann Ratio for an array of fluxes

    See `von Neumann 1941 <https://www.jstor.org/stable/2235951>`_ and
    `Sokolovsky+2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.464..274S/abstract>`_ Eq. 19

    Parameters
    ----------
    flux : :class:`~numpy.ndarray`
        Array of fluxes

    Returns
    -------
    ratio : `float`
        Von Neumann Ratio
    """
    eta = np.sum(np.ediff1d(flux)**2) / np.var(flux)
    return 1 / eta


def J_stetson(time, mag, mag_err, pair_threshold=0.021):
    """Calculate the J Stetson statistic as a measure of variability

    See `Stetson+1996 <https://ui.adsabs.harvard.edu/abs/1996PASP..108..851S/abstract>`_ and Eq. 12 of
    `Sokolovsky+2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.464..274S/abstract>`_

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time of observations
    mag : :class:`~numpy.ndarray`
        Magnitudes of each observation
    mag_err : :class:`~numpy.ndarray`
        Errors on the magnitudes
    pair_threshold : `float`, optional
        Threshold for whether to consider observations close enough, by default 0.021 days (=~30 min)

    Returns
    -------
    J : `float`
        J Stetson Statistic
    """
    # get the difference in time between observations
    delta_times = np.concatenate((np.ediff1d(time), [1]))

    # calculate the delta for each observation
    n = len(time)
    delta = (np.sqrt(n / (n - 1)) * ((mag - np.mean(mag)) / mag_err))

    # start the P and weights
    P = np.empty(len(delta_times))
    w = np.ones(len(delta_times))

    # mask for which observations are part of a pair (0.021 day=~30min)
    pairs = delta_times < pair_threshold
    inds = np.argwhere(pairs).flatten()

    # calculate P and w
    P[inds] = delta[inds] * delta[inds]
    P[~pairs] = delta[~pairs]**2 - 1
    w[~pairs] = 0.25

    # combine into the J stetson stat
    J = np.sum(w * np.sign(P) * np.sqrt(np.abs(P))) / np.sum(w)
    return J


def periodogram(time, flux, flux_err, frequencies, n_bootstrap=100, max_peaks=25, freq_thresh=5):
    """Generate a bootstrapped Lomb-Scargle periodogram and various associated statistics

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Times of observations
    flux : :class:`~numpy.ndarray`
        Flux for each observations
    flux_err : :class:`~numpy.ndarray`
        Errors on the fluxes
    frequencies : :class:`~numpy.ndarray`
        Frequencies at which to evaluate the periodogram
    n_bootstrap : `int`, optional
        How many bootstraps to perform, by default 100
    max_peaks : `int`, optional
        Fixed number of max peaks to return, by default 25. If None is given then the number of peaks that
        exist will be returned rather than truncated/padded to this fixed number.
    freq_thresh : `int`, optional
        Threshold for high vs. low frequency, by default 5 days since this represents a threshold of how
        different stars vary
        `McQuillan+2012 <https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..11M/abstract>`_.

    Returns
    -------
    med : :class:`~numpy.ndarray`
        Median power at each frequency
    percentiles : :class:`~numpy.ndarray`
        1 and 2-sigma power at each frequency (in order of 2-, 1-, 1+, 2+) with shape `(4, len(frequencies))`
    lsp_stats : `dict`
        Dictionary of various statistics related to the periodogram. Keys are: ["max_power",
        "freq_at_max_power", "peak_freqs", "power_at_peaks", "n_peaks", "ratio_of_power_at_high_v_low_freq"].
        Many of these are self explanatory but we note that `peak_freqs` and `power_at_peaks` will always have
        a length of `max_peaks` may need to be masked to just take the first `n_peaks`. (This feature is
        useful when saving to a FITS file with a required fixed column width).
    """
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

    _, _, peak_left, peak_right = peak_widths(one_below, peak_inds, rel_height=0.5)
    peak_left = frequencies[np.floor(peak_left).astype(int)]
    peak_right = frequencies[np.ceil(peak_right).astype(int)]

    # calculate how much of our power is coming from timescales shorter/longer than threshold
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

    ls = LombScargle(time, flux, flux_err)
    ls.power(frequencies)
    fap = ls.false_alarm_probability(power=max_power)

    lsp_stats = {
        "max_power": max_power,
        "freq_at_max_power": freq_at_max_power,
        "peak_freqs": peak_freqs,
        "peak_left_edge": peak_left,
        "peak_right_edge": peak_right,
        "power_at_peaks": power_at_peaks,
        "n_peaks": n_peaks,
        "ratio_of_power_at_high_v_low_freq": ratio_of_power_at_high_v_low_freq,
        "FAP": fap
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


def autocorr(time, flux, largest_gap_allowed=0.25, plot=False, **plot_kwargs):
    """Calculate the autocorrelation function for an array of fluxes

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Times of observations
    flux : :class:`~numpy.ndarray`
        Flux for each observations
    largest_gap_allowed : `float`, optional
        Largest gap in the data which is allowed in a chunk (in days), by default 0.25
    plot : `bool`, optional
        Whether to show a plot of the autocorrelation, by default False
    plot_kwargs : `various`
        Arguments to pass to `elk.plot.plot_acf`

    Returns
    -------
    ac_time : :class:`~numpy.ndarray`
        Times evaluated for the autocorrelation function
    acf : :class:`~numpy.ndarray`
        The autocorrelation function
    confint : :class:`~numpy.ndarray`
        1-sigma confidence interval around the autocorrelation function with shape (len(ac_time), 2)
    acf_stats : `dict`
        Some simple stats for the autocorrelation function
    fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
        Figure and axis on which the correlation function has been plotted
    """
    chunk_mask = longest_contiguous_chunk(time, largest_gap_allowed=largest_gap_allowed)

    ac_time, ac_flux = time[chunk_mask], flux[chunk_mask]
    acf, confint = calc_acf(ac_flux, nlags=len(ac_flux) - 1, alpha=0.3173)
    plot_times = ac_time - ac_time[0]

    # find max autocorrelation (cut first value since it is meaningless in this context)
    max_ac = max(acf[plot_times > plot_times[argrelextrema(acf, np.less)[0][0]]])
    acf_stats = {
        "max_autocorrelation": max_ac,
        "time_of_max_autocorrelation": plot_times[acf == max_ac],
        "rms": np.sqrt(np.mean(np.square(acf)))
    }

    if plot:
        fig, ax = plot_acf(plot_times, acf, confint, **plot_kwargs)
        return ac_time, acf, confint, acf_stats, fig, ax
    else:
        return ac_time, acf, confint, acf_stats
