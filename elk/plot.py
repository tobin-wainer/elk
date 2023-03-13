import matplotlib.pyplot as plt
import numpy as np


def plot_periodogram(frequencies, power, power_percentiles, peak_freqs,
                     fig=None, ax=None, show=True, title=None, save_path=None):
    """Plot a periodogram

    Parameters
    ----------
    frequencies : :class:`~numpy.ndarray`
        Frequencies at which periodogram evaluated
    power : :class:`~numpy.ndarray`
        Power at each frequency
    power_percentiles : :class:`~numpy.ndarray`
        Percentiles on the power, if shape is (2, len(power)) then assumed to be 1-sigma values, if shape is
        (4, len(power)) then assumed to be 1- and 2-sigma values in order of (2-, 1-, 1+, 2+)
    peak_freqs : :class:`~numpy.ndarray`
        Frequencies at which peaks occur
    fig : :class:`~matplotlib.pyplot.Figure`, optional
        Figure on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    ax : :class:`~matplotlib.pyplot.AxesSubplot`, optional
        Axis on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    show : bool, optional
        Whether to show the plot, by default True
    title : `str`, optional
        A title for the plot, by default None
    save_path : `str`, optional
        Where to save the plot, if None then plot is not saved, by default None

    Returns
    -------
    fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
        Figure and axis on which the periodogram has been plotted
    """
    # create figure if necessary and add title
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)

    # shade any percentiles as desired
    if power_percentiles is not None:
        if len(power_percentiles) == 2:
            one_below, one_above = power_percentiles
            ax.fill_between(frequencies, one_below, one_above, color="grey", alpha=0.3)
        else:
            two_below, one_below, one_above, two_above = power_percentiles
            ax.fill_between(frequencies, two_below, two_above, color="grey", alpha=0.15)
            ax.fill_between(frequencies, one_below, one_above, color="grey", alpha=0.3)

    # plot the actual power
    ax.plot(frequencies, power, color="black", linewidth=1.25)

    # add vertical lines at each of the peaks
    ax.vlines(x=peak_freqs, ymin=np.zeros(len(peak_freqs)),
              ymax=power[np.in1d(frequencies, peak_freqs)], linestyle='--', color="tab:red")

    # set scales and labels
    ax.set_xscale('log')
    ax.set_xlabel(r'Frequency $[1 / {\rm days}]$')
    ax.set_ylabel('Power')

    # save and show if desired
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_acf(time, acf=None, acf_percentiles=None, title=None, fig=None, ax=None, show=False, save_path=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)

    ax.plot(time, acf)
    ax.fill_between(time, acf_percentiles[:, 0], acf_percentiles[:, 1], color='grey', alpha=.5)
    ax.set_xlabel(r'Time $[\rm days]$')
    ax.set_ylabel("Autocorrelation")

    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
