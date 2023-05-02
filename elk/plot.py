import matplotlib.pyplot as plt
import numpy as np


__all__ = ["plot_periodogram", "plot_acf", "plot_lightcurve"]


def plot_periodogram(frequencies, power, power_percentiles, peak_freqs, fap=None,
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
    fap : `float`, optional
        False alarm probability for the maximum power peak - plotted as horizontal line if provided, by default None
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

    if fap is not None:
        ax.axhline(fap, color="gold", label=f"False alarm probability ({fap:1.2e})")

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


def plot_acf(time, acf, acf_percentiles=None, title=None, fig=None, ax=None, show=True, save_path=None):
    """Plot an autocorrelation function

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Times at which autocorrelation function is evaluated
    acf : :class:`~numpy.ndarray`, optional
        The autocorrelation function, by default None
    acf_percentiles : :class:`~numpy.ndarray`, optional
        1-sigma confidence interval on the autocorrelation function, by default None
    title : `str`, optional
        A title for the plot, by default None
    fig : :class:`~matplotlib.pyplot.Figure`, optional
        Figure on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    ax : :class:`~matplotlib.pyplot.AxesSubplot`, optional
        Axis on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    show : bool, optional
        Whether to show the plot, by default True
    save_path : `str`, optional
        Where to save the plot, if None then plot is not saved, by default None

    Returns
    -------
    fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
        Figure and axis on which the periodogram has been plotted
    """
    # create figure if necessary and add title
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)

    # shade percentiles if desired
    if acf_percentiles is not None:
        ax.fill_between(time, acf_percentiles[:, 0], acf_percentiles[:, 1], color='grey', alpha=.5)

    # plot autocorrelation function and label axes
    ax.plot(time, acf)
    ax.set_xlabel(r'Time $[\rm days]$')
    ax.set_ylabel("Autocorrelation")

    # save and show as desired
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_lightcurve(time, flux, fold_period=None, title=None, fig=None, ax=None, show=True, save_path=None,
                    color='k', linewidth=0.5, **kwargs):
    """Plot a lightcurve

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Times of observations
    flux : :class:`~numpy.ndarray`
        Flux of each observation
    fold_period : `float`, optional
        Period on which to fold the light curve, if None then no folding is done, by default None
    title : `str`, optional
        A title for the plot, by default None
    fig : :class:`~matplotlib.pyplot.Figure`, optional
        Figure on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    ax : :class:`~matplotlib.pyplot.AxesSubplot`, optional
        Axis on which to plot, if either `fig` or `ax` is None then new ones are created, by default None
    show : bool, optional
        Whether to show the plot, by default True
    save_path : `str`, optional
        Where to save the plot, if None then plot is not saved, by default None
    color : `str`, optional
        Colour to use for the lightcurve line, by default 'k'
    linewidth : `float`, optional
        Linewidth for the lightcurve, by default 0.5
    **kwargs
        Keyword arguments to pass to the plotting of the lightcurve line

    Returns
    -------
    fig, ax : :class:`~matplotlib.pyplot.Figure`, :class:`~matplotlib.pyplot.AxesSubplot`
        Figure and axis on which the lightcurve has been plotted
    """
    # create figure if necessary and add title
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)

    # if a period is given then fold based on that period and re-sort
    if fold_period is not None:
        time = (time - time[0]) % fold_period
        order = np.argsort(time)
        time = time[order]
        flux = flux[order]

    ax.plot(time, flux, color=color, linewidth=linewidth, **kwargs)

    ax.set_xlabel(r'Time $\rm [days]$')
    ax.set_ylabel(r'Flux $[e / {\rm s}]$')

    # save and show as desired
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
