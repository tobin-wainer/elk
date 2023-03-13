import matplotlib.pyplot as plt
import numpy as np


def plot_periodogram(frequencies, power, power_percentiles, peak_freqs,
                     fig=None, ax=None, show=True, title=None, save_path=None):
    fig, ax = plt.subplots()
    ax.set_title(title)

    if power_percentiles is not None:
        if len(power_percentiles) == 2:
            one_below, one_above = power_percentiles
            ax.fill_between(frequencies, one_below, one_above, color="grey", alpha=0.3)
        else:
            two_below, one_below, one_above, two_above = power_percentiles
            ax.fill_between(frequencies, two_below, two_above, color="grey", alpha=0.15)
            ax.fill_between(frequencies, one_below, one_above, color="grey", alpha=0.3)

    ax.plot(frequencies, power, color="black", linewidth=1.25)

    ax.vlines(x=peak_freqs, ymin=np.zeros(len(peak_freqs)),
              ymax=power[np.in1d(frequencies, peak_freqs)], linestyle='--', color="tab:red")
    ax.set_xscale('log')
    ax.set_xlabel(r'Frequency $[1 / {\rm days}]$')
    ax.set_ylabel('Power')

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
