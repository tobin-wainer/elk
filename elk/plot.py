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


def plot_acf(time, acf=None, flux=None, fig=None, ax=None):
    fig, ax = plt.subplots()
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