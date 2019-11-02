#!/usr/bin/env python
# coding: utf-8


from google.cloud import bigquery
from math import log10, floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import matplotlib.lines as mlines
import os
import matplotlib as mpl
from PIL import Image
from io import BytesIO
mpl.rc('figure', dpi=400, figsize=(10, 10))
mpl.rc('savefig', dpi=400)


def round_sig(x, sig=1):
    """Round a value to N sig fig.

    Parameters
    ----------
    x : float
        Value to round
    sig : int, optional
        Number of sig figs, default is 1

    Returns
    -------
    float
        Rounded value

    """
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


project_id = 'hypothermia-bayescmd'
# Explicitly use service account credentials by specifying the private
# key file. All clients in google-cloud-python have this helper.
client = bigquery.Client.from_service_account_json(
    "../gcloud/hypothermia-auth.json"
)


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


priors = {
    "phi": [
        "uniform",
        [
            0.0288,
            0.0432
        ]
    ],
    "r_n": [
        "uniform",
        [
            0.01496,
            0.02244
        ]
    ],
    "r_0": [
        "uniform",
        [
            0.01008,
            0.015119999999999998
        ]
    ],
    "k_aut": [
        "uniform",
        [
            0.0,
            1.0
        ]
    ],
    "Q_10_haemo": [
        "uniform",
        [
            0.1,
            6.0
        ]
    ],
    "n_m": [
        "uniform",
        [
            1.464,
            2.1959999999999997
        ]
    ],
    "r_m": [
        "uniform",
        [
            0.0216,
            0.0324
        ]
    ],
    "CBFn": [
        "uniform",
            [
                0.0064,
                0.0096
            ]
    ],
    "Q_10_met": [
        "uniform",
        [
            0.1,
            6.0
        ]
    ],
    "CMRO2_n": [
        "uniform",
        [
            0.016,
            0.024
        ]
    ]
}


def plot_comparison_diag_medians(g, df, medians):

    for i, var in enumerate(g.x_vars):
        ax = g.axes[i][i]
        neo007_median = np.median(df[var][df['Neonate'] == 'neo007'])
        ax.axvline(neo007_median, color=sns.color_palette()[0], linewidth=2)

        neo021_median = np.median(df[var][df['Neonate'] == 'neo021'])
        ax.axvline(neo021_median, color=sns.color_palette()[1], linewidth=2)

        ax.text(
            0.05,
            1.45,
            "neo007: {:.3g}\nneo021: {:.3g}".format(neo007_median,
                                                    neo021_median),
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=8)

    return True


def medians_comparison_kde_plot(x, y, medians, **kws):
    """Plot bivariate KDE with median of distribution marked on,
    comparing between groups.

    Parameters
    ----------
    x : array-like
        Array-like of data to plot.
    y : array-like
        Array-like of data to plot.
    medians : :obj:`dict`
        Dictionary of parameter, median pairings.
    kws : key, value pairings.
        Other keyword arguments to pass to :obj:`sns.distplot`.

    Returns
    -------
    ax : :obj:`matplotlib.AxesSubplot`
        AxesSubplot object of univariate KDE and bar plot with median marked
        on as well as text.

    """
    ax = plt.gca()
    ax = sns.kdeplot(x, y, ax=ax, **kws)
    # color = infer_from_cmap(kws['cmap'])
    x_median = x.median()
    y_median = y.median()
    ax.plot(x_median, y_median, 'X', markerfacecolor=kws['color'],
            markeredgecolor='k', markeredgewidth=1.5, markersize=6)
    return ax


signals = ['CCO', 'HbT', 'Hbdiff']

for SIGNAL in [''] + signals:
    print("Working on {} ".format(SIGNAL if SIGNAL != '' else "TOTAL"))
    posterior_size = 4000
    if SIGNAL != '':
        distance = SIGNAL + "_NRMSE"
    else:
        distance = "NRMSE"

    neo007_query = """
    SELECT
    phi,r_n,r_0,k_aut,Q_10_haemo,n_m,r_m,CBFn,Q_10_met,CMRO2_n,
    NRMSE,
    "neo007" as Neonate
    FROM
    neo_desat.neo007_gradient
    ORDER BY
    {} ASC
    LIMIT
    {}
    """.format(distance, posterior_size)

    # In[ ]:

    neo021_query = """
    SELECT
    phi,r_n,r_0,k_aut,Q_10_haemo,n_m,r_m,CBFn,Q_10_met,CMRO2_n,
    NRMSE,
    "neo021" as Neonate
    FROM
    neo_desat.neo021_gradient
    ORDER BY {} ASC
    LIMIT {}
    """.format(distance, posterior_size)

    with Timer("Pulling Posterior from Big Query"):
        neo007 = client.query(neo007_query).to_dataframe()

        neo021 = client.query(neo021_query).to_dataframe()

        df = pd.concat([neo007, neo021])

    parameters = list(priors.keys())
    medians = {}
    with sns.plotting_context("paper", rc={"xtick.labelsize": 12,
                                           "ytick.labelsize": 12,
                                           "axes.labelsize": 8}):
        g = sns.PairGrid(df,
                         vars=parameters,
                         diag_sharey=False,
                         height=0.5,
                         hue='Neonate')

        with Timer("Plotting diagonals"):

            g.map_diag(sns.distplot, hist_kws=dict(alpha=0.5))
            plot_comparison_diag_medians(g, df, medians=medians)
        with Timer("Plotting lower triangle"):
            g.map_lower(medians_comparison_kde_plot, medians=medians)
        n_ticks = 4
        with Timer("Formatting figure"):
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)
            for ii, ax in enumerate(g.axes.flat):
                for label in ax.get_xticklabels():
                    label.set_rotation(75)
                ax.xaxis.labelpad = 5
                ax.yaxis.labelpad = 5
                ii_y = ii // len(parameters)
                ii_x = ii % len(parameters)
                ax.set_ylim(priors[parameters[ii_y]][1])
                ax.set_xlim(priors[parameters[ii_x]][1])
                xmax = priors[parameters[ii_x]][1][1]
                xmin = priors[parameters[ii_x]][1][0]
                xticks = np.arange(xmin, xmax,
                                   round_sig((xmax - xmin) / n_ticks, sig=1))
                ax.set_xticks(xticks)
                ax.set_xlabel(ax.get_xlabel(), labelpad=1,
                              rotation=30, fontsize=8)
                ax.set_ylabel(ax.get_ylabel(), labelpad=15,
                              rotation=45, fontsize=8)
            lines = []
            # lines.append(('True Value', mlines.Line2D([], [], color='black')))
            lines.append(('Posterior Median - neo007',
                          mlines.Line2D([], [], color=sns.color_palette()[0])))
            lines.append(('Posterior Median - neo021',
                          mlines.Line2D([], [], color=sns.color_palette()[1])))
            g.set(yticklabels=[])
            g.set(xticklabels=[])
            g.fig.legend(labels=[l[0] for l in lines],
                         handles=[l[1] for l in lines],
                         bbox_to_anchor=(0.995, 0.995), loc=1, prop={"size": 11})

            # g.fig.tight_layout()
            g.fig.subplots_adjust(wspace=0.15, hspace=0.25)

    figPath = "/home/buck06191/Dropbox/phd/desat_neonate/ABC/Figures/"
    g.savefig(figPath+'comparison_posteriors_neonate_desat_{}.png'.format(distance),
              dpi=250, bbox_inches='tight', transparent=True)


# %%
