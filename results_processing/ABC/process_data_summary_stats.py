#!/usr/bin/env python
# coding: utf-8

from bayescmd.results_handling import get_output
from google.cloud import bigquery
from bayescmd.abc import priors_creator
import pickle
from pprint import pprint
from copy import copy
from distutils import dir_util
from pathlib import Path
import statsmodels.api as sm
import scipy.stats as stats
from bayescmd.abc import inputParse
from bayescmd.abc import import_actual_data
from bayescmd.abc import SummaryStats
from bayescmd.results_handling import get_output
from io import BytesIO
from PIL import Image
import matplotlib as mpl
import os
from math import ceil
import json
import random
import matplotlib.lines as mlines
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')


# Google BigQuery
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')
client = bigquery.Client.from_service_account_json(
    "../gcloud/hypothermia-auth.json"
)


mpl.rc('figure', dpi=300, figsize=(7.5, 8))

mpl.rcParams["xtick.labelsize"] = 8

mpl.rcParams["ytick.labelsize"] = 8

mpl.rcParams["axes.labelsize"] = 10

mpl.rcParams["figure.titlesize"] = 12


def TIFF_exporter(fig, fname, fig_dir='.', extra_artists=()):
    """
    Parameters
    ----------
    fig: matplotlib figure
    """

    # save figure
    # (1) save the image in memory in PNG format
    # png1 = BytesIO()
    fig.savefig(os.path.join(fig_dir, '{}.png'.format(fname)), format='png', bbox_inches='tight', bbox_extra_artists=extra_artists,
                dpi=300, transparent=False)

#     # (2) load this image into PIL
#     png2 = Image.open(png1)

#     # (3) save as TIFF
#     png2.save(os.path.join(fig_dir, '{}.tiff'.format(fname)),
#               compression='tiff_deflate')
#     png1.close()
    return True


# Explicitly use service account credentials by specifying the private
# key file. All clients in google-cloud-python have this helper.


def generate_histogram_query(project, neonate, n_bins, distance):
    histogram_query = """
    SELECT
      MIN(data.{distance}) AS min,
      MAX(data.{distance}) AS max,
      COUNT(data.{distance}) AS num,
      INTEGER((data.{distance}-value.min)/(value.max-value.min)*{n_bins}) AS group_
    FROM
      [{project}:neo_desat.{neonate}] data
    CROSS JOIN (
      SELECT
        MAX({distance}) AS max,
        MIN({distance}) AS min
      FROM
        [{project}:neo_desat.{neonate}_gradient]) value
    GROUP BY
      group_
    ORDER BY
      group_
    """.format(neonate=neonate, n_bins=n_bins, distance=distance, project=project)
    return histogram_query


# In[4]:
def generate_posterior_query(project, neonate, distance, parameters, limit=50000):
    unpacked_params = ",\n".join(parameters)
    histogram_query = """
    SELECT
        {unpacked_params},
        {distance},
        idx
    FROM
    `{project}.neo_desat.{neonate}_gradient`
    ORDER BY
    {distance} ASC
    LIMIT
    {limit}
    """.format(project=project, neonate=neonate, unpacked_params=unpacked_params, distance=distance, limit=limit)
    return histogram_query


def generate_posterior_query(project, neonate, distance, parameters, limit=50000):
    unpacked_params = ",\n".join(parameters)
    histogram_query = """
SELECT
    {unpacked_params},
    {distance},
    idx
FROM
  `{project}.neo_desat.{neonate}_gradient`
ORDER BY
  {distance} ASC
LIMIT
  {limit}
    """.format(project=project, neonate=neonate, unpacked_params=unpacked_params, distance=distance, limit=limit)
    return histogram_query


def load_configuration(neonate, verbose=False):
    current_file = Path(os.path.abspath(''))
    config_file = os.path.join(current_file.parents[1],
                               'config_files',
                               'abc',
                               'neo_config.json'
                               )

    with open(config_file, 'r') as conf_f:
        conf = json.load(conf_f)

    params = conf['priors']

    input_path = os.path.join(current_file.parents[1],
                              'data',
                              'formatted_data',
                              '{}_formatted.csv'.format(neonate))

    d0 = import_actual_data(input_path)

    targets = conf['targets']
    model_name = conf['model_name']
    inputs = conf['inputs']

    config = {
        "model_name": model_name,
        "targets": targets,
        "times": d0['t'],
        "inputs": inputs,
        "parameters": params,
        "input_path": input_path,
        "zero_flag": conf['zero_flag'],
    }

    if verbose:
        pprint(config)

    return config, d0


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def run_model(model):
    """Run a BCMD Model.

    Parameters
    ----------
    model : :obj:`bayescmd.bcmdModel.ModelBCMD`
        An initialised instance of a ModelBCMD class.

    Returns
    -------
    output : :obj:`dict`
        Dictionary of parsed model output.

    """

    model.create_initialised_input()

    model.run_from_buffer()

    output = model.output_parse()
    return output


def get_runs(posterior, conf, n_repeats=50):
    rand_selection = random.sample(range(posterior.shape[0]), n_repeats)
    outputs_list = []
    p_names = list(conf['parameters'].keys())
    posteriors = posterior[p_names].values
    d0 = import_actual_data(conf['input_path'])
    input_data = inputParse(d0, conf['inputs'])
    while len(outputs_list) < n_repeats:
        idx = rand_selection.pop()
        print("\tSample {}, idx:{}".format(len(outputs_list), idx))
        p = dict(zip(p_names, posteriors[idx]))

        _, output = get_output(
            conf['model_name'],
            p,
            conf['times'],
            input_data,
            d0,
            conf['targets'],
            distance="NRMSE",
            zero_flag=conf['zero_flag'])
        outputs_list.append(output)
    return outputs_list


def get_repeated_outputs(df,
                         model_name,
                         parameters,
                         input_path,
                         inputs,
                         targets,
                         n_repeats,
                         zero_flag,
                         neonate,
                         tolerance=None,
                         limit=None,
                         frac=None,
                         openopt_path=None,
                         offset=None,
                         distance='euclidean'
                         ):
    """Generate model output and distances multiple times.

    Parameters
    ----------
    model_name : :obj:`str`
        Names of model. Should match the modeldef file for model being generated
        i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.
    parameters : :obj:`dict` of :obj:`str`: :obj:`tuple`
        Dict of model parameters to compare, with value tuple of the prior max
        and min.
    input_path : :obj:`str`
        Path to the true data file
    inputs : :obj:`list` of :obj:`str`
        List of model inputs.
    targets : :obj:`list` of :obj:`str`
        List of model outputs against which the model is being optimised.
    n_repeats : :obj: `int`
        Number of times to generate output data
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
    zero_flag : dict
        Dictionary of form target(:obj:`str`): bool, where bool indicates
        whether to zero that target.

        Note: zero_flag keys should match targets list.
    openopt_path : :obj:`str` or :obj:`None`
        Path to the openopt data file if it exists. Default is None.
    offset : :obj:`dict`
        Dictionary of offset parameters if they are needed
    distance : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.

    Returns
    -------
    fig : :obj:`matplotlib.figure`
        Figure containing all axes.

    """
    p_names = list(parameters.keys())
    sorted_df = df.sort_values(by=distance)

    if tolerance:
        accepted_limit = sum(df[distance].values < tolerance)
    elif limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    df_list = []
    if n_repeats > accepted_limit:
        print(
            "Setting number of repeats to quarter of the posterior size\n",
            file=sys.stderr)
        n_repeats = int(accepted_limit / 4)
    d0 = import_actual_data(input_path)
    input_data = inputParse(d0, inputs)

    true_data = pd.read_csv(input_path)
    times = true_data['t'].values

    if openopt_path:
        openopt_data = pd.read_csv(openopt_path)

    if n_repeats > accepted_limit:
        raise ValueError(
            "Number of requested model runs greater than posterior size:"
            "\n\tPosterior Size: {}\n\tNumber of runs: {}".format(
                accepted_limit, n_repeats))

    rand_selection = list(range(accepted_limit))
    random.shuffle(rand_selection)

    outputs_list = []

    posteriors = sorted_df.iloc[:accepted_limit][p_names].values
    select_idx = 0
    with Timer("Running repeat outputs"):
        for i in range(n_repeats):
            try:
                idx = rand_selection.pop()
                p = dict(zip(p_names, posteriors[idx]))
                if offset:
                    p = {**p, **offset}
                output = get_output(
                    model_name,
                    p,
                    times,
                    input_data,
                    d0,
                    targets,
                    distance=distance,
                    zero_flag=zero_flag)
                outputs_list.append(output)
                print("Sample {}, idx:{}".format(len(outputs_list), idx))

            except (TimeoutError, TimeoutExpired) as e:
                print("Timed out for Sample {}, idx:{}".format(
                    len(outputs_list), idx))
                pprint.pprint(p)
                rand_selection.insert(0, idx)
            except (CalledProcessError) as e:
                print("CalledProcessError for Sample {}, idx:{}".format(
                    len(outputs_list), idx))
                pprint.pprint(p)
                rand_selection.insert(0, idx)

        print("Final number of runs is: {}".format(len(outputs_list)))

    d = {"Errors": {}, "Outputs": {}}
    d['Errors']['Average'] = np.nanmean(
        [o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.nanmean(
            [o[0][target] for o in outputs_list])
        d['Outputs'][target] = [o[1][target] for o in outputs_list]

    for ii, target in enumerate(targets):
        x = [j for j in times for n in range(len(d['Outputs'][target]))]
        with Timer('Transposing {}'.format(target)):
            y = np.array(d['Outputs'][target]).transpose()
            y = y.ravel()
        with Timer("Crafting DataFrame for {}".format(target)):
            model_name_col = [neonate]*len(x)
            target_col = [target]*len(x)
            df1 = pd.DataFrame(
                {"Time": x, "Posterior": y, "Neonate": model_name_col, "Output": target_col})
        with Timer("Appending dataframe for {}".format(target)):
            df_list.append(df1.copy())
            del df1
    return pd.concat(df_list), true_data


def reduce_dataframe(df):
    before = copy(df.memory_usage(deep=True, index=True).sum())
    print("BEFORE\t", before)
    df['Signal'] = df['Signal'].astype('category')
    df['Batch'] = df['Batch'].apply(pd.to_numeric, downcast='unsigned')
    df['Time (sec)'] = df['Time (sec)'].apply(
        pd.to_numeric, downcast='unsigned')
    if 'Residuals' in df.columns:
        df['Residuals'] = df['Residuals'].apply(
            pd.to_numeric, downcast='float')
    if 'Data' in df.columns:
        df['Data'] = df['Data'].apply(pd.to_numeric, downcast='float')
    after = copy(df.memory_usage(deep=True, index=True).sum())
    print("AFTER\t", after)
    print("SAVING:\t {}%".format(100-(after/before*100)))
    return df


labels = {"t": "Time (sec)",
          "SaO2sup": "SaO2 (%)",
          "P_a": "ABP (mmHg)",
          "PaCO2": "PaCO$_2$ (mmHg)",
          "temp": "Temperature ($^{\circ}$C)",
          "TOI": "TOI (%)",
          "HbT": "$\Delta$HbT $(\mu M)$",
          "Hbdiff": "$\Delta$HbD $(\mu M)$",
          "CCO": "$\Delta$CCO $(\mu M)$"
          }
LIM = 10000

signals = ['CCO', 'HbT', 'Hbdiff']


configuration = {}

neonates = ['neo007', 'neo021']

signals = ['CCO', 'HbT', 'Hbdiff']

for SIGNAL in [''] + signals:
    print("Working on {} ".format(SIGNAL if SIGNAL != '' else "TOTAL"))
    posterior_size = 4000
    if SIGNAL != '':
        distance = SIGNAL + "_NRMSE"
    else:
        distance = "NRMSE"
    for NEONATE in neonates:

        print("Working on {} ".format(NEONATE))
        # Set config and create figure path

        configuration[NEONATE] = {}

        config, d0 = load_configuration(NEONATE)
        configuration[NEONATE]['bayescmd_config'] = config
        configuration[NEONATE]['original_data'] = d0

        configuration[NEONATE]['histogram_query'] = generate_histogram_query('hypothermia-bayescmd',
                                                                             NEONATE,
                                                                             100,
                                                                             distance)

        configuration[NEONATE]['posterior_query'] = generate_posterior_query('hypothermia-bayescmd',
                                                                             NEONATE,
                                                                             distance,
                                                                             list(
                                                                                 configuration[NEONATE]['bayescmd_config']['parameters'].keys()),
                                                                             limit=posterior_size)

        figPath = "/home/buck06191/Dropbox/phd/desat_neonate/ABC/Figures/Combined_gradient/{}/{}".format(
            NEONATE, 'NRMSE')
        dir_util.mkpath(figPath)

        # Get posterior
        print("\tRunning SQL query")
        df_post = client.query(
            configuration[NEONATE]['posterior_query']).to_dataframe()
        N = int(posterior_size)
        # Plot posterior predictive
        config["offset"] = {}
        print("\tGetting Posterior Predictive")

        with Timer("Getting outputs"):
            outputs_list = get_runs(
                df_post, config, n_repeats=N)
            results = {}
            print("\n")

        for i, output in enumerate(outputs_list):
            results[i] = {}
            summary_creator = SummaryStats(
                output, config['targets'], config['zero_flag'], observed_data=d0)
            summary_creator.get_stats()
            results[i]['data'] = summary_creator.d0
            results[i]['residuals'] = summary_creator.residuals
            results[i]['stats'] = summary_creator.summary_stats

        resid_formatted = [{'Batch': i, 'Signal': j, 'Residuals': v, 'Time (sec)': idx+1} for i in results.keys(
        ) for j in results[i]['residuals'].keys() for idx, v in enumerate(results[i]['residuals'][j])]
        data_formatted = [{'Batch': i, 'Signal': j, 'Data': v, 'Time (sec)': idx+1} for i in results.keys(
        ) for j in results[i]['data'].keys() for idx, v in enumerate(results[i]['data'][j])]

        print("Residuals Dataframe")
        residuals = reduce_dataframe(pd.DataFrame(resid_formatted))
        del resid_formatted
        print("Data Dataframe")
        data = reduce_dataframe(pd.DataFrame(data_formatted))

        del data_formatted
        fig1, axes1 = plt.subplots(2, 2, figsize=(7, 7))
        fig2, axes2 = plt.subplots(2, 2, figsize=(7, 7))

        for ii, s in enumerate(config['targets']):
            signal_resid = residuals[residuals['Signal'] == s]['Residuals']
            ax1 = axes1.flatten()[ii]
            sns.distplot(signal_resid, ax=ax1)
            resid_mu, resid_sigma = np.mean(signal_resid), np.std(signal_resid)
            print("\t{}: Mean $(\mu$): {:.3g}\n\tStandard Deviation ($\sigma$): {:.3g}".format(
                s.upper(), resid_mu, resid_sigma))
            mean = ax1.axvline(resid_mu, color='k',
                               label='Mean', linestyle='--')
            std = ax1.axvline(resid_mu-resid_sigma, color='g',
                              label='Standard Deviation', linestyle='--')
            ax1.axvline(resid_mu+resid_sigma, color='g', linestyle='--')
            ax1.set_title("{}".format(s), fontsize=12)

            ax2 = axes2.flatten()[ii]
            resid = signal_resid.values.copy()
            sm.qqplot(resid, line='s', ax=ax2)
            ax2.axhline(0, color='k', linestyle='--')
            sample_mean = ax2.axhline(
                resid_mu, color='xkcd:orange', linestyle=':', label="Sample Mean")
            theoretical_mean = ax2.axvline(
                0, color='k', linestyle='--', label="Theoretical Mean")
            ax2.set_title("{}".format(s), fontsize=12)
            # print(stats.anderson(resid, dist='norm'))

        axes1[-1, -1].axis('off')
        axes2[-1, -1].axis('off')

        lgd1 = fig1.legend(handles=[mean, std], bbox_to_anchor=(
            0.55, 0.4), loc=2, fontsize=14)
        fig1.tight_layout()
        fig1.subplots_adjust(top=0.85)
        TIFF_exporter(fig1, 'residuals_dist_{}'.format(NEONATE),
                      fig_dir=figPath, extra_artists=(lgd1,))

        lgd2 = fig2.legend(handles=[theoretical_mean, sample_mean], bbox_to_anchor=(
            0.55, 0.4), loc=2, fontsize=14)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.85)
        TIFF_exporter(fig2, 'residuals_qq_{}'.format(NEONATE),
                      fig_dir=figPath, extra_artists=(lgd2,))

        posterior = {}
        prior = {}
        entropy = {}
        bins = {}
        fig4, axes4 = plt.subplots(
            ceil(len(config['parameters'])/3), 3, figsize=(7, 8))
        i = 0
        for k, v in config['parameters'].items():
            ax = axes4[i//3][i % 3]

            prior[k], bins[k] = np.histogram(np.random.uniform(
                v[1][0], v[1][1], LIM), 50, density=True)
            posterior[k], _ = np.histogram(
                df_post[k].values, bins=bins[k], density=True)

            entropy[k] = stats.entropy(posterior[k], prior[k])
            line_post = ax.bar(bins[k][:-1], posterior[k], width=bins[k]
                               [1]-bins[k][0], align='edge', label='Posterior')
            line_prior = ax.bar(bins[k][:-1], prior[k], width=bins[k]
                                [1]-bins[k][0], align='edge', alpha=.75, label='Prior')
            # ax.text(0.7,0.965, "Entropy: {:.3g}".format(entropy[k]), transform=ax.transAxes, size=16)
            ax.set_title(
                "K-L Divergence: {:.3g}".format(entropy[k]), y=1.01, fontsize=12)
            ax.set_xlabel(k)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)

            fig4.tight_layout()
            i += 1

        n_emptyAxes = 3-len(config['parameters']) % 3
        if n_emptyAxes > 0:
            for n in range(1, n_emptyAxes+1):
                axes4[-1, int(-1*n)].axis('off')

        lgd4 = fig4.legend(handles=[line_post, line_prior],
                           bbox_to_anchor=(0.7, 0.2), loc=2, fontsize=12)

        TIFF_exporter(fig4, 'kl_div_{}'.format(NEONATE),
                      fig_dir=figPath, extra_artists=(lgd4,))
        plt.close('all')
