#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import config
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
ds = tfp.distributions

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import ShuffleSplit
from higgs_geant import load_data
from higgs_geant import split_data_label_weights
from higgs_geant import split_train_test
from higgs_geant import normalize_weight

from higgs_4v_pandas import tau_energy_scale
from higgs_4v_pandas import jet_energy_scale
from higgs_4v_pandas import lep_energy_scale
from higgs_4v_pandas import soft_term
from higgs_4v_pandas import nasty_background

from nll import HiggsNLL
from higgs_example import HiggsExample
from higgs_inferno import HiggsInferno

from higgs_template_model import HiggsTemplateModel

sns.set()
sns.set_style("whitegrid")
sns.set_context("poster")

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


data = load_data()
data = data.drop('DER_mass_MMC',axis=1)

statistical_error = []
full_error = []

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.info('Hello')


cv_sim_xp = ShuffleSplit(n_splits=config.N_CV, test_size=0.2, random_state=config.RANDOM_STATE)
for i, (idx_sim, idx_xp) in enumerate(cv_sim_xp.split(data, data['Label'])):
    # SPLIT DATA
    #-----------
    data_sim, data_xp = split_train_test(data, idx_sim, idx_xp)
    cv_train_test = ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    idx_train, idx_test = next(cv_train_test.split(data_sim, data_sim['Label']))
    data_train, data_test = split_train_test(data_sim, idx_train, idx_test)

    data_train = data_train.copy()
    data_test = data_test.copy()
    # data_train["origWeight"] = data_train["Weight"]
    # data_train['Weight'] = normalize_weight(data_train['Weight'], data_train['Label'])
    data_test["origWeight"] = data_test["Weight"]
    data_test['Weight'] = normalize_weight(data_test['Weight'], data_test['Label'])

    X_train, y_train, W_train = split_data_label_weights(data_train)
    X_test, y_test, W_test = split_data_label_weights(data_test)

    data_infer = data_xp.copy()
    data_infer["origWeight"] = data_infer["Weight"]
    data_infer['Weight'] = normalize_weight(data_infer['Weight'], data_infer['Label'])
    # tau_energy_scale(data_infer, scale=config.TRUE_TAU_ENERGY_SCALE)
    # jet_energy_scale(data_infer, scale=config.TRUE_JET_ENERGY_SCALE)
    # lep_energy_scale(data_infer, scale=config.TRUE_LEP_ENERGY_SCALE)
    # soft_term(data_infer, config.TRUE_SIGMA_SOFT)
    # nasty_background(data_infer, config.TRUE_NASTY_BKG)
    X_infer, y_infer, W_infer = split_data_label_weights(data_infer)

    # NOT TRAINING INFERNO
    #-----------------
    model_name ='{}-{}-{}-{}'.format('higgs_default', 2000, 300, i) 
    model_path = os.path.join(config.SAVING_DIR, model_name)
    pars = ["mu",
            "tau_energy_sc",
            "jet_energy_sc",
            "lep_energy_sc",
            "sigma_met",
            "nasty_background_sc",
            ]
    aux = {"tau_energy_sc" : ds.Normal(loc=config.CALIBRATED_TAU_ENERGY_SCALE,
                                        scale=config.CALIBRATED_TAU_ENERGY_SCALE_ERROR),
            "jet_energy_sc" : ds.Normal(loc=config.CALIBRATED_JET_ENERGY_SCALE,
                                        scale=config.CALIBRATED_JET_ENERGY_SCALE_ERROR),
            "lep_energy_sc" : ds.Normal(loc=config.CALIBRATED_LEP_ENERGY_SCALE,
                                        scale=config.CALIBRATED_LEP_ENERGY_SCALE_ERROR),
            "sigma_met" : ds.Normal(loc=config.CALIBRATED_SIGMA_SOFT,
                                        scale=config.CALIBRATED_SIGMA_SOFT_ERROR),
            "nasty_background_sc" : ds.Normal(loc=config.CALIBRATED_NASTY_BKG,
                                        scale=config.CALIBRATED_NASTY_BKG_ERROR),
            }

    logger.info('loading {}'.format(model_name))
    model = HiggsInferno(model_path=model_path,
                            poi="mu", pars=pars, seed=17, aux=aux)

    s_f_matrix = data_train.loc[:, HiggsExample().features].values
    mean_and_std = (s_f_matrix.mean(axis=0),s_f_matrix.std(axis=0))
    model.phs_scale = {model.scale_means : mean_and_std[0],
                      model.scale_stds : mean_and_std[1]}

    # model.NOTRAIN = True
    # model.fit(data_train, data_test, n_epochs=300, lr=1e-3,
    #               temperature=0.1, batch_size=2000, seed=17)

    data_test['numLabel'] = data_test['Label']
    data_test['Label'].loc[data_test['numLabel'] == 0] = 'b'
    data_test['Label'].loc[data_test['numLabel'] == 0] = 's'
    negative_log_likelihood = HiggsNLL(model, data_test, X_infer, W_infer, N_BIN=10)

    higgs_template = HiggsTemplateModel(multiple_pars=False)

    tau_energy=[
              config.CALIBRATED_TAU_ENERGY_SCALE,
              config.CALIBRATED_TAU_ENERGY_SCALE+config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
              config.CALIBRATED_TAU_ENERGY_SCALE-config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
              ]
    jet_energy=[
              config.CALIBRATED_JET_ENERGY_SCALE,
              config.CALIBRATED_JET_ENERGY_SCALE+config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
              config.CALIBRATED_JET_ENERGY_SCALE-config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
              ]
    lep_energy=[
              config.CALIBRATED_LEP_ENERGY_SCALE,
              config.CALIBRATED_LEP_ENERGY_SCALE+config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
              config.CALIBRATED_LEP_ENERGY_SCALE-config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
              ]
    sigma_soft=[
              config.CALIBRATED_SIGMA_SOFT,
              config.CALIBRATED_SIGMA_SOFT+config.CALIBRATED_SIGMA_SOFT_ERROR,
              config.CALIBRATED_SIGMA_SOFT-config.CALIBRATED_SIGMA_SOFT_ERROR,
              ]
    nasty_bkg=[
              config.CALIBRATED_NASTY_BKG,
              config.CALIBRATED_NASTY_BKG+config.CALIBRATED_NASTY_BKG_ERROR,
              config.CALIBRATED_NASTY_BKG-config.CALIBRATED_NASTY_BKG_ERROR,
              ]
    all_pars = [tau_energy, jet_energy, lep_energy, sigma_soft, nasty_bkg]

    # FILL TEMPLATE DICT
    # ------------------
    template_dict = {}

    skew_data = negative_log_likelihood.get_skew_data(tau_energy[0], jet_energy[0], lep_energy[0], sigma_soft[0], nasty_bkg[0])
    s_histo, b_histo, _ = model.compute_summaries(skew_data)
    template_dict[('b', tau_energy[0], jet_energy[0], lep_energy[0], sigma_soft[0], nasty_bkg[0])] = b_histo
    template_dict[('s', tau_energy[0], jet_energy[0], lep_energy[0], sigma_soft[0], nasty_bkg[0])] = s_histo


    N_pars = len(all_pars)
    for i in range(N_pars):
        tmp = [all_pars[j][0] for j in range(N_pars)]
        tmp[i] = all_pars[i][1]
        skew_data = negative_log_likelihood.get_skew_data(*tmp)
        s_histo, b_histo, _ = model.compute_summaries(skew_data)
        template_dict[('b', *tmp)] = b_histo
        template_dict[('s', *tmp)] = s_histo


    N_pars = len(all_pars)
    for i in range(N_pars):
        tmp = [all_pars[j][0] for j in range(N_pars)]
        tmp[i] = all_pars[i][2]
        skew_data = negative_log_likelihood.get_skew_data(*tmp)
        s_histo, b_histo, _ = model.compute_summaries(skew_data)
        template_dict[('b', *tmp)] = b_histo
        template_dict[('s', *tmp)] = s_histo


    _ =  higgs_template.templates_from_dict(template_dict)
    
    with tf.Session() as sess:
        fisher_matrix = higgs_template.asimov_hess(sess=sess)    

    stat_error = fisher_matrix.marginals(['mu'])
    f_error = fisher_matrix.marginals(higgs_template.all_pars.keys())
    print('Satistical error', stat_error)
    print('Full Uncertainties')
    print(f_error)

    statistical_error.append(stat_error)
    full_error.append(f_error)

print()
print()
print('statistical_error')
print()
print([e['mu'] for e in statistical_error])
print()
print('full error')
print()
print([d['mu'] for d in full_error])
