#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import os
import argparse
import logging
import config
import higgs_geant

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
ds = tfp.distributions

from sklearn.model_selection import ShuffleSplit
from higgs_geant import normalize_weight
from higgs_geant import split_train_test
from higgs_geant import split_data_label_weights

from higgs_4v_pandas import tau_energy_scale
from higgs_4v_pandas import jet_energy_scale
from higgs_4v_pandas import lep_energy_scale
from higgs_4v_pandas import soft_term
from higgs_4v_pandas import nasty_background

from higgs_inferno import HiggsInferno


def parse_args():
    # TODO : more descriptive msg.
    parser = argparse.ArgumentParser(description="Training launcher")

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument('--n-epochs', help='number of epochs',
                        default=100, type=int)

    parser.add_argument('--batch-size', help='batch size',
                        default=2000, type=int)

    parser.add_argument("--name", type=str,
                        default="higgs_default", help="directory name where to save the model")

    args = parser.parse_args()
    return args



def main():
    # GET LOGGER
    #-----------
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.info('Hello')

    args = parse_args()
    logger.info(args)
    logger.handlers[0].flush()

    logger.info('Loading data ...')
    data = higgs_geant.load_data()
    data = data.drop( ["DER_mass_MMC"], axis=1 )
    data['numLabel'] = data['Label']
    data['Label'].loc[data['Label'] == 1] = 's'
    data['Label'].loc[data['Label'] == 0] = 'b'

    # CROSS VALIDATION
    #-----------------
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

        pars = ["mu",
                "tau_energy_sc",
                # "jet_energy_sc",
                # "lep_energy_sc",
                # "sigma_met",
                # "nasty_background_sc",
                ]
        aux = {"tau_energy_sc" : ds.Normal(loc=config.CALIBRATED_TAU_ENERGY_SCALE,
                                            scale=config.CALIBRATED_TAU_ENERGY_SCALE_ERROR),
                # "jet_energy_sc" : ds.Normal(loc=config.CALIBRATED_JET_ENERGY_SCALE,
                #                             scale=config.CALIBRATED_JET_ENERGY_SCALE_ERROR),
                # "lep_energy_sc" : ds.Normal(loc=config.CALIBRATED_LEP_ENERGY_SCALE,
                #                             scale=config.CALIBRATED_LEP_ENERGY_SCALE_ERROR),
                # "sigma_met" : ds.Normal(loc=config.CALIBRATED_SIGMA_SOFT,
                #                             scale=config.CALIBRATED_SIGMA_SOFT_ERROR),
                # "nasty_background_sc" : ds.Normal(loc=config.CALIBRATED_NASTY_BKG,
                #                             scale=config.CALIBRATED_NASTY_BKG_ERROR),
                }

        model_name ='{}-{}-{}-{}'.format(args.name, args.batch_size, args.n_epochs, i) 
        model_path = os.path.join(config.SAVING_DIR, model_name)
        os.makedirs(model_path, exist_ok=True)

        # TRAINING INFERNO
        #-----------------
        logger.info('Start training : {}'.format(model_name))
        inferno = HiggsInferno(model_path=model_path,
                                poi="mu", pars=pars, seed=17, aux=aux)
        inferno.fit(data_train, data_test, n_epochs=args.n_epochs, lr=1e-3,
                  temperature=0.1, batch_size=args.batch_size, seed=17)
        logger.info('End of training {}'.format(model_name))
        
        # COMPUTE SUMMARIES
        #------------------
        logger.info('Computing summary statistic for test data')
        sig_weighted_counts, bkg_weighted_counts, total_weighted_counts = inferno.compute_summaries(data_test)
        logger.info('Ploting...')
        try:
            plt.bar(np.arange(10)+0.1, bkg_weighted_counts, width=0.4, label='b')
            plt.bar(np.arange(10)+0.5, sig_weighted_counts, width=0.4, label='s')
            plt.yscale('log')
            plt.xticks(np.arange(10)+0.5, labels=range(10))
            plt.legend()
            plt.savefig(os.path.join(model_path, 'test_counts_sig_bkg.png'))
            plt.clf()
        except:
            logger.warn('Error in ploting test_counts_sig_bkg')
            pass
        logger.info('Computing summary statistic for Exp data')
        sig_weighted_counts, bkg_weighted_counts, total_weighted_counts = inferno.compute_summaries(data_xp)
        logger.info('Ploting...')
        try:
            plt.bar(np.arange(10)+0.1, bkg_weighted_counts, width=0.4, label='b')
            plt.bar(np.arange(10)+0.5, sig_weighted_counts, width=0.4, label='s')
            plt.yscale('log')
            plt.xticks(np.arange(10)+0.5, labels=range(10))
            plt.legend()
            plt.savefig(os.path.join(model_path, 'xp_counts_sig_bkg.png'))
            plt.clf()
        except:
            logger.warn('Error in ploting xp_counts_sig_bkg')
            pass
        try:
            plt.bar(np.arange(10)+0.1, total_weighted_counts, width=0.7, label='total')
            plt.yscale('log')
            plt.xticks(np.arange(10)+0.5, labels=range(10))
            plt.legend()
            plt.savefig(os.path.join(model_path, 'xp_counts_total.png'))
            plt.clf()
        except:
            logger.warn('Error in ploting xp_counts_total')
            pass



if __name__ == '__main__':
    main()