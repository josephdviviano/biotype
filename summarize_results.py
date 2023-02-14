#!/usr/bin/env python

import os, sys
from glob import glob
import pandas as pd
import numpy as np

FIELDS = ['auc_test_macro', 'accuracy_test_macro', 'recall_test_macro',
          'precision_test_macro', 'f1_test_macro']
header = 'Analyses, Data, # of variables, AUC, Accuracy, Recall, Precision, f1\n'

f = open('all_stats.csv', 'wb')
f.write(header)

folders = os.listdir('./')
folders.sort()
folders = filter(lambda x: '.csv' not in x, folders)

for experiment in folders:
    try:
        means = pd.read_csv(os.path.join(experiment, 'xbrain_results_final_mean.csv'))
        sems = pd.read_csv(os.path.join(experiment, 'xbrain_results_final_sem.csv'))
    except:
        print('{} failed, skipping'.format(experiment))
        continue

    idx = means['auc_test_macro'] == np.max(means['auc_test_macro'])

    auc = (means['auc_test_macro'].loc[idx], sems['auc_test_macro'].loc[idx])
    acc = (means['accuracy_test'].loc[idx],  sems['accuracy_test'].loc[idx])
    rec = (means['recall_test_macro'].loc[idx], sems['recall_test_macro'].loc[idx])
    pre = (means['precision_test_macro'].loc[idx], sems['precision_test_macro'].loc[idx])
    f1r = (means['f1_test_macro'].loc[idx], sems['f1_test_macro'].loc[idx])
    #import IPython; IPython.embed()

    line = '{0},,,{1:.2f}+/-{2:.2f},{3:.2f}+/-{4:.2f},{5:.2f}+/-{6:.2f},{7:.2f}+/-{8:.2f},{9:.2f}+/-{10:.2f}\n'.format(
        experiment, float(auc[0]), float(auc[1]),
                    float(acc[0]), float(acc[1]),
                    float(rec[0]), float(rec[1]),
                    float(pre[0]), float(pre[1]),
                    float(f1r[0]), float(f1r[1]))

    f.write(line)

f.close()



