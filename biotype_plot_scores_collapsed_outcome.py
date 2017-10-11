#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def zscore(X):
    return((X - X.mean(axis=0)) / X.std(axis=0))


def zscore_by_group(X, labels, group):
    """z-scores each column of X by the mean and standard deviation of group."""
    assert(X.shape[0] == len(labels))
    idx = np.where(labels == group)[0]
    X_group_mean = np.mean(X.loc[idx], axis=0)
    X_group_std = np.std(X.loc[idx], axis=0)
    return((X - X_group_mean) / X_group_std)

input_df = pd.read_csv('xbrain_database_with_biotypes.csv')
cols = ['bsfs_total', 'qls_factor_interpersonal', 'qls_factor_instrumental_role', 'qls_factor_intrapsychic', 'qls_factor_comm_obj_activities', 'qls_total', 'sans_total_sc', 'sans_dim_exp_avg', 'sans_dim_mot_avg', 'bprs_factor_total']
flipper = [False, False, False, False, False, False, True, True, True, True]

flip = False
try:
    if sys.argv[1] == 'flip':
        flip = True
except:
    pass

# plot each score seperately
db = pd.DataFrame()
try:
    db['id'] = input_df['ID']
except:
    db['id'] = input_df['record_id']
db['biotype'] = input_df['biotype']
db['diagnosis'] = input_df['Diagnosis']

#healthy_group = 'HC'  # healthy controls
healthy_group = 1
labels = db['diagnosis']

if flip:
    input_df['biotype'] = np.abs(input_df['biotype']-1) # works because we only ever have 2 biotypes

input_df = input_df.loc[input_df['Diagnosis'] != healthy_group]

for i, col in enumerate(cols):
    if flipper[i]:
        db[col] = zscore(input_df[col]*-1)
    else:
        db[col] = zscore(input_df[col])

db = pd.melt(db, id_vars=['id', 'biotype', 'diagnosis'], value_vars=cols)

# show diagnostic distributions for each biotype seperarely
sns.set_style('white')

d_0 = db['value'].loc[db['biotype'] == 0]
d_1 = db['value'].loc[db['biotype'] == 1]

import IPython; IPython.embed()

sns.distplot(d_0, hist=False, color="r", kde_kws={"shade": True}, label='biotype 0')
sns.distplot(d_1, hist=False, color="b", kde_kws={"shade": True}, label='biotype 1')
sns.plt.legend()

sns.plt.savefig('biotype_scores_collapsed_outcome.pdf')
sns.plt.close()

