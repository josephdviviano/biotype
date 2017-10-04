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
cols = ['scog_er40_crt_columnqcrt_value_inv', 'Part1_TotalCorrect', 'Part2_TotalCorrect', 'Part3_TotalCorrect', 'RMET total', 'rad_total', 'np_domain_tscore_process_speed', 'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning', 'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps', 'np_domain_tscore_att_vigilance']
names = ['ER40 RT (inv)', 'Tasit 1', 'Tasit 2', 'Tasit 3', 'RMET', 'RAD', 'Processing Speed', 'Working Memory', 'Verbal Learning', 'Visual Learning', 'Reasoning', 'Attention/Vigilance']

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

healthy_group = 1  # healthy controls
labels = db['diagnosis']

if flip:
    input_df['biotype'] = np.abs(input_df['biotype']-1) # works because we only ever have 2 biotypes

for col in cols:
    db[col] = zscore_by_group(input_df[col], labels, healthy_group)

db = pd.melt(db, id_vars=['id', 'biotype', 'diagnosis'], value_vars=cols)

# show diagnostic distributions for each biotype seperarely
sns.set_style('white')

d_0 = db['value'].loc[db['biotype'] == 0]
d_1 = db['value'].loc[db['biotype'] == 1]

sns.distplot(d_0, hist=False, color="r", kde_kws={"shade": True}, label='biotype 0')
sns.distplot(d_1, hist=False, color="b", kde_kws={"shade": True}, label='biotype 1')
sns.plt.legend()

sns.plt.savefig('biotype_scores_collapsed.pdf')
sns.plt.close()

