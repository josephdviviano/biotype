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
fig, (ax1, ax2) = plt.subplots(figsize=(10, 7), nrows=2, sharex=True)
plt.subplots_adjust(left=0.125, bottom=0.15, right=0.9, top=0.85, wspace=0.25, hspace=0.25)
plt.suptitle('Diagnosis distribution per biotype')

sns.swarmplot(x="variable", y="value", hue="diagnosis", data=db.loc[db['biotype'] == 0], ax=ax1)
ax1.set_ylim([-4, 4])
ax1.set_title('Average-performing biotype')
ax1.set_xticklabels([], rotation=45, ha='right')
ax1.hlines(0,  ax1.xaxis.get_majorticklocs()[0],  ax1.xaxis.get_majorticklocs()[-1])

sns.swarmplot(x="variable", y="value", hue="diagnosis", data=db.loc[db['biotype'] == 1], ax=ax2)
ax2.set_ylim([-4, 4])
ax2.set_title('Poor-performing biotype')
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.hlines(0,  ax1.xaxis.get_majorticklocs()[0],  ax1.xaxis.get_majorticklocs()[-1])

sns.plt.savefig('biotype_yscores_per_diagnosis_and_biotype.pdf')
sns.plt.close()

