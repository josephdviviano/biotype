#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import seaborn as sns
import matplotlib.pyplot as plt

def zscore(X):
    return((X - X.mean(axis=0)) / X.std(axis=0))

merged = pd.read_csv('xbrain_database_with_biotypes.csv')
cols = ['scog_er40_crt_columnqcrt_value_inv', 'Part1_TotalCorrect', 'Part2_TotalCorrect',
        'Part3_TotalCorrect', 'RMET total', 'rad_total', 'np_domain_tscore_process_speed',
        'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning',
        'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps',
        'np_domain_tscore_att_vigilance']

# plot each score seperately
db = pd.DataFrame()
try:
    db['id'] = merged['ID']
except:
    db['id'] = merged['record_id']
db['biotype'] = merged['biotype']
db['diagnosis'] = merged['Diagnosis']

for col in cols:
    db[col] = zscore(merged[col])

db = pd.melt(db, id_vars=['id', 'biotype', 'diagnosis'], value_vars=cols)

# show diagnostic distributions for each biotype seperarely
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

sns.swarmplot(x="variable", y="value", hue="diagnosis", data=db.loc[db['biotype'] == 0], ax=ax1)
ax1.set_ylim([-4, 4])
ax1.set_title('normally-performing biotype')

sns.swarmplot(x="variable", y="value", hue="diagnosis", data=db.loc[db['biotype'] == 1], ax=ax2)
ax2.set_ylim([-4, 4])
ax2.set_title('sub-performing biotype')
sns.plt.xticks(rotation=45)

sns.plt.savefig('biotype_yscores_per_diagnosis_and_biotype.pdf')
sns.plt.close()

