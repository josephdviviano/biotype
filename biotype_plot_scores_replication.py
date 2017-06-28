#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

def zscore(X):
    return((X - X.mean(axis=0)) / X.std(axis=0))

merged = pd.read_csv('xbrain_database_with_biotypes.csv')
cols = ['scog_er40_crt_columnqcrt_value_inv', 'Part1_TotalCorrect', 'Part2_TotalCorrect', 'Part3_TotalCorrect', 'RMET total', 'np_domain_tscore_process_speed', 'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning', 'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps', 'np_domain_tscore_att_vigilance']

# plot each score seperately
db = pd.DataFrame()
db['id'] = merged['record_id']
db['biotype'] = merged['biotype']
#db['cluster'] = merged['cluster']

#for col in cols:
#    print('{} nans: {}'.format(col, np.sum(np.isnan(merged[col]))))

for i, col in enumerate(cols):
    db[col] = zscore(merged[col])

db = pd.melt(db, id_vars=['id', 'biotype'], value_vars=cols)
#db = pd.melt(db, id_vars=['id', 'biotype', 'cluster'], value_vars=cols)

sns.set(style='ticks')
sns.violinplot(x="variable", y="value", hue="biotype", data=db, split=True, palette="RdBu")
sns.plt.xticks(rotation=45)
sns.plt.savefig('biotype_violin.pdf')
sns.plt.close()

sns.boxplot(x="variable", y="value", hue="biotype", data=db, palette="RdBu")
sns.plt.xticks(rotation=45)
sns.plt.savefig('biotype_box.pdf')
sns.plt.close()

#sns.boxplot(x="variable", y="value", hue="cluster", data=db, palette="RdBu")
#sns.plt.xticks(rotation=45)
#sns.plt.savefig('cluster_box.pdf')
#sns.plt.close()

# plot just PCs
db = pd.DataFrame()
db['id'] = merged['record_id']
db['biotype'] = merged['biotype']
#db['cluster'] = merged['cluster']
X = merged[cols].as_matrix()
X = zscore(X)
clf = PCA(n_components=3)
pca_comp = clf.fit_transform(X)

# invert components if required to maintain interpretability
if np.corrcoef(pca_comp[:, 0], np.mean(X, axis=1))[0][1] < 0:
    pca_comp = pca_comp * -1

db['pc1'] = pca_comp[:, 0]
db['pc2'] = pca_comp[:, 1]
db['pc3'] = pca_comp[:, 2]
db = pd.melt(db, id_vars=['id', 'biotype'], value_vars=['pc1', 'pc2', 'pc3'])
#db = pd.melt(db, id_vars=['id', 'biotype', 'cluster'], value_vars=['pc1', 'pc2', 'pc3'])

sns.boxplot(x="variable", y="value", hue="biotype", data=db, palette="RdBu")
sns.plt.xticks(rotation=45)
sns.plt.savefig('biotype_box_pcs.pdf')
sns.plt.close()

#sns.boxplot(x="variable", y="value", hue="cluster", data=db, palette="RdBu")
#sns.plt.xticks(rotation=45)
#sns.plt.savefig('cluster_box_pcs.pdf')
#sns.plt.close()


