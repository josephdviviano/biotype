#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data etc

# good agreement
#input_a = 'xbrain_biotype_rois_rest_dp_conn.csv'
#input_b = 'xbrain_biotype_rois_rest-replication_dp_conn.csv'

# no relationship
#input_a = 'xbrain_biotype_rois_imob_dp_conn.csv'
#input_b = 'xbrain_biotype_rois_imob-replication_dp_conn.csv'

# no relationship
#input_a = 'xbrain_biotype_rois_imob_dp_conn.csv'
#input_b = 'xbrain_biotype_rois_ea_dp_conn.csv'

# questionable relationship
#input_a = 'xbrain_biotype_rois_rest_dp_conn.csv'
#input_b = 'xbrain_biotype_rois_ea_dp_conn.csv'

# no relationship
#input_a = 'xbrain_biotype_rois_rest_dp_conn.csv'
#input_b = 'xbrain_biotype_rois_imob_dp_conn.csv'

flip_a = False
flip_b = False

dat_a = np.genfromtxt(input_a, delimiter=',')
dat_b = np.genfromtxt(input_b, delimiter=',')

if flip_a:
    dat_a = dat_a * -1
if flip_b:
    dat_b = dat_b * -1

dims = dat_a.shape
dat_a = dat_a.reshape(dims[0]*dims[1], 1)
dat_b = dat_b.reshape(dims[0]*dims[1], 1)

idx_a = np.where(dat_a)[0]
idx_b = np.where(dat_b)[0]
idx = np.intersect1d(idx_a, idx_b)

xa = dat_a[idx]
xb = dat_b[idx]

print(idx)

idx_pos = np.where(xa > 0)[0]
idx_neg = np.where(xa < 0)[0]

sns.set(style="white")
if len(idx_pos) > 4:
    print('positive r={}'.format(np.corrcoef(xa[idx_pos].T, xb[idx_pos].T)[0][1]))
    g = sns.jointplot(xa[idx_pos], xb[idx_pos], kind="reg", color="r", size=7, space=0)
    g.savefig('biotype_orig-vs-replication_pos.pdf')
else:
    print('no shared positive corrs')

if len(idx_neg) > 4:
    print('negative r={}'.format(np.corrcoef(xa[idx_neg].T, xb[idx_neg].T)[0][1]))
    g = sns.jointplot(xa[idx_neg], xb[idx_neg], kind="reg", color="r", size=7, space=0)
    g.savefig('biotype_orig-vs-replication_neg.pdf')
else:
    print('no shared negative corrs')

