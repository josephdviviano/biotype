#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys

# load data etc
input_a = sys.argv[1]
flip_a = sys.argv[2]
input_b = sys.argv[3]
flip_b = sys.argv[4]
output = sys.argv[5]

# flip input diff mats (if required)
if flip_a == 'flip':
    flip_a = True
else:
    flip_a = False

if flip_b == 'flip':
    flip_b = True
else:
    flip_b = False

# load matrices
dat_a = np.genfromtxt(input_a, delimiter=',')
dat_b = np.genfromtxt(input_b, delimiter=',')

# flip if required
if flip_a:
    dat_a = dat_a * -1
if flip_b:
    dat_b = dat_b * -1

# take upper triangle so we don't double-count connections
idx_triu = np.triu_indices(len(dat_a), k=1)
dat_a = dat_a[idx_triu]
dat_b = dat_b[idx_triu]

# find connections that are significant in both datasets
idx_a = np.where(dat_a)[0]
idx_b = np.where(dat_b)[0]
idx = np.intersect1d(idx_a, idx_b)

# plot the positive and negative corrs seperately
xa = dat_a[idx]
xb = dat_b[idx]

idx_pos = np.where(xa > 0)[0]
idx_neg = np.where(xa < 0)[0]

sns.set(style="white")
if len(idx) > 2:
    r, p = pearsonr(xa, xb)
    sns.regplot(xa, xb, color='r')
    sns.plt.title('r={}, p={}'.format(r, p))
    sns.plt.savefig('{}.pdf'.format(output))
else:
    print('no shared corrs')
#if len(idx_pos) > 2:
#    print('positive r={}'.format(np.corrcoef(xa[idx_pos].T, xb[idx_pos].T)[0][1]))
#    g = sns.jointplot(xa[idx_pos], xb[idx_pos], kind="reg", color="r", size=7, space=0)
#    g.savefig('{}_pos.pdf'.format(output))
#else:
#    print('no shared positive corrs')
#
#if len(idx_neg) > 2:
#    print('negative r={}'.format(np.corrcoef(xa[idx_neg].T, xb[idx_neg].T)[0][1]))
#    g = sns.jointplot(xa[idx_neg], xb[idx_neg], kind="reg", color="r", size=7, space=0)
#    g.savefig('{}_neg.pdf'.format(output))
#else:
#    print('no shared negative corrs')

