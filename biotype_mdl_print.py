#!/usr/bin/env python

import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt

files = glob.glob('biotype-stability-rest*/*.npz')
data = np.zeros((len(files), 2))
for i, f in enumerate(files):
    mdl = np.load(f)
    data[i, 0] = mdl['n_cc']
    data[i, 1] = mdl['reg']

fig = plt.figure()
ax = plt.gca()
ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.1, edgecolors='none')
ax.set_yscale('log')
fig.savefig('biotype_reg_vs_ncomps.pdf')
