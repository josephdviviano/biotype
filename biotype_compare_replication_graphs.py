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
output_name = sys.argv[5]

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

idx_a = np.where(dat_a)
idx_b = np.where(dat_b)
idx_both = np.logical_and(dat_a != 0, dat_b != 0)

output = np.zeros(dat_a.shape)
output[output == 0] = np.nan
output[idx_a] = 1
output[idx_b] = 2
output[idx_both] = 3

sns.set(style="white")
plt.imshow(output, cmap=plt.cm.viridis, interpolation='nearest')
#sns.heatmap(output, cmap=plt.cm.Set1)
plt.title('shared connections: {} vs {}'.format(input_a, input_b))
plt.savefig('{}.pdf'.format(output_name))

