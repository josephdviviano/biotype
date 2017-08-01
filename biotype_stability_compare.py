#!/usr/bin/env python

import pandas as pd
from copy import copy
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


diff = lambda l1,l2: [x for x in l1 if x not in l2]


def zscore(X):
    return((X - X.mean(axis=0)) / X.std(axis=0))


def zscore_by_group(X, labels, group):
    """z-scores each column of X by the mean and standard deviation of group."""
    assert(X.shape[0] == len(labels))
    idx = np.where(labels == group)[0]
    X_group_mean = np.mean(X.loc[idx], axis=0)
    X_group_std = np.std(X.loc[idx], axis=0)
    return((X - X_group_mean) / X_group_std)


def match_labels(a, b):

    ids_a = np.unique(a)
    ids_b = np.unique(b)
    n = len(ids_a)
    D = np.zeros((n, n))

    for x in np.arange(n):
        for y in np.arange(n):
            idx_a = np.where(a == x)[0]
            idx_b = np.where(b == y)[0]
            n_int = len(np.intersect1d(idx_a, idx_b))
            # distance = (# in cluster) - 2*sum(# in intersection)
            D[x,y] = (len(idx_a) + len(idx_b) - 2*n_int)

    # permute labels w/ minimum weighted bipartite matching (hungarian method)
    idx_D_x, idx_D_y = linear_sum_assignment(D)
    mappings = np.hstack((np.atleast_2d(idx_D_x).T, np.atleast_2d(idx_D_y).T))

    return(mappings)

#exp_names = ['rest', 'imob', 'imob-gm', 'ea', 'ea-gm', 'all']
exp_names = ['rest', 'imob-gm', 'imob-stat', 'ea-gm', 'ea-stat']
n_exp = 100

f1 = open('biotype_stability_compare.csv', 'wb')
f1.write('test,t,p\n')

X = []

for exp_name in exp_names:

    experiments = []
    for i in range(n_exp):
        experiments.append('biotype-stability-{}_{}'.format(exp_name, str(i).zfill(2)))

    cols = ['ID']
    failed = []
    for i, experiment in enumerate(experiments):
        try:
            mdl = np.load(os.path.join(experiment, 'xbrain_biotype.npz'))
            db = pd.read_csv(os.path.join(experiment, 'xbrain_database.csv'))
        except:
            print('failed {}'.format(experiment))
            failed.append(os.path.join(experiment, 'xbrain_biotype.npz'))
            continue

        colname = 'biotype_{}'.format(i)
        db[colname] = mdl['clusters']
        cols.append(colname)
        if i == 0:
            db_master = copy(db)
        else:
            db_master = db_master.merge(db, on='ID')

    db_master.to_csv('biotypes_combined.csv')
    db_master = db_master[cols]
    db_master.to_csv('biotypes_reduced.csv')

    # remove 'ID' from cols, so we can use them for x,y labels
    cols.reverse()
    cols.pop()
    cols.reverse()

    data = db_master[cols].as_matrix()

    for i in range(data.shape[1]):
        if i == 0:
            continue
        mappings = match_labels(data[:, 0], data[:, i])
        tmp = np.zeros(data.shape[0])

        for j in mappings:
            idx = np.where(data[:, i] == j[0])[0]
            tmp[idx] = j[1]
        data[:, i] = tmp

    # compute distances
    distances = pdist(data.T, metric='hamming')
    X.append(distances)

# compare all GLM with CONN
for glm_test in [2, 4]:
    for conn_test in [0,1,3]:
       test = mannwhitneyu(X[glm_test], X[conn_test])
       testname = '{} vs {}'.format(exp_names[glm_test], exp_names[conn_test])
       f1.write('{},{},{}\n'.format(testname, test[0], test[1]))

f1.close()

