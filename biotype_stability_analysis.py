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
exp_names = ['rest', 'imob-gm', 'imob-stat', 'ea-gm', 'ea-stat', 'all']
n_exp = 100

f1 = open('biotype_assignment_stability.csv', 'wb')
f1.write('experiment,n_total,n_unstable,mean_instability,n_ctl,mean_instability_ctl,n_scz,mean_instability_scz\n')

f2 = open('biotype_stability_scores.csv', 'wb')
f2.write('experiment,mean,median,std\n')

for exp_name in exp_names:

    experiments = []
    if exp_name != 'all':
        for i in range(n_exp):
            experiments.append('biotype-stability-{}_{}'.format(exp_name, str(i).zfill(2)))
    else:
        all_exp_names = diff(exp_names, ['all'])
        for e in all_exp_names:
            for i in range(n_exp):
                experiments.append('biotype-stability-{}_{}'.format(e, str(i).zfill(2)))

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

    # plot all pairwise distances
    distances = squareform(pdist(data.T, metric='hamming'))
    sns.heatmap(distances, vmin=0, vmax=0.35, square=True, linewidths=0)
    sns.plt.savefig('biotype_distance_{}.pdf'.format(exp_name))
    sns.plt.close()

    mean = np.mean(pdist(data.T, metric='hamming'))
    median = np.median(pdist(data.T, metric='hamming'))
    std = np.std(pdist(data.T, metric='hamming'))
    f2.write('{},{},{},{}\n'.format(exp_name, mean, median, std))

    pairwise_distances = distances[np.triu_indices(np.shape(distances)[0], k=1)]
    print('distance between biotypes {}: {}+/-{}'.format(exp_name,
        np.mean(pairwise_distances), np.std(pairwise_distances)))

    # calculate % of times the subject flipped
    instability = np.sum(np.abs(np.diff(data, axis=1)), axis=1) / (float(data.shape[1])-1)
    instability[instability == 0] = np.nan
    db_master['instability'] = instability
    db = pd.read_csv('/projects/jviviano/data/xbrain/assets/database_xbrain.csv')
    db_final = db.merge(db_master, on='ID')

    scores = ['scog_er40_crt_columnqcrt_value_inv', 'Part1_TotalCorrect', 'Part2_TotalCorrect',
              'Part3_TotalCorrect', 'RMET total', 'rad_total', 'np_domain_tscore_process_speed',
              'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning',
              'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps',
              'np_domain_tscore_att_vigilance']

    # zscore scores (normalize against healthy controls)
    for score in scores:
        db_final[score] = zscore_by_group(db_final[score], db_final['Diagnosis'], 1)

    db_melt = pd.melt(db_final, id_vars=['ID', 'Diagnosis', 'instability'], value_vars=scores)
    db_melt = db_melt.dropna()

    # show instability vs score (collapsed across 12 tests)
    sns.set(style="white")
    sns.kdeplot(db_melt['value'], db_melt['instability'], cmap='Reds', shade=True, shade_lowest=False)
    sns.plt.xlim((-3,3))
    sns.plt.ylim((0,1))
    sns.plt.savefig('biotype_assignment_instability_combined_{}'.format(exp_name))
    sns.plt.close()

    # show instability vs score for each test
    g = sns.FacetGrid(db_melt, col="variable", margin_titles=True, xlim=[-3,3], ylim=[0,1])
    g.map(sns.kdeplot, 'value', 'instability', cmap='Reds', shade=True, shade_lowest=False)
    g.savefig('biotype_assignment_instability_individual_{}.pdf'.format(exp_name))
    sns.plt.close()

    # save stats
    f1.write('{experiment},{n_total},{n_unstable},{mean_instability},{n_ctl},{mean_instability_ctl},{n_scz},{mean_instability_scz}\n'.format(
        experiment=exp_name,
        n_total=len(db_final),
        n_unstable=np.sum(np.isfinite(db_final['instability'])),
        mean_instability=np.nanmean(db_final['instability']),
        n_ctl=np.sum(np.isfinite(db_final['instability'].loc[db_final['Diagnosis'] == 1])),
        mean_instability_ctl=np.nanmean(db_final['instability'].loc[db_final['Diagnosis'] == 1]),
        n_scz=np.sum(np.isfinite(db_final['instability'].loc[db_final['Diagnosis'] == 4])),
        mean_instability_scz=np.nanmean(db_final['instability'].loc[db_final['Diagnosis'] == 4])))

    for failure in failed:
        print('redo: {}'.format(failure))
    print('done: {}'.format(exp_name))

f1.close()
f2.close()

