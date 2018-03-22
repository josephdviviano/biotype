#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats.mstats import kruskalwallis
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests

# load data etc
mask = 'shen_1mm_268_parcellation.nii.gz'
fdr = False
input_mdls = ['biotype-restconn_test-restconn',
              'biotype-restconn_test-restconn_replication',
              'biotype-imobconn_test-imobconn',
              'biotype-imobconn_test-imobconn_replication',
              'biotype-eaconn_test-eaconn',
              'biotype-restconn_test-restconn',
              'biotype-restconn_test-restconn_replication',
              'biotype-imobconn_test-imobconn',
              'biotype-imobconn_test-imobconn_replication',
              'biotype-eaconn_test-eaconn']
names = ['rest', 'rest-replication', 'imob', 'imob-replication', 'ea',
         'rest', 'rest-replication', 'imob', 'imob-replication', 'ea']
types = ['b', 'b', 'b', 'b', 'b', 'd', 'd', 'd', 'd', 'd']
#input_mdls = ['biotype-restconn_test-restconn_motion', 'biotype-restconn_test-restconn_motion']
#names = ['motion', 'motion']
#types = ['b', 'd']

for i, input_mdl in enumerate(input_mdls):

    if types[i] == 'b':
        prefix = 'xbrain_biotype_rois_{}'.format(names[i])
    elif types[i] == 'd':
        prefix = 'xbrain_diagnosis_rois_{}'.format(names[i])
    else:
        print('invalid type: {}'.format(types[i]))
        sys.exit(1)
    output = '{}.nii.gz'.format(prefix)

    mdl = np.load(os.path.join(input_mdl, 'xbrain_biotype.npz'))
    db = pd.read_csv(os.path.join(input_mdl, 'xbrain_database_with_biotypes.csv'))
    nii = nib.load(mask)
    nii_data = nii.get_data()
    rois = np.unique(nii_data[nii_data > 0])

    clf = LabelEncoder()
    if types[i] == 'b':
         labels = clf.fit_transform(db['biotype'])
    elif types[i] == 'd':
         labels = clf.fit_transform(db['Diagnosis'])

    correlations = np.zeros((mdl['X'].shape[1], len(np.unique(labels))))

    # split data into groups
    group_data = []
    groups = np.unique(labels)
    for group in groups:
        idx = np.where(labels == group)[0]
        group_corrs = mdl['X'][idx, :]
        group_data.append(group_corrs)

    # collect pvalues for all possible pair-wise comparisons (if more than 2 groups)
    pairs = [",".join(map(str, comb)) for comb in combinations(groups, 2)]
    pvals = np.zeros((mdl['X'].shape[1], len(pairs)))

    # save mean connectivity for each group pair, and their differences
    output_list = []
    for j, p in enumerate(pairs):
        t1 = int(p.split(',')[0])
        t2 = int(p.split(',')[1])
        for k in np.arange(mdl['X'].shape[1]):
            #pvals[k, j] = kruskalwallis(group_data[t1][:, k], group_data[t2][:, k]).pvalue
            pvals[k, j] = ttest_ind(group_data[t1][:, k], group_data[t2][:, k])[1]

        # binarize with fdr correction or a uncorrected threshold
        if fdr:
            corrected = multipletests(np.ravel(pvals[:, j]), alpha=0.05, method='fdr_bh')
            passed = corrected[0]
            pvals_corrected = corrected[1]
        else:
            threshold = 1
            passed = pvals[:, j] < threshold

        # skip comparisons with no significant contrasts
        if len(passed) == 0:
            print('SKIPPING DUE TO NO SIG. DIFFERENCES')
            continue

        try:
            threshold = np.max(pvals[passed])
            print('input {}: pair {}/{}, threshold {}, n_connections {}'.format(
                names[i], j+1, len(pairs), threshold, sum(passed)))

            pvals[passed] = 1
            pvals[~passed] = 0

        except:
            pvals[:] = 0

        # take mean of thresholded group data
        d1 = np.mean(group_data[t1] * pvals[:, j].T, axis=0)
        d2 = np.mean(group_data[t2] * pvals[:, j].T, axis=0)

        # straight differences
        dd = d1 - d2

        # differences in absoloute connectivity
        dd_abs = np.abs(d1) - np.abs(d2)

        # differences in + correlations only
        dd_pos = np.zeros(d1.shape)
        dd_pos[d1 > 0] = d1[d1 > 0]
        dd_pos[d2 > 0] = dd_pos[d2 > 0] - d2[d2 > 0]

        # differences in - correlations only
        dd_neg = np.zeros(d1.shape)
        dd_neg[d1 < 0] = d1[d1 < 0]
        dd_neg[d2 < 0] = dd_neg[d2 < 0] - d2[d2 < 0]

        atlas_1 = np.zeros(nii_data.shape)
        atlas_2 = np.zeros(nii_data.shape)
        atlas_dd = np.zeros(nii_data.shape)
        atlas_da = np.zeros(nii_data.shape)
        atlas_dp = np.zeros(nii_data.shape)
        atlas_dn = np.zeros(nii_data.shape)

        # reconstruct correlation matrix (now representing relationships between
        # connectivity features in X and the components found for X).
        idx_triu = np.triu_indices(len(rois), k=1)
        n_idx = len(idx_triu[0])
        n_images = len(d1) / n_idx

        # we do this for multi-image inputs (IM/OB)
        for image in range(n_images):

            mat_1 = np.zeros((len(rois), len(rois)))
            mat_2 = np.zeros((len(rois), len(rois)))
            mat_dd = np.zeros((len(rois), len(rois)))
            mat_da = np.zeros((len(rois), len(rois)))
            mat_dp = np.zeros((len(rois), len(rois)))
            mat_dn = np.zeros((len(rois), len(rois)))

            # take sum in each ROI for each image (across all edges)
            mat_1[idx_triu] = d1[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_1 = mat_1 + mat_1.T
            np.savetxt('{}_group1_conn.csv'.format(prefix), mat_1, delimiter=',')
            mat_1 = np.sum(mat_1, axis=1)

            mat_2[idx_triu] = d2[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_2 = mat_2 + mat_2.T
            np.savetxt('{}_group2_conn.csv'.format(prefix), mat_2, delimiter=',')
            mat_2 = np.sum(mat_2, axis=1)

            mat_dd[idx_triu] = dd[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_dd = mat_dd + mat_dd.T
            np.savetxt('{}_dd_conn.csv'.format(prefix), mat_dd, delimiter=',')
            mat_dd = np.sum(mat_dd, axis=1)

            mat_da[idx_triu] = dd_abs[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_da = mat_da + mat_da.T
            np.savetxt('{}_da_conn.csv'.format(prefix), mat_da, delimiter=',')
            mat_da = np.sum(mat_da, axis=1)

            mat_dp[idx_triu] = dd_pos[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_dp = mat_dp + mat_dp.T
            np.savetxt('{}_dp_conn.csv'.format(prefix), mat_dp, delimiter=',')
            mat_dp = np.sum(mat_dp, axis=1)

            mat_dn[idx_triu] = dd_neg[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_dn = mat_dn + mat_dn.T
            np.savetxt('{}_dn_conn.csv'.format(prefix), mat_dn, delimiter=',')
            mat_dn = np.sum(mat_dn, axis=1)

            # load these connectivity values into the ROI mask
            for k, roi in enumerate(rois):
                atlas_1[nii_data == roi] = mat_1[k]
                atlas_2[nii_data == roi] = mat_2[k]
                atlas_dd[nii_data == roi] = mat_dd[k]
                atlas_da[nii_data == roi] = mat_da[k]
                atlas_dp[nii_data == roi] = mat_dp[k]
                atlas_dn[nii_data == roi] = mat_dn[k]

            output_list.append(atlas_1)
            output_list.append(atlas_2)
            output_list.append(atlas_dd)
            output_list.append(atlas_da)
            output_list.append(atlas_dp)
            output_list.append(atlas_dn)

    if len(output_list) > 0:
        output_nii = np.stack(output_list, axis=3)
        output_nii = nib.nifti1.Nifti1Image(output_nii, nii.affine, header=nii.header)
        output_nii.update_header()
        output_nii.header_class(extensions=())
        output_nii.to_filename(output)
    else:
        print('NO SIG DIFFERENCES FOR {}'.format(output))


