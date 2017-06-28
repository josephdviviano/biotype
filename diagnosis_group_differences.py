#!/usr/bin/env python

import numpy as np
import nibabel as nib
from scipy.stats.mstats import kruskalwallis
from scipy.stats import ttest_ind
from itertools import combinations
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from statsmodels.sandbox.stats.multicomp import multipletests

# load data etc
mask = 'shen_1mm_268_parcellation.nii.gz'

input_folders = ['biotype-restconn_test-restconn',
                 'biotype-restconn_test-restconn_replication',
                 'biotype-imobconn_test-imobconn',
                 'biotype-imobconn_test-imobconn_replication',
                 'biotype-eaconn_test-eaconn']

names = ['rest', 'rest-replication', 'imob', 'imob-replication', 'ea']

for ii, input_folder in enumerate(input_folders):

    output = 'xbrain_diagnosis_rois_{}.nii.gz'.format(names[ii])

    mdl = np.load(os.path.join(input_folder, 'xbrain_biotype.npz'))
    db = pd.read_csv(os.path.join(input_folder, 'xbrain_database_with_biotypes.csv'))
    nii = nib.load(mask)
    nii_data = nii.get_data()
    rois = np.unique(nii_data[nii_data > 0])

    clf = LabelEncoder()
    diagnosis = clf.fit_transform(db['Diagnosis'])

    correlations = np.zeros((mdl['X'].shape[1], len(np.unique(diagnosis))))

    # split data into groups
    group_data = []
    groups = np.unique(diagnosis)
    for group in groups:
        idx = np.where(diagnosis == group)[0]
        group_corrs = mdl['X'][idx, :]
        group_data.append(group_corrs)

    # collect pvalues for all possible pair-wise comparisons (if more than 2 groups)
    pairs = [",".join(map(str, comb)) for comb in combinations(groups, 2)]
    pvals = np.zeros((mdl['X'].shape[1], len(pairs)))

    for i, p in enumerate(pairs):
        t1 = int(p.split(',')[0])
        t2 = int(p.split(',')[1])
        for j in np.arange(mdl['X'].shape[1]):
            #pvals[j, i] = kruskalwallis(group_data[t1][:, j], group_data[t2][:, j]).pvalue
            pvals[k, j] = ttest_ind(group_data[t1][:, k], group_data[t2][:, k])[1]

    # binarize with fdr correction
    corrected = multipletests(np.ravel(pvals), alpha=0.05, method='fdr_bh')
    passed = corrected[0]
    pvals_corrected = corrected[1]
    threshold = np.max(pvals[passed])
    pvals[passed] = 1
    pvals[~passed] = 0

    print('input {}: threshold {}, n_connections {}'.format(
        names[i], threshold, sum(passed)))

    for i in range(len(group_data)):
        group_data[i] = group_data[i] * pvals.T

    # save mean connectivity for each group pair, and their differences
    output_list = []
    for i, p in enumerate(pairs):
        t1 = int(p.split(',')[0])
        t2 = int(p.split(',')[1])

        d1 = np.mean(group_data[t1], axis=0)
        d2 = np.mean(group_data[t2], axis=0)
        dd = d1-d2

        atlas_1 = np.zeros(nii_data.shape)
        atlas_2 = np.zeros(nii_data.shape)
        atlas_d = np.zeros(nii_data.shape)

        # reconstruct correlation matrix (now representing relationships between
        # connectivity features in X and the components found for X).
        idx_triu = np.triu_indices(len(rois), k=1)
        n_idx = len(idx_triu[0])
        n_images = len(d1) / n_idx

        for image in range(n_images):

            mat_1 = np.zeros((len(rois), len(rois)))
            mat_2 = np.zeros((len(rois), len(rois)))
            mat_d = np.zeros((len(rois), len(rois)))

            # take mean in each ROI for each image (across all edges)
            mat_1[idx_triu] = d1[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_1 = mat_1 + mat_1.T
            mat_1 = np.sum(mat_1, axis=1)
            mat_2[idx_triu] = d2[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_2 = mat_2 + mat_2.T
            mat_2 = np.sum(mat_2, axis=1)
            mat_d[idx_triu] = dd[n_idx*image:n_idx*(image+1)] # for multi-image inputs (e.g., imob)
            mat_d = mat_d + mat_d.T
            mat_d = np.sum(mat_d, axis=1)

            # load these connectivity values into the ROI mask
            for k, roi in enumerate(rois):
                atlas_1[nii_data == roi] = mat_1[k]
                atlas_2[nii_data == roi] = mat_2[k]
                atlas_d[nii_data == roi] = mat_d[k]

            output_list.append(atlas_1)
            output_list.append(atlas_2)
            output_list.append(atlas_d)


    output_nii = np.stack(output_list, axis=3)
    output_nii = nib.nifti1.Nifti1Image(output_nii, nii.affine, header=nii.header)
    output_nii.update_header()
    output_nii.header_class(extensions=())
    output_nii.to_filename(output)

