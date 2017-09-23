#!/usr/bin/env python
"""
must be run in an xbrain output folder
"""


import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

flip = True

def split_name(x):
    return('_'.join(x.split('_')[:3]))


def get_site(x):
    return(x.split('_')[1])


def gather_covariates(covs):
    """covs a list of covariates. returns an n samples by m features matrix"""
    covs = np.vstack(covs).T
    clf = Imputer(verbose=1)
    covs = clf.fit_transform(covs)
    return(covs)


def choens_d(x, y):
    """estimation of effect size: uses pooled standard deviation"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    d = (np.mean(x)-np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return(d)


def covary(db, col, y):
    """
    calculates multiple linear regression between y
    (matrix of standardized variables), and X, which is the defined
    column 'col' in the pandas dataframe 'db'.

    takes linear regression terms m and b, and subtracts that
    prediciton from X, which is returned in place.
    """
    scaler_X = StandardScaler()
    scaler_X.fit(db[col].reshape(-1, 1))
    X = scaler_X.transform(db[col].reshape(-1, 1))
    lm = LinearRegression(normalize=False)
    lm.fit(X, y)

    # regress each linear fit from data
    for m, b in zip(lm.coef_, lm.intercept_):
        X -= m*X + b

    db[col] = X

    return(db)


def demean_by_site(db, col):
    for sitename in ['CMH', 'ZHH', 'MRC']:
        idx = merged['site'] == sitename
        db[col].loc[idx] -= np.mean(db[col].loc[idx])

    return(db)


# subcortical volumes and cortical thickness of main ROIs
# covary for age, and for TBV (when doing subcortical volumes only), and
# parental education (if you have it).

assets = '/projects/jviviano/data/xbrain/assets'
db = pd.read_csv('xbrain_database_with_biotypes.csv')
enigma_lut = os.path.join(assets, 'enigma_lut.csv')

# used for pretty printing enigma ROI names later on
f = open(enigma_lut, 'rb')
f = f.readlines()
enigma_dict = {}
for i, line in enumerate(f):
    if i > 0:
        enigma_dict[line.split(',')[0]] = line.split(',')[1].strip()

# used for cortical surface
surf_dict = {'L_bankssts': 'Left Bank of Superior Temporal Sulcus',
             'L_caudalanteriorcingulate': 'Left Caudal Anterior Cingulate',
             'L_caudalmiddlefrontal': 'Left Caudal Middle Frontal Gyrus',
             'L_cuneus': 'Left Cuneus',
             'L_entorhinal': 'Left Entorhinal Cortex',
             'L_fusiform': 'Left Fusiform Gyrus',
             'L_inferiorparietal': 'Left Inferior Parietal Cortex',
             'L_inferiortemporal': 'Left Inferior Temporal Cortex',
             'L_isthmuscingulate': 'Left Isthmus Cingulate Cortex',
             'L_lateraloccipital': 'Left Lateral Occipital Cortex',
             'L_lateralorbitofrontal': 'Left Lateral Orbitofrontal Cortex',
             'L_lingual': 'Left Lingual Gyrus',
             'L_medialorbitofrontal': 'Left Medial Orbitofrontal Cortex',
             'L_middletemporal': 'Left Middle Temporal Cortex',
             'L_parahippocampal': 'Left Parahippocampal Gyrus',
             'L_paracentral': 'Left Paracentral Sulcus',
             'L_parsopercularis': 'Left Pars Opercularis',
             'L_parsorbitalis': 'Left Pars Orbitalis',
             'L_parstriangularis': 'Left Pars Triangularis',
             'L_pericalcarine': 'Left Pericalcarine',
             'L_postcentral': 'Left Postcentral Sulcus',
             'L_posteriorcingulate': 'Left Posterior Cingulate',
             'L_precentral': 'Left Precentral Sulcus',
             'L_precuneus': 'Left Precuneus',
             'L_rostralanteriorcingulate': 'Left Rostral Anterior Cingulate',
             'L_rostralmiddlefrontal': 'Left Rostral Middle Frontal Gyrus',
             'L_superiorfrontal': 'Left Superior Frontal Gyrus',
             'L_superiorparietal': 'Left Superior Parietal Gyrus',
             'L_superiortemporal': 'Left Superior Temporal Gyrus',
             'L_supramarginal': 'Left Supramarginal Gyrus',
             'L_frontalpole': 'Left Frontal Pole',
             'L_temporalpole': 'Left Temporal Pole',
             'L_transversetemporal': 'Left Transverse Temporal Cortex',
             'L_insula': 'Left Insula',
             'R_bankssts': 'Right Bank of Superior Temporal Sulcus',
             'R_caudalanteriorcingulate': 'Right Caudal Anterior Cingulate',
             'R_caudalmiddlefrontal': 'Right Caudal Middle Frontal Gyrus',
             'R_cuneus': 'Right Cuneus',
             'R_entorhinal': 'Right Entorhinal Cortex',
             'R_fusiform': 'Right Fusiform Gyrus',
             'R_inferiorparietal': 'Right Inferior Parietal Cortex',
             'R_inferiortemporal': 'Right Inferior Temporal Cortex',
             'R_isthmuscingulate': 'Right Isthmus Cingulate Cortex',
             'R_lateraloccipital': 'Right Lateral Occipital Cortex',
             'R_lateralorbitofrontal': 'Right Lateral Orbitofrontal Cortex',
             'R_lingual': 'Right Lingual Gyrus',
             'R_medialorbitofrontal': 'Right Medial Orbitofrontal Cortex',
             'R_middletemporal': 'Right Middle Temporal Cortex',
             'R_parahippocampal': 'Right Parahippocampal Gyrus',
             'R_paracentral': 'Right Paracentral Sulcus',
             'R_parsopercularis': 'Right Pars Opercularis',
             'R_parsorbitalis': 'Right Pars Orbitalis',
             'R_parstriangularis': 'Right Pars Triangularis',
             'R_pericalcarine': 'Right Pericalcarine',
             'R_postcentral': 'Right Postcentral Sulcus',
             'R_posteriorcingulate': 'Right Posterior Cingulate',
             'R_precentral': 'Right Precentral Sulcus',
             'R_precuneus': 'Right Precuneus',
             'R_rostralanteriorcingulate': 'Right Rostral Anterior Cingulate',
             'R_rostralmiddlefrontal': 'Right Rostral Middle Frontal Gyrus',
             'R_superiorfrontal': 'Right Superior Frontal Gyrus',
             'R_superiorparietal': 'Right Superior Parietal Gyrus',
             'R_superiortemporal': 'Right Superior Temporal Gyrus',
             'R_supramarginal': 'Right Supramarginal Gyrus',
             'R_frontalpole': 'Right Frontal Pole',
             'R_temporalpole': 'Right Temporal Pole',
             'R_transversetemporal': 'Right Transverse Temporal Cortex',
             'R_insula': 'Right Insula'}

# used for subcortical volumes
sub_dict = {' LLatVent': 'Left Lateral Ventricle',
            'RLatVent': 'Right Lateral Ventricle',
            'Lthal': 'Left Thalamus',
            'Rthal': 'Right Thalamus',
            'Lcaud': 'Left Caudate',
            'Rcaud': 'Right Caudate',
            'Lput': 'Left Putamin',
            'Rput': 'Right Putamin',
            '  Lpal': 'Left Globus Pallidus',
            'Rpal': 'Right Globus Pallidus',
            'Lhippo': 'Left Hippocampus',
            'Rhippo': 'Right Hippocampus',
            'Lamyg': 'Left Amygdala',
            'Ramyg': 'Right Amygdala',
            'Laccumb': 'Left Nucleus Accumbens',
            'Raccumb': 'Right Nucleus Accumbens'}




# thickness analysis
cols = np.array(['L_bankssts_thickavg', 'L_caudalanteriorcingulate_thickavg', 'L_caudalmiddlefrontal_thickavg', 'L_cuneus_thickavg', 'L_entorhinal_thickavg', 'L_fusiform_thickavg', 'L_inferiorparietal_thickavg', 'L_inferiortemporal_thickavg', 'L_isthmuscingulate_thickavg', 'L_lateraloccipital_thickavg', 'L_lateralorbitofrontal_thickavg', 'L_lingual_thickavg', 'L_medialorbitofrontal_thickavg', 'L_middletemporal_thickavg', 'L_parahippocampal_thickavg', 'L_paracentral_thickavg', 'L_parsopercularis_thickavg', 'L_parsorbitalis_thickavg', 'L_parstriangularis_thickavg', 'L_pericalcarine_thickavg', 'L_postcentral_thickavg', 'L_posteriorcingulate_thickavg', 'L_precentral_thickavg', 'L_precuneus_thickavg', 'L_rostralanteriorcingulate_thickavg', 'L_rostralmiddlefrontal_thickavg', 'L_superiorfrontal_thickavg', 'L_superiorparietal_thickavg', 'L_superiortemporal_thickavg', 'L_supramarginal_thickavg', 'L_frontalpole_thickavg', 'L_temporalpole_thickavg', 'L_transversetemporal_thickavg', 'L_insula_thickavg', 'R_bankssts_thickavg', 'R_caudalanteriorcingulate_thickavg', 'R_caudalmiddlefrontal_thickavg', 'R_cuneus_thickavg', 'R_entorhinal_thickavg', 'R_fusiform_thickavg', 'R_inferiorparietal_thickavg', 'R_inferiortemporal_thickavg', 'R_isthmuscingulate_thickavg', 'R_lateraloccipital_thickavg', 'R_lateralorbitofrontal_thickavg', 'R_lingual_thickavg', 'R_medialorbitofrontal_thickavg', 'R_middletemporal_thickavg', 'R_parahippocampal_thickavg', 'R_paracentral_thickavg', 'R_parsopercularis_thickavg', 'R_parsorbitalis_thickavg', 'R_parstriangularis_thickavg', 'R_pericalcarine_thickavg', 'R_postcentral_thickavg', 'R_posteriorcingulate_thickavg', 'R_precentral_thickavg', 'R_precuneus_thickavg', 'R_rostralanteriorcingulate_thickavg', 'R_rostralmiddlefrontal_thickavg', 'R_superiorfrontal_thickavg', 'R_superiorparietal_thickavg', 'R_superiortemporal_thickavg', 'R_supramarginal_thickavg', 'R_frontalpole_thickavg', 'R_temporalpole_thickavg', 'R_transversetemporal_thickavg', 'R_insula_thickavg'])

stats = pd.read_csv(os.path.join(assets, 'all_thickness.csv'))
stats['ID'] = stats['SubjID'].apply(split_name)
merged = db.merge(stats, on='ID')
# get unique site fields
merged['site'] = merged['ID'].apply(get_site)

covs = gather_covariates([
    merged['Education'], merged['Age at Enrollment']])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(covs)

for col in cols:
    merged = covary(merged, col, y)

ts = np.zeros((len(cols), 2)) # ts, ps (per column)
ds = np.zeros(len(cols)) # choens d
for i, col in enumerate(cols):

    if flip:
        idx_0 = np.where(merged['biotype'] == 1)[0]
        idx_1 = np.where(merged['biotype'] == 0)[0]
    else:
        idx_0 = np.where(merged['biotype'] == 0)[0]
        idx_1 = np.where(merged['biotype'] == 1)[0]

    ttest = ttest_ind(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    ts[i, 0] = ttest[0] # tvals
    ts[i, 1] = ttest[1] # pvals
    ds[i] = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])

corrected = multipletests(np.ravel(ts[:, 1]), alpha=0.05, method='fdr_bh')
H1 = np.where(corrected[0])[0]

f = open('cortical_thickness.csv', 'wb')
f.write('roi,t value,p value,choens d,significant\n')
for i, col in enumerate(cols):
    f.write('{},{},{},{},{}\n'.format(surf_dict['_'.join(col.split('_')[:2])], np.around(ts[i, 0], decimals=2), np.around(ts[i, 1], decimals=2), np.around(ds[i], decimals=2), str(int(corrected[0][i]))))
f.close()
print('cortical thickness: {}/{} tests reject H0'.format(len(H1), len(cols)))



# surface space analysis
cols = np.array(['L_bankssts_surfavg', 'L_caudalanteriorcingulate_surfavg', 'L_caudalmiddlefrontal_surfavg', 'L_cuneus_surfavg', 'L_entorhinal_surfavg', 'L_fusiform_surfavg', 'L_inferiorparietal_surfavg', 'L_inferiortemporal_surfavg', 'L_isthmuscingulate_surfavg', 'L_lateraloccipital_surfavg', 'L_lateralorbitofrontal_surfavg', 'L_lingual_surfavg', 'L_medialorbitofrontal_surfavg', 'L_middletemporal_surfavg', 'L_parahippocampal_surfavg', 'L_paracentral_surfavg', 'L_parsopercularis_surfavg', 'L_parsorbitalis_surfavg', 'L_parstriangularis_surfavg', 'L_pericalcarine_surfavg', 'L_postcentral_surfavg', 'L_posteriorcingulate_surfavg', 'L_precentral_surfavg', 'L_precuneus_surfavg', 'L_rostralanteriorcingulate_surfavg', 'L_rostralmiddlefrontal_surfavg', 'L_superiorfrontal_surfavg', 'L_superiorparietal_surfavg', 'L_superiortemporal_surfavg', 'L_supramarginal_surfavg', 'L_frontalpole_surfavg', 'L_temporalpole_surfavg', 'L_transversetemporal_surfavg', 'L_insula_surfavg', 'R_bankssts_surfavg', 'R_caudalanteriorcingulate_surfavg', 'R_caudalmiddlefrontal_surfavg', 'R_cuneus_surfavg', 'R_entorhinal_surfavg', 'R_fusiform_surfavg', 'R_inferiorparietal_surfavg', 'R_inferiortemporal_surfavg', 'R_isthmuscingulate_surfavg', 'R_lateraloccipital_surfavg', 'R_lateralorbitofrontal_surfavg', 'R_lingual_surfavg', 'R_medialorbitofrontal_surfavg', 'R_middletemporal_surfavg', 'R_parahippocampal_surfavg', 'R_paracentral_surfavg', 'R_parsopercularis_surfavg', 'R_parsorbitalis_surfavg', 'R_parstriangularis_surfavg', 'R_pericalcarine_surfavg', 'R_postcentral_surfavg', 'R_posteriorcingulate_surfavg', 'R_precentral_surfavg', 'R_precuneus_surfavg', 'R_rostralanteriorcingulate_surfavg', 'R_rostralmiddlefrontal_surfavg', 'R_superiorfrontal_surfavg', 'R_superiorparietal_surfavg', 'R_superiortemporal_surfavg', 'R_supramarginal_surfavg', 'R_frontalpole_surfavg', 'R_temporalpole_surfavg', 'R_transversetemporal_surfavg', 'R_insula_surfavg'])

stats = pd.read_csv(os.path.join(assets, 'all_surfavg.csv'))
stats['ID'] = stats['SubjID'].apply(split_name)
merged = db.merge(stats, on='ID')
# get unique site fields
merged['site'] = merged['ID'].apply(get_site)

covs = gather_covariates([
    merged['Education'], merged['Age at Enrollment']])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(covs)

for col in cols:
    merged = covary(merged, col, y)

ts = np.zeros((len(cols), 2)) # ts, ps (per column)
ds = np.zeros(len(cols)) # choens d
for i, col in enumerate(cols):

    if flip:
        idx_0 = np.where(merged['biotype'] == 1)[0]
        idx_1 = np.where(merged['biotype'] == 0)[0]
    else:
        idx_0 = np.where(merged['biotype'] == 0)[0]
        idx_1 = np.where(merged['biotype'] == 1)[0]

    ttest = ttest_ind(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    ts[i, 0] = ttest[0] # tvals
    ts[i, 1] = ttest[1] # pvals
    ds[i] = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])

corrected = multipletests(np.ravel(ts[:, 1]), alpha=0.05, method='fdr_bh')
H1 = np.where(corrected[0])[0]

f = open('cortical_surface_space.csv', 'wb')
f.write('roi,t value,p value,choens d,significant\n')
for i, col in enumerate(cols):
    f.write('{},{},{},{},{}\n'.format(surf_dict['_'.join(col.split('_')[:2])], np.around(ts[i, 0], decimals=2), np.around(ts[i, 1], decimals=2), np.around(ds[i], decimals=2), str(int(corrected[0][i]))))
f.close()
print('cortical surface space: {}/{} tests reject H0'.format(len(H1), len(cols)))



# subcortical volumes analysis
cols = np.array([' LLatVent', 'RLatVent', 'Lthal', 'Rthal', 'Lcaud', 'Rcaud', 'Lput', 'Rput', '  Lpal', 'Rpal', 'Lhippo', 'Rhippo', 'Lamyg', 'Ramyg', 'Laccumb', 'Raccumb'])

stats = pd.read_csv(os.path.join(assets, 'all_volumes.csv'))
stats['ID'] = stats['SubjID'].apply(split_name)
merged = db.merge(stats, on='ID')
# get unique site fields
merged['site'] = merged['ID'].apply(get_site)

covs = gather_covariates([
    merged['Education'], merged['ICV'], merged['Age at Enrollment']])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(covs)

for col in cols:
    merged = covary(merged, col, y)

ts = np.zeros((len(cols), 2)) # ts, ps (per column)
ds = np.zeros(len(cols)) # choens d
for i, col in enumerate(cols):

    if flip:
        idx_0 = np.where(merged['biotype'] == 1)[0]
        idx_1 = np.where(merged['biotype'] == 0)[0]
    else:
        idx_0 = np.where(merged['biotype'] == 0)[0]
        idx_1 = np.where(merged['biotype'] == 1)[0]

    ttest = ttest_ind(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    ts[i, 0] = ttest[0] # tvals
    ts[i, 1] = ttest[1] # pvals
    ds[i] = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])

corrected = multipletests(np.ravel(ts[:, 1]), alpha=0.05, method='fdr_bh')
H1 = np.where(corrected[0])[0]

f = open('subcortical_volumes.csv', 'wb')
f.write('roi,t value,p value,choens d,significant\n')
for i, col in enumerate(cols):
    f.write('{},{},{},{},{}\n'.format(sub_dict[col], np.around(ts[i, 0], decimals=2), np.around(ts[i, 1], decimals=2), np.around(ds[i], decimals=2), str(int(corrected[0][i]))))
f.close()
print('subcortical volumes: {}/{} tests reject H0'.format(len(H1), len(cols)))



# FA analysis
cols = np.array(['ACR-L_FA', 'ACR-R_FA', 'ALIC-L_FA', 'ALIC-R_FA', 'BCC_FA', 'CC_FA', 'CGC-L_FA', 'CGC-R_FA', 'CGH-L_FA', 'CGH-R_FA', 'CR-L_FA', 'CR-R_FA', 'CST-L_FA', 'CST-R_FA', 'EC-L_FA', 'EC-R_FA', 'FX/ST-L_FA', 'FX/ST-R_FA', 'FXST_FA', 'GCC_FA', 'IC-L_FA', 'IC-R_FA', 'IFO-L_FA', 'IFO-R_FA', 'PCR-L_FA', 'PCR-R_FA', 'PLIC-L_FA', 'PLIC-R_FA', 'PTR-L_FA', 'PTR-R_FA', 'RLIC-L_FA', 'RLIC-R_FA', 'SCC_FA', 'SCR-L_FA', 'SCR-R_FA', 'SFO-L_FA', 'SFO-R_FA', 'SLF-L_FA', 'SLF-R_FA', 'SS-L_FA', 'SS-R_FA', 'UNC-L_FA', 'UNC-R_FA'])

stats = pd.read_csv(os.path.join(assets, 'all_fa.csv'))
stats['ID'] = stats['id'].apply(split_name)
merged = db.merge(stats, on='ID')
# get unique site fields
merged['site'] = merged['ID'].apply(get_site)

covs = gather_covariates([
    merged['Education'], merged['Age at Enrollment']])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(covs)

for col in cols:
    merged = demean_by_site(merged, col)
    merged = covary(merged, col, y)

ts = np.zeros((len(cols), 2)) # ts, ps (per column)
ds = np.zeros(len(cols)) # choens d
for i, col in enumerate(cols):

    if flip:
        idx_0 = np.where(merged['biotype'] == 1)[0]
        idx_1 = np.where(merged['biotype'] == 0)[0]
    else:
        idx_0 = np.where(merged['biotype'] == 0)[0]
        idx_1 = np.where(merged['biotype'] == 1)[0]

    ttest = ttest_ind(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    ts[i, 0] = ttest[0] # tvals
    ts[i, 1] = ttest[1] # pvals
    ds[i] = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])

corrected = multipletests(np.ravel(ts[:, 1]), alpha=0.05, method='fdr_bh')
H1 = np.where(corrected[0])[0]

f = open('dti_fa.csv', 'wb')
f.write('roi,t value,p value,choens d,significant\n')
for i, col in enumerate(cols):
    f.write('{},{},{},{},{}\n'.format(enigma_dict[col.split('_')[0]], np.around(ts[i, 0], decimals=2), np.around(ts[i, 1], decimals=2), np.around(ds[i], decimals=2), str(int(corrected[0][i]))))
f.close()
print('dti fa: {}/{} tests reject H0'.format(len(H1), len(cols)))




# MD analysis
cols = np.array(['ACR-L_MD', 'ACR-R_MD', 'ALIC-L_MD', 'ALIC-R_MD', 'BCC_MD', 'CC_MD', 'CGC-L_MD', 'CGC-R_MD', 'CGH-L_MD', 'CGH-R_MD', 'CR-L_MD', 'CR-R_MD', 'CST-L_MD', 'CST-R_MD', 'EC-L_MD', 'EC-R_MD', 'FX/ST-L_MD', 'FX/ST-R_MD', 'FXST_MD', 'GCC_MD', 'IC-L_MD', 'IC-R_MD', 'IFO-L_MD', 'IFO-R_MD', 'PCR-L_MD', 'PCR-R_MD', 'PLIC-L_MD', 'PLIC-R_MD', 'PTR-L_MD', 'PTR-R_MD', 'RLIC-L_MD', 'RLIC-R_MD', 'SCC_MD', 'SCR-L_MD', 'SCR-R_MD', 'SFO-L_MD', 'SFO-R_MD', 'SLF-L_MD', 'SLF-R_MD', 'SS-L_MD', 'SS-R_MD', 'UNC-L_MD', 'UNC-R_MD'])

stats = pd.read_csv(os.path.join(assets, 'all_md.csv'))
stats['ID'] = stats['id'].apply(split_name)
merged = db.merge(stats, on='ID')
# get unique site fields
merged['site'] = merged['ID'].apply(get_site)

covs = gather_covariates([
    merged['Education'], merged['Age at Enrollment']])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(covs)

for col in cols:
    merged = demean_by_site(merged, col)
    merged = covary(merged, col, y)

ts = np.zeros((len(cols), 2)) # ts, ps (per column)
ds = np.zeros(len(cols)) # choens d
for i, col in enumerate(cols):

    if flip:
        idx_0 = np.where(merged['biotype'] == 1)[0]
        idx_1 = np.where(merged['biotype'] == 0)[0]
    else:
        idx_0 = np.where(merged['biotype'] == 0)[0]
        idx_1 = np.where(merged['biotype'] == 1)[0]

    ttest = ttest_ind(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    d = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])
    ts[i, 0] = ttest[0] # tvals
    ts[i, 1] = ttest[1] # pvals
    ds[i] = choens_d(merged[col].iloc[idx_0], merged[col].iloc[idx_1])

corrected = multipletests(np.ravel(ts[:, 1]), alpha=0.05, method='fdr_bh')
H1 = np.where(corrected[0])[0]

f = open('dti_md.csv', 'wb')
f.write('roi,t value,p value,choens d,significant\n')
for i, col in enumerate(cols):
    f.write('{},{},{},{},{}\n'.format(enigma_dict[col.split('_')[0]], np.around(ts[i, 0], decimals=2), np.around(ts[i, 1], decimals=2), np.around(ds[i], decimals=2),  str(int(corrected[0][i]))))
f.close()
print('dti md: {}/{} tests reject H0'.format(len(H1), len(cols)))


