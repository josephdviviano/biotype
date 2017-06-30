#!/usr/bin/env python
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import image
from nilearn import plotting
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import seaborn as sns
import glob
import sys

def make_3d(img, n, flip=False):
    """returns a 3d map from a 4d file of maps"""
    data = image.load_img(img)
    hdr = data.get_header()
    aff = data.get_affine()
    data = data.get_data()[:,:,:,n]
    if flip:
        data = data*-1

    data = nib.nifti1.Nifti1Image(data, aff, header=hdr)

    return(data)


def find_threshold(data, pct=75):
    """for all nonzero values in input statmap, find the pct percentile"""
    data = data.get_data()
    data = data[np.where(data)]
    try:
        threshold = np.percentile(np.abs(data), pct)
    except:
        threshold = -1000000
    return(threshold)


def calc_y_loadings(mdl):
    """plots relationship between each y component and the submitted y scores"""
    correlations = np.zeros((mdl['y'].shape[1], mdl['comps_y'].shape[1]))
    pvals = np.zeros((mdl['y'].shape[1], mdl['comps_y'].shape[1]))
    for i, score in enumerate(mdl['y'].T):
        for j, comp in enumerate(mdl['comps_y'].T):
            correlations[i, j] = pearsonr(score, comp)[0]
            pvals[i, j] = pearsonr(score, comp)[1]
    return(correlations, pvals)


def bonferonni(ps):
    """returns a pass/fail mask for a matrix of pvals"""
    tests = reduce(lambda x, y: x*y, ps.shape)
    threshold = 0.05 / tests
    passed = ps < threshold
    return(passed)


n_comps = 3
try:
    flip = [sys.argv[1], sys.argv[2], sys.argv[3]]
except:
    flip = [False, False, False]


#cols = ['scog_er40_crt_columnqcrt_value_inv', 'Part1_TotalCorrect', 'Part2_TotalCorrect', 'Part3_TotalCorrect', 'RMET total', 'rad_total', 'np_domain_tscore_process_speed', 'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning', 'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps', 'np_domain_tscore_att_vigilance']
#names = ['Part1_TotalCorrect','Part2_TotalCorrect', 'Part3_TotalCorrect', 'RMET total', 'scog_er40_crt_columnqcrt_value_inv', 'rad_total', 'np_domain_tscore_process_speed', 'np_domain_tscore_att_vigilance', 'np_domain_tscore_work_mem', 'np_domain_tscore_verbal_learning', 'np_domain_tscore_visual_learning', 'np_domain_tscore_reasoning_ps']
names = ['Tasit 1', 'Tasit 2', 'Tasit 3', 'RMET', 'ER40 RT (inv)', 'RAD', 'Processing Speed', 'Attention/Vigilance', 'Working Memory', 'Verbal Learning', 'Visual Learning', 'Reasoning']


mdl = np.load('xbrain_biotype.npz')
db = pd.read_csv('xbrain_database_with_biotypes.csv')
loadings = glob.glob('xbrain_biotype_X_*_loadings_*.nii.gz')[0] # assumes only one

y_loadings, y_loadings_p = calc_y_loadings(mdl)
y_passed = bonferonni(y_loadings_p)

# y
for comp in range(n_comps):
    if flip[comp] == 'flip':
        y_loadings[:, comp] = y_loadings[:, comp] * -1

plt.imshow(y_loadings, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
plt.yticks(range(len(mdl['y_names'])), names)
plt.colorbar()
plt.savefig('cca_y_thesholded.pdf')
plt.close()

# X
comps = mdl['comps_X']

for comp in range(n_comps):
    if flip[comp] == 'flip':
        data = make_3d(loadings, comp+comp+1, flip=True)
        comps[:, comp] = comps[:, comp] * -1
    else:
        data = make_3d(loadings, comp+comp)
    threshold = find_threshold(data)
    plotting.plot_glass_brain(data, threshold=threshold, colorbar=True, plot_abs=False, display_mode='lzry')
    plotting.matplotlib.pyplot.savefig('cca_X_thresholded_comp_{}.pdf'.format(comp+1))


sns.clustermap(comps, method='ward', metric='euclidean', col_cluster=False)
sns.plt.savefig('cca_X_clusters.pdf')
sns.plt.close()

f = open('cca_corrs.csv', 'wb')
for comp in range(n_comps):
    f.write('{}\n'.format(mdl['cancorrs'][comp]))
f.close()


# plot each score seperately
#db = pd.DataFrame()
#db['id'] = merged['ID']
#db['biotype'] = merged['biotype']
#db['cluster'] = merged['cluster']

#for col in cols:
#    print('{} nans: {}'.format(col, np.sum(np.isnan(merged[col]))))

#for i, col in enumerate(cols):
#    db[col] = zscore(merged[col])

#db = pd.melt(db, id_vars=['id', 'biotype'], value_vars=cols)
#db = pd.melt(db, id_vars=['id', 'biotype', 'cluster'], value_vars=cols)

