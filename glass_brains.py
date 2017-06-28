#!/usr/bin/env python

import nibabel as nib
from nilearn import image
from nilearn import plotting
import numpy as np

def make_3d(img, n):
    """returns a 3d map from a 4d file of maps"""
    data = image.load_img(img)
    hdr = data.get_header()
    aff = data.get_affine()
    data = data.get_data()[:,:,:,n]
    data = nib.nifti1.Nifti1Image(data, aff, header=hdr)

    return(data)


def invert(img):
    """inverts all values contained in the data"""
    data = image.load_img(img)
    hdr = data.get_header()
    aff = data.get_affine()
    data = data.get_data()*-1
    data = nib.nifti1.Nifti1Image(data, aff, header=hdr)

    return(data)


def find_threshold(data, pct=75):
    """for all nonzero values in input statmap, find the pct percentile"""
    data = data.get_data()
    data = data[np.where(data)]
    threshold = np.percentile(np.abs(data), pct)
    return(threshold)

def plot(filename, flip=False):
    title = filename.split('.')[0]
    # 2 = sum(A) - sum(B), 3 = sum(abs(A)) - sum(abs(B))
    # 4 = sum(ispos(A)) - sum(ispos(B))
    # 5 = sum(isneg(A)) - sum(isneg(B))
    data = make_3d(filename, 4) # change this to get different sub-brick
    if flip:
        data = invert(data)
    threshold = find_threshold(data, 0)
    plotting.plot_glass_brain(data, threshold=threshold, colorbar=True, plot_abs=False, display_mode='lzry')
    plotting.matplotlib.pyplot.savefig('{}.pdf'.format(title))
    plotting.matplotlib.pyplot.savefig('{}.jpg'.format(title))

# if true, inverts colormap
filenames = {
    'xbrain_biotype_rois_ea.nii.gz': True,
    'xbrain_biotype_rois_imob.nii.gz': False,
    'xbrain_biotype_rois_imob-replication.nii.gz': True,
    'xbrain_biotype_rois_rest.nii.gz': True,
    'xbrain_biotype_rois_rest-replication.nii.gz': True
}
# non significant: xbrain_diagnosis_rois_imob-replication.nii.gz
   #'xbrain_diagnosis_rois_ea.nii.gz': False,
   #'xbrain_diagnosis_rois_imob.nii.gz': False,
   #'xbrain_diagnosis_rois_rest.nii.gz': False,
   #'xbrain_diagnosis_rois_rest-replication.nii.gz': False

for filename in filenames.keys():
    print(filename)
    plot(filename, flip=filenames[filename])
