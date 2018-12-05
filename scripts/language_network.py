"""
This script grab IBC RSVP language data and extract language related regions
out of that.

Need to install the ibc_public package
git clone git@github.com:hbp-brain-charting/public_analysis_code.git

Need functions in conjunction.py file
runfile('./conjunction.py')

Author: Bertrand Thirion, Herve Lemaitre, 2018
"""
import os
from joblib import Memory
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.input_data import NiftiMasker
import nibabel as nib
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from ibc_public.utils_data import (data_parser, SMOOTH_DERIVATIVES, SUBJECTS, LABELS, CONTRASTS, CONDITIONS, THREE_MM, DERIVATIVES)
import ibc_public

# cache = '/neurospin/tmp/bthirion'
cache = '/volatile/lemaitre/IBC'

# caching
mem = Memory(cachedir=cache, verbose=0, bytes_limit=4.e9)

# output directory
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# mask of grey matter across subjects
package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

#  select task fMRI
subject_list = SUBJECTS
task_list = ['rsvp_language']
df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=task_list)
df = df[df.acquisition == 'ffx']
conditions = df[df.modality == 'bold'].contrast.unique()
n_conditions = len(conditions)

# function for conjunction
def conjunction_img(imgs, masker, percentile=50, threshold=3.1,
                    height_control='none'):
    """ generate conjunction statsitics of the contrat images in dataframe

    Parameters
    ----------
    imgs: list of niimg,
          Images on which conjunction analysis is run
    masker: niftimasker instance,
            to define the spatial context of the analysis
    precentile:  float,
        Percentile used for the conjunction analysis
    """
    from conjunction import _conjunction_inference_from_z_values
    from nistats.thresholding import map_threshold
    Z = masker.transform(imgs).T
    pos_conj = _conjunction_inference_from_z_values(Z, percentile * .01)
    neg_conj = _conjunction_inference_from_z_values(-Z, percentile * .01)
    conj = pos_conj
    conj[conj < 0] = 0
    conj[neg_conj > 0] = - neg_conj[neg_conj > 0]
    conj_img = masker.inverse_transform(conj)
    conj_img, threshold_ = map_threshold(
        conj_img, level=threshold, height_control=height_control,
        cluster_threshold=100)
    conj = conj * (np.abs(conj) > threshold)
    return conj_img, threshold_, conj

# define parameters
contrast = 'sentence-jabberwocky'
imgs = df[df.contrast == contrast].path.values
percentile = 50
zmaps = []
masker = NiftiMasker(mask_img=mask_gm).fit()

# conjunctions analysis
conj_img, threshold_, conj = conjunction_img(
    imgs, masker, percentile=percentile, threshold=.05,
    height_control='fdr')

# display and save T map
fname = os.path.join(write_dir, 'conj50_%s.png' % contrast)
plotting.plot_glass_brain(conj_img, vmax=10, title=contrast)
plotting.plot_glass_brain(conj_img, vmax=10, title=contrast,
                          output_file=fname)
fname = os.path.join(write_dir, 'conj50_%s.nii.gz' % contrast)
conj_img.to_filename(fname)

# display and save label map
values = conj_img.get_data()
local_maxi = peak_local_max(values, min_distance=5, indices=False)
markers = ndi.label(local_maxi)[0]
labels = watershed(-values, markers, mask=values > 0)
label_img = nib.Nifti1Image(labels, affine=conj_img.get_affine())
plotting.plot_glass_brain(label_img, title='extracted regions')
fname = os.path.join(write_dir, 'language_rois.png')
plotting.plot_glass_brain(label_img, title='extracted regions',
                          output_file=fname)
fname = os.path.join(write_dir, 'language_rois.nii.gz')
label_img.to_filename(fname)

# close display
plt.show(block=False)
