import sys
import os
import nibabel as nib
import numpy as np

t1_nii=nib.load(str(sys.argv[1]))
t1=t1_nii.get_fdata()
aseg_nii = nib.load(str(sys.argv[2]))
aseg = aseg_nii.get_fdata()

aseg_lwm_coord = np.asarray(list(zip(np.where(aseg==2)[0], np.where(aseg==2)[1], np.where(aseg==2)[2])))
aseg_lwm_mean=0
for i in range(0, aseg_lwm_coord.shape[0]):
    aseg_lwm_mean += t1[aseg_lwm_coord[i][0], aseg_lwm_coord[i][1], aseg_lwm_coord[i][2]]
aseg_lwm_mean=aseg_lwm_mean/aseg_lwm_coord.shape[0]
print ('left wm mean:', aseg_lwm_mean)

aseg_rwm_coord = np.asarray(list(zip(np.where(aseg==41)[0], np.where(aseg==41)[1], np.where(aseg==41)[2])))
aseg_rwm_mean=0
for i in range(0, aseg_rwm_coord.shape[0]):
    aseg_rwm_mean += t1[aseg_rwm_coord[i][0], aseg_rwm_coord[i][1], aseg_rwm_coord[i][2]]
aseg_rwm_mean=aseg_rwm_mean/aseg_rwm_coord.shape[0]
print ('right wm mean:',aseg_rwm_mean)

total_wm_mean=(aseg_lwm_mean+aseg_rwm_mean)/2
print ('total wm mean:', total_wm_mean)

scale_factor = 110/total_wm_mean
scaled_t1 = t1 * scale_factor
scaled_t1_img = nib.Nifti1Image(scaled_t1, t1_nii.affine, t1_nii.header)
nib.save(scaled_t1_img, str(sys.argv[3]))


aseg_lwm_mean=0
for i in range(0, aseg_lwm_coord.shape[0]):
    aseg_lwm_mean += scaled_t1[aseg_lwm_coord[i][0], aseg_lwm_coord[i][1], aseg_lwm_coord[i][2]]
aseg_lwm_mean=aseg_lwm_mean/aseg_lwm_coord.shape[0]
print ('scaled_left wm mean:', aseg_lwm_mean)

aseg_rwm_mean=0
for i in range(0, aseg_rwm_coord.shape[0]):
    aseg_rwm_mean += scaled_t1[aseg_rwm_coord[i][0], aseg_rwm_coord[i][1], aseg_rwm_coord[i][2]]
aseg_rwm_mean=aseg_rwm_mean/aseg_rwm_coord.shape[0]
print ('scaled_right wm mean:',aseg_rwm_mean)

total_wm_mean=(aseg_lwm_mean+aseg_rwm_mean)/2
print ('scaled_total wm mean:', total_wm_mean)
