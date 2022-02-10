import sys
import nibabel as nib
import numpy as np

t1_nii=nib.load(sys.argv[1])
t1=t1_nii.get_fdata()
swi_nii=nib.load(sys.argv[2])
swi=swi_nii.get_fdata()
# firstseg= initial (T1w only segmentation mask)
firstseg=nib.load(sys.argv[3]).get_fdata()

# mean value of MNI152 thalamus, putamen, pallidus
mni_lthal_mean=6132.8704143573505
mni_lputa_mean=6448.697039050207
mni_lpall_mean=6842.335451080051

t1_lputa=[]
swi_lputa=[]
lputa_coord = np.asarray(list(zip(np.where(firstseg==12)[0], np.where(firstseg==12)[1], np.where(firstseg==12)[2])))
for coord in lputa_coord:
    t1_lputa.append(t1[coord[0]][coord[1]][coord[2]])
    swi_lputa.append(swi[coord[0]][coord[1]][coord[2]])

t1_lpall=[]
swi_lpall=[]
lpall_coord = np.asarray(list(zip(np.where(firstseg==13)[0], np.where(firstseg==13)[1], np.where(firstseg==13)[2])))
for coord in lpall_coord:
    t1_lpall.append(t1[coord[0]][coord[1]][coord[2]])
    swi_lpall.append(swi[coord[0]][coord[1]][coord[2]])

# print ('lputa_mean: t1 ', np.mean(t1_lputa), ' swi ', np.mean(swi_lputa), ' mni ', mni_lputa_mean)
# print ('lpall_mean: t1 ', np.mean(t1_lpall), ' swi ', np.mean(swi_lpall), ' mni ', mni_lpall_mean)

A=np.array([[np.mean(t1_lputa), np.mean(swi_lputa)],[np.mean(t1_lpall), np.mean(swi_lpall)]])
B=np.array([mni_lputa_mean, mni_lpall_mean])
C=np.linalg.solve(A,B)
# print ('weight: ', C)
hc= C[0] * t1  +  C[1] * swi

hc_img = nib.Nifti1Image(hc, t1_nii.affine, t1_nii.header)
nib.save(hc_img, sys.argv[4])
