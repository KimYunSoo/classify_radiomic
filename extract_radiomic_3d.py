import os
import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase
from sklearn import svm
import csv
import sys
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, getTestCase, gldm, ngtdm
import radiomics
import numpy as np

image = sitk.ReadImage(sys.argv[1])
mask_left = sitk.ReadImage(sys.argv[2])
mask_right = sitk.ReadImage(sys.argv[3])
params = sys.argv[4]
params4 = sys.argv[5]
params7 = sys.argv[6]
extractor = featureextractor.RadiomicsFeatureExtractor(params)
f=open(sys.argv[7],'a')
wr=csv.writer(f)

left_bb, left_correctedMask = imageoperations.checkMask(image, mask_left, label=1)
if left_correctedMask is not None:
    mask_left = left_correctedMask
left_croppedImage, left_croppedMask = imageoperations.cropToTumorMask(image, mask_left, left_bb)
right_bb, right_correctedMask = imageoperations.checkMask(image, mask_right, label=1)
if right_correctedMask is not None:
    mask_right = right_correctedMask
right_croppedImage, right_croppedMask = imageoperations.cropToTumorMask(image, mask_right, right_bb)

lputa_result = extractor.execute(left_croppedImage, left_croppedMask, label=1)
lputa_result2 = lputa_result.copy()
rputa_result = extractor.execute(right_croppedImage, right_croppedMask, label=1)
rputa_result2 = rputa_result.copy()

extractor4 = featureextractor.RadiomicsFeatureExtractor(params4)  
lputa_result4 = extractor4.execute(left_croppedImage, left_croppedMask, label=1)
for (key, val) in six.iteritems(lputa_result4):
    if 'original_glcm_' in key:
        lputa_result2[key+'4']=val
    elif 'original_ngtdm_' in key:
        lputa_result2[key+'4']=val
rputa_result4 = extractor4.execute(right_croppedImage, right_croppedMask, label=1)
for (key, val) in six.iteritems(rputa_result4):
    if 'original_glcm_' in key:
        rputa_result2[key+'4']=val
    elif 'original_ngtdm_' in key:
        rputa_result2[key+'4']=val

extractor7 = featureextractor.RadiomicsFeatureExtractor(params7)  
lputa_result7 = extractor7.execute(left_croppedImage, left_croppedMask, label=1)
for (key, val) in six.iteritems(lputa_result7):
    if 'original_glcm_' in key:
        lputa_result2[key+'7']=val
    elif 'original_ngtdm_' in key:
        lputa_result2[key+'7']=val
rputa_result7 = extractor7.execute(right_croppedImage, right_croppedMask, label=1)
for (key, val) in six.iteritems(rputa_result7):
    if 'original_glcm_' in key:
        rputa_result2[key+'7']=val
    elif 'original_ngtdm_' in key:
        rputa_result2[key+'7']=val

lputa_result2['leftRight']=12
lputa_result2['subject_number']=sys.argv[2]
rputa_result2['leftRight']=51
rputa_result2['subject_number']=sys.argv[2]
drop_names=['diagnostics_Versions_PyRadiomics','diagnostics_Versions_Numpy','diagnostics_Versions_SimpleITK','diagnostics_Versions_PyWavelet','diagnostics_Versions_Python','diagnostics_Configuration_Settings','diagnostics_Configuration_EnabledImageTypes','diagnostics_Image-original_Hash','diagnostics_Image-original_Dimensionality','diagnostics_Image-original_Spacing','diagnostics_Image-original_Size','diagnostics_Mask-original_Hash','diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size','diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex','diagnostics_Mask-original_CenterOfMass']
for name in drop_names:
    del(lputa_result2[name])
    del(rputa_result2[name])
wr.writerow(lputa_result2.keys())
wr.writerow(lputa_result2.values())
wr.writerow(rputa_result2.values())

f.close()
