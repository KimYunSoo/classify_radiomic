# Automated Differentiation of Atypical Parkinsonian Syndromes Using Brain Iron Patterns in Susceptibility Weighted Imaging
An implementation of 3D brain MRI segmentation, radiomic feature extraction and selection, and disease classification

## Publication
Kim Yun Soo, Jae-Hyeok Lee, and Jin Kyu Gahm. "Automated Differentiation of Atypical Parkinsonian Syndromes Using Brain Iron Patterns in Susceptibility Weighted Imaging." Diagnostics 12.3 (2022): 637.

Paper Link : https://doi.org/10.3390/diagnostics12030637

Abstract
In recent studies, iron overload has been reported in atypical parkinsonian syndromes. The topographic patterns of iron distribution in deep brain nuclei vary by each subtype of parkinsonian syndrome, which is affected by underlying disease pathologies. In this study, we developed a novel framework that automatically analyzes the disease-specific patterns of iron accumulation using susceptibility weighted imaging (SWI). We constructed various machine learning models that can classify diseases using radiomic features extracted from SWI, representing distinctive iron distribution patterns for each disorder. Since radiomic features are sensitive to the region of interest, we used a combination of T1-weighted MRI and SWI to improve the segmentation of deep brain nuclei. Radiomics was applied to SWI from 34 patients with a parkinsonian variant of multiple system atrophy, 21 patients with cerebellar variant multiple system atrophy, 17 patients with progressive supranuclear palsy, and 56 patients with Parkinson’s disease. The machine learning classifiers that learn the radiomic features extracted from iron-reflected segmentation results produced an average area under receiver operating characteristic curve (AUC) of 0.8607 on the training data and 0.8489 on the testing data, which is superior to the conventional classifier with segmentation using only T1-weighted images. Our radiomic model based on the hybrid images is a promising tool for automatically differentiating atypical parkinsonian syndromes.

## Requirements
+ Package Required: numpy, scipy, sklearn, skfeature, pandas, nibabel, ants, SimpleITK, radiomics

## Running
+ git clone https://github.com/KimYunSoo/classify_radiomic.git


+ Run following commmand to make Hybrid Contrast Image by combining T1w and SWI

  python make_hc.py T1w.nii.gz SWI.nii.gz initial_mask.nii.gz save_HC.nii.gz
 

+ Run following commmand to extract radiomic feature (Multiple parameter files can be used to change neighbor voxel distance)

  python extract_radiomic_3d.py volume.nii.gz left_putamen_mask.nii.gz right_putamen_mask.nii.gz parameter1.yaml parameter2.yaml parameter3.yaml feature_table.csv
  

+ Run following commmand to build and test machine learning classification (If you classify MSA-P, MSA-C, PD, and PSP)

  python radiomic_clf.py MSAP_hc.csv MSAP_fs.csv MSAC_hc.csv MSAC_fs.csv PD_hc.csv PD_fs.csv PSP_hc.csv PSP_fs.csv your_save_path train_iter_num test_iter_num
  

## Result Visualization
+ Deep Gray Matter (DGM) axial slice in T1w, SWI and Hybrid Contrast (HC) images.

![image](https://user-images.githubusercontent.com/45022470/153359259-28cff295-c955-42f8-88c5-79ea179fe21d.png)


+ Putamen mask of segmentation result of using only T1w image (FreeSurfer) and using T1w and SWI (proposed method) with SWI overlayed.

![image](https://user-images.githubusercontent.com/45022470/153359381-cea01a75-07ec-4715-9f5b-4d404cef3882.png)




## License
GNU General Public License 3.0 License
