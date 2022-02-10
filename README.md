# Automated Differentiation of Atypical Parkinsonian Syndromes Using Brain Iron Patterns in Susceptibility Weighted Imaging
An implementation of 3D brain MRI segmentation, radiomic feature extraction and selection, and disease classification

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
