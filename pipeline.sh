#!/bin/bash
export FREESURFER_HOME=/mnt/c/Users/Owner/Desktop/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

'''     setting for your own path
dir=
swi_256_dir=
fs_mask_dir=
fsl_mask_dir=
hc_dir=
hc_mask_dir=
'''

for subject in $dir/*
do
    '''    copy the raw data, freesurfer recon-all results    '''
    echo ${subject}
    mkdir $dir/${subject}_hc
    cp $dir/${subject}_t1_ras_256_n4.nii.gz  $dir/${subject}_hc/${subject}_t1_ras_256_n4.nii.gz
    cp $dir/${subject}_swi_ras_256.nii.gz  $dir/${subject}_hc/${subject}_swi_ras_256.nii.gz
    cp $dir/${subject}_first/${subject}_t1_ras_256_n4_all_fast_firstseg.nii.gz $dir/${subject}_hc/${subject}_t1_ras_256_n4_all_fast_firstseg.nii.gz
    cd $dir/${subject}_first


    '''    pre-processing    '''    
    # (freesurfer) mri_convert input.nii output.nii -c (-c = conform, make T1 image size 256 x 256 x 256, resolution 1mm x 1mm x 1mm)
    mri_convert "$subject" $dir/${subject}_256.nii.gz -c
    
    # (freesurfer) mri_convert aseg.mgz aseg.nii  (convert .mgz to .nii, aseg.mgz = freesurfer recon-all output segmentation mask label)
    mri_convert $dir/${subject}_hc/aseg.mgz $dir/${subject}_hc/aseg.nii.gz

    # python normalize.py input.nii aseg.nii output.nii  (normalize T1)
    python normalize.py   $dir/${subject}_hc/${subject}_t1_ras_256_n4.nii.gz  $dir/${subject}_hc/aseg.nii.gz $dir/${subject}_hc/${subject}_t1_ras_256_n4_norm.nii.gz

    # python n4itk.py intput.nii output.nii  (bias-correction T1, SWI)
    python n4itk.py $dir/${subject}_t1_ras_256.nii.gz  $dir/t1_nii_256_n4/${subject}_t1_ras_256_n4.nii.gz


    '''    T1-SWI registration, making hybrid contrast image (HC)    '''  
    # (freesurfer) mri_vol2vol --targ T1.nii --mov SWI.nii --reg output_transform_matrix.lta --o output_registered_SWI.nii (affine registration SWI to T1)
    mri_vol2vol --targ $dir/${subject}_t1_ras_256_n4.nii.gz --mov $dir/${subject}_swi_ras.nii.gz --reg $dir/${subject}_swi_to_t1.lta --o $dir/${subject}_swi_ras_256.nii.gz 
    
    # python make_hc.py T1w.nii.gz SWI.nii.gz freesurfer_mask_aseg.nii.gz output_HC.nii.gz (making hybrid contrast image, HC = w1 * T1 + w2 * SWI)
    python make_hc.py T1w.nii.gz SWI.nii.gz initial_mask.nii.gz save_HC.nii.gz


    '''    HC-MNI registration and segmentation    '''  
    # python ants.py image MNI152.nii HC.nii HC_in_MNI.nii (HC to MNI152 non-linear registration by ants, to disable FSL's registration method FLIRT and use ANTs instead)
    python ants_reg.py image  MNI152_T1_1mm.nii.gz  $dir/${subject}_hc/${subject}_hc_ras_256.nii.gz    $dir/${subject}_hc/${subject}_hc_in_mni_ants.nii.gz
   
    #(fsl) run_first_all -i HC_in_MNI152.nii -o fsl_segmentation_label_mask_result -a identity_matrix.mat (disable FIRST's registration FLIRT and proceed only segmentation for HC in MNI152 space)
    run_first_all -i $dir/${subject}_hc_in_mni_ants.nii.gz -o $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants -a identity_dim4.mat
    
    # python ants.py mask  HC.nii  MNI152.nii  output_inverse_warped_to_original_T1space_label  label_in_MNIspace_result_of_fsl_first (ants mask registration to get final label mask, HC segmentation label in MNI space --> inverse warp --> original T1 space)
    python ants_reg.py mask MNI152_T1_1mm.nii.gz $dir/${subject}_t1_ras_256_n4_norm.nii.gz $dir/${subject}_fs_aseg_in_mni_ants.nii.gz $dir/${subject}_fs_aseg.nii.gz 
    
        
    '''    making figures for paper publication    '''
    # (freesurfer) mri_binarize --i mask --o bin_mask --match 12 (make bin_mask, 12 for left putamen or 51 for right putamen)
    mri_binarize --i $fsl_mask_dir/${subject}/${subject}_first_all_fast_firstseg.nii.gz --o  $fsl_mask_dir/first_bin/${subject}_first_all_fast_firstseg_left.nii.gz --match 12
    mri_binarize --i $fsl_mask_dir/${subject}/${subject}_first_all_fast_firstseg.nii.gz --o  $fsl_mask_dir/first_bin/${subject}_first_all_fast_firstseg_right.nii.gz --match 51
    mri_binarize --i $fs_mask_dir/${subject}_fs_aseg.nii.gz --o  $fs_mask_dir/${subject}_fs_aseg_left.nii.gz --match 12
    mri_binarize --i $fs_mask_dir/${subject}_fs_aseg.nii.gz --o $fs_mask_dir/${subject}_fs_aseg_right.nii.gz --match 51
    
    # (freesurfer) mri_tessellate mask.nii label mask_surf.surf (to mask surface of segmented structure for visualization)
    mri_tessellate $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg.nii.gz 12 $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_lputa.surf
    mri_tessellate $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg.nii.gz 51 $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_rputa.surf
    
    # (freesurfer) mris_smmoth mask_surf.surf smoothed_mask_surf.smooth (to smooth surface file)
    mris_smooth $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_lputa.surf $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_lputa.smooth
    mris_smooth $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_rputa.surf $dir/${subject}_hc_in_mni_ants_first/${subject}_hc_in_mni_ants_all_fast_firstseg_rputa.smooth
    
    # (freesurfer) freeview -v swi.nii -viewport axial -slice x_slice_num y_slice_num z_slice_num -f left_putamen_smoothed_surf:edgecolor=red right_putamen_smoothed_surf:edgecolor=red -ss screenshot_save_path.png (take screenshot for overlaying SWI and putamen mask, to compare freesurfer mask and proposed method's mask)
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 70  -f $dir/${subject}_surf/${subject}_hc_rputa.smooth:edgecolor=red $dir/${subject}_surf/${subject}_hc_lputa.smooth:edgecolor=red -ss $dir/${subject}_surf/${subject}_hc70_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 75  -f $dir/${subject}_surf/${subject}_hc_rputa.smooth:edgecolor=red $dir/${subject}_surf/${subject}_hc_lputa.smooth:edgecolor=red -ss $dir/${subject}_surf/${subject}_hc75_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 80  -f $dir/${subject}_surf/${subject}_hc_rputa.smooth:edgecolor=red $dir/${subject}_surf/${subject}_hc_lputa.smooth:edgecolor=red -ss $dir/${subject}_surf/${subject}_hc80_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 65  -f $dir/${subject}_surf/${subject}_fs_aseg_rputa.smooth:edgecolor=blue $dir/${subject}_surf/${subject}_fs_aseg_lputa.smooth:edgecolor=blue -ss $dir/${subject}_surf/${subject}_fs65_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 70  -f $dir/${subject}_surf/${subject}_fs_aseg_rputa.smooth:edgecolor=blue $dir/${subject}_surf/${subject}_fs_aseg_lputa.smooth:edgecolor=blue -ss $dir/${subject}_surf/${subject}_fs70_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 75  -f $dir/${subject}_surf/${subject}_fs_aseg_rputa.smooth:edgecolor=blue $dir/${subject}_surf/${subject}_fs_aseg_lputa.smooth:edgecolor=blue -ss $dir/${subject}_surf/${subject}_fs75_put.png
    freeview -v  $dir/${subject}_swi_in_mni_ants.nii.gz -viewport axial -slice 100 120 80  -f $dir/${subject}_surf/${subject}_fs_aseg_rputa.smooth:edgecolor=blue $dir/${subject}_surf/${subject}_fs_aseg_lputa.smooth:edgecolor=blue -ss $dir/${subject}_surf/${subject}_fs80_put.png

    
    '''    extract radiomic features and compare with conventional FreeSurfer method    '''
    # python extract_radiomic_3d.py volume.nii.gz left_putamen_mask.nii.gz right_putamen_mask.nii.gz parameter1.yaml parameter2.yaml parameter3.yaml feature_table.csv (extract radiomic features in 3D volume)
    python extract_radiomic_3d.py volume.nii.gz left_putamen_mask.nii.gz right_putamen_mask.nii.gz parameter1.yaml parameter2.yaml parameter3.yaml feature_table.csv

    # python radiomic_clf.py MSAP_hc.csv MSAP_fs.csv MSAC_hc.csv MSAC_fs.csv PD_hc.csv PD_fs.csv PSP_hc.csv PSP_fs.csv your_save_path train_iter_num test_iter_num (to compare radiomic features's acc, b-acc, AUC, ROC, sensitivity, specifity by 10 machine learning classifier algorithms)
    python radiomic_clf.py MSAP_hc.csv MSAP_fs.csv MSAC_hc.csv MSAC_fs.csv PD_hc.csv PD_fs.csv PSP_hc.csv PSP_fs.csv your_save_path train_iter_num test_iter_num

    