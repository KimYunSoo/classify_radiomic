import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import SimpleITK as sitk
import ants
import sys
import nibabel as nib
import numpy as np

maksOrNot = sys.argv[1]
fix= ants.image_read(sys.argv[2])
mov= ants.image_read(sys.argv[3])
out= str(sys.argv[4])

reg_output = ants.registration(fix, mov, type_of_transforme = 'SyN')
if maksOrNot=='mask':
    mask = ants.image_read(sys.argv[5])
    # out_mask= ants.apply_transforms(fix, mask, transformlist=reg_output['fwdtransforms'], interpolator='genericLabel')
    out_mask= ants.apply_transforms(fix, mask, transformlist=reg_output['fwdtransforms'])
    ants.image_write(out_mask, out)
elif maksOrNot=='swi':
    swi = ants.image_read(sys.argv[5])
    out_swi= ants.apply_transforms(fix, swi, transformlist=reg_output['fwdtransforms'])
    ants.image_write(out_swi, out)
else:
    ants.image_write(reg_output['warpedmovout'], out)

# print (maksOrNot, fix, mov, out, mask)

