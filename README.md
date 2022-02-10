# classify_radiomic
An implementation of 3D brain MRI extracted radiomic feature selection and disease classification

## Requirements
Package Required: scipy, sklearn, skfeature, pandas, numpy 

## Running
git clone https://github.com/KimYunSoo/classify_radiomic.git

(If you classify MSA-P, MSA-C, PD, and PSP)

python radiomic_clf.py MSAP_hc.csv MSAP_fs.csv MSAC_hc.csv MSAC_fs.csv PD_hc.csv PD_fs.csv PSP_hc.csv PSP_fs.csv your_save_path train_iter_num test_iter_num

## Result Visualization
![image](https://user-images.githubusercontent.com/45022470/153359259-28cff295-c955-42f8-88c5-79ea179fe21d.png)


## License
GNU General Public License 3.0 License
