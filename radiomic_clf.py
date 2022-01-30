from scipy import stats, io
import sklearn
import pandas as pd 
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import roc_curve, recall_score, make_scorer, roc_auc_score, precision_score, accuracy_score, classification_report, f1_score, auc, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score 
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.linear_model import Lasso, LassoCV, Ridge
import random
from itertools import cycle
import sys

MSAP_hc, MSAP_fs = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
MSAC_hc, MSAC_fs = pd.read_csv(sys.argv[3]), pd.read_csv(sys.argv[4])
PDDD_hc, PDDD_fs = pd.read_csv(sys.argv[5]), pd.read_csv(sys.argv[6])
PSPP_hc, PSPP_fs = pd.read_csv(sys.argv[7]), pd.read_csv(sys.argv[8])
save_dir = sys.argv[9]
train_num, test_num = int(sys.argv[10]), int(sys.argv[11])

MSAP_hc['disease'], MSAP_fs['disease'] = 0,0
MSAC_hc['disease'], MSAC_fs['disease'] = 1,1
PDDD_hc['disease'], PDDD_fs['disease'] = 2,2
PSPP_hc['disease'], PSPP_fs['disease'] = 3,3

def concat2(table1, table2):
    out = pd.concat([table1, table2])
    out = out.drop(['subject_number', 'leftRight', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile','original_firstorder_Maximum', 'original_firstorder_Mean', 'original_firstorder_Minimum', 'original_firstorder_Median','diagnostics_Mask-original_VolumeNum', 'original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance'], axis=1)
    out_class_drop = out.drop(['disease'], axis=1)
    out_class = out['disease'].values
    return out_class_drop, out_class 
MSAP_MSAC_hc_class_drop, MSAP_MSAC_hc_class = concat2(MSAP_hc, MSAC_hc)    
MSAP_PDDD_hc_class_drop, MSAP_PDDD_hc_class = concat2(MSAP_hc, PDDD_hc)
MSAP_PSPP_hc_class_drop, MSAP_PSPP_hc_class = concat2(MSAP_hc, PSPP_hc)
MSAC_PDDD_hc_class_drop, MSAC_PDDD_hc_class = concat2(MSAC_hc, PDDD_hc)
MSAC_PSPP_hc_class_drop, MSAC_PSPP_hc_class = concat2(MSAC_hc, PSPP_hc)
PDDD_PSPP_hc_class_drop, PDDD_PSPP_hc_class = concat2(PDDD_hc, PSPP_hc)
MSAP_MSAC_fs_class_drop, MSAP_MSAC_fs_class = concat2(MSAP_fs, MSAC_fs)    
MSAP_PDDD_fs_class_drop, MSAP_PDDD_fs_class = concat2(MSAP_fs, PDDD_fs)
MSAP_PSPP_fs_class_drop, MSAP_PSPP_fs_class = concat2(MSAP_fs, PSPP_fs)
MSAC_PDDD_fs_class_drop, MSAC_PDDD_fs_class = concat2(MSAC_fs, PDDD_fs)
MSAC_PSPP_fs_class_drop, MSAC_PSPP_fs_class = concat2(MSAC_fs, PSPP_fs)
PDDD_PSPP_fs_class_drop, PDDD_PSPP_fs_class = concat2(PDDD_fs, PSPP_fs)

def dict2csv(hc_result_val, fs_result_val, hc_result_test, fs_result_test, result_path, names):
    hc_result_mean_val={'knn':{}, "LinSVM":{},"RBFSVM":{},"GP":{},"DT":{},"RF":{},"MLP":{},"Ada":{},"NB":{},"QDA":{}}
    fs_result_mean_val={'knn':{}, "LinSVM":{},"RBFSVM":{},"GP":{},"DT":{},"RF":{},"MLP":{},"Ada":{},"NB":{},"QDA":{}}
    hc_result_mean_test={'knn':{}, "LinSVM":{},"RBFSVM":{},"GP":{},"DT":{},"RF":{},"MLP":{},"Ada":{},"NB":{},"QDA":{}}
    fs_result_mean_test={'knn':{}, "LinSVM":{},"RBFSVM":{},"GP":{},"DT":{},"RF":{},"MLP":{},"Ada":{},"NB":{},"QDA":{}}
    for name in names:
        for i in range(6):
            hc_result_mean_val[name][i]=np.mean(np.asarray(hc_result_val[name])[:,i])
            fs_result_mean_val[name][i]=np.mean(np.asarray(fs_result_val[name])[:,i])
            hc_result_mean_test[name][i]=np.mean(np.asarray(hc_result_test[name])[:,i])
            fs_result_mean_test[name][i]=np.mean(np.asarray(fs_result_test[name])[:,i])
    pd.DataFrame.from_dict(hc_result_mean_val, orient='index').to_csv(result_path)
    pd.DataFrame.from_dict(fs_result_mean_val, orient='index').to_csv(result_path, mode= 'a')
    pd.DataFrame.from_dict(hc_result_mean_test, orient='index').to_csv(result_path, mode= 'a')
    pd.DataFrame.from_dict(fs_result_mean_test, orient='index').to_csv(result_path, mode= 'a')
    print ('CSV', result_path, 'saved')

def plot_roc(hc_result, fs_result, png_path):    
    plt.figure(figsize=(5,5))
    plt.plot(np.sort(np.concatenate(np.asarray(hc_result['RBFSVM'])[:,6])), np.sort(np.concatenate(np.asarray(hc_result['RBFSVM'])[:,7])), label="HC (AUC:{0:0.4f})".format(np.mean(np.asarray(hc_result['RBFSVM'])[:,4])))
    plt.plot(np.sort(np.concatenate(np.asarray(fs_result['RBFSVM'])[:,6])), np.sort(np.concatenate(np.asarray(fs_result['RBFSVM'])[:,7])), label="FS (AUC:{0:0.4f})".format(np.mean(np.asarray(fs_result['RBFSVM'])[:,4])))
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")     
    plt.tight_layout()
    plt.savefig(png_path)
    print ('ROC', png_path, 'saved')

def svm_build(X_train, y_train, X_test, y_test, fea_idx, clf):
    clf.fit(X_train[:, fea_idx], y_train)
    fpr, tpr, thr = roc_curve(y_test, clf.predict_proba(X_test[:, fea_idx])[:,1])
    auc_hc = auc(fpr, tpr)
    report = classification_report(clf.predict(X_test[:, fea_idx]), y_test, output_dict=True)
    recall=[]
    for label in ['0','1','2','3','4']:
        if label in report.keys():
            recall.append(report[label]['recall'])
    sen, spe, baccu = recall[0], recall[1], (recall[0]+recall[1])/2
    return [report['accuracy'], baccu, sen, spe, auc_hc, fea_idx, fpr, tpr]

def svm_train_test(hc_class_drop_table, hc_class_array, fs_class_drop_table, fs_class_array, train_num, test_num, result_csv_path, train_png_path, test_png_path):
    hc_class_array = np.where(hc_class_array==np.unique(hc_class_array)[0], 0, hc_class_array)
    hc_class_array = np.where(hc_class_array==np.unique(hc_class_array)[1], 1, hc_class_array)
    fs_class_array = np.where(fs_class_array==np.unique(fs_class_array)[0], 0, fs_class_array)
    fs_class_array = np.where(fs_class_array==np.unique(fs_class_array)[1], 1, fs_class_array)
    names = ["knn","LinSVM","RBFSVM","GP","DT","RF","MLP","Ada","NB","QDA"]
    classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025, probability=True), SVC(C=1, probability=True), GaussianProcessClassifier(1.0 * RBF(1.0)), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1, max_iter=1000), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]
    hc_result_val={'knn':[], "LinSVM":[],"RBFSVM":[],"GP":[],"DT":[],"RF":[],"MLP":[],"Ada":[],"NB":[],"QDA":[]}
    fs_result_val={'knn':[], "LinSVM":[],"RBFSVM":[],"GP":[],"DT":[],"RF":[],"MLP":[],"Ada":[],"NB":[],"QDA":[]}
    hc_result_test={'knn':[], "LinSVM":[],"RBFSVM":[],"GP":[],"DT":[],"RF":[],"MLP":[],"Ada":[],"NB":[],"QDA":[]}
    fs_result_test={'knn':[], "LinSVM":[],"RBFSVM":[],"GP":[],"DT":[],"RF":[],"MLP":[],"Ada":[],"NB":[],"QDA":[]}

    for i in range(train_num):
        print ('************ train', i, '************')
        hX_train, hX_test, hy_train, hy_test = train_test_split(np.asarray(hc_class_drop_table), hc_class_array, test_size=0.3)
        fX_train, fX_test, fy_train, fy_test = train_test_split(np.asarray(fs_class_drop_table), fs_class_array, test_size=0.3)
        n_samples, n_features = np.asarray(hc_class_drop_table).shape    
        num_fea = 10    
        hc_score = fisher_score.fisher_score(hX_train, hy_train)
        hc_idx = fisher_score.feature_ranking(hc_score)
        fs_score = fisher_score.fisher_score(fX_train, fy_train)
        fs_idx = fisher_score.feature_ranking(fs_score)         
        kfold_num = 0
        for kfold_train, kfold_test in KFold(n_splits=3).split(hX_train): 
            for name, clf in zip(names, classifiers):
                hc_result_val[name].append(svm_build(hX_train[kfold_train], hy_train[kfold_train], hX_train[kfold_test], hy_train[kfold_test], hc_idx[0:num_fea], clf))
                fs_result_val[name].append(svm_build(fX_train[kfold_train], fy_train[kfold_train], fX_train[kfold_test], fy_train[kfold_test], fs_idx[0:num_fea], clf))
            kfold_num += 1    
    hc_best_idx = np.asarray(hc_result_val['RBFSVM'])[np.argmax(np.asarray(hc_result_val['RBFSVM'])[:,1]),5]
    fs_best_idx = np.asarray(hc_result_val['RBFSVM'])[np.argmax(np.asarray(fs_result_val['RBFSVM'])[:,1]),5]
    
    for i in range(test_num):
        print ('************ test', i, '************')
        hX_train, hX_test, hy_train, hy_test = train_test_split(np.asarray(hc_class_drop_table), hc_class_array, test_size=0.3)
        fX_train, fX_test, fy_train, fy_test = train_test_split(np.asarray(fs_class_drop_table), fs_class_array, test_size=0.3)
        n_samples, n_features = np.asarray(hc_class_drop_table).shape    
        num_fea = 10    
        for name, clf in zip(names, classifiers):
            hc_result_test[name].append(svm_build(hX_train, hy_train, hX_test, hy_test, hc_best_idx, clf))
            fs_result_test[name].append(svm_build(fX_train, fy_train, fX_test, fy_test, fs_best_idx, clf))
            
    dict2csv(hc_result_val, fs_result_val, hc_result_test, fs_result_test, result_csv_path, names)
    plot_roc(hc_result_val, fs_result_val, train_png_path)
    plot_roc(hc_result_test, fs_result_test, test_png_path)

svm_train_test(MSAP_MSAC_hc_class_drop, MSAP_MSAC_hc_class, MSAP_MSAC_fs_class_drop, MSAP_MSAC_fs_class, train_num, test_num, save_dir+'MSAP_MSAC.csv', save_dir+'MSAP_MSAC_train.png', save_dir+'MSAP_MSAC_test.png')
print ('^^^^^^^^^ MSAP_MSAC done ^^^^^^^^^')
svm_train_test(MSAP_PDDD_hc_class_drop, MSAP_PDDD_hc_class, MSAP_PDDD_fs_class_drop, MSAP_PDDD_fs_class, train_num, test_num, save_dir+'MSAP_PDDD.csv', save_dir+'MSAP_PDDD_train.png', save_dir+'MSAP_PDDD_test.png')
print ('^^^^^^^^^ MSAP_PDDD done ^^^^^^^^^')
svm_train_test(MSAP_PSPP_hc_class_drop, MSAP_PSPP_hc_class, MSAP_PSPP_fs_class_drop, MSAP_PSPP_fs_class, train_num, test_num, save_dir+'MSAP_PSPP.csv', save_dir+'MSAP_PSPP_train.png', save_dir+'MSAP_PSPP_test.png')
print ('^^^^^^^^^ MSAP_PSPP done ^^^^^^^^^')
svm_train_test(MSAC_PDDD_hc_class_drop, MSAC_PDDD_hc_class, MSAC_PDDD_fs_class_drop, MSAC_PDDD_fs_class, train_num, test_num, save_dir+'MSAC_PDDD.csv', save_dir+'MSAC_PDDD_train.png', save_dir+'MSAC_PDDD_test.png')
print ('^^^^^^^^^ MSAC_PDDD done ^^^^^^^^^')
svm_train_test(MSAC_PSPP_hc_class_drop, MSAC_PSPP_hc_class, MSAC_PSPP_fs_class_drop, MSAC_PSPP_fs_class, train_num, test_num, save_dir+'MSAC_PSPP.csv', save_dir+'MSAC_PSPP_train.png', save_dir+'MSAC_PSPP_test.png')
print ('^^^^^^^^^ MSAC_PSPP done ^^^^^^^^^')
svm_train_test(PDDD_PSPP_hc_class_drop, PDDD_PSPP_hc_class, PDDD_PSPP_fs_class_drop, PDDD_PSPP_fs_class, train_num, test_num, save_dir+'PDDD_PSPP.csv', save_dir+'PDDD_PSPP_train.png', save_dir+'PDDD_PSPP_test.png')
print ('^^^^^^^^^ PDDD_PSPP done ^^^^^^^^^')
