# -*- coding: utf-8 -*-
"""
PREPROCESSING: REWEIGHING
Created on Mon Feb  3 14:47:12 2020

The custom pre-processing function is adapted from
https://github.com/IBM/AIF360

@author: Johannes
"""
output_path = 'C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\2_output\\'
# Load all necessary packages
import sys
sys.path.append("../")
sys.path.append("C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\1_code")
import numpy as np
from tqdm import tqdm

from load_germandata import load_GermanDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

#import matplotlib.pyplot as plt


## import dataset
dataset_orig = load_GermanDataset() # customize loading function to all vars
# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'sex'
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
print(dataset_orig.feature_names)

all_metrics =  ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True) #should be stratified for target

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Statistical parity difference between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

methods = ["reweighing", 
           "lfr", 
           "disp_impact_remover"
           ]

for m in methods:
    if m == "reweighing":
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)
        w_train = dataset_transf_train.instance_weights.ravel()
        
        out = dataset_transf_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        out = out.sample(n=out.shape[0], replace=True, weights=w_train)
        
        # Code testing that transformation worked
        assert np.abs(dataset_transf_train.instance_weights.sum()-dataset_orig_train.instance_weights.sum())<1e-6
    elif m == "lfr":
        TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
        TR = TR.fit(dataset_orig_train)
        dataset_transf_train = TR.transform(dataset_orig_train)
        
    elif m == "disp_impact_remover":     
        di = DisparateImpactRemover(repair_level=1, sensitive_attribute='sex')
        dataset_transf_train = di.fit_transform(dataset_orig_train)

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print(m + "achieved a statistical parity difference between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

        
    out.to_csv(output_path + 'pre_' + m + '2.csv', index = None, header=True)



e = dataset_orig_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

e[0].to_csv(output_path + 'pre_' + 'test' + '.csv', index = None, header=True)





