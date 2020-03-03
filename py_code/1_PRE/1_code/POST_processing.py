# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:19:48 2020

@author: Johannes
"""

output_path = 'C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\3_wipResults\\'
# Load all necessary packages
import sys
sys.path.append("../")
sys.path.append("C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\1_code")
import numpy as np
import pandas as pd

from load_germandata import load_GermanDataset
from load_taiwandata import load_TaiwanDataset
from load_gmscdata import load_GMSCDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from aif360.algorithms.postprocessing.eq_odds_postprocessing\
        import EqOddsPostprocessing

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

## import prediction results from R BASED ON THE SAME PREDICTIONS FOR COMPARISON
pred = pd.read_csv(output_path + 'POST_Rprediction.csv')

## import dataset
dataset_orig = load_TaiwanDataset() 

# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'AGE'
privileged_groups = [{'AGE': 1}] 
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)

# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True) #should be stratified for target

# Scale data
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

# Postprocessing
model_names = ['glm', "svmRadial", "rf", "xgbTree", "nnet"]
ROC_thresholds = pd.DataFrame()
EOP_thresholds = pd.DataFrame()

for m in model_names:
    scores = np.array(pred[m+'_scores']).reshape(len(pred.index),1)
    labels = np.where(pred[m+'_class']=='Good', 2.0, 1.0).reshape(len(pred.index),1)
    
    dataset_orig_test_pred.scores = scores
    dataset_orig_test_pred.labels = labels
    
    # Reject Option Classification
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_test, dataset_orig_test_pred)
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
        
    ROC_thresholds[m+"_fairScores"] = dataset_transf_test_pred.scores.flatten()
    label_names = np.where(dataset_transf_test_pred.labels==1,'Good','Bad')
    ROC_thresholds[m+"_fairLabels"] = label_names
    
    # Equality of Odds
    EOP = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups)
    EOP = EOP.fit(dataset_orig_test, dataset_orig_test_pred)
    dataset_transf_test_pred = EOP.predict(dataset_orig_test_pred)
    
    EOP_thresholds[m+"_fairScores"] = dataset_transf_test_pred.scores.flatten()
    label_names = np.where(dataset_transf_test_pred.labels==1,'Good','Bad')
    EOP_thresholds[m+"_fairLabels"] = label_names  

ROC_thresholds.to_csv(output_path + 'ROC_POST_thresholds.csv', index = None, header=True)
EOP_thresholds.to_csv(output_path + 'EOP_POST_thresholds.csv', index = None, header=True)







