# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:31:13 2020

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
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing \
    import CalibratedEqOddsPostprocessing

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

## import prediction results from R BASED ON THE SAME PREDICTIONS FOR COMPARISON
pred = pd.read_csv(output_path + 'POST_allModels_Rprediction.csv')

## import dataset
dataset_orig = load_TaiwanDataset() 

# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'AGE'
privileged_groups = [{'AGE': 1}] 
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)

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
model_names = ['glm', "svmLinear", "rf", "xgbTree", "nnet"]
CEO_thresholds = pd.DataFrame()

for m in model_names:
    scores = np.array(pred[m+'_scores']).reshape(len(pred.index),1)
    labels = np.where(pred[m+'_class']=='Good', 1.0, 2.0).reshape(len(pred.index),1)
    
    dataset_orig_test_pred.scores = scores
    dataset_orig_test_pred.labels = labels
    
    # Reject Option Classification
    CEO = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups, 
                                         privileged_groups=privileged_groups, 
                                         cost_constraint = "weighted"
                                         )
    CEO = CEO.fit(dataset_orig_test, dataset_orig_test_pred)
    
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        dataset_transf_test_pred = CEO.predict(dataset_orig_test_pred, t)
        # print best threshold
        CEO_thresholds[m+"_fairScores_at_"+str(t)] = dataset_transf_test_pred.scores.flatten()
        label_names = np.where(dataset_transf_test_pred.labels==1,'Good','Bad')
        CEO_thresholds[m+"_fairLabels_at_"+str(t)] = label_names

CEO_thresholds.to_csv(output_path + 'CEO_POST_thresholds.csv', index = None, header=True)

