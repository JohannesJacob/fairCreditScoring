"""
INPROCESSING:
    Independence - Prejudice remover
    Separation - Meta Algorithm
Created on Mon Feb  3 14:47:12 2020

ATTENTION: Data will be scaled!

The custom pre-processing function is adapted from
https://github.com/IBM/AIF360

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
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats
from aif360.algorithms.inprocessing import PrejudiceRemover


from sklearn.preprocessing import StandardScaler, MaxAbsScaler


## import dataset
dataset_orig = load_TaiwanDataset() 

# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'AGE'
privileged_groups = [{'AGE': 1}] 
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)

all_metrics =  ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True) #should be stratified for target

#testdata = dataset_orig_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
#testdata[0].to_csv(output_path + 'taiwan_in_' + 'test' + '.csv', index = None, header=True)

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Statistical parity difference between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

# Scale data and check that the Difference in mean outcomes didn't change
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())

#### PREJUDICE REMOVER ########################################################
pr_predictions = pd.DataFrame()

all_eta = [1, 5, 15, 30, 50, 70, 100, 150]

for eta in all_eta:
    print("Eta: %.2f" % eta)
    colname = "eta_" + str(eta)

    debiased_model = PrejudiceRemover(eta=eta, sensitive_attr=protected, class_attr = "TARGET")
    debiased_model.fit(dataset_orig_train)
    
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    scores = dataset_debiasing_test.scores
    pr_predictions[colname] = sum(scores.tolist(), [])
    
pr_predictions.to_csv(output_path + 'taiwan_pr_predictions' + '.csv', index = None, header=True)

###############################################################################


#### META ALGORITHM ###########################################################
meta_predictions = pd.DataFrame()

all_tau = np.linspace(0.1, 0.9, 4)
for tau in all_tau:
    print("Tau: %.2f" % tau)
    colname = "tau_" + str(tau)

    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=protected)
    debiased_model.fit(dataset_orig_train)
    
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    scores = dataset_debiasing_test.scores
    meta_predictions[colname] = sum(scores.tolist(), [])
    
meta_predictions.to_csv(output_path + 'taiwan_meta_predictions_tau_09' + '.csv', index = None, header=True)

    
###############################################################################
    

#### ADVERSIAL LEARNING #######################################################
    
    
###############################################################################