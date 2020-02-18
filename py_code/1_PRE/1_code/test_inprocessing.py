# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:09:42 2020

@author: Johannes
"""

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_orig = load_preproc_data_german()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

biased_model = MetaFairClassifier(tau=0, sensitive_attr="sex")
biased_model.fit(dataset_orig_train)

# Apply the unconstrained model to test data
dataset_bias_test = biased_model.predict(dataset_orig_test)

predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in list(dataset_bias_test.labels)]
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)["sex"]

acc, sr, unconstrainedFDR = getStats(y_test, predictions, x_control_test)
print(unconstrainedFDR)

# Learn debiased classifier
tau = 0.8
debiased_model = MetaFairClassifier(tau=tau, sensitive_attr="sex")
debiased_model.fit(dataset_orig_train)

# Apply the debiased model to test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

### Testing 
predictions = list(dataset_debiasing_test.labels)

#### Running the algorithm for different tau values

tau_predictions = pd.DataFrame()
accuracies, false_discovery_rates, statistical_rates = [], [], []
s_attr = "sex"
# Converting to form used by celisMeta.utils file
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)[s_attr]

all_tau = np.linspace(0.1, 0.9, 9)
for tau in all_tau:
    print("Tau: %.2f" % tau)
    colname = "tau_" + str(tau)

    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr)
    debiased_model.fit(dataset_orig_train)
    
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    predictions = dataset_debiasing_test.labels
    tau_predictions[colname] = sum(predictions.tolist(), [])
    predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in predictions]
    
    acc, sr, fdr = getStats(y_test, predictions, x_control_test)
    
    ## Testing
    assert (tau < unconstrainedFDR) or (fdr >= unconstrainedFDR)
    
    accuracies.append(acc)
    false_discovery_rates.append(fdr)
    statistical_rates.append(sr)

