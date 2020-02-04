# -*- coding: utf-8 -*-
"""
PREPROCESSING: REWEIGHING
Created on Mon Feb  3 14:47:12 2020

The custom pre-processing function is adapted from
https://github.com/IBM/AIF360

@author: Johannes
"""

# Load all necessary packages
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

#from common_utils import compute_metrics

XD_features = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker']
D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes
Y_features = ['credit']
X_features = list(set(XD_features)-set(D_features))
categorical_features = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']

all_privileged_classes = {"sex": [1.0], "age": [1.0]}
all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "age": {1.0: 'Old', 0.0: 'Young'}}
def default_preprocessing(df):
    status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
    df['sex'] = df['personal_status'].replace(status_map)
    return df

gd = GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=default_preprocessing)
print(gd.feature_names)
g = gd.convert_to_dataframe


## import dataset
dataset_orig = load_preproc_data_german(['sex']) # customize loading function to all vars
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
print(dataset_orig.feature_names)

all_metrics =  ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Statistical parity difference between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

methods = ["reweighing", "lfr", "disp_impact_remover"]

for m in methods:
    if m == "reweighing":
        # REWEIGHING ## --------------------------------------------
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)
        
        # Code testing that transformation worked
        assert np.abs(dataset_transf_train.instance_weights.sum()-dataset_orig_train.instance_weights.sum())<1e-6
        # ----------------------------------------------------------
    elif m == lfr:
        

e = dataset_transf_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

e[0].to_csv(r'C:\Users\Johannes\OneDrive\Dokumente\Humboldt-Universität\Msc WI\1_4. Sem\Master Thesis II\fairCreditScoring\py_code\1_PRE\2_output\pre_' + m + '.csv', index = None, header=True)









