# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:32:53 2020

LOADING GERMAN CREDIT DATASET

@author: Johannes
"""

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import GermanDataset
import numpy as np

def load_GermanDataset():
    def default_preprocessing(df):
        ''' KEEP LABELS AS  IN ORIG
        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'
    
        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'
    
        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'
    
        def group_status(x):
            if x in ['A11']:
                return '<=0'
            elif x in ['A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'
    
        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        '''
        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                        'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)
        
        return df
    
    XD_features = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker']
    D_features = ['sex']
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
    
    all_privileged_classes = {"sex": [1.0], "age": [1.0]}
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                        "age": {1.0: 'Old', 0.0: 'Young'}}
    
    df = GermanDataset(
            label_name=Y_features[0],
            favorable_classes=[1],
            protected_attribute_names=D_features,
            privileged_classes=[all_privileged_classes[x] for x in D_features],
            instance_weights_name=None,
            categorical_features=categorical_features,
            features_to_keep=X_features+Y_features+D_features,
            metadata={ 'label_maps': [{1.0: 'Good', 2.0: 'Bad'}],
                       'protected_attribute_maps': [all_protected_attribute_maps[x]
                                    for x in D_features]},
            custom_preprocessing=default_preprocessing)
    
    return df
