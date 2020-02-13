# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:45:28 2020

@author: Johannes
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:32:53 2020

LOADING GERMAN CREDIT DATASET

@author: Johannes
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np

def load_GMSCDataset():
    
    filepath = "C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\2_raw data\\GiveMeSomeCredit\\cs-training.csv"
    df = pd.read_csv(filepath, sep=',', na_values=[])
    
    del df['Unnamed: 0']
    df['age'] = df['age'].apply(lambda x: np.where(x >= 26, 1.0, 0.0))
    df = df.rename(columns={'SeriousDlqin2yrs': 'TARGET', 'age': 'AGE'})
    df = df[df.MonthlyIncome.notnull()]


    XD_features = ['RevolvingUtilizationOfUnsecuredLines', 'AGE',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = []
    
    privileged_class = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}
    
  
    def default_preprocessing(df):
              
        # Good credit == 1
        status_map = {0: 1.0, 1: 2.0}
        df['TARGET'] = df['TARGET'].replace(status_map)

                
        return df
    
    df_standard = StandardDataset(
        df = df,
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[privileged_class["AGE"]],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'Good', 2.0: 'Bad'}],
                   'protected_attribute_maps': [protected_attribute_map]},
        custom_preprocessing=default_preprocessing)
    
    return df_standard
