# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:03:47 2020

@author: Johannes
"""

import pandas as pd
import numpy as np

import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from catboost import Pool, CatBoostClassifier

from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint

# Read orig data
filepath = "C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\2_raw data\\"

data = pd.read_csv(
    filepath+"lending-club\\accepted_2007_to_2018q4.csv\\accepted_2007_to_2018Q4.csv",
    parse_dates=['issue_d'], infer_datetime_format=True)

# Selecting loans issued in 2018
data = data[(data.issue_d >= '2018-01-01 00:00:00') & (data.issue_d < '2019-01-01 00:00:00')]
data = data.reset_index(drop=True)

# Read data description
browse_notes = pd.read_excel(filepath+'\\lending-club-loan-data\\LCDataDictionary.xlsx',
                             sheet_name=1)

# Match column names with colnames in data
browse_feat = browse_notes['BrowseNotesFile'].dropna().values
browse_feat = [re.sub('(?<![0-9_])(?=[A-Z0-9])', '_', x).lower().strip() for x in browse_feat]

data_feat = data.columns.values
# np.setdiff1d(browse_feat, data_feat)
# np.setdiff1d(data_feat, browse_feat)

wrong = ['is_inc_v', 'mths_since_most_recent_inq', 'mths_since_oldest_il_open',
         'mths_since_recent_loan_delinq', 'verified_status_joint']
correct = ['verification_status', 'mths_since_recent_inq', 'mo_sin_old_il_acct',
           'mths_since_recent_bc_dlq', 'verification_status_joint']

browse_feat = np.setdiff1d(browse_feat, wrong)
browse_feat = np.append(browse_feat, correct)

avail_feat = np.intersect1d(browse_feat, data_feat)
X = data[avail_feat].copy()
X.info()

# Clean date cols
X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'], infer_datetime_format=True)
X['sec_app_earliest_cr_line'] = pd.to_datetime(X['sec_app_earliest_cr_line'], infer_datetime_format=True)

# Clean employment length
X['emp_length'] = X['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})
X['emp_length'] = X['emp_length'].str.extract('(\d+)').astype('float')
X['id'] = X['id'].astype('float')

# Fixing missing values
X = X.drop(['desc', 'member_id'], axis=1, errors='ignore') # Drop empty features

fill_empty = ['emp_title', 'verification_status_joint'] 
fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
            'mths_since_last_major_derog', 'mths_since_last_record',
            'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
            'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
            'pct_tl_nvr_dlq','sec_app_mths_since_last_major_derog'] 
fill_min = np.setdiff1d(X.columns.values, np.append(fill_empty, fill_max)) 

X[fill_empty] = X[fill_empty].fillna('') # NA replaced with empty string
X[fill_max] = X[fill_max].fillna(X[fill_max].max()) # fill with maximum value
X[fill_min] = X[fill_min].fillna(X[fill_min].min()) # fill with min


# Drop constant features and features with many unique values
X = X.drop(['num_tl_120dpd_2m', 'id'], axis=1, errors='ignore')
X = X.drop(['url', 'emp_title'], axis=1, errors='ignore')

# Drop high collinearity features (numeric -> Pearson)
num_feat = X.select_dtypes('number').columns.values

comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(X[comb[0]], X[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)
    
high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.9]

# Drop first feature from each highly correlated feature pair
X = X.drop(np.unique(high_corr_num[:, 0]), axis=1, errors='ignore')

# Drop high collinearity features (categorical -> Kramers V)
cat_feat = X.select_dtypes('object').columns.values

# Calculate Kramers V
cat_feat = X.select_dtypes('object').columns.values

comb_cat_feat = np.array(list(combinations(cat_feat, 2)))
corr_cat_feat = np.array([])
for comb in comb_cat_feat:
    table = pd.pivot_table(X, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
    corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
    corr_cat_feat = np.append(corr_cat_feat, corr)
    
high_corr_cat = comb_cat_feat[corr_cat_feat >= 0.9]

# Drop first feature from each highly correlated feature pair
X = X.drop(np.unique(high_corr_cat[:, 1]), axis=1, errors='ignore')

# Create TARGET feature
status_map = {'Current': 1.0, 'Fully Paid': 1.0, 'In Grace Period': 1.0,
              'Charged Off': 2.0, 'Late (31-120 days)': 2.0, 'Late (16-30 days)': 2.0,
              'Default': 2.0,}
X['TARGET'] = data['loan_status'].replace(status_map)

# Write new data frame
X.to_csv(filepath + 'accepted2018_reduced' + '.csv', index = None, header=True)









