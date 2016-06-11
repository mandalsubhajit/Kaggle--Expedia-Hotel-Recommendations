# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:41:24 2016

@author: subhajit
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import h5py
import os

os.chdir('D:\Data Science Competitions\Kaggle\Expedia Hotel Recommendations\codes')

submission = pd.read_csv('../input/sample_submission.csv')

# read in RF results
with h5py.File('../output/probs/allpreds.h5', 'r') as hf:
        predshf = hf['preds_latest']
        preds = 0.31*normalize(predshf.value, norm='l1', axis=1)

# read in XGB results
with h5py.File('../output/probs/allpreds_xgb.h5', 'r') as hf:
        predshf = hf['preds']
        preds += 0.39*normalize(predshf.value, norm='l1', axis=1)

# read in SGD results
with h5py.File('../output/probs/allpreds_sgd.h5', 'r') as hf:
        predshf = hf['preds']
        preds += 0.27*normalize(predshf.value, norm='l1', axis=1)

# read in Bernoulli NB results
with h5py.File('../output/probs/allpreds_bnb.h5', 'r') as hf:
        predshf = hf['preds']
        preds += 0.03*normalize(predshf.value, norm='l1', axis=1)

print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('../output/pred_sub.csv', index=False)