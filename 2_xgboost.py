# -*- coding: utf-8 -*-
"""
Created on Mon May 30 06:44:09 2016

@author: subhajit
"""

import pandas as pd
import datetime
from sklearn import cross_validation
import xgboost as xgb
import numpy as np
import h5py
import os

os.chdir('D:\Data Science Competitions\Kaggle\Expedia Hotel Recommendations\codes')

def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', -metric

def pre_process(data):
    try:
        data.loc[data.srch_ci.str.endswith('00'),'srch_ci'] = '2015-12-31'
        data['srch_ci'] = data.srch_ci.astype(np.datetime64)
        data.loc[data.date_time.str.endswith('00'),'date_time'] = '2015-12-31'
        data['date_time'] = data.date_time.astype(np.datetime64)
    except:
        pass
    data.fillna(0, inplace=True)
    data['srch_duration'] = data.srch_co-data.srch_ci
    data['srch_duration'] = data['srch_duration'].apply(lambda td: td/np.timedelta64(1, 'D'))
    data['time_to_ci'] = data.srch_ci-data.date_time
    data['time_to_ci'] = data['time_to_ci'].apply(lambda td: td/np.timedelta64(1, 'D'))
    data['ci_month'] = data['srch_ci'].apply(lambda dt: dt.month)
    data['ci_day'] = data['srch_ci'].apply(lambda dt: dt.day)
    #data['ci_year'] = data['srch_ci'].apply(lambda dt: dt.year)
    data['bk_month'] = data['date_time'].apply(lambda dt: dt.month)
    data['bk_day'] = data['date_time'].apply(lambda dt: dt.day)
    #data['bk_year'] = data['date_time'].apply(lambda dt: dt.year)
    data['bk_hour'] = data['date_time'].apply(lambda dt: dt.hour)
    data.drop(['date_time', 'user_id', 'srch_ci', 'srch_co'], axis=1, inplace=True)

if os.path.exists('../output/srch_dest_hc_hm_agg.csv'):
    agg1 = pd.read_csv('../output/srch_dest_hc_hm_agg.csv')
else:
    reader = pd.read_csv('../input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=200000)
    pieces = [chunk.groupby(['srch_destination_id','hotel_country','hotel_market','hotel_cluster'])['is_booking'].agg(['sum','count']) for chunk in reader]
    agg = pd.concat(pieces).groupby(level=[0,1,2,3]).sum()
    del pieces
    agg.dropna(inplace=True)
    agg['sum_and_cnt'] = 0.85*agg['sum'] + 0.15*agg['count']
    agg = agg.groupby(level=[0,1,2]).apply(lambda x: x.astype(float)/x.sum())
    agg.reset_index(inplace=True)
    agg1 = agg.pivot_table(index=['srch_destination_id','hotel_country','hotel_market'], columns='hotel_cluster', values='sum_and_cnt').reset_index()
    agg1.to_csv('../output/srch_dest_hc_hm_agg.csv', index=False)
    del agg

destinations = pd.read_csv('../input/destinations.csv')
submission = pd.read_csv('../input/sample_submission.csv')

clf = xgb.XGBClassifier(#missing=9999999999,
                objective = 'multi:softmax',
                max_depth = 5,
                n_estimators=300,
                learning_rate=0.01,
                nthread=4,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight = 3,
                #scale_pos_weight = ratio,
                #reg_alpha=0.03,
                #seed=1301,
                silent=False)


if os.path.exists('rows_complete.txt'):
    with open('rows_complete.txt', 'r') as f:
        skipsize = int(f.readline())
else:
    skipsize = 0

skip = 0 if skipsize==0 else range(1, skipsize)
tchunksize = 1000000
print('%d rows will be skipped and next %d rows will be used for training' % (skipsize, tchunksize))
train = pd.read_csv('../input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=skip, nrows=tchunksize)
train = train[train.is_booking==1]
train = pd.merge(train, destinations, how='left', on='srch_destination_id')
train = pd.merge(train, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
pre_process(train)
#chunk = chunk[chunk.ci_year==2014]
y = train.hotel_cluster
train.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, y, stratify=y, test_size=0.2)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric=map5eval, eval_set=[(X_train, y_train),(X_test, y_test)])


count = 0
chunksize = 10000
preds = np.empty((submission.shape[0],clf.n_classes_))
#chunk = pd.read_csv('../input/test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=10000)
reader = pd.read_csv('../input/test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
    chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
    chunk.drop(['id'], axis=1, inplace=True)
    pre_process(chunk)
    
    pred = clf.predict_proba(chunk)
    preds[count:(count + chunk.shape[0]),:] = pred
    count = count + chunksize
    print('%d rows completed' % count)

del clf
del agg1
if os.path.exists('../output/probs/allpreds_xgb.h5'):
    with h5py.File('../output/probs/allpreds_xgb.h5', 'r+') as hf:
        print('reading in and combining probabilities')
        predshf = hf['preds']
        preds += predshf.value
        print('writing latest probabilities to file')
        predshf[...] = preds
else:
    with h5py.File('../output/probs/allpreds_xgb.h5', 'w') as hf:
        print('writing latest probabilities to file')
        hf.create_dataset('preds', data=preds)

print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('../output/pred_sub.csv', index=False)


skipsize += tchunksize
with open('rows_complete.txt', 'w') as f:
    f.write(str(skipsize))