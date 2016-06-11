# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:08:01 2016

@author: subhajit
"""

import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import h5py
import os

os.chdir('D:\Data Science Competitions\Kaggle\Expedia Hotel Recommendations\codes')

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
#agg1 = agg.set_index(['srch_destination_id','hotel_cluster'])
#df1 = train.groupby(['srch_destination_id','hotel_cluster'])['is_booking'].agg(['sum', 'count'])


#test = pd.read_csv('../input/test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'])
#test['srch_ci'] = test.apply(lambda row: datetime.datetime.strptime(row['srch_ci']+ ':00:00:00', '%Y-%m-%d:%H:%M:%S'), axis=1)
destinations = pd.read_csv('../input/destinations.csv')
submission = pd.read_csv('../input/sample_submission.csv')

#test = pd.merge(test, destinations, how='left', on='srch_destination_id')
#id = test.id
#test.drop('id', axis=1, inplace=True)
#pre_process(test)

clf = RandomForestClassifier(n_estimators=0, n_jobs=-1, warm_start=True)
count = 0
chunksize = 200000
#preds = np.zeros((test.shape[0], 100))

#train = pd.read_csv('../input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=0, nrows=1000)
#train.columns.difference(test.columns)
reader = pd.read_csv('../input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    try:
        chunk = chunk[chunk.is_booking==1]
        chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
        chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
        pre_process(chunk)
        #chunk = chunk[chunk.ci_year==2014]
        y = chunk.hotel_cluster
        chunk.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)
        
        if len(y.unique()) == 100:
            clf.set_params(n_estimators=clf.n_estimators+1)
            clf.fit(chunk, y)
        #chunk.columns[np.argsort(clf.feature_importances_)[::-1]]
        
        #preds += np.vstack(tuple([clf.predict_proba(test.loc[i*chunksize:min((i+1)*chunksize,test.shape[0]),:]) for i in range(int(test.shape[0]/100000))]))
        #preds += clf.predict_proba(test)
        
        count = count + chunksize
        print('%d rows completed' % count)
        if(count/chunksize == 300):
            break
    except Exception as e:
        #e = sys.exc_info()[0]
        print('Error: %s' % str(e))
        pass

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
print('writing current probabilities to file')
if os.path.exists('../output/probs/allpreds.h5'):
    with h5py.File('../output/probs/allpreds.h5', 'r+') as hf:
            print('reading in and combining probabilities')
            predslatesthf = hf['preds_latest']
            preds += predslatesthf.value
            print('writing latest probabilities to file')
            predslatesthf[...] = preds
else:
    with h5py.File('../output/probs/allpreds.h5', 'w') as hf:
        print('writing latest probabilities to file')
        hf.create_dataset('preds_latest', data=preds)



print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('../output/pred_sub.csv', index=False)