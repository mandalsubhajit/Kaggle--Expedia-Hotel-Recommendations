# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:17:11 2016

@author: subhajit
"""

import pandas as pd
import os

os.chdir('D:\Data Science Competitions\Kaggle\Expedia Hotel Recommendations\codes')


match_pred = pd.read_csv('../output/match_pred.csv')
match_pred.fillna('', inplace=True)
match_pred = match_pred['hotel_cluster'].tolist()
match_pred = [s.split(' ') for s in match_pred]

pred_sub = pd.read_csv('../output/pred_sub.csv')
ids = pred_sub.id
pred_sub = pred_sub['hotel_cluster'].tolist()
pred_sub = [s.split(' ') for s in pred_sub]

def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if (marker in seen) or (marker == ''): continue
        seen[marker] = 1
        result.append(item)
    return result
    
full_preds = [f5(match_pred[p] + pred_sub[p])[:5] for p in range(len(pred_sub))]

write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(ids[i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("../output/predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))