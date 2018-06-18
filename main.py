#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:38:05 2018

@author: linsam
"""

import os,sys
import pandas as pd
import numpy as np
#from random import seed
from collections import Counter
import gc
from datetime import datetime
#from graphviz import Digraph

os.chdir('/home/linsam/kaggle/kaggle_Grupo_Bimbo_Inventory_Demand')
sys.path.append('/home/linsam/kaggle/kaggle_Grupo_Bimbo_Inventory_Demand')
import function
import xgboost as xgb

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#----------------------------------------------
'''separate data, because it is time series'''
train_y = train_data[train_data['Semana']==9]
Counter(train_data['Semana'])
train_x = train_data[train_data['Semana']<9]

real_train_x = train_data
del train_data
gc.collect()
#----------------------------------------------
# feature engineering for train data

log_due = pd.DataFrame( np.log1p( train_y['Demanda_uni_equil'] ) )
log_due.columns = ['log_due']
train_y['log_due'] = log_due
#-----------------------------------------------
train_x = function.data_preprocess(train_x)

train_x = function.change_type2cate(train_x)

tem = function.feature_engineering(train_x)

test = function.merge_feature(train_y,tem,'train')

#---------------------------------------------
# feature engineering for test data ( test will predict our target  )
#-----------------------------------------------
real_train_x = function.data_preprocess(real_train_x)

real_train_x = function.change_type2cate(real_train_x)
tem = function.feature_engineering(real_train_x)

real_test = function.merge_feature(test_data,tem,'test')
test_id = real_test['id']
real_test.drop(['id','Semana'], axis=1, inplace=True)

#dreal_test = xgb.DMatrix(real_test)
#------------------------------------------
# build model
# test = test2
# model = build_model(test)# xgb.train
#del model
model = function.build_model(test) # xgb.XGBRegressor
#plot_fun(model)
gc.collect()
#xgb.plot_importance(model)
#---------------------------------------
# pred 

y_pred = model.predict(xgb.DMatrix(real_test))

y_pred_exp = np.expm1(y_pred).round().astype(int)

result1 = pd.DataFrame( {'id':test_id,
           'Demanda_uni_equil':y_pred_exp} )

result1.to_csv('pred.csv',index=False)





       
              
              