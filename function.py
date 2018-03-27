#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 22:24:45 2018

@author: linsam
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from random import sample, seed
import copy
from datetime import datetime
import gc
#-----------------------------------------

def cbind(base,data):
    #base = train_y
    #data = log_due
    data.index = base.index
    bind_data = pd.concat([base, data], axis=1)
    
    return bind_data

def add_group_col(data):
    #data = mean_due_age
    name = data.index.name
    data[name]=data.index
    data.index  = range(len(data))
    
    return data



def change_type2cate(train_x):
    
    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 
           'Cliente_ID' , 'Producto_ID'  ]
    for col in colname:
        train_x[col] = train_x[col].astype('category')
    
    return train_x
    
    #-----------------------------------------------

def feature_engineering(train_x):
    #s = datetime.datetime.now()

    #mean_due_age = 
    #mean.due.age = main.train.x[, .(mean.due.age = mean(log.due)), by = .(Agencia_ID)]
    s = datetime.now()
    mean_due_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_due'].agg({'mean_due_age':np.mean})
    mean_due_can = train_x.groupby(['Canal_ID'], as_index=False)['log_due'].agg({'mean_due_can':np.mean})
    mean_due_rut = train_x.groupby(['Ruta_SAK'], as_index=False)['log_due'].agg({'mean_due_rut':np.mean})
    mean_due_cli = train_x.groupby(['Cliente_ID'], as_index=False)['log_due'].agg({'mean_due_cli':np.mean})
    mean_due_pro = train_x.groupby(['Producto_ID'], as_index=False)['log_due'].agg({'mean_due_pro':np.mean})

	#--------------------------------------------------------------------------------------------
    mean_vh_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_vh'].agg({'mean_vh_age':np.mean})
	#--------------------------------------------------------------------------------------------
    mean_due_pa = train_x.groupby(['Producto_ID','Agencia_ID'], as_index=False)['log_due'].agg({'mean_due_pa':np.mean})
    mean_due_pr = train_x.groupby(['Producto_ID','Ruta_SAK'], as_index=False)['log_due'].agg({'mean_due_pr':np.mean})
    mean_due_pcli = train_x.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_due'].agg({'mean_due_pcli':np.mean})
    mean_due_pcan = train_x.groupby(['Producto_ID','Canal_ID'], as_index=False)['log_due'].agg({'mean_due_pcan':np.mean})
    
    mean_due_pca = train_x.groupby(['Producto_ID','Cliente_ID','Agencia_ID'], as_index=False)
    mean_due_pca = mean_due_pca['log_due'].agg({'mean_due_pca':np.mean})
	#--------------------------------------------------------------------------------------------
    s = datetime.now()
    mean_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    mean_due_acrcp = mean_due_acrcp['log_due'].agg({'mean_due_acrcp':np.mean})
    sd_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    sd_due_acrcp = sd_due_acrcp['log_due'].agg({'sd_due_acrcp':np.std})
    t = datetime.now() - s
    print(t) 
    # 0:00:04.097429
    
    tem = [ mean_due_age,
            mean_due_can,
            mean_due_rut ,
            mean_due_cli ,
            mean_due_pro ,
            mean_due_pa,
            mean_due_pr,
            mean_due_pcli,
            mean_due_pcan,
            mean_due_pca,	
            mean_vh_age,
            mean_due_acrcp,
            sd_due_acrcp]

    return tem
    
def data_preprocess(real_train_x):
    log_vh = pd.DataFrame( np.log1p( real_train_x['Venta_hoy'] ) )
    log_vh.columns = ['log_vh']
    
    real_train_x['log_vh'] = log_vh
    
    log_due = pd.DataFrame( np.log1p( real_train_x['Demanda_uni_equil'] ) )
    log_due.columns = ['log_due']
    
    real_train_x['log_due'] = log_due
    
    return real_train_x

def merge_feature(train_y,tem,date_name ):
    
    mean_due_age    = tem[0]
    mean_due_can    = tem[1]
    mean_due_rut    = tem[2]
    mean_due_cli    = tem[3]
    mean_due_pro    = tem[4]
    mean_due_pa     = tem[5]
    mean_due_pr     = tem[6]
    mean_due_pcli   = tem[7]
    mean_due_pcan   = tem[8]
    mean_due_pca    = tem[9]	
    mean_vh_age     = tem[10]
    mean_due_acrcp  = tem[11]
    sd_due_acrcp    = tem[12]
    
    if date_name == 'train':
        seed(100)
        # first, we build model by frac 0.5 data
        # that cost more less time
        #test = train_y.sample(frac=0.5,replace=False)    
        # in final build model, we use all data
        test = train_y
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 
                   'Cliente_ID' , 'Producto_ID'  , 
                   'Demanda_uni_equil', 'log_due']
        test = test[colname]
        
    elif date_name == 'test':
        test = train_y
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 
                   'Cliente_ID' , 'Producto_ID'  ]
    
    #len(test)#6895
    #-------------------------------------------------------------------------
    test = pd.merge(test, mean_due_age,how = 'left', on='Agencia_ID')
    
    test = pd.merge(test, mean_due_can,how = 'left', on='Canal_ID')
    test = pd.merge(test, mean_due_rut,how = 'left', on='Ruta_SAK')
    test = pd.merge(test, mean_due_cli,how = 'left', on='Cliente_ID')
    #-------------------------------------------------------------------------
    test = pd.merge(test , mean_due_pa, how = 'left', on = ["Producto_ID", "Agencia_ID"])
    test = pd.merge(test , mean_due_pr, how = 'left', on = ["Producto_ID", "Ruta_SAK"])
    test = pd.merge(test , mean_due_pcli, how = 'left', on = ["Producto_ID", "Cliente_ID"])
    test = pd.merge(test , mean_due_pcan, how = 'left', on = ["Producto_ID", "Canal_ID"])
    #-------------------------------------------------------------------------
    test = pd.merge(test , mean_due_pca, how = 'left', 
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID"])
    #-----------------------------------------------------------------
    test = pd.merge(test , mean_vh_age, how = 'left', on = ["Agencia_ID"])
    test = pd.merge(test , sd_due_acrcp, how = 'left', 
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",	"Canal_ID" , "Ruta_SAK"])
                    
    test = pd.merge(test , mean_due_acrcp, how = 'left', 
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",	"Canal_ID" , "Ruta_SAK"])
      
    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 'Cliente_ID' , 'Producto_ID']
        
    test.drop(colname, axis=1, inplace=True)# inplace is update test (drop)  
    
    return test


def rmsle_eval(y, y0):
    
    y0=y0.get_label()    
    assert len(y) == len(y0)
    return 'rmsle',np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
    
def build_model(test):
    
    test_y = copy.deepcopy( test['log_due'] )
    test_x = copy.deepcopy( test )
    test_x.drop('log_due', axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        test_x, test_y, test_size=0.5, random_state=0)

    real_train_y = copy.deepcopy( X_train['Demanda_uni_equil'] )
    real_test_y = copy.deepcopy( X_test['Demanda_uni_equil'] )
    
    X_train.drop(['Demanda_uni_equil'], axis=1, inplace=True)
    X_test.drop(['Demanda_uni_equil'], axis=1, inplace=True)   

    dtrain = xgb.DMatrix(X_train,y_train)
    dtest = xgb.DMatrix(X_test,y_test)
    
    xgb_params={
            'eta':0.1,
            'max_depth':7,
            'subsample':0.7,
            'colsample_bytree':0.7,
            'objective':'reg:linear',
            'nfold':2,
            'eval_metric': 'rmse',
            'silent': 1
            ,'tree_method':'gpu_hist'
            }
    
    '''cv_output = xgb.cv(xgb_params,
                       dtrain,
                       num_boost_round=75,
                       early_stopping_rounds=5,
                       verbose_eval=5,
                       show_stdv=False)'''
    
    watchlist  = [ (dtrain,'train'),(dtest,'eval')]
                 
    try:
        del model
        gc.collect()
    except:
        123
      
    seed(100)
    #s = datetime.now()
    model = xgb.train(xgb_params, 
                      dtrain, 
                      evals = watchlist,
                      num_boost_round = 100,
                      early_stopping_rounds = 1,
                      verbose_eval = 1
                      #feval = rmsle_eval
                      )
    #t = datetime.now() -s 
    #print(t)


    #model.save_model('0001.model')
    #model = xgb.Booster({'nthread': 4})  # init model
    #model.load_model('model.bin')  # load data              
                      
    #------------------------------------------
    train_pred = model.predict(dtrain)
    train_pred = np.expm1(train_pred).astype(int)
    
    test_pred = model.predict(dtest)
    test_pred = np.expm1(test_pred).astype(int)
    
    temp = np.array( real_train_y )
    temp = np.log1p( temp ) -  np.log1p( train_pred )
    train_rmsle = np.sqrt( np.mean( np.power(temp, 2) ) )

    temp = np.array( real_test_y )
    temp = np.log1p( temp ) -  np.log1p( test_pred )
    test_rmsle = np.sqrt( np.mean( np.power(temp, 2) ) )

    print('train_rmsle :' + str( train_rmsle ) + '\ntest_rmsle :' + str( test_rmsle ) )
    
    ''' compare gpu vs cpu
    gpu cost time : 0:00:44.152872
    cpu cost time : 0:10:30.306069
    -------------------gpu------------------------                   
    train_rmsle :0.48128475735646753
    test_rmsle :0.4830263706408986
    kaggle Private Score    : 0.47381
    kaggle Public Score     : 0.45840
    -------------------cpu------------------------
    train_rmsle :0.4852542921969597
    test_rmsle :0.4867282065421481
    kaggle Private Score    : 0.47636
    kaggle Public Score     : 0.46137
    ===================conclusion==============
    gpu is faster 10 times than cpu, and gpu error is smaller than cpu
    end'''
    

    return model


def build_model2(test):
    
    test_y = copy.deepcopy( test['log_due'] )
    test_x = copy.deepcopy( test )
    test_x.drop(['log_due'], axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        test_x, test_y, test_size=0.5, random_state=0)
    
    real_train_y = copy.deepcopy( X_train['Demanda_uni_equil'] )
    real_test_y = copy.deepcopy( X_test['Demanda_uni_equil'] )
    
    X_train.drop(['Demanda_uni_equil'], axis=1, inplace=True)
    X_test.drop(['Demanda_uni_equil'], axis=1, inplace=True)    
    
    #dtrain = xgb.DMatrix(X_train,y_train)
    #dtest = xgb.DMatrix(X_test,y_test)    
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    xgb_param = {'max_depth':7, 
                 'subsample':0.7,
                 'colsample_bytree':0.7, 
                 'learning_rate':0.1
                 ,'tree_method':'gpu_hist'
                 ,'max_bin' : 16
                 }    
    try:
        del model
        gc.collect()
    except:
        123
        
    model = xgb.XGBRegressor(**xgb_param, n_estimators=100)  
     
    seed(100)
    model.fit(X_train,y_train,
              eval_set = eval_set,
              verbose=True,
              eval_metric = ['rmse']
              #,early_stopping_rounds = 1
              )
    #model.save_model('0001.model')
    #model = xgb.Booster({'nthread': 4})  # init model
    #model.load_model('model.bin')  # load data
    #------------------------------------------
    train_pred = model.predict(X_train)
    train_pred = np.expm1(train_pred).round().astype(int)
    
    test_pred = model.predict(X_test)
    test_pred = np.expm1(test_pred).round().astype(int)
    
    temp = np.array( real_train_y )
    temp = np.log1p( temp ) -  np.log1p( train_pred )
    train_rmsle = np.sqrt( np.mean( np.power(temp, 2) ) )

    temp = np.array( real_test_y )
    temp = np.log1p( temp ) -  np.log1p( test_pred )
    test_rmsle = np.sqrt( np.mean( np.power(temp, 2) ) )

    print('train_rmsle :' + str( train_rmsle ) + '\ntest_rmsle :' + str( test_rmsle ) )
    
    return model,train_rmsle,test_rmsle


def plot_fun(model):
    
    results = model.evals_result()
    
    from matplotlib import pyplot
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

    ax.legend()
    pyplot.ylabel('rmse')
    pyplot.title('XGBoost rmse')
    pyplot.show()    









