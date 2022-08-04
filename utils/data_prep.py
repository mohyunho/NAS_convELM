'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt


## Define function for data reading
def load_data (data_path_list, columns_ts, columns_rul):
  train_FD = pd.read_csv(data_path_list[0], sep= ' ', header=None, 
                            names=columns_ts, index_col=False)
  test_FD = pd.read_csv(data_path_list[1], sep= ' ', header=None, 
                            names=columns_ts, index_col=False)
  RUL_FD = pd.read_csv(data_path_list[2], sep= ' ', header=None, 
                            names=columns_rul, index_col=False)
  
  return train_FD, test_FD, RUL_FD


def rul_mapper (train_FD, test_FD, piecewise_lin_ref):
    # get the time of the last available measurement for each unit
    mapper = {}
    for unit_nr in train_FD['unit_nr'].unique():
        mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()
        
    # calculate RUL = time.max() - time_now for each unit
    train_FD['RUL'] = train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles']
    # piecewise linear for RUL labels
    train_FD['RUL'].loc[(train_FD['RUL'] > piecewise_lin_ref)] = piecewise_lin_ref

    ## Excluse columns which only have NaN as value
    # nan_cols = ['sensor_{0:02d}'.format(s + 22) for s in range(5)]
    cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
    # print('Columns with all nan: \n' + str(cols_nan) + '\n')
    cols_const = [ col for col in train_FD.columns if len(train_FD[col].unique()) <= 2 ]
    # print('Columns with all const values: \n' + str(cols_const) + '\n')

    ## Drop exclusive columns
    # train_FD = train_FD.drop(columns=cols_const + cols_nan)
    # test_FD = test_FD.drop(columns=cols_const + cols_nan)

    train_FD = train_FD.drop(columns=cols_const + cols_nan + ['sensor_01','sensor_05','sensor_06',
                                                            'sensor_10','sensor_16','sensor_18','sensor_19'])

    test_FD = test_FD.drop(columns=cols_const + cols_nan + ['sensor_01','sensor_05','sensor_06',
                                                            'sensor_10','sensor_16','sensor_18','sensor_19'])


    return train_FD, test_FD


### function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 142 192 -> from row 142 to 192
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


        
        
def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]
        

### Normalize sensor measurement data 
def df_preprocessing(df, train=True):
    if train==True:
        cols_normalize = df.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])
    else : 
        cols_normalize = df.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns=df.columns)
    if train==True:
        pass
    else :
        df = df.reset_index(drop=True)
    
    return df