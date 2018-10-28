#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:45:58 2018

@author: raghuvansh
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np

datapath = '/home/raghuvansh/DL/SocialCops Land Classification Challenge/data'

def load_data():
    df_train = pd.read_csv(datapath + '/land_train.csv')
    df_test = pd.read_csv(datapath + '/land_test.csv')
    train_data = df_train.values[:,:-1]
    train_labels = to_one_hot(df_train.values[:,-1])
    test_data = df_test.values
    train_data_std = preprocessing.scale(train_data)
    test_data_std = preprocessing.scale(test_data)
    
    return train_data_std, train_labels, test_data_std

def to_one_hot(test_labels):
    one_hot_matrix = np.zeros((len(test_labels), 4))
    for i in range(len(one_hot_matrix)):
        one_hot_matrix[i, int(test_labels[i]) - 1] = 1
        
    return one_hot_matrix