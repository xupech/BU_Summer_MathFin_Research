#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Wed Aug 19 15:39:47 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from cleandata import *

import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

import sklearn
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


class LSTM_Model:
    """
    LSTM machine class.
    """    
    def __init__(self, train_x, train_y, test_x, test_y):
        """
        LSTM init class

        param: train_x, np.array of training data 

        param: train_y, np.array of training labels 

        param: test_x, np.array of testing data

        param: test_y, np.array of testing_labels     

        """        
        self.train_x = np.reshape(train_x.values, (train_x.shape[0], 1, train_x.shape[1]))
        self.train_y = train_y.values
        
        self.test_x = np.reshape(test_x.values, (test_x.shape[0], 1, test_x.shape[1]))
        self.test_y = test_y.values

        model = Sequential()
        
        dropout_rate = 0.45

        model.add(LSTM(units=16, return_sequences=True))
        model.add(Dropout(rate=dropout_rate))
        model.add(LSTM(units=8, return_sequences=False))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=8))
        model.add(Activation("relu"))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=1))
        model.add(Activation("linear"))
        rms=optimizers.RMSprop(lr=2e-3, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=rms)
        
        self.model = model
        
        
        
    def training(self):
        """
            Function to train the model
        """
        
        self.model.fit(self.train_x, self.train_y, epochs=100, batch_size=300, verbose = 0)
        
    def tunning(self, nfolds=5):        
        """
            Function to hyper tunning the parameters.
        """
        # self.model = KerasRegressor(build_fn=self.create_model(),epochs = 100, batch_size = 1000, verbose=0)
        pass 
        # , 'softmax','selu', 'exponential'
        #dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #optimizers = ['rmsprop', 'adam', 'adagrad', 'adadelta', 'adam', 'sgd']
        #kernel_initializers = ['glorot_uniform', 'normal', 'uniform']
        #activations = ['relu', 'tanh', 'sigmoid'] 
        #activation = activations,
        #epochs = np.array([50, 100, 150, 200])
        #batches = np.array([500, 1000, 2000,3000,4000,5000])
        #param_grid = {'dropout_rate':dropout_rates, 'optimizer':optimizers, 'epochs':epochs, 
        #                  'batch_size':batches, 'activation':activations,
        #                  'recurrent_activation': activations, 'kernel_initializer':kernel_initializers}
        #grid_search = GridSearchCV(self.model, param_grid, cv=nfolds)
        #grid_search.fit(self.train_x, self.train_y)
        
        #self.model.Param = grid_search.best_params_
        #return grid_search.best_params_
        
        
    def predict(self):
        """
            Function to predict y_hat using the trained model
        """        
        return self.model.predict(self.test_x).reshape(-1)
        
        

from sklearn.model_selection import train_test_split
    
if __name__ == '__main__':
    
    raw_data = pd.read_csv("constituents_2013_fund_tech.csv")
    clean_data = cleandata(raw_data, 'median', small_sample= True)
    clean_data.__main__()
    
    clean_data=clean_data.data_float
    
    y = clean_data['RET']
    x = clean_data.drop(['RET'], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)    
    
    model = LSTM_Model(x_train, y_train, x_test, y_test)    
    model.tunning()        
    model.training()    
    y_hat = model.predict()
    
    plt.plot(range(len(y_hat)), y_hat, label='predict')
    
#    plt.plot(range(len(y_hat)) , y_train, label='Real')    
    plt.plot(range(len(y_hat)) , y_test, label='Real')    
    
    plt.legend()