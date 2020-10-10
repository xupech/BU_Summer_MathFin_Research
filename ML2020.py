#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Wed Jun  3 18:44:48 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""

from cleandata import *  # analysis:ignore
import pandas as pd
from SVM import *  # analysis:ignore
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.mixture import BayesianGaussianMixture as BGM
from LSTM import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cycler import cycler

def one_step_function(raw_data, param_dict):
    
    ML_object = MLEngineer(raw_data, num_stocks = param_dict['num_stocks'],
                       trading_window = param_dict['trading_window'], algorithm = param_dict['algorithm'], small_sample = False, method = param_dict['fill_in_method'], pca = param_dict['pca'])
    ML_object.data_processing()
    
    returns = ML_object.profit_construct(param_dict['time_list'], strategy = param_dict['strategy'])
    
    ML_object.return_plot(returns)
    
    return ML_object


class MLEngineer(cleandata):  # analysis:ignore
    def __init__(self, raw_data, num_stocks, trading_window, algorithm, method='mean', selected_columns=None, small_sample=False, pca =False):
        """
            Method to initalize Machine Learning Algorithm Object,
            param: raw_data, raw data outputted from sas as pandas dataframe, must have columns Date in sas format
            param: method, method used to fill in missing value
            param: num_stocks, number of stocks in the optimial portfolio            
            param: trading window, trading frequency with unit day            
            param: algorithm, string variable indicating what algorithm wants to implement

        """

        cleandata.__init__(self, raw_data, method,
                           selected_columns, small_sample)  # analysis:ignore
        self.num_stocks = num_stocks
        self.trading_window = trading_window
        if algorithm == 'SVM':
            self.model = SVM_Machine  # analysis:ignore
        if algorithm == 'LSTM':
            self.model = LSTM_Model  # analysis:ignore            
        self.descriptive = ['Unnamed: 0', 'permno', 'gvkey', 'fyear']
        self.count =0 

        self.pca=pca
        
    def data_processing(self):
        """
            Method to clean data
        """
        self.__main__()
        self.get_clean_data()
        self.date_list = self.clean_data.DATE.unique()
        
        
    def return_prediction(self, time_step):
        """
            Method to predict returns
        """
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_date(time_step)

        if self.pca:
            self.x_train, self.y_train, self.x_test, self.y_test = self.PCA(time_step)
    
        self.return_clustering(time_step)

        y_hat = self.clustering.predict(self.x_test).copy()
        for c in set(self.clustering.predict(self.x_test)):

            flag_train = self.clustering.predict(self.x_train) == c
            flag_test =  self.clustering.predict(self.x_test) == c
            self.x_train.loc[flag_train]
            self.y_train.loc[flag_train]            
            model = self.model(self.x_train.loc[flag_train],self.y_train.loc[flag_train], self.x_test[flag_test], self.y_test[flag_test])
            #model.tunning()
            model.training()            
            y_hat[flag_test] = model.predict()
                      
        return y_hat, self.y_test.values

    def return_clustering(self, time_step, n_clusters = 10):
        
        
        c_model = BGM(n_clusters, covariance_type='diag').fit(self.x_train)    
        
        self.clustering = c_model


        if self.count % self.trading_window  == 0:
            X_principal = pd.DataFrame(self.PCA(time_step, 2)[0]) 
            X_principal.columns = ['P1', 'P2']
            X_principal['clusters'] = self.clustering.predict(self.x_train)
            sns.pairplot(x_vars=["P1"], y_vars=["P2"], data=X_principal, hue="clusters", size=5)
            plt.show()
        self.count+=1

        

    def PCA(self, time_step, pc = 60, verbose = False):
        '''pc: number of PC components.'''
 
        X_train_set, y_train, X_test_set, y_test = self.split_date(time_step)
        scaler = StandardScaler().fit(X_train_set)
        X_train_set = scaler.transform(X_train_set)
        X_test_set = scaler.transform(X_test_set)

        pca = PCA(n_components=pc, svd_solver='full')
        pca.fit(X_train_set)

        X_train_set = pca.transform(X_train_set)
        X_test_set = pca.transform(X_test_set)

        if verbose:
            print('number of PCs: ', pc)
            print(sum(pca.explained_variance_ratio_[:pc]))
        return pd.DataFrame(X_train_set), y_train, pd.DataFrame(X_test_set), y_test


    def split_date(self, time_step):
        """
            Method to split data on specific date into training and testing
        """

        month_of_looking_back = self.trading_window
        time = time_step
        time_train_start = time - month_of_looking_back
        time_test_end = time + 1

        dataset = self.clean_data

        train = dataset.loc[(dataset['DATE'] > time_train_start)
                            & (dataset['DATE'] <= time)]
        test = dataset.loc[dataset['DATE'] == time_test_end]

        '''Drop gvkeys, date and all other irrelevant columns.'''
        self.tickers = test['permno'].values
        
        x_train = train.drop(self.descriptive + ['RET'] + ['DATE'], axis=1)
        y_train = train['RET']
        x_test = test.drop(self.descriptive + ['RET'] + ['DATE'], axis=1)
        y_test = test['RET']

        return x_train, y_train, x_test, y_test

    def opti_stocks(self, time_step):
        """
            Method to output optimally selected stocks 
        """
        y_hat, y_test = self.return_prediction(time_step)
        indx_rank = np.argsort(y_hat)
#        print(y_hat)
        portfoli_permno = self.tickers[indx_rank[-self.num_stocks:]]

        return portfoli_permno, indx_rank

    def profit_construct(self, time_list, strategy):
        """
            Method to do performance mesaurement attribution    
        """

        time = pd.to_datetime(time_list[0]).to_period('M')
        time_start = time - self.trading_window
        time_end = pd.to_datetime(time_list[1]).to_period('M') + 1
                
        '''Check if looking back period has gone out of range.'''
        if time_start not in self.date_list:
            raise SystemExit("Invalid start date, please enter a valid date. ")

        '''Check if looking ahead period has gone out of range.'''
        if time_end not in self.date_list:
            raise SystemExit("Invalid end date, please enter a valid date. ")
  
        return_list = []                
        time_step = time
        
        while time_step != time_end:
            print(time_step)                
            port_id, indx_rank = self.opti_stocks(time_step)
            temp = (self.y_test * self.y_std) + self.y_mean
            
            if strategy == 'long_equal_weighted':            
#                return_list.append(self.y_test.iloc[indx_rank[-self.num_stocks:]].mean())
                return_list.append(temp.iloc[indx_rank[-self.num_stocks:]].mean())                
            if strategy == 'short_equal_weighted':            
#                return_list.append(self.y_test.iloc[indx_rank[:self.num_stocks]].mean())
                return_list.append(temp.iloc[indx_rank[:self.num_stocks]].mean())                
            time_step += 1
#        return (np.array(return_list) + self.y_mean)*self.y_std
        return np.array(return_list) 

    def return_plot(self, return_list):
        """
            Method to generate return density plot and cumulative return trace plot
        """
        return_series = np.array(return_list)
        cumulative_ret = (return_series + 1).cumprod()              
        plt.plot(cumulative_ret, label='return')
        plt.xlabel('time_step')
        plt.ylabel('returns')
        plt.title('Cumulative Return Plot')
        plt.show()
        
        
        sns.distplot(return_series)
        plt.title('Return Distribution Plot')       
        plt.show()
       # self.cumulative_ret = cumulative_ret
        
      #  self.cumulative_ret = return_series 
                        
    def calc_sharpe_ratio(self):
        shortrate_data = pd.read_excel("macro_data.xlsx", "short_rate")
        shortrate_data = shortrate_data[::-1].reset_index(drop=True)

        start_date = self.ret_df['test_date'].iloc[0]
        end_date = self.ret_df['test_date'].iloc[-1]

        start_index = next(i for i in range(len(shortrate_data['Date'])) if (
            shortrate_data['Date'].iloc[i].month == start_date.month) & (shortrate_data['Date'].iloc[i].year == start_date.year))
        end_index = next(i for i in range(len(shortrate_data['Date'])) if (
            shortrate_data['Date'].iloc[i].month == end_date.month) & (shortrate_data['Date'].iloc[i].year == end_date.year))

        risk_free_rate = shortrate_data.iloc[start_index:end_index +
                                             1]['short_rate'].values / 100.
        risk_free_rate = np.reshape(risk_free_rate, [len(risk_free_rate), 1])

        ret_avg = np.mean(
            self.ret_df[self.ret_cols].values * 12 - risk_free_rate, axis=0)
        ret_std = np.sqrt(
            12) * np.std(self.ret_df[self.ret_cols].values.astype(float), axis=0)
        sharpe_ratio = ret_avg / ret_std

        sharpe_ratio_result = pd.DataFrame([ret_avg, ret_std, sharpe_ratio], index=["Expected Excess Return", "Standard Deviation", "Sharpe Ratio"],
                                           columns=self.ret_cols).T
        return sharpe_ratio_result

    def calc_max_drawdown(self):
        drawdown_df = (self.ret_cumu[self.ret_cols] - self.ret_cumu[self.ret_cols].cummax(
        )) / self.ret_cumu[self.ret_cols].cummax()
        max_drawdown = drawdown_df.min(axis=0)
        return max_drawdown


if __name__ == '__main__':

    raw_data = pd.read_csv("fun_tech_2013_median.csv")
    ML_object = MLEngineer(raw_data, num_stocks=40,
                           trading_window = 12, algorithm='LSTM', small_sample=False, pca =True)
    ML_object.data_processing()
       

##
#
#    plt.subplots(figsize=(10,10))
##    raw_data = pd.read_sas('/Users/caesa/Desktop/Research/Machine Learning/ML in Math Finance MK/2018/raw_data.sas7bdat')
#    raw_data = pd.read_sas('/Users/caesa/Desktop/Research/Machine Learning/ML in Math Finance MK/2020/Summer_ML/rpsdata_rfs.sas7bdat')
#    a = raw_data.groupby('DATE').count()['RET']
#    raw_data_2018 = pd.read_sas('/Users/caesa/Desktop/Research/Machine Learning/ML in Math Finance MK/2018/raw_data.sas7bdat')
#    b = raw_data_2018.groupby('DATE').count()['RET']
#    
#    raw_data.groupby('DATE').count()['RET'].plot()
#    raw_data_2018.groupby('DATE').count()['RET'].plot()
#    
#    
#    for i in a.index:
#        print(i)
#        print(b.loc[i])
#        print(a.loc[i])
        
    
    
    time_list =['2015-10', '2015-12']
    returns_long = ML_object.profit_construct(time_list, strategy ='long_equal_weighted')
    
    ML_object.return_plot(returns_long)
    
    returns_short = - np.array(ML_object.profit_construct(time_list, strategy ='short_equal_weighted'))

    ML_object.return_plot(returns_short)

