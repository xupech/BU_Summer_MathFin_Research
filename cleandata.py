#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Sun Jun  7 14:57:04 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

sns.set(color_codes=True)


class cleandata:

    def __init__(self, raw_data, method='mean', selected_columns=None, small_sample=False):
        """
            Method to initliaze data cleanning

            param: raw_data, raw data outputted from sas as pandas dataframe, must have columns Date in sas format

            param: method, method used to fill in missing value

            param: seletec_clumns, a set of columns specified by the user, default is using all the data
        """
        if small_sample:
            raw_data = raw_data.iloc[-5000:,:]
        
        self.fill_in_method = method
        
        raw_data = raw_data.dropna(how='all',axis=1)
        
        if selected_columns is not None:
            try:
                self.raw_data = raw_data[selected_columns]

            except:

                print(
                    'Warnings: raw_data does not have all the specified selected columns')

                self.raw_data = raw_data
        else:
            self.raw_data = raw_data
        
    def get_float_factor(self):
        
        """
            Method to get continuous predictive factors
        """

        non_float = self.raw_data.dtypes[self.raw_data.dtypes !=
                                 'float64'].index                         
        self.data_float = self.raw_data.drop(non_float, axis=1)        
    
           
        return self.data_float     
    def get_category_factor(self):
        """
            Method to get category/indicator variables
        """
        indicator_flag = self.raw_data.dtypes == 'int64'
        
        data_indicator = self.raw_data.loc[:,indicator_flag]
        
        self.data_indicator = data_indicator
        
        return data_indicator
                
    def get_clean_data(self, standardrize= True):
        """
            Method to aggregate all claen data
        """
        
        if standardrize:
           self.y_mean = np.mean(self.data_float['RET'])
           self.y_std = np.std(self.data_float['RET'])
           self.data_float = (self.data_float - self.data_float.mean())/self.data_float.std()
           
        self.clean_data = pd.concat([self.data_indicator,self.data_float, self.raw_data['DATE']],axis=1)
        
        
        

    def fill_in_missing(self, method='mean'):
        """
            Method to fill in missing continuous data by using fill in method defined by the user
        """
        if self.fill_in_method == 'mean':

            self.data_float.fillna(self.data_float.mean(), inplace=True)
            
        elif self.fill_in_method == 'median':

            self.data_float.fillna(self.data_float.median(), inplace=True)
            
        
    def cap_inf(self):
        """
            Method to cap data outside of 3 sigma to infinity
            
            currently implemented to drop the infinity data as there are only two
        """
        self.get_float_factor()
        self.data_float = self.data_float.loc[:, np.sum(np.isinf(self.data_float))==0]

    def correct_date(self):
        """
            Method to correct date into pandas dt64 format
        """
        if self.raw_data['DATE'].dtypes=='O':
            self.raw_data['DATE'] = pd.to_datetime(self.raw_data['DATE'], format = '%Y-%m').dt.to_period('M')
            
    def calc_vif(self, show_dist=False):
        """
            Method to calculate varaince inflation factors
        """

        
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
            self.data_float.values, i) for i in range(self.data_float.shape[1])]
        vif["features"] = self.data_float.columns
        
        return vif
    
    def del_missing_indicators(self):
        """
            Method to drop indicator variables that have missing data (implemented since fill_in_method doesn't apply for indicator variables)
        """

        self.get_category_factor()
        self.data_indicator.dropna(how='any', axis=1)
        

    def __main__(self):
        
        self.correct_date()
        print('date format corrected')        
        
        # cleanining for category variables
        self.del_missing_indicators()
        print('missing indicator variables deleted')        

        
        self.cap_inf()
        print('continous variables containing infinity deleted')        
        
        # cleanining for continuous variables        
        self.fill_in_missing()
        print('missing continuous variables filled in with ' + str(self.fill_in_method) + '')


#        return self.raw_data

if __name__ == '__main__':
#    col_selected = pd.read_excel("selected_column.xlsx")
#    col_set = set(col_selected['col_name'])
    
    
    
    raw_data = pd.read_csv('fun_tech_2013_median.csv')
    clean_data = cleandata(raw_data, method = 'median', small_sample = True)
    clean_data.__main__()



#    raw_data2 = pd.read_csv('fun_tech_2013_median.csv')
#    # To show vif distribution
#    vif_df = clean_data.calc_vif()    
#    vif_inf_flags = np.isinf(vif_df['VIF Factor'])
#    vif_df.loc[vif_inf_flags==False,:].plot.density()
    
    clean_data.get_clean_data()
    
    clean_data.clean_data.groupby('DATE').count()['ret'].plot()
    
    



