"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Wed Jun  3 18:44:48 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from cleandata import *
class SVM_Machine:
    """
    SVM machine class
    """
    def __init__(self, train_x, train_y, test_x, test_y, Tunning_Cs=[0.001, 0.01, 0.1, 1, 10]):
        """
        SVM machine init class

        param: train_x, np.array of training data 

        param: train_y, np.array of training labels 

        param: test_x, np.array of testing data

        param: test_y, np.array of testing_labels     

        """        
        self.Cs = Tunning_Cs
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x        
        self.test_y = test_y
        self.model = svm.SVR(kernel='rbf', gamma='auto')
        
    def training(self):
        """
            Function to train the model
        """
        self.model.fit(self.train_x, self.train_y)
        
    def tunning(self, nfolds=5):        
        """
            Function to hyper tunning the parameter C
        """
        Cs = self.Cs
        param_grid = {'C': Cs}
        grid_search = GridSearchCV(self.model, param_grid, cv=nfolds)
        grid_search.fit(self.train_x, self.train_y)
        
        self.model.C = grid_search.best_params_['C']
        return grid_search.best_params_
        
        
    def predict(self):
        """
            Function to predict y_hat using the trained model
        """        
        return self.model.predict(self.test_x)
        
        

from sklearn.model_selection import train_test_split
    
if __name__ == '__main__':
    
    raw_data = pd.read_csv("constituents_2013_fund_tech.csv")
    clean_data = cleandata(raw_data, 'median', small_sample= True)
    clean_data.__main__()
    
    clean_data=clean_data.data_float
    
    y = clean_data['ret']
    x = clean_data.drop(['ret'], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)    
    
    model = SVM_Machine(x_train, y_train, x_test, y_test, Tunning_Cs=[0.001, 0.01, 0.1, 1, 10])    
    model.tunning()        
    model.training()    
    y_hat = model.predict()
    
    plt.plot(range(len(y_hat)), y_hat, label='predict')
    
#    plt.plot(range(len(y_hat)) , y_train, label='Real')    
    plt.plot(range(len(y_hat)) , y_test, label='Real')    
    
    plt.legend()


"""
    flags = (clean_data.dtypes == 'float64') | (clean_data.dtypes == 'int64')
    
    clean_data = clean_data.loc[:,flags]
    clean_data = clean_data.dropna(axis=1)
    
    clean_data = clean_data.loc[:, np.sum(np.isinf(clean_data))==0]
"""