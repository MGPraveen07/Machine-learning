# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:50:29 2019

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Computer_data.csv')
dataset.info() 

# Encoding catagorical data
dataset.cd[dataset.cd == 'no'] = 0
dataset.cd[dataset.cd == 'yes'] = 1
dataset['cd'] = pd.to_numeric(dataset['cd'], errors ='coerce')
dataset.multi[dataset.multi == 'no'] = 0
dataset.multi[dataset.multi == 'yes'] = 1
dataset['multi'] = pd.to_numeric(dataset['multi'], errors ='coerce')
dataset.premium[dataset.premium == 'no'] = int(0)
dataset.premium[dataset.premium == 'yes'] = int(1)
dataset['premium'] = pd.to_numeric(dataset['premium'], errors ='coerce')
dataset.info()
X = dataset.iloc[: , 2:11].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)


# Fitting the multiple linear regression model to training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)

# Building optimal solution useing Backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((6259,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()