# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:39:59 2018
@author: GMittal
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


def create_dataset():
    dataset = pd.read_csv("Returns.csv")
    dataset = dataset.dropna(how='all')
    
    return dataset

if __name__ == "__main__":
   
   dataset = create_dataset()
    
   # Analyze Data
   print(dataset.shape)
   print(dataset.dtypes)
   print(dataset.head(20))
    
    # descriptions
   set_option('precision', 1)
   print(dataset.describe())

    # correlation
   set_option('precision', 2)
   print(dataset.corr(method='pearson'))
   
  # Data Visualization
   # histograms
   dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
   pyplot.show()
   
   # density
   dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False,
fontsize=1)
   pyplot.show()

   # box and whisker plots
   dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,
fontsize=8)
   pyplot.show()
   
   
   # scatter plot matrix
   scatter_matrix(dataset,figsize=[20,20],marker='x')
   pyplot.show()
   
   
 # Validation dataset
   isDataset = dataset[dataset['In Sample']==1]
   isDataset = isDataset.drop('In Sample', 1)
   
   # Split-out validation dataset
   isDataset = isDataset.dropna()
   array = isDataset.values
   XDataset = pd.DataFrame()
   XDataset['Percentage of ShOutI'] = isDataset['Percentage of ShOutI']
   XDataset['Percentage Change'] = isDataset['Percentage Change']
   XDataset = XDataset.dropna()
   X = XDataset.values                
   Y =  isDataset['Excess Return'].values
                 
   validation_size = 0.20
   seed = 7
   X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
   test_size=validation_size, random_state=seed)
   
   # Evaluate Algorithms
   # Test options and evaluation metric
   num_folds = 10
   seed = 7
   scoring = 'neg_mean_squared_error'

   
   # Spot-Check Algorithms
   models = []
   models.append(('LR', LinearRegression()))
   models.append(('LASSO', Lasso()))
   models.append(('EN', ElasticNet()))
   models.append(('KNN', KNeighborsRegressor()))
   models.append(('CART', DecisionTreeRegressor()))
   models.append(('SVR', SVR()))
   
   
   # evaluate each model in turn
   results = []
   names = []
   for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    X_train = X_train.reshape(len(X_train), 2)
    Y_train = Y_train.reshape(len(Y_train), 1)
    
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

   
   # Compare Algorithms
   fig = pyplot.figure()
   fig.suptitle('Algorithm Comparison')
   ax = fig.add_subplot(111)
   pyplot.boxplot(results)
   ax.set_xticklabels(names)
   pyplot.show()

   
   # Standardize the dataset
   pipelines = []
   pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
   pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
   pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
   pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
   pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
   pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
   results = []
   names = []
   for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

   # Compare Algorithms
   fig = pyplot.figure()
   fig.suptitle('Scaled Algorithm Comparison')
   ax = fig.add_subplot(111)
   pyplot.boxplot(results)
   ax.set_xticklabels(names)
   pyplot.show()

   
   # ensembles
   ensembles = []
   ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostRegressor())])))
   ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingRegressor())])))
   ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestRegressor())])))
   ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',
ExtraTreesRegressor())])))
   results = []
   names = []
   for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

   # Compare Algorithms
   fig = pyplot.figure()
   fig.suptitle('Scaled Ensemble Algorithm Comparison')
   ax = fig.add_subplot(111)
   pyplot.boxplot(results)
   ax.set_xticklabels(names)
   pyplot.show()
   
   # prepare the model
   scaler = StandardScaler().fit(X_train)
   rescaledX = scaler.transform(X_train)
   model = AdaBoostRegressor(random_state=seed, n_estimators=400)
   model.fit(rescaledX, Y_train)
   
   # transform the validation dataset
   X_validation = X_validation.reshape(len(X_validation),2)
   Y_validation = Y_validation.reshape(len(Y_validation), 1)
   rescaledValidationX = scaler.transform(X_validation)
   predictions = model.predict(rescaledValidationX)
   print(mean_squared_error(Y_validation, predictions))
   
   # live trading   

   osDataset = dataset[dataset['In Sample']==0]
   XosDataset = pd.DataFrame()
   XosDataset['Percentage of ShOutI'] = osDataset['Percentage of ShOutI']
   XosDataset['Percentage Change'] = osDataset['Percentage Change']
   XosDataset = XosDataset.dropna()
   X = XosDataset.values                
   #X = osDataset['Percentage of ShOutI'].values
   #X = X[:, None]
   osDataset['PredictedNextQuarterreturns'] = model.predict(X)
   osDataset['invest'] = np.where(abs(osDataset['PredictedNextQuarterreturns']) > 0.001, np.sign(osDataset['PredictedNextQuarterreturns']), 0)
   osDataset['pnl'] = osDataset['invest']*osDataset['Excess Return']*100000

   pnlSeries = osDataset.groupby('Date')['pnl'].sum()
   totalPNL = pnlSeries.sum()
   sharpe = pnlSeries.mean()/pnlSeries.std()*np.sqrt(252)