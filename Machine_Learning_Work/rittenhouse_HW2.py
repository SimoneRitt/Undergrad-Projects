#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:51:50 2022

@author: simonerittenhouse
"""

# Introduction to Machine Learning
# Homework Assignment 1
# 3/9/2022

# importing packages

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

#%%
# loading data

df = pd.read_csv('/Users/simonerittenhouse/Desktop/techSalaries2017.csv')

#%%
# Exploratory Analysis

print(df.head())

corrUnprocessed = df.corr()
print(corrUnprocessed['totalyearlycompensation'])

# collinearity
print('\nr (SAT & GPA) =', np.corrcoef(df['GPA'].to_numpy(), df['SAT'].to_numpy())[1,0], '\n')

#print(pd.plotting.scatter_matrix(df, figsize = (30,30)), '\n')

#%% 
#Recoding Gender and Zodiac

def recode(gender):
    if gender == 'Male':
        return 0
    elif gender == 'Female':
        return 1
    else:
        return None

df['gender'] = df['gender'].apply(recode)

zodiac = pd.get_dummies(df['Zodiac'])
df = pd.concat([df, zodiac], axis = 1)
df = df.drop(['Zodiac'], axis = 1)

#%%
# Dropping Missing Values and Extra Columns

# dropping categorical and uninformative variables
dfPred = df.drop(['company', 'location', 'basesalary', 'bonus', 'stockgrantvalue', 
                  'Race', 'title', 'Education'], axis = 1)

# finding NaNs
print(df.isnull().sum(axis = 0), '\n')

dfPrednotNan = dfPred[(~df['Education'].isna()) & (~df['Race'].isna()) & (~df['gender'].isna())]
print(dfPrednotNan.shape, '\n')

# preventing overdetermined models (by dropping one dummy variable for education, race, and zodiac)
print(dfPrednotNan.corr()['totalyearlycompensation'])
# lowest correlated race dummy var = Race_Two_Or_More
# lowest correlated education dummy var = Highschool
# lowest correlated zodiac dummy var = 4

dfPrednotNan = dfPrednotNan.drop(['Race_Two_Or_More', 'Highschool', 4], axis = 1)

#%% Standardizing

dataProcessed= dfPrednotNan.to_numpy()
dataNorm = (dataProcessed - np.mean(dataProcessed, axis=0))/np.std(dataProcessed, axis=0)

#%%
# QUESTION ONE

# multiple regression model

y = dataNorm[:,0]
X = dataNorm[:,1:]

multReg = LinearRegression().fit(X,y)

print('\nAll Predictors in Multiple Regression Model')
print('R-squared:', multReg.score(X,y))
print('Intercept:', multReg.intercept_)
print('Betas:', multReg.coef_, '\n')

# plotting results

yHat = multReg.predict(X)
plt.figure()
plt.plot(yHat, y, 'o', markersize = .75)
plt.xlabel('Prediction from model')
plt.ylabel('Actual Total Yearly Compensation')
plt.title('R^2 for All Predictors (OLS) = {:.3f}'.format(multReg.score(X,y)))
plt.show()

# single regression models 

RSqrs = []
index = 0
for i in range(1,len(list(dfPrednotNan.columns))):
    feature = dataNorm[:,i:i+1]
    
    model = LinearRegression().fit(feature, y)
    rSqr = model.score(feature, y)
    yHat = model.predict(feature)
    
    RSqrs.append(rSqr)
    index += 1

RSqrDf = pd.DataFrame({'R-Squared': RSqrs}, index = dfPrednotNan.columns[1:])
print(RSqrDf)

yearsexperience = dataNorm[:,1:2]
plt.figure()
plt.plot(yearsexperience, y, 'o', markersize = .75)
m,b = np.polyfit(yearsexperience[:,0], y, 1)
plt.plot(yearsexperience, m*yearsexperience + b, 'r')
plt.xlabel('Years of Experience')
plt.ylabel('Total Yearly Compensation')
plt.title('R^2 for Years of Experience = {:.3f}'.format(RSqrs[0]))
plt.show()

#%%
# QUESTION TWO

# splitting data
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

# finding optimal lambda
lambdas = np.linspace(0,400,1001)
cont = np.empty([len(lambdas),2])*np.NaN

for ii in range(len(lambdas)):
    ridgeModel = Ridge(alpha=lambdas[ii]).fit(xTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = metrics.mean_squared_error(yTest,ridgeModel.predict(xTest))
    cont[ii,1] = error

lambdaOp = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]

plt.plot(cont[:,0],cont[:,1])
plt.plot(lambdaOp, cont[np.argmax(cont[:,1]==np.min(cont[:,1])), 1], 'ro', markersize = 5)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Ridge regression')
plt.show()
print('\nOptimal lambda:',lambdaOp)

# ridge regression
ridgeReg = Ridge(alpha = lambdaOp).fit(X,y)

print('\nAll Predictors in Ridge Regression Model')
print('R-squared:', ridgeReg.score(X,y))
print('Intercept:', ridgeReg.intercept_)
print('Betas:', ridgeReg.coef_)

# plotting results

yHat = ridgeReg.predict(X)
plt.figure()
plt.plot(yHat, y, 'o', markersize = .75)
plt.xlabel('Prediction from model')
plt.ylabel('Actual Total Yearly Compensation')
plt.title('R^2 for All Predictors (Ridge) = {:.3f}'.format(ridgeReg.score(X,y)))
plt.show()

#%%
# QUESTION THREE
print()

# splitting data
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

# finding optimal lambda
lambdas = np.linspace(-1,5,1001)
cont = np.empty([len(lambdas),2])*np.NaN

for ii in range(len(lambdas)):
    lassoModel = Lasso(alpha=lambdas[ii], tol=1).fit(xTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = metrics.mean_squared_error(yTest,lassoModel.predict(xTest))
    cont[ii,1] = error

lambdaOp = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]

plt.plot(cont[:,0],cont[:,1])
plt.plot(lambdaOp, cont[np.argmax(cont[:,1]==np.min(cont[:,1])), 1], 'ro', markersize = 5)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Lasso regression')
plt.show()
print('\nOptimal lambda:',lambdaOp)

# lasso regression
lassoReg = Lasso(alpha = lambdaOp).fit(X,y)

print('\nAll Predictors in Lasso Regression Model')
print('R-squared:', lassoReg.score(X,y))
print('Intercept:', lassoReg.intercept_)
print('Betas:', lassoReg.coef_, '\n')

# plotting results

yHat = lassoReg.predict(X)
plt.figure()
plt.plot(yHat, y, 'o', markersize = .75)
plt.xlabel('Prediction from model')
plt.ylabel('Actual Total Yearly Compensation')
plt.title('R^2 for All Predictors (Lasso) = {:.3f}'.format(lassoReg.score(X,y)))
plt.show()

# finding number of variables dropped by model

index = 0
droppedCount = 0
for i in lassoReg.coef_:
    if i.round(8) == 0.0:
        droppedCount += 1
        
    index += 1
print('Number of variables dropped:', droppedCount)
#%%
# QUESTION FOUR

# model with just income as predictor
print('\nPREDICTING GENDER USING LOGISTIC REGRESSION:')

dfGender = dfPrednotNan

# showing imbalance
plt.figure()
plt.bar([0,1], dfGender['gender'].value_counts(ascending = True), color = ['orange', 'c'], tick_label = ['Women', 'Men'])
plt.ylabel('Count')
plt.title('Gender Class Counts')
plt.show()

dfGender = dfGender.to_numpy()

# splitting data and normalizing predictors
train_data, val_data = model_selection.train_test_split(dfGender, test_size = 0.2)

y_train = train_data[:,3].reshape(-1,1)
y_val  = val_data[:,3].reshape(-1,1)

train_data = np.delete(train_data, 3, axis = 1)
val_data = np.delete(val_data, 3, axis = 1)

# storing betas for compensation and AUC
betasCompensation = []
numPredictors = []
AUCs = []
largeBeta = 0

# iterating through models with different numbers of predictors
for i in range(len(train_data[0,:])):
    print()
    X_train = train_data[:,0:i+1]
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_val = val_data[:,0:i+1]
    X_val = (X_val - np.mean(X_val)) / np.std(X_val)
    
    # building model and predicting
    model = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
    model.fit(X_train, y_train.ravel())
    
    pred = model.predict(X_val)
    
    numPred = len(X_train[0,:])
    print('Number of Predictor(s) = {}'.format(numPred))
    
    # betas
    print('Betas (Normalized): ', model.coef_)
    betasCompensation.append(model.coef_[0][0])
    
    # finding threshold where beta for compensation is smaller than other betas
    if len(model.coef_[0]) > 1 and abs(model.coef_[0][0]) == max(map(abs, model.coef_[0])):
        largeBeta = numPred
    
    AUC = metrics.roc_auc_score(y_val, pred)
    print('Area Under Curve =', AUC)
    AUCs.append(AUC)
    
    if numPred == 1:
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_val, pred))
        precision = metrics.precision_score(y_val, pred)
        print("Precision = {:0.1f}%".format(100 * precision))
        recall = metrics.recall_score(y_val, pred)
        print("Recall    = {:0.1f}%".format(100 * recall))
    
    numPredictors.append(numPred)
 
# plotting beta for compensation across models
plt.figure(figsize = (10,5))
plt.plot(numPredictors, betasCompensation)
plt.axvline(x=largeBeta, color = 'r', linestyle = '--', label='threshold where model has larger betas than compensation')
plt.xlabel('Number of Predictors in Model')
plt.ylabel('Beta Value of Total Yearly Compensation')
plt.title('Beta of Total Compensation Across Number of Predictors')
plt.legend(loc = 'best')
plt.show()

#%%
# QUESTION FIVE

print('\nPREDICTING HIGH AND LOW PAY:')

# getting high/low outcome variable
median = np.median(df['totalyearlycompensation'])

df['highLowPay'] = np.where(df['totalyearlycompensation'] > median, 1, 0)
dfHighLowPay = df[['highLowPay', 'yearsofexperience', 'Age', 'Height', 'SAT', 'GPA']]

dataPay = dfHighLowPay.to_numpy()

# splitting data
train_data, val_data = model_selection.train_test_split(dataPay, test_size = 0.2)

y_train = train_data[:,0]
X_train = train_data[:,1:]

y_val = val_data[:,0]
X_val = val_data[:,1:]

# building a model and predicting
model = LogisticRegression(solver = 'liblinear').fit(X_train, y_train)
pred = model.predict(X_val)

# betas
print('Betas:', model.coef_)

# metrics
accuracy = metrics.accuracy_score(y_val, pred) 
print("Accuracy = {:0.1f}%".format(accuracy * 100))

precision = metrics.precision_score(y_val, pred)
print("Precision = {:0.1f}%".format(100 * precision))

recall = metrics.recall_score(y_val, pred)
print("Recall    = {:0.1f}%".format(100 * recall))

AUC = metrics.roc_auc_score(y_val, pred)
print('Area Under Curve =', AUC)

# confusion matrix

conf_matrix = metrics.confusion_matrix(y_val, pred)
print("\nConfusion matrix = ")
print(conf_matrix)


#%%
# EXTRA CREDIT A

# plotting histograms
fig, axs = plt.subplots(1, 3, figsize = (10,5))
dfHist = df[['basesalary', 'Height', 'Age']]
titles = list(enumerate(dfHist))

for i, feature in enumerate(dfHist):
    axs[i].hist(dfHist[feature], bins = 50)
    axs[i].set(title = titles[i][1])
  
axs[0].set(ylabel = 'Frequency')
plt.show()

#%%
# EXTRA CREDIT B
print()

# getting company names
companies = ['Amazon', 'Apple', 'Microsoft', 'Google', 'Netflix', 'eBay', 
             'Facebook', 'Uber', 'Intel', 'Salesforce', 'Sony', 'Airbnb',
             'LinkedIn', 'Samsung', 'Cisco', 'Tesla']

# getting mean income and number of software engineers per company
companyMeanIncome = []
numWomen = []

for i in range(len(companies)):
    tempDf = df[df['company'] == companies[i]]
    companyMeanIncome.append(tempDf['totalyearlycompensation'].mean())
    numWomen.append(tempDf['gender'].value_counts()[1])

# plotting income and scatter plot
plt.figure(figsize = (15, 6))
plt.bar(companies, companyMeanIncome)
plt.title('Mean Total Yearly Compensation Across Major Companies')
plt.show()

plt.scatter(companyMeanIncome, numWomen)
plt.xlabel('Mean Yearly Compensation')
plt.ylabel('Number of Women')
plt.title('Mean Compensation vs. Number of Women: r = ' 
          + str(np.corrcoef(companyMeanIncome, numWomen)[0,1].round(2)))
plt.show()

# getting extreme values
for i in range(len(companyMeanIncome)):
    if companyMeanIncome[i] == min(companyMeanIncome):
        print('Lowest Average Yearly Compensation (${}):'.format(round(min(companyMeanIncome),2)), 
              companies[i])
    if companyMeanIncome[i] == max(companyMeanIncome):
        print('Highest Average Yearly Compensation (${}):'.format(round(max(companyMeanIncome),2)), 
              companies[i])
        
# conducting t-test for high vs. low income companies (Netflix and Samsung)
import scipy.stats as stats

dfNet = df[df['company'] == 'Netflix']
dfSam = df[df['company'] == 'Samsung']

t, p = stats.ttest_ind(dfNet['totalyearlycompensation'], dfSam['totalyearlycompensation'])
print('\nt statistic for independent samples:', t)
print('associated p-value:', p)
