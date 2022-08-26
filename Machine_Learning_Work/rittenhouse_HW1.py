#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:35:07 2022

@author: Simone Rittenhouse
"""

# Introduction to Machine Learning
# Homework Assignment 1
# 2/16/2022

# importing packages

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# loading data

data = np.genfromtxt('/Users/simonerittenhouse/Desktop/housingUnits.csv', delimiter = ',')
df = pd.read_csv('/Users/simonerittenhouse/Desktop/housingUnits.csv')

# inspecting data
print(df.head()) 

#%%

# QUESTION ONE
print()

# plotting correlations and getting Pearson's r
print(pd.plotting.scatter_matrix(df, figsize = (15,15)), '\n')

correlations = df.corr()
print(correlations)
print(correlations['median_house_value'])

# plotting histograms
fig, axs = plt.subplots(1, df.shape[1], figsize = (30,4))

for i, feature in enumerate(df):
    axs[i].hist(df[feature], bins = 21)

index = 0
titles = list(enumerate(df))
for ax in axs:
    ax.set(title = titles[index][1])
    index += 1
  
axs[0].set(ylabel = 'Frequency')
plt.show()

#%%

# QUESTION TWO
print()

# testing both ways of standardizing

standardizeFeatures = ['population', 'households']

for feature in standardizeFeatures:
    df_temp = df.copy()
    df_temp['total_rooms'] = df_temp['total_rooms']/df_temp[feature]
    df_temp['total_bedrooms'] = df_temp['total_bedrooms']/df_temp[feature]
    
    print(df_temp.corr()['median_house_value'])
    
    plt.figure(figsize = (15,7))
    
    x = df_temp['total_rooms'].to_numpy()
    y = df_temp['median_house_value'].to_numpy()
    a, b = np.polyfit(x,y, 1)
    plt.subplot(1,2,1)
    plt.scatter(x, y)
    plt.plot(x, a*x+b, color = 'r')
    plt.xlabel('Total Rooms/'+feature, fontsize = 15)
    plt.ylabel('Median House Value', fontsize = 15)
    plt.title("r = {:.3f}".format(np.corrcoef(x,y)[0,1]))

    x = df_temp['total_bedrooms'].to_numpy()
    y = df_temp['median_house_value'].to_numpy()
    a, b = np.polyfit(x,y, 1)
    plt.subplot(1,2,2)
    plt.scatter(x, y)
    plt.plot(x, a*x+b, color = 'r')
    plt.xlabel('Total Bedrooms/'+feature, fontsize = 15)
    plt.ylabel('Median House Value', fontsize = 15)
    plt.title("r = {:.3f}".format(np.corrcoef(x,y)[0,1]))
    plt.show()
    
    X = df_temp.loc[:,['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']]
    y = df_temp.loc[:,'median_house_value']

    multReg = LinearRegression().fit(X,y)

    print('\nAll Predictors in ' + feature + ' Standardized Model')
    print('R-squared:', multReg.score(X,y))
    print('Intercept:', multReg.intercept_)
    print('Betas:', multReg.coef_, '\n')

# final decision (standardizing by population)
data[:,1] = data[:,1]/data[:,3]
data[:,2] = data[:,2]/data[:,3]

#%%

# QUESTION THREE
print()

y = data[1:,7]
RSqrs = []
feature_names = ['House Age', 'Total Rooms', 'Total Bedrooms', 'Population', 'Households', 'Income', 'Distance to Ocean']

for i in range(7):
    feature = data[1:,i:i+1]
    
    model = LinearRegression().fit(feature, y)
    rSqr = model.score(feature, y)
    yHat = model.predict(feature)

    plt.figure()
    plt.plot(yHat, y, 'o', markersize = .75)
    plt.xlabel('Prediction from model')
    plt.ylabel('Actual House Value')
    plt.title('R^2 for {} = {:.3f}'.format(feature_names[i], rSqr))
    plt.show()
    
    plt.figure()
    plt.plot(feature, y, 'o', markersize = .75)
    m,b = np.polyfit(feature[:,0], y, 1)
    plt.plot(feature, m*feature + b, 'r')
    plt.xlabel(feature_names[i])
    plt.ylabel('Actual House Value')
    plt.title('R^2 for {} = {:.3f}'.format(feature_names[i], rSqr))
    plt.show()
    
    RSqrs.append(rSqr)

RSqrDf = pd.DataFrame({'R-Squared': RSqrs}, index = [column for column in df.columns if column != 'median_house_value'])

print(RSqrDf)

#%%

# QUESTION FOUR
print()

X = data[1:,:7]

multReg = LinearRegression().fit(X,y)

print('\nAll Predictors in Multiple Regression Model')
print('R-squared:', multReg.score(X,y))
print('Intercept:', multReg.intercept_)
print('Betas:', multReg.coef_)

# plotting results

yHat = multReg.predict(X)
plt.figure()
plt.plot(yHat, y, 'o', markersize = .75)
plt.xlabel('Prediction from model')
plt.ylabel('Actual house value')
plt.title('R^2 for All Predictors = {:.3f}'.format(multReg.score(X,y)))
plt.show()

#%%

# QUESTION FIVE
print()

plt.figure(figsize = (15,7))

# correlation between standardized variables 2 and 3
x = data[1:,1]
y = data[1:,2]
a, b = np.polyfit(x,y, 1)
plt.subplot(1,2,1)
plt.scatter(x, y)
plt.plot(x, a*x+b, color = 'r')
plt.xlabel('Total Rooms', fontsize = 15)
plt.ylabel('Total Bedrooms', fontsize = 15)
plt.title("r = {:.3f}".format(np.corrcoef(x,y)[0,1]))

# correlation between variables 4 and 5
x = data[1:,3]
y = data[1:,4]
a, b = np.polyfit(x,y, 1)
plt.subplot(1,2,2)
plt.scatter(x, y)
plt.plot(x, a*x+b, color = 'r')
plt.xlabel('Population', fontsize = 15)
plt.ylabel('Number of Households', fontsize = 15)
plt.title("r = {:.3f}".format(np.corrcoef(x,y)[0,1]))
plt.show()

df_pop = df.copy()
df_pop['total_rooms'] = df_pop['total_rooms']/df_pop['population']
df_pop['total_bedrooms'] = df_pop['total_bedrooms']/df_pop['population']

correlationsStandard = df_pop.corr()
print(correlationsStandard)

#%%

# EXTRA CREDIT A

fig, axs = plt.subplots(2, int(df_pop.shape[1]/2), figsize = (20,10))
titles = list(enumerate(df_pop))

row = 0
col = 0
index = 0
for i, feature in enumerate(df_pop):
    axs[row, col].hist(df_pop[feature], bins = 15)
    axs[row, col].set(title = titles[index][1])
    col += 1
    if col == 4:
        row = 1
        col = 0
    index += 1
  
axs[0,0].set(ylabel = 'Frequency')
axs[1,0].set(ylabel = 'Frequency')
plt.show()

#%%

# EXTRA CREDIT B

# histogram of outcome
plt.figure()
plt.hist(df['median_house_value'], bins = 31)
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Frequency of Median House Values')
plt.show()
