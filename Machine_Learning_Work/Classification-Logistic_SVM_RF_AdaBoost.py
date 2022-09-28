#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:58:02 2022

@author: simonerittenhouse
"""

# Introduction to Machine Learning
# 4/6/2022

# importing packages

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics, model_selection, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

# loading data
df = pd.read_csv('/Users/simonerittenhouse/Desktop/diabetes.csv')
data = df.to_numpy()

# inspecting data
print(df.columns)
print(df.shape)

# exploratory analysis
correlations = df.corr()
print(correlations)
#print(pd.plotting.scatter_matrix(df, figsize = (30,30)), '\n')

# setting random seed
np.random.seed(13)

# splitting data (test size = 30%)
train_data, val_data = model_selection.train_test_split(data, test_size = 0.3)

y_train = train_data[:,0]
y_val  = val_data[:,0]

X_train = np.delete(train_data, 0, axis = 1)
X_val = np.delete(val_data, 0, axis = 1)

# checking imbalance in outcome variable (there should be fewer diabetes patients)
print(df['Diabetes'].value_counts())
df['Diabetes'].value_counts().plot(kind = 'bar', color = ['mediumturquoise', 'orangered'])
plt.title('Class Counts for Target Variable: Diabetes')
plt.ylabel('Count')
plt.xlabel('Class')
plt.xticks(ticks = [0,1], labels = ['not diagnosed', 'diagnosed'], rotation = 0)
plt.show()


#%% FUNCTIONS FOR DROPPING PREDICTORS AND PLOTTING

def predDrop(metric, model, X_train = X_train, X_val = X_val, y_train = y_train, y_val = y_val):
    performance = []
    totalPred = X_train.shape[1]
    for col in range(totalPred):
        X_train_temp = X_train.copy()
        X_val_temp = X_val.copy()
    
        X_train_temp = np.delete(X_train_temp, col, axis = 1)
        X_val_temp = np.delete(X_val_temp, col, axis = 1)
    
        modelPredDrop = model.fit(X_train_temp, y_train)
        
        if metric == 'AUC':
            performance.append(metrics.roc_auc_score(y_val, modelPredDrop.predict_proba(X_val_temp)[:,1]))
        elif metric == 'AUC SVM':
            performance.append(metrics.roc_auc_score(y_val, modelPredDrop.decision_function(X_val_temp))) 
        elif metric == 'Accuracy':
            performance.append(metrics.accuracy_score(y_val, modelPredDrop.predict(X_val_temp)))
        # to see progress of function (because of long runtime)
        complete = ((col+1)/totalPred)*100
        print(format(str(col+1), '>2s')+' of '+str(totalPred)+' predictors dropped -', str(round(complete, 2))+'% complete')
    return performance
            
def bestPred(performance, metric, df = df):
    decrease = [(performance[0] - x) for x in performance[1:]]
    print('\nBest Predictor:', list(df.columns[1:])[decrease.index(max(decrease))])
    print(metric+' of full model:', performance[0])

def plottingPerformanceOverPred(performance, title, metric, df = df):
    predDropped = list(df.columns)
    predDropped[0] = 'None Dropped'

    plt.figure(figsize = (10,4))
    plt.plot(np.arange(len(performance)), performance)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Predictor Dropped')
    plt.xticks(ticks = np.arange(len(performance)), labels = predDropped, rotation = 80)
    plt.show()

#%% QUESTION ONE
print('\nPREDICTING WITH LOGISTIC REGRESSION:')

# storing AUC for all models
AUCsLog = []

# building model
modelLogReg = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
modelLogReg.fit(X_train, y_train)
pred = modelLogReg.predict(X_val)

AUCsLog.append(metrics.roc_auc_score(y_val, modelLogReg.predict_proba(X_val)[:,1]))
LogAcc = metrics.accuracy_score(y_val, pred)

# dropping predictors
AUCsLog.extend(predDrop('AUC', modelLogReg))

# finding predictor that dropped performance most (best predictor)
bestPred(AUCsLog, 'AUC')

# plotting trend in AUC
plottingPerformanceOverPred(AUCsLog, 'Logistic Regression: AUC While Dropping Predictors', 'AUC')

#%% QUESTION TWO
print('\nPREDICTING WITH SUPPORT VECTOR MACHINES:')

# getting good value for slack variable
C = np.linspace(1.5, 1.001, 5)
C = np.flip(np.append(C, 1/(2**np.arange(0, 10))))

k = 5 # cross-validation (k fold)
cv_scores = np.zeros(len(C))
split = model_selection.KFold(k)
for i in range(len(C)):
    svm = LinearSVC(C = C[i], dual = False)
    cv_scores[i] = np.mean(model_selection.cross_val_score(svm, X_train, y_train, cv = split, scoring = 'roc_auc'))

# finding optimal C value (where CV Score plateaus)
optC = C[0]
threshold = 0.00001
for i in range(1,len(C)-1):
    if cv_scores[i] - cv_scores[np.where(C == optC)] < threshold:
        break
    else:
        optC = C[i]

print('Optimal Slack Value:', optC)
# plotting slack variable
plt.plot(C.astype('str'), cv_scores, 'b-x')
plt.xlabel(r'$C$')
plt.ylabel(r'Score')
plt.xticks(range(len(C)), labels=np.round_(C, decimals = 3).astype('str'), rotation = 30)
plt.title(r'{:d}-Fold CV Score for Linear SVM'.format(k))
plt.grid()
plt.show()

#%%

# storing AUC for all models
AUCsSVM = []

# building model
modelSVM = LinearSVC(C = optC, dual = False).fit(X_train, y_train)
pred = modelSVM.predict(X_val)

AUCsSVM.append(metrics.roc_auc_score(y_val, modelSVM.decision_function(X_val)))
SVMAcc = metrics.accuracy_score(y_val, pred)

# dropping predictors
AUCsSVM.extend(predDrop('AUC SVM', LinearSVC(C = optC, dual = False)))

# finding predictor that dropped performance most (best predictor)
bestPred(AUCsSVM, 'AUC')

# plotting trend in AUC
plottingPerformanceOverPred(AUCsSVM, 'SVM: AUC While Dropping Predictors', 'AUC')

#%% QUESTION THREE
print('\nPREDICTING WITH SINGLE DECISION TREE:')

# storing AUC for all models
AUCsTree = []

# building model
modelTree = tree.DecisionTreeClassifier(criterion='gini', random_state=0).fit(X_train, y_train)
pred = modelTree.predict(X_val)

AUCsTree.append(metrics.roc_auc_score(y_val, modelTree.predict_proba(X_val)[:,1]))
TreeAcc = metrics.accuracy_score(y_val, pred)

# dropping predictors
AUCsTree.extend(predDrop('AUC', tree.DecisionTreeClassifier(criterion='gini', random_state=0)))

# finding predictor that dropped performance most (best predictor)
bestPred(AUCsTree, 'AUC')

# plotting trend in AUC
plottingPerformanceOverPred(AUCsTree, 'Single Decision Tree: AUC While Dropping Predictors', 'AUC')

#%% QUESTION FOUR
print('\nPREDICTING WITH RANDOM FOREST:')

# storing AUC for all models
AUCsForest = []

# grid search for best parameters
grid=dict()
grid['n_estimators'] = [50, 100]
grid['max_samples'] = [.5, .75, 0.999]
grid['max_features'] = [0.25, 0.5, 0.75]
model = RandomForestClassifier(bootstrap=True, criterion='gini')
gridSearch = model_selection.GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='roc_auc')
result = gridSearch.fit(X_train, y_train)
params = result.best_params_
print('Highest AUC =', result.best_score_)
print('Optimal Parameters:', params)

#%%

# building model
modelForest = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                     max_samples=params['max_samples'], 
                                     max_features=params['max_features'],
                                     bootstrap=True, criterion='gini')
modelForest.fit(X_train, y_train)
pred = modelForest.predict(X_val)

AUCsForest.append(metrics.roc_auc_score(y_val, modelForest.predict_proba(X_val)[:,1]))
ForestAcc = metrics.accuracy_score(y_val, pred)

# dropping predictors
AUCsForest.extend(predDrop('AUC', modelForest))

# finding predictor that dropped performance most (best predictor)
bestPred(AUCsForest, 'AUC')

# plotting trend in AUC
plottingPerformanceOverPred(AUCsForest, 'Random Forest: AUC While Dropping Predictors', 'AUC')

#%% QUESTION FIVE
print('\nPREDICTING WITH ADABOOST:')

# storing AUC for all models
AUCsBoost = []

# grid search for best parameters
grid=dict()
grid['n_estimators'] = [100, 500]
grid['learning_rate'] = [0.001, 0.01, 0.1, 1]
model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1))
gridSearch = model_selection.GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='roc_auc')
result = gridSearch.fit(X_train, y_train)
params = result.best_params_
print('Highest AUC =', result.best_score_)
print('Optimal Parameters:', params)

#%%

# building model
modelBoost = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), 
                                n_estimators=params['n_estimators'], 
                                learning_rate=params['learning_rate'])
modelBoost.fit(X_train, y_train)
pred = modelBoost.predict(X_val)

AUCsBoost.append(metrics.roc_auc_score(y_val, modelBoost.predict_proba(X_val)[:,1]))
BoostAcc = metrics.accuracy_score(y_val, pred)

# dropping predictors
AUCsBoost.extend(predDrop('AUC', modelBoost))

# finding predictor that dropped performance most (best predictor)
bestPred(AUCsBoost, 'AUC')

# plotting trend in AUC
plottingPerformanceOverPred(AUCsBoost, 'Gradient Boosting: AUC While Dropping Predictors', 'AUC')

#%% EXTRA CREDIT A
print('\nBEST MODEL:')

# plotting AUC and accuracy
Models = ['Logistic Regression', 'SVM', 'Single Decision Tree', 'Random Forest', 'AdaBoost']
Accuracies = [LogAcc, SVMAcc, TreeAcc, ForestAcc, BoostAcc]
AUCs = [AUCsLog[0], AUCsSVM[0], AUCsTree[0], AUCsForest[0], AUCsBoost[0]]

barColors = ('mediumturquoise '*len(Models)).rstrip().split(' ')
AccColors = barColors.copy()
AccColors[Accuracies.index(max(Accuracies))] = 'orangered'
AUCColors = barColors.copy()
AUCColors[AUCs.index(max(AUCs))] = 'orangered'

plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
pd.Series(Accuracies, index = Models).plot(kind = 'bar', color = AccColors)
plt.title('Validation Set Accuracy Across Models')
plt.xlabel('Model')
plt.xticks(rotation=50)
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
pd.Series(AUCs, index = Models).plot(kind = 'bar', color = AUCColors)
plt.title('Validation Set AUC Across Models')
plt.xlabel('Model')
plt.xticks(rotation=50)
plt.ylabel('AUC')

# finding model with best performance across metrics
print('\nHighest Accuracy:', Models[Accuracies.index(max(Accuracies))])
print('Highest AUC:', Models[AUCs.index(max(AUCs))])

plt.show()

#%% EXTRA CREDIT B
print('\nOTHER ANALYSES:\n')

# predicting general health from physical and mental health (Random Forest)

import seaborn as sns
sns.reset_orig()

X = df[['PhysicalHealth', 'MentalHealth']].to_numpy()
y = df['GeneralHealth'].to_numpy()

print('Correlation Between General Health and Physical/Mental Health:')
print(df.corr()['GeneralHealth'][['PhysicalHealth', 'MentalHealth']])

df['GeneralHealth'].value_counts(sort=False).plot(kind = 'bar')
plt.title('Distribution of General Health Variable')
plt.ylabel('Frequency')
plt.xlabel('Health Rating')
plt.xticks(rotation = 0)
plt.show()

# building model
trainX, testX, trainy, testy = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 36)
forest = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators = 500).fit(trainX, trainy)

# accuracy
Accs = []
acc = metrics.accuracy_score(testy, forest.predict(testX))
print('\nAccuracy:', acc)
Accs.append(acc)

# confusion matrix
matrix = metrics.confusion_matrix(testy, forest.predict(testX))
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap=plt.cm.Blues)
class_names = ['1 - bad health', '2 - poor health', 
               '3 - fair health', '4 - good health', 
               '5 - great health']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
plt.show()

# feature importance
for feature in range(X.shape[1]):
    trainX_temp = trainX.copy()
    trainX_temp = np.delete(trainX_temp, feature, axis = 1)
    
    testX_temp = testX.copy()
    testX_temp = np.delete(testX_temp, feature, axis = 1)
    forest = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators = 500)
    forest.fit(trainX_temp, trainy)
    Accs.append(metrics.accuracy_score(testy, forest.predict(testX_temp)))
 
label = ['None Dropped', 'Physical Health', 'Mental Health']
sns.reset_orig()
plt.figure(figsize = (5,3))
plt.plot(np.arange(len(Accs)), Accs)
plt.title('Random Forest Accuracy While Dropping Predictors')
plt.xlabel('Predictor Dropped')
plt.ylabel('Accuracy')
plt.xticks(ticks = np.arange(len(Accs)), 
           labels = label, 
           rotation = 80)
plt.show()

# finding predictor that is most important (i.e. dropped accuracy the most)
print('Best Predictor of General Health:', label[Accs.index(min(Accs))])
