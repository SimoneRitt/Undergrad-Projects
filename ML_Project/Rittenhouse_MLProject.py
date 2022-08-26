#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:10:15 2022

@author: simonerittenhouse
"""

# Simone Rittenhouse

# Introduction to Machine Learning
# Capstone Project
# 5/17/2022

# importing packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics, model_selection, tree
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.cm as cm

# loading dataset
df = pd.read_csv('/Users/simonerittenhouse/Desktop/Intro to ML/musicData.csv')

# setting seed
seed = 17274546
np.random.seed(seed)
random.seed(seed)

#%% EXPLORATORY ANALYSIS

print('Shape:', df.shape)
print('Features:', df.columns)
print(df.describe())

# seeing pairs of large correlations
correlations = df.corr()
highCorr = correlations.where((correlations > 0.3) & (correlations < 1))
lowCorr = correlations.where(correlations < -0.3)
largeCorr = []
largeCorrPairs = []
for x in highCorr:
    temp = highCorr[x].dropna()
    if len(temp) > 0:
        for y in range(len(temp)):
            if temp[y] not in largeCorr:
                largeCorr.append(temp[y])
                largeCorrPairs.append((temp.index[y], x))
for x in lowCorr:
    temp = lowCorr[x].dropna()
    if len(temp) > 0:
        for y in range(len(temp)):
            if temp[y] not in largeCorr:
                largeCorr.append(temp[y])
                largeCorrPairs.append((temp.index[y], x))
for val in range(len(largeCorr)):
    print(largeCorrPairs[val], largeCorr[val])
    
print('\nHighest Correlation:', largeCorrPairs[largeCorr.index(max(largeCorr))], max(largeCorr))
print('Lowest Correlation:', largeCorrPairs[largeCorr.index(min(largeCorr))], min(largeCorr))

#%% PRE-PROCESSING

# finding NaNs
print('\nNULL VALUES:')
print(df.isnull().sum(axis = 0), '\n')
nulls = df[df.isnull().any(axis=1)]
print(nulls)

# dropping nulls
df.dropna(inplace=True)
print('\nNew Shape (Dropping Null Values):')
print(df.shape)

# label encoding genres (making classes numeric)
le = LabelEncoder()
le.fit(df['music_genre'])
orig_genres = le.classes_
df['music_genre'] = le.transform(df['music_genre'])

# dropping artist and song (from model dataset)
df_model = df.copy()
df_model.drop(['artist_name', 'track_name'], inplace=True, axis = 1)

# seeing df datatypes
print(df_model.dtypes)

# tempo needs to be changed to numeric (not categorical)
missingCount = 0
for val in df_model['tempo']:
    if val == '?':
        missingCount += 1
print('\nNumber of missing Tempo Values:', missingCount)

# train/test split
test_ind = []
for genre in df_model['music_genre'].dropna().unique():
    genre_df = df_model[df_model['music_genre'] == genre]
    test_indices = genre_df.sample(500, replace=False, random_state=seed)
    test_ind.extend(list(test_indices.index))
train_ind = [i for i in df_model.index if i not in test_ind]

print('\nTrain/Test Data Shapes:')
test = df_model.loc[test_ind]
print('test -', test.shape)

train = df_model.loc[train_ind]
print('train -', train.shape)

# creating target labels
test_labels = test['music_genre']
train_labels = train['music_genre']

test.drop(['music_genre'], axis=1, inplace=True)
train.drop(['music_genre'], axis=1, inplace=True)

# re-encoding tempo
numericTempoTrain = [float(tempo) for tempo in train['tempo'] if tempo != '?']
imputeTempoTrain = np.median(numericTempoTrain)

newTempoTrain = []
for val in train['tempo']:
    if val == '?':
        newTempoTrain.append(imputeTempoTrain)
    else:
        newTempoTrain.append(float(val))
        
numericTempoTest = [float(tempo) for tempo in test['tempo'] if tempo != '?']
imputeTempoTest = np.median(numericTempoTest)

newTempoTest = []
for val in test['tempo']:
    if val == '?':
        newTempoTest.append(imputeTempoTest)
    else:
        newTempoTest.append(float(val))
        
# median imputation of tempo     
train['tempo'] = newTempoTrain
test['tempo'] = newTempoTest

# encoding categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# making sure columns are aligned (same columns in both sets) - this drops date value of '0/4'
train, test = train.align(test, join = 'inner', axis = 1)

print('\nTrain/Test Data Shapes (with Dummies):')
print('test -', test.shape)
print('train -', train.shape)

# normalization (of columns that aren't binary)
notBinary = [col for col in train.columns if len(train[col].unique()) != 2]

for col in train.columns:
    if col in notBinary:
        train[col] = (train[col] - train[col].mean()) / train[col].std()
        test[col] = (test[col] - test[col].mean()) / test[col].std()

#%% DIMENSIONALITY REDUCTION

# dividing one-hot encoded columns by the square root of its probability and centering (for FAMD)
binary = [col for col in train.columns if len(train[col].unique()) == 2]
for col in train.columns:
    if col in binary:
        prob_train = len(train[train[col] == 1])/len(train[col])
        prob_test = len(test[test[col] == 1])/len(test[col])

        train[col] = train[col]/np.sqrt(prob_train)
        train[col] = train[col] - train[col].mean()
        
        test[col] = test[col]/np.sqrt(prob_test)
        test[col] = test[col] - test[col].mean()
   
pca_train = PCA(random_state=seed).fit(train)
train_pca = pca_train.transform(train)
test_pca = pca_train.transform(test)

plt.scatter(train_pca[:,0], train_pca[:,1])
plt.title('First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# scree plot
print('\nEigenvalues:', pca_train.explained_variance_)
print('Number of Features to use:', len([e for e in pca_train.explained_variance_ if e > 1]))
PC_values = np.arange(pca_train.n_components_) + 1
plt.plot(PC_values, pca_train.explained_variance_, 'o-', linewidth=2, color='blue')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# interpreting first two components using loadings
loadings = pca_train.components_[:2].T
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=train.columns)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
loading_matrix['PC1'].plot(kind='bar')
plt.title('Loadings Plot For PC1')
plt.xlabel('Feature')
plt.ylabel('Feature Loading onto PC1')

plt.subplot(1,2,2)
loading_matrix['PC2'].plot(kind='bar')
plt.title('Loadings Plot For PC2')
plt.xlabel('Feature')
plt.ylabel('Feature Loading onto PC2')
plt.show()
    
#%% CLUSTERING

print()

# silhouette method
sil_data = train_pca[:,:12]
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    # silhouette plot
    plt.xlim([-0.1, 0.8])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(sil_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value
    clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = clusterer.fit_predict(sil_data)

    silhouette_avg = silhouette_score(sil_data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample (each point)
    sample_silhouette_values = silhouette_samples(sil_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx( # this plots silhouettes - canonical way of plotting silhouette scores
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])

    plt.title(
        "Silhouette Plot for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters)

    plt.show()
    
#%%
    
# performing k-means clustering for n_clusters= 8
k_means = KMeans(n_clusters=8, n_init=100, max_iter=10000, random_state=seed)
trainCluster_labels = k_means.fit_predict(sil_data)
testCluster_labels = k_means.predict(test_pca[:,:12])

# plot clusters formed
col = {0:'r',1:'g',2:'b',3:'m',4:'c',5:'greenyellow',6:'orangered',7:'gold'}

plt.figure(figsize=(10,6))
for label in np.unique(trainCluster_labels):
    cluster = sil_data[np.where(trainCluster_labels==label)]
    plt.scatter(cluster[:,0], cluster[:,1], color=col[label], label=label, alpha=0.5)
plt.legend(loc='best')
plt.title('K-Means Clustering on PCA-reduced Data, k={}'.format(8))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#%% FINAL TRAIN/TEST DATASETS

# normalizing train cluster labels (to avoid issues of scaling)
trainClusters = (trainCluster_labels - np.mean(trainCluster_labels, axis=0)) / np.std(trainCluster_labels, axis=0)

# normalizing test cluster labels (to avoid issues of scaling)
testClusters = (testCluster_labels - np.mean(testCluster_labels, axis=0)) / np.std(testCluster_labels, axis=0)

# adding in cluster columns to train/test
train = np.column_stack((train_pca[:,:12], trainClusters))
test = np.column_stack((test_pca[:,:12], testClusters))

print('\nFinal train shape:', train.shape)
print('Final test shape:', test.shape)

#%% PLOTTING ORIGINAL LABELS IN 2D SPACE

col = {0:'r',1:'g',2:'b',3:'m',4:'c',5:'greenyellow',6:'orangered',7:'gold',
       8:'firebrick',9:'deeppink'}

plt.figure(figsize=(10,6))
for label in np.unique(train_labels):
    musicGenre = train[np.where(train_labels==label)]
    plt.scatter(musicGenre[:,0], musicGenre[:,1], color=col[label], label=orig_genres[label], alpha=0.5)
plt.legend(loc='best')
plt.title('Music Genres in PCA-reduced Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#%% RANDOM FOREST

print('\nPREDICTING WITH RANDOM FOREST:')

# grid search for best parameters
grid=dict()
grid['n_estimators'] = [50, 100]
grid['max_samples'] = [.5, .75, 0.999]
grid['max_features'] = [0.25, 0.5, 0.75]
model = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=seed)
gridSearch = model_selection.GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='f1_macro')
result = gridSearch.fit(train, train_labels)
params = result.best_params_
print('Highest F1 Score =', result.best_score_)
print('Optimal Parameters:', params)

# building model
modelForest = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                     max_samples=params['max_samples'], 
                                     max_features=params['max_features'],
                                     bootstrap=True, criterion='gini',
                                     random_state=seed)
modelForest.fit(train, train_labels)

#%%
# METRICS
pred_rf = modelForest.predict(test)

acc_rf = metrics.accuracy_score(test_labels, pred_rf)
print('\nRandom Forest Accuracy:', acc_rf)

pred_prob_rf = modelForest.predict_proba(test)
auc_rf = metrics.roc_auc_score(test_labels, pred_prob_rf, multi_class='ovr')
print('Average Random Forest AUC:', auc_rf)

# plotting single class AUCs (and finding average)
rf_class_auc = []

plt.figure(figsize=(10,6))

index = 0
for singleClass in np.unique(test_labels):
    classLabels = np.where(test_labels==singleClass,1,0)
    classAUC = metrics.roc_auc_score(classLabels, pred_prob_rf[:,index])
    rf_class_auc.append(classAUC)
    
    # plotting them:
    fpr, tpr, thresholds = metrics.roc_curve(classLabels, pred_prob_rf[:,index])
    plt.plot(fpr, tpr, label='{}, AUC={:.3f}'.format(orig_genres[index], classAUC))
    index += 1
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

print('Average of Single Class AUCs:', np.mean(rf_class_auc))

#%% ADABOOST

print('\nPREDICTING WITH ADABOOST:')

# grid search for best parameters
grid=dict()
grid['n_estimators'] = [100, 500]
grid['learning_rate'] = [0.001, 0.01, 0.1, 1]
model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1,random_state=seed), random_state=seed)
gridSearch = model_selection.GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='f1_macro')
result = gridSearch.fit(train, train_labels)
params = result.best_params_
print('Highest F1 Score =', result.best_score_)
print('Optimal Parameters:', params)

# building model
modelBoost = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1, random_state=seed), 
                                n_estimators=params['n_estimators'], 
                                learning_rate=params['learning_rate'],
                                random_state=seed)
modelBoost.fit(train, train_labels)

#%%
# METRICS
pred_boost = modelBoost.predict(test)

acc_boost = metrics.accuracy_score(test_labels, pred_boost)
print('\nAdaboost Accuracy:', acc_boost)

pred_prob_boost = modelBoost.predict_proba(test)
auc_boost = metrics.roc_auc_score(test_labels, pred_prob_boost, multi_class='ovr')
print('Average AdaBoost AUC:', auc_boost)

# plotting single class AUCs (and finding average)
boost_class_auc = []

plt.figure(figsize=(10,6))

index = 0
for singleClass in np.unique(test_labels):
    classLabels = np.where(test_labels==singleClass,1,0)
    classAUC = metrics.roc_auc_score(classLabels, pred_prob_boost[:,index])
    boost_class_auc.append(classAUC)
    
    # plotting them:
    fpr, tpr, thresholds = metrics.roc_curve(classLabels, pred_prob_boost[:,index])
    plt.plot(fpr, tpr, label='{}, AUC={:.3f}'.format(orig_genres[index], classAUC))
    index += 1
plt.title('AdaBoost ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

print('Average of Single Class AUCs:', np.mean(boost_class_auc))

#%% COMPARING MODELS

Accuracies = [acc_rf, acc_boost]
AUCs = [auc_rf, auc_boost]

barColors = ['mediumturquoise']*2
AccColors = barColors.copy()
AccColors[Accuracies.index(max(Accuracies))] = 'orangered'
AUCColors = barColors.copy()
AUCColors[AUCs.index(max(AUCs))] = 'orangered'

plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.bar(np.arange(2), AUCs, tick_label = ['Random Forest', 'AdaBoost'], color=AUCColors)
plt.title('Average AUC of Models')
plt.xlabel('Model')
plt.ylabel('AUC')

plt.subplot(1,2,2)
plt.bar(np.arange(2), Accuracies, tick_label = ['Random Forest', 'AdaBoost'], color=AccColors)
plt.title('Accuracy Of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

#%% EXTRA CREDIT

exploreDf = pd.read_csv('/Users/simonerittenhouse/Desktop/Intro to ML/musicData.csv')

exploreDf= exploreDf.dropna().sort_values(by=['popularity'], ascending=False)

genrePop = []
genres = []
for genre in exploreDf['music_genre'].unique():
    genres.append(genre)
    genreDf = exploreDf[exploreDf['music_genre']==genre]
    genrePop.append(genreDf['popularity'].mean())
    
# plotting popularity for each genre
plt.figure(figsize=(10,6))
plt.bar(np.arange(len(genrePop)), genrePop, tick_label=genres)
plt.title('Average Popularity of Songs in Each Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.show()

print('\nMost Popular Genre:', genres[genrePop.index(max(genrePop))])
print('Least Popular Genre:', genres[genrePop.index(min(genrePop))])

# testing significance between genres
mostPop = exploreDf[exploreDf['music_genre']==genres[genrePop.index(max(genrePop))]]['popularity']
leastPop = exploreDf[exploreDf['music_genre']==genres[genrePop.index(min(genrePop))]]['popularity']
tStat, pValue = stats.ttest_ind(mostPop, leastPop)

print('p-value =', pValue)
print('t-Statistic =', tStat)
    
print('\nMost Popular Song: {} by {}'.format(exploreDf['track_name'].iloc[0],
                                             exploreDf['artist_name'].iloc[0]))
print('Least Popular Song: {} by {}'.format(exploreDf['track_name'].iloc[-1],
                                             exploreDf['artist_name'].iloc[-1]))