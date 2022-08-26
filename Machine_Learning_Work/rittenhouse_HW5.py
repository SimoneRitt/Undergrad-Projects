#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:23:22 2022

@author: simonerittenhouse
"""

# Introduction to Machine Learning
# Homework Assignment 5
# 5/11/2022

# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN

# loading data
df = pd.read_csv('/Users/simonerittenhouse/Desktop/Intro to ML/wines.csv')
features = df.columns

# inspecting data
print('Features:', features)
print('Shape:', df.shape)

# exploratory analysis
correlations = df.corr()
print(correlations)

# normalizing the data
data = df.to_numpy()
data = (data - np.mean(data, axis=0))/np.std(data, axis=0)

#%% QUESTION ONE

# running PCA
pca = PCA(whiten=True, random_state=0).fit(data)
data_pca = pca.transform(data)

# scree plot
print('\nEigenvalues:', pca.explained_variance_)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_, 'o-', linewidth=2, color='blue')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# scatter plot of first 2 PC
plt.scatter(data_pca[:,0], data_pca[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: Wines in 2-D')
plt.show()

# variance explained by first two principal components
print('\nVariance explained by PC1 ({:.2%}) and PC2 ({:.2%}) = {:.2%}'.format(pca.explained_variance_ratio_[0], 
                                                                              pca.explained_variance_ratio_[1], 
                                                                              sum(pca.explained_variance_ratio_[:2])))

# interpreting first two components using loadings
loadings = pca.components_[:2].T
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)
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

#%% QUESTION TWO

KL_div = []

for p in range(5,151):
    model = TSNE(perplexity=p, random_state=0)
    data_tsne = model.fit_transform(data)
    kl = model.kl_divergence_
    
    KL_div.append(kl)
    
    if p == 20:
        # scatter plot of first 2 PC
        plt.scatter(data_tsne[:,0], data_tsne[:,1])
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('t-SNE (perplexity=20): Wines in 2-D')
        plt.show()
        
        print('\nKL-divergence for perplexity = 20:', kl)
       
plt.plot(range(5,151), KL_div)
plt.title('KL-Divergence Scores for t-SNE Across Perplexity Values')
plt.xlabel('Perplexity')
plt.ylabel('KL-Divergence')
plt.show()

#%% QUESTION THREE

dist_euclid = euclidean_distances(data)

# 2 components (poor fit)
mds = MDS(n_components=2, n_init=100, max_iter=10000, dissimilarity='precomputed', random_state=0)
data_mds = mds.fit_transform(dist_euclid)

stress_orig = np.sqrt(mds.stress_ / (0.5 * np.sum(dist_euclid**2)))

# testing stress for higher dimensional space:
stress = [stress_orig]
for x in range(3,11):
    test = MDS(n_components=x, n_init=100, max_iter=10000, dissimilarity='precomputed', random_state=0)
    test_MDS = test.fit_transform(dist_euclid)
    stress.append(np.sqrt(test.stress_ / (0.5 * np.sum(dist_euclid**2))))

# plotting
plt.scatter(data_mds[:,0], data_mds[:,1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('MDS: Wines in 2-D')
plt.show()

# plotting stress
plt.plot(range(2,11), stress)
plt.title('Stress Over The Number of Components')
plt.xlabel('N Components')
plt.ylabel('MDS Stress')

print('\nStress of MDS:', stress[0], '\n')

#%% QUESTION FOUR

# Using PCA data for silhouette plots (2D-solution)
sil_data = data_pca[:,:2]

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    # silhouette plot
    plt.xlim([-0.1, 0.8])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(sil_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(sil_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
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
        % n_clusters,
        fontweight="bold",
    )

    plt.show()

#%%

# performing k-means clustering for n_clusters= 3

k_means = KMeans(n_clusters=3, n_init=100, max_iter=10000, random_state=0)
cluster_labels = k_means.fit_predict(sil_data)

# plot clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)

plt.figure()
plt.scatter(
    sil_data[:, 0], sil_data[:, 1], alpha=0.7, c=colors, edgecolor="k"
    )

# Labeling the clusters
centers = k_means.cluster_centers_
# Draw white circles at cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="o",
    c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker="$%d$" % (i+1), alpha=1, s=50, edgecolor="k")

plt.title("K-Means Clustering on PCA Data (k=3)")
plt.xlabel("Feature space for the 1st feature")
plt.ylabel("Feature space for the 2nd feature")
plt.show()

print('Squared-Sum of Distance From all Points to Their Clusters =', k_means.inertia_)

#%%

from math import dist

total_dist = 0
for label in range(len(set(cluster_labels))):
    cluster = sil_data[np.where(cluster_labels == label)]
    for pt in range(len(cluster)):
        total_dist += dist(cluster[pt,:], centers[label,:])
        
print('Total Distance of all Points to Their Cluster Centers =', total_dist)

#%% QUESTION FIVE

# dBScan for t-SNE data with perplexity of 20
dB_data = TSNE(perplexity=20, random_state=0).fit_transform(data)

eps_sil = []
test_eps = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]
for i in test_eps:
    db = DBSCAN(eps=i, min_samples=2).fit(dB_data)
    labels = db.labels_
    sil_avg = silhouette_score(dB_data, labels)
    eps_sil.append(sil_avg)
    
plt.plot(test_eps, eps_sil)
plt.title('Average Silhouette Score Across Epsilon Values (min_samples=2)')
plt.xlabel('Epsilon Value')
plt.ylabel('Average Silhouette Score')
plt.show()
    
# seeing which values of min_samples give optimal number of clusters
for e in [1.25, 2.25, 3.5, 4.0]:
    e_clust = []
    outliers = []
    test_minSamples = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for i in test_minSamples:
        db = DBSCAN(eps=e, min_samples=i).fit(dB_data)
        # ignore -1 since it's for outliers
        labels = set([x for x in db.labels_ if x>-1])
        noise = [x for x in db.labels_ if x == -1]
        e_clust.append(len(labels))
        outliers.append(len(noise))
    # plotting number of clusters
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(test_minSamples, e_clust)
    plt.title('Number of Clusters Across min_samples (epsilon={})'.format(e))
    plt.xlabel('Min Samples Value')
    plt.ylabel('Number of Clusters')
    plt.xticks(test_minSamples)
 
    # plotting number of outliers
    plt.subplot(1,2,2)
    plt.plot(test_minSamples, outliers)
    plt.title('Number of Outliers Across min_samples (epsilon={})'.format(e))
    plt.xlabel('Min Samples Value')
    plt.ylabel('Number of Outliers')
    plt.xticks(test_minSamples)
    plt.show()

col = {0:'r',1:'g',2:'b',3:'m',-1:'k'}
# testing epsilon == 3.5
for m in [3,9,18]:
    db = DBSCAN(eps=3.5, min_samples=m).fit(dB_data)
    dB_pred = db.fit_predict(dB_data)
    
    # plotting
    for label in np.unique(db.labels_):
        cluster = dB_data[np.where(dB_pred==label)]
        plt.scatter(cluster[:,0], cluster[:,1], color=col[label], label=label)
    plt.legend(loc='best')
    plt.title('dBScan for epsilon={}, min_samples={}'.format(3.5, m))
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()

# testing epsilon == 4.0
for m in [3,4,10]:
    db = DBSCAN(eps=4, min_samples=m).fit(dB_data)
    dB_pred = db.fit_predict(dB_data)
    
    # plotting
    for label in np.unique(db.labels_):
        cluster = dB_data[np.where(dB_pred==label)]
        plt.scatter(cluster[:,0], cluster[:,1], color=col[label], label=label)
    plt.legend(loc='best')  
    plt.title('dBScan for epsilon={}, min_samples={}'.format(4.0, m))
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()

#%% EXTRA CREDIT B

# examining results of PCA loadings

df['Flavonoids'].hist()
plt.title('Distribution of Flavonoids')
plt.xlabel('Flavonoids')
plt.ylabel('Frequency')
plt.show()

df['Alcohol'].hist()
plt.title('Distribution of Alcohol')
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.show()

df['Color_Intensity'].hist()
plt.title('Distribution of Color Intensity')
plt.xlabel('Color Intensity')
plt.ylabel('Frequency')
plt.show()

print('\nCorrelation between Color Intensity and Alcohol:', 
      df[['Color_Intensity', 'Alcohol']].corr()['Alcohol'][0])

print('\nCorrelation between Color Intensity and Flavonoids:', 
      df[['Color_Intensity', 'Flavonoids']].corr()['Flavonoids'][0])

print('\nCorrelation between Color Intensity and Hue:', 
      df[['Color_Intensity', 'Hue']].corr()['Hue'][0])