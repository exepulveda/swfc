import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib as mpl
mpl.use('agg')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cluster_utils import create_clusters_dict

from plotting import scatter_clusters

import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename = "ds4"
    
    X = np.loadtxt("../../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    locations = X[:,0:2]
    values = X[:,2:6] #0,1,2,5 are real; 3 and 4 are cats
    true_clusters = X[:,6]
    
    n,p = values.shape
    nclusters = 4
    numerical = [0,1,2,3]
    data = values[:,numerical]
    standadize = StandardScaler()
    data = standadize.fit_transform(data)

    clustering = KMeans(n_clusters=4)
    kmeans_clusters = clustering.fit_predict(data)
        
    pca = PCA(n_components=2,whiten=True)
    pca_X = pca.fit_transform(data)
    clustering_pca = KMeans(n_clusters=4)
    clusters_pca = clustering_pca.fit_predict(pca_X)
    #save data
    new_data = np.c_[X,pca_X,kmeans_clusters,clusters_pca]
    np.savetxt("../../data/{dataset}_clusters.csv".format(dataset=filename),new_data,delimiter=",",fmt="%.4f")
