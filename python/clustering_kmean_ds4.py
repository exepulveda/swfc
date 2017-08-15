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
from sklearn.metrics import silhouette_score

from cluster_utils import create_clusters_dict, recode_categorical_values
from plotting import scatter_clusters
import matplotlib.pyplot as plt

import clusteringlib as cl

from case_study_2d import attributes,setup_case_study,setup_distances

if __name__ == "__main__":
    filename = 'ds4'

    X = np.loadtxt("../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    locations = X[:,0:2]
    data = X[:,2:6] #0,1,2,5 are continues
    true_clusters = X[:,6]

    N,ND = data.shape
    
    var_types = np.ones(ND)

    seed = 1634120
    np.random.seed(seed)

    standadize = StandardScaler()
    data_scaled = standadize.fit_transform(data)
    scale = standadize.scale_

    data_F = np.asfortranarray(data,dtype=np.float32)

    for NC in range(2,11):
        clustering = KMeans(n_clusters=NC)
        kmeans_clusters = clustering.fit_predict(data_scaled)
        
        sc = silhouette_score(data_scaled,kmeans_clusters)
            
        #save data
        #new_data = np.c_[locations,kmeans_clusters_all]
        #np.savetxt("../results/{dataset}_clusters_kmeans_{nc}.csv".format(dataset=filename,nc=NC),new_data,delimiter=",",fmt="%.4f")
        centroids_F = np.asfortranarray(np.empty((NC,ND)),dtype=np.float32)
        for k in range(NC):
            indices = np.where(kmeans_clusters == k)[0]
            centroids_F[k,:] = np.mean(data[indices,:],axis=0)
        
        #stats
        cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
        cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)

        #KMeans
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND) 
        clusters = np.asfortranarray(kmeans_clusters,dtype=np.int8)
        
        ret_kmeans = cl.clustering.dbi_index(centroids_F,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        print("KMeans DB Index:",NC,ret_kmeans,ret_sill,sc)
        cl.distances.reset()

