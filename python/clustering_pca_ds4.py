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
    
    #now all are continues variables
    var_types = np.ones(ND)

    seed = 1634120   
    np.random.seed(seed)

    standadize = StandardScaler()
    data_scaled = standadize.fit_transform(data)
    scale = standadize.scale_
    ND_PCA = 2

    pca = PCA(n_components=ND_PCA,whiten=True)
    pca_X = pca.fit_transform(data_scaled)

    data_F = np.asfortranarray(data,dtype=np.float32)

    for NC in range(2,11):
        clustering_pca = KMeans(n_clusters=NC)
        clusters_pca = clustering_pca.fit_predict(pca_X)

        #print("Calculating centroids")
        centroids_F = np.asfortranarray(np.empty((NC,ND)),dtype=np.float32)
        for k in range(NC):
            indices = np.where(clusters_pca == k)[0]
            centroids_F[k,:] = np.mean(data[indices,:],axis=0)
            #print(k,len(indices)) #,centroids_F[k,:])        

        #PCA
        cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
        cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
        
        clusters = np.asfortranarray(clusters_pca,dtype=np.int8)
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND) 
        
        ret_pca = cl.clustering.dbi_index(centroids_F,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        print('2D PCA',NC,ret_pca,ret_sill,sep=',')
        cl.distances.reset()
