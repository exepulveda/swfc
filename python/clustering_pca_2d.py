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
    filename = '2d'
    locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()

    N,ND = data_ore.shape
    
    
    #recode binary clay
    binary_clay = np.zeros((N,categories[0]))
    for i in range(categories[0]):
        indices = np.where(data_ore[:,0] == i)[0]
        binary_clay[indices,i] = 1.0

    values = np.c_[binary_clay,data_ore[:,1:]]
    N,ND = values.shape

    #now all are continues variables
    var_types = np.ones(ND)
    seed = 1634120
    
    np.random.seed(seed)
    n,p = values.shape

    standadize = StandardScaler()
    data = standadize.fit_transform(values)
    scale = standadize.scale_
    ND_PCA = 3
    scale = np.ones(ND_PCA)
    var_types = np.ones(ND_PCA)

    pca = PCA(n_components=ND_PCA,whiten=True)
    pca_X = pca.fit_transform(data)
    data_F = np.asfortranarray(pca_X,dtype=np.float32)

    pca_clusters_all = np.empty(len(locations))

    for NC in range(2,11):
        clustering_pca = KMeans(n_clusters=NC)
        clusters_pca = clustering_pca.fit_predict(pca_X)

        #save data
        pca_clusters_all.fill(NC) #waste
        pca_clusters_all[ore_indices] = clusters_pca #ore
        
        new_data = np.c_[locations,pca_clusters_all]
        np.savetxt("../results/{dataset}_clusters_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),new_data,delimiter=",",fmt="%.4f")
        
        #stats
        setup_distances(scale,var_types,categories,targets=None)

        #PCA
        centroid = np.asfortranarray(clustering_pca.cluster_centers_,dtype=np.float32)
        clusters = np.asfortranarray(clusters_pca,dtype=np.int8)
        weights = np.asfortranarray(np.ones((NC,ND_PCA),dtype=np.float32)/ ND_PCA) 
        
        ret_pca = cl.clustering.dbi_index(centroid,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        print('2D PCA',NC,ret_pca,ret_sill,sep=' ')
        cl.distances.reset()

    #save data
    NC = 4
    clustering_pca = KMeans(n_clusters=NC)
    clusters_pca = clustering_pca.fit_predict(pca_X)

    centroids = np.empty((NC,ND))
    for k in range(NC):
        indices = np.where(clusters_pca == k)[0]
        centroids[k,:] = np.mean(values[indices,:],axis=0)

    print('centroids',centroids)

    np.savetxt("../results/final_{dataset}_clusters_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),clusters_pca,delimiter=",",fmt="%.4f")

    np.savetxt("../results/final_{dataset}_centroids_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),centroids,delimiter=",",fmt="%.4f")
