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
    binary_clay = np.zeros((N,3))
    for i in range(categories[0]):
        indices = np.where(np.int32(data_ore[:,0]) == i)[0]
        print(i,len(indices))
        binary_clay[indices,i] = 1.0

    values = np.c_[binary_clay,data_ore[:,1:]]
    N,ND = values.shape
    
    var_types = np.ones(ND)
    seed = 1634120
    np.random.seed(seed)
    
    n,p = values.shape

    standadize = StandardScaler()
    data = standadize.fit_transform(values)
    scale = standadize.scale_

    data_F = np.asfortranarray(data,dtype=np.float32)

    kmeans_clusters_all = np.empty(len(locations))
    
    for NC in range(2,11):
        clustering = KMeans(n_clusters=NC)
        kmeans_clusters = clustering.fit_predict(data)
            
        #save data
        kmeans_clusters_all.fill(NC) #waste
        kmeans_clusters_all[ore_indices] = kmeans_clusters #ore
        
        new_data = np.c_[locations,kmeans_clusters_all]
        np.savetxt("../results/{dataset}_clusters_kmeans_{nc}.csv".format(dataset=filename,nc=NC),new_data,delimiter=",",fmt="%.4f")
        
        #stats
        setup_distances(scale,var_types,categories,targets=None)

        #KMeans
        centroid = np.asfortranarray(clustering.cluster_centers_,dtype=np.float32)
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND) 
        clusters = np.asfortranarray(kmeans_clusters,dtype=np.int8)
        
        ret_kmeans = cl.clustering.dbi_index(centroid,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        print("KMeans DB Index:",NC,ret_kmeans,ret_sill)
        cl.distances.reset()
        

    #
    NC = 4
    clustering = KMeans(n_clusters=NC)
    kmeans_clusters = clustering.fit_predict(data)
    centroids = np.empty((NC,ND))
    for k in range(NC):
        indices = np.where(kmeans_clusters == k)[0]
        centroids[k,:] = np.mean(values[indices,:],axis=0)

    print('centroids',centroids)

    new_data = np.c_[locations_ore,kmeans_clusters]
    np.savetxt("../results/final_{dataset}_clusters_kmeans_{nc}.csv".format(dataset=filename,nc=NC),kmeans_clusters,delimiter=",",fmt="%.4f")
    np.savetxt("../results/final_{dataset}_centroids_kmeans_{nc}.csv".format(dataset=filename,nc=NC),centroids,delimiter=",",fmt="%.4f")
