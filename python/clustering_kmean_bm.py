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

from case_study_bm import attributes,setup_case_study_ore,setup_case_study_all,setup_distances

if __name__ == "__main__":
    filename = 'bm'

    locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore()
    
    N,ND = data.shape
    
    #recode binary clay
    binary_rocktype = np.zeros((N,categories[0]))
    for i in range(categories[0]):
        indices = np.where(data[:,0] == i)[0]
        print(i,len(indices))
        binary_rocktype[indices,i] = 1.0

    values = np.c_[binary_rocktype,data[:,1:]]
    
    #var_types = np.ones(ND)
    seed = 1634120
    np.random.seed(seed)
    
    n,p = values.shape

    standadize = StandardScaler()
    data_std = standadize.fit_transform(values)
    #scale = np.ones(ND) #standadize.scale_

    data[:,0] += 0.999
    data_F = np.asfortranarray(data,dtype=np.float32)
    
    print('var_types',var_types)

    for NC in range(2,1):
        #print("KMeans DB Index:",NC)
        clustering = KMeans(n_clusters=NC)
        kmeans_clusters = np.int8(clustering.fit_predict(data_std))
        #centroid = np.asfortranarray(clustering.cluster_centers_,dtype=np.float32)
        #print("KMeans Centroids:",centroid)
        centroids_F = np.asfortranarray(np.empty((NC,ND)),dtype=np.float32)
        
        #calculate centroids back 
        #print("Calculating centroids")
        for k in range(NC):
            indices = np.where(kmeans_clusters == k)[0]
            centroids_F[k,:] = np.mean(data[indices,:],axis=0)
            print(k,len(indices),centroids_F[k,:])
        
        #new_data = np.c_[locations,kmeans_clusters]
        #np.savetxt("../results/{dataset}_clusters_kmeans_{nc}.csv".format(dataset=filename,nc=NC),new_data,delimiter=",",fmt="%.4f")
        
        #stats
        setup_distances(scale,var_types,use_cat=True,targets=None)
        #print("setup_distances DONE")

        #KMeans
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND) 
        clusters = np.asfortranarray(kmeans_clusters,dtype=np.int8)
        
        ret_kmeans = cl.clustering.dbi_index(centroids_F,data_F,clusters,weights)
        #print("KMeans DB Index:",NC,ret_kmeans)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        print("KMeans:",NC,ret_kmeans,ret_sill,sep=',')
        cl.distances.reset()
        

    #
    NC = 3
    clustering = KMeans(n_clusters=NC)
    kmeans_clusters = clustering.fit_predict(data)
    np.savetxt("../results/final_{dataset}_clusters_kmeans_{nc}.csv".format(dataset=filename,nc=NC),kmeans_clusters,delimiter=",",fmt="%.4f")
