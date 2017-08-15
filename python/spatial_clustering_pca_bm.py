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

from bm_case_study import bm_variables,setup_case_study,setup_distances

if __name__ == "__main__":
    filename = 'bm'
    locations,data,scale,var_types,targets = setup_case_study()


    N,ND = data.shape
    
    #recode binary rocktype
    binary_rocktype = np.zeros((N,3))
    for i in range(3):
        indices = np.where(data[i,0] == i)[0]
        binary_rocktype[indices,i] = 1.0
    
    values = np.c_[binary_rocktype,data[:,1:]]
    N,ND = values.shape
    
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

    for NC in range(2,21):
        clustering_pca = KMeans(n_clusters=NC)
        clusters_pca = clustering_pca.fit_predict(pca_X)
        #save data
        new_data = np.c_[locations,pca_X,clusters_pca]
        np.savetxt("../results/{dataset}_clusters_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),new_data,delimiter=",",fmt="%.4f")
        
        #stats
        setup_distances(scale,var_types,distances_cat=None,targets=None)

        #PCA
        centroid = np.asfortranarray(clustering_pca.cluster_centers_,dtype=np.float32)
        clusters = np.asfortranarray(clusters_pca,dtype=np.int8)
        weights = np.asfortranarray(np.ones((NC,ND_PCA),dtype=np.float32)/ ND_PCA) 
        
        ret_pca = cl.clustering.dbi_index(centroid,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        print("PCA DB Index:",NC,ret_pca,ret_sill)
        cl.distances.reset()
