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

    #now all are continues variables
    seed = 1634120
    
    np.random.seed(seed)
    standadize = StandardScaler()
    data_std = standadize.fit_transform(values)
    #scale = standadize.scale_
    ND_PCA = 3
    #scale = np.ones(ND_PCA)
    #var_types = np.ones(ND_PCA)

    pca = PCA(n_components=ND_PCA,whiten=True)
    pca_X = pca.fit_transform(data_std)


    data[:,0] += 0.999
    data_F = np.asfortranarray(data,dtype=np.float32)

    for NC in range(2,1):
        clustering_pca = KMeans(n_clusters=NC)
        clusters_pca = np.int8(clustering_pca.fit_predict(pca_X))

        centroids_F = np.asfortranarray(np.empty((NC,ND)),dtype=np.float32)

        print("Calculating centroids")
        for k in range(NC):
            indices = np.where(clusters_pca == k)[0]
            centroids_F[k,:] = np.mean(data[indices,:],axis=0)
            print(k,len(indices),centroids_F[k,:])

        #save data
        new_data = np.c_[locations,clusters_pca]
        np.savetxt("../results/{dataset}_clusters_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),new_data,delimiter=",",fmt="%.4f")
        
        #stats
        setup_distances(scale,var_types,use_cat=True,targets=None)

        #PCA
        clusters = np.asfortranarray(clusters_pca,dtype=np.int8)
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND) 
        
        ret_pca = cl.clustering.dbi_index(centroids_F,data_F,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data_F,clusters,weights)
        
        #print('2D PCA',NC,ret_pca,sep=' ')
        print('2D PCA',NC,ret_pca,ret_sill,sep=',')
        cl.distances.reset()
        

    #save data
    NC = 3
    clustering_pca = KMeans(n_clusters=NC)
    clusters_pca = clustering_pca.fit_predict(pca_X)
    np.savetxt("../results/final_{dataset}_clusters_pca_{nclusters}.csv".format(dataset=filename,nclusters=NC),clusters_pca,delimiter=",",fmt="%.4f")
