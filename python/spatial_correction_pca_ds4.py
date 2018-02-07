import sys
import random
import logging
import collections
import math
import sys
import json

sys.path += ['..']

import numpy as np
import scipy.stats


from graph_labeling import graph_cut, make_neighbourhood
from scipy.spatial import cKDTree

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import clusteringlib as cl

from cluster_utils import crisp_to_fuzzy, crisp_centroids

import matplotlib as mpl
#mpl.use('agg')

import matplotlib.pyplot as plt


'''
This script if for adpat yhe Kmeans results in order to use spatail correction
'''

CHECK_VALID = False

if __name__ == "__main__":
    seed = 1634120

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

    NC = 4
    ND_PCA = 2

    pca = PCA(n_components=ND_PCA,whiten=True)
    pca_X = pca.fit_transform(data_scaled)
    data_F = np.asfortranarray(pca_X,dtype=np.float32)

    clustering_pca = KMeans(n_clusters=NC)
    clusters_pca = clustering_pca.fit_predict(pca_X)

    #Generate centroids
    centroids = crisp_centroids(pca_X,clusters_pca,NC)
    #print('centroids',centroids)

    #Generate U matrix for kmeas
    u = crisp_to_fuzzy(pca_X,centroids,p=2.0)
    #print('u',u)

    knn = 8
    
    #create neighbourdhood EW
    print("building neighbourhood, location.shape=",locations.shape)
    kdtree = cKDTree(locations)
    neighbourhood,distances = make_neighbourhood(kdtree,locations,knn,max_distance=np.inf)
    distances = np.array(distances)

    #spatial EW
    verbose_level = 2
    clusters_graph = graph_cut(locations,neighbourhood,u,unary_constant=100.0,smooth_constant=80.0,verbose=1)
    
    for i in range(N):
        print(i,clusters_pca[i],u[i],clusters_graph[i],sep=',')
        
    np.savetxt("../results/final_ds4_clusters_spca_4.csv",clusters_graph,delimiter=",",fmt="%.4f")

    quit()
    color = ['r','b','g','c']

    fig, ax = plt.subplots() 

    for c in range(NC):
        c1 = np.where(clusters_graph_ew == c)[0]

        plt.plot(locations[c1,0],locations[c1,1],marker='o',color=color[c],linestyle='None')

    plt.show()
    #plt.savefig("../../figures/scatter-{cm}-ds4".format(cm=clustering_method[i]))
        
    
    #np.savetxt("../results/final_2d_clusters_sfcew_4.csv",clusters_graph_ew,delimiter=",",fmt="%.4f")

    #spatial 
    #verbose_level = 2
    #clusters_graph = graph_cut(locations_ore,neighbourhood,best_u,unary_constant=100.0,smooth_constant=50.0,verbose=1)
    
    #for i in range(N):
    #    print(i,cluster[i],best_u[i],cluster[neighbourhood[i]],clusters_graph[i],sep=',')

    #np.savetxt("../results/final_2d_clusters_sfc_4.csv",clusters_graph,delimiter=",",fmt="%.4f")

