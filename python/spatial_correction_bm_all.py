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

import sys
sys.path += ['..']

import clusteringlib as cl

from graph_labeling import graph_cut, make_neighbourhood
from scipy.spatial import cKDTree

from case_study_bm import setup_case_study_ore, setup_distances

CHECK_VALID = False

if __name__ == "__main__":
    locations,data,min_values,max_values,scale,var_types,targets = setup_case_study_ore()
    seed = 1634120
    N,ND = data.shape
        
    np.random.seed(seed)
    lambda_value = 0.25
    targets = np.asfortranarray(np.percentile(data[:,-1], [15,50,85]),dtype=np.float32)
    var_types[-1] = 2
    force = (ND-1,0.15) #weight to target 15%
    
    knn = 15
    
    #create neighbourdhood
    print("building neighbourhood, location.shape=",locations.shape)
    kdtree = cKDTree(locations)
    neighbourhood,distances = make_neighbourhood(kdtree,locations,knn,max_distance=2.0)
    distances = np.array(distances)
    
    debug = False
    
    for NC in range(2,11):
        file_template = '../results/bm_{set}_wfc_%d.csv'%NC
        
        if debug: print("reading WFC results...")
        best_centroids = np.loadtxt(file_template.format(set='centroids'),delimiter=",")
        best_weights = np.loadtxt(file_template.format(set='weights'),delimiter=",")
        best_u = np.loadtxt(file_template.format(set='u'),delimiter=",")
        clusters = np.argmax(best_u,axis=1) 

        weights_F = np.asfortranarray(best_weights,dtype=np.float32)
        centroids_F = np.asfortranarray(best_centroids,dtype=np.float32)

        if debug: print("setup_distances...")
        setup_distances(scale,var_types,use_cat=True,targets=targets)        
        if debug: print("setup_distances...DONE")

        #spatial
        if debug: print("executing  graph_cut...")
        clusters_graph = np.int32(graph_cut(locations,neighbourhood,best_u,unary_constant=100.0,smooth_constant=10.0,verbose=0))
        np.savetxt("../results/final_bm_clusters_swfc_%d.csv"%NC,clusters_graph,delimiter=",",fmt="%d")
        

        if debug: print("executing  graph_cut...DONE")

        #computing indices
        clusters = np.asfortranarray(clusters,dtype=np.int8)
        ret_wfc_dbi = cl.clustering.dbi_index(centroids_F,data,clusters,weights_F)
        ret_wfc_sill= cl.clustering.silhouette_index(data,clusters,weights_F)

        #Spatial correction
        centroids_F = np.asfortranarray(np.empty((NC,ND)),dtype=np.float32)
        #calculate centroids back 
        for k in range(NC):
            indices = np.where(clusters_graph == k)[0]
            centroids_F[k,:] = np.mean(data[indices,:],axis=0)
        
        clusters = np.asfortranarray(clusters_graph,dtype=np.int8)
        
        #computing indices
        ret_swfc_dbi = cl.clustering.dbi_index(centroids_F,data,clusters,weights_F)
        ret_swfc_sill= cl.clustering.silhouette_index(data,clusters,weights_F)

        print("WFC/SWFC: DB,Sil:",NC,ret_wfc_dbi,ret_wfc_sill,ret_swfc_dbi,ret_swfc_sill,sep=',')
        
        cl.distances.reset()
