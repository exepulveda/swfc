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

from bm_case_study import setup_case_study

CHECK_VALID = False

if __name__ == "__main__":
    locations,data,scale,var_types,targets = setup_case_study()
    seed = 1634120
    
    np.random.seed(seed)
    lambda_value = 0.25
    NC = 4
    target = False
    force = False
    
    file_template = '../../data/bm_{set}_%s_%s_sfc_%s.csv'%(target,force,NC)

    best_centroids = np.loadtxt(file_template.format(set='centroids'),delimiter=",")
    best_weights = np.loadtxt(file_template.format(set='weights'),delimiter=",")
    best_u = np.loadtxt(file_template.format(set='u'),delimiter=",")
    
    clusters = np.argmax(best_u,axis=1)    
    
    N,ND = data.shape

    knn = 25
    
    #create neighbourdhood
    print("building neighbourhood, location.shape=",locations.shape)
    kdtree = cKDTree(locations)
    neighbourhood,distances = make_neighbourhood(kdtree,locations,knn,max_distance=2.0)
    distances = np.array(distances)
    
    #spatial
    verbose_level = 2
    clusters_graph = graph_cut(locations,neighbourhood,best_u,unary_constant=100.0,smooth_constant=10.0,verbose=1)
    
    for i in range(N):
        print(i,clusters[i],best_u[i],clusters_graph[i],sep=',')
    
    #np.savetxt("../../data/{dataset}_clusters.csv".format(dataset=filename),new_data,delimiter=",",fmt="%.4f")
