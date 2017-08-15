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

from case_study_2d import attributes,setup_case_study,setup_distances

CHECK_VALID = False

if __name__ == "__main__":
    locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
    data = data_ore.copy()

    seed = 1634120
    
    np.random.seed(seed)
    lambda_value = 0.25
    NC = 4
    target = False
    force = False

    filename_template_ew = "../results/final_2d_{tag}_fcew_4.csv"
    filename_template = "../results/final_2d_{tag}_fc_4.csv"

    #clusters_ew = np.loadtxt(filename_template_ew.format(tag='clusters'),delimiter=",")[:,2]
    #clusters = np.loadtxt(filename_template.format(tag='clusters'),delimiter=",")[:,2]

    best_u = np.loadtxt(filename_template.format(tag='u'),delimiter=",")
    best_u_ew = np.loadtxt(filename_template_ew.format(tag='u'),delimiter=",")
    
    cluster_ew = np.argmax(best_u_ew,axis=1)
    cluster = np.argmax(best_u,axis=1)
    
    N,ND = data.shape

    knn = 8
    
    #create neighbourdhood EW
    print("building neighbourhood, location.shape=",locations.shape)
    kdtree = cKDTree(locations_ore)
    neighbourhood,distances = make_neighbourhood(kdtree,locations_ore,knn,max_distance=np.inf)
    distances = np.array(distances)


    #spatial EW
    verbose_level = 2
    clusters_graph_ew = graph_cut(locations_ore,neighbourhood,best_u_ew,unary_constant=100.0,smooth_constant=80.0,verbose=1)
    
    for i in range(N):
        print(i,cluster_ew[i],best_u_ew[i],cluster_ew[neighbourhood[i]],clusters_graph_ew[i],sep=',')
    
    np.savetxt("../results/final_2d_clusters_sfcew_4.csv",clusters_graph_ew,delimiter=",",fmt="%.4f")

    #spatial 
    verbose_level = 2
    clusters_graph = graph_cut(locations_ore,neighbourhood,best_u,unary_constant=100.0,smooth_constant=50.0,verbose=1)
    
    for i in range(N):
        print(i,cluster[i],best_u[i],cluster[neighbourhood[i]],clusters_graph[i],sep=',')

    np.savetxt("../results/final_2d_clusters_sfc_4.csv",clusters_graph,delimiter=",",fmt="%.4f")

