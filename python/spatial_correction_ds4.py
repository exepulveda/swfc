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


CHECK_VALID = False

if __name__ == "__main__":
    np.random.seed(1634120)
    
    lambda_value = 0.25

    filename = "ds4"
    X = np.loadtxt("../../data/{dataset}_clusters.csv".format(dataset=filename),delimiter=",",skiprows=0)

    best_centroids = np.loadtxt("../../data/ds4_sfk_centroids_{lambda_value}.csv".format(lambda_value=lambda_value),delimiter=",")
    best_weights = np.loadtxt("../../data/ds4_sfk_weights_{lambda_value}.csv".format(lambda_value=lambda_value),delimiter=",")
    best_u = np.loadtxt("../../data/ds4_sfk_u_{lambda_value}.csv".format(lambda_value=lambda_value),delimiter=",")
    
    locations = X[:,0:2]
    
    
    #filter just 0-50
    #indices = np.where(locations[:,0] <= 50)[0]
    #indices = np.where(locations[indices,1] <= 50)[0]
    #locations = locations[indices,:]
    #X = X[indices,:]
    
    N,ND = X.shape
    NC = 4
    knn = 8
    
    #create neighbourdhood
    print("building neighbourhood")
    
    kdtree = cKDTree(locations)
    neighbourhood,distances = make_neighbourhood(kdtree,locations,knn)
    distances = np.array(distances)
    
    #create U fake
    #best_u = np.zeros((N,NC))
    #
    #for i in range(N):
    #    best_u[i,int(X[i,6])] = 1.0
    

    #spatial
    verbose_level = 2
    clusters_graph = graph_cut(locations,neighbourhood,best_u,unary_constant=100.0,smooth_constant=100.0,verbose=0)
    
    if ND > 12:
        X[:,12] = clusters_graph
        new_data = X
    else:
        new_data = np.c_[X,clusters_graph]
        
    np.savetxt("../../data/{dataset}_clusters.csv".format(dataset=filename),new_data,delimiter=",",fmt="%.4f")
