import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import pickle
import logging
import copy

from pygco import cut_simple, cut_from_graph, Graph #~/installing/gco_python

from collections import defaultdict,Counter
import itertools


import clustering

def average_distance(values):
    y = pdist(values, 'euclidean')
    return np.mean(y)


def smooth_cost(i,j,labels,values,beta):
    if labels[i] == labels[j]:
        return 0
    else:
        dist_values = np.linalg.norm(values[i]-values[j])
        return beta*np.exp(-dist_values)
        
def make_neighbourhood(kdtree,locations,k=5,max_distance=np.inf):
    #build edges as KNN
    ret = []
    dist = []
    n = len(locations)
    for i in range(n):
        #find KNN
        distances,indices = kdtree.query(locations[i],k=(k+1),distance_upper_bound=max_distance) #k+1 to avoid finding its self

        ret += [[indices[j] for j in range(k+1) if indices[j] != i and not np.isinf(distances[j])]]
        dist += [[distances[j] for j in range(k+1) if indices[j] != i and not np.isinf(distances[j])]]

    return ret,dist
        
def graph_cut(locations,neighbourhood,prob,unary_constant=100.0,smooth_constant=100.0,verbose=0):
    n,nlabels  = prob.shape
    assert locations.shape[0] == n
    

    edges = []
    processed_pairs = set()

    #build edges as KNN
    for i in range(n):
        #find KNN
        indices = neighbourhood[i]
        
        if verbose > 1: print("indices",i,indices)
        
        for index in indices:
            if index != i:
                edges += [[i,index]]
                
                #print "edges",i,index,d

    unary_cost = -np.log(prob+1e-5)

    smooth_cost = np.ones((nlabels,nlabels))
    for c1 in range(nlabels):
        smooth_cost[c1,c1] = 0

    edges = np.int32(edges)
    unary_cost = np.int32(unary_cost*unary_constant)
    smooth_cost = np.int32(smooth_cost*smooth_constant)

    graph = Graph(edges,unary_cost,smooth_cost)
    
    if verbose > 0: print("Energy",graph.energy())
    if verbose > 0: print("Data Energy",graph.data_energy())
    if verbose > 0: print("Label Energy",graph.label_energy())
    if verbose > 0: print("Smooth Energy",graph.smooth_energy())
    if verbose > 0: print("All Energy",graph.data_energy() + graph.label_energy() + graph.smooth_energy())

   
    ret = graph.cut()#,algorithm="swap")

    if verbose > 0: print("Energy",graph.energy())
    if verbose > 0: print("Data Energy",graph.data_energy())
    if verbose > 0: print("Label Energy",graph.label_energy())
    if verbose > 0: print("Smooth Energy",graph.smooth_energy())
    if verbose > 0: print("All Energy",graph.data_energy() + graph.label_energy() + graph.smooth_energy())
    

    if verbose > 1: print(ret)

    return ret
