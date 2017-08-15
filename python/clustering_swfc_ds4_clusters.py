import sys
import random
import logging
import collections
import math
import sys
import argparse

sys.path += ['..']

import clusteringlib as cl
import numpy as np
import scipy.stats

import clustering_ga

from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from cluster_utils import fix_weights

from graph_labeling import graph_cut, make_neighbourhood
from scipy.spatial import cKDTree

CHECK_VALID = False

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

if __name__ == "__main__":
    filename = 'ds4'

    X = np.loadtxt("../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    locations = X[:,0:2]
    data = X[:,2:6] #0,1,2,5 are continues
    true_clusters = X[:,6]
    N,ND = data.shape

    print(N,ND)
    
    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)
    scale = np.std(data,axis=0)
    var_types = np.ones(ND)

    seed = 1634120

    m = 2.0

    verbose=0
    lambda_value = 0.25
    
    #filename_template = "../../data/bm_{tag}_%s_%s_sfc_{nc}.csv"%(args.target,args.force)

    #spatial correction
    
    kdtree = cKDTree(locations)
    neighbourhood,distances = make_neighbourhood(kdtree,locations,8,max_distance=np.inf)


    ngen=50
    npop=50
    cxpb=0.8
    mutpb=0.4
    stop_after=20

    for NC in range(2,11):
        np.random.seed(seed)
        random.seed(seed)
        cl.utils.set_seed(seed)
        
        cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
        cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
        
        
        #initial centroids at random
        indices = np.random.choice(N,size=NC,replace=False)
        current_centroids = np.asfortranarray(np.empty((NC,ND)))
        current_centroids[:,:] = data[indices,:]

        #initial weights are uniform
        weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND)
        
        for k in range(100):
            best_centroids,best_u,best_energy_centroids,best_jm,current_temperature,evals = clustering_ga.optimize_centroids(
                    data,
                    current_centroids,
                    weights,
                    m,
                    lambda_value,
                    var_types,
                    {},
                    ngen=ngen,npop=npop,cxpb=cxpb,mutpb=mutpb,stop_after=stop_after,
                    min_values = min_values,
                    max_values = max_values,
                    verbose=verbose)

            #print("centroids",best_centroids,best_energy_centroids,"jm",best_jm)
                    
                    
            u = best_u
            N,NC = u.shape
            
            clusters = np.argmax(u,axis=1)
            
            centroids = best_centroids.copy()
            
            #print("centroids",centroids)
            #print("u",u)
            counter = collections.Counter(clusters)
            #print("number of clusters: ",counter.most_common())

            best_weights,best_u,best_energy_weights,evals = clustering_ga.optimize_weights(
                    data,
                    centroids,
                    weights,
                    m,
                    lambda_value,
                    ngen=ngen,npop=npop,cxpb=cxpb,mutpb=mutpb,stop_after=stop_after,
                    force=None,
                    verbose=verbose)

            clusters = np.argmax(best_u,axis=1)

            weights = best_weights.copy()

            current_centroids = best_centroids.copy()
            #print(lambda_value,k,best_energy_centroids,best_energy_weights,"jm",best_jm)

            #print('iteration',k,best_energy_centroids,best_energy_weights)

            #save data
            new_data = np.c_[locations,clusters]
            
            
            #np.savetxt(filename_template.format(tag='clusters',nc=NC),new_data,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='centroids',nc=NC),current_centroids,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='u',nc=NC),best_u,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='weights',nc=NC),best_weights,delimiter=",",fmt="%.4f")
            
            if abs(best_energy_centroids - best_energy_weights) < 1e-2:
                break

        #spatial correction
        clusters_graph = graph_cut(locations,neighbourhood,best_u,unary_constant=100.0,smooth_constant=30.0,verbose=0)
        
        centroid = np.asfortranarray(best_centroids,dtype=np.float32)
        weights = np.asfortranarray(best_weights,dtype=np.float32)
        clusters = np.asfortranarray(clusters,dtype=np.int8)
        
        ret_fc = cl.clustering.dbi_index(centroid,data,clusters,weights)
        ret_sill= cl.clustering.silhouette_index(data,clusters,weights)

        clusters = np.asfortranarray(clusters_graph,dtype=np.int8)
        ret_fc_sc = cl.clustering.dbi_index(centroid,data,clusters,weights)
        ret_sill_sc = cl.clustering.silhouette_index(data,clusters,weights)
        
        print("DB Index:",NC,ret_fc,ret_sill,ret_fc_sc,ret_sill_sc,sep=',')
        cl.distances.reset()
