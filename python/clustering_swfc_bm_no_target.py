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

CHECK_VALID = False

from case_study_bm import attributes,setup_case_study_ore,setup_case_study_all,setup_distances

if __name__ == "__main__":
    locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore(a=0.999)
    N,ND = data.shape

    print(N,ND)
    #print(min_values)
    #print(max_values)
    #print(scale)

    seed = 1634120

    #targets = np.asfortranarray(np.percentile(data[:,-1], [15,50,85]),dtype=np.float32)
    #var_types[-1] = 2
    
    #print('targets',targets)
    
    m = 2.0
    force=None

    verbose=1
    lambda_value = 0.25
    
    filename_template = "../results/bm_{tag}_swfc_{nc}_no_target.csv"

    ngen=300
    npop=200
    cxpb=0.8
    mutpb=0.4
    stop_after=20

    NC = 3
    np.random.seed(seed)
    random.seed(seed)
    cl.utils.set_seed(seed)
    
    cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
    cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
    distances_cat = np.asfortranarray(np.ones((3,3)),dtype=np.float32)
    distances_cat[0,0] = 0.0
    distances_cat[1,1] = 0.0
    distances_cat[2,2] = 0.0
    cl.distances.set_categorical(1, 3,distances_cat)

    #cl.distances.set_targeted(23,targets,False)
    #force = (22,0.15)
    force = None

    #initial centroids at random
    indices = np.random.choice(N,size=NC,replace=False)
    current_centroids = np.asfortranarray(np.empty((NC,ND)))
    current_centroids[:,:] = data[indices,:]

    #initial weights are uniform
    weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND)
    
    for c in range(NC):
        weights[c,:] = fix_weights(weights[c,:],force=force)
            
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
                force=force,
                verbose=verbose)

        clusters = np.argmax(best_u,axis=1)

        weights = best_weights.copy()

        current_centroids = best_centroids.copy()
        #print(lambda_value,k,best_energy_centroids,best_energy_weights,"jm",best_jm)

        print('iteration',k,best_energy_centroids,best_energy_weights)

        #save data
        new_data = np.c_[locations,clusters]
        
        
        np.savetxt(filename_template.format(tag='clusters',nc=NC),new_data,delimiter=",",fmt="%.4f")
        np.savetxt(filename_template.format(tag='centroids',nc=NC),current_centroids,delimiter=",",fmt="%.4f")
        np.savetxt(filename_template.format(tag='u',nc=NC),best_u,delimiter=",",fmt="%.4f")
        np.savetxt(filename_template.format(tag='weights',nc=NC),best_weights,delimiter=",",fmt="%.4f")
        
        if abs(best_energy_centroids - best_energy_weights) < 1e-2:
            break


    centroid = np.asfortranarray(best_centroids,dtype=np.float32)
    weights = np.asfortranarray(best_weights,dtype=np.float32)
    clusters = np.asfortranarray(clusters,dtype=np.int8)
    
    ret_fc = cl.clustering.dbi_index(centroid,data,clusters,weights)
    ret_sill= cl.clustering.silhouette_index(data,clusters,weights)
    
    print("DB Index:",NC,ret_fc,ret_sill,sep=',')
    cl.distances.reset()
