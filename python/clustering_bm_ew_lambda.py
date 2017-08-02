import sys
import random
import logging
import collections
import math
import sys

import clusteringlib as cl
import numpy as np
import scipy.stats

import clustering_sa as csa

from clustering_ga_ew import optimize_weights,optimize_centroids

from bm_case_study import bm_variables,setup_case_study,setup_distances

import sklearn.cluster
import sklearn.metrics

CHECK_VALID = False

if __name__ == "__main__":

    locations,data,scale,var_types,targets = setup_case_study()

    NC = 4
    N,ND = data.shape
    
    seed = 1634120

    distances_cat = np.asfortranarray(np.ones((3,3)),dtype=np.int32)
    distances_cat[0,0] = 0.0
    distances_cat[1,1] = 0.0
    distances_cat[2,2] = 0.0

    weights = np.empty(ND,dtype=np.float32)
    
    setup_distances(scale,var_types,distances_cat,targets=None)
    
    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)    
    
    m = 2.5
    #force=(3,0.3)
    force=None
    
    #assing centroids at random from data
    #generate initial solution
    current_centroids = np.empty((NC,ND))

    print('lambda,weights1,weights2,weights3,weights4',sep=",")

    seed = 1634120
    
    ngen=50
    npop=100
    cxpb=0.8
    mutpb=0.4
    stop_after=10
    verbose=0

    for lambda_value in np.arange(0.0,1.0,0.01):
        np.random.seed(seed)
        cl.utils.set_seed(seed)
        random.seed(seed)

        weights = np.ones(ND) / ND
        for i in range(NC):
            j = np.random.choice(N)
            current_centroids[i,:] = data[j,:]
        #print('lambda_value',lambda_value)
        #print('initial_weights',weights)

        for k in range(100):
            best_centroids,best_u,best_energy_centroids,best_jm,current_temperature,evals = optimize_centroids(
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

            best_weights,best_u,best_energy_weights,evals = optimize_weights(
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

            #print('iteration',k,best_energy_centroids,best_energy_weights)

            #save data
            #new_data = np.c_[locations,clusters]
            
            
            #np.savetxt(filename_template.format(tag='clusters',nc=NC),new_data,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='centroids',nc=NC),current_centroids,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='u',nc=NC),best_u,delimiter=",",fmt="%.4f")
            #np.savetxt(filename_template.format(tag='weights',nc=NC),best_weights,delimiter=",",fmt="%.4f")
            
            if abs(best_energy_centroids - best_energy_weights) < 1e-2:
                break

        #cl.distances.reset()

        weights = list(weights)
        print(lambda_value,",",",".join([str(x) for x in weights]))
