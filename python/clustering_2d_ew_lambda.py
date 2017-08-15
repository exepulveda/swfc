import sys
import random
import logging
import collections
import math
import sys

import clusteringlib as cl
import numpy as np
import scipy.stats

from clustering_ga_ew import optimize_weights,optimize_centroids

from case_study_2d import attributes,setup_case_study,setup_distances

import sklearn.cluster
import sklearn.metrics

CHECK_VALID = False

if __name__ == "__main__":

    locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
    data = data_ore.copy()
    
    NC = 3
    N,ND = data.shape
    
    weights = np.empty(ND,dtype=np.float32)
    setup_distances(scale,var_types,categories,targets=None)
    
    m = 2.5
    #force=(3,0.3)
    force=None
    
    #assing centroids at random from data
    #generate initial solution
    current_centroids = np.empty((NC,ND))

    seed = 1634120
    
    ngen=10
    npop=50
    cxpb=0.8
    mutpb=0.4
    stop_after=10
    verbose=0

    for lambda_value in np.arange(0.02,1.0,0.02):
        #print('performin clustering, lambda',lambda_value)
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
        print("LAMBDA 2D EW,",lambda_value,",",",".join([str(x) for x in weights]))
