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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-t', "--target",     action='store_true', dest="target", required=False, help="use target distance on recovery")
parser.add_argument('-f', "--force",     action='store_true', dest="force", required=False, help="force weights")

from case_study_2d import attributes,setup_case_study,setup_distances


if __name__ == "__main__":
    args = parser.parse_args()
    
    locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
    data = data_ore.copy()
    
    N,ND = data.shape
    
    m = 2.0
    force=None

    #targets = np.asfortranarray(np.percentile(values[:,-1], [15,50,85]),dtype=np.float32)
    targets = None
    print('targets',targets)

    seed = 1634120
    verbose=1
    lambda_value = 0.20
    
    ngen=100
    npop=200
    cxpb=0.8
    mutpb=0.4
    stop_after=10
    
    filename_template = "../results/final_2d_{tag}_fc_{nc}.csv"
    fc_clusters_all = np.empty(len(locations))

    NC = 4

    np.random.seed(seed)
    random.seed(seed)
    cl.utils.set_seed(seed)
    
    setup_distances(scale,var_types,categories,targets=targets)
    
    #initial centroids
    kmeans_method = KMeans(n_clusters=NC,random_state=seed)
    kmeans_method.fit(data)
    
    current_centroids = np.asfortranarray(np.empty((NC,ND)))
    current_centroids[:,:] = kmeans_method.cluster_centers_
    
    for i in range(ND):
        current_centroids[:,i] =  np.clip(current_centroids[:,i],min_values[i],max_values[i])


    #initial weights are uniform
    weights = np.asfortranarray(np.ones((NC,ND),dtype=np.float32)/ ND)
    
    if args.target:
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

        #print('2D FC iteration',k,best_energy_centroids,best_energy_weights)

        if abs(best_energy_centroids - best_energy_weights) < 1e-2:
            break


    centroid = np.asfortranarray(best_centroids,dtype=np.float32)
    weights = np.asfortranarray(best_weights,dtype=np.float32)
    clusters = np.asfortranarray(clusters,dtype=np.int8)
    
    ret_fc = cl.clustering.dbi_index(centroid,data,clusters,weights)
    ret_sill= cl.clustering.silhouette_index(data,clusters,weights)
    
    print("2D FC Index:",NC,ret_fc,ret_sill,sep=',')
    cl.distances.reset()

    #save data
    np.savetxt(filename_template.format(tag='clusters',nc=NC),clusters,delimiter=",",fmt="%.4f")
    np.savetxt(filename_template.format(tag='centroids',nc=NC),current_centroids,delimiter=",",fmt="%.4f")
    np.savetxt(filename_template.format(tag='u',nc=NC),best_u,delimiter=",",fmt="%.4f")
    np.savetxt(filename_template.format(tag='weights',nc=NC),best_weights,delimiter=",",fmt="%.4f")            
    
    
