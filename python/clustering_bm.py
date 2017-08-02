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

import clustering_pso as clustering
import clustering_ga

from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from cluster_utils import fix_weights

CHECK_VALID = False

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-t', "--target",     action='store_true', dest="target", required=False, help="use target distance on recovery")
parser.add_argument('-f', "--force",     action='store_true', dest="force", required=False, help="force weights")

if __name__ == "__main__":
    args = parser.parse_args()
    
    
    print(args.target)
    print(args.force)
    
    X = np.loadtxt("../../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv",delimiter=",",skiprows=1)
    locations = X[:,1:4]
    
    rocktype = [4]
    mineralogy = [x for x in range(5,11)]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [69]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    seed = 1634120
    N,ND = values.shape

    values[:,0] = clustering.recode_categorical_values(values[:,0],[300,400,500])
    data = np.asfortranarray(np.float32(values))
    scale = np.std(data,axis=0)
    means = np.mean(data,axis=0)

    var_types = np.ones(ND)
    var_types[0] = 3

    if args.target:
        targets = np.asfortranarray(np.percentile(values[:,-1], [25,50,75]),dtype=np.float32)
        var_types[-1] = 2
    
    m = 2.5
    force=None

    targets = np.asfortranarray(np.percentile(values[:,-1], [15,50,85]),dtype=np.float32)
    print('targets',targets)

    verbose=1
    lambda_value = 0.25
    
    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)
    
    #hack because is categorical
    min_values[0] = 0.0
    
    filename_template = "../../data/bm_{tag}_%s_%s_sfc_{nc}.csv"%(args.target,args.force)

    ngen=100
    npop=100
    cxpb=0.8
    mutpb=0.4
    stop_after=10

    for NC in range(9,20):
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

        if args.target:
            cl.distances.set_targeted(23,targets,False)
            if args.force:
                force = (22,0.2)


        #initial centroids
        kmeans_method = KMeans(n_clusters=NC,random_state=seed)
        kmeans_method.fit(values)
        
        current_centroids = np.asfortranarray(np.empty((NC,ND)))
        current_centroids[:,:] = kmeans_method.cluster_centers_

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
