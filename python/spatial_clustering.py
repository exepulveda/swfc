import numpy as np
import pickle
import logging
import argparse

from cosa import calculate_weights
from cosa import calculate_weights_fuzzy
from cosa import fuzzy_dispersion_loop
from cosa import dispersion_loop

from cmean import fuzzy_cmeans
from categorical_utils import categorical_to_simplex
from cluster_utils import create_clusters_dict

from cluster_metrics import *

from distance_utils import simplex_distance
from distance_utils import standard_distance
from distance_utils import DistanceCalculator
from distance_utils import WeightedDistanceCalculator

from graph_labeling import graph_cut

from scipy.spatial import cKDTree

def swc(data,locations,ndim,nclusters,distance_function_weighted,
        adj_set,
        alpha=1.2,
        lambda_value = 0.2,
        inc_eta=0.1,
        max_iterations=50,
        verbose=False,
        apply_spatial_step = False,
        apply_spatial_end = False,
        full_stats =False
        ):
    #this function applies spatial constraint at the end only
    n,p = data.shape

    logging.info("Spatial Weighted clustering")    
    logging.info("clustres: %d",nclusters)    
    logging.info("data shape: %dx%d",n,p)    
    logging.info("alpha: %f",alpha)    
    logging.info("lambda_value: %f",lambda_value)    


    #calculate dissimilarity matrix setting weights = 1.0
    logging.info("Calculating dissimilarity matrix...")    
    weights = np.ones(ndim)
    dissimilarity = distance_function_weighted.dissimilarity(data,data,debug=False)
    logging.info("Calculating dissimilarity matrix. DONE")    
    
    #np.savetxt("../outputs/diss_targets.csv",dissimilarity[:,:,4])
    #quit()

    #initial weights
    weights = np.ones((nclusters,ndim)) / ndim
    
    #initial partition is random
    u=None

    iterations = 0
    eta = lambda_value
    
    init_u = None
    
    #outter loop is for eta
    while True:
        #inner loop looks at weights to stabilise
        #logging.info("current iteration: %d",iterations)
        weights_old = weights.copy()
        #set new weights
        #distance_function_weighted.set_weights(weights)
        #distance_function_weighted.set_lambda(eta)
        
        distance_function_tmp = lambda x,y: distance_function_weighted.distance_centroids(x,y,weights=weights)


        prototype, u, u0, d, jm, p, fpc = fuzzy_cmeans(data, nclusters, alpha, distance_function_tmp,init=init_u,verbose=verbose)
        init_u = u.copy()

        logging.info("fuzzy cmeans: inital jm=%f, last jm=%f, fpc=%f",jm[0],jm[-1],fpc)

        stat = {
                "step":iterations,
                "eta":eta,
                "iterations":p,
                "jm":jm[-1],
                "fpc":fpc,
            }

        #clusters
        clusters = np.argmax(u,axis=1)
        clusters_dict = create_clusters_dict(clusters)
        
        
        print('prototype',iterations,prototype)
        #np.savetxt("../outputs/u-{0}.csv".format(iterations),u)
        #quit()

        #fuzzy dispersion
        #skl = fuzzy_dispersion_loop(dissimilarity,u**alpha)
        #print skl
        
        #skl = dispersion_loop(dissimilarity,nclusters,clusters_dict)
        #print skl
        
       
        #fuzzy score
        #try:
        #except Exception as e:
        #    print e
        #    fuzzy_score = -666
            
        #logging.info("fuzzy score: %f",fuzzy_score)


        #report
        #np.savetxt("../outputs/D.csv",D)
        
        #try:
        #D = distance_function_weighted.distance_max(data,data,nclusters,clusters,weights)
        #score = silloutte(D,clusters,nclusters)
        #except Exception as e:
        #    print e
        #    score = -666


        #vpc = partition_coefficient(u)
        #vpe = partition_entropy(u)
        compact = 0.0 #compactness(nclusters,clusters,adj_set)




        if full_stats:
            centroids = prototype.T
            
            score = silhouette_score(data,clusters)

            score_vpc = vpc(u)
            score_mvpc = mvpc(u,pre_vpc=score_vpc)
            score_vpe = vpe(u)
            score_vavcd = vavcd(data,u,alpha,centroids)
            score_vfs = vfs(data,u,alpha,centroids)
            score_vxb = vxb(data,u,alpha,centroids)
            score_vmcd = vmcd(centroids)

            stat["score_vpc"] = score_vpc
            stat["score_mvpc"] = score_mvpc
            stat["score_vpe"] = score_vpe
            stat["score_vavcd"] = score_vavcd
            stat["score_vfs"] = score_vfs
            stat["score_vxb"] = score_vxb
            stat["score_vmcd"] = score_vmcd
                
        logging.info("stats before spatial=%s",stat)
        
        
        #update weiths
        if apply_spatial_step:
            #spatial
            clusters_graph = graph_cut(locations,adj_set,u)
            clusters_graph_dict = create_clusters_dict(clusters_graph)
            clusters = clusters_graph.copy()
            weights = calculate_weights(dissimilarity,nclusters,clusters_graph_dict,lambda_value)
        else:
            weights = calculate_weights_fuzzy(dissimilarity,u**alpha,lambda_value,debug=False)

        for i in range(nclusters):
            logging.debug("weights[cluster=%d]=%s",i,weights[i,:])
        #print weights.shape,weights_old.shape
        
        #print "weights",weights
        
        #np.testing.assert_allclose(np.sum(weights,axis=1),1.0)

        iterations += 1
        eta = eta + inc_eta*lambda_value
        diff = np.sum((weights - weights_old)**2)
        
        logging.info("diff weights=%f",diff)
        #logging.info("weights=%s",weights)
        #stop condition
        if iterations >= max_iterations:
            break
        else:
            #calculate weights difference
            if diff < 1e-3:
                break
        
    #stats += [stat]        

    if apply_spatial_end:
        #spatial at end
        clusters_graph = graph_cut(locations,adj_set,u)
    else:
        clusters_graph = None

    return clusters,prototype,u,weights,stat,clusters_graph
    

