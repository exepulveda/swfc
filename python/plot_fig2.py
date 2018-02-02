import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters


parser = argparse.ArgumentParser(description="spatial clustering")
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    filename = "ds4"

    X = np.loadtxt("../data/{dataset}_clusters.csv".format(dataset=filename),delimiter=",",skiprows=0)
    
    
    locations = X[:,0:2]
    true_clusters = X[:,6]
    kmeans_clusters = X[:,9]
    pca_clusters = X[:,10]
    fc_clusters = X[:,11]
    sfc_clusters = X[:,12]
    

    nclusters = 4
    nvars = 4
    
    clustering_method = [
        ('TC','fig1'),
        ('KMeans','fig2a'),
        ('PCA','fig2b'),
        ('WFC','fig2c'),
        ('SWFC','fig2d'),
    ]
    
    equivalences = {}
    #equivalences[1] = {0:0,1:1,2:2,3:3}
    equivalences[2] = {0:3, 1:1, 2:0, 3:2}
    equivalences[3] = {0:2, 1:0, 2:3, 3:1}
    equivalences[4] = equivalences[3]    
    
    color = ['r','b','g','c']
    
    for i,clusters in enumerate([true_clusters,kmeans_clusters,pca_clusters,fc_clusters,sfc_clusters]): #clustering method
        fig, ax = plt.subplots() 
        ax.set_aspect('equal')


        if i in equivalences:
            adjust_cluster = adjust_clusters(clusters,equivalences[i])
        else:
            adjust_cluster =  clusters
            
            
        #print("adjust_cluster",i,adjust_cluster)

        for c in range(nclusters):
            c1 = np.where(adjust_cluster == c)[0]

            plt.plot(locations[c1,0],locations[c1,1],marker='o',color=color[c],linestyle='None')

        plt.savefig("../figures/case_ds4/{cm}.pdf".format(cm=clustering_method[i][1]),bbox_inches='tight')
