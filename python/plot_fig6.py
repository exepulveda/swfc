import numpy as np
import pickle
import logging
import argparse
import csv
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters

if __name__ == "__main__":
    filename = "ds4"
    X = np.loadtxt("../data/{dataset}_clusters.csv".format(dataset=filename),delimiter=",",skiprows=0)
    locations = X[:,0:2]
    values = X[:,2:6]
    true_clusters = X[:,6]
    kmeans_clusters = X[:,9]
    pca_clusters = X[:,10]
    fc_clusters = X[:,11]
    sfc_clusters = X[:,12]
    

    nclusters = 4
    nvars = 4
    
    labels = ["All","C1","C2","C3","C4"]
    mins = [0,0,0,60]
    maxs = [2.0,5.0,50.0,100.0]
    #plot
    
    clustering_method = [
        ('TC','fig6'),
        ('KMeans','fig6a'),
        ('PCA','fig6b'),
        ('WFC','fig6c'),
        ('SWFC','fig6d'),
    ]    
    
    equivalences = {}
    #equivalences[1] = {0:0,1:1,2:2,3:3}
    equivalences[2] = {0:3, 1:1, 2:0, 3:2}
    equivalences[3] = {0:2, 1:0, 2:3, 3:1}
    equivalences[4] = equivalences[3]
    
    variables = ['Cu','Fe','As','Recovery']

    for i,clusters in enumerate([true_clusters,kmeans_clusters,pca_clusters,fc_clusters,sfc_clusters]): #clustering method
        fig, axarr = plt.subplots(nvars,figsize=(6, 6), sharex=True) 
        
        if i in equivalences:
            adjust_cluster = adjust_clusters(clusters,equivalences[i])
        else:
            adjust_cluster =  clusters
                    
        for v in range(nvars):
            ax = axarr[v]
            ax.set_ylim([mins[v],maxs[v]])
            ax.set_title(variables[v],fontsize = 8)
            d = [values[:,v]]

            for c in range(nvars):
                d += [values[adjust_cluster==c,v]]

            bx = ax.boxplot(d,labels=labels)

            #plt.legend(['Cluster 1','Cluster 2','Cluster 3','Cluster 4'],loc='upper right',bbox_to_anchor=(0, 1))
        #plt.title(clustering_method[i])
        plt.savefig("../figures/case_ds4/{cm}.pdf".format(cm=clustering_method[i][1]),bbox_inches='tight')
