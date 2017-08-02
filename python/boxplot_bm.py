import numpy as np
import pickle
import logging
import argparse
import csv
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters

bm_variables = ['RockType','Mgt','Hem','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']


if __name__ == "__main__":
    X = np.loadtxt("../../data/bm_clusters.csv",delimiter=",",skiprows=0)
    locations = X[:,0:3]

    kmeans_clusters = X[:,6]
    pca_clusters = X[:,7]
    fc1_clusters = X[:,8]
    fc2_clusters = X[:,9]
    fc3_clusters = X[:,10]
    
    clusters = np.c_[kmeans_clusters,pca_clusters,fc1_clusters,fc2_clusters,fc3_clusters]
    
    X = np.loadtxt("../../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv",delimiter=",",skiprows=1)
    locations = X[:,1:4]
    rocktype = [4]
    mineralogy = [x for x in range(5,11)]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [69]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats
    NC = 6
    N,ND = values.shape
    
    labels = ["All","C1","C2","C3","C4","C5","C6"]
    mins = [0,0,0,60]
    maxs = [2.0,5.0,50.0,100.0]
    #plot
    
    clustering_method = ['K-Means','PCA + K-Means','FC1','FC2','FC3']
    
    equivalences = {}
    #equivalences[1] = {0:0,1:1,2:2,3:3}
    #equivalences[2] = {0:3, 1:1, 2:0, 3:2}
    #equivalences[3] = {0:2, 1:0, 2:3, 3:1}
    #equivalences[4] = equivalences[3]

    for i,clusters in enumerate([kmeans_clusters,pca_clusters,fc1_clusters,fc2_clusters,fc3_clusters]): #clustering method
        
        if i in equivalences:
            adjust_cluster = adjust_clusters(clusters,equivalences[i])
        else:
            adjust_cluster =  clusters
                    
        for v in range(ND):
            fig, ax = plt.subplots(figsize=(6, 6)) 
            #ax.set_ylim([mins[v],maxs[v]])
            d = [values[:,v]]

            for c in range(NC):
                d += [values[adjust_cluster==c,v]]

            bx = ax.boxplot(d,labels=labels,showmeans=True)

            #plt.legend(['Cluster 1','Cluster 2','Cluster 3','Cluster 4'],loc='upper right',bbox_to_anchor=(0, 1))
        #plt.title(clustering_method[i])
            plt.savefig("../../figures/boxplot-{var}-{cm}-bm".format(var=bm_variables[v],cm=clustering_method[i]))
