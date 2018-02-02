import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cluster_utils import create_clusters_dict
from cluster_metrics import separation

from plotting import scatter_clusters, plot_confusion_matrix

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters

if __name__ == "__main__":
    mpl.use('agg')
    filename = "ds4"
    X = np.loadtxt("../data/{dataset}_clusters.csv".format(dataset=filename),delimiter=",",skiprows=0)
    locations = X[:,0:2]
    values = X[:,2:6]
    true_clusters = X[:,6]
    kmeans_clusters = X[:,9]
    pca_clusters = X[:,10]
    fc_clusters = X[:,11]
    sfc_clusters = X[:,12]
    
    #compute centroids
    NC = 4
    N,ND = values.shape
    
    true_centroids = np.empty((NC,ND))
    kmeans_centroids = np.empty((NC,ND))
    pca_centroids = np.empty((NC,ND))
    fc_centroids = np.empty((NC,ND))
    sfc_centroids = np.empty((NC,ND))
    
    for c in range(NC):
        c1 = np.where(true_clusters == c)[0]
        data_clustered = values[c1,:]
        centroid = np.mean(data_clustered,axis=0)
        true_centroids[c,:] = centroid

        c1 = np.where(kmeans_clusters == c)[0]
        data_clustered = values[c1,:]
        centroid = np.mean(data_clustered,axis=0)
        kmeans_centroids[c,:] = centroid

        c1 = np.where(pca_clusters == c)[0]
        data_clustered = values[c1,:]
        centroid = np.mean(data_clustered,axis=0)
        pca_centroids[c,:] = centroid

        c1 = np.where(fc_clusters == c)[0]
        data_clustered = values[c1,:]
        centroid = np.mean(data_clustered,axis=0)
        fc_centroids[c,:] = centroid

        c1 = np.where(sfc_clusters == c)[0]
        data_clustered = values[c1,:]
        centroid = np.mean(data_clustered,axis=0)
        sfc_centroids[c,:] = centroid

    print("true_centroids",true_centroids)
    print("kmeans_centroids",kmeans_centroids)
    print("pca_centroids",pca_centroids)
    print("fc_centroids",fc_centroids)
    print("sfc_centroids",sfc_centroids)

    
    print("separation true_centroids",separation(true_centroids))
    print("separation kmeans_centroids",separation(kmeans_centroids))
    print("separation pca_centroids",separation(pca_centroids))
    print("separation fc_centroids",separation(fc_centroids))
    print("separation sfc_centroids",separation(sfc_centroids))
    
    print("silhouette_score true_centroids",silhouette_score(X,true_clusters))
    print("silhouette_score kmeans_centroids",silhouette_score(X,kmeans_clusters))
    print("silhouette_score pca_centroids",silhouette_score(X,pca_clusters))
    print("silhouette_score fc_centroids",silhouette_score(X,fc_clusters))
    print("silhouette_score sfc_centroids",silhouette_score(X,sfc_clusters))
    
    equivalences = {}
    #equivalences[1] = {0:0,1:1,2:2,3:3}
    equivalences[2] = {0:3, 1:1, 2:0, 3:2}
    equivalences[3] = {0:2, 1:0, 2:3, 3:1}
    equivalences[4] = equivalences[3]    
    
    #confusion matrix
    def build_cm(NC,true_clusters,clusters,equivalence={}):
        cm = np.zeros((NC,NC))
        for i in range(NC):
            true_c = np.where(true_clusters == i)[0]
            for j in range(NC):
                if j in equivalence:
                    c = equivalence[j]
                else:
                    c = j
                
                cluster = np.where(clusters[true_c] == c)[0]
                cm[i,j] = len(cluster)
    
        return cm
    
    classes = ['C1','C2','C3','C4']
    #KMeans
    i = 1
    if i in equivalences:
        adjust_cluster = adjust_clusters(kmeans_clusters,equivalences[i])
    else:
        adjust_cluster =  kmeans_clusters

    cm_kmeans = build_cm(NC,true_clusters,adjust_cluster)
    
    fig, ax = plt.subplots() 
    plot_confusion_matrix(cm_kmeans, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,xlabel="Predicted",ylabel="True")
                          
    plt.savefig("../figures/case_ds4/fig4a.pdf",bbox_inches='tight')
    
    #PCA
    i = 2
    if i in equivalences:
        adjust_cluster = adjust_clusters(pca_clusters,equivalences[i])
    else:
        adjust_cluster =  pca_clusters
    cm_pca = build_cm(NC,true_clusters,adjust_cluster)
    
    fig, ax = plt.subplots() 
    plot_confusion_matrix(cm_pca, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,xlabel="Predicted",ylabel="True")
    plt.savefig("../figures/case_ds4/fig4b.pdf",bbox_inches='tight')
    
    #FC
    i = 3
    if i in equivalences:
        adjust_cluster = adjust_clusters(fc_clusters,equivalences[i])
    else:
        adjust_cluster =  fc_clusters
    cm_fc = build_cm(NC,true_clusters,adjust_cluster)
    fig, ax = plt.subplots() 
    plot_confusion_matrix(cm_fc, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,xlabel="Predicted",ylabel="True")
    plt.savefig("../figures/case_ds4/fig4c.pdf",bbox_inches='tight')

    #SFC
    i = 4
    if i in equivalences:
        adjust_cluster = adjust_clusters(sfc_clusters,equivalences[i])
    else:
        adjust_cluster =  sfc_clusters
    cm_sfc = build_cm(NC,true_clusters,adjust_cluster)
    fig, ax = plt.subplots() 
    plot_confusion_matrix(cm_sfc, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,xlabel="Predicted",ylabel="True")
    plt.savefig("../figures/case_ds4/fig4d.pdf",bbox_inches='tight')
