import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib as mpl

mpl.use('agg')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

from cluster_utils import create_clusters_dict,recode_categorical_values

from plotting import scatter_clusters, plot_confusion_matrix

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters

import clusteringlib as cl

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
    
    N,K = clusters.shape
    NC = 6

    print("adjusted_rand_score:")
    
    for i in range(K):
        c1 = clusters[:,i]
        for j in range(i+1,K):
            c2 = clusters[:,j]
            ret = adjusted_rand_score(c1,c2)
            print(i,j,ret)

    print("cluster proportions:")
            
    #clusters stats
    for i in range(K):
        c1 = clusters[:,i]
        for j in range(NC):
            indices = np.where(c1==j)[0]
            print(i,j,len(indices)/N*100.0)
            
            
    

    X = np.loadtxt("../../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv",delimiter=",",skiprows=1)
    locations = X[:,1:4]
    rocktype = [4]
    mineralogy = [x for x in range(5,11)]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [69]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    N,ND = values.shape
    
    values[:,0] = recode_categorical_values(values[:,0],[300,400,500])
    
    data = np.asfortranarray(np.float32(values))
    scale = np.std(data,axis=0)
    
    var_types = np.ones(ND)
    var_types[0] = 3

    cl.clustering.sk_setup(np.asfortranarray(np.float32(scale)))
    cl.clustering.set_variables(np.asfortranarray(np.int32(var_types)),False)
    distances_cat = np.asfortranarray(np.ones((3,3)),dtype=np.float32)
    distances_cat[0,0] = 0.0
    distances_cat[1,1] = 0.0
    distances_cat[2,2] = 0.0
    cl.clustering.set_categorical(1, 3,distances_cat)

    s = np.asfortranarray(np.zeros((NC,ND),dtype=np.float32))
    cluster = np.asfortranarray(fc1_clusters,dtype=np.int8)
    cluster += 1

    print("cluster stats to each variable:")
    for i in range(ND):
        mean_all = np.mean(values[:,i])
        std_all = np.std(values[:,i])

        ret = [bm_variables[i],mean_all,std_all]
        for k in range(NC):
            indices = np.where(fc1_clusters == k)[0]
            mean_c = np.mean(values[indices,i])
            std_c = np.std(values[indices,i])
            ret += [mean_c,std_c]
            
        
        print(*ret,sep=',')

    quit()

    cl.clustering.dispersion(data,cluster,s,0)

    
    for i in range(NC):
        print(i,s[i,:])
        e = np.max(s[i,:])
        #relevance = np.sort(s[i,:])
        relevance = s[i,:].copy()
        relevance = 1.0/(relevance + 0.001)
        relevance /= np.sum(relevance)
        relevance *= 100.0
        
    
