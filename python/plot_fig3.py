import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib as mpl
mpl.use('agg')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from plotting import scatter_clusters
import matplotlib.pyplot as plt

import clusteringlib as cl

from case_study_2d import attributes,setup_case_study,setup_distances

if __name__ == "__main__":
    filename = 'ds4'

    X = np.loadtxt("../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    locations = X[:,0:2]
    data = X[:,2:6] #0,1,2,5 are continues
    true_clusters = np.int8(X[:,6])

    N,ND = data.shape
    
    #now all are continues variables
    var_types = np.ones(ND)

    seed = 1634120   
    np.random.seed(seed)

    standadize = StandardScaler()
    data_scaled = standadize.fit_transform(data)
    #data_scaled = data.copy() #standadize.fit_transform(data)
    scale = standadize.scale_
    ND_PCA = 2

    pca = PCA(n_components=ND_PCA,whiten=True)
    pca_X = pca.fit_transform(data_scaled)
    print(pca_X.shape)

    fig, ax = plt.subplots() 
    nclusters = 4
    color = ['r','b','g','c']
    #ax.set_aspect('equal')    
    
    for c in range(nclusters):
        c1 = np.where(true_clusters == c)[0]

        plt.plot(pca_X[c1,0],pca_X[c1,1],marker='o',color=color[c],linestyle='None')


    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.savefig("../figures/case_ds4/fig3.pdf",bbox_inches='tight')
