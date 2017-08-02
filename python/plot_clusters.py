'''
http://stackoverflow.com/questions/9651940/determining-and-storing-voronoi-cell-adjacency
'''
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist

import pickle
import logging

from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    #read muestras
    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    
    data = np.loadtxt("../data/muestras-cluster.csv",skiprows=0,delimiter=",")
    
    locations = data[:,0:2]
    values = data[:,3:5]
    clusters = np.int32(data[:,-1])
    
    print np.min(clusters)
    print np.max(clusters)
    
    n,m = values.shape
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    diff = set(clusters)
    
    ax.scatter(locations[:,0], locations[:,1],c=clusters)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.show()        

