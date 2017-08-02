import numpy as np
import pickle
import logging
import argparse
import csv

import matplotlib.pyplot as plt
from plotting import plot_confusion_matrix

parser = argparse.ArgumentParser(description="spatial clustering")
parser.add_argument('--verbose', help='output debug info',default=False,action='store_true',required=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    filename = "ds4"
    data = np.loadtxt("../../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    
    locations = data[:,0:2]
    clusters = np.int8(data[:,-1])

    nclusters = 4
    
    #plot
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    color = ['r','b','g','c']
    for c in range(4):
        c1 = np.where(clusters == c)[0]
        plt.plot(locations[c1,0],locations[c1,1],marker='o',color=color[c],linestyle='None')
    #plt.legend(['Cluster 1','Cluster 2','Cluster 3','Cluster 4'],loc='upper right',bbox_to_anchor=(0, 1))
    plt.show()
