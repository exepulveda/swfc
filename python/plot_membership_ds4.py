'''
This script plots the membership output of SWFC method.
Blue color --> low probability 
Red color  --> high probability 
'''
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
    X = np.loadtxt("../data/{dataset}.csv".format(dataset=filename),skiprows=1,delimiter=",")
    
    NC = 4

    template = "../results/%s_{tag}_swfc_%d.csv"%(filename,NC)    

    u = np.loadtxt(template.format(tag="u"),delimiter=",",skiprows=0)
    locations = X[:,0:2]
    
    fig, ax = plt.subplots() 

    plt.scatter(locations[:,0],locations[:,1],marker='o',s=20,c=np.max(u,axis=1),cmap='jet')
    plt.colorbar()
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    #plt.show()
    plt.savefig("../figures/ds4_membership")
