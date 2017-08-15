import sys
import random
import logging
import collections
import math
import sys
import json
import matplotlib as mpl
#mpl.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

if __name__ == "__main__":
    data = np.loadtxt("../data/sfc_weights_bm.csv",delimiter="\t",skiprows=0)
    
    N,ND = data.shape
    
    print(data.shape)
    
    lambda_values  = data[:,0]
    weights = data[:,1:]
    n_weights = [np.count_nonzero(np.int8(row > 1.0/ND/4.0)) for row in weights]
    
    print(weights.shape)
    print(n_weights)
    
    n,ND = weights.shape
    
    fig, ax1 = plt.subplots()    
    
    for i in range(ND):
        ax1.plot(lambda_values,weights[:,i])
    
    #ax1.legend(['Cu','Fe','As','Rec'])
    ax1.legend(['Var'+str(k+1) for k in range(ND)])

    ax2 = ax1.twinx()
    ax2.plot(lambda_values,n_weights,'k.')
    ax2.set_ylim(0,ND+1)
        
    plt.show()
