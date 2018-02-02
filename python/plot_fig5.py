import sys
import random
import logging
import collections
import math
import sys
import json
import matplotlib as mpl

mpl.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

if __name__ == "__main__":
    filename = "../data/wfc_weights_ds4.csv"
    output = "../figures/case_ds4/fig5-ds4_weights.pdf"

    data = np.loadtxt(filename,delimiter=",",skiprows=1)
    
    N,ND = data.shape
    
    print(data.shape)
    
    lambda_values  = data[:,0]
    weights = data[:,1:]
    threshold = 1.0/ND/4.0
    n_weights = [np.count_nonzero(np.int8(row > threshold)) for row in weights]
    
    print(weights.shape)
    print(n_weights,1.0/ND/4.0)
    
    n,ND = weights.shape
    
    fig, ax1 = plt.subplots()    
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
    for i in range(ND):
        ax1.plot(lambda_values,weights[:,i])
    
    ax1.legend(['Cu','Fe','As','Rec'])
    ax2 = ax1.twinx()
    ax2.plot(lambda_values,n_weights,'k.')
    ax2.set_ylim(0,ND+1)
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("Weight value")
    ax2.set_ylabel("Number of weights greater than %.2f"%(threshold,))
        
    plt.savefig(output,bbox_inches='tight')
