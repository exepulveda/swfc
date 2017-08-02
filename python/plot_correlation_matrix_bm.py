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

import pandas as pd


bm_variables= ['RockType','Mgt','Hem','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']


def correlation_matrix(corr,labels):
    fig,ax = plt.subplots() 
    cax = ax.imshow(corr, interpolation="nearest", cmap='jet')
    #ax.grid(True)
    #ax.set_xticks(labels)
    #ax.set_yticklabels(labels,fontsize=6)

    tickslocs = np.arange(len(labels))    
    ax.set_xticks(tickslocs)
    ax.set_xticklabels(labels,rotation=90,fontsize=8,ha='left')    
    ax.set_yticks(tickslocs)
    ax.set_yticklabels(labels,fontsize=8)    
    fig.colorbar(cax, ax=ax)    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    return fig,ax


if __name__ == "__main__":
    data = np.loadtxt("../../data/bm_clusters.csv",delimiter=",")
    X = np.loadtxt("../../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv",delimiter=",",skiprows=1)
    locations = X[:,1:4]
    rocktype = [4]
    mineralogy = [x for x in range(5,11)]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [69]
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    print(len(all_columns),len(bm_variables))
    assert len(all_columns) == len(bm_variables)

    values = X[:,all_columns]
    

    df = pd.DataFrame(data=values[:,1:],columns=bm_variables[1:])
    fig,ax=correlation_matrix(df.corr(),bm_variables[1:])
    
    plt.show()
