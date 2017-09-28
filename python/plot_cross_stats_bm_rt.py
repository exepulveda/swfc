'''
This script plots statistcis grouped by rock type
'''
import sys
import random
import logging
import collections
import math
import sys
from scipy.stats import gaussian_kde

import matplotlib as mpl
#mpl.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

import pandas as pd

from case_study_bm import setup_case_study_ore, attributes

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
    locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore()

    #'RockType','Mgt','Hem','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']
    #    0        1     2     3    4     5   6    7   8   9    10   11   12   13  14   15  16   17  18   19   20   21   22

    N,ND = data.shape


    fig,axs = plt.subplots(5,5,figsize=(20,20))
    fig.subplots_adjust(wspace=0.20)

    ax = axs[0,2]
    ax.set_axis_off()
    ax = axs[4,4]
    ax.set_axis_off()
            
    #first plot is the pie chart plot of Rock type
    ax = axs[0,0]
    x = data[:,0]
    c = collections.Counter(np.int32(x))
    
    mc = c.most_common()
    mc.sort()
    codes,count = zip(*mc)
    
    labels = [str(int(k) + 1) for k in codes]
    ax.pie(count,autopct='%1.1f%%',labels=labels)
    ax.set_title('Lithology')

    #second plot is the correlation matrix of 22 vars
    corr = np.zeros((ND-1,ND-1))
    for i in range(1,ND):
      for j in range(i+1,ND):
          corr[i-1,j-1] = np.corrcoef(data[:,i],data[:,j])[0,1]
          corr[j-1,i-1] = corr[i-1,j-1]
    
    ax = axs[0,1]
    cax = ax.imshow(corr, interpolation="nearest", cmap='jet')
    #ax.grid(True)
    #ax.set_xticks(labels)
    #ax.set_yticklabels(labels,fontsize=6)

    tickslocs = np.arange(ND-1)    
    ax.set_xticks(tickslocs)
    ax.set_xticklabels(attributes[1:],rotation=90,fontsize=8,ha='left')    
    ax.set_yticks(tickslocs)
    ax.set_yticklabels(attributes[1:],fontsize=8)    
    ax.set_title('Correlation matrix')
    fig.colorbar(cax, ax=ax)    

    
    #plot boxplot of lithology vs. others
    offset = 2
    for k in range(1,ND):
        y = data[:,k]
        
        #i,j for figure
        i = (k + offset) // 5
        j = (k + offset) - i*5
        print(k,i,j)
        ax = axs[i,j]
        ax.set_title('Lithology vs. ' + attributes[k])

        #boxplot
        d = []

        for k,c in enumerate(codes):
            indices = np.where(x == c)[0]
            yindices = y[indices]
            
            d += [yindices]
        
        ax.boxplot(d,showmeans=True,showfliers=False)


    plt.savefig("../figures/case_bm/bm_cross_stats_rt.jpg", format="jpg")
    
