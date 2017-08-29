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

    variables = (0,1,2,3,5,6,20,22)
    NV = len(variables)
    
    N,ND = data.shape


    fig,axs = plt.subplots(NV,NV,figsize=(30,30))
    fig.subplots_adjust(wspace=0.1)

    for i in range(NV):
        for j in range(NV):
            ax = axs[i,j]
            #ax.set_axis_off()

    #plot histograms on diagonal
    for i,v in enumerate(variables):
        ax = axs[i,i]
        x = data[:,v]
        ax.set_title(attributes[v])
        if var_types[v] != 3:
            n, bins, patches = ax.hist(x,color='blue', alpha=0.5,normed=True)
            
            #print(i,v,bins)
            min_x = np.min(x)
            max_x = np.max(x)
            
            x_grid = np.linspace(bins[0], bins[-1], 1000)
            #KDE
            bandwidth=(max_x - min_x) * 0.05
            kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
            pdf = kde.evaluate(x_grid)
            ax.plot(x_grid, pdf, color='red', alpha=0.5, lw=1)
        else:
            #pie chart
            c = collections.Counter(np.int32(x))
            codes,count = zip(*c.most_common())
            ax.pie(count,autopct='%1.1f%%')
            
    #ploting scatter plot in the right upper 
    sample_size = 5000
    for i in range(NV):
        v = variables[i]
        x = data[:,variables[i]]

        if var_types[v] == 3:
            x = np.int32(x)
            c = collections.Counter(x)
            mc = c.most_common()
            mc.sort()
            codes,count = zip(*mc)

        for j in range(i+1,NV):
            w = variables[j]
            y = data[:,variables[j]]
            ax = axs[i,j]
            ax2 = axs[j,i]
            ax2.set_axis_off()
            if var_types[v] != 3 and var_types[w] != 3:
                indices = np.random.choice(N,size=sample_size,replace=False)
                ax.scatter(x[indices],y[indices],s=1)
                # add stats table
                #clust_data = []
                #label = ['Min','Mean','Median','Max','Std']
                #clust_data = [[np.min(x)],[np.mean(x)],[np.median(x)],np.max(x),[np.std(x)]]
                #the_table = ax.table(cellText=clust_data,rowLabels=label,loc='center')      
                #ax.text(2, 10, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
                #ax2.text(2, 10, r'$min$={min}'.format(min=np.min(x)))
                #ax2.text(0.1,0.6,'$min={:0.3f}$'.format(np.min(x)),fontsize=12)
                #ax2.text(0.1,0.5,'$mean={:0.3f}$'.format(np.mean(x)),fontsize=12)
                #ax2.text(0.1,0.4,'r$median={:0.3f}$'.format(np.median(x)),fontsize=12)
                #ax2.text(0.1,0.3,'$max={:0.3f}$'.format(np.max(x)),fontsize=12)
                #ax2.text(0.1,0.2,'$\sigma={:0.3f}$'.format(np.std(x)),fontsize=12)
                #table version
                row_labels= ['min','mean','median','max','std']

                celldata= [
                        ['{:0.3f}'.format(np.min(x))],
                        ['{:0.3f}'.format(np.mean(x))],
                        ['{:0.3f}'.format(np.median(x))],
                        ['{:0.3f}'.format(np.max(x))],
                        ['{:0.3f}'.format(np.std(x))]
                    ]
                ax2.table(cellText=celldata,rowLabels=row_labels,loc='center left',fontsize=24,colWidths = [0.4])
                #row_labels=['min','mean','median','max','$\sigma$']
                #table_vals=['${:0.3f}$'.format(np.min(x)),'${:0.3f}$'.format(np.min(x)),'${:0.3f}$'.format(np.min(x)),'${:0.3f}$'.format(np.min(x))]
                #table = r'''\begin{tabular}{ c | c | c | c } & col1 & col2 & col3 \\\hline row1 & 11 & 12 & 13 \\\hline row2 & 21 & 22 & 23 \\\hline  row3 & 31 & 32 & 33 \end{tabular}'''
                #plt.text(0.1,0.8,table,size=12)                
                
            elif var_types[v] == 3 and var_types[w] != 3:
                #boxplot
                d = []

                row_labels= ['min','mean','median','max','std']
                col_labels= [str(x) for x in codes]

                celldata = [ [ None for x in range(len(codes))] for y in range(5)]
                for k,c in enumerate(codes):
                    indices = np.where(x == c)[0]
                    
                    yindices = y[indices]
                    
                    d += [yindices]
                
                    celldata[0][k] = '{:0.3f}'.format(np.min(yindices))
                    celldata[1][k] = '{:0.3f}'.format(np.mean(yindices))
                    celldata[2][k] = '{:0.3f}'.format(np.median(yindices))
                    celldata[3][k] = '{:0.3f}'.format(np.max(yindices))
                    celldata[4][k] = '{:0.3f}'.format(np.std(yindices))

                ax.boxplot(d) #,showmeans=True)
                table = ax2.table(cellText=celldata,loc='center left',rowLabels=row_labels,colLabels=col_labels,fontsize=24,colWidths = [0.2]*len(codes))     
                
                #cell = table._cells[(1, 0)]
                #cell.set_text_props(ha='left')
                
                #print(i,j,celldata) 

    #ploting main stats as text lower
    sample_size = 5000
    for i in range(NV):
        v = variables[i]
        x = data[:,variables[i]]

        if var_types[v] == 3:
            x = np.int32(x)
            c = collections.Counter(x)
            codes,count = zip(*c.most_common())

        for j in range(i+1,NV):
            w = variables[j]
            y = data[:,variables[j]]
            ax = axs[i,j]
            if var_types[v] != 3 and var_types[w] != 3:
                indices = np.random.choice(N,size=sample_size,replace=False)
                ax.scatter(x[indices],y[indices],s=1)
            elif var_types[v] == 3 and var_types[w] != 3:
                #boxplot
                d = []
            
    #
    #plt.savefig("../figures/bm_cross_stats.svg", format="svg")
    plt.savefig("../figures/bm_cross_stats.jpg", format="jpg")
    
