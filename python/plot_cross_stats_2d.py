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

from case_study_2d import attributes,setup_case_study,setup_distances

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
    locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
    data = data_ore.copy()

    N,ND = data.shape


    fig,axs = plt.subplots(ND,ND,figsize=(40,40))
    fig.subplots_adjust(wspace=0.5)

    for i in range(ND):
        for j in range(ND):
            ax = axs[i,j]
            #ax.set_axis_off()

    #plot histograms on diagonal
    for i,v in enumerate(range(ND)):
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
            #bar chart
            bar_width = 0.35
            pos = np.arange(1,4) - bar_width/2.0
            c = collections.Counter(np.int32(x))
            codes,count = zip(*c.most_common())
            ax.bar(pos, count, bar_width)
            #ax.pie(count,autopct='%1.1f%%')
            
    #ploting scatter plot in the right upper 
    for i in range(ND):
        v = i
        x = data[:,i]

        if var_types[v] == 3:
            x = np.int32(x)
            c = collections.Counter(x)
            mc = c.most_common()
            mc.sort()
            codes,count = zip(*mc)

        for j in range(i+1,ND):
            w = j
            y = data[:,j]
            ax = axs[i,j]
            ax.set_title(attributes[v] + " vs. " + attributes[w])
            ax2 = axs[j,i]
            ax2.set_axis_off()
            ax2.set_title(attributes[v] + " vs. " + attributes[w])
            if var_types[v] != 3 and var_types[w] != 3:
                ax.scatter(x,y,s=3)
                ax.set_xlabel(attributes[v])
                ax.set_ylabel(attributes[w])
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
    for i in range(ND):
        v = i
        x = data[:,i]

        if var_types[v] == 3:
            x = np.int32(x)
            c = collections.Counter(x)
            codes,count = zip(*c.most_common())

        for j in range(i+1,ND):
            w = j
            y = data[:,j]
            ax = axs[i,j]
            if var_types[v] != 3 and var_types[w] != 3:
                ax.scatter(x,y,s=1)
            elif var_types[v] == 3 and var_types[w] != 3:
                #boxplot
                d = []
            
    #
    plt.savefig("../figures/2d_cross_stats.svg", format="svg")
    plt.savefig("../figures/2d_cross_stats.jpg", format="jpg")
    
