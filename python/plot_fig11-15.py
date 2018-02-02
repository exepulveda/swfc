import numpy as np
import math

from collections import Counter

import matplotlib as mpl
mpl.use('agg')


from case_study_bm import attributes,setup_case_study_ore,setup_distances

import matplotlib.pyplot as plt

from cluster_utils import relabel



if __name__ == "__main__":
    NC = 3


    kmeans = np.int32(np.loadtxt('../results/final_bm_clusters_kmeans_%d.csv'%NC,delimiter=","))
    pca = np.int32(np.loadtxt('../results/final_bm_clusters_pca_%d.csv'%NC,delimiter=","))
    #fcew = np.loadtxt('../results/final_2d_clusters_fcew_4.csv',delimiter=",")
    #fc = np.loadtxt('../results/final_2d_clusters_fc_4.csv',delimiter=",")
    #sfcew = np.loadtxt('../results/final_2d_clusters_sfcew_4.csv',delimiter=",")
    #swfc_no_target = np.loadtxt('../results/bm_clusters_swfc_3_no_target.csv',delimiter=",")[:,-1]
    wfc  = np.int32(np.loadtxt('../results/bm_clusters_wfc_%d.csv'%NC,delimiter=",")[:,-1])
    swfc = np.int32(np.loadtxt('../results/final_bm_clusters_swfc_%d.csv'%NC,delimiter=","))

    locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore()

    N,ND = data.shape
    
    variables_include = set(['RockType','Fe','Fe_Rec','Ap','Mgt'])
    
    print(kmeans.shape,pca.shape,N,ND)

    #names = ['kmeans','pca','fcew','sfcew','fc','sfc']
    names = ['kmeans','pca','WFC','SWFC']
    labels = ["All"] + ["C"+str(k+1) for k in range(NC)]

    cluster_target = swfc
    
    clay_color = ['r','b','g','c','m','y']

    #for i,cluster in enumerate([kmeans,pca,fcew,sfcew,fc,sfc]):
    for i,cluster in enumerate([kmeans,pca,wfc,swfc]):

        adjust_cluster = relabel(cluster_target,cluster,NC)

        for var in range(ND):
            if attributes[var] in variables_include:
                fig, ax = plt.subplots(figsize=(6, 6)) 
                #ax.set_ylim([mins[v],maxs[v]])
                if var_types[var] == 3:
                    counter = Counter(data[:,var])
                    mostc =  counter.most_common()
                    cats = len(mostc)
                    width = 0.4
                    ind = np.arange(NC+1)
                    
                    counts = np.zeros((cats,NC+1))
                        
                    for k,vc in mostc:
                        k = int(math.floor(k))
                        counts[k,0] = vc
                        
                    for c in range(NC):
                        indices = np.where(np.int8(adjust_cluster)==c)[0]
                        counter = Counter(data[indices,var])
                        for k,vc in counter.most_common():
                            k = int(math.floor(k))
                            counts[k,c+1] = vc

                    sumc = np.zeros(NC+1)
                    for k in range(cats):
                        ax.bar(ind, counts[k,:], width, bottom=sumc,color=clay_color[k])
                        sumc += counts[k,:]
                    
                    ax.set_xticks(ind+width/2)
                    ax.set_xticklabels(labels)
                    ax.set_xlim(-width,ind[-1]+2*width)
                else:
                    d = [data[:,var]]

                    for c in range(NC):
                        d += [data[adjust_cluster==c,var]]

                    bx = ax.boxplot(d,labels=labels,showmeans=True)

                plt.savefig("../figures/case_bm/fig11-{var}-bm-{cm}.pdf".format(var=attributes[var],cm=names[i]),bbox_inches='tight')
                plt.close('all')
