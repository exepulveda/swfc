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
    wfc  = np.int32(np.loadtxt('../results/final_bm_clusters_wfc_%d.csv'%NC,delimiter=",")[:,-1])
    swfc = np.int32(np.loadtxt('../results/final_bm_clusters_swfc_%d.csv'%NC,delimiter=","))

    locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore()

    N,ND = data.shape
    
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

            plt.savefig("../figures/case_bm/boxplot-{var}-bm-{cm}".format(var=attributes[var],cm=names[i]),bbox_inches='tight')
            plt.close('all')

#from case_study_bm import attributes,setup_case_study_ore,setup_case_study_all,setup_distances

#if __name__ == "__main__":
    #locations,data,min_values,max_values,scale,var_types,categories = setup_case_study_ore(a=0.999)
    #N,ND = data.shape

    #locations = X[:,0:3]

    #kmeans_clusters = X[:,6]
    #pca_clusters = X[:,7]
    #fc1_clusters = X[:,8]
    #fc2_clusters = X[:,9]
    #fc3_clusters = X[:,10]
    
    #clusters = np.c_[kmeans_clusters,pca_clusters,fc1_clusters,fc2_clusters,fc3_clusters]
    
    #X = np.loadtxt("../../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv",delimiter=",",skiprows=1)
    #locations = X[:,1:4]
    #rocktype = [4]
    #mineralogy = [x for x in range(5,11)]
    #elements = [x for x in range(11,25)]
    #sg = [39]
    #rec = [69]
    
    #all_columns = rocktype + mineralogy + elements + sg + rec
    
    #values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats
    #NC = 6
    #N,ND = values.shape
    
    #labels = ["All","C1","C2","C3","C4","C5","C6"]
    #mins = [0,0,0,60]
    #maxs = [2.0,5.0,50.0,100.0]
    ##plot
    
    #clustering_method = ['K-Means','PCA + K-Means','FC1','FC2','FC3']
    
    #equivalences = {}
    ##equivalences[1] = {0:0,1:1,2:2,3:3}
    ##equivalences[2] = {0:3, 1:1, 2:0, 3:2}
    ##equivalences[3] = {0:2, 1:0, 2:3, 3:1}
    ##equivalences[4] = equivalences[3]

    #for i,clusters in enumerate([kmeans_clusters,pca_clusters,fc1_clusters,fc2_clusters,fc3_clusters]): #clustering method
        
        #if i in equivalences:
            #adjust_cluster = adjust_clusters(clusters,equivalences[i])
        #else:
            #adjust_cluster =  clusters
                    
        #for v in range(ND):
            #fig, ax = plt.subplots(figsize=(6, 6)) 
            ##ax.set_ylim([mins[v],maxs[v]])
            #d = [values[:,v]]

            #for c in range(NC):
                #d += [values[adjust_cluster==c,v]]

            #bx = ax.boxplot(d,labels=labels,showmeans=True)

            ##plt.legend(['Cluster 1','Cluster 2','Cluster 3','Cluster 4'],loc='upper right',bbox_to_anchor=(0, 1))
        ##plt.title(clustering_method[i])
            #plt.savefig("../../figures/boxplot-{var}-{cm}-bm".format(var=bm_variables[v],cm=clustering_method[i]))
