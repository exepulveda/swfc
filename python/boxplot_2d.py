import numpy as np

from collections import Counter

import matplotlib as mpl
mpl.use('agg')


from case_study_2d import attributes,setup_case_study,setup_distances

import matplotlib.pyplot as plt

from cluster_utils import adjust_clusters

locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
data = data_ore.copy()


if __name__ == "__main__":
    kmeans = np.loadtxt('../results/final_2d_clusters_kmeans_4.csv',delimiter=",")
    pca = np.loadtxt('../results/final_2d_clusters_pca_4.csv',delimiter=",")
    fcew = np.loadtxt('../results/final_2d_clusters_fcew_4.csv',delimiter=",")
    fc = np.loadtxt('../results/final_2d_clusters_fc_4.csv',delimiter=",")
    sfcew = np.loadtxt('../results/final_2d_clusters_sfcew_4.csv',delimiter=",")
    sfc = np.loadtxt('../results/final_2d_clusters_sfc_4.csv',delimiter=",")

    N,ND = data_ore.shape
    NC = 4

    names = ['kmeans','pca','fcew','sfcew','fc','sfc']
    labels = ["All","C1","C2","C3","C4"]
    
    equivalences = {}
    equivalences[0] = {0:0, 1:1, 2:2, 3:3, 4:4}

    equivalences[1] = {0:3, 1:1, 2:2, 3:0, 4:4}

    equivalences[2] = {0:2, 1:0, 2:1, 3:3, 4:4}
    equivalences[3] = {0:2, 1:0, 2:1, 3:3, 4:4}

    equivalences[4] = {0:1, 1:0, 2:2, 3:3, 4:4}
    equivalences[5] = {0:1, 1:0, 2:2, 3:3, 4:4}
    
    clay_color = ['r','b','g']

    for i,cluster in enumerate([kmeans,pca,fcew,sfcew,fc,sfc]):
        
        if i in equivalences:
            adjust_cluster = adjust_clusters(cluster,equivalences[i])
        else:
            adjust_cluster =  cluster
                    
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

            plt.savefig("../figures/boxplot-{var}-{cm}-2d".format(var=attributes[var],cm=names[i]),bbox_inches='tight')
            plt.close('all')
