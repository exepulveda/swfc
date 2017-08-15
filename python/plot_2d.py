import numpy as np
from skimage import data, io, transform
from skimage import img_as_float
import matplotlib.patches as patches

from case_study_2d import attributes,setup_case_study,setup_distances

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
from cluster_utils import adjust_clusters

locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale,var_types,categories = setup_case_study()
data = data_ore.copy()

print('ore size',len(ore_indices))

'''
min  : 14700	106980	2200
max  : 18660	106980	3220
range: 3960	0	1020
bsize: 40	40	20
blocks: 100	1	52
'''


N,K = locations.shape

#block model size
nx = 100
nz = 52
sx = 40.0
sz = 20.0

col = 4

#readinf clusters
kmeans = np.loadtxt('../results/final_2d_clusters_kmeans_4.csv',delimiter=",")
pca = np.loadtxt('../results/final_2d_clusters_pca_4.csv',delimiter=",")
fcew = np.loadtxt('../results/final_2d_clusters_fcew_4.csv',delimiter=",")
fc = np.loadtxt('../results/final_2d_clusters_fc_4.csv',delimiter=",")
sfcew = np.loadtxt('../results/final_2d_clusters_sfcew_4.csv',delimiter=",")
sfc = np.loadtxt('../results/final_2d_clusters_sfc_4.csv',delimiter=",")

names = ['kmeans','pca','fcew','sfcew','fc','sfc']

equivalences = {}
equivalences[0] = {0:0, 1:1, 2:2, 3:3, 4:4}

equivalences[1] = {0:3, 1:1, 2:2, 3:0, 4:4}

equivalences[2] = {0:2, 1:0, 2:1, 3:3, 4:4}
equivalences[3] = {0:2, 1:0, 2:1, 3:3, 4:4}

equivalences[4] = {0:1, 1:0, 2:2, 3:3, 4:4}
equivalences[5] = {0:1, 1:0, 2:2, 3:3, 4:4}


facecolor = {0: 'c', 1: 'b', 2: 'r', 3: 'y', 4: 'k', 5: 'w'}

img = np.zeros((nx,nz),dtype=int)
img.fill(5)

#air
for ind,(x,z) in enumerate(locations):
    i = int((x - 14700)//40)
    k = int((z - 2200)//20)

    img[i,k] = 4

for c,cluster in enumerate([kmeans,pca,fcew,sfcew,fc,sfc]):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, aspect='equal')

    #ore
    new_clusters = adjust_clusters(cluster,equivalences[c])

    for ind,(x,z) in enumerate(locations_ore):
        i = int((x - 14700)//40)
        k = int((z - 2200)//20)

        img[i,k] = new_clusters[ind]


    new_img = img[20:80,10:]
    nx2,nz2 = new_img.shape

    ax.set_xlim(0,nx2*sx)
    ax.set_ylim(0,nz2*sz)

    for i in range(nx2):
        for k in range(nz2):
            ax.add_patch(
                patches.Rectangle(
                    (i*sx, k*sz),   # (x,y)
                    sx,          # width
                    sz,          # height
                    facecolor=facecolor[new_img[i,k]],
                    edgecolor=facecolor[new_img[i,k]]
                )
            )    
    plt.savefig("../figures/projections/2d-xz-%s.jpg"%names[c],bbox_inches='tight')
    plt.close('all')
