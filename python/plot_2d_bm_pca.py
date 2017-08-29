import numpy as np
from skimage import data, io
from case_study_bm import setup_case_study_ore

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

locations,data,min_values,max_values,scale,var_types,targets = setup_case_study_ore()

N,K = data.shape

#block model size
sx,ex = 8,42
sy,ey = 8,41
sz,ez = 1,48

#number of blocks
nx = ex-sx+1
ny = ey-sy+1
nz = ez-sz+1

NC = 4

file_template = np.int32(np.loadtxt('../results/final_bm_clusters_pca_%d.csv'%NC,delimiter=","))

clusters = np.loadtxt(file_template,delimiter=",",dtype=np.int32)

#cluster_color = { 0: 50, 1: 100, 2: 150, 3: 200 }
#cluster_color = { 0: 50, 1: 100, 3: 150, 2: 200 }
#cluster_color = { 0: 50, 2: 100, 3: 150, 1: 200 }
#cluster_color = { 0: 50, 2: 100, 1: 150, 3: 200 }
#cluster_color = { 0: 50, 3: 100, 1: 150, 2: 200 }
#cluster_color = { 0: 50, 3: 100, 2: 150, 1: 200 }

cluster_color = { 3: 50, 2: 100, 0: 150, 1: 200 }


N,ND = data.shape


#make images of all XY for all Z
for k in range(nz):
    fig,ax = plt.subplots()
    img = np.zeros((nx,ny),dtype=int)
    z = locations[:,2] - sz
    indices = np.where(z == k)[0]
    for ind in indices:
        x,y = np.int32(locations[ind,(0,1)])
        #print(k,ind,x,y,x-sx,y-sy)
        img[x-sx,y-sy] = cluster_color[clusters[ind]]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../figures/projections/bm-xy-%d-pca"%(k+1))
    plt.clf()
    plt.close('all')
    if k == 24: quit()
    

#make images of all XZ for all Y
for j in range(ny):
    fig,ax = plt.subplots()
    img = np.zeros((nx,nz),dtype=int)
    y = locations[:,1] - sy
    indices = np.where(y == j)[0]
    for ind in indices:
        x,z = np.int32(locations[ind,(0,2)])
        img[x-sx,z-sz] = cluster_color[clusters[ind]]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../figures/projections/bm-xz-%d-pca"%(j+1))
    plt.clf()
    plt.close('all')

#make images of all YZ for all X
for i in range(ny):
    fig,ax = plt.subplots()
    img = np.zeros((ny,nz),dtype=int)
    x = locations[:,0] - sx
    indices = np.where(x == i)[0]
    for ind in indices:
        y,z = np.int32(locations[ind,(1,2)])
        img[y-sy,z-sz] = cluster_color[clusters[ind]]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../figures/projections/bm-yz-%d-pca"%(i+1))
    plt.clf()
    plt.close('all')
    
#plt.show()
