import numpy as np
from skimage import data, io
from bm_case_study import setup_case_study

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

locations,data,scale,var_types,targets = setup_case_study()

N,K = data.shape

#block model size
sx,ex = 8,42
sy,ey = 8,41
sz,ez = 1,48

#number of blocks
nx = ex-sx+1
ny = ey-sy+1
nz = ez-sz+1

col = 4


lambda_value = 0.25
NC = 4
target = False
force = False

file_template = '../../data/bm_{set}_%s_%s_sfc_%s.csv'%(target,force,NC)

best_centroids = np.loadtxt(file_template.format(set='centroids'),delimiter=",")
best_weights = np.loadtxt(file_template.format(set='weights'),delimiter=",")
best_u = np.loadtxt(file_template.format(set='u'),delimiter=",")

clusters = np.argmax(best_u,axis=1)    

N,ND = data.shape


#make images of all XY for all Z
for k in range(nz):
    fig,ax = plt.subplots()
    img = np.zeros((nx,ny),dtype=int)
    z = locations[:,2] - sz
    indices = np.where(z == k)[0]
    for ind in indices:
        x,y = locations[ind,(0,1)]
        img[x-sx,y-sy] = clusters[ind]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../../figures/projections/bm-xy-%d-bm"%(k+1))

#make images of all XZ for all Y
for j in range(ny):
    fig,ax = plt.subplots()
    img = np.zeros((nx,nz),dtype=int)
    y = locations[:,1] - sy
    indices = np.where(y == j)[0]
    for ind in indices:
        x,z = locations[ind,(0,2)]
        img[x-sx,z-sz] = clusters[ind]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../../figures/projections/bm-xz-%d-bm"%(j+1))

#make images of all YZ for all X
for i in range(ny):
    fig,ax = plt.subplots()
    img = np.zeros((ny,nz),dtype=int)
    x = locations[:,0] - sx
    indices = np.where(x == i)[0]
    for ind in indices:
        y,z = locations[ind,(1,2)]
        img[y-sy,z-sz] = clusters[ind]
    
    ax.imshow(img, interpolation='nearest')
    plt.savefig("../../figures/projections/bm-yz-%d-bm"%(i+1))
    
#plt.show()
