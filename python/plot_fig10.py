'''
This script plots the histogram and cumulative histogram of membership output of SWFC method.
'''
import numpy as np
from skimage import data, io
from case_study_bm import setup_case_study_ore

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

locations,data,min_values,max_values,scale,var_types,targets = setup_case_study_ore()

NC = 3
target = False
force = False

file_template = '../results/bm_{set}_wfc_%d.csv'%NC

best_centroids = np.loadtxt(file_template.format(set='centroids'),delimiter=",")
best_weights = np.loadtxt(file_template.format(set='weights'),delimiter=",")
best_u = np.loadtxt(file_template.format(set='u'),delimiter=",")

fig,ax1 = plt.subplots()

max_u = np.max(best_u,axis=1)

n_bins = 31

ax1.hist(max_u,n_bins)
ax2 = ax1.twinx()

n_bins = 100

ax2.hist(max_u,n_bins,normed=True, histtype='step', cumulative=True,color='red')

ax1.set_xlabel('Membership')
ax1.set_ylabel('Frequeny')
ax2.set_ylabel('Cumulated frequeny')
plt.savefig("../figures/case_bm/fig10.pdf")
