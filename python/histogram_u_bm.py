import numpy as np
from skimage import data, io
from case_study_bm import setup_case_study_ore

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

locations,data,min_values,max_values,scale,var_types,targets = setup_case_study_ore()

NC = 6
target = False
force = False

file_template = '../results/bm_{set}_swfc_%d.csv'%NC

best_centroids = np.loadtxt(file_template.format(set='centroids'),delimiter=",")
best_weights = np.loadtxt(file_template.format(set='weights'),delimiter=",")
best_u = np.loadtxt(file_template.format(set='u'),delimiter=",")

fig,ax = plt.subplots()

max_u = np.max(best_u,axis=1)

ax.hist(max_u,31)

plt.savefig("../figures/u_hist_bm_%d.png"%NC)
plt.clf()
