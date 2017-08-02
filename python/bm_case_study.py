import sys

sys.path += ['..']

import clusteringlib as cl
import numpy as np
import scipy.stats

from cluster_utils import fix_weights,recode_categorical_values

bm_variables = ['RockType','Mgt','Hem','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']


def setup_case_study(inputfile=None,target=False):
    if inputfile is None:
        inputfile = "../data/SOB_Big_data_base_Assays_SG_OPEX_HSC_5iter.csv"
         
    X = np.loadtxt(inputfile,delimiter=",",skiprows=1)
    locations = X[:,1:4]
    
    rocktype = [4]
    mineralogy = [x for x in range(5,11)]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [69]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    values[:,0] = recode_categorical_values(values[:,0],[300,400,500])

    data = np.asfortranarray(np.float32(values))
    scale = np.std(data,axis=0)

    N,ND = data.shape

    var_types = np.ones(ND)
    var_types[0] = 3

    if target:
        targets = np.asfortranarray(np.percentile(values[:,-1], [15,50,85]),dtype=np.float32)
        var_types[-1] = 2
        print('targets',targets)
    else:
        targets = None
        
    return locations,data,scale,var_types,targets


def setup_distances(scale,var_types,distances_cat=None,targets=None):
    cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
    cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
    if distances_cat is not None:
        distances_cat = np.asfortranarray(np.ones((3,3)),dtype=np.float32)
        distances_cat[0,0] = 0.0
        distances_cat[1,1] = 0.0
        distances_cat[2,2] = 0.0
        cl.distances.set_categorical(1, 3,distances_cat)

    if targets is not None:
        cl.distances.set_targeted(23,targets,False)
