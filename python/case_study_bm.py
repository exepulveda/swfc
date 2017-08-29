import sys

sys.path += ['..']

import clusteringlib as cl
import numpy as np
import scipy.stats

from cluster_utils import fix_weights,recode_categorical_values

#attributes = ['RockType','Mgt','Hem','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']

attributes = ['RockType','Mgt','Ab','Act','Ap','Bt','O','F','Na','Mg','Al','Si','P','Cl','K','Ca','Ti','V','Mn','Fe','SG','Fe_Rec']

def setup_case_study_ore(inputfile=None,targets=None,a=0):
    if inputfile is None:
        inputfile = "../data/synthetic_bm_small.csv"
         
    X = np.loadtxt(inputfile,delimiter=",",skiprows=1)
    locations = X[:,1:4]
    
    rocktype = [4]
    mineralogy = [5,7,8,9,10]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [40]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    values[:,0] = recode_categorical_values(np.int32(values[:,0]),[300,400,500],a=a)

    data = np.asfortranarray(np.float32(values))
    scale = np.std(data,axis=0)

    N,ND = data.shape

    var_types = np.ones(ND,dtype=np.int32)
    var_types[0] = 3 #rocktype is a categorical attribute

    if targets is not None:
        targets = np.asfortranarray(targets,dtype=np.float32)#np.asfortranarray(np.percentile(values[:,-1], [15,50,85]),dtype=np.float32) #using here default percentils
        var_types[-1] = 2 #recovery will be targeted

    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)
    
    #rocktype is a categorical attribute
    var_types[0] = 3 
    min_values[0] = 0    

    return locations,data,min_values,max_values,scale,var_types,{0:3}


def setup_case_study_all(inputfile=None,targets=None):
    if inputfile is None:
        inputfile = "../data/synthetic_bm.csv"
         
    X = np.loadtxt(inputfile,delimiter=",",skiprows=1)
    locations = X[:,1:4]
    
    rocktype = [4]
    mineralogy = [5,7,8,9,10]
    elements = [x for x in range(11,25)]
    sg = [39]
    rec = [40]
    
    all_columns = rocktype + mineralogy + elements + sg + rec
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    values[:,0] = recode_categorical_values(np.int32(values[:,0]),[100,200,300,400,500],a=0)

    data = np.asfortranarray(np.float32(values))
    scale = np.std(data,axis=0)

    N,ND = data.shape

    var_types = np.ones(ND)
    var_types[0] = 3 #rocktype is a categorical attribute

    if targets is not None:
        targets = np.asfortranarray(targets,dtype=np.float32)#np.asfortranarray(np.percentile(values[:,-1], [15,50,85]),dtype=np.float32) #using here default percentils
        var_types[-1] = 2 #recovery will be targeted

    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)
    
    #rocktype is a categorical attribute
    var_types[0] = 3 
    min_values[0] = 0    

    return locations,data,min_values,max_values,scale,var_types,{0:5}



def setup_distances(scale,var_types,use_cat=False,targets=None):
    cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
    cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
    if use_cat:
        distances_cat = np.asfortranarray(np.ones((3,3)),dtype=np.float32)
        distances_cat[0,0] = 0.0
        distances_cat[1,1] = 0.0
        distances_cat[2,2] = 0.0
        cl.distances.set_categorical(1, 3,distances_cat)

    if targets is not None:
        cl.distances.set_targeted(22,targets,False)
