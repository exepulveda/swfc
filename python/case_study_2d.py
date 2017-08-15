import sys

sys.path += ['..']

import clusteringlib as cl
import numpy as np
import scipy.stats

from cluster_utils import fix_weights,recode_categorical_values

#attributes = ['CuT','As','Rec','wi','fminz','arci']
attributes = ['Clay','CuT','As','Rec','Wi']

#arci 0: 4265, 1: 4220, 2: 4211, 3: 3495
#scale= 0.7484396218950435 as mean distance



def setup_case_study(inputfile=None,targets=None):
    if inputfile is None:
        inputfile = "../data/2D.csv"
         
    X = np.loadtxt(inputfile,delimiter=",",skiprows=1)

    locations = X[:,(0,2)] #y is fixed
    
    orewaste = [7]
    clay = [8]
    elements = [3,4,5,6]
    
    all_columns = clay + elements
    
    values = X[:,all_columns] #0,1,2,5 are real; 3 and 4 are cats

    #split data ore/waste
    waste_indices = np.where(np.int8(X[:,7]) == 0)[0]
    ore_indices = np.where(np.int8(X[:,7]) == 1)[0]
    
    #print(len(waste_indices),len(ore_indices),len(X))
    

    locations_ore = locations[ore_indices,:]
    values_ore = values[ore_indices,:]

    values_ore[:,0] = recode_categorical_values(np.int32(values_ore[:,0]),[1,2,3],a=0)


    data_ore = np.asfortranarray(np.float32(values_ore))
    scale_ore = np.std(data_ore,axis=0)
    
    N,ND = data_ore.shape

    N2 = N**2
    ncat2 = 0.0
    for i in range(3):
        indices = np.where(np.int8(values_ore[:,0]) == i)[0]
        ncat2 += len(indices)**2
        
        #print('cat clay',i,len(indices),N)
        
    scale_ore[0] = 1.0 - ncat2/N2
    print('scale clay',scale_ore[0])

    var_types = np.ones(ND)

    min_values = np.min(data_ore,axis=0)
    max_values = np.max(data_ore,axis=0)
    
    #clay is a categorical attribute
    var_types[0] = 3 
    min_values[0] = 0    


    return locations,ore_indices,locations_ore,data_ore,min_values,max_values,scale_ore,var_types,{0:3}


def setup_distances(scale,var_types,categories=None,targets=None):
    cl.distances.sk_setup(np.asfortranarray(np.float32(scale)))
    cl.distances.set_variables(np.asfortranarray(np.int32(var_types)),False)
    
    for i in range(len(var_types)):
        if var_types[i] == 3 and categories is not None and i in categories:
            ncat = categories[i]
            distances_cat = np.asfortranarray(np.ones((ncat,ncat)),dtype=np.float32)
            for j in range(ncat):
                distances_cat[j,j] = 0.0
            
            cl.distances.set_categorical(i+1,ncat,distances_cat) #array fortran starts in 1

    if targets is not None:
        cl.distances.set_targeted(23,targets,False)
