#metrics for clustering

import numpy as np
import collections
import math

def kmeans(data,nclusters,distance_function,categorical = [4,5]):
    n,p = data.shape
    
    centroids = np.empty((nclusters,p))
    #min and max values
    minval = np.min(data,axis=0)
    maxval = np.max(data,axis=0)
    #randomise centroids
    for i in range(p):
        if i not in categorical:
            centroids[:,i] = np.random.uniform(low=minval[i],high=maxval[i],size=nclusters)
        else:
            centroids[:,i] = np.random.randint(low=int(minval[i]),high=int(maxval[i])+1,size=nclusters)
            
    #main loop
    clusters = np.empty(n,dtype=np.int32)
    for iterations in range(1000):
        #to each point find the closet centroid to
        for i in range(n):
            dmax = distance_function(data[i,:],centroids[0])
            cluster = 0
            for k in range(1,nclusters):
                d = distance_function(data[i,:],centroids[k,:])
                if d < dmax:
                    dmax = d
                    cluster = k
                
            clusters[i] = cluster
        #generate centroids
        clusters_dict = create_clusters_dict(clusters)
        c_c = cluster_centroids(clusters_dict,data,categorical = categorical)
        for k,c in enumerate(c_c):
            centroids[k,:] = c_c[c]
            
    return clusters

def kmeans_weighted(data,nclusters,distance_function,W,categorical = [4,5]):
    n,p = data.shape
    
    centroids = np.empty((nclusters,p))
    #min and max values
    minval = np.min(data,axis=0)
    maxval = np.max(data,axis=0)
    #randomise centroids
    for i in range(p):
        if i not in categorical:
            centroids[:,i] = np.random.uniform(low=minval[i],high=maxval[i],size=nclusters)
        else:
            centroids[:,i] = np.random.randint(low=int(minval[i]),high=int(maxval[i])+1,size=nclusters)
            
    #main loop
    clusters = np.empty(n,dtype=np.int32)
    for iterations in range(1000):
        #to each point find the closet centroid to
        for i in range(n):
            dmax = distance_function(data[i,:],centroids[0],W[0,:])
            cluster = 0
            for k in range(1,nclusters):
                d = distance_function(data[i,:],centroids[k,:],W[k,:])
                if d < dmax:
                    dmax = d
                    cluster = k
                
            clusters[i] = cluster
        #generate centroids
        clusters_dict = create_clusters_dict(clusters)
        c_c = cluster_centroids(clusters_dict,data,categorical = categorical)
        for k,c in enumerate(c_c):
            centroids[k,:] = c_c[c]
            
    return clusters

            
def cluster_centroids(clusters_dict,data,categorical = [4,5]):
    ret = {}
    p = data.shape[1]
    for c,clusters_indices in clusters_dict.items():
        centroid = np.empty(p)
        for j in range(p):
            if j not in categorical:
                centroid[j] = np.mean(data[clusters_indices,j])
            else:
                elem = [int(x) for x in data[clusters_indices,j]]
                counter = collections.Counter(elem)
                centroid[j] = counter.most_common(1)[0][0]
    
        ret[c] = centroid
        
    return ret

def global_centroids(data,categorical = [4,5]):
    p = data.shape[1]
    centroid = np.empty(p)
    for j in range(p):
        if j not in categorical:
            centroid[j] = np.mean(data[:,j])
        else:
            counter = collections.Counter(np.int32(data[:,j]))
            centroid[j] = counter.most_common(1)[0][0]

    return centroid

def clusters_stats(clusters,data,categorical = [4,5]):
    clusters_dict = create_clusters_dict(clusters)
    
    #order keys
    keys = list(clusters_dict.keys())
    keys.sort()
    
    for j in range(data.shape[1]):
        for i,c in enumerate(keys):
            #get idx
            idx = clusters_dict[c]
            if j not in categorical:
                #calculate stats
                minval=np.min(data[idx,j])
                maxval=np.max(data[idx,j])
                meanval=np.mean(data[idx,j])
                std=np.std(data[idx,j])
                print("variable:{var}, cluster: {cluster}. min={minval}, max={maxval}, mean={meanval}, std={std}".format(var=j,cluster=c,minval=minval,maxval=maxval,meanval=meanval,std=std))
            else:
                ele = [int(x) for x in data[idx,j]]
                ele_size = float(len(ele))
                counter = collections.Counter(ele)
                print("variable:{var}, cluster: {cluster}. ".format(var=j,cluster=c), end=' ')
                for k,v in counter.most_common():
                    print(k,v/ele_size*100.0, end=' ')
                print("")

def create_clusters_dict(clusters):
    #clusters is a list
    
    clusters_dict = {}
    for i,c in enumerate(clusters):
        if c not in clusters_dict:
            clusters_dict[c] = []

        clusters_dict[c] += [i]
       
    return clusters_dict

def recode_clusters(clusters):
    #recode clusters codes in sequential order
    clusters_dict = create_clusters_dict(clusters)
    #order keys
    keys = list(clusters_dict.keys())
    keys.sort()
    
    equivalence = {}
    for i,k in enumerate(keys):
        equivalence[k] = i
    
    recoded_clusters = np.empty_like(clusters)
    
    for i,c in enumerate(clusters):
        recoded_clusters[i] = equivalence[c]
        
    return recoded_clusters

def adjust_clusters(clusters,equivalences):
    '''
    Different algorithms produce different clusters clode
    '''
    new_clusters = np.empty_like(clusters)
    nclusters  = len(equivalences)
    for i in range(len(clusters)):
        new_clusters[i] = equivalences[int(clusters[i])]
    
    return new_clusters

def fuzziness_estimation(n,d):
    return 1.0 + (1418.0/n + 22.05)*d**-2 + (12.33/n + 0.243)*d**(-0.0406*math.log(n)-0.1134)
    
    
def fix_weights(weights,force=None):
    w = weights.copy()
    
    if force is not None:
        idx,value = force

        w[idx] = 0.0
        s = np.sum(w)
        
        w /= s
        w *= (1.0-value)
        
        w[idx] = value
    else:
        w = w / np.sum(w) #normalize
    
    return w 
    

def recode_categorical_values(values,cats,a=0.999):
    new_values = np.empty_like(values,dtype=np.float32)
    for i,c in enumerate(cats):
        indices = np.where(values == int(c))[0]
        print(i,c,len(indices),i+a)
        new_values[indices] = (i + a)

    return new_values


import itertools
import numpy as np


def relabel(x,y,nclusters,verbose=False):
    n = len(x)

    z = np.empty_like(x)
    
    per = itertools.permutations(range(nclusters),nclusters)
    min_diff = n
    min_per = None
    for k,indices in enumerate(per):
        z.fill(-9)
        #copy values of y in z but changing index
        for i,j in enumerate(indices):
            pos = np.where(y == i)[0]
            z[pos] = j
        #calculate difference compared to x
        diff = np.count_nonzero(x != z)
        if verbose: print(k,indices,diff,x,y,z)
        if diff < min_diff:
            min_per = z.copy()
            min_diff = diff
            
    return min_per

    
if __name__ == "__main__":
    
    #generate cluestrs
    n =100
    clusters = np.empty(n)
    clusters[:40] = 20
    clusters[40:] = 50
    
    recoded_clusters = recode_clusters(clusters)
    
    dict1 = create_clusters_dict(clusters)
    dict2 = create_clusters_dict(recoded_clusters)
    
    assert len(dict1[20]) == len(dict2[0])
    assert len(dict1[50]) == len(dict2[1])

    assert np.sum(np.array(dict1[20]) - np.array(dict2[0])) == 0.0
    assert np.sum(np.array(dict1[50]) - np.array(dict2[1])) == 0.0


