#metrics for clustering
import numpy as np
import math

from sklearn.metrics import silhouette_score

#dbi
def dbi(data,clusters_dict,centroids):
    nclusters = len(clusters_dict)
    assert nclusters == len(centroids)
    
    s = np.empty(nclusters)
    k = 0
    for c,indices in clusters_dict.items():
        nc = len(indices)
        s[k] = np.sqrt(np.sum((data[indices,:] - centroids[k,:])**2)/nc)
        k+=1

    m = np.empty((nclusters,nclusters))
    for k1 in range(nclusters):
        for k2 in range(nclusters):
            m[k1,k2] = np.sqrt(np.sum((centroids[k1,:] - centroids[k2,:])**2))

    r = np.zeros((nclusters,nclusters))
    for k1 in range(nclusters):
        for k2 in range(nclusters):
            if k1 != k2:
                r[k1,k2] = (s[k1] + s[k2])/m[k1,k2]

    ret = 0.0
    for k in range(nclusters):
        rc = r[k,:].copy()
        rc[k] = 0
        ret += np.max(rc)
    
    return ret/nclusters

#compactness
def compactness(nclusters,clusters,neighbours):
    n = {}
    nd = {}
    for k in range(nclusters):
        n[k] = 0.0
        nd[k] = 0.0
        
        
    for i,c in enumerate(clusters):
        neighbour = neighbours[i]
        #count how many different to me
        nd[c] += len([x for x in neighbour if clusters[x] != c])
        n[c] += 1
        
        
    ret1 = 0.0
    ret2 = 0.0
    for k in range(nclusters):
        ret1 += nd[k]
        ret2 += n[k] 
        
    return ret1/ret2

def validity_k(u,data,centroids,distance_functions=None,weights=None,full=False):
    n,c = u.shape
    
    #distance among all centroids
    mind = None
    for i in range(c):
        for j in range(i+1,c):
            if distance_functions is None or weights is None:
                d = np.linalg.norm(centroids[:,i]-centroids[:,j])
            else:
                w = np.maximum(weights[i,:],weights[j,:])
                d = distance_functions.distance(centroids[:,i],centroids[:,j],w)
                
            d = d**2
            if mind is None or d < mind:
                mind = d

            #print i,j,d
    
    #intraclass similarity
    intraclass_similarity = 0.0
    for k in range(c):
        for j in range(n):
            if distance_functions is None or weights is None:
                d = np.linalg.norm(data[j,:]-centroids[:,k])
            else:
                w = weights[k,:]
                d = distance_functions.distance(data[j,:],centroid,w)
            intraclass_similarity += (u[j,k]**2) * (d**2)    

    #penalty
    penalty = 0.0
    mean_all = np.mean(data,axis=0)
    for k in range(c):
        if distance_functions is None or weights is None:
            d = np.linalg.norm(mean_all-centroids[:,k])
        else:
            w = weights[k,:]
            d = distance_functions.distance(mean_all,centroids[:,k],w)
        penalty += d**2
    penalty /= c
        
    if full:
        return intraclass_similarity,penalty,mind,(intraclass_similarity+penalty)/mind
    else:
        return (intraclass_similarity+penalty)/mind

def vpc(u):
    n,c = u.shape
    
    return np.sum(u**2)/float(n)

def mvpc(u,pre_vpc=None):
    n,c = u.shape
    
    if pre_vpc is None:
        pre_vpc = vpc(u)
    
    return 1.0 - c/(c-1.0)*(1.0- pre_vpc)


def vpe(u):
    n,c = u.shape
    
    return -np.sum(u*np.log(u))/float(n)

def vavcd(x,u,m,centroids):
    n,c = u.shape
    nc,ndim = centroids.shape
    um = u**m

    assert c == nc, "clusters are not similar in vavcd"
    
    ret = 0.0
    for i in range(c):
        num = np.sum(um[:,i] * np.linalg.norm(x-centroids[i,:]))
        den = np.sum(um[:,i])
        ret += num/den
        
    return ret/(n*c)

def vfs(x,u,m,centroids):
    n,c = u.shape
    nc,ndim = centroids.shape
    um = u**m
    assert c == nc, "clusters are not similar in vfs"
    
    mean_x = np.mean(x,axis=0)
    
    ret = 0.0
    for i in range(c):
        ret += np.sum(um[:,i] * (np.linalg.norm(x-centroids[i,:]) - np.linalg.norm(mean_x-centroids[i,:])))
        
    return ret

def vxb(x,u,m,centroids):
    n,c = u.shape
    nc,ndim = centroids.shape
    um = u**m
    
    assert c == nc, "clusters are not similar in vxb"
    
    ret = 0.0
    for i in range(c):
        ret += np.sum(um[:,i] * np.linalg.norm(x-centroids[i,:]))
        
    return ret/(n*vmcd(centroids))

def vmcd(centroids):
    nc,ndim = centroids.shape

    mind = np.inf
    for i in range(nc):
        for j in range(i+1,nc):
            d = np.linalg.norm(centroids[i,:]-centroids[j,:])
            if d < mind:
                mind = d
        
    return mind


def probabilities(diss,clusters):
    n,m = diss.shape
    assert n == m
    assert len(clusters) == n
    
    codes = sort(list(clusters.keys()))
    
    nclusters = len(codes)
    
    probs = np.empty((n,nclusters))
    
    
    
    #calculate average distance to all clusters
    for i in range(n):
        mean_distance_to = {}
        #select clusters
        for k,cluster in enumerate(codes):
            idx = clusters[cluster]
            #calculate distances to all elements in cluster
            mean_distance = np.mean(diss[i,idx])
            mean_distance_to[cluster] = mean_distance
            
            probs[i,k] = mean_distance

    #now normalize each row
    probs[:,:] = probs[:,:] / np.sum(probs,axis=1)
    
    return probs

def silloutte(diss,clusters,nclusters):
    n,m = diss.shape
    assert n == m
    assert len(clusters) == n
    
    return silhouette_score(diss,clusters,metric="precomputed")
    
    a = np.empty(n)
    b = np.empty(n)
    s = np.empty(n)

    #calculate average distance to all clusters
    for i in range(n):
        #a
        idx = np.argwhere(clusters == clusters[i])
        
        a[i] = np.mean(diss[i,idx])

        x = []
        for c in range(1,nclusters+1):
            #b
            if c != clusters[i]:
                idx = np.argwhere(clusters == c)
                x += [np.mean(diss[i,idx])]
                
        b[i] = np.min(x)
        
        #print("a,b",i,a[i],b[i])
        
        s[i] = (b[i] - a[i])/max(b[i] , a[i])
        
    
    return np.mean(s)

def bcd(clusters_dict, centroids,centroid,distance_functions,debug=False):
    #centroids is a dict with centroids of each cluster
    #centroid is the global centroid
    #distance_functions is a dic with distance function of each cluster 
    nclusters = len(clusters_dict)
    
    assert nclusters == len(centroids)
    
    ret = 0.0
    n = 0
    for c,indices in clusters_dict.items():
        d = distance_functions[c](centroids[c],centroid)
        if debug: print("bcd",c,len(indices),centroids[c],centroid,d)
        ret += d*d*len(clusters_dict[c])
        n += len(clusters_dict[c])
        
    ret = ret/(n*nclusters)
    
    return ret

def wcd(values,clusters_dict, centroids,centroid,distance_functions,debug=False):
    #values are all values
    #centroids is a dict with centroids of each cluster
    #centroid is the global centroid
    #distance_function 
    nclusters = len(clusters_dict)
    
    assert nclusters == len(centroids)
    
    ret = 0.0
    for c,indices in clusters_dict.items():
        n = len(indices)
        accum = 0.0
        for i in indices:
            if debug: print("wcd",c,i,n,centroids[c],values[i,:])
            d = distance_functions[c](centroids[c],values[i,:])
            accum += d*d
        
        accum /= n
        
        ret += math.sqrt(accum)
        
    ret = ret/nclusters
    
    return ret

def sf(values,clusters_dict, centroids,centroid,distance_functions,debug=False):
    #values are all values
    #centroids is a dict with centroids of each cluster
    #centroid is the global centroid
    #distance_function 
    bcd_ = bcd(clusters_dict, centroids,centroid,distance_functions,debug=debug)
    wcd_ = wcd(values,clusters_dict, centroids,centroid,distance_functions,debug=debug)
    if debug: print("bcd",bcd_,"wcd",wcd_)
    
    return 1.0 - 1.0/np.exp(np.exp(bcd_ - wcd_))
    #return math.exp(bcd_)/math.exp(wcd_)

def separation(centroids):
    NC,ND = centroids.shape
    
    ret = 0.0
    for i in range(NC):
        for j in range(i+1,NC):
            d = np.linalg.norm(centroids[i,:]-centroids[j,:])
            ret += d
            
    return d
