import numpy as np
from queue import Queue
from scipy.spatial import cKDTree
import logging


    

def energy(solution,values):
    logger = logging.getLogger("energy")    
    ret = 0.0
    clusters = set(solution)
    for cluster in clusters:
        logger.debug("evaluating cluster %d",cluster)
        
        indices = np.where(solution == cluster)[0]
        n = len(indices)

        if n > 1:
            #centroid
            centroid = np.mean(values[indices,:],axis=0)
            #WCSS
            cost = np.sum(np.linalg.norm(values[indices,:] - centroid))
            
            #distances = pdist(values[indices,:])
            #var = np.var(distances)
            logger.debug("cluster %d, size=%d, cost=%f",cluster,n,cost)
            
            ret += cost
        else:
            ret += 100000
    return ret

def eps_neighborhood(p,eps):
    pass
    
def is_core(p,eps,minpts):
    return len(eps_neighborhood(p,eps)) >= minpts
    
def directly_density_reachable(p,q,eps,minpts):
    neigbours = eps_neighborhood(q,eps)
    return p in neigbours and len(neigbours) >= minpts
    
def density_reachable(p,q,eps,minpts):
    neigbours = eps_neighborhood(q,eps)
    return p in neigbours and len(neigbours) >= minpts
    
def get_neighbors(i,l,v, kdtree_locations, kdtree_values, eps_locations, eps_values):
    q1 = kdtree_locations.query_ball_point(l, eps_locations)
    q2 = kdtree_values.query_ball_point(v, eps_values)
    
    #print len(q1), len(q2)
    
    all_sets = set(q1) & set(q2)
    
    all_sets.discard(i)
    return list(all_sets)
    
    
def st_dbscan(locations,values,eps1,eps2,minpts,epsilon):
    cluster_label = 0
    
    n = len(locations)
    assert n == len(values)

    #build a KDTree for both sets
    kdtree_locations = cKDTree(locations)
    kdtree_values = cKDTree(values)

    cluster = [None]*n
    noise = [False]*n
    
    all_neigbors = [get_neighbors(i,locations[i],values[i],kdtree_locations,kdtree_values,eps1,eps2) for i in range(n)]
    
    for i in range(n):
        print("processing",i)
        if cluster[i] is None:
            print("get_neighbors...")
            neigbors = all_neigbors[i]
            print("get_neighbors..",len(neigbors))
            
            m = len(neigbors)
            if m < minpts:
                noise[i] = True
            else:
                cluster_label += 1
                for j in neigbors:
                    cluster[j] = cluster_label
                
                queue = set(neigbors)

                print("processing queue..")
                while len(queue) > 0:
                    y = queue.pop()
                    #print "processing element from queue..",y

                    y_neigbors = all_neigbors[y]
                    #print "y_neigbors..",len(y_neigbors)
                    
                    y_size = len(y_neigbors)
                    if y_size >= minpts:
                        # temporary cluster
                        tmp_cluster = [k for k,x in enumerate(cluster) if x == cluster_label]
                        for o in y_neigbors:
                            #calculating cluster_average()
                            cluster_average = np.mean(np.sqrt(np.sum((values[tmp_cluster,:] - values[o,:])**2,axis=1)))
                            #print y,o,"cluster_average",cluster_average,"noise",noise[o],"cluster",cluster[o]
                            if (not noise[o] or cluster[o] is None) and cluster_average < epsilon:
                                cluster[o] = cluster_label
                                queue.add(o)
                                #print y,o,"cluster_average",cluster_average,"noise",noise[o],"cluster",cluster[o]

                print("processing queue..done")
    
    return cluster

if __name__ == "__main__":
    #read muestras
    
    data = np.loadtxt("../data/muestras.csv",skiprows=1,delimiter=",")
    #x,y,z,cut,au,ug1,ug2,Ica_1,Ica_2
    locations = data[:,0:3]
    values = data[:,3:5]
    
    eps1 = 50.0
    eps2 = 5.0
    minpts = 10
    epsilon = 1.0
    
    ret = st_dbscan(locations,values,eps1,eps2,minpts,epsilon)
    
    for i,c in enumerate(ret):
        print(i,",",c)
