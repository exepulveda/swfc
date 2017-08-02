from numba import jit,float64, int32,autojit
from numba.numpy_support import from_dtype

import numpy as np
import logging


#distances metrics
def standard_distance(x,y,*args,**kargs):
    dim = args[0]
    if len(args) > 1:
        scale = args[1]
    else:
        scale = 1.0
    #print "standard_distance",x.shape,y.shape,args,kargs
    ret = np.abs(x[dim]-y[dim])/scale    
    #print "standard_distance ret.shape",ret.shape
    
    return ret

def simplex_distance(x,y,*args,**kargs):
    #args should contain slice and scale optional
    start = args[0]
    end = args[1]
    sl = slice(start,end)
    if len(args) > 2:
        sk = args[2]
    else:
        sk = 1.0
        
        
    #print "slice,sk",sl,sk
    #print "x[sl]",x[sl]
    #print "y[sl]",y[sl]
    
    #quit()
    
    d = np.linalg.norm(x[sl]-y[sl],axis=0)
        
    return d/sk

def categorical_distance(x,y,*args,**kargs):
    #args should contain dim and scale optional
    dim = args[0]
    if len(args) > 1:
        sk = args[1]
    else:
        sk = 1.0
        
    d = (x[dim] == y[dim])
        
    return d.astype(float)/sk
    
@jit(float64(float64,float64,float64[:]))
def targeted_distance_simple(x,y,targets):
    T = len(targets)
    tdx =np.empty(T)
    tdy =np.empty_like(tdx)

    for i,t in enumerate(targets):
        tdx[i] = abs(x-t)
        tdy[i] = abs(y-t)

    td = np.maximum(tdx,tdy)

    return np.min(td)
    
def targeted_distance(x,y,*args,**kargs):
    dim = args[0]
    target = args[1]
    if len(args) > 2:
        scale = args[2]
    else:
        scale = 1.0

    if len(target) > 0:
        #accoding shape of x
        x_shape = x.shape
        #print "x_shape",x_shape
        if len(x_shape) > 1:
            #print "targeted_distance:",dim,target,scale
            
            tdx =np.empty((x_shape[1],len(target)))
            tdy =np.empty_like(tdx)
            
            for i,t in enumerate(target):
                tdx[:,i] = np.abs((x[dim,:]-t)/scale)
                tdy[:,i] = np.abs((y[dim,:]-t)/scale)
            
            
            #print "tdx",tdx
            #print "tdy",tdy
            #target distance is max x, y to each target
            td = np.maximum(tdx,tdy)
            #td = np.abs(tdx-tdy)
            #print "td",td
            #target distance is minimum target
            d =  np.min(td,axis=1)
            #print "d",d
            
            #quit()
            
            return d
        elif len(x_shape) == 1:
            tdx =np.empty(len(target))
            tdy =np.empty_like(tdx)
            
            for i,t in enumerate(target):
                tdx[i] = abs((x[dim]-t)/scale)
                tdy[i] = abs((y[dim]-t)/scale)
                
            #target distance is max x, y to each target
            #td = np.maximum(tdx,tdy)
            td = np.abs(tdx-tdy)
            #target distance is minimum target
            d =  np.min(td)
            
            #print d.shape
            
            return d
        else:
            assert False, "invalida shape in targeted_distance"
        
    else:
        return standard_distance(x,y,args,kargs)
        

class DistanceCalculator:
    def __init__(self,metric,*args,**kargs):
        #print "args",args
        #print "kargs",kargs
        self.args = args
        self.kargs = kargs
        self.metric = metric
        
    def distance(self,x,y):
        return self.metric(x,y,*self.args,**self.kargs)
        
    def __call__(self,x,y):
        return self.distance(x,y)

class DistanceMetricCalculator:
    def __init__(self,ndim):
        self.ndim = ndim
        self.distance_functions = [None]*ndim
           
    def set_distance(self,i,distance_function,*args,**kargs):
        assert i < self.ndim
        assert i >= 0
        
        self.distance_functions[i] = DistanceCalculator(distance_function,*args,**kargs)

    def distance_dim(self,x,y,weights=None,debug=False):
        if debug: print("x-y.shape",x.shape,y.shape)

        _x, _y = np.broadcast_arrays(x,y)
        
        if debug: print(_x.shape)
        if debug: print(_y.shape)
        
        if weights is None:
            weights = np.ones(self.ndim)#/self.ndim
        else:
            assert len(weights) == self.ndim, "weights length =[%d], ndim=[%d]"%(len(weights),self.ndim)

        if debug: print("weights.shape",weights.shape)


        #fix dim
        ss = list(_x.T.shape)
        ss[0] = self.ndim
        ss = tuple(ss)
        
        
        d = np.empty(ss)
        for i in range(self.ndim):
            if debug: print(self.distance_functions[i])
            d_tmp = self.distance_functions[i](_x.T,_y.T) #dimension to be the latest axe
            
            if debug: print("A",d_tmp.shape,d.shape)

            d[i] = d_tmp * weights[i]

        if debug: print("distance_dim",d,weights)

        return d

    def dissimilarity(self,x,y,debug=False):
        #just use euclidian distance over expended dimesions
        n,p = x.shape
        m,p2 = y.shape
        
        assert p == p2

        dxy = np.zeros((n,m,self.ndim))

        for j in range(m):
            tmp = self.distance_dim(x[:,:],y[j,:],debug=debug)
            
            dxy[:,j,:] = tmp.T
            
        #quit()
        return dxy

    def distance(self,x,y,weights=None,debug=False):
        d = self.distance_dim(x,y,weights=weights,debug=debug)
        #print "d.shape",d.shape
        return np.sum(self.distance_dim(x,y,weights=weights,debug=debug),axis=0).T

    def distance_centroids(self,x,centroids,weights,debug=False):
        #x is N,P and centroid is C,P
        #return a distance matrix NxC
        n,p = x.shape
        c,p2 = centroids.shape

        assert p == p2
        assert (c,self.ndim) == weights.shape

        dxy = np.zeros((n,c))
        
        #addition = self.lambda_value*np.log(ndim)

        for j in range(c):
            tmp = self.distance_dim(x[:,:],centroids[j,:],debug=debug)
            tmp2 = tmp.T * weights[j,:]
            
            dxy[:,j] = np.sum(tmp2,axis=1)
            
                
        #dxy[:,:] += addition

            
        return dxy

    def distance_max(self,x,y,nclusters,clusters,weights,debug=False):
        #x is N,P and y is C,P
        #return a distance matrix NxC
        n,p = x.shape
        c,p2 = y.shape
        
        assert p == p2
        assert n == c

        dxy = np.zeros((n,n))
        cls = clusters[:]
        
        #print clusters[0:10]
        #print weights


        for j in range(c):
            #maximum weight
            w1 = weights[cls,:] #weights of all x on all clusters
            w2 = weights[clusters[j],:] #weights of y[j] on all clusters
            w = np.maximum(w1,w2)
            

            tmp = self.distance_dim(x[:,:],y[j,:],debug=debug)

            #print j,w1.shape,w2.shape,w.shape,tmp.shape

            tmp2 = tmp.T * w
            
            dxy[:,j] = np.sum(tmp2,axis=1)

        return dxy

        
    def __call__(self,x,weights=None):
        return self.distance(x,y,weights)

class DefaultMetricCalculator:
    def __init__(self,ndim):
        self.ndim = ndim
           
    def distance(self,x,y,weights=None,debug=False):
        n,p = x.shape
        c,p2 = y.shape

        dxy = np.zeros((n,c))
        
        for j in range(c):
            tmp = np.linalg.norm(x[:,:]-y[j,:],axis=1)
            
            dxy[:,j] = tmp

        return dxy

        
    def __call__(self,x,y):
        return self.distance(x,y)


class WeightedDistanceCalculator:
    def __init__(self,distance_functions,numerical_size,categorical_size):
        self.distance_functions = distance_functions
        self.weights = None
        self.numerical_size = numerical_size
        self.categorical_size = categorical_size
        self.lambda_value = 1.0
        self.ndim = self.numerical_size+len(self.categorical_size)
        
    def set_weights(self,w):
        self.weights = w

    def set_lambda(self,l):
        self.lambda_value = l
        
    def dispersion(self,values):
        n = len(values)
        n2 =float(n**2)
        sk = np.empty(self.ndim)
        for i in range(self.ndim):
            d_tmp = 0
            if i < self.numerical_size:
                for j in range(n):
                    d_tmp += np.sum(self.distance_functions[i](values[:,i],values[j,i]))
            else:
                for j in range(n):
                    d_tmp += np.sum(self.distance_functions[i](values[:,:],values[j,:]))
            
            sk[i] = np.sum(d_tmp)/ n2

        return sk
        
    def distance_two(self,x,y,weights=None,debug=False):
        #x and y are on dim size n
        assert x.shape == y.shape
        assert len(x.shape) == 1

        n = x.shape[0]

        ndim = self.numerical_size+len(self.categorical_size)
        assert len(weights) == ndim

        d = 0.0
        for i in range(ndim):
            if i < self.numerical_size:
                d_tmp = self.distance_functions[i](x[i],y[i])
            else:
                d_tmp = self.distance_functions[i](x[:],y[:])

            d += d_tmp * weights[i]

        return d

    def distance_xy(self,x,y,debug=False):
        #x and y are on dim size n
        assert x.shape == y.shape
        assert len(x.shape) == 1

        n = x.shape[0]
        
        ndim = self.numerical_size+len(self.categorical_size)
        
        d = np.empty(ndim)

        for k in range(ndim):
            if k < self.numerical_size:
                d_tmp = self.distance_functions[k](x[k],y[k])
            else:
                d_tmp = self.distance_functions[k](x[:],y[:])
            d[k] = d_tmp

        return d


    def distance(self,x,y,weights=None,debug=False):
        #x is N,P and y is C,P
        #return a distance matrix NxC
        if len(x.shape) == 1:
            x = x[np.newaxis,:]
        if len(y.shape) == 1:
            y = y[np.newaxis,:]
        
        n,p = x.shape
        c,p2 = y.shape
        
        assert p == p2

        ndim = self.numerical_size+len(self.categorical_size)
        dxy = np.zeros((n,c,ndim))
        
        if weights is None:
            weights = self.weights

        for j in range(c):
            #calculate all distance between x and y[j]
            #distance for numerical variables
            #for i in xrange(self.numerical_size):
            for i in range(ndim):
                #print j,i,x.shape,y.shape,self.weights.shape
                if i < self.numerical_size:
                    d_tmp = self.distance_functions[i](x[:,i],y[j,i])
                else:
                    d_tmp = self.distance_functions[i](x[:,:],y[np.newaxis,j,:])

                

                dxy[:,j,i] = d_tmp * weights[j,i]

            #fake categorical
            #offset = self.numerical_size
            #for i,k in enumerate(self.categorical_size):
            #    d_tmp = self.distance_functions[i](x[:,i],y[j,i])

            #    #x_cat = x[:,offset:offset+k]
            #    #y_cat = y[j,offset:offset+k]
                
            #    #print "(x_cat-y_cat).shape",(x_cat-y_cat).shape
            #    #print "np.linalg.norm(x_cat-y_cat,axis=1).shape",np.linalg.norm(x_cat-y_cat,axis=1).shape
            #    #d = (np.linalg.norm(x_cat-y_cat,axis=1) * self.weights[j,self.numerical_size+i])
            #    #move offset to next
            #    #offset += k
            
            #    #dxy[:,j,i] += d
            
        return dxy

    def distance_cosa(self,x,centroids,debug=False):
        #x is N,P and centroid is C,P
        #return a distance matrix NxC
        n,p = x.shape
        c,p2 = centroids.shape

        ndim = self.numerical_size+len(self.categorical_size)
        
        assert p == p2
        assert (c,ndim) == self.weights.shape

        dxy = np.zeros((n,c))
        
        addition = self.lambda_value*np.log(ndim)

        for j in range(c):
            #calculate all distance between x and centroids[j]
            #distance for numerical variables
            #for i in xrange(self.numerical_size):
            #j is the cluster as well
            for i in range(ndim):
                #print j,i,x.shape,y.shape,self.weights.shape
                if i < self.numerical_size:
                    d_tmp = self.distance_functions[i](x[:,i],centroids[j,i])
                else:
                    d_tmp = self.distance_functions[i](x[:,:],centroids[np.newaxis,j,:])

                #print d_tmp.shape,j,i,self.weights.shape

                part1 = d_tmp * self.weights[j,i]
                part2 = self.lambda_value*self.weights[j,i]*np.log(self.weights[j,i])


                dxy[:,j] += part1 # part2
                
        #dxy[:,:] += addition

            
        return dxy

    def distance_clusters(self,x,y,nclusters,clusters,debug=False):
        #x is N,P and y is C,P
        #return a distance matrix NxC
        n,p = x.shape
        c,p2 = y.shape
        
        assert p == p2
        assert n == c

        ndim = self.numerical_size+len(self.categorical_size)
        dxy = np.zeros((n,n))
        
        addition = self.lambda_value*np.log(ndim)

        for j in range(n):
            for i in range(ndim):
                #get all clusters of j
                cls = clusters[:]
                w = self.weights[cls,i]
                
                w = np.maximum(w,self.weights[clusters[j],i])
                
                #part1 = self.lambda_value*w*np.log(w)
            
                #distance of all x[:,i] in indices to y[j,i]
                if i < self.numerical_size:
                    d_tmp = self.distance_functions[i](x[:,i],y[j,i])
                else:
                    d_tmp = self.distance_functions[i](x[:,:],y[np.newaxis,j,:])

                part2 = d_tmp * w

                #print j,i,part1,part2
                
                dxy[:,j] += part2

            #dxy[:,j] += addition
            
        #quit()
        #symmetric
        #dxy = dxy + dxy.T
        return dxy

        
    def dissimilarity(self,x,y,clusters):
        #x is N,P and y is C,P
        #return a distance matrix NxC
        n,p = x.shape
        c,p2 = y.shape
        
        assert p == p2

        dxy = np.zeros((n,c,p))

        for j in range(c):
            #calculate all distance between x and y[j]
            #distance for numerical variables
            for i in range(self.numerical_size):
                #print j,i,x.shape,y.shape,self.weights.shape
                d_tmp = distance_functions[i](x[:,i],y[j,i])
                dxy[:,j,i] = d_tmp * self.weights[j,i]

            #fake categorical
            offset = self.numerical_size
            for i,k in enumerate(self.categorical_size):
                x_cat = x[:,offset:offset+k]
                y_cat = y[j,offset:offset+k]
                
                #print "(x_cat-y_cat).shape",(x_cat-y_cat).shape
                #print "np.linalg.norm(x_cat-y_cat,axis=1).shape",np.linalg.norm(x_cat-y_cat,axis=1).shape
                d = (np.linalg.norm(x_cat-y_cat,axis=1) * self.weights[j,self.numerical_size+i])
                #move offset to next
                offset += k
            
                dxy[:,j,i] += d
            
        return dxy

        
    def __call__(self,x,y):
        return self.distance(x,y)

if __name__ == "__main__":
    ndim = 4
    nrealdim = 4
    n = 3
    dmc = DistanceMetricCalculator(ndim)
    
    dmc.set_distance(0,standard_distance,0)
    dmc.set_distance(1,standard_distance,1)
    dmc.set_distance(2,standard_distance,2)
    dmc.set_distance(3,standard_distance,3)
    
    x = np.random.random((n,nrealdim))
    y = np.random.random((n,nrealdim))
    ret = dmc.distance(x,y)
    
    assert ret.shape == (n,), str(ret)

    x = np.random.random((n,nrealdim))
    y = np.random.random(nrealdim)
    ret = dmc.distance(x,y)
    assert ret.shape == (n,)

    x = np.random.random(nrealdim)
    y = np.random.random(nrealdim)
    ret = dmc.distance(x,y)
    assert ret.shape == ()

    x = np.random.random(nrealdim)
    y = np.random.random((n,nrealdim))
    ret = dmc.distance(x,y)
    assert ret.shape == (n,)

    x = np.random.random((n,n+2,nrealdim))
    y = np.random.random((n,n+2,nrealdim))
    ret = dmc.distance(x,y)
    assert ret.shape == (n,n+2)

    #second case
    ndim = 4
    nrealdim = 6
    n = 3
    dmc = DistanceMetricCalculator(ndim)
    
    dmc.set_distance(0,standard_distance,0)
    dmc.set_distance(1,standard_distance,1)
    dmc.set_distance(2,standard_distance,2)
    dmc.set_distance(3,simplex_distance,3,5)
    
    x = np.random.random((n,nrealdim))
    y = np.random.random((n,nrealdim))
    ret = dmc.distance(x,y)
    assert ret.shape == (n,)

    x = np.random.random((n,nrealdim))
    y = np.random.random(nrealdim)
    ret = dmc.distance(x,y)
    assert ret.shape == (n,)

    x = np.random.random(nrealdim)
    y = np.random.random(nrealdim)
    ret = dmc.distance(x,y)
    assert ret.shape == ()

    x = np.random.random(nrealdim)
    y = np.random.random((n,nrealdim))
    ret = dmc.distance(x,y)
    assert ret.shape == (n,)

    x = np.random.random((n,n+2,nrealdim))
    y = np.random.random((n,n+2,nrealdim))
    ret = dmc.distance(x,y)
    assert ret.shape == (n,n+2)
