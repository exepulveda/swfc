import numpy as np
import pickle 
#from graph_utils import build_adjacent_set

nc1 = 200
nc2 = 200
nc3 = 400
nc4 = 100

np.random.seed(1634120)

nsamples = nc1+nc2+nc3+nc4

def make_cat(categories, percentages, n):
    sizes = np.int32(np.cumsum(np.array(percentages) * n))
    
    if sizes[-1] < n:
        sizes[-1] = n
    
    ret =np.empty(n,dtype=np.int32)
    
    offset = 0
    for i,c in enumerate(categories):
        ret[offset:sizes[i]] = c

        offset = sizes[i]
        
    return ret

m = 4



cut_all = np.random.normal(size=nsamples,loc=0.68,scale=0.3) #cut
fe_all = np.random.normal(size=nsamples,loc=2.56,scale=1.15) #fe
as_all = np.random.normal(size=nsamples,loc=21.5,scale=11.08) #as
rec_all  = np.random.normal(size=nsamples,loc=81.43,scale=6.04) #rec

data = np.empty(shape=(nsamples,m))
clusters = np.empty(nsamples)

data[:,0] = cut_all
data[:,1] = fe_all
data[:,2] = as_all
data[:,3] = rec_all

'''clusters:
1: high cut, low as, high rec, alt=((1,80%),(2,20%), litho=*)
2: high cut, low  fe, low as, alt=((2,70%),(4,30%), litho=((1,15%),(2,85%))
3: low  cut, low fe, * as low, low rec, alt=*, litho=((1,30%),(2,70%))
4: * cut, high fe,  high as, * rec, alt=((1,90%))
'''


#real cluster
slice_c1 = slice(0,nc1)
slice_c2 = slice(nc1,nc1+nc2)
slice_c3 = slice(nc1+nc2,nc1+nc2+nc3)
slice_c4 = slice(nc1+nc2+nc3,nc1+nc2+nc3+nc4)
clusters[slice_c1] = 0
clusters[slice_c2] = 1
clusters[slice_c3] = 2
clusters[slice_c4] = 3

data[slice_c1,0] = np.random.normal(size=nc1,loc=0.8,scale=0.05)
data[slice_c1,2] = np.random.normal(size=nc1,loc=30.0,scale=1.0)
data[slice_c1,3] = np.random.normal(size=nc1,loc=88.0,scale=1.0)

data[slice_c2,0] = np.random.normal(size=nc2,loc=0.9,scale=0.05)
data[slice_c2,1] = np.random.normal(size=nc2,loc=1.5,scale=0.1)
data[slice_c2,2] = np.random.normal(size=nc2,loc=15.0,scale=1.0)

data[slice_c3,0] = np.random.normal(size=nc3,loc=0.4,scale=0.05)
data[slice_c3,1] = np.random.normal(size=nc3,loc=1.2,scale=0.1)
data[slice_c3,3] = np.random.normal(size=nc3,loc=70.0,scale=2.0)

data[slice_c4,1] = np.random.normal(size=nc4,loc=4.0,scale=0.1)
data[slice_c4,2] = np.random.normal(size=nc4,loc=40.0,scale=1.0)


data[:,0] = np.clip(data[:,0],0.0,4.0)
data[:,1] = np.clip(data[:,1],0.0,10.0)
data[:,2] = np.clip(data[:,2],0.0,60.0)
data[:,3] = np.clip(data[:,3],40.0,99.0)

#locations
locations = np.random.uniform(size=(nsamples,2)) * 100.0
#c1
locations[0:nc1//2  ,0] = np.random.uniform(low=10.0,high=35.0,size=nc1//2)
locations[0:nc1//2  ,1] = np.random.uniform(low=10.0,high=35.0,size=nc1//2)
locations[nc1//2:nc1,0] = np.random.uniform(low=65.0,high=85.0,size=nc1//2)
locations[nc1//2:nc1,1] = np.random.uniform(low=65.0,high=85.0,size=nc1//2)
#c2
locations[slice_c2  ,0] = np.random.uniform(low=0.0,high=100.0,size=nc2)
locations[slice_c2  ,1] = np.random.uniform(low=30.0,high=70.0,size=nc2)
#c3
locations[slice_c3  ,0] = np.random.uniform(low=5.0,high=45.0,size=nc3)
locations[slice_c3  ,1] = np.random.uniform(low=65.0,high=100.0,size=nc3)

locations[slice_c4  ,0] = np.random.uniform(low=55.0,high=100.0,size=nc4)
locations[slice_c4  ,1] = np.random.uniform(low=0.0,high=35.0,size=nc4)

#50 at ramdom
random_samples = 100
idx = np.random.choice(np.array(nsamples),size=random_samples)
locations[idx,:] = np.random.uniform(size=(random_samples,2)) * 100.0

X = np.c_[locations,data,clusters]

np.savetxt("../../data/ds4_noise.csv",idx,fmt="%d",delimiter=",")
np.savetxt("../../data/ds4.csv",X,fmt="%.5f",delimiter=",",header="x,y,cut,fe,as,rec,cluster",comments="")
np.save("../../data/ds4.npy",X)
#clip recoveries
#data[:,5] = np.clip(data[:,5],60.0,98.0)


#ret,triangulation = build_adjacent_set(locations)

#pickle.dump(ret,open("adj_set_ds4.dump","w"))
#pickle.dump(triangulation,open("voronoi_ds4.dump","w"))


#calculate historgrmas
#import matplotlib.pyplot as plt
#plt.scatter(X[:,0],X[:,1],c=X[:,6])
#plt.show()
