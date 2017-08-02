import numpy as np

def fix_weights(w,force):
    idx,value = force
    
    w[idx] = 0.0
    s = np.sum(w)
    
    w /= s
    w *= (1.0-value)
    
    w[idx] = value
    
ND = 6

force = (5,0.3)

for i in range(100):
    w = np.random.random(ND)
    fix_weights(w,force)
    
    print(w)
    print(np.sum(w))
