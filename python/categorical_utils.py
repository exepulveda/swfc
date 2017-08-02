import numpy as np
import scipy.spatial
import scipy.optimize
import math

def categorical_keys(data):
    #build keys
    k = list(set(np.int32(data)))
    k.sort()
        
    return k

def simplex_coordinates1 ( m ):

#*****************************************************************************80
#
## SIMPLEX_COORDINATES1 computes the Cartesian coordinates of simplex vertices.
#
#  Discussion:
#
#    The simplex will have its centroid at 0
#
#    The sum of the vertices will be zero.
#
#    The distance of each vertex from the origin will be 1.
#
#    The length of each edge will be constant.
#
#    The dot product of the vectors defining any two vertices will be - 1 / M.
#    This also means the angle subtended by the vectors from the origin
#    to any two distinct vertices will be arccos ( - 1 / M ).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 June 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, the spatial dimension.
#
#    Output, real X(M,M+1), the coordinates of the vertices
#    of a simplex in M dimensions.
#
  import numpy as np

  x = np.zeros ( [ m, m + 1 ] )

  for k in range ( 0, m ):
#
#  Set X(K,K) so that sum ( X(1:K,K)^2 ) = 1.
#
    s = 0.0
    for i in range ( 0, k ):
      s = s + x[i,k] ** 2

    x[k,k] = np.sqrt ( 1.0 - s )
#
#  Set X(K,J) for J = K+1 to M+1 by using the fact that XK dot XJ = - 1 / M.
#
    for j in range ( k + 1, m + 1 ):
      s = 0.0
      for i in range ( 0, k ):
        s = s + x[i,k] * x[i,j]

      x[k,j] = ( - 1.0 / float ( m ) - s ) / x[k,k]

  return x

def simplex_coordinates2 ( m ):
#*****************************************************************************80
#
## SIMPLEX_COORDINATES2 computes the Cartesian coordinates of simplex vertices.
#
#  Discussion:
#
#    This routine uses a simple approach to determining the coordinates of
#    the vertices of a regular simplex in n dimensions.
#
#    We want the vertices of the simplex to satisfy the following conditions:
#
#    1) The centroid, or average of the vertices, is 0.
#    2) The distance of each vertex from the centroid is 1.
#       By 1), this is equivalent to requiring that the sum of the squares
#       of the coordinates of any vertex be 1.
#    3) The distance between any pair of vertices is equal (and is not zero.)
#    4) The dot product of any two coordinate vectors for distinct vertices
#       is -1/M; equivalently, the angle subtended by two distinct vertices
#       from the centroid is arccos ( -1/M).
#
#    Note that if we choose the first M vertices to be the columns of the
#    MxM identity matrix, we are almost there.  By symmetry, the last column
#    must have all entries equal to some value A.  Because the square of the
#    distance between the last column and any other column must be 2 (because
#    that's the distance between any pair of columns), we deduce that
#    (A-1)^2 + (M-1)*A^2 = 2, hence A = (1-sqrt(1+M))/M.  Now compute the
#    centroid C of the vertices, and subtract that, to center the simplex
#    around the origin.  Finally, compute the norm of one column, and rescale
#    the matrix of coordinates so each vertex has unit distance from the origin.
#
#    This approach devised by John Burkardt, 19 September 2010.  What,
#    I'm not the first?
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 June 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, the spatial dimension.
#
#    Output, real X(M,M+1), the coordinates of the vertices
#    of a simplex in M dimensions.
#
  import numpy as np

  x = np.zeros ( [ m, m + 1 ] )

  for j in range ( 0, m ):
    x[j,j] = 1.0

  a = ( 1.0 - np.sqrt ( float ( 1 + m ) ) ) / float ( m )

  for i in range ( 0, m ):
    x[i,m] = a
#
#  Adjust coordinates so the centroid is at zero.
#
  c = np.zeros ( m )
  for i in range ( 0, m ):
    s = 0.0
    for j in range ( 0, m + 1 ):
      s = s + x[i,j]
    c[i] = s / float ( m + 1 )

  for j in range ( 0, m + 1 ):
    for i in range ( 0, m ):
      x[i,j] = x[i,j] - c[i]
#
#  Scale so each column has norm 1.
#
  s = 0.0
  for i in range ( 0, m ):
    s = s + x[i,0] ** 2
  s = np.sqrt ( s )

  for j in range ( 0, m + 1 ):
    for i in range ( 0, m ):
      x[i,j] = x[i,j] / s

  return x

def simplex_coordinates2_test ( m ):

#*****************************************************************************80
#
## SIMPLEX_COORDINATES2_TEST tests SIMPLEX_COORDINATES2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 June 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, the spatial dimension.
#
  import numpy as np
  import platform
  from r8_factorial import r8_factorial
  from r8mat_transpose_print import r8mat_transpose_print
  from simplex_volume import simplex_volume

  print ( '' )
  print ( 'SIMPLEX_COORDINATES2_TEST' )
  print(( '  Python version: %s' % ( platform.python_version ( ) ) ))
  print ( '  Test SIMPLEX_COORDINATES2' )

  x = simplex_coordinates2 ( m )

  r8mat_transpose_print ( m, m + 1, x, '  Simplex vertex coordinates:' )

  s = 0.0
  for i in range ( 0, m ):
    s = s + ( x[i,0] - x[i,1] ) ** 2

  side = np.sqrt ( s )

  volume = simplex_volume ( m, x )

  volume2 = np.sqrt ( m + 1 ) / r8_factorial ( m ) \
    / np.sqrt ( 2.0 ** m ) * side ** m

  print ( '' )
  print(( '  Side length =     %g' % ( side ) ))
  print(( '  Volume =          %g' % ( volume ) ))
  print(( '  Expected volume = %g' % ( volume2 ) ))

  xtx = np.dot ( np.transpose ( x ), x )

  r8mat_transpose_print ( m + 1, m + 1, xtx, '  Dot product matrix:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'SIMPLEX_COORDINATES2_TEST' )
  print ( '  Normal end of execution.' )
  return


def get_new_point(p1,p2,d):
    x0,y0 = p1
    x1,y1 = p2
    m = (y1 - y0) / (x1 - x0)
    n = y0 - m*x0
    
    x = x0 - math.sqrt(d**2-(y0-m*x0 - n))
    y = m*x+n
    
    return (x,y)

def create_irregular_simplex(ncat,target_distances):
    simplex = simplex_coordinates2 (ncat-1)
    simplex_shape = simplex.shape
    #simplex is (ROWS,COLS) each COLS represents a category
    #now we want to move the vertex to match distances
    #print "initial simplex",simplex
    
    #print get_new_point(simplex[:,0],simplex[:,1],target_distances[0])
    #quit()
    
    max_target_distances = np.max(target_distances)
    
    simplex *= max_target_distances
    
    def min_func(coords,simplex_shape,target_distances):
        #reshape
        x = coords.reshape(simplex_shape)
        #now calculate pair distance 
        distances = scipy.spatial.distance.pdist(x.T)
        
        #print "distances",np.sum(target_distances - distances),distances
        
        return (target_distances - distances)
        
    simplex_flat = simplex.flatten()
    n = len(simplex_flat)
    
    #ret = min_func(simplex_flat,simplex_shape,target_distances)
    #print ret
    
    minval = np.empty(n)
    minval.fill(-np.max(target_distances))
    maxval = np.empty(n)
    maxval.fill(np.max(target_distances))
    
    opt = scipy.optimize.least_squares(min_func,simplex_flat,args=(simplex_shape,target_distances),verbose=0,xtol=1e-10,gtol=1e-10,ftol=1e-15,bounds=(minval, maxval))
    
    return np.array(opt.x).reshape(simplex_shape).T
    
    #print "lsq_solution",opt.x.copy()
    initial_solution = list(opt.x.copy())
    #initial_solution = list(simplex_flat)
    
    print("inital_solution",initial_solution)
    
    from annealing import Annealer
    class IrregularSimplexProblem(Annealer):
        def move(self):
            #select a coordinate and perturbe it
            i = np.random.randint(0, len(self.state))
            r = np.random.normal(loc=0,scale=max_target_distances)
            
            #print "moving",i,r,self.state[i]
            self.state[i] += r

        def energy(self):
            x = np.array(self.state).reshape(simplex_shape)
            #now calculate pair distance 
            distances = scipy.spatial.distance.pdist(x.T)
            
            return np.sum((target_distances - distances)**2)

    sa = IrregularSimplexProblem(initial_solution)
    xbest, cost = sa.anneal()
    xbest = np.array(xbest)

    print(xbest, cost)
    
    return xbest.reshape(simplex_shape).T

def create_regular_simplex(ncat):
    return simplex_coordinates2 (ncat-1).T
    
def categorical_to_simplex(values,distances=None):
    n = len(values)
    keys = categorical_keys(values)
    ncat = len(keys)
    
    if distances is None:
        x = create_regular_simplex(ncat)
        
    else:
        assert ncat*(ncat-1)/2 == len(distances),"%d cdifferent cats, but only %d"%((ncat*(ncat-1)/2),len(distances))
        x = create_irregular_simplex (ncat,distances)
    
    ret = np.empty((n,ncat-1))
    
    for i,k in enumerate(keys):
        #find key k
        indices = np.argwhere(values == k)
        ret[indices,:] = x[i,:]
    
    return ret,ncat-1
    
    
    

if ( __name__ == '__main__' ):
    ncat = 3
    target_distances = np.array([0.4,0.6,1.0])
    #make it symmetric
    #target_distances = target_distances + target_distances.T
    
    x = create_irregular_simplex (ncat,target_distances)
    
    print("target_distances",target_distances)
    print("final distances",scipy.spatial.distance.pdist(x.T))
    #print x
