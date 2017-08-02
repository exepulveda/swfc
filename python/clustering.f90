module clustering
  implicit none
  !integer, parameter :: dp = kind(1.d0)
  !integer, parameter :: sp = kind(1.0)

contains

subroutine dispersion(data,cluster,s,verbose)
  use distances
  implicit none
  real(kind=4), intent(in)   :: data(:,:) !NxND
  integer(kind=8), intent(in):: cluster(:)
  real(kind=4), intent(inout):: s(:,:) !NCxND
  integer, intent(in) :: verbose

  !locals
  real(kind=8) d(size(data,2))
  integer i,j,k,NC,ND,N,c1,c2

  NC = size(s,1)
  N = size(data,1)
  ND = size(data,2)
  
  s = 0.0
  
  do i=1,N
    c1 = cluster(i)
    k = 0
    do j=1,N
      c2 = cluster(j)
      if (c1 == c2) then
        call distance_function_dim(data(i,:),data(j,:),d)
        
        
        s(c1,:) = s(c1,:) + d
        
        k = k+1
      end if
    end do
    s(c1,:) = s(c1,:)/k
    !print *,i,c1,k,s(c1,:)
  end do
end subroutine

subroutine update_membership(data,centroid,m,u,weights,verbose)
  use distances
  implicit none
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(inout) :: u(:,:) !NxNC
  real(kind=4), intent(in) :: weights (:,:)
  integer, intent(in) :: verbose
  !locals
  real(kind=8) distance_cache(size(centroid,1),size(data,1)) !NCxN
  real(kind=8) p,sum_tmp,d1,d2
  integer i,j,k,NC,ND,N
  logical flag

  NC = size(centroid,1)
  ND = size(data,2)
  N = size(data,1)
  
  if (verbose > 0) then
      print *, "NC",NC
      print *, "ND",ND
      print *, "Centroid",size(centroid,1),size(centroid,2)
      print *, "data",size(data,1),size(data,2)
      print *, "u",size(u,1),size(u,2)
      print *, "weights",size(weights,1),size(weights,2)
  end if  
  
  p = 2.0/(m-1.0)
  !print *, "p",p
  
  do i=1,N
    do j=1,NC
      distance_cache(j,i) = distance_function(centroid(j,:),data(i,:),weights(j,:))
      if (verbose > 1) then
        print *,i,j,distance_cache(j,i)
        print *,centroid(j,:)
        print *,data(i,:)
        print *,weights(j,:)
      end if
    end do
  end do

  do i=1,N
    do j=1,NC
      d1 = distance_cache(j,i)
      sum_tmp = 0.0
      flag = .true.
      do k=1,NC
        d2 = distance_cache(k,i)
        if (verbose > 1) print *,i,k,j,"d1",d1,"d2",d2
        
        if (d2 == 0.0) then
          flag = .false.
          exit
        else
          sum_tmp = sum_tmp + (d1/d2)**p
        end if
        
      end do
      
      if (flag) then
        u(i,j) = 1.0/sum_tmp
      else
        u(i,:) = 0.0
        u(i,k) = 1.0
        exit
      end if
    end do
  end do

  !do i=1,N
  !  do j=1,NC
  !    if (verbose > 1) print *,i,j,u(i,j)
  !  end do
  !end do

end subroutine

subroutine update_membership_ew(data,centroid,m,u,weights,verbose)
  use distances
  implicit none
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(inout) :: u(:,:) !NxNC
  real(kind=4), intent(in) :: weights (:) !NC
  logical, intent(in) :: verbose
  !locals
  real(kind=8) distance_cache(size(centroid,1),size(data,1)) !NCxN
  real(kind=8) p,sum_tmp,d1,d2
  integer i,j,k,NC,ND,N
  logical flag

  NC = size(centroid,1)
  ND = size(data,2)
  N = size(data,1)
  
  if (verbose) then
      print *, "NC",NC
      print *, "ND",ND
      print *, "Centroid",size(centroid,1),size(centroid,2)
      print *, "data",size(data,1),size(data,2)
      print *, "u",size(u,1),size(u,2)
      print *, "weights",size(weights)
  end if  
  
  p = 2.0/(m-1.0)
  !print *, "p",p
  
  do i=1,N
    do j=1,NC
      distance_cache(j,i) = distance_function(centroid(j,:),data(i,:),weights)
      !print *,i,k,distance_cache(i,k) !,centroid(i,:),data(k,:)
    end do
  end do

  do i=1,N
    do j=1,NC
      d1 = distance_cache(j,i)
      sum_tmp = 0.0
      flag = .true.
      do k=1,NC
        d2 = distance_cache(k,i)
        !print *,i,k,j,"d1",d1,"d2",d2
        
        if (d2 == 0.0) then
          flag = .false.
          exit
        else
          sum_tmp = sum_tmp + (d1/d2)**p
        end if
        
      end do
      
      if (flag) then
        u(i,j) = 1.0/sum_tmp
      else
        u(i,:) = 0.0
        u(i,k) = 1.0
        exit
      end if
    end do
  end do
end subroutine

function compactness(centroid,data,m,u,weights)
  use distances
  implicit none
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(in) :: u(:,:) !NxNC
  real(kind=4), intent(in) :: weights (:)
  !return
  real(kind=8) compactness
  !locals
  real(kind=8) d,inner_num,inner_den
  integer i,k,NC,N

  NC = size(centroid,1)
  N = size(data,1)
  
  compactness = 0.0
  
  do i=1,NC
    inner_num = 0.0
    inner_den = 0.0
    do k=1,N
      d = distance_function(centroid(i,:),data(k,:),weights)
      inner_num = inner_num + d*u(k,i)
      inner_den = inner_den + u(k,i)
    end do

    compactness = compactness + inner_num/inner_den
  end do

end function

function separation(centroid,m,weights)
  use distances
  implicit none
  real(kind=4), intent(in) :: centroid(:,:)!NCxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(in) :: weights (:)
  !return
  real(kind=8) separation
  !locals
  real(kind=8) d,d1,d2,sum,p
  integer i,j,l,NC
  real(kind=4) u(size(centroid,1),size(centroid,1))!NCxNC

  NC = size(centroid,1)

  p = 1.0/(m-1.0)

  !compute u
  do i=1,NC
    do j=1,NC
      if (i /= j) then
        d1 = distance_function(centroid(i,:),centroid(j,:),weights)
        sum = 0.0
        do l=1,NC
          if (j /= l) then
            d2 = distance_function(centroid(j,:),centroid(l,:),weights)
            sum = sum + (d1/d2)**p
          end if
        end do
        
        u(j,i) = 1.0/sum
      end if
    end do
  end do

  
  separation = 0.0
  
  do i=1,NC
    do j=1,NC
      if (i/=j) then
        d = distance_function(centroid(i,:),centroid(j,:),weights)
        separation = separation + d*u(j,i)**m
      end if
    end do

  end do

end function

function compactness_weighted(data,centroid,m,u,weights,regularization,lambda,jm,mean_distance,verbose)
  use distances
  implicit none
  real(kind=4), intent(in)  :: data(:,:) !NxND
  real(kind=4), intent(in)  :: centroid(:,:) !NCxND
  real(kind=4), intent(in)  :: m
  real(kind=4), intent(in)  :: u(:,:) !NxNC
  real(kind=4), intent(in)  :: weights (:,:) !NCxND
  integer, intent(in)       :: regularization
  real(kind=4), intent(in)  :: lambda
  real(kind=4), intent(out) :: jm
  real(kind=4), intent(out) :: mean_distance
  integer, intent(in)       :: verbose
  !return
  real(kind=8) compactness_weighted
  !locals
  real(kind=8) d,inner_cluster,w,constant
  integer i,j,k,NC,N,ND
  real(kind=4) :: weights_cluster (size(data,2)), cent(size(data,2))

  NC = size(centroid,1)
  N = size(data,1)
  ND = size(data,2)
  
  inner_cluster = 0.0
  
  constant = lambda * log(real(ND))
  
  jm = 0.0

  !inner cluster
  do j=1,NC
    weights_cluster = weights(j,:)
    cent(:) = centroid(j,:)

    if (regularization > 0) then
      w = lambda * sum(weights_cluster * log(weights_cluster))
    end if
    do i=1,N
      d = distance_function(cent,data(i,:),weights_cluster)

      if (regularization > 0) then
        d = d+w+constant
      end if

      !inner_cluster = inner_cluster + (d+w+constant)*u(i,j)**m 

      !if (verbose > 2) print *,i,j,d,w,constant,jm
      
      !if (u(i,j) < 0 .or. u(i,j) > 1.0) then
      !  print *,"PROBLEM",i,j,u(i,j)
      !  stop
      !end if

      !if (d > 1e10) then
      !  print *,"PROBLEM",i,j,d
      !  print *,"CENTROID",cent
      !  print *,"DATA",data(i,:)
      !  stop
      !end if
      
      !if (regularization > 0) then
      !  d = d + w + constant
      !end if
      
      jm = jm + (d**2)*(u(i,j)**m)
    end do
  end do

  !if (verbose > 0) print *,"inner_cluster",inner_cluster
  !if (verbose > 0) print *,"jm",jm
  !separation
  mean_distance = 0.0
  do i=1,NC
    do j=1,NC
      if (i /= j) then
        do k=1,ND
          weights_cluster(k) = max(weights(j,k),weights(i,k))
        end do

        d = distance_function(centroid(i,:),centroid(j,:),weights_cluster)

        !if (verbose > 2) print *,i,j,d,mean_distance
        !if (d < min_distance) then
        !  min_distance = d
        !end if
        mean_distance = mean_distance + d
      end if
    end do
  end do
  if (verbose>0) print *,"mean_distance",mean_distance

  
  compactness_weighted = inner_cluster

end function

function compactness_weighted_ew(data,centroid,m,u,weights,regularization,lambda,jm,mean_distance,verbose)
  use distances
  implicit none
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(in) :: u(:,:) !NxNC
  real(kind=4), intent(in) :: weights (:) !ND
  integer, intent(in)       :: regularization
  real(kind=4), intent(in) :: lambda
  real(kind=4), intent(out) :: jm
  real(kind=4), intent(out) :: mean_distance
  logical, intent(in) :: verbose
  !return
  real(kind=8) compactness_weighted_ew,w,constant
  !locals
  real(kind=8) d,inner_cluster,penalty
  integer i,j,NC,N,ND
  real(kind=4) :: cent(size(data,2))

  NC = size(centroid,1)
  N = size(data,1)
  ND = size(data,2)
  
  inner_cluster = 0.0
  
  constant = lambda * log(real(ND))
  
  if (verbose) print *,"regularization",regularization,"lambda",lambda
  
  jm = 0.0

  !inner cluster
  do j=1,NC
    cent(:) = centroid(j,:)
    if (regularization == 1) then !friedman
        w = lambda * sum(weights * log(weights))
    else
        w = 0.0
    end if

    do i=1,N
      d = distance_function(cent,data(i,:),weights)
      inner_cluster = inner_cluster + (d+w+constant)*u(i,j)**m 
      
      jm = jm + ((d+w+constant)**2)*(u(i,j)**m)
    end do
  end do

  if (verbose) print *,"inner_cluster",inner_cluster
  !separation
  mean_distance = 0.0
  do i=1,NC
    do j=1,NC
      if (i /= j) then
        d = distance_function(centroid(i,:),centroid(j,:),weights)
        mean_distance = mean_distance + d
        !if (d < min_distance) then
        !  min_distance = d
        !end if
      end if
    end do
  end do
  if (verbose) print *,"mean_distance",mean_distance

  
  compactness_weighted_ew = inner_cluster

end function

!from https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
function dbi_index(centroid,data,clusters,weights)
  use distances
  implicit none
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: weights (:,:)
  integer(kind=1), intent(in) :: clusters(:)
  !return
  real(kind=4) :: dbi_index
  !locals
  real(kind=8) dist,mean_d
  integer i,j,k,NC,ND,N,elements
  real(kind=8) s(size(centroid,1))
  real(kind=8) r(size(centroid,1),size(centroid,1))
  real(kind=8) d(size(centroid,1))
  real(kind=4) w(size(centroid,2))

  NC = size(centroid,1)
  ND = size(centroid,2)
  N = size(data,1)
  
  !print *,NC,ND,N

  !S(k): mean distance between centroid and all data within the cluster
  do k=1,NC
    mean_d = 0.0
    elements = 0
    do i=1,N
      if (clusters(i) == (k-1)) then !k-1 because clusters start in 0
        dist = distance_function(centroid(k,:),data(i,:),weights(k,:))

        !print *,"dist",k,i,dist

        mean_d = mean_d + dist
        elements = elements + 1
      end if
    end do
    
    s(k) = mean_d/elements
    !print *,"S",k,s(k)
  end do
  !M(k,j):  is a measure of separation between cluster {\displaystyle C_{i}} C_{i} and cluster {\displaystyle C_{j}} C_{j}.
  do k=1,NC
    do j=1,NC
      do i=1,ND
        w(i) = max(weights(k,i),weights(j,i))
      end do
      dist = distance_function(centroid(k,:),centroid(j,:),w)
      !m(k,j) = d
      
      r(k,j) = (s(k) + s(j))/dist
      !print *,"R",k,j,r(k,j),dist
    end do
  end do

  !compute D
  do k=1,NC
    d(k) = -1000
    do j=1,NC
      if (j /= k .and. d(k) < r(k,j)) then
        d(k) = r(k,j)
      end if
    end do
    !print *,"D",k,d(k)
  end do
  
  dbi_index = sum(d)/NC
  
end function

!https://en.wikipedia.org/wiki/Silhouette_(clustering)
function silhouette_index(data,clusters,weights)
  use distances
  implicit none
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: weights (:,:)
  integer(kind=1), intent(in) :: clusters(:)
  !return
  real(kind=4) :: silhouette_index
  !locals
  real(kind=8) dist
  integer i,j,k,l,NC,ND,N,c1,c2
  real(kind=8) s(size(data,1))
  real(kind=8) a(size(data,1))
  real(kind=8) b(size(data,1))
  real(kind=8) bb(size(data,1),size(weights,1))
  integer(kind=4) cs(size(weights,1))
  real(kind=4) w(size(weights,2),size(weights,1),size(weights,1))

  NC = size(weights,1)
  ND = size(weights,2)
  N = size(data,1)

  !counting clusters elements
  cs = 0
  do i=1,N
    c1 = clusters(i)
    cs(c1+1) = cs(c1+1) + 1
  end do

  !weights among clusters
  do k=1,NC
    do j=1,NC
      do i=1,ND
        w(i,j,k) = max(weights(k,i),weights(j,i))
      end do
    end do
  end do

  !a(i): be the average dissimilarity of {\displaystyle i} i with all other data within the same cluster
  bb = 0.0
  a = 0.0
  do i=1,N
    c1 = clusters(i)
    do j=1,N
      c2 = clusters(j)
      if (c2 == c1) then
        dist = distance_function(data(i,:),data(j,:),w(:,c1+1,c2+1))
        a(i) = a(i) + dist
        bb(i,c1+1) = huge(dist)
      else
        dist = distance_function(data(i,:),data(j,:),w(:,c1+1,c2+1))
        bb(i,c2+1) = bb(i,c2+1) + dist
      end if
    end do
    !print *, a(i),bb(i,:)
    a(i) = a(i)/cs(c1+1)
    !normalize b with cluster size
    do k=1,NC
      bb(i,k) = bb(i,k) / cs(k)
    end do
  end do
  b = minval(bb,dim=2)
  
  !compute silluette
  do i=1,N
    s(i) = (b(i) - a(i))/max(a(i),b(i))
    !print *,i,a(i),b(i),s(i)
  end do

  silhouette_index = sum(s)/N
  
end function

subroutine fix_weights(weights,dim_force,weight_force)
  implicit none
  real(kind=4), intent(inout) :: weights (:)
  integer, intent(in)         :: dim_force
  real(kind=4), intent(in)    :: weight_force

  if (dim_force > 0) then
      weights(dim_force) = 0.0
      weights = weights / sum(weights) 
      weights = weights * (1.0-weight_force)
      weights(dim_force) = weight_force
  else
    weights = weights / sum(weights) 
  end if
end subroutine

end module
