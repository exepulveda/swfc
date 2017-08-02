module sfc_opt
  implicit none
  !integer, parameter :: dp = kind(1.d0)
  !integer, parameter :: sp = kind(1.0)

contains

subroutine mutate_centroid(ind,minvalues,maxvalues,randn,randn_index)
  use utils
  implicit none
  !argmunents
  real(kind=4), intent(inout) :: ind(:,:)
  real(kind=4), intent(in)    :: minvalues(:)
  real(kind=4), intent(in)    :: maxvalues(:)
  real(kind=4), intent(in)    :: randn(:)
  integer, intent(inout)      :: randn_index
  !locals
  integer NC,ND,i,j
  real(kind=4) pert

  NC = size(ind,1)
  ND = size(ind,2)
  
  i = randint(1,NC)
  j = randint(1,ND)
  
  pert = randn(randn_index)
  randn_index = randn_index + 1

  print *,i,j,pert,ind(i,j)

  ind(i,j) = ind(i,j) + pert
  if (ind(i,j) < minvalues(j)) then
    ind(i,j) = minvalues(j)
  elseif (ind(i,j) > maxvalues(j)) then
    ind(i,j) = maxvalues(j)
  end if
end subroutine

subroutine crossover_centroid(ind1,ind2)
  use utils
  implicit none
  !argmunents
  real(kind=4), intent(inout) :: ind1(:,:)
  real(kind=4), intent(inout) :: ind2(:,:)
  !locals
  integer N,NC,ND,i,j
  real(kind=4) v,w
  real rnd(size(ind1,1),size(ind1,2))
  
  NC = size(ind1,1)
  ND = size(ind1,2)
  
  do i=1,ND
    do j=1,NC
      if (randu() < 0.5) then
        v = ind1(j,i)
        w = ind2(j,i)
      else
        v = ind2(j,i)
        w = ind1(j,i)
      end if
      ind1(j,i) = v
      ind2(j,i) = w
    end do
  end do
end subroutine

function evaluate_centroid(data,centroid,m,u,weights,lambda,verbose)
  use clustering
  implicit none
  !argmunents
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: m
  real(kind=4), intent(inout) :: u(:,:) !NxNC
  real(kind=4), intent(in)    :: weights (:,:) !NCxND
  real(kind=4), intent(in)    :: lambda
  integer(kind=4), intent(in) :: verbose
  !return
  real(kind=4) evaluate_centroid  
  !locals
  integer regularization
  real(kind=4) jm
  real(kind=4) mean_distance,ret

  call update_membership(data,centroid,m,u,weights,verbose)
  !centroid,data,m,u,weights,verbose
  ret = compactness_weighted(data,centroid,m,u,weights,1,lambda,jm,mean_distance,verbose)
  
  evaluate_centroid = jm + 10.0/mean_distance

end function

function tournament(fitness,k)
  use utils
  implicit none
  !argmunents
  real(kind=4), intent(in) :: fitness(:)
  integer, intent(in)      :: k
  !return
  integer tournament  
  !locals
  real(kind=4) s
  integer npop,i,j,pos(1)
  
  npop = size(fitness)

  j = randint(1,npop)
  s = fitness(j)
  tournament = j
  !print *,1,j,fitness(j),s,tournament
  do i=2,k
    j = randint(1,npop)
    if (fitness(j) < s) then
      s = fitness(j)
      tournament = j
    end if
    
    !print *,i,j,fitness(j),s,tournament
  end do
  
end function

function optimize_centroids(data, min_values, max_values, &
    current_centroids,m,weights,lambda,npop,ngen,cxpb,mutpb,max_no_improvement,verbose,randn,&
    best_u)
  use utils
  use clustering
  implicit none
  real(kind=4), intent(in)    :: data(:,:) !NxND
  real(kind=4), intent(inout) :: current_centroids (:,:) !NCxND
  real(kind=4), intent(in)    :: m
  real(kind=4), intent(in)    :: weights (:,:) !NCxND
  real(kind=4), intent(in)    :: lambda
  integer, intent(in)         :: npop,ngen,max_no_improvement
  real(kind=4), intent(in)    :: cxpb,mutpb
  integer, intent(in)         :: verbose
  real(kind=4), intent(in)    :: randn(:)
  real(kind=4), intent(in)    :: min_values(:)
  real(kind=4), intent(in)    :: max_values(:)
  real(kind=4), intent(inout) :: best_u(:,:) !NxNC
  !return
  real(kind=4) optimize_centroids
  !locals
  real(kind=4) population(npop,size(weights,1),size(weights,2))
  real(kind=4) matingpool(npop,size(weights,1),size(weights,2))
  real(kind=4) fitness(npop), offspring_fitness(npop)
  real(kind=4) :: u(size(data,1),size(weights,2)) !NxNC
  !best
  real(kind=4) best_individual (size(weights,1),size(weights,2))
  real(kind=4) best_fitness,prev_fitness
  integer NC,N,ND,gen,i,j,k,no_improvement,pos(1),randn_index
  logical changed(npop)
  real rnd
  
  NC = size(weights,1)
  ND = size(weights,2)
  N = size(data,1)
  
  !initialize population
  population(1,:,:) = current_centroids
  do i=2,npop
    !select at random an element from data
    do j=1,NC
      k = randint(1,N)
      population(i,j,:) = data(k,:)
    end do
  end do
  !evaluate
  print *,"init evaluation"
  do i=1,npop
    fitness(i) = evaluate_centroid(data,population(i,:,:),m,u,weights,lambda,verbose)
    print *,"init",i,fitness(i)
  end do
  
  !update best
  pos = minloc(fitness)
  best_individual = population(pos(1),:,:)
  best_fitness = fitness(pos(1))
  
  no_improvement = 0
  randn_index = 1
  print *,0,best_fitness,minval(fitness),maxval(fitness)
  !
  do gen=1,ngen
    prev_fitness = best_fitness

    !create a mating pool by tournament selection
    do i=1,npop
      j = tournament(fitness,5)
      matingpool(i,:,:) = population(j,:,:)
      offspring_fitness(i) = fitness(j)
    end do
    
    changed = .false.
    
    !crossover mating pool
    do i=1,npop,2
      rnd = randu()
      if (rnd < cxpb) then
        !print *,'crossover_centroid',i,i+1
        call crossover_centroid(matingpool(i,:,:),matingpool(i+1,:,:))
        changed(i) = .true.
        changed(i+1) = .true.
      end if
    end do
    
    !mutate mating pool
    do i=1,npop
      rnd = randu()
      !print *,i,rnd,mutpb
      if (rnd < mutpb) then
        !print *,i,matingpool(i,:,:)
        call mutate_centroid(matingpool(i,:,:),min_values,max_values,randn,randn_index)
        !print *,i,matingpool(i,:,:)
        changed(i) = .true.
      end if
    end do
    
    !evaluate
    do i=1,npop
      if (changed(i)) then
        offspring_fitness(i) = evaluate_centroid(data,matingpool(i,:,:),m,u,weights,lambda,verbose)
        !print *,"offspring_fitness",gen,i,offspring_fitness(i)
      end if
    end do
    
    do i=1,npop
      print *,i,fitness(i),offspring_fitness(i)
    end do
    
    !replace population
    population = matingpool
    fitness = offspring_fitness
    
    !update best
    pos = minloc(fitness)
    if (fitness(pos(1)) < best_fitness) then
      best_individual = population(pos(1),:,:)
      best_fitness = fitness(pos(1))
    else
      !replace the worse individual with the best
      pos = maxloc(fitness)
      population(pos(1),:,:) = best_individual
      fitness(pos(1)) = best_fitness
    end if

    print *,"optimize_centroids",gen,best_fitness,minval(fitness),maxval(fitness)
    
    !check if not improvements
    if (best_fitness >= prev_fitness) then
      no_improvement = no_improvement + 1
      if (no_improvement > max_no_improvement) exit
    else
      no_improvement = 0
    end if
  end do
  
  current_centroids = best_individual
  call update_membership(current_centroids,data,m,best_u,weights,verbose)
  optimize_centroids = best_fitness
end function

!!!!!! weights
subroutine mutate_weights(ind,dim_force,weight_force,randn,randn_index)
  use utils
  use clustering
  implicit none
  !argmunents
  real(kind=4), intent(inout) :: ind(:,:)
  integer, intent(in)         :: dim_force
  real(kind=4), intent(in)    :: weight_force
  real(kind=4), intent(in)    :: randn(:)
  integer, intent(inout)      :: randn_index
  !locals
  integer NC,ND,i,j
  real(kind=4) pert

  NC = size(ind,1)
  ND = size(ind,2)
  
  i = randint(1,NC)
  j = randint(1,ND)
  
  pert = randn(randn_index)
  randn_index = randn_index + 1

  !print *,"ind(i,j)",i,j,ind(i,j),pert,minvalues(j),maxvalues(j)

  ind(i,j) = ind(i,j) + pert
  
  call fix_weights(ind(i,:),dim_force,weight_force)
end subroutine

subroutine crossover_weights(ind1,ind2,dim_force,weight_force)
  use clustering
  use utils
  implicit none
  !argmunents
  real(kind=4), intent(inout) :: ind1(:,:)
  real(kind=4), intent(inout) :: ind2(:,:)
  integer, intent(in)         :: dim_force
  real(kind=4), intent(in)    :: weight_force
  !locals
  integer NC,ND,i,j
  real(kind=4) v,w
  real rnd(size(ind1,1),size(ind1,2))
  
  NC = size(ind1,1)
  ND = size(ind1,2)
  
  do i=1,ND
    do j=1,NC
      if (randu() < 0.5) then
        v = ind1(j,i)
        w = ind2(j,i)
      else
        v = ind2(j,i)
        w = ind1(j,i)
      end if
      ind1(j,i) = v
      ind2(j,i) = w
    end do
  end do
  
  do i=1,NC
    call fix_weights(ind1(i,:),dim_force,weight_force)
    call fix_weights(ind2(i,:),dim_force,weight_force)
  end do
  
end subroutine

function evaluate_weights(data,centroid,m,u,weights,lambda,verbose)
  use clustering
  implicit none
  !argmunents
  real(kind=4), intent(in) :: centroid(:,:) !NCxND
  real(kind=4), intent(in) :: data(:,:) !NxND
  real(kind=4), intent(in) :: m, lambda
  real(kind=4), intent(inout) :: u(:,:) !NxNC
  real(kind=4), intent(in) :: weights (:,:) !NCxND
  integer(kind=4), intent(in) :: verbose
  !return
  real(kind=4) evaluate_weights  
  !locals
  integer regularization
  real(kind=4) jm
  real(kind=4) mean_distance,ret

  call update_membership(centroid,data,m,u,weights,verbose)
  ret = compactness_weighted(centroid,data,m,u,weights,1,lambda,jm,mean_distance,verbose)
  
  evaluate_weights = jm + 10.0/mean_distance

end function

function optimize_weights(data, lambda, dim_force,weight_force, &
    centroids,m,weights,npop,ngen,cxpb,mutpb,max_no_improvement,verbose,randn,&
    best_u)
  use utils
  use clustering
  implicit none
  real(kind=4), intent(in)    :: data(:,:) !NxND
  real(kind=4), intent(in)    :: lambda
  integer, intent(in)         :: dim_force
  real(kind=4), intent(in)    :: weight_force
  real(kind=4), intent(in)    :: centroids (:,:) !NCxND
  real(kind=4), intent(in)    :: m
  real(kind=4), intent(inout) :: weights (:,:) !NCxND
  integer, intent(in)         :: npop,ngen,max_no_improvement
  real(kind=4), intent(in)    :: cxpb,mutpb
  integer, intent(in)         :: verbose
  real(kind=4), intent(in)    :: randn(:)
  real(kind=4), intent(inout) :: best_u(:,:) !NxNC
  !return
  real(kind=4) optimize_weights
  !locals
  real(kind=4) population(npop,size(weights,1),size(weights,2))
  real(kind=4) matingpool(npop,size(weights,1),size(weights,2))
  real(kind=4) fitness(npop), offspring_fitness(npop)
  real(kind=4) :: u(size(data,1),size(weights,2)) !NxNC
  !best
  real(kind=4) best_individual (size(weights,1),size(weights,2))
  real(kind=4) best_fitness,prev_fitness
  integer NC,N,ND,gen,i,j,k,no_improvement,pos(1),randn_index
  logical changed(npop)
  real rnd(npop)
  
  NC = size(weights,1)
  ND = size(weights,2)
  N = size(data,1)
  
  !initialize population
  population(1,:,:) = weights
  do i=2,npop
    !select at random an element from data
    do j=1,NC
      do k=1,ND
        population(i,j,k) = randu()
      end do
      call fix_weights(population(i,j,:),dim_force,weight_force)
    end do
  end do
  !evaluate
  do i=1,npop
    fitness(i) = evaluate_weights(data,centroids,m,u,population(i,:,:),lambda,verbose)
    !print *,"init",i,fitness(i)
  end do
  !update best
  pos = minloc(fitness)
  best_individual = population(pos(1),:,:)
  best_fitness = fitness(pos(1))
  
  no_improvement = 0
  randn_index = 1
  print *,0,best_fitness,minval(fitness),maxval(fitness)
  !
  do gen=1,ngen
    prev_fitness = best_fitness

    !create a mating pool by tournament selection
    do i=1,npop
      j = tournament(fitness,5)
      matingpool(i,:,:) = population(j,:,:)
      offspring_fitness(i) = fitness(j)
    end do
    
    changed = .false.
    
    !crossover mating pool
    do i=1,npop,2
      if (randu() < cxpb) then
        call crossover_weights(matingpool(i,:,:),matingpool(i+1,:,:),dim_force,weight_force)
        changed(i) = .true.
        changed(i+1) = .true.
      end if
    end do
    
    !mutate mating pool
    do i=1,npop
      if (randu() < mutpb) then
        call mutate_weights(matingpool(i,:,:),dim_force,weight_force,randn,randn_index)
        changed(i) = .true.
      end if
    end do
    
    !evaluate
    do i=1,npop
      if (changed(i)) then
        offspring_fitness(i) = evaluate_weights(data,centroids,m,u,matingpool(i,:,:),lambda,verbose)
        !print *,"offspring_fitness",gen,i,offspring_fitness(i)
      end if
    end do
    !replace population
    population = matingpool
    fitness = offspring_fitness
    
    !update best
    pos = minloc(fitness)
    if (fitness(pos(1)) < best_fitness) then
      best_individual = population(pos(1),:,:)
      best_fitness = fitness(pos(1))
    else
      !replace the worse individual with the best
      pos = maxloc(fitness)
      population(pos(1),:,:) = best_individual
      fitness(pos(1)) = best_fitness
    end if

    print *,"optimize_weights",gen,best_fitness,minval(fitness),maxval(fitness)
    
    !check if not improvements
    if (best_fitness >= prev_fitness) then
      no_improvement = no_improvement + 1
      if (no_improvement > max_no_improvement) exit
    else
      no_improvement = 0
    end if
  end do
  
  weights = best_individual
  call update_membership(centroids,data,m,best_u,weights,verbose)
  optimize_weights = best_fitness
end function

end module
