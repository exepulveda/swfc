module distances
  implicit none

  public 
  real(kind=4), allocatable :: sk (:)
  
  type target_info
    real(kind=4), allocatable :: targets(:)
  end type

  type categorical_info
    integer :: number_categories
    real(kind=4), allocatable :: distance_matrix(:,:)
  end type
  
  integer, allocatable :: variable_types(:,:)
  !1: continue, 2: targeted continue, 3: categorical

  type(target_info), allocatable :: targeted_types(:)
  type(categorical_info), allocatable :: categorical_types(:)

  private sk
  private targeted_types
  private categorical_types
  private variable_types
  private target_info
  private categorical_info
contains

subroutine sk_setup(sk_values)
  real(kind=4), intent(in) :: sk_values (:)
  integer i,N
  
  N = size(sk_values)
  
  allocate(sk(N))

  do i=1,N
    sk(i) = sk_values(i)
    !print *,'sk',i,sk_values(i)
  end do
  
end subroutine

subroutine reset()
  !
  integer n,i
  n = size(categorical_types)
  do i=1,n
    deallocate(categorical_types(i)%distance_matrix)
  end do
  n = size(targeted_types)
  do i=1,n
    deallocate(targeted_types(i)%targets)
  end do

  deallocate(sk)
  deallocate(targeted_types)
  deallocate(categorical_types)
  deallocate(variable_types)
end subroutine


subroutine set_variables(variable_type,verbose)
  integer, intent(in) :: variable_type(:)
  logical, intent(in) :: verbose

  !locals
  integer i,ntargeted,ncategorical
  !
  allocate(variable_types(size(variable_type),2))
  
  ntargeted = 0
  ncategorical = 0
  do i=1,size(variable_type)
    variable_types(i,1) = variable_type(i)
    variable_types(i,2) = 0
    if (verbose) then
      print *,"set_variables:",i,"variable_types",variable_types(i,1)
    end if
    if (variable_type(i) == 2) then
      ntargeted = ntargeted + 1
      variable_types(i,2) = ntargeted
      if (verbose) then
        print *,"ntargeted:",ntargeted
      end if
    else if (variable_type(i) == 3) then
      ncategorical = ncategorical + 1
      variable_types(i,2) = ncategorical
      if (verbose) then
        print *,"ncategorical:",ncategorical
      end if
    end if
    
  end do
  !allocate targetetd structure
  allocate(targeted_types(ntargeted))
  !allocate categorical structure
  allocate(categorical_types(ncategorical))
end subroutine

subroutine set_categorical(index, number_categories,distances_cat)
  integer, intent(in) :: index
  integer, intent(in) :: number_categories
  real(kind=4), intent(in) :: distances_cat(:,:)

  allocate(categorical_types(variable_types(index,2))%distance_matrix(number_categories,number_categories))

  categorical_types(variable_types(index,2))%number_categories = number_categories
  categorical_types(variable_types(index,2))%distance_matrix(:,:) = distances_cat(:,:)
end subroutine

subroutine set_targeted(index, targets,verbose)
  integer, intent(in) :: index
  real(kind=4), intent(in) :: targets(:)
  logical, intent(in) :: verbose

  if (verbose) then
      print *, "set_targeted","index",index,"variable_types",variable_types(index,1), variable_types(index,2)
  end if

  allocate(targeted_types(variable_types(index,2))%targets(size(targets)))
  targeted_types(variable_types(index,2))%targets(:) = targets(:)

  if (verbose) then
      print *, "set_targeted",targeted_types(variable_types(index,2))%targets(:)
  end if
end subroutine


pure function distance_cat(x,y,distance_matrix)
  implicit none
  real(kind=4), intent(in) :: x
  real(kind=4), intent(in) :: y
  real(kind=4), intent(in) :: distance_matrix(:,:)
  
  real(kind=4) distance_cat
  distance_cat = distance_matrix(floor(x)+1,floor(y)+1)
end function

pure function distance_targeted(x,y,targets)
  implicit none
  real(kind=4), intent(in) :: x
  real(kind=4), intent(in) :: y
  real(kind=4), intent(in) :: targets (:)
  real(kind=4) distance_targeted
  !locals
  integer i,N
  real(kind=4) dt(size(targets))
  
  N = size(targets)

  do i=1,N
    dt(i) = abs(x-targets(i)) + abs(y-targets(i))
  end do
  
  distance_targeted = minval(dt)
end function

pure function distance_abs(x,y)
  implicit none
  real(kind=4), intent(in) :: x
  real(kind=4), intent(in) :: y
  real(kind=4) distance_abs
  !locals
  distance_abs = abs(x-y)
end function

pure function distance_function_clustering(x,y,weights)
  implicit none
  real(kind=4), intent(in) :: x(:)
  real(kind=4), intent(in) :: y(:)
  real(kind=4), intent(in) :: weights (:)
  real(kind=8) distance_function_clustering
  !locals
  integer i,N,var_type,var_type_index
  real(kind=8) s,d
  
  N = size(x)

  !print *,"distance_function_clustering","N",N

  s = 0.0
  do i=1,N
    var_type = variable_types(i,1)
    var_type_index = variable_types(i,2)

    !print *,"i",i,"var_type",var_type,"var_type_index",var_type_index
    !print *,"x,y",x(i), y(i), weights(i) / sk(i)

    if (var_type == 1) then
      !print *,"abs"
      d = abs(x(i)-y(i))
    else if (var_type == 2) then
      !print *,"distance_targeted"
      d = distance_targeted(x(i), y(i), targeted_types(var_type_index)%targets)
    else if (var_type == 3) then
      !print *,"distance_cat"
      d = distance_cat(x(i), y(i), categorical_types(var_type_index)%distance_matrix)
    end if
    s = s + d * weights(i) / sk(i)
  end do
  
  distance_function_clustering = s
  
end function

subroutine distance_function_dim(x,y,ret)
  implicit none
  real(kind=4), intent(in) :: x(:)
  real(kind=4), intent(in) :: y(:)
  real(kind=8), intent(inout) :: ret(:)
  !locals
  integer i,N,var_type,var_type_index
  real(kind=8) s,d
  
  N = size(x)

  !print *,"distance_function_clustering","N",N

  ret = 0.0
  do i=1,N
    var_type = variable_types(i,1)
    var_type_index = variable_types(i,2)

    if (var_type == 1) then
      !print *,"abs"
      d = abs(x(i)-y(i))
    else if (var_type == 2) then
      !print *,"distance_targeted"
      d = distance_targeted(x(i), y(i), targeted_types(var_type_index)%targets)
    else if (var_type == 3) then
      !print *,"distance_cat"
      d = distance_cat(x(i), y(i), categorical_types(var_type_index)%distance_matrix)
    end if
    
    ret(i) = d
  end do
  
end subroutine
    
function distance_function_squared(x,y,weights)
  implicit none
  real(kind=4), intent(in) :: x(:)
  real(kind=4), intent(in) :: y(:)
  real(kind=4), intent(in) :: weights (:)
  real(kind=8) distance_function_squared
  
  
  integer i,N
  real(kind=8) s
  
  N = size(x)
  
  s = 0.0
  do i=1,N
    s = s + (x(i) - y(i))**2
  end do
  
  distance_function_squared = sqrt(s)
  
end function

function distance_function(x,y,weights)
  implicit none
  real(kind=4), intent(in) :: x(:)
  real(kind=4), intent(in) :: y(:)
  real(kind=4), intent(in) :: weights (:)
  real(kind=8) distance_function

  distance_function = distance_function_clustering(x,y,weights)
end function


end module 
