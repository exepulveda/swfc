module utils
  implicit none
  
contains

subroutine set_seed(seed)
  implicit none
  integer, intent(in) :: seed
  
  call srand(seed)
end subroutine

function randint(a,b)
  implicit none
  integer, intent(in) :: a,b
  real randint
  
  real rnd
  
  !call RANDOM_NUMBER(rnd)
  rnd = rand()

  randint = int(rnd*(b+1-a))+a

end function

function randu()
  implicit none
  real randu
  
  real rnd
  
  !call RANDOM_NUMBER(rnd)
  rnd = rand()

  randu = rnd

end function


end module
