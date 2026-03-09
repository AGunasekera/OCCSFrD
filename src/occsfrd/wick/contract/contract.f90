 program test
  implicit none
  integer, parameter :: n=4, ncmax=100
  integer :: list(n,n), nlist(n)
  integer :: clist(n,ncmax)
  integer :: nc
  logical :: sgnflps(ncmax)
  ! logical :: fermsgn(n)

  list = 0
  nlist = 0

  nlist(1) = 3 ; list(1,1)=2 ; list(2,1)=3 ; list(3,1)=4
  nlist(2) = 1 ; list(1,2)=3 !; list(2,2)=4
  nlist(3) = 1 ; list(1,3)=4

  call contract(list,nlist,n,clist,nc,ncmax,sgnflps)

 end program 

 subroutine contract(list,nlist,n,clist,nc,ncmax,sgnflps)
  implicit none
  integer, intent(in) :: n          ! number of elements
  integer, intent(in) :: nlist(n)   ! element i contracts with nlist(i) elements
  integer, intent(in) :: list(n,n)  ! list(1:nlist(i),i) list of elements i contracts with
  integer, intent(in) :: ncmax
  integer :: idx, iseq
  logical :: fermsgn(n), sgnflp(n)

  integer :: seq(2,n), i
  integer, intent(out) :: nc, clist(n,ncmax)
  logical, intent(out) :: sgnflps(ncmax)

  nc = 0
  clist = 0
  iseq = 0
  seq = 0
  sgnflps = .false.
  fermsgn = .false.
  sgnflp = .false.
 
  idx = 1

  do i = 2,n
   fermsgn(i) = .not. fermsgn(i-1)
  enddo
  call cdrive(idx,list,nlist,n,nc,clist,ncmax,seq,iseq,sgnflps,fermsgn,sgnflp)

  ! write(6,*) nc
  ! do i = 1,nc
   ! write(6,'(4i7)') clist(:,i)
  ! enddo

 end subroutine


 recursive subroutine cdrive(idx,list,nlist,n,nc,clist,ncmax,seq,iseq,sgnflps,fermsgn,sgnflp)
  implicit none
  integer :: idx, n, iseq
  integer :: nlist(n), list(n,n), nc, ncmax, clist(n,ncmax), seq(2,n)
  integer :: i, j, jdx, jseq, k
  logical :: lnew, sgnflp(n), sgnflps(ncmax), fermsgn(n)

  if (idx.gt.n) return

  ! write(6,*) 'currently at ',idx,iseq
  ! write(6,'(4i7)') seq(1,:)
  ! write(6,'(4i7)') seq(2,:)

  if (2*iseq==n) then ! full contraction
   ! write(6,*) 'found full contraction'
   nc = nc+1 ! increment full contraction count
   if (nc.gt.ncmax) then 
     write(6,*) 'number of contractions exceeded ',ncmax
     stop 'increase ncmax'
   end if
   ! store it
   do i = 1,iseq
    clist(seq(1,i),nc) = seq(2,i)
    ! clist(seq(2,i),nc) = seq(1,i)
    sgnflps(nc) = sgnflps(nc) .neqv. sgnflp(i)
   enddo
   return
  end if

  ! currently have iseq contractions involving elements up to idx-1

  ! make sure idx not in sequence
  lnew=.true.
  do j = 1,iseq
   if (idx==seq(2,j)) then
    lnew=.false.
    ! sgnflps(nc+1) = sgnflps(nc+1) * (-1)
    ! write(6,*) 'already present in sequence'
    exit
   end if
  enddo 

  if (lnew) then

  ! search over contractions for element idx
  do i = 1,nlist(idx)

   ! write(6,*) 'testing contraction ',idx,list(i,idx)

   ! make sure candidate - list(i,idx) - is not already in sequence
   lnew=.true.
   do j = 1,iseq
    if (list(i,idx)==seq(2,j)) then 
     lnew=.false.
     ! sgnflps(nc+1) = sgnflps(nc+1) * (-1)
     ! write(6,*) 'already present in sequence'
     exit
    end if
   enddo
   if (.not.lnew) cycle

   ! write(6,*) 'adding to the sequence'

   ! add to the sequence
   jseq = iseq+1
   seq(1,jseq) = idx
   seq(2,jseq) = list(i,idx)
   ! write(6,*) 'old sign flips:', sgnflps(nc+1)
   ! sgnflps(nc+1) = sgnflps(nc+1) * ((-1) ** (list(i,idx) - idx))
   do k = seq(1,jseq)+1,seq(2,jseq)-1
    fermsgn(k) = .not. fermsgn(k)
   enddo
   ! write(6,*) idx, fermsgn(idx), list(i,idx), fermsgn(list(i,idx)), sgnflps(nc+1)
   ! write(6,*) 'new sign flips:', sgnflps(nc+1) .neqv. (fermsgn(list(i,idx)) .eqv. fermsgn(idx))
   ! sgnflp = sgnflp .neqv. (fermsgn(list(i,idx)) .eqv. fermsgn(idx))
   sgnflp(jseq) = fermsgn(seq(1,jseq)) .eqv. fermsgn(seq(2,jseq))
   ! write(6,*) 'sign flip', seq(1,jseq), seq(2,jseq), sgnflp(jseq)

   ! move to next idx
   jdx = idx + 1
   call cdrive(jdx,list,nlist,n,nc,clist,ncmax,seq,jseq,sgnflps,fermsgn,sgnflp)

   ! finished searching, remove sequence
   do k = seq(1,jseq)+1,seq(2,jseq)-1
    fermsgn(k) = .not. fermsgn(k)
   enddo
   seq(:,jseq) = 0
   sgnflp(jseq) = .false.

  enddo

  else

   ! move to next idx
   jdx = idx + 1
   jseq = iseq
   call cdrive(jdx,list,nlist,n,nc,clist,ncmax,seq,jseq,sgnflps,fermsgn,sgnflp)

  end if

  ! if (idx==n) write(6,*) 'search path exhausted'

 end subroutine
