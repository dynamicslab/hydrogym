!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
!!
!!   Author: Prabal Negi
!!   Implementation of Relaxation Term (RT) filtering
!!
!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
!!----------------------------------------------------------------------  
!     read parameters relaxation term filtering 
      subroutine rtfil_param_in(fid)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            !
cc MA:      include 'PARALLEL_DEF' 
      include 'PARALLEL'        ! ISIZE, WDSIZE, LSIZE,CSIZE
      include 'RTFILTER'

!     argument list
      integer fid               ! file id

!     local variables
      integer ierr

!     namelists
      namelist /RTFILTER/ rt_kai,rt_kut,rt_wght,rt_ifboyd


!     default values
! ! !       rt_kut        = 1            ! total number of modes 
! ! !       rt_kai        = 100.
! ! !       rt_wght       = 0.1
! ! !       rt_ifboyd     =.true. 

!     read the file
      ierr=0
! ! !       if (NID.eq.0) then
! ! !          read(unit=fid,nml=RTFILTER,iostat=ierr)
! ! !       endif
! ! !       call err_chk(ierr,'Error reading RTFILTER parameters.$')

!     broadcast data
! ! !       call bcast(rt_kut,      ISIZE)
! ! !       call bcast(rt_kai,      WDSIZE)
! ! !       call bcast(rt_wght,     WDSIZE)
! ! !       call bcast(rt_ifboyd,   LSIZE)

      return
      end
!-----------------------------------------------------------------------
!     write parameters relaxation term filtering 
      subroutine rtfil_param_out(fid)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            !
      include 'RTFILTER'

!     argument list
      integer fid               ! file id

!     local variables
      integer ierr

!     namelists
      namelist /RTFILTER/ rt_kai,rt_kut,rt_wght,rt_ifboyd

!     read the file
      ierr=0
      if (NID.eq.0) then
         write(unit=fid,nml=RTFILTER,iostat=ierr)
      endif
      call err_chk(ierr,'Error writing RTFILTER parameters.$')

      return
      end
!-----------------------------------------------------------------------
!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
!    Main interface for RT filter
      subroutine make_RTF

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
      include 'RTFILTER'

      integer nxyz
      parameter (nxyz=lx1*ly1*lz1)
      integer n
      parameter (n=nxyz*lelt)

      integer lm,lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)
      real f_filter(lm2)
      
      integer newfilter          ! 0 => same shape in 2D/3D as the NEK filter
                                 ! 1 => filter funtion applied directly to 3D field  
      real op_mat(lx1,lx1)
      save op_mat

      integer icalld
      save icalld
      data icalld /0/
!------------------------------ 

      if (rt_wght.eq.0) then
           call rzero(rtfx,n)
           call rzero(rtfy,n)
           if (if3d) call rzero(rtfz,n)
           return
      endif

      if (icalld.eq.0) then
!    Create the filter
!           call spectrnsfm(f_filter,rt_kut,rt_wght)
           newfilter = 0           ! 0 => nek formulation           
           call make_fil_new(f_filter,rt_kut,rt_wght,newfilter)

           call build_RT_MAT(op_mat,f_filter,rt_ifboyd)
           icalld=icalld+1
      endif

      call opcopy(rtfx,rtfy,rtfz,vx,vy,vz)

      call build_RT(rtfx,op_mat,nx1,nz1,if3d,newfilter)
      call build_RT(rtfy,op_mat,nx1,nz1,if3d,newfilter)
      if (if3d) call build_RT(rtfz,op_mat,nx1,nz1,if3d,newfilter)

      call cmult(rtfx,rt_kai,n)
      call cmult(rtfy,rt_kai,n)
      if (if3d) call cmult(rtfz,rt_kai,n)

      return
      end subroutine make_RTF

c-----------------------------------------------------------------------
      subroutine spectrnsfm(f_filter,kut,wght)
!     Not in use.
!     Test before using.
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'

      real intv(lx1*lx1)
      real diag(lx1*lx1)

      real f_filter(lx1*lx1)

      real amp,wght
      integer kut
      integer k,k0,kk

!     Set up transfer function

      call ident (diag,lx1)
!      kut  = param(105)+1
!      wght = param(106)
      k0 = lx1-kut
      do k=k0+1,lx1
         kk = k+lx1*(k-1)
         amp = wght*(k-k0)*(k-k0)/(kut*kut)   ! quadratic growth
         diag(kk) = 1.-amp
      enddo
 
      call build_filter_inv(intv,diag,lx1)
 
      call mxm(intv,lx1,diag,lx1,f_filter,lx1)
 
      do k=1,lx1*lx1
         intv(k) = 1.-f_filter(k)
      enddo
      k0 = lx1+1
      if (nio.eq.0) then
         write(6,6) 'INP G:',(diag(k),k=1,lx1*lx1,k0)
         write(6,6) 'RT trn:',(intv(k),k=1,lx1*lx1,k0)
         write(6,6) 'QnG trn:',(f_filter(k),k=1,lx1*lx1,k0)
   6   format(a8,16f9.6,6(/,8x,16f9.6))
      endif

!    testing_prabal
!      if (nio.eq.0) then
!      do k=1,lx1
!          write(6,8) (f_filter(k0),k0=(k-1)*lx1+1,k*lx1)
!   8   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      endif

      return
      end subroutine spectrnsfm

c---------------------------------------------------------------------- 

      subroutine build_filter_inv(intvv,intv,lx1)

      implicit none

      integer trunc
      parameter (trunc = 5)

      integer lx1
      real intv(lx1,lx1)
      real intvv(lx1,lx1)
      real wk1(lx1,lx1)
      real wk2(lx1,lx1)
      real wk3(lx1,lx1)
      integer ii,jj,lt

      lt = lx1*lx1

      call ident(intvv,lx1)
      call sub2(intvv,intv,lt)
      call rzero(wk1,lt)
     
      do ii=0,trunc

      call ident(wk2,lx1)

      do jj=1,ii
          call mxm(wk2,lx1,intvv,lx1,wk3,lx1)
          call copy(wk2,wk3,lt)
      enddo

      call add2(wk1,wk2,lt)

      enddo

      call copy(intvv,wk1,lt)

      return
      end subroutine build_filter_inv

c---------------------------------------------------------------------- 

      subroutine build_RT_MAT(op_mat,f_filter,ifboyd)

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'

      logical IFBOYD 
      integer n
      parameter (n=lx1*lx1)
      integer lm, lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)

      real f_filter(lm2)
      real op_mat(lx1,lx1)

      real ref_xmap(lm2)
      real wk_xmap(lm2)

      real wk1(lm2),wk2(lm2)
      real indr(lm),ipiv(lm),indc(lm)

      real rmult(lm)
      integer ierr

      integer i,j

!    testing_prabal
!      if (nio.eq.0) then
!      write(6,*) 'FIL:'
!      do i=1,lx1
!          write(6,8) (f_filter(j),j=(i-1)*lx1+1,i*lx1)
!      enddo
!      endif

      call spec_coeff_init(ref_xmap,ifboyd)
      
      call copy(wk_XMAP,REF_XMAP,lm2)
      call copy(wk1,wk_XMAP,lm2)

      call gaujordf  (wk1,lx1,lx1,indr,indc,ipiv,ierr,rmult)  !xmap inverse


      call mxm  (f_filter,lx1,wk1,lx1,wk2,lx1)        !          -1
      call mxm  (wk_XMAP,lx1,wk2,lx1,op_mat,lx1)      !     V D V

!    testing_prabal
!      if (nio.eq.0) then
!      write(6,*) 'OP_MAT'
!      do i=1,lx1
!          write(6,9) (op_mat(i,j),j=1,lx1)
!   9   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      endif

      return
      end subroutine build_RT_MAT

c---------------------------------------------------------------------- 

      subroutine build_RT(v,f,nx,nz,if3d,newfilter)
c
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'

      integer nxyz 
      parameter (nxyz=lx1*ly1*lz1)

      real v(nxyz,lelt),w1(nxyz*lelt),w2(nxyz*lelt)
      logical if3d
c
      integer nx,nz
      integer newfilter

      real f(nx,nx),ft(nx,nx)
      real id(nx,nx)
c
      integer e,i,j,k
c
      call transpose(ft,nx,f,nx)
      call ident(id,nx)
c
      if (if3d) then
         do e=1,nelv
c           Filter
            call copy(w2,v(1,e),nxyz)
            call mxm(f,nx,w2,nx,w1,nx*nx)
            i=1
            j=1
            do k=1,nx
               call mxm(w1(i),nx,ft,nx,w2(j),nx)
               i = i+nx*nx
               j = j+nx*nx
            enddo
            call mxm (w2,nx*nx,ft,nx,w1,nx)

            if (newfilter.eq.1) then
                 call copy(v(1,e),w1,nxyz)
            else 
                 call sub3(w2,v(1,e),w1,nxyz)
                 call copy(v(1,e),w2,nxyz)
            endif

         enddo
c
      else
         do e=1,nelv
c           Filter
            call copy(w1,v(1,e),nxyz)
            call mxm(f ,nx,w1,nx,w2,nx)
            call mxm(w2,nx,ft,nx,w1,nx)

            if (newfilter.eq.1) then
                 call copy(v(1,e),w1,nxyz)
            else 
                 call sub3(w2,v(1,e),w1,nxyz)
                 call copy(v(1,e),w2,nxyz)
            endif

         enddo
      endif
c
      return
      end subroutine build_RT


c---------------------------------------------------------------------- 

      subroutine make_fil_new(diag,kut,wght,newfilter)

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'

      real diag(lx1*lx1)
      real intv(lx1*lx1)
      integer nx,k0,kut,kk,k

      integer newfilter
      real amp,wght

c     Set up transfer function
c
      nx = lx1
      call ident   (diag,nx)
      call rzero   (intv,nx*nx) 
c
!      kut  = param(105)+1
!      wght = param(106)

      k0 = nx-kut
      do k=k0+1,nx
         kk = k+nx*(k-1)
         amp = wght*(k-k0)*(k-k0)/(kut*kut)   ! quadratic growth
         diag(kk) = 1.-amp
         intv(kk) = amp          
      enddo

!      do k=1,lx1*lx1
!         intv(k) = 1.-diag(k)
!         intv(k) = diag(k)
!      enddo

      k0 = lx1+1
      if (nio.eq.0) then
         write(6,6) 'RT trn:',(intv(k),k=1,lx1*lx1,k0)
         write(6,6) 'QnG trn:',(diag(k),k=1,lx1*lx1,k0)
   6   format(a8,16f9.6,6(/,8x,16f9.6))
      endif

      if (newfilter.eq.1) then
          call copy(diag,intv,lx1,lx1)
      endif

!    testing_prabal
!      if (nio.eq.0) then
!      write(6,*) 'intv:'
!      do k=1,lx1
!          write(6,8) (intv(k0),k0=(k-1)*lx1+1,k*lx1)
!   8   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      write(6,*) 'Diag:'
!      do k=1,lx1
!          write(6,8) (diag(k0),k0=(k-1)*lx1+1,k*lx1)
!      enddo
!      endif

      return
      end subroutine make_fil_new

c---------------------------------------------------------------------- 

      subroutine spec_coeff_init(ref_xmap,ifboyd)
!     Initialise spectral coefficient mapping
!     Modified by Prabal for relaxation term implementation.

      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'WZ_DEF'
      include 'WZ'

      integer lm, lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)

!     local variables
      integer i, j, k, n, nx, kj
!     Legendre polynomial
      real plegx(lm)
      real z
      real REF_XMAP(lm2)
      real pht(lm2)

!     Change of basis
      logical IFBOYD
!-----------------------------------------------------------------------

!     initialise arrays
!     X - direction
!      n = NX1-1
!      do j= 1, NX1
!!     Legendre polynomial
!        z = ZGM1(j,1)
!        call legendre_poly(plegx,z,n)
!        do i=1, NX1
!            REF_XMAP(j,i) = plegx(i) !*WXM1(j)     ! already transposed here
!        enddo
!      enddo
      nx = LX1
      kj = 0
      n  = nx-1
      do j=1,nx
         z = ZGM1(j,1)
         call legendre_poly(plegx,z,n)
         kj = kj+1
         pht(kj) = plegx(1)
         kj = kj+1
         pht(kj) = plegx(2)

         if (IFBOYD) then        ! change basis to conserve boundary values
              do k=3,nx
                 kj = kj+1
                 pht(kj) = plegx(k)-plegx(k-2)
              enddo
         else
              do k=3,nx
                 kj = kj+1
                 pht(kj) = plegx(k)
              enddo         
         endif
      enddo

      call transpose (ref_xmap,nx,pht,nx)

!    testing_prabal
!      if (nio.eq.0) then
!      write(6,*) 'REF_XMAP'
!      do i=1,lx1
!          write(6,8) (REF_XMAP(j),j=(i-1)*lx1+1,i*lx1)
!   8   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      endif

!    testing_prabal
!      if (nio.eq.0) then
!      do i=1,lx1
!          write(6,8) (REF_XMAP(i,j),j=1,lx1)
!   8   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      endif

!      if (nio.eq.0) then
!      write(6,*) 'EEST_XMAP'
!      do i=1,lx1
!          write(6,9) (EEST_XMAP(i,j),j=1,lx1)
!   9   format(16f10.6,6(/,8x,16f10.6))
!      enddo
!      endif

      return
      end subroutine spec_coeff_init

!=======================================================================
c

