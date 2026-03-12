!> @file math_tools.f
!! @ingroup math
!! @brief Set of math related tools for KTH modules
!! @author Adam Peplinski
!! @date Jan 31, 2017
!=======================================================================
!> @brief Step function
!! @ingroup math
!! @details Continuous step function:
!!  \f{eqnarray*}{
!!    stepf(x) = \left\{ \begin{array}{ll}
!!  0 &\mbox{ if $x \leq x_{min}$} \\
!!  \left(1+e^{\left((x-1)^{-1} + x^{-1}\right)}\right)^{-1} &\mbox{ if $x \leq x_{max}$} \\
!!  1 &\mbox{ if $x >  x_{max}$}
!!       \end{array} \right.
!!  \f}
!!  with \f$ x_{min} = 0.02\f$ and \f$ x_{max}=0.98\f$
!! @param[in] x       function argument
!! @return math_stepf
      real function math_stepf(x)
      implicit none

      ! argument list
      real x

      ! local variables
      real xdmin, xdmax
      parameter (xdmin = 0.001, xdmax = 0.999)
!-----------------------------------------------------------------------
      ! get function vale
      if (x.le.xdmin) then
         math_stepf = 0.0
      else if (x.le.xdmax) then
         math_stepf = 1./( 1. + exp(1./(x - 1.) + 1./x) )
      else
         math_stepf = 1.
      end if

      return
      end function math_stepf
!=======================================================================
!> @brief Give random distribution depending on position
!! @ingroup math
!! @details The original Nek5000 rundom number generator is implementted
!!  in @ref ran1. This totally ad-hoc random number generator below
!!  could be preferable to the origina one for the simple reason that it
!!  gives the same initial cindition independent of the number of
!!  processors, which is important for code verification.
!! @param[in] ix,iy,iz     GLL point index
!! @param[in] ieg          global element number
!! @param[in] xl           physical point coordinates
!! @param[in] fcoeff       function coefficients
!! @return  random distribution
      real function math_ran_dst(ix,iy,iz,ieg,xl,fcoeff)
      implicit none

      include 'SIZE'
      include 'INPUT'       ! IF3D

      ! argument list
      integer ix,iy,iz,ieg
      real xl(LDIM)
      real fcoeff(3)
!-----------------------------------------------------------------------
      math_ran_dst = fcoeff(1)*(ieg+xl(1)*sin(xl(2))) +
     $     fcoeff(2)*ix*iy + fcoeff(3)*ix
      if (IF3D) math_ran_dst =
     $     fcoeff(1)*(ieg +xl(NDIM)*sin(math_ran_dst)) +
     $     fcoeff(2)*iz*ix + fcoeff(3)*iz
      math_ran_dst = 1.e3*sin(math_ran_dst)
      math_ran_dst = 1.e3*sin(math_ran_dst)
      math_ran_dst = cos(math_ran_dst)

      return
      end function math_ran_dst
!=======================================================================
!> @brief Give random number in the defined range
!! @ingroup math
!! @param[in] lower, upper     range for random numer
!! @return  random number in the defined range
      real function math_ran_rng(lower, upper)
      implicit none

      ! argument list
      real lower, upper

      ! functions
      real math_zbqlu01
!-----------------------------------------------------------------------
      math_ran_rng = lower + math_zbqlu01()*(upper-lower)

      return
      end function math_ran_rng
!=======================================================================
!> @brief Marsaglia-Zaman random number generator
!! @ingroup math
!! @details Returns a uniform random number between 0 & 1, using
!!   a Marsaglia-Zaman type subtract-with-borrow generator.
!! @remarks Uses double precision, rather than integer, arithmetic 
!!   throughout because MZ's INTEGER constants overflow
!!   32-bit INTEGER storage (which goes from -2^31 to 2^31).
!!   Ideally, we would explicitly truncate all INTEGER 
!!   quantities at each stage to ensure that the DOUBLE
!!   PRECISION representations DO not accumulate approximation
!!   error; however, on some machines the USE of DNINT to
!!   accomplish this is *seriously* slow (run-time increased
!!   by a factor of about 3). This DOUBLE PRECISION version 
!!   has been tested against an INTEGER implementation that
!!   uses long integers (non-standard and, again, slow) -
!!   the output was identical up to the 16th decimal place
!!   after 10^10 calls, so we're probably OK ...
!!   In current implementation we follow Nek5000 compilation
!!   rulles prolonging all reals to doulbe with compiler flaggs.
!! @author Richard Chandler (richard@stats.ucl.ac.uk)
!! @author Paul Northrop (northrop@stats.ox.ac.uk)
!! @return  random number
      real function math_zbqlu01()
      implicit none

      ! global variables
      real zbqlix(43), br, cr
      common /math_zbql01/ zbqlix, br, cr

      ! local variables
      integer curpos,id22,id43
      save curpos,id22,id43
      data curpos,id22,id43 /1,22,43/
      real xr,b2,binv

!-----------------------------------------------------------------------
      b2 = br
      binv = 1.0/br

      do
         xr = zbqlix(id22) - zbqlix(id43) - cr
         if (xr.lt.0.0) then
            xr = xr + br
            cr = 1.0
         else
            cr = 0.0
         endif
         zbqlix(id43) = xr

         ! Update array pointers. Do explicit check for bounds of each to
         ! avoid expense of modular arithmetic. If one of them is 0
         ! the others won't be   
         curpos = curpos - 1
         id22 = id22 - 1
         id43 = id43 - 1
         if (curpos.eq.0) then
            curpos=43
         elseif (id22.eq.0) then
            id22 = 43
         elseif (id43.eq.0) then
            id43 = 43
         endif

         ! The INTEGER arithmetic there can yield X=0, which can cause 
         ! problems in subsequent routines (e.g. ZBQLEXP). The problem
         ! is simply that X is discrete whereas U is supposed to 
         ! be continuous - hence IF X is 0, go back and generate another
         ! X and RETURN X/B^2 (etc.), which will be uniform on (0,1/B). 
         if (xr.ge.binv) exit
         b2 = b2*br       
      enddo
      
      math_zbqlu01 = xr/b2
      return
      end function math_zbqlu01
!=======================================================================
!> @brief Initialise Marsaglia-Zaman random number generator
!! @ingroup math
!! @details To initialise the random number generator - either repeatably
!!    or nonrepeatably. 
!! @param[in] seed     number which generates elements of the array ZBQLIX
      subroutine math_zbqlini(seed)
      implicit none

      ! argiment list
      integer seed

      ! global variables
      real zbqlix(43), br, cr
      common /math_zbql01/ zbqlix, br, cr

      ! local variables
      integer icall
      save    icall
      data    icall  /0/

      integer il

      !  Initialisation data for random number generator.
      !  The values below have themselves been generated using the
      !  NAG generator.      
      real zbq_ini(43),b_ini,c_ini
      data (zbq_ini(il),il=1,43) /8.001441d7,5.5321801d8,
     $   1.69570999d8,2.88589940d8,2.91581871d8,1.03842493d8,
     $   7.9952507d7,3.81202335d8,3.11575334d8,4.02878631d8,
     $   2.49757109d8,1.15192595d8,2.10629619d8,3.99952890d8,
     $   4.12280521d8,1.33873288d8,7.1345525d7,2.23467704d8,
     $   2.82934796d8,9.9756750d7,1.68564303d8,2.86817366d8,
     $   1.14310713d8,3.47045253d8,9.3762426d7 ,1.09670477d8,
     $   3.20029657d8,3.26369301d8,9.441177d6,3.53244738d8,
     $   2.44771580d8,1.59804337d8,2.07319904d8,3.37342907d8,
     $   3.75423178d8,7.0893571d7 ,4.26059785d8,3.95854390d8,
     $   2.0081010d7,5.9250059d7,1.62176640d8,3.20429173d8,
     $   2.63576576d8/
      data b_ini / 4.294967291d9 /
      data c_ini / 0.0d0 /

      real tmpvar1

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      if (icall.gt.0) return
      icall = icall + 1

      br = b_ini
      cr = c_ini
      ! if seed==0 use cpu time as seed
      if (seed.eq.0) then
         tmpvar1 = mod(dnekclock(),br)
      else
         tmpvar1 = mod(real(seed),br)
      endif
      zbqlix(1) = tmpvar1
      do il= 2, 43
         tmpvar1 = zbq_ini(il-1)*3.0269d4
         tmpvar1 = mod(tmpvar1,br)       
         zbqlix(il) = tmpvar1
      enddo
     
      return
      end subroutine math_zbqlini
!=======================================================================
!> @brief Give bounds for loops to extract edge
!! @ingroup math
!! @note All edge related routines used IXCN and ESKIP, but this
!!   caused some problems as these arrays have to be continuosly
!!   updated for different levels in pressure solver, so I add this
!!   routine.
!! @param[out]   istart    lower loop bound
!! @param[out]   istop     upper loop bound
!! @param[out]   iskip     stride
!! @param[in]    iedg      edge number
!! @param[in]    nx,ny,nz  element size
      subroutine math_edgind(istart,istop,iskip,iedg,nx,ny,nz)
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'TOPOL'

!     argument list
      integer istart,istop,iskip,iedg,nx,ny,nz

!     local variables
      integer ivrt, icx, icy, icz
!-----------------------------------------------------------------------
!     find vertex position
!     start
      ivrt = icedg(1,iedg)
      icx = mod(ivrt +1,2)
      icy = mod((ivrt-1)/2,2)
      icz = (ivrt-1)/4
      istart =  1 + (nx-1)*icx + nx*(ny-1)*icy + nx*ny*(nz-1)*icz

!     stop
      ivrt = icedg(2,iedg)
      icx = mod(ivrt +1,2)
      icy = mod((ivrt-1)/2,2)
      icz = (ivrt-1)/4
      istop =  1 + (nx-1)*icx + nx*(ny-1)*icy + nx*ny*(nz-1)*icz

!     find stride
      if (iedg.le.4) then
         iskip = 1
      elseif (iedg.le.8) then
         iskip = nx
      else
         iskip =nx*nx
      endif

      return
      end subroutine
!=======================================================================
!> @brief Extract element edge
!! @ingroup math
!! @note This routine works on singe element not whole field.
!! @param[out]   vec       vector containg edge values
!! @param[in]    edg       edge number
!! @param[in]    vfld      pointer to singe element in the field
!! @param[in]    nx,ny,nz  element size
      subroutine math_etovec(vec,edg,vfld,nx,ny,nz)
      implicit none

      include 'SIZE'
      include 'TOPOL'

!     argument list
      real vfld(nx*ny*nz)
      integer edg,nx,ny,nz
      real vec(nx)

!     local variables
      integer is, ie, isk, il, jl
!-----------------------------------------------------------------------
!      is  = IXCN(icedg(1,edg))
!      ie  = IXCN(icedg(2,edg))
!      isk = ESKIP(edg,3)
      call math_edgind(is,ie,isk,edg,nx,ny,nz)
      jl = 1
      do il=is,ie,isk
        vec(jl) = vfld(il)
        jl = jl + 1
      enddo

      return
      end subroutine
!=======================================================================
!> @brief 3D rotation of a vector along given axis
!! @ingroup math
!! @param[in]  vo     output vector
!! @param[in]  vi     input vector
!! @param[in]  va     rotation axis
!! @param[in]  an     rotation angle
      subroutine math_rot3da(vo,vi,va,an)
      implicit none

      ! parameters
      integer ldim
      parameter (ldim=3)

      ! argument list
      real vo(ldim), vi(ldim), va(ldim), an

      ! local variables
      integer il, jl
      real mat(ldim,ldim), ta(ldim)
      real rtmp, can, can1, san
!-----------------------------------------------------------------------
      if (an.eq.0.0) then
         call copy(vo,vi,ldim)
      else
         ! make sure the axis vector is normalised
         rtmp = 0.0
         do il=1,ldim
            rtmp = rtmp + va(il)**2
         enddo
         ! add check if rtmp.gt.0
         rtmp = 1.0/sqrt(rtmp)
         do il=1,ldim
            ta(il) = rtmp*va(il)
         enddo
         ! fill in rotation mtrix
         can = cos(an)
         can1 = 1.0 - can
         san = sin(an)
         mat(1,1) = can + ta(1)*ta(1)*can1
         mat(2,1) = ta(1)*ta(2)*can1 - ta(3)*san
         mat(1,2) = ta(1)*ta(2)*can1 + ta(3)*san
         mat(3,1) = ta(1)*ta(3)*can1 + ta(2)*san
         mat(1,3) = ta(1)*ta(3)*can1 - ta(2)*san
         mat(2,2) = can + ta(2)*ta(2)*can1
         mat(3,2) = ta(2)*ta(3)*can1 - ta(1)*san
         mat(2,3) = ta(2)*ta(3)*can1 + ta(1)*san
         mat(3,3) = can + ta(3)*ta(3)*can1
         ! perform rotation
         do il = 1, ldim
            vo(il) = 0.0
            do jl=1,ldim
               vo(il) = vo(il) + mat(jl,il)*vi(jl)
            enddo
         enddo
      endif

      return
      end subroutine
!=======================================================================
