!-----------------------------------------------------------------------
!
!     user subroutines required by nek5000
!
!     Parameters used by this set of subroutines:
!
!-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,ieg)
      include 'SIZE'
      include 'NEKUSE'          ! UDIFF, UTRANS

      UDIFF =0.0
      UTRANS=0.0

      return
      end
!-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,ieg)
      
      include 'SIZE'
      include 'NEKUSE'          ! FF[XYZ]
      include 'PARALLEL'

      integer ix,iy,iz,ieg

      ffx = 0.0
      ffy = 0.0
      ffz = 0.0

      return
      end
!-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,ieg)
      include 'SIZE'
      include 'NEKUSE'          ! QVOL
      
      QVOL   = 0.0

      return
      end
!-----------------------------------------------------------------------
      subroutine userchk
      implicit none
      include 'SIZE'            !
      include 'TSTEP'           ! ISTEP, lastep, time
      include 'INPUT'           ! IF3D, PARAM
#ifdef TSRS
      logical ifsave
#endif
!     start framework
      if (ISTEP.eq.0) call frame_start

!     monitor simulation
      call frame_monitor

!     save/load files for full-restart
      call chkpt_main

!     for statistics
      call stat_avg

#ifdef TSRS
      !     for Time Series 
            ifsave = .FALSE.
            call tsrs_main(ifsave)
#endif
!     for OPPO ctrl 
#ifdef CTRL 
      call oppo_ctrl
#endif 

#ifdef DRL
      
      call DRL_main
      
#endif 
      

!     finalise framework
      if (ISTEP.eq.NSTEPS.or.LASTEP.eq.1) then
         call frame_end
      endif
     
      return
      end
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,eg)
      implicit none
      include 'SIZE'
      include 'NEKUSE'          ! UX, UY, UZ, TEMP, X, Y
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      include 'TOPOL'
      include 'GEOM' !unx, uny, unz

      integer ix,iy,iz,iside,eg
      integer iel,f     
      real vf,snx,sny,snz
      logical isfind

      iel=gllel(eg)
      vf=0.0
      isfind=.FALSE.

! #ifdef CTRL
!       ! call actuate_vel(ix,iy,iz,iside,iel,vf,isfind)
!       if (isfind) then 
!       ux = 0.0
!       uy = -vf
!       uz = 0.0      
!       ! call record_actuate(ix,iy,iz,iel,iside,x,y,z,vf)
!       endif
! #endif

#ifdef DRL

      if (y.le.0.1) then

            call ACTUATE_JET(vf,isfind,ix,iy,iz,iside,eg)

!            if (isfind) then 
!            f=eface1(iside)
!            if (f.eq.1.or.f.eq.2) then ! "r face"
!            snx=unx(iy,iz,iside,iel) ! Note:  iy,iz
!            sny=uny(iy,iz,iside,iel)
!            snz=unz(iy,iz,iside,iel)
!            elseif (f.eq.3.or.f.eq.4) then ! "s face"
!            snx=unx(ix,iz,iside,iel) !  ix,iz
!            sny=uny(ix,iz,iside,iel)
!            snz=unz(ix,iz,iside,iel)
!            elseif (f.eq.5.or.f.eq.6) then ! "t face"
!            snx=unx(ix,iy,iside,iel) ! ix,iy
!            sny=uny(ix,iy,iside,iel)
!            snz=unz(ix,iy,iside,iel)
!            endif

            ux=0.0
            uy=vf
            uz=0.0

! #ifndef YWDEBUG
!       print *,"NID",NID,"ISTEP",ISTEP,
!      $ "x",x,"y",y,"z",z,
!      $"iel",iel,"ix",ix,"iy",iy,"iz",iz,
!      $ "ux",ux,"uy",uy
! #endif

!            endif ! if isfind: We no longer use that
      endif

#endif

c$$$       ! Print for test 
!       print *,"NID",NID,"ISTEP",ISTEP,
!      $ "x",x,"y",y,"z",z,
!      $"iel",iel,"ix",ix,"iy",iy,"iz",iz,
!      $ "vf",vf,"ux",ux,"uy",uy
!      $ "BC:",cbi
c$$$
 
      return
      end
! -----------------------------------------------------------------------

!-----------------------------------------------------------------------
      subroutine useric (ix,iy,iz,ieg)
      include 'SIZE'
      include 'NEKUSE'          ! UX, UY, UZ, TEMP, Z
      include "PARALLEL"
!     argument list
      integer ix,iy,iz,ieg

!     Initialization 
      real viscosity, Ubar, Ret, channelh, PI
      real utau, sig, dup, betab, alphap, eps   
      real deviation, dy, zp, yp, xp
      real expsig, expsig2, cosbeta, sinalpha
      real streamp, spanp

!     Test 
      integer icount
      
      PI=3.14159265359
      
      viscosity = 2e-5
      Ubar = 0.135
      Ret = 180 

      ! YW Modified here 
      channelh = 1.0

      utau = Ret*viscosity/channelh
      sig = 0.00055
      dup = Ubar*0.25/utau;

      ! Wave Length TO Wave Number 
      betap = 2.0*PI*(1.0/100.0);
      alphap = 2.0*PI*(1.0/250.0);
      eps = Ubar/100.0;      
      deviation = 1.0 + 0.2*rand();
      
      ! Inner-scaled Coordinates 
      zp = z*Ret/channelh
      ! dy = MIN(y+1,2.0*channelh-(y+1))      
      
      ! Half-channel
      !--------------------------------
      dy = MIN(y,channelh-(y+1))      
      !------------------------------
      yp = dy*Ret/channelh
      xp = x*Ret/channelh 
      
      expsig=EXP(-sig*yp*yp+0.5)
      cosbeta=COS(betap*zp)

      ! Perturbation in streamwise velocity 
      streamper=(utau*dup/2.0)*(yp/40.0)*expsig*cosbeta*deviation
      expsig2=EXP(-sig*yp*yp)
      sinalpha=SIN(alphap*xp)
      spanper=eps*sinalpha*yp*expsig2*deviation
      
      ! bulk velocity + perturbation
      !------------------------------------------------------------
      ! ux = 3.0*Ubar*(dy/channelh - 0.5*(dy/channelh)*(dy/channelh))
      
      ! half-channel 
      !------------------------------------------------------------
      ux = 3.0*Ubar*((dy/channelh)*(dy/channelh))
      !------------------------------------------------------------
      ! Adding perturbation here
      ux = ux+streamper*25
      uz = spanper*25   
      uy = 0
            
!     temperature and scalars            
      temp = 0.0
      
      icount = icount +1 

      return
      end
!-----------------------------------------------------------------------
      subroutine usrdat
      include 'SIZE'

      integer,parameter :: seed = 42
      call srand(seed)

      return
      end
!-----------------------------------------------------------------------
      subroutine usrdat2
      implicit none
      include 'SIZE'

      return
      end
!-----------------------------------------------------------------------
      subroutine usrdat3
      include 'SIZE'
      include 'INPUT'           ! param, if3d
      include 'MASS'            ! volvm1      
      
      param(54) = -1  ! use >0 for const flowrate or <0 bulk vel
                      ! flow direction is given by (1=x, 2=y, 3=z) 
      param(55) = 1.0 ! flowrate/bulk-velocity 

      return
      end
c-----------------------------------------------------------------------

!======================================================================
!> @brief Register user specified modules
      subroutine frame_usr_register
      implicit none

      include 'SIZE'
      include 'FRAMELP'
!-----------------------------------------------------------------------
!     register modules
      call io_register
      call chkpt_register
      call stat_register
#ifdef TSRS
      call tsrs_register
#endif
      return
      end subroutine
!======================================================================
!> @brief Initialise user specified modules
      subroutine frame_usr_init
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'SOLN'
!-----------------------------------------------------------------------
!     initialise modules
      call chkpt_init
      call stat_init
#ifdef TSRS
      call tsrs_init
#endif
      return
      end subroutine
!======================================================================
!> @brief Finalise user specified modules
      subroutine frame_usr_end
      implicit none

      include 'SIZE'
      include 'FRAMELP'
!-----------------------------------------------------------------------
!     finalise modules
      call stat_end()
#ifdef TSRS
      call tsrs_end()
#endif
      return
      end subroutine
!======================================================================
!> @brief Provide element coordinates and local numbers (user interface)
!! @param[out]  idir              mapping (uniform) direction
!! @param[out]  ctrs              2D element centres
!! @param[out]  cell              local element numberring
!! @param[in]   lctrs1,lctrs2     array sizes
!! @param[out]  nelsort           number of local 3D elements to sort
!! @param[out]  map_xm1, map_ym1  2D coordinates of mapped elements
!! @param[out]  ierr              error flag
      subroutine user_map2d_get(idir,ctrs,cell,lctrs1,lctrs2,nelsort,
     $     map_xm1,map_ym1,ierr)
      implicit none

      include 'SIZE'
      include 'INPUT'           ! [XYZ]C
      include 'GEOM'            ! [XYZ]M1

!     argument list
      integer idir
      integer lctrs1,lctrs2
      real ctrs(lctrs1,lctrs2)  ! 2D element centres  and diagonals 
      integer cell(lctrs2)      ! local element numberring
      integer nelsort           ! number of local 3D elements to sort
      real map_xm1(lx1,lz1,lelt), map_ym1(lx1,lz1,lelt)
      integer ierr              ! error flag

!     local variables
      integer ntot              ! tmp array size for copying
      integer el ,il ,jl        ! loop indexes
      integer nvert             ! vertex number
      real rnvert               ! 1/nvert
      real xmid,ymid            ! 2D element centre
      real xmin,xmax,ymin,ymax  ! to get approximate element diagonal
      integer ifc               ! face number

!     dummy arrays
      real xcoord(8,LELT), ycoord(8,LELT) ! tmp vertex coordinates

#ifdef DEBUG
!     for testing
      character*3 str1, str2
      integer iunit, ierrl
      ! call number
      integer icalldl
      save icalldl
      data icalldl /0/
#endif

!-----------------------------------------------------------------------
!     initial error flag
      ierr = 0
!     set important parameters
!     uniform direction; should be taken as input parameter
!     x-> 1, y-> 2, z-> 3
      idir = 3
      
!     get element midpoints
!     vertex number
      nvert = 2**NDIM
      rnvert= 1.0/real(nvert)

!     eliminate uniform direction
      ntot = 8*NELV
      if (idir.EQ.1) then  ! uniform X
         call copy(xcoord,YC,ntot) ! copy y
         call copy(ycoord,ZC,ntot) ! copy z
      elseif (idir.EQ.2) then  ! uniform Y
         call copy(xcoord,XC,ntot) ! copy x
         call copy(ycoord,ZC,ntot) ! copy z
      elseif (idir.EQ.3) then  ! uniform Z
         call copy(xcoord,XC,ntot) ! copy x
         call copy(ycoord,YC,ntot) ! copy y
      endif

!     set initial number of elements to sort
      nelsort = 0
      call izero(cell,NELT)

!     for every element
      do el=1,NELV
!     element centre
         xmid = xcoord(1,el)
         ymid = ycoord(1,el)
!     element diagonal
         xmin = xmid
         xmax = xmid
         ymin = ymid
         ymax = ymid
         do il=2,nvert
            xmid=xmid+xcoord(il,el)
            ymid=ymid+ycoord(il,el)
            xmin = min(xmin,xcoord(il,el))
            xmax = max(xmax,xcoord(il,el))
            ymin = min(ymin,ycoord(il,el))
            ymax = max(ymax,ycoord(il,el))
         enddo
         xmid = xmid*rnvert
         ymid = ymid*rnvert

!     count elements to sort
            nelsort = nelsort + 1
!     2D position
!     in general this coud involve some curvilinear transform
            ctrs(1,nelsort)=xmid
            ctrs(2,nelsort)=ymid
!     reference distance
            ctrs(3,nelsort)=sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
            if (ctrs(3,nelsort).eq.0.0) then
               ierr = 1
               return
            endif
!     element index
            cell(nelsort) = el
      enddo

!     provide 2D mesh
!     in general this coud involve some curvilinear transform
      if (idir.EQ.1) then  ! uniform X
         ifc = 4
         do el=1,NELV
            call ftovec(map_xm1(1,1,el),ym1,el,ifc,nx1,ny1,nz1)
            call ftovec(map_ym1(1,1,el),zm1,el,ifc,nx1,ny1,nz1)
         enddo
      elseif (idir.eq.2) then  ! uniform y
         ifc = 1
         do el=1,nelv
            call ftovec(map_xm1(1,1,el),xm1,el,ifc,nx1,ny1,nz1)
            call ftovec(map_ym1(1,1,el),zm1,el,ifc,nx1,ny1,nz1)
         enddo
      elseif (idir.eq.3) then  ! uniform z
         ifc = 5
         do el=1,nelv
            call ftovec(map_xm1(1,1,el),xm1,el,ifc,nx1,ny1,nz1)
            call ftovec(map_ym1(1,1,el),ym1,el,ifc,nx1,ny1,nz1)
         enddo
      endif

#ifdef DEBUG
!     testing
      ! to output refinement
      icalldl = icalldl+1
      call io_file_freeid(iunit, ierrl)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalldl
      open(unit=iunit,file='map2d_usr.txt'//str1//'i'//str2)
      
      write(iunit,*) idir, NELV, nelsort
      write(iunit,*) 'Centre coordinates and cells'
      do el=1,nelsort
         write(iunit,*) el, ctrs(:,el), cell(el)
      enddo
      write(iunit,*) 'GLL coordinates'
      do el=1,nelsort
         write(iunit,*) 'Element ', el
         write(iunit,*) 'XM1'
         do il=1,nz1
            write(iunit,*) (map_xm1(jl,il,el),jl=1,nx1)
         enddo
         write(iunit,*) 'YM1'
         do il=1,nz1
            write(iunit,*) (map_ym1(jl,il,el),jl=1,nx1)
         enddo
      enddo
      close(iunit)
#endif

      return
      end subroutine
!=======================================================================
!> @brief Provide velocity, deriv. and vort. in required coordinates and normalise pressure
!! @param[out]   lvel             velocity
!! @param[out]   dudx,dvdx,dwdx   velocity derivatives
!! @param[out]   vort             vorticity
!! @param[inout] pres             pressure
      subroutine user_stat_trnsv(lvel,dudx,dvdx,dwdx,vort,pres)
      implicit none

      include 'SIZE'
      include 'SOLN'
      include 'INPUT'               ! if3d
      include 'GEOM'

      ! argument list
      real lvel(LX1,LY1,LZ1,LELT,3) ! velocity array
      real dudx(LX1,LY1,LZ1,LELT,3) ! velocity derivatives; U
      real dvdx(LX1,LY1,LZ1,LELT,3) ! V
      real dwdx(LX1,LY1,LZ1,LELT,3) ! W
      real vort(LX1,LY1,LZ1,LELT,3) ! vorticity
      real pres(LX1,LY1,LZ1,LELT)   ! pressure

      ! local variables
      integer itmp              ! dummy variable
      integer il, jl            ! loop index
      integer ifll              ! field number for object definition
      real vrtmp(lx1*lz1)       ! work array for face
      real vrtmp2(2)            ! work array
      
      ! functions
      real vlsum
!-----------------------------------------------------------------------
      ! Velocity transformation; simple copy
      itmp = NX1*NY1*NZ1*NELV
      call copy(lvel(1,1,1,1,1),VX,itmp)
      call copy(lvel(1,1,1,1,2),VY,itmp)
      call copy(lvel(1,1,1,1,3),VZ,itmp)

      ! Derivative transformation
      ! No transformation
      call gradm1(dudx(1,1,1,1,1),dudx(1,1,1,1,2),dudx(1,1,1,1,3),
     $      lvel(1,1,1,1,1))
      call gradm1(dvdx(1,1,1,1,1),dvdx(1,1,1,1,2),dvdx(1,1,1,1,3),
     $      lvel(1,1,1,1,2))
      call gradm1(dwdx(1,1,1,1,1),dwdx(1,1,1,1,2),dwdx(1,1,1,1,3),
     $      lvel(1,1,1,1,3))

      ! get vorticity
      if (IF3D) then
         ! curlx
         call sub3(vort(1,1,1,1,1),dwdx(1,1,1,1,2),
     $        dvdx(1,1,1,1,3),itmp)
         ! curly
         call sub3(vort(1,1,1,1,2),dudx(1,1,1,1,3),
     $        dwdx(1,1,1,1,1),itmp)
      endif
      ! curlz
      call sub3(vort(1,1,1,1,3),dvdx(1,1,1,1,1),dudx(1,1,1,1,2),itmp)
      
      ! normalise pressure
      ! in this example I integrate pressure over top faces marked "W"
      ifll = 1     ! I'm interested in velocity bc
      ! relying on mesh structure given by genbox set face number
      jl = 3
      call rzero(vrtmp2,2)  ! zero work array
      itmp = LX1*LZ1
      do il=1,nelv   ! element loop
         if (cbc(jl,il,ifll).eq.'W  ') then
            vrtmp2(1) = vrtmp2(1) + vlsum(area(1,1,jl,il),itmp)
            call ftovec(vrtmp,pres,il,jl,lx1,ly1,lz1)
            call col2(vrtmp,area(1,1,jl,il),itmp)
            vrtmp2(2) = vrtmp2(2) + vlsum(vrtmp,itmp)
         endif
      enddo
      ! global communication
      call gop(vrtmp2,vrtmp,'+  ',2)
      ! missing error check vrtmp2(1) == 0
      vrtmp2(2) = -vrtmp2(2)/vrtmp2(1)
      ! remove mean pressure
      itmp = LX1*LY1*LZ1*NELV
      call cadd(pres,vrtmp2(2),itmp)

      return
      end subroutine
!======================================================================

c automatically added by makenek
      subroutine usrdat0() 

      return
      end

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end
