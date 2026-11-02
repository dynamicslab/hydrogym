!=======================================================================
! Name        : time_series
! Author      : Adam Peplinski
! Version     : last modification 2015.05.22
! Copyright   : GPL
! Description : This is a set of routines to generate time series for 
!     wing simulations. It is just a slight modification of hpts. It 
!     is written to save some memory and directly couple to arrays 
!     generated in statistics.
!=======================================================================
!     Initialise points for statistics point time history
!     I read 2D distribution and generate 3D on
      subroutine stat_pts_init

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'GEOM_DEF'
      include 'GEOM'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
      include 'PTSTAT'          ! point time series for 2D statistics

!     reading buffer
      integer lt2
      parameter (lt2=LX1*LY1*LZ1*LELT)
      real xyz(LDIM,lt2)        ! coordinate buffer
      integer ixyz(lt2)         ! global numberring
      common /scrns/ xyz
      common /scrmg/ ixyz

!     dummy arrays
      integer mid(2,lt2)        ! integer transfer array
      integer mprocid(lt2)      ! poin-processor mapping
      common /scruz/ mid, mprocid


!     local variables
      integer pts_npt2d         ! point number in 2D slice
      integer pts_nlev          ! number of layers in uniform direction
      integer pts_dir           ! uniform direction
      real pts_xmin, pts_xmax   ! min/max position of the layer
      real pts_dx               ! distance between layers

      integer ierr              ! error mark
      integer nxyz, ntot        ! array size
      integer il, jl, kl        ! loop index
      integer itmp_dx, itmp_dm, itmp_dp ! dummy variables
      real xlmin, xlmax         ! min/max grid position in uniform direction

!     rows and collumns to read in for given processor
      integer pts_nxi, pts_ndx, pts_nyi, pts_ndy
!     initial processor division; pts_nx*pts_ny + pts_nm = NP
      integer pts_nx            ! processors in 2D slice
      integer pts_ny            ! processors in uniform direction
      integer pts_nm            ! unused processors; pts_nm = mod(NP,pts_nx)

!     statistics
      integer pts_glmax, pts_glmin, pts_glnp, pts_glsum,
     $     pts_glnl

!     file reading
      logical ifascii           ! file format
      character*132 fname       ! file name
      character*132 hdr         ! header
      integer wdsizli           ! single/double precision
      real*4 test_pattern       ! big/little endian test
      logical ifbswap           ! big/little endian
      integer ivdum(4)          ! dummy array

!     point redistribution
      integer nptimb            ! allowed point imbalance

      integer nfail             ! number of unmapped points
      
!     functions
      integer iglsum, iglmax, iglmin
      real glmin, glmax
      logical if_byte_swap_test

c$$$!     for testing
c$$$      integer itl1, itl2
c$$$      character*2 str
 
!     stamp the log
      if(NIO.eq.0) write(6,*) 'Point time history init.; start'


!     set file format
!     for now text file
      ifascii = .TRUE.

!     get information about 2D point number, number of layers in 
!     uniform direction and the chosen direction (1-x, 2-y, 3-z)
!     x_init and delta x
      if(NIO.eq.0) write(6,*) 'reading stat_pts.in'

      ierr = 0
      if (ifascii) then
!     text file
         if(NID.eq.0) then
            open(50,file='stat_pts.in',status='old',err=100)
            read(50,*,err=100) pts_npt2d, pts_nlev, pts_dir
            
            ! print *, 'The pts_nlev is', pts_nlev

            read(50,*,err=100) pts_xmin, pts_dx
            close(50)
            goto 101
 100        ierr = 1
 101        continue
         endif
      else                      ! ifascii
!     binary file
         fname = 'stat_pts.in'//char(0)
         call byte_open(fname,ierr)
         if (ierr.eq.0) then
!     integer header
            call byte_read(hdr,10,ierr)
            if (ierr.ne.0) goto 201
            read (hdr,150) pts_npt2d, pts_nlev, pts_dir, wdsizli
 150        format(i11,i11,i3,i3)
            wdsizli = wdsizli/4

!     big/little endian test
            il = 1
            call byte_read(test_pattern,il,ierr)
            if (ierr.ne.0) goto 201
            ifbswap = if_byte_swap_test(test_pattern,ierr)
            if (ierr.ne.0) goto 201

!     real part
            il=wdsizli*2
            call byte_read(ivdum,il,ierr)
            if (ierr.ne.0) goto 201
!     copy variables
            if (wdsizli.eq.2) then
!     big/little endian
               if (ifbswap) then
                  call byte_reverse8(ivdum,il,ierr)
                  if (ierr.ne.0) goto 201
               endif
               call copy(pts_xmin,ivdum(1),1)
               call copy(pts_dx,ivdum(3),1)
            elseif (wdsizli.eq.1) then
!     big/little endian
               if (ifbswap) then
                  call byte_reverse(ivdum,il,ierr)
                  if (ierr.ne.0) goto 201
               endif
               call copy4r(pts_xmin,ivdum(1),1)
               call copy4r(pts_dx,ivdum(2),1)
            endif

!     close file
            call byte_close(ierr)
            if (ierr.ne.0.and.NIO.eq.0) write(6,*)
     $              'Error closing stat_pts.in in stat_pts_in'

         endif                  ! ierr.eq.0

 201     continue
      endif                     ! ifascii

      ierr=iglsum(ierr,1)
      if(ierr.gt.0) then
        if(NIO.eq.0) write(6,*)
     $        'Error reading stat_pts.in in stat_pts_in'
        call exitt
      endif

!     distribute 
      call bcast(pts_npt2d,isize)
      call bcast(pts_nlev,isize)
      call bcast(pts_dir,isize)
      call bcast(pts_xmin,wdsize)
      call bcast(pts_dx,wdsize)

!     make sure some numbers are positive
      pts_dx = abs(pts_dx)

!     check consistency comparing box size
      if (pts_nlev.le.0) then
         if(NIO.eq.0)
     $        write(6,*) 'ERROR: stat_pts_in; wrong layer number'
         call exitt
      endif
      if (pts_dir.le.0.or.pts_dir.gt.3) then
         if(NIO.eq.0)
     $        write(6,*) 'ERROR: stat_pts_in; wrong uniform direction'
         call exitt
      endif

      pts_xmax = pts_xmin + (pts_nlev - 1)*pts_dx

      nxyz  = NX1*NY1*NZ1
      ntot  = nxyz*NELT

      if (pts_dir.eq.1) then
         xlmin = glmin(XM1,ntot)          
         xlmax = glmax(XM1,ntot)
      elseif (pts_dir.eq.2) then
         xlmin = glmin(YM1,ntot)          
         xlmax = glmax(YM1,ntot)
      else
         xlmin = glmin(ZM1,ntot)          
         xlmax = glmax(ZM1,ntot)
      endif
      if (pts_xmin.lt.xlmin.or.pts_xmax.gt.xlmax) then
         if(NIO.eq.0)
     $        write(6,*) 'ERROR: stat_pts_in; wrong layer position'
         call exitt
      endif

!     calculate global number of points and check consistency
      npoints = pts_npt2d*pts_nlev
      ! print *, "The number of npoint is", npoints
      ! print *, "The NP equals to", NP
      ! print *, "The LHIS is", LHIS
      if (npoints.gt.NP*LHIS) then
         if(NIO.eq.0) then
            write(6,*) 'ERROR: stat_pts_in; wrong global point number'
            write(6,*) 'Unifomrm direction ',pts_dir
            write(6,*) 'Mesh limits        ',xlmin,xlmax
            write(6,*) 'Pts layer limits   ',pts_xmin,pts_xmax
         endif
         call exitt
      endif

!     create intial points partitioning
!     this one is a crude one and will be corrected later

!     processor partitioning
!     take into account ratios
      xlmin = pts_npt2d/real(pts_nlev)
!     should I use nint instead of int? 
      pts_nx = int(sqrt(xlmin*NP))
!     correct bounds
      pts_nx = max(1,pts_nx)
      pts_nx = min(NP,pts_nx)
!     uniform direction
      pts_ny = NP/pts_nx
      pts_nm = mod(NP,pts_nx)
!     in case pts_nx is much bigger than pts_ny
      if (pts_ny.lt.pts_nm) then
         pts_nx = pts_nx + pts_nm/pts_ny
         pts_nm = NP - pts_nx*pts_ny
      endif

!     divide points neglecting last pts_nm processors
      if (NID.lt.pts_nx*pts_ny) then
!     divide 2D points
         itmp_dx = pts_npt2d/pts_nx
         itmp_dm = mod(pts_npt2d,pts_nx)

         itmp_dp = mod(NID,pts_nx)
         pts_nxi = itmp_dp*itmp_dx + 1 + min(itmp_dp,itmp_dm)
         pts_ndx = itmp_dx
         if (itmp_dp.lt.itmp_dm) pts_ndx = pts_ndx +1

!     divide levels
         itmp_dx = pts_nlev/pts_ny
         itmp_dm = mod(pts_nlev,pts_ny)

         itmp_dp = NID/pts_nx
         pts_nyi = itmp_dp*itmp_dx + 1 + min(itmp_dp,itmp_dm)
         pts_ndy = itmp_dx
         if (itmp_dp.lt.itmp_dm) pts_ndy = pts_ndy +1

      else
         pts_nxi = 0
         pts_ndx = 0

         pts_nyi = 0
         pts_ndy = 0
      endif

!     get local point number
      npts = pts_ndx*pts_ndy
      if (npts.eq.0) then
         pts_nxi = 0
         pts_ndx = 0

         pts_nyi = 0
         pts_ndy = 0
      endif

!     statistics
      pts_glmax = iglmax(npts,1)
      pts_glmin = iglmin(npts,1)
!     number of nodes with npts exceeding LHIS
      ierr = 0
      if (npts.gt.LHIS) ierr = 1
      pts_glnp = iglsum(ierr,1)
!     number of points to be redistributed
      ierr = 0
      if (npts.gt.LHIS) ierr = npts - LHIS
      pts_glsum = iglsum(ierr,1)
!     number of empty processors
      ierr = 0
      if (npts.eq.0) ierr = 1
      pts_glnl = iglsum(ierr,1)

!     stamp logs
      if (NIO.eq.0) then
         write(6,*)
         write(6,*) 'Time series; stat_pts_in:'
         write(6,*) '   Initial distribution:'
         write(6,'(A22,I7)') 'Global point nr       ',npoints
         write(6,'(A22,I7,I7)') 'Local point nr min/max',pts_glmin,
     $        pts_glmax
         write(6,'(A22,I7)') 'Nr of pts to redist.  ',pts_glsum
         write(6,'(A22,I7)') 'Nr of overfilled nodes',pts_glnp
         write(6,'(A22,I7)') 'Nr of empty nodes     ',pts_glnl
         write(6,*)
      endif

!     check consistency of the local number of points

      if(pts_glmax.gt.LHIS) then
!     I have to redistribute points

!     Check if the buffer has enough space
         if(pts_glmax.gt.lt2) then
!     we cannot read everyting at once; for now just exit
            if(NIO.eq.0) write(6,*)
     $           'ERROR: stat_pts_in; too many local points'
            call exitt
         endif

!     load the 2D mesh to the buffer
         call stat_pts_read(xyz,lt2,mid,lt2,pts_nx,pts_ny,pts_dir,
     $        pts_nxi,pts_ndx,pts_npt2d,pts_nlev,ifascii)

!     generate whole 3D distribution
         call stat_pts_fill(xyz,ixyz,lt2,pts_dir,pts_npt2d,pts_nxi,
     $        pts_ndx,pts_nyi,pts_ndy,pts_xmin,pts_dx)

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='xyz_fill.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, LHIS, lt2
c$$$         write(10001,*) pts_npt2d, pts_nlev, pts_dir
c$$$         write(10001,*) pts_xmin, pts_dx
c$$$         write(10001,*) NP, pts_nx, pts_ny, pts_nm
c$$$         write(10001,*) pts_nxi, pts_ndx, pts_nyi, pts_ndy
c$$$         do itl1=1,npts
c$$$            write(10001,*) itl1, ixyz(itl1), mid(1,itl1),
c$$$     $           (xyz(itl2,itl1),itl2 = 1,LDIM)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

!     redistribute points

!     prepare mapping
!     fill in mid
         do il=1,npts
            mprocid(il) = NID
         enddo
!     set imbalance
         nptimb = 0

         call pts_map_all(mprocid,lt2,npts,npoints,nptimb)

!     fill in redistribution array
         do il=1,npts
            mid(1,il) = mprocid(il)
            mid(2,il) = ixyz(il)
         enddo

!     redistribute points
         call pts_transfer(xyz,NDIM,mid,2,lt2,npts)

!     copy arrays
         do il=1,npts
            ipts(il) = mid(2,il)
         enddo

         call copy(pts,xyz,NDIM*npts)

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='xyz_rdst.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, LHIS, lt2
c$$$         write(10001,*) pts_npt2d, pts_nlev, pts_dir
c$$$         write(10001,*) pts_xmin, pts_dx
c$$$         write(10001,*) NP, pts_nx, pts_ny, pts_nm
c$$$         write(10001,*) pts_nxi, pts_ndx, pts_nyi, pts_ndy
c$$$         do itl1=1,npts
c$$$            write(10001,*) itl1, mid(2,itl1), mid(1,itl1),
c$$$     $           (xyz(itl2,itl1),itl2 = 1,LDIM)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

      else                      ! pts_glmax.gt.LHIS

         if (LHIS.gt.lt2) then
            if(NID.eq.0) write(6,*) 
     $           'ERROR: stat_pts_in; LHIS greater than lt2'
            call exitt
         endif

!     load 2D mesh to pts array
         call stat_pts_read(pts,LHIS,mid,lt2,pts_nx,pts_ny,pts_dir,
     $        pts_nxi,pts_ndx,pts_npt2d,pts_nlev,ifascii)

!     generate whole 3D distribution
         call stat_pts_fill(pts,ipts,LHIS,pts_dir,pts_npt2d,pts_nxi,
     $        pts_ndx,pts_nyi,pts_ndy,pts_xmin,pts_dx)


c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='pts_in.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, LHIS, lt2
c$$$         write(10001,*) pts_npt2d, pts_nlev, pts_dir
c$$$         write(10001,*) pts_xmin, pts_dx
c$$$         write(10001,*) NP, pts_nx, pts_ny, pts_nm
c$$$         write(10001,*) pts_nxi, pts_ndx, pts_nyi, pts_ndy
c$$$         do itl1=1,npts
c$$$            write(10001,*) itl1, ipts(itl1), mid(1,itl1),
c$$$     $           (pts(itl2,itl1),itl2 = 1,LDIM)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

!     redistribute points

!     prepare mapping
!     fill in mid
         do il=1,npts
            proc(il) = NID
         enddo
!     set imbalance
         nptimb = 0

         call pts_map_all(proc,LHIS,npts,npoints,nptimb)

!     fill in redistribution array
         do il=1,npts
            mid(1,il) = proc(il)
            mid(2,il) = ipts(il)
         enddo

!     redistribute points
         call pts_transfer(pts,NDIM,mid,2,LHIS,npts)

!     copy arrays
         do il=1,npts
            ipts(il) = mid(2,il)
         enddo

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='pts_rdst.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, LHIS, lt2
c$$$         write(10001,*) pts_npt2d, pts_nlev, pts_dir
c$$$         write(10001,*) pts_xmin, pts_dx
c$$$         write(10001,*) NP, pts_nx, pts_ny, pts_nm
c$$$         write(10001,*) pts_nxi, pts_ndx, pts_nyi, pts_ndy
c$$$         do itl1=1,npts
c$$$            write(10001,*) itl1, ipts(itl1), mid(1,itl1),
c$$$     $           (pts(itl2,itl1),itl2 = 1,LDIM)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

      endif                     ! pts_glmax.gt.LHIS

!     initialise findpts

cc MA: note the different name of the routine:
cc MA:      call intpts_setup(-1.0,inth_hpts) ! use default tolerance

cc MA: too recent...
cc      call intp_setup(-1.0)

cc MA: December version:
      call intpts_setup(-1.0,inth_hpts) ! use default tolerance


!     find point-element mapping and reshuffle the points

!     run findpts to find point-node distribution and to check 
!     the number of failures
      call findpts(inth_hpts,rcode,1,
     &     proc,1,
     &     elid,1,
     &     rst,NDIM,
     &     dist,1,
     &     pts(1,1),NDIM,
     &     pts(2,1),NDIM,
     &     pts(3,1),NDIM,npts)

      nfail = 0
      do il=1,npts
!     check return code 
         if(rcode(il).eq.1) then
            if (dist(il).gt.1e-12) then
               nfail = nfail + 1
               if (nfail.le.5) write(6,'(a,1p4e15.7)') 
     &      ' WARNING: point on boundary or outside the mesh xy[z]d^2:'
     &              ,(pts(kl,il),kl=1,NDIM),dist(il)
            endif   
         elseif(rcode(il).eq.2) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p3e15.7)') 
     &           ' WARNING: point not within mesh xy[z]: !',
     &           (pts(kl,il),kl=1,NDIM)
         endif
      enddo

      nfail = iglsum(nfail,1)
      if (nfail.gt.0) then
         if (NIO.eq.0) write(6,*) 'Error: stat_pts; non-zero nfail'
         call exitt
      endif

c$$$!     test performance
c$$$!     set imbalance
c$$$         nptimb = 0
c$$$         do il=1,4
c$$$            call pts_timing(nptimb)
c$$$!     increase imbalance
c$$$            nptimb =  nptimb + 10
c$$$         enddo

!     redistribue points
      nptimb = 0
      call pts_rdst(nptimb)

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='pts_fnl.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, LHIS
c$$$         do itl1=1,npts
c$$$            write(10001,*) itl1, ipts(itl1),
c$$$     $           (pts(itl2,itl1),itl2 = 1,LDIM)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

!     stamp the log
      if(NIO.eq.0) write(6,*) 'Point time history init.; end'


      return
      end
c-----------------------------------------------------------------------
      subroutine stat_pts_compute(lvel,vort,p0)
c
c     evaluate velocity, pressure and vorticity
c     for list of points (2D distribution read from stat_pts.in and 
c     extended in uniform direction) and dump results
c     into a file (stat_pts.out).
c     I/O modified
c
      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
      include 'PTSTAT'          ! point time series for 2D statistics

!     arrays to save V[XYZ], PR, VORTICITY[XYZ]
!     to save some memory and adjust to statistics code
      real wrk(LX1*LY1*LZ1*LELT)
      common /outtmp/ wrk

!     argument list
      real lvel(LX1,LY1,LZ1,LELT,LDIM) ! velocity array
      real vort(LX1,LY1,LZ1,LELT,LDIM) ! vorticity
      real p0(LX1,LY1,LZ1,LELT) ! mapped pressure

!     local variables
      integer ntot, nxyz        ! array sizes
      integer i                 ! loop index
      integer ifld              ! field number

      nxyz  = NX1*NY1*NZ1
      ntot  = nxyz*NELV
      if(NIO.eq.0) write(6,*) 'collect history points'

!     evaluate fields start
!     VX
      ifld = 1
      call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &     rcode,1,
     &     proc,1,
     &     elid,1,
     &     rst,NDIM,npts,
     &     lvel(1,1,1,1,1))
c$$$!     testing begin
c$$$      do i=1,npts
c$$$         fieldout(ifld,i) = pts(1,i)
c$$$      enddo
c$$$!     testing end

!     VY
      ifld = ifld +1
      call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &     rcode,1,
     &     proc,1,
     &     elid,1,
     &     rst,NDIM,npts,
     &     lvel(1,1,1,1,2))
c$$$!     testing begin
c$$$      do i=1,npts
c$$$         fieldout(ifld,i) = pts(2,i)
c$$$      enddo
c$$$!     testing end

!     VZ
      if (IF3D) then
         ifld = ifld +1
         call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &        rcode,1,
     &        proc,1,
     &        elid,1,
     &        rst,NDIM,npts,
     &        lvel(1,1,1,1,3))
c$$$!     testing begin
c$$$         do i=1,npts
c$$$            fieldout(ifld,i) = pts(3,i)
c$$$         enddo
c$$$!     testing end
      endif

!     pressure
      ifld = ifld +1
      call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &     rcode,1,
     &     proc,1,
     &     elid,1,
     &     rst,NDIM,npts,
     &     p0)
c$$$!     testing begin
c$$$      do i=1,npts
c$$$         fieldout(ifld,i) = ipts(i)
c$$$      enddo
c$$$!     testing end

!     curlx
      if (IF3D) then
         ifld = ifld +1
         call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &        rcode,1,
     &        proc,1,
     &        elid,1,
     &        rst,NDIM,npts,
     &        vort(1,1,1,1,1))
c$$$!     testing begin
c$$$         do i=1,npts
c$$$            fieldout(ifld,i) = NID
c$$$         enddo
c$$$!     testing end
      endif

!     curly
      if (IF3D) then
         ifld = ifld +1
         call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &        rcode,1,
     &        proc,1,
     &        elid,1,
     &        rst,NDIM,npts,
     &        vort(1,1,1,1,2))
      endif

!     curlz
      ifld = ifld +1
      call findpts_eval(inth_hpts,fieldout(ifld,1),nfldm,
     &     rcode,1,
     &     proc,1,
     &     elid,1,
     &     rst,NDIM,npts,
     &     vort(1,1,1,1,3))
!     evaluate fields end

!     write interpolation results to hpts.out
      call stat_pts_out(fieldout,ifld,npts,npoints)

      if(NIO.eq.0) write(6,*) 'done :: collect history points'

      return
      end
c-----------------------------------------------------------------------
!     read 2D points from the file
      subroutine stat_pts_read(pts,lpts,ivdum,lvdum,pts_nx,pts_ny,
     $     pts_dir, pts_nxi,pts_ndx,pts_npt2d,pts_nlev,ifascii)

      implicit none 

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'

!     argument list
      real    pts(LDIM,lpts)    ! point list
      integer lpts              ! array size
      integer ivdum(2*lvdum)    ! buffer for reading
      integer lvdum             ! array size
      integer pts_nx            ! processors in 2D slice
      integer pts_ny            ! processors in uniform direction
      integer pts_dir           ! uniform direction
      integer pts_nxi,pts_ndx   ! collumns to read
      integer pts_npt2d         ! point number in 2D slice
      integer pts_nlev          ! number of layers in uniform direction
      logical ifascii           ! file format

!     local variables
      integer ip, il, ik        ! loop index
      integer idum, idum1, idum2, idum3 ! dummy variable
      integer maxrd, mread, iread ! reading order
      real rdum                 ! dummy variable
      integer msg_id            ! message id for non-blocking send
      integer len               ! buffer lenght

!     file reading
      integer ierr              ! error mark
      character*132 fname       ! file name
      character*132 hdr         ! header
      integer wdsizli           ! single/double precision
      real*4 test_pattern       ! big/little endian test
      logical ifbswap           ! big/little endian
      integer itmp_dx, itmp_dm, itmp_dp ! dummy variables

!     functions
      integer iglsum, irecv
      logical if_byte_swap_test

!     zero array
      ip = LDIM*lpts
      call rzero(pts,ip)

!     first pts_nx processors reads from the file and send messages
!     receiving nodes; non-blocking communication
      if (NID.ge.pts_nx.and.pts_ndx.gt.0) then
         len = wdsize*NDIM*pts_ndx
         msg_id=irecv(pts_nxi,pts,len)
      endif
      call nekgsync()

      if (ifascii) then
!     text file

!     first pts_nx processors reads from the file
         maxrd = 32             ! max # procs to read at once
         mread = (pts_nx-1)/maxrd+1 ! mod param
         iread = 0              ! mod param

         do ip=0,pts_nx-1,maxrd ! loop over processors
            call nekgsync()
            if (NID.lt.pts_nx) then
               if (pts_ndx.gt.0) then
                  if ((mod(NID,mread).eq.iread)) then

!     no need for error check
                     open(50,file='stat_pts.in',status='old',ERR=400)
                     read(50,*,ERR=400) idum
                     read(50,*,ERR=400) rdum

!     skeep unnecessary lines
                     do il=1,pts_nxi-1
                        read(50,*,ERR=500,END=600) rdum
                     enddo
!     read required lines
                     if (pts_dir.eq.1) then
!     x - unuform direction; read y,z
                        do il=1,pts_ndx
                           read(50,*,ERR=500,END=600)
     $                          pts(2,il),pts(3,il)
                        enddo
                     elseif (pts_dir.eq.2) then
!     y - unuform direction; read x,z
                        do il=1,pts_ndx
                           read(50,*,ERR=500,END=600)
     $                          pts(1,il),pts(3,il)
                        enddo
                     else
!     z - unuform direction; read x,y
                        do il=1,pts_ndx
                           read(50,*,ERR=500,END=600)
     $                          pts(1,il),pts(2,il)
                        enddo
                     endif
                     close(50)

                  endif         ! processor group
               endif            ! pts_ndx.gt.0
            endif               ! NID.lt.pts_nx
            iread = iread + 1
         enddo                  ! loop over processors

      else                      ! ifascii
!     binary file

!     this time only pts_nx-1 reads from the file and sends data to 
!     nodes with smaller nid
!     open the file
         ierr = 0
         if (NID.eq.(pts_nx-1)) then

            fname = 'stat_pts.in'//char(0)
         call byte_open(fname,ierr)
            if (ierr.eq.0) then
!     integer header
               call byte_read(hdr,10,ierr)
               if (ierr.ne.0) goto 251
               read (hdr,150) ip, ip, ip, wdsizli
 150           format(i11,i11,i3,i3)
               wdsizli = wdsizli/4

!     big/little endian test
               il = 1
               call byte_read(test_pattern,il,ierr)
               if (ierr.ne.0) goto 251
               ifbswap = if_byte_swap_test(test_pattern,ierr)
               if (ierr.ne.0) goto 251

!     real part
               il=wdsizli*2
               call byte_read(ivdum,il,ierr)
               if (ierr.ne.0) goto 251

            endif               ! ierr.eq.0

 251        continue
         endif                  ! NID.eq.(pts_nx-1)

!     error check
         ierr=iglsum(ierr,1)
         if(ierr.gt.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error opening stat_pts.in in stat_pts_read'
            call exitt
         endif

!     broadcast
         call ibcastn(wdsizli,1,pts_nx-1)
         if (ifbswap) then
            il = 1
         else
            il = 0
         endif
         call ibcastn(il,1,pts_nx-1)
         if (il.eq.1) then
            ifbswap = .TRUE.
         else
            ifbswap = .FALSE.
         endif

!     receiving nodes; non-blocking communication
         ierr = 0
         if (NID.lt.(pts_nx-1).and.pts_ndx.gt.0) then
            len = 4*wdsizli*2*pts_ndx
            msg_id=irecv(pts_nxi,ivdum,len)
         endif
         call nekgsync()

!     read and distribute points
         if (NID.eq.(pts_nx-1)) then

!     read point to distribute
!     for point number caculation
            itmp_dx = pts_npt2d/pts_nx
            itmp_dm = mod(pts_npt2d,pts_nx)
            il = 1
!     node buffer
            idum1 = 0

            do ip=0,NID-1

!     check how many nodes one can buffer
               idum3 = 0
               do ik=idum1,NID-1
!     how many points
                  itmp_dp = mod(ik,pts_nx)

                  if (itmp_dp.lt.itmp_dm) then
                     idum = itmp_dx + 1
                  else
                     idum = itmp_dx
                  endif
                  idum3 = idum3 + idum
!     check buffer size
                  if ((wdsizli*idum3).lt.lvdum) then
                     idum2 = ik
                  else
!     remove last node
                     idum3 = idum3 - idum
                     goto 260
                  endif
               enddo

 260           continue

!     check limits
               if (idum1.eq.idum2.and.(wdsizli*idum3).ge.lvdum) then
                  ierr = 1
                  goto 280
               endif

               if (idum3.gt.0) then
!     read buffer
                  len = wdsizli*2*idum3
                  call byte_read(ivdum,len,ierr)

!     redistribute data form buffer
                  iread = 1
                  do ik=idum1,idum2

!     how many points
                     itmp_dp = mod(ik,pts_nx)

                     if (itmp_dp.lt.itmp_dm) then
                        idum = itmp_dx + 1
                     else
                        idum = itmp_dx
                     endif

                     if (idum.gt.0) then

!     send the buffer
                        len = 4*wdsizli*2*idum
                        call csend(il,ivdum(iread),len,ik,0)

!     update array position
                        iread = iread + wdsizli*2*idum
!     update message tag
                        il = il+idum

                     endif      ! idum.gt.0
                  enddo         ! ik
               endif            ! idum3.gt.0

!     check and update limits
               if (idum2.eq.(NID-1)) goto 270
               idum1 = idum2 +1
               
            enddo               ! ip

 270        continue

!     read own points
            if(pts_ndx.gt.0) then
               il = wdsizli*2*pts_ndx
               call byte_read(ivdum,il,ierr)
            endif

!     close file
            call byte_close(ierr)
            if (ierr.ne.0.and.NIO.eq.0) write(6,*)
     $              'Error closing stat_pts.in in stat_pts_read'

         endif

!     error check
 280     continue
         ierr = iglsum(ierr,1)
         if (ierr.gt.0) then
            if (NIO.eq.0)
     $           write(6,*) 'Error: stat_pts_read lvdum too small'
            call exitt
         endif

         if (NID.lt.(pts_nx-1).and.pts_ndx.gt.0) then
!     receive messages from node pts_nx-1
            call msgwait(msg_id)
         endif

         ierr = 0
!     copy variables
         if (NID.lt.pts_nx.and.pts_ndx.gt.0) then
            if (wdsizli.eq.2) then
!     big/little endian
               if (ifbswap) then
                  len = wdsizli*2*pts_ndx
                  call byte_reverse8(ivdum,len,ierr)
                  if (ierr.ne.0) goto 301
               endif
               if (pts_dir.eq.1) then
!     x - unuform direction; read y,z
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy(pts(2,il),ivdum(idum),1)
                     call copy(pts(3,il),ivdum(idum+wdsizli),1)
                  enddo
               elseif (pts_dir.eq.2) then
!     y - unuform direction; read x,z
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy(pts(1,il),ivdum(idum),1)
                     call copy(pts(3,il),ivdum(idum+wdsizli),1)
                  enddo
               else
!     z - unuform direction; read x,y
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy(pts(1,il),ivdum(idum),1)
                     call copy(pts(2,il),ivdum(idum+wdsizli),1)
                  enddo
               endif
            elseif (wdsizli.eq.1) then
!     big/little endian
               if (ifbswap) then
                  len = wdsizli*2*pts_ndx
                  call byte_reverse(ivdum,len,ierr)
                  if (ierr.ne.0) goto 301
               endif
               if (pts_dir.eq.1) then
!     x - unuform direction; read y,z
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy4r(pts(2,il),ivdum(idum),1)
                     call copy4r(pts(3,il),ivdum(idum+wdsizli),1)
                  enddo
               elseif (pts_dir.eq.2) then
!     y - unuform direction; read x,z
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy4r(pts(1,il),ivdum(idum),1)
                     call copy4r(pts(3,il),ivdum(idum+wdsizli),1)
                  enddo
               else
!     z - unuform direction; read x,y
                  do il=1,pts_ndx
                     idum = (il-1)*wdsizli*2 +1
                     call copy4r(pts(1,il),ivdum(idum),1)
                     call copy4r(pts(2,il),ivdum(idum+wdsizli),1)
                  enddo
               endif
            endif
         endif                  ! NID.lt.pts_nx.and.pts_ndx.gt.0

 301     continue
         ierr=iglsum(ierr,1)
         if(ierr.gt.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error: stat_pts_read; wrong byte swap'
            call exitt
         endif
      endif                     ! ifascii

!     redistribute data
      if (NID.lt.pts_nx) then
!     sending nodes
         if(pts_ndx.gt.0) then

!     level division
            itmp_dx = pts_nlev/pts_ny
            itmp_dm = mod(pts_nlev,pts_ny)
            
            do ip=2,pts_ny
               idum = NID + (ip-1)*pts_nx

               itmp_dp = idum/pts_nx
               if (itmp_dp.lt.itmp_dm) then
                  len = itmp_dx + 1
               else
                  len = itmp_dx
               endif

               if(len.gt.0) then
                  len = wdsize*NDIM*pts_ndx
                  call csend(pts_nxi,pts,len,idum,0)
               endif
            enddo
         endif
      elseif (pts_ndx.gt.0) then
!     receiving nodes
         call msgwait(msg_id)
      endif

      return

 400  CONTINUE
      if(NIO.eq.0) write(6,*)
     $     'Error opening stat_pts.in in stat_pts_read'
      call exitt

 500  CONTINUE
      if(NIO.eq.0) write(6,501) pts_nxi+il-1
 501  FORMAT(2X,'ERROR reading 2D point data at point ',I12
     $    ,/,2X,'ABORTING in routine stat_pts_read.')
      call exitt

 600  CONTINUE
      if(NIO.eq.0) write(6,601) pts_nxi+il-1
 601  FORMAT(2X,'ERROR 2 reading 2D point data at point ',I12
     $    ,/,2X,'ABORTING in routine stat_pts_read.')
      call exitt

      return
      end
c-----------------------------------------------------------------------
!     fill missing coordinates based on pts_dir
      subroutine stat_pts_fill(pts,ipts,lpts,pts_dir,pts_npt2d,pts_nxi,
     $     pts_ndx,pts_nyi,pts_ndy,pts_xmin,pts_dx)

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'

!     argument list
      real    pts(LDIM,lpts)    ! point list
      integer ipts(lpts)        ! global ordering
      integer lpts              ! array size
      integer pts_dir           ! uniform direction
      integer pts_npt2d         ! pts number in single layer
      integer pts_ndx ,pts_nxi  ! loop arguments
      integer pts_ndy, pts_nyi  ! loop arguments
      real pts_xmin,pts_dx      ! point dist in uniform direction

!     local variables
      integer il, ip            ! loop index
      integer itmp, itmpgl      ! dummy variable
      real rtmp                 ! dummy variable


      call izero(ipts,lpts)

      if (pts_ndx.gt.0.and.pts_ndy.gt.0) then
         if (pts_dir.eq.1) then
!     first layer
!     x - unuform direction; y,z already set
!     fill x on a first layer
            rtmp = pts_xmin + (pts_nyi -1)*pts_dx
!     global ordering
            itmpgl = (pts_nyi-1)*pts_npt2d + pts_nxi -1
            do ip=1,pts_ndx
!     coordinates
               pts(1,ip) = rtmp
!     global ordering
               ipts(ip) = itmpgl +ip
            enddo
!     the rest
!     copy y,z coordinates and fill x at the rest of layers
            do il=2,pts_ndy
               rtmp = rtmp + pts_dx
               itmpgl = itmpgl + pts_npt2d
               do ip=1,pts_ndx
                  itmp = (il -1)*pts_ndx + ip
!     coordinates
                  pts(1,itmp) = rtmp
                  pts(2,itmp) = pts(2,ip)
                  pts(3,itmp) = pts(3,ip)
!     global ordering
                  ipts(itmp) = itmpgl +ip
               enddo
            enddo

         elseif (pts_dir.eq.2) then
!     first layer
!     y - unuform direction; x,z already set
!     fill y on a first layer
            rtmp = pts_xmin + (pts_nyi -1)*pts_dx
!     global ordering
            itmpgl = (pts_nyi-1)*pts_npt2d + pts_nxi -1
            do ip=1,pts_ndx
!     coordinates
               pts(2,ip) = rtmp
!     global ordering
               ipts(ip) = itmpgl +ip
            enddo
!     the rest
!     copy x,z coordinates and fill y at the rest of layers
            do il=2,pts_ndy
               rtmp = rtmp + pts_dx
               itmpgl = itmpgl + pts_npt2d
               do ip=1,pts_ndx
                  itmp = (il -1)*pts_ndx + ip
!     coordinates
                  pts(2,itmp) = rtmp
                  pts(1,itmp) = pts(1,ip)
                  pts(3,itmp) = pts(3,ip)
!     global ordering
                  ipts(itmp) = itmpgl +ip
               enddo
            enddo

         else
!     first layer
!     z - unuform direction; x,y already set
!     fill z on a first layer
            rtmp = pts_xmin + (pts_nyi -1)*pts_dx
!     global ordering
            itmpgl = (pts_nyi-1)*pts_npt2d + pts_nxi -1
            do ip=1,pts_ndx
!     coordinates
               pts(3,ip) = rtmp
!     global ordering
               ipts(ip) = itmpgl +ip
            enddo
!     the rest
!     copy y,z coordinates and fill x at the rest of layers
            do il=2,pts_ndy
               rtmp = rtmp + pts_dx
               itmpgl = itmpgl + pts_npt2d
               do ip=1,pts_ndx
                  itmp = (il -1)*pts_ndx + ip
!     coordinates
                  pts(3,itmp) = rtmp
                  pts(1,itmp) = pts(1,ip)
                  pts(2,itmp) = pts(2,ip)
!     global ordering
                  ipts(itmp) = itmpgl +ip
               enddo
            enddo

         endif
      endif

      return
      end
c-----------------------------------------------------------------------
!     Output points for statistics point time history
!     To reduce number of wirting to the disc I collect some number of 
!     time snapshots
!     and later on write the whole set to the disc
      subroutine stat_pts_out(fieldout,nflds,npts,npoints)

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'

      integer lfldm             ! max number of fields
      parameter(lfldm=2*LDIM+1)

!     arguments
      real fieldout(lfldm,LHIS)
      integer nflds             ! number of fields
      integer npts              ! local number of points
      integer npoints           ! global number of points

!     local variables
      integer ltsnap            ! number of snapshots (MA: per pts file?) 
      parameter (ltsnap = 100)
      real buffer(lfldm,LHIS,ltsnap) ! buffer for snapshots
      real tmlist(ltsnap)       ! snapshot time
      save buffer, tmlist
      integer istcount          ! step interval for data collection

      integer icall             ! call counter
      save icall
      data icall /0/

      integer isize             !
      parameter (isize = lfldm*LHIS)
      
c$$$!     for testing
c$$$      integer itl1, itl2, itl3
c$$$      character*2 str

!     collect data
!     count calls
   
      icall = icall + 1
      call copy(buffer(1,1,icall),fieldout,isize)
      tmlist(icall) = TIME

!     this is directly related to statistics code; probably should 
!     be changed in the future
      istcount = int(PARAM(51))
      
      if(NIO.eq.0) write(6,*) 'YN Check:',icall,ltsnap,istcount   
!     buffer is full or this is the last step; save data
      if(icall.eq.ltsnap.or.(NSTEPS-ISTEP).lt.istcount) then
         if(NIO.eq.0) write(6,*) 'dump history points'

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='pts_out.txt'//str)
c$$$         write(10001,*) NID, npoints, npts, nflds, icall
c$$$         do itl3=1,icall
c$$$            write(10001,*) itl3, tmlist(itl3)
c$$$            do itl1=1,npts
c$$$               write(10001,*) itl1, (buffer(itl2,itl1,itl3),itl2=1,
c$$$     $              nflds)
c$$$            enddo
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

         call stat_mfo_outpts(buffer,tmlist,ltsnap,nflds,icall)

!     reset counter
         icall = 0
      endif

      return
      end
c-----------------------------------------------------------------------
! ! MA: substituted by intp_setup in Nek5000/core/intp.f
cc MA: copied from nek1093_dong/trunk/nek/postpro.f
c-----------------------------------------------------------------------
! !       subroutine intpts_setup(tolin,ih)
! ! c
! ! c setup routine for interpolation tool
! ! c tolin ... stop point seach interation if 1-norm of the step in (r,s,t) 
! ! c           is smaller than tolin 
! ! c
! !       include 'SIZE'
! !       include 'GEOM'
! ! 
! !       common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal
! ! 
! !       tol = tolin
! !       if (tolin.lt.0) tol = 1e-13 ! default tolerance 
! ! 
! !       n       = lx1*ly1*lz1*lelt 
! !       npt_max = 256
! !       nxf     = 2*nx1 ! fine mesh for bb-test
! !       nyf     = 2*ny1
! !       nzf     = 2*nz1
! !       bb_t    = 0.1 ! relative size to expand bounding boxes by
! ! c
! !       if(nidd.eq.0) write(6,*) 'initializing intpts(), tol=', tol
! !       call findpts_setup(ih,nekcomm,npp,ndim,
! !      &                     xm1,ym1,zm1,nx1,ny1,nz1,
! !      &                     nelt,nxf,nyf,nzf,bb_t,n,n,
! !      &                     npt_max,tol)
! ! c       
! !       return
! !       end
! ! c-----------------------------------------------------------------------
