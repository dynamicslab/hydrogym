!=======================================================================
! Name        : statistics_2D
! Author      : Prabal Singh Negi, Adam Peplinski
! Version     : last modification 2015.05.22
! Copyright   : GPL
! Description : This is a set of routines to calculate 2D statistics
!=======================================================================
c----------------------------------------------------------------------
!     Driver for statistics computation
      subroutine stat_avg_all

!      implicit none    ! TSTEP does not have declarations

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'STATS'

      integer istcount
      save istcount
      integer iocount
      save iocount

      if (ISTEP.eq.0) then
!     take step interval for data collection
!     this is directly related to the old code;
!     probably should be changed later
            istcount = int(PARAM(51))
            iocount  = int(PARAM(50))

!     Initialize 2D statistics
            call stat_init

            STAT_ATIME = 0.
            STAT_TSTART = time

!     Debugging only
            call vis_ownership

      else
!           average
            if (mod(ISTEP,istcount).eq.0) call stat_compute
            if (mod(ISTEP,iocount).eq.0) then

                  call stat_gl_collation        ! Communication
                  call stat_mfo_outfld2D        ! Output
                  STAT_ATIME = 0.
                  STAT_TSTART = time

            endif
      endif

      if (ISTEP.eq.NSTEPS) call stat_end        ! finalize

      return
      end subroutine stat_avg_all

!====================================================================== 

!     main interface for statistics initialisation
      subroutine stat_init()
      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables
      include 'INPUT'           ! if3d

!     work arrays
      integer lctrs1 ,lctrs2    ! array sizes
      parameter (lctrs1=3,lctrs2=2*LX1*LY1*LZ1*LELT)
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      integer owner(lctrs2)     ! mark node with smallest id
      logical ifseg(lctrs2)     ! segment borders
      common /SCRNS/ ctrs
      common /SCRVH/ ifseg
      common /SCRUZ/ cell, ninseg, ind, owner


C     COMMON /CTMP0/ DUMMY0(LX1,LY1,LZ1,LELT,2)
C     COMMON /CTMP1/ DUMMY1(LX1,LY1,LZ1,LELT,4)
C     COMMON /SCRNS/ DUMMY2(LX1,LY1,LZ1,LELT,7)
C     COMMON /SCRUZ/ DUMMY3(LX1,LY1,LZ1,LELT,4)
C     COMMON /SCREV/ DUMMY4(LX1,LY1,LZ1,LELT,2)
C     COMMON /SCRVH/ DUMMY5(LX1,LY1,LZ1,LELT,2)
C     COMMON /SCRMG/ DUMMY6(LX1,LY1,LZ1,LELT,4)
C     COMMON /SCRCH/ DUMMY7(LX1,LY1,LZ1,LELT,2)
!     used in stat_init_int1D
C     COMMON /SCRSF/ DUMMY8(LX1,LY1,LZ1,LELT,3)
!     used in stat_init_coord
C     COMMON /CTMP0/ DUMMY0(LX1,LY1,LZ1,LELT,2)

      integer nseg              ! segments number

      integer i                 ! loop index

!     simple timing
      real ltim_init

!     functions
      real dnekclock

!     stamp logs
      if (NIO.eq.0) write(6,*) '2D statistics initialisation'
!     simple timing
      ltim_init = dnekclock()

!     reset timing and counters
      STAT_TINI = 0.0
      STAT_TEV = 0.0
      STAT_TCMM = 0.0
      STAT_TIO = 0.0
      STAT_ION = 0
      STAT_CNN = 0
      STAT_EVN = 0

!     generate local 3D => 2D mapping
      if (NIO.eq.0) write(6,*) 'Calls; stat_init_local'
      call stat_init_local(ctrs,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg)

!     reshuffle coordinate arrays
      if (NIO.eq.0) write(6,*) 'Calls; stat_init_coord'
      call stat_init_coord

!     generate local => global mapping
!     fill ownership array
      do i=1,nseg
         cell(i) = i
         owner(i) = NID
      enddo
      if (NIO.eq.0) write(6,*) 'Calls; stat_init_global'
      call stat_init_global(ctrs,owner,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg)

!     create communicator
!     will be used by stat_init_int1D
!     to be added
c------------------------------ 
!     Communication map
      call stat_comm_map

!      call stat_output_mapping
!      call exitt
!      stat_tstart=time
!      stat_atime=0.


!     get local integration coefficients
      if (NIO.eq.0) write(6,*) 'Calls; stat_init_int1D'
      call stat_init_int1D

!     initalisation of point time series
!     this routine uses number of scratch common blocks
      if (NIO.eq.0) write(6,*) 'Calls; stat_pts_init'
      if (if3d) then
!          call stat_pts_init                              !not for 2D yet
      endif
!     simple timing
      ltim_init = dnekclock() - ltim_init

!     save timing
      STAT_TINI = ltim_init

      return
      end
c----------------------------------------------------------------------
!     main interface for statistics finalisation
      subroutine stat_end()
      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables

!     local variables
!     simple timing
      real ltim_min, ltim_max

!     functions
      real glmax, glmin

!     should one check if all the data are wirtten?

!     simple timing
      ltim_min  = glmin(STAT_TINI,1)
      ltim_max  = glmax(STAT_TINI,1)
!     stamp logs
      if (NIO.eq.0) then
         write(6,*) '2D statistics timning:'
         write(6,*) 'initialisation min/max: ',ltim_min, ltim_max
      endif
      ltim_min  = glmin(STAT_TEV,1)
      ltim_max  = glmax(STAT_TEV,1)
!     stamp logs
      if (NIO.eq.0) then
         write(6,*) 'evolution min/max:      ',ltim_min, ltim_max,
     $        STAT_EVN
      endif
      ltim_min  = glmin(STAT_TCMM,1)
      ltim_max  = glmax(STAT_TCMM,1)
!     stamp logs
      if (NIO.eq.0) then
         write(6,*) 'communication min/max:  ',ltim_min, ltim_max,
     $        STAT_CNN
      endif

      ltim_min  = glmin(STAT_TIO,1)
      ltim_max  = glmax(STAT_TIO,1)
!     stamp logs
      if (NIO.eq.0) then
         write(6,*) 'I/O min/max:            ',ltim_min, ltim_max,
     $        STAT_ION
      endif

      return
      end
c----------------------------------------------------------------------
!     generate local 3D => 2D mapping
      subroutine stat_init_local(ctrs,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg)
      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'INPUT'           ! [XYZ]C
      include 'STATS'           ! 2D statistics speciffic variables

!     argument list
      integer lctrs1,lctrs2     ! array sizes
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer nseg              ! segments number
!     work arrays
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      logical ifseg(lctrs2)     ! segment borders

!     local variables
      integer e,i,j             ! loop indexes
      integer nelsort           ! number of local 3D elements to sort

!     local sorting
      integer key               ! sorting key
      integer ipass, iseg       ! loop index
      real aa(lctrs1)           ! dummy array

c$$$!     for testing
c$$$      integer itl1, itl2
c$$$      character*2 str

!     We can sort only part of the domain, so first mark and copy
!     all elements in the region you are interested in
!     set uniform direction, cell centres and diagonals
      call user_stat_init(ctrs,cell,lctrs1,lctrs2,nelsort)

!     check array size
      if (lctrs2.lt.nelsort) then
         if (NIO.eq.0) write(6,*) 'Error: stat_init_local; lctrs2'
         call exitt
      endif

!     check array size
      if (NELT.lt.nelsort) then
         if (NIO.eq.0) write(6,*) 'Error: stat_init_local; nelsort'
         call exitt
      endif

!     check uniform direction
      if (STAT_IDIR.gt.3) then
         if (NIO.eq.0) write(6,*) 'Error: stat_init_local; STAT_IDIR'
         call exitt
      endif

!     for every element
      do e=1,nelsort
!     reset segments borders
         ifseg(e) = .FALSE.
      enddo

!     perform local sorting to identify unique set sorting by directions
!     first run => whole set is one segment
      nseg        = 1
      ifseg(1)    = .TRUE.
      ninseg(1)   = nelsort

!     Multiple passes eliminates false positives
      do ipass=1,2
         do j=1,2          ! Sort within each segment !#2D

            i =1
            do iseg=1,nseg
               call tuple_sort(ctrs(1,i),lctrs1,ninseg(iseg),j,1,
     $              ind,aa)     ! key = j
               call iswap_ip(cell(i),ind,ninseg(iseg)) ! Swap position
               i = i + ninseg(iseg)
            enddo
 
            do i=2,nelsort
!     find segments borders
               if (abs(ctrs(j,i)-ctrs(j,i-1)).gt.
     $              STAT_TOL*min(ctrs(3,i),ctrs(3,i-1)))
     $              ifseg(i)=.TRUE.
            enddo

!  Count up number of different segments
            nseg = 0
            do i=1,nelsort
               if (ifseg(i)) then
                  nseg = nseg+1
                  ninseg(nseg) = 1
               else
                  ninseg(nseg) = ninseg(nseg) + 1
               endif
            enddo
         enddo                  ! j=1,2
      enddo                     ! ipass=1,2

!     sorting end

!     generate local 3D => 2D mapping
!     local number of unique
      STAT_LNUM = nseg

!     mark all elements as unwanted
      call ifill(STAT_LMAP,-1,NELT)

!     for all segments
!     count 3D elements
      j=1
      do iseg=1,nseg
!     within segment
         do i=1,ninseg(iseg)
            STAT_LMAP(cell(j)) = iseg
            j=j+1
         enddo
      enddo

!     contract coordinate set
!     for all segments
!     count 3D elements
      j=ninseg(1) +1
      do iseg=2,nseg
         do i = 1,lctrs1
            ctrs(i,iseg) = ctrs(i,j)
         enddo
         j = j + ninseg(iseg)
      enddo

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='local_init.txt'//str)
c$$$         write(10001,*) NID, nseg
c$$$         do itl1=1,nseg
c$$$            write(10001,*) itl1, ctrs(1,itl1),ctrs(2,itl1),ctrs(3,itl1)
c$$$         enddo
c$$$         do itl1=1,NELV
c$$$            write(10001,*) itl1, STAT_LMAP(itl1)
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end


      return
      end
c----------------------------------------------------------------------
!     reshuffle coordinate arrays
      subroutine stat_init_coord
      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables

!     local variables
      integer len               ! buffer size
      integer i                 ! loop index
      integer el                ! destination element
      integer imark(LELT)       ! element mark
      real rtmp(STAT_LM1,STAT_LM1,LELT,2) ! dummy arrays
      common /ctmp0/ rtmp

c$$$!     for testing
c$$$      integer itl1, itl2
c$$$      character*2 str

      len = STAT_LM1*STAT_LM1
      call ifill(imark, -1,NELV)
      do i=1,NELV
         el = STAT_LMAP(i)
         if (el.gt.0) then
            if (imark(el).eq.-1) then
               imark(el) = 1
               call copy(rtmp(1,1,el,1),STAT_XM1(1,1,i),len)
               call copy(rtmp(1,1,el,2),STAT_YM1(1,1,i),len)
            endif
         endif
      enddo

!     copy arrays back
      len = len*STAT_LNUM
      call copy(STAT_XM1,rtmp(1,1,1,1),len)
      call copy(STAT_YM1,rtmp(1,1,1,2),len)

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='usr_init_sort.txt'//str)
c$$$         write(10001,*) NID, STAT_IDIR, NELV,STAT_NM1,STAT_LNUM
c$$$         do itl1=1,NELV
c$$$            write(10001,*) itl1
c$$$            do j=1,NX1
c$$$               do i=1,NX1
c$$$                  write(10001,*) i,j,STAT_XM1(i,j,itl1),
c$$$     $                 STAT_YM1(i,j,itl1)
c$$$               enddo
c$$$            enddo
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

      return
      end
c----------------------------------------------------------------------
!     generate local to global mapping
      subroutine stat_init_global(ctrs,owner,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg)
      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'PARALLEL'
      include 'STATS'           ! 2D statistics speciffic variables

!     argument list
      integer lctrs1,lctrs2     ! array sizes
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer nseg              ! segments number
!     work arrays
      integer owner(lctrs2)     ! mark node with smallest id
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      logical ifseg(lctrs2)     ! segment borders

!     local variables
      integer csteps            ! numer of steps in the cycle
      integer lwork             ! working array size
      integer umrkgl            ! global number of unmarked zones
!     local coppies
      real lctrs(lctrs1,LELT)   ! local copy 2D element centres
      integer nsort             ! number of elements to sort
      integer nsorted           ! number of sorted elements

      integer igpass            ! numer of executed cycles
      integer igpass_max        ! max numer of cycles
      parameter (igpass_max = 100)

      integer icstep            ! loop index

!     communication
      integer msg_id1, msg_id2  ! message id for non-blocking receive
      integer srcid, dstid      ! source and destination node id
      integer len               ! buffer size
      integer cnsort            ! number of elements to receive

!     local sorting
      integer key               ! sorting key
      integer ipass, iseg, i, j, k ! loop index
      real aa(lctrs1)           ! dummy array

!     error mark
      integer ierror

!     functions
      integer iglsum, irecv, iglmin, iglmax

!     for testing
      integer itl1, itl2
      character*2 str

!     get number of steps to exchange all the data in the ring
      csteps=int(log(NP+0.)/log(2.))
      if(NP.gt.2**csteps) csteps=csteps+1

!     check initial number of sections and the size of work arrays
      lwork = lctrs2/2
      if (lwork.lt.nseg) then
         if (NIO.eq.0) write(6,*) 'Error: init_stat_global; lwork'
         call exitt
      endif

!     get global number of unmarked zones
      umrkgl = iglsum(nseg,1)
!     initial number of elements to sort
      nsort = nseg
!     initial number of sorted elements
      nsorted = 0

!     make a local copy of initial set
      if (LELT.lt.nseg) then
         if (NIO.eq.0) write(6,*) 'Error: init_stat_global; nseg'
         call exitt
      endif
      i = lctrs1*nseg
      call copy(lctrs,ctrs,i)

!     check STAT_LNUM
      if (STAT_LNUM.ne.nseg) then
         if (NIO.eq.0) write(6,*) 'Warning: init_stat_global; STAT_LNUM'
         STAT_LNUM = nseg
      endif

!     reset ownership and local => global map
!     this routine will produce the simplest ownership without 
!     taking into account work ballancing
      call ifill(STAT_GMAP,-1,nseg)
      call ifill(STAT_OWN,-1,nseg)

!     following loop has to be executed as long as unmarked zones 
!     exists
!     count global passes
      igpass = 1
      do

!     stamp log
         if (NIO.eq.0) write(6,*) 'Cycle ', igpass,
     $        'globally unmarked = ',umrkgl

!     collect information within the ring
         do icstep=1,csteps

!     exchange information between processors
!     source and destination
            i = 2**(icstep-1)
            srcid = NID - i
            dstid = NID + i
            if (srcid.lt.0) srcid = srcid + NP
            if (dstid.ge.NP) dstid = dstid - NP

!     set buffer for the number of elements to receive
            len = ISIZE
            msg_id1 = irecv(0,cnsort,len)

!     send local size of the buffer
            call csend(0,nsort,len,dstid,0)

!     finish communication
            call msgwait(msg_id1)

!     exchange coordinates and ownership
!     receive
            len = WDSIZE*lctrs1*cnsort
            msg_id1 = irecv(1,ctrs(1,nsort+1),len)

            len = ISIZE*cnsort
            msg_id2 = irecv(2,owner(nsort+1),len)

!     send
            len = WDSIZE*lctrs1*nsort
            call csend(1,ctrs,len,dstid,0)

            len = ISIZE*nsort
            call csend(2,owner,len,dstid,0)

!     reset cell for the received elements to -1
!     this way all the non-local elements are marked
            do i=nsort + 1,nsort + cnsort
               cell(i) = -1
            enddo
         
!     finish communication
            call msgwait(msg_id1)
            call msgwait(msg_id2)

!     update number of elements to sort
            nsort = nsort + cnsort

!     perform local sorting to identify unique set
!     sorting by directions
!     reset section boudarry mark
            do i=1,nsort
               ifseg(i) = .FALSE.
            enddo
!     first run => whole set is one segment
            nseg        = 1
            ifseg(1)    = .TRUE.
            ninseg(1)   = nsort

! Multiple passes eliminates false positives
            do ipass=1,2
               do j=1,2    ! Sort within each segment !#2D

                  i =1
                  do iseg=1,nseg
                     call tuple_sort(ctrs(1,i),lctrs1,ninseg(iseg),j,1,
     $                    ind,aa) ! key = j
!     Swap position
                     call iswap_ip(cell(i),ind,ninseg(iseg)) 
                     call iswap_ip(owner(i),ind,ninseg(iseg))
                     i = i + ninseg(iseg)
                  enddo
 
                  do i=2,nsort
!     find segments borders
                     if (abs(ctrs(j,i)-ctrs(j,i-1)).gt.
     $                    STAT_TOL*min(ctrs(3,i),ctrs(3,i-1)))
     $                    ifseg(i)=.TRUE.
                  enddo

!     Count up number of different segments
                  nseg = 0      
                  do i=1,nsort
                     if (ifseg(i)) then
                        nseg = nseg+1
                        ninseg(nseg) = 1
                     else
                        ninseg(nseg) = ninseg(nseg) + 1
                     endif
                  enddo
               enddo            ! j=1,2
            enddo               ! ipass=1,2

!     local sorting end

!     contract coordinate set
!     for all segments
            j=ninseg(1) +1
            do iseg=2,nseg
               do i = 1,lctrs1
                  ctrs(i,iseg) = ctrs(i,j)
               enddo
               j = j + ninseg(iseg)
            enddo
!     contract ownership
!     for all segments
            j=1
            do iseg=1,nseg
               owner(iseg) = owner(j)
               j = j+1
!     within segment
               do i=2,ninseg(iseg)
                  if (owner(iseg).gt.owner(j)) owner(iseg) = owner(j)
                  j=j+1
               enddo
            enddo
!     contract cell
            ierror = 0
!     for all segments
            j=1
            do iseg=1,nseg
               cell(iseg) = cell(j)
               j = j+1
!     within segment
!     for checking consistency
!     in every section can be only 1 non negative cell entrance
               k = 0
               if (cell(iseg).ne.-1) k = k+1
               do i=2,ninseg(iseg)
                  if (cell(iseg).lt.cell(j)) cell(iseg) = cell(j)
                  if (cell(j).ne.-1) k = k+1
                  j=j+1
               enddo
               if (k.gt.1) ierror = ierror +1
            enddo

!     check consistency
            ierror = iglsum(ierror,1)
            if (ierror.gt.0) then
               if (NIO.eq.0) write(6,*) 'Error: init_stat_global; cell'
               call exitt
            endif

!     update number of elements to sort
            nsort = min(nseg,lwork)

         enddo                  ! icstep
!     global exchange and sort end
         ierror = 0
!     mark elements that can be mapped
         do i=1,nsort
            if (cell(i).ne.-1) then
!     check consistency; was this cell mapped previously
               if(STAT_GMAP(cell(i)).ne.-1) then
                  ierror = ierror +1
               endif
               STAT_GMAP(cell(i)) = nsorted + i
               STAT_OWN(cell(i))  = owner(i)
            endif
         enddo

!     check consistency
         ierror = iglsum(ierror,1)
         if (ierror.gt.0) then
            if (NIO.eq.0) write(6,*) 'Error: init_stat_global; cell2'
            call exitt
         endif

!     update number of sorted elements
         nsorted = nsorted + nsort

!     count local unmarked zones
         nseg = 0
         do i=1,STAT_LNUM
            if(STAT_GMAP(i).eq.-1) then
               nseg = nseg + 1
!     fill in coordinates and mark initial ownership
               call copy(ctrs(1,nseg),lctrs(1,i),lctrs1)
               owner(nseg) = NID
               cell(nseg) = i
            endif
         enddo

!     get global number of unmarked zones
         umrkgl = iglsum(nseg,1)

         if (umrkgl.eq.0) goto 500

!     update number of elements to sort
         nsort = nseg

!     count global passes
         igpass = igpass +1

!     is igpass too big; something is wrong exit
         if (igpass.gt.igpass_max) then
            if (NIO.eq.0) write(6,*) 'Error: init_stat_global; igpass'
            call exitt
         endif

      enddo                     ! infinite loop

 500  continue

!     find number of ef elements owned
      STAT_LOWN = 0
      do i=1,STAT_LNUM
         if (STAT_OWN(i).eq.NID) STAT_LOWN = STAT_LOWN + 1
      enddo
!     global number of unique 2D elements
      STAT_GNUM = iglsum(STAT_LOWN,1)
!     imbalance
      j = iglmin(STAT_LOWN,1)
      k = iglmax(STAT_LOWN,1)
      
!     stamp logs
      if (NIO.eq.0) then
         write(6,*) 'Global number of unique 2D elements: ',STAT_GNUM
         write(6,*) 'Owned element imbalance min/max: ',j, k
      endif

!     testing
!         write(str,'(i2.2)') NID
!         open(unit=10001,file='global_init.txt'//str)
!         write(10001,*) NID, STAT_LNUM, STAT_GNUM, STAT_LOWN, igpass
!         do itl1=1,STAT_LNUM
!            write(10001,*) itl1,lctrs(1,itl1),lctrs(2,itl1),
!     $           lctrs(3,itl1),STAT_GMAP(itl1),STAT_OWN(itl1)
!         enddo
!         do itl1=1,NELV
!            write(10001,*) itl1, STAT_LMAP(itl1)
!         enddo
!         close(10001)
!     testing end

      return
      end
c----------------------------------------------------------------------
!     get local integration coefficients
!     This version does 1D integration over one of the directions
!     R,S,T. It supports curved coordinate systems, however 
!     axisymmetric cases are not supported
      subroutine stat_init_int1D()

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'WZ'              ! W?M1
      include 'GEOM'            ! ?M1
      include 'INPUT'           ! IFAXIS
      include 'DXYZ'            ! D?M1, D?TM1
      include 'STATS'           ! 2D statistics speciffic variables

!     scratch space
      real lxyzd(LX1,LY1,LZ1,LELT,3)
      common /SCRSF/ lxyzd      ! coordinate derivatives

!     local variables
      integer i, j, k, e        ! loop index
      integer el                ! index of 2D element
      real lwm1(STAT_LM1)       ! wieghts for 1D integration

      if(IFAXIS) then
         if (NIO.eq.0) then
            write(6,*) 'Error: stat_init_int1D; IFAXIS not supported'
         endif
         call exitt
      endif

!     copy wieghts depending on the uniform direction
      if (STAT_IDIR.eq.1) then
         call copy(lwm1,WXM1,NX1)
         STAT_NM1 = NX1
         STAT_NM2 = NY1
         STAT_NM3 = NZ1
!     get coordinates derivatives d[XYZ]/dr
         i = NY1*NZ1
         do e = 1, NELT
            if(STAT_LMAP(e).ne.-1) then
               call mxm(DXM1,NX1,XM1(1,1,1,e),NX1,lxyzd(1,1,1,e,1),i)
               call mxm(DXM1,NX1,YM1(1,1,1,e),NX1,lxyzd(1,1,1,e,2),i)
               call mxm(DXM1,NX1,ZM1(1,1,1,e),NX1,lxyzd(1,1,1,e,3),i)
            endif
         enddo
      elseif (STAT_IDIR.eq.2) then
         call copy(lwm1,WYM1,NY1)
         STAT_NM1 = NY1
         STAT_NM2 = NX1
         STAT_NM3 = NZ1
!     get coordinates derivatives d[XYZ]/ds
         do e = 1, NELT
            if(STAT_LMAP(e).ne.-1) then
               do i=1, NZ1
                  call mxm(XM1(1,1,i,e),NX1,DYTM1,NY1,
     $                 lxyzd(1,1,i,e,1),NY1)
                  call mxm(YM1(1,1,i,e),NX1,DYTM1,NY1,
     $                 lxyzd(1,1,i,e,2),NY1)
                  call mxm(ZM1(1,1,i,e),NX1,DYTM1,NY1,
     $                 lxyzd(1,1,i,e,3),NY1)
               enddo
            endif
         enddo
      else
         if (if3d) then            ! #2D 
              call copy(lwm1,WZM1,NZ1)
              STAT_NM1 = NZ1
              STAT_NM2 = NX1
              STAT_NM3 = NY1
          else
               call rone(lwm1,STAT_LM1)
               STAT_NM1=NX1
               STAT_NM2=NY1
               STAT_NM3=NZ1
          endif
!     get coordinates derivatives d[XYZ]/dt
         i = NX1*NY1
         do e = 1, NELT
            if(STAT_LMAP(e).ne.-1) then
               call mxm(XM1(1,1,1,e),i,DZTM1,NZ1,lxyzd(1,1,1,e,1),NZ1)
               call mxm(YM1(1,1,1,e),i,DZTM1,NZ1,lxyzd(1,1,1,e,2),NZ1)
               call mxm(ZM1(1,1,1,e),i,DZTM1,NZ1,lxyzd(1,1,1,e,3),NZ1)
            endif
         enddo

      endif

!     for now I assume STAT_LM1=STAT_NM2=STAT_NM3
!     check if that is true
      if (if3d) then               ! #2D
      if(STAT_LM1.ne.STAT_NM2.or.STAT_LM1.ne.STAT_NM3) then
         if (NIO.eq.0) then
            write(6,*) 'Error: stat_init_int1D; unequal array sizes'
         endif
         call exitt
      endif
      endif

!     get 1D mass matrix ordering directions in such a way that 
!     the uniform direction corresponds to the the first index
      i = STAT_NM1*STAT_NM2*STAT_NM3
!     get arc length
      do e = 1, NELT
         if(STAT_LMAP(e).ne.-1) then
            call vsq(lxyzd(1,1,1,e,1),i)
            call vsq(lxyzd(1,1,1,e,2),i)
            call vsq(lxyzd(1,1,1,e,3),i)
      
            call add2(lxyzd(1,1,1,e,1),lxyzd(1,1,1,e,2),i)
            call add2(lxyzd(1,1,1,e,1),lxyzd(1,1,1,e,3),i)

            call vsqrt(lxyzd(1,1,1,e,1),i)
         endif
      enddo

      i=i*NELT
      call rzero(STAT_BM1D,i)

!     reshuffle array
      if (if3d) then               ! #2D
          call stat_reshufflev(STAT_BM1D,lxyzd,NELT)
      else
          call copy (STAT_BM1D,lxyzd,i)
      endif

!     multiply by wieghts
      do e=1, NELT
         if(STAT_LMAP(e).ne.-1) then
            do k=1, STAT_NM3
               do j=1, STAT_NM2
                  do i=1, STAT_NM1
                     STAT_BM1D(i,j,k,e) = lwm1(i)*STAT_BM1D(i,j,k,e)
                  enddo
               enddo
            enddo
         endif
      enddo

!     get total line length
!     sum contributions from different 3D elements to get 
!     local arc length

      i = STAT_NM2*STAT_NM3*NELT
      call rzero(STAT_ABM1D,i)

      do e = 1, NELV
         el = STAT_LMAP(e)
         if(el.gt.0) then
            do k=1, STAT_NM3
               do j=1, STAT_NM2
                  do i=1, STAT_NM1
                     STAT_ABM1D(j,k,el) = STAT_ABM1D(j,k,el) +
     $                    STAT_BM1D(i,j,k,e)
                  enddo
               enddo
            enddo
         endif
      enddo

!     Global communication to sum local contributions
!     this should be added


      return
      end
c----------------------------------------------------------------------
!     order directions in such a way that the uniform direction 
!     corresponds to the the first index
      subroutine stat_reshufflev(rvar, var, n)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables

!     argument list
      real rvar(STAT_NM1,STAT_NM2,STAT_NM3,LELT) !reshuffled array
      real var(LX1,LY1,LZ1,LELT) ! input array
      integer n                 ! element number to reshuffle

!     local variables
      integer i, j, k, e        ! loop index

      if (STAT_IDIR.eq.1) then
         do e=1, n
            if(STAT_LMAP(e).ne.-1) then
               do k=1, STAT_NM3
                  do j=1, STAT_NM2
                     do i=1, STAT_NM1
                        rvar(i,j,k,e) = var(i,j,k,e)
                     enddo
                  enddo
               enddo
            endif
         enddo
      elseif (STAT_IDIR.eq.2) then
         do e=1, n
            if(STAT_LMAP(e).ne.-1) then
               do k=1, STAT_NM3
                  do j=1, STAT_NM2
                     do i=1, STAT_NM1
                        rvar(i,j,k,e) = var(j,i,k,e)
                     enddo
                  enddo
               enddo
            endif
         enddo
      else
         do e=1, n
            if(STAT_LMAP(e).ne.-1) then
               do k=1, STAT_NM3
                  do j=1, STAT_NM2
                     do i=1, STAT_NM1
                        rvar(i,j,k,e) = var(j,k,i,e)
                     enddo
                  enddo
               enddo
            endif
         enddo
      endif


      return
      end
c----------------------------------------------------------------------
!     perform local 1D integration on 1 variable
      subroutine stat_compute_1Dav1(lvar,npos,alpha,beta)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables
      include 'INPUT'           ! IF3D

!     argument list
      real lvar(LX1,LY1,LZ1,LELT) ! integrated variable
      integer npos              ! position in STAT_RUAVG
      real alpha, beta          ! time averaging parameters

!     local variables
      integer i, j, k, e        ! loop index
      integer el                ! index of 2D element
      real rtmp(STAT_LM1,STAT_LM1,LELT) ! dummy array
      common /CTMP0/ rtmp

!     zero work array
!      if (if3d) then     ! #2D
!          e = STAT_NM2*STAT_NM3*STAT_LNUM
!      else
!          e = STAT_NM1*STAT_NM2*STAT_LNUM
!      endif
      e = STAT_LM1*STAT_LM1*LELT
      call rzero(rtmp,e)

!     perform 1D integral
      if (if3d) then          ! #2D
           do e = 1, NELV
              el = STAT_LMAP(e)
              if(el.gt.0) then
                 do k=1, STAT_NM3
                    do j=1, STAT_NM2
                       do i=1, STAT_NM1
                          rtmp(j,k,el) = rtmp(j,k,el) +
     $                         STAT_BM1D(i,j,k,e)*lvar(i,j,k,e)
                       enddo
                    enddo
                 enddo
              endif
           enddo
      else
           do e = 1, NELV
              el = STAT_LMAP(e)
              if(el.gt.0) then
                 do i=1, STAT_NM1
                    do j=1, STAT_NM2
                       do k=1, STAT_NM3
                          rtmp(i,j,el) = rtmp(i,j,el)+lvar(i,j,k,e)
!     $                         STAT_BM1D(i,j,k,e)*lvar(i,j,k,e)
                     enddo
                    enddo
                 enddo
              endif
           enddo
      endif

!     consistency check
      if(npos.gt.STAT_LVAR) then
         if (NIO.eq.0) write(6,*) 'Error; inconsistent npos ',npos
         call exitt
      endif

!     time average
      if (if3d) then          ! #2D
          e = STAT_NM2*STAT_NM3*STAT_LNUM
      else
          e = STAT_NM1*STAT_NM2*STAT_LNUM
      endif
      call add2sxy(STAT_RUAVG(1,1,1,npos),alpha,rtmp,beta,e)

      return
      end
c----------------------------------------------------------------------
!     perform local 1D integration on 2 variables
      subroutine stat_compute_1Dav2(lvar1,lvar2,npos,alpha,beta)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'           ! 2D statistics speciffic variables
      include 'INPUT'           ! IF3D
!     argument list
      real lvar1(LX1,LY1,LZ1,LELT) ! integrated variable
      real lvar2(LX1,LY1,LZ1,LELT) ! integrated variable
      integer npos              ! position in STAT_RUAVG
      real alpha, beta          ! time averaging parameters

!     local variables
      integer i, j, k, e        ! loop index
      integer el                ! index of 2D element
      real rtmp(STAT_LM1,STAT_LM1,LELT) ! dummy array
      common /CTMP0/ rtmp

!     zero work array
!      e = STAT_NM2*STAT_NM3*STAT_LNUM
      e = STAT_LM1*STAT_LM1*LELT
      call rzero(rtmp,e)

!     perform 1D integral
      if (if3d) then          ! #2D
      do e = 1, NELV
         el = STAT_LMAP(e)
         if(el.gt.0) then
            do k=1, STAT_NM3
               do j=1, STAT_NM2
                  do i=1, STAT_NM1
                     rtmp(j,k,el) = rtmp(j,k,el) +
     $                 STAT_BM1D(i,j,k,e)*lvar1(i,j,k,e)*lvar2(i,j,k,e)
                  enddo
               enddo
            enddo
         endif
      enddo
      else
      do e = 1, NELV
         el = STAT_LMAP(e)
         if(el.gt.0) then
            do i=1, STAT_NM1
               do j=1, STAT_NM2
                  do k=1, STAT_NM3
                     rtmp(i,j,el) = rtmp(i,j,el) +
     $                        lvar1(i,j,k,e)*lvar2(i,j,k,e)
!     $                 STAT_BM1D(i,j,k,e)*lvar1(i,j,k,e)*lvar2(i,j,k,e)
                  enddo
               enddo
            enddo
         endif
      enddo
      endif

!     consistency check
      if(npos.gt.STAT_LVAR) then
         if (NIO.eq.0) write(6,*) 'Error; inconsistent npos ',npos
         call exitt
      endif

!     time average
      if (if3d) then          ! #2D
          e = STAT_NM2*STAT_NM3*STAT_LNUM
      else
          e = STAT_NM1*STAT_NM2*STAT_LNUM
      endif
      call add2sxy(STAT_RUAVG(1,1,1,npos),alpha,rtmp,beta,e)

      return
      end
c----------------------------------------------------------------------
!     compute statistics
      subroutine stat_compute

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'
      include 'STATS'           ! 2D statistics speciffic variables
      include 'INPUT'           ! if3d

!     local variables
!     simple timing
      real ltim_init

      integer npos              ! position in STAT_RUAVG
      real alpha, beta,dtime    ! time averaging parameters
      integer lnvar             ! count number of variables
      integer i                 ! loop index
      integer itmp              ! dummy variable
      real rtmp                 ! dummy variable

!     work arrays
      real slvel(LX1,LY1,LZ1,LELT,3), slp(LX1,LY1,LZ1,LELT)
      common /SCRMG/ slvel, slp
      real tmpvel(LX1,LY1,LZ1,LELT,3), tmppr(LX1,LY1,LZ1,LELT)
      common /SCRUZ/ tmpvel, tmppr

      real dudx(LX1,LY1,LZ1,LELT,3) ! velocity derivatives; U
      real dvdx(LX1,LY1,LZ1,LELT,3) ! V
      real dwdx(LX1,LY1,LZ1,LELT,3) ! W
      common /SCRNS/ dudx, dvdx
      common /SCRSF/ dwdx

!     functions
      real dnekclock

!     simple timing
      ltim_init = dnekclock()

!     stamp logs
      if (NIO.eq.0) write(6,*) '2D statistics compute average'

!     reset varaible counter
      lnvar = 0

!     get time averaging parameters
      dtime=time-stat_atime-stat_tstart
      stat_atime=time-stat_tstart
      beta = dtime/stat_atime             ! for testing
      alpha = 1.0 - beta

!     pressure map, derivatives and vorticity are calculated on the 
!     whole mesh due to stat_pts
!     get pressure
!     map pressure to velocity mesh
      call mappr(tmppr,PR,tmpvel(1,1,1,1,2),tmpvel(1,1,1,1,3))
!     remove mean at the wall

!     get derivatives   **** has been shifted to user_stat_trnsv() ***
!      call gradm1(dudx(1,1,1,1,1),dudx(1,1,1,1,2),dudx(1,1,1,1,3),VX)
!      call gradm1(dvdx(1,1,1,1,1),dvdx(1,1,1,1,2),dvdx(1,1,1,1,3),VY)
!      call gradm1(dwdx(1,1,1,1,1),dwdx(1,1,1,1,2),dwdx(1,1,1,1,3),VZ)

!     get vorticity and transfom velocity and derivatives
      call user_stat_trnsv(tmpvel,dudx,dvdx,dwdx,slvel)

!     call time series
!     adding time series here
!     notice I use tmpvel, so the mean pressure is subtracted from 
!     pressure
      if (if3d) then               ! #2D
!          call stat_pts_compute(tmpvel,slvel,tmppr)
      endif

!     reshuffle velocity arrays
      itmp = LX1*LY1*LZ1*LELT*3

      if (if3d) then               ! #2D
      call stat_reshufflev(slvel(1,1,1,1,1),tmpvel(1,1,1,1,1),NELV)
      call stat_reshufflev(slvel(1,1,1,1,2),tmpvel(1,1,1,1,2),NELV)
      call stat_reshufflev(slvel(1,1,1,1,3),tmpvel(1,1,1,1,3),NELV)
      else
          call copy(slvel,tmpvel,itmp)
      endif

!     reshuffle pressure
      if (if3d) then          ! #2D
           call stat_reshufflev(slp,tmppr,NELV)
      else
           itmp=LX1*LY1*LZ1*LELT
           call copy(slp,tmppr,itmp)
      endif

!     reshuffle velocity derivatives
!     VX
      if (if3d) then               ! #2D
      call stat_reshufflev(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),dudx(1,1,1,1,2),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,3),dudx(1,1,1,1,3),NELV)
!     copy
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call copy(dudx,tmpvel,itmp)
      endif

!     VY
      if (if3d) then               ! #2D
      call stat_reshufflev(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),dvdx(1,1,1,1,2),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,3),dvdx(1,1,1,1,3),NELV)
!     copy
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call copy(dvdx,tmpvel,itmp)
      endif

!     VZ
      if (if3d) then               ! #2D
      call stat_reshufflev(tmpvel(1,1,1,1,1),dwdx(1,1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),dwdx(1,1,1,1,2),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,3),dwdx(1,1,1,1,3),NELV)
!     copy
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call copy(dwdx,tmpvel,itmp)
      endif

     
c============================================================  
!     notation
!     U , V, W - velocity components
!     P - pressure
!     U,X = dU/dX - velocity derivative

!     average U
      lnvar = lnvar + 1
      npos = lnvar
c$$$!     for testing
c$$$      itmp = NX1*NY1*NZ1
c$$$      do i=1,NELV
c$$$         rtmp = STAT_LMAP(i)
c$$$         call cfill(slvel(1,1,1,i,1),rtmp,itmp)
c$$$      enddo
c$$$!     for testing; end
      call stat_compute_1Dav1(slvel(1,1,1,1,1),npos,alpha,beta)

!    lnvar = 1
c------------------------------------------------------------ 

!     average V
      lnvar = lnvar + 1
      npos = lnvar
c$$$!     for testing
c$$$      itmp = NX1*NY1*NZ1
c$$$      do i=1,NELV
c$$$         rtmp = STAT_GMAP(STAT_LMAP(i))
c$$$         call cfill(slvel(1,1,1,i,2),rtmp,itmp)
c$$$      enddo
c$$$!     for testing; end
      call stat_compute_1Dav1(slvel(1,1,1,1,2),npos,alpha,beta)

c------------------------------------------------------------ 

!     average W
      lnvar = lnvar + 1
      npos = lnvar
c$$$!     for testing
c$$$      itmp = NX1*NY1*NZ1
c$$$      do i=1,NELV
c$$$         rtmp = STAT_OWN(STAT_LMAP(i))
c$$$         call cfill(slvel(1,1,1,i,3),rtmp,itmp)
c$$$      enddo
c$$$!     for testing; end
      call stat_compute_1Dav1(slvel(1,1,1,1,3),npos,alpha,beta)

c------------------------------------------------------------ 

!     average P
      lnvar = lnvar + 1
      npos = lnvar
c$$$!     for testing
c$$$      itmp = NX1*NY1*NZ1
c$$$      do i=1,NELV
c$$$         rtmp = NID
c$$$         call cfill(slp(1,1,1,i),rtmp,itmp)
c$$$      enddo
c$$$!     for testing; end
      call stat_compute_1Dav1(slp(1,1,1,1),npos,alpha,beta)

!     get the rest

c============================================================  
! variables below not tested.

!    <uu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,1),
     $         npos,alpha,beta)
!npos==5
      
c------------------------------------------------------------ 
!    <vv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,2),slvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <ww>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,3),slvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <pp>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),slp(1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <uv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <vw>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,2),slvel(1,1,1,1,3),
     $         npos,alpha,beta)
!npos==10
      
c------------------------------------------------------------ 
!    <uw>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <pu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <pv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    <pw>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!    Variable: pdu/dx. Notation: <p(u_x)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,1),
     $         npos,alpha,beta)
!npos==15

c------------------------------------------------------------ 
!     <p(u_y)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(u_z)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(v_x)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(v_y)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(v_z)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,3),
     $         npos,alpha,beta)
!npos==20

c------------------------------------------------------------ 
!     <p(w_x)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(w_y)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <p(w_z)>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
c------------------------------------------------------------ 

!     UU,VV,WW
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),slvel(1,1,1,1,1),slvel(1,1,1,1,1),
     $         itmp)
c------------------------------------------------------------ 
c------------------------------------------------------------ 

!     <uuu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,1),tmpvel(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <vvv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,2),tmpvel(1,1,1,1,2),
     $         npos,alpha,beta)
!npos==25

c------------------------------------------------------------ 
!     <www>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(slvel(1,1,1,1,3),tmpvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <uuv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <uuw>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),slvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <vvu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),slvel(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <vvw>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),slvel(1,1,1,1,3),
     $         npos,alpha,beta)
!npos==30

c------------------------------------------------------------ 
!     <wwu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),slvel(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <wwv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),slvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <ppp>
      lnvar = lnvar + 1
      npos = lnvar

!      copy pp to tmppr 
      itmp = LX1*LY1*LZ1*LELT
      call col3(tmppr(1,1,1,1),slp(1,1,1,1),slp(1,1,1,1),
     $         itmp) 
     
      call stat_compute_1Dav2(tmppr(1,1,1,1),slp(1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <pppp>
      lnvar = lnvar + 1
      npos = lnvar

!     tmppr==pp
      call stat_compute_1Dav2(tmppr(1,1,1,1),tmppr(1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <uvw>
      lnvar = lnvar + 1
      npos = lnvar
      
!     copy uv to tmppr (don't need pp anymore) 
      itmp = LX1*LY1*LZ1*LELT
      call col3(tmppr(1,1,1,1),slvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $         itmp) 
      
      call stat_compute_1Dav2(tmppr(1,1,1,1),slvel(1,1,1,1,3),
     $         npos,alpha,beta)
!npos==35

c------------------------------------------------------------ 
!     <uuuu>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <vvvv>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),tmpvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <wwww>
      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),tmpvel(1,1,1,1,3),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <uuuv>
      lnvar = lnvar + 1
      npos = lnvar

!     tmppr = uv            
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),tmppr(1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     <uuvv>
      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),
     $         npos,alpha,beta)
!npos==40

c------------------------------------------------------------ 
!     <uvvv>
      lnvar = lnvar + 1
      npos = lnvar

!     tmppr = uv      
     
      call stat_compute_1Dav2(tmppr(1,1,1,1),tmpvel(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     Variable: (du/dx)^2 + (du/dy)^2 + (du/dz)^2.
!     Notation: e11

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dudx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

c------------------------------------------------------------ 
!     (dv/dx)^2 + (dv/dy)^2 + (dv/dz)^2.
!     e22

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

c------------------------------------------------------------ 
!     (dw/dx)^2 + (dw/dy)^2 + (dw/dz)^2.
!     e33

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dwdx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dx)*(dv/dx) + (du/dy)*(dv/dy) + (du/dz)*(dv/dz).
!     e12

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
!npos==45

c------------------------------------------------------------ 
!     (du/dx)*(dw/dx) + (du/dy)*(dw/dy) + (du/dz)*(dw/dz).
!     e13

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

c------------------------------------------------------------ 
!     (dv/dx)*(dw/dx) + (dv/dy)*(dv/dy) + (dv/dz)*(dw/dz).
!     e23

      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $         itmp)

      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dx)*(du/dx)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,1),dudx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dy)*(du/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,2),dudx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dx)*(du/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,1),dudx(1,1,1,1,2),
     $         npos,alpha,beta)
!npos==50

c------------------------------------------------------------ 
!     (dv/dx)*(dv/dx)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dvdx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (dv/dy)*(dv/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dvdx(1,1,1,1,2),dvdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (dv/dx)*(dv/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dvdx(1,1,1,1,1),dvdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (dw/dx)*(dw/dx)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dwdx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (dw/dy)*(dw/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dwdx(1,1,1,1,2),dwdx(1,1,1,1,2),
     $         npos,alpha,beta)
!npos==55

c------------------------------------------------------------ 
!     (dw/dx)*(dw/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dwdx(1,1,1,1,1),dwdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dx)*(dv/dx)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dy)*(dv/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,2),dvdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dx)*(dv/dy)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,1),dvdx(1,1,1,1,2),
     $         npos,alpha,beta)

c------------------------------------------------------------ 
!     (du/dy)*(dv/dx)

      lnvar = lnvar + 1
      npos = lnvar
      
      call stat_compute_1Dav2(dudx(1,1,1,1,2),dvdx(1,1,1,1,1),
     $         npos,alpha,beta)
!npos==60

c============================================================  

!     save number of variables
      STAT_NVAR = lnvar

!     simple timing
      ltim_init = dnekclock() - ltim_init

!     update timing and counters
      STAT_TEV = STAT_TEV + ltim_init
      STAT_EVN = STAT_EVN + 1

      return
      end
c----------------------------------------------------------------------

      subroutine stat_comm_map()

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
      include 'PARALLEL'
      include 'mpif.h'

      integer e,e2,ii,arr_size
      parameter (arr_size=lelt)             ! place holder

      integer collsteps,poffset,dstid,recno,tosend
      integer nt,csteps,lsend,nprocs,tmp_max,pos_cnt,ivlmax
      integer logi_procpos(arr_size),tmp_procpos(arr_size)
      integer tmp2_procpos(arr_size)

c------------------------------ transfer to STAT common block
c      integer stat_procid(stat_lnum), stat_procpos(stat_lnum)
c      integer stat_sendno(stat_lnum)
c      integer stat_maxrec,stat_snd_cnt

c      logical stat_init

c------------------------------
      call ifill(stat_procid,-1,lelt)
      call izero(stat_procpos,lelt)

      collsteps = np/arr_size
      if (mod(np,arr_size).gt.0) then
            collsteps=collsteps+1
      endif

      nprocs=arr_size
      stat_snd_cnt=0
      pos_cnt=0
      recno=0
      stat_maxrec=0

      do ii=0,collsteps-1

      call izero(tmp_procpos,nprocs)
      call izero(logi_procpos,nprocs)

      if (ii.eq.collsteps-1) then
            nprocs=mod(np,arr_size)
      endif
      poffset=ii*arr_size

      do e=1,nprocs
            dstid=poffset+e-1
            
            if (dstid.eq.nid) then
                  tosend=0
            else
                  call get_send_no(dstid,tosend,stat_own,
     $            stat_lnum,nid)
            endif
            
            if (tosend.gt.0) then
                  stat_snd_cnt=stat_snd_cnt+1
c                  stat_sendno=tosend
                  tosend=1
                  stat_procid(stat_snd_cnt)=dstid
            endif

            tmp_procpos(e)=tosend
            logi_procpos(e)=tosend

      enddo       ! e=1,nprocs
      
      call ivgl_running_sum(tmp2_procpos,tmp_procpos,nprocs)
      call icopy(tmp_procpos,tmp2_procpos,nprocs)
     
      call ibcastn(tmp2_procpos,nprocs,np-1)

      if ((nid.ge.poffset).and.(nid.lt.poffset+nprocs)) then
            stat_recno=tmp2_procpos(nid+1-poffset)    ! get how many NID receives
            
            tmp_max=ivlmax(tmp2_procpos,nprocs)
            if (tmp_max.gt.stat_maxrec) then
                  stat_maxrec=tmp_max           ! update max number of receives
            endif
      endif

      do e2=1,nprocs
            tmp2_procpos(e2)=tmp_procpos(e2)*logi_procpos(e2)
            if (logi_procpos(e2).gt.0) then
                  pos_cnt=pos_cnt+1
                  stat_procpos(pos_cnt)=tmp2_procpos(e2)
            endif
      enddo       ! end e2=1,nprocs

      enddo       ! end ii=1:collsteps-1
      
      
      stat_ini=.false.

      end subroutine stat_comm_map

c--------------------------------------------------

      subroutine stat_gl_collation

      implicit none

      include 'mpif.h'
cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'PARALLEL'
      include 'STATS'
      include 'INPUT' !if3d

      integer e,e2,cnt2,cnt3,len
      integer recsize
      integer recmap(lelt)

      integer status(mpi_status_size)
      integer msg_id1,msg_id2,msg_id3          ! request handles for irecv
      integer ierr
      integer dstid
      integer maxpos
     
      integer ivlmax          ! function

      integer snds(lelt)
      integer irecv

      real recdata(stat_lm1,stat_lm1,lelt*stat_lvar)
      real arclen(stat_lm1,stat_lm1,lelt)

      integer pos

      if (nid.eq.0) then
            write(6,*) 'Computing Global Stats'
      endif

      maxpos=ivlmax(stat_procpos,stat_maxrec)     ! max position number

      do e=1,stat_maxrec
            
            if (e.le.stat_recno) then
                  len=(stat_lnum)*isize 
                  msg_id1 = irecv(0,recmap,len)       ! rec mapping
            endif

c---------- Send mapping 
            cnt2=0                  ! counter for processes to send to in this step
            if (e.le.maxpos) then
            do e2=1,stat_snd_cnt                       ! could send to multiple
                                                       ! processes in one step
                  if (stat_procpos(e2).eq.e) then
                        dstid=stat_procid(e2)
                        cnt2=cnt2+1
                        snds(cnt2)=dstid

                        call sendmap(dstid)            ! send mapping 
                  endif
            enddo
            endif
           
            if (e.le.stat_recno) then
                  call msgwaitstat(msg_id1,status)    ! wait to receive mapping

                  call mpi_get_count(status,mpi_byte,recsize,ierr)
                  recsize=recsize/isize                ! no of elements

                  call trnsfm_recmap(recmap,recsize)

                  len=wdsize*recsize*stat_nm2*stat_nm3*stat_lvar
                  msg_id2 = irecv(0,recdata,len)      ! rec data

                  len=wdsize*recsize*stat_nm2*stat_nm3
                  msg_id3 = irecv(1,arclen,len)       ! rec arclength
                 
            endif

c---------- Send data
            do e2=1,cnt2
                  dstid=snds(e2)
                  call senddata(dstid)
            enddo

c---------- Receive data and arclength
            if (e.le.stat_recno) then
                  call msgwait(msg_id2)               ! wait to receive data

                  call colldata(recmap,recsize,recdata)
                  call msgwait(msg_id3)               ! wait to receive arclength

c---------- Sum up arc lengths                   
                  if (stat_ini.eqv..false.) then
                        len=stat_nm2*stat_nm3
                        do e2=1,recsize                        
                              call add2(stat_abm1d(1,1,recmap(e2)),
     $                        arclen(1,1,e2),len)
                        enddo
                  endif

            endif
            call nekgsync            
      enddo             ! end e=1,stat_maxrec

      stat_ini=.true.

c---------- Divide data by arc length

      if (if3d) then
           len=stat_nm2*stat_nm3
           do e=1,stat_lnum
           if (stat_own(e).eq.nid) then
                 do e2=1,stat_lvar
                 call invcol2(stat_ruavg(1,1,e,e2),
     $                   stat_abm1d(1,1,e),len)
                 enddo
           endif
           enddo
      endif

      end subroutine stat_gl_collation

c-------------------------------------------------- 

      subroutine msgwaitstat(imsg,status)
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
c
c     write(6,*) nid,' msgwait:',imsg
c
      call mpi_wait (imsg,status,ierr)
c
      return
      end

c-------------------------------------------------- 

      subroutine senddata(dstid)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
      include 'PARALLEL'

      integer dstid
      integer len
      integer indmap(lelt)
      integer e,e2,lsels
      integer pos

      real snddata(stat_lm1,stat_lm1,lelt*stat_lvar)
      real arclen(stat_lm1,stat_lm1,lelt)
      
      lsels=0                       ! number of local stat elements to send
      do e=1,stat_lnum
      if (stat_own(e).eq.dstid) then
            lsels=lsels+1
            indmap(lsels)=e
      endif
      enddo

      len=stat_nm2*stat_nm3
      do e=1,stat_lvar
      do e2=1,lsels
      pos = (e-1)*lsels+e2
      call copy(snddata(1,1,pos),stat_ruavg(1,1,indmap(e2),e),len)
      call rzero(stat_ruavg(1,1,indmap(e2),e),len)
      enddo
      enddo

      do e2=1,lsels
      call copy(arclen(1,1,e2),stat_abm1d(1,1,indmap(e2)),len)
      enddo

      len=stat_nm2*stat_nm3*lsels*stat_lvar*wdsize
      call csend(0,snddata,len,dstid,0)         ! send data
      
      len=stat_nm2*stat_nm3*lsels*wdsize
      call csend(1,arclen,len,dstid,0)          ! send arc length


      end subroutine senddata

c--------------------------------------------------

      subroutine sendmap(dstid)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
      include 'PARALLEL'

      integer dstid
      integer len
      integer e,lsels

      integer sndmap(lelt)
      
      lsels=0                       ! local number of stat elements to send
      do e=1,stat_lnum
      if (stat_own(e).eq.dstid) then
            lsels=lsels+1
            sndmap(lsels)=stat_gmap(e)
      endif
      enddo

      if (nid.eq.1) then
          write(*,*) (sndmap(e),e=1,lsels)
      endif
      
      len=lsels*isize
      call csend(0,sndmap,len,dstid,0)

      end subroutine sendmap

c--------------------------------------------------

      subroutine colldata(recmap,n,recdata)

      implicit none

cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
      
      integer e,e2
      integer n,len
      integer recmap(n)
      integer pos
      integer lpos

      real recdata(stat_lm1,stat_lm1,lelt*stat_lvar)
!      real recdata(1)

      len=stat_nm2*stat_nm3
   
      do e=1,stat_lvar
      do e2=1,n
       pos = (e-1)*n + e2
       call add2(stat_ruavg(1,1,recmap(e2),e),recdata(1,1,pos),len)

      enddo
      enddo


      end subroutine colldata 
c-------------------------------------------------- 

      subroutine get_send_no(dstid,tosend,stat_own,stat_lnum,nid)

      implicit none

      integer dstid,tosend,e,stat_lnum,nid
      integer stat_own(1)

      tosend=0
      do e=1,stat_lnum
            if (stat_own(e).eq.dstid) then
                  tosend=tosend+1
            endif

      enddo

      end subroutine get_send_no

c----------------------------------------------------------------------

      subroutine trnsfm_recmap(recmap,recsize)

      implicit none
cc MA:     include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'

      integer recsize
      integer recmap(lelt)

      integer i,j,cnt
     
      cnt=0
      do i = 1,recsize
      do j = 1,stat_lnum
          if (stat_gmap(j).eq.recmap(i)) then
               recmap(i) = j
               cnt=cnt+1
               exit
          endif
          if (cnt.eq.recsize) return

      enddo
      enddo

      end subroutine trnsfm_recmap

c----------------------------------------------------------------------
    


