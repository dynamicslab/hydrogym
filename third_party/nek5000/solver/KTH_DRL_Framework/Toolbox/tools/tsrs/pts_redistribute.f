!=======================================================================
! Description : Set of rutines to redistribute points between processors
!     IMPORTANT!!! This vesion of the code does not take into account
!     error code rcode. This shuld be added later.
!     IMPORTANT!!! This vesion of the code does not work for number of 
!     processors bigger than 2*LX1*LY1*LZ1*LELT
!=======================================================================
!     Adam Peplinski 2021.05.28
!     This is a very old version barely touched by me right now due to lack of time.
!     For now I just slightly refresh the old stuff and hope for the best.
c-----------------------------------------------------------------------
!     redistribute points
!     I assume findpts aws alleready called and proc array is filled
!     IMPORTANT!!! This routine uses scratch arrays in
!     scrmg and scrns
      subroutine pts_rdst(nptimb)

      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'TSRSD'

!     argument list
      integer nptimb            ! allowed point imbalance

!     local variables
      integer il                ! loop index
      integer itmp

!     work arrays; I use scratch so be carefull
      integer libuf, lrbuf, lptn
      parameter (libuf = 5, lrbuf = 2*LDIM, lptn=LX1*LY1*LZ1*LELT)
      integer ibuf(libuf,lptn)
      real rbuf(lrbuf,lptn)
      integer mid(lptn)
      common /scrmg/ ibuf, mid
      common /scrns/ rbuf

!     check size of transfer arrays
      if(LHIS.gt.lptn) then
         if (NIO.eq.0)
     $        write(6,*) 'Error: pts_rdst; insufficient buffer size'
         call exitt
      endif

!     adjust point -processor mapping

!     IMPORTANT!!!
!     Place to use information from rcode(i)

!     fill in mid
      do il=1,tsrs_npts
         mid(il) = tsrs_proc(il)
      enddo

!     prepare mapping
      call pts_map_all(mid,lptn,tsrs_npts,tsrs_nptot,nptimb)

!     fill in redistribution array
!     integer
      do il=1,tsrs_npts
         ibuf(1,il) = mid(il)
         ibuf(2,il) = tsrs_ipts(il)
         ibuf(3,il) = tsrs_proc(il)
         ibuf(4,il) = tsrs_elid(il)
         ibuf(5,il) = tsrs_rcode(il)
      enddo

!     real
      if (IF3D) then
         do il=1,tsrs_npts
            itmp = (il-1)*NDIM
            rbuf(1,il) = tsrs_pts(1,il)
            rbuf(2,il) = tsrs_pts(2,il)
            rbuf(3,il) = tsrs_pts(3,il)
            rbuf(4,il) = tsrs_rst(itmp+1)
            rbuf(5,il) = tsrs_rst(itmp+2)
            rbuf(6,il) = tsrs_rst(itmp+3)
         enddo
      else
         do il=1,tsrs_npts
            itmp = (il-1)*NDIM
            rbuf(1,il) = tsrs_pts(1,il)
            rbuf(2,il) = tsrs_pts(2,il)
            rbuf(3,il) = tsrs_rst(itmp+1)
            rbuf(4,il) = tsrs_rst(itmp+2)
         enddo
      endif

!     redistribute points
      call pts_transfer(rbuf,lrbuf,ibuf,libuf,lptn,tsrs_npts)

!     copy arrays back
!     integer
      do il=1,tsrs_npts
         tsrs_ipts(il) = ibuf(2,il)
         tsrs_proc(il) = ibuf(3,il)
         tsrs_elid(il) = ibuf(4,il)
         tsrs_rcode(il) = ibuf(5,il)
      enddo

!     real
      if (IF3D) then
         do il=1,tsrs_npts
            itmp = (il-1)*NDIM
            tsrs_pts(1,il) = rbuf(1,il)
            tsrs_pts(2,il) = rbuf(2,il)
            tsrs_pts(3,il) = rbuf(3,il)
            tsrs_rst(itmp+1) = rbuf(4,il)
            tsrs_rst(itmp+2) = rbuf(5,il)
            tsrs_rst(itmp+3) = rbuf(6,il)
         enddo
      else
         do il=1,tsrs_npts
            itmp = (il-1)*NDIM
            tsrs_pts(1,il) = rbuf(1,il)
            tsrs_pts(2,il) = rbuf(2,il)
            tsrs_rst(itmp+1) = rbuf(3,il)
            tsrs_rst(itmp+2) = rbuf(4,il)
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
!     redistibute points between processors
!     ibuf and rbuf have to be filled in outside this routine
      subroutine pts_transfer(rbuf,lrbuf,ibuf,libuf,lpts,npts)

      implicit none

! !       include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
! !       include 'PARALLEL_DEF'
      include 'PARALLEL'

!     argument list
      real    rbuf(lrbuf,lpts)    ! point list
      integer ibuf(libuf,lpts) ! target proc id; global ordering
      integer lrbuf, libuf, lpts ! array sizes
      integer npts              ! local number of points

!     local variables
      integer itmp1, itmp2
!     required by crystal router
      integer*8 vl

!     timing
      real ltime1, ltime2, timemaxs, timemins

!     functions
      integer iglmin, iglmax
      real dnekclock, glmax, glmin

!     timing
      ltime1 = dnekclock()

!     send points
      call fgslib_crystal_tuple_transfer
     $     (cr_h,npts,lpts,ibuf,libuf,vl,0,rbuf,lrbuf,1)

!     statistics after redistribution
      itmp1 = iglmin(npts,1)
      itmp2 = iglmax(npts,1)

!     timing
      ltime2 = dnekclock()
      ltime2 = ltime2 - ltime1
      timemaxs = glmax(ltime2,1)
      timemins = glmin(ltime2,1)

!     stamp logs
      if (NIO.eq.0) then
         write(6,*)
         write(6,*) 'Point redistribution; pts_transfer_min:'
         write(6,'(A22,I7,I7)') 'New loc pts nr min/max', itmp1, itmp2
         write(6,'(A22,g13.5,g13.5)') 'Sending time min/max  ', 
     $        timemins, timemaxs
         write(6,*)
      endif

      return
      end
c-----------------------------------------------------------------------
!     redistibute points between processors
      subroutine pts_map_all(mid,lpts,npts,npoints,nptimb)

      implicit none

! !       include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
! !       include 'PARALLEL_DEF'
      include 'PARALLEL'

!     dummy arrays
!     I use nek5000 scratch arrays so be careful
      integer lptn
      parameter (lptn=2*LX1*LY1*LZ1*LELT)
      integer npts_plist(lptn) ! list of processors used
      common /scrvh/ npts_plist

!     argument list
      integer mid(lpts)       ! target proc id; global ordering
      integer lpts              ! array size
      integer npts              ! local number of points
      integer npoints           ! global number of points
      integer nptimb            ! allowed point imbalance

!     specific common blocks
      integer nptav, nptmod     ! average point number per proc 
      integer nptmax            ! max point number per proc
      common /istat_pts_avm/ nptav, nptmod, nptmax
!     global/local counters
      integer nptgdone, nptgundone ! done/undone points number; global
      integer nptldone, nptlundone ! done/undone points number; local
      common /istat_pts_done/ nptgdone, nptgundone, nptldone,
     $     nptlundone
!     to keep track of nodes overflow and free space
      integer nptover, nptempty, nptshift   
      common /istat_pts_oes/ nptover, nptempty, nptshift

!     local variables
      integer ierr              ! error mark
      integer itmp              ! dummy variables

      integer nptav1            ! local average point number per proc 
      integer nptshift1         ! global number of shifts in all calls
      integer nloop, nloopmod   ! processor loop
      integer nplist            ! number of processors in the list
      integer ipr, ipt          ! loop index

!     node status; positive - overflow (points to send)
!                  negative - empty space
      integer istatus
      integer indasg            ! number of points asigned to the node

!     timing
      real ltime1, ltime2, timemax, timemin

!     functions
      integer iglsum, iglmin, iglmax
      real dnekclock, glmax, glmin

!     timing
      ltime1 = dnekclock()

!     Generate global point mapping for point redistribution

!     Average points per processor; this value has to be 
!     dependent on nptmod
!     This has to be done here as pts_map_set can be called 
!     number of times
      nptav = npoints/NP
      nptmod = mod(npoints,NP)
      if (nptmod.gt.0) then
         nptav1 = nptav + 1
      else
         nptav1 = nptav
      endif
!     nptimb must be positive
      nptimb = abs(nptimb)
!     max points per processor
      nptmax = min(nptav1 + nptimb,LHIS)

      ierr = 0
      if (nptav1.gt.LHIS) ierr = 1
      ierr=iglsum(ierr,1)
      if(ierr.gt.0) then
        if(NIO.eq.0) write(6,*)
     $        'Error: pts_map_all; wrong nptav'
        call exitt
      endif

!     There can be more processors than the array size and the 
!     distribution of points does not gurantee the nodes with 
!     the highies number of points are located at the beginning
!     of the processor list.
!     In general additional sorting is necessary.

!     Initialise done and undone points number
!     This has to be done here as pts_map_set can be called 
!     number of times.
!     global
      nptgdone = 0
      nptgundone = npoints
!     local
      nptldone = 0
      nptlundone = npts

!     Is there more processors than the array size
      nloop = NP/lptn + 1
      nloopmod = mod(NP,lptn)

!     If processor number is small no sorting is necessary
      if (nloop.eq.1) then
         nplist = NP

!     fill in processor array
         do ipr=1,nplist
            npts_plist(ipr) = ipr-1
         enddo

!     mark all processors as undone (negative number of points 
!     assigned to the node)
         indasg = -1

!     is syncronisatoin necessary?
         call nekgsync()

         call pts_map_set(mid,lpts,istatus,indasg,
     $        npts_plist,lptn,npts,nplist)

!     check consistency
!     are all points done
         if (nptgdone.ne.npoints) then
            if(NIO.eq.0) write(6,*)
     $              'Error: pts_map_all; not all points redist.'
            call exitt
         endif

!     is the local number consistent
         ierr = 0
         if (nptldone.ne.npts) ierr = 1
         ierr=iglsum(ierr,1)
         if(ierr.gt.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error: pts_map_all; wrong nptldone'
            call exitt
         endif

!     is there global overflow
         if(nptover.ne.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error: pts_map_all; global overflow'
            call exitt
         endif

!     does any node report overflow
         ierr = 0
         if (istatus.gt.0) ierr = istatus
         ierr=iglsum(ierr,1)
         if(ierr.gt.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error: pts_map_all; node overflow'
            call exitt
         endif

!     get min/max assigned points
         ipr = iglmin(indasg,1)
         ipt = iglmax(indasg,1)

         if(ipr.lt.0) then
            if(NIO.eq.0) write(6,*)
     $           'Error: pts_map_all; untuched nodes'
            call exitt
         endif

!     save number of shifts
         nptshift1 = nptshift

      else                      ! nloop.eq.1
!     Big number of processors; sorting necessary

!     not done yet
         if(NIO.eq.0) write(6,*)
     $        'Error: pts_map_all unsupported option'
         call exitt

      endif                     ! nloop.eq.1

!     timing
      ltime2 = dnekclock()
      ltime1 = ltime2 - ltime1
      timemax = glmax(ltime1,1)
      timemin = glmin(ltime1,1)

!     statistics before redistribution
      ierr = iglmin(npts,1)
      itmp = iglmax(npts,1)

!     stamp logs
      if (NIO.eq.0) then
         write(6,*)
         write(6,*) 'Point redistribution; pts_map_all:'
         write(6,'(A22,I7)')    'Global point nr       ', npoints
         write(6,'(A22,I7)')    'Average point nr      ', nptav
         write(6,'(A22,I7,I7)') 'Old loc pts nr min/max', ierr, itmp
         write(6,'(A22,I7,I7)') 'Assgn point nr min/max', ipr, ipt
         write(6,'(A22,I7)') 'Nr of shifted points  ', nptshift1
         write(6,'(A22,g13.5,g13.5)') 'Mapping time min/max  ', 
     $        timemin, timemax
         write(6,*)
      endif

      return
      end
c-----------------------------------------------------------------------
!     map points to processors on the working processor set
      subroutine pts_map_set(mid,lpts,istatus,indasg,npts_plist,
     $     lplist,npts,nplist)

      implicit none

! !       include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
! !       include 'PARALLEL_DEF'
      include 'PARALLEL'

!     dummy arrays
!     I use nek5000 scratch arrays so be careful
      integer lptn
      parameter (lptn=2*LX1*LY1*LZ1*LELT)
      integer npts_node_u(lptn) ! undone point-node distribution
      integer npts_node_l(lptn) ! local point-node distribution
      integer npts_node_r(lptn) ! running point-node distribution
      integer npts_node_g(lptn) ! global point-node distribution
      integer npts_node_h(lptn) ! hidden point-node distribution
      common /scrch/ npts_node_u
      common /screv/ npts_node_l
      common /ctmp0/ npts_node_r
      common /ctmp1/ npts_node_g
      common /scrsf/ npts_node_h

!     argument list
      integer mid(lpts)       ! target proc id; global ordering
      integer lpts              ! array size
      integer istatus           ! node status
      integer indasg            ! number of points asigned to the node
      integer npts_plist(lplist) ! list of processors used
      integer lplist            ! array size
      integer npts              ! local number of points
      integer nplist            ! number of processors in the list

!     specific common blocks
      integer nptav, nptmod     ! average point number per proc 
      integer nptmax            ! max point number per proc
      common /istat_pts_avm/ nptav, nptmod, nptmax
!     global/local counters
      integer nptgdone, nptgundone ! done/undone points number; global
      integer nptldone, nptlundone ! done/undone points number; local
      common /istat_pts_done/ nptgdone, nptgundone, nptldone,
     $     nptlundone
! to keep track of nodes overflow and  free space
      integer nptover, nptempty, nptshift   
      common /istat_pts_oes/ nptover, nptempty, nptshift
      

!     local variables
      integer ierr              ! error mark
      integer itmp, itmp1, itmp2, itmp3, itmp4, itmp5 ! dummy variables
      integer idummy(2)         ! dummy arrays

      integer nptav1            ! local average point number per proc 

      integer ipr, ipt, ipt2, ipt3 ! loop index
      integer nodeid,  nodeid1  ! poc id

      integer nptupg, nptupl    ! global/local updated points

!     functions
      integer iglsum

c$$$!     for testing
c$$$      integer itl1, itl2
c$$$      character*2 str
c$$$      character*3 str2
c$$$      write(str,'(i2.2)') NID
c$$$      str2=str//'p'

!     nptav, nptmod, nptmax are set outside this routine, as 
!     this routine can be executed many times

!     Initialise done and undone points number
!     Done outside this routine, as this routine can be executed 
!     many times
!     nptgdone, nptgundone, nptldone, nptlundone

!     keep track of redistribution; overflows, empty spaces, points
!     shifted to other processor
!     these variables are specific to given processor set
      nptover = 0
      nptempty = 0
      nptshift = 0

!     gather distribution information about all processors in the set
      call izero(npts_node_l,nplist)
!     local loop over points and processors
      do ipr = 1,nplist
         do ipt=1,npts
            if(npts_plist(ipr).eq.mid(ipt))
     $           npts_node_l(ipr) = npts_node_l(ipr) + 1
         enddo
      enddo
 
!     global communication
      call ivgl_running_sum(npts_node_r,npts_node_l,nplist)

!     last node has global sum; broadcast it
      if (NID.eq.(NP-1)) call icopy(npts_node_g,npts_node_r,nplist)
      call ibcastn(npts_node_g,nplist,NP-1)

!     remove local numbers from the running sum
      do ipr = 1, nplist
         npts_node_r(ipr) = npts_node_r(ipr) - npts_node_l(ipr)
      enddo

!     mark everything as undone
      call icopy(npts_node_u,npts_node_g,nplist)
!     zero hidden array
      call izero(npts_node_h,nplist)

!     extract information
!     loop over processors
      do ipr = 1, nplist

!     destination proc id
         nodeid = npts_plist(ipr)

!     destination average number
         if (nodeid.lt.nptmod) then
            nptav1 = nptav + 1
         else
            nptav1 = nptav
         endif

!     check global number of points
         if (npts_node_g(ipr).gt.nptmax) then
!     too many points; some have to be redistributed

!     check how many points are allready placed on the node ipr
            idummy(1) = 0
            idummy(2) = 0
            itmp = 0
            nptupg = nptmax
            if (NID.eq.nodeid) then
               itmp = min(npts_node_l(ipr),nptupg)

!     update local variables
               npts_node_l(ipr) = npts_node_l(ipr) - itmp
               npts_node_r(ipr) = npts_node_r(ipr) + itmp
               idummy(1) = itmp
               idummy(2) = npts_node_r(ipr)
            endif

!     broadcast values and get new number of point to collect
            call ibcastn(idummy,2,nodeid)
            itmp = nptupg - idummy(1)

!     shift points
            if (npts_node_r(ipr).ge.idummy(2))
     $           npts_node_r(ipr) = npts_node_r(ipr) - idummy(1)
!     set hidden points
            npts_node_h(ipr) = npts_node_h(ipr) + idummy(1)

            nptupl = 0
            if(itmp.gt.0) then
!     check how many local points can be unchanged

               if(npts_node_l(ipr).gt.0) then
!     what point range
                  itmp1 = 1 - npts_node_r(ipr)
                  itmp2 = itmp1 + itmp - 1
!     does current node fit into this range
                  itmp1 = max(1,itmp1)
                  itmp2 = min(npts_node_l(ipr),itmp2)
                  if ((itmp1.le.npts_node_l(ipr)).and.
     $                 (itmp2.ge.1)) nptupl = itmp2 - itmp1 + 1

!     update local variables
                  npts_node_l(ipr) = npts_node_l(ipr) - nptupl
                  npts_node_r(ipr) = npts_node_r(ipr) + nptupl
               endif            ! npts_node_l(ipr).gt.0

!     check consistency
               ierr = iglsum(nptupl,1)
               if(ierr.ne.itmp) then
                  if(NIO.eq.0) write(6,*)
     $                 'Error: pts_map_set; wrong nptupl 1'
                  call exitt
               endif

            endif               ! itmp.gt.0

!     add hidden points to the local done
         if (NID.eq.nodeid) nptupl = nptupl + npts_node_h(ipr)

!     initial update done/undone point number; related to this processor
!     global
            nptgdone = nptgdone + nptupg
            nptgundone = nptgundone - nptupg
!     local
            nptldone = nptldone + nptupl
            nptlundone = nptlundone - nptupl

!     mark processor
!     sending - positive, receiving -negative, zeor - done
            itmp = npts_node_g(ipr) - nptupg
            npts_node_u(ipr) = itmp
!     count overflow/empty/shift
            nptover = nptover + itmp

!     check if threre are some empty places to redistribute points
!     count the points
            nptupg = 0          ! global
            nptupl = 0          ! local
            if (nptempty.gt.0) then

!     loop over all nodes with smaller id
               do ipt2=1,ipr-1
                  if(npts_node_u(ipt2).lt.0) then
!     how many points
                     itmp2 = - npts_node_u(ipt2)
                     itmp1 = min(itmp2,itmp)

!     update mid; local operation
                     if(npts_node_l(ipr).gt.0) then
!     what point range
                        itmp2 = npts_node_g(ipr) -
     $                       npts_node_u(ipr) - npts_node_h(ipr) -
     $                       npts_node_r(ipr) + 1
                        itmp3 = itmp2 + itmp1 - 1

!     does current node fit into this range
                        itmp2 = max(1,itmp2)
                        itmp3 = min(npts_node_l(ipr),itmp3)

                        if ((itmp2.le.npts_node_l(ipr)).and.
     $                       (itmp3.ge.1)) then
!     node id; to send
                           nodeid1 = npts_plist(ipt2)
!     count local points
                           itmp4 = 0
                           itmp5 = 0
                           do ipt3 = 1,npts
                              if (mid(ipt3).eq.nodeid) then
                                 itmp4 = itmp4 + 1
                                 if (itmp4.ge.itmp2.and.
     $                                itmp4.le.itmp3) then
!     redirect point
                                    mid(ipt3) = nodeid1
!     count local points
                                    itmp5 = itmp5 + 1
                                 endif
                              endif
                           enddo ! ipt3

!     update local variables
                           nptupl = nptupl + itmp5
                           npts_node_l(ipr) = npts_node_l(ipr)  - itmp5
                           npts_node_l(ipt2)= npts_node_l(ipt2) + itmp5
                           npts_node_r(ipr) = npts_node_r(ipr)  + itmp5

                        endif   ! npts_node_l(ipr).gt.0

                     endif      ! npts_node_l(ipt2).gt.0

!     update global variables
                     npts_node_u(ipt2) = npts_node_u(ipt2) + itmp1
                     npts_node_u(ipr)  = npts_node_u(ipr)  - itmp1
                     itmp = itmp - itmp1
                     nptupg = nptupg + itmp1

                  endif         ! npts_node_u(ipt2).lt.0

                  if (itmp.eq.0) goto 100
               enddo            ! ipt2

 100           continue

!     check consistency
               ierr = iglsum(nptupl,1)
               if(ierr.ne.nptupg) then
                  if(NIO.eq.0) write(6,*)
     $                 'Error: pts_map_set; wrong nptupl 2'
                  call exitt
               endif

!     final update
!     update done/undone point number; related to this processor
!     global
               nptgdone   = nptgdone   + nptupg
               nptgundone = nptgundone - nptupg
!     local
               nptldone   = nptldone   + nptupl
               nptlundone = nptlundone - nptupl
!     count overflow/empty/shift
               nptempty = nptempty - nptupg
               nptover  = nptover  - nptupg
               nptshift = nptshift + nptupg

            endif               ! nptempty.gt.0

         else                   ! npts_node_g(ipr).gt.nptmax
!     all directed points can be send to given proc
!     check average number of points
            if (npts_node_g(ipr).gt.nptav1) then
!     no more space for points
!     do not change mid
!     update done/undone point number
!     global
               nptgdone   = nptgdone   + npts_node_g(ipr)
               nptgundone = nptgundone - npts_node_g(ipr)
!     local
               nptldone   = nptldone   + npts_node_l(ipr)
               nptlundone = nptlundone - npts_node_l(ipr)
!     mark processor as done
!     sending - positive, receiving - negative, zeor - done
               npts_node_u(ipr) = 0

            else                ! npts_node_g(ipr).gt.nptav1
!     this processor should receive more points

!     initial update done/undone point number; related to this processor
!     global
               nptgdone = nptgdone + npts_node_g(ipr)
               nptgundone = nptgundone - npts_node_g(ipr)
!     local
               nptldone = nptldone + npts_node_l(ipr)
               nptlundone = nptlundone - npts_node_l(ipr)

!     mark processor
!     sending - positive, receiving - negative, zero - done
               itmp = nptav1 - npts_node_g(ipr)
               npts_node_u(ipr) = - itmp
!     count overflow/empty/shift
               nptempty = nptempty + itmp

!     check if threre are any points to redistribute
!     count the points
               nptupg = 0       ! global
               nptupl = 0       ! local
               if (nptover.gt.0) then

!     loop over all nodes with smaller id
                  do ipt2=1,ipr - 1
                     if(npts_node_u(ipt2).gt.0) then
!     how many points
                        itmp2 = npts_node_u(ipt2)
                        itmp1 = min(itmp2,itmp)

!     update mid; local operation
                        if(npts_node_l(ipt2).gt.0) then
!     what point range
                           itmp2 = npts_node_g(ipt2) -
     $                          npts_node_u(ipt2) - npts_node_h(ipt2) -
     $                          npts_node_r(ipt2) + 1
                           itmp3 = itmp2 + itmp1 - 1

!     does current node fit into this range
                           itmp2 = max(1,itmp2)
                           itmp3 = min(npts_node_l(ipt2),itmp3)
                           if ((itmp2.le.npts_node_l(ipt2)).and.
     $                          (itmp3.ge.1)) then
!     node id
                              nodeid1 = npts_plist(ipt2)
!     count local points
                              itmp4 = 0
                              itmp5 = 0
                              do ipt3 = 1,npts
                                 if (mid(ipt3).eq.nodeid1) then
                                    itmp4 = itmp4 + 1
                                    if (itmp4.ge.itmp2.and.
     $                                   itmp4.le.itmp3) then
!     redirect point
                                       mid(ipt3) = nodeid
!     count local points
                                       itmp5 = itmp5 + 1
                                    endif
                                 endif
                              enddo ! ipt3

!     update local variables
                              nptupl = nptupl + itmp5
                              npts_node_l(ipr) = npts_node_l(ipr)
     $                             + itmp5
                              npts_node_l(ipt2)= npts_node_l(ipt2)
     $                             - itmp5
                              npts_node_r(ipt2)= npts_node_r(ipt2)
     $                             + itmp5

                           endif ! npts_node_l(ipt2).gt.0

                        endif   ! npts_node_l(ipt2).gt.0

!     update global variables
                        npts_node_u(ipt2) = npts_node_u(ipt2) - itmp1
                        npts_node_u(ipr)  = npts_node_u(ipr) + itmp1
                        itmp = itmp - itmp1
                        nptupg = nptupg + itmp1

                     endif      ! npts_node_u(ipt2).gt.0

                     if (itmp.eq.0) goto 200
                  enddo         ! ipt2

 200              continue

!     check consistency
                  ierr = iglsum(nptupl,1)
                  if(ierr.ne.nptupg) then
                     if(NIO.eq.0) write(6,*)
     $                    'Error: pts_map_set; wrong nptupl 3'
                     call exitt
                  endif

!     final update
!     update done/undone point number; related to this processor
!     global
                  nptgdone   = nptgdone   + nptupg
                  nptgundone = nptgundone - nptupg
!     local
                  nptldone   = nptldone   + nptupl
                  nptlundone = nptlundone - nptupl
!     count overflow/empty/shift
                  nptempty = nptempty - nptupg
                  nptover  = nptover  - nptupg
                  nptshift = nptshift + nptupg

               endif            ! nptover.gt.0

            endif               ! npts_node_g(ipr).gt.nptav1

         endif                  ! npts_node_g(ipr).gt.nptmax
      enddo                     ! ipr

!     set node status
      do ipr =1,nplist
         if (NID.eq.npts_plist(ipr)) then
            istatus = npts_node_u(ipr)
            indasg  = npts_node_g(ipr)
            go to 300
         endif
      enddo

 300  continue

      return
      end
c-----------------------------------------------------------------------
!     global scan
      subroutine ivgl_running_sum(out,in,n)
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
      integer n
      integer out(n),in(n)

      call mpi_scan(in,out,n,mpi_integer,mpi_sum,nekcomm,ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine ibcastn(buf,len,sid)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer len,sid
      integer buf(len)

      call mpi_bcast (buf,len,mpi_integer,sid,nekcomm,ierr)

      return
      end
c-----------------------------------------------------------------------
