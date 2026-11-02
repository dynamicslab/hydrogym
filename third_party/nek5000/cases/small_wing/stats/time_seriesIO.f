!=======================================================================
! Name        : time_seriesIO
! Author      : Adam Peplinski
! Version     : last modification 2015.05.20
! Copyright   : GPL
! Description : This is a set of routines to write time series to 
!     the file. They are modiffication of the existing nek5000 routines
!=======================================================================
!     this is just modification of mfo_outfld
!     I have to write the header, global point numberring, point 
!     position and the interior of the buffer
!     muti-file output
      subroutine stat_mfo_outpts(fieldbuff,tmlist,ltsnap,nflds,ntlist)  

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'PTSTAT'          ! point time series for 2D statistics

!     arguments
      real fieldbuff(nfldm,LHIS,ltsnap) ! fields to write
      real tmlist(ltsnap)       ! snapshot time
      integer ltsnap            ! array size
      integer nflds             ! number of fields
      integer ntlist            ! number of snapshots

!     local variables
      real tiostart, tio        ! timing
      integer ierr              ! error mark
      integer il, jl, kl        ! loop index
      character*3 prefix        ! file prefix
      
      integer wdsizol           ! store global wdsizo
      integer nptsB             ! running sum for npts
      integer nelBl             ! store global nelB

      integer*8 offs0, offs     ! offset      
      integer*8 stride,strideB  ! stride

      integer ioflds            ! fields count

!     dummy arrays
      real ur1(LHIS)
      common /SCRCH/  ur1

!     byte sum
      real dnbyte

!     functions
      integer igl_running_sum
      real dnekclock_sync, glsum

!     simple timing
      tiostart=dnekclock_sync()

!     force double precission
      wdsizol = WDSIZO
      WDSIZO = WDSIZE

!     get number of points on proceesor with smaller nid
      ierr = npts
      nptsB = igl_running_sum(ierr)
      nptsB = nptsB - npts

!     replace value
      nelBl = NELB
      NELB = nptsB

!     open files on i/o nodes
      prefix='pts'
      ierr=0
      if (NID.eq.PID0) call mfo_open_files(prefix,ierr)

      call err_chk(ierr,'Error opening file in mfo_open_files. $')

!     write header, byte key, global ordering, time snapshots
      call stat_mfo_write_hdrpts(tmlist,ntlist,nflds)

!     initial offset: header, test pattern, tiem list, global ordering
      offs0 = iHeaderSize + 4 + WDSIZO*ntlist + ISIZE*npoints
      offs = offs0

!     stride
      strideB = NELB * wdsizo
      stride  = npoints * wdsizo

!     count fields
      ioflds = 0

!     write coordinates
      do jl=1,NDIM
!     copy vector
         do il=1,npts
            ur1(il) = pts(jl,il)
         enddo
!     offset
         offs = offs0 + stride*ioflds + strideB
         call byte_set_view(offs,IFH_MBYTE)
         call stat_mfo_outspts(ur1,npts)
         ioflds = ioflds + 1
      enddo

!     write fields
      do kl=1,ntlist
!     field loop
         do jl=1,nflds
!     copy vector
            do il=1,npts
               ur1(il) = fieldbuff(jl,il,kl)
            enddo
!     offset
            offs = offs0 + stride*ioflds + strideB
            call byte_set_view(offs,IFH_MBYTE)
            call stat_mfo_outspts(ur1,npts)
            ioflds = ioflds + 1
         enddo
      enddo

!     count bytes
      dnbyte = 1.*ioflds*npts*WDSIZO

      ierr = 0
      if (NID.eq.PID0) 
#ifdef MPIIO
     &     call byte_close_mpi(IFH_MBYTE,ierr)
#else
     &     call byte_close(ierr)
#endif
      call err_chk(ierr,
     $     'Error closing file in stat_mfo_outpts. Abort. $')

      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4 + WDSIZO * ntlist +
     $     ISIZE*npoints
      dnbyte = dnbyte/1024/1024
      if(NIO.eq.0) write(6,7) ISTEP,TIME,dnbyte,dnbyte/tio,
     &     NFILEO
    7 format(/,i9,1pe12.4,' done :: Write checkpoint',/,
     &     30X,'file size = ',3pG12.2,'MB',/,
     &     30X,'avg data-throughput = ',0pf7.1,'MB/s',/,
     &     30X,'io-nodes = ',i5,/)

!     restore old values
      WDSIZO = wdsizol
      NELB = nelBl

      return
      end
c-----------------------------------------------------------------------
!     based on mfo_write_hdr
!     write hdr, byte key, global ordering, time list
      subroutine stat_mfo_write_hdrpts(tmlist,ntlist,nflds)

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'PTSTAT'          ! point time series for 2D statistics

!     argumet list
      real tmlist(ntlist)       ! snapshot time
      integer ntlist            ! number of snapshots
      integer nflds             ! number of fields

!     local variables
      real*4 test_pattern       ! byte key
      integer lglist(0:LHIS) ! dummy array
      common /ctmp0/ lglist
      integer idum, inelp
      integer nelo              ! number of points to write
      integer nfileoo           ! number of files to create
      
      integer j                 ! loop index
      integer mtype             ! tag

      integer ierr              ! error mark
      integer ibsw_out, len
      integer*8 ioff            ! offset

      character*132 hdr         ! header

      call nekgsync()
      idum = 1

#ifdef MPIIO
      nfileoo = 1   ! all data into one file
      nelo = npoints
#else
      nfileoo = NFILEO
      if(NID.eq.PID0) then                ! how many elements to dump
        nelo = npts
        do j = PID0+1,PID1
           mtype = j
           call csend(mtype,idum,ISIZE,j,0)   ! handshake
           call crecv(mtype,inelp,ISIZE)
           nelo = nelo + inelp
        enddo
      else
        mtype = NID
        call crecv(mtype,idum,ISIZE)          ! hand-shake
        call csend(mtype,npts,ISIZE,PID0,0)   ! u4 :=: u8
      endif 
#endif

!     write header
      ierr = 0
      if(NID.eq.PID0) then
         call blank(hdr,132)    
 
         write(hdr,1) WDSIZO,nelo,npoints,ntlist,nflds,TIME,
     $        fid0,nfileoo
 1       format('#std',1x,i1,1x,i10,1x,i10,1x,i10,1x,i4,1x,e20.13,
     $        1x,i6,1x,i6)

!     if we want to switch the bytes for output
!     switch it again because the hdr is in ASCII
         call get_bytesw_write(ibsw_out)
c      if (ibsw_out.ne.0) call set_bytesw_write(ibsw_out)
         if (ibsw_out.ne.0) call set_bytesw_write(0)  

!     write test pattern for byte swap
         test_pattern = 6.54321 

#ifdef MPIIO
! only rank0 (pid00) will write hdr + test_pattern + time list
         call byte_write_mpi(hdr,iHeaderSize/4,PID00,IFH_MBYTE,ierr)
         call byte_write_mpi(test_pattern,1,PID00,IFH_MBYTE,ierr)
         len = WDSIZO/4 * ntlist
         call byte_write_mpi(tmlist,len,PID00,IFH_MBYTE,ierr)
#else
         call byte_write(hdr,iHeaderSize/4,ierr)
         call byte_write(test_pattern,1,ierr)
         len = WDSIZO/4 * ntlist
         call byte_write(tmlist,len,ierr)
#endif

      endif

      call err_chk(ierr,
     $     'Error writing header in stat_mfo_write_hdrpts. $')

      ! write global point numbering for this group
      if(NID.eq.PID0) then
#ifdef MPIIO
         ioff = iHeaderSize + 4 + WDSIZO * ntlist + NELB*ISIZE
         call byte_set_view (ioff,IFH_MBYTE)
         call byte_write_mpi(ipts,npts,-1,IFH_MBYTE,ierr)
#else
         call byte_write(ipts,npts,ierr)
#endif
         do j = PID0+1,PID1
            mtype = j
            call csend(mtype,idum,ISIZE,j,0) ! handshake
            len = ISIZE*(LHIS+1)
            call crecv(mtype,lglist,len)
            if(ierr.eq.0) then
#ifdef MPIIO
               call byte_write_mpi
     $              (lglist(1),lglist(0),-1,IFH_MBYTE,ierr)
#else
               call byte_write(lglist(1),lglist(0),ierr)
#endif
            endif
         enddo
      else
         mtype = NID
         call crecv(mtype,idum,ISIZE) ! hand-shake
        
         lglist(0) = npts
         call icopy(lglist(1),ipts,npts)

         len = ISIZE*(npts+1)
         call csend(mtype,lglist,len,PID0,0)  
      endif 

      call err_chk(ierr,
     $     'Error writing global nums in stat_mfo_write_hdrpts. $')
      return
      end
c-----------------------------------------------------------------------
!     based on mfo_outs
      subroutine stat_mfo_outspts(u,nel)

      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'

!     argument list
      real u(nel)
      integer nel

!     local variables
      integer len, leo, nout
      integer idum, ierr
      integer k, mtype

!     dummy arrays
      integer lw2
      parameter (lw2=LX1*LY1*LZ1*LELT)
      common /SCRVH/ u4(2+2*lw2)
      real*4         u4
      real*8         u8(1+lw2)
      equivalence    (u4,u8)

      call nekgsync() ! clear outstanding message queues.
      if(LHIS.gt.lw2) then
        if(NIO.eq.0) write(6,*) 'ABORT: lw2 too small'
        call exitt
      endif

      len  = 8 + 8*LHIS  ! recv buffer size
      leo  = 8 + WDSIZO*nel

      idum = 1
      ierr = 0

      if (NID.eq.PID0) then

         if (WDSIZO.eq.4) then             ! 32-bit output
             call copyx4 (u4,u,nel)
         else
             call copy   (u8,u,nel)
         endif
         nout = WDSIZO/4 * nel
         if(ierr.eq.0) 
#ifdef MPIIO
     &     call byte_write_mpi(u4,nout,-1,ifh_mbyte,ierr)
#else
     &     call byte_write(u4,nout,ierr)          ! u4 :=: u8
#endif

         ! write out the data of my childs
         idum  = 1
         do k=PID0+1,PID1
            mtype = k
            call csend(mtype,idum,4,k,0)       ! handshake
            call crecv(mtype,u4,len)
            nout  = WDSIZO/4 * u8(1)
            if (WDSIZO.eq.4.and.ierr.eq.0) then
#ifdef MPIIO
               call byte_write_mpi(u4(3),nout,-1,IFH_MBYTE,ierr)
#else
               call byte_write(u4(3),nout,ierr)
#endif
            elseif(ierr.eq.0) then
#ifdef MPIIO
               call byte_write_mpi(u8(2),nout,-1,IFH_MBYTE,ierr)
#else
               call byte_write(u8(2),nout,ierr)
#endif
            endif
         enddo

      else

         u8(1)= nel
         if (WDSIZO.eq.4) then             ! 32-bit output
             call copyx4 (u4(3),u,nel)
         else
             call copy   (u8(2),u,nel)
         endif

         mtype = NID
         call crecv(mtype,idum,4)            ! hand-shake
         call csend(mtype,u4,leo,pid0,0)     ! u4 :=: u8

      endif

      call err_chk(ierr,
     $     'Error writing data to .f00 in stat_mfo_outspts. $')

      return
      end
c-----------------------------------------------------------------------
