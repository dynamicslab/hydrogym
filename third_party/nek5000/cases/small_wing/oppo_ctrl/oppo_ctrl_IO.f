c==============================================
c I/O for saving relevent file for opposition control  
c Yuning Wang 
c==============================================


c-----------------------------------------------------------------------
!     Output points for statistics point time history
!     To reduce number of wirting to the disc I collect some number of 
!     time snapshots
!     and later on write the whole set to the disc
      subroutine ctrl_pts_out()
c=============================================
c       Define variable
c=============================================
      implicit none
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include "OPPO_CTL"
      
!     arguments
      integer npts              ! local number of points
      real fieldout(nfldc,totctrl)
      integer nflds             ! number of fields
      
!     local variables
      integer ltsnap            ! number of snapshots (MA: per pts file?) 
      parameter(ltsnap=2)
      real buffer(nfldc,totctrl,ltsnap) ! buffer for snapshots
      real tmlist(ltsnap)       ! snapshot time
      save buffer, tmlist
      integer istcount          ! step interval for data collection

      integer icl             ! call counter for control 
      save icl
      data icl /0/            ! Save it to another common block, differ from PTS

      integer isize             !
      ! parameter (isize = nfldc*totctrl)

!     I/O variables
      integer wrt_ctrl,outp_ctrl

!     collect data
!     count calls
c=============================================
c       Function
c=============================================

c Step 1: Parameter 
c------------------------------
      isize = nfldc*totctrl

! YW: Added 2 New Parameters in .rea File
      wrt_ctrl  = int(PARAM(90))
      outp_ctrl = int(PARAM(91))

! YW: this is directly related to statistics code; 
! probably should be changed in the future
      istcount = int(PARAM(51))

c Step 2: Add current series to buffer 
c------------------------------
! Note that If Buffer is full, we stop writting 
      if (mod(ISTEP,wrt_ctrl).eq.0 .and. 
     $      icl.lt.ltsnap) then
            
            icl = icl + 1
      
            call copy(buffer(1,1,icl),vctl(icl,1),isize)
            
            tmlist(icl) = TIME
      
      endif ! if(mod(ISTEP,wrt_ctrl).eq.0 .and. icl.le.ltsnap)
      
c Step 3: Dumping the buffer 
c------------------------------
cc    IF this is the time step to dump OR this is the last step

      if( mod(ISTEP,outp_ctrl).eq.0 .or. 
     $   (NSTEPS-ISTEP).lt.istcount) then

         call oppo_mfo_outpts(buffer,tmlist,ltsnap,nfldc,icl)
      
         if(NIO.eq.0) print *,"CTRL PTS Saved!"

      ! reset counter
         icl = 0
      
      endif 

      return
      end




      subroutine oppo_mfo_outpts(fieldbuff,tmlist,ltsnap,nflds,ntlist)
cc: Save the opposition control output
cc Args: 
c=============================================
c       Define variable
c=============================================
                
        implicit none
cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
        include 'SIZE'
cc MA:      include 'PARALLEL_DEF'
        include 'PARALLEL'
cc MA:      include 'RESTART_DEF'
        include 'RESTART'
cc MA:      include 'TSTEP_DEF'
        include 'TSTEP'
        include 'OPPO_CTL'          !  Common Block for Opposition control
!       arguments
        
        integer ltsnap            ! array size
        integer nflds             ! number of fields
        integer ntlist            ! number of snapshots
        character*3 prefix        ! Prefix for the file 
        real fieldbuff(nflds,totctrl,ltsnap) ! fields to write
        real tmlist(ltsnap)       ! snapshot time

!       local variables
        real tiostart, tio        ! timing
        integer ierr              ! error mark
        integer il, jl, kl        ! loop index
        integer wdsizol           ! store global wdsizo
        integer nptsB             ! running sum for npts
        integer nelBl             ! store global nelB

        integer*8 offs0, offs     ! offset      
        integer*8 stride,strideB  ! stride
        integer ioflds            ! fields count
!       dummy arrays
        real ur2(totctrl)
        common /SCRCH/  ur2

!       byte sum
        real dnbyte

!       functions
        integer igl_running_sum
        real dnekclock_sync, glsum
    
c=============================================
c       Function
c=============================================
!       simple timing
        tiostart=dnekclock_sync()
!       force double precission
        wdsizol = WDSIZO
        WDSIZO = WDSIZE
!       get number of points on proceesor with smaller nid
        ierr = numctrl
        nptsB = igl_running_sum(ierr)
        nptsB = nptsB - numctrl
!     replace value
        nelBl = NELB
        NELB = nptsB

cc Step 1: initialization 
c--------------------------------------------
!     open files on i/o nodes
      prefix='vfl'
      ierr=0
      if (NID.eq.PID0) call mfo_open_files(prefix,ierr)

      call err_chk(ierr,'Error opening file in mfo_open_files. $')

!     write header, byte key, global ordering, time snapshots
      call oppo_mfo_write_hdrpts(tmlist,ntlist,nflds) ! YW modified here 
      ! Note after this function we should know 
      ! how many ctrl points we have in total

!     initial offset: header, test pattern, tiem list, global ordering
      offs0 = iHeaderSize + 4 + WDSIZO*ntlist + ISIZE*npoints
      offs = offs0

!     stride
      strideB = NELB * wdsizo
      stride  = npoints * wdsizo

!     count fields
      ioflds = 0

cc Step 2: Write the coordinates
c---------------------------------------------
cc YW     write coordinates for Detection Plane 
      do jl=1,NDIM
            call rzero(ur2,totctrl)
            !     copy vector
            if (numctrl.gt.0) then 
                  do il=1,numctrl
                        ur2(il) = crdctl(jl,il) ! YW modified here 
                  enddo
            endif
!     offset
         offs = offs0 + stride*ioflds + strideB
         call byte_set_view(offs,IFH_MBYTE)
         call oppo_mfo_outspts(ur2,numctrl) ! YW modified here 
         ioflds = ioflds + 1
      enddo ! 
c---------------------------------------------

cc Step 3: Write the quantities 
c---------------------------------------------

!     write fields
      do kl=1,ntlist
      !     field loop
         call rzero(ur2,totctrl)
         do jl=1,nflds ! YW: modified here 
!     copy vector
            if (numctrl.gt.0) then
            do il=1,numctrl ! YW: modified here 
               ur2(il) = fieldbuff(jl,il,kl)
            enddo
            endif 
!     offset
            offs = offs0 + stride*ioflds + strideB
            call byte_set_view(offs,IFH_MBYTE)
            call oppo_mfo_outspts(ur2,numctrl) ! YW: modified here 
            ioflds = ioflds + 1
         enddo
      enddo
c---------------------------------------------

cc Step 4: Close the file and write down 
c---------------------------------------------
!     count bytes
      dnbyte = 1.*ioflds*totctrl*WDSIZO ! YW: modified here 

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
     $     ISIZE*npoints ! YW: modified here note the npoints is not the on in PTSTAT
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
c---------------------------------------------

      return
      end
c-----------------------------------------------------------------------




c-----------------------------------------------------------------------
!     based on mfo_write_hdr
!     write hdr, byte key, global ordering, time list
      subroutine oppo_mfo_write_hdrpts(tmlist,ntlist,nflds)
cc YW: Write down the title, time, byte key, etc 
c=============================================
c       Define variable
c=============================================
      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'OPPO_CTL'          ! Common block for opposition control 

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
c=============================================
c       Function
c=============================================

      call nekgsync()
      idum = 1
c Step 1: Get Title Info
c-------------------------------
      
c 1.1 Counting how many ctrl pts in total 
cc YW: As the distribution on MPI RANK is not uniformed as TimeSeries
cc We can do a Communication here 
c----------------------------------------
c Calculation of points
c--------------------------------------
      if (npoints.eq.0) call count_total_ctrlpts
      ! npoints = igl_running_sum(numctrl)

#ifdef MPIIO
      nfileoo=1
      nelo=npoints
#else
      nfileoo = NFILEO
      if(NID.eq.PID0) then                ! how many elements to dump
            nelo = totctrl
            do j = PID0+1,PID1
            mtype = j
            call csend(mtype,idum,ISIZE,j,0)   ! handshake
            call crecv(mtype,inelp,ISIZE)
            nelo = nelo + inelp
            enddo
      else
            mtype = NID
            call crecv(mtype,idum,ISIZE)          ! hand-shake
            call csend(mtype,totctrl,ISIZE,PID0,0)   ! u4 :=: u8
      endif !(if (NID.eq.PID0))
#endif


c----------------------------------------

c 1.2 Write the title here 
c--------------------------------------------------------------
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

      endif ! (IF NID.eq.PID0)

      
      call err_chk(ierr,
     $     'Error writing header in stat_mfo_write_hdrpts. $')
c-------------------------------------------------------------


c Step 3: Write the Global indicies of the control points
cc This corresponds to the working array for findpts(), I name it as iptctrl 
c--------------------------------------------------------------
      ! write global point numbering for this group
      if(NID.eq.PID0) then
#ifdef MPIIO
         ioff = iHeaderSize + 4 + WDSIZO * ntlist + NELB*ISIZE
         call byte_set_view (ioff,IFH_MBYTE)
         call byte_write_mpi(iptctl,totctrl,-1,IFH_MBYTE,ierr) !YW modified 
#else
         call byte_write(iptctl,totctrl,ierr)!YW modified 
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
        
         lglist(0) = numctrl
         call icopy(lglist(1),iptctl,numctrl) !YW modified 

         len = ISIZE*(numctrl+1)

         call csend(mtype,lglist,len,PID0,0)  
      endif ! if (NID.eq.PID0)

      call err_chk(ierr,
     $     'Error writing global nums in stat_mfo_write_hdrpts. $')
      return
      end
c----------------------------------------------------------------




c-----------------------------------------------------------------------
!     based on mfo_outs
      subroutine oppo_mfo_outspts(u,nel)
cc YW: Write the data into the current binary file
c=============================================
c       Define variable
c=============================================
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
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


c=============================================
c       Function
c=============================================
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



c-----------------------------------------------------------------------
      subroutine count_total_ctrlpts
c=============================================
c       Define variable
c=============================================
      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'OPPO_CTL'          ! Common block for opposition control 

      real listp1(LP)           ! for Summup the control points
      real listp2(LP)           ! For summup the control points
      integer j 
c=============================================
c     Function
c=============================================
      call nekgsync()

      listp1(NID)=numctrl
      npoints=0
      
      call gop(listp1,listp2,"+  ",LP)

      do j=1,LP
            npoints=npoints + int(listp1(j))
      enddo
      ! A useless parameter for keep consistency with the old code 
      if (NID.eq.0) print *, "Total control points =",npoints
      
      
      return
      end
c-----------------------------------------------------------------------