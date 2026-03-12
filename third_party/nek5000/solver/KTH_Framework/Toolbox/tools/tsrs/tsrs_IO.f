!> @file tsrs_IO.f
!! @ingroup tsrs
!! @brief I/O routines for time series module
!! @details This is a set of routines to write statistics correlated history  
!!  points to the file. They are modiffication of the existing nek5000 routines
!! @author Adam Peplinski
!! @date May 15, 2021
!=======================================================================
!> @brief Write a point history file 
!! @ingroup tsrs
!! @details This routine is based on mfo_outfld adopted for point data
!! @param[in]   bff    buffer to write
!! @param[in]   lbff   buffer size 
      subroutine tsrs_mfo_outfld(bff,lbff)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSTEP'
      include 'TSRSD'

      ! argument list
      integer lbff
      real bff(lbff)

      ! local variables
      real tiostart, tio        ! timing
      integer itmp
      integer ierr              ! error flag
      character*3 prefix        ! file prefix
      
      integer wdsizol           ! store global wdsizo
      integer nptsB             ! running sum for npts
      integer nelBl             ! store global nelB

      integer*8 offs0, offs     ! offset      
      integer*8 stride,strideB  ! stride

      integer ioflds            ! fields count

!     byte sum
      real dnbyte

!     functions
      integer igl_running_sum
      real dnekclock_sync, glsum
!----------------------------------------------------------------------
      tiostart=dnekclock_sync()
      call io_init
      ! this is mesh speciffic, so some variables must be overwritten
      ! how many ele are present up to rank nid
      itmp = tsrs_npts
      nptsB = igl_running_sum(itmp)
      nptsB = nptsB - tsrs_npts
      ! replace value
      nelBl = NELB
      NELB = nptsB

      ! force double precission
      wdsizol = WDSIZO
      WDSIZO = WDSIZE

      ! open files on i/o nodes
      prefix='pts'
      ierr=0
      if (nid.eq.pid0) call mfo_open_files(prefix,ierr)
      call mntr_check_abort(tsrs_id,ierr,
     $     'Error openning file')

      ! write header
      call tsrs_mfo_write_hdr()

      ! initial offset: header, test pattern, tiem list, global ordering
      offs0 = iheadersize + 4 + wdsizo*tsrs_ntsnap + isize*tsrs_nptot
      offs = offs0
      ! stride
      strideb = nelb * wdsizo
      stride  = tsrs_nptot * wdsizo

      ! count fields
      ioflds = 0

      ! write coordinates in a single call
      ! offset
      offs = offs0 + stride*ioflds + strideb*ndim
      call byte_set_view(offs,ifh_mbyte)
      itmp = tsrs_npts*ndim
      call tsrs_mfo_outs(tsrs_pts,itmp,ierr)
      call mntr_check_abort(tsrs_id,ierr,
     $     'Error writing point coordinates')
      ioflds = ioflds + ndim

      ! write fields in a single call
      ! offset
      offs = offs0 + stride*ioflds + strideB*tsrs_nfld*tsrs_ntsnap
      call byte_set_view(offs,ifh_mbyte)
      itmp = tsrs_npts*tsrs_nfld*tsrs_ntsnap
      call tsrs_mfo_outs(bff,itmp,ierr)
      call mntr_check_abort(tsrs_id,ierr,
     $     'Error writing interpolated field')
      ioflds = ioflds + tsrs_nfld*tsrs_ntsnap

      if (nid.eq.pid0) then 
         if(ifmpiio) then
           call byte_close_mpi(ifh_mbyte,ierr)
         else
           call byte_close(ierr)
         endif
      endif
      call mntr_check_abort(tsrs_id,ierr,
     $     'Error closing file')

      ! count bytes
      dnbyte = 1.*ioflds*tsrs_npts*wdsizo
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + wdsizo * tsrs_ntsnap +
     $     isize*tsrs_nptot

      dnbyte = dnbyte/1024/1024
      if(nio.eq.0) write(6,7) istep,time,dnbyte,dnbyte/tio,
     &             nfileo
    7 format(/,i9,1pe12.4,' done :: Write checkpoint',/,
     &       30X,'file size = ',3pG12.2,'MB',/,
     &       30X,'avg data-throughput = ',0pf7.1,'MB/s',/,
     &     30X,'io-nodes = ',i5,/)

      ! restore old values
      wdsizo = wdsizol
      nelb = nelbl

      return
      end subroutine
!=======================================================================
!> @brief Write file header 
!! @ingroup tsrs
!! @details This routine is based on mfo_write_hdr adopted for point data
      subroutine tsrs_mfo_write_hdr()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'PARALLEL'
      include 'TSTEP'
      include 'TSRSD'

      ! local variables
      integer ierr              ! error flag
      real*4 test_pattern       ! byte key
      integer lglist(0:lhis)    ! dummy array

      integer idum, inelp
      integer nelo              ! number of points to write
      integer nfileoo           ! number of files to create
      
      integer jl                ! loop index
      integer mtype             ! tag

      integer ibsw_out, len
      integer*8 ioff            ! offset

      character*132 hdr         ! header
!----------------------------------------------------------------------
      ierr = 0

      call nekgsync()
      idum = 1

      if(ifmpiio) then
        nfileoo = 1   ! all data into one file
        nelo = tsrs_nptot
      else
        nfileoo = nfileo
        if(nid.eq.pid0) then                ! how many elements to dump
          nelo = tsrs_npts
          do jl = pid0+1,pid1
             mtype = jl
             call csend(mtype,idum,4,jl,0)   ! handshake
             call crecv(mtype,inelp,4)
             nelo = nelo + inelp
          enddo
        else
          mtype = nid
          call crecv(mtype,idum,4)          ! hand-shake
          call csend(mtype,tsrs_npts,4,pid0,0)   ! u4 :=: u8
        endif 
      endif

      ! write a header
      if(nid.eq.pid0) then
         call blank(hdr,132)    
 
         write(hdr,1) wdsizo,ldim,nelo,tsrs_nptot,tsrs_ntsnap,
     $        tsrs_nfld,time,fid0,nfileoo
 1       format('#std',1x,i1,1x,i1,1x,i10,1x,i10,1x,i10,1x,i4,1x,e20.13,
     $        1x,i6,1x,i6)

         ! write test pattern for byte swap
         test_pattern = 6.54321

         len = wdsizo/4 * tsrs_ntsnap
         if(ifmpiio) then
            ! only rank0 (pid00) will write hdr + test_pattern + time list
            call byte_write_mpi(hdr,iHeaderSize/4,pid00,ifh_mbyte,ierr)
            call byte_write_mpi(test_pattern,1,pid00,ifh_mbyte,ierr)
            call byte_write_mpi(tsrs_tmlist,len,pid00,ifh_mbyte,ierr)
         else
            call byte_write(hdr,iheadersize/4,ierr)
            call byte_write(test_pattern,1,ierr)
            call byte_write(tsrs_tmlist,len,ierr)
         endif
      endif

      call mntr_check_abort(tsrs_id,ierr,
     $     'Error writing header')

      ! write global point number
      if(nid.eq.pid0) then
         if(ifmpiio) then
            ioff = iheadersize + 4 + wdsizo * tsrs_ntsnap + nelb*isize
            call byte_set_view (ioff,ifh_mbyte)
            call byte_write_mpi(tsrs_ipts,tsrs_npts,-1,ifh_mbyte,ierr)
         else
            call byte_write(tsrs_ipts,tsrs_npts,ierr)
         endif

         do jl = pid0+1,pid1
            mtype = jl
            call csend(mtype,idum,isize,jl,0) ! handshake
            len = isize*(lhis+1)
            call crecv(mtype,lglist,len)
            if(ierr.eq.0) then
               if(ifmpiio) then
                  call byte_write_mpi
     $                 (lglist(1),lglist(0),-1,ifh_mbyte,ierr)
               else
                  call byte_write(lglist(1),lglist(0),ierr)
               endif
            endif
         enddo
      else
         mtype = nid
         call crecv(mtype,idum,isize) ! hand-shake
        
         lglist(0) = tsrs_npts
         call icopy(lglist(1),tsrs_ipts,tsrs_npts)

         len = isize*(tsrs_npts+1)
         call csend(mtype,lglist,len,pid0,0)  
      endif

      call mntr_check_abort(tsrs_id,ierr,
     $     'Error writing global point numbers')

      return
      end subroutine
!=======================================================================
!> @brief Write single field for a local set of points. 
!! @ingroup tsrs
!! @details This routine is just a modification of mfo_outs. The only reason
!!   to have it is to control dummy array sizes, as mfo_outs could fail 
!!   in this case. Otherwise it would be completely redundant.
!! @param[in]   ul    input array
!! @param[in]   lpts  array length
!! @param[out]  ierr  error flagg
!! @remark This routine uses global scratch space \a SCRVH
      subroutine tsrs_mfo_outs(ul,lpts,ierr)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSRSD'

      ! global memory space
      ! dummy arrays
      integer lw2
      parameter (lw2=tsrs_nfld*tsrs_ltsnap*lhis)
      real*4 u4(2+2*lw2)
      common /SCRVH/ u4
      real*8         u8(1+lw2)
      equivalence    (u4,u8)

      ! argument list
      real ul(lpts)
      integer lpts, ierr

      ! local variables
      integer len, leo, nout
      integer idum
      integer kl, mtype
!----------------------------------------------------------------------
      ! clear outstanding message queues
      call nekgsync()

      len  = 8 + 8*lw2  ! recv buffer size
      leo  = 8 + wdsizo*lpts
      idum = 1
      ierr = 0

      if (nid.eq.pid0) then

         if (wdsizo.eq.4) then             ! 32-bit output
             call copyx4 (u4,ul,lpts)
         else
             call copy   (u8,ul,lpts)
         endif
         nout = wdsizo/4 * lpts
         if(ierr.eq.0) then 
            if(ifmpiio) then
               call byte_write_mpi(u4,nout,-1,ifh_mbyte,ierr)
            else
               call byte_write(u4,nout,ierr) ! u4 :=: u8
            endif
         endif

         ! write out the data of my childs
         idum  = 1
         do kl= pid0+1, pid1
            mtype = kl
            call csend(mtype,idum,4,kl,0)       ! handshake
            call crecv(mtype,u4,len)
            nout  = wdsizo/4 * u8(1)
            if (wdsizo.eq.4.and.ierr.eq.0) then
               if(ifmpiio) then
                  call byte_write_mpi(u4(3),nout,-1,ifh_mbyte,ierr)
               else
                  call byte_write(u4(3),nout,ierr)
               endif
            elseif(ierr.eq.0) then
               if(ifmpiio) then
                  call byte_write_mpi(u8(2),nout,-1,ifh_mbyte,ierr)
               else
                  call byte_write(u8(2),nout,ierr)
               endif
            endif
         enddo

      else

         u8(1)= lpts
         if (wdsizo.eq.4) then  ! 32-bit output
            call copyx4 (u4(3),ul,lpts)
         else
            call copy   (u8(2),ul,lpts)
         endif

         mtype = nid
         call crecv(mtype,idum,4) ! hand-shake
         call csend(mtype,u4,leo,pid0,0) ! u4 :=: u8

      endif

      return
      end subroutine
!=======================================================================
!> @brief Read interpolation points positions, number and redistribute them
!! @ingroup tsrs
      subroutine tsrs_mfi_points()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'TSRSD'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer il, jl            ! loop index
      integer ierr              ! error flag
      integer ldiml             ! dimesion of interpolation file
      integer nptsr             ! number of points in the file
      integer npass             ! number of messages to send
      integer itmp_ipts(lhis)
      real rtmp_pts(ldim,lhis)
      real*4 rbuffl(2*ldim*lhis)
      real rtmp1, rtmp2
      character*132 fname       ! file name
      integer hdrl
      parameter (hdrl=32)
      character*32 hdr          ! file header
      character*4 dummy
      real*4 bytetest
      integer iglob             ! global point numbering

      ! functions
      logical if_byte_swap_test
!-----------------------------------------------------------------------
      ! master opens files and gets point number
      ierr = 0
      if (nid.eq.pid00) then
         !open the file
         fname='int_pos'
         call byte_open(fname,ierr)

         ! read header
         call blank     (hdr,hdrl)
         call byte_read (hdr,hdrl/4,ierr)
         if (ierr.ne.0) goto 101

         ! big/little endian test
         call byte_read (bytetest,1,ierr)
         if(ierr.ne.0) goto 101
         if_byte_sw = if_byte_swap_test(bytetest,ierr)
         if(ierr.ne.0) goto 101

         ! extract header information
         read(hdr,*,iostat=ierr) dummy, wdsizr, ldiml, nptsr
      endif

 101  continue

      call mntr_check_abort(tsrs_id,ierr,
     $       'tsrs_mfi_points: Error opening point files')

      ! broadcast header data
      call bcast(wdsizr,isize)
      call bcast(ldiml,isize)
      call bcast(nptsr,isize)
      call bcast(if_byte_sw,lsize)

      ! check dimension consistency
      if (ldim.ne.ldiml) call mntr_abort(tsrs_id,
     $       'tsrs_mfi_points: Inconsisten dimension.')

      ! calculate point distribution; I assume it is post-processing
      ! done on small number of cores, so I assume nptsr >> mp
      tsrs_nptot = nptsr
      tsrs_npts = nptsr/mp
      if (tsrs_npts.gt.0) then
         tsrs_npt1 = mod(tsrs_nptot,mp)
      else
         tsrs_npt1 = tsrs_nptot
      endif
      if (nid.lt.tsrs_npt1) tsrs_npts = tsrs_npts +1

      ! stamp logs
      call mntr_logi(tsrs_id,lp_prd,
     $          'Interpolation point number :', tsrs_nptot)

      ierr = 0
      if (tsrs_npts.gt.lhis) ierr = 1
      call mntr_check_abort(tsrs_id,ierr,
     $       'tsrs_mfi_points: lhis too small')

      ! read and redistribute points
      ! this part is not optimised, but it is post-processing
      ! done locally, so I don't care
      if (nid.eq.pid00) then
         if (tsrs_nptot.gt.0) then
            ! read points for the master rank
            ldiml = ldim*tsrs_npts*wdsizr/4
            call byte_read (rbuffl,ldiml,ierr)

            ! get byte shift
            if (if_byte_sw) then
               if(wdsizr.eq.8) then
                  call byte_reverse8(rbuffl,ldiml,ierr)
               else
                  call byte_reverse(rbuffl,ldiml,ierr)
               endif
            endif

            ! copy data
            ldiml = ldim*tsrs_npts
            if (wdsizr.eq.4) then
               call copy4r(tsrs_pts,rbuffl,ldiml)
            else
               call copy(tsrs_pts,rbuffl,ldiml)
            endif
            ! provide global point numbers
            iglob = 0
            do il = 1, tsrs_npts
               iglob = iglob + 1
               tsrs_ipts(il) = iglob
            enddo

            ! redistribute rest of points
            npass = min(mp,tsrs_nptot)
            do il = 1,npass-1
               nptsr = tsrs_npts
               if (tsrs_npt1.gt.0.and.il.ge.tsrs_npt1) then
                  nptsr = tsrs_npts -1
               endif
               ! read points for the slave rank
               ldiml = ldim*nptsr*wdsizr/4
               call byte_read (rbuffl,ldiml,ierr)

               ! get byte shift
               if (if_byte_sw) then
                  if(wdsizr.eq.8) then
                     call byte_reverse8(rbuffl,ldiml,ierr)
                  else
                     call byte_reverse(rbuffl,ldiml,ierr)
                  endif
               endif

               ! copy data
               ldiml = ldim*nptsr
               if (wdsizr.eq.4) then
                  call copy4r(rtmp_pts,rbuffl,ldiml)
               else
                  call copy(rtmp_pts,rbuffl,ldiml)
               endif
               ! provide global point numbers
               do jl = 1, nptsr
                  iglob = iglob + 1
                  itmp_ipts(jl) = iglob
               enddo

               ! send data
               ldiml = ldiml*wdsizr
               call csend(il,rtmp_pts,ldiml,il,jl)
               ldiml = nptsr*isize
               call csend(il,itmp_ipts,ldiml,il,jl)
            enddo
         endif
      else
         if (tsrs_npts.gt.0) then
            call crecv2(nid,tsrs_pts,ldim*tsrs_npts*wdsize,0)
            call crecv2(nid,tsrs_ipts,tsrs_npts*isize,0)
         endif
      endif

      ! master closes files
      if (nid.eq.pid00) then
        call byte_close(ierr)
      endif

      return
      end subroutine
!=======================================================================




      
