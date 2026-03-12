!> @file arn_arp_io.f
!! @ingroup arn_arp
!! @brief Set of checkpointing routines for arn_arp module.
!=======================================================================
!> @brief Write restart files
!! @ingroup arn_arp
      subroutine arn_rst_save
      implicit none

      include 'SIZE'            ! NIO
      include 'TSTEP'           ! LASTEP
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! local variables
      character(20) str
!-----------------------------------------------------------------------
      ! save checkpoint for idoarp=-2
      if (idoarp.eq.-2) then
         call mntr_logi(arna_id,lp_prd,
     $       'Writing checkpoint; ido = ',idoarp)

         ! save parameters and workla; independent on processor;
         ! serial output
         call arn_write_par('ARP')

         ! save big arrays; parallel output
         call mfo_arnv('ARV')

         ! this is the last step
         LASTEP=1
      endif

      return
      end
!=======================================================================
!> @brief Read from checkpoints
!! @ingroup arn_arp
      subroutine arn_rst_read
      implicit none

      include 'SIZE'            ! NIO
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'
!-----------------------------------------------------------------------
      call mntr_log(arna_id,lp_prd,
     $       'Reading checkpoint.')

      ! read parameters and WORKLA; independent on processor; serial input
      call arn_read_par('ARP')

      ! read big arrays; parallel input
      call mfi_arnv('ARV')

      return
      end
!=======================================================================
!> @brief Write procesor independent data
!! @ingroup arn_arp
!! @param[in]   prefix    prefix
      subroutine arn_write_par(prefix)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! argument list
      character*3 prefix

      ! local variables
      character*132 fname
      character*6  str
      integer lwdsizo, lfid0, ierr
      logical lifreguo, lifmpiio
!-----------------------------------------------------------------------
      call nekgsync()
      call io_init()

      ! copy and set output parameters
      lwdsizo= WDSIZO
      WDSIZO = 8

      lifreguo= IFREGUO
      IFREGUO = .false.

      ! this is done by master node only, so serial writing
      lifmpiio = IFMPIIO
      IFMPIIO = .false.
      lfid0 = FID0
      FID0 = 0

      ierr = 0
      if (NID.eq.0) then
         ! create file name
         call io_mfo_fname(fname,SESSION,prefix,ierr)
         if (ierr.eq.0) then
            write(str,'(i5.5)') mod(arna_fnum,2) + 1
            fname = trim(fname)//trim(str)
            ! open file
            call io_mbyte_open(fname,ierr)
         endif

         if (ierr.eq.0) then
            call mfo_arnp
            ! close the file; only serial
            call byte_close(ierr)
         endif
      endif

      call  mntr_check_abort(arna_id,ierr,
     $       'arn_write_par: Error writing par file.')

      ! put output variables back
      WDSIZO = lwdsizo
      IFREGUO = lifreguo
      IFMPIIO = lifmpiio
      FID0 = lfid0

      return
      end
!=======================================================================
!> @brief Write procesor independent variables
!! @ingroup arn_arp
      subroutine mfo_arnp
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! local variables
      character*16 hdr
      integer ahdsize
      parameter (ahdsize=16)

      integer il, itmp(33), ierr

      real*4 test_pattern

      real*4 rtmp4(6), workla4(2*wldima)
      real*8 rtmp8(3), workla8(wldima)
      equivalence (rtmp4,rtmp8)
      equivalence (workla4,workla8)
!-----------------------------------------------------------------------
      ! write idoarp and character varialbes
      call blank(hdr,ahdsize)

      write(hdr,1) idoarp,bmatarp,whicharp,tst_mode! 14
 1    format('#arp',1x,i2,1x,a1,1x,a2,1x,i1)

      call byte_write(hdr,ahdsize/4,ierr)

      ! write test pattern for byte swap
      test_pattern = 6.54321

      call byte_write(test_pattern,1,ierr)

      ! collect and write integer varialbes
      itmp(1) = arna_ns
      itmp(2) = arna_negv
      itmp(3) = arna_nkrl
      itmp(4) = nwlarp
      itmp(5) = infarp
      itmp(6) = nparp
      itmp(7) = ncarp
      itmp(8) = tst_step
      do il=1,11
         itmp(8+il) = iparp(il)
      enddo
      do il=1,14
          itmp(19+il) = ipntarp(il)
      enddo

      call byte_write(itmp,33,ierr)

      ! collect and write real variables
      rtmp8(1) = tst_tol
      rtmp8(2) = RNMARP
      rtmp8(3) = DT

      call byte_write(rtmp4,6,ierr)

      ! write workla
      call copy(workla8,workla,nwlarp)

      call byte_write(workla4,2*nwlarp,ierr)

      return
      end
!=======================================================================
!> @brief Read procesor independent data
!! @ingroup arn_arp
!! @param[in]   prefix    prefix
      subroutine arn_read_par(prefix)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! argument list
      character*3 prefix

      ! local variables
      character*132 fname
      character*6  str
      integer lwdsizo, lfid0, ierr, il
      logical lifreguo, lifmpiio
!-----------------------------------------------------------------------
      call nekgsync()
      call io_init()

      !copy and set output parameters
      lwdsizo= WDSIZO
      WDSIZO = 8

      lifreguo= IFREGUO
      IFREGUO = .false.

      ! this is done by master node only, so serial reading
      lifmpiio = IFMPIIO
      IFMPIIO = .false.
      lfid0 = FID0
      FID0 = 0

      ierr = 0
      if (NID.eq.0) then
         ! create file name
         call io_mfo_fname(fname,SESSION,prefix,ierr)
         if (ierr.eq.0) then
            write(str,'(i5.5)') arna_fnum
            fname = trim(fname)//trim(str)
            ! open file
            call io_mbyte_open(fname, ierr)
         endif

         if (ierr.eq.0) then
            ! read parameters
            call mfi_arnp
            ! close the file
            call byte_close(ierr)
         endif
      endif

      call  mntr_check_abort(arna_id,ierr,
     $       'arn_read_par: Error opening par file.')

      ierr = 0
      if (NID.eq.0) then
         ! check and copy parameters
         ! is it correct restart
         if (idoarp0.ne.-2) then
            call mntr_error(arna_id,
     $           'arn_read_par, wrong idoarp0')
            call mntr_logi(arna_id,lp_err,'idoarp0 = ', idoarp0)
            ierr=1
         endif

         ! is it the same ARPACK mode
         if (bmatarp0.ne.bmatarp) then
            call mntr_error(arna_id,
     $           'arn_read_par, different ARPACK modes')
            call mntr_logi(arna_id,lp_err,'bmatarp0 = ', bmatarp0)
            call mntr_logi(arna_id,lp_err,'bmatarp  = ', bmatarp)
            ierr=1
         endif

         ! do we look for the same eigenvectors
         if (whicharp0.ne.whicharp) then
            call mntr_error(arna_id,
     $           'arn_read_par, different mode selsction')
            call mntr_logi(arna_id,lp_err,'whicharp0 = ', whicharp0)
            call mntr_logi(arna_id,lp_err,'whicharp  = ', whicharp)
            ierr=1
         endif

         ! is it the same integration mode
         if (tst_mode0.ne.tst_mode) then
            call mntr_error(arna_id,
     $           'arn_read_par, wrong simulation mode')
            call mntr_logi(arna_id,lp_err,'tst_mode0 = ', tst_mode0)
            call mntr_logi(arna_id,lp_err,'tst_mode  = ', tst_mode)
            ierr=1
         endif

         ! this should be removed later as it does not allow to change processor number
         ! is the length of the vector the same
         if (arna_ns0.ne.arna_ns) then
            call mntr_error(arna_id,
     $           'arn_read_par, different vector length (IFHEAT?)')
            call mntr_logi(arna_id,lp_err,'arna_ns0 = ', arna_ns0)
            call mntr_logi(arna_id,lp_err,'arna_ns  = ', arna_ns)
            ierr=1
         endif

         ! what is the size of krylov space
         ! would it be possible to change this?; related nparp, nwlarp,
         ! ipntarp
         if (arna_nkrl0.ne.arna_nkrl) then
            call mntr_error(arna_id,
     $           'arn_read_par, different Krylov space size')
            call mntr_logi(arna_id,lp_err,'arna_nkrl0 = ', arna_nkrl0)
            call mntr_logi(arna_id,lp_err,'arna_nkrl  = ', arna_nkrl)
            ierr=1
         endif

         if (nwlarp0.ne.nwlarp) then
            call mntr_error(arna_id,
     $           'arn_read_par, different size of work array')
            call mntr_logi(arna_id,lp_err,'nwlarp0 = ', nwlarp0)
            call mntr_logi(arna_id,lp_err,'nwlarp  = ', nwlarp)
            ierr=1
         endif

         ! stopping criterion
         if (tst_tol0.ne.tst_tol) then
            call mntr_warn(arna_id,
     $       'arn_read_par, different stopping criterion')
            call mntr_logi(arna_id,lp_err,'tst_tol0 = ', tst_tol0)
            call mntr_logi(arna_id,lp_err,'tst_tol  = ', tst_tol)
         endif

         ! number of eigenvalues
         if (arna_negv0.ne.arna_negv) then
            call mntr_warn(arna_id,
     $       'arn_read_par, different number of eigenvalues')
            call mntr_logi(arna_id,lp_err,'arna_negv0 = ', arna_negv0)
            call mntr_logi(arna_id,lp_err,'arna_negv  = ', arna_negv)
         endif

         ! stepper phase length
         if (dtarp0.ne.DT) then
            call mntr_warn(arna_id,
     $       'arn_read_par, different time step')
            call mntr_logi(arna_id,lp_err,'dtarp0 = ', dtarp0)
            call mntr_logi(arna_id,lp_err,'dt     = ', DT)
         endif

         if (tst_step0.ne.tst_step) then
            call mntr_warn(arna_id,
     $       'arn_read_par, different number of steps instepper phase')
            call mntr_logi(arna_id,lp_err,'tst_step0 = ', tst_step0)
            call mntr_logi(arna_id,lp_err,'tst_step  = ', tst_step)
         endif

         ! check IPARP
         if (iparp0(1).ne.iparp(1)) then
            call mntr_error(arna_id,
     $           'arn_read_par, different shift in ARPACK')
            call mntr_logi(arna_id,lp_err,'iparp0(1) = ', iparp0(1))
            call mntr_logi(arna_id,lp_err,'iparp(1)  = ', iparp(1))
            ierr=1
         endif

         if (iparp0(3).ne.iparp(3)) then
            call mntr_warn(arna_id,
     $           'arn_read_par, different cycle number')
            call mntr_logi(arna_id,lp_err,'iparp0(3) = ', iparp0(3))
            call mntr_logi(arna_id,lp_err,'iparp(3)  = ', iparp(3))
         endif

         if (IPARP0(7).ne.IPARP(7)) then
            call mntr_error(arna_id,
     $           'arn_read_par, different ARPACK modes')
            call mntr_logi(arna_id,lp_err,'iparp0(7) = ', iparp0(7))
            call mntr_logi(arna_id,lp_err,'iparp(7)  = ', iparp(7))
            ierr=1
         endif

         ! copy rest of parameters
         nparp = nparp0
         ncarp = ncarp0
         infarp= infarp0
         rnmarp= rnmarp0
         do il=4,11
            iparp(il) = iparp0(il)
         enddo
         iparp(2) = iparp0(2)
         do il=1,14
            ipntarp(il) = ipntarp0(il)
         enddo
      endif                     ! NID

      call  mntr_check_abort(arna_id,ierr,
     $       'arn_read_par: Error reading par file.')

      idoarp = -2
      ! broadcast
      call bcast(nparp,ISIZE)
      call bcast(ncarp,ISIZE)
      call bcast(infarp,ISIZE)
      call bcast(iparp,11*ISIZE)
      call bcast(ipntarp,14*ISIZE)
      call bcast(rnmarp,WDSIZE)

      call bcast(workla,nwlarp*WDSIZE)

      ! put output variables back
      WDSIZO = lwdsizo
      IFREGUO = lifreguo

      return
      end
!=======================================================================
!> @brief Read procesor independent variables
!! @ingroup arn_arp
      subroutine mfi_arnp
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSTEP'
      include 'FRAMELP'
      include 'TSTEPPERD'
      INCLUDE 'ARN_ARPD'

      ! local variables
      character*16 hdr
      character*4 dummy
      integer ahdsize
      parameter (ahdsize=16)

      integer ibsw_out, il, itmp(33), ierr

      real*4 test_pattern

      real*4 rtmp4(6), workla4(2*WLDIMA)
      real*8 rtmp8(3), workla8(WLDIMA)
      equivalence (rtmp4,rtmp8)
      equivalence (workla4,workla8)

      logical if_byte_swap_test, if_byte_sw_loc

      ! functions
      integer indx2
!-----------------------------------------------------------------------
      ! read idoarp and character varialbes
      call blank(hdr,ahdsize)

      call byte_read(hdr,ahdsize/4,ierr)

      if (indx2(hdr,132,'#arp',4).eq.1) then
         read(hdr,*) dummy,idoarp0,bmatarp0,whicharp0,tst_mode0! 14
      else
         call  mntr_abort(arna_id,
     $       'mfi_arnp; Error reading header')
      endif

      ! read test pattern for byte swap
      call byte_read(test_pattern,1,ierr)
      ! determine endianess
      if_byte_sw_loc = if_byte_swap_test(test_pattern,ierr)

      ! read integer varialbes
      call byte_read(itmp,33,ierr)
      if (if_byte_sw) call byte_reverse(itmp,33,ierr)

      arna_ns0 = itmp(1)
      arna_negv0  = itmp(2)
      arna_nkrl0  = itmp(3)
      nwlarp0 = itmp(4)
      infarp0 = itmp(5)
      nparp0  = itmp(6)
      ncarp0  = itmp(7)
      tst_step0 = itmp(8)
      do il=1,11
         iparp0(il) = itmp(8+il)
      enddo
      do il=1,14
         ipntarp0(il) = itmp(19+il)
      enddo

      ! read real variables
      call byte_read(rtmp4,6,ierr)
      if (if_byte_sw) call byte_reverse(rtmp4,6,ierr)

      tst_tol0 = rtmp8(1)
      rnmarp0 = rtmp8(2)
      dtarp0  = rtmp8(3)

      ! read workla
      if (nwlarp0.le.wldima) then
         call byte_read(workla4,2*nwlarp0,ierr)
         if (if_byte_sw) call byte_reverse(workla4,2*nwlarp0,ierr)
         call copy(workla,workla8,nwlarp0)
      else
         call  mntr_abort(arna_id,
     $       'mfi_arnp; Wrong work array size nwlarp0.le.wldima')
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write procesor dependent data (long vectors)
!! @ingroup arn_arp
!! @param[in]   prefix    prefix
!! @remark This routine uses global scratch space SCRUZ
      subroutine mfo_arnv(prefix)  ! muti-file output
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      include 'RESTART'
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! argument list
      character*3 prefix

      ! local variables
      integer*8 offs0,offs,nbyte,stride,strideB,nxyzo8
      integer lwdsizo, il, ierr
      integer ioflds, nout
      real dnbyte, tio, tiostart
      character*132  fname
      character*6  str
      logical lifxyo, lifpo, lifvo, lifto, lifreguo, lifpso(LDIMT1)

      ! functions
      real dnekclock_sync, glsum

      ! scratch space
      real UR1(LXO*LXO*LXO*LELT), UR2(LXO*LXO*LXO*LELT),
     $     UR3(LXO*LXO*LXO*LELT)
      common /SCRUZ/  UR1, UR2, UR3
!-----------------------------------------------------------------------
      tiostart=dnekclock_sync()
      call io_init

      ! set array and elelemnt size
      nout = NELT
      NXO  = NX1
      NYO  = NY1
      NZO  = NZ1

      ! copy and set output parameters
      lwdsizo= WDSIZO
      WDSIZO = 8

      lifreguo= IFREGUO
      IFREGUO = .false.
      lifxyo= IFXYO
      IFXYO = .false.
      lifpo= IFPO
      IFPO = .false.
      lifvo= IFVO
      IFVO = .true.
      lifto= IFTO
      IFTO = .false.
      do il=1,LDIMT1
         lifpso(il)= IFPSO(il)
         IFPSO(il) = .false.
      enddo

      ! open files on i/o nodes
      ierr = 0
      if (nid.eq.pid0) then
         ! create file name
         call io_mfo_fname(fname,SESSION,prefix,ierr)
         if (ierr.eq.0) then
            write(str,'(i5.5)') mod(arna_fnum,2) + 1
            fname = trim(fname)//trim(str)
            call io_mbyte_open(fname,ierr)
         endif
      endif

      call mntr_check_abort(arna_id,ierr,'mfo_arnv; file not opened.')

      ! write a header and create element mapping
      call mfo_write_hdr

      ! set offset
      offs0 = iHeaderSize + 4 + isize*nelgt
      nxyzo8  = NXO*NYO*NZO
      strideB = nelB * nxyzo8*WDSIZO
      stride  = nelgt* nxyzo8*WDSIZO
      ioflds = 0

      ! dump all fields based on the t-mesh to avoid different
      ! topologies in the post-processor

      ! resid array
      call  mfo_singlev(ioflds,nout,offs0,stride,strideB,
     $     UR1,UR2,UR3,RESIDA)

      ! workd array
      do il=0,2
         call  mfo_singlev(ioflds,nout,offs0,stride,strideB,
     $     UR1,UR2,UR3,WORKDA(1+arna_ns*il))
      enddo

      ! krylov space
      do il=1,arna_nkrl
         call  mfo_singlev(ioflds,nout,offs0,stride,strideB,
     $     UR1,UR2,UR3,VBASEA(1,il))
      enddo

      dnbyte = 1.*ioflds*nout*WDSIZO*NXO*NYO*NZO

      ! put output variables back
      WDSIZO = lwdsizo

      IFREGUO = lifreguo
      IFXYO = lifxyo
      IFPO = lifpo
      IFVO = lifvo
      IFTO = lifto
      do il=1,LDIMT1
         IFPSO(il) = lifpso(il)
      enddo

      call io_mbyte_close(ierr)
      call mntr_check_abort(arna_id,ierr,'mfo_arnv; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024

      call mntr_log(arna_id,lp_prd,'Checkpoint written:')
      call mntr_logr(arna_id,lp_vrb,'file size (MB) = ',dnbyte)
      call mntr_logr(arna_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(arna_id,lp_vrb,'io-nodes = ',nfileo)

      return
      end
!=======================================================================
!> @brief Read procesor dependent data (long vectors)
!! @ingroup arn_arp
!! @param[in]   prefix    prefix
!! @remark This routine uses global scratch space SCRNS, SCRUZ
      subroutine mfi_arnv(prefix)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'INPUT'
      include 'TSTEP'
      include 'RESTART'
      include 'FRAMELP'
      include 'TSTEPPERD'
      INCLUDE 'ARN_ARPD'

      ! argument list
      character*3 prefix

      ! local variables
      character*132  fname

      character*6  str

      integer e, il, iofldsr, ierr
      integer*8 offs0,offs,nbyte,stride,strideB,nxyzr8
      real dnbyte, tio, tiostart

      ! functions
      real dnekclock_sync, glsum

      ! scratch space
      integer lwk
      parameter (lwk = 7*lx1*ly1*lz1*lelt)
      real wk(lwk)
      common /scrns/ wk

      real UR1(LX1,LY1,LZ1,LELT), UR2(LX1,LY1,LZ1,LELT),
     $     UR3 (LX1,LY1,LZ1,LELT)
      COMMON /scruz/ UR1, UR2, UR3
!-----------------------------------------------------------------------
      tiostart=dnekclock_sync()
      call io_init

      ! create file name
      ierr = 0
      if (nid.eq.pid0r) then         ! open files on i/o nodes
         call io_mfo_fname(fname,SESSION,prefix,ierr)
         if (ierr.eq.0) then
            write(str,'(i5.5)') arna_fnum
            fname = trim(fname)//trim(str)
         endif
      endif

      call mntr_check_abort(arna_id,ierr,'mfi_arnv; file not opened.')

      call mfi_prepare(fname)       ! determine reader nodes +
                                    ! read hdr + element mapping

      offs0   = iHeadersize + 4 + ISIZE*NELGR
      nxyzr8  = NXR*NYR*NZR
      strideB = nelBr* nxyzr8*WDSIZR
      stride  = nelgr* nxyzr8*WDSIZR

      ! read arrays
      iofldsr = 0

      ! resid array
      call mfi_singlev(iofldsr,offs0,stride,strideB,
     $     UR1,UR2,UR3,RESIDA(1))

      ! workd array
      do il=0,2
         call mfi_singlev(iofldsr,offs0,stride,strideB,
     $        UR1,UR2,UR3,WORKDA(1+arna_ns*il))
      enddo

      ! krylov space
      do il=1,arna_nkrl
         call mfi_singlev(iofldsr,offs0,stride,strideB,
     $        UR1,UR2,UR3,VBASEA(1,il))
      enddo

      ! close files
      call io_mbyte_close(ierr)
      call mntr_check_abort(arna_id,ierr,'mfi_arnv; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      if(nid.eq.pid0r) then
         dnbyte = 1.*iofldsr*nelr*wdsizr*nxr*nyr*nzr
      else
         dnbyte = 0.0
      endif

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024

      call mntr_log(arna_id,lp_prd,'Checkpoint read:')
      call mntr_logr(arna_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(arna_id,lp_vrb,'io-nodes = ',nfileo)

      return
      end subroutine
!=======================================================================
!> @brief Write single Krylov vector to the file
!! @ingroup arn_arp
!! @param[inout] ioflds         Vector counter
!! @param[in]    nout           local number of elements to write
!! @param[in]    offs0          global file offset (header +...)
!! @param[in]    stride         single vector length
!! @param[in]    strideB        space saved for processes with lower nid
!! @param[in]    ur1,ur2,ur3    output arrays
!! @param[in]    vect           Krylov vector
      subroutine mfo_singlev(ioflds,nout,offs0,stride,strideB,
     $     ur1,ur2,ur3,vect)
      implicit none

      include 'SIZE'
      include 'INPUT'           ! IF3D, IFHEAT
      include 'RESTART'
      include 'TSTEPPERD'
      INCLUDE 'ARN_ARPD'

      ! argument list
      integer ioflds,nout
      integer*8 offs0,stride,strideB
      real UR1(LXO*LXO*LXO*LELT), UR2(LXO*LXO*LXO*LELT),
     $     UR3(LXO*LXO*LXO*LELT)
      real VECT(arna_ls)

      ! local variables
      integer*8 offs
!-----------------------------------------------------------------------
      offs = offs0 + ioflds*stride + NDIM*strideB
      call byte_set_view(offs,ifh_mbyte)

      call copy(UR1,VECT(1),tst_nv)
      call copy(UR2,VECT(1+tst_nv),tst_nv)
      if (IF3D) call copy(UR3,VECT(1+2*tst_nv),tst_nv)

      call mfo_outv(UR1,UR2,UR3,nout,NXO,NYO,NZO)
      ioflds = ioflds + NDIM

      if (IFHEAT) then
         offs = offs0 + ioflds*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call copy(UR1,VECT(1+NDIM*tst_nv),tst_nt)

         call mfo_outs(UR1,nout,NXO,NYO,NZO)
         ioflds = ioflds + 1
      endif

      return
      end subroutine
!=======================================================================
!> @brief Read single Krylov vector from the file
!! @ingroup arn_arp
!! @param[inout] iofldr         Vector counter
!! @param[in]    offs0          global file offset (header +...)
!! @param[in]    stride         single vector length
!! @param[in]    strideB        space saved for processes with lower nid
!! @param[in]    ur1,ur2,ur3    input arrays
!! @param[out]   vect           Krylov vector
!! @remark This routine uses global scratch space SCRNS
      subroutine mfi_singlev(iofldr,offs0,stride,strideB,
     $     ur1,ur2,ur3,vect)
      implicit none

      include 'SIZE'
      include 'INPUT'           ! IF3D, IFHEAT
      include 'RESTART'
      include 'TSTEPPERD'
      INCLUDE 'ARN_ARPD'

      ! argument list
      integer iofldr
      integer*8 offs0,stride,strideB
      real UR1(LX1*LX1*LX1*LELT), UR2(LX1*LX1*LX1*LELT),
     $     UR3(LX1*LX1*LX1*LELT)
      real VECT(arna_ls)

      integer lwk
      parameter (lwk = 7*lx1*ly1*lz1*lelt)
      real wk(lwk)
      common /scrns/ wk

      ! local variables
      integer*8 offs
!-----------------------------------------------------------------------
      offs = offs0 + iofldr*stride + NDIM*strideB
      call byte_set_view(offs,ifh_mbyte)
      call mfi_getv(UR1,UR2,UR3,wk,lwk,.false.)

      call copy(VECT(1),UR1,tst_nv)
      call copy(VECT(1+tst_nv),UR2,tst_nv)
      if (IF3D) call copy(VECT(1+2*tst_nv),UR3,tst_nv)
      iofldr = iofldr + NDIM

      if (IFHEAT) then
         offs = offs0 + iofldr*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         call mfi_gets(UR1,wk,lwk,.false.)

         call copy(VECT(1+NDIM*tst_nv),UR1,tst_nt)
         iofldr = iofldr + 1
      endif

      return
      end subroutine
!=======================================================================
