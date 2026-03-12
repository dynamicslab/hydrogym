!> @file chkpt_mstp.f
!! @ingroup chkpoint_mstep
!! @brief Set of multi-file checkpoint routines for DNS, MHD and
!!    perturbation simulations
!=======================================================================
!> @brief Register multi step checkpointing module
!! @ingroup chkpoint_mstep
      subroutine chkpts_register()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'

      ! local variables
      integer lpmid
!-----------------------------------------------------------------------
      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,chpm_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(chpm_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,chpt_name)
      if (lpmid.le.0) then
        lpmid = 1
        call mntr_abort(lpmid,
     $   'Parent ['//trim(chpt_name)//'] module not registered')
      endif

      ! register module
      call mntr_mod_reg(chpm_id,lpmid,chpm_name,
     $         'Multi-file checkpointing')

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'CHP_TOT')
      call mntr_tmr_reg(chpm_tread_id,lpmid,chpm_id,
     $      'CHP_READ','Checkpointing reading time',.true.)

      call mntr_tmr_reg(chpm_twrite_id,lpmid,chpm_id,
     $      'CHP_WRITE','Checkpointing writing time',.true.)

      ! adjust step delay
      call mntr_set_step_delay(chpm_snmax)

      return
      end subroutine
!=======================================================================
!> @brief Initialise multi-file checkpoint routines
!! @ingroup chkpoint_mstep
!! @note This interface is defined in @ref chkpt_main
      subroutine chkpts_init
      implicit none

      include 'SIZE'            ! NID, NPERT
      include 'TSTEP'           ! ISTEP, NSTEPS
      include 'INPUT'           ! IFPERT, PARAM
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (chpm_ifinit) then
         call mntr_warn(chpm_id,
     $        'module ['//trim(chpm_name)//'] already initiaised.')
         return
      endif

      ! get number of snapshots in a set
      if (PARAM(27).lt.0) then
         chpm_nsnap = NBDINP
      else
         chpm_nsnap = chpm_snmax
      endif

      ! we support only one perturbation
      if (IFPERT) then
         if (NPERT.gt.1) call mntr_abort(chpm_id,
     $         'only single perturbation supported')
      endif

      chpm_ifinit = .true.

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup chkpoint_mstep
!! @return chkpts_is_initialised
      logical function chkpts_is_initialised()
      implicit none

      include 'SIZE'
      include 'CHKPTMSTPD'
!-----------------------------------------------------------------------
      chkpts_is_initialised = chpm_ifinit

      return
      end function
!=======================================================================
!> @brief Write full file restart set
!! @ingroup chkpoint_mstep
!! @note This interface is defined in @ref chkpt_main.
!! @note This is version of @ref full_restart_save routine.
      subroutine chkpts_write()
      implicit none

      include 'SIZE'            !
      include 'TSTEP'           ! ISTEP, NSTEPS
      include 'INPUT'           ! IFMVBD, IFREGUO
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'

      ! local variables
      integer il, ifile, fnum
      real ltim
      character*132 fname(CHKPTNFMAX)
      logical ifcoord
      logical ifreguol

      character*2 str
      character*200 lstring

      integer icalldl
      save    icalldl
      data    icalldl  /0/

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! no regular mesh
      ifreguol= IFREGUO
      IFREGUO = .false.

      ! do we write a snapshot
      ifile = 0
      if (chpt_stepc.gt.0.and.chpt_stepc.le.chpm_nsnap) then
      ! timing
         ltim = dnekclock()

         ! file number
         ifile = chpm_nsnap - chpt_stepc +1
         if (ifile.eq.1) call mntr_log(chpm_id,lp_inf,
     $                             'Writing checkpoint snapshot')

         ! initialise I/O data
         call io_init

         ! get set of file names in the snapshot
         call chkpt_set_name(fname, fnum, chpt_set_o, ifile)

         ! do we wtrite coordinates; we save coordinates in DNS files only
         if (IFMVBD) then  ! moving boundaries save in every file
            ifcoord = .true.
         elseif (ifile.eq.1) then
            ! perturbation mode with constant base flow - only 1 rsX written
            if (ifpert.and.(.not.ifbase)) then
               if (icalldl.eq.0.and.(.not.chpt_ifrst)) then
                  icalldl = 1
                  ifcoord = .true.
               else
                  call chcopy (fname(1),fname(fnum),132)
                  fnum = 1
                  ifcoord = .false.
               endif
            ! DNS, MHD, perturbation with changing base flow - every first rsX file in snapshot
            else
               ifcoord = .true.
            endif
         else
            if (ifpert.and.(.not.ifbase)) then
               call chcopy (fname(1),fname(fnum),132)
               fnum = 1
            endif
            ifcoord = .false.
         endif

         ! write down files
         call chkpt_restart_write(fname, fnum, ifcoord)

         ! update output set number
         ! we do it after the last file in the set was sucsesfully written
         if (ifile.eq.chpm_nsnap) then
            write(str,'(I2)') chpt_set_o+1
            lstring = 'Written checkpoint snapshot number: '//trim(str)
            call mntr_log(chpm_id,lp_prd,lstring)
         endif

         ! timing
         ltim = dnekclock() - ltim
         call mntr_tmr_add(chpm_twrite_id,1,ltim)
      endif

      ! put parameters back
      IFREGUO = ifreguol

      return
      end subroutine
!=======================================================================
!> @brief Read full file restart set.
!! @ingroup chkpoint_mstep
!! @note This interface is defined in @ref chkpt_main
!! @note This is version of @ref full_restart routine.
      subroutine chkpts_read()
      implicit none

      include 'SIZE'            !
      include 'TSTEP'           ! ISTEP, IF_FULL_PRES
      include 'INPUT'           ! IFREGUO, INITC
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'

      ! local variables
      integer ifile, fnum, fnuml, il
      real dtratio, epsl, ltim
      parameter (epsl = 0.0001)
      logical ifreguol
      character*132 fname(CHKPTNFMAX),fnamel(CHKPTNFMAX)
      character*200 lstring
      integer icalld
      save icalld
      data icalld /0/

      !functions
      real dnekclock
!-----------------------------------------------------------------------
      ! no regular mesh; important for file name generation
      ifreguol= IFREGUO
      IFREGUO = .false.

      ! this is multi step restart so check for timestep consistency is necessary
      ! this routine gets the information of pressure mesh as well
      if (chpt_ifrst.and.icalld.eq.0) then
         call chkpt_dt_get
         icalld = 1
      endif

      if (chpt_ifrst.and.(ISTEP.lt.chpm_nsnap)) then

         ! timing
         ltim = dnekclock()

         ifile = ISTEP+1  ! snapshot number

         ! initialise I/O data
         call io_init

         ! get set of file names in the snapshot
         call chkpt_set_name(fname, fnum, chpt_set_i, ifile)

         ! perturbation mode with constant base flow - only 1 rsX written
         if (ifpert.and.(.not.ifbase)) then
            if (ifile.eq.1) then
               il = 0
               call chkpt_set_name(fnamel, fnuml, il, ifile)
               call chcopy (fname(1),fnamel(1),132)
               fnum = 2
            else
               call chcopy (fname(1),fname(fnum),132)
               fnum = 1
            endif
         endif

         call chkpt_restart_read(fname, fnum)

         ! check time step consistency
         if(ifile.gt.1.and.chpm_dtstep(ifile).gt.0.0) then
            dtratio = abs(DT-chpm_dtstep(ifile))/chpm_dtstep(ifile)
            if (dtratio.gt.epsl) then
                write(lstring,*) 'Time step inconsistent, new=',
     $            DT,', old=',chpm_dtstep(ifile)
               call mntr_warn(chpm_id,lstring)
              ! possible place to exit if this should be trerated as error
            endif
         endif

         ! timing
         ltim = dnekclock() - ltim
         call mntr_tmr_add(chpm_tread_id,1,ltim)
      endif

      ! put parameters back
      IFREGUO = ifreguol
      IF_FULL_PRES=.false.

      return
      end subroutine
!=======================================================================
!> @brief Get old simulation time steps and pressure mesh marker.
!! @ingroup chkpoint_mstep
!! @todo Different files could have different chpm_if_pmesh, so it is
!!    not the best place to read it
      subroutine chkpt_dt_get
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'

      ! local variables
      integer ifile, ierr
      real timerl(chpm_snmax), p0thr
      character*132 fname, header
      character*3 prefix
      character*4 dummy
!-----------------------------------------------------------------------
      ! which set of files should be used
      if (ifpert.and.(.not.ifbase)) then
         prefix(1:2)='rp'
      else
         prefix(1:2)='rs'
      endif

      ! initialise I/O data
      call io_init

      ! collect simulation time from file headers
      do ifile=1,chpm_nsnap
         call chkpt_fname(fname, prefix, chpt_set_i, ifile, ierr)
         call mntr_check_abort(chpm_id,ierr,'dt get; file name error')

         ierr = 0
         if (NID.eq.pid00) then
            ! open file
            call addfid(fname,fid0)
            ! add ending character; required by C
            fname = trim(fname)//CHAR(0)
            call byte_open(fname,ierr)
            ! read header
            if (ierr.eq.0) then
               call blank     (header,iHeaderSize)
               call byte_read (header,iHeaderSize/4,ierr)
            endif
            ! close the file
            if (ierr.eq.0) call byte_close(ierr)
         endif
         call mntr_check_abort(chpm_id,ierr,
     $       'dt get; error reading header')

         call bcast(header,iHeaderSize)
         ierr = 0
         if (index(header,'#std').eq.1) then
            read(header,*,iostat=ierr) dummy
     $         ,  wdsizr,nxr,nyr,nzr,nelr,nelgr,timer,istpr
     $         ,  ifiler,nfiler
     $         ,  rdcode      ! 74+20=94
     $         ,  p0thr, chpm_if_pmesh
         else
            ierr = 1
         endif
         call mntr_check_abort(chpm_id,ierr,
     $       'dt get; error extracting timer')

         timerl(ifile) = timer
      enddo

      ! get dt
      do ifile=2,chpm_nsnap
         chpm_dtstep(ifile) = timerl(ifile) - timerl(ifile-1)
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Generate set of restart file names in snapshot
!! @ingroup chkpoint_mstep
!! @param[out] fname  restart file names
!! @param[out] fnum   number of files in snapshot
!! @param[in]  nset   set number
!! @param[in]  ifile  snupshot numer
      subroutine chkpt_set_name(fname, fnum, nset, ifile)
      implicit none

      include 'SIZE'            ! NIO
      include 'INPUT'           ! IFMHD, IFPERT, IFBASE
      include 'FRAMELP'
      include 'CHKPTMSTPD'

      ! argument list
      character*132 fname(CHKPTNFMAX)
      integer fnum, nset, ifile

      ! local variables
      integer ifilel, ierr

      character*132 bname
      character*3 prefix
!-----------------------------------------------------------------------
      ! fill fname array with 'rsX' (DNS), 'rpX' (pert.) and 'rbX' (MHD) file names
      if (IFMHD) then
         ! file number
         fnum = 2

         ! prefix and name for fluid (DNS)
         prefix(1:2)='rs'
         call chkpt_fname(fname(1), prefix, nset, ifile, ierr)
         call mntr_check_abort(chpm_id,ierr,
     $        'chkpt_set_name; DNS file name error')

         ! prefix and name for magnetic field (MHD)
         prefix(1:2)='rb'
         call chkpt_fname(fname(2), prefix, nset, ifile, ierr)
         call mntr_check_abort(chpm_id,ierr,
     $        'chkpt_set_name; MHD file name error')

      elseif (IFPERT) then
         ! file number
         ! I assume only single perturbation
         fnum = 2

         ! prefix and name for base flow (DNS)
         prefix(1:2)='rs'
         if (IFBASE) then
            ifilel = ifile
         else
            ifilel =1
         endif
         call chkpt_fname(fname(1), prefix, nset, ifilel, ierr)
         call mntr_check_abort(chpm_id,ierr,
     $        'chkpt_set_name; base flow file name error')

         ! prefix and name for perturbation
         prefix(1:2)='rp'
         call chkpt_fname(fname(2), prefix, nset, ifile, ierr)
         call mntr_check_abort(chpm_id,ierr,
     $        'chkpt_set_name; perturbation file name error')

      else                ! DNS
         fnum = 1

         ! create prefix and name for DNS
         prefix(1:2)='rs'
         call chkpt_fname(fname(1), prefix, nset, ifile, ierr)
         call mntr_check_abort(chpm_id,ierr,
     $        'chkpt_set_name; DNS file name error')

      endif

      return
      end subroutine
!=======================================================================
!> @brief Generate single restart file name
!! @ingroup chkpoint_mstep
!! @param[out] fname  restart file name
!! @param[in]  prefix prefix
!! @param[in]  nset   set number
!! @param[in]  ifile  snupshot numer
!! @param[out] ierr   error mark
      subroutine chkpt_fname(fname, prefix, nset, ifile, ierr)
      implicit none

      include 'SIZE'            ! NIO
      include 'INPUT'           ! SESSION
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'

      ! argument list
      character*132 fname
      character*3 prefix
      integer nset, ifile, ierr

      ! local variables
      character*132 bname    ! base name
      character*132 fnamel   ! local file name
      character*3 prefixl    ! local prefix
      integer itmp

      character*6  str

      character*(*) kst
      parameter(kst='0123456789abcdefx')
!-----------------------------------------------------------------------
      ! create prefix and name for DNS
      ierr = 0
      prefixl(1:2) = prefix(1:2)
      itmp=min(17,chpt_nset*chpm_nsnap) + 1
      prefixl(3:3)=kst(itmp:itmp)

      ! get base name (SESSION)
      bname = trim(adjustl(SESSION))

      call io_mfo_fname(fnamel,bname,prefixl,ierr)
      if (ierr.ne.0) then
         call mntr_error(chpm_id,'chkpt_fname; file name error')
         return
      endif

      write(str,'(i5.5)') chpm_nsnap*nset+ifile
      fname = trim(fnamel)//trim(str)

      return
      end subroutine
!=======================================================================
!> @brief Write checkpoint snapshot.
!! @ingroup chkpoint_mstep
!! @param[out] fname   restart file name
!! @param[in]  fnum    number of files in snapshot
!! @param[in]  ifcoord do we save coordinates
      subroutine chkpt_restart_write(fname, fnum, ifcoord)
      implicit none

      include 'SIZE'
      include 'RESTART'
      include 'TSTEP'
      include 'INPUT'
      include 'CHKPTMSTPD'

      ! argument list
      character*132 fname(CHKPTNFMAX)
      integer fnum
      logical ifcoord

      ! local variables
      integer lwdsizo
      integer ipert, il
      ! which set of variables do we write: DNS (1), MHD (2) or perturbation (3)
      integer chktype

      logical lif_full_pres, lifxyo, lifpo, lifvo, lifto,
     $        lifpsco(LDIMT1)
!-----------------------------------------------------------------------
      ! adjust I/O parameters
      lwdsizo = WDSIZO
      WDSIZO  = 8
      lif_full_pres = IF_FULL_PRES
      IF_FULL_PRES = .true.
      lifxyo = IFXYO
      lifpo= IFPO
      IFPO = .TRUE.
      lifvo= IFVO
      IFVO = .TRUE.
      lifto= IFTO
      IFTO = IFHEAT
      do il=1,NPSCAL
         lifpsco(il)= IFPSCO(il)
         IFPSCO(il) = .TRUE.
      enddo

      if (IFMHD) then
         ! DNS first
         IFXYO = ifcoord
         chktype = 1
         call chkpt_mfo(fname(1),chktype,ipert)

         ! MHD
         IFXYO = .FALSE.
         chktype = 2
         call chkpt_mfo(fname(2),chktype,ipert)

      elseif (IFPERT) then
         ! DNS first
         if (fnum.eq.2) then
            IFXYO = ifcoord
            chktype = 1
            call chkpt_mfo(fname(1),chktype,ipert)
         endif

         ! perturbation
         IFXYO = .FALSE.
         chktype = 3
         ipert = 1
         call chkpt_mfo(fname(fnum),chktype,ipert)
      else ! DNS
         ! write only one set of files
         if (fnum.ne.1) call mntr_abort(chpm_id,
     $        'chkpt_restart_save; too meny files for DNS')

         IFXYO = ifcoord
         chktype = 1
         call chkpt_mfo(fname(1),chktype,ipert)
      endif

      ! restore I/O parameters
      WDSIZO = lwdsizo
      IF_FULL_PRES = lif_full_pres
      IFXYO = lifxyo
      IFPO = lifpo
      IFVO = lifvo
      IFTO = lifto
      do il=1,NPSCAL
         IFPSCO(il) = lifpsco(il)
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Read checkpoint snapshot.
!! @ingroup chkpoint_mstep
!! @param[out] fname   restart file name
!! @param[in]  fnum    number of files in snapshot
      subroutine chkpt_restart_read(fname, fnum)
      implicit none

      include 'SIZE'
      include 'RESTART'
      include 'TSTEP'
      include 'INPUT'
      include 'CHKPTMSTPD'

      ! argument list
      character*132 fname(CHKPTNFMAX)
      integer fnum

      ! local variables
      integer ndumps, ipert, il
      ! which set of variables do we write: DNS (1), MHD (2) or perturbation (3)
      integer chktype
      character*132 fnamel
!-----------------------------------------------------------------------
      if (IFMHD) then
         ! DNS first
         chktype = 1
         call sioflag(ndumps,fnamel,fname(1))
         call chkpt_mfi(fnamel,chktype,ipert)

         ! MHD
         chktype = 2
         call sioflag(ndumps,fnamel,fname(2))
         call chkpt_mfi(fnamel,chktype,ipert)

      elseif (IFPERT) then
         ! DNS first
         if (fnum.eq.2) then
            chktype = 1
            call sioflag(ndumps,fnamel,fname(1))
            call chkpt_mfi(fnamel,chktype,ipert)
         endif

         ! perturbation
         chktype = 3
         ipert = 1
         call sioflag(ndumps,fnamel,fname(fnum))
         call chkpt_mfi(fnamel,chktype,ipert)
      else ! DNS
         ! read only one set of files
         if (fnum.ne.1) call mntr_abort(chpm_id,
     $        'chkpt_restart_read; too meny files for DNS')

         chktype = 1
         call sioflag(ndumps,fnamel,fname(1))
         call chkpt_mfi(fnamel,chktype,ipert)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write field to the file
!! @details This routine is based on @ref mfo_outfld but does not assume
!!    any file numbering. It is optimised for chekpoint writing.
!! @ingroup chkpoint_mstep
!! @param[in]   fname      file name
!! @param[in]   chktype    data type to write (DNS, MHD, perturbation)
!! @param[in]   ipert      index of perturbation field
!! @note Only one set of data (DNS, MHD or perturbation) can be saved in
!!    single file
!! @remark This routine uses global scratch space \a SCRCG.
      subroutine chkpt_mfo(fname,chktype,ipert)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSTEP'
      include 'GEOM'
      include 'SOLN'
      include 'FRAMELP'
      include 'CHKPTMSTPD'

      ! argumnt list
      character*132 fname
      integer chktype, ipert

      ! local variables
      integer ierr, il, itmp
      integer ioflds
      integer*8 offs,nbyte
      real dnbyte, tiostart, tio

      ! functions
      real dnekclock_sync, glsum

      real pm1(lx1,ly1,lz1,lelv)
      common /SCRCG/ pm1
!-----------------------------------------------------------------------
      ! simple timing
      tiostart=dnekclock_sync()

      ! set elelemnt size
      NXO  = NX1
      NYO  = NY1
      NZO  = NZ1

      ! open file
      call io_mbyte_open(fname,ierr)
      call mntr_check_abort(chpm_id,ierr,'chkpt_mfo; file not opened.')

      ! write a header and create element mapping
      call mfo_write_hdr

      ! set header offset
      offs = iHeaderSize + 4 + isize*nelgt
      ioflds = 0

      ! write fields
      ! coordinates
      if (ifxyo) then
         call io_mfov(offs,xm1,ym1,zm1,nx1,ny1,nz1,nelt,nelgt,ndim)
         ioflds = ioflds + ndim
      endif

      ! velocity, magnetic field, perturbation velocity
      if (ifvo) then
         if (chktype.eq.1) then
            call io_mfov(offs,vx,vy,vz,nx1,ny1,nz1,nelt,nelgt,ndim)
         elseif(chktype.eq.2) then
            call io_mfov(offs,bx,by,bz,nx1,ny1,nz1,nelt,nelgt,ndim)
         elseif(chktype.eq.3) then
            call io_mfov(offs,vxp(1,ipert),vyp(1,ipert),
     $                       vzp(1,ipert),nx1,ny1,nz1,nelt,nelgt,ndim)
         endif
         ioflds = ioflds + ndim
      endif

      ! pressure
      if (ifpo) then
         if (chktype.eq.1) then
            ! copy array if necessary
            if (ifsplit) then
               itmp = nx1*ny1*nz1*lelv
               call io_mfos(offs,pr,nx2,ny2,nz2,nelt,nelgt,ndim)
            else
               itmp = nx1*ny1*nz1*lelv
               call rzero(pm1,itmp)
               itmp = nx2*ny2*nz2
               do il=1,nelv
                  call copy(pm1(1,1,1,il),pr(1,1,1,il),itmp)
               enddo
               call io_mfos(offs,pm1,nx1,ny1,nz1,nelt,nelgt,ndim)
            endif
         elseif(chktype.eq.2) then
            ! copy array
            itmp = nx1*ny1*nz1*lelv
            call rzero(pm1,itmp)
            itmp = nx2*ny2*nz2
            do il=1,nelv
               call copy(pm1(1,1,1,il),pm(1,1,1,il),itmp)
            enddo
            call io_mfos(offs,pm1,nx1,ny1,nz1,nelt,nelgt,ndim)
         elseif(chktype.eq.3) then
            ! copy array
            itmp = nx1*ny1*nz1*lelv
            call rzero(pm1,itmp)
            itmp = nx2*ny2*nz2
            do il=1,nelv
               call copy(pm1(1,1,1,il),prp(1+itmp*(il-1),ipert),itmp)
            enddo
            call io_mfos(offs,pm1,nx1,ny1,nz1,nelt,nelgt,ndim)
         endif
         ioflds = ioflds + 1
      endif

      if (chktype.ne.2) then
         if (ifto) then
            if (chktype.eq.1) then
               call io_mfos(offs,t,nx1,ny1,nz1,nelt,nelgt,ndim)
            elseif(chktype.eq.3) then
               call io_mfos(offs,tp(1,1,ipert),
     $                          nx1,ny1,nz1,nelt,nelgt,ndim)
            endif
            ioflds = ioflds + 1
         endif

         do il=1,ldimt-1
            if (ifpsco(il)) then
               if (chktype.eq.1) then
                  call io_mfos(offs,t(1,1,1,1,il+1),
     $                 nx1,ny1,nz1,nelt,nelgt,ndim)
               elseif(chktype.eq.3) then
                  call io_mfos(offs,tp(1,il+1,ipert),
     $                 nx1,ny1,nz1,nelt,nelgt,ndim)
               endif
               ioflds = ioflds + 1
            endif
         enddo
      endif
      dnbyte = 1.*ioflds*nelt*wdsizo*nx1*ny1*nz1

      ! possible place for metadata

      ! close file
      call io_mbyte_close(ierr)
      call mntr_check_abort(chpm_id,ierr,'chkpt_mfo; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024

      call mntr_log(chpm_id,lp_prd,'Checkpoint written:')
      call mntr_logr(chpm_id,lp_vrb,'file size (MB) = ',dnbyte)
      call mntr_logr(chpm_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(chpm_id,lp_vrb,'io-nodes = ',nfileo)

      return
      end subroutine
!=======================================================================
!> @brief Read field to the file
!! @details This routine is based on @ref mfi but supports perturbation
!!    as well. It is optimised for chekpoint reading.
!! @ingroup chkpoint_mstep
!! @param[in]   fname      file name
!! @param[in]   chktype    data type to read (DNS, MHD, preturbation)
!! @param[in]   ipert      index of perturbation field
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine chkpt_mfi(fname,chktype,ipert)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSTEP'
      include 'GEOM'
      include 'SOLN'
      include 'FRAMELP'
      include 'CHKPTMSTPD'

      ! argumnt list
      character*132 fname
      integer chktype, ipert

      ! local variables
      integer ierr, il, jl, kl
      integer itmp1, itmp2, itmp3
      integer ioflds
      integer*8 offs0,offs,nbyte
      real dnbyte, tiostart, tio
      logical ifskip

      !     scratch arrays
      integer lwkv
      parameter (lwkv = lx1*ly1*lz1*lelt)
      real wkv1(lwkv),wkv2(lwkv),wkv3(lwkv)
      common /scruz/ wkv1,wkv2,wkv3

      ! functions
      real dnekclock_sync, glsum
!-----------------------------------------------------------------------
      ! simple timing
      tiostart=dnekclock_sync()

      ! open file and read header; some operations related to header are
      ! performed in chkpts_read; it is not the optimal way, but I don't want
      ! to modify mfi_prepare
      call mfi_prepare(fname)

      ! set header offset
      offs = iHeaderSize + 4 + isize*nelgr
      ioflds = 0

      ! read fields
      ! coordinates
      if (ifgetxr) then
         ifskip = .not.ifgetx
         ! skip coordinates if you need to interpolate them;
         ! I assume current coordinates are more exact than the interpolated one
         if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
            call io_mfiv(offs,xm1,ym1,zm1,lx1,ly1,lz1,lelt,ifskip)
         else
            ifskip=.TRUE.
            call io_mfiv(offs,wkv1,wkv2,wkv3,nxr,nyr,nzr,lelt,ifskip)
         endif
         ioflds = ioflds + ldim
      endif

      ! velocity, magnetic field, perturbation velocity
      if (ifgetur) then
         ifskip = .not.ifgetu
         if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
            ! unchanged resolution
            ! read field directly to the variables
            if (chktype.eq.1) then
               call io_mfiv(offs,vx,vy,vz,lx1,ly1,lz1,lelv,ifskip)
            elseif(chktype.eq.2) then
               call io_mfiv(offs,bx,by,bz,lx1,ly1,lz1,lelv,ifskip)
            elseif(chktype.eq.3) then
               call io_mfiv(offs,vxp(1,ipert),vyp(1,ipert),vzp(1,ipert)
     $                   ,lx1,ly1,lz1,lelv,ifskip)
            endif
         else
            ! modified resolution
            ! read field to tmp array
            call io_mfiv(offs,wkv1,wkv2,wkv3,nxr,nyr,nzr,lelt,ifskip)

            ! interpolate
            if (ifgetu) then
               if (chktype.eq.1) then
                  call chkpt_map_gll(vx,wkv1,nxr,nzr,nelv)
                  call chkpt_map_gll(vy,wkv2,nxr,nzr,nelv)
                  if (if3d) call chkpt_map_gll(vz,wkv3,nxr,nzr,nelv)
               elseif(chktype.eq.2) then
                  call chkpt_map_gll(bx,wkv1,nxr,nzr,nelv)
                  call chkpt_map_gll(by,wkv2,nxr,nzr,nelv)
                  if (if3d) call chkpt_map_gll(bz,wkv3,nxr,nzr,nelv)
               elseif(chktype.eq.3) then
                  call chkpt_map_gll(vxp(1,ipert),wkv1,nxr,nzr,nelv)
                  call chkpt_map_gll(vyp(1,ipert),wkv2,nxr,nzr,nelv)
                  if (if3d) call chkpt_map_gll(vzp(1,ipert),wkv3,
     $                        nxr,nzr,nelv)
               endif
            endif
         endif
         ioflds = ioflds + ndim
      endif

      ! pressure
      if (ifgetpr) then
         ifskip = .not.ifgetp

         if (chpm_if_pmesh) then
            ! file contains pressure on nxr-2 (GL) mesh
            ! read field to tmp array
            call io_mfis(offs,wkv1,nxr,nyr,nzr,lelt,ifskip)

            if (ifgetp) then
               if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
                  ! unchanged resolution
                  if (ifsplit) then
                     if (chktype.eq.1) then
                        !interpolate GL to GLL
                        itmp1 = nx1*ny1*nz1
                        jl = 1
                        do il=1,nelv
                           call map21t (pr(1,1,1,il),wkv1(jl),il)
                           jl = jl +itmp1
                        enddo
                     endif
                  else
                     ! remove zeros
                     itmp1 = nx1*ny1*nz1
                     itmp2 = nx2*ny2*nz2
                     jl = 1
                     if(chktype.eq.1) then
                        do il=1,nelv
                           call copy(pr(1,1,1,il),wkv1(jl),itmp2)
                           jl = jl + itmp1
                        enddo
                     elseif(chktype.eq.2) then
                        do il=1,nelv
                           call copy(pm(1,1,1,il),wkv1(jl),itmp2)
                           jl = jl + itmp1
                        enddo
                     elseif(chktype.eq.3) then
                        kl = 1
                        do il=1,nelv
                           call copy(prp(kl,ipert),wkv1(jl),itmp2)
                           jl = jl + itmp1
                           kl = kl + itmp2
                        enddo
                     endif
                  endif
               else
                  ! modified resolution
                  ! remove zeros
                  itmp1 = nxr*nyr*nzr
                  itmp3 = max(nzr-2,1)
                  itmp2 = (nxr-2)*(nyr-2)*itmp3
                  jl = 1
                  kl = 1
                  do il=1,nelv
                     call copy(wkv2(kl),wkv1(jl),itmp2)
                     jl = jl + itmp1
                     kl = kl + itmp2
                  enddo

                  if (ifsplit) then
                     if (chktype.eq.1) then
                        ! interpolate on GL mesh
                        call chkpt_map_gl(wkv1,wkv2,nxr-2,itmp3,nelv)

                        !interpolate GL to GLL
                        itmp2 = nx2*ny2*nz2
                        jl = 1
                        do il=1,nelv
                           call map21t (pr(1,1,1,il),wkv1(jl),il)
                           jl = jl +itmp2
                        enddo
                     endif
                  else
                     ! interpolate on GL mesh
                     if (chktype.eq.1) then
                        call chkpt_map_gl(pr,wkv2,nxr-2,itmp3,nelv)
                     elseif(chktype.eq.2) then
                        call chkpt_map_gl(pm,wkv2,nxr-2,itmp3,nelv)
                     elseif(chktype.eq.3) then
                        call chkpt_map_gl(prp(1,ipert),wkv2,nxr-2,
     $                       itmp3,nelv)
                     endif
                  endif
               endif
            endif

         else
            ! file contains pressure on nxr (GLL) mesh
            if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
               ! unchanged resolution
               if (ifsplit) then
                  ! read field directly to the variables
                  if (chktype.eq.1) then
                     call io_mfis(offs,pr,lx1,ly1,lz1,lelv,ifskip)
                  endif
               else
                  ! read field to tmp array
                  call io_mfis(offs,wkv1,nxr,nyr,nzr,lelt,ifskip)

                  ! interpolate GLL to GL
                  if (ifgetp) then
                     itmp2 = nx1*ny1*nz1
                     itmp2 = nx2*ny2*nz2
                     jl = 1
                     if (chktype.eq.1) then
                        do il=1,nelv
                           call map12 (pr(1,1,1,il),wkv1(jl),il)
                           jl = jl +itmp1
                        enddo
                     elseif(chktype.eq.2) then
                        do il=1,nelv
                           call map12 (pm(1,1,1,il),wkv1(jl),il)
                           jl = jl +itmp1
                        enddo
                     elseif(chktype.eq.3) then
                        kl = 1
                        do il=1,nelv
                           call map12 (prp(kl,ipert),wkv1(jl),il)
                           jl = jl +itmp1
                           kl = kl +itmp2
                        enddo
                     endif
                  endif
               endif

            else
               ! modified resolution
               ! read field to tmp array
               call io_mfis(offs,wkv1,nxr,nyr,nzr,lelt,ifskip)

               if (ifgetp) then
                  if (ifsplit) then
                     ! interpolate on GLL
                     call chkpt_map_gll(pr,wkv1,nxr,nzr,nelv)
                  else
                     ! interpolate on GLL
                     call chkpt_map_gll(wkv2,wkv1,nxr,nzr,nelv)

                     ! interpolate GLL to GL
                     itmp2 = nx1*ny1*nz1
                     itmp2 = nx2*ny2*nz2
                     jl = 1
                     if (chktype.eq.1) then
                        do il=1,nelv
                           call map12 (pr(1,1,1,il),wkv2(jl),il)
                           jl = jl +itmp1
                        enddo
                     elseif(chktype.eq.2) then
                        do il=1,nelv
                           call map12 (pm(1,1,1,il),wkv2(jl),il)
                           jl = jl +itmp1
                        enddo
                     elseif(chktype.eq.3) then
                        kl = 1
                        do il=1,nelv
                           call map12 (prp(kl,ipert),wkv2(jl),il)
                           jl = jl +itmp1
                           kl = kl +itmp2
                        enddo
                     endif
                  endif
               endif
            endif
         endif

         ioflds = ioflds + 1
      endif

      if (chktype.ne.2) then
         if (ifgettr) then
            ifskip = .not.ifgett
            if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
               ! unchanged resolution
               ! read field directly to the variables
               if (chktype.eq.1) then
                  call io_mfis(offs,t,lx1,ly1,lz1,lelt,ifskip)
               elseif(chktype.eq.3) then
                  call io_mfis(offs,tp(1,1,ipert),lx1,ly1,lz1,lelt,
     $                 ifskip)
               endif
            else
               ! modified resolution
               ! read field to tmp array
               call io_mfis(offs,wkv1,nxr,nyr,nzr,lelt,ifskip)

               ! interpolate
               if (ifgett) then
                  if (chktype.eq.1) then
                     call chkpt_map_gll(t,wkv1,nxr,nzr,nelt)
                  elseif(chktype.eq.3) then
                     call chkpt_map_gll(tp(1,1,ipert),wkv1,nxr,nzr,nelt)
                  endif
               endif
            endif
            ioflds = ioflds + 1
         endif

         do il=1,ldimt-1
            if (ifgtpsr(il)) then
               ifskip = .not.ifgtps(il)
               if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
                  ! unchanged resolution
                  ! read field directly to the variables
                  if (chktype.eq.1) then
                     call io_mfis(offs,t(1,1,1,1,il+1),lx1,ly1,lz1,
     $                            lelt,ifskip)
                  elseif(chktype.eq.3) then
                     call io_mfis(offs,tp(1,il+1,ipert),lx1,ly1,lz1,
     $                            lelt,ifskip)
                  endif
               else
                  ! modified resolution
                  ! read field to tmp array
                  call io_mfis(offs,wkv1,nxr,nyr,nzr,lelt,ifskip)

                  ! interpolate
                  if (ifgtps(il)) then
                     if (chktype.eq.1) then
                        call chkpt_map_gll(t(1,1,1,1,il+1),wkv1,
     $                                    nxr,nzr,nelt)
                     elseif(chktype.eq.3) then
                        call chkpt_map_gll(tp(1,il+1,ipert),wkv1,
     $                                    nxr,nzr,nelt)
                     endif
                  endif
               endif
               ioflds = ioflds + 1
            endif
         enddo
      endif

      if (ifgtim) time = timer

      ! close file
      call io_mbyte_close(ierr)
      call mntr_check_abort(chpm_id,ierr,'chkpt_mfi; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      if(nid.eq.pid0r) then
         dnbyte = 1.*ioflds*nelr*wdsizr*nxr*nyr*nzr
      else
         dnbyte = 0.0
      endif

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024

      call mntr_log(chpm_id,lp_prd,'Checkpoint read:')
      call mntr_logr(chpm_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(chpm_id,lp_vrb,'io-nodes = ',nfileo)

      if (ifaxis) call chkpt_axis_interp_ic()

      return
      end subroutine
!=======================================================================
!> @brief Interpolate input on velocity mesh
!! @details This is version of @ref mapab with corrected array sizes.
!!    It iterpolates fields defined on GLL points. Like
!!    the orginal routine I assume NXR=NYR=NZR, or NXR=NYR, NZR=1
!! @ingroup chkpoint_mstep
!! @param[out]   xf           output field on velocity mesh
!! @param[in]    yf           input field on velocity mesh
!! @param[in]    nxr, nzr     array sizes
!! @param[in]    nel          element number
!! @remarks This routine uses global scratch space CTMP0, CTMPABM1
      subroutine chkpt_map_gll(xf,yf,nxr,nzr,nel)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'IXYZ'
      include 'WZ'

      ! argument lists
      integer nxr, nzr, nel
      real xf(lx1,ly1,lz1,nel), yf(nxr,nxr,nzr,nel)

      ! local variables
      integer nyzr, nxy2
      integer ie, iz, izoff

      ! moddificaion flag
      integer nold
      save    nold
      data    nold /0/

      ! work arrays
      integer lxr, lyr, lzr, lxyzr
      parameter (lxr=lx1+6)
      parameter (lyr=ly1+6)
      parameter (lzr=lz1+6)
      parameter (lxyzr=lxr*lyr*lzr)
      real txa(lxyzr),txb(lx1,ly1,lzr),zgmr(lxr),wgtr(lxr)
      common /ctmp0/  txa, txb, zgmr, wgtr

      ! interpolation arrays
      real ires(lxr*lxr)  ,itres(lxr*lxr)
      common /ctmpabm1/ ires, itres
!-----------------------------------------------------------------------
      nyzr = nxr*nzr
      nxy2 = lx1*ly1

      if (nxr.ne.nold) then
         nold=nxr
         call zwgll(zgmr,wgtr,nxr)
         call igllm(ires,itres,zgmr,zgm1,nxr,lx1,nxr,lx1)
      endif

      do ie=1,nel
         call mxm (ires,lx1,yf(1,1,1,ie),nxr,txa,nyzr)
         do iz=1,nzr
            izoff = 1 + (iz-1)*lx1*nxr
            call mxm (txa(izoff),lx1,itres,nxr,txb(1,1,iz),ly1)
         enddo
         if (if3d) then
            call mxm (txb,nxy2,itres,nzr,xf(1,1,1,ie),lz1)
         else
            call copy(xf(1,1,1,ie),txb,nxy2)
         endif
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Interpolate pressure input
!! @details This is version of @ref mapab modified to work with pressure
!!    mesh. It iterpolates fields defined on GL points. Like
!!    the orginal routine I assume NXR=NYR=NZR, or NXR=NYR, NZR=1
!! @ingroup chkpoint_mstep
!! @param[out]   xf           output field on pressure mesh
!! @param[in]    yf           input field on pressure mesh
!! @param[in]    nxr, nzr     array sizes
!! @param[in]    nel          element number
!! @remarks This routine uses global scratch space CTMP0, CTMPABM2
      subroutine chkpt_map_gl(xf,yf,nxr,nzr,nel)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'IXYZ'
      include 'WZ'

      ! argument lists
      integer nxr, nzr, nel
      real xf(lx2,ly2,lz2,nel), yf(nxr,nxr,nzr,nel)

      ! local variables
      integer nyzr, nxy2
      integer ie, iz, izoff

      ! moddificaion flag
      integer nold
      save    nold
      data    nold /0/

      ! work arrays
      integer lxr, lyr, lzr, lxyzr
      parameter (lxr=lx2+6)
      parameter (lyr=ly2+6)
      parameter (lzr=lz2+6)
      parameter (lxyzr=lxr*lyr*lzr)
      real txa(lxyzr),txb(lx2,ly2,lzr),zgmr(lxr),wgtr(lxr)
      common /ctmp0/  txa, txb, zgmr, wgtr

      ! interpolation arrays
      real ires(lxr,lxr)  ,itres(lxr,lxr)
      common /ctmpabm2/ ires, itres
!-----------------------------------------------------------------------
      nyzr = nxr*nzr
      nxy2 = lx2*ly2

      if (nxr.ne.nold) then
         nold=nxr
         call zwgl   (zgmr,wgtr,nxr)
         call iglm   (ires,itres,zgmr,zgm2,nxr,lx2,nxr,lx2)
      endif

      do ie=1,nel
         call mxm (ires,lx2,yf(1,1,1,ie),nxr,txa,nyzr)
         do iz=1,nzr
            izoff = 1 + (iz-1)*lx2*nxr
            call mxm (txa(izoff),lx2,itres,nxr,txb(1,1,iz),ly2)
         enddo
         if (if3d) then
            call mxm (txb,nxy2,itres,nzr,xf(1,1,1,ie),lz2)
         else
            call copy(xf(1,1,1,ie),txb,nxy2)
         endif
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Map loaded variables from velocity to axisymmetric mesh
!! @ingroup chkpoint_mstep
!! @note This is version of @ref axis_interp_ic taking into account fact
!! pressure does not have to be written on velocity mesh.
!! @remark This routine uses global scratch space \a CTMP0.
      subroutine chkpt_axis_interp_ic()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'RESTART'
      include 'SOLN'
      include 'GEOM'
      include 'IXYZ'

      ! scratch space
      real axism1 (lx1,ly1), axism2 (lx2,ly2), ialj2 (ly2,ly2),
     $     iatlj2(ly2,ly2), tmp(ly2,ly2)
      common /ctmp0/ axism1, axism2, ialj2, iatlj2, tmp

      ! local variables
      integer el, ips, is1
!-----------------------------------------------------------------------
      if (.not.ifaxis) return

      ! get interpolation operators between Gauss-Lobatto Jacobi
      ! and and Gauss Legendre poits (this is missing in genwz)
      !call invmt(iajl2 ,ialj2 ,tmp ,ly2)
      call invmt(iatjl2,iatlj2,tmp,ly2)

      do el=1,nelv
         if (ifrzer(el)) then
           if (ifgetx) then
             call mxm(xm1(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
             call copy(xm1(1,1,1,el),axism1,nx1*ny1)
             call mxm(ym1(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
             call copy(ym1(1,1,1,el),axism1,nx1*ny1)
           endif
           if (ifgetu) then
             call mxm(vx(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
             call copy(vx(1,1,1,el),axism1,nx1*ny1)
             call mxm(vy(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
             call copy(vy(1,1,1,el),axism1,nx1*ny1)
           endif
           if (ifgetw) then
             call mxm(vz(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
             call copy(vz(1,1,1,el),axism1,nx1*ny1)
           endif
           if (ifgetp) then
             if (ifsplit) then
                call mxm(pr(1,1,1,el),nx1,iatlj1,ny1,axism1,ny1)
                call copy(pr(1,1,1,el),axism1,nx1*ny1)
             else
                call mxm(pr(1,1,1,el),nx2,iatlj2,ny2,axism2,ny2)
                call copy(pr(1,1,1,el),axism2,nx2*ny2)
             endif
           endif
           if (ifgett) then
             call mxm(t(1,1,1,el,1),nx1,iatlj1,ny1,axism1,ny1)
             call copy(t(1,1,1,el,1),axism1,nx1*ny1)
           endif
           do ips=1,npscal
            is1 = ips + 1
            if (ifgtps(ips)) then
             call mxm(t(1,1,1,el,is1),nx1,iatlj1,ny1,axism1,ny1)
             call copy(t(1,1,1,el,is1),axism1,nx1*ny1)
            endif
           enddo
         endif
      enddo

      return
      end subroutine
!=======================================================================
