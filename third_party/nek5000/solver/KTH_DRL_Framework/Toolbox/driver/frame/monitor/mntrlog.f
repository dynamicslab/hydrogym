!> @file mntrlog.f
!! @ingroup monitor
!! @brief Set of module register and logging routines for KTH framework
!! @author Adam Peplinski
!! @date Sep 28, 2017
!=======================================================================
!> @brief Initialise monitor by registering framework and monitor
!! @ingroup monitor
!! @param[in]  log_thr   initial log threshold
      subroutine mntr_register_mod(log_thr)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'
      include 'MNTRTMRD'

      ! argument list
      integer log_thr

      ! local variables
      character*2 str
      character*200 lstring

      ! functions
      integer frame_get_master
      real dnekclock
!-----------------------------------------------------------------------
      ! simple timing
      mntr_frame_tmini = dnekclock()

      ! set master node
      mntr_pid0 = frame_get_master()

      ! first register framework
      mntr_frame_id = 1
      mntr_mod_id(mntr_frame_id) = 0
      mntr_mod_name(mntr_frame_id) = mntr_frame_name
      mntr_mod_dscr(mntr_frame_id) = 'Framework backbone'
      mntr_mod_num = mntr_mod_num + 1
      mntr_mod_mpos = mntr_mod_mpos + 1

      ! next monitor
      mntr_id = 2
      mntr_mod_id(mntr_id) = mntr_frame_id
      mntr_mod_name(mntr_id) = mntr_name
      mntr_mod_dscr(mntr_id) = 'Monitoring module'
      mntr_mod_num = mntr_mod_num + 1
      mntr_mod_mpos = mntr_mod_mpos + 1

      ! set log threshold
      mntr_lp_def = log_thr

      ! log changes
      lstring ='Registered module ['//trim(mntr_mod_name(mntr_frame_id))
      lstring= trim(lstring)//']: '//trim(mntr_mod_dscr(mntr_frame_id))
      call mntr_log(mntr_id,lp_inf,trim(lstring))

      lstring = 'Registered module ['//trim(mntr_mod_name(mntr_id))
      lstring= trim(lstring)//']: '//trim(mntr_mod_dscr(mntr_id))
      call mntr_log(mntr_id,lp_inf,trim(lstring))

      ! register framework timer and get initiaisation time
      call mntr_tmr_reg(mntr_frame_tmr_id,0,mntr_frame_id,
     $     'FRM_TOT','Total elapsed framework time',.false.)

      write(str,'(I2)') mntr_lp_def
      call mntr_log(mntr_id,lp_inf,
     $     'Initial log threshold set to: '//trim(str))

      return
      end subroutine
!=======================================================================
!> @brief Register monitor runtime parameters
!! @ingroup monitor
      subroutine mntr_register_par
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! local variables
      integer rpid,itmp
      real rtmp
      logical ltmp
      character*20 ctmp
!-----------------------------------------------------------------------
      ! register and set active section
      call rprm_sec_reg(mntr_sec_id,mntr_id,'_'//adjustl(mntr_name),
     $     'Runtime parameter section for monitor module')
      call rprm_sec_set_act(.true.,mntr_sec_id)

      ! register parameters
      call rprm_rp_reg(mntr_lp_def_id,mntr_sec_id,'LOGLEVEL',
     $     'Logging threshold for toolboxes',rpar_int,mntr_lp_def,
     $      0.0,.false.,' ')

      call rprm_rp_reg(mntr_iftdsc_id,mntr_sec_id,'IFTIMDSCR',
     $     'Write timer description in the summary',rpar_log,0,
     $      0.0,.false.,' ')

      call rprm_rp_reg(mntr_wtime_id,mntr_sec_id,'WALLTIME',
     $     'Simulation wall time',rpar_str,0,0.0,.false.,'00:00')

      return
      end subroutine
!=======================================================================
!> @brief Initialise monitor module
!! @ingroup monitor
      subroutine mntr_init
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! local variables
      integer ierr, nhour, nmin
      integer itmp
      real rtmp
      logical ltmp
      character*20 ctmp
      character*2 str
!-----------------------------------------------------------------------
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,mntr_lp_def_id,rpar_int)
      mntr_lp_def = itmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,mntr_iftdsc_id,rpar_log)
      mntr_iftdsc = ltmp

      write(str,'(I2)') mntr_lp_def
      call mntr_log(mntr_id,lp_inf,
     $     'Reseting log threshold to: '//trim(str))

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,mntr_wtime_id,rpar_str)
      mntr_wtimes = ctmp

      ! get wall clock
      ctmp = trim(adjustl(mntr_wtimes))
      ! check string format
      ierr = 0
      if (ctmp(3:3).ne.':') ierr = 1
      if (.not.(LGE(ctmp(1:1),'0').and.LLE(ctmp(1:1),'9'))) ierr = 1
      if (.not.(LGE(ctmp(2:2),'0').and.LLE(ctmp(2:2),'9'))) ierr = 1
      if (.not.(LGE(ctmp(4:4),'0').and.LLE(ctmp(4:4),'9'))) ierr = 1
      if (.not.(LGE(ctmp(5:5),'0').and.LLE(ctmp(5:5),'9'))) ierr = 1

      if (ierr.eq.0) then
         read(ctmp(1:2),'(I2)') nhour
         read(ctmp(4:5),'(I2)') nmin
         mntr_wtime = 60.0*(nmin +60*nhour)
      else
         call mntr_log(mntr_id,lp_inf,'Wrong wall time format')
      endif

      ! write summary
      call mntr_mod_summary_print()

      mntr_ifinit = .true.

      return
      end subroutine
!=======================================================================
!> @brief Monitor simulation wall clock
!! @ingroup monitor
      subroutine mntr_wclock
      implicit none

      include 'SIZE'
      include 'TSTEP'           ! ISTEP, NSTEPS, LASTEP
      include 'INPUT'           ! IFMVBD, IFREGUO
      include 'PARALLEL'        ! WDSIZE
      include 'CTIMER'          ! ETIMES
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! local variables
      integer il, lstdl
      real rtmp
!-----------------------------------------------------------------------
      ! double delay step as monitoing routine does not know when checkpointing
      ! starts and wall clock can be reached in the middle of writnig process
      lstdl = 2*mntr_stdl+1

      ! check simulation wall time
      if (mntr_wtime.gt.0.0) then

         ! save wall time of the current step
         do il=lstdl,2,-1
            mntr_wtstep(il) = mntr_wtstep(il-1)
         enddo
         mntr_wtstep(1) = dnekclock() - ETIMES
         ! check if simulation is going to exceed wall time, but
         ! first let read all checkpointing files (necessary for multi file
         ! checkpointing)
         if (ISTEP.gt.lstdl) then
            ! it should be enough for the master to check condition
            if (NID.eq.mntr_pid0) rtmp = 2.0*mntr_wtstep(1) -
     $         mntr_wtstep(lstdl)
            ! broadcast predicted time
            il = WDSIZE
            call bcast(rtmp,il)

            if (rtmp.gt.mntr_wtime.and.(NSTEPS-ISTEP).gt.lstdl) then
               call mntr_log(mntr_id,lp_inf,
     $                 'Wall clock reached; adjust NSTEPS')
               NSTEPS = ISTEP+lstdl
            endif
         endif
      endif

      ! check convergence flag
      if (mntr_ifconv.and.(NSTEPS-ISTEP).gt.lstdl) then
         call mntr_log(mntr_id,lp_inf,
     $            'Simulation converged; adjust NSTEPS')
         NSTEPS = ISTEP+lstdl
      endif

      ! just to take into account there is istep and kstep,
      ! and kstep is just a local variable
      if (ISTEP.ge.NSTEPS) LASTEP=1

      return
      end subroutine
!=======================================================================
!> @brief Set number of steps necessary to write proper checkpointing
!! @ingroup monitor
!! @param[in] dstep   step delay
      subroutine mntr_set_step_delay(dstep)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer dstep

!-----------------------------------------------------------------------
      if (dstep.gt.mntr_stdl_max) then
         call mntr_abort(mntr_id,"Step delay exceeds mntr_stdl_max")
      else
         mntr_stdl = max(mntr_stdl,dstep)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Get step delay
!! @ingroup monitor
!! @param[out] dstep   step delay
      subroutine mntr_get_step_delay(dstep)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer dstep
!-----------------------------------------------------------------------
      dstep = mntr_stdl

      return
      end subroutine
!=======================================================================
!> @brief Set convergence flag to shorten simulation
!! @ingroup monitor
      subroutine mntr_set_conv(ifconv)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      logical ifconv

!-----------------------------------------------------------------------
      mntr_ifconv = ifconv

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup monitor
!! @return mntr_is_initialised
      logical function mntr_is_initialised()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'
!-----------------------------------------------------------------------
      mntr_is_initialised = mntr_ifinit

      return
      end function
!=======================================================================
!> @brief Get logging threashold
!! @ingroup monitor
!! @return mntr_lp_def_get
      integer function mntr_lp_def_get()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'
!-----------------------------------------------------------------------
      mntr_lp_def_get = mntr_lp_def

      return
      end function
!=======================================================================
!> @brief Register new module
!! @ingroup monitor
!! @param[out] mid      current module id
!! @param[in]  pmid     parent module id
!! @param[in]  mname    module name
!! @param[in]  mdscr    module description
      subroutine mntr_mod_reg(mid,pmid,mname,mdscr)
      implicit none

      include 'SIZE'
      include 'PARALLEL'        ! ISIZE
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer mid, pmid
      character*(*) mname, mdscr

      ! local variables
      character*10  lname
      character*132 ldscr
      integer slen,slena

      integer il, ipos
!-----------------------------------------------------------------------
      ! check name length
      slena = len_trim(adjustl(mname))
      ! remove trailing blanks
      slen = len_trim(mname) - slena + 1
      if (slena.gt.mntr_lstl_mnm) then
         call mntr_log(mntr_id,lp_deb,
     $        'too long module name; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(lname,mntr_lstl_mnm)
      lname= mname(slen:slen+slena- 1)
      call capit(lname,slena)

      ! check description length
      slena = len_trim(adjustl(mdscr))
      ! remove trailing blanks
      slen = len_trim(mdscr) - slena + 1
      if (slena.ge.mntr_lstl_mds) then
         call mntr_log(mntr_id,lp_deb,
     $        'too long module description; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(ldscr,mntr_lstl_mds)
      ldscr= mdscr(slen:slen + slena - 1)

      ! find empty space
      ipos = 0

      ! to ensure consistency I do it on master and broadcast result
      if (nid.eq.mntr_pid0) then

         ! check if module is already registered
         do il=1,mntr_mod_mpos
            if (mntr_mod_id(il).ge.0.and.
     $         mntr_mod_name(il).eq.lname) then
               ipos = -il
               exit
            endif
         enddo

         ! find empty spot
         if (ipos.eq.0) then
            do il=1,mntr_id_max
               if (mntr_mod_id(il).eq.-1) then
                  ipos = il
                  exit
               endif
            enddo
         endif
      endif

      ! broadcast mid
      call bcast(ipos,isize)

      ! error; no free space found
      if (ipos.eq.0) then
         mid = ipos
         call mntr_abort(mntr_id,
     $        'module ['//trim(lname)//'] cannot be registered')
      !  module already registered
      elseif (ipos.lt.0) then
         mid = abs(ipos)
         call mntr_abort(mntr_id,
     $    'Module ['//trim(lname)//'] is already registered')
      ! new module
      else
         mid = ipos
         ! check if parent module is registered
         if (pmid.gt.0) then
            if (mntr_mod_id(pmid).ge.0) then
               mntr_mod_id(ipos) = pmid
            else
               mntr_mod_id(ipos) = 0
               call mntr_log(mntr_id,lp_inf,
     $       "Module's ["//trim(lname)//"] parent not registered.")
            endif
         else
            mntr_mod_id(ipos) = 0
         endif
         mntr_mod_name(ipos)=lname
         mntr_mod_dscr(ipos)=ldscr
         mntr_mod_num = mntr_mod_num + 1
         if (mntr_mod_mpos.lt.ipos) mntr_mod_mpos = ipos
         call mntr_log(mntr_id,lp_inf,
     $       'Registered module ['//trim(lname)//']: '//trim(ldscr))
      endif

      return
      end subroutine
!=======================================================================
!> @brief Check if module name is registered and return its id.
!! @ingroup monitor
!! @param[out] mid      module id
!! @param[in]  mname    module name
      subroutine mntr_mod_is_name_reg(mid,mname)
      implicit none

      include 'SIZE'
      include 'PARALLEL'        ! ISIZE
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer mid
      character*(*) mname

      ! local variables
      character*10  lname
      character*3 str
      integer slen,slena

      integer il, ipos
!-----------------------------------------------------------------------
      ! check name length
      slena = len_trim(adjustl(mname))
      ! remove trailing blanks
      slen = len_trim(mname) - slena + 1
      if (slena.gt.mntr_lstl_mnm) then
         call mntr_log(mntr_id,lp_deb,
     $          'too long module name; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(lname,mntr_lstl_mnm)
      lname= mname(slen:slen+slena- 1)
      call capit(lname,slena)

      ! find module
      ipos = 0

      ! to ensure consistency I do it on master and broadcast result
      if (nid.eq.mntr_pid0) then
         ! check if module is already registered
         do il=1,mntr_mod_mpos
            if (mntr_mod_id(il).ge.0.and.
     $         mntr_mod_name(il).eq.lname) then
               ipos = il
               exit
            endif
         enddo
      endif

      ! broadcast ipos
      call bcast(ipos,isize)

      if (ipos.eq.0) then
         mid = -1
         call mntr_log(mntr_id,lp_inf,
     $        'Module ['//trim(lname)//'] not registered')
      else
         mid = ipos
         write(str,'(I3)') ipos
         call mntr_log(mntr_id,lp_vrb,
     $        'Module ['//trim(lname)//'] registered with mid='//str)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Check if module id is registered. This operation is performed locally
!! @ingroup monitor
!! @param[in] mid      module id
!! @return mntr_mod_is_id_reg
      logical function mntr_mod_is_id_reg(mid)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer mid
!-----------------------------------------------------------------------
      mntr_mod_is_id_reg = mntr_mod_id(mid).ge.0

      return
      end function
!=======================================================================
!> @brief Get number of registered modules. This operation is performed locally
!! @ingroup monitor
!! @param[out]    nmod     module number
!! @param[out]    mmod     max module id
      subroutine mntr_mod_get_number(nmod,mmod)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer nmod, mmod
!-----------------------------------------------------------------------
      nmod = mntr_mod_num
      mmod = mntr_mod_mpos

      return
      end subroutine
!=======================================================================
!> @brief Get module name an parent id for given module id. This operation is performed locally
!! @ingroup monitor
!! @param[out]    pmid     parent module id
!! @param[out]    mname    module name
!! @param[inout]  mid      module id
      subroutine mntr_mod_get_info(mname, pmid,mid)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      character*10 mname
      integer mid, pmid

      ! local variables
      character*5 str
!-----------------------------------------------------------------------
      if (mntr_mod_id(mid).ge.0) then
         pmid = mntr_mod_id(mid)
         mname = mntr_mod_name(mid)
      else
         mid = -1
         write(str,'(I3)') mid
         call mntr_log(mntr_id,lp_vrb,
     $        'Module id'//trim(str)//' not registered')
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write log message
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] priority  log priority
!! @param[in] logs      log body
      subroutine mntr_log(mid,priority,logs)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer mid,priority
      character*(*) logs

      ! local variables
      character*200 llogs
      character*5 str
      integer slen, slena
!-----------------------------------------------------------------------
      ! check log priority
      if (priority.lt.mntr_lp_def) return

      ! done only by master
      if (nid.eq.mntr_pid0) then

         ! check description length
         slena = len_trim(adjustl(logs))
         ! remove trailing blanks
         slen = len_trim(logs) - slena + 1
         if (slena.ge.mntr_lstl_log) then
            if (mntr_lp_def.le.lp_deb) write(*,*)' ['//mntr_name//'] ',
     $       'too long log string; shortenning'
            slena = min(slena,mntr_lstl_log)
         endif
         call blank(llogs,mntr_lstl_mds)
         llogs= logs(slen:slen + slena - 1)

         ! check module id
         if (mntr_mod_id(mid).ge.0) then
            ! add module name
            write(*,*) ' ['//trim(mntr_mod_name(mid))//'] '//trim(llogs)
         else
            write(str,'(I3)') mid
            write(*,*) ' ['//trim(mntr_name)//'] ',
     $      ' WARNING: module'//trim(str)//' not registered;'
            write(*,*) 'Log body: '//trim(llogs)
         endif
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write log message from given process
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] priority  log priority
!! @param[in] logs      log body
!! @param[in] prid      process id
      subroutine mntr_log_local(mid,priority,logs,prid)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer mid,priority, prid
      character*(*) logs

      ! local variables
      character*200 llogs
      character*5 str
      integer slen, slena
!-----------------------------------------------------------------------
      ! check log priority
      if (priority.lt.mntr_lp_def) return

      ! done only by given process

      ! check description length
      slena = len_trim(adjustl(logs))
      ! remove trailing blanks
      slen = len_trim(logs) - slena + 1
      if (slena.ge.mntr_lstl_log) then
         if (mntr_lp_def.le.lp_deb) write(*,*)' ['//mntr_name//'] ',
     $       'too long log string; shortenning'
         slena = min(slena,mntr_lstl_log)
      endif
      call blank(llogs,mntr_lstl_mds)
      llogs= logs(slen:slen + slena - 1)

      ! check module id
      if (mntr_mod_id(mid).ge.0) then
      ! add module name
       write(*,*) ' ['//trim(mntr_mod_name(mid))//'] nid= ',prid,
     $      ' '//trim(llogs)
      else
         write(str,'(I3)') mid
         write(*,*) ' ['//trim(mntr_name)//'] ',
     $   ' WARNING: module'//trim(str)//' not registered;'
         write(*,*) 'Log body: nid= ',prid,' '//trim(llogs)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write log message adding single integer
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] priority  log priority
!! @param[in] logs      log body
!! @param[in] ivar      integer variable
      subroutine mntr_logi(mid,priority,logs,ivar)
      implicit none

      ! argument list
      integer mid,priority,ivar
      character*(*) logs

      ! local variables
      character*10 str
!-----------------------------------------------------------------------
      write(str,'(I8)') ivar
      call mntr_log(mid,priority,trim(logs)//' '//trim(str))

      return
      end subroutine
!=======================================================================
!> @brief Write log message adding single real
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] priority  log priority
!! @param[in] logs      log body
!! @param[in] rvar      real variable
      subroutine mntr_logr(mid,priority,logs,rvar)
      implicit none

      ! argument list
      integer mid,priority
      character*(*) logs
      real rvar

      ! local variables
      character*20 str
!-----------------------------------------------------------------------
      write(str,'(E15.8)') rvar
      call mntr_log(mid,priority,trim(logs)//' '//trim(str))

      return
      end subroutine
!=======================================================================
!> @brief Write log message adding single logical
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] priority  log priority
!! @param[in] logs      log body
!! @param[in] lvar      logical variable
      subroutine mntr_logl(mid,priority,logs,lvar)
      implicit none

      ! argument list
      integer mid,priority
      character*(*) logs
      logical lvar

      ! local variables
      character*2 str
!-----------------------------------------------------------------------
      write(str,'(L2)') lvar
      call mntr_log(mid,priority,trim(logs)//' '//trim(str))

      return
      end subroutine
!=======================================================================
!> @brief Write warning message
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] logs      log body
      subroutine mntr_warn(mid,logs)
      implicit none

      include 'FRAMELP'

      ! argument list
      integer mid,priority
      character*(*) logs
!-----------------------------------------------------------------------
      call mntr_log(mid,lp_inf,'WARNING: '//logs)
      return
      end subroutine
!=======================================================================
!> @brief Write error message
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] logs      log body
      subroutine mntr_error(mid,logs)
      implicit none

      include 'FRAMELP'

      ! argument list
      integer mid
      character*(*) logs
!-----------------------------------------------------------------------
      call mntr_log(mid,lp_err,'ERROR: '//logs)
      return
      end subroutine
!=======================================================================
!> @brief Abort simulation
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] logs      log body
      subroutine mntr_abort(mid,logs)
      implicit none

      include 'FRAMELP'

      ! argument list
      integer mid
      character*(*) logs
!-----------------------------------------------------------------------
      call mntr_log(mid,lp_err,'ABORT: '//logs)
      call exitt
      return
      end subroutine
!=======================================================================
!> @brief Abort simulation
!! @ingroup monitor
!! @param[in] mid       module id
!! @param[in] ierr      error flag
!! @param[in] logs      log body
      subroutine mntr_check_abort(mid,ierr,logs)
      implicit none

      include 'FRAMELP'

      ! argument list
      integer mid,ierr
      character*(*) logs

      ! local variables
      integer imax, imin, itest
      character*5 str
      ! functions
      integer iglmax, iglmin
!-----------------------------------------------------------------------
      imax = iglmax(ierr,1)
      imin = iglmin(ierr,1)
      if (imax.gt.0) then
         itest = imax
      else
         itest = imin
      endif

      if (itest.ne.0) then
         write(str,'(I3)') itest
         call mntr_log(mid,lp_err,
     $         'ABORT: '//trim(logs)//' ierr='//trim(str))
         call exitt
      endif
      return
      end subroutine
!=======================================================================
!> @brief Print registered modules showing tree structure
!! @ingroup monitor
      subroutine mntr_mod_summary_print()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! local variables
      integer il, stride
      parameter (stride=4)
      integer olist(2,mntr_id_max), ierr
      character*25 ftm
      character*3 str
!-----------------------------------------------------------------------
      call mntr_log(mntr_id,lp_prd,
     $         'Summary of registered modules')

      if (nid.eq.mntr_pid0) then
         ! get ordered list
         call mntr_mod_get_olist(olist, ierr)


         if(ierr.eq.0.and.mntr_lp_def.le.lp_prd) then
            do il=1,mntr_mod_num
               write(str,'(I3)') stride*(olist(2,il))
               ftm = '('//trim(str)//'X,"[",A,"] : ",A)'
               write(*,ftm) mntr_mod_name(olist(1,il)),
     $                mntr_mod_dscr(olist(1,il))
            enddo
         endif
      endif

      return
      end subroutine
!=======================================================================
!> @brief Provide ordered list of registered modules for printing.
!! @ingroup monitor
!! @param[out]   olist    ordered list
!! @param[out]   ierr     error flag
      subroutine mntr_mod_get_olist(olist,ierr)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'

      ! argument list
      integer olist(2,mntr_id_max), ierr

      ! local variables
      integer ind(mntr_id_max), level, parent, ipos
      integer slist(2,mntr_id_max), itmp1(2)
      integer npos, key
      integer il, jl
      integer istart, in, itest
!-----------------------------------------------------------------------
      ierr = 0

      ! sort module index array
      ! copy data removing possible empty slots
      npos=0
      do il=1,mntr_mod_mpos
         if (mntr_mod_id(il).ge.0) then
            npos = npos + 1
            slist(1,npos) = mntr_mod_id(il)
            slist(2,npos) = il
         endif
      enddo
      if(npos.ne.mntr_mod_num) then
         ierr = 1
         call mntr_log_local(mntr_id,lp_inf,
     $         'Inconsistent module number; return',mntr_pid0)
         return
      endif

      ! sort with respect to parent id
      key = 1
      call ituple_sort(slist,2,npos,key,1,ind,itmp1)

      ! sort within children of single parent with respect to children id
      istart = 1
      itest = slist(1,istart)
      do il=1,npos
         if(itest.ne.slist(1,il).or.il.eq.npos) then
           if (il.eq.npos.and.itest.eq.slist(1,il)) then
              jl = npos + 1
           else
              jl = il
           endif
           in = jl - istart
           if (itest.eq.0.and.in.ne.1) then
              call mntr_log_local(mntr_id,lp_inf,
     $         'Must be single root of the graph; return',mntr_pid0)
              ierr = 2
              return
           endif
           if (in.gt.1) then
              key = 2
              call ituple_sort(slist(1,istart),2,in,key,1,ind,itmp1)
           endif
           if (il.ne.npos) then
              itest = slist(1,il)
              istart = il
           endif
         endif
      enddo

      parent = 0
      level = 0
      ipos = 1
      call mntr_build_ord_list(olist,slist,npos,ipos,parent,level)

      return
      end subroutine
!=======================================================================
!> @brief Build ordered list reflecting graph structure
!! @ingroup monitor
!! @param[out]   olist    ordered list
!! @param[inout] slist    list sorted with respect to parent
!! @param[in]    nlist    lists length
!! @param[inout] npos     position in olist array
!! @param[in]    parent   parent id
!! @param[in]    level    parent level
      recursive subroutine mntr_build_ord_list(olist,slist,nlist,npos,
     $     parent,level)
      implicit none

      ! argument list
      integer nlist, npos, parent, level
      integer olist(2,nlist),slist(2,nlist)

      ! local variables
      integer il
      integer lparent, llevel
!-----------------------------------------------------------------------
      llevel = level + 1
      do il=1, nlist
         if (slist(1,il).eq.parent) then
            slist(1,il) = - parent
            lparent = slist(2,il)
            olist(1,npos) = lparent
            olist(2,npos) = llevel
            npos = npos +1
            call mntr_build_ord_list(olist,slist,nlist,npos,lparent,
     $           llevel)
         endif
      enddo

      return
      end subroutine
!=======================================================================
