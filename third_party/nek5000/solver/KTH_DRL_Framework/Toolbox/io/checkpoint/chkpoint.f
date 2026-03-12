!> @file chkpoint.f
!! @ingroup chkpoint
!! @brief Set of checkpoint routines
!! @details This is a main interface reading/writing runtime parameters
!! and calling proper submodule.
!=======================================================================
!> @brief Register checkpointing module
!! @ingroup chkpoint
!! @note This routine should be called in frame_usr_register
      subroutine chkpt_register()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'CHKPOINTD'

      ! local variables
      integer lpmid
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,chpt_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(chpt_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'Parent module ['//'FRAME'//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(chpt_id,lpmid,chpt_name,'Checkpointing I/O')

      ! register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(chpt_ttot_id,lpmid,chpt_id,
     $      'CHP_TOT','Checkpointing total time',.false.)

      call mntr_tmr_reg(chpt_tini_id,chpt_ttot_id,chpt_id,
     $      'CHP_INI','Checkpointing initialisation time',.true.)

      ! register and set active section
      call rprm_sec_reg(chpt_sec_id,chpt_id,'_'//adjustl(chpt_name),
     $     'Runtime paramere section for checkpoint module')
      call rprm_sec_set_act(.true.,chpt_sec_id)

      ! register parameters
      call rprm_rp_reg(chpt_ifrst_id,chpt_sec_id,'READCHKPT',
     $     'Restat from checkpoint',rpar_log,0,0.0,.false.,' ')

      call rprm_rp_reg(chpt_fnum_id,chpt_sec_id,'CHKPFNUMBER',
     $     'Restart file number',rpar_int,1,0.0,.false.,' ')

      call rprm_rp_reg(chpt_step_id,chpt_sec_id,'CHKPINTERVAL',
     $     'Checkpiont saving frequency (number of time steps)',
     $      rpar_int,500,0.0,.false.,' ')

      ! set initial step delay
      call mntr_set_step_delay(1)

      ! call submodule registration
      call chkpts_register

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(chpt_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise checkpointing module
!! @ingroup chkpoint
!! @note This routine should be called in frame_usr_init
      subroutine chkpt_init()
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'FRAMELP'
      include 'CHKPOINTD'

      ! local variables
      integer itmp, lstdl
      real rtmp, ltim
      logical ltmp
      character*20 ctmp
      character*2 str
      character*200 lstring

      ! functions
      logical chkpts_is_initialised
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (chpt_ifinit) then
         call mntr_warn(chpt_id,
     $        'module ['//trim(chpt_name)//'] already initiaised.')
         ! check submodule intialisation
         if (.not.chkpts_is_initialised()) then
            call mntr_abort(chpt_id,
     $        'required submodule module not initiaised.')
         endif
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_ifrst_id,rpar_log)
      chpt_ifrst = ltmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_fnum_id,rpar_int)
      chpt_fnum = itmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_step_id,rpar_int)
      chpt_step = itmp

      if (chpt_ifrst) then
      ! get input set number
         chpt_set_i = chpt_fnum - 1
         if (chpt_set_i.ge.chpt_nset) then
            write(str,'(I2)') chpt_nset + 1
            lstring = 'chpt_fnum must be in the range: 1-'//trim(str)
            call mntr_abort(chpt_id,lstring)
         endif

         chpt_set_o = mod(chpt_set_i+1,chpt_nset)
      else
         chpt_set_o = 0
      endif

      ! set reset flag
      chpt_reset = -1

      ! check number of steps
      call mntr_get_step_delay(lstdl)
      if (NSTEPS.lt.3*lstdl) then
         call mntr_abort(chpt_id,'too short run for multi-file restart')
      endif

      ! check checkpoint frequency
      if (chpt_step.lt.2*lstdl.or.chpt_step.gt.NSTEPS) then
         chpt_step = NSTEPS
         call mntr_warn(chpt_id,'wrong chpt_step; resetting to NSTEPS')
      endif

      ! set min and max ISTEP for cyclic checkpoint writning
      ! timesteps outside this bouds require special treatment
      chpt_istep = lstdl
      chpt_nstep = NSTEPS - lstdl - 1
      ! check if chpt_nstep is in the middle of writing cycle
      itmp = chpt_nstep + chpt_step -1
      if (mod(itmp,chpt_step).ge.(chpt_step-lstdl)) then
         itmp = lstdl + mod(itmp,chpt_step) + 1 - chpt_step
         chpt_nstep = chpt_nstep - itmp
      endif

      ! call submodule initialisation
      call chkpts_init

      ! is everything initialised
      if (chkpts_is_initialised()) chpt_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(chpt_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup chkpoint
!! @return chkpt_is_initialised
      logical function chkpt_is_initialised()
      implicit none

      include 'SIZE'
      include 'CHKPOINTD'
!-----------------------------------------------------------------------
      chkpt_is_initialised = chpt_ifinit

      return
      end function
!=======================================================================
!> @brief Main checkpoint interface
!! @ingroup chkpoint
!! @note This routine should be called in userchk as a first framework call
!     after frame_monitor
      subroutine chkpt_main
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'CHKPOINTD'

      ! local variables
      integer itmp, lstdl
!-----------------------------------------------------------------------
      if(chpt_ifrst.and.ISTEP.le.chpt_istep) then
         call chkpts_read
      elseif (ISTEP.gt.chpt_istep) then

      ! adjust max ISTEP for cyclic checkpoint writning
         call mntr_get_step_delay(lstdl)
         chpt_nstep = NSTEPS - lstdl -1
      ! check if chpt_nstep is in the middle of writing cycle
         itmp = chpt_nstep + chpt_step -1
         if (mod(itmp,chpt_step).ge.(chpt_step-lstdl)) then
            itmp = lstdl + mod(itmp,chpt_step) + 1 - chpt_step
            chpt_nstep = chpt_nstep - itmp
         endif

      ! count steps to the end of wrting stage
         itmp = ISTEP + chpt_step -1
         if (ISTEP.gt.(NSTEPS-lstdl)) then
            chpt_stepc = NSTEPS-ISTEP+1
         elseif (ISTEP.lt.chpt_nstep.and.
     $    mod(itmp,chpt_step).ge.(chpt_step-lstdl)) then
            chpt_stepc = chpt_step - mod(itmp,chpt_step)
         else
            chpt_stepc = -1
         endif

      ! get the checkpoint set number
      ! to avoid conflicts with dependent packages I reset chpt_set_o
      ! during the step after checkpointing
         if (ISTEP.lt.chpt_nstep.and.mod(ISTEP,chpt_step).eq.0) then
            chpt_reset = mod(chpt_set_o+1,chpt_nset)
         elseif (chpt_reset.ge.0) then
            chpt_set_o = chpt_reset
            chpt_reset = -1
         endif

         call chkpts_write
      endif

      return
      end subroutine
!=======================================================================
!> @brief Get step count to the checkpoint and a set number
!! @ingroup chkpoint
!! @param[out] step_cnt   decreasing step count in checkpoint writinh phase (otherwise -1)
!! @param[out] set_out    set number
      subroutine chkpt_get_fset(step_cnt, set_out)
      implicit none

      include 'SIZE'
      include 'CHKPOINTD'

      ! argument list
      integer step_cnt, set_out
!-----------------------------------------------------------------------
      step_cnt = chpt_stepc
      set_out  = chpt_set_o

      return
      end subroutine
!=======================================================================
