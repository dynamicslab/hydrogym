!> @file powerit.f
!! @ingroup powerit
!! @brief Set of subroutines to perform power iterations within time stepper
!! @author Adam Peplinski
!! @date Mar 7, 2016
!=======================================================================
!> @brief Register power iteration module
!! @ingroup powerit
!! @note This interface is called by @ref tst_register
      subroutine stepper_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'POWERITD'

      ! local variables
      integer lpmid, il
      real ltim
      character*2 str

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,pwi_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(pwi_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,tst_name)
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'parent module ['//trim(tst_name)//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(pwi_id,lpmid,pwi_name,
     $      'Power iterations for time stepper')

      ! register timers
      ! initialisation
      call mntr_tmr_reg(pwi_tmr_ini_id,tst_tmr_ini_id,pwi_id,
     $     'PWI_INI','Power iteration initialisation time',.true.)
      ! submodule operation
      call mntr_tmr_reg(pwi_tmr_evl_id,tst_tmr_evl_id,pwi_id,
     $     'PWI_EVL','Power iteration evolution time',.true.)

      ! register and set active section
      call rprm_sec_reg(pwi_sec_id,pwi_id,'_'//adjustl(pwi_name),
     $     'Runtime paramere section for power iteration module')
      call rprm_sec_set_act(.true.,pwi_sec_id)

      ! register parameters
      call rprm_rp_reg(pwi_l2n_id,pwi_sec_id,'L2N',
     $     'Vector initial norm',rpar_real,0,1.0,.false.,' ')

      ! set initialisation flag
      pwi_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pwi_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise power iteration module
!! @ingroup powerit
!! @note This interface is called by @ref tst_init
      subroutine stepper_init()
      implicit none

      include 'SIZE'
      include 'SOLN'            ! V[XYZ]P, TP
      include 'MASS'            ! BM1
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'TSTEPPERD'
      include 'POWERITD'

      ! local variables
      integer itmp, il, set_in
      real rtmp, ltim, lnorm
      logical ltmp
      character*20 ctmp

      ! functions
      real dnekclock, cht_glsc2_wt
      logical chkpts_is_initialised
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (pwi_ifinit) then
         call mntr_warn(pwi_id,
     $        'module ['//trim(pwi_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pwi_l2n_id,rpar_real)
      pwi_l2n = rtmp

      ! get restart options
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_ifrst_id,rpar_log)
      pwi_ifrst = ltmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_fnum_id,rpar_int)
      pwi_fnum = itmp
      set_in = pwi_fnum -1

      ! initial growth rate
      pwi_grw = 0.0

      ! place to read checkpoint file
      if (pwi_ifrst) then
         if(.not.chkpts_is_initialised()) call mntr_abort(pwi_id,
     $        'Checkpointing module not initialised')
         call stepper_read(set_in)
      endif

      ! normalise vector
      lnorm = cht_glsc2_wt(VXP,VYP,VZP,TP,VXP,VYP,VZP,TP,BM1)
      lnorm = sqrt(pwi_l2n/lnorm)
      call cht_opcmult (VXP,VYP,VZP,TP,lnorm)

      ! make sure the velocity and temperature fields are continuous at
      ! element faces and edges
      call tst_dssum

      ! save intial vector
      call cht_opcopy (pwi_vx,pwi_vy,pwi_vz,pwi_t,VXP,VYP,VZP,TP)

      ! stamp log file
      call mntr_log(pwi_id,lp_prd,'POWER ITERATIONS initialised')
      call mntr_logr(pwi_id,lp_prd,'L2NORM = ',pwi_l2n)

      ! everything is initialised
      pwi_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pwi_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup powerit
!! @return stepper_is_initialised
      logical function stepper_is_initialised()
      implicit none

      include 'SIZE'
      include 'POWERITD'
!-----------------------------------------------------------------------
      stepper_is_initialised = pwi_ifinit

      return
      end function
!=======================================================================
!> @brief Renormalise vector and check convergence.
!! @ingroup powerit
!! @note This interface is defined in @ref tstpr_solve
!! @remarks This routine uses global scratch space SCRUZ
      subroutine stepper_vsolve
      implicit none

      include 'SIZE'            ! NIO
      include 'TSTEP'           ! TIME, LASTEP, NSTEPS
      include 'INPUT'           ! IFHEAT
      include 'MASS'            ! BM1
      include 'SOLN'            ! V[XYZ]P, TP
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'TSTEPPERD'
      include 'POWERITD'

      ! scratch space
      real  TA1 (LPX1*LPY1*LPZ1*LELV), TA2 (LPX1*LPY1*LPZ1*LELV),
     $     TA3 (LPX1*LPY1*LPZ1*LELV), TAT (LPX1*LPY1*LPZ1*LELT)
      COMMON /SCRUZ/ TA1, TA2, TA3, TAT

      ! local variables
      integer itmp
      real lnorm, grth_old, ltim

      ! functions
      real dnekclock, cht_glsc2_wt
!-----------------------------------------------------------------------
      ! timing
      ltim=dnekclock()

      ! normalise vector
      lnorm = cht_glsc2_wt(VXP,VYP,VZP,TP,VXP,VYP,VZP,TP,BM1)
      lnorm = sqrt(pwi_l2n/lnorm)
      call cht_opcmult (VXP,VYP,VZP,TP,lnorm)

      ! make sure the velocity and temperature fields are continuous at
      ! element faces and edges
      call tst_dssum

      ! compare current and prevoius growth rate
      grth_old = pwi_grw
      pwi_grw = 1.0/lnorm
      grth_old = pwi_grw - grth_old

      ! get L2 norm of the update
      call cht_opsub3 (TA1,TA2,TA3,TAT,pwi_vx,pwi_vy,pwi_vz,pwi_t,
     $     VXP,VYP,VZP,TP)
      lnorm = cht_glsc2_wt(TA1,TA2,TA3,TAT,TA1,TA2,TA3,TAT,BM1)
      lnorm = sqrt(lnorm)

      ! log stamp
      call mntr_log(pwi_id,lp_prd,'POWER ITERATIONS: convergence')
      call mntr_logr(pwi_id,lp_prd,'||V-V_old|| = ',lnorm)
      call mntr_logr(pwi_id,lp_prd,'Growth ',pwi_grw)

      itmp = 0
      if (IFHEAT) itmp = 1

      !write down current field
      call outpost2(VXP,VYP,VZP,PRP,TP,itmp,'PWI')

      ! write down field difference
      call outpost2(TA1,TA2,TA3,PRP,TAT,itmp,'VDF')

      ! check convergence
      if(lnorm.lt.tst_tol.and.grth_old.lt.tst_tol) then
         call mntr_log(pwi_id,lp_prd,'Reached stopping criteria')
         ! mark the last step
         LASTEP = 1
      else
         ! save current vector and restart stepper
         call cht_opcopy (pwi_vx,pwi_vy,pwi_vz,pwi_t,VXP,VYP,VZP,TP)
      endif

      ! save checkpoint
      if (LASTEP.eq.1.or.tst_cmax.eq.tst_vstep) then
         call stepper_write()

         ! mark the last step
         LASTEP = 1
      endif

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pwi_tmr_evl_id,1,ltim)

      if (LASTEP.eq.1) then
         ! final log stamp
         call mntr_log(pwi_id,lp_prd,'POWER ITERATIONS finalised')
         call mntr_logr(pwi_id,lp_prd,'||V-V_old|| = ',lnorm)
         call mntr_logr(pwi_id,lp_prd,'Growth ',pwi_grw)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Read restart files
!! @ingroup powerit
!! @param[in]  set_in  restart set number
      subroutine stepper_read(set_in)
      implicit none

      include 'SIZE'            ! NIO
      include 'TSTEP'           ! TIME, LASTEP, NSTEPS
      include 'INPUT'           ! IFMVBD, IFREGUO
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'
      include 'POWERITD'

      ! argument list
      integer set_in

      ! local variables
      integer ifile, step_cnt, fnum
      character*132 fname(CHKPTNFMAX)
      logical ifreguol
!-----------------------------------------------------------------------
      ! no regular mesh
      ifreguol= IFREGUO
      IFREGUO = .false.

      call mntr_log(pwi_id,lp_inf,'Reading checkpoint snapshot')

      ! initialise I/O data
      call io_init

      ! get set of file names in the snapshot
      ifile = 1
      call chkpt_set_name(fname, fnum, set_in, ifile)

      ! read files
      call chkpt_restart_read(fname, fnum)

      ! put parameters back
      IFREGUO = ifreguol

      return
      end subroutine
!=======================================================================
!> @brief Write restart files
!! @ingroup powerit
      subroutine stepper_write
      implicit none

      include 'SIZE'            ! NIO
      include 'TSTEP'           ! TIME, LASTEP, NSTEPS
      include 'INPUT'           ! IFMVBD, IFREGUO
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'CHKPTMSTPD'
      include 'POWERITD'

      ! local variables
      integer ifile, step_cnt, set_out, fnum
      character*132 fname(CHKPTNFMAX)
      logical ifcoord
      logical ifreguol
!-----------------------------------------------------------------------
      ! no regular mesh
      ifreguol= IFREGUO
      IFREGUO = .false.

      call mntr_log(pwi_id,lp_inf,'Writing checkpoint snapshot')

      ! initialise I/O data
      call io_init

      ! get set of file names in the snapshot
      ifile = 1
      call chkpt_get_fset(step_cnt, set_out)
      call chkpt_set_name(fname, fnum, set_out, ifile)

      ifcoord = .true.
      ! write down files
      call chkpt_restart_write(fname, fnum, ifcoord)

      ! put parameters back
      IFREGUO = ifreguol

      return
      end subroutine
!=======================================================================

