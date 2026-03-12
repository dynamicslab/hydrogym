!> @file tstepper.f
!! @ingroup tstepper
!! @brief Set of subroutines to use time steppers for e.g. power
!!    iterations or solution of eigenvalue problem with Arnoldi algorithm
!! @author Adam Peplinski
!! @date Mar 7, 2016
!=======================================================================
!> @brief Register time stepper module
!! @ingroup tstepper
!! @note This routine should be called in frame_usr_register
      subroutine tst_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'TSTEPPERD'

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
      call mntr_mod_is_name_reg(lpmid,tst_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(tst_name)//'] already registered')
         return
      endif

      ! check if conjugated heat transfer module was registered
      call mntr_mod_is_name_reg(lpmid,'CONJHT')
      if (lpmid.gt.0)  then
         call mntr_warn(lpmid,
     $        'module ['//'CONJHT'//'] already registered')
      else
         call cht_register()
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'parent module ['//'FRAME'//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(tst_id,lpmid,tst_name,
     $      'Time stepper')

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      ! total time
      call mntr_tmr_reg(tst_tmr_tot_id,lpmid,tst_id,
     $     'TST_TOT','Time stepper total time',.false.)
      lpmid = tst_tmr_tot_id
      ! initialisation
      call mntr_tmr_reg(tst_tmr_ini_id,lpmid,tst_id,
     $     'TST_INI','Time stepper initialisation time',.true.)
      ! submodule operation
      call mntr_tmr_reg(tst_tmr_evl_id,lpmid,tst_id,
     $     'TST_EVL','Time stepper evolution time',.true.)

      ! register and set active section
      call rprm_sec_reg(tst_sec_id,tst_id,'_'//adjustl(tst_name),
     $     'Runtime paramere section for time stepper module')
      call rprm_sec_set_act(.true.,tst_sec_id)

      ! register parameters
      call rprm_rp_reg(tst_mode_id,tst_sec_id,'MODE',
     $     'Simulation mode',rpar_str,10,0.0,.false.,'DIR')

      call rprm_rp_reg(tst_step_id,tst_sec_id,'STEPS',
     $     'Length of stepper phase',rpar_int,40,0.0,.false.,' ')

      call rprm_rp_reg(tst_cmax_id,tst_sec_id,'MAXCYC',
     $     'Max number of stepper cycles',rpar_int,10,0.0,.false.,' ')

      call rprm_rp_reg(tst_tol_id,tst_sec_id,'TOL',
     $    'Convergence threshold',rpar_real,0,1.0d-6,.false.,' ')

      ! place for submodule registration
      ! register arnoldi or power iterations
      call stepper_register()

      ! set initialisation flag
      tst_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tst_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise time stepper module
!! @ingroup tstepper
!! @note This routine should be called in frame_usr_init
      subroutine tst_init()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'TSTEP'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'ADJOINT'
      include 'TSTEPPERD'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      ! functions
      real dnekclock, cht_glsc2_wt
      logical cht_is_initialised
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (tst_ifinit) then
         call mntr_warn(tst_id,
     $        'module ['//trim(tst_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! intialise conjugated heat transfer
      if (.not.cht_is_initialised()) call cht_init

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,tst_mode_id,rpar_str)
      if (trim(ctmp).eq.'DIR') then
        tst_mode = 1
      else if (trim(ctmp).eq.'ADJ') then
        tst_mode = 2
      else if (trim(ctmp).eq.'OIC') then
        tst_mode = 3
      else
        call mntr_abort(tst_id,
     $        'wrong simulation mode; possible values: DIR, ADJ, OIC')
      endif

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,tst_step_id,rpar_int)
      tst_step = itmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,tst_cmax_id,rpar_int)
      tst_cmax = itmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,tst_tol_id,rpar_real)
      tst_tol = rtmp

      ! check simulation parameters
      if (.not.IFTRAN) call mntr_abort(tst_id,
     $   'time stepper requres transient simulation; IFTRAN=.T.')

      if (NSTEPS.eq.0) call mntr_abort(tst_id,
     $   'time stepper requres NSTEPS>0')

      if (PARAM(12).ge.0) call mntr_abort(tst_id,
     $   'time stepper assumes constant dt')

      if (.not.IFPERT) call mntr_abort(tst_id,
     $   'time stepper has to be run in perturbation mode')

      if (IFBASE)  call mntr_abort(tst_id,
     $   'time stepper assumes constatnt base flow')

      if (NPERT.ne.1) call mntr_abort(tst_id,
     $   'time stepper requires NPERT=1')

      ! initialise cycle counters
      tst_istep = 0
      tst_vstep = 0

      ! vector length
      tst_nv  = NX1*NY1*NZ1*NELV ! velocity single component
      if (IFHEAT) then        !temperature
         tst_nt  = NX1*NY1*NZ1*NELT
      else
         tst_nt  = 0
      endif
      tst_np  = NX2*NY2*NZ2*NELV ! presure

      ! place for submodule initialisation
      ! arnoldi or power iterations
      call stepper_init

      ! zero presure
      call rzero(PRP,tst_np)

      ! set initial time
      TIME=0.0

      ! make sure NSTEPS is bigger than the possible number of iterations
      ! in time stepper phase; multiplication by 2 for OIC
      NSTEPS = max(NSTEPS,tst_step*tst_cmax*2+10)

      IFADJ = .FALSE.
      if (tst_mode.eq.2) then
         ! Is it adjoint mode
         IFADJ = .TRUE.
      elseif  (tst_mode.eq.3) then
         ! If it is optimal initial condition save initial L2 norm
         tst_L2ini = cht_glsc2_wt(VXP,VYP,VZP,TP,VXP,VYP,VZP,TP,BM1)

         if (tst_L2ini.eq.0.0) call mntr_abort(tst_id,
     $   'tst_init, tst_L2ini = 0')

         call mntr_log(tst_id,lp_prd,
     $  'Optimal initial condition; direct phase start')
      endif

      ! set cpfld for conjugated heat transfer
      if (IFHEAT) call cht_cpfld_set

!     should be the first step of every cycle performed with Uzawa
!     turned on?
!         IFUZAWA = tst_ifuz

      ! everything is initialised
      tst_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tst_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup tstepper
!! @return tst_is_initialised
      logical function tst_is_initialised()
      implicit none

      include 'SIZE'
      include 'TSTEPPERD'
!-----------------------------------------------------------------------
      tst_is_initialised = tst_ifinit

      return
      end function
!=======================================================================
!> @brief Control time stepper after every nek5000 step and call suitable
!! stepper_vsolve of required submodule
!! @ingroup tstepper
      subroutine tst_solve()
      implicit none

      include 'SIZE'            ! NIO
      include 'TSTEP'           ! ISTEP, TIME
      include 'INPUT'           ! IFHEAT, IF3D, IFUZAWA ????????
      include 'MASS'            ! BM1
      include 'SOLN'            ! V[XYZ]P, PRP, TP, VMULT, V?MASK
      include 'ADJOINT'         ! IFADJ
      include 'FRAMELP'
      include 'TSTEPPERD'

      ! global comunication in nekton
      integer NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL
      common /nekmpi/ NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL

      ! local variables
      real grw         ! growth rate
      real ltim        ! timing

      ! functions
      real dnekclock, cht_glsc2_wt
!-----------------------------------------------------------------------
      if (ISTEP.eq.0) return

      ! timing
      ltim = dnekclock()

!     turn off Uzawa after first step
!         IFUZAWA = .FALSE.

      ! step counting
      tst_istep = tst_istep + 1

      ! stepper phase end
      if (mod(tst_istep,tst_step).eq.0) then
         ! check for the calculation mode
         if (tst_mode.eq.3.and.(.not.IFADJ)) then
            ! optimal initial condition

            call mntr_log(tst_id,lp_prd,
     $      'Optimal initial condition; adjoint phase start')

            IFADJ = .TRUE.

            ! itaration count
            tst_istep = 0

!           ! should be the first step of every cycle performed with Uzawa
!     turned on?
!               IFUZAWA = tst_ifuz

            ! set time and iteration number
            TIME=0.0
            ISTEP=0

            ! get L2 norm after direct phase
            tst_L2dir = cht_glsc2_wt(VXP,VYP,VZP,TP,
     $         VXP,VYP,VZP,TP,BM1)
             ! normalise vector
             grw = sqrt(tst_L2ini/tst_L2dir)
             call cht_opcmult (VXP,VYP,VZP,TP,grw)

            ! zero presure
            call rzero(PRP,tst_np)

            ! set cpfld for conjugated heat transfer
               if (IFHEAT) call cht_cpfld_set
         else
            !stepper phase counting
            tst_istep = 0
            tst_vstep = tst_vstep +1

            call mntr_logi(tst_id,lp_prd,'Finished stepper phase:',
     $           tst_vstep)

            if (tst_mode.eq.3) then
               ! optimal initial condition
               call mntr_log(tst_id,lp_prd,
     $         'Optimal initial condition; rescaling solution')

               ! get L2 norm after direct phase
               tst_L2adj = cht_glsc2_wt(VXP,VYP,VZP,TP,
     $                 VXP,VYP,VZP,TP,BM1)
               ! normalise vector after whole cycle
               grw = sqrt(tst_L2dir/tst_L2ini)! add direct growth
               call cht_opcmult (VXP,VYP,VZP,TP,grw)

            endif

            ! run vector solver (arpack, power iteration)
            call stepper_vsolve

            if (LASTEP.ne.1) then
               ! stepper restart;
               ! set time and iteration number
               TIME=0.0
               ISTEP=0

!     should be the first step of every cycle performed with Uzawa 
!     turned on?
!               IFUZAWA = tst_ifuz

               ! zero presure
               call rzero(PRP,tst_np)

               if (tst_mode.eq.3) then
                  ! optimal initial condition
                  call mntr_log(tst_id,lp_prd,
     $            'Optimal initial condition; direct phase start')

                  IFADJ = .FALSE.

                  ! get initial L2 norm
                  tst_L2ini = cht_glsc2_wt(VXP,VYP,VZP,TP,
     $                    VXP,VYP,VZP,TP,BM1)
                  ! set cpfld for conjugated heat transfer
                  if (IFHEAT) call cht_cpfld_set

               endif
            endif

         endif               ! tst_mode.eq.3.and.(.not.IFADJ)
      endif                  ! mod(tst_istep,tst_step).eq.0

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tst_tmr_evl_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Average velocity and temperature at element faces.
!! @ingroup tstepper
      subroutine tst_dssum
      implicit none

      include 'SIZE'            ! N[XYZ]1
      include 'INPUT'           ! IFHEAT
      include 'SOLN'            ! V[XYZ]P, TP, [VT]MULT
      include 'TSTEP'           ! IFIELD
      include 'TSTEPPERD'       ! tst_nt

      ! local variables
      integer ifield_tmp
!-----------------------------------------------------------------------
      ! make sure the velocity and temperature fields are continuous at
      ! element faces and edges
      ifield_tmp = IFIELD
      IFIELD = 1
#ifdef AMR
      call amr_oph1_proj(vxp,vyp,vzp,nx1,ny1,nz1,nelv)
#else
      call opdssum(VXP,VYP,VZP)
      call opcolv (VXP,VYP,VZP,VMULT)
#endif

      if(IFHEAT) then
         IFIELD = 2
#ifdef AMR
         call h1_proj(tp,nx1,ny1,nz1)
#else
         call dssum(TP,NX1,NY1,NZ1)
         call col2 (TP,TMULT,tst_nt)
#endif
      endif
      IFIELD = ifield_tmp

      return
      end subroutine
!=======================================================================
