!> @file arn_arp.f
!! @ingroup arn_arp
!! @brief Set of subroutines to solve eigenvalue problem with Arnoldi
!!   algorithm using PARPACK/ARPACK
!! @warning There is no restart option for serial ARPACK version. It is
!!   supported by parallel PARPACK only.
!! @author Adam Peplinski
!! @date Mar 7, 2016
!
!
! To define ARPACK mode: direct or inverse:
! Notice that simulation with temperature or passive scalars has to be
! performed in inverse mode due to speciffic inner product.
!
! Direct eigenvalue problem A*x = lambda*x
!#define ARPACK_DIRECT
! Generalized (inverse) eigenvalue problem A*x = lambda*B*x
#undef ARPACK_DIRECT
!=======================================================================
!> @brief Register Arnoldi ARPACK module
!! @ingroup arn_arp
!! @note This interface is called by @ref tst_register
      subroutine stepper_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

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
      call mntr_mod_is_name_reg(lpmid,arna_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(arna_name)//'] already registered')
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
      call mntr_mod_reg(arna_id,lpmid,arna_name,
     $      'Arnoldi ARPACK spectra calculation')

      ! register timers
      ! initialisation
      call mntr_tmr_reg(arna_tmr_ini_id,tst_tmr_ini_id,arna_id,
     $     'ARNA_INI','Arnoldi ARPACK initialisation time',.true.)
      ! submodule operation
      call mntr_tmr_reg(arna_tmr_evl_id,tst_tmr_evl_id,arna_id,
     $     'ARNA_EVL','Arnoldi ARPACK evolution time',.true.)

      ! register and set active section
      call rprm_sec_reg(arna_sec_id,arna_id,'_'//adjustl(arna_name),
     $     'Runtime paramere section for Arnoldi ARPACK module')
      call rprm_sec_set_act(.true.,arna_sec_id)

      ! register parameters
      call rprm_rp_reg(arna_nkrl_id,arna_sec_id,'NKRL',
     $     'Krylov space size',rpar_int,50,0.0,.false.,' ')

      call rprm_rp_reg(arna_negv_id,arna_sec_id,'NEGV',
     $     'Number of eigenvalues',rpar_int,10,0.0,.false.,' ')

      ! set initialisation flag
      arna_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(arna_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise Arnoldi ARPACK module
!! @ingroup arn_arp
!! @note This interface is called by @ref tst_init
      subroutine stepper_init()
      implicit none

      include 'SIZE'            ! NIO, NDIM, NPERT
      include 'TSTEP'           ! NSTEPS
      include 'INPUT'           ! IF3D, IFHEAT
      include 'SOLN'            ! V?MASK, TMASK, V[XYZ]P, TP
      include 'FRAMELP'
      include 'CHKPOINTD'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! ARPACK include file
      INCLUDE 'debug.h'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (arna_ifinit) then
         call mntr_warn(arna_id,
     $        'module ['//trim(arna_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,arna_nkrl_id,rpar_int)
      arna_nkrl = itmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,arna_negv_id,rpar_int)
      arna_negv = itmp

      ! get restart options
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_ifrst_id,rpar_log)
      arna_ifrst = ltmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,chpt_fnum_id,rpar_int)
      arna_fnum = itmp

      ! check simulation parameters
#ifdef ARPACK_DIRECT
      ! standard eigenvalue problem A*x = lambda*x
      if(IFHEAT) call mntr_abort(arna_id,
     $   'IFHEAT requires #undef ARPACK_DIRECT')
#endif

      if (arna_nkrl.gt.arna_lkrl) call mntr_abort(arna_id,
     $   'arna_nkrl bigger than arna_lkrl')

      if (arna_negv.ge.(arna_nkrl/2)) call mntr_abort(arna_id,
     $   'arna_negv > arna_nkrl/2')

      ! make sure NSTEPS is bigger than the possible number of iteration in arnoldi
      ! multiplication by 2 for OIC
      NSTEPS = max(NSTEPS,tst_step*arna_nkrl*tst_cmax*2+10)

      ! related to restart
      nparp = 0
      ncarp = 0
      rnmarp= 0.0

      ! initialise ARPACK parameters
#ifdef ARPACK_DIRECT
      ! direct eigenvalue problem A*x = lambda*x
      bmatarp='I'
#else
      ! generalised eigenvalue problem A*x = lambda*B*x
      bmatarp='G'
#endif
      ! eigenvalues of largest magnitude
      whicharp='LM'

      call izero(iparp,11)
      call izero(ipntarp,14)
      ! exact shifts with respect to the current Hessenberg matrix
      iparp(1)=1
      ! maximum number of Arnoldi update iterations allowed
      iparp(3)=tst_cmax
#ifdef ARPACK_DIRECT
      ! A*x = lambda*x
      iparp(7)=1
#else
      ! A*x = lambda*M*x, M symmetric positive definite; bmatarp='G'
      iparp(7)=2
#endif
      ! used size of workla
      nwlarp = (3*arna_nkrl+6)*arna_nkrl

      ! user supplied initial conditions
      infarp=1

      ! get eigenvectors
      rvarp=.true.
      ! compute Ritz vectors
      howarp='A'
      ! select should be specifird for howarp='S'

      ! no shift
      sigarp(1) = 0.0
      sigarp(2) = 0.0

      ! vector lengths
      ! single vector length in Krylov space
      ! velocity
      arna_ns = tst_nv*NDIM
      ! temperature
      if(IFHEAT) then
         arna_ns = arna_ns + tst_nt
      endif
      if (arna_ns.gt.arna_ls) call mntr_abort(arna_id,
     $   'arna_ns too big; arna_ns > arna_ls')

      ! initialise arrays
      call rzero(workda,wddima)
      call rzero(workla,wldima)
      call rzero(workea,wedima)
      call rzero(vbasea,arna_ls*arna_lkrl)
      call rzero(resida,arna_ls)
      call rzero(driarp,arna_lkrl*4)

      ! info level from ARPACK
      ndigit = -3
      logfil = 6
      mngets = 0
      mnaitr = 2
      mnapps = 0
      mnaupd = 2
      mnaup2 = 2
      mneupd = 0

      ! restart
      if (arna_ifrst) then
         ! read checkpoint
         call arn_rst_read
      else
         ! if no restatrt fill RESIDA with initial conditions
         ! V?MASK removes points at the wall and inflow
#ifdef ARPACK_DIRECT
         ! A*x = lambda*x
         ! velocity
         call col3(resida(1),VXP,V1MASK,tst_nv)
         call col3(resida(1+tst_nv),VYP,V2MASK,tst_nv)
         if (IF3D) call col3(resida(1+2*tst_nv),VZP,V3MASK,tst_nv)
         ! no temperature here
#else
         ! A*x = lambda*M*x
         ! velocity
         call copy(resida(1),VXP,tst_nv)
         call copy(resida(1+tst_nv),VYP,tst_nv)
         if (IF3D) call copy(resida(1+2*tst_nv),VZP,tst_nv)
         ! temperature
         if(IFHEAT) call copy(resida(1+NDIM*tst_nv),TP,tst_nt)
#endif

         ! initialise rest of variables
         ! firts call
         idoarp=0
      endif

      ! ARPACK interface
      call arn_naupd

      ! we should start stepper here
      if (idoarp.ne.-1.and.idoarp.ne.1) then
         write(ctmp,*) idoarp
         call mntr_abort(arna_id,
     $   'stepper_init; error with arn_naupd, ido = '//trim(ctmp))
      endif

      ! print info
      call mntr_log(arna_id,lp_prd,'ARPACK initialised')
      call mntr_log(arna_id,lp_prd,'Parameters:')
      call mntr_log(arna_id,lp_prd,'BMAT = '//trim(bmatarp))
      call mntr_log(arna_id,lp_prd,'WHICH = '//trim(whicharp))
      call mntr_logr(arna_id,lp_prd,'TOL = ',tst_tol)
      call mntr_logi(arna_id,lp_prd,'NEV = ',arna_negv)
      call mntr_logi(arna_id,lp_prd,'NCV = ',arna_nkrl)
      call mntr_logi(arna_id,lp_prd,'IPARAM(1) = ',iparp(1))
      call mntr_logi(arna_id,lp_prd,'IPARAM(3) = ',iparp(3))
      call mntr_logi(arna_id,lp_prd,'IPARAM(7) = ',iparp(7))
      call mntr_logl(arna_id,lp_prd,'RVEC = ',rvarp)
      call mntr_log(arna_id,lp_prd,'HOWMNY = '//trim(howarp))

      ! everything is initialised
      arna_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(arna_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup arn_arp
!! @return stepper_is_initialised
      logical function stepper_is_initialised()
      implicit none

      include 'SIZE'
      include 'ARN_ARPD'
!-----------------------------------------------------------------------
      stepper_is_initialised = arna_ifinit

      return
      end function
!=======================================================================
!> @brief Create Krylov space, get Ritz values and restart
!!  stepper phase.
!! @ingroup arn_arp
!! @note This interface is called by @ref tst_solve
      subroutine stepper_vsolve
      implicit none

      include 'SIZE'            ! NIO
      include 'INPUT'           ! IFHEAT, IF3D
      include 'SOLN'            ! V[XYZ]P, TP, V?MASK
      include 'MASS'            ! BM1
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! local variables
      real  ltim        ! timing
      character(20) str

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim=dnekclock()

      ! fill work array with velocity
      ! V?MASK removes points at the boundary
#ifdef ARPACK_DIRECT
      ! A*x = lambda*x
      ! velocity
      call col3(workda(ipntarp(2)),VXP,V1MASK,tst_nv)
      call col3(workda(ipntarp(2)+tst_nv),VYP,V2MASK,tst_nv)
      if (IF3D) call col3(workda(ipntarp(2)+2*tst_nv),VZP,V3MASK,tst_nv)
      ! no temperature here
#else
      ! velocity
      ! A*x = lambda*M*x
      call copy(workda(ipntarp(2)),VXP,tst_nv)
      call copy(workda(ipntarp(2)+tst_nv),VYP,tst_nv)
      if (IF3D) call copy(workda(ipntarp(2)+2*tst_nv),VZP,tst_nv)
      ! temperature
      if(IFHEAT) call copy(workda(ipntarp(2)+NDIM*tst_nv),TP,tst_nt)
      ! this may be not necessary, but ARPACK manual is not clear about it
      !call col3(workda(ipntarp(1)),VXP,BM1,tst_nv)
      !call col3(workda(ipntarp(1)+tst_nv),VYP,BM1,tst_nv)
      !if (IF3D) call col3(workda(ipntarp(1)+2*tst_nv),VZP,BM1,tst_nv)
#endif

      ! ARPACK interface
      call arn_naupd

      if (idoarp.eq.-2) then
         ! checkpoint
         call arn_rst_save
      elseif (idoarp.eq.99) then
         ! finalise
         call arn_esolve
      elseif (idoarp.eq.-1.or.idoarp.eq.1) then
         ! stepper restart, nothing to do
      else
         write(str,*) idoarp
         call mntr_abort(arna_id,
     $    'stepper_vsolve; error with arn_naupd, ido = '//trim(str))
      endif

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(arna_tmr_evl_id,1,ltim)

      return
      end
!=======================================================================
!> @brief ARPACK postprocessing
!! @ingroup arn_arp
      subroutine arn_esolve
      implicit none

      include 'SIZE'            ! NIO, NID, LDIMT1
      include 'TSTEP'           ! ISTEP, DT, LASTEP
      include 'SOLN'            ! VX, VY, VZ, VMULT, V?MASK
      include 'INPUT'           ! IFXYO,IFPO,IFVO,IFTO,IFPSO,IF3D,IFHEAT
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! local variables
      integer il, iunit, ierror
      real dumm
      logical lifxyo, lifpo, lifvo, lifto, lifpso(LDIMT1)
      character(20) str

      ! global comunication in nekton
      integer NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL
      common /nekmpi/ NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL
!-----------------------------------------------------------------------
      if (idoarp.eq.99) then

         call mntr_logi(arna_id,lp_prd,
     $       'Postprocessing converged eigenvectors NV= ',iparp(5))

#ifdef MPI
         call pdneupd(NEKCOMM,rvarp,howarp,selarp,driarp,driarp(1,2),
     $     vbasea,arna_ls,sigarp(1),sigarp(2),workea,bmatarp,arna_ns,
     $     whicharp,arna_negv,tst_tol,resida,arna_nkrl,vbasea,
     $     arna_ls,iparp,ipntarp,workda,workla,nwlarp,ierrarp)
#else
         call dneupd(rvarp,howarp,selarp,driarp,driarp(1,2),
     $     vbasea,arna_ls,sigarp(1),sigarp(2),workea,bmatarp,arna_ns,
     $     whicharp,arna_negv,tst_tol,resida,arna_nkrl,vbasea,
     $     arna_ls,iparp,ipntarp,workda,workla,nwlarp,ierrarp)
#endif

         if (ierrarp.eq.0) then
            call mntr_log(arna_id,lp_prd,
     $       'Writing eigenvalues and eigenvectors')
            ierror=0
            ! open file
            if (NID.eq.0) then
               ! find free unit
               call io_file_freeid(iunit, ierror)
               if (ierror.eq.0) then
                  open (unit=iunit,file='eigenvalues.txt',
     $                 action='write', iostat=ierror)
                  write(unit=iunit,FMT=410,iostat=ierror)
 410              FORMAT(10x,'I',17x,'re(RITZ)',17x,'im(RITZ)',17x,
     $                 'ln|RITZ|',16x,'arg(RITZ)')
               endif
            endif
            ! error check
            call  mntr_check_abort(arna_id,ierror,
     $       'Error opening eigenvalue file.')

            ! integration time
            dumm = DT*tst_step
            dumm = 1.0/dumm

            ! copy and set output parameters
            lifxyo = IFXYO
            IFXYO = .TRUE.
            lifpo= IFPO
            IFPO = .false.
            lifvo= IFVO
            IFVO = .true.
            lifto= IFTO
            if (IFHEAT) then
               IFTO = .TRUE.
            else
               IFTO = .FALSE.
            endif
            do il=1,LDIMT1
               lifpso(il)= IFPSO(il)
               IFPSO(il) = .false.
            enddo

      ! We have to take into account storrage of imaginary and real
      ! parts of eigenvectors in arpack.
      ! The complex Ritz vector associated with the Ritz value
      ! with positive imaginary part is stored in two consecutive
      ! columns.  The first column holds the real part of the Ritz
      ! vector and the second column holds the imaginary part.  The
      ! Ritz vector associated with the Ritz value with negative
      ! imaginary part is simply the complex conjugate of the Ritz
      ! vector associated with the positive imaginary part.

            ierror=0
            do il=1,IPARP(5)
               !copy eigenvectors to perturbation variables
               call copy(VXP,vbasea(1,il),tst_nv)
               call copy(VYP,vbasea(1+tst_nv,il),tst_nv)
               if (IF3D) call copy(VZP,vbasea(1+2*tst_nv,il),tst_nv)
               if(IFHEAT) then
                  call copy(TP,vbasea(1+NDIM*tst_nv,il),tst_nt)
                  call outpost2(VXP,VYP,VZP,PRP,TP,1,'egv')
               else
                  call outpost2(VXP,VYP,VZP,PRP,TP,0,'egv')
               endif
               ! possible place to test error

               ! get growth rate; get eigenvalues of continuous operator
               driarp(il,3) = log(sqrt(driarp(il,1)**2+
     $              driarp(il,2)**2))*dumm
               driarp(il,4) = atan2(driarp(il,2),driarp(il,1))*dumm

               if (NID.eq.0)  write(unit=iunit,fmt=*,iostat=ierror)
     $           il,driarp(il,1),driarp(il,2),driarp(il,3),driarp(il,4)
            enddo
            ! error check
            call  mntr_check_abort(arna_id,ierror,
     $       'Error writing eigenvalue file.')

            ! put output variables back
            IFXYO = lifxyo
            IFPO = lifpo
            IFVO = lifvo
            IFTO = lifto
            do il=1,LDIMT1
               IFPSO(il) = lifpso(il)
            enddo

            ! close eigenvalue file
            if (NID.eq.0)  close(unit=iunit)

         else                   ! ierrarp
            write(str,*) ierrarp
            call  mntr_abort(arna_id,
     $       'arn_esolve; error with _neupd, info = '//trim(str))
         endif                  ! ierrarp

         ! finish run
         LASTEP=1
      endif

      return
      end
!=======================================================================
!> @brief Interface to pdnaupd
!! @ingroup arn_arp
      subroutine arn_naupd
      implicit none

      include 'SIZE'            ! NIO, NDIM, N[XYZ]1
      include 'INPUT'           ! IF3D, IFHEAT
      include 'SOLN'            ! V?MASK, TMASK, V[XYZ]P, TP
      include 'MASS'            ! BM1
      include 'FRAMELP'
      include 'TSTEPPERD'
      include 'ARN_ARPD'

      ! local variables
      character(20) str

      ! global comunication in nekton
      integer NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL
      common /nekmpi/ NIDD,NPP,NEKCOMM,NEKGROUP,NEKREAL
!-----------------------------------------------------------------------
#ifdef MPI
      call pdnaupd(NEKCOMM,idoarp,bmatarp,arna_ns,whicharp,arna_negv,
     $  tst_tol,resida,arna_nkrl,vbasea,arna_ls,iparp,ipntarp,workda,
     $  workla,nwlarp,infarp,nparp,rnmarp,ncarp)
#else
      call dnaupd(Idoarp,bmatarp,arna_ns,whicharp,arna_negv,
     $  tst_tol,resida,arna_nkrl,vbasea,arna_ls,iparp,ipntarp,workda,
     $  workla,nwlarp,infarp)
#endif

      ! error check
      if (infarp.lt.0) then
         write(str,*) infarp
         call  mntr_abort(arna_id,
     $       'arn_naupd; error with _naupd, info = '//trim(str))
      endif

      if (idoarp.eq.2) then
         do
            ! A*x = lambda*M*x
            ! multiply by weights and masks
            ! velocity
            call col3(workda(ipntarp(2)),BM1,V1MASK,tst_nv)
            call col3(workda(ipntarp(2)+tst_nv),BM1,V2MASK,tst_nv)
            if (IF3D) call col3(workda(ipntarp(2)+2*tst_nv),
     $           BM1,V3MASK,tst_nv)

            ! temperature
            if(IFHEAT) then
               call col3(workda(ipntarp(2)+NDIM*tst_nv),
     $              BM1,TMASK,tst_nt)

               !coefficients
               call cht_weight_fun (workda(ipntarp(2)),
     $              workda(ipntarp(2)+tst_nv),
     $              workda(ipntarp(2)+2*tst_nv),
     $              workda(ipntarp(2)+NDIM*tst_nv),1.0)
            endif

            call col2(workda(ipntarp(2)),workda(ipntarp(1)),arna_ns)

#ifdef MPI
            call pdnaupd(NEKCOMM,idoarp,bmatarp,arna_ns,whicharp,
     $        arna_negv,tst_tol,resida,arna_nkrl,vbasea,arna_ls,
     $        iparp,ipntarp,workda,workla,nwlarp,infarp,nparp,rnmarp,
     $        ncarp)
#else
            call dnaupd(idoarp,bmatarp,arna_ns,whicharp,arna_negv,
     $        tst_tol,resida,arna_nkrl,vbasea,arna_ls,iparp,ipntarp,
     $        workda,workla,nwlarp,infarp)
#endif

            ! error check
            if (infarp.lt.0) then
               write(str,*) infarp
               call  mntr_abort(arna_id,
     $  'arn_naupd; inner prod. error with _naupd, info = '//trim(str))
            endif
            if (idoarp.ne.2) exit
         enddo
      endif                     ! idoarp.eq.2

      ! restart stepper
      if (idoarp.eq.-1.or.idoarp.eq.1) then
         call mntr_log(arna_id,lp_prd,'Restarting stepper')

         ! move renormed data back to nekton
         ! velocity
         call copy(VXP,workda(ipntarp(1)),tst_nv)
         call copy(VYP,workda(ipntarp(1)+tst_nv),tst_nv)
         if (IF3D) call copy(VZP,workda(ipntarp(1)+2*tst_nv),tst_nv)
         ! temperature
         if(IFHEAT) call copy(TP,workda(ipntarp(1)+NDIM*tst_nv),tst_nt)

         ! make sure the velocity and temperature fields are continuous at
         ! element faces and edges
         call tst_dssum
      endif                     ! idoarp.eq.-1.or.idoarp.eq.1

      return
      end
!=======================================================================
