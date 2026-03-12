c-----------------------------------------------------------------------
      subroutine nek_init(intracomm)

      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'ZPER'
c
      include 'OPCTR'
      include 'CTIMER'

C     used scratch arrays
C     NOTE: no initial declaration needed. Linker will take 
c           care about the size of the CBs automatically
c
c      COMMON /CTMP1/ DUMMY1(LCTMP1)
c      COMMON /CTMP0/ DUMMY0(LCTMP0)
c
c      COMMON /SCRNS/ DUMMY2(LX1,LY1,LZ1,LELT,7)
c      COMMON /SCRUZ/ DUMMY3(LX1,LY1,LZ1,LELT,4)
c      COMMON /SCREV/ DUMMY4(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRVH/ DUMMY5(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRMG/ DUMMY6(LX1,LY1,LZ1,LELT,4)
c      COMMON /SCRCH/ DUMMY7(LX1,LY1,LZ1,LELT,2)
c      COMMON /SCRSF/ DUMMY8(LX1,LY1,LZ1,LELT,3)
c      COMMON /SCRCG/ DUMM10(LX1,LY1,LZ1,LELT,1)
  
      real kwave2
      real*8 t0, tpp

      logical ifemati,ifsync_

      call get_session_info(intracomm)

      etimes = dnekclock()
      istep  = 0
      tpp    = 0.0

      call opcount(1)

      call initdim         ! Initialize / set default values.
      call initdat
      call files

      etime = dnekclock()
      call readat          ! Read .rea +map file
      etims0 = dnekclock_sync()
      if (nio.eq.0) then
         write(6,12) 'nelgt/nelgv/lelt:',nelgt,nelgv,lelt
         write(6,12) 'lx1  /lx2  /lx3 :',lx1,lx2,lx3
         write(6,'(A,g13.5,A,/)')  ' done :: read .rea file ',
     &                             etims0-etime,' sec'
 12      format(1X,A,4I12,/,/)
      endif 

      ifsync_ = ifsync
      ifsync = .true.

      call setvar          ! Initialize most variables

#ifdef MOAB
      if (ifmoab) call nekMOAB_bcs  !   Map BCs
#endif

      instep=1             ! Check for zero steps
      if (nsteps.eq.0 .and. fintim.eq.0.) instep=0

      igeom = 2
      call setup_topo      ! Setup domain topology  

      call genwz           ! Compute GLL points, weights, etc.

      call io_init         ! Initalize io unit

      if(nio.eq.0) write(6,*) 'call usrdat'
      call usrdat
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat' 

      call gengeom(igeom)  ! Generate geometry, after usrdat 

      if (ifmvbd) call setup_mesh_dssum ! Set mesh dssum (needs geom)

      if(nio.eq.0) write(6,*) 'call usrdat2'
      call usrdat2
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2' 

      call geom_reset(1)    ! recompute Jacobians, etc.
      call vrdsmsh          ! verify mesh topology

      call setlog  ! Initalize logical flags

      call bcmask  ! Set BC masks for Dirichlet boundaries.

      if (fintim.ne.0.0.or.nsteps.ne.0) 
     $   call geneig(igeom) ! eigvals for tolerances

      call vrdsmsh     !     Verify mesh topology

      call dg_setup    !     Setup DG, if dg flag is set.

      if (ifflow.and.(fintim.ne.0.or.nsteps.ne.0)) then    ! Pressure solver 
         call estrat                                       ! initialization.
         if (iftran.and.solver_type.eq.'itr') then         ! Uses SOLN space 
            call set_overlap                               ! as scratch!
         elseif (solver_type.eq.'fdm'.or.solver_type.eq.'pdm')then
            ifemati = .true.
            kwave2  = 0.0
            if (ifsplit) ifemati = .false.
            call gfdm_init(nx2,ny2,nz2,ifemati,kwave2)
         elseif (solver_type.eq.'25D') then
            call g25d_init
         endif
      endif

      if(ifcvode) call cv_setsize

      if(nio.eq.0) write(6,*) 'call usrdat3'
      call usrdat3
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'

#ifdef CMTNEK
        call nek_cmt_init
        if (nio.eq.0) write(6,*)'Initialized DG machinery'
#endif

      call setics      !     Set initial conditions 
      call setprop     !     Compute field properties

      if (instep.ne.0) then !USRCHK
        if(nio.eq.0) write(6,*) 'call userchk'
         if (ifneknek) call userchk_set_xfer
         if (ifneknek) call bcopy
         if (ifneknek) call chk_outflow
         call userchk
         if(nio.eq.0) write(6,'(A,/)') ' done :: userchk' 
      endif

      if (ifcvode .and. nsteps.gt.0) call cv_init

      call comment
      call sstest (isss) 

      call dofcnt

      jp = 0            ! Set perturbation field count to 0 for baseline flow

      call in_situ_init()

      call time00       !     Initalize timers to ZERO
      call opcount(2)

      etims0 = dnekclock_sync()
      if (nio.eq.0) then
        write (6,*) ' '
        if (time.ne.0.0) write (6,'(a,e14.7)') ' Initial time:',time
        write (6,'(a,g13.5,a)') 
     &              ' Initialization successfully completed ',
     &              etims0-etimes, ' sec'
      endif

      ifsync = ifsync_ ! restore initial value

      return
      end
c-----------------------------------------------------------------------
      subroutine nek_solve

      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'CTIMER'

      real*4 papi_mflops
      integer*8 papi_flops

      call nekgsync()

      if (instep.eq.0) then
        if(nid.eq.0) write(6,'(/,A,/,A,/)') 
     &     ' nsteps=0 -> skip time loop',
     &     ' running solver in post processing mode'
      else
        if(nio.eq.0) write(6,'(/,A,/)') 'Starting time loop ...'
      endif

      isyc  = 0
      if(ifsync) isyc=1
      itime = 0
#ifdef TIMER
      itime = 1
#endif
      call nek_comm_settings(isyc,itime)
      call nek_comm_startstat()

      istep  = 0
      msteps = 1

      do kstep=1,nsteps,msteps
         call nek__multi_advance(kstep,msteps)
         call userchk
         call prepost (.false.,'his')
         call in_situ_check()
         if (lastep .eq. 1) goto 1001
      enddo
 1001 lastep=1


      call nek_comm_settings(isyc,0)

      call comment

c     check for post-processing mode
      if (instep.eq.0) then
         nsteps=0
         istep=0
         if(nio.eq.0) write(6,*) 'call userchk'
         call userchk
         if(nio.eq.0) write(6,*) 'done :: userchk'
         call prepost (.true.,'his')
      else
         if (nio.eq.0) write(6,'(/,A,/)') 
     $      'end of time-step loop' 
      endif


      RETURN
      END

c-----------------------------------------------------------------------
      subroutine nek_advance

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common /cgeom/ igeom


      call nekgsync

      call setup_convect(2) ! Save conv vel

      if (iftran) call settime
      if (ifmhd ) call cfl_check
      call setsolv
      call comment

#ifdef CMTNEK
      if (nio.eq.0.and.istep.le.1) write(6,*) 'CMT branch active'
      call cmt_nek_advance
      return
#endif


      if (ifsplit) then   ! PN/PN formulation


         do igeom=1,ngeom


         ! within cvode we use the lagged wx for 
         ! extrapolation, that's why we have to call it before gengeom 
         if (ifheat .and. ifcvode) call heat_cvode (igeom)   

         if (ifgeom) then
               call gengeom (igeom)
               call geneig  (igeom)
         endif

         if (ifheat)               call heat (igeom)

         if (igeom.eq.2) then  
                                   call setprop
            if (iflomach)          call qthermal(.true.)
         endif

         if (ifflow)               call fluid         (igeom)
         if (ifmvbd)               call meshv         (igeom)
         if (param(103).gt.0)      call q_filter      (param(103))
         enddo

      else                ! PN-2/PN-2 formulation

         call setprop
         do igeom=1,ngeom

            if (igeom.gt.2) call userchk_set_xfer

            if (ifgeom) then
               call gengeom (igeom)
               call geneig  (igeom)
            endif

            if (ifneknekm.and.igeom.eq.2) call multimesh_create

            if (ifmhd) then
               if (ifheat)      call heat     (igeom)
                                call induct   (igeom)
            elseif (ifpert) then
               if (ifbase.and.ifheat)  call heat          (igeom)
               if (ifbase.and.ifflow)  call fluid         (igeom)
               if (ifflow)             call fluidp        (igeom)
               if (ifheat)             call heatp         (igeom)
            else  ! std. nek case
               if (ifheat)             call heat          (igeom)
               if (ifflow)             call fluid         (igeom)
               if (ifmvbd)             call meshv         (igeom)
            endif

            if (igeom.eq.ngeom.and.param(103).gt.0) 
     $          call q_filter(param(103))
         enddo
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine nek_end

      include 'SIZE'
      include 'TSTEP'
      include 'PARALLEL'
      include 'OPCTR'

      if(instep.ne.0)  call runstat
      if(xxth(1).gt.0) call crs_stats(xxth(1))

   
      call in_situ_end()
      return
      end
c-----------------------------------------------------------------------
      subroutine nek__multi_advance(kstep,msteps)

      include 'SIZE'
      include 'TOTAL'

      do i=1,msteps
         istep = istep+i
         call nek_advance

         if (ifneknek) call userchk_set_xfer
         if (ifneknek) call bcopy
         if (ifneknek) call chk_outflow

      enddo

      return
      end
c-----------------------------------------------------------------------
