c-----------------------------------------------------------------------
      subroutine nek_init(comm)
c
      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
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

      integer comm
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
  
      common /rdump/ ntdump

      real kwave2
      logical ifemati

      real rtest
      integer itest
      integer*8 itest8
      character ctest
      logical ltest 

      common /c_is1/ glo_num(lx1 * ly1 * lz1, lelt)
      common /ivrtx/ vertex((2 ** ldim) * lelt)
      integer*8 glo_num, ngv
      integer vertex

      ! set word size for REAL
      wdsize = sizeof(rtest)
      ! set word size for INTEGER
      isize = sizeof(itest)
      ! set word size for INTEGER*8
      isize8 = sizeof(itest8) 
      ! set word size for LOGICAL
      lsize = sizeof(ltest) 
      ! set word size for CHARACTER
      csize = sizeof(ctest)

      call setupcomm(comm,newcomm,newcommg,'','')
      intracomm   = newcomm   ! within a session
      nekcomm     = newcomm
      iglobalcomm = newcommg  ! across all sessions
      call iniproc()

      if (nid.eq.nio) call printHeader

      etimes = dnekclock()
      istep  = 0

      call opcount(1)

      call initdim         ! Initialize / set default values.
      call initdat
      call files

      call readat          ! Read .rea +map file

      call setvar          ! Initialize most variables

      instep=1             ! Check for zero steps
      if (nsteps.eq.0 .and. fintim.eq.0.) instep=0

      igeom = 2
      call setup_topo      ! Setup domain topology  

      call genwz           ! Compute GLL points, weights, etc.

      if(nio.eq.0) write(6,*) 'call usrdat'
      call usrdat
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat' 

      call gengeom(igeom)  ! Generate geometry, after usrdat 

      if (ifmvbd) call setup_mesh_dssum ! Set mesh dssum (needs geom)

      if(nio.eq.0) write(6,*) 'call usrdat2'
      call usrdat2
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2' 

      call fix_geom
      call geom_reset(1)    ! recompute Jacobians, etc.

      call vrdsmsh          ! verify mesh topology
      call mesh_metrics     ! print some metrics

      call setlog(.true.)   ! Initalize logical flags

      if (ifneknekc) call neknek_setup

      call bcmask  ! Set BC masks for Dirichlet boundaries.

      if (fintim.ne.0.0 .or. nsteps.ne.0) 
     $   call geneig(igeom) ! eigvals for tolerances

      call dg_setup ! Setup DG, if dg flag is set.

      if (ifflow.and.iftran) then ! Init pressure solver 
         if (fintim.ne.0 .or. nsteps.ne.0) call prinit
      endif

      if(ifcvode) call cv_setsize

      if(nio.eq.0) write(6,*) 'call usrdat3'
      call usrdat3
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'

#ifdef CMTNEK
        call nek_cmt_init
#endif

      call setics
      call setprop

      if (instep.ne.0) then
         if (ifneknekc) call neknek_exchange
         if (ifneknekc) call chk_outflow

         if (nio.eq.0) write(6,*) 'call userchk'
         call userchk
         if(nio.eq.0) write(6,'(A,/)') ' done :: userchk' 
      endif

      call setprop      ! call again because input has changed in userchk

      if (ifcvode .and. nsteps.gt.0) call cv_init

      call comment
      call sstest (isss) 

      call dofcnt

      jp = 0  ! Set perturbation field count to 0 for baseline flow
      p0thn = p0th

      call in_situ_init()

      call time00       !     Initalize timers to ZERO
      call opcount(2)

      ntdump=0
      if (timeio.ne.0.0) ntdump = int( time/timeio )

      tinit = dnekclock_sync() - etimes
      if (nio.eq.0) then
        write (6,*) ' '
        if (time.ne.0.0) write (6,'(a,e14.7)') ' Initial time:',time
        write (6,'(a,g13.5,a)') 
     &     ' Initialization successfully completed ', tinit, ' sec'
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine nek_solve

      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'CTIMER'

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

      ! start measurements
      dtmp = dnekgflops()

      istep  = 0
      msteps = 1

      irstat = int(param(120))

      do kstep=1,nsteps,msteps
         call nek__multi_advance(kstep,msteps)
         if(kstep.ge.nsteps) lastep = 1
         call check_ioinfo  
         call set_outfld
         etime1 = dnekclock()
         call userchk
         tuchk = tuchk + dnekclock()-etime1
         call prepost (ifoutfld,'his')
         call in_situ_check()
         if (mod(kstep,irstat).eq.0 .and. lastep.eq.0) call runstat 
         if (lastep .eq. 1) goto 1001
      enddo
 1001 lastep=1

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

      ntot = lx1*ly1*lz1*nelv

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

         if (ifneknekc .and. igeom.gt.2) then
            if (ifneknekm.and.igeom.eq.3) call neknek_setup
            call neknek_exchange
         endif

         ! call here before we overwrite wx 
         if (ifheat .and. ifcvode) call heat_cvode (igeom)   

         if (ifgeom) then
            call gengeom (igeom)
            call geneig  (igeom)
         endif

         if (ifheat) call heat (igeom)

         if (igeom.eq.2) then  
            call setprop
            call rzero(qtl,ntot)
            if (iflomach) call qthermal
         endif

         if (ifflow)          call fluid    (igeom)
         if (ifmvbd)          call meshv    (igeom)
         if (igeom.eq.ngeom.and.filterType.eq.1)
     $                        call q_filter(param(103))

         enddo

      else                ! PN-2/PN-2 formulation
         call setprop
         do igeom=1,ngeom

            if (ifneknekc .and. igeom.gt.2) then
              if (ifneknekm.and.igeom.eq.3) call neknek_setup
              call neknek_exchange
            endif

            ! call here before we overwrite wx 
            if (ifheat .and. ifcvode) call heat_cvode (igeom)   

            if (ifgeom) then
               if (.not.ifrich) call gengeom (igeom)
               call geneig  (igeom)
            endif

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
            if (igeom.eq.ngeom.and.filterType.eq.1)
     $         call q_filter(param(103))
         enddo
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine nek_end

      include 'SIZE'
      include 'TOTAL'

      if(instep.ne.0) call runstat

c      if (ifstrs) then
c         call fgslib_crs_free(xxth_strs) 
c      else
c         call fgslib_crs_free(xxth(1))
c      endif

      call in_situ_end()
      call exitt0()

      return
      end
c-----------------------------------------------------------------------
      subroutine nek__multi_advance(kstep,msteps)

      include 'SIZE'
      include 'TOTAL'

      do i=1,msteps
         istep = istep+i
         call nek_advance

         if (ifneknekc) then 
            call neknek_exchange
            call bcopy
            call chk_outflow
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
