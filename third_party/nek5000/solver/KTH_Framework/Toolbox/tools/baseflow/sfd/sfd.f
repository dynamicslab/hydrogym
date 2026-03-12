!> @file sfd.f
!! @ingroup sfd
!! @brief Selective frequency damping (SFD) in nekton
!! @author Adam Peplinski
!! @date Feb 6, 2017
!=======================================================================
!> @brief Register SFD module
!! @ingroup sfd
!! @note This routine should be called in frame_usr_register
      subroutine sfd_register()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'SFDD'

      ! local variables
      integer lpmid
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,sfd_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(sfd_name)//'] already registered')
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
      call mntr_mod_reg(sfd_id,lpmid,sfd_name,
     $    'Selective Frequency Damping')

      ! register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(sfd_ttot_id,lpmid,sfd_id,
     $      'SFD_TOT','SFD total time',.false.)

      call mntr_tmr_reg(sfd_tini_id,sfd_ttot_id,sfd_id,
     $      'SFD_INI','SFD initialisation time',.true.)

      call mntr_tmr_reg(sfd_tevl_id,sfd_ttot_id,sfd_id,
     $      'SFD_EVL','SFD evolution time',.true.)

      call mntr_tmr_reg(sfd_tchp_id,sfd_ttot_id,sfd_id,
     $      'SFD_CHP','SFD checkpoint saving time',.true.)

      call mntr_tmr_reg(sfd_tend_id,sfd_ttot_id,sfd_id,
     $      'SFD_END','SFD finalilsation time',.true.)

      ! register and set active section
      call rprm_sec_reg(sfd_sec_id,sfd_id,'_'//adjustl(sfd_name),
     $     'Runtime paramere section for SFD module')
      call rprm_sec_set_act(.true.,sfd_sec_id)

      ! register parameters
      call rprm_rp_reg(sfd_dlt_id,sfd_sec_id,'FILTERWDTH',
     $     'SFD filter width',rpar_real,0,1.0,.false.,' ')

      call rprm_rp_reg(sfd_chi_id,sfd_sec_id,'CONTROLCFF',
     $     'SFD control coefficient',rpar_real,0,1.0,.false.,' ')

      call rprm_rp_reg(sfd_tol_id,sfd_sec_id,'RESIDUALTOL',
     $     'SFD tolerance for residual',rpar_real,0,1.0,.false.,' ')

      call rprm_rp_reg(sfd_cnv_id,sfd_sec_id,'LOGINTERVAL',
     $     'SFD frequency for logging convegence data',
     $      rpar_int,0,1.0,.false.,' ')

      call rprm_rp_reg(sfd_ifrst_id,sfd_sec_id,'SFDREADCHKPT',
     $     'SFD; restat from checkpoint',
     $      rpar_log,0,1.0,.false.,' ')

      ! initialisation flag
      sfd_ifinit = .false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(sfd_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initialise SFD module
!! @ingroup sfd
!! @note This routine should be called in frame_usr_init
      subroutine sfd_init()
      implicit none

      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'TSTEP'
      include 'FRAMELP'
      include 'SFDD'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp
      character*2 str
      character*200 lstring

      ! to get checkpoint runtime parameters
      integer ierr, lmid, lsid, lrpid

      ! functions
      integer frame_get_master
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (sfd_ifinit) then
         call mntr_warn(sfd_id,
     $        'module ['//trim(sfd_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,sfd_dlt_id,rpar_real)
      sfd_dlt = abs(rtmp)
      if (sfd_dlt.gt.0.0) then
         sfd_dlt = 1.0/sfd_dlt
      else
         call mntr_abort(sfd_id,
     $            'sfd_init; Filter width must be positive.')
      endif

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,sfd_chi_id,rpar_real)
      sfd_chi = abs(rtmp)
      if (sfd_chi.le.0.0) call mntr_abort(sfd_id,
     $            'sfd_init; Forcing control must be positive.')

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,sfd_tol_id,rpar_real)
      sfd_tol = abs(rtmp)
      if (sfd_tol.le.0.0) call mntr_abort(sfd_id,
     $            'sfd_init; Residual tolerance must be positive.')

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,sfd_cnv_id,rpar_int)
      sfd_cnv = abs(itmp)

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,sfd_ifrst_id,rpar_log)
      sfd_ifrst = ltmp

      ! check if checkpointing module was registered and take parameters
      ierr = 0
      call mntr_mod_is_name_reg(lmid,'CHKPOINT')
      if (lmid.gt.0) then
         call rprm_sec_is_name_reg(lsid,lmid,'_CHKPOINT')
         if (lsid.gt.0) then
            ! restart flag
            call rprm_rp_is_name_reg(lrpid,lsid,'READCHKPT',rpar_log)
            if (lrpid.gt.0) then
               call rprm_rp_get(itmp,rtmp,ltmp,ctmp,lrpid,rpar_log)
               sfd_chifrst = ltmp
            else
               ierr = 1
               goto 30
            endif
            if (sfd_chifrst) then
               ! checkpoint set number
               call rprm_rp_is_name_reg(lrpid,lsid,'CHKPFNUMBER',
     $              rpar_int)
               if (lrpid.gt.0) then
                  call rprm_rp_get(itmp,rtmp,ltmp,ctmp,lrpid,rpar_int)
                  sfd_fnum = itmp
               else
                  ierr = 1
                  goto 30
               endif
            endif
         else
            ierr = 1
         endif
      else
         ierr = 1
      endif

 30   continue

      ! check for errors
      call mntr_check_abort(sfd_id,ierr,
     $            'Error reading checkpoint parameters')

      ! check restart flags
      if (sfd_ifrst.and.(.not.sfd_chifrst)) call mntr_abort(sfd_id,
     $            'sfd_init; Restart flagg missmatch.')

      ! check simulation parameters
      if (.not.IFTRAN) call mntr_abort(sfd_id,
     $            'sfd_init; SFD requres transient equations')

      if (IFPERT) call mntr_abort(sfd_id,
     $            'sfd_init; SFD cannot be run in perturbation mode')

      ! get number of snapshots in a set
      if (PARAM(27).lt.0) then
         sfd_nsnap = NBDINP
      else
         sfd_nsnap = 3
      endif

      ! initialise module arrays
      if (sfd_ifrst) then
         ! read checkpoint
         call sfd_rst_read
      else
         itmp = nx1*ny1*nz1*nelv
         do il=1,3
            call rzero(sfd_vxlag(1,1,1,1,il),itmp)
            call rzero(sfd_vylag(1,1,1,1,il),itmp)
            if (if3d )call rzero(sfd_vzlag(1,1,1,1,il),itmp)
         enddo

         call opcopy (sfd_vx,sfd_vy,sfd_vz,vx,vy,vz)
      endif

      ! get the forcing (difference between v? and sfd_v?)
      call opsub3(sfd_bfx,sfd_bfy,sfd_bfz,vx,vy,vz,sfd_vx,sfd_vy,sfd_vz)

      ! open the file for convergence history
      ierr = 0
      if (nid.eq.frame_get_master()) then
         call io_file_freeid(sfd_fid, ierr)
         if (ierr.eq.0) then
            open (unit=sfd_fid,file='SFDconv.out',status='new',
     $           action='write',iostat=ierr)
         endif
      endif

      ! check for errors
      call mntr_check_abort(sfd_id,ierr,
     $            'Error opening convergence file.')

      ! initialise storage arrays
      call rzero(sfd_abx1,itmp)
      call rzero(sfd_abx2,itmp)
      call rzero(sfd_aby1,itmp)
      call rzero(sfd_aby2,itmp)
      call rzero(sfd_abz1,itmp)
      call rzero(sfd_abz2,itmp)

      ! initialisation flag
      sfd_ifinit = .true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(sfd_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup sfd
!! @return sfd_is_initialised
      logical function sfd_is_initialised()
      implicit none

      include 'SIZE'
      include 'SFDD'
!-----------------------------------------------------------------------
      sfd_is_initialised = sfd_ifinit

      return
      end function
!=======================================================================
!> @brief Finalise SFD
!! @ingroup sfd
!! @note This routine should be called in frame_usr_end
      subroutine sfd_end
      implicit none

      include 'SIZE'            ! NID, NDIM, NPERT
      include 'INPUT'           ! IF3D
      include 'RESTART'
      include 'SOLN'
      include 'FRAMELP'
      include 'SFDD'

      ! local variables
      integer ntot1, il
      real ltim
      real ab0, ab1, ab2

      logical lifxyo, lifpo, lifvo, lifto, lifreguo, lifpso(LDIMT1)

      ! functions
      integer frame_get_master
      real dnekclock, gl2norm
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      ! final convergence
      ntot1 = nx1*ny1*nz1*nelv

      ! calculate L2 norms
      ab0 = gl2norm(sfd_bfx,ntot1)
      ab1 = gl2norm(sfd_bfy,ntot1)
      if (if3d) ab2 = gl2norm(sfd_bfz,ntot1)

      ! stamp the log
      call mntr_log(sfd_id,lp_prd,
     $              'Final convergence (L2 norm per grid point):')
      call mntr_logr(sfd_id,lp_prd,'DVX = ',ab0)
      call mntr_logr(sfd_id,lp_prd,'DVY = ',ab1)
      if (if3d) call mntr_logr(sfd_id,lp_prd,'DVZ = ',ab2)
      call mntr_log(sfd_id,lp_prd,'Saving velocity difference')

      ! save the velocity difference for inspection
      lifxyo= ifxyo
      ifxyo = .true.
      lifpo= ifpo
      ifpo = .false.
      lifvo= ifvo
      ifvo = .true.
      lifto= ifto
      ifto = .false.
      do il=1,ldimt1
         lifpso(il)= ifpso(il)
         ifpso(il) = .false.
      enddo

      call outpost2(sfd_bfx,sfd_bfy,sfd_bfz,pr,t,0,'vdf')

      ifxyo = lifxyo
      ifpo = lifpo
      ifvo = lifvo
      ifto = lifto
      do il=1,ldimt1
         ifpso(il) = lifpso(il)
      enddo

      ! close file with convergence history
      if (nid.eq.frame_get_master()) close(sfd_fid)

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(sfd_tend_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Main SFD interface
!! @ingroup sfd
      subroutine sfd_main
      implicit none
!-----------------------------------------------------------------------
      call sfd_solve
      call sfd_rst_write

      return
      end subroutine
!=======================================================================
!> @brief Calcualte SFD forcing
!! @ingroup sfd
!! @param[inout] ffx,ffy,ffz     forcing; x,y,z component
!! @param[in]    ix,iy,iz        GLL point index
!! @param[in]    ieg             global element number
      subroutine sfd_forcing(ffx,ffy,ffz,ix,iy,iz,ieg)
      implicit none

      include 'SIZE'            !
      include 'INPUT'           ! IF3D
      include 'PARALLEL'        ! GLLEL
      include 'SFDD'

      ! argument list
      real ffx, ffy, ffz
      integer ix,iy,iz,ieg

      ! local variables
      integer iel
!-----------------------------------------------------------------------
      iel=GLLEL(ieg)
      ffx = ffx - sfd_chi*sfd_bfx(ix,iy,iz,iel)
      ffy = ffy - sfd_chi*sfd_bfy(ix,iy,iz,iel)
      if (if3d) ffz = ffz - sfd_chi*sfd_bfz(ix,iy,iz,iel)

      return
      end subroutine
!=======================================================================
!> @brief Update filtered velocity field.
!! @ingroup sfd
!! @details Sum up contributions to kth order extrapolation scheme and
!!   get new filtered velocity field.
!!   This subroutine is based on @ref makeabf and  @ref makebdf
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine sfd_solve
      implicit none

      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'           ! ISTEP, TIME, NSTEPS, LASTEP
      include 'INPUT'           ! IF3D
      include 'FRAMELP'
      include 'SFDD'

      ! temporary storage
      real  TA1 (LX1,LY1,LZ1,LELV), TA2 (LX1,LY1,LZ1,LELV),
     $      TA3 (LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/ TA1, TA2, TA3

      ! local variables
      integer ntot1, ilag
      real ltim
      real ab0, ab1, ab2

      ! functions
      integer frame_get_master
      real dnekclock, gl2norm
!-----------------------------------------------------------------------
      if (istep.eq.0) return

      ! timing
      ltim = dnekclock()

      ! active array length
      ntot1 = NX1*NY1*NZ1*NELV

      ! A-B part
      ! current rhs
      ! I use BFS? vectors generated during the convergence tests so skip this step
!          call opsub3(sfd_bfx,sfd_bfy,sfd_bfz,vxlag,vylag,vzlag,
!     $                sfd_vx,sfd_vy,sfd_vz)
      ! finish rhs
      call opcmult(sfd_bfx,sfd_bfy,sfd_bfz,sfd_dlt)

      ! old time steps
      ab0 = AB(1)
      ab1 = AB(2)
      ab2 = AB(3)
      call add3s2(ta1,sfd_abx1,sfd_abx2,ab1,ab2,ntot1)
      call add3s2(ta2,sfd_aby1,sfd_aby2,ab1,ab2,ntot1)
      ! save rhs
      call copy(sfd_abx2,sfd_abx1,ntot1)
      call copy(sfd_aby2,sfd_aby1,ntot1)
      call copy(sfd_abx1,sfd_bfx,ntot1)
      call copy(sfd_aby1,sfd_bfy,ntot1)
      ! current
      call add2s1 (sfd_bfx,ta1,ab0,ntot1)
      call add2s1 (sfd_bfy,ta2,ab0,ntot1)
      if (if3d) then
         call add3s2 (ta3,sfd_abz1,sfd_abz2,ab1,ab2,ntot1)
         call copy   (sfd_abz2,sfd_abz1,ntot1)
         call copy   (sfd_abz1,sfd_bfz,ntot1)
         call add2s1 (sfd_bfz,ta3,ab0,ntot1)
      endif

      ! multiplication by time step
      call opcmult(sfd_bfx,sfd_bfy,sfd_bfz,dt)

      ! BD part
      ab0 = bd(2)
      call opadd2cm(sfd_bfx,sfd_bfy,sfd_bfz,sfd_vx,sfd_vy,sfd_vz,ab0)

      do ilag=2,nbd
         ab0 = bd(ilag+1)
         call opadd2cm(sfd_bfx,sfd_bfy,sfd_bfz,
     $           sfd_vxlag (1,1,1,1,ilag-1),sfd_vylag (1,1,1,1,ilag-1),
     $           sfd_vzlag (1,1,1,1,ilag-1),ab0)
      enddo

      ! take into account restart option
      if (sfd_ifrst.and.(istep.lt.sfd_nsnap))
     $   call opcopy (ta1,ta2,ta3,sfd_vxlag(1,1,1,1,sfd_nsnap-1),
     $   sfd_vylag(1,1,1,1,sfd_nsnap-1),sfd_vzlag(1,1,1,1,sfd_nsnap-1))

      ! keep old filtered velocity fields
      do ilag=3,2,-1
         call opcopy(sfd_vxlag(1,1,1,1,ilag),sfd_vylag(1,1,1,1,ilag),
     $        sfd_vzlag(1,1,1,1,ilag),sfd_vxlag(1,1,1,1,ilag-1),
     $        sfd_vylag(1,1,1,1,ilag-1),sfd_vzlag(1,1,1,1,ilag-1))
      enddo

      call opcopy (sfd_vxlag,sfd_vylag,sfd_vzlag,sfd_vx,sfd_vy,sfd_vz)

      ! calculate new filtered velocity field
      ! take into account restart option
      if (sfd_ifrst.and.(istep.lt.sfd_nsnap)) then
         call opcopy (sfd_vx,sfd_vy,sfd_vz,ta1,ta2,ta3)
      else
         ab0 = 1.0/bd(1)
         call opcopy (sfd_vx,sfd_vy,sfd_vz,sfd_bfx,sfd_bfy,sfd_bfz)
         call opcmult(sfd_vx,sfd_vy,sfd_vz,ab0)
      endif

      ! convergence test
      ! find the difference between V? and VS?
      call opsub3(sfd_bfx,sfd_bfy,sfd_bfz,vx,vy,vz,sfd_vx,sfd_vy,sfd_vz)

      ! calculate L2 norms
      ab0 = gl2norm(sfd_bfx,ntot1)
      ab1 = gl2norm(sfd_bfy,ntot1)
      if (if3d) ab2 = gl2norm(sfd_bfz,ntot1)

      ! for tracking convergence
      if (mod(istep,sfd_cnv).eq.0) then
         if (nid.eq.frame_get_master()) then
            if (if3d) then
               write(sfd_fid,'(4e13.5)') time, ab0, ab1, ab2
            else
               write(sfd_fid,'(3e13.5)') time, ab0, ab1
            endif
         endif

         ! stamp the log
         call mntr_log(sfd_id,lp_prd,
     $              'Convergence (L2 norm per grid point):')
         call mntr_logr(sfd_id,lp_prd,'DVX = ',ab0)
         call mntr_logr(sfd_id,lp_prd,'DVY = ',ab1)
         if (if3d) call mntr_logr(sfd_id,lp_prd,'DVZ = ',ab2)
      endif

      ! check stopping criteria
      if (istep.gt.sfd_nsnap) then ! to ensure proper restart
         ab0 = max(ab0,ab1)
         if (if3d) ab0 = max(ab0,ab2)
         if (ab0.lt.sfd_tol) then
            call mntr_log(sfd_id,lp_ess,'Stopping criteria reached')
            call mntr_set_conv(.TRUE.)
         endif
      endif

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(sfd_tevl_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Create checkpoint
!! @ingroup sfd
      subroutine sfd_rst_write
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'FRAMELP'
      include 'SFDD'

      ! local variables
      integer step_cnt, set_out
      real ltim
      integer lwdsizo
      integer ipert, il, ierr
      logical lif_full_pres, lifxyo, lifpo, lifvo, lifto,
     $        lifpsco(LDIMT1), ifreguol

      character*132 fname, bname
      character*3 prefix
      character*6  str

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! avoid writing during possible restart reading
      call mntr_get_step_delay(step_cnt)
      if (istep.le.step_cnt) return

      ! get step count and file set number
      call chkpt_get_fset(step_cnt, set_out)

      ! we write everything in single step
      if (step_cnt.eq.1) then
         ltim = dnekclock()

         call mntr_log(sfd_id,lp_inf,'Writing checkpoint snapshot')

         ! adjust I/O parameters
         lwdsizo = WDSIZO
         WDSIZO  = 8
         lif_full_pres = IF_FULL_PRES
         IF_FULL_PRES = .false.
         lifxyo = IFXYO
         IFXYO = .false.
         lifpo= IFPO
         IFPO = .false.
         lifvo= IFVO
         IFVO = .true.
         lifto= IFTO
         IFTO = .false.
         do il=1,NPSCAL
            lifpsco(il)= IFPSCO(il)
            IFPSCO(il) = .TRUE.
         enddo
         ifreguol= IFREGUO
         IFREGUO = .false.

         ! initialise I/O data
         call io_init

         ! get file name
         prefix = 'SFD'
         bname = trim(adjustl(SESSION))
         call io_mfo_fname(fname,bname,prefix,ierr)
         call mntr_check_abort(sfd_id,ierr,
     $       'sfd_rst_write; file name error')
         write(str,'(i5.5)') set_out + 1
         fname=trim(fname)//trim(str(1:5))

         ! save filtered valocity field
         call sfd_mfo(fname)

         ! put parameters back
         WDSIZO = lwdsizo
         IF_FULL_PRES = lif_full_pres
         IFXYO = lifxyo
         IFPO = lifpo
         IFVO = lifvo
         IFTO = lifto
         do il=1,NPSCAL
            IFPSCO(il) = lifpsco(il)
         enddo
         IFREGUO = ifreguol

         ! timing
         ltim = dnekclock() - ltim
         call mntr_tmr_add(sfd_tchp_id,1,ltim)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Read from checkpoint
!! @ingroup sfd
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine sfd_rst_read
      implicit none

      include 'SIZE'            ! NID, NDIM, NPERT
      include 'INPUT'
      include 'FRAMELP'
      include 'SFDD'

      ! temporary storage
      real TA1 (LX1,LY1,LZ1,LELV), TA2 (LX1,LY1,LZ1,LELV),
     $     TA3 (LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/ TA1, TA2, TA3

      ! local variables
      integer ilag, ierr
      logical ifreguol

      character*132 fname, bname
      character*3 prefix
      character*6  str
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_log(sfd_id,lp_inf,'Reading checkpoint')

      ! no regular mesh
      ifreguol= ifreguo
      ifreguo = .false.

      ! initialise I/O data
      call io_init

      ! create file name
      prefix = 'SFD'
      bname = trim(adjustl(SESSION))
      call io_mfo_fname(fname,bname,prefix,ierr)
      call mntr_check_abort(sfd_id,ierr,'sfd_rst_read; file name error')
      write(str,'(i5.5)') sfd_fnum
      fname=trim(fname)//trim(str(1:5))

      ! read filtered velocity field
      call sfd_mfi(fname)

      ! put parameters back
      ifreguo = ifreguol

      ! move velcity fields to sotre oldest one in VS?
      call opcopy (ta1,ta2,ta3,sfd_vxlag(1,1,1,1,sfd_nsnap-1),
     $   sfd_vylag(1,1,1,1,sfd_nsnap-1),sfd_vzlag(1,1,1,1,sfd_nsnap-1))

      do ilag=sfd_nsnap-1,2,-1
         call opcopy(sfd_vxlag(1,1,1,1,ilag),sfd_vylag(1,1,1,1,ilag),
     $        sfd_vzlag(1,1,1,1,ilag),sfd_vxlag(1,1,1,1,ilag-1),
     $        sfd_vylag(1,1,1,1,ilag-1),sfd_vzlag(1,1,1,1,ilag-1))
      enddo

      call opcopy (sfd_vxlag,sfd_vylag,sfd_vzlag,sfd_vx,sfd_vy,sfd_vz)
      call opcopy (sfd_vx,sfd_vy,sfd_vz,ta1,ta2,ta3)

      return
      end subroutine
!=======================================================================
!> @brief Store SFD restart file
!! @ingroup sfd
!! @details This rouotine is version of @ref mfo_outfld adjusted for SFD restart.
!! @param[in]   fname      file name
!! @note This routine uses standard header wirter so cannot pass additional
!!    information in the file header. That is why I save whole lag spce
!!    irrespective of  chpm_nsnap value.
      subroutine sfd_mfo(fname)
      implicit none

      include 'SIZE'
      include 'RESTART'
      include 'PARALLEL'
      include 'INPUT'
      include 'FRAMELP'
      INCLUDE 'SFDD'

      ! argument list
      character*132 fname

      ! local variables
      integer il, ierr, ioflds, nout
      integer*8 offs

      real tiostart, tio, dnbyte

      ! functions
      real dnekclock_sync, glsum
!-----------------------------------------------------------------------
      ! simple timing
      tiostart=dnekclock_sync()

      nout = NELT
      NXO  = NX1
      NYO  = NY1
      NZO  = NZ1

      ! open file
      ierr = 0
      call io_mbyte_open(fname,ierr)
      call mntr_check_abort(sfd_id,ierr,'sfd_mfo; file not opened.')

      ! write a header and create element mapping
      call mfo_write_hdr

      ! set offset
      offs = iHeaderSize + 4 + isize*nelgt
      ioflds = 0

      ! write fields
      ! current filtered velocity field
      call io_mfov(offs,sfd_vx,sfd_vy,sfd_vz,nx1,ny1,nz1,nelt,
     $      nelgt,ndim)
      ioflds = ioflds + ndim

      ! history
      do il=1,3
         call io_mfov(offs,sfd_vxlag(1,1,1,1,il),sfd_vylag(1,1,1,1,il),
     $           sfd_vzlag(1,1,1,1,il),nx1,ny1,nz1,nelt,nelgt,ndim)
         ioflds = ioflds + ndim
      enddo

      dnbyte = 1.*ioflds*nelt*wdsizo*nx1*ny1*nz1

      ! close file
      call io_mbyte_close(ierr)
      call mntr_check_abort(sfd_id,ierr,'sfd_mfo; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. +ISIZE*NELGT
      dnbyte = dnbyte/1024/1024

      call mntr_log(sfd_id,lp_prd,'Checkpoint written:')
      call mntr_logr(sfd_id,lp_vrb,'file size (MB) = ',dnbyte)
      call mntr_logr(sfd_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(sfd_id,lp_vrb,'io-nodes = ',nfileo)

      return
      end subroutine
!=======================================================================
!> @brief Load SFD restart file
!! @ingroup sfd
!! @details This rouotine is version of @ref mfi adjusted ofr SFD restart.
!! @param[in]   fname      file name
!! @note This routine uses standard header reader and cannot pass additiona
!!    information in the file. That is why I read whole lag spce irrespective
!!    of  sfd_nsnap value.
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine sfd_mfi(fname)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'FRAMELP'
      include 'SFDD'

      ! argument lilst
      character*132 fname

      ! local variables
      integer il, ioflds, ierr
      integer*8 offs
      real tiostart, tio, dnbyte
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

      ! open file, get header information and read mesh data
      call mfi_prepare(fname)

      ! set header offset
      offs = iHeaderSize + 4 + isize*nelgr
      ioflds = 0
      ifskip = .FALSE.

      ! read arrays
      ! filtered velocity
      if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
         ! unchanged resolution
         ! read field directly to the variables
         call io_mfiv(offs,sfd_vx,sfd_vy,sfd_vz,lx1,ly1,lz1,lelv,ifskip)
      else
         ! modified resolution
         ! read field to tmp array
         call io_mfiv(offs,wkv1,wkv2,wkv3,nxr,nyr,nzr,lelt,ifskip)

         ! interpolate
         call chkpt_map_gll(sfd_vx,wkv1,nxr,nzr,nelv)
         call chkpt_map_gll(sfd_vy,wkv2,nxr,nzr,nelv)
         if (if3d) call chkpt_map_gll(sfd_vz,wkv3,nxr,nzr,nelv)
      endif
      ioflds = ioflds + ndim

      ! history
      do il=1,3
         if ((nxr.eq.lx1).and.(nyr.eq.ly1).and.(nzr.eq.lz1)) then
            ! unchanged resolution
            ! read field directly to the variables
            call io_mfiv(offs,sfd_vxlag(1,1,1,1,il),
     $           sfd_vylag(1,1,1,1,il),sfd_vzlag(1,1,1,1,il),
     $           lx1,ly1,lz1,lelv,ifskip)
         else
            ! modified resolution
            ! read field to tmp array
            call io_mfiv(offs,wkv1,wkv2,wkv3,nxr,nyr,nzr,lelt,ifskip)

            ! interpolate
            call chkpt_map_gll(sfd_vxlag(1,1,1,1,il),wkv1,nxr,nzr,nelv)
            call chkpt_map_gll(sfd_vylag(1,1,1,1,il),wkv2,nxr,nzr,nelv)
            if (if3d) call chkpt_map_gll(sfd_vzlag(1,1,1,1,il),wkv3,
     $                     nxr,nzr,nelv)
         endif
         ioflds = ioflds + ndim
      enddo

      ! close file
      call io_mbyte_close(ierr)
      call mntr_check_abort(sfd_id,ierr,'sfd_mfi; file not closed.')

      ! stamp the log
      tio = dnekclock_sync()-tiostart
      if (tio.le.0.0) tio=1.

      if(nid.eq.pid0r) then
         dnbyte = 1.*ioflds*nelr*wdsizr*nxr*nyr*nzr
      else
         dnbyte = 0.0
      endif

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024

      call mntr_log(sfd_id,lp_prd,'Checkpoint read:')
      call mntr_logr(sfd_id,lp_vrb,'avg data-throughput (MB/s) = ',
     $     dnbyte/tio)
      call mntr_logi(sfd_id,lp_vrb,'io-nodes = ',nfileo)

      if (ifaxis) call mntr_abort(sfd_id,
     $                 'sfd_mfi; axisymmetric case not supported')

      return
      end subroutine
!=======================================================================
