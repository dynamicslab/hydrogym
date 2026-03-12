!> @file pstat3D.f
!! @ingroup pstat3d
!! @brief Post processing for statistics module
!! @author Adam Peplinski
!! @date Mar 13, 2019
!=======================================================================
!> @brief Register post processing statistics module
!! @ingroup pstat3d
!! @note This routine should be called in frame_usr_register
      subroutine pstat3d_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'PSTAT3D'

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
      call mntr_mod_is_name_reg(lpmid,pstat_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(pstat_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'parent module ['//'FRAME'//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(pstat_id,lpmid,pstat_name,
     $      'Post processing for 3D statistics')

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      ! total time
      call mntr_tmr_reg(pstat_tmr_tot_id,lpmid,pstat_id,
     $     'PSTAT_TOT','Pstat total time',.false.)
      lpmid = pstat_tmr_tot_id
      ! initialisation
      call mntr_tmr_reg(pstat_tmr_ini_id,lpmid,pstat_id,
     $     'PSTAT_INI','Pstat initialisation time',.true.)
      ! averaging
      call mntr_tmr_reg(pstat_tmr_avg_id,lpmid,pstat_id,
     $     'PSTAT_AVG','Pstat averaging time',.true.)
      ! new field calculation
      call mntr_tmr_reg(pstat_tmr_new_id,lpmid,pstat_id,
     $     'PSTAT_NEW','Pstat new field calculation time',.true.)
      ! interpolation
      call mntr_tmr_reg(pstat_tmr_int_id,lpmid,pstat_id,
     $     'PSTAT_INT','Pstat interpolation time',.true.)

      ! register and set active section
      call rprm_sec_reg(pstat_sec_id,pstat_id,'_'//adjustl(pstat_name),
     $     'Runtime paramere section for pstat module')
      call rprm_sec_set_act(.true.,pstat_sec_id)

      ! register parameters
      call rprm_rp_reg(pstat_nfile_id,pstat_sec_id,'STS_NFILE',
     $ 'Number of stat files',rpar_int,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_stime_id,pstat_sec_id,'STS_STIME',
     $ 'Statistics starting time',rpar_real,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_nstep_id,pstat_sec_id,'STS_NSTEP',
     $ 'Number of steps between averaging (in sts file)',
     $  rpar_int,10,0.0,.false.,' ')

      ! set initialisation flag
      pstat_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_tot_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise pstat module
!! @ingroup pstat3d
!! @note This routine should be called in frame_usr_init
      subroutine pstat3d_init()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp
      integer tmp_swfield(pstat_svar)

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
!     check if the module was already initialised
      if (pstat_ifinit) then
         call mntr_warn(pstat_id,
     $        'module ['//trim(pstat_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_nfile_id,rpar_int)
      pstat_nfile = abs(itmp)
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_stime_id,rpar_real)
      pstat_stime = rtmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_nstep_id,rpar_int)
      pstat_nstep = abs(itmp)

      ! set field swapping array
      do il = 1,26
         tmp_swfield(il) = il
      enddo
      do il = 27,32
         tmp_swfield(il) = il+1
      enddo
      tmp_swfield(33) = 27
      tmp_swfield(34) = 38
      do il = 35,38
         tmp_swfield(il) = il-1
      enddo
      do il = 39,44
         tmp_swfield(il) = il
      enddo
      call icopy(pstat_swfield,tmp_swfield,pstat_svar)

      ! everything is initialised
      pstat_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup pstat3d
!! @return pstat3d_is_initialised
      logical function pstat3d_is_initialised()
      implicit none

      include 'SIZE'
      include 'PSTAT3D'
!-----------------------------------------------------------------------
      pstat3d_is_initialised = pstat_ifinit

      return
      end function
!=======================================================================
!> @brief Main interface of pstat module
!! @ingroup pstat3d
      subroutine pstat3d_main
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! local variables
      integer il

      ! simple timing
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! read and average fields
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Field averaging')
      call pstat3d_sts_avg
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_avg_id,1,ltim)

      ! calculate new fields
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'New field calculation')
      call pstat3d_nfield
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_new_id,1,ltim)

      ! interpolate into the set of points
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Point interpolation')
      call pstat3d_interp
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_int_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Read in fields and average them
!! @ingroup pstat3d
      subroutine pstat3d_sts_avg
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'SOLN'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! local variables
      integer il, jl            ! loop index
      integer ierr              ! error mark
      integer nvec              ! single field length
      character*3 prefix        ! file prefix
      character*2 fins          ! number of a file in a section
      character*132 fname       ! file name
      character*132 bname       ! base name
      character*6  str          ! file number
      integer nps1,nps0,npsr    ! number of passive scalars in the file
      integer istepr            ! number of time step in restart files
      real ltime, dtime         ! simulation time and time update
      real rtmp

!-----------------------------------------------------------------------
      ! no regular mesh; important for file name generation
      ifreguo = .false.

      call io_init

      ierr=0
      ! open files on i/o nodes
      ! get base name (SESSION)
      bname = trim(adjustl(session))

      ! mark variables to be read
      ifgetx=.false.
      ifgetz=.false.
      ifgetu=.true.
      ifgetw=.true.
      ifgetp=.false.
      ifgett=.true.
      do il=1, npscal
          ifgtps(il)=.false.
      enddo

      ! initial time and step count
      ltime = pstat_stime
      istepr = 0

      ! initilise vectors
      call rzero(pstat_ruavg,lx1**ldim*lelt*pstat_svar)
      nvec = lx1*ly1*lz1*nelt

      ! loop over stat files
      do il = 1,pstat_nfile

         do jl=1, pstat_finset

            ! get prefix
            write(fins,'(i2.2)') jl
            prefix = 's'//trim(fins)

            ! get file name
            call io_mfo_fname(fname,bname,prefix,ierr)
            write(str,'(i5.5)') il
            fname = trim(fname)//trim(str)

            fid0 = 0
            call addfid(fname,fid0)

            ! add directory name
            fname = 'DATA/'//trim(fname)

            !call load_fld(fname)
            call mfi(fname,il)

            ! calculate interval and update time
            dtime = timer - ltime

            ! accumulate fileds
            call add2s2(pstat_ruavg(1,1,pstat_swfield(1,jl)),
     $           vx,dtime,nvec)
            call add2s2(pstat_ruavg(1,1,pstat_swfield(2,jl)),
     $           vy,dtime,nvec)
            call add2s2(pstat_ruavg(1,1,pstat_swfield(3,jl)),
     $           vz,dtime,nvec)
            call add2s2(pstat_ruavg(1,1,pstat_swfield(4,jl)),
     $           t,dtime,nvec)
         enddo

         ! sum number of time steps
         istepr = istepr + istpr

         ! update time
         ltime = timer

      enddo

      ! save data for file header
      pstat_etime = ltime
      pstat_istepr = istepr

      ! divide by time span
      if (ltime.ne.pstat_stime) then
         rtmp = 1.0/(ltime-pstat_stime)
      else
         rtmp = 1.0
      endif

      do il = 1,pstat_svar
         call cmult(pstat_ruavg(1,1,il),rtmp,nvec)
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Calculate new fileds
!! @ingroup pstat3d
      subroutine pstat3d_nfield
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! local variables
      integer il, jl            ! loop index
      integer nvec              ! single field length
      integer itmp
      real rtmp, rho
!-----------------------------------------------------------------------
      ! update derivative arrays
      !call geom_reset(1)

      ! initilise vectors
      call rzero(pstat_runew,lx1**ldim*lelt*pstat_dvar)
      call rzero(pstat_rutmp,lx1**ldim*lelt*pstat_tvar)
      nvec = lx1*ly1*lz1*nelt
      rho = 1.0

      ! mean variables U, V, W, P already calculated and stored under
      ! U (avg 1), V (avg 2), W (avg 3), P (avg 4)
      ! Reynolds-stress tensor diagonal terms and pressure RMS
      ! U*U (tmp 1), V*V (tmp 2), W*W (tmp 3), P*P (tmp 4)
      ! uu (avg 5), vv (avg 6), ww (avg 7), pp (avg 8)
      do il = 1, 4
         call col3(pstat_rutmp(1,1,il),pstat_ruavg(1,1,il),
     $        pstat_ruavg(1,1,il),nvec)
         call sub2(pstat_ruavg(1,1,4+il),pstat_rutmp(1,1,il),nvec)
      enddo

      ! Reynolds-stress tensor off diagonal terms
      call subcol3(pstat_ruavg(1,1,9),pstat_ruavg(1,1,1), ! uv (avg 9)
     $     pstat_ruavg(1,1,2),nvec)
      call subcol3(pstat_ruavg(1,1,10),pstat_ruavg(1,1,2), ! vw (avg 10)
     $     pstat_ruavg(1,1,3),nvec)
      call subcol3(pstat_ruavg(1,1,11),pstat_ruavg(1,1,1), ! uw (avg 11)
     $     pstat_ruavg(1,1,3),nvec)

      ! pressure skewness
      call col3(pstat_rutmp(1,1,5),pstat_rutmp(1,1,4), ! P*P*P (tmp 5)
     $     pstat_ruavg(1,1,4),nvec)
      call sub2(pstat_ruavg(1,1,27),pstat_rutmp(1,1,5),nvec) ! ppp (avg 27)
      rtmp = -3.0
      call admcol3(pstat_ruavg(1,1,27),pstat_ruavg(1,1,4),
     $     pstat_ruavg(1,1,8),rtmp,nvec)

      ! pressure flattness
      rtmp = -4.0
      call admcol3(pstat_ruavg(1,1,38),pstat_ruavg(1,1,4), ! pppp (avg 38)
     $     pstat_ruavg(1,1,27),rtmp,nvec)
      rtmp = -6.0
      call admcol3(pstat_ruavg(1,1,38),pstat_ruavg(1,1,8),
     $     pstat_rutmp(1,1,4),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,38),pstat_ruavg(1,1,4),
     $     pstat_rutmp(1,1,5),nvec)

      ! Skewness tensor
      ! diagonal terms
      ! uuu (avg 24), vvv (avg 25), www (avg 26)
      rtmp = -3.0
      do il= 1, 3
         call admcol3(pstat_ruavg(1,1,23+il),pstat_ruavg(1,1,il),
     $        pstat_ruavg(1,1,4+il),rtmp,nvec)
         call subcol3(pstat_ruavg(1,1,23+il),pstat_ruavg(1,1,il),
     $        pstat_rutmp(1,1,il),nvec)
      enddo

      ! off diagonal terms
      rtmp = -2.0
      call admcol3(pstat_ruavg(1,1,28),pstat_ruavg(1,1,1), ! uuv (avg 28)
     $     pstat_ruavg(1,1,9),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,28),pstat_ruavg(1,1,2),
     $     pstat_ruavg(1,1,5),nvec)
      call subcol3(pstat_ruavg(1,1,28),pstat_ruavg(1,1,2),
     $     pstat_rutmp(1,1,1),nvec)
      call admcol3(pstat_ruavg(1,1,29),pstat_ruavg(1,1,1), ! uuw (avg 29)
     $     pstat_ruavg(1,1,11),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,29),pstat_ruavg(1,1,3),
     $     pstat_ruavg(1,1,5),nvec)
      call subcol3(pstat_ruavg(1,1,29),pstat_ruavg(1,1,3),
     $     pstat_rutmp(1,1,1),nvec)
      call admcol3(pstat_ruavg(1,1,30),pstat_ruavg(1,1,2), ! uvv (avg 30)
     $     pstat_ruavg(1,1,9),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,30),pstat_ruavg(1,1,1),
     $     pstat_ruavg(1,1,6),nvec)
      call subcol3(pstat_ruavg(1,1,30),pstat_ruavg(1,1,1),
     $     pstat_rutmp(1,1,2),nvec)
      call admcol3(pstat_ruavg(1,1,31),pstat_ruavg(1,1,2), ! vvw (avg 31)
     $     pstat_ruavg(1,1,10),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,31),pstat_ruavg(1,1,3),
     $     pstat_ruavg(1,1,6),nvec)
      call subcol3(pstat_ruavg(1,1,31),pstat_ruavg(1,1,3),
     $     pstat_rutmp(1,1,2),nvec)
      call admcol3(pstat_ruavg(1,1,32),pstat_ruavg(1,1,3), ! uww (avg 32)
     $     pstat_ruavg(1,1,11),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,32),pstat_ruavg(1,1,1),
     $     pstat_ruavg(1,1,7),nvec)
      call subcol3(pstat_ruavg(1,1,32),pstat_ruavg(1,1,1),
     $     pstat_rutmp(1,1,3),nvec)
      call admcol3(pstat_ruavg(1,1,33),pstat_ruavg(1,1,3), ! vww (avg 33)
     $     pstat_ruavg(1,1,10),rtmp,nvec)
      call subcol3(pstat_ruavg(1,1,33),pstat_ruavg(1,1,2),
     $     pstat_ruavg(1,1,7),nvec)
      call subcol3(pstat_ruavg(1,1,33),pstat_ruavg(1,1,2),
     $     pstat_rutmp(1,1,3),nvec)
      call subcol3(pstat_ruavg(1,1,34),pstat_ruavg(1,1,1), ! uvw (avg 34)
     $     pstat_ruavg(1,1,10),nvec)
      call subcol3(pstat_ruavg(1,1,34),pstat_ruavg(1,1,2),
     $     pstat_ruavg(1,1,11),nvec)
      call subcol3(pstat_ruavg(1,1,34),pstat_ruavg(1,1,3),
     $     pstat_ruavg(1,1,9),nvec)
      call subcol4(pstat_ruavg(1,1,34),pstat_ruavg(1,1,1),
     $     pstat_ruavg(1,1,2),pstat_ruavg(1,1,3),nvec)

      ! Velocity gradient tensor
      ! dUdx (new 1), dUdy (new 2), dUdz (new 3),
      ! dVdx (new 4), dVdy (new 5), dVdz (new 6), 
      ! dWdx (new 7), dWdy (new 8), dWdz (new 9)
      do il=1,3
         itmp = (il-1)*3
         call gradm1(pstat_runew(1,1,itmp+1),pstat_runew(1,1,itmp+2), 
     $     pstat_runew(1,1,itmp+3),pstat_ruavg(1,1,il))
      enddo

      ! Dissipation tensor
      rtmp = -2.0*param(2)
      ! diagonal terms
      ! Dxx (avg 39), Dyy (avg 40), Dzz (avg 41)
      do il=1,3
         itmp = (il-1)*3
         do jl = 1,3
            call subcol3(pstat_ruavg(1,1,38+il),
     $           pstat_runew(1,1,itmp+jl),pstat_runew(1,1,itmp+jl),
     $           nvec)
         enddo
         call cmult(pstat_ruavg(1,1,38+il),rtmp,nvec)
      enddo
      ! off diagonal terms
      do jl = 1,3
         call subcol3(pstat_ruavg(1,1,42),pstat_runew(1,1,jl), ! Dxy (avg 42)
     $     pstat_runew(1,1,3+jl),nvec)
      enddo
      call cmult(pstat_ruavg(1,1,42),rtmp,nvec)
      do jl = 1,3
         call subcol3(pstat_ruavg(1,1,43),pstat_runew(1,1,jl), ! Dxz (avg 43)
     $     pstat_runew(1,1,6+jl),nvec)
      enddo
      call cmult(pstat_ruavg(1,1,43),rtmp,nvec)
      do jl = 1,3
         call subcol3(pstat_ruavg(1,1,44),pstat_runew(1,1,3+jl), ! Dyz (avg 44)
     $     pstat_runew(1,1,6+jl),nvec)
      enddo
      call cmult(pstat_ruavg(1,1,44),rtmp,nvec)

      ! Derivatives of the Reynolds-stress tensor
      ! duudx (tmp 2), duudy (tmp 3), duudz (tmp 4),
      ! dvvdx (tmp 5), dvvdy (tmp 6), dvvdz (tmp 7),
      ! dwwdx (tmp 8), dwwdy (tmp 9), dwwdz (tmp 10),
      ! duvdx (tmp 11), duvdy (tmp 12), duvdz (tmp 13),
      ! dvwdx (tmp 14), dvwdy (tmp 15), dvwdz (tmp 16),
      ! duwdx (tmp 17), duwdy (tmp 18), duwdz (tmp 19)
      do il = 1,3
         itmp = (il-1)*3
         call gradm1(pstat_rutmp(1,1,itmp+2),pstat_rutmp(1,1,itmp+3),
     $        pstat_rutmp(1,1,itmp+4),pstat_ruavg(1,1,4+il))
         itmp = (il+2)*3
         call gradm1(pstat_rutmp(1,1,itmp+2),pstat_rutmp(1,1,itmp+3),
     $        pstat_rutmp(1,1,itmp+4),pstat_ruavg(1,1,8+il))
      enddo

      ! Mean convection tensor
      ! Cxx (new 10), Cyy (new 11), Czz (new 12), Cxy (new 13), Cyz (new 14), Cxz (new 15)
      do il = 1,6
         itmp = (il-1)*3
         call vdot3 (pstat_runew(1,1,9+il),
     $        pstat_ruavg(1,1,1),pstat_ruavg(1,1,2),pstat_ruavg(1,1,3),
     $        pstat_rutmp(1,1,itmp+2),pstat_rutmp(1,1,itmp+3),
     $        pstat_rutmp(1,1,itmp+4),nvec)
      enddo

      ! Second derivatives of the Reynolds-stress tensor components
      call gradm1(pstat_rutmp(1,1,1),pstat_rutmp(1,1,20), ! d2uudx2 (tmp 1)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,2))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,2), ! d2uudy2 (tmp 2)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,3))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2uudz2 (tmp 3)
     $     pstat_rutmp(1,1,3),pstat_rutmp(1,1,4))
      call gradm1(pstat_rutmp(1,1,4),pstat_rutmp(1,1,20), ! d2vvdx2 (tmp 4)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,5))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,5), ! d2vvdy2 (tmp 5)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,6))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2vvdz2 (tmp 6)
     $     pstat_rutmp(1,1,6),pstat_rutmp(1,1,7))
      call gradm1(pstat_rutmp(1,1,7),pstat_rutmp(1,1,20), ! d2wwdx2 (tmp 7)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,8))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,8), ! d2wwdy2 (tmp 8)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,9))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2wwdz2 (tmp 9)
     $     pstat_rutmp(1,1,9),pstat_rutmp(1,1,10))
      call gradm1(pstat_rutmp(1,1,10),pstat_rutmp(1,1,20), ! d2uvdx2 (tmp 10)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,11))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,11), ! d2uvdy2 (tmp 11)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,12))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2uvdz2 (tmp 12)
     $     pstat_rutmp(1,1,12),pstat_rutmp(1,1,13))
      call gradm1(pstat_rutmp(1,1,13),pstat_rutmp(1,1,20), ! d2vwdx2 (tmp 13)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,14))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,14), ! d2vwdy2 (tmp 14)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,15))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2vwdz2 (tmp 15)
     $     pstat_rutmp(1,1,15),pstat_rutmp(1,1,16))
      call gradm1(pstat_rutmp(1,1,16),pstat_rutmp(1,1,20), ! d2uwdx2 (tmp 16)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,17))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,17), ! d2uwdy2 (tmp 17)
     $     pstat_rutmp(1,1,21),pstat_rutmp(1,1,18))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! d2uwdz2 (tmp 18)
     $     pstat_rutmp(1,1,18),pstat_rutmp(1,1,19))

      ! Viscous diffusion tensor
      ! VDxx (new 16), VDyy (new 17), VDzz (new 18), VDxy (new 19), VDyz (new 20), VDxz (new 21)
      rtmp = param(2)
      do il = 1,6
         itmp = (il-1)*3
         call add4(pstat_runew(1,1,15+il),pstat_rutmp(1,1,itmp+1),
     $        pstat_rutmp(1,1,itmp+2),pstat_rutmp(1,1,itmp+3),nvec)
         call cmult(pstat_runew(1,1,15+il),rtmp,nvec)
      enddo

      ! Derivatives of the triple-product terms
      call gradm1(pstat_rutmp(1,1,1),pstat_rutmp(1,1,20), ! duuudx (tmp 1)
     $     pstat_rutmp(1,1,21),pstat_ruavg(1,1,24))
      call gradm1(pstat_rutmp(1,1,4),pstat_rutmp(1,1,11), ! duvvdx (tmp 4), duvvdy (tmp 11)
     $     pstat_rutmp(1,1,20),pstat_ruavg(1,1,30))
      call gradm1(pstat_rutmp(1,1,7),pstat_rutmp(1,1,20), ! duwwdx (tmp 7), duwwdz (tmp 18)
     $     pstat_rutmp(1,1,15),pstat_ruavg(1,1,32))
      call gradm1(pstat_rutmp(1,1,10),pstat_rutmp(1,1,2), ! duuvdx (tmp 10), duuvdy (tmp 2)
     $     pstat_rutmp(1,1,20),pstat_ruavg(1,1,28))
      call gradm1(pstat_rutmp(1,1,13),pstat_rutmp(1,1,20), ! duuwdx (tmp 16), duuwdz (tmp 3)
     $     pstat_rutmp(1,1,3),pstat_ruavg(1,1,29))
      call gradm1(pstat_rutmp(1,1,16),pstat_rutmp(1,1,14), ! duvwdx (tmp 13), duvwdy (tmp 17), duvwdz (tmp 12)
     $     pstat_rutmp(1,1,12),pstat_ruavg(1,1,34))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,5), ! dvvvdy (tmp 5)
     $     pstat_rutmp(1,1,21),pstat_ruavg(1,1,25))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,8), ! dvwwdy (tmp 8), dvwwdz (tmp 15)
     $     pstat_rutmp(1,1,18),pstat_ruavg(1,1,33))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,17), ! dvvwdy (tmp 14), dvvwdz (tmp 6)
     $     pstat_rutmp(1,1,6),pstat_ruavg(1,1,31))
      call gradm1(pstat_rutmp(1,1,20),pstat_rutmp(1,1,21), ! dwwwdz (tmp 9)
     $     pstat_rutmp(1,1,9),pstat_ruavg(1,1,26))

      ! Turbulent transport tensor
      ! Txx (new 22), Tyy (new 23), Tzz (new 24), Txy (new 25), Tyz (new 26), Txz (new 27)
      do il = 1,6
         itmp = (il-1)*3
         call add4(pstat_runew(1,1,21+il),pstat_rutmp(1,1,itmp+1),
     $        pstat_rutmp(1,1,itmp+2),pstat_rutmp(1,1,itmp+3),nvec)
         call chsign(pstat_runew(1,1,21+il),nvec)
      enddo

      ! Pressure strain tensor
      ! P*dUdx (tmp 13), P*dUdy (tmp 14), P*dUdz (tmp 15), P*dVdx (tmp 16), P*dVdy (tmp 17), P*dVdz (tmp 18),
      ! P*dWdx (tmp 19), P*dWdy (tmp 20), P*dWdz (tmp 21)
      ! pdudx (tmp 1), pdudy (tmp 2), pdudz (tmp 3), pdvdx (tmp 4), pdvdy (tmp 5), pdvdz (tmp 6),
      ! pdwdx (tmp 7), pdwdy (tmp 8), pdwdz (tmp 9)
      do il = 1,9
         call col3(pstat_rutmp(1,1,12+il),pstat_ruavg(1,1,4), ! P*d[UVW]d[xyz]
     $        pstat_runew(1,1,il),nvec)
         call sub3(pstat_rutmp(1,1,il),pstat_ruavg(1,1,14+il),
     $        pstat_rutmp(1,1,12+il),nvec)
      enddo

      ! diagonal terms
      ! PSxx (avg 18), PSyy (avg 19), PSzz (avg 20)
      rtmp = -2.0/rho
      do il = 1,3
         itmp = (il-1)*4
         call copy(pstat_ruavg(1,1,17+il),pstat_rutmp(1,1,itmp+1),nvec)
         call cmult(pstat_ruavg(1,1,17+il),rtmp,nvec)
      enddo

      ! off diagonal terms
      rtmp = -1.0/rho
      call add3(pstat_ruavg(1,1,21),pstat_rutmp(1,1,2), ! PSxy (avg 21)
     $     pstat_rutmp(1,1,4),nvec)
      call cmult(pstat_ruavg(1,1,21),rtmp,nvec)
      call add3(pstat_ruavg(1,1,22),pstat_rutmp(1,1,3), ! PSxz (avg 22)
     $     pstat_rutmp(1,1,7),nvec)
      call cmult(pstat_ruavg(1,1,22),rtmp,nvec)
      call add3(pstat_ruavg(1,1,23),pstat_rutmp(1,1,6), ! PSyz (avg 23)
     $     pstat_rutmp(1,1,8),nvec)
      call cmult(pstat_ruavg(1,1,23),rtmp,nvec)

      ! Derivatives of the pressure-velocity products
      ! dpudx (tmp 1), dpudy (tmp 2), dpudz (tmp 3), dpvdx (tmp 4), dpvdy (tmp 5), dpvdz (tmp 6),
      ! dpwdx (tmp 7), dpwdy (tmp 8), dpwdz (tmp 9)
      do il = 1,3
         itmp = (il-1)*3
         call gradm1(pstat_rutmp(1,1,itmp+1),pstat_rutmp(1,1,itmp+2),
     $     pstat_rutmp(1,1,itmp+3),pstat_ruavg(1,1,11+il))
      enddo

      ! Derivatives of the mean pressure field
      call gradm1(pstat_rutmp(1,1,10),pstat_rutmp(1,1,11), ! dPdx (tmp 10), dPdy (tmp 11), dPdz (tmp 12)
     $     pstat_rutmp(1,1,12),pstat_ruavg(1,1,4))

      ! Pressure transport tensor
      ! update dp[uvw]d[xyz]
      do il = 1,9
         call sub2(pstat_rutmp(1,1,il),pstat_rutmp(1,1,12+il),nvec)
      enddo
      do il = 1,3
         itmp = (il-1)*3
         do jl = 1,3
            call subcol3(pstat_rutmp(1,1,itmp+jl),pstat_ruavg(1,1,il),
     $           pstat_rutmp(1,1,9+jl),nvec)
         enddo
      enddo

      ! diagonal terms
      ! PTxx (avg 12), PTyy (avg 13) PTzz (avg 14)
      rtmp = -2.0/rho
      do il=1,3
         itmp = (il-1)*4
         call copy(pstat_ruavg(1,1,11+il),pstat_rutmp(1,1,itmp+1),nvec)
         call cmult(pstat_ruavg(1,1,11+il),rtmp,nvec)
      enddo
      ! off diagonal terms
      rtmp = -1.0/rho
      call add3(pstat_ruavg(1,1,15),pstat_rutmp(1,1,2), ! PTxy (avg 15)
     $     pstat_rutmp(1,1,4),nvec)
      call cmult(pstat_ruavg(1,1,15),rtmp,nvec)
      call add3(pstat_ruavg(1,1,16),pstat_rutmp(1,1,3), ! PTxz (avg 16)
     $     pstat_rutmp(1,1,7),nvec)
      call cmult(pstat_ruavg(1,1,16),rtmp,nvec)
      call add3(pstat_ruavg(1,1,17),pstat_rutmp(1,1,6), ! PTyz (avg 17)
     $     pstat_rutmp(1,1,8),nvec)
      call cmult(pstat_ruavg(1,1,17),rtmp,nvec)

      ! Production tensor
      rtmp = -2.0
      call vdot3(pstat_rutmp(1,1,1), ! Pxx (tmp 1)
     $     pstat_ruavg(1,1,5),pstat_ruavg(1,1,9),pstat_ruavg(1,1,11),
     $     pstat_runew(1,1,1),pstat_runew(1,1,2),pstat_runew(1,1,3),
     $     nvec)
      call cmult(pstat_rutmp(1,1,1),rtmp,nvec)
      call vdot3(pstat_rutmp(1,1,2), ! Pyy (tmp 2)
     $     pstat_ruavg(1,1,9),pstat_ruavg(1,1,6),pstat_ruavg(1,1,10),
     $     pstat_runew(1,1,4),pstat_runew(1,1,5),pstat_runew(1,1,6),
     $     nvec)
      call cmult(pstat_rutmp(1,1,2),rtmp,nvec)
      call vdot3(pstat_rutmp(1,1,3), ! Pzz (tmp 3)
     $     pstat_ruavg(1,1,11),pstat_ruavg(1,1,10),pstat_ruavg(1,1,7),
     $     pstat_runew(1,1,7),pstat_runew(1,1,8),pstat_runew(1,1,9),
     $     nvec)
      call cmult(pstat_rutmp(1,1,3),rtmp,nvec)
      call vdot3(pstat_rutmp(1,1,4), ! Pxy (tmp 4)
     $     pstat_ruavg(1,1,9),pstat_ruavg(1,1,6),pstat_ruavg(1,1,10),
     $     pstat_runew(1,1,1),pstat_runew(1,1,2),pstat_runew(1,1,3),
     $     nvec)
      call vdot3(pstat_rutmp(1,1,5),
     $     pstat_ruavg(1,1,5),pstat_ruavg(1,1,9),pstat_ruavg(1,1,11),
     $     pstat_runew(1,1,4),pstat_runew(1,1,5),pstat_runew(1,1,6),
     $     nvec)
      call add2(pstat_rutmp(1,1,4),pstat_rutmp(1,1,5),nvec)
      call chsign(pstat_rutmp(1,1,4),nvec)
      call vdot3(pstat_rutmp(1,1,5), ! Pxz (tmp 5)
     $     pstat_ruavg(1,1,11),pstat_ruavg(1,1,10),pstat_ruavg(1,1,7),
     $     pstat_runew(1,1,1),pstat_runew(1,1,2),pstat_runew(1,1,3),
     $     nvec)
      call vdot3(pstat_rutmp(1,1,6),
     $     pstat_ruavg(1,1,5),pstat_ruavg(1,1,9),pstat_ruavg(1,1,11),
     $     pstat_runew(1,1,7),pstat_runew(1,1,8),pstat_runew(1,1,9),
     $     nvec)
      call add2(pstat_rutmp(1,1,5),pstat_rutmp(1,1,6),nvec)
      call chsign(pstat_rutmp(1,1,5),nvec)
      call vdot3(pstat_rutmp(1,1,6), ! Pyz (tmp 6)
     $     pstat_ruavg(1,1,11),pstat_ruavg(1,1,10),pstat_ruavg(1,1,7),
     $     pstat_runew(1,1,4),pstat_runew(1,1,5),pstat_runew(1,1,6),
     $     nvec)
      call vdot3(pstat_rutmp(1,1,7),
     $     pstat_ruavg(1,1,9),pstat_ruavg(1,1,6),pstat_ruavg(1,1,10),
     $     pstat_runew(1,1,7),pstat_runew(1,1,8),pstat_runew(1,1,9),
     $     nvec)
      call add2(pstat_rutmp(1,1,6),pstat_rutmp(1,1,7),nvec)
      call chsign(pstat_rutmp(1,1,6),nvec)

      ! Velocity-pressure-gradient tensor
      ! Pixx (tmp 7), Piyy (tmp 8), Pizz (tmp 9), Pixy (tmp 10), Pixz (tmp 11), Piyz (tmp 12)
      do il=1,6
         call sub3(pstat_rutmp(1,1,6+il),pstat_ruavg(1,1,11+il),
     $        pstat_ruavg(1,1,17+il),nvec)
      enddo

      ! TKE budget
      rtmp = 0.5
      call add4(pstat_rutmp(1,1,13),pstat_rutmp(1,1,1), ! Pk (tmp 13)
     $     pstat_rutmp(1,1,2),pstat_rutmp(1,1,3),nvec)
      call cmult(pstat_rutmp(1,1,13),rtmp,nvec)
      call add4(pstat_rutmp(1,1,14),pstat_ruavg(1,1,39), ! Dk (tmp 14)
     $     pstat_ruavg(1,1,40),pstat_ruavg(1,1,41),nvec)
      call cmult(pstat_rutmp(1,1,14),rtmp,nvec)
      call add4(pstat_rutmp(1,1,15),pstat_runew(1,1,22), ! Tk (tmp 15)
     $     pstat_runew(1,1,23),pstat_runew(1,1,24),nvec)
      call cmult(pstat_rutmp(1,1,15),rtmp,nvec)
      call add4(pstat_rutmp(1,1,16),pstat_runew(1,1,16), ! VDk (tmp 16)
     $     pstat_runew(1,1,17),pstat_runew(1,1,18),nvec)
      call cmult(pstat_rutmp(1,1,16),rtmp,nvec)
      call add4(pstat_rutmp(1,1,17),pstat_rutmp(1,1,7), ! Pik (tmp 17)
     $     pstat_rutmp(1,1,8),pstat_rutmp(1,1,9),nvec)
      call cmult(pstat_rutmp(1,1,17),rtmp,nvec)
      call add4(pstat_rutmp(1,1,18),pstat_runew(1,1,10), ! Ck (tmp 18)
     $     pstat_runew(1,1,11),pstat_runew(1,1,12),nvec)
      call cmult(pstat_rutmp(1,1,18),rtmp,nvec)
      call add4(pstat_rutmp(1,1,19),pstat_rutmp(1,1,13), ! Resk (tmp 19)
     $     pstat_rutmp(1,1,14),pstat_rutmp(1,1,15),nvec)
      call add3(pstat_rutmp(1,1,20),pstat_rutmp(1,1,16),
     $     pstat_rutmp(1,1,17),nvec)
      call sub2(pstat_rutmp(1,1,20),pstat_rutmp(1,1,18),nvec)
      call add2(pstat_rutmp(1,1,19),pstat_rutmp(1,1,20),nvec)

      ! write down fields
      call pstat3d_mfo()

      return
      end subroutine
!=======================================================================
!> @brief Interpolate int the set of points
!! @ingroup pstat3d
      subroutine pstat3d_interp
      implicit none

      include 'SIZE'
      include 'GEOM'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer ifpts       ! findpts flag
      real tol            ! interpolation tolerance
      integer nt
      integer npt_max     ! max communication set
      integer nxf, nyf, nzf  ! fine mesh for bb-test
      real bb_t           ! relative size to expand bounding boxes by
      real toldist
      parameter (toldist = 5e-6)

      integer rcode(lhis), proc(lhis),elid(lhis)
      real dist(lhis), rst(ldim*lhis)

      integer nfail, npass

      integer il                    ! loop index
      !integer nvec                  ! single field length

      ! functions
      integer iglsum

!#define DEBUG
#ifdef DEBUG
      character*3 str1, str2
      integer iunit, ierr, jl
      ! call number
      integer icalld
      save icalld
      data icalld /0/
#endif
!-----------------------------------------------------------------------
      ! read point position
      call pstat3d_mfi_interp

      ! initialise interpolation tool
      tol     = 5e-13
      nt       = lx1*ly1*lz1*lelt
      npt_max = 256
      nxf     = 2*lx1
      nyf     = 2*ly1
      nzf     = 2*lz1
      bb_t    = 0.01

      ! start interpolation tool on given mesh
      call fgslib_findpts_setup(ifpts,nekcomm,mp,ldim,xm1,ym1,zm1,
     &     lx1,ly1,lz1,nelt,nxf,nyf,nzf,bb_t,nt,nt,npt_max,tol)

      ! identify points
      call fgslib_findpts(ifpts,rcode,1,proc,1,elid,1,rst,ldim,dist,1,
     &     pstat_int_pts(1,1),ldim,pstat_int_pts(2,1),ldim,
     &     pstat_int_pts(ldim,1),ldim,pstat_npt)

      ! find problems with interpolation
      nfail = 0
      do il = 1,pstat_npt
         ! check return code
         if (rcode(il).eq.1) then
            if (sqrt(dist(il)).gt.toldist) nfail = nfail + 1
         elseif(rcode(il).eq.2) then
            nfail = nfail + 1
         endif
      enddo
      nfail = iglsum(nfail,1)

#ifdef DEBUG
      ! for testing
      ! to output refinement
      icalld = icalld+1
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='INTfpts.txt'//str1//'i'//str2)

      write(iunit,*) pstat_nptot, pstat_npt, nfail
      do il=1,pstat_npt
         write(iunit,*) il, proc(il), elid(il), rcode(il), dist(il),
     $        (rst(jl+(il-1)*ldim),jl=1,ldim)
      enddo

         close(iunit)
#endif

      if (nfail.gt.0) call mntr_abort(pstat_id,
     $     'pstat_interp: Points not mapped')

      ! Interpolate averaged fields
      do il=1,pstat_svar
         call fgslib_findpts_eval(ifpts,pstat_int_avg (1,il),1,
     &        rcode,1,proc,1,elid,1,rst,ndim,pstat_npt,
     &        pstat_ruavg(1,1,il))
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='INTavg.txt'//str1//'i'//str2)

      write(iunit,*) pstat_nptot, pstat_npt
      do il=1,pstat_npt
         write(iunit,*) il, (pstat_int_avg(il,jl),jl=1,4)
      enddo

      close(iunit)
#endif

      ! Interpolate tmp fields
      do il=1,pstat_tvar
         call fgslib_findpts_eval(ifpts,pstat_int_tmp (1,il),1,
     &        rcode,1,proc,1,elid,1,rst,ndim,pstat_npt,
     &        pstat_rutmp(1,1,il))
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='INTtmp.txt'//str1//'i'//str2)

      write(iunit,*) pstat_nptot, pstat_npt
      do il=1,pstat_npt
         write(iunit,*) il, (pstat_int_tmp(il,jl),jl=1,4)
      enddo

      close(iunit)
#endif

      ! Interpolate new fields
      do il=1,pstat_dvar
         call fgslib_findpts_eval(ifpts,pstat_int_new (1,il),1,
     &        rcode,1,proc,1,elid,1,rst,ndim,pstat_npt,
     &        pstat_runew(1,1,il))
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='INTnew.txt'//str1//'i'//str2)

      write(iunit,*) pstat_nptot, pstat_npt
      do il=1,pstat_npt
         write(iunit,*) il, (pstat_int_new(il,jl),jl=1,4)
      enddo

      close(iunit)
#endif
         

      ! finalise interpolation tool
      call fgslib_findpts_free(ifpts)

      ! write down interpolated values
      call pstat3d_mfo_interp

#undef DEBUG
      return
      end subroutine
!=======================================================================
