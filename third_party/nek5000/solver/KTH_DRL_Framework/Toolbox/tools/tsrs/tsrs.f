!> @file tsrs.f
!! @ingroup tsrs
!! @brief Routines for time series module
!! @details This is a set of routines to generate statistics related history  
!!  points. It is just a modiffication of hpts, but uses binary files and point localisation
!! @author Adam Peplinski
!! @date May 24, 2021
!=======================================================================
!> @brief Register point time seriesmodule for statistics tool
!! @ingroup tsrs
!! @note This routine should be called in frame_usr_register
      subroutine tsrs_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'TSRSD'

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
      call mntr_mod_is_name_reg(lpmid,tsrs_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(tsrs_name)//'] already registered')
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
      call mntr_mod_reg(tsrs_id,lpmid,tsrs_name,
     $      'point time series')

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      ! total time
      call mntr_tmr_reg(tsrs_tmr_tot_id,lpmid,tsrs_id,
     $     'TSRS_TOT','Time series total time',.false.)
      lpmid = tsrs_tmr_tot_id
      ! initialisation
      call mntr_tmr_reg(tsrs_tmr_ini_id,lpmid,tsrs_id,
     $     'TSRS_INI','Time seires initialisation time',.true.)
      call mntr_tmr_reg(tsrs_tmr_cvp_id,lpmid,tsrs_id,
     $     'TSRS_CVP','Vorticity and pressure calc. time',.true.)
      ! interpolation
      call mntr_tmr_reg(tsrs_tmr_int_id,lpmid,tsrs_id,
     $     'TSRS_INT','Time series interpolation time',.true.)
      ! buffering
      call mntr_tmr_reg(tsrs_tmr_bfr_id,lpmid,tsrs_id,
     $     'TSRS_BFR','Time series buffering time',.true.)
      ! I/O
      call mntr_tmr_reg(tsrs_tmr_io_id,lpmid,tsrs_id,
     $     'TSRS_IO','Time series I/O time',.true.)

      ! register and set active section
      call rprm_sec_reg(tsrs_sec_id,tsrs_id,'_'//adjustl(tsrs_name),
     $     'Runtime paramere section for time series module')
      call rprm_sec_set_act(.true.,tsrs_sec_id)

      ! register parameters
      call rprm_rp_reg(tsrs_smpstep_id,tsrs_sec_id,'SMPSTEP',
     $     'Frequency of sampling',rpar_int,10,0.0,.false.,' ')

      ! set initialisation flag
      tsrs_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tsrs_tmr_tot_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise time series module
!! @ingroup tsrs
!! @note This routine should be called in frame_usr_init
      subroutine tsrs_init()
      implicit none

      include 'SIZE'
      include 'GEOM'
      include 'TSTEP'
      include 'FRAMELP'
      include 'TSRSD'

      ! global memory access
      integer nidd,npp,nekcomm,nekgroup,nekreal
      common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal

      ! local variables
      integer itmp
      real rtmp
      logical ltmp
      character*20 ctmp
      integer ntot
      integer npt_max, nxf, nyf, nzf
      real ltim
      real tol, bb_t            ! interpolation tolerance and relative size to expand bounding boxes by
      parameter (tol = 5.0E-13, bb_t = 0.1)

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (tsrs_ifinit) then
         call mntr_warn(tsrs_id,
     $        'module ['//trim(tsrs_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,tsrs_smpstep_id,rpar_int)
      tsrs_smpstep = itmp

      ! initialise findpts
      ntot = lx1*ly1*lz1*lelt 
      npt_max = 256
      nxf = 2*lx1 ! fine mesh for bb-test
      nyf = 2*ly1
      nzf = 2*lz1

      call fgslib_findpts_setup(tsrs_handle,nekcomm,npp,ldim,
     $     xm1,ym1,zm1,nx1,ny1,nz1,nelt,nxf,nyf,nzf,bb_t,ntot,ntot,
     $     npt_max,tol)

      ! read and redistribute points among processors
      call tsrs_read_redistribute()

      ! clean the buffer
      tsrs_ntsnap = 0
      ntot = tsrs_nfld*lhis*tsrs_ltsnap
      call rzero(tsrs_buff,ntot)
      call rzero(tsrs_tmlist,tsrs_ltsnap)

      ! everything is initialised
      tsrs_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tsrs_tmr_ini_id,1,ltim)
      
      return
      end subroutine
!=======================================================================
!> @brief Finalise time series module
!! @ingroup tsrs
!! @note This routine should be called in frame_usr_end
      subroutine tsrs_end()
      implicit none

      include 'SIZE'
      include 'TSRSD'

      ! local variables
      logical ifapp, ifsave
!-----------------------------------------------------------------------
      ! make sure all data in the buffer is saved
      ifapp = .FALSE.
      ifsave = .TRUE.
      call tsrs_buffer_save(ifapp, ifsave)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup tsrs
!! @return tsrs_is_initialised
      logical function tsrs_is_initialised()
      implicit none

      include 'SIZE'
      include 'TSRSD'
!-----------------------------------------------------------------------
      tsrs_is_initialised = tsrs_ifinit

      return
      end function
!=======================================================================
!> @brief Main interface of time series module 
!! @ingroup tsrs
!! @details This routine calls interpolation routine at proper step and
!!   allows a user to write down data in a current step if correlation
!!   with some other package is required.
!! @param[in]   ifsave     force I/O operation at a given step
      subroutine tsrs_main(ifsave)
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'TSRSD'

      ! argument list
      logical ifsave

      ! local variables
      integer itmp

      ! simple timing
      real ltim

      ! number of steps to be descarded in the simulation beginning
      ! This is necessary due to multistep restart scheme as 2-3 first steps
      ! are in general repeated from the previous simulation.
      ! It does not produce any problem in case of AMR, as in this case
      ! those first steps are skept anyway.
      ! For now the number is simply hard-coded to the max of time integration
      ! order
      integer step_skip
      parameter (step_skip=lorder)

      logical ifapp

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! skip initial steps
      if (ISTEP.gt.step_skip) then
        itmp = ISTEP - step_skip

        ! sample fields
        if (mod(itmp,tsrs_smpstep).eq.0) then
           call tsrs_get()
        endif
        ! is I/O required
        if (ifsave) then
           ifapp = .FALSE.
           call tsrs_buffer_save(ifapp, ifsave)
        endif
      endif
      
      return
      end subroutine
!=======================================================================
!> @brief Perform interpolation and data buffering
!! @ingroup tsrs
!! @details This routine performs interpolation on set of points, buffering
!!     and file writing.
      subroutine tsrs_get()
      implicit none

      include 'SIZE'
      include 'SOLN'
      include 'TSRSD'

      ! global variables
      ! work arrays
      real slvel(LX1,LY1,LZ1,LELT,3)
      common /SCRMG/ slvel
      real tmpvel(LX1,LY1,LZ1,LELT,3), tmppr(LX1,LY1,LZ1,LELT)
      common /SCRUZ/ tmpvel, tmppr
      real dudx(LX1,LY1,LZ1,LELT,3) ! du/dx, du/dy and du/dz
      real dvdx(LX1,LY1,LZ1,LELT,3) ! dv/dx, dv/dy and dv/dz
      real dwdx(LX1,LY1,LZ1,LELT,3) ! dw/dx, dw/dy and dw/dz
      common /SCRNS/ dudx, dvdx
      common /SCRSF/ dwdx

      ! local variables
      real ltim
      logical ifapp, ifsave

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ltim = dnekclock()
      ! Map pressure to velocity mesh
      call mappr(tmppr,PR,tmpvel(1,1,1,1,2),tmpvel(1,1,1,1,3))

      ! Compute derivative tensor and normalise pressure
      call user_stat_trnsv(tmpvel,dudx,dvdx,dwdx,slvel,tmppr)
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tsrs_tmr_cvp_id,1,ltim)

      ltim = dnekclock()
      ! interpolate variables
      call tsrs_interpolate(tmpvel,slvel,tmppr)
      ltim = dnekclock() - ltim
      call mntr_tmr_add(tsrs_tmr_int_id,1,ltim)

      ifapp = .TRUE.
      ifsave = .FALSE.
      call tsrs_buffer_save(ifapp, ifsave)

      return
      end subroutine
!=======================================================================
!> @brief Read and redistribute points among mpi ranks
!! @ingroup tsrs
!! @details
      subroutine tsrs_read_redistribute()
      implicit none

      include 'SIZE'
      include 'TSRSD'
!#define DEBUG
#ifdef DEBUG
      include 'INPUT'
      include 'GEOM'
#endif

      ! local variables
      integer il
      integer nfail, npass
      real toldist
      parameter (toldist = 5e-6)
      integer nptimb            ! assumed point imbalance

      ! functions
      integer iglsum

#ifdef DEBUG
      character*3 str1, str2
      integer iunit, ierr, jl
      ! call number
      integer icalld
      save icalld
      data icalld /0/
      real coord_int(ldim,lhis)
      integer rcode(lhis),proc(lhis),elid(lhis)
      real dist(lhis),rst(ldim*lhis)
      ! timng variables
      real ltime1, ltime2
      ! functions
      real dnekclock
#endif
!-----------------------------------------------------------------------
      ! read in point position
      call tsrs_mfi_points()

      ! identify points
      call fgslib_findpts(tsrs_handle,tsrs_rcode,1,tsrs_proc,1,
     $     tsrs_elid,1,tsrs_rst,ldim,tsrs_dist,1,
     &     tsrs_pts(1,1),ldim,tsrs_pts(2,1),ldim,
     &     tsrs_pts(ldim,1),ldim,tsrs_npts)

      ! find problems with interpolation
      nfail = 0
      do il = 1,tsrs_npts
         ! check return code
         if (tsrs_rcode(il).eq.1) then
            if (sqrt(tsrs_dist(il)).gt.toldist) nfail = nfail + 1
         elseif(tsrs_rcode(il).eq.2) then
            nfail = nfail + 1
         endif
      enddo
      nfail = iglsum(nfail,1)

#ifdef DEBUG
      ! for testing
      ! to output findpts data
      icalld = icalld+1

      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSrpos.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts
      do il=1, tsrs_npts
         write(iunit,*) il, tsrs_ipts(il), (tsrs_pts(jl,il),jl=1,ldim)
      enddo

      close(iunit)

      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSfpts.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts, nfail
      do il=1,tsrs_npts
         write(iunit,*) il, tsrs_ipts(il), tsrs_proc(il), tsrs_elid(il),
     $        tsrs_rcode(il), tsrs_dist(il),
     $        (tsrs_rst(jl+(il-1)*ldim),jl=1,ldim)
      enddo

      close(iunit)

      ! timing
      call nekgsync()
      ltime1 = dnekclock()
      ! interpolate coordinates to see interpolation quality
      call fgslib_findpts_eval(tsrs_handle,coord_int(1,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,xm1)
      call fgslib_findpts_eval(tsrs_handle,coord_int(2,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,ym1)
      if (if3d) call fgslib_findpts_eval(tsrs_handle,coord_int(ldim,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,zm1)
      call nekgsync()
      ltime1 = dnekclock() - ltime1
      
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSintp.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts
      do il=1, tsrs_npts
         write(iunit,*) il, tsrs_ipts(il),
     $        (tsrs_pts(jl,il)-coord_int(jl,il),jl=1,ldim)
      enddo

      close(iunit)
#endif

      ! redistribute points to minimise communication
      nptimb = 0
      call pts_rdst(nptimb)

#ifdef DEBUG
      ! to check interpolation parametes correctess after transfer
      call icopy(rcode,tsrs_rcode,lhis)
      call icopy(proc,tsrs_proc,lhis)
      call icopy(elid,tsrs_elid,lhis)
      call copy(dist,tsrs_dist,lhis)
      call copy(rst,tsrs_rst,ldim*lhis)
#endif
      
      ! Interesting; Even though all interpolation data seems to be fine I have to reinitialise it here
      ! It looks like gslib has some internal data that has to be wahsed up; Must be new stuff not present in older gslib versions
      call fgslib_findpts(tsrs_handle,tsrs_rcode,1,tsrs_proc,1,
     $     tsrs_elid,1,tsrs_rst,ldim,tsrs_dist,1,
     &     tsrs_pts(1,1),ldim,tsrs_pts(2,1),ldim,
     &     tsrs_pts(ldim,1),ldim,tsrs_npts)

#ifdef DEBUG
      ! for testing
      ! to output findpts data after redistribution
      icalld = icalld+1

      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSrpos.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts
      do il=1, tsrs_npts
         write(iunit,*) il, tsrs_ipts(il), (tsrs_pts(jl,il),jl=1,ldim)
      enddo

      close(iunit)

      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSfpts.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts, nfail
      do il=1,tsrs_npts
         write(iunit,*) il, tsrs_ipts(il), tsrs_proc(il), tsrs_elid(il), ! new interpolation parameters
     $        tsrs_rcode(il), tsrs_dist(il),
     $        (tsrs_rst(jl+(il-1)*ldim),jl=1,ldim)
         write(iunit,*) il, tsrs_ipts(il), proc(il), elid(il), ! old interpolation parameters transferred by pts_rdst
     $        rcode(il), dist(il),                             ! notice dist can be different as it is not transferred (not needed findpts_eval)
     $        (rst(jl+(il-1)*ldim),jl=1,ldim)
      enddo

      close(iunit)

      ! timing
      call nekgsync()
      ltime2 = dnekclock()
      ! interpolate coordinates to see interpolation quality
      call fgslib_findpts_eval(tsrs_handle,coord_int(1,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,xm1)
      call fgslib_findpts_eval(tsrs_handle,coord_int(2,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,ym1)
      if (if3d) call fgslib_findpts_eval(tsrs_handle,coord_int(ldim,1),
     $     ldim,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,zm1)
      call nekgsync()
      ltime2 = dnekclock() - ltime2
      
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='TSRSintp.txt'//str1//'i'//str2)

      write(iunit,*) tsrs_nptot, tsrs_npts,ltime1,ltime2
      do il=1, tsrs_npts
         write(iunit,*) il, tsrs_ipts(il),
     $        (tsrs_pts(jl,il)-coord_int(jl,il),jl=1,ldim)
      enddo

      close(iunit)
#endif
#undef DEBUG

      if (nfail.gt.0) call mntr_abort(tsrs_id,
     $     'tsrs_read_redistribute: Points not mapped')

      return
      end subroutine
!=======================================================================
!> @brief Interpolate fields on a set of points
!! @ingroup tsrs
!! @param[in]   vlct    velocity field
!! @param[in]   vort    vorticity filed
!! @param[in]   pres    pressure filed
      subroutine tsrs_interpolate(vlct,vort,pres)
      implicit none

      include 'SIZE'
      include 'TSRSD'
      
      ! argument list
      real vlct(lx1*ly1*lz1*lelt,ldim)
      real vort(lx1*ly1*lz1*lelt,ldim)
      real pres(lx1*ly1*lz1*lelt)

      ! local variables
      integer ifld, il
!-----------------------------------------------------------------------
      ifld = 0
      ! velocity
      do il = 1, ldim
         ifld = ifld + 1
         call fgslib_findpts_eval(tsrs_handle,tsrs_fld(ifld,1),
     $        tsrs_nfld,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $        tsrs_rst,ldim,tsrs_npts,vlct(1,il))
      enddo

      ! pressure
      ifld = ifld + 1
      call fgslib_findpts_eval(tsrs_handle,tsrs_fld(ifld,1),
     $     tsrs_nfld,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $     tsrs_rst,ldim,tsrs_npts,pres)

      ! vorticity
      do il = 1, ldim
         ifld = ifld + 1
         call fgslib_findpts_eval(tsrs_handle,tsrs_fld(ifld,1),
     $        tsrs_nfld,tsrs_rcode,1,tsrs_proc,1,tsrs_elid,1,
     $        tsrs_rst,ldim,tsrs_npts,vort(1,il))
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Buffer and save interpolated fields.
!! @ingroup tsrs
!! @param[in]   ifapp       do we append buffer
!! @param[in]   ifsave      save and clean the buffer in current call
!! @remark This routine uses global scratch space \a SCRCH
      subroutine tsrs_buffer_save(ifapp, ifsave)
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'TSRSD'

      ! global memory space
      ! dummy arrays
      integer lbff, nbff
      parameter (lbff=tsrs_nfld*lhis*tsrs_ltsnap)
      real bff(lbff)
      common /SCRCH/  bff

      ! argument list
      logical ifapp, ifsave

      ! local variables
      integer ntot, il, jl
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! append buffer
      if (ifapp) then
         ltim = dnekclock()
         tsrs_ntsnap = tsrs_ntsnap + 1
         ntot = tsrs_nfld*tsrs_npts
         call copy(tsrs_buff(1,1,tsrs_ntsnap),tsrs_fld,ntot)
         tsrs_tmlist(tsrs_ntsnap) = time
         ltim = dnekclock() - ltim
         call mntr_tmr_add(tsrs_tmr_bfr_id,1,ltim)
      endif

      ! save data and clean buffer
      if(ifsave.or.(tsrs_ntsnap.eq.tsrs_ltsnap)) then
         if (tsrs_ntsnap.gt.0) then
            ltim = dnekclock()

            ! reshuffle storage to simplify I/O
            nbff = 0
            do il = 1, tsrs_npts
               do jl = 1, tsrs_ntsnap
                  call copy(bff(nbff+1),tsrs_buff(1,il,jl),tsrs_nfld)
                  nbff = nbff + tsrs_nfld
               enddo
            enddo
            call tsrs_mfo_outfld(bff,nbff)

            ! clean the buffer
            tsrs_ntsnap = 0
            ntot = tsrs_nfld*lhis*tsrs_ltsnap
            call rzero(tsrs_buff,ntot)
            call rzero(tsrs_tmlist,tsrs_ltsnap)

            ltim = dnekclock() - ltim
            call mntr_tmr_add(tsrs_tmr_io_id,1,ltim)
         endif
      endif

      return
      end subroutine
!=======================================================================







