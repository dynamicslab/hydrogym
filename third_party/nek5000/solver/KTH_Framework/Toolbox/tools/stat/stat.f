!> @file stat.f
!! @ingroup stat
!! @brief 2D and 3D statistics module
!! @details 
!! @note This code works for extruded meshes only
!! @author Prabal Negi, Adam Peplinski
!! @date Aug 15, 2018
!=======================================================================
!> @brief Register 2D and 3D statistics module
!! @ingroup stat
!! @note This routine should be called in frame_usr_register
      subroutine stat_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'STATD'

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
      call mntr_mod_is_name_reg(lpmid,stat_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(stat_name)//'] already registered')
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
      call mntr_mod_reg(stat_id,lpmid,stat_name,
     $      '2D and 3D statistics')

      ! check if 2D mapping module is registered
      if (stat_rdim.eq.1) then
         call mntr_mod_is_name_reg(lpmid,'MAP2D')
         ! if not, register module
         if (lpmid.le.0) then
            call map2D_register()
         endif
      endif

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      ! total time
      call mntr_tmr_reg(stat_tmr_tot_id,lpmid,stat_id,
     $     'STAT_TOT','Statistics total time',.false.)
      lpmid = stat_tmr_tot_id
      ! initialisation
      call mntr_tmr_reg(stat_tmr_ini_id,lpmid,stat_id,
     $     'STAT_INI','Statistics initialisation time',.true.)
      ! averagign
      call mntr_tmr_reg(stat_tmr_avg_id,lpmid,stat_id,
     $     'STAT_AVG','Statistics averaging time',.true.)

      if (stat_rdim.eq.1) then
      ! communication
         call mntr_tmr_reg(stat_tmr_cmm_id,lpmid,stat_id,
     $        'STAT_CMM','Statistics communication time',.true.)
      endif
      ! IO
      call mntr_tmr_reg(stat_tmr_io_id,lpmid,stat_id,
     $     'STAT_IO','Statistics IO time',.true.)

      ! register and set active section
      call rprm_sec_reg(stat_sec_id,stat_id,'_'//adjustl(stat_name),
     $     'Runtime paramere section for statistics module')
      call rprm_sec_set_act(.true.,stat_sec_id)

      ! register parameters
      call rprm_rp_reg(stat_avstep_id,stat_sec_id,'AVSTEP',
     $     'Frequency of averaging',rpar_int,10,0.0,.false.,' ')

      call rprm_rp_reg(stat_IOstep_id,stat_sec_id,'IOSTEP',
     $     'Frequency of filed saving',rpar_int,100,0.0,.false.,' ')
      
      ! set initialisation flag
      stat_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(stat_tmr_tot_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise statistics module
!! @ingroup stat
!! @note This routine should be called in frame_usr_init
      subroutine stat_init()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'TSTEP'
      include 'MAP2D'
      include 'STATD'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (stat_ifinit) then
         call mntr_warn(stat_id,
     $        'module ['//trim(stat_name)//'] already initiaised.')
         return
      endif

      ! check if map2d module is initialised
      if (stat_rdim.eq.1) then
         if (.not.map2d_ifinit) call map2d_init()
      endif

      ! timing
      ltim = dnekclock()
      
      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,stat_avstep_id,rpar_int)
      stat_avstep = itmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,stat_IOstep_id,rpar_int)
      stat_IOstep = itmp

      ! initialise time averaging variables for a given statistics file
      stat_atime = 0.0
      stat_tstart = time

      if (stat_rdim.eq.1) then
         ! finish 3D => 2D mapping operations
         ! get local integration coefficients
         call mntr_log(stat_id,lp_vrb,'Getting loacal int. coeff.')
         call stat_init_int1D
      endif

      ! reset statistics variables
      itmp = lx1**(LDIM-stat_rdim)*lelt*stat_lvar
      call rzero(stat_ruavg,itmp)
      
      ! everything is initialised
      stat_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(stat_tmr_ini_id,1,ltim)
      
      return
      end subroutine
!=======================================================================
!> @brief Finalise statistics module
!! @ingroup stat
!! @note This routine should be called in frame_usr_end
      subroutine stat_end()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'STATD'

      ! local variables
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! make sure all data in the buffer is saved
      if (stat_atime.gt.0.0) then
         if (stat_rdim.eq.1) then
            ltim = dnekclock()
            call mntr_log(stat_id,lp_inf,'Global sum')
            call stat_gs_sum
            ltim = dnekclock() - ltim
            call mntr_tmr_add(stat_tmr_cmm_id,1,ltim)
         endif

         ltim = dnekclock()
         call mntr_log(stat_id,lp_inf,'Writing stat file')
         call stat_mfo
         ltim = dnekclock() - ltim
         call mntr_tmr_add(stat_tmr_io_id,1,ltim)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup stat
!! @return stat_is_initialised
      logical function stat_is_initialised()
      implicit none

      include 'SIZE'
      include 'STATD'
!-----------------------------------------------------------------------
      stat_is_initialised = stat_ifinit

      return
      end function
!=======================================================================
!> @brief Main interface of statistics module
!! @ingroup stat
!! @details This routine performs time averaging and file writing.
!! @note This routine should be called in userchk during every step
      subroutine stat_avg
      implicit none 

      include 'SIZE'
      include 'FRAMELP'
      include 'TSTEP'
      include 'STATD'

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

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! skip initial steps
      if (ISTEP.gt.step_skip) then
        itmp = ISTEP - step_skip

        ! average
        if (mod(itmp,stat_avstep).eq.0) then
          ! simple timing
          ltim = dnekclock()
          call mntr_log(stat_id,lp_inf,'Average compute')
          call stat_compute
          ! timing
          ltim = dnekclock() - ltim
          call mntr_tmr_add(stat_tmr_avg_id,1,ltim)
        endif

        ! save statistics file and restart statistics variables
        if (mod(itmp,stat_IOstep).eq.0) then
          if (stat_rdim.eq.1) then

              ltim = dnekclock()
              call mntr_log(stat_id,lp_inf,'Global sum')
              call stat_gs_sum
              ltim = dnekclock() - ltim
              call mntr_tmr_add(stat_tmr_cmm_id,1,ltim)
          endif

          ltim = dnekclock()
          call mntr_log(stat_id,lp_inf,'Writing stat file')
          call stat_mfo
          ltim = dnekclock() - ltim
          call mntr_tmr_add(stat_tmr_io_id,1,ltim)


          ! clean up array
          itmp = lx1**(LDIM-stat_rdim)*lelt*stat_lvar
          call rzero(stat_ruavg,itmp)

          ! reset averaging parameters
          stat_atime = 0.0
          stat_tstart = time
        endif
      else
         ! set averaging parameters
         stat_atime = 0.0
         stat_tstart = time
      endif

      return
      end subroutine
!====================================================================== 
!> @brief Get local integration coefficients
!! @ingroup stat
!! @details This version does 1D integration over one of the directions
!!  R,S,T. It supports curved coordinate systems, however 
!!  axisymmetric 2.5D cases are not supported
!! @remark This routine uses global scratch space \a SCRSF
      subroutine stat_init_int1D()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'WZ'              ! W?M1
      include 'GEOM'            ! ?M1
      include 'INPUT'           ! IFAXIS, IF3D
      include 'DXYZ'            ! D?M1, D?TM1
      include 'MAP2D'
      include 'STATD'

      ! global variables
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      ! scratch space
      real lxyzd(lx1,ly1,lz1,lelt,3)
      common /SCRSF/ lxyzd      ! coordinate derivatives

      ! local variables
      integer il, jl, kl, el    ! loop index
      integer el2               ! index of 2D element
      real lwm1(lx1)            ! wieghts for 1D integration

      ! global communication
      integer gs_handle         ! gather-scatter handle
      integer*8 unodes(lx1*lz1*lelt)    ! unique local nodes
!-----------------------------------------------------------------------
      if(ifaxis) call mntr_abort(stat_id,
     $        'stat_init_int1D; IFAXIS not supported')

      ! copy wieghts depending on the uniform direction
      if (map2d_idir.eq.1) then
         call copy(lwm1,wxm1,nx1)
         stat_nm1 = nx1
         stat_nm2 = ny1
         stat_nm3 = nz1
         ! get coordinates derivatives d[XYZ]/dr
         il = ny1*nz1
         do el = 1, nelt
            if(map2d_lmap(el).ne.-1) then
               call mxm(dxm1,nx1,xm1(1,1,1,el),nx1,lxyzd(1,1,1,el,1),il)
               call mxm(dxm1,nx1,ym1(1,1,1,el),nx1,lxyzd(1,1,1,el,2),il)
               if (if3d) call mxm(dxm1,nx1,zm1(1,1,1,el),nx1,
     $              lxyzd(1,1,1,el,3),il)
            endif
         enddo
      elseif (map2d_idir.eq.2) then
         call copy(lwm1,wym1,ny1)
         stat_nm1 = ny1
         stat_nm2 = nx1
         stat_nm3 = nz1
         ! get coordinates derivatives d[XYZ]/ds
         do el = 1, nelt
            if(map2d_lmap(el).ne.-1) then
               do il=1, nz1
                  call mxm(xm1(1,1,il,el),nx1,dytm1,ny1,
     $                 lxyzd(1,1,il,el,1),ny1)
                  call mxm(ym1(1,1,il,el),nx1,dytm1,ny1,
     $                 lxyzd(1,1,il,el,2),ny1)
                  if (if3d) call mxm(zm1(1,1,il,el),nx1,dytm1,ny1,
     $                 lxyzd(1,1,il,el,3),ny1)
               enddo
            endif
         enddo
      else
         if (if3d) then
            call copy(lwm1,wzm1,nz1)
            stat_nm1 = nz1
            stat_nm2 = nx1
            stat_nm3 = ny1
            ! get coordinates derivatives d[XYZ]/dt
            il = nx1*ny1
            do el = 1, nelt
               if(map2d_lmap(el).ne.-1) then
                  call mxm(xm1(1,1,1,el),il,dztm1,nz1,
     $                 lxyzd(1,1,1,el,1),nz1)
                  call mxm(ym1(1,1,1,el),il,dztm1,nz1,
     $                 lxyzd(1,1,1,el,2),nz1)
                  call mxm(zm1(1,1,1,el),il,dztm1,nz1,
     $                 lxyzd(1,1,1,el,3),nz1)
               endif
            enddo
         else
            call mntr_abort(stat_id,'2D run cannot be z averaged.')
         endif
      endif

      ! for now I assume lx1=stat_nm1=stat_nm2=stat_nm3
      ! check if that is true
      if (if3d) then
         if(lx1.ne.stat_nm1.or.lx1.ne.stat_nm2.or.lx1.ne.stat_nm3) then
            call mntr_abort(stat_id,
     $           'stat_init_int1D; unequal array sizes')
         endif
      else
         if(lx1.ne.stat_nm1.or.lx1.ne.stat_nm2.or.1.ne.stat_nm3) then
            call mntr_abort(stat_id,
     $           'stat_init_int1D; unequal array sizes')
         endif
      endif

      ! get 1D mass matrix ordering directions in such a way that
      ! the uniform direction corresponds to the the first index
      il = stat_nm1*stat_nm2*stat_nm3
      ! get arc length
      do el = 1, nelt
         if(map2d_lmap(el).ne.-1) then
            call vsq(lxyzd(1,1,1,el,1),il)
            call vsq(lxyzd(1,1,1,el,2),il)
            if (if3d) call vsq(lxyzd(1,1,1,el,3),il)
      
            call add2(lxyzd(1,1,1,el,1),lxyzd(1,1,1,el,2),il)
            if (if3d) call add2(lxyzd(1,1,1,el,1),lxyzd(1,1,1,el,3),il)

            call vsqrt(lxyzd(1,1,1,el,1),il)
         endif
      enddo

      il=il*nelt
      call rzero(stat_bm1d,il)

      ! reshuffle array
      call stat_reshufflev(stat_bm1d,lxyzd,nelt)

      ! multiply by wieghts
      do el=1, nelt
         if(map2d_lmap(el).ne.-1) then
            do kl=1, stat_nm3
               do jl=1, stat_nm2
                  do il=1, stat_nm1
                     stat_bm1d(il,jl,kl,el) =
     $                    lwm1(il)*stat_bm1d(il,jl,kl,el)
                  enddo
               enddo
            enddo
         endif
      enddo

      ! get total line length
      ! sum contributions from different 3D elements to get
      ! local arc length

      il = stat_nm2*stat_nm3*nelt
      call rzero(stat_abm1d,il)

      do el = 1, nelv
         el2 = map2d_lmap(el)
         if(el2.gt.0) then
            do kl=1, stat_nm3
               do jl=1, stat_nm2
                  do il=1, stat_nm1
                     stat_abm1d(jl,kl,el2) = stat_abm1d(jl,kl,el2) +
     $                    stat_bm1d(il,jl,kl,el)
                  enddo
               enddo
            enddo
         endif
      enddo

      ! Global communication to sum local contributions for arc lenght
      ! set up communicator
      el = stat_nm2*stat_nm3
      do il = 1,map2d_lnum
         kl = map2d_gmap(il) - 1
         do jl=1,el
            unodes(el*(il-1) + jl) = int(el,8)*int(kl,8) + jl
         enddo
      enddo
      kl = el*map2d_lnum
      call fgslib_gs_setup(gs_handle,unodes,kl,nekcomm,mp)

      call fgslib_gs_op(gs_handle,stat_abm1d,1,1,0)

      ! destroy communicator
      call fgslib_gs_free (gs_handle)

      return
      end subroutine
!=======================================================================
!> @brief Array reshuffle
!! @ingroup stat
!! @details Reorder directions in such a way that the uniform direction 
!!   corresponds to the the first index
!! @param[out]  rvar   reshuffled array
!! @param[in]   var    input array
!! @param[in]   nl     element number to reshuffle
      subroutine stat_reshufflev(rvar, var, nl)
      implicit none

      include 'SIZE'
      include 'MAP2D'
      include 'STATD'

      ! argument list
      real rvar(lx1,ly1,lz1,lelt) !reshuffled array
      real var(lx1,ly1,lz1,lelt) ! input array
      integer nl                ! element number to reshuffle

      ! local variables
      integer il, jl, kl, el    ! loop index
!-----------------------------------------------------------------------
      ! if no space averaging copy
      if (stat_rdim.eq.0) then
         el = lx1*ly1*lz1*lelt
         call copy(rvar,var,el)
      else
         ! if space averagign swap data
         if (map2d_idir.eq.1) then
            do el=1, nl
               if(map2d_lmap(el).ne.-1) then
                  do kl=1, stat_nm3
                     do jl=1, stat_nm2
                        do il=1, stat_nm1
                           rvar(il,jl,kl,el) = var(il,jl,kl,el)
                        enddo
                     enddo
                  enddo
               endif
            enddo
         elseif (map2d_idir.eq.2) then
            do el=1, nl
               if(map2d_lmap(el).ne.-1) then
                  do kl=1, stat_nm3
                     do jl=1, stat_nm2
                        do il=1, stat_nm1
                           rvar(il,jl,kl,el) = var(jl,il,kl,el)
                        enddo
                     enddo
                  enddo
               endif
            enddo
         else
            do el=1, nl
               if(map2d_lmap(el).ne.-1) then
                  do kl=1, stat_nm3
                     do jl=1, stat_nm2
                        do il=1, stat_nm1
                           rvar(il,jl,kl,el) = var(jl,kl,il,el)
                        enddo
                     enddo
                  enddo
               endif
            enddo
         endif
      endif
         
      return
      end subroutine
!=======================================================================
!> @brief Perform local 1D integration on 1 variable
!! @ingroup stat
!! @param[in]   lvar        integrated variable
!! @param[in]   npos        position in stat_ruavg
!! @param[in]   alpha,beta  time averaging parameters
!! @remark This routine uses global scratch space \a CTMP0
      subroutine stat_compute_1Dav1(lvar,npos,alpha,beta)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'STATD'

      ! argument list
      real lvar(lx1,ly1,lz1,lelt)
      integer npos
      real alpha, beta

      ! global variables
      real rtmp(lx1,lz1,lelt) ! dummy array
      common /CTMP0/ rtmp

      ! local variables
      integer il, jl, kl, el    ! loop index
!-----------------------------------------------------------------------
      ! consistency check
      if(npos.gt.stat_lvar)
     $     call mntr_abort(stat_id,'inconsistent npos ')

      if (stat_rdim.eq.1) then
         ! zero work array
         el = lx1*lz1*lelt
         call rzero(rtmp,el)

         ! perform 1D integral
         do el = 1, nelv
            do kl=1, stat_nm3
               do jl=1, stat_nm2
                  do il=1, stat_nm1
                     rtmp(jl,kl,el) = rtmp(jl,kl,el) +
     $                    stat_bm1d(il,jl,kl,el)*lvar(il,jl,kl,el)
                  enddo
               enddo
            enddo
         enddo

         ! time average
         el = stat_nm2*stat_nm3*nelv
         call add2sxy(stat_ruavg(1,1,npos),alpha,rtmp,beta,el)
      else
         el = lx1**(LDIM)*lelt
         call add2sxy(stat_ruavg(1,1,npos),alpha,lvar,beta,el)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Perform local 1D integration on multiplication of 2 variables
!! @ingroup stat
!! @param[in]   lvar1,lvar2 integrated variable
!! @param[in]   npos        position in stat_ruavg
!! @param[in]   alpha,beta  time averaging parameters
!! @remark This routine uses global scratch space \a CTMP0
      subroutine stat_compute_1Dav2(lvar1,lvar2,npos,alpha,beta)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'STATD'

      ! argument list
      real lvar1(lx1,ly1,lz1,lelt), lvar2(lx1,ly1,lz1,lelt)
      integer npos
      real alpha, beta

      ! global variables
      real rtmp(lx1,lz1,lelt)   ! dummy array
      real rtmp2(lx1,ly1,lz1,lelt) ! dummy array
      common /CTMP0/ rtmp, rtmp2

      ! local variables
      integer il, jl, kl, el    ! loop index
!-----------------------------------------------------------------------
      ! consistency check
      if(npos.gt.stat_lvar)
     $     call mntr_abort(stat_id,'inconsistent npos ')

      if (stat_rdim.eq.1) then
         ! zero work array
         el = lx1*lz1*lelt
         call rzero(rtmp,el)

         ! perform 1D integral
         do el = 1, nelv
            do kl=1, stat_nm3
               do jl=1, stat_nm2
                  do il=1, stat_nm1
                     rtmp(jl,kl,el) = rtmp(jl,kl,el) +
     $                    stat_bm1d(il,jl,kl,el)*
     $                    lvar1(il,jl,kl,el)*lvar2(il,jl,kl,el)
                  enddo
               enddo
            enddo
         enddo

         ! time average
         el = stat_nm2*stat_nm3*nelv
         call add2sxy(stat_ruavg(1,1,npos),alpha,rtmp,beta,el)
      else
         el = lx1**(LDIM)*lelt
         call col3(rtmp2,lvar1,lvar2,el)
         call add2sxy(stat_ruavg(1,1,npos),alpha,rtmp2,beta,el)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Global statistics summation
!! @ingroup stat
      subroutine stat_gs_sum
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'PARALLEL'
      include 'INPUT'           ! if3d
      include 'MAP2D'
      include 'STATD'

      ! global variables
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer gs_handle         ! gather-scatter handle
      integer*8 unodes(lx1*lz1*lelt) ! unique local nodes
      integer il, jl, el        ! loop index
      integer el2               ! index of 2D element
      integer itmp1, itmp2
      real rtmp_ruavg(lx1*lz1,lelt) ! tmp array for local 2D element aggragation
!-----------------------------------------------------------------------
      ! if no space averaging return
      if (stat_rdim.eq.0) return
      
      ! stamp logs
      call mntr_log(stat_id,lp_vrb,'Global statistics summation.')

      ! perform local summation of 2D contributions; not required for 3D version
      do il = 1, stat_lvar
         el = lx1*lz1*lelt
         call rzero(rtmp_ruavg,el)
         do el=1,nelv
            el2 = map2d_lmap(el)
            if(el2.gt.0) then
               do jl = 1,stat_nm2*stat_nm3
                  rtmp_ruavg(jl,el2) = rtmp_ruavg(jl,el2) +
     $                 stat_ruavg(jl,el,il)
               enddo
            endif
         enddo
         ! copy data back
         el = stat_nm2*stat_nm3*map2d_lnum
         call copy(stat_ruavg(1,1,il),rtmp_ruavg,el)
      enddo

      ! set up communicator
      itmp1 = stat_nm2*stat_nm3
      do il = 1, map2d_lnum
         itmp2 = map2d_gmap(il) - 1
         do jl=1,itmp1
            unodes(itmp1*(il-1) + jl) = int(itmp1,8)*int(itmp2,8) + jl
         enddo
      enddo
      itmp2 = itmp1*map2d_lnum
      call fgslib_gs_setup(gs_handle,unodes,itmp2,nekcomm,mp)

      ! sum variables
      do il=1,stat_lvar
         call fgslib_gs_op(gs_handle,stat_ruavg(1,1,il),1,1,0)
      enddo

      ! destroy communicator
      call fgslib_gs_free (gs_handle)

      ! divide data by arc length
      if (if3d) then
         itmp1=stat_nm2*stat_nm3
         do il=1, map2d_lnum
            if (map2d_own(il).eq.nid) then
               do jl=1, stat_lvar
                  call invcol2(stat_ruavg(1,il,jl),
     $                 stat_abm1d(1,1,il),itmp1)
               enddo
            endif
         enddo
      endif
      
      return
      end subroutine
!=======================================================================
!> @brief Compute statistics
!! @ingroup stat
!! @remark This routine uses global scratch space \a SCRMG, \a SCRUZ, \a SCRNS, \a SCRSF
      subroutine stat_compute()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'SOLN'
      include 'TSTEP'
      include 'STATD'           ! Variables from the statistics
      include 'INPUT'           ! if3d

      ! global variables
      ! work arrays
      real slvel(LX1,LY1,LZ1,LELT,3), slp(LX1,LY1,LZ1,LELT)
      common /SCRMG/ slvel, slp
      real tmpvel(LX1,LY1,LZ1,LELT,3), tmppr(LX1,LY1,LZ1,LELT)
      common /SCRUZ/ tmpvel, tmppr
      real dudx(LX1,LY1,LZ1,LELT,3) ! du/dx, du/dy and du/dz
      real dvdx(LX1,LY1,LZ1,LELT,3) ! dv/dx, dv/dy and dv/dz
      real dwdx(LX1,LY1,LZ1,LELT,3) ! dw/dx, dw/dy and dw/dz
      common /SCRNS/ dudx, dvdx
      common /SCRSF/ dwdx

      ! local variables
      integer npos              ! position in STAT_RUAVG
      real alpha, beta,dtime    ! time averaging parameters
      integer lnvar             ! count number of variables
      integer i                 ! loop index
      integer itmp              ! dummy variable
      real rtmp                 ! dummy variable
      integer ntot

!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_log(stat_id,lp_vrb,'Average fields.')
      
      ! Calculate time span of current statistical sample
      dtime=time-stat_atime-stat_tstart

      ! Update total time over which the current stat file is averaged
      stat_atime=time-stat_tstart

      ! Time average is compuated as:
      ! Accumulated=alpha*Accumulated+beta*New
      ! Calculate alpha and beta
      beta=dtime/STAT_ATIME
      alpha=1.0-beta
      
      ! Map pressure to velocity mesh
      call mappr(tmppr,PR,tmpvel(1,1,1,1,2),tmpvel(1,1,1,1,3))

      ! Compute derivative tensor and normalise pressure
      call user_stat_trnsv(tmpvel,dudx,dvdx,dwdx,slvel,tmppr)

      ! reset varaible counter
      lnvar = 0

      ! reshuffle arrays
      ! velocity
      call stat_reshufflev(slvel(1,1,1,1,1),tmpvel(1,1,1,1,1),NELV)
      call stat_reshufflev(slvel(1,1,1,1,2),tmpvel(1,1,1,1,2),NELV)
      if (if3d) call stat_reshufflev(slvel(1,1,1,1,3),
     $     tmpvel(1,1,1,1,3),NELV)

      ! pressure
      call stat_reshufflev(slp,tmppr,NELV)

      ! reshuffle velocity derivatives
      ! VX
      call stat_reshufflev(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),dudx(1,1,1,1,2),NELV)
      if (if3d) call stat_reshufflev(tmpvel(1,1,1,1,3),
     $     dudx(1,1,1,1,3),NELV)
      ! copy
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call copy(dudx,tmpvel,itmp)

      ! VY
      call stat_reshufflev(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),dvdx(1,1,1,1,2),NELV)
      if (if3d) call stat_reshufflev(tmpvel(1,1,1,1,3),
     $     dvdx(1,1,1,1,3),NELV)
      ! copy
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call copy(dvdx,tmpvel,itmp)

      ! VZ
      if (if3d) then
         call stat_reshufflev(tmpvel(1,1,1,1,1),dwdx(1,1,1,1,1),NELV)
         call stat_reshufflev(tmpvel(1,1,1,1,2),dwdx(1,1,1,1,2),NELV)
         call stat_reshufflev(tmpvel(1,1,1,1,3),dwdx(1,1,1,1,3),NELV)
      ! copy
         itmp = LX1*LY1*LZ1*LELT*LDIM
         call copy(dwdx,tmpvel,itmp)
      endif

!=======================================================================
      ! Computation of statistics
      
      ! <u>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(slvel(1,1,1,1,1),npos,alpha,beta)

      ! <v>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(slvel(1,1,1,1,2),npos,alpha,beta)

      ! <w>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(slvel(1,1,1,1,3),npos,alpha,beta)

      ! <p>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(slp(1,1,1,1),npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <uu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <vv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,2),slvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <ww>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,3),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

      ! <pp>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),slp(1,1,1,1),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <uv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <vw>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,2),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

      ! <uw>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,1),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <pu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <pv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <pw>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <pdudx>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <pdudy>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <pdudz>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dudx(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <pdvdx>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <pdvdy>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <pdvdz>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dvdx(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <pdwdx>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <pdwdy>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <pdwdz>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slp(1,1,1,1),dwdx(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! UU, VV, WW
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),slvel(1,1,1,1,1),slvel(1,1,1,1,1),
     $     itmp)

      ! <uuu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,1),tmpvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <vvv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,2),tmpvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <www>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(slvel(1,1,1,1,3),tmpvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <uuv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <uuw>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

      ! <vvu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),slvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <vvw>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <wwu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),slvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <wwv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),slvel(1,1,1,1,2),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <ppp>t
      lnvar = lnvar + 1
      npos = lnvar
      itmp = LX1*LY1*LZ1*LELT
      call col3(tmppr(1,1,1,1),slp(1,1,1,1),slp(1,1,1,1),
     $     itmp) 
      call stat_compute_1Dav2(tmppr(1,1,1,1),slp(1,1,1,1),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <pppp>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmppr(1,1,1,1),tmppr(1,1,1,1),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <uvw>t
      lnvar = lnvar + 1
      npos = lnvar
      ! copy uv to tmppr (do not need pp anymore)
      itmp = LX1*LY1*LZ1*LELT
      call col3(tmppr(1,1,1,1),slvel(1,1,1,1,1),slvel(1,1,1,1,2),
     $     itmp) 
      call stat_compute_1Dav2(tmppr(1,1,1,1),slvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <uuuu>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,1),
     $     npos,alpha,beta)

      ! <vvvv>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),tmpvel(1,1,1,1,2),
     $     npos,alpha,beta)

      ! <wwww>t
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),tmpvel(1,1,1,1,3),
     $     npos,alpha,beta)

!-----------------------------------------------------------------------
      ! <e11>t : (du/dx)^2 + (du/dy)^2 + (du/dz)^2
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dudx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

      ! <e22>t: (dv/dx)^2 + (dv/dy)^2 + (dv/dz)^2
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
      
      ! <e33>t: (dw/dx)^2 + (dw/dy)^2 + (dw/dz)^2
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dwdx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
      
!-----------------------------------------------------------------------
      ! <e12>t: (du/dx)*(dv/dx) + (du/dy)*(dv/dy) + (du/dz)*(dv/dz)
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dvdx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)

      ! <e13>t: (du/dx)*(dw/dx) + (du/dy)*(dw/dy) + (du/dz)*(dw/dz)
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dudx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
      
      ! <e23>t: (dv/dx)*(dw/dx) + (dv/dy)*(dw/dy) + (dv/dz)*(dw/dz)
      itmp = LX1*LY1*LZ1*LELT*LDIM
      call col3(tmpvel(1,1,1,1,1),dvdx(1,1,1,1,1),dwdx(1,1,1,1,1),
     $     itmp)
      itmp = LX1*LY1*LZ1*LELT
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),itmp)
      call add2(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,3),itmp)
      lnvar = lnvar + 1
      npos = lnvar
      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
      
!=======================================================================
      !End of local compute

      ! save number of variables
      stat_nvar = lnvar

      return
      end subroutine
!=======================================================================
