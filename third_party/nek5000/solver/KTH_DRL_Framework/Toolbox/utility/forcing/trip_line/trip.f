!> @file trip.f
!! @ingroup trip_line
!! @brief Tripping function for AMR version of nek5000
!! @note  This version uses developed framework parts. This is because
!!   I'm in a hurry and I want to save some time writing the code. So
!!   I reuse already tested code and focuse important parts. For the
!!   same reason for now only lines parallel to z axis are considered. 
!!   The tripping is based on a similar implementation in the SIMSON code
!!   (Chevalier et al. 2007, KTH Mechanics), and is described in detail 
!!   in the paper Schlatter & Örlü, JFM 2012, DOI 10.1017/jfm.2012.324.
!! @author Adam Peplinski
!! @date May 03, 2018
!=======================================================================
!> @brief Register tripping module
!! @ingroup trip_line
!! @note This routine should be called in frame_usr_register
      subroutine trip_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'TRIPD'

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
      call mntr_mod_is_name_reg(lpmid,trip_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(trip_name)//'] already registered')
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
      call mntr_mod_reg(trip_id,lpmid,trip_name,
     $      'Tripping along the line')

      ! register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(trip_tmr_id,lpmid,trip_id,
     $     'TRIP_TOT','Tripping total time',.false.)

      ! register and set active section
      call rprm_sec_reg(trip_sec_id,trip_id,'_'//adjustl(trip_name),
     $     'Runtime paramere section for tripping module')
      call rprm_sec_set_act(.true.,trip_sec_id)

      ! register parameters
      call rprm_rp_reg(trip_nline_id,trip_sec_id,'NLINE',
     $     'Number of tripping lines',rpar_int,0,0.0,.false.,' ')

      call rprm_rp_reg(trip_tiamp_id,trip_sec_id,'TIAMP',
     $     'Time independent amplitude',rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(trip_tdamp_id,trip_sec_id,'TDAMP',
     $     'Time dependent amplitude',rpar_real,0,0.0,.false.,' ')

      do il=1, trip_nline_max
         write(str,'(I2.2)') il

         call rprm_rp_reg(trip_spos_id(1,il),trip_sec_id,'SPOSX'//str,
     $     'Starting point X',rpar_real,0,0.0,.false.,' ')
         
         call rprm_rp_reg(trip_spos_id(2,il),trip_sec_id,'SPOSY'//str,
     $     'Starting point Y',rpar_real,0,0.0,.false.,' ')

         if (IF3D) then
            call rprm_rp_reg(trip_spos_id(ldim,il),trip_sec_id,
     $           'SPOSZ'//str,'Starting point Z',
     $           rpar_real,0,0.0,.false.,' ')
         endif
        
         call rprm_rp_reg(trip_epos_id(1,il),trip_sec_id,'EPOSX'//str,
     $     'Ending point X',rpar_real,0,0.0,.false.,' ')
         
         call rprm_rp_reg(trip_epos_id(2,il),trip_sec_id,'EPOSY'//str,
     $     'Ending point Y',rpar_real,0,0.0,.false.,' ')

         if (IF3D) then
            call rprm_rp_reg(trip_epos_id(ldim,il),trip_sec_id,
     $           'EPOSZ'//str,'Ending point Z',
     $           rpar_real,0,0.0,.false.,' ')
         endif

         call rprm_rp_reg(trip_smth_id(1,il),trip_sec_id,'SMTHX'//str,
     $     'Smoothing length X',rpar_real,0,0.0,.false.,' ')
         
         call rprm_rp_reg(trip_smth_id(2,il),trip_sec_id,'SMTHY'//str,
     $     'Smoothing length Y',rpar_real,0,0.0,.false.,' ')

         if (IF3D) then
            call rprm_rp_reg(trip_smth_id(ldim,il),trip_sec_id,
     $           'SMTHZ'//str,'Smoothing length Z',
     $           rpar_real,0,0.0,.false.,' ')
         endif

         call rprm_rp_reg(trip_lext_id(il),trip_sec_id,'LEXT'//str,
     $        'Line extension',rpar_log,0,0.0,.false.,' ')
      
         call rprm_rp_reg(trip_rota_id(il),trip_sec_id,'ROTA'//str,
     $        'Rotation angle',rpar_real,0,0.0,.false.,' ')
         call rprm_rp_reg(trip_nmode_id(il),trip_sec_id,'NMODE'//str,
     $     'Number of Fourier modes',rpar_int,0,0.0,.false.,' ')
         call rprm_rp_reg(trip_tdt_id(il),trip_sec_id,'TDT'//str,
     $     'Time step for tripping',rpar_real,0,0.0,.false.,' ')
      enddo

      ! set initialisation flag
      trip_ifinit=.false.
      
      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(trip_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise tripping module
!! @ingroup trip_line
!! @note This routine should be called in frame_usr_init
      subroutine trip_init()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'FRAMELP'
      include 'TRIPD'

      ! local variables
      integer itmp
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      integer il, jl

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (trip_ifinit) then
         call mntr_warn(trip_id,
     $        'module ['//trim(trip_name)//'] already initiaised.')
         return
      endif
      
      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_nline_id,rpar_int)
      trip_nline = itmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_tiamp_id,rpar_real)
      trip_tiamp = rtmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_tdamp_id,rpar_real)
      trip_tdamp = rtmp
      do il=1,trip_nline
         do jl=1,LDIM
            call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_spos_id(jl,il),
     $           rpar_real)
            trip_spos(jl,il) = rtmp
            call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_epos_id(jl,il),
     $           rpar_real)
            trip_epos(jl,il) = rtmp
            call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_smth_id(jl,il),
     $           rpar_real)
            trip_smth(jl,il) = abs(rtmp)
         enddo
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_lext_id(il),
     $        rpar_log)
         trip_lext(il) = ltmp
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_rota_id(il),
     $        rpar_real)
         trip_rota(il) = rtmp
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_nmode_id(il),
     $        rpar_int)
         trip_nmode(il) = itmp
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,trip_tdt_id(il),
     $        rpar_real)
         trip_tdt(il) = rtmp
      enddo

      ! get sure z position of stating point is lower than ending point position
      do il=1,trip_nline
         if (trip_spos(ldim,il).gt.trip_epos(ldim,il)) then
            do jl=1,LDIM
               rtmp = trip_spos(jl,il)
               trip_spos(jl,il) = trip_epos(jl,il)
               trip_epos(jl,il) = rtmp
            enddo
         endif
      enddo

      ! get inverse line lengths and smoothing radius
      do il=1,trip_nline
         trip_ilngt(il) = 0.0
         do jl=1,LDIM
            trip_ilngt(il) = trip_ilngt(il) + (trip_epos(jl,il)-
     $           trip_spos(jl,il))**2
         enddo
         if (trip_ilngt(il).gt.0.0) then
            trip_ilngt(il) = 1.0/sqrt(trip_ilngt(il))
         else
            trip_ilngt(il) = 1.0
         endif
         do jl=1,LDIM
            if (trip_smth(jl,il).gt.0.0) then
               trip_ismth(jl,il) = 1.0/trip_smth(jl,il)
            else
               trip_ismth(jl,il) = 1.0
            endif
         enddo
      enddo

      ! get 1D projection and array mapping
      call trip_1dprj

      ! initialise random generator seed and number of time intervals
      do il=1,trip_nline
         trip_seed(il) = -32*il
         trip_ntdt(il) = 1 - trip_nset_max
         trip_ntdt_old(il) = trip_ntdt(il)
      enddo
      
      ! generate random phases (time independent and time dependent)
      call trip_rphs_get

      ! get forcing
      call trip_frcs_get(.true.)
      
      ! everything is initialised
      trip_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(trip_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup trip_line
!! @return trip_is_initialised
      logical function trip_is_initialised()
      implicit none

      include 'SIZE'
      include 'TRIPD'
!-----------------------------------------------------------------------
      trip_is_initialised = trip_ifinit

      return
      end function
!=======================================================================
!> @brief Update tripping
!! @ingroup trip_line
      subroutine trip_update()
      implicit none

      include 'SIZE'
      include 'TRIPD'

      ! local variables
      real ltim
      
      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()      

      ! update random phases (time independent and time dependent)
      call trip_rphs_get

      ! update forcing
      call trip_frcs_get(.false.)

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(trip_tmr_id,1,ltim)

      return
      end subroutine      
!=======================================================================
!> @brief Compute tripping forcing
!! @ingroup trip_line
!! @param[inout] ffx,ffy,ffz     forcing; x,y,z component
!! @param[in]    ix,iy,iz        GLL point index
!! @param[in]    ieg             global element number
      subroutine trip_forcing(ffx,ffy,ffz,ix,iy,iz,ieg)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'TRIPD'

      ! argument list
      real ffx, ffy, ffz
      integer ix,iy,iz,ieg

      ! local variables
      integer ipos,iel,il
      real ffn
!-----------------------------------------------------------------------
      iel=GLLEL(ieg)

      do il= 1, trip_nline
         ffn = trip_fsmth(ix,iy,iz,iel,il)
         if (ffn.gt.0.0) then
            ipos = trip_map(ix,iy,iz,iel,il)
            ffn = trip_ftrp(ipos,il)*ffn

            ffx = ffx - ffn*sin(trip_rota(il))
            ffy = ffy + ffn*cos(trip_rota(il))
         endif
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Reset tripping
!! @ingroup trip_line
      subroutine trip_reset()
      implicit none

      include 'SIZE'
      include 'TRIPD'

      ! local variables
      real ltim
      
      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()      

      ! get 1D projection and array mapping
      call trip_1dprj
      
      ! update forcing
      call trip_frcs_get(.true.)

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(trip_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Get 1D projection, array mapping and forcing smoothing
!! @ingroup trip_line
!! @details This routine is just a simple version supporting only lines
!!   paralles to z axis. In future it can be generalised.
!! @remark This routine uses global scratch space \a CTMP0 and \a CTMP1
      subroutine trip_1dprj()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'TRIPD'

      ! global memory access
      real lcoord(LX1*LY1*LZ1*LELT)
      common /CTMP0/ lcoord
      integer lmap(LX1*LY1*LZ1*LELT)
      common /CTMP1/ lmap

      ! local variables
      integer npxy, npel, nptot, itmp, jtmp, ktmp, eltmp, istart
      integer il, jl
      real xl, yl, zl, xr, yr, rota, rtmp, ptmp, epsl
      parameter (epsl = 1.0e-10)
!-----------------------------------------------------------------------
      npxy = NX1*NY1
      npel = npxy*NZ1
      nptot = npel*NELV
      
      ! for each line
      do il=1,trip_nline
         ! reset mapping array
         call ifill(trip_map(1,1,1,1,il),-1,nptot)

         ! Get coordinates and sort them
         call copy(lcoord,zm1,nptot)
         call sort(lcoord,lmap,nptot)

         ! if we do not extend a line exclude points below line start (z coordinate matters only)
         ! this cannot be mixed with Gauss profile
         istart = 1
         if (.not.trip_lext(il)) then
            do jl=1,nptot
               if (lcoord(jl).lt.
     $              (trip_spos(ldim,il)-3.0*trip_smth(ldim,il))) then
                  istart = istart+1
               else
                  exit
               endif
            enddo
         endif

         ! find unique entrances and provide mapping
         trip_npoint(il) = 1
         trip_prj(trip_npoint(il),il) = lcoord(istart)
         itmp = lmap(istart)-1
         eltmp = itmp/npel + 1
         itmp = itmp - npel*(eltmp-1)
         ktmp = itmp/npxy + 1
         itmp = itmp - npxy*(ktmp-1)
         jtmp = itmp/nx1 + 1
         itmp = itmp - nx1*(jtmp-1) + 1
         trip_map(itmp,jtmp,ktmp,eltmp,il) = trip_npoint(il)
         do jl=istart+1,nptot
            ! if line is not extended finish at proper position
            if (.not.trip_lext(il).and.(lcoord(jl).gt.
     $           (trip_epos(ldim,il)+3.0*trip_smth(ldim,il)))) exit

            if((lcoord(jl)-trip_prj(trip_npoint(il),il)).gt.
     $           max(epsl,abs(epsl*lcoord(jl)))) then
               trip_npoint(il) = trip_npoint(il) + 1
               trip_prj(trip_npoint(il),il) = lcoord(jl)
            endif

            itmp = lmap(jl)-1
            eltmp = itmp/npel + 1
            itmp = itmp - npel*(eltmp-1)
            ktmp = itmp/npxy + 1
            itmp = itmp - npxy*(ktmp-1)
            jtmp = itmp/nx1 + 1
            itmp = itmp - nx1*(jtmp-1) + 1
            trip_map(itmp,jtmp,ktmp,eltmp,il) = trip_npoint(il)
         enddo
             
         ! rescale 1D array
         do jl=1,trip_npoint(il)
            trip_prj(jl,il) = (trip_prj(jl,il) - trip_spos(ldim,il))
     $           *trip_ilngt(il)
         enddo
         
         ! get smoothing profile
         rota = trip_rota(il)
         ! initialize smoothing factor
         call rzero(trip_fsmth(1,1,1,1,il),nptot)
         
         do jl=1,nptot
            itmp = jl-1
            eltmp = itmp/npel + 1
            itmp = itmp - npel*(eltmp-1)
            ktmp = itmp/npxy + 1
            itmp = itmp - npxy*(ktmp-1)
            jtmp = itmp/nx1 + 1
            itmp = itmp - nx1*(jtmp-1) + 1

            ! take only mapped points
            istart = trip_map(itmp,jtmp,ktmp,eltmp,il)
            if (istart.gt.0) then

               ! rotation
               xl = xm1(itmp,jtmp,ktmp,eltmp)-trip_spos(1,il)
               yl = ym1(itmp,jtmp,ktmp,eltmp)-trip_spos(2,il)

               xr = xl*cos(rota)+yl*sin(rota)
               yr = -xl*sin(rota)+yl*cos(rota)

               rtmp = (xr*trip_ismth(1,il))**2+(yr*trip_ismth(2,il))**2
               ! do we extend a line beyond its ends
               if (.not.trip_lext(il)) then
                  if (trip_prj(istart,il).lt.0.0) then
                     zl = zm1(itmp,jtmp,ktmp,eltmp)-trip_spos(ldim,il)
                     rtmp = rtmp+(zl*trip_ismth(ldim,il))**2
                  elseif(trip_prj(istart,il).gt.1.0) then
                     zl = zm1(itmp,jtmp,ktmp,eltmp)-trip_epos(ldim,il)
                     rtmp = rtmp+(zl*trip_ismth(ldim,il))**2
                  endif
               endif
               ! Gauss; cannot be used with lines not extended beyond their ending points
               !trip_fsmth(itmp,jtmp,ktmp,eltmp,il) = exp(-4.0*rtmp)
               ! limited support
               if (rtmp.lt.1.0) then
                  trip_fsmth(itmp,jtmp,ktmp,eltmp,il) =
     $                 exp(-rtmp)*(1-rtmp)**2
               else
                  trip_fsmth(itmp,jtmp,ktmp,eltmp,il) = 0.0
               endif
            endif

         enddo
      enddo

      return
      end subroutine      
!=======================================================================
!> @brief Generate set of random phases
!! @ingroup trip_line
      subroutine trip_rphs_get
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'PARALLEL'
      include 'TRIPD'
      
      ! local variables
      integer il, jl, kl
      integer itmp
      real trip_ran2

#ifdef DEBUG
      character*3 str1, str2
      integer iunit, ierr
      ! call number
      integer icalldl
      save icalldl
      data icalldl /0/
#endif
!-----------------------------------------------------------------------
      ! time independent part
      if (trip_tiamp.gt.0.0.and..not.trip_ifinit) then
         do il = 1, trip_nline
            do jl=1, trip_nmode(il)
               trip_rphs(jl,1,il) = 2.0*pi*trip_ran2(il)
            enddo
         enddo
      endif

      ! time dependent part
      do il = 1, trip_nline
         itmp = int(time/trip_tdt(il))
         call bcast(itmp,ISIZE) ! just for safety
         do kl= trip_ntdt(il)+1, itmp
            do jl= trip_nset_max,3,-1
               call copy(trip_rphs(1,jl,il),trip_rphs(1,jl-1,il),
     $              trip_nmode(il))
            enddo
            do jl=1, trip_nmode(il)
               trip_rphs(jl,2,il) = 2.0*pi*trip_ran2(il)
            enddo
         enddo
         ! update time interval
         trip_ntdt_old(il) = trip_ntdt(il)
         trip_ntdt(il) = itmp
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      icalldl = icalldl+1
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalldl
      open(unit=iunit,file='trp_rps.txt'//str1//'i'//str2)

      do il=1,trip_nmode(1)
         write(iunit,*) il,trip_rphs(il,1:4,1)
      enddo

      close(iunit)
#endif

      return
      end subroutine
!=======================================================================
!> @brief A simple portable random number generator
!! @ingroup trip_line
!! @details  Requires 32-bit integer arithmetic. Taken from Numerical
!!   Recipes, William Press et al. Gives correlation free random
!!   numbers but does not have a very large dynamic range, i.e only
!!   generates 714025 different numbers. Set seed negative for
!!   initialization
!! @param[in]   il      line number
!! @return      ran
      real function trip_ran2(il)
      implicit none

      include 'SIZE'
      include 'TRIPD'
      
      ! argument list
      integer il

      ! local variables
      integer iff(trip_nline_max), iy(trip_nline_max)
      integer ir(97,trip_nline_max)
      integer m,ia,ic,j
      real rm
      parameter (m=714025,ia=1366,ic=150889,rm=1./m)
      save iff,ir,iy
      data iff /trip_nline_max*0/
!-----------------------------------------------------------------------
      ! initialise
      if (trip_seed(il).lt.0.or.iff(il).eq.0) then
         iff(il)=1
         trip_seed(il)=mod(ic-trip_seed(il),m)
         do j=1,97
            trip_seed(il)=mod(ia*trip_seed(il)+ic,m)
            ir(j,il)=trip_seed(il)
         end do
         trip_seed(il)=mod(ia*trip_seed(il)+ic,m)
         iy(il)=trip_seed(il)
      end if
      
      ! generate random number
      j=1+(97*iy(il))/m
      iy(il)=ir(j,il)
      trip_ran2=iy(il)*rm
      trip_seed(il)=mod(ia*trip_seed(il)+ic,m)
      ir(j,il)=trip_seed(il)

      end function
!=======================================================================
!> @brief Generate forcing along 1D line
!! @ingroup trip_line
!! @param[in] ifreset    reset flag
      subroutine trip_frcs_get(ifreset)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'TRIPD'

      ! argument list
      logical ifreset

#ifdef TRIP_PR_RST
      ! variables necessary to reset pressure projection for P_n-P_n-2
      integer nprv(2)
      common /orthbi/ nprv

      ! variables necessary to reset velocity projection for P_n-P_n-2
      include 'VPROJ'
#endif      
      ! local variables
      integer il, jl, kl, ll
      integer istart
      real theta0, theta
      logical ifntdt_dif

#ifdef DEBUG
      character*3 str1, str2
      integer iunit, ierr
      ! call number
      integer icalldl
      save icalldl
      data icalldl /0/
#endif
!-----------------------------------------------------------------------
      ! reset all
      if (ifreset) then
         if (trip_tiamp.gt.0.0) then
            istart = 1
         else
            istart = 2
         endif
         do il= 1, trip_nline
            do jl = istart, trip_nset_max
               call rzero(trip_frcs(1,jl,il),trip_npoint(il))
               do kl= 1, trip_npoint(il)
                  theta0 = 2*pi*trip_prj(kl,il)
                  do ll= 1, trip_nmode(il)
                     theta = theta0*ll
                     trip_frcs(kl,jl,il) = trip_frcs(kl,jl,il) +
     $                    sin(theta+trip_rphs(ll,jl,il))
                  enddo
               enddo
            enddo
         enddo
         ! rescale time independent part
         if (trip_tiamp.gt.0.0) then
            do il= 1, trip_nline
               call cmult(trip_frcs(1,1,il),trip_tiamp,trip_npoint(il))
            enddo
         endif
      else
         ! reset only time dependent part if needed
         ifntdt_dif = .FALSE.
         do il= 1, trip_nline
            if (trip_ntdt(il).ne.trip_ntdt_old(il)) then
               ifntdt_dif = .TRUE.
               do jl= trip_nset_max,3,-1
                  call copy(trip_frcs(1,jl,il),trip_frcs(1,jl-1,il),
     $                 trip_npoint(il))
               enddo
               call rzero(trip_frcs(1,2,il),trip_npoint(il))
               do jl= 1, trip_npoint(il)
                  theta0 = 2*pi*trip_prj(jl,il)
                  do kl= 1, trip_nmode(il)
                     theta = theta0*kl
                     trip_frcs(jl,2,il) = trip_frcs(jl,2,il) +
     $                    sin(theta+trip_rphs(kl,2,il))
                  enddo
               enddo
            endif
         enddo
         if (ifntdt_dif) then
#ifdef TRIP_PR_RST
            ! reset projection space
            ! pressure
            if (int(PARAM(95)).gt.0) then
               PARAM(95) = ISTEP
               nprv(1) = 0      ! veloctiy field only
            endif
            ! velocity
            if (int(PARAM(94)).gt.0) then
               PARAM(94) = ISTEP!+2
               ivproj(2,1) = 0
               ivproj(2,2) = 0
               if (IF3D) ivproj(2,3) = 0
            endif
#endif
         endif
      endif
      
      ! get tripping for current time step
      if (trip_tiamp.gt.0.0) then
         do il= 1, trip_nline
           call copy(trip_ftrp(1,il),trip_frcs(1,1,il),trip_npoint(il))
         enddo
      else
         do il= 1, trip_nline
            call rzero(trip_ftrp(1,il),trip_npoint(il))
         enddo
      endif
      ! interpolation in time
      do il = 1, trip_nline
         theta0= time/trip_tdt(il)-real(trip_ntdt(il))
         if (theta0.gt.0.0) then
            theta0=theta0*theta0*(3.0-2.0*theta0)
            !theta0=theta0*theta0*theta0*(10.0+(6.0*theta0-15.0)*theta0)
            do jl= 1, trip_npoint(il)
               trip_ftrp(jl,il) = trip_ftrp(jl,il) +
     $              trip_tdamp*((1.0-theta0)*trip_frcs(jl,3,il) +
     $              theta0*trip_frcs(jl,2,il))
            enddo
         else
            theta0=theta0+1.0
            theta0=theta0*theta0*(3.0-2.0*theta0)
            !theta0=theta0*theta0*theta0*(10.0+(6.0*theta0-15.0)*theta0)
            do jl= 1, trip_npoint(il)
               trip_ftrp(jl,il) = trip_ftrp(jl,il) +
     $              trip_tdamp*((1.0-theta0)*trip_frcs(jl,4,il) +
     $              theta0*trip_frcs(jl,3,il))
            enddo
         endif
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      icalldl = icalldl+1
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalldl
      open(unit=iunit,file='trp_fcr.txt'//str1//'i'//str2)

      do il=1,trip_npoint(1)
         write(iunit,*) il,trip_prj(il,1),trip_ftrp(il,1),
     $        trip_frcs(il,1:4,1)
      enddo

      close(iunit)
#endif
      
      return
      end subroutine
!=======================================================================
