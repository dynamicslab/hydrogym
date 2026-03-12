!> @file spongebx.f
!! @ingroup sponge_box
!! @brief Sponge/fringe for simple box mesh
!! @author Adam Peplinski
!! @date Feb 1, 2017
!=======================================================================
!> @brief Register sponge_box module
!! @ingroup sponge_box
!! @note This routine should be called in frame_usr_register
      subroutine spng_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'SPONGEBXD'

      ! local variables
      integer lpmid
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,spng_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(spng_name)//'] already registered')
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
      call mntr_mod_reg(spng_id,lpmid,spng_name,
     $          'Sponge/fringe for rectangular domain')

      ! register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(spng_tmr_id,lpmid,spng_id,
     $     'SPNG_INI','Sponge calculation initialisation time',.false.)

      ! register and set active section
      call rprm_sec_reg(spng_sec_id,spng_id,'_'//adjustl(spng_name),
     $     'Runtime paramere section for sponge_box module')
      call rprm_sec_set_act(.true.,spng_sec_id)

      ! register parameters
      call rprm_rp_reg(spng_str_id,spng_sec_id,'STRENGTH',
     $     'Sponge strength',rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_wl_id(1),spng_sec_id,'WIDTHLX',
     $     'Sponge left section width; dimension X ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_wl_id(2),spng_sec_id,'WIDTHLY',
     $     'Sponge left section width; dimension Y ',
     $     rpar_real,0,0.0,.false.,' ')

      if (IF3D) call rprm_rp_reg(spng_wl_id(ndim),spng_sec_id,
     $     'WIDTHLZ','Sponge left section width; dimension Z ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_wr_id(1),spng_sec_id,'WIDTHRX',
     $     'Sponge right section width; dimension X ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_wr_id(2),spng_sec_id,'WIDTHRY',
     $     'Sponge right section width; dimension Y ',
     $     rpar_real,0,0.0,.false.,' ')

      if (IF3D) call rprm_rp_reg(spng_wr_id(ndim),spng_sec_id,
     $     'WIDTHRZ','Sponge right section width; dimension Z ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_dl_id(1),spng_sec_id,'DROPLX',
     $     'Sponge left drop/rise section width; dimension X ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_dl_id(2),spng_sec_id,'DROPLY',
     $     'Sponge left drop/rise section width; dimension Y ',
     $     rpar_real,0,0.0,.false.,' ')

      if (IF3D) call rprm_rp_reg(spng_dl_id(ndim),spng_sec_id,
     $    'DROPLZ','Sponge left drop/rise section width; dimension Z ',
     $    rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_dr_id(1),spng_sec_id,'DROPRX',
     $     'Sponge right drop/rise section width; dimension X ',
     $     rpar_real,0,0.0,.false.,' ')

      call rprm_rp_reg(spng_dr_id(2),spng_sec_id,'DROPRY',
     $     'Sponge right drop/rise section width; dimension Y ',
     $     rpar_real,0,0.0,.false.,' ')

      if (IF3D) call rprm_rp_reg(spng_dr_id(ndim),spng_sec_id,
     $   'DROPRZ','Sponge right drop/rise section width; dimension Z ',
     $    rpar_real,0,0.0,.false.,' ')

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(spng_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise sponge_box module
!! @ingroup sponge_box
!! @param[in] lvx, lvy, lvz   velocity field to be stored as reference field
!! @note This routine should be called in frame_usr_init
!! @remark This routine uses global scratch space \a SCRUZ
      subroutine spng_init(lvx,lvy,lvz)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'FRAMELP'
      include 'SPONGEBXD'

      ! argument list
      real lvx(LX1*LY1*LZ1*LELV),lvy(LX1*LY1*LZ1*LELV),
     $     lvz(LX1*LY1*LZ1*LELV)

      ! local variables
      integer ierr, nhour, nmin
      integer itmp
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      integer ntot, il, jl
      real bmin(LDIM), bmax(LDIM)

      real xxmax, xxmax_c, xxmin, xxmin_c, arg
      real lcoord(LX1*LY1*LZ1*LELV)
      common /SCRUZ/ lcoord

      ! functions
      real dnekclock, glmin, glmax, math_stepf
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (spng_ifinit) then
         call mntr_warn(spng_id,
     $        'module ['//trim(spng_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_str_id,rpar_real)
      spng_str = rtmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wl_id(1),rpar_real)
      spng_wl(1) = rtmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wl_id(2),rpar_real)
      spng_wl(2) = rtmp

      if (IF3D) then
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wl_id(ndim),
     $        rpar_real)
         spng_wl(ndim) = rtmp
      endif

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wr_id(1),rpar_real)
      spng_wr(1) = rtmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wr_id(2),rpar_real)
      spng_wr(2) = rtmp

      if (IF3D) then
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_wr_id(ndim),
     $        rpar_real)
         spng_wr(ndim) = rtmp
      endif

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dl_id(1),rpar_real)
      spng_dl(1) = rtmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dl_id(2),rpar_real)
      spng_dl(2) = rtmp

      if (IF3D) then
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dl_id(ndim),
     $        rpar_real)
         spng_dl(ndim) = rtmp
      endif

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dr_id(1),rpar_real)
      spng_dr(1) = rtmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dr_id(2),rpar_real)
      spng_dr(2) = rtmp

      if (IF3D) then
         call rprm_rp_get(itmp,rtmp,ltmp,ctmp,spng_dr_id(ndim),
     $        rpar_real)
         spng_dr(ndim) = rtmp
      endif

      ! initialise sponge variables

      ! get box size
      ntot = NX1*NY1*NZ1*NELV
      bmin(1) = glmin(XM1,ntot)
      bmax(1) = glmax(XM1,ntot)
      bmin(2) = glmin(YM1,ntot)
      bmax(2) = glmax(YM1,ntot)
      if(IF3D) then
         bmin(NDIM) = glmin(ZM1,ntot)
         bmax(NDIM) = glmax(ZM1,ntot)
      endif

      ! zero spng_fun
      call rzero(spng_fun,ntot)


      if(spng_str.gt.0.0) then
         call mntr_log(spng_id,lp_inf,"Sponge turned on")

         ! save reference field
         call copy(spng_vr(1,1),lvx, ntot)
         call copy(spng_vr(1,2),lvy, ntot)
         if (IF3D) call copy(spng_vr(1,NDIM),lvz, ntot)

         ! for every dimension
         do il=1,NDIM

            if (spng_wl(il).gt.0.0.or.spng_wr(il).gt.0.0) then
               if (spng_wl(il).lt.spng_dl(il).or.
     $              spng_wr(il).lt.spng_dr(il)) then
                  call mntr_abort(spng_id,"Wrong sponge parameters")
               endif

               ! sponge beginning (rise at xmax; right)
               xxmax = bmax(il) - spng_wr(il)
               ! end (drop at xmin; left)
               xxmin = bmin(il) + spng_wl(il)
               ! beginnign of constant part (right)
               xxmax_c = xxmax + spng_dr(il)
               ! beginnign of constant part (left)
               xxmin_c = xxmin - spng_dl(il)

               ! get SPNG_FUN
               if (xxmax.le.xxmin) then
                  call mntr_abort(spng_id,"Sponge too wide")
               else
                  ! this should be done by pointers, but for now I avoid it
                  if (il.eq.1) then
                     call copy(lcoord,XM1, ntot)
                  elseif (il.eq.2) then
                     call copy(lcoord,YM1, ntot)
                  elseif (il.eq.3) then
                     call copy(lcoord,ZM1, ntot)
                  endif

                  do jl=1,ntot
                     rtmp = lcoord(jl)
                     if(rtmp.le.xxmin_c) then ! constant; xmin
                        rtmp=spng_str
                     elseif(rtmp.lt.xxmin) then ! fall; xmin
                        arg = (xxmin-rtmp)/(spng_wl(il)-spng_dl(il))
                        rtmp = spng_str*math_stepf(arg)
                     elseif (rtmp.le.xxmax) then ! zero
                        rtmp = 0.0
                     elseif (rtmp.lt.xxmax_c) then ! rise
                        arg = (rtmp-xxmax)/(spng_wr(il)-spng_dr(il))
                        rtmp = spng_str*math_stepf(arg)
                     else    ! constant
                        rtmp = spng_str
                     endif
                     spng_fun(jl)=max(spng_fun(jl),rtmp)
                  enddo

               endif         ! xxmax.le.xxmin

            endif            ! spng_w(il).gt.0.0
         enddo


      endif

#ifdef DEBUG
      ! for debugging
      ltmp = ifto
      ifto = .TRUE.
      call outpost2(spng_vr,spng_vr(1,2),spng_vr(1,NDIM),spng_fun,
     $              spng_fun,1,'spg')
      ifto = ltmp
#endif

      ! is everything initialised
      spng_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(spng_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup sponge_box
!! @return spng_is_initialised
      logical function spng_is_initialised()
      implicit none

      include 'SIZE'
      include 'SPONGEBXD'
!-----------------------------------------------------------------------
      spng_is_initialised = spng_ifinit

      return
      end function
!=======================================================================
!> @brief Get sponge forcing
!! @ingroup sponge_box
!! @param[inout] ffx,ffy,ffz     forcing; x,y,z component
!! @param[in]    ix,iy,iz        GLL point index
!! @param[in]    ieg             global element number
      subroutine spng_forcing(ffx,ffy,ffz,ix,iy,iz,ieg)
      implicit none

      include 'SIZE'            !
      include 'INPUT'           ! IF3D
      include 'PARALLEL'        ! GLLEL
      include 'SOLN'            ! JP
      include 'SPONGEBXD'

      ! argument list
      real ffx, ffy, ffz
      integer ix,iy,iz,ieg

      ! local variables
      integer iel, ip
!-----------------------------------------------------------------------
      iel=GLLEL(ieg)
      if (SPNG_STR.gt.0.0) then
         ip=ix+NX1*(iy-1+NY1*(iz-1+NZ1*(iel-1)))

         if (JP.eq.0) then
            ! dns
            ffx = ffx + SPNG_FUN(ip)*(SPNG_VR(ip,1) - VX(ix,iy,iz,iel))
            ffy = ffy + SPNG_FUN(ip)*(SPNG_VR(ip,2) - VY(ix,iy,iz,iel))
            if (IF3D) ffz = ffz + SPNG_FUN(ip)*
     $           (SPNG_VR(ip,NDIM) - VZ(ix,iy,iz,iel))
         else
            ! perturbation
            ffx = ffx - SPNG_FUN(ip)*VXP(ip,JP)
            ffy = ffy - SPNG_FUN(ip)*VYP(ip,JP)
            if(IF3D) ffz = ffz - SPNG_FUN(ip)*VZP(ip,JP)
         endif

      endif

      return
      end subroutine
!=======================================================================
