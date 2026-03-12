!> @file gSyEM.f
!! @ingroup gsyn_eddy
!! @brief Generalised Synthetic Eddy Method
!! @author J. Canton
!! @author L. Hufnagel
!! @author F. Magagnato
!! @author Adam Peplinski
!! @date Apr 07, 2020
!=======================================================================
!> @brief Register gSyEM module
!! @ingroup gsyn_eddy
!! @note This routine should be called in frame_usr_register
      subroutine gsem_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'GSYEMD'

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
      call mntr_mod_is_name_reg(lpmid,gsem_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(gsem_name)//'] already registered')
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
      call mntr_mod_reg(gsem_id,lpmid,gsem_name,
     $     'Generalised Synthetic Eddy Method')

      ! check compilation parameters
      if(ldim.ne.3) call mntr_abort(gsem_id,
     $     'This code must be compiled with ldim=3')

      ! register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(gsem_ttot_id,lpmid,gsem_id,
     $      'GSEM_TOT','GSEM total time',.false.)

      call mntr_tmr_reg(gsem_tini_id,gsem_ttot_id,gsem_id,
     $      'GSEM_INI','GSEM initialisation time',.true.)

      call mntr_tmr_reg(gsem_tevl_id,gsem_ttot_id,gsem_id,
     $      'GSEM_EVL','GSEM evolution time',.true.)

      call mntr_tmr_reg(gsem_tchp_id,gsem_ttot_id,gsem_id,
     $      'GSEM_CHP','GSEM checkpoint saving time',.true.)

      ! register and set active section
      call rprm_sec_reg(gsem_sec_id,gsem_id,'_'//adjustl(gsem_name),
     $     'Runtime paramere section for gSyEM module')
      call rprm_sec_set_act(.true.,gsem_sec_id)

      ! register parameters
      call rprm_rp_reg(gsem_mode_id,gsem_sec_id,'MODE',
     $     'gSyEM mode',rpar_int,1,0.0,.false.,' ')

      call rprm_rp_reg(gsem_nfam_id,gsem_sec_id,'NFAM',
     $     'Family number',rpar_int,1,0.0,.false.,' ')

      do il=1, gsem_nfam_max
         write(str,'(I2.2)') il
         call rprm_rp_reg(gsem_neddy_id(il),gsem_sec_id,'NEDDY'//str,
     $        'Numer of eddies per family',rpar_int,10,0.0,.false.,' ')
         call rprm_rp_reg(gsem_fambc_id(il),gsem_sec_id,'FAMBC'//str,
     $        'Family BC index',rpar_int,1,0.0,.false.,' ')
         call rprm_rp_reg(gsem_sig_max_id(il),gsem_sec_id,
     $        'FAMASIG'//str,'Family max edge size',rpar_real,
     $        1,0.025,.false.,' ')
         call rprm_rp_reg(gsem_sig_min_id(il),gsem_sec_id,
     $        'FAMISIG'//str,'Family min edge size',rpar_real,
     $        1,0.001,.false.,' ')
         call rprm_rp_reg(gsem_dir_id(1,il),gsem_sec_id,'FAMDIRX'//str,
     $        'Family normal X component',rpar_real,1,0.0,.false.,' ')
         call rprm_rp_reg(gsem_dir_id(2,il),gsem_sec_id,'FAMDIRY'//str,
     $        'Family normal Y component',rpar_real,1,0.0,.false.,' ')
         call rprm_rp_reg(gsem_dir_id(3,il),gsem_sec_id,'FAMDIRZ'//str,
     $        'Family normal Z component',
     $        rpar_real,1,0.0,.false.,' ')
      enddo

      ! set initialisation flag
      gsem_ifinit=.false.
      
      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(gsem_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise gSyEM module
!! @ingroup gsyn_eddy
!! @note This routine should be called in frame_usr_init
      subroutine gsem_init()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'GEOM'
      include 'FRAMELP'
      include 'GSYEMD'

      ! local variables
      integer itmp
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      ! to get checkpoint runtime parameters
      integer ierr, lmid, lsid, lrpid
      
      integer il, jl, iel, iface, nface
      integer itmp2
      character*3 cbcl

      ! face to face mapping
      integer iffmap(4,6)
      save iffmap
      data iffmap / 2,4,5,6, 1,3,5,6, 2,4,5,6, 1,3,5,6,
     $              1,2,3,4, 1,2,3,4 /

      ! face to edge mapping
      integer ifemap(4,6)
      save ifemap
      data ifemap / 10,9,1,3, 10,12,6,8, 12,11,2,4, 9,11,5,7,
     $              1,6,2,5, 3,8,4,7 /

      ! functions
      integer iglsum
      integer frame_get_master
      real dnekclock
!-----------------------------------------------------------------------
      ! check if the module was already initialised
      if (gsem_ifinit) then
         call mntr_warn(gsem_id,
     $        'module ['//trim(gsem_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_mode_id,rpar_int)
      gsem_mode = itmp

      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_nfam_id,rpar_int)
      gsem_nfam = itmp
      if(gsem_nfam.gt.gsem_nfam_max) call mntr_abort(gsem_id,
     $     'Too many families')

      do il=1,gsem_nfam
        call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_neddy_id(il),rpar_int)
        gsem_neddy(il) = itmp
        if(itmp.gt.gsem_neddy_max) call mntr_abort(gsem_id,
     $       'Too many edies per family')

        call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_fambc_id(il),rpar_int)
        gsem_fambc(il)=itmp

        call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_sig_max_id(il),
     $       rpar_real)
        gsem_sig_max(il)=rtmp

        call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_sig_min_id(il),
     $       rpar_real)
        gsem_sig_min(il)=rtmp

        do jl=1,ldim
           call rprm_rp_get(itmp,rtmp,ltmp,ctmp,gsem_dir_id(jl,il),
     $          rpar_real)
           gsem_dir(jl,il)=rtmp
        enddo
      enddo

      ! check the restart flag
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
               gsem_chifrst = ltmp
            else
               ierr = 1
               goto 30
            endif
            if (gsem_chifrst) then
               ! checkpoint set number
               call rprm_rp_is_name_reg(lrpid,lsid,'CHKPFNUMBER',
     $              rpar_int)
               if (lrpid.gt.0) then
                  call rprm_rp_get(itmp,rtmp,ltmp,ctmp,lrpid,rpar_int)
                  gsem_fnum = itmp
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
      call mntr_check_abort(gsem_id,ierr,
     $            'Error reading checkpoint parameters')
      
      ! get number of snapshots in a set
      if (PARAM(27).lt.0) then
         gsem_nsnap = NBDINP
      else
         gsem_nsnap = 3
      endif

      ! set calculated family normal falg
      do il=1,gsem_nfam
         gsem_dirl(il) = .true.
         do jl=1,ldim
            if (gsem_dir(jl,il).ne.0.0) gsem_dirl(il) = .false.
         enddo
      enddo

      ! calculate eddy offset
      gsem_eoff(1) = 1
      do il=1,gsem_nfam
         gsem_eoff(il+1) = gsem_eoff(il) + gsem_neddy(il)
      enddo

      ! generate faces/edges mapping and check if all families
      ! correspond to Dirichlet bc
      ierr = 0
      nface = 2*ndim
      call izero(gsem_lfnum,gsem_nfam_max)
      call izero(gsem_gfnum,gsem_nfam_max)
      call izero(gsem_lenum,gsem_nfam_max)
      call izero(gsem_genum,gsem_nfam_max)
      itmp = 2*lelt*gsem_nfam_max
      call izero(gsem_lfmap,itmp)
      call izero(gsem_lemap,itmp)
      do il=1,gsem_nfam
         itmp=0                 ! count family faces
         itmp2=0                ! count family edges
         do iel=1,nelv
            do iface = 1, nface
               if(boundaryID(iface,iel).eq.gsem_fambc(il)) then
                  itmp = itmp +1
                  if(cbc(iface,iel,1).eq.'v  ') then
                     gsem_lfmap(1,itmp,il) = iel
                     gsem_lfmap(2,itmp,il) = iface
                     ! get edges
                     do jl=1,(ndim-1)*2
                        cbcl = cbc(iffmap(jl,iface),iel,1)
                        if(cbcl(1:1).eq.'W'.or.cbcl(1:1).eq.'m') then
                           itmp2 = itmp2 +1
                           gsem_lemap(1,itmp2,il) = iel
                           gsem_lemap(2,itmp2,il) = ifemap(jl,iface)
                        endif
                     enddo
                  else
                     ierr = 1
                  endif
               endif
            enddo
         enddo
         gsem_lfnum(il) = itmp
         gsem_gfnum(il) = iglsum(gsem_lfnum(il),1)
         if (gsem_gfnum(il).eq.0)  call mntr_abort(gsem_id,
     $        'Empty family BC')
         gsem_lenum(il) = itmp2
         gsem_genum(il) = iglsum(gsem_lenum(il),1)
      enddo
      call mntr_check_abort(gsem_id,ierr,'Non Dirichlet family BC')

      ! calculate face offset
      gsem_foff(1) = 1
      do il=1,gsem_nfam
         gsem_foff(il+1) = gsem_foff(il) + gsem_lfnum(il)
      enddo

      ! generate family information
      call gsem_fam_setup()
      
      ! initialise eddy position and orientation; possible rfestart
      if (gsem_chifrst) then
         ! read checkpoint
         call gsem_rst_read()
      else
         if (gsem_mode.eq.1) then
            do il=1,gsem_nfam
               call gsem_gen_eall(il)
            enddo
         elseif (gsem_mode.eq.2) then
         elseif (gsem_mode.eq.3) then
            do il=1,gsem_nfam
               call gsem_gen_mall(il)
            enddo
         endif
      endif
      
      ! everything is initialised
      gsem_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(gsem_tini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup gsyn_eddy
!! @return gsem_is_initialised
      logical function gsem_is_initialised()
      implicit none

      include 'SIZE'
      include 'GSYEMD'
!-----------------------------------------------------------------------
      gsem_is_initialised = gsem_ifinit

      return
      end function
!=======================================================================
!> @brief Main gSyEM interface
!! @ingroup gsyn_eddy
      subroutine gsem_main
      implicit none

      include 'SIZE'
      include 'TSTEP'
      include 'GSYEMD'

      ! local variables
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! take into account restart
      if ((gsem_chifrst.and.(istep.lt.(gsem_nsnap-1))).or.
     $     istep.eq.0) return

      ! to be consistent with saved velocity field first write checkpoint
      call gsem_rst_write

      ! update vertices position
      ! timing
      ltim = dnekclock()
      call gsem_evolve()
      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(gsem_tevl_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Create checkpoint
!! @ingroup gsyn_eddy
!! @details In current inplementation all data is duplicated over all
!!    nodes, so only master performs I/O
      subroutine gsem_rst_write
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'FRAMELP'
      include 'GSYEMD'

      ! local variables
      integer step_cnt, set_out
      integer ierr
      real ltim
      character*132 fname
      character*6  str

      character*132 hdr
      integer ahdsize
      parameter (ahdsize=132)

      real*4 test_pattern

      integer il
      integer itmp(4), itmp2

      real*4 workla4(2*ldim*gsem_neddy_max)
      real*8 workla8(ldim*gsem_neddy_max)
      equivalence (workla4,workla8)
      
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

         call mntr_log(gsem_id,lp_inf,'Writing checkpoint snapshot')

         ierr = 0
         ! for current implementation I/O on master only
         if(nid.eq.0) then
            ! get file name
            fname='GSEM'//trim(adjustl(SESSION))//'_rs.'
            write(str,'(i5.5)') set_out + 1
            fname=trim(fname)//trim(str(1:5))//CHAR(0)

            ! open file
            call byte_open(fname,ierr)

            if (ierr.eq.0) then
               ! write number of families
               call blank(hdr,ahdsize)

               write(hdr,10) gsem_mode, gsem_nfam, TIME
 10            format('#gsem',1x,i2,1x,i2,1x,1pe14.7)

               call byte_write(hdr,ahdsize/4,ierr)
               if (ierr.gt.0) goto 20

               ! write test pattern for byte swap
               test_pattern = 6.54321

               call byte_write(test_pattern,1,ierr)
               if (ierr.gt.0) goto 20

               ! loop over families
               do il=1,gsem_nfam
                  ! collect and write integer varialbes
                  itmp(1) = gsem_fambc(il)
                  itmp(2) = gsem_neddy(il)
                  itmp(3) = gsem_gfnum(il)
                  itmp(4) = gsem_genum(il)

                  call byte_write(itmp,4,ierr)
                  if (ierr.gt.0) goto 20

                  itmp2 = ldim*gsem_neddy(il)
                  call copy(workla8,gsem_epos(1,gsem_eoff(il)),itmp2)
                  call byte_write(workla4,2*itmp2,ierr)
                  if (ierr.gt.0) goto 20

                  call byte_write(gsem_eps(1,gsem_eoff(il)),itmp2,ierr)
                  if (ierr.gt.0) goto 20                 
               enddo
               
               ! close file
               call byte_close(ierr)
            endif
         endif

 20      continue

         call  mntr_check_abort(gsem_id,ierr,
     $        'gsem_rst_write: Error writing restart file.')

         call nekgsync()
         ! timing
         ltim = dnekclock() - ltim
         call mntr_tmr_add(gsem_tchp_id,1,ltim)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Read from checkpoint
!! @ingroup gsyn_eddy
!! @details In current inplementation all data is duplicated over all
!!    nodes, so only master performs I/O
      subroutine gsem_rst_read
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'GSYEMD'

      ! local variables
      integer ierr
      character*132 fname
      character*6  str

      ! local variable copies
      integer lgsem_mode, lgsem_nfam, lgsem_fambc, lgsem_neddy
      integer lgsem_gfnum, lgsem_genum
      real ltime
      
      character*132 hdr
      character*5 dummy
      integer ahdsize
      parameter (ahdsize=132)

      real*4 test_pattern

      integer il
      integer itmp(4), itmp2

      real*4 workla4(2*ldim*gsem_neddy_max)
      real*8 workla8(ldim*gsem_neddy_max)
      equivalence (workla4,workla8)
      logical if_byte_swap_test, if_byte_sw_loc

      ! functions
      integer indx2
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_log(gsem_id,lp_inf,'Reading checkpoint')

      ierr = 0
      ! for current implementationI/O on master only
      if(nid.eq.0) then
         ! get file name
         fname='GSEM'//trim(adjustl(SESSION))//'_rs.'
         write(str,'(i5.5)') gsem_fnum
         fname=trim(fname)//trim(str(1:5))//CHAR(0)

         ! open file
         call byte_open(fname,ierr)

         if (ierr.eq.0) then
            ! read header
            call blank(hdr,ahdsize)

            call byte_read(hdr,ahdsize/4,ierr)
            if (ierr.gt.0) goto 20

            if (indx2(hdr,132,'#gsem',5).eq.1) then
               read(hdr,*) dummy,lgsem_mode,lgsem_nfam,ltime
            else
               call  mntr_log(gsem_id,lp_err,
     $              'gsem_rst_read; Error reading header')
               ierr=1
               goto 20
            endif

            if(lgsem_mode.ne.gsem_mode.or.lgsem_nfam.ne.gsem_nfam) then
               call  mntr_log(gsem_id,lp_err,
     $              'gsem_rst_read; Inconsistent header values')
               ierr=1
               goto 20
            endif

            ! read test pattern for byte swap
            call byte_read(test_pattern,1,ierr)
            if (ierr.gt.0) goto 20
            ! determine endianess
            if_byte_sw_loc = if_byte_swap_test(test_pattern,ierr)
            if (ierr.gt.0) goto 20

            ! loop over families
            do il=1,gsem_nfam
               ! read integer varialbes
               call byte_read(itmp,4,ierr)
               if (if_byte_sw_loc) then
                  call byte_reverse(itmp,4,ierr)
                  if (ierr.gt.0) goto 20
               endif
               if(itmp(1).ne.gsem_fambc(il).or.
     $              itmp(2).ne.gsem_neddy(il).or.
     $              itmp(3).ne.gsem_gfnum(il).or.
     $              itmp(4).ne.gsem_genum(il)) then
                  call  mntr_log(gsem_id,lp_err,
     $                 'gsem_rst_read; Inconsistent family data')
                  ierr=1
                  goto 20
               endif

               itmp2 = ldim*gsem_neddy(il)
               call byte_read(workla4,2*itmp2,ierr)
               if (ierr.gt.0) goto 20
               if (if_byte_sw_loc) then
                  call byte_reverse(workla4,2*itmp2,ierr)
                  if (ierr.gt.0) goto 20
               endif
               call copy(gsem_epos(1,gsem_eoff(il)),workla8,itmp2)

               call byte_read(gsem_eps(1,gsem_eoff(il)),itmp2,ierr)
               if (ierr.gt.0) goto 20
               if (if_byte_sw_loc) then
                  call byte_reverse(gsem_eps(1,gsem_eoff(il)),itmp2,
     $                 ierr)
                  if (ierr.gt.0) goto 20
               endif
            enddo

            ! close file
            call byte_close(ierr)
         endif
      endif

 20   continue

      call  mntr_check_abort(gsem_id,ierr,
     $       'gsem_rst_read: Error reading restart file.')

      ! broadcast data
      itmp2 = ldim*(gsem_eoff(gsem_nfam + 1) - 1)
      call bcast(gsem_epos,itmp2*WDSIZE)
      call bcast(gsem_eps,itmp2*ISIZE)
      
      return
      end subroutine
!=======================================================================
!> @brief Generate family information
!! @ingroup gsyn_eddy
!! @details This routine generates family information
!!    including bounding box size, average coordinates, normal vector
      subroutine gsem_fam_setup()
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'TSTEP'           ! PI
      include 'FRAMELP'
      include 'GSYEMD'

      ! local variables
      integer il, jl, kl, ll, ml, nl
      integer iel, ifc, ied, ifn, itmp
      real rtmp, epsl
      parameter (epsl=1.0E-06)

      ! profiles for given family
      ! number of points in profile per family
      integer npoint(gsem_nfam_max)
      ! family profile array offset
      integer poff(gsem_nfam_max+1)
      ! position in a profile
      real rpos(gsem_npoint_max*gsem_nfam_max)
      ! mean velocity
      real umean(gsem_npoint_max*gsem_nfam_max)
      ! turbulence kinetic energy
      real tke(gsem_npoint_max*gsem_nfam_max)
      ! dissipation rate
      real dss(gsem_npoint_max*gsem_nfam_max)

      ! work arrays
      real xyz_ewall(ldim,lx1,gsem_edge_max,gsem_nfam_max)
      real vrtmp(lx1*lz1), work(lx1*ldim*gsem_edge_max)
      real drtmp(ldim,lx1*lz1)

      ! functions
      integer igl_running_sum
      real vlmin, vlmax, vlsum
!-----------------------------------------------------------------------
      call mntr_log(gsem_id,lp_inf,'Generate family information')

      ! initialise random number genrator with clock time
      itmp = 0
      call math_zbqlini(itmp)

      ! read profile data
      call gsem_prof_read(npoint,poff,rpos,umean,tke,dss)

      ! extract edge position;
      ! This is necessary to get distance of the point from the edge
      ! Not very fancy, but done only once, so not urgent to improve
      call mntr_log(gsem_id,lp_inf,'Extract edge position')
      ! initial values
      call rzero(xyz_ewall,ldim*lx1*gsem_edge_max*gsem_nfam_max)
      ! loop over families
      do il=1,gsem_nfam
         ! check array sizes
         if (gsem_genum(il).gt.gsem_edge_max) call mntr_abort(gsem_id,
     $     'Too many edges per family')
         ! get offset
         itmp = igl_running_sum(gsem_lenum(il)) - gsem_lenum(il)
         do jl=1,gsem_lenum(il)
            iel = gsem_lemap(1,jl,il)
            ied = gsem_lemap(2,jl,il)

            ! copy coordinates
            call math_etovec(vrtmp,ied,xm1(1,1,1,iel),lx1,ly1,lz1)
            do kl=1,lx1
               xyz_ewall(1,kl,itmp+jl,il) = vrtmp(kl)
            enddo
            call math_etovec(vrtmp,ied,ym1(1,1,1,iel),lx1,ly1,lz1)
            do kl=1,lx1
               xyz_ewall(2,kl,itmp+jl,il) = vrtmp(kl)
            enddo
            call math_etovec(vrtmp,ied,zm1(1,1,1,iel),lx1,ly1,lz1)
            do kl=1,lx1
               xyz_ewall(ldim,kl,itmp+jl,il) = vrtmp(kl)
            enddo
         enddo
         ! global distribution
         call gop(xyz_ewall(1,1,1,il),work,'+  ',ldim*lx1*gsem_edge_max)
      enddo

      ! extract bounding box, average coordinate, normal, area
      call mntr_log(gsem_id,lp_inf,'Extract bounding box information')
      ! initial values
      do il=1,gsem_nfam
         rtmp = 99.0E20
         call cfill(gsem_bmin(1,il),rtmp,ldim)
         rtmp = -99.0E20
         call cfill(gsem_bmax(1,il),rtmp,ldim)
      enddo
      call rzero(gsem_bcrd,ldim*gsem_nfam_max)
      call rzero(gsem_bnrm,ldim*gsem_nfam_max)
      call rzero(gsem_area,gsem_nfam_max)
      itmp=lx1*lz1
      ! face loop
      do il=1,gsem_nfam
         do jl=1,gsem_lfnum(il)
            iel = gsem_lfmap(1,jl,il)
            ifc = gsem_lfmap(2,jl,il)

            ! bnd. box, centre crd., normal
            call ftovec(vrtmp,xm1,iel,ifc,lx1,ly1,lz1)
            rtmp = vlmin(vrtmp,itmp)
            gsem_bmin(1,il) = min(gsem_bmin(1,il),rtmp)
            rtmp = vlmax(vrtmp,itmp)
            gsem_bmax(1,il) = max(gsem_bmax(1,il),rtmp)
            rtmp = vlsum(vrtmp,itmp)
            gsem_bcrd(1,il) = gsem_bcrd(1,il) + rtmp
            rtmp = vlsum(unx(1,1,ifc,iel),itmp)
            gsem_bnrm(1,il) = gsem_bnrm(1,il) + rtmp

            call ftovec(vrtmp,ym1,iel,ifc,lx1,ly1,lz1)
            rtmp = vlmin(vrtmp,itmp)
            gsem_bmin(2,il) = min(gsem_bmin(2,il),rtmp)
            rtmp = vlmax(vrtmp,itmp)
            gsem_bmax(2,il) = max(gsem_bmax(2,il),rtmp)
            rtmp = vlsum(vrtmp,itmp)
            gsem_bcrd(2,il) = gsem_bcrd(2,il) + rtmp
            rtmp = vlsum(uny(1,1,ifc,iel),itmp)
            gsem_bnrm(2,il) = gsem_bnrm(2,il) + rtmp

            call ftovec(vrtmp,zm1,iel,ifc,lx1,ly1,lz1)
            rtmp = vlmin(vrtmp,itmp)
            gsem_bmin(ldim,il) = min(gsem_bmin(ldim,il),rtmp)
            rtmp = vlmax(vrtmp,itmp)
            gsem_bmax(ldim,il) = max(gsem_bmax(ldim,il),rtmp)
            rtmp = vlsum(vrtmp,itmp)
            gsem_bcrd(ldim,il) = gsem_bcrd(ldim,il) + rtmp
            rtmp = vlsum(unz(1,1,ifc,iel),itmp)
            gsem_bnrm(ldim,il) = gsem_bnrm(ldim,il) + rtmp

            ! family area
            gsem_area(il) = gsem_area(il)+vlsum(area(1,1,ifc,iel),itmp)
         enddo
      enddo

      ! global average
      call gop(gsem_bmin,work,'m  ',ldim*gsem_nfam_max)
      call gop(gsem_bmax,work,'M  ',ldim*gsem_nfam_max)
      call gop(gsem_bcrd,work,'+  ',ldim*gsem_nfam_max)
      call gop(gsem_bnrm,work,'+  ',ldim*gsem_nfam_max)
      call gop(gsem_area,work,'+  ',gsem_nfam_max)
      
      do il=1,gsem_nfam
         rtmp = 1.0/real(lx1*lx1*gsem_gfnum(il))
         call cmult(gsem_bcrd(1,il),rtmp,ldim)
         ! as nek normal point outwards change normal sign ???????????
         rtmp=-1.0*rtmp
         call cmult(gsem_bnrm(1,il),rtmp,ldim)
         ! this shouldn't be necessary, but just in case normalise vector
         rtmp = 0.0
         do jl=1,ldim
            rtmp = rtmp + gsem_bnrm(jl,il)**2
         enddo
         rtmp = 1.0/sqrt(rtmp)
         call cmult(gsem_bnrm(1,il),rtmp,ldim)
      enddo

      ! rotation axis and angle
      do il=1,gsem_nfam
         if(gsem_bnrm(3,il).gt.(1.0-epsl)) then
            gsem_raxs(1,il) = 1.0
            gsem_raxs(2,il) = 0.0
            gsem_raxs(3,il) = 0.0
            gsem_rang(il) = 0.0
         elseif(gsem_bnrm(3,il).lt.(epsl-1.0)) then
            gsem_raxs(1,il) = 1.0
            gsem_raxs(2,il) = 0.0
            gsem_raxs(3,il) = 0.0
            gsem_rang(il) = pi
         else
            gsem_raxs(1,il) = -gsem_bnrm(2,il)
            gsem_raxs(2,il) = gsem_bnrm(1,il)
            gsem_raxs(3,il) = 0.0
            rtmp = sqrt(gsem_bnrm(1,il)**2 + gsem_bnrm(2,il)**2)
            gsem_rang(il) = atan2(rtmp,gsem_bnrm(3,il))
         endif
      enddo

      ! Redefine bounding box using average normal and current box extend
      ! In this step I simply project bounding box vectors on a surface
      ! orthogonal to normal vector moving coordinate centre to
      ! averaged family position
      do il=1,gsem_nfam
         ! min
         rtmp = 0.0
         do jl=1,ldim
            rtmp = rtmp + (gsem_bmin(jl,il) - gsem_bcrd(jl,il))*
     $           gsem_bnrm(jl,il)
         enddo
         do jl=1,ldim
            gsem_bmin(jl,il) = (gsem_bmin(jl,il) - gsem_bcrd(jl,il)) -
     $           rtmp*gsem_bnrm(jl,il)
         enddo
         ! max
         rtmp = 0.0
         do jl=1,ldim
            rtmp = rtmp + (gsem_bmax(jl,il) - gsem_bcrd(jl,il))*
     $           gsem_bnrm(jl,il)
         enddo
         do jl=1,ldim
            gsem_bmax(jl,il) = (gsem_bmax(jl,il) - gsem_bcrd(jl,il)) -
     $           rtmp*gsem_bnrm(jl,il)
         enddo

         ! rotate bounding box to get 2D constraint; THIS IS NOT FINISHED
         call math_rot3Da(drtmp,gsem_bmin(1,il),
     $        gsem_raxs(1,il),-gsem_rang(il))
         call math_rot3Da(drtmp,gsem_bmax(1,il),
     $        gsem_raxs(1,il),-gsem_rang(il))

         ! Should I shift edges to the surface normal to gsem_bcrd???????

         ! update max allowed vertex size looking for min distance
         ! between family averaged centre and the edge
         ! This part is different form the original code !!!!!!
         gsem_cln_min(il) = 99.0E20
         gsem_cln_max(il) = -99.0E20
         do jl = 1,gsem_genum(il)           
            do ll=1,lx1
               rtmp = 0.0
               do kl=1,ldim
                  rtmp = rtmp +
     $                 (gsem_bcrd(kl,il)-xyz_ewall(kl,ll,jl,il))**2
               enddo
               rtmp = sqrt(rtmp)
               gsem_cln_min(il) = min(gsem_cln_min(il),rtmp)
               gsem_cln_max(il) = max(gsem_cln_max(il),rtmp)
            enddo
         enddo
      enddo
      call gop(gsem_cln_min,work,'m  ',gsem_nfam_max)
      call gop(gsem_cln_max,work,'M  ',gsem_nfam_max)
      ! In the origianl code this should be ralated to sigma_max

      ! Generate face information
      call mntr_log(gsem_id,lp_inf,'Generate face information')
      ! initial values
      rtmp = -99.0E20
      call cfill(gsem_mdst,rtmp,gsem_nfam_max)
      call rzero(gsem_sigma,lx1*lz1*6*lelt)
      call rzero(gsem_umean,lx1*lz1*6*lelt)
      call rzero(gsem_intn,lx1*lz1*6*lelt)
      call rzero(gsem_ub,ldim*gsem_nfam_max)
      itmp=lx1*lz1
      ! face loop
      do il=1,gsem_nfam
         do jl=1,gsem_lfnum(il)
            iel = gsem_lfmap(1,jl,il)
            ifc = gsem_lfmap(2,jl,il)

            ! extract coordinates
            call ftovec(vrtmp,xm1,iel,ifc,lx1,ly1,lz1)
            do kl=1,itmp
               drtmp(1,kl) = vrtmp(kl)
            enddo
            call ftovec(vrtmp,ym1,iel,ifc,lx1,ly1,lz1)
            do kl=1,itmp
               drtmp(2,kl) = vrtmp(kl)
            enddo
            call ftovec(vrtmp,zm1,iel,ifc,lx1,ly1,lz1)
            do kl=1,itmp
               drtmp(ldim,kl) = vrtmp(kl)
            enddo

            ! Should I shift drtmp to the surface normal to gsem_bcrd?????

            ! get min distance from the wall edge
            ! Not very fancy, but done only once, so not urgent to improve
            rtmp = 99.0E20
            call cfill(vrtmp,rtmp,lx1*lz1)
            do kl=1,gsem_genum(il)
               do ll=1,lx1
                  do ml=1,lx1*lz1
                     rtmp = 0.0
                     do nl=1,ldim
                        rtmp = rtmp +
     $                       (xyz_ewall(nl,ll,kl,il)-drtmp(nl,ml))**2
                     enddo
                     rtmp = sqrt(rtmp)
                     vrtmp(ml) = min(vrtmp(ml),rtmp)
                  enddo
               enddo
            enddo
            ! fill face arrays with profile values
            ifn = gsem_foff(il) + jl - 1
            call gsem_face_prof(il,poff,rpos,umean,tke,dss,
     $           iel,ifc,ifn,vrtmp)

            ! update max distance
            rtmp = vlmax(vrtmp,lx1*lz1)
            gsem_mdst(il) = max(gsem_mdst(il),rtmp)
         enddo
      enddo
      ! global operations
      call gop(gsem_ub,work,'+  ',ldim*gsem_nfam_max)
      call gop(gsem_mdst,work,'M  ',gsem_nfam_max)
      
      ! average bulk velocity and project it on normal direction
      do il=1,gsem_nfam
         rtmp = 1.0/gsem_area(il)
         do jl=1,ldim
            gsem_ub(jl,il) = gsem_ub(jl,il)*rtmp
         enddo
         if (gsem_dirl(il)) then
            gsem_un(il) = 0.0
            do jl=1,ldim
               gsem_un(il) = gsem_un(il) +
     $              gsem_ub(jl,il)*gsem_bnrm(jl,il)
            enddo
         else
            gsem_un(il) = 0.0
            do jl=1,ldim
               gsem_un(il) = gsem_un(il) + gsem_ub(jl,il)**2
            enddo
            gsem_un(il) = sqrt(gsem_un(il))
         endif

         ! box extent in normal direction
         gsem_bext(il) = min(gsem_sig_max(il),gsem_cln_min(il))
      enddo
      
      ! stamp logs
      call mntr_log(gsem_id,lp_inf,'General family information')
      do il=1,gsem_nfam
         call mntr_logi(gsem_id,lp_inf,'Family: ',il)
         call mntr_log(gsem_id,lp_inf,'Boundary box:')
         call mntr_logr(gsem_id,lp_inf,'min x=',gsem_bmin(1,il))
         call mntr_logr(gsem_id,lp_inf,'min y=',gsem_bmin(2,il))
         call mntr_logr(gsem_id,lp_inf,'min z=',gsem_bmin(3,il))
         call mntr_logr(gsem_id,lp_inf,'max x=',gsem_bmax(1,il))
         call mntr_logr(gsem_id,lp_inf,'max y=',gsem_bmax(2,il))
         call mntr_logr(gsem_id,lp_inf,'max z=',gsem_bmax(3,il))
         call mntr_log(gsem_id,lp_inf,'Average centre coordiantes:')
         call mntr_logr(gsem_id,lp_inf,'x=',gsem_bcrd(1,il))
         call mntr_logr(gsem_id,lp_inf,'y=',gsem_bcrd(2,il))
         call mntr_logr(gsem_id,lp_inf,'z=',gsem_bcrd(3,il))
         call mntr_log(gsem_id,lp_inf,'Average normal:')
         call mntr_logr(gsem_id,lp_inf,'x=',gsem_bnrm(1,il))
         call mntr_logr(gsem_id,lp_inf,'y=',gsem_bnrm(2,il))
         call mntr_logr(gsem_id,lp_inf,'z=',gsem_bnrm(3,il))
         call mntr_log(gsem_id,lp_inf,'Rotation axis:')
         call mntr_logr(gsem_id,lp_inf,'rx=',gsem_raxs(1,il))
         call mntr_logr(gsem_id,lp_inf,'ry=',gsem_raxs(2,il))
         call mntr_logr(gsem_id,lp_inf,'rz=',gsem_raxs(3,il))
         call mntr_log(gsem_id,lp_inf,'Rotation angle:')
         call mntr_logr(gsem_id,lp_inf,'ang=',gsem_rang(il))
         call mntr_log(gsem_id,lp_inf,'Characteristic length:')
         call mntr_logr(gsem_id,lp_inf,'min=',gsem_cln_min(il))
         call mntr_logr(gsem_id,lp_inf,'max=',gsem_cln_max(il))
         call mntr_log(gsem_id,lp_inf,'Box extent in norm. dir.:')
         call mntr_logr(gsem_id,lp_inf,'bext=',2.0*gsem_bext(il))
         call mntr_log(gsem_id,lp_inf,'Max point distance:')
         call mntr_logr(gsem_id,lp_inf,'min=',gsem_mdst(il))
         call mntr_log(gsem_id,lp_inf,'Surface area:')
         call mntr_logr(gsem_id,lp_inf,'area=',gsem_area(il))
         call mntr_log(gsem_id,lp_inf,'Bulk velocity:')
         call mntr_logr(gsem_id,lp_inf,'ubx=',gsem_ub(1,il))
         call mntr_logr(gsem_id,lp_inf,'uby=',gsem_ub(2,il))
         call mntr_logr(gsem_id,lp_inf,'ubz=',gsem_ub(3,il))
         call mntr_log(gsem_id,lp_inf,'Projected bulk velocity:')
         call mntr_logr(gsem_id,lp_inf,'un=',gsem_un(il))
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Read profiles for all familes
!! @ingroup gsyn_eddy
!! @param[out]   npoint      number of points in profile per family
!! @param[out]   poff        family profile array offset
!! @param[out]   rpos        position in a profile
!! @param[out]   umean       profile mean velocity
!! @param[out]   tke         profile turbulence kinetic energy
!! @param[out]   dss         profile dissipation rate
      subroutine gsem_prof_read(npoint,poff,rpos,umean,tke,dss)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer npoint(gsem_nfam_max), poff(gsem_nfam_max+1)
      real rpos(gsem_npoint_max*gsem_nfam_max)
      real umean(gsem_npoint_max*gsem_nfam_max)
      real tke(gsem_npoint_max*gsem_nfam_max)
      real dss(gsem_npoint_max*gsem_nfam_max)

      ! local variables
      integer itmp, il, jl, ierr, iunit
      character*132 fname
      character*3  str
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_log(gsem_id,lp_prd,'Reading family profiles')

      ! zero arrays
      itmp = gsem_npoint_max*gsem_nfam_max
      call rzero(rpos,itmp)
      call rzero(umean,itmp)
      call rzero(tke,itmp)
      call rzero(dss,itmp)

      ! reading done by master only
      ierr = 0
      if (nid.eq.0) then
         ! loop over families
         poff(1) = 1
         do il=1,gsem_nfam
            call io_file_freeid(iunit, ierr)
            if (ierr.eq.0) then
               write(str,'(i2.2)') gsem_fambc(il)
               fname='gsem_prof_'//trim(adjustl(str))//'.txt'
               open(unit=iunit,file=fname,status='old',iostat=ierr)
               if (ierr.eq.0) then
                  read(iunit,*,iostat=ierr) ! skip header
                  read(iunit,*,iostat=ierr)  npoint(il)
                  if (npoint(il).gt.gsem_npoint_max.or.
     $                 npoint(il).le.0) then
                     call mntr_log_local(gsem_id,lp_prd,
     $               'Wrong piont number in '//trim(adjustl(fname)),nid)
                     ierr=1
                     exit
                  endif
                  read(iunit,*) ! skip header
                  do jl=0,npoint(il) - 1
                     itmp = poff(il) + jl
                     read(iunit,*,iostat=ierr) rpos(itmp),umean(itmp),
     $                    tke(itmp),dss(itmp)
                     if(ierr.ne.0) exit
                  enddo
                  close(iunit)
                  poff(il+1) = poff(il) + npoint(il)
               endif
            endif
            if (ierr.ne.0) exit
         enddo
      endif
      call mntr_check_abort(gsem_id,ierr,'Error reading profile file')

      ! broadcast data
      call bcast(npoint,gsem_nfam*ISIZE)
      call bcast(poff,(gsem_nfam+1)*ISIZE)
      itmp = (poff(gsem_nfam+1)-1)*WDSIZE
      call bcast(rpos,itmp)
      call bcast(umean,itmp)
      call bcast(tke,itmp)
      call bcast(dss,itmp)

      return
      end subroutine
!=======================================================================
!> @brief Generate profile information for given family face
!! @ingroup gsyn_eddy
!! @param[in]   nfam        family number
!! @param[in]   poff        family profile array offset
!! @param[in]   rpos        position in a profile
!! @param[in]   umean       profile mean velocity
!! @param[in]   tke         profile turbulence kinetic energy
!! @param[in]   dss         profile dissipation rate
!! @param[in]   iel         local element number
!! @param[in]   ifc         face number
!! @param[in]   ifn         face position in gsem_sigma,gsem_umean,gsem_intn arrays
!! @param[in]   dist        point distance form the edge
      subroutine gsem_face_prof(nfam,poff,rpos,umean,tke,dss,
     $     iel,ifc,ifn,dist)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam, iel, ifc, ifn
      integer poff(gsem_nfam_max+1)
      real rpos(gsem_npoint_max*gsem_nfam_max)
      real umean(gsem_npoint_max*gsem_nfam_max)
      real tke(gsem_npoint_max*gsem_nfam_max)
      real dss(gsem_npoint_max*gsem_nfam_max)
      real dist(lx1,lz1)

      ! local variables
      integer itmp, il, jl, kl
      real nrtmp1(lx1*lz1), nrtmp2(lx1*lz1)
      real pvel, ptke, pdss, int1, int2, rtmp
      real twothr
      parameter (twothr=2.0/3.0)

      ! functions
      real vlsum
!-----------------------------------------------------------------------
      ! get profile values
      do jl=1,lz1
         do il=1,lx1
            ! find position in a profile
            ! not very efficient, but done only dutring intialisation
            ! so improvemnt not urgent
            itmp = poff(nfam+1)-1
            pvel = umean(itmp)
            ptke = tke(itmp)
            pdss = dss(itmp)
            do kl=poff(nfam)+1,itmp
               if (dist(il,jl).le.rpos(kl)) then
                  rtmp = 1.0/(rpos(kl) - rpos(kl-1))
                  int1 = (rpos(kl) - dist(il,jl))*rtmp
                  int2 = (dist(il,jl)-rpos(kl-1))*rtmp
                  pvel = int1*umean(kl-1) + int2*umean(kl)
                  ptke = int1*tke(kl-1) + int2*tke(kl)
                  pdss = int1*dss(kl-1) + int2*dss(kl)
                  exit
               endif
            enddo
            ! find vertex size
            ! formula 8.2.3 of Poletto's PhD
            rtmp = ptke**1.5/pdss

            ! Limit eddy size far away from wall
            ! to Prandtl's mixing length
            ! kappa is .41, max_range is the distance to the wall
            ! THIS PART OF THE CODE IS MODIFIED
            rtmp = min(rtmp,0.41*gsem_cln_max(nfam))

            ! make eddies no larger than box
            rtmp = min(rtmp,dist(il,jl))

            ! avoid creating too small eddies (0.001 is chosen arbitrarly)
            ! THIS SHOULD BE RELATED TO RESOLUTION
            rtmp = max(rtmp,0.001*gsem_cln_min(nfam))

            ! apply user defined cutoff
            rtmp = min(max(rtmp,gsem_sig_min(nfam)),gsem_sig_max(nfam))

            ! fill in profiles
            gsem_sigma(il,jl,ifn) = rtmp
            gsem_umean(il,jl,ifn) = pvel
            gsem_intn(il,jl,ifn) = sqrt(twothr*ptke)
         enddo
      enddo

      ! get bulk velocity depending on family normal flag
      itmp = lx1*lz1
      call col3(nrtmp1,gsem_umean(1,1,ifn),area(1,1,ifc,iel),itmp)
      if (gsem_dirl(nfam)) then        
         call col3(nrtmp2,nrtmp1,unx(1,1,ifc,iel),itmp)
         gsem_ub(1,nfam) = gsem_ub(1,nfam) - vlsum(nrtmp2,itmp)
         call col3(nrtmp2,nrtmp1,uny(1,1,ifc,iel),itmp)
         gsem_ub(2,nfam) = gsem_ub(2,nfam) - vlsum(nrtmp2,itmp)
         call col3(nrtmp2,nrtmp1,unz(1,1,ifc,iel),itmp)
         gsem_ub(3,nfam) = gsem_ub(3,nfam) - vlsum(nrtmp2,itmp)
      else
         do il=1,ldim
            call copy(nrtmp2,nrtmp1,itmp)
            call cmult(nrtmp2,gsem_dir(il,nfam),itmp)
            gsem_ub(il,nfam) = gsem_ub(il,nfam) + vlsum(nrtmp2,itmp)
         enddo
      endif

      return
      end subroutine
!=======================================================================
!> @brief Generate eddies from the list for a given family
!! @ingroup gsyn_eddy
!! @note This routine redistributes data among mpi ranks
!! @param[in]  elist      list of eddies to be geneated
!! @param[in]  neddy      number of eddies
!! @param[in]  nfam       family number
!! @param[in]  ifinit     intial distribution
      subroutine gsem_gen_elist(elist,neddy,nfam,ifinit)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer neddy, nfam, elist(neddy)
      logical ifinit

      ! local variables
      integer il, itmp1, itmp2, ierr
      character*50 str

      ! work arrays  USE SCRATCH IN THE FUTURE !!!!!!
      real eposl(ldim,gsem_neddy_max)
      integer epsl(ldim,gsem_neddy_max)

      ! function
      integer mntr_lp_def_get
!-----------------------------------------------------------------------
      ! generate set of eddies
      if (nid.eq.0) then
         do il=1,neddy
            call usr_gen_eddy(eposl(1,il),epsl(1,il),nfam,ifinit)
         enddo
      endif

      ! broadcast eddy data
      call bcast(neddy,ISIZE)
      call bcast(elist,neddy*ISIZE)
      call bcast(eposl,ldim*neddy*WDSIZE)
      call bcast(epsl,ldim*neddy*ISIZE)

      ! put data on place
      itmp1 = gsem_eoff(nfam)
      itmp2 = gsem_eoff(nfam+1)-1
      ierr = 0
      do il=1,neddy
         ! sanity check
         if (elist(il).lt.itmp1.or.elist(il).gt.itmp2) then
            ierr = 1
            exit
         endif
         call copy(gsem_epos(1,elist(il)),eposl(1,il),ldim)
         call icopy(gsem_eps(1,elist(il)),epsl(1,il),ldim)
      enddo
      call mntr_check_abort(gsem_id,ierr,'Eddy outside family range')
      
      ! stamp logs
      call mntr_logi(gsem_id,lp_prd,'Generated new eddies neddy=',neddy)
      ! verbose output
      if (mntr_lp_def_get().lt.lp_inf) then
         do il=1,neddy
            write(str,20) elist(il),eposl(1,il),eposl(2,il),eposl(3,il)
 20         format('edn=',i6,' ex=',f9.5,' ey=',f9.5,' ez=',f9.5)
            call mntr_log(gsem_id,lp_vrb,str)
         enddo
      endif

      return
      end subroutine
!=======================================================================
!> @brief Inital generation of all eddies (gsem_mode.eq.1) for a given family
!! @ingroup gsyn_eddy
!! @param[in]  nfam       family number
      subroutine gsem_gen_eall(nfam)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam

      ! local variables
      integer neddy, il
      logical ifinit

      ! work arrays  USE SCRATCH IN THE FUTURE !!!!!!
      integer elist(gsem_neddy_max)
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_logi(gsem_id,lp_prd,'Initilise eddies for family=',nfam)

      ! take all eddies in the family and set initialisation flag
      neddy = gsem_neddy(nfam)
      do il=1,neddy
         elist(il) = gsem_eoff(nfam) + il - 1
      enddo
      ifinit = .true.

      call gsem_gen_elist(elist,neddy,nfam,ifinit)

      return
      end subroutine
!=======================================================================
!> @brief Advect/recycle eddies (gsem_mode.eq.1) for a given family
!! @ingroup gsyn_eddy
!! @param[in]  nfam       family number
      subroutine gsem_adv_eddy(nfam)
      implicit none

      include 'SIZE'
      include 'TSTEP'           ! DT
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam

      ! local variables
      integer neddy, il, jl, itmp
      real rtmp
      logical ifinit

      ! work arrays  USE SCRATCH IN THE FUTURE !!!!!!
      integer elist(gsem_neddy_max)
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_logi(gsem_id,lp_prd,'Advect eddies for family=',nfam)

      ! advect eddies and mark the once to be recycled
      neddy=0
      do il=0,gsem_neddy(nfam)-1
         itmp = gsem_eoff(nfam)+il
         rtmp = 0.0
         do jl=1,ldim
            gsem_epos(jl,itmp) = gsem_epos(jl,itmp) +
     $           DT*gsem_un(nfam)*gsem_bnrm(jl,nfam)
            ! get distance with respect to family centre
            rtmp = rtmp + (gsem_epos(jl,itmp)-gsem_bcrd(jl,nfam))*
     $             gsem_bnrm(jl,nfam)
         enddo
         if (rtmp.gt.gsem_bext(nfam)) then
            neddy=neddy+1
            elist(neddy) = itmp
         endif
      enddo

      ! recreate eddies
      ifinit = .false.
      call gsem_gen_elist(elist,neddy,nfam,ifinit)

      return
      end subroutine
!=======================================================================
!> @brief Inital generation of all modes (gsem_mode.eq.3) for a given family
!! @ingroup gsyn_eddy
!! @param[in]  nfam       family number
      subroutine gsem_gen_mall(nfam)
      implicit none

      include 'SIZE'
      include 'TSTEP'           ! pi
      include 'PARALLEL'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam

      ! local variables
      integer il, jl, itmp

      ! functions
      real math_ran_rng
!-----------------------------------------------------------------------
      ! stamp logs
      call mntr_logi(gsem_id,lp_prd,'Initilise modes for family=',nfam)

      ! set all modes
      if (nid.eq.0) then
         do il=0,gsem_neddy(nfam)-1
            itmp = gsem_eoff(nfam)+il
            do jl=1,ldim
               gsem_eamp(jl,itmp) = math_ran_rng(0.0,1.0)
               gsem_ephs(jl,itmp) = math_ran_rng(0.0,2.0*pi)
            enddo
         enddo
      endif

      ! broadcast modes
      call bcast(gsem_eamp(1,gsem_eoff(nfam)),
     $     ldim*gsem_neddy(nfam)*WDSIZE)
      call bcast(gsem_ephs(1,gsem_eoff(nfam)),
     $     ldim*gsem_neddy(nfam)*WDSIZE)

      return
      end subroutine
!=======================================================================
!> @brief Time evolution of eddies
!! @ingroup gsyn_eddy
      subroutine gsem_evolve()
      implicit none

      include 'SIZE'
      include 'GSYEMD'

      ! local variables
      integer il
!-----------------------------------------------------------------------
      ! for different gSyEM modes
      if (gsem_mode.eq.1) then
         do il=1,gsem_nfam
            call gsem_adv_eddy(il)
            call gsem_iso(il)
         enddo
      elseif (gsem_mode.eq.2) then
      elseif (gsem_mode.eq.3) then
         do il=1,gsem_nfam
            call gsem_Dairay(il)
         enddo
      endif

      return
      end subroutine
!=======================================================================
!> @brief Fill velocity array with eddies; ISO
!! @ingroup gsyn_eddy
!! @param[in]  nfam       family number
!! @todo This routine is way too slow due to work imbalance
      subroutine gsem_iso(nfam)
      implicit none

      include 'SIZE'
      include 'TSTEP'           ! PI
      include 'GEOM'
      include 'SOLN'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam

      ! local variables
      integer il, jl, kl, ll, ml
      integer iel, ifc, ifn, itmp, itmp2
      real vrtmp(lx1*lz1,ldim), drtmp(ldim,lx1*lz1)
      real sqrt32
      parameter (sqrt32=sqrt(1.5))
      real bulk_vel(ldim), vb
      real sqrtn, dr(ldim), rtmp, rtmp2, fc(ldim), fcm
      ! to spped up eddy search
      integer icount, ipos, iepos(gsem_neddy_max)
      real el_dist, el_cnt(ldim)

      ! functions
      real vlsum, vlmax
!-----------------------------------------------------------------------
      call rzero(bulk_vel,ldim)
      ! this definition can differ slightly from origianl code !!!!!!
      ! approximated volume of controlled region
      vb = 2.0*gsem_area(nfam)*gsem_bext(nfam)
      ! weighting parameter
      sqrtn = sqrt(16.0*vb/15/pi/real(gsem_neddy(nfam)))

      ! loop over faces
      itmp = lx1*lz1
      do il=1,gsem_lfnum(nfam)
         ! local element and face number
         iel = gsem_lfmap(1,il,nfam)
         ifc = gsem_lfmap(2,il,nfam)
         ! position in face arrays
         ifn = gsem_foff(nfam) + il - 1

         ! extract coordinates and get element centre
         rtmp = 1.0/real(itmp)
         call ftovec(vrtmp,xm1,iel,ifc,lx1,ly1,lz1)
         do jl=1,itmp
            drtmp(1,jl) = vrtmp(jl,1)
         enddo
         el_cnt(1) = rtmp*vlsum(vrtmp,itmp)
         call ftovec(vrtmp,ym1,iel,ifc,lx1,ly1,lz1)
         do jl=1,itmp
            drtmp(2,jl) = vrtmp(jl,1)
         enddo
         el_cnt(2) = rtmp*vlsum(vrtmp,itmp)
         call ftovec(vrtmp,zm1,iel,ifc,lx1,ly1,lz1)
         do jl=1,itmp
            drtmp(ldim,jl) = vrtmp(jl,1)
         enddo
         el_cnt(ldim) = rtmp*vlsum(vrtmp,itmp)

         ! to speed up eddy search
         ! get characteristic element lenght (max centre-vertex distance)
         call rzero(vrtmp,itmp)
         do jl=1,ldim
            vrtmp(1,1) = vrtmp(1,1)+(drtmp(jl,1)-el_cnt(jl))**2
            vrtmp(2,1) = vrtmp(2,1)+(drtmp(jl,lx1)-el_cnt(jl))**2
            vrtmp(3,1) = vrtmp(3,1)+(drtmp(jl,itmp-lx1+1)-el_cnt(jl))**2
            vrtmp(4,1) = vrtmp(4,1)+(drtmp(jl,itmp)-el_cnt(jl))**2
         enddo
         el_dist = sqrt(vlmax(vrtmp,4))
         ! add max vertex size for given face
         el_dist = el_dist + vlmax(gsem_sigma(1,1,ifn),itmp)
         ! safety buffer
         el_dist = 1.1*el_dist

         ! find eddies inside a sphere surrounding element centre
         ! loop over all eddies
         icount = 0
         call izero(iepos,gsem_neddy_max)
         do kl=0,gsem_neddy(nfam)-1
            ipos = gsem_eoff(nfam)+kl
            rtmp = 0.0
            do ll=1,ldim
               dr(ll) = el_cnt(ll) - gsem_epos(ll,ipos)
               rtmp = rtmp + dr(ll)**2
            enddo
            rtmp = sqrt(rtmp)
            if (rtmp.le.el_dist) then
               icount = icount + 1
               iepos(icount) = ipos
            endif
         enddo

         ! add velocity profile
         if (gsem_dirl(nfam)) then
            call rzero(vrtmp,lx1*lz1*ldim)
            call subcol3(vrtmp(1,1),unx(1,1,ifc,iel),
     $           gsem_umean(1,1,ifn),itmp)
            call subcol3(vrtmp(1,2),uny(1,1,ifc,iel),
     $           gsem_umean(1,1,ifn),itmp)
            call subcol3(vrtmp(1,3),unz(1,1,ifc,iel),
     $           gsem_umean(1,1,ifn),itmp)
         else
            do kl=1,ldim
               call copy(vrtmp(1,kl),gsem_umean(1,1,ifn),itmp)
               call cmult(vrtmp(1,kl),gsem_dir(kl,nfam),itmp)
            enddo
         endif
         
         ! loop over all points in the face
         do jl=1,lz1
            itmp2 = (jl-1)*lz1
            do kl=1,lx1
               ! loop over all marked eddies
               do ll=1,icount
                  rtmp = 0.0
                  do ml=1,ldim
                     dr(ml)=drtmp(ml,itmp2+kl)-gsem_epos(ml,iepos(ll))
                     rtmp = rtmp + dr(ml)**2
                  enddo
                  rtmp = sqrt(rtmp)
                  rtmp2 = gsem_sigma(kl,jl,ifn)
                  if (rtmp.le.rtmp2) then
                     rtmp2 = 1.0/rtmp2

                     ! vector product of eddy relative position and
                     ! eddy orientation
                     ! NOTICE eddy orientation is not rotated
                     fc(1) = dr(2)*gsem_eps(3,iepos(ll)) -
     $                    dr(3)*gsem_eps(2,iepos(ll))
                     fc(2) = dr(3)*gsem_eps(1,iepos(ll)) -
     $                    dr(1)*gsem_eps(3,iepos(ll))
                     fc(3) = dr(1)*gsem_eps(2,iepos(ll)) -
     $                    dr(2)*gsem_eps(1,iepos(ll))

                     ! perturbation amplitude
                     ! original code looks strange !!!!!!!!!
                     ! short version
                     fcm = sqrtn*gsem_intn(kl,jl,ifn)*sqrt(rtmp2)*
     $                    sin(pi*rtmp*rtmp2)**2/rtmp**2

                     ! update purturbed velocity
                     do ml=1,ldim
                        vrtmp(itmp2+kl,ml) = vrtmp(itmp2+kl,ml) +
     $                       fcm*fc(ml)
                     enddo
                  endif
               enddo
            enddo
         enddo

         ! fill in velocity array
         call vectof(vx,vrtmp(1,1),iel,ifc,lx1,ly1,lz1)
         call vectof(vy,vrtmp(1,2),iel,ifc,lx1,ly1,lz1)
         call vectof(vz,vrtmp(1,3),iel,ifc,lx1,ly1,lz1)

         ! get current bulk velocity
         do ml=1,ldim
            call col2(vrtmp(1,ml),area(1,1,ifc,iel),itmp)
            bulk_vel(ml) = bulk_vel(ml) + vlsum(vrtmp(1,ml),itmp)
         enddo
      enddo

      ! global operations
      call gop(bulk_vel,fc,'+  ',ldim)

      ! average bulk velocity
      rtmp = 1.0/gsem_area(nfam)
      call cmult(bulk_vel,rtmp,ldim)

      ! correct inflow velocity
      itmp = lx1*lz1
      do ml=1,ldim
         rtmp = gsem_ub(ml,nfam) - bulk_vel(ml)
         call cfill(vrtmp(1,ml),rtmp,itmp)
      enddo
      ! loop over faces
      do il=1,gsem_lfnum(nfam)
         ! local element and face number
         iel = gsem_lfmap(1,il,nfam)
         ifc = gsem_lfmap(2,il,nfam)
         call vectof_add(vx,vrtmp(1,1),iel,ifc,lx1,ly1,lz1)
         call vectof_add(vy,vrtmp(1,2),iel,ifc,lx1,ly1,lz1)
         call vectof_add(vz,vrtmp(1,3),iel,ifc,lx1,ly1,lz1)
      enddo

!      call nekgsync()

      return
      end subroutine
!=======================================================================
!> @brief Fill velocity array with eddies; Dairay
!! @ingroup gsyn_eddy
!! @param[in]  nfam       family number
!! @todo this routine is not done yet
      subroutine gsem_Dairay(nfam)
      implicit none

      include 'SIZE'
!      include 'TSTEP'           ! PI
      include 'GEOM'
      include 'SOLN'
      include 'FRAMELP'
      include 'GSYEMD'

      ! argument list
      integer nfam

      ! local variables
!-----------------------------------------------------------------------


      return
      end subroutine
!=======================================================================
