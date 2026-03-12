!-----------------------------------------------------------------------
      subroutine lpm_init(rparam,yp,nyp,pp,npp,npart,time_)
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
#     include "LPM"

      real     yp(*)
      real     pp(*)
      real     rparam(*)

      lpm_d2chk(2) = 0.0
      lpm_npart = npart
      lpm_timef = time_

      if (nyp.gt.LPM_LRS)
     $   call exitti('nyp > LPM_LRS$',nyp)

      if (npp.gt.LPM_LRP)
     $   call exitti('npp > LPM_LRP$',npp)

      if (lpm_npart.gt.LPM_LPART)
     $   call exitti('lpm_npart > LPM_LPART$',lpm_npart)

      call copy(lpm_y    ,yp,lpm_npart*nyp)
      call copy(lpm_rprop,pp,lpm_npart*npp)

      if (nio.eq.0) then
         write(6,*) ' '
         write(6,*) 'initialize LPM'
         if (int(rparam(1)) .ne. 0) 
     $   write(6,*) 'overwrite default settings'
      endif

      call lpm_rparam_set(rparam)
      call lpm_tag_init
      call lpm_tag_set

      ! get domain bounds
      call domain_size( lpm_xdrange(1,1),lpm_xdrange(2,1)
     >                 ,lpm_xdrange(1,2),lpm_xdrange(2,2)
     >                 ,lpm_xdrange(1,3),lpm_xdrange(2,3))

      ! send particles to correct rank
      call lpm_init_filter

      call lpm_comm_setup
      call lpm_interpolate_setup

      ! two-way coupling init
      if (int(lpm_rparam(4)) .ne. 1) then
         call lpm_project
      endif

      call nekgsync()
      if (nio.eq.0) then
        write(6,*) 'done :: initialize LPM' 
        write(6,*) ' '
      endif

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_rparam_set(rparam)
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      real rparam(*)

      if (rparam(1) .eq. 0) then ! set defaults
         lpm_rparam(2)  = 1      ! time integration method
         lpm_rparam(3)  = lx1-1  ! polynomial order of mesh
         lpm_rparam(4)  = 0      ! use 1 for tracers only
         lpm_rparam(5)  = 0      ! index of filter non-dim in rprop
         lpm_rparam(6)  = 0      ! non-dimensional Gaussian filter width
         lpm_rparam(7)  = 0      ! percent decay of Gaussian filter
         lpm_rparam(8)  = 0      ! periodic in x
         lpm_rparam(9)  = 0      ! periodic in y
         lpm_rparam(10) = 0      ! periodic in z
      else ! custom values
         do i=2,lpm_nparam
            lpm_rparam(i) = rparam(i)
         enddo
      endif

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_init_filter
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      rsig  = 0.0
      jdp   = int(lpm_rparam(5))
      filt  = lpm_rparam(6)
      alph  = lpm_rparam(7)

      rdx_max = 0.0

      lx1m1 = max(1,lx1-1)
      ly1m1 = max(1,ly1-1)
      lz1m1 = max(1,lz1-1)

      ! first, compute what is the smallest filter size for the grid
      do ie=1,nelt
      do k=1,lz1m1
      do j=1,ly1m1
      do i=1,lx1m1
         ip1 = i+1
         jp1 = j+1
         kp1 = k+1

         rdx_test = xm1(ip1,j,k,ie) - xm1(i,j,k,ie)
         rdy_test = ym1(i,jp1,k,ie) - ym1(i,j,k,ie)
         if (if3d) 
     >   rdz_test = zm1(i,j,kp1,ie) - zm1(i,j,k,ie)

         rdist = max(rdx_test,rdy_test)
         if (if3d)
     >   rdist = max(rdz_test,rdist)

         rdx_dy_test = sqrt(rdx_test**2 + rdy_test**2)
         rdist = max(rdx_dy_test,rdist)
         if (if3d) then
            rdy_dz_test = sqrt(rdy_test**2 + rdz_test**2)
            rdist = max(rdy_dz_test,rdist)
            rdx_dz_test = sqrt(rdx_test**2 + rdz_test**2)
            rdist = max(rdx_dz_test,rdist)
         endif

         if (rdist .gt. rdx_max) rdx_max = rdist
      enddo
      enddo
      enddo
      enddo
      rdx_max = glmax(rdx_max,1)

      lpm_rdx_max = rdx_max

      rsig_dx_min_set  = 0.25
      rfilt_dp_min_set = 1.0
      rsig_dx_min      = 1E12

      tol = 1.e-12
      if (wdsize.eq.4) tol = 1.e-6

      rfilt = lpm_rparam(6)
      lpm_d2chk(2) = -1
      if (abs(alph) .lt. tol) then
         alph = 1E-3
         lpm_rparam(7) = alph
      endif

      do i=1,lpm_npart

         if (filt .lt. tol) then
            rsig_test_grid = rsig_dx_min_set*rdx_max + tol
            rsig_test_diam = rfilt_dp_min_set*lpm_rprop(jdp,i)
     >                       /(2.*sqrt(2.*log(2.)))
            rsig_test = max(rsig_test_grid,rsig_test_diam)
            filt      = rsig_test/lpm_rprop(jdp,i)*2.*sqrt(2.*log(2.))
            rfilt = filt
         else
            rsig_test = filt*lpm_rprop(jdp,i)/(2.*sqrt(2.*log(2.)))
         endif

         rsig_dx = rsig_test/rdx_max
         if (rsig_dx .lt. rsig_dx_min) rsig_dx_min = rsig_dx

         if (rsig_test .gt. rsig) rsig = rsig_test
      enddo
      rsig = glmax(rsig,1)
      rsig_dx_min = glmin(rsig_dx_min,1)
         
      lpm_d2chk(2) = rsig*sqrt(-2*log(alph))

      if (rsig_dx_min .lt. rsig_dx_min_set) then
         if (nid .eq. 0) then
            filt_new = filt/(rsig_dx_min/rsig_dx_min_set)
            write(6,100) filt, filt_new
 100        format('Reset Gaussian filter width from', E14.7
     >             ' to', E14.7, ' or larger')
         endif
         call exitt
      endif

      lpm_rparam(6) = glmax(rfilt,1)
      lpm_d2chk(2)  = glmax(lpm_d2chk(2),1)
      if (int(lpm_rparam(4)) .eq. 1) lpm_d2chk(2) = 0

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_tag_set
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      do i=1,lpm_npart
         if (lpm_iprop(5,i) .eq. -1) lpm_iprop(5,i) = nid
         if (lpm_iprop(6,i) .eq. -1) lpm_iprop(6,i) = istep
         if (lpm_iprop(7,i) .eq. -1) lpm_iprop(7,i) = i
      enddo

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_tag_init
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      if (.not. LPM_RESTART) then
         do i=1,LPM_LPART
            lpm_iprop(5,i) = -1
            lpm_iprop(6,i) = -1
            lpm_iprop(7,i) = -1
         enddo
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_solve(time_)
      include 'SIZE'
      include 'TSTEP'
#     include "LPM"

      real time_

      ts   = dnekclock()

      isol = lpm_rparam(2)
      dt_  = time_ - lpm_timef 
      n    = LPM_NPART*LPM_LRS

      ! save previous solution
      call copy(lpm_y1,lpm_y,n)

      if (isol .eq. 1) then
         call lpm_rk3_driver(time_,dt_,lpm_y1,lpm_y,lpm_ydot,n)
      else
         call exitti('unknown LPM integrator$',isol)
      endif

      lpm_timef = time_

      if(nio.eq.0)
     &   write(*,'(4x,i7,a,1p3e12.4)')
     &   istep,'  LPM-solver done',time,dnekclock()-ts, lpm_timef

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_rk3_driver(time_,dt_,y0,y,ydot,n)
      include 'SIZE'
#     include "LPM"

      real time_
      real y0(*)
      real y(*)
      real ydot(*)

      parameter(nstage = 3)
      real tcoef(3,3)

      call lpm_rk3_coeff(tcoef,dt_)

      do istage=1,nstage
         call lpm_fun(time_,y,ydot)
         do i=1,n
            y(i) = tcoef(1,istage)*y0  (i)
     >           + tcoef(2,istage)*y   (i)
     >           + tcoef(3,istage)*ydot(i)
         enddo
      enddo

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_interpolate_setup
      include 'SIZE'
#     include "LPM"

      call lpm_move_outlier
      call lpm_comm_bin_setup
      call lpm_comm_findpts

      call lpm_comm_crystal

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_interpolate_fld(jp,fld)
      include 'SIZE'
#     include "LPM"

      common /intp_h/ ih_intp(2,1)

      real fld(*)

      ih_intp1 = ih_intp(1,i_fp_hndl)

c     call fgslib_findpts_eval_local( ih_intp1
c    >                               ,lpm_rprop (jp,1)
c    >                               ,LPM_LRP
c    >                               ,lpm_iprop (2,1)
c    >                               ,LPM_LIP
c    >                               ,lpm_rprop2(1,1)
c    >                               ,LPM_LRP2
c    >                               ,LPM_NPART
c    >                               ,fld)

         call fgslib_findpts_eval( ih_intp1
     >                           ,lpm_rprop (jp,1)
     >                           ,LPM_LRP
     >                           ,lpm_iprop (1,1)
     >                           ,LPM_LIP
     >                           ,lpm_iprop (3,1)
     >                           ,LPM_LIP
     >                           ,lpm_iprop (2,1)
     >                           ,LPM_LIP
     >                           ,lpm_rprop2(1,1)
     >                           ,LPM_LRP2
     >                           ,LPM_NPART
     >                           ,fld)

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_project
      include 'SIZE'
#     include "LPM"

      if (int(lpm_rparam(4)) .ne. 1) then
         call lpm_comm_ghost_create
         call lpm_comm_ghost_send
         call lpm_solve_project
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_solve_project
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      real               phig(lx1,ly1,lz1,lelt)
      common /otherpvar/ phig

      real    multfci
      integer e

      common /lpm_fix/ phigdum,phigvdum
      real phigdum(lx1,ly1,lz1,lelt,3),phigvdum(lx1,ly1,lz1,lelt)

      real xrun(lx1),yrun(ly1),zrun(lz1)

      integer xrune(lx1),yrune(ly1),zrune(lz1)

      real    rproj(1+LPM_LRP_GP,LPM_LPART+LPM_LPART_GP)
      integer iproj(4,LPM_LPART+LPM_LPART_GP)

      logical partl

      nxyz = lx1*ly1*lz1

      nxyzdum = nxyz*LPM_LRP_PRO*lpm_neltb
      call rzero(lpm_pro_fldb,nxyzdum)

      d2chk2_sq = lpm_d2chk(2)**2

      ! real particles
      lpm_jxgp  = 1
      lpm_jygp  = 2
      lpm_jzgp  = 3
      lpm_jdp   = 4
      filt      = lpm_rparam(6)
      alph      = lpm_rparam(7)

      nxyze = lx1*ly1*lz1*nelt
      nxyz = lx1*ly1*lz1

c     lpm_npart_gp = 0

      ! real particles
      do ip=1,lpm_npart
         rsig    = filt*lpm_cp_map(lpm_jdp,ip)/(2.*sqrt(2.*log(2.)))
         multfci = 1./(sqrt(2.*pi)**2 * rsig**2) 
         if (if3d) multfci = multfci**(1.5d+0)
         ralph   = filt/2.*sqrt(-log(alph)/log(2.))*
     >             lpm_cp_map(lpm_jdp,ip)
         ralph2   = ralph**2
         rbexpi   = 1./(-2.*rsig**2)

c        if (.not. if3d) 
c    >          multfci = multfci/lpm_cp_map(lpm_jdp,ip)

         rproj(1 ,ip) = rbexpi
         rproj(2 ,ip) = lpm_cp_map(lpm_jxgp,ip)
         rproj(3 ,ip) = lpm_cp_map(lpm_jygp,ip)
         rproj(4 ,ip) = lpm_cp_map(lpm_jzgp,ip)


         do j=5,LPM_LRP_GP
            rproj(j,ip) = lpm_cp_map(j,ip)*multfci
         enddo
                    
         iproj(1,ip)  = lpm_iprop(8,ip)
         iproj(2,ip)  = lpm_iprop(9,ip)
         iproj(3,ip)  = lpm_iprop(10,ip)
         iproj(4,ip)  = lpm_iprop(11,ip)
      enddo

      ! ghost particles
      do ip=1,lpm_npart_gp
         rsig    = filt*lpm_rprop_gp(lpm_jdp,ip)/(2.*sqrt(2.*log(2.)))
         multfci = 1./(sqrt(2.*pi)**2 * rsig**2) 
         if (if3d) multfci = multfci**(1.5d+0)
         ralph   = filt/2.*sqrt(-log(alph)/log(2.))*
     >             lpm_rprop_gp(lpm_jdp,ip)
         ralph2   = ralph**2
         rbexpi   = 1./(-2.*rsig**2)

c        if (.not. if3d) 
c    >          multfci = multfci/lpm_rprop_gp(lpm_jdp,ip)

         rproj(1 ,ip+lpm_npart) = rbexpi
         rproj(2 ,ip+lpm_npart) = lpm_rprop_gp(lpm_jxgp,ip)
         rproj(3 ,ip+lpm_npart) = lpm_rprop_gp(lpm_jygp,ip)
         rproj(4 ,ip+lpm_npart) = lpm_rprop_gp(lpm_jzgp,ip)

         do j=5,LPM_LRP_GP
            rproj(j,ip+lpm_npart) = lpm_rprop_gp(j,ip)*multfci
         enddo
                    
         iproj(1,ip+lpm_npart)  = lpm_iprop_gp(2,ip)
         iproj(2,ip+lpm_npart)  = lpm_iprop_gp(3,ip)
         iproj(3,ip+lpm_npart)  = lpm_iprop_gp(4,ip)
         iproj(4,ip+lpm_npart)  = lpm_iprop_gp(5,ip)
      enddo

      ndum = lpm_npart+lpm_npart_gp
c     ndum = lpm_npart_gp

      if (int(lpm_rparam(3)) .eq. 1) then
         do ip=1,ndum
            ii    = iproj(1,ip)
            jj    = iproj(2,ip)
            kk    = iproj(3,ip)
            ndum  = iproj(4,ip)
          
            ilow  = ii-1
            ihigh = ii+1
            jlow  = jj-1
            jhigh = jj+1
            klow  = kk-1
            khigh = kk+1
         
            do ie=1,lpm_neltb
         
               if (lpm_el_map(1,ie) .gt. ndum) exit
               if (lpm_el_map(2,ie) .lt. ndum) cycle 
         
               if (lpm_el_map(3,ie) .gt. ihigh) cycle
               if (lpm_el_map(4,ie) .lt. ilow)  cycle
               if (lpm_el_map(5,ie) .gt. jhigh) cycle
               if (lpm_el_map(6,ie) .lt. jlow)  cycle
               if (lpm_el_map(7,ie) .gt. khigh) cycle
               if (lpm_el_map(8,ie) .lt. klow)  cycle

               do i=1,lx1
                  xrun(i) = (lpm_xm1b(i,1,1,1,ie) - rproj(2,ip))**2
                  xrune(i) = 1
                  if (xrun(i) .gt. d2chk2_sq) xrune(i) = -1
               enddo
               do j=1,ly1
                  yrun(j) = (lpm_xm1b(1,j,1,2,ie) - rproj(3,ip))**2
                  yrune(j) = 1
                  if (yrun(j) .gt. d2chk2_sq) yrune(j) = -1
               enddo
               do k=1,lz1
                  zrun(k) = (lpm_xm1b(1,1,k,3,ie) - rproj(4,ip))**2
                  if (.not. if3d) zrun(k) = 0.0
                  zrune(k) = 1
                  if (zrun(k) .gt. d2chk2_sq) zrune(k) = -1
               enddo
         
               do k=1,lz1
                  if (zrune(k) .lt. 0) cycle
               do j=1,ly1
                  if (yrune(j) .lt. 0) cycle
                  rdum1 = yrun(j) + zrun(k)
               do i=1,lx1
                  if (lpm_modgp(i,j,k,ie,4).ne.iproj(4,ip)) cycle
                  rdist2 = rdum1 + xrun(i)
                  if (rdist2 .gt. d2chk2_sq) cycle

                  rexp = exp(rdist2*rproj(1,ip))
                  
                  do jj=1,LPM_LRP_PRO
                     j1 = jj+4
                     LPM_PRO_FLDB(i,j,k,jj,ie) = 
     >                                   LPM_PRO_FLDB(i,j,k,jj,ie) 
     >                                 + rproj(j1,ip)*rexp
                  enddo
               enddo
               enddo
               enddo
            enddo
         enddo
      else
         do ip=1,ndum
            ii    = iproj(1,ip)
            jj    = iproj(2,ip)
            kk    = iproj(3,ip)
            ndum  = iproj(4,ip)
          
            ilow  = ii-1
            ihigh = ii+1
            jlow  = jj-1
            jhigh = jj+1
            klow  = kk-1
            khigh = kk+1
         
            do ie=1,lpm_neltb
         
               if (lpm_el_map(1,ie) .gt. ndum) exit
               if (lpm_el_map(2,ie) .lt. ndum) cycle 
         
               if (lpm_el_map(3,ie) .gt. ihigh) cycle
               if (lpm_el_map(4,ie) .lt. ilow)  cycle
               if (lpm_el_map(5,ie) .gt. jhigh) cycle
               if (lpm_el_map(6,ie) .lt. jlow)  cycle
               if (lpm_el_map(7,ie) .gt. khigh) cycle
               if (lpm_el_map(8,ie) .lt. klow)  cycle
         
               do i=1,nxyz
                  if (lpm_modgp(i,1,1,ie,4).ne.iproj(4,ip)) cycle
                  rdist2  = (lpm_xm1b(i,1,1,1,ie) - rproj(2,ip))**2 +
     >                      (lpm_xm1b(i,1,1,2,ie) - rproj(3,ip))**2
                  if(if3d) rdist2 = rdist2                      +
     >                      (lpm_xm1b(i,1,1,3,ie) - rproj(4,ip))**2
                  if (rdist2 .gt. d2chk2_sq) cycle
               
                  rexp = exp(rdist2*rproj(1,ip))
                  
                  do j=1,LPM_LRP_PRO
                     j1 = j+4
                     LPM_PRO_FLDB(i,1,1,j,ie) = 
     >                               LPM_PRO_FLDB(i,1,1,j,ie) 
     >                             + rproj(j1,ip)*rexp
                  enddo
                enddo
             enddo
          enddo
      endif

      ! now send xm1b to the processors in nek that hold xm1

      neltbc = lpm_neltb
      ndum = LPM_LRMAX*neltbc
      call icopy(lpm_er_mapc,lpm_er_map,ndum)
      do ie=1,neltbc
         lpm_er_mapc(5,ie) = lpm_er_mapc(2,ie)
         lpm_er_mapc(6,ie) = lpm_er_mapc(2,ie)
      enddo
      nl = 0
      nii = LPM_LRMAX
      njj = 6
      nrr = nxyz*LPM_LRP_PRO
      call fgslib_crystal_tuple_transfer(i_cr_hndl,neltbc,lpm_lbmax
     >                  , lpm_er_mapc,nii,partl,nl,lpm_pro_fldb,nrr,njj)

      ! add the fields from the bins to ptw array
      nlxyze = lx1*ly1*lz1*lelt
      call rzero(LPM_PRO_FLD,nlxyze*LPM_LRP_PRO)
      do ie=1,neltbc
         iee = lpm_er_mapc(1,ie)
         do ip=1,LPM_LRP_PRO
         do k=1,nz1
         do j=1,ny1
         do i=1,nx1
            LPM_PRO_FLD(i,j,k,iee,ip) = LPM_PRO_FLD(i,j,k,iee,ip) +
     >                                  LPM_PRO_FLDB(i,j,k,ip,ie)
         enddo
         enddo
         enddo
         enddo
      enddo

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_rk3_coeff(tcoef,rdt)
      include 'SIZE'
      include 'TOTAL'

      real tcoef(*)
      real rdt

      tcoef(1) = 0.0
      tcoef(2) = 1.0 
      tcoef(3) = rdt
      tcoef(4) = 3.0/4.0
      tcoef(5) = 1.0/4.0 
      tcoef(6) = rdt/4.0 
      tcoef(7) = 1.0/3.0
      tcoef(8) = 2.0/3.0 
      tcoef(9) = rdt*2.0/3.0 

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_qtl_pvol(divin,phipin)
c
c     Computes modified divergence constraint for multiphase dense
c     incompressible flow
c
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      common /phig_qtl_blk/ phig_last
      real phig_last(lx1,ly1,lz1,lelt)

      real divin(lx2,ly2,lz2,lelv), phipin(lx1,ly1,lz1,lelt)

      COMMON /SCRNS/ ur(lx1,ly1,lz1,lelt)
     >              ,us(lx1,ly1,lz1,lelt)
     >              ,ut(lx1,ly1,lz1,lelt)
     >              ,phigin(lx1,ly1,lz1,lelt)
     >              ,phig_qtl(lx1,ly1,lz1,lelt)
     >              ,grad_dot(lx1,ly1,lz1,lelt)

      integer icalld
      save    icalld
      data    icalld  /-1/

      icalld = icalld + 1
      nxyze = lx1*ly1*lz1*lelt

      rdt_in = 1./dt

      call rzero(phig_qtl,nxyze)

      if (icalld .eq. 0) then
         call rone(phig_last,nxyze)
         call sub2(phig_last,phipin,nxyze)
      endif

      call rone(phigin,nxyze)
      call sub2(phigin,phipin,nxyze)
      
c     if (icalld .lt. 5) goto 123

      call opgrad(ur,us,ut,phigin)
      call sub3(phig_qtl,phigin,phig_last,nxyze)
      call cmult(phig_qtl,rdt_in,nxyze)
      call vdot3(grad_dot,vx,vy,vz,ur,us,ut,nxyze)
      call add2(phig_qtl,grad_dot,nxyze)
      call invcol2(phig_qtl,phigin,nxyze)
      call chsign(phig_qtl,nxyze)

      call copy(phig_last,phigin,nxyze)

      do ie=1,nelt
         call map12(divin(1,1,1,ie),phig_qtl(1,1,1,ie),ie)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_move_outlier
      include 'SIZE'
#     include "LPM"

      integer in_part(LPM_LPART)
      integer jj(3)

      iperiodicx = int(lpm_rparam(8))
      iperiodicy = int(lpm_rparam(9))
      iperiodicz = int(lpm_rparam(10))

      jj(1) = 1
      jj(2) = 2
      jj(3) = 3

      do i=1,lpm_npart
         isl = (i -1) * LPM_LRS + 1
         in_part(i) = 0
         do j=0,ndim-1
            jchk = jj(j+1)
            if (lpm_y(jchk,i).lt.lpm_xdrange(1,j+1))then
               if (((iperiodicx.eq.1) .and. (j.eq.0)) .or.   ! periodic
     >             ((iperiodicy.eq.1) .and. (j.eq.1)) .or.     
     >             ((iperiodicz.eq.1) .and. (j.eq.2)) ) then
                   lpm_y(jchk,i) = lpm_xdrange(2,j+1) - 
     &                         abs(lpm_xdrange(1,j+1) - lpm_y(jchk,i))
                   lpm_y1(isl+j)   = lpm_xdrange(2,j+1) +
     &                         abs(lpm_xdrange(1,j+1) - lpm_y1(isl+j))
                  goto 1512
                endif
            endif
            if (lpm_y(jchk,i).gt.lpm_xdrange(2,j+1))then
               if (((iperiodicx.eq.1) .and. (j.eq.0)) .or.   ! periodic
     >             ((iperiodicy.eq.1) .and. (j.eq.1)) .or.     
     >             ((iperiodicz.eq.1) .and. (j.eq.2)) ) then
                   lpm_y(jchk,i) = lpm_xdrange(1,j+1) +
     &                         abs(lpm_y(jchk,i) - lpm_xdrange(2,j+1))
                   lpm_y1(isl+j)   = lpm_xdrange(1,j+1) +
     &                         abs(lpm_y1(isl+j) - lpm_xdrange(2,j+1))
                  goto 1512
                endif
            endif
            if (lpm_iprop(1,i) .eq. 2) then
               in_part(i) = -1 ! only if periodic check fails it will get here
            endif
 1512 continue
         enddo
      enddo

      ic = 0
      do i=1,lpm_npart
         if (in_part(i).eq.0) then
            ic = ic + 1 
            if (i .ne. ic) then
               isl = (i -1) * LPM_LRS + 1
               isr = (ic-1) * LPM_LRS + 1
               call copy(lpm_y     (1,ic),lpm_y(1,i)     ,LPM_LRS)
               call copy(lpm_y1    (isr) ,lpm_y1(isl)    ,LPM_LRS)
               call copy(lpm_ydot  (1,ic),lpm_ydot(1,i)  ,LPM_LRS)
               call copy(lpm_ydotc (1,ic),lpm_ydotc(1,i) ,LPM_LRS)
               call copy(lpm_rprop (1,ic),lpm_rprop(1,i) ,LPM_LRP)
               call copy(lpm_rprop2(1,ic),lpm_rprop2(1,i),LPM_LRP2)
               call icopy(lpm_iprop(1,ic),lpm_iprop(1,i) ,LPM_LIP)
            endif
         endif
      enddo
      lpm_npart = ic

      return
      end
!-----------------------------------------------------------------------
