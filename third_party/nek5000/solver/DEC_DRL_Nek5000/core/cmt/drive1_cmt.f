c-----------------------------------------------------------------------
      subroutine cmt_nek_advance
c
c     Solve the Euler equations

      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      include 'SOLN'
      include 'GEOM'
      include 'CTIMER'
      include 'CMTDATA'
      include 'CMTTIMERS'
      
      integer e,eq
      character*32 dumchars

      ftime_dum = dnekclock()
      nxyz1=lx1*ly1*lz1
      n = nxyz1*lelt*toteq
      nfldpart = ndim*npart

      if(istep.eq.1) then 
         call cmt_ics
!        do e=1,nelt
!           do i=1,nxyz1
!              write(401,'(7e17.8)')  xm1(i,1,1,1),ym1(i,1,1,1),
!    >      u(i,1,1,1,e),u(i,1,1,2,e),u(i,1,1,3,e),u(i,1,1,4,e),
!    >      u(i,1,1,5,e)
!           enddo
!        enddo
!        call exitt
         call set_tstep_coef
         call cmt_flow_ics
         call init_cmt_timers
c all point particles are initialized and 
c preprocessing of interpolation step 
         call usr_particles_init
      endif

      nstage = 3
      do stage=1,nstage
         if (stage.eq.1) call copy(res3(1,1,1,1,1),U(1,1,1,1,1),n)

         rhst_dum = dnekclock()
         call compute_rhs_and_dt
         rhst = rhst + dnekclock() - rhst_dum
c particle equations of motion are solved (also includes forcing)
c In future this subroutine may compute the back effect of particles
c on the fluid and suitably modify the residue computed by 
c compute_rhs_dt for the 5 conserved variables
         call usr_particles_solver

! JH111815 soon....
! JH082316 someday...maybe?
!        do eq=1,toteq
!           call fbinvert(res1(1,1,1,1,eq))
!        enddo

         do e=1,nelt
            do eq=1,toteq
            do i=1,nxyz1
c multiply u with bm1 as res has been multiplied by bm1 in compute_rhs
               u(i,1,1,eq,e) = bm1(i,1,1,e)*tcoef(1,stage)
     >                     *res3(i,1,1,eq,e)+bm1(i,1,1,e)*
     >                     tcoef(2,stage)*u(i,1,1,eq,e)-
     >                     tcoef(3,stage)*res1(i,1,1,e,eq)
c              u(i,1,1,eq,e) = bm1(i,1,1,e)*u(i,1,1,eq,e) - DT *
c    >                        (c1*res1(i,1,1,e,eq) + c2*res2(i,1,1,e,eq)
c    >                       + c3*res3(i,1,1,e,eq))
c-----------------------------------------------------------------------
c this completely stops working if B become nondiagonal for any reason.
! JH111815 in fact, I'd like to redo the time marching stuff above and
!          have an fbinvert call for res1
               u(i,1,1,eq,e) = u(i,1,1,eq,e)/bm1(i,1,1,e)
c that completely stops working if B become nondiagonal for any reason.
!-----------------------------------------------------------------------
            enddo
            enddo
         enddo
      enddo

      call compute_primitive_vars
      ftime = ftime + dnekclock() - ftime_dum

      if (mod(istep,iostep).eq.0.or.istep.eq.1)then
         call out_pvar_nek
         call out_fld_nek
         call mass_balance(if3d)
c dump out particle information. 
         call usr_particles_io(istep)
      end if

      call print_cmt_timers

 101  format(4(2x,e18.9))
      return
      end

c-----------------------------------------------------------------------

      subroutine compute_rhs_and_dt
!> doxygen comments look like this
      include 'SIZE'
      include 'TOTAL'
      include 'DG'
      include 'CMTDATA'
      include 'CTIMER'

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*3*lfq) ! might not need ldim
! not sure if viscous surface fluxes can live here yet
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      real graduf

      integer e,eq
      real wkj(lx1+lxd)
      character*32  dumchars

      if (nxd.gt.nx1) then
         call set_dealias_face
      else
         call set_alias_rx(istep)
      endif
!     call set_dealias_rx ! done in set_convect_cons,
! JH113015                ! now called from compute_primitive_variables

!     filter the conservative variables before start of each
!     time step
!     if(IFFLTR)  call filter_cmtvar(IFCNTFILT)
!        primitive vars = rho, u, v, w, p, T, phi_g
      if (istep.eq.1) then
         call compute_primitive_vars
!-----------------------------------------------------------------------
! JH082216 Transport properties are for the higher-derivative operators.
!          Artificial viscosity is now computed in the branches of
!          compute_transport_props
!-----------------------------------------------------------------------
         call compute_transport_props
      else
         if(stage.gt.1) then
            call compute_primitive_vars
            call compute_transport_props
         endif
      endif
!-----------------------------------------------------------------------
! JH072914 We can really only proceed with dt once we have current
!          primitive variables. Only then can we compute CFL and/or dt.
!-----------------------------------------------------------------------
      if(stage.eq.1) call setdtcmt
      ntot = lx1*ly1*lz1*lelt*toteq
      call rzero(res1,ntot)

!     !Total_eqs = 5 (we will set this up so that it can be a user 
!     !defined value. 5 will be its default value)
!     !eq = 1 -------- Mass balance
!     !eq = 2 -------- x  momentum 
!     !eq = 3 -------- y  momentum 
!     !eq = 4 -------- z  momentum 
!     !eq = 5 -------- Energy Equation 

!-----------------------------------------------------------------------
! JH060314 Compute inviscid surface fluxes now that we have the
!          primitive variables.
!-----------------------------------------------------------------------
      call fluxes_full_field

      nstate=nqq
      nfq=nx1*nz1*2*ndim*nelt
      iqm =1
      iqp =iqm+nstate*nfq
      iflx=iqp+nstate*nfq
      do eq=1,toteq
         ieq=(eq-1)*ndg_face+iflx
         call surface_integral_full(res1(1,1,1,1,eq),flux(ieq))
      enddo

               !                   -
      iuj=iflx ! overwritten with U -{{U}}
!-----------------------------------------------------------------------
!                          /     1  T \
! JH082316 imqqtu computes | I - -QQ  | U for all 5 conserved variables
!                          \     2    /
! which I now make the following be'neon-billboarded assumption:
!***********************************************************************
! ASSUME CONSERVED VARS U1 THROUGH U5 ARE CONTIGUOUSLY STORED
! SEQUENTIALLY IN /CMTSURFLX/ i.e. that iu2=iu1+1, etc.
! CMTDATA BETTA REFLECT THIS!!!
!***********************************************************************
      ium=(iu1-1)*nfq+iqm
      iup=(iu1-1)*nfq+iqp
      call   imqqtu(flux(iuj),flux(ium),flux(iup))
      call igtu_cmt(flux(iqm),flux(iuj),graduf) ! [[u]].{{gradv}}

      do e=1,nelt
!-----------------------------------------------------------------------
! JH082216 Since the dawn of CMT-nek we have called this particular loop
!***********************************************************************
!*         "THE" ELEMENT LOOP                                          *
!***********************************************************************
!          since it does several operations, mostly for volume integrals,
!          for all equations, one element at a time. If we get memory
!          under control and GPUs really need to act on gigabytes all
!          at once, then this and its dependents can still have their
!          loop order flipped and things like totalh declared for
!          15 full fields or more.
!-----------------------------------------------------------------------
! Get user defined forcing from userf defined in usr file
         call cmtusrf(e)
         call compute_gradients(e) ! gradU
         do eq=1,toteq
            call convective_cmt(e,eq)        ! convh & totalh -> res1
            call    viscous_cmt(e,eq) ! diffh -> half_iku_cmt -> res1
                                             !       |
                                             !       -> diffh2graduf
! Compute the forcing term in each of the 5 eqs
            call compute_forcing(e,eq)
         enddo
      enddo

! get the rest of Hij^{d*}
      call igu_cmt(flux(iqm),graduf)
      do eq=1,toteq
         ieq=(eq-1)*ndg_face+iqm
!Finally add viscous surface flux functions of derivatives to res1.
         call surface_integral_full(res1(1,1,1,1,eq),flux(ieq))
      enddo

      return
      end
!-----------------------------------------------------------------------

      subroutine set_tstep_coef
      include 'SIZE'
      include 'TSTEP'

      COMMON /TIMESTEPCOEf/ tcoef(3,3)

      COMMON /TSTEPSTAGE/  stage,nstage
      integer              stage,nstage

      nstage = 3
      do stage=1,nstage
         if(stage.eq.1)then
            tcoef(1,stage) = 0.0
            tcoef(2,stage) = 1.0 
            tcoef(3,stage) = dt 
         endif
         if(stage.eq.2)then
            tcoef(1,stage) = 3.0/4.0
            tcoef(2,stage) = 1.0/4.0 
            tcoef(3,stage) = dt/4.0 
         endif
         if(stage.eq.3)then
            tcoef(1,stage) = 1.0/3.0
            tcoef(2,stage) = 2.0/3.0 
            tcoef(3,stage) = dt*2.0/3.0 
         endif
      enddo
      return
      end
!-----------------------------------------------------------------------

      subroutine cmt_flow_ics
      include 'SIZE'
      include 'CMTDATA'
      include 'SOLN'

      integer e
      nxyz1 = nx1*ny1*nz1
      n     = nxyz1*lelt*toteq
      if (ifrestart)then
         do e=1,nelt
            call copy(U(1,1,1,2,e),vx(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,3,e),vy(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,4,e),vz(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,5,e),t(1,1,1,e,1),nxyz1) 
            call copy(U(1,1,1,1,e),pr(1,1,1,e),nxyz1) 
         enddo
      endif
      call rzero(res1,n)
      call rzero(res2,n)
      return
      end
!-----------------------------------------------------------------------

      subroutine print_cmt_timers
      include 'SIZE'
      include 'CMTTIMERS'
      include 'TSTEP'
      include 'PARALLEL'

c we need our own IO features. Until then we use the default nek routines
      if ((mod(istep,flio_freq).eq.0.and.istep.gt.0)
     $                               .or.istep.eq.nstep)then
         dmtime1 = ftime/istep
         dtime_ = glsum(dmtime1,1)
         if(nio.eq.0) write(6,*) 'fluid rhs compute time(Avg)  '
     $               ,dtime_/np
      endif
      return 
      end
!-----------------------------------------------------------------------

      subroutine init_cmt_timers
      include 'CMTTIMERS'

      rhst    = 0.00
      ftime   = 0.00

      return
      end
!-----------------------------------------------------------------------
