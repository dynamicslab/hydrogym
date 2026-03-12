C> @file diffusive_cmt.f routines for diffusive fluxes.
C> Some surface. Some volume. All pain. Jacobians and other factorizations.

C> \ingroup vsurf
C> @{
C> ummcu = \f$\mathbf{U}^--\{\{\mathbf{U}\}\}\f$
      subroutine imqqtu(ummcu,uminus,uplus)
! Computes (I-0.5*QQT)U for all five conserved variables.
! See call in compute_rhs_and_dt for important documentation
!                                     -
! Spoiler: for SEM this comes out to U -{{U}}, which when
!          spoken is "UMinus Minus the Central flux of U" which I
!          then abbreviate as ummcu
      include 'SIZE'

      real ummcu (lx1*lz1*2*ldim*nelt,toteq) ! intent(out)
      real uminus(lx1*lz1*2*ldim*nelt,toteq) ! intent(in)
      real uplus (lx1*lz1*2*ldim*nelt,toteq) ! intent(in)
      integer ivar

      nf = lx1*lz1*2*ldim*nelt
      const=-0.5

! U-{{U}} on interior faces. first just do it on all faces.
      do ivar=1,toteq
         call add3(ummcu(1,ivar),uminus(1,ivar),uplus(1,ivar),nf)
         call cmult(ummcu(1,ivar),const,nf)        !         -
         call add2(ummcu(1,ivar),uminus(1,ivar),nf)!ummcu = U -{{U}}
      enddo

C> @}
      return
      end

!-----------------------------------------------------------------------

C> \ingroup bcond
C> @{
C> umubc = \f$\mathbf{U}^--\mathbf{U}^D\f$
      subroutine imqqtu_dirichlet(umubc,wminus,wplus)
! v+ undefined on boundary faces, so (I-0.5QQ^T) degenerates to 
! [[U]] with respect to the Dirichlet boundary state
      include 'SIZE'
      include 'INPUT' ! do we need this?
      include 'TSTEP' ! for ifield
      include 'CMTDATA'
      real umubc (lx1*lz1,2*ldim,nelt,toteq) ! intent(out)
      real wminus(lx1*lz1,2*ldim,nelt,nqq),
     >     wplus (lx1*lz1,2*ldim,nelt,nqq)
      real nTol
      integer e,f
      character*132 deathmessage
      common /nekcb/ cb
      character*3 cb

      nTol = 1.0E-14

      nxz = lx1*lz1
      nface=2*ldim
      ifield= 1 ! You need to figure out the best way of dealing with
                ! this variable

      do e=1,nelt
      do f=1,nface

         cb=cbc(f,e,ifield)
         if (cb.ne.'E  '.and.cb.ne.'P  ') then ! cbc bndy. this routine
                                               ! had better not touch any
                                               ! interior face
! JH031315 flux added to argument list. BC routines preserve wminus for
!          obvious reasons and fill wplus with good stuff for everybody:
!          imposed states for Dirichlet conditions, and important things
!          for viscous numerical fluxes.
! JH060215 added SYM bc. Just use it as a slip wall hopefully.
! JH111416 This may look like lazy duplication, but there is a good chance
!          that this may replace/consolidate BC calls in InviscidFlus.
!          basically, nothing in wplus is trustworthy, so we are going to
!          recompute and overwrite iu1 through iu5 in wplus with stuff we
!          do trust (UBC, to be exact).
!           if (cb.eq.'v  ' .or. cb .eq. 'V  ') then
!             call inflow2(nqq,f,e,wminus,wplus)
            if (cb .eq. 'W  ' .or. cb .eq.'I  ')then
              call wallbc2(nqq,f,e,wminus,wplus)
            endif

!  -
! U - UBC
            do ivar=1,toteq
               call sub3(umubc(1,f,e,ivar),wminus(1,f,e,iu1+ivar-1),
     >                                      wplus(1,f,e,iu1+ivar-1),nxz)
            enddo

         endif 
      enddo
      enddo

C> @}
      return
      end

!-----------------------------------------------------------------------

C> \ingroup vfjac
C> @{
C> flux = \f$\mathscr{A}\f$ dU = \f$\left(\mathscr{A}^{\mbox{NS}}+\mathscr{A}^{\mbox{EVM}}\right) \f$dU 
      subroutine agradu(flux,du,e,eq)
      include 'SIZE'
      include 'CMTDATA'
!JH122716 Apply viscous flux jacobian \mathscr{A} to a notional gradient
!         of the unknowns U (gradU in viscous_cmt and iku, U-{{U}} and
!         U-U_D in igtu_cmt, [U] in ihu (CHECK LI'S NOTES AGAIN))
! Yes, I know theoretically that EVM and NSE have different diffusive
! fluxes for different physical reasons. But I am combining them because
! 1. s_{ij} is shared by both and cuts down on code redundancy
! 2. we may wish to run EVM AND NSE together (I know this is frowned upon)

! but let's consider the normal use case of either NSE OR EVM
! Compressible Navier-Stokes      |
! mu=mu(T)                        |
! lambda=-2/3mu                   |
! nu_s=0                          |
! kappa=kappa(T)                  |  All of this 
!                                 |  should be done in
! Entropy visosity method (EVM)   |  uservp. very error-prone
! mu=mu_s(R_s,|u|+c,h)            |  requires deliberate user attention
! lambda=0 for EVM                |
! nu_s=nu_s(mu_s)                 |
! kappa=0                         |
! I need a flag, ifevm, for controlling calls (entropy_residual
! and longitudinal viscous fluxes, mostly due to grad rho) SOMEDAY
!-----------------------------------------------------------------------
! constructive feedback is always welcome
! flux is zero on entry
!-----------------------------------------------------------------------
      integer e, eq
      real flux(lx1*ly1*lz1,ldim),du(lx1*ly1*lz1,toteq,ldim)

C> \f$\tau_{ij}\f$ and \f$u_j \tau_{ij}\f$.  \f$\lambda=0\f$ and \f$\kappa=0\f$
C> for EVM
      call fluxj_ns (flux,du,e,eq)
C> \f$\nu_s \nabla \rho\f$, \f$\nu_s \left(\nabla \rho \right) \otimes \mathbf{u}\f$
C> and \f$\nu_s \nabla \left(\rho e\right)\f$.  \f$\nu_s=0\f$ for Navier-Stokes
      call fluxj_evm(flux,du,e,eq)

! no idea where phi goes. put it out front
c     call col2(flux,phig(1,1,1,e),lx1*ly1*lz1)

C> @}
      return
      end

!-----------------------------------------------------------------------

C> \f$ \tau_{ij}=2 \mu\sigma_{ij} + \lambda \Delta \delta_{ij}\f$
C> Navier-Stokes, so no mass diffusion. uservp provides properties.
C> Implemented via maxima-generated code
      subroutine fluxj_ns(flux,gradu,e,eq)
! viscous flux jacobian for compressible Navier-Stokes equations (NS)
! SOLN and CMTDATA are indexed, assuming vdiff has been filled by uservp
! somehow. In serious need of debugging and replacement.
      include 'SIZE'
      include 'INPUT'! TRIAGE?
      include 'SOLN' ! TRIAGE?

      parameter (ldd=lx1*ly1*lz1)
      common /ctmp1/ viscscr(lx1,ly1,lz1)
      real viscscr

      integer e,eq
      real flux(lx1*ly1*lz1,ldim),gradu(lx1*ly1*lz1,toteq,ldim)
      integer eijk3(3,3,3)
!     data eijk2 / 0, -1, 1, 0/
      data eijk3
     >/0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0/

      n=lx1*ly1*lz1

! This is a disaster that I might want to program less cleverly
      if (eq .lt. toteq) then ! TRIAGE. CAN'T GET AGRADU_NS to WORK
                              ! for ENERGY EQUATION. MAXIMA ROUTINES
                              ! BELOW
!!      do j=1,ldim
!!         do k=1,ldim
!!            ieijk=0
!!!           if (eq .lt. toteq .and. eq .gt. 1) ieijk=eijk3(eq-1,j,k) ! does this work in 2D?
!!            if (eq.gt.1)ieijk=eijk3(eq-1,j,k) ! does this work in 2D?
!!
!!            if (ieijk .eq. 0) then
!!              call agradu_ns(flux(1,j),gradu(1,1,k),viscscr,e,
!!     >                           eq,j,k)
!!            endif
!!         enddo
!!      enddo
! JH110716 Maxima routines added for every viscous flux.
!          agradu_ns has failed all verification checks for homentropic vortex
!          initialization.
!          start over
        if (eq.eq.2) then
           call A21kldUldxk(flux(1,1),gradu,e)
           call A22kldUldxk(flux(1,2),gradu,e)
           call A23kldUldxk(flux(1,3),gradu,e)
        elseif (eq.eq.3) then
           call A31kldUldxk(flux(1,1),gradu,e)
           call A32kldUldxk(flux(1,2),gradu,e)
           call A33kldUldxk(flux(1,3),gradu,e)
        elseif (eq.eq.4) then
           call A41kldUldxk(flux(1,1),gradu,e)
           call A42kldUldxk(flux(1,2),gradu,e)
           call A43kldUldxk(flux(1,3),gradu,e)
        endif

      else ! Energy equation courtesy of thoroughly-checked maxima
           ! until I can get agradu_ns working correctly
         if (if3d) then
            call a53kldUldxk(flux(1,3),gradu,e)
         else
            call rzero(gradu(1,1,3),lx1*ly1*lz1*toteq)
            call rzero(vz(1,1,1,e),lx1*ly1*lz1)
         endif
         call a51kldUldxk(flux(1,1),gradu,e)
         call a52kldUldxk(flux(1,2),gradu,e)
      endif

      return
      end

!-----------------------------------------------------------------------
!     subroutine agradu_ns(gijklu,dut,visco,e,eq,jflux,kdir) ! in junkyard...
!-----------------------------------------------------------------------

C> viscous flux jacobian for entropy viscosity Euler regularization of
C> Guermond and Popov (2014) SIAM JAM 74(2) that do NOT overlap with
C> the compressible Navier-Stokes equations (NS).
      subroutine fluxj_evm(flux,du,e,eq)
! SOLN and CMTDATA are indexed, assuming vdiff has been filled by uservp
! somehow. In serious need of debugging and replacement.
      include 'SIZE'
      include 'PARALLEL'
      include 'INPUT'! TRIAGE?
      include 'SOLN' ! TRIAGE?
      include 'CMTDATA'

      parameter (ldd=lx1*ly1*lz1)
      common /ctmp1/ viscscr(lx1,ly1,lz1) ! I'ma keep this
      real viscscr

      integer e,eq,eq2
      real flux(lx1*ly1*lz1,ldim),du(lx1*ly1*lz1,toteq,ldim)

      n=lx1*ly1*lz1

! diffusion due to grad rho
      if (eq .eq. 1) then
         do j=1,ldim ! flux+= viscscr*nu_s*grad (rho)
            call addcol3(flux(1,j),vdiff(1,1,1,e,inus),du(1,1,j),n)
         enddo
      else
         if (eq.lt.toteq) then
            call copy(viscscr,du(1,1,eq-1),n)
            call col2(viscscr,vdiff(1,1,1,e,inus),n)
            call addcol3(flux(1,1),viscscr,vx(1,1,1,e),n)
            call addcol3(flux(1,2),viscscr,vy(1,1,1,e),n)
            if (if3d) call addcol3(flux(1,3),viscscr,vz(1,1,1,e),n)

         else ! energy equation

            if(if3d) then ! mass diffusion term
               call vdot3(viscscr,vx(1,1,1,e),vy(1,1,1,e),vz(1,1,1,e),
     >                            vx(1,1,1,e),vy(1,1,1,e),vz(1,1,1,e),n)
            else
               call vdot2(viscscr,vx(1,1,1,e),vy(1,1,1,e),
     >                            vx(1,1,1,e),vy(1,1,1,e),n)
            endif
            call col2(viscscr,vdiff(1,1,1,e,inus),n)
            do j=1,ldim
               call addcol3(flux(1,j),du(1,1,j),viscscr,n)
            enddo

            do j=1,ldim
               do eq2=2,ldim+1
                  call col4(viscscr,du(1,eq2,j),u(1,1,1,eq2,e),
     >                           vdiff(1,1,1,e,inus),n)
                  call invcol2(viscscr,vtrans(1,1,1,e,irho),n) ! scr=nu_s*U/rho
                  call sub2(flux(1,j),viscscr,n)
               enddo
               call addcol3(flux(1,j),du(1,toteq,j),vdiff(1,1,1,e,inus),
     >                      n)
            enddo
         endif ! eq<toteq?

      endif ! eq==1?

      return
      end

!-----------------------------------------------------------------------

      subroutine half_iku_cmt(res,diffh,e)
      include 'SIZE'
      include 'MASS'
! diffh has D AgradU. half_iku_cmt applies D^T BM1 to it and increments
! the residual res with the result
      integer e ! lopsided. routine for one element must reference bm1
      real res(lx1,ly1,lz1),diffh(lx1*ly1*lz1,ldim)

      n=lx1*ly1*lz1

      do j=1,ldim
         call col2(diffh(1,j),bm1(1,1,1,e),n)
!        call col2(diffh(1,j),phig(1,1,1,e),n) ! still no idea where phi goes
      enddo

!     const=-1.0 ! I0
      const=1.0  ! *-1 in time march
      call gradm11_t(res,diffh,const,e)

      return
      end

!-----------------------------------------------------------------------

      subroutine compute_transport_props
! get vdiff props
! viscosity in imu
! second viscosity in ilam; second viscosity is usually -2/3mu
! but we refuse to assume Stokes' hypothesis for the user
! second viscosity=0 in the EVM for Euler gas dynamics
! thermal conductivity in iknd;
! mass diffusivity for EVM in inus
! via nekasgn
      include 'SIZE'
      include 'PARALLEL'
      include 'NEKUSE'
      include 'SOLN'
      include 'CMTDATA'

      integer   e

      do e=1,nelt
         ieg=lglel(e)
         do k=1,lz1
         do j=1,ly1
         do i=1,lx1
            call nekasgn(i,j,k,e)
            call cmtasgn(i,j,k,e)
            call uservp(i,j,k,ieg)
            vdiff(i,j,k,e,imu)  = mu   ! NEKUSE
            vdiff(i,j,k,e,ilam) = lambda!NEKUSE
            vdiff(i,j,k,e,iknd) = udiff! NEKUSE
            vdiff(i,j,k,e,inus) = nu_s ! CMTDATA
         enddo
         enddo
         enddo
      enddo

      return
      end

!-----------------------------------------------------------------------
! TRIAGE BELOW UNTIL I CAN FIX AGRADU_NS
!-----------------------------------------------------------------------
      subroutine a51kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU3x=dU(i,3,1)
         dU4x=dU(i,4,1)
         dU5x=dU(i,5,1)
         dU1y=dU(i,1,2)
         dU2y=dU(i,2,2)
         dU3y=dU(i,3,2)
         dU4y=dU(i,4,2)
         dU5y=dU(i,5,2)
         dU1z=dU(i,1,3)
         dU2z=dU(i,2,3)
         dU3z=dU(i,3,3)
         dU4z=dU(i,4,3)
         dU5z=dU(i,5,3)
         rho   =vtrans(i,1,1,ie,irho)
         cv    =vtrans(i,1,1,ie,icv)/rho
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         K     =vdiff(i,1,1,ie,iknd)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         E     =U(i,1,1,toteq,ie)/rho
         lambdamu=lambda+mu
         kmcvmu=K-cv*mu
         flux(i)=
     >(K*dU5x+cv*lambda*u1*dU4z-kmcvmu*u3*dU4x+cv*lambda*u1*dU3y
     1   -kmcvmu*u2*dU3x+cv*mu*u3*dU2z+cv*mu*u2*dU2y+(cv*lambda-
     2   K+2*cv*mu)*u1*dU2x-cv*lambdamu*u1*u3*dU1z-cv*lambdamu
     3   *u1*u2*dU1y+(K*u3**2-cv*mu*u3**2+K*u2**2-cv*mu*u2**2-cv*la
     4   mbda*u1**2+K*u1**2-2*cv*mu*u1**2-E*K)*dU1x)/(cv*rho)
      enddo
      return
      end

      subroutine a52kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU3x=dU(i,3,1)
         dU4x=dU(i,4,1)
         dU5x=dU(i,5,1)
         dU1y=dU(i,1,2)
         dU2y=dU(i,2,2)
         dU3y=dU(i,3,2)
         dU4y=dU(i,4,2)
         dU5y=dU(i,5,2)
         dU1z=dU(i,1,3)
         dU2z=dU(i,2,3)
         dU3z=dU(i,3,3)
         dU4z=dU(i,4,3)
         dU5z=dU(i,5,3)
         rho   =vtrans(i,1,1,ie,irho)
         cv    =vtrans(i,1,1,ie,icv)/rho
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         K     =vdiff(i,1,1,ie,iknd)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         E     =U(i,1,1,toteq,ie)/rho
         lambdamu=lambda+mu
         kmcvmu=K-cv*mu
         flux(i)=
     >(K*dU5y+cv*lambda*u2*dU4z-kmcvmu*u3*dU4y+cv*mu*u3*dU3z+(cv
     1   *lambda-K+2*cv*mu)*u2*dU3y+cv*mu*u1*dU3x-kmcvmu*u1*dU2y+
     2   cv*lambda*u2*dU2x-cv*lambdamu*u2*u3*dU1z+(K*u3**2-cv*mu
     3   *u3**2-cv*lambda*u2**2+K*u2**2-2*cv*mu*u2**2+K*u1**2-cv*mu*
     4   u1**2-E*K)*dU1y-cv*lambdamu*u1*u2*dU1x)/(cv*rho)
      enddo
      return
      end
      subroutine a53kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU3x=dU(i,3,1)
         dU4x=dU(i,4,1)
         dU5x=dU(i,5,1)
         dU1y=dU(i,1,2)
         dU2y=dU(i,2,2)
         dU3y=dU(i,3,2)
         dU4y=dU(i,4,2)
         dU5y=dU(i,5,2)
         dU1z=dU(i,1,3)
         dU2z=dU(i,2,3)
         dU3z=dU(i,3,3)
         dU4z=dU(i,4,3)
         dU5z=dU(i,5,3)
         rho   =vtrans(i,1,1,ie,irho)
         cv    =vtrans(i,1,1,ie,icv)/rho
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         K     =vdiff(i,1,1,ie,iknd)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         E     =U(i,1,1,toteq,ie)/rho
         lambdamu=lambda+mu
         kmcvmu=K-cv*mu
         flux(i)=
     >(K*(dU5z-E*dU1z)+cv*u3*(lambda*dU4z+2*mu*dU4z+lambda*dU3y+lambda
     1   *dU2x)-K*u3*dU4z+cv*mu*u2*(dU4y+dU3z)+cv*mu*u1*(dU4x+dU2z)-
     2   K*u2*dU3z-K*u1*dU2z-cv*(lambda+2*mu)*u3**2*dU1z+K*u3**2*dU1z+
     3   K*u2**2*dU1z-cv*mu*u2**2*dU1z+K*u1**2*dU1z-cv*mu*u1**2*dU1z-c
     4   v*(lambda+mu)*u2*u3*dU1y-cv*(lambda+mu)*u1*u3*dU1x)/(cv*rho)
      enddo
      return
      end

      subroutine A21kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU1y=dU(i,1,2)
         dU3y=dU(i,3,2)
         dU1z=dU(i,1,3)
         dU4z=dU(i,4,3)
         rho   =vtrans(i,1,1,ie,irho)
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         lambdamu=lambda+2.0*mu
         flux(i)=
     >(lambda*(dU4z+dU3y-u3*dU1z-u2*dU1y)+lambdamu*(dU2x-u1*dU1x))/rho
      enddo
      return
      end
      subroutine A22kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU3x=dU(i,3,1)
         dU1y=dU(i,1,2)
         dU2y=dU(i,2,2)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         flux(i)=mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho
      enddo
      return
      end
      subroutine A23kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU4x=dU(i,4,1)
         dU1z=dU(i,1,3)
         dU2z=dU(i,2,3)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         flux(i)=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho
      enddo
      return
      end

      subroutine A31kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU3x=dU(i,3,1)
         dU1y=dU(i,1,2)
         dU2y=dU(i,2,2)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         flux(i)=mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho
      enddo
      return
      end
      subroutine A32kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU1y=dU(i,1,2)
         dU3y=dU(i,3,2)
         dU1z=dU(i,1,3)
         dU4z=dU(i,4,3)
         rho   =vtrans(i,1,1,ie,irho)
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         lambdamu=lambda+2.0*mu
         flux(i)=(lambda*(dU4z+dU2x-u3*dU1z-u1*dU1x)+
     >   lambdamu*(dU3y-u2*dU1y))/rho
      enddo
      return
      end
      subroutine A33kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1y=dU(i,1,2)
         dU4y=dU(i,4,2)
         dU1z=dU(i,1,3)
         dU3z=dU(i,3,3)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         flux(i)=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho
      enddo
      return
      end

      subroutine A41kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU4x=dU(i,4,1)
         dU1z=dU(i,1,3)
         dU2z=dU(i,2,3)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         flux(i)=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho
      enddo
      return
      end
      subroutine A42kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1y=dU(i,1,2)
         dU4y=dU(i,4,2)
         dU1z=dU(i,1,3)
         dU3z=dU(i,3,3)
         rho   =vtrans(i,1,1,ie,irho)
         mu    =vdiff(i,1,1,ie,imu)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         flux(i)=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho
      enddo
      return
      end
      subroutine A43kldUldxk(flux,dU,ie)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA' ! gradu lurks
      real K,E,kmcvmu,lambdamu
      real dU(lx1*ly1*lz1,toteq,3)
      real flux(lx1*ly1*lz1)
      npt=lx1*ly1*lz1
      do i=1,npt
         dU1x=dU(i,1,1)
         dU2x=dU(i,2,1)
         dU1y=dU(i,1,2)
         dU3y=dU(i,3,2)
         dU1z=dU(i,1,3)
         dU4z=dU(i,4,3)
         rho   =vtrans(i,1,1,ie,irho)
         lambda=vdiff(i,1,1,ie,ilam)
         mu    =vdiff(i,1,1,ie,imu)
         u1    =vx(i,1,1,ie)
         u2    =vy(i,1,1,ie)
         u3    =vz(i,1,1,ie)
         lambdamu=lambda+2.0*mu
         flux(i)=(lambda*(dU3y+dU2x-u2*dU1y-u1*dU1x)+
     >lambdamu*(dU4z-u3*dU1z))/rho
      enddo
      return
      end
