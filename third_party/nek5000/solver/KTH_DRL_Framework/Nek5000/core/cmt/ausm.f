C> @file ausm.f Riemann solvers and other rocflu miscellany
! ******************************************************************************
!
! Purpose: Compute convective fluxes using AUSM+ scheme.
!
! Description: None.
!
! Input: 
!   nx          x-component of face normal
!   ny          y-component of face normal
!   nz          z-component of face normal
!   nm          Magnitude of face normal
!   fs          Face speed
!   rl          Density of left state
!   ul          x-component of velocity of left state
!   vl          y-component of velocity of left state
!   wl          z-component of velocity of left state   
!   Hl		Total enthalpy of left state
!   al		Speed of sound of left state
!   pl          Pressure of left state
!   rr          Density of right state
!   ur          x-component of velocity of right state
!   vr          y-component of velocity of right state
!   wr          z-component of velocity of right state  
!   pr          Pressure of right state
!   Hr		Total enthalpy of right state
!   ar		Speed of sound of right state
!
! Output: 
!   flx         Fluxes
!   vf          Face velocities ! NOT USED IN CMT-NEK YET
!
! Notes: 
!   1. Liou M.-S., Progress towards an improved CFD method: AUSM+, AIAA Paper
!      95-1701, 1995
!   2. Do not use computation of face speed of sound which leads to exact 
!      capturing of isolated normal shock waves because of robustness problems
!      for unsteady flows and because that formulation is not applicable to 
!      anything but calorically and thermally perfect gases.
!
! ******************************************************************************

C> \ingroup isurf
C> @{
C> Computes inviscid numerical surface flux from AUSM+ Riemann solver
      SUBROUTINE AUSM_FluxFunction(ntot,nx,ny,nz,nm,fs,rl,ul,vl,wl,pl,
     >                         al,tl,rr,ur,vr,wr,pr,ar,tr,flx,el,er)

!     IMPLICIT NONE ! HAHAHHAHHAHA
! ******************************************************************************
! Definitions and declarations
! ******************************************************************************
      real     MixtJWL_Enthalpy
      external MixtJWL_Enthalpy

! ==============================================================================
! Arguments
! ==============================================================================
      integer ntot
      REAL al(ntot),ar(ntot),fs(ntot),nm(ntot),nx(ntot),ny(ntot),
     >     nz(ntot),pl(ntot),pr(ntot),rl(ntot),rr(ntot),ul(ntot),
     >     ur(ntot),vl(ntot),vr(ntot),wl(ntot),wr(ntot),el(ntot),
     >     er(ntot),tl(ntot),tr(ntot)! INTENT(IN) ::
      REAL flx(ntot,5)!,vf(3) ! INTENT(OUT) ::

! ==============================================================================
! Locals
! ==============================================================================

      REAL af,mf,mfa,mfm,mfp,ml,mla,mlp,mr,mra,mrm,pf,ql,qr,vml,vmr,
     >        wtl,wtr,Hl,Hr

! ******************************************************************************
! Start, compute face state
! ******************************************************************************

      do i=1,ntot
!        Change the Enthalpy 
         Hl = MixtJWL_Enthalpy(rl(i),pl(i),ul(i),vl(i),wl(i),el(i))
         Hr = MixtJWL_Enthalpy(rr(i),pr(i),ur(i),vr(i),wr(i),er(i))

         ql = ul(i)*nx(i) + vl(i)*ny(i) + wl(i)*nz(i) - fs(i)
         qr = ur(i)*nx(i) + vr(i)*ny(i) + wr(i)*nz(i) - fs(i)

         af = 0.5*(al(i)+ar(i)) ! NOTE not using original formulation, see note
         ml  = ql/af
         mla = ABS(ml)

         mr  = qr/af
         mra = ABS(mr)    

         IF ( mla .le. 1.0 ) THEN 
            mlp = 0.25*(ml+1.0)*(ml+1.0) + 0.125*(ml*ml-1.0)*(ml*ml-1.0)
            wtl = 0.25*(ml+1.0)*(ml+1.0)*(2.0-ml) +
     >            0.1875*ml*(ml*ml-1.0)*(ml*ml-1.0)
         ELSE
            mlp = 0.5*(ml+mla)
            wtl = 0.5*(1.0+ml/mla)
         END IF ! mla

         IF ( mra .le. 1.0 ) THEN 
            mrm = -0.25*(mr-1.0)*(mr-1.0)-0.125*(mr*mr-1.0)*(mr*mr-1.0)
            wtr = 0.25*(mr-1.0)*(mr-1.0)*(2.0+mr) -
     >            0.1875*mr*(mr*mr-1.0)*(mr*mr-1.0)
         ELSE
            mrm = 0.5*(mr-mra)
            wtr = 0.5*(1.0-mr/mra)
         END IF ! mla

         mf  = mlp + mrm
         mfa = ABS(mf)
         mfp = 0.5*(mf+mfa)
         mfm = 0.5*(mf-mfa)

         pf = wtl*pl(i) + wtr*pr(i)

! ******************************************************************************
! Compute fluxes
! ******************************************************************************

!        vf(1) = mfp*ul + mfm*ur ! I'm sure we'll need this someday
!        vf(2) = mfp*vl + mfm*vr
!        vf(3) = mfp*wl + mfm*wr

         flx(i,1)=(af*(mfp*rl(i)      +mfm*rr(i)   )        )*nm(i)
         flx(i,2)=(af*(mfp*rl(i)*ul(i)+mfm*rr(i)*ur(i))+pf*nx(i))*
     >            nm(i)
         flx(i,3)=(af*(mfp*rl(i)*vl(i)+mfm*rr(i)*vr(i))+pf*ny(i))*
     >            nm(i)
         flx(i,4)=(af*(mfp*rl(i)*wl(i)+mfm*rr(i)*wr(i))+pf*nz(i))*
     >            nm(i)
         flx(i,5)=(af*(mfp*rl(i)*Hl   +mfm*rr(i)*Hr) + pf*fs(i))*
     >            nm(i)
      enddo
C> @}
      return
      END

!-----------------------------------------------------------------------
! NOT LONG FOR THIS WORLD

      SUBROUTINE CentralInviscid_FluxFunction(ntot,nx,ny,nz,fs,ul,pl,
     >                                     ur,pr,flx)
! JH081915 More general, more obvious
! JH111815 HEY GENIUS THIS MAY BE SECOND ORDER AND THUS KILLING YOUR
!          CONVERGENCE. REPLACE WITH AUSM AND SHITCAN IT
! JH112015 This isn't why walls aren't converging. There's something
!          inherently second-order about your wall pressure. Think!
      real nx(ntot),ny(ntot),nz(ntot),fs(ntot),ul(ntot,5),pl(ntot),
     >     ur(ntot,5),pr(ntot) ! intent(in)
      real flx(ntot,5)! intent(out),dimension(5) ::

      do i=1,ntot
         rl =ul(i,1)
         rul=ul(i,2)
         rvl=ul(i,3)
         rwl=ul(i,4)
         rel=ul(i,5)

         rr =ur(i,1)
         rur=ur(i,2)
         rvr=ur(i,3)
         rwr=ur(i,4)
         rer=ur(i,5)

         ql = (rul*nx(i) + rvl*ny(i) + rwl*nz(i))/rl - fs(i)
         qr = (rur*nx(i) + rvr*ny(i) + rwr*nz(i))/rr - fs(i)

         flx(i,1) = 0.5*(ql* rl+ qr*rr               )
         flx(i,2) = 0.5*(ql* rul+pl(i)*nx(i) + qr* rur     +pr(i)*nx(i))
         flx(i,3) = 0.5*(ql* rvl+pl(i)*ny(i) + qr* rvr     +pr(i)*ny(i))
         flx(i,4) = 0.5*(ql* rwl+pl(i)*nz(i) + qr* rwr     +pr(i)*nz(i))
         flx(i,5) = 0.5*(ql*(rel+pl(i))+pl(i)*fs(i)+qr*(rer+pr(i))+
     >               pr(i)*fs(i))
      enddo

      return
      end
