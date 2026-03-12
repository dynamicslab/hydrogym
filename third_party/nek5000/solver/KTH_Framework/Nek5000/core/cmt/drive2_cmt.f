C> @file drive2_cmt.f mid-level initialization drivers. Not long for this world.
c-----------------------------------------------------------------------
      subroutine nek_cmt_init
      include 'SIZE'
      include 'TOTAL'
      include 'DG'
      include 'CMTDATA'
      if (nio.eq.0) write(6,*)'Set up CMT-Nek'    
      if (toteq.ne.5) then
         if (nio.eq.0) write(6,*)'toteq is low ! toteq = ',toteq
         if (nio.eq.0) write(6,*) 'Reset toteq in SIZE to 5'
         call exitt
      endif
      if (ifrestart) then
         ifheat = .true. ! almost certainly incorrect
      endif
      call setup_cmt_commo

      iostep2=iostep
      iostep=9999999
      
c     call setup_cmt_param
      return
      end

!-----------------------------------------------------------------------

      subroutine izero8(a,n)
      integer*8 a(1)
      do i=1,n
         a(i)=0
      enddo
      return
      end
!-----------------------------------------------------------------------
      subroutine setup_cmt_param
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'CMTDATA'
      INCLUDE 'CMTBCDATA'

      real  MixtPerf_R_CpG, MixtPerf_T_DPR, MixtPerf_C_GRT
     >                 ,MixtPerf_Ho_CpTUVW,MixtPerf_Cp_CvR,MixtPerf_R_M
     >                 ,MixtPerf_G_CpR      
      external MixtPerf_R_CpG, MixtPerf_T_DPR, MixtPerf_C_GRT
     >                 ,MixtPerf_Ho_CpTUVW,MixtPerf_Cp_CvR,MixtPerf_R_M
     >                 ,MixtPerf_G_CpR      

      cip_adhoc=10.0
      cvgref     = param(104)
c     gmaref     = param(105)
      molmass    = param(106)
      muref      = param(107)
      coeflambda = param(108)
      suthcoef   = param(109)
      reftemp    = param(110)
      prlam      = param(111)
      pinfty     = param(112)
      rgasref    = MixtPerf_R_M(molmass,dum)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 
! put these in rea file someday
      return
      end
c------------------------------------------------------------------------

      subroutine limiter
! EBDG Stuff. WHERE'S PHI????
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'NEKUSE'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ scr(lxyz),avstate(toteq)
      real scr,avstate
      integer e,eg
      real kemax

      nxyz=lx1*ly1*lz1
      ntot=nxyz*nelt

      epslon=1.0e-9


!     rgam=rgasref/(gmaref-1.0)
!      do i=1,ntot
!         rho=max(vtrans(i,1,1,1,irho),1.0e-10)
!!        scr(i,1)=rgam*log(pr(i,1,1,1)/(rho**gmaref))
!         scr(i,1)=log(pr(i,1,1,1)/(rho**gmaref))
!      enddo
!!     call dsop(scr,'MIN',lx1,ly1,lz1)
!      call copy(t(1,1,1,1,4),scr,ntot)
!! elemental entropy minimum
!!     do e=1,nelt
!!        se0(e)=vlmin(scr(1,e),nxyz)
!!     enddo

      do e=1,nelt

!        rhomin=vlmin(vtrans(1,1,1,e,irho),nxyz)
         rhomin=vlmin(u(1,1,1,1,e),nxyz)

! positivity-preserving limiter of Zhang and Shu: density
         rho=vlsc2(bm1(1,1,1,e),u(1,1,1,1,e),nxyz)/volel(e)
!        if (abs(rho-rhomin) .gt. epslon) then
         theta=min((rho-epslon)/(rho-rhomin+epslon),1.0)
         if (rho .lt. epslon) then
            write(6,'(a33,2i5,1p3e15.7)')
     >        'rho<epslon,nid,e,rho,rhomin,theta',nid,e,rho,rhomin,theta
            do i=1,nxyz
               uold=u(i,1,1,1,e)
               u(i,1,1,1,e)=rho+theta*(uold-rho)
            enddo
         else
            do i=1,nxyz
               uold=u(i,1,1,1,e)
               u(i,1,1,1,e)=rho+abs(theta)*(uold-rho)
            enddo
         endif
         call cfill(t(1,1,1,e,4),theta,nxyz)
!        rho=vlsc2(bm1(1,1,1,e),u(1,1,1,1,e),nxyz)/volel(e)

! positivity-preserving limiter of Lv & Ihme: internal energy
! first compute kinetic energy density
         if (if3d) then
            call vdot3(scr,
     >             u(1,1,1,irpu,e),u(1,1,1,irpv,e),u(1,1,1,irpw,e),
     >             u(1,1,1,irpu,e),u(1,1,1,irpv,e),u(1,1,1,irpw,e),nxyz)
         else
            call vdot2(scr,u(1,1,1,irpu,e),u(1,1,1,irpv,e),
     >                     u(1,1,1,irpu,e),u(1,1,1,irpv,e),nxyz)
         endif
         call invcol2(scr,u(1,1,1,irg,e),nxyz)
         call cmult(scr,0.5,nxyz)
!-----------------------------------------------------------------------
! JH091818 thinking about limiter pegged to max KE. explore this alternate
!          diagnostically
!         kemax=vlmax(scr,nxyz)
!-----------------------------------------------------------------------
! then subtract it off to get internal energy density in scr
         call sub2(scr,u(1,1,1,iret,e),nxyz)
         call chsign(scr,nxyz)
! violation if negative energy density
         tau=vlmin(scr,nxyz)
         tau=min(tau,0.0)
!-----------------------------------------------------------------------
! JH091818
! overwrite scr to explore alternate limiter below
!-----------------------------------------------------------------------
!         kemax=-kemax
!         call cadd2(scr,u(1,1,1,5,e),kemax,nxyz)
!-----------------------------------------------------------------------

! now for rhoe(avstate)
         do m=1,toteq
            avstate(m)=vlsc2(bm1(1,1,1,e),u(1,1,1,m,e),nxyz)/volel(e)
         enddo
         rhoeavg=avstate(5)-
     >              0.5*(avstate(2)**2+avstate(3)**2+avstate(4)**2)/
     >                   avstate(1)

         if (rhoeavg .lt. 0.0) then
            write(6,*) 'duh sir , can''t get positive e(avg)',e,rhoeavg,
     >                 tau,nid
            epsebdg(e)=1.0
         else
            epsebdg(e)=tau/(tau-rhoeavg)
         endif
!        epsebdg(e)=1.0 ! Godunov test
         do m=1,toteq
            do i=1,nxyz
               uold=u(i,1,1,m,e)
               u(i,1,1,m,e)=uold+epsebdg(e)*(avstate(m)-uold)
            enddo
         enddo

!-----------------------------------------------------------------------
! diagnostics
!-----------------------------------------------------------------------
         call cfill(t(1,1,1,e,5),epsebdg(e),nxyz)
! JH091818
! alternate limiter trying to keep U5 > kemax (at best ultraconservative)
!         tau=vlmin(scr,nxyz)
!         tau=min(tau,0.0)
!         epsalot=tau/(tau-(avstate(5)+kemax))
!         call cfill(t(1,1,1,e,5),epsalot,nxyz)
!-----------------------------------------------------------------------

!         rho=avstate(1)
!! Entropy-bounded limiter of Lv and Ihme
!-----------------------------------------------------------------------
! JH091118 This isn't ready for non-ideal state equations or volume fraction
!-----------------------------------------------------------------------
!         e_internal=avstate(5)-
!     >              0.5*(avstate(2)**2+avstate(3)**2+avstate(4)**2)/
!     >                   rho
!         e_internal=e_internal/rho
!         eg=gllel(e)
!         call cmt_userEOS(1,1,1,eg) ! assigns elm avg to  pres and temp
!         do i=1,nxyz
!            scr(i)=pr(i,1,1,e)-exp(se0const)*
!     >                         (u(i,1,1,1,e)**gmaref)
!         enddo
!         tau=vlmin(scr,nxyz)
!         tau=min(tau,0.0)
!         epsebdg(e)=tau/
!     >          (tau-(pres-exp(se0const)*rho**gmaref))
!         epsebdg(e)=min(epsebdg(e),1.0)
!         epsebdg(e)=max(epsebdg(e),0.0)
!! diagnostic
!         call cfill(t(1,1,1,e,5),epsebdg(e),nxyz)
!
!         do m=1,toteq
!            do i=1,nxyz
!               uold=u(i,1,1,m,e)
!               u(i,1,1,m,e)=uold+epsebdg(e)*(avstate(m)-uold)
!            enddo
!         enddo
!-----------------------------------------------------------------------

      enddo
      
      return
      end

!-----------------------------------------------------------------------
! JH082118 shock detectors. Test shock detectors in usr file extensively.
!          port them here after they have earned our trust.
!-----------------------------------------------------------------------

      subroutine AVeverywhere(shkdet)
      include 'SIZE'
      include 'TOTAL'
      real shkdet(nelt)
      call rone(shkdet,nelt)
      return
      end

      subroutine limiter_only(shkdet)
! worst ad hoc shock detector ever. read a limiter function from CMTDATA
! epsebdg is the only one we have so far. If it's bigger than some threshold,
! apply AV there too.
      include 'SIZE'
      include 'CMTDATA'
      real shkdet(nelt)
      integer e

      call rzero(shkdet,nelt)
      tol=1.0e-5

      if (.not.time4av) then
         do e=1,nelt
            if (abs(epsebdg(e)) .gt. tol) shkdet(e)=1.0
         enddo
      endif

      emax=glamax(shkdet,nelt)
      if (nio.eq.0) write(6,*) 'max shock detector =',emax

      return
      end
