c--------------------------------------------------------------------
      subroutine inflow(nvar,f,e,faceq,bcq,flux)
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'CMTBCDATA'
      integer nvar,f,e
      real faceq(nx1,nz1,2*ldim,nelt,nvar)
      real bcq(nx1,nz1,2*ldim,nelt,nvar)
      real flux(nx1*nz1,2*ldim,nelt,*)  

      ! Assume that the user knows inflow is sub or super sonic

      call inflow_rflu(nvar,f,e,faceq,bcq,flux)
      return
      end
!--------------------------------------------------------------------

      subroutine inflow_rflu(nvar,f,e,faceq,bcq,flux1)
      include 'SIZE'
      include 'NEKUSE'
      include 'CMTDATA'
      include 'GEOM'
      include 'PARALLEL'
      include 'DG'
      include 'PERFECTGAS'

      integer f,e,fdim ! intent(in)
      integer i,bcOptType
      real faceq(nx1*nz1,2*ldim,nelt,nvar) ! intent(in)
      real bcq  (nx1*nz1,2*ldim,nelt,nvar)   ! intent(out)
      real flux1(nx1*nz1,2*ldim,nelt,*)    ! intent(out)
      real snx,sny,snz,rhou,rhov,rhow,pl,rhob,rhoub,rhovb,rhowb
     >     ,rhoeb, mach
      parameter (lfd1=lxd*lzd,lfc1=lx1*lz1)
      common /SCRNS/ nxf(lfd1),nyf(lfd1),nzf(lfd1),fs(lfd1),
     >               ufacel(lfd1,5),plc(lfc1),ufacer(lfd1,5),prc(lfd1),
     >               flx(lfd1,5),plf(lfd1),prf(lfd1),jaco_c(lfc1),
     >               jaco_f(lfd1)
      real nxf,nyf,nzf,fs,ufacel,ufacer,plc,prc,plf,prf,flx
     >              ,jaco_c,jaco_f

      real phirc(lfc1),phirf(lfd1),molmrf(lfd1),molmrc(lfc1),cvgrc(lfc1)
      real cvgrf(lfd1),cpgrc(lfc1),cpgrf(lfd1),p0inrc(lfc1),p0inrf(lfd1)
     >     ,t0inrc(lfc1),t0inrf(lfd1),uxrf(lfd1),uyrf(lfd1),uzrf(lfd1)
     >     ,csndrf(lfd1),philf(lfd1)

      nxz=nx1*nz1
      nxzd=nxd*nzd
      fdim=ndim-1
      ieg=lglel(e)

      call facind(i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)    
      l=0
      do iz=k0,k1
      do iy=j0,j1
      do ix=i0,i1
         call nekasgn(ix,iy,iz,e)
         call cmtasgn(ix,iy,iz,e)
         call userbc (ix,iy,iz,f,ieg) ! get molarmass asnd phi t0in p0in cp cv
c                                     !     ux,uy,uz
         l=l+1
         phirc(l) = phi
         molmrc(l) = molarmass
         cpgrc(l)=cp
         cvgrc(l)=cv
         p0inrc(l) = p0in
         t0inrc(l) = t0in
         bcq(l,f,e,iux)  = ux
         bcq(l,f,e,iuy)  = uy
         bcq(l,f,e,iuz)  = uz
         bcq(l,f,e,isnd) = asnd

      enddo
      enddo
      enddo

      if (nxd.gt.nx1) then
         call map_faced(cpgrf,cpgrc,nx1,nxd,fdim,0)
         call map_faced(cvgrf,cvgrc,nx1,nxd,fdim,0)
         call map_faced(molmrf,molmrc,nx1,nxd,fdim,0)
         call map_faced(p0inrf,p0inrc,nx1,nxd,fdim,0)
         call map_faced(t0inrf,t0inrc,nx1,nxd,fdim,0)
         call map_faced(phirf,phirc,nx1,nxd,fdim,0)

         call map_faced(nxf,unx(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nyf,uny(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nzf,unz(1,1,f,e),nx1,nxd,fdim,0)

         call map_faced(ufacel(1,1),faceq(1,f,e,iu1),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,2),faceq(1,f,e,iu2),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,3),faceq(1,f,e,iu3),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,4),faceq(1,f,e,iu4),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,5),faceq(1,f,e,iu5),nx1,nxd,fdim,0)

         call map_faced(philf,faceq(1,f,e,iph),nx1,nxd,fdim,0)

         call map_faced(uxrf,bcq(1,f,e,iux), nx1,nxd,fdim,0)
         call map_faced(uyrf,bcq(1,f,e,iuy), nx1,nxd,fdim,0)
         call map_faced(uzrf,bcq(1,f,e,iuz), nx1,nxd,fdim,0)
         call map_faced(csndrf,bcq(1,f,e,isnd), nx1,nxd,fdim,0)

         call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz)
         call map_faced(jaco_f,jaco_c,nx1,nxd,fdim,0) 
         call col2(jaco_f,wghtf,nxzd)
      else
         call copy(cpgrf,cpgrc,nxz)
         call copy(cvgrf,cvgrc,nxz)
         call copy(molmrf,molmrc,nxz)
         call copy(p0inrf,p0inrc,nxz)
         call copy(t0inrf,t0inrc,nxz)
         call copy(phirf,phirc,nxz)

         call copy(nxf,unx(1,1,f,e),nxz)
         call copy(nyf,uny(1,1,f,e),nxz)
         call copy(nzf,unz(1,1,f,e),nxz)

         call copy(ufacel(1,1),faceq(1,f,e,iu1),nxz)
         call copy(ufacel(1,2),faceq(1,f,e,iu2),nxz)
         call copy(ufacel(1,3),faceq(1,f,e,iu3),nxz)
         call copy(ufacel(1,4),faceq(1,f,e,iu4),nxz)
         call copy(ufacel(1,5),faceq(1,f,e,iu5),nxz)

         call copy(philf,faceq(1,f,e,iph),nxz)

         call copy(uxrf,bcq(1,f,e,iux), nxz)
         call copy(uyrf,bcq(1,f,e,iuy), nxz)
         call copy(uzrf,bcq(1,f,e,iuz), nxz)
         call copy(csndrf,bcq(1,f,e,isnd), nxz)

         call copy(jaco_f,area(1,1,f,e),nxz) 
      endif

      do l=1,lfd1
         bcOptType=0
         snx  = nxf(l)
         sny  = nyf(l)
         snz  = nzf(l)

         rho  = ufacel(l,1)/philf(l) 
         rhou = ufacel(l,2)/philf(l)
         rhov = ufacel(l,3)/philf(l)
         rhow = ufacel(l,4)/philf(l)
         rhoe = ufacel(l,5)/philf(l)

         ux   = uxrf(l) 
         uy   = uyrf(l)
         uz   = uzrf(l)
         asnd = csndrf(l)
         mach = sqrt(ux**2+uy**2+uz**2)/asnd
         if (mach.lt.1.0) bcOptType=1
         betah = atan2(uy,ux)
         if (ldim.eq.3) betav = atan2(uz,ux)
         if (ldim.eq.2) betav = atan2(0.0,ux)

         call BcondInflowPerf(bcOptType,0,p0inrf(l),t0inrf(l)
     >                       ,betah,betav,mach,snx,sny,snz,cpgrf(l)
     >                       ,molmrf(l),rho,rhou,rhov,rhow,rhob,rhoub
     >                       ,rhovb,rhowb,rhoeb,pres)
         
         ufacer(l,1) = rhob*phirf(l)
         ufacer(l,2) = rhoub*phirf(l)
         ufacer(l,3) = rhovb*phirf(l)
         ufacer(l,4) = rhowb*phirf(l)
         ufacer(l,5) = rhoeb*phirf(l)
         prf(l)      = pres*phirf(l)
      enddo

      call rzero(fs,nxzd)

!-----------------------------------------------------------------------
! Inviscid flux at inflow can probably just be hardcoded instead of
! derived from a trivial call of CentralInviscid_FluxFunction
c     call CentralInviscid_FluxFunction(nxzd,nxf,nyf,nzf,fs,ufacel,plf,
c    >                                  ufacer,prf,flx)
c MS010716 This central flux call is trivial. Flux computation is based 
c solely on the right state or the ufacer array. 
c This is the most stable way of incorporating
c inflow boundary codntion (so far!). This was tested for 
c  ---  uniform flow and subsonic flow over a cylinder. 
c  (Need to test supersonic uniform flow !)
c Recall that ST had mentioned that he uses something like this in his 
c code. Also note that this change was important for inflow BC
c Outflow BC is not sensitive to method used to compute the flux. 
      call CentralInviscid_FluxFunction(nxzd,nxf,nyf,nzf,fs,ufacer,prf,
     >                                  ufacer,prf,flx)

      do ieq=1,toteq
         call col2(flx(1,ieq),jaco_f,nxzd)
      enddo

      if (nxd.gt.nx1) then
         do j=1,toteq
            call map_faced(flux1(1,f,e,j),flx(1,j),nx1,nxd,fdim,1)
         enddo
      else
         do j=1,toteq
            call copy(flux1(1,f,e,j),flx(1,j),nxz)
         enddo
      endif

      return
      end

!*********************************************************************
!* Illinois Open Source License                                      *
!*                                                                   *
!* University of Illinois/NCSA                                       * 
!* Open Source License                                               *
!*                                                                   *
!* Copyright@2008, University of Illinois.  All rights reserved.     *
!*                                                                   *
!*  Developed by:                                                    *
!*                                                                   *
!*     Center for Simulation of Advanced Rockets                     *
!*                                                                   *
!*     University of Illinois                                        *
!*                                                                   *
!*     www.csar.uiuc.edu                                             *
!*                                                                   *
!* Permission is hereby granted, free of charge, to any person       *
!* obtaining a copy of this software and associated documentation    *
!* files (the "Software"), to deal with the Software without         *
!* restriction, including without limitation the rights to use,      *
!* copy, modify, merge, publish, distribute, sublicense, and/or      *
!* sell copies of the Software, and to permit persons to whom the    *
!* Software is furnished to do so, subject to the following          *
!* conditions:                                                       *
!*                                                                   *
!*                                                                   *
!* @ Redistributions of source code must retain the above copyright  * 
!*   notice, this list of conditions and the following disclaimers.  *
!*                                                                   * 
!* @ Redistributions in binary form must reproduce the above         *
!*   copyright notice, this list of conditions and the following     *
!*   disclaimers in the documentation and/or other materials         *
!*   provided with the distribution.                                 *
!*                                                                   *
!* @ Neither the names of the Center for Simulation of Advanced      *
!*   Rockets, the University of Illinois, nor the names of its       *
!*   contributors may be used to endorse or promote products derived * 
!*   from this Software without specific prior written permission.   *
!*                                                                   *
!* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,   *
!* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   *
!* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          *
!* NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR           *
!* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       * 
!* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,   *
!* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE    *
!* USE OR OTHER DEALINGS WITH THE SOFTWARE.                          *
!*********************************************************************
!* Please acknowledge The University of Illinois Center for          *
!* Simulation of Advanced Rockets in works and publications          *
!* resulting from this software or its derivatives.                  *
!*********************************************************************
!******************************************************************************
!
! Purpose: set inflow boundary condition for one cell.
!
! Description: The subsonic boundary condition is based on the extrapolation of 
!   the Riemann invariant from the interior field (Holmes, D.G.: Inviscid 2D 
!   Solutions on Unstructured, Adaptive Grids. VKI Lecture Series 1989-06, 1989). 
!   The supersonic inflow boundary condition computes the conservative variables
!   from given velocity components, density and pressure.
!
! Input: bcOptType  = boundary treatment: subsonic, supersonic, or mixed
!        bcOptFixed = whether _computed_ inflow angle should be fixed or not
!        ptot       = given total pressure
!        ttot       = given total temperature
!        betah      = given inlet angle wrp. to y-axis
!        betav      = given inlet angle wrp. to z-axis
!        sx/y/zn    = components of normalized face vector (outward facing)
!        cpgas      = specific heat at constant pressure (boundary cell)
!        mm         = molecular mass at boundary cell
!        rl         = given density
!        ru/v/wl    = given velocity components
!
! Output: rr      = density at boundary
!         ru/v/wr = density * velocity components at boundary
!         rer     = density * total internal energy at boundary
!         pr      = pressure at boundary
!
! Notes: 
!   1. This condition is valid only for thermally and calorically perfect  
!      gas.
!   2. Important to avoid division by MakeNonZero(sl) when computing eta 
!      because that computation can become undefined for reservoir inflow
!      conditions, i.e., for a vanishing velocity vector. 
!
!******************************************************************************
!
! $Id: BcondInflowPerf.F90,v 1.1.1.1 2014/05/05 21:47:47 tmish Exp $
!
! Copyright: (c) 2002-2004 by the University of Illinois
!
!******************************************************************************

      SUBROUTINE BcondInflowPerf(bcOptType,bcOptFixed,ptot,ttot,betah,
     >                           betav,mach,sxn,syn,szn,cpgas,mm,rl,rul,
     >                           rvl,rwl,rr,rur,rvr,rwr,rer,pr)
      IMPLICIT NONE

      integer BCOPT_SUBSONIC, BCOPT_MIXED, BCOPT_FIXED_NO
! faked flags
      parameter (BCOPT_SUBSONIC = 1)
      parameter (BCOPT_MIXED = 2)
      parameter (BCOPT_FIXED_NO = 2)
      real  MixtPerf_C_Co2GUVW, MixtPerf_C_DGP, MixtPerf_C_GRT,
     >  MixtPerf_Co2_CGUVW, MixtPerf_C2_GRT,MixtPerf_D_PRT,
     >  MixtPerf_Eo_DGPUVW, MixtPerf_Eo_DGPVm, MixtPerf_G_CpR,
     >  MixtPerf_P_GMaPo, MixtPerf_P_GPoTTo, MixtPerf_Po_GPTTo,
     >  MixtPerf_Po_CGPUVW, MixtPerf_R_M, MixtPerf_T_CGR,
     >  MixtPerf_T_GMaTo,  MixtPerf_Vm_C2Co2G
      external  MixtPerf_C_Co2GUVW, MixtPerf_C_DGP, MixtPerf_C_GRT,
     >  MixtPerf_Co2_CGUVW, MixtPerf_C2_GRT,MixtPerf_D_PRT,
     >  MixtPerf_Eo_DGPUVW, MixtPerf_Eo_DGPVm, MixtPerf_G_CpR,
     >  MixtPerf_P_GMaPo, MixtPerf_P_GPoTTo, MixtPerf_Po_GPTTo,
     >  MixtPerf_Po_CGPUVW, MixtPerf_R_M, MixtPerf_T_CGR,
     >  MixtPerf_T_GMaTo,  MixtPerf_Vm_C2Co2G


! ... parameters
      INTEGER bcOptFixed,bcOptType!, INTENT(IN) ::

      REAL betah, betav, cpgas, mach, mm, sxn, syn, szn, !, INTENT(IN) ::
     >                        ptot, rl, rul, rvl, rwl, ttot

      REAL rer, rr, rur, rvr, rwr, pr!, INTENT(OUT) ::

! ... local variables
      REAL al, ar, a02, cp, disc, eta, g, gm1, igm1, ql, rgas, Rm,
     >            sl, sr , tr, ul, ur, vl, vr, wl, wr

!******************************************************************************
! gas properties

      rgas = MixtPerf_R_M(mm)
      g    = MixtPerf_G_CpR(cpgas,rgas)

! subsonic or mixed -----------------------------------------------------------   

      IF ( bcOptType.eq.BCOPT_SUBSONIC.OR.bcOptType.eq.BCOPT_MIXED) THEN
         gm1  = g - 1.0
         igm1 = 1.0/gm1

         ul = rul/rl
         vl = rvl/rl
         wl = rwl/rl

         a02 = MixtPerf_C2_GRT(g,rgas,ttot)

         al = MixtPerf_C_Co2GUVW(a02,g,ul,vl,wl) ! make al consistent with a02
         ql = ul*sxn + vl*syn + wl*szn

! - subsonic

         IF ( ABS(ql) .lt. al ) THEN
            sl = SQRT(ul*ul + vl*vl + wl*wl)
         
            IF ( bcOptFixed .eq. BCOPT_FIXED_NO ) THEN 
               IF ( sl .gt. 1.0E-6 ) THEN ! Avoid ill-defined angle computation
                  eta = ql/sl        
               ELSE 
                  eta = -1.0
               END IF ! sl
            ELSE 
               eta = -1.0
            END IF ! bcOptFixed

            Rm   = al - 0.5*gm1*ql
            disc = 0.5*gm1*eta**2*
     >       (a02/(Rm*Rm)*(1.0 + 0.5*gm1*eta**2) - 1.0)     
        
            IF ( disc .lt. 0.0 ) THEN ! discriminant cannot be negative
               ar = SQRT(a02)
               tr = ttot
               pr = ptot
               sr = 0.0
            ELSE
               ar = Rm/(1.0 + 0.5*gm1*eta*eta)*(1.0+SQRT(disc))
               tr = MixtPerf_T_CGR(ar,g,rgas)
               pr = MixtPerf_P_GPoTTo(g,ptot,tr,ttot)
               sr = MixtPerf_Vm_C2Co2G(ar*ar,a02,g)
            END IF ! disc    
                     
            rr = MixtPerf_D_PRT( pr,rgas,tr )
            ur = sr*COS(betah)*COS(betav)
            vr = sr*SIN(betah)
            wr = sr*COS(betah)*SIN(betav)

            rer = rr*MixtPerf_Eo_DGPVm(rr,g,pr,sr)
            rur = rr*ur
            rvr = rr*vr
            rwr = rr*wr

! - supersonic

         ELSE
            IF ( bcOptType .eq. BCOPT_MIXED ) THEN
               pr = mixtPerf_P_GMaPo(g,mach,ptot)
               tr = mixtPerf_T_GMaTo(g,mach,ttot)
               rr = mixtPerf_D_PRT(pr,rgas,tr)
               ar = mixtPerf_C_GRT(g,rgas,tr)

               ur = mach*ar*COS(betah)*COS(betav)
               vr = mach*ar*SIN(betah)
               wr = mach*ar*COS(betah)*SIN(betav)

               rer = rr*MixtPerf_Eo_DGPUVW(rr,g,pr,ur,vr,wr)
               rur = rr*ur
               rvr = rr*vr
               rwr = rr*wr
            END IF ! bcOptType
         END IF ! ql < al

! supersonic ------------------------------------------------------------------

      ELSE ! bcOptType .eq. BCOPT_SUPERSONIC
         pr = mixtPerf_P_GMaPo(g,mach,ptot)
         tr = mixtPerf_T_GMaTo(g,mach,ttot)
         rr = mixtPerf_D_PRT(pr,rgas,tr)
         ar = mixtPerf_C_GRT(g,rgas,tr)

         ur = mach*ar*COS(betah)*COS(betav)
         vr = mach*ar*SIN(betah)
         wr = mach*ar*COS(betah)*SIN(betav) 

         rer = rr*MixtPerf_Eo_DGPUVW(rr,g,pr,ur,vr,wr)
         rur = rr*ur
         rvr = rr*vr
         rwr = rr*wr
      END IF ! bcOptType

      return
      end

!******************************************************************************
!
! RCS Revision history:
!
! $Log: BcondInflowPerf.F90,v $
! Revision 1.1.1.1  2014/05/05 21:47:47  tmish
! Initial checkin for rocflu macro.
!
! Revision 1.3  2008/12/06 08:43:31  mtcampbe
! Updated license.
!
! Revision 1.2  2008/11/19 22:16:46  mtcampbe
! Added Illinois Open Source License/Copyright
!
! Revision 1.1  2007/04/09 18:48:32  haselbac
! Initial revision after split from RocfloMP
!
! Revision 1.1  2007/04/09 17:59:24  haselbac
! Initial revision after split from RocfloMP
!
! Revision 1.1  2004/12/01 16:47:56  haselbac
! Initial revision after changing case
!
! Revision 1.7  2004/01/29 22:52:36  haselbac
! Added bcOptFixed, fixed bug, clean-up
!
! Revision 1.6  2003/12/04 03:22:56  haselbac
! Fixed bug in formulation, added partial fix for eta
!
! Revision 1.5  2003/11/20 16:40:35  mdbrandy
! Backing out RocfluidMP changes from 11-17-03
!
! Revision 1.2  2002/06/22 00:49:50  jblazek
! Modified interfaces to BC routines.
!
! Revision 1.1  2002/06/10 21:19:34  haselbac
! Initial revision
!
!******************************************************************************
