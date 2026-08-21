c--------------------------------------------------------------------
      subroutine outflow(nvar,f,e,faceq,bcq,flux) ! don't really need nvar anymore
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'

      integer  nvar,f,e
      real faceq(nx1,nz1,2*ldim,nelt,nvar)
      real bcq(nx1,nz1,2*ldim,nelt,nvar)
      real flux(nx1*nz1,2*ldim,nelt,*)

      call outflow_rflu(nvar,f,e,faceq,bcq,flux)

      return
      end

!--------------------------------------------------------------------

      subroutine outflow_rflu(nvar,f,e,faceq,bcq,flux1)
      include 'SIZE'
      include 'NEKUSE'
      include 'CMTDATA'
      include 'CMTBCDATA'
      include 'INPUT'
      include 'GEOM'
      include 'PARALLEL'
      include 'DG'

      integer i,bcOpt
      integer  f,e,fdim
      real faceq(nx1*nz1,2*ldim,nelt,nvar)
      real bcq(nx1*nz1,2*ldim,nelt,nvar)
      real flux1(nx1*nz1,2*ldim,nelt,*)
      real sxn,syn,szn,rhou,rhov,rhow,pl,rhob,rhoub,rhovb,rhowb,rhoeb
      parameter (lfd1=lxd*lzd,lfc1=lx1*lz1)
      common /SCRNS/ nxf(lfd1),nyf(lfd1),nzf(lfd1),fs(lfd1),
     >               ufacel(lfd1,5),plc(lfc1),ufacer(lfd1,5),prc(lfc1),
     >               flx(lfd1,5),plf(lfd1),prf(lfd1),jaco_c(lfc1),
     >               jaco_f(lfd1)
      real nxf,nyf,nzf,fs,ufacel,ufacer,plc,prc,plf,prf,jaco_c,jaco_f
      real philc(lfc1),philf(lfd1),molmlf(lfd1),molmlc(lfc1),cvglc(lfc1)
      real cvglf(lfd1),cpglc(lfc1),cpglf(lfd1)

      nxz=nx1*nz1
      nxzd=nxd*nzd
      fdim=ndim-1
      ieg=lglel(e)

      call facind(i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)    
      l=0
      do iz=k0,k1
      do iy=j0,j1
      do ix=i0,i1
         call nekasgn(ix,iy,iz,e)     ! gives us phi- and rho-
         call cmtasgn(ix,iy,iz,e)     ! gives us phi- and rho-
         call userbc (ix,iy,iz,f,ieg) ! just for molarmass
         l=l+1
         sxn = unx(l,1,f,e)
         syn = uny(l,1,f,e)
         szn = unz(l,1,f,e)
         philc(l) = phi
         molmlc(l) = molarmass
         rhou= faceq(l,f,e,iu2)/phi
         rhov= faceq(l,f,e,iu3)/phi
         rhow= faceq(l,f,e,iu4)/phi
         rhoe= faceq(l,f,e,iu5)/phi
         plc(l)= faceq(l,f,e,ipr)
         cpglc(l)=faceq(l,f,e,icpf)/rho
         cvglc(l)=faceq(l,f,e,icvf)/rho
c        fs = 0.0
         if(outflsub)then
            pres= pinfty
         else
            pres= plc(l)
         endif
         call BcondOutflowPerf(1,pres,sxn,syn,szn,cpglc(l),molmlc(l),
     >                         rho,rhou,rhov,rhow,rhoe,plc(l),
     >                         rhob,rhoub,rhovb,rhowb,rhoeb )
         bcq(l,f,e,irho)=rhob
         bcq(l,f,e,iux)=rhoub/rhob
         bcq(l,f,e,iuy)=rhovb/rhob
         bcq(l,f,e,iuz)=rhowb/rhob
! dammit fix this. tdstate to the rescue?
         bcq(l,f,e,ithm)=(rhoeb-0.5*(rhoub**2+rhovb**2+rhowb**2)/rhob)/
     >                   cvglc(l)
! dammit fix that
         bcq(l,f,e,iu1)=rhob*phi
         bcq(l,f,e,iu2)=rhoub*phi
         bcq(l,f,e,iu3)=rhovb*phi
         bcq(l,f,e,iu4)=rhowb*phi
         bcq(l,f,e,iu5)=rhoeb*phi
         bcq(l,f,e,iph)=phi
         bcq(l,f,e,ipr)=pres
         plc(l)=plc(l)*phi
         prc(l)=phi*pres ! needs phi. U in bcq has phi already
      enddo
      enddo
      enddo

      if (nxd.gt.nx1) then
         call map_faced(nxf,unx(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nyf,uny(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nzf,unz(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(plf,plc,nx1,nxd,fdim,0)
         call map_faced(prf,prc,nx1,nxd,fdim,0)
         call map_faced(philf,philc,nx1,nxd,fdim,0)
c     call map_faced(cvglf,cvglc,nx1,nxd,fdim,0)
         call map_faced(cpglf,cpglc,nx1,nxd,fdim,0)
         call map_faced(molmlf,molmlc,nx1,nxd,fdim,0)
         call map_faced(ufacel(1,1),faceq(1,f,e,iu1),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,2),faceq(1,f,e,iu2),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,3),faceq(1,f,e,iu3),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,4),faceq(1,f,e,iu4),nx1,nxd,fdim,0)
         call map_faced(ufacel(1,5),faceq(1,f,e,iu5),nx1,nxd,fdim,0)
         call map_faced(ufacer(1,1),bcq(1,f,e,iu1),  nx1,nxd,fdim,0)
         call map_faced(ufacer(1,2),bcq(1,f,e,iu2),  nx1,nxd,fdim,0)
         call map_faced(ufacer(1,3),bcq(1,f,e,iu3),  nx1,nxd,fdim,0)
         call map_faced(ufacer(1,4),bcq(1,f,e,iu4),  nx1,nxd,fdim,0)
         call map_faced(ufacer(1,5),bcq(1,f,e,iu5),  nx1,nxd,fdim,0)

         call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz)
         call map_faced(jaco_f,jaco_c,nx1,nxd,fdim,0) 
         call col2(jaco_f,wghtf,nxzd)
      else

         call copy(nxf,unx(1,1,f,e),nxz)
         call copy(nyf,uny(1,1,f,e),nxz)
         call copy(nzf,unz(1,1,f,e),nxz)
         call copy(plf,plc,nxz)
         call copy(prf,prc,nxz)
         call copy(philf,philc,nxz)
         call copy(cpglf,cpglc,nxz)
         call copy(molmlf,molmlc,nxz)
         call copy(ufacel(1,1),faceq(1,f,e,iu1),nxz)
         call copy(ufacel(1,2),faceq(1,f,e,iu2),nxz)
         call copy(ufacel(1,3),faceq(1,f,e,iu3),nxz)
         call copy(ufacel(1,4),faceq(1,f,e,iu4),nxz)
         call copy(ufacel(1,5),faceq(1,f,e,iu5),nxz)
         call copy(ufacer(1,1),bcq(1,f,e,iu1),  nxz)
         call copy(ufacer(1,2),bcq(1,f,e,iu2),  nxz)
         call copy(ufacer(1,3),bcq(1,f,e,iu3),  nxz)
         call copy(ufacer(1,4),bcq(1,f,e,iu4),  nxz)
         call copy(ufacer(1,5),bcq(1,f,e,iu5),  nxz)

         call copy(jaco_f,area(1,1,f,e),nxz) 
      endif

      do l=1,lfd1
         sxn  = nxf(l)
         syn  = nyf(l)
         szn  = nzf(l)
         rho  = ufacel(l,1)/philf(l) 
         rhou = ufacel(l,2)/philf(l)
         rhov = ufacel(l,3)/philf(l)
         rhow = ufacel(l,4)/philf(l)
         rhoe = ufacel(l,5)/philf(l)
         plf(l) = plf(l)/philf(l)
c        fs   = 0.0
         if(outflsub)then
            pres= pinfty
         else
            pres= plf(l)
         endif
         call BcondOutflowPerf(1,pres,sxn,syn,szn,cpglf(l),molmlf(l)
     >                        ,rho,rhou,rhov,rhow,rhoe,plf(l)
     >                        ,rhob,rhoub,rhovb,rhowb,rhoeb )
         ufacer(l,1) = rhob*philf(l)
         ufacer(l,2) = rhoub*philf(l)
         ufacer(l,3) = rhovb*philf(l)
         ufacer(l,4) = rhowb*philf(l)
         ufacer(l,5) = rhoeb*philf(l)

         plf(l)      = plf(l)*philf(l)
         prf(l)      = pres*philf(l)
      enddo

      call rzero(fs,nxzd)

!-----------------------------------------------------------------------
! Inviscid flux at inflow can probably just be hardcoded instead of
! derived from a trivial call of CentralInviscid_FluxFunction
c     call CentralInviscid_FluxFunction(nxzd,nxf,nyf,nzf,fs,ufacel,plf,
c    >                                  ufacer,prf,flx)
c MS010716 This central flux call is trivial. Flux computation is based 
c solely on the right state or the ufacer array. 
c Note that this change was important for inflow BC.
c Outflow BC is not sensitive to method used to compute the flux. 
c This was tested for 
c  ---  uniform flow and subsonic flow over a cylinder. 
c  (Need to test supersonic uniform flow !)
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


!******************************************************************************
!
! Purpose: set outflow boundary condition for one cell.
!
! Description: the subsonic boundary condition is based on non-reflecting,
!              characteristics method of Whitfield and Janus: Three-Dimensional
!              Unsteady Euler Equations Solution Using Flux Vector Splitting.
!              AIAA Paper 84-1552, 1984. The supersonic boundary condition
!              consists of simple extrapolation.
!
! Input: bcOpt    = boundary treatment: subsonic, supersonic, or mixed
!        pout     = given static outlet pressure
!        sx/y/zn  = components of ortho-normalized face vector (outward facing)
!        cpgas    = specific heat at constant pressure (boundary cell)
!        mol      = molecular mass at boundary cell
!        rho      = density at boundary cell
!        rhou/v/w = density * velocity components at boundary cell
!        rhoe     = density * total energy at boundary cell
!        press    = static pressure at boundary cell
!
! Output: rhob      = density at boundary
!         rhou/v/wb = density * velocity components at boundary
!         rhoeb     = density * total energy at boundary
!
! Notes: this condition is valid only for thermally and calorically
!        perfect gas (supersonic outflow valid for all gases).
!
!******************************************************************************
!
! $Id: BcondOutflowPerf.F90,v 1.1.1.1 2014/05/05 21:47:47 tmish Exp $
!
! Copyright: (c) 2002 by the University of Illinois
!
!******************************************************************************

      SUBROUTINE BcondOutflowPerf( bcOpt,pout,sxn,syn,szn,cpgas,mol,
     >                       rho,rhou,rhov,rhow,rhoe,press,
     >                       rhob,rhoub,rhovb,rhowb,rhoeb )

      IMPLICIT NONE
      integer bcopt_subsonic,bcopt_mixed
      parameter (BCOPT_SUBSONIC=1)
      parameter (BCOPT_MIXED=0)
      real MixtPerf_C_DGP,MixtPerf_Eo_DGPUVW,MixtPerf_G_CpR,MixtPerf_R_M
     >   , MixtPerf_P_DEoGVm2
      external MixtPerf_C_DGP,MixtPerf_Eo_DGPUVW,MixtPerf_G_CpR,
     >     MixtPerf_R_M,MixtPerf_P_DEoGVm2

! ... parameters
      INTEGER bcOpt

      REAL pout
      REAL rho, rhou, rhov, rhow, rhoe, press
      REAL sxn, syn, szn, cpgas, mol
      REAL rhob, rhoub, rhovb, rhowb, rhoeb

! ... local variables
      REAL csound, rgas, gamma, gam1, u, v, w, mach, rrhoc, deltp,
     >            ub, vb, wb, vnd

!******************************************************************************
! gas properties; velocity components; Mach number

      rgas  = MixtPerf_R_M( mol )
      gamma = MixtPerf_G_CpR( cpgas,rgas )
      gam1  = gamma - 1.0

      u      = rhou/rho
      v      = rhov/rho
      w      = rhow/rho
      csound = MixtPerf_C_DGP( rho,gamma,press )
      mach   = SQRT(u*u+v*v+w*w)/csound

! subsonic flow ---------------------------------------------------------------

      IF (mach .lt. 1.0 .AND.
     >(bcOpt .eq. BCOPT_SUBSONIC .OR. bcOpt .eq. BCOPT_MIXED)) THEN
         rrhoc = 1.0/(rho*csound)
         deltp = press - pout
         rhob  = rho - deltp/(csound*csound)
         ub    = u + sxn*deltp*rrhoc
         vb    = v + syn*deltp*rrhoc
         wb    = w + szn*deltp*rrhoc

! - special treatment to prevent "deltp" from changing the sign
!   of velocity components. This may happen for very small u, v, w.

         vnd = ub*sxn + vb*syn + wb*szn
         IF ( vnd .lt. 0.0 ) THEN ! inflow at outflow boundary
            ub = SIGN(1.0,u)*MAX(ABS(ub),ABS(u))
            vb = SIGN(1.0,v)*MAX(ABS(vb),ABS(v))
            wb = SIGN(1.0,w)*MAX(ABS(wb),ABS(w))
         END IF ! vnd

         rhoub = rhob*ub
         rhovb = rhob*vb
         rhowb = rhob*wb
         rhoeb = rhob*MixtPerf_Eo_DGPUVW( rhob,gamma,pout,ub,vb,wb )

! supersonic flow -------------------------------------------------------------

      ELSE
         rhob  = rho
         rhoub = rhou
         rhovb = rhov
         rhowb = rhow
         rhoeb = rhoe
      END IF ! mach

      end

!******************************************************************************
!
! RCS Revision history:
!
! $Log: BcondOutflowPerf.F90,v $
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
! Revision 1.2  2006/03/26 20:21:09  haselbac
! Fix mistake in declarations
!
! Revision 1.1  2004/12/01 16:48:04  haselbac
! Initial revision after changing case
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

