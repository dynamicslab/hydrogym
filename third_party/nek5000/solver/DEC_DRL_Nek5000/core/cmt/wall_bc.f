      subroutine wallbc(nstate,f,e,faceq,bcq,flux)
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      include 'CMTBCDATA'

      integer nstate,f,e
      real    faceq(nx1*nz1,2*ndim,nelt,nstate)
      real    bcq(nx1*nz1,2*ndim,nelt,nstate) 
      real    flux(nx1*nz1,2*ndim,nelt,*)

      call slipwall_rflu(nstate,f,e,faceq,bcq,flux) ! calls RindState stuff

      return
      end

!--------------------------------------------------------------------

      subroutine slipwall_rflu(nvar,f,e,faceq,bcq,fluxw)
      include 'SIZE'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'GEOM'
      include 'NEKUSE'
      include 'INPUT'
      include 'PARALLEL'
      include 'DG'
      include 'MASS'
      integer  f,e
! JH091614 faceq now has intent(inout)...
! JH031315 not anymore. nobody changes qminus here. that's dumb
      real faceq(nx1*nz1,2*ndim,nelt,nvar)
      real bcq(nx1*nz1,2*ndim,nelt,nvar)
      real fluxw(nx1*nz1,2*ndim,nelt,*)
      integer i, nxz, fdim
      real nx,ny,nz,rl,ul,vl,wl,pl,fs
      parameter (lfd1=lxd*lzd,lfc1=lx1*lz1)
      common /SCRNS/ nxf(lfd1),nyf(lfd1),nzf(lfd1),fs2(lfd1),
     >               ufacel(lfd1,5),plc(lfc1),ufacer(lfd1,5),prc(lfd1),
     >               flx(lfd1,5),plf(lfd1),jaco_c(lfc1),
     >               jaco_f(lfd1),dumminus(lfd1,5)
      real nxf,nyf,nzf,ufacel,ufacer,plc,prc,plf,jaco_c,jaco_f,dumminus

      nxz=nx1*nz1
      nxzd=nxd*nzd
      fdim=ndim-1
      ieg=lglel(e)

! I know this says slipwall, but to the inviscid terms all walls are
! slip. or something.
      call facind(i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)    
      l=0
      do iz=k0,k1
      do iy=j0,j1
      do ix=i0,i1
         call nekasgn(ix,iy,iz,e)
         call cmtasgn(ix,iy,iz,e)
         call userbc (ix,iy,iz,f,ieg) ! get molarmass, phi
c                                     ! ux,uy,uz
         l=l+1
         nx = unx(l,1,f,e)
         ny = uny(l,1,f,e)
         nz = unz(l,1,f,e)
         rl = faceq(l,f,e,irho)
         ul = faceq(l,f,e,iux)
         vl = faceq(l,f,e,iuy)
         wl = faceq(l,f,e,iuz)
         plc(l)= faceq(l,f,e,ipr)
         fs = 0.0 ! no moving grid for awhile, and it will not look anything
                  ! like RocFlu
         call RFLU_SetRindStateSlipWallPerf(cp,molarmass,nx,ny,nz,
     >                                      rl,ul,vl,wl,fs,plc(l))
         bcq(l,f,e,irho)=rl ! probably shouldn't be setting these
         bcq(l,f,e,iux)=ul  ! on the other hand, it ensures [[]]=0
         bcq(l,f,e,iuy)=vl  ! THINK!!!
         bcq(l,f,e,iuz)=wl
         bcq(l,f,e,ipr)=plc(l)! from RFLU_SetRindStateSlipWallPerf
         bcq(l,f,e,iph)=phi
         bcq(l,f,e,iu1)=faceq(l,f,e,iu1)
         bcq(l,f,e,iu2)=faceq(l,f,e,iu2)
         bcq(l,f,e,iu3)=faceq(l,f,e,iu3)
         bcq(l,f,e,iu4)=faceq(l,f,e,iu4)
         bcq(l,f,e,iu5)=faceq(l,f,e,iu5)
! need a different place to set dirichlet BC for viscous fluxes
!           bcq(l,f,e,iux)=ux ! better b
!           bcq(l,f,e,iuy)=uy
!           bcq(l,f,e,iuz)=uz
!        if (cbc(f,e,2) .eq. 'W  ') bcq(l,f,e,ithm)=temp
         plc(l)=plc(l)*phi
      enddo
      enddo
      enddo

! Inviscid flux at walls is due to pressure only. should probably just
! hardcode that instead of calling CentralInviscid so trivially
      if (nxd.gt.nx1) then
         call map_faced(nxf,unx(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nyf,uny(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(nzf,unz(1,1,f,e),nx1,nxd,fdim,0)
         call map_faced(plf,plc,nx1,nxd,fdim,0)

         call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz)
         call map_faced(jaco_f,jaco_c,nx1,nxd,fdim,0)
         call col2(jaco_f,wghtf,nxzd)
      else
         call copy(nxf,unx(1,1,f,e),nxz)
         call copy(nyf,uny(1,1,f,e),nxz)
         call copy(nzf,unz(1,1,f,e),nxz)
         call copy(plf,plc,nxz)

         call copy(jaco_f,area(1,1,f,e),nxz)
      endif
      call rzero(dumminus,toteq*nxzd)
      call map_faced(dumminus(1,1),faceq(1,f,e,iu1),nx1,nxd,fdim,0)
      call rzero(fs2,nxzd)
      call CentralInviscid_FluxFunction(nxzd,nxf,nyf,nzf,fs2,dumminus,
     >                                    plf,dumminus,plf,flx)

      do ieq=1,toteq-1
         call col2(flx(1,ieq),jaco_f,nxzd)
      enddo

      if (nxd.gt.nx1) then
         do j=1,toteq-1
            call map_faced(fluxw(1,f,e,j),flx(1,j),nx1,nxd,fdim,1)
         enddo
         if (cbc(f,e,2).ne.'I  ') call map_faced(fluxw(1,f,e,toteq),
     >                              flx(1,toteq),nx1,nxd,fdim,1)
      else
         do j=1,toteq-1
            call copy(fluxw(1,f,e,j),flx(1,j),nxz)
         enddo
         if (cbc(f,e,2).ne.'I  ') call copy(fluxw(1,f,e,toteq),
     >                              flx(1,toteq),nxz)
      endif

      return
      end

!-----------------------------------------------------------------------
! ******************************************************************************
!
! Purpose: Set rind state for slip-wall boundaries and perfect gas.
!
! Description: Torn bleeding from RocFlu. I think "rind" means the same thing as
!              "ghost," but I gotta admit that it's a better way of putting it.
!              Not sure if low-order reconstruction lurks here.
! Input:
!   cpGas       Specific heat at constant pressure
!   mmGas       Molecular mass
!   nx,ny,nz    Components of unit normal vector
!   rl          Density
!   ul         x-velocity component
!   vl         y-velocity component
!   wl         z-velocity component
!   fs          Grid speed
!   pl          Pressure
!
! Output: 
!   pl          Pressure
!
! Notes: 
!   1. Valid only for thermally and calorically perfect gas.
!
! ******************************************************************************

      SUBROUTINE RFLU_SetRindStateSlipWallPerf(cpGas,mmGas,nx,ny,nz,rl,
     >                                         ul,vl,wl,fs,pl)

! ******************************************************************************
!   Declarations and definitions
! ******************************************************************************

      real MixtPerf_R_M, MixtPerf_G_CpR, MixtPerf_C_DGP
      external MixtPerf_R_M, MixtPerf_G_CpR, MixtPerf_C_DGP

! ==============================================================================  
!   Arguments 
! ==============================================================================  

      REAL cpGas,fs,mmGas,nx,ny,nz,rl,ul,vl,wl
      REAL pl

! ==============================================================================  
!   Locals 
! ==============================================================================  

      REAL al,gGas,irl,ql,rGas,term
          
! ******************************************************************************
!   Compute wall pressure
! ******************************************************************************

      rGas = MixtPerf_R_M(mmGas)
      gGas = MixtPerf_G_CpR(cpGas,rGas)
 
      irl = 1.0/rl
      ql  = ul*nx + vl*ny + wl*nz - fs
 
      al  = MixtPerf_C_DGP(rl,gGas,pl)

      IF ( ql .lt. 0.0 ) THEN
         term = 1.0 + 0.5*(gGas-1.0)*ql/al
         pl   = pl*term**(2.0*gGas/(gGas-1.0))
      ELSE
         term = (gGas+1.0)/4.0
         pl   = pl + term*rl*ql*(ql + SQRT(al*al+term*term*ql*ql)/term)
      END IF ! ql
 
! ******************************************************************************
!   End
! ******************************************************************************

      end
