C> @file wall_bc.f Dirichlet states for wall boundary conditions
! FUN FACT: Did you know that bdry.f has a subroutine called
!           BCNEUSC
      subroutine wallbc2(nstate,f,e,facew,wbc)
! DIRICHLET WALL CONDITIONS BECAUSE I DONT KNOW HOW TO INDEX
! UNX in userbc with volume indices instead of face indices
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      include 'SOLN'
      include 'NEKUSE'
      include 'PARALLEL'
      include 'CMTBCDATA'
      include 'CMTDATA'

      integer nstate,f,e
      real    facew(lx1*lz1,2*ldim,nelt,nstate)
      real    wbc(lx1*lz1,2*ldim,nelt,nstate) 
      common /nekcb/ cb
      character*3 cb

      tol=1.0e-10
! JH112116
! rind state for inviscid fluxes is different from viscous fluxes? not
! sure what the right thing to do is.
! JH031617 Collis (CTR 2002-ish), Hartmann & Houston (2006-8) probably BR
!          and Dolejsi and Feistatuer (2015) (check that)
!          all say YES, inviscid rind and viscous rind are different.
      call facind(i0,i1,j0,j1,k0,k1,lx1,ly1,lz1,f)    
      ieg=lglel(e)
      l=0
      do iz=k0,k1
      do iy=j0,j1
      do ix=i0,i1
         call nekasgn(ix,iy,iz,e)
         call cmtasgn(ix,iy,iz,e)
         call userbc (ix,iy,iz,f,ieg)
         l=l+1

! bring this outside of the face point loop you moron
         if (abs(vdiff(ix,iy,iz,e,ilam)) .gt. tol) then ! physical not artvisc

            wbc(l,f,e,iux)=ux
            wbc(l,f,e,iuy)=uy
            wbc(l,f,e,iuz)=uz
!-----------------------------------------------------------------
! JH112116 I need to check wbc(:,ipr) to make sure it is unchanced
!          from inviscid computation (assuming that's the right
!          answer for general viscous BC).
!-----------------------------------------------------------------
            wbc(l,f,e,iph) =phi
            wbc(l,f,e,ithm)=temp
            wbc(l,f,e,iu1) =facew(l,f,e,iu1)
            wbc(l,f,e,iu2) =wbc(l,f,e,iu1)*ux
            wbc(l,f,e,iu3) =wbc(l,f,e,iu1)*uy
            wbc(l,f,e,iu4) =wbc(l,f,e,iu1)*uz
            if (cb .eq. 'W  ') then ! consider taking properties from userbc too
!           wbc(l,f,e,iu5)=wbc(l,f,e,iu1)*facew(l,f,e,icvf)
               wbc(l,f,e,iu5)=phi*facew(l,f,e,icvf)*temp+
     >         0.5/wbc(l,f,e,iu1)*(wbc(l,f,e,iu2)**2+wbc(l,f,e,iu3)**2+
     >                          wbc(l,f,e,iu4)**2)
            else ! BETTA JUST BE 'I  '
!-------------------------------------------------------------
! JH111516 HARDCODING ADIABATIC WALL. DO SMARTER SOON
!          METHOD "B"
!           wbc(l,f,e,iu5)=facew(l,f,e,iu5)-0.5/facew(l,f,e,iu1)*
!    >     (facew(l,f,e,iu2)**2+facew(l,f,e,iu3)**2+facew(l,f,e,iu4)**2)
!          METHOD "A"
               wbc(l,f,e,iu5)=facew(l,f,e,iu5)
            endif
! JH111516 INVISCID HARDCODING ADIABATIC WALL. DO SMARTER SOON
!-------------------------------------------------------------
         else ! artvisc only

! JH031617 For now, artificial viscosity does not directly contribute to
!          boundary fluxes at all. This means dU=0 for IGTU and gradU is
!          strictly interior for IGU
            do m=1,nqq ! TEST FOR vDIFF OUTSIDE WHAT THE EHLLO IS WRONG WITH YOU
               wbc(l,f,e,m)=facew(l,f,e,m)
            enddo

         endif ! physical viscosity or artvisc
      enddo
      enddo
      enddo

      return
      end

      subroutine wallbc_inviscid(nstate,f,e,facew,wbc)
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      include 'CMTBCDATA'

      integer nstate,f,e
      real    facew(lx1*lz1,2*ldim,nelt,nstate)
      real    wbc(lx1*lz1,2*ldim,nelt,nstate) 

! JH102016
! rind state for inviscid fluxes is different from viscous fluxes
      call reflect_rind(nstate,f,e,facew,wbc)

      return
      end

      subroutine reflect_rind(nvar,f,e,facew,wbc)
      include 'SIZE'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'GEOM'
      include 'INPUT'
      include 'PARALLEL'
      include 'DG'
      include 'MASS'
      include 'TSTEP'
      integer  f,e
! JH091614 facew now has intent(inout)...
! JH031315 not anymore. nobody changes qminus here. that's dumb
      real facew(lx1*lz1,2*ldim,nelt,nvar)
      real wbc(lx1*lz1,2*ldim,nelt,nvar)
      integer i, nxz, fdim
      real nx,ny,nz,rl,ul,vl,wl,pl,fs
      parameter (lfd1=lxd*lzd,lfc1=lx1*lz1)

      nxz=lx1*lz1
      nxzd=lxd*lzd
      fdim=ldim-1
      ieg=lglel(e)
      ifield=1

! I know this says slipwall, but to the inviscid terms all walls are
! slip. or something.
      call facind(i0,i1,j0,j1,k0,k1,lx1,ly1,lz1,f)    
      do l=1,nxz
         nx = unx(l,1,f,e)
         ny = uny(l,1,f,e)
         nz = unz(l,1,f,e)
         rl = facew(l,f,e,irho)
         rr = rl
         ul = facew(l,f,e,iux)
         vl = facew(l,f,e,iuy)
         wl = facew(l,f,e,iuz)
         fs = 0.0 ! no moving grid for awhile, and it will not look anything
                  ! like RocFlu

! JH111516 Mirror a la' Dolejsi & Feistauer (2015) section 8.3.1.2
! JH021717 This is for inviscid fluxes, which are produced by the Riemann
!          solver. Presently, this is AUSM, which acts on primitive variables
!          as-coded. Still, it always makes sense to form UBC, so we do so
!          here even though it is different for viscous BC
         udotn = ul*nx+vl*ny+wl*nz
         ur = ul-2.0*udotn*nx
         vr = vl-2.0*udotn*ny
         wr = wl-2.0*udotn*nz
         wbc(l,f,e,irho)= rr
         wbc(l,f,e,iux) = ur
         wbc(l,f,e,iuy) = vr
         wbc(l,f,e,iuz) = wr
         wbc(l,f,e,ipr) = facew(l,f,e,ipr)
         wbc(l,f,e,ithm)= facew(l,f,e,ithm)
         wbc(l,f,e,isnd)= facew(l,f,e,isnd)
         wbc(l,f,e,iph) = facew(l,f,e,iph)
         wbc(l,f,e,icvf)= facew(l,f,e,icvf)
         wbc(l,f,e,icpf)= facew(l,f,e,icpf)
         wbc(l,f,e,iu1)= facew(l,f,e,iu1)
         wbc(l,f,e,iu2)= rr*ur
         wbc(l,f,e,iu3)= rr*vr
         wbc(l,f,e,iu4)= rr*wr
         wbc(l,f,e,iu5)= facew(l,f,e,iu5)
      enddo

      return
      end

!--------------------------------------------------------------------
! NOT LONG FOR THIS WORLD

      subroutine slipwall_rflu(nvar,f,e,facew,wbc,fluxw)
      include 'SIZE'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'GEOM'
      include 'NEKUSE'
      include 'INPUT'
      include 'PARALLEL'
      include 'DG'
      include 'MASS'
      include 'TSTEP'
      integer  f,e
! JH091614 facew now has intent(inout)...
! JH031315 not anymore. nobody changes qminus here. that's dumb
      real facew(lx1*lz1,2*ldim,nelt,nvar)
      real wbc(lx1*lz1,2*ldim,nelt,nvar)
      real fluxw(lx1*lz1,2*ldim,nelt,*)
      integer i, nxz, fdim
      real nx,ny,nz,rl,ul,vl,wl,pl,fs
      parameter (lfd1=lxd*lzd,lfc1=lx1*lz1)
      common /SCRNS/ nxf(lfd1),nyf(lfd1),nzf(lfd1),fs2(lfd1),
     >               ufacel(lfd1,5),plc(lfc1),ufacer(lfd1,5),prc(lfd1),
     >               flx(lfd1,5),plf(lfd1),jaco_c(lfc1),
     >               jaco_f(lfd1),dumminus(lfd1,5)
      real nxf,nyf,nzf,ufacel,ufacer,plc,prc,plf,jaco_c,jaco_f,dumminus

      nxz=lx1*lz1
      nxzd=lxd*lzd
      fdim=ldim-1
      ieg=lglel(e)
      ifield=1

! I know this says slipwall, but to the inviscid terms all walls are
! slip. or something.
      call facind(i0,i1,j0,j1,k0,k1,lx1,ly1,lz1,f)    
      l=0
      do iz=k0,k1
      do iy=j0,j1
      do ix=i0,i1
         call nekasgn(ix,iy,iz,e)
         call userbc (ix,iy,iz,f,ieg) ! get molarmass, phi
c                                     ! ux,uy,uz someday
         l=l+1
         nx = unx(l,1,f,e)
         ny = uny(l,1,f,e)
         nz = unz(l,1,f,e)
         rl = facew(l,f,e,irho)
         ul = facew(l,f,e,iux)
         vl = facew(l,f,e,iuy)
         wl = facew(l,f,e,iuz)
         plc(l)= facew(l,f,e,ipr)
         fs = 0.0 ! no moving grid for awhile, and it will not look anything
                  ! like RocFlu
         call RFLU_SetRindStateSlipWallPerf(cp,molarmass,nx,ny,nz,
     >                                      rl,ul,vl,wl,fs,plc(l))
         wbc(l,f,e,irho)=rl

!-----------------------------------------------------------------
! JH111516 INVISCID HARDCODING SLIP WALL. DO THIS SMARTER SOON
! JH111516 Mirror a la' Dolejsi & Feistauer (2015) section 8.3.1.2
!-----------------------------------------------------------------
         udotn=ul*nx+vl*ny+wl*nz
         ur=ul-2.0*udotn*nx
         vr=vl-2.0*udotn*ny
         wr=wl-2.0*udotn*nz
         wbc(l,f,e,iux)=ur
         wbc(l,f,e,iuy)=vr
         wbc(l,f,e,iuz)=wr
! JH111516 SHOULD BE SET TO WALL SPEED i.e. 0 FOR NO-SLIP WALLS!!!
!-----------------------------------------------------------------
         wbc(l,f,e,ipr)=plc(l)! from RFLU_SetRindStateSlipWallPerf
         wbc(l,f,e,iph)=phi
         wbc(l,f,e,iu1)=facew(l,f,e,iu1)
         wbc(l,f,e,iu2)=wbc(l,f,e,iu1)*ur
         wbc(l,f,e,iu3)=wbc(l,f,e,iu1)*vr
         wbc(l,f,e,iu4)=wbc(l,f,e,iu1)*wr
!-------------------------------------------------------------
! JH111516 INVISCID HARDCODING ADIABATIC WALL. DO SMARTER SOON
         wbc(l,f,e,iu5)=facew(l,f,e,iu5)
! JH111516 INVISCID HARDCODING ADIABATIC WALL. DO SMARTER SOON
!-------------------------------------------------------------
! need a different place to set dirichlet BC for viscous fluxes
!           wbc(l,f,e,iux)=ux ! better b
!           wbc(l,f,e,iuy)=uy
!           wbc(l,f,e,iuz)=uz
!        if (cbc(f,e,ifield) .eq. 'W  ') wbc(l,f,e,ithm)=temp
         plc(l)=plc(l)*phi
      enddo
      enddo
      enddo

! Inviscid flux at walls is due to pressure only. should probably just
! hardcode that instead of calling CentralInviscid so trivially
      if (lxd.gt.lx1) then
         call map_faced(nxf,unx(1,1,f,e),lx1,lxd,fdim,0)
         call map_faced(nyf,uny(1,1,f,e),lx1,lxd,fdim,0)
         call map_faced(nzf,unz(1,1,f,e),lx1,lxd,fdim,0)
         call map_faced(plf,plc,lx1,lxd,fdim,0)

         call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz)
         call map_faced(jaco_f,jaco_c,lx1,lxd,fdim,0)
         call col2(jaco_f,wghtf,nxzd)
      else
         call copy(nxf,unx(1,1,f,e),nxz)
         call copy(nyf,uny(1,1,f,e),nxz)
         call copy(nzf,unz(1,1,f,e),nxz)
         call copy(plf,plc,nxz)

         call copy(jaco_f,area(1,1,f,e),nxz)
      endif
      call rzero(dumminus,toteq*nxzd)
      call map_faced(dumminus(1,1),facew(1,f,e,iu1),lx1,lxd,fdim,0)
      call rzero(fs2,nxzd)
! START BY GETTING RID OF THESE TRIVIAL CENTRAL CALLS AND CENTRAL ALTOGETHER
      call CentralInviscid_FluxFunction(nxzd,nxf,nyf,nzf,fs2,dumminus,
     >                                    plf,dumminus,plf,flx)

      do ieq=1,toteq-1
         call col2(flx(1,ieq),jaco_f,nxzd)
      enddo

      if (lxd.gt.lx1) then
         do j=1,toteq-1
            call map_faced(fluxw(1,f,e,j),flx(1,j),lx1,lxd,fdim,1)
         enddo
         if(cbc(f,e,ifield).ne.'I  ') call map_faced(fluxw(1,f,e,toteq),
     >                              flx(1,toteq),lx1,lxd,fdim,1)
      else
         do j=1,toteq-1
            call copy(fluxw(1,f,e,j),flx(1,j),nxz)
         enddo
         if (cbc(f,e,ifield).ne.'I  ') call copy(fluxw(1,f,e,toteq),
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
! NOT LONG FOR THIS WORLD
!--------------------------------------------------------------------
