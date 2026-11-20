      subroutine fluxes_full_field
!-----------------------------------------------------------------------
! JH060314 First, compute face fluxes now that we have the primitive variables
! JH091514 renamed from "surface_fluxes_inviscid" since it handles all fluxes
!          that we compute from variables stored for the whole field (as
!          opposed to one element at a time).
!-----------------------------------------------------------------------
      include 'SIZE'
      include 'DG'
      include 'SOLN'
      include 'CMTDATA'
      include 'INPUT'

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*3*lfq) ! might not need ldim
! JH070214 OK getting different answers whether or not the variables are
!          declared locally or in common blocks. switching to a different
!          method of memory management that is more transparent to me.
      common /CMTSURFLX/ fatface(heresize),notyet(hdsize)
      real fatface,notyet
      integer eq
      character*32 cname
      nfq=nx1*nz1*2*ndim*nelt
      nstate = nqq
! where different things live
      iqm =1
      iqp =iqm+nstate*nfq
      iflx=iqp+nstate*nfq

      call fillq(irho,vtrans,fatface(iqm),fatface(iqp))
      call fillq(iux, vx,    fatface(iqm),fatface(iqp))
      call fillq(iuy, vy,    fatface(iqm),fatface(iqp))
      call fillq(iuz, vz,    fatface(iqm),fatface(iqp))
      call fillq(ipr, pr,    fatface(iqm),fatface(iqp))
      call fillq(ithm,t,     fatface(iqm),fatface(iqp))
      call fillq(isnd,csound,fatface(iqm),fatface(iqp))
      call fillq(iph, phig,  fatface(iqm),fatface(iqp))
      call fillq(icvf,vtrans(1,1,1,1,icv),fatface(iqm),fatface(iqp))
      call fillq(icpf,vtrans(1,1,1,1,icp),fatface(iqm),fatface(iqp))
      call fillq(imuf, vdiff(1,1,1,1,imu), fatface(iqm),fatface(iqp))
      call fillq(ikndf,vdiff(1,1,1,1,iknd),fatface(iqm),fatface(iqp))
      call fillq(ilamf,vdiff(1,1,1,1,ilam),fatface(iqm),fatface(iqp))

      i_cvars=(iu1-1)*nfq+1
      do eq=1,toteq
         call faceu(eq,fatface(i_cvars))
         i_cvars=i_cvars+nfq
      enddo

      call face_state_commo(fatface(iqm),fatface(iqp),nfq,nstate
     >                     ,dg_hndl)

      call InviscidFlux(fatface(iqm),fatface(iqp),fatface(iflx)
     >                 ,nstate,toteq)

!     call face_flux_commo(fatface(iflx),fatface(iflx),ndg_face,toteq,
!    >                     flux_hndl) ! for non-symmetric gs_op someday

      return
      end

!-----------------------------------------------------------------------

      subroutine faceu(ivar,yourface)
! get faces of conserved variables stored contiguously
      include 'SIZE'
      include 'CMTDATA'
      include 'DG'
      integer e
      real yourface(nx1,nz1,2*ldim,nelt)

      do e=1,nelt
         call full2face_cmt(1,nx1,ny1,nz1,iface_flux(1,e),
     >                      yourface(1,1,1,e),u(1,1,1,ivar,e))
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine fillq(ivar,field,qminus,yourface)
      include 'SIZE'
      include 'DG'

      integer ivar! intent(in)
      real field(nx1,ny1,nz1,nelt)! intent(in)
!     real, intent(out)qminus(7,nx1*nz1*2*ldim*nelt) ! gs_op no worky
      real qminus(nx1*nz1*2*ndim*nelt,*)! intent(out)
      real yourface(nx1,nz1,2*ndim,*)
      integer e,f

      nxz  =nx1*nz1
      nface=2*ndim

      call full2face_cmt(nelt,nx1,ny1,nz1,iface_flux,yourface,field)

      do i=1,ndg_face
         qminus(i,ivar)=yourface(i,1,1,1)
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine face_state_commo(mine,yours,nf,nstate,handle)

! JH060414 if we ever want to be more intelligent about who gets what,
!          who gives what and who does what, this is the place where all
!          that is done. At the very least, gs_op may need the transpose
!          flag set to 1. Who knows. Everybody duplicates everything for
!          now.
! JH070714 figure out gs_op_fields, many, vec, whatever (and the
!          corresponding setup) to get this done for the transposed
!          ordering of state variables. I want variable innermost, not
!          grid point.

      integer handle,nf,nstate ! intent(in)
      real yours(*),mine(*)

      ntot=nf*nstate
      call copy(yours,mine,ntot)
!-----------------------------------------------------------------------
! operation flag is second-to-last arg, an integer
!                                                1 ==> +
      call gs_op_fields(handle,yours,nf,nstate,1,1,0)
      call sub2 (yours,mine,ntot)
      return
      end

!-----------------------------------------------------------------------

      subroutine face_flux_commo(flux1,flux2,nf,neq,handle)
! JH060514 asymmetric transposed gs_op, gs_unique magic may be needed if
!          we ever decide to avoid redundancy. For now, this routine
!          doesn't need to do anything.
      integer ntot,handle
      real flux1(*),flux2(*)
! JH061814 It doesn't need to do anything, but a sanity check would be
!          wise.
      return
      end

!-------------------------------------------------------------------------------

      subroutine InviscidFlux(qminus,qplus,flux,nstate,nflux)
!-------------------------------------------------------------------------------
! JH091514 A fading copy of RFLU_ModAUSM.F90 from RocFlu
!-------------------------------------------------------------------------------

!#ifdef SPEC
!      USE ModSpecies, ONLY: t_spec_type
!#endif
      include 'SIZE'
      include 'INPUT' ! do we need this?
      include 'GEOM' ! for unx
      include 'CMTDATA' ! do we need this without outflsub?
      include 'DG'

! ==============================================================================
! Arguments
! ==============================================================================
      integer nstate,nflux
      real qminus(nx1*nz1,2*ndim,nelt,nstate),
     >     qplus(nx1*nz1,2*ndim,nelt,nstate),
     >     flux(nx1*nz1,2*ndim,nelt,nflux)

! ==============================================================================
! Locals
! ==============================================================================

      integer e,f,fdim,i,k,nxz,nface,ifield
      parameter (lfd=lxd*lzd)
! JH111815 legacy rocflu names.
!
! nx,ny,nz : outward facing unit normal components
! fs       : face speed. zero until we have moving grid
! jaco_c   : fdim-D GLL grid Jacobian
! nm       : jaco_c, fine grid
!
! State on the interior (-, "left") side of the face
! rl       : density
! ul,vl,wl : velocity
! tl       : temperature
! al       : sound speed
! pl       : pressure, then phi
! cpl      : rho*cp
! State on the exterior (+, "right") side of the face
! rr       : density
! ur,vr,wr : velocity
! tr       : temperature
! ar       : sound speed
! pr       : pressure
! cpr      : rho*cp

      COMMON /SCRNS/ nx(lfd), ny(lfd), nz(lfd), rl(lfd), ul(lfd),
     >               vl(lfd), wl(lfd), pl(lfd), tl(lfd), al(lfd),
     >               cpl(lfd),rr(lfd), ur(lfd), vr(lfd), wr(lfd),
     >               pr(lfd),tr(lfd), ar(lfd),cpr(lfd),phl(lfd),fs(lfd),
     >               jaco_f(lfd),flx(lfd,toteq),jaco_c(lx1*lz1)
      real nx, ny, nz, rl, ul, vl, wl, pl, tl, al, cpl, rr, ur, vr, wr,
     >                pr,tr, ar,cpr,phl,fs,jaco_f,flx,jaco_c

!     REAL vf(3)
      real nTol
      character*132 deathmessage
      character*3 cb

      nTol = 1.0E-14

      fdim=ndim-1
      nface = 2*ndim
      nxz   = nx1*nz1
      nxzd  = nxd*nzd
      ifield= 1

!     if (outflsub)then
!        call maxMachnumber
!     endif
      do e=1,nelt
      do f=1,nface

         cb=cbc(f,e,ifield)
         if (cb.ne.'E  '.and.cb.ne.'P  ') then ! cbc bndy

!-----------------------------------------------------------------------
! compute flux for weakly-enforced boundary condition
!-----------------------------------------------------------------------

            do j=1,nstate
               do i=1,nxz
                  if (abs(qplus(i,f,e,j)) .gt. ntol) then
                  write(6,*) nid,j,i,qplus(i,f,e,j),qminus(i,f,e,j),cb,
     > nstate
                  write(deathmessage,*)  'GS hit a bndy,f,e=',f,e,'$'
! Make sure you are not abusing this error handler
                  call exitti(deathmessage,f)
                  endif
               enddo
            enddo
! JH031315 flux added to argument list. BC routines preserve qminus for
!          obvious reasons and fill qplus with good stuff for everybody:
!          imposed states for Dirichlet conditions, and important things
!          for viscous numerical fluxes.
! JH060215 added SYM bc. Just use it as a slip wall hopefully.
            if (cb.eq.'v  ' .or. cb .eq. 'V  ') then
              call inflow(nstate,f,e,qminus,qplus,flux)
            elseif (cb.eq.'O  ') then
              call outflow(nstate,f,e,qminus,qplus,flux)
            elseif (cb .eq. 'W  ' .or. cb .eq.'I  '.or.cb .eq.'SYM')then
              call wallbc(nstate,f,e,qminus,qplus,flux)
            endif 

         else ! cbc(f,e,ifield) == 'E  ' or 'P  ' below; interior face

! JH111715 now with dealiased surface integrals. I am too lazy to write
!          something better

            if (nxd.gt.nx1) then
               call map_faced(nx,unx(1,1,f,e),nx1,nxd,fdim,0)
               call map_faced(ny,uny(1,1,f,e),nx1,nxd,fdim,0)
               call map_faced(nz,unz(1,1,f,e),nx1,nxd,fdim,0)

               call map_faced(rl,qminus(1,f,e,irho),nx1,nxd,fdim,0)
               call map_faced(ul,qminus(1,f,e,iux),nx1,nxd,fdim,0)
               call map_faced(vl,qminus(1,f,e,iuy),nx1,nxd,fdim,0)
               call map_faced(wl,qminus(1,f,e,iuz),nx1,nxd,fdim,0)
               call map_faced(pl,qminus(1,f,e,ipr),nx1,nxd,fdim,0)
               call map_faced(tl,qminus(1,f,e,ithm),nx1,nxd,fdim,0)
               call map_faced(al,qminus(1,f,e,isnd),nx1,nxd,fdim,0)
               call map_faced(cpl,qminus(1,f,e,icpf),nx1,nxd,fdim,0)

               call map_faced(rr,qplus(1,f,e,irho),nx1,nxd,fdim,0)
               call map_faced(ur,qplus(1,f,e,iux),nx1,nxd,fdim,0)
               call map_faced(vr,qplus(1,f,e,iuy),nx1,nxd,fdim,0)
               call map_faced(wr,qplus(1,f,e,iuz),nx1,nxd,fdim,0)
               call map_faced(pr,qplus(1,f,e,ipr),nx1,nxd,fdim,0)
               call map_faced(tr,qplus(1,f,e,ithm),nx1,nxd,fdim,0)
               call map_faced(ar,qplus(1,f,e,isnd),nx1,nxd,fdim,0)
               call map_faced(cpr,qplus(1,f,e,icpf),nx1,nxd,fdim,0)

               call map_faced(phl,qminus(1,f,e,iph),nx1,nxd,fdim,0)

               call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz)
               call map_faced(jaco_f,jaco_c,nx1,nxd,fdim,0) 
               call col2(jaco_f,wghtf,nxzd)
            else

               call copy(nx,unx(1,1,f,e),nxz)
               call copy(ny,uny(1,1,f,e),nxz)
               call copy(nz,unz(1,1,f,e),nxz)

               call copy(rl,qminus(1,f,e,irho),nxz)
               call copy(ul,qminus(1,f,e,iux),nxz)
               call copy(vl,qminus(1,f,e,iuy),nxz)
               call copy(wl,qminus(1,f,e,iuz),nxz)
               call copy(pl,qminus(1,f,e,ipr),nxz)
               call copy(tl,qminus(1,f,e,ithm),nxz)
               call copy(al,qminus(1,f,e,isnd),nxz)
               call copy(cpl,qminus(1,f,e,icpf),nxz)

               call copy(rr,qplus(1,f,e,irho),nxz)
               call copy(ur,qplus(1,f,e,iux),nxz)
               call copy(vr,qplus(1,f,e,iuy),nxz)
               call copy(wr,qplus(1,f,e,iuz),nxz)
               call copy(pr,qplus(1,f,e,ipr),nxz)
               call copy(tr,qplus(1,f,e,ithm),nxz)
               call copy(ar,qplus(1,f,e,isnd),nxz)
               call copy(cpr,qplus(1,f,e,icpf),nxz)

               call copy(phl,qminus(1,f,e,iph),nxz)

               call copy(jaco_f,area(1,1,f,e),nxz) 
            endif
            call rzero(fs,nxzd) ! moving grid stuff later

            call AUSM_FluxFunction(nxzd,nx,ny,nz,jaco_f,fs,rl,ul,vl,wl,
     >                        pl,al,tl,rr,ur,vr,wr,pr,ar,tr,flx,cpl,cpr)

            do j=1,toteq
               call col2(flx(1,j),phl,nxzd)
            enddo

            if (nxd.gt.nx1) then
               do j=1,toteq
                  call map_faced(flux(1,f,e,j),flx(1,j),nx1,nxd,fdim,1)
               enddo
            else
               do j=1,toteq
                  call copy(flux(1,f,e,j),flx(1,j),nxz)
               enddo
            endif

         endif ! cbc(f,e,ifield)
      enddo
      enddo

      end

!-----------------------------------------------------------------------

      subroutine surface_integral_full(vol,flux)
! Integrate surface fluxes for an entire field. Add contribution of flux
! to volume according to add_face2full_cmt
      include 'SIZE'
      include 'GEOM'
      include 'DG'
      include 'CMTDATA'
      real vol(nx1*ny1*nz1*nelt),flux(*)
      integer e,f

! weak form until we get the time loop rewritten
!     onem=-1.0
!     ntot=nx1*nz1*2*ndim*nelt
!     call cmult(flux,onem,ntot)
! weak form until we get the time loop rewritten
      call add_face2full_cmt(nelt,nx1,ny1,nz1,iface_flux,vol,flux)

      return
      end

!-------------------------------------------------------------------------------

      subroutine diffh2graduf(e,eq,graduf)
! peels off diffusiveH into contiguous face storage via restriction operator R
! for now, stores {{gradU}} for igu
      include  'SIZE'
      include  'DG' ! iface
      include  'CMTDATA'
      include  'GEOM'
      integer e,eq
      real graduf(nx1*nz1*2*ndim,nelt,toteq)
      common /scrns/ hface(lx1*lz1,2*ldim)
     >              ,normal(lx1*ly1,2*ldim)
      real hface, normal

      integer f

      nf    = nx1*nz1*2*ndim*nelt
      nfaces=2*ndim
      nxz   =nx1*nz1
      nxzf  =nxz*nfaces
      nxyz  = nx1*ny1*nz1

      call rzero(graduf(1,e,eq),nxzf) !   . dot nhat -> overwrites beginning of flxscr
      do j =1,ndim
         if (j .eq. 1) call copy(normal,unx(1,1,1,e),nxzf)
         if (j .eq. 2) call copy(normal,uny(1,1,1,e),nxzf)
         if (j .eq. 3) call copy(normal,unz(1,1,1,e),nxzf)
         call full2face_cmt(1,nx1,ny1,nz1,iface_flux,hface,diffh(1,j)) 
         call addcol3(graduf(1,e,eq),hface,normal,nxzf)
      enddo
      call col2(graduf(1,e,eq),area(1,1,1,e),nxzf)

      return
      end

!-----------------------------------------------------------------------

      subroutine igu_cmt(flxscr,gdudxk)
! gets central-flux contribution to interior penalty numerical flux
! Hij^{d*}
      include 'SIZE'
      include 'CMTDATA'
      include 'DG'

      real gdudxk(nx1*nz1*2*ndim,nelt,toteq)
      real flxscr(nx1*nz1*2*ndim*nelt,toteq)
      real const
      integer e,eq,f

      nxz = nx1*nz1
      nfaces=2*ndim
      nxzf=nxz*nfaces
      nfq =nx1*nz1*nfaces*nelt
      ntot=nfq*toteq

      call copy (flxscr,gdudxk,ntot) ! save gradU.n
      const = 0.5
      call cmult(gdudxk,const,ntot)
!-----------------------------------------------------------------------
! supa huge gs_op to get {{gdu}}
! operation flag is second-to-last arg, an integer
!                                                   1 ==> +
      call gs_op_fields(dg_hndl,gdudxk,nfq,toteq,1,1,0)
!-----------------------------------------------------------------------
      call bcflux(gdudxk)
      call sub2  (flxscr,gdudxk,ntot) ! overwrite flxscr with a- - {{a}}
! I wish it were that easy, but [v] changes character on dirichlet boundaries
      call igu_dirichlet(flxscr,gdudxk)
      call chsign(flxscr,ntot)

      return
      end

!-----------------------------------------------------------------------

      subroutine igu_dirichlet(flux,fminus)
      include 'SIZE'
      include 'TOTAL'
      integer e,eq,f
      real flux(nx1*nz1,2*ndim,nelt,toteq)
      real fminus(nx1*nz1,2*ndim,nelt,toteq)
      character*3 cb2

      nxz=nx1*nz1
      nfaces=2*ndim

      ifield=1
      do e=1,nelt
         do f=1,nfaces
            cb2=cbc(f, e, ifield)
            if (cb2 .eq. 'W  ') then
               do eq=1,toteq
                  call copy(flux(1,f,e,eq),fminus(1,f,e,eq),nxz)
               enddo
            endif
         enddo
      enddo

      return
      end
