c-----------------------------------------------------------------------
      subroutine fluidp (igeom)
c
c     Driver for perturbation velocity
c
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'TURBO'
      include 'SOLN'

      do jp=1,npert

         if (nio.eq.0.and.igeom.eq.2) write(6,1) istep,time,jp
   1     format(i9,1pe14.7,' Perturbation Solve:',i5)

         call perturbv (igeom)

      enddo

      jp=0   ! set jp to zero, for baseline flow

      return
      end
c-----------------------------------------------------------------------
      subroutine perturbv (igeom)
c
c
c     Solve the convection-diffusion equation for the perturbation field, 
c     with projection onto a div-free space.
c
c
      include 'SIZE'
      include 'INPUT'
      include 'EIGEN'
      include 'SOLN'
      include 'TSTEP'
      include 'MASS'
C
      COMMON /SCRNS/  RESV1 (LX1,LY1,LZ1,LELV)
     $ ,              RESV2 (LX1,LY1,LZ1,LELV)
     $ ,              RESV3 (LX1,LY1,LZ1,LELV)
     $ ,              DV1   (LX1,LY1,LZ1,LELV)
     $ ,              DV2   (LX1,LY1,LZ1,LELV)
     $ ,              DV3   (LX1,LY1,LZ1,LELV)
      COMMON /SCRVH/  H1    (LX1,LY1,LZ1,LELV)
     $ ,              H2    (LX1,LY1,LZ1,LELV)
c
      ifield = 1
c
      if (igeom.eq.1) then
c
c        Old geometry, old velocity
c
         call makefp
         call lagfieldp
c
      else
c
c        New geometry, new velocity
c
         intype = -1
         call sethlm   (h1,h2,intype)
         call cresvipp (resv1,resv2,resv3,h1,h2)
         call ophinv   (dv1,dv2,dv3,resv1,resv2,resv3,h1,h2,tolhv,nmxh)
         call opadd2   (vxp(1,jp),vyp(1,jp),vzp(1,jp),dv1,dv2,dv3)
         call incomprp (vxp(1,jp),vyp(1,jp),vzp(1,jp),prp(1,jp))
c
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lagfieldp
c
c     Keep old Vp-field(s)
c
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
c
      do ilag=nbdinp-1,2,-1
         call opcopy
     $     (vxlagp(1,ilag  ,jp),vylagp(1,ilag  ,jp),vzlagp(1,ilag  ,jp)
     $     ,vxlagp(1,ilag-1,jp),vylagp(1,ilag-1,jp),vzlagp(1,ilag-1,jp))
      enddo
      call opcopy(vxlagp(1,1,jp),vylagp(1,1,jp),vzlagp(1,1,jp)
     $           ,vxp   (1,jp)  ,vyp   (1,jp)  ,vzp   (1,jp) )
c
      return
      end
c-----------------------------------------------------------------------
      subroutine makefp
c
c     Make rhs for velocity perturbation equation
c
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'INPUT'
      include 'TSTEP'
      include 'ADJOINT'
 
                                              call makeufp
      if (ifnav.and.(.not.ifchar).and.(.not.ifadj))call advabp
      if (ifnav.and.(.not.ifchar).and.(     ifadj))call advabp_adjoint
      if (iftran)                              call makextp
                                               call makebdfp
c
      return
      end
c--------------------------------------------------------------------
      subroutine makeufp
c
c     Compute and add: (1) user specified forcing function (FX,FY,FZ)
c
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      time = time-dt
      call nekuf   (bfxp(1,jp),bfyp(1,jp),bfzp(1,jp))
      call opcolv2 (bfxp(1,jp),bfyp(1,jp),bfzp(1,jp)
     $                              ,vtrans(1,1,1,1,ifield),bm1)
      time = time+dt
c
      return
      end
c--------------------------------------------------------------------
      subroutine advabp
C
C     Eulerian scheme, add convection term to forcing function
C     at current time step.
C
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      COMMON /SCRNS/ TA1 (LX1*LY1*LZ1*LELV)
     $ ,             TA2 (LX1*LY1*LZ1*LELV)
     $ ,             TA3 (LX1*LY1*LZ1*LELV)
     $ ,             TB1 (LX1*LY1*LZ1*LELV)
     $ ,             TB2 (LX1*LY1*LZ1*LELV)
     $ ,             TB3 (LX1*LY1*LZ1*LELV)
C
      ntot1 = nx1*ny1*nz1*nelv
      ntot2 = nx2*ny2*nz2*nelv
c
      if (if3d) then
         call opcopy  (tb1,tb2,tb3,vx,vy,vz)                   ! Save velocity
         call opcopy  (vx,vy,vz,vxp(1,jp),vyp(1,jp),vzp(1,jp)) ! U <-- dU
         call convop  (ta1,tb1)                                ! du.grad U
         call convop  (ta2,tb2)
         call convop  (ta3,tb3)
         call opcopy  (vx,vy,vz,tb1,tb2,tb3)  ! Restore velocity
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
            bfzp(i,jp) = bfzp(i,jp)-tmp*ta3(i)
         enddo
c
         call convop  (ta1,vxp(1,jp))       !  U.grad dU
         call convop  (ta2,vyp(1,jp))
         call convop  (ta3,vzp(1,jp))
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
            bfzp(i,jp) = bfzp(i,jp)-tmp*ta3(i)
         enddo
c
      else  ! 2D
c
         call opcopy  (tb1,tb2,tb3,vx,vy,vz)                   ! Save velocity
         call opcopy  (vx,vy,vz,vxp(1,jp),vyp(1,jp),vzp(1,jp)) ! U <-- dU
         call convop  (ta1,tb1)                                ! du.grad U
         call convop  (ta2,tb2)
         call opcopy  (vx,vy,vz,tb1,tb2,tb3)  ! Restore velocity
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
         enddo
c
         call convop  (ta1,vxp(1,jp))       !  U.grad dU
         call convop  (ta2,vyp(1,jp))
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
         enddo
c
      endif
c
      return
      end
c--------------------------------------------------------------------
      subroutine advabp_adjoint
C
C     Eulerian scheme, add convection term to forcing function
C     at current time step for backward part of adjoint: 
C     Convective term is now (U.Grad)u - (Grad U)^T .u
C     instead of (u.Grad)U + (U.Grad)u in above subroutine advabp
C
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
      include 'GEOM'
      include 'ADJOINT'
C
      COMMON /SCRNS/ TA1 (LX1*LY1*LZ1*LELV)
     $ ,             TA2 (LX1*LY1*LZ1*LELV)
     $ ,             TA3 (LX1*LY1*LZ1*LELV)
     $ ,             TB1 (LX1*LY1*LZ1*LELV)
     $ ,             TB2 (LX1*LY1*LZ1*LELV)
     $ ,             TB3 (LX1*LY1*LZ1*LELV)


      real fact,factx,facty
C
      ntot1 = nx1*ny1*nz1*nelv
      ntot2 = nx2*ny2*nz2*nelv   !dimensionn arrays 
      NTOT      = NX1*NY1*NZ1*NELT

c
      if (if3d) then
         call opcopy  (tb1,tb2,tb3,vx,vy,vz)                   ! Save velocity
         call opcopy  (vx,vy,vz,vxp(1,jp),vyp(1,jp),vzp(1,jp)) ! U <-- u
c     
         call convop_adj  (ta1,ta2,ta3,tb1,tb2,tb3,vx,vy,vz) ! u.grad U^T

         call opcopy  (vx,vy,vz,tb1,tb2,tb3)  ! Restore velocity
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
            bfzp(i,jp) = bfzp(i,jp)-tmp*ta3(i)
         enddo
c
c               
c
         call convop  (ta1,vxp(1,jp))       !  U.grad u
         call convop  (ta2,vyp(1,jp))
         call convop  (ta3,vzp(1,jp))
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)+tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)+tmp*ta2(i)
            bfzp(i,jp) = bfzp(i,jp)+tmp*ta3(i)
         enddo
c
         if (ifheat) then       ! dt.(grad T)
c                               ! term coming from the temperature convection
            call opcolv3 (ta1,ta2,ta3,dTdx,dTdy,dTdz,tp)
c
            do i=1,ntot1
               tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,2)
               bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
               bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
               bfzp(i,jp) = bfzp(i,jp)-tmp*ta3(i)
            enddo
         endif
c
      else  ! 2D

         call opcopy  (tb1,tb2,tb3,vx,vy,vz)                   ! Save velocity
         call opcopy  (vx,vy,vz,vxp(1,jp),vyp(1,jp),vzp(1,jp)) 

         call convop_adj  (ta1,ta2,ta3,tb1,tb2,tb3,vx,vy,vz) ! u.((grad U)^T)

         call opcopy  (vx,vy,vz,tb1,tb2,tb3) ! Restore velocity

         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
         enddo

         call convop  (ta1,vxp(1,jp))       !  U.grad u
         call convop  (ta2,vyp(1,jp))
c
         do i=1,ntot1
            tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,ifield)
            bfxp(i,jp) = bfxp(i,jp)+tmp*ta1(i)
            bfyp(i,jp) = bfyp(i,jp)+tmp*ta2(i)
         enddo

         if (ifheat) then       ! dt.(grad T)^T
                                ! term coming from the temperature convection
            call opcolv3 (ta1,ta2,ta3,dTdx,dTdy,dTdz,tp)
c
            do i=1,ntot1
               tmp = bm1(i,1,1,1)*vtrans(i,1,1,1,2)
               bfxp(i,jp) = bfxp(i,jp)-tmp*ta1(i)
               bfyp(i,jp) = bfyp(i,jp)-tmp*ta2(i)
            enddo
         endif
         
      endif
c
      return
      end
c--------------------------------------------------------------------
      subroutine convop_adj  (bdux,bduy,bduz,udx,udy,udz,cx,cy,cz)

      include 'SIZE'
      include 'TOTAL'

      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)
      common /scrcv/ fx(ltd),fy(ltd),fz(ltd)
     $     , uf1(ltd),uf2(ltd),uf3(ltd),uf4(ltd),uf5(ltd),uf6(ltd)
      real urx(ltd),usx(ltd),utx(ltd)
      real ury(ltd),usy(ltd),uty(ltd)
      real urz(ltd),usz(ltd),utz(ltd)
      real bdux(1),bduy(1),bduz(1),u(1),cx(1),cy(1),cz(1)
      real udx(1),udy(1),udz(1)
      logical ifuf,ifcf            ! u and/or c already on fine mesh?
      integer e
      real bdrx(1), bdry(1),bdrz (1)

      call set_dealias_rx
      nxyz1 = nx1*ny1*nz1
c     AM DING DING 
      nxyzd = lxd*lyd*lzd
      nxyzu = nxyz1
      nxyzc = nxyz1
      ntot1=nx1*ny1*nz1*nelv
      ic = 1                    ! pointer to vector field C
      iu = 1                    ! pointer to scalar field u
      ib = 1                    ! pointer to scalar field Bdu
      if(if3d)then
         do e=1,nelv
                                ! map coarse velocity to fine mesh (C-->F)
            call intp_rstd(fx,cx(ic),nx1,nxd,if3d,0) ! 0 --> forward
            call intp_rstd(fy,cy(ic),nx1,nxd,if3d,0) 
            call intp_rstd(fz,cz(ic),nx1,nxd,if3d,0) 
               
            call intp_rstd(uf1,udx(iu),nx1,nxd,if3d,0) ! 0 --> forward
            call grad_rst(urx,usx,utx,uf1,nxd,if3d)
            
            call intp_rstd(uf2,udy(iu),nx1,nxd,if3d,0) 
            call grad_rst(ury,usy,uty,uf2,nxd,if3d)
            
            call intp_rstd(uf3,udz(iu),nx1,nxd,if3d,0) 
            call grad_rst(urz,usz,utz,uf3,nxd,if3d)
            
            do i=1,nxyzd        ! mass matrix included, per DFM (4.8.5)
               uf4(i)=fx(i)*(rx(i,1,e)*urx(i)+rx(i,4,e)*usx(i)
     $              +rx(i,7,e)*utx(i))+
     $              fy(i)*(rx(i,1,e)*ury(i)+rx(i,4,e)*usy(i)
     $              +rx(i,7,e)*uty(i))+
     $              fz(i)*(rx(i,1,e)*urz(i)+rx(i,4,e)*usz(i)
     $              +rx(i,7,e)*utz(i))
               uf5(i)=fx(i)*(rx(i,2,e)*urx(i)+rx(i,5,e)*usx(i)
     $              +rx(i,8,e)*utx(i))+
     $              fy(i)*(rx(i,2,e)*ury(i)+rx(i,5,e)*usy(i)
     $              +rx(i,8,e)*uty(i))+
     $              fz(i)*(rx(i,2,e)*urz(i)+rx(i,5,e)*usz(i)
     $              +rx(i,8,e)*utz(i))
               uf6(i)=fx(i)*(rx(i,3,e)*urx(i)+rx(i,6,e)*usx(i)
     $              +rx(i,9,e)*utx(i))+
     $              fy(i)*(rx(i,3,e)*ury(i)+rx(i,6,e)*usy(i)
     $              +rx(i,9,e)*uty(i))+
     $              fz(i)*(rx(i,3,e)*urz(i)+rx(i,6,e)*usz(i)
     $              +rx(i,9,e)*utz(i))
            enddo

            call intp_rstd(bdux(ib),uf4,nx1,nxd,if3d,1) ! Project back to coarse
            call intp_rstd(bduy(ib),uf5,nx1,nxd,if3d,1)
            call intp_rstd(bduz(ib),uf6,nx1,nxd,if3d,1)

            ic = ic + nxyzc
            iu = iu + nxyzu
            ib = ib + nxyz1
         enddo
         call invcol2     (bdux,bm1,ntot1) ! local mass inverse
         call invcol2     (bduy,bm1,ntot1) ! local mass inverse
         call invcol2     (bduz,bm1,ntot1) ! local mass inverse
      else
         do e=1,nelv

                               ! map coarse velocity to fine mesh (C-->F)
            call intp_rstd(fx,cx(ic),nx1,nxd,if3d,0) ! 0 --> forward
            call intp_rstd(fy,cy(ic),nx1,nxd,if3d,0) 

            call intp_rstd(uf1,udx(iu),nx1,nxd,if3d,0) 
            call grad_rst(urx,usx,utx,uf1,nxd,if3d)

            call intp_rstd(uf2,udy(iu),nx1,nxd,if3d,0) 
            call grad_rst(ury,usy,uty,uf2,nxd,if3d)

            do i=1,nxyzd       
               uf4(i) = fx(i)*(rx(i,1,e)*urx(i)+rx(i,3,e)*usx(i))+
     $              fy(i)*(rx(i,1,e)*ury(i)+rx(i,3,e)*usy(i))
               uf5(i) = fx(i)*(rx(i,2,e)*urx(i)+rx(i,4,e)*usx(i))+
     $              fy(i)*(rx(i,2,e)*ury(i)+rx(i,4,e)*usy(i))
            enddo

            call intp_rstd(bdux(ib),uf4,nx1,nxd,if3d,1)
            call intp_rstd(bduy(ib),uf5,nx1,nxd,if3d,1)

            ic = ic + nxyzc
            iu = iu + nxyzu
            ib = ib + nxyz1
         end do
         call invcol2     (bdux,bm1,ntot1) ! local mass inverse
         call invcol2     (bduy,bm1,ntot1) ! local mass inverse
      end if


      return
      end
c--------------------------------------------------------------------
      subroutine makextp
c
c     Add extrapolation terms to perturbation source terms
c
c     (nek5 equivalent for velocity is "makeabf")
c
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      common /scrns/ ta1 (lx1,ly1,lz1,lelv)
     $ ,             ta2 (lx1,ly1,lz1,lelv)
     $ ,             ta3 (lx1,ly1,lz1,lelv)
c
      ntot1 = nx1*ny1*nz1*nelv
c
      ab0 = ab(1)
      ab1 = ab(2)
      ab2 = ab(3)
      call add3s2 (ta1,exx1p(1,jp),exx2p(1,jp),ab1,ab2,ntot1)
      call add3s2 (ta2,exy1p(1,jp),exy2p(1,jp),ab1,ab2,ntot1)
      call copy   (exx2p(1,jp),exx1p(1,jp),ntot1)
      call copy   (exy2p(1,jp),exy1p(1,jp),ntot1)
      call copy   (exx1p(1,jp),bfxp (1,jp),ntot1)
      call copy   (exy1p(1,jp),bfyp (1,jp),ntot1)
      call add2s1 (bfxp(1,jp),ta1,ab0,ntot1)
      call add2s1 (bfyp(1,jp),ta2,ab0,ntot1)
      if (if3d) then
         call add3s2 (ta3,exz1p(1,jp),exz2p(1,jp),ab1,ab2,ntot1)
         call copy   (exz2p(1,jp),exz1p(1,jp),ntot1)
         call copy   (exz1p(1,jp),bfzp (1,jp),ntot1)
         call add2s1 (bfzp(1,jp),ta3,ab0,ntot1)
      endif
c
      return
      end
c--------------------------------------------------------------------
      subroutine makebdfp
C
C     Add contributions to perturbation source from lagged BD terms.
C
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
C
      COMMON /SCRNS/ TA1(LX1,LY1,LZ1,LELV)
     $ ,             TA2(LX1,LY1,LZ1,LELV)
     $ ,             TA3(LX1,LY1,LZ1,LELV)
     $ ,             TB1(LX1,LY1,LZ1,LELV)
     $ ,             TB2(LX1,LY1,LZ1,LELV)
     $ ,             TB3(LX1,LY1,LZ1,LELV)
     $ ,             H2 (LX1,LY1,LZ1,LELV)
C
      ntot1 = nx1*ny1*nz1*nelv
      const = 1./dt
      call cmult2(h2,vtrans(1,1,1,1,ifield),const,ntot1)
      call opcolv3c (tb1,tb2,tb3
     $              ,vxp(1,jp),vyp(1,jp),vzp(1,jp),bm1,bd(2))
C
      do ilag=2,nbd
         if (ifgeom) then
            call opcolv3c(ta1,ta2,ta3,vxlagp(1,ilag-1,jp),
     $                                vylagp(1,ilag-1,jp),
     $                                vzlagp(1,ilag-1,jp),
     $                                bm1lag(1,1,1,1,ilag-1),bd(ilag+1))
         else
            call opcolv3c(ta1,ta2,ta3,vxlagp(1,ilag-1,jp),
     $                                vylagp(1,ilag-1,jp),
     $                                vzlagp(1,ilag-1,jp),
     $                                bm1                   ,bd(ilag+1))
         endif
         call opadd2  (tb1,tb2,tb3,ta1,ta2,ta3)
      enddo
      call opadd2col (bfxp(1,jp),bfyp(1,jp),bfzp(1,jp),tb1,tb2,tb3,h2)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cresvipp(resv1,resv2,resv3,h1,h2)
c
c     Account for inhomogeneous Dirichlet boundary contributions 
c     in rhs of perturbation eqn.
c                                               n
c     Also, subtract off best estimate of grad p
c
      include 'SIZE'
      include 'TOTAL'
      real           resv1 (lx1,ly1,lz1,1)
      real           resv2 (lx1,ly1,lz1,1)
      real           resv3 (lx1,ly1,lz1,1)
      real           h1    (lx1,ly1,lz1,1)
      real           h2    (lx1,ly1,lz1,1)
      common /scruz/ w1    (lx1,ly1,lz1,lelv)
     $ ,             w2    (lx1,ly1,lz1,lelv)
     $ ,             w3    (lx1,ly1,lz1,lelv)
     $ ,             prextr(lx2,ly2,lz2,lelv)
c
      ntot1 = nx1*ny1*nz1*nelv
      ntot2 = nx2*ny2*nz2*nelv
c
      call bcdirvc (vxp(1,jp),vyp(1,jp),vzp(1,jp)
     $             ,v1mask,v2mask,v3mask)
      call extrapprp(prextr)
      call opgradt(resv1,resv2,resv3,prp(1,jp))
      call opadd2(resv1,resv2,resv3,bfxp(1,jp),bfyp(1,jp),bfzp(1,jp))
      call ophx  (w1,w2,w3,vxp(1,jp),vyp(1,jp),vzp(1,jp),h1,h2)
      call opsub2(resv1,resv2,resv3,w1,w2,w3)
c
      return
      end
c--------------------------------------------------------------------
      subroutine heatp (igeom)
C
C     Driver for temperature or passive scalar.
C
C     Current version:
C     (1) Varaiable properties.
C     (2) Implicit time stepping.
C     (3) User specified tolerance for the Helmholtz solver
C         (not based on eigenvalues).
C     (4) A passive scalar can be defined on either the 
C         temperatur or the velocity mesh.
C     (5) A passive scalar has its own multiplicity (B.C.).  
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'TURBO'
      INCLUDE 'SOLN'
C
      do jp=1,npert
         do ifield=2,nfield
            INTYPE        = -1
            IF (.NOT.IFTMSH(IFIELD)) IMESH = 1
            IF (     IFTMSH(IFIELD)) IMESH = 2
            CALL UNORM
            CALL SETTOLT
            CALL CDSCALP (IGEOM)
         enddo
      enddo
c
      jp=0   ! set jp to zero, for baseline flow
c
      return
      end
C
C-----------------------------------------------------------------------
      subroutine cdscalp (igeom)
C-----------------------------------------------------------------------
C
C     Solve the convection-diffusion equation for passive scalar IPSCAL
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'MVGEOM'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      LOGICAL          IFCONV
C
      COMMON /SCRNS/ TA(LX1,LY1,LZ1,LELT)
     $              ,TB(LX1,LY1,LZ1,LELT)
      COMMON /SCRVH/ H1(LX1,LY1,LZ1,LELT)
     $              ,H2(LX1,LY1,LZ1,LELT)
c
      include 'ORTHOT'
      napproxt(1) = laxtt
C
      IF (IGEOM.EQ.1) THEN
C
C        Old geometry
C
         CALL MAKEQP
         CALL LAGSCALP
C
      ELSE
C
         IF (IFPRINT) THEN
            IF (IFIELD.EQ.2.AND.NIO.EQ.0) THEN
               WRITE (6,*) ' Temperature/Passive scalar solution'
            ENDIF
         ENDIF
         if1=ifield-1
         write(name4t,1) if1
    1    format('TEM',i1)
C
C        New geometry
C
         NEL    = NELFLD(IFIELD)
         NTOT   = NX1*NY1*NZ1*NEL
C
         INTYPE = 0
         IF (IFTRAN) INTYPE = -1
         CALL SETHLM  (H1,H2,INTYPE)
         CALL BCNEUSC (TA,-1)
         CALL ADD2    (H2,TA,NTOT)
         CALL BCDIRSC (TP(1,IFIELD-1,jp))
         CALL AXHELM  (TA,TP (1,IFIELD-1,jp),H1,H2,IMESH,1)
         CALL SUB3    (TB,BQP(1,IFIELD-1,jp),TA,NTOT)
         CALL BCNEUSC (TA,1)
         CALL ADD2    (TB,TA,NTOT)
c
         CALL HMHOLTZ (name4t,TA,TB,H1,H2
     $                 ,TMASK(1,1,1,1,IFIELD-1)
     $                 ,TMULT(1,1,1,1,IFIELD-1)
     $                 ,IMESH,TOLHT(IFIELD),NMXH,1)
c
c        call hsolve  (name4t,TA,TB,H1,H2 
c    $                 ,TMASK(1,1,1,1,IFIELD-1)
c    $                 ,TMULT(1,1,1,1,IFIELD-1)
c    $                 ,IMESH,TOLHT(IFIELD),NMXH,1
c    $                 ,approxt,napproxt,bintm1)
c
         CALL ADD2    (TP(1,IFIELD-1,jp),TA,NTOT)
C
         CALL BCNEUSC (TA,1)
         CALL ADD2 (BQP(1,IFIELD-1,jp),TA,NTOT)
C
      ENDIF
C
      return
      end
C
      subroutine makeqp
C----------------------------------------------------------------------
C
C     Generate forcing function for the solution of a passive scalar.
C     !! NOTE: Do not change the content of the array BQ until the current
C              time step is completed 
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      include 'SOLN'
      LOGICAL  IFTURB
C
                                                       CALL MAKEUQP
      IF (IFADVC(IFIELD).AND.(.NOT.IFCHAR))            CALL CONVABP
      IF (IFTRAN)                                      CALL MAKEABQP
      IF ((IFTRAN.AND..NOT.IFCHAR).OR.
     $    (IFTRAN.AND..NOT.IFADVC(IFIELD).AND.IFCHAR)) CALL MAKEBDQP
c     IF (IFADVC(IFIELD).AND.IFCHAR)                   CALL CONVCHP
      IF (IFADVC(IFIELD).AND.IFCHAR) write(6,*) 'no convchp'
      IF (IFADVC(IFIELD).AND.IFCHAR) call exitt
c
      return
      end
C
      subroutine makeuqp
C---------------------------------------------------------------------
C
C     Fill up user defined forcing function and collocate will the
C     mass matrix on the Gauss-Lobatto mesh.
C
C---------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'MASS'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      NTOT = NX1*NY1*NZ1*NELFLD(IFIELD)
C
      time = time-dt                           ! time is tn
c
      call rzero   ( bqp(1,ifield-1,jp) ,    ntot)
      call setqvol ( bqp(1,ifield-1,jp)          )
      call col2    ( bqp(1,ifield-1,jp) ,bm1,ntot)
c
      time = time+dt                           ! restore time
C
      return
      end
C
      subroutine convabp
C---------------------------------------------------------------
C
C     Eulerian scheme, add convection term to forcing function 
C     at current time step.
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'ADJOINT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
c
      common /scruz/ ta (lx1,ly1,lz1,lelt)
     $             , ua (lx1,ly1,lz1,lelt)
     $             , ub (lx1,ly1,lz1,lelt)
     $             , uc (lx1,ly1,lz1,lelt)
      real coeff
c
      nel = nelfld(ifield)
      ntot1 = nx1*ny1*nz1*nel
c
      if (.not.ifadj) then
         call opcopy(ua,ub,uc,vx,vy,vz)
         call opcopy(vx,vy,vz,vxp(1,jp),vyp(1,jp),vyp(1,jp))
         call convop(ta,t(1,1,1,1,ifield-1)) ! dU.grad T
         call opcopy(vx,vy,vz,ua,ub,uc)
      endif
c
      call convop  (ua,tp(1,ifield-1,jp)) ! U.grad dT
c
      if (.not.ifadj) then
         call add2 (ta,ua,ntot1)          ! U.grad dT + dU.grad T
      else
         call copy (ta,ua,ntot1)
         coeff=-1.0
         call cmult(ta,coeff,ntot1)       ! -U.grad dT
                                          ! the second term depends on the buoyancy
      endif
c
      call col2    (ta,vtrans(1,1,1,1,ifield),ntot1) !vtrans (U.grad dT + dU.grad T)
      call subcol3 (bqp(1,ifield-1,jp),bm1,ta,ntot1)
c
      return
      end
C
      subroutine makeabqp
C-----------------------------------------------------------------------
C
C     Sum up contributions to 3rd order Adams-Bashforth scheme.
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      COMMON /SCRUZ/ TA (LX1,LY1,LZ1,LELT)
C
      AB0   = AB(1)
      AB1   = AB(2)
      AB2   = AB(3)
      NEL   = NELFLD(IFIELD)
      NTOT1 = NX1*NY1*NZ1*NEL
C
      CALL ADD3S2 (TA,VGRADT1P(1,IFIELD-1,jp),
     $                VGRADT2P(1,IFIELD-1,jp),AB1,AB2,NTOT1)
      CALL COPY   (   VGRADT2P(1,IFIELD-1,jp),
     $                VGRADT1P(1,IFIELD-1,jp),NTOT1)
      CALL COPY   (   VGRADT1P(1,IFIELD-1,jp),
     $                     BQP(1,IFIELD-1,jp),NTOT1)
      CALL ADD2S1 (BQP(1,IFIELD-1,jp),TA,AB0,NTOT1)
C
      return
      end
C
      subroutine makebdqp
C-----------------------------------------------------------------------
C
C     Add contributions to Q from lagged BD terms.
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
C
      COMMON /SCRNS/ TA (LX1,LY1,LZ1,LELT)
     $ ,             TB (LX1,LY1,LZ1,LELT)
     $ ,             H2 (LX1,LY1,LZ1,LELT)
C
      NEL   = NELFLD(IFIELD)
      NTOT1 = NX1*NY1*NZ1*NEL
      CONST = 1./DT
      CALL COPY  (H2,VTRANS(1,1,1,1,IFIELD),NTOT1)
      CALL CMULT (H2,CONST,NTOT1)
C
      CALL COL3  (TB,BM1,TP(1,IFIELD-1,jp),NTOT1)
      CALL CMULT (TB,BD(2),NTOT1)
C
      DO 100 ILAG=2,NBD
         IF (IFGEOM) THEN
            CALL COL3 (TA,BM1LAG(1,1,1,1,ILAG-1),
     $                    TLAGP (1,ILAG-1,IFIELD-1,jp),NTOT1)
         ELSE
            CALL COL3 (TA,BM1,
     $                    TLAGP (1,ILAG-1,IFIELD-1,jp),NTOT1)
         ENDIF
         CALL ADD2S2(TB,TA,BD(ilag+1),NTOT1)
 100  CONTINUE
C
      CALL COL2 (TB,H2,NTOT1)
      CALL ADD2 (BQP(1,IFIELD-1,jp),TB,NTOT1)
C
      return
      end
C
      subroutine lagscalp
C-----------------------------------------------------------------------
C
C     Keep old passive scalar field(s) 
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      NTOT1 = NX1*NY1*NZ1*NELFLD(IFIELD)
C
      DO 100 ILAG=NBDINP-1,2,-1
         CALL COPY (TLAGP(1,ILAG  ,IFIELD-1,jp),
     $              TLAGP(1,ILAG-1,IFIELD-1,jp),NTOT1)
 100  CONTINUE
C
      CALL COPY (TLAGP(1,1,IFIELD-1,jp),TP(1,IFIELD-1,jp),NTOT1)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine incomprp (ux,uy,uz,up)
c
c     Project U onto the closest incompressible field
c
c     Input:  U     := (ux,uy,uz)
c
c     Output: updated values of U, iproj, proj; and
c             up    := pressure correction req'd to impose div U = 0
c
c
c     Dependencies: ifield ==> which "density" (vtrans) is used.
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
c
      common /scrns/ w1    (lx1,ly1,lz1,lelv)
     $ ,             w2    (lx1,ly1,lz1,lelv)
     $ ,             w3    (lx1,ly1,lz1,lelv)
     $ ,             dv1   (lx1,ly1,lz1,lelv)
     $ ,             dv2   (lx1,ly1,lz1,lelv)
     $ ,             dv3   (lx1,ly1,lz1,lelv)
     $ ,             dp    (lx2,ly2,lz2,lelv)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $ ,             h2    (lx1,ly1,lz1,lelv)
      common /scrhi/ h2inv (lx1,ly1,lz1,lelv)
      COMMON /SCRCH/ PREXTR(LX2,LY2,LZ2,LELV)
      logical ifprjp

c
      if (icalld.eq.0) tpres=0.0
      icalld=icalld+1
      npres=icalld
c
      ntot1  = nx1*ny1*nz1*nelv
      ntot2  = nx2*ny2*nz2*nelv
      intype = 1
      dtbd   = bd(1)/dt

      call rzero   (h1,ntot1)
c     call copy    (h2,vtrans(1,1,1,1,ifield),ntot1)
      call cmult2  (h2,vtrans(1,1,1,1,ifield),dtbd,ntot1)
      call invers2 (h2inv,h2,ntot1)

      call opdiv   (dp,ux,uy,uz)
      call chsign  (dp,ntot2)
      call ortho   (dp)


C******************************************************************


      ifprjp=.false.    ! project out previous pressure solutions?
      istart=param(95)  
      if (istep.ge.istart.and.istart.ne.0) ifprjp=.true.

      ! Most likely, the following can be commented out. (pff, 1/6/2010)
      if (npert.gt.1.or.ifbase)            ifprjp=.false.

      if (ifprjp)   call setrhs  (dp,h1,h2,h2inv)
                    call esolver (dp,h1,h2,h2inv,intype)
      if (ifprjp)   call gensoln (dp,h1,h2,h2inv)


C******************************************************************

      call opgradt (w1 ,w2 ,w3 ,dp)
      call opbinv  (dv1,dv2,dv3,w1 ,w2 ,w3 ,h2inv)
      call opadd2  (ux ,uy ,uz ,dv1,dv2,dv3)
c
      call extrapprp(prextr)
      call lagpresp
      call add3(up,prextr,dp,ntot2)
c
      return
      end
c------------------------------------------------------------------------
      subroutine extrapprp (prextr)
C
C     Pressure extrapolation
C
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      COMMON /CTMP0/ DPR (LX2,LY2,LZ2,LELV)
      REAL        PREXTR (LX2,LY2,LZ2,LELV)
C
      ntot2 = nx2*ny2*nz2*nelv
      if (nbd.eq.1.or.nbd.eq.2) then
         call copy (prextr,prp(1,JP),ntot2)
      elseif (nbd.eq.3) then
         const = dtlag(1)/dtlag(2)
         call sub3 (dpr,prp(1,JP),prlagp(1,1,JP),ntot2)
         call cmult(dpr,const,ntot2)
         call add3 (prextr,prp(1,JP),dpr,ntot2)
      elseif (nbd.gt.3) then
         write (6,*) 'Pressure extrapolation cannot be completed'
         write (6,*) 'Try a lower-order temporal scheme'
         call exitt
      endif
      return
      end
C-------------------------------------------------------------------
      subroutine lagpresp
C
C     save old pressure values
C
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      if (nbdinp.eq.3) then
         ntot2 = nx2*ny2*nz2*nelv
         call copy (prlagp(1,1,JP),prp(1,JP),ntot2)
      endif
      return
      end
C-------------------------------------------------------------------
      subroutine lyap_scale ! Rescale / orthogonalize perturbation fields
c
c
      include 'SIZE'
      include 'TOTAL'
c
      real sigma(0:lpert)
c
      ntotv = nx1*ny1*nz1*nelv
      ntotp = nx2*ny2*nz2*nelv
      ntott = nx1*ny1*nz1*nelt
c
      do j=1,npert
         call normvc(h1,semi,pl2,plinf,vxp(1,j),vyp(1,j),vzp(1,j))
         sigma(j) = pl2
         if (pl2.gt.0) then
            write(6,*) 'this is pl2:',pl2
            scale = 1./pl2
            call opcmult(vxp(1,j),vyp(1,j),vzp(1,j),scale)
            call   cmult(tp(1,1,j),scale,ntott)
            call   cmult(prp(1,j) ,scale,ntotp)
         endif
c
c        Have to do lag terms as well, etc
c        
c        Also, must orthogonalize
      enddo
c
      if (nio.eq.0) then
         if (npert.eq.1) write(6,1) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.2) write(6,2) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.3) write(6,3) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.4) write(6,4) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.5) write(6,5) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.6) write(6,6) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.7) write(6,7) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.8) write(6,8) istep,time,(sigma(j),j=1,npert)
         if (npert.eq.9) write(6,9) istep,time,(sigma(j),j=1,npert)
      endif
    1 format(i8,1p2e12.4,' pgrow')
    2 format(i8,1p3e12.4,' pgrow')
    3 format(i8,1p4e12.4,' pgrow')
    4 format(i8,1p5e12.4,' pgrow')
    5 format(i8,1p6e12.4,' pgrow')
    6 format(i8,1p7e12.4,' pgrow')
    7 format(i8,1p8e12.4,' pgrow')
    8 format(i8,1p9e12.4,' pgrow')
    9 format(i8,1p10e12.4,' pgrow')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_pert  ! dump perturbation .fld files
c
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
c
      character*3 s3
c
      do jpp=1,npert
         write(s3,3) jpp
 3       format('p',i2.2)
         call outpost2
     $   (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),prp(1,jpp),tp(1,1,jpp),1,s3)
c        call writehist
c    $     (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),jpp)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine pert_add2s2(i,j,scale)   ! xi = xi + scale * xj
      include 'SIZE'
      include 'TOTAL'

      ntotp = nx2*ny2*nz2*nelv
      ntotv = nx1*ny1*nz1*nelv
      ntott = nx1*ny1*nz1*nelt

      call add2s2(vxp(1,i),vxp(1,j),scale,ntotv)
      call add2s2(vyp(1,i),vyp(1,j),scale,ntotv)
      if (if3d) call add2s2(vzp(1,i),vzp(1,j),scale,ntotv)
      call add2s2(prp(1,i),prp(1,j),scale,ntotp)

      do l=1,lorder-1
         call add2s2(vxlagp(1,l,i),vxlagp(1,l,j),scale,ntotv)
         call add2s2(vylagp(1,l,i),vylagp(1,l,j),scale,ntotv)
         if (if3d) call add2s2(vzlagp(1,l,i),vzlagp(1,l,j),scale,ntotv)
         if (l.le.lorder-2) 
     $      call add2s2(prlagp(1,l,i),prlagp(1,l,j),scale,ntotp)
      enddo

      call add2s2(exx1p(1,i),exx1p(1,j),scale,ntotv)
      call add2s2(exy1p(1,i),exy1p(1,j),scale,ntotv)
      if (if3d) call add2s2(exz1p(1,i),exz1p(1,j),scale,ntotv)

      call add2s2(exx2p(1,i),exx2p(1,j),scale,ntotv)
      call add2s2(exy2p(1,i),exy2p(1,j),scale,ntotv)
      if (if3d) call add2s2(exz2p(1,i),exz2p(1,j),scale,ntotv)

      if (ifheat) then
        do k=0,npscal
          k1=k+1
          ntotk = nx1*ny1*nz1*nelfld(k+2)
          call add2s2(tp(1,k1,i),tp(1,k1,j),scale,ntotk)
          do l=1,lorder-1
            call add2s2(tlagp(1,l,k1,i),tlagp(1,l,k1,j),scale,ntotk)
          enddo
          call add2s2(vgradt1p(1,k1,i),vgradt1p(1,k1,j),scale,ntotk)
          call add2s2(vgradt2p(1,k1,i),vgradt2p(1,k1,j),scale,ntotk)
        enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      function pert_inner_prod(i,j) ! weighted inner product vi^T vj
      include 'SIZE'
      include 'TOTAL'

      common/normset/pran, ray, rayc

      ntotv=nx1*ny1*nz1*nelv
      ntott=nx1*ny1*nz1*nelt

      s1 = rayc*glsc3(vxp(1,i),bm1,vxp(1,j),ntotv)
      s2 = rayc*glsc3(vyp(1,i),bm1,vyp(1,j),ntotv)
      s3 = 0
      if (if3d) s3 = rayc*glsc3(vzp(1,i),bm1,vzp(1,j),ntotv)

      t1 = 0
      if (ifheat) t1=pran*ray*ray*glsc3(tp(1,1,i),bm1,tp(1,1,j),ntott)

      pert_inner_prod = (s1+s2+s3+t1)/volvm1

      return
      end
c-----------------------------------------------------------------------
      subroutine pert_ortho_norm ! orthogonalize and rescale pert. arrays
      include 'SIZE'
      include 'TOTAL'

      do k=1,npert
         call pert_ortho_norm1(k)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine pert_ortho_norm1 (k) ! orthogonalize k against 1...k-1
      include 'SIZE'
      include 'TOTAL'

      do j=1,k-1
         scale = -pert_inner_prod(j,k)
         call pert_add2s2(k,j,scale)   ! xi = xi + scale * xj
      enddo
      scale = pert_inner_prod(k,k)
      if (scale.gt.0) scale = 1./scale
      call rescalepert(pertnorm,scale,k)

      return
      end
c-----------------------------------------------------------------------
      function opnorm2(v1,v2,v3)
      include 'SIZE'
      include 'TOTAL'
c
      real v1(1) , v2(1), v3(1)
      real normsq1,normsq2,normsq3,opnorm
c
      ntotv=nx1*ny1*nz1*nelv
      normsq1=glsc3(v1,bm1,v1,ntotv)
      normsq2=glsc3(v2,bm1,v2,ntotv)
      if(if3d) then
         normsq3=glsc3(v3,bm1,v3,ntotv)
      else
         normsq3=0
      endif

      opnorm2=normsq1+normsq2+normsq3
      if (opnorm2.gt.0) opnorm2=sqrt(opnorm2/volvm1)
      return
      end
c-----------------------------------------------------------------------

      function Tnorm(temp)
      include 'SIZE'
      include 'TOTAL'

      real temp(*)
c
      ntotv = nx1*ny1*nz1*nelv
      Tnorm = sqrt( glsc3(temp,BM1,temp,ntotv) /voltm1)
c
      return
      end
c--------------------------------------------
      function dmnorm(v1,v2,v3,temp)
c     Norm weighted by mass matrix
      include 'SIZE'
      include 'TOTAL'

      real v1(1),v2(1),v3(1),temp(1)
      real normsq1,normsq2,normsq3,normsqT,dMnorm
      common/normset/pran, ray, rayc

      ntotv=nx1*ny1*nz1*nelv
      normsq1=(rayc)*glsc3(v1,BM1,v1,ntotv)
      normsq2=(rayc)*glsc3(v2,BM1,v2,ntotv)
      if(if3d) then
         normsq3=(rayc)*glsc3(v3,BM1,v3,ntotv)
      else
         normsq3=0
      endif

      if(ifheat) then
         normsqT = (pran*ray*ray)*glsc3(temp,BM1,temp,ntotv)
      else
         normsqT = 0
      endif

      dmnorm=sqrt((normsq1+normsq2+normsq3+normsqT)/volvm1)

      return
      end

c---------------------------------------------------------------
      subroutine opscale(v1,v2,v3,temp,alpha)
c     v =  alpha*v
      include 'SIZE'
      include 'INPUT'

      real alpha
      real v1(1),v2(1),v3(1),temp(1)

      ltotv=lx1*ly1*lz1*lelv
      ltott=lx1*ly1*lz1*lelt

      call cmult(v1,alpha,ltotv)
      call cmult(v2,alpha,ltotv)
      if (if3d)   call cmult(v3,alpha,ltotv)
      if (ifheat) call cmult(temp,alpha,ltott*ldimt)

      return
      end

c---------------------------------------------------------------
      subroutine opscaleV(v1,v2,v3,alpha)
c     v =  alpha*v
      include 'SIZE'
      include 'INPUT'
      real alpha
      real v1(*),v2(*),v3(*)

      ntotv=nx1*ny1*nz1*nelv

      call cmult(v1,alpha,ntotv)
      call cmult(v2,alpha,ntotv)

      if (if3d)   call cmult(v3,alpha,ntotv)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine computelyap
      include 'SIZE'
      include 'TOTAL'

      do jpp=1,npert         ! Loop through each Lyapunov eigenvalue
         call computelyap1
     $     (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),jpp)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine computelyap1(vxq,vyq,vzq,tq,jpp)
      include 'SIZE'
      include 'TOTAL'

      real vxq(1),vyq(1),vzq(1),tq(1)
      real lyapsum,twt
      common /pertsave/ timestart,tinitial

      integer icount
      save    icount
      data    icount /0/

      logical       if_restart,if_ortho_lyap
      common/restar/if_restart,if_ortho_lyap

      character*132 lyprestart
      common/restflename/lyprestart  !file for restart data

      twt = param(126) !time to wait to start computing exponents

      if (nio.eq.0) 
     $  write(6,8) istep,icount,time,twt,(lyap(k,jpp),k=1,3),jpp
    8 format(i9,i4,2f8.4,1p3e12.4,i3,'clyap')

      if(time.lt.twt) then
c
c        For a  fresh simulation, then we wait 5 vertical diffusion 
c        times before we start measuring, so in this case just rescale
c
         pertnorm = dmnorm(vxq,vyq,vzq,tq)
         pertinvnorm = 1.0/pertnorm
         call rescalepert(pertnorm,pertinvnorm,jpp)
         lyap(3,jpp) = pertnorm !store latest norm
         timestart   = time
         tinitial    = time
         icount      = 0
         return
      else
         if (jpp.eq.1) icount = icount + 1
      endif

      irescale = param(128)
      if (icount.eq.irescale) then

         lyapsum     = lyap(2,jpp)
         oldpertnorm = lyap(3,jpp)
         pertnorm=dmnorm(vxq,vyq,vzq,tq)
c
         lyap(1,jpp) = log(pertnorm/oldpertnorm)/(time-timestart)
         lyapsum     = lyapSum + log(pertnorm/oldpertnorm)
         lyap(2,jpp) = lyapSum

         if(nid.eq.0) then        ! write out results to the .lyp file
 
           write(6 ,1) istep,time,lyap(1,jpp),lyapsum,pertnorm,jpp
           write(79,2) time,lyap(1,jpp),lyapsum,pertnorm,oldpertnorm,jpp
 1          format(i9,1p4e17.8,i4,'lyap')
 2          format(1p5e17.8,i4,'lyap')
c           call flushbuffer(79)
 
            if (jpp.eq.1) open(unit=96,file=lyprestart)
            write(96,9) lyapsum,timestart,jpp
 9          format(1p2e19.11,i9)
            if (jpp.eq.npert) close(96)
         endif

         pertinvnorm = 1.0/pertnorm
         call rescalepert(pertnorm,pertinvnorm,jpp)
         lyap(3,jpp) = pertnorm  !save current pertnorm as old pertnorm

         if (jpp.eq.npert) then
            icount    = 0
            timestart = time
         endif

         if_ortho_lyap = .false.
         if (param(125).gt.0) if_ortho_lyap = .true.
         if (jpp.eq.npert .and. if_ortho_lyap) call pert_ortho_norm

      endif

      return
      end
c-----------------------------------------------------------------------

      subroutine rescalepert(pertnorm,pertinvnorm,jpp)
      include 'SIZE'
      include 'TOTAL'

      ntotp = nx2*ny2*nz2*nelv

      call opscale                     !normalize vectors to unit norm
     $      (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),pertinvnorm)
      call cmult(prp(1,jpp),pertinvnorm,ntotp)

      call opscale(exx1p(1,jpp),exy1p(1,jpp),exz1p(1,jpp)
     $                           ,vgradt1p(1,1,jpp),pertinvnorm)
      call opscale(exx2p(1,jpp),exy2p(1,jpp),exz2p(1,jpp)
     $                           ,vgradt2p(1,1,jpp),pertinvnorm)

      ltotv = lx1*ly1*lz1*lelv
      ltotp = lx2*ly2*lz2*lelv

      call cmult( tlagp(1,1,1,jpp),pertinvnorm,ltotv*(lorder-1)*ldimt)
      call cmult(vxlagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(vylagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(vzlagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(prlagp(1,1,jpp),pertinvnorm,ltotp*(Lorder-2))

      if (nio.eq.0) write(6,1) istep,pertnorm,pertinvnorm,jpp,'PNORM'
  1   format(i4,1p2e12.4,i4,a5)
      pertnorm = pertnorm*pertinvnorm

      return
      end
c-----------------------------------------------------------------------
