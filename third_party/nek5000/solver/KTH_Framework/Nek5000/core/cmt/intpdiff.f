C> @file intpdiff.f interpolation and differentiation routines not already provided
C> by nek5000
!--------------------------------------------------------------------
! JH061914 propose name change to intpdiff since that is what is in
!          here.
! JH081816 went to local_grad. redimensioned gradu. wish I could use
!          gradm11, but stride
!--------------------------------------------------------------------

      subroutine compute_gradients(e)
      include 'SIZE'
      include 'INPUT'
      include 'DXYZ'
      include 'GEOM'
      include 'SOLN'
      include 'CMTDATA'
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ud(ldd),tu(ldd)

      integer eq,e

!     !  Compute d/dx, d/dy and d/dz of all the cons vars

      nxy1  = lx1*ly1
      nyz1  = ly1*lz1
      nxyz1 = lx1*ly1*lz1
      m0 = lx1-1

      do eq=1,toteq
         call invcol3(ud,u(1,1,1,eq,e),phig(1,1,1,e),nxyz1)

         if (if3d) then
            call local_grad3(ur,us,ut,ud,m0,1,dxm1,dxtm1)
            do i=1,nxyz1
               gradu(i,eq,1) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)+
     >                                     sxm1(i,1,1,e)*us(i)+
     >                                     txm1(i,1,1,e)*ut(i))
            enddo
            do i=1,nxyz1
               gradu(i,eq,2) = jacmi(i,e)*(rym1(i,1,1,e)*ur(i)+
     >                                     sym1(i,1,1,e)*us(i)+
     >                                     tym1(i,1,1,e)*ut(i))
            enddo
            do i=1,nxyz1
               gradu(i,eq,3) = jacmi(i,e)*(rzm1(i,1,1,e)*ur(i)+
     >                                     szm1(i,1,1,e)*us(i)+
     >                                     tzm1(i,1,1,e)*ut(i))
            enddo

         else

            call local_grad2(ur,us   ,ud,m0,1,dxm1,dxtm1)
            do i=1,nxyz1
               gradu(i,eq,1) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)+
     >                                     sxm1(i,1,1,e)*us(i))
            enddo
            do i=1,nxyz1
               gradu(i,eq,2) = jacmi(i,e)*(rym1(i,1,1,e)*ur(i)+
     >                                     sym1(i,1,1,e)*us(i))
            enddo

         endif

      enddo ! equation loop

      return
      end

!-----------------------------------------------------------------------

      subroutine set_dealias_face

!-----------------------------------------------------------------------
! JH111815 needed for face Jacobian and surface integrals
!-----------------------------------------------------------------------

      include 'SIZE'
      include 'INPUT' ! for if3d
      include 'GEOM'  ! for ifgeom
      include 'TSTEP' ! for istep
      include 'WZ'    ! for wxm1
      include 'DG'    ! for facewz

      integer ilstep
      save    ilstep
      data    ilstep /-1/

      if (.not.ifgeom.and.ilstep.gt.1) return  ! already computed
      if (ifgeom.and.ilstep.eq.istep)  return  ! already computed
      ilstep = istep

      call zwgl(zptf,wgtf,lxd)

      if (if3d) then
         k=0
         do j=1,ly1
         do i=1,lx1
            k=k+1
            wghtc(k)=wxm1(i)*wzm1(j)
         enddo
         enddo
         k=0
         do j=1,lyd
         do i=1,lxd
            k=k+1
            wghtf(k)=wgtf(i)*wgtf(j)
         enddo
         enddo
      else
         call copy(wghtc,wxm1,lx1)
         call copy(wghtf,wgtf,lxd)
      endif

      return
      end
!-----------------------------------------------------------------------

      subroutine set_alias_rx(istp)
! note that set_alias_rx will be called only when lxd = lx1
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
c     include 'TSTEP' ! for istep

      common /dealias1/ zd(lx1),wd(lx1)
      integer e

      integer ilsp
      save    ilsp
      data    ilsp /-1/

      if (.not.ifgeom.and.ilsp.gt.1) return  ! already computed
      if (ifgeom.and.ilsp.eq.istp)  return  ! already computed
      ilsp = istp
      nxyz = lx1*ly1*lz1
      call zwgll (zd,wd,lx1)  

      if (if3d)then
         do e=1,nelv

            call copy(rx(1,1,e),rxm1(1,1,1,e),nxyz) 
            call copy(rx(1,2,e),rym1(1,1,1,e),nxyz) 
            call copy(rx(1,3,e),rzm1(1,1,1,e),nxyz) 
            call copy(rx(1,4,e),sxm1(1,1,1,e),nxyz) 
            call copy(rx(1,5,e),sym1(1,1,1,e),nxyz) 
            call copy(rx(1,6,e),szm1(1,1,1,e),nxyz) 
            call copy(rx(1,7,e),txm1(1,1,1,e),nxyz) 
            call copy(rx(1,8,e),tym1(1,1,1,e),nxyz) 
            call copy(rx(1,9,e),tzm1(1,1,1,e),nxyz) 

            l = 0
            do k=1,lz1
            do j=1,ly1
            do i=1,lx1
               l = l+1
               w = wd(i)*wd(j)*wd(k)
               do ii=1,9
                  rx(l,ii,e) = w*rx(l,ii,e)
               enddo
            enddo
            enddo
            enddo
            enddo
      else

         do e=1,nelv

c           Interpolate z+ and z- into fine mesh, translate to r-s-t coords

            call copy(rx(1,1,e),rxm1(1,1,1,e),nxyz) 
            call copy(rx(1,2,e),rym1(1,1,1,e),nxyz) 
            call copy(rx(1,3,e),sxm1(1,1,1,e),nxyz) 
            call copy(rx(1,4,e),sym1(1,1,1,e),nxyz) 

            l = 0
            do j=1,ly1
            do i=1,lx1
               l = l+1
               w = wd(i)*wd(j)
               do ii=1,4
                  rx(l,ii,e) = w*rx(l,ii,e)
               enddo
            enddo
            enddo
         enddo

      endif

      return
      end

!-----------------------------------------------------------------------

      subroutine gradm11_t(grad,uxyz,csgn,e) ! grad is incremented, not overwritten
c     source: . gradm1, from navier5.f 
c             . gradm1, from navier5.f 
c     Compute divergence^T of ux,uy,uz -- mesh 1 to mesh 1 (vel. to vel.)
!     single element, but references jacmi and the metrics

      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'

      parameter (lxyz=lx1*ly1*lz1)
      real grad(lxyz),uxyz(lxyz,ldim)

      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz),ud(lxyz),tmp(lxyz)
      real ur,us,ut,tmp

      integer e

      nxyz = lx1*ly1*lz1
      call rzero(ud,nxyz)

      N = lx1-1
      if (if3d) then

         do i=1,lxyz
            ur(i) = jacmi(i,e)*(uxyz(i,1)*rxm1(i,1,1,e)
     >                        + uxyz(i,2)*rym1(i,1,1,e)
     >                        + uxyz(i,3)*rzm1(i,1,1,e) )
            us(i) = jacmi(i,e)*(uxyz(i,1)*sxm1(i,1,1,e)
     >                        + uxyz(i,2)*sym1(i,1,1,e)
     >                        + uxyz(i,3)*szm1(i,1,1,e) )
            ut(i) = jacmi(i,e)*(uxyz(i,1)*txm1(i,1,1,e)
     >                        + uxyz(i,2)*tym1(i,1,1,e)
     >                        + uxyz(i,3)*tzm1(i,1,1,e) )
         enddo
         call local_grad3_t(ud,ur,us,ut,N,1,dxm1,dxtm1,tmp)
      else
         do i=1,lxyz
            ur(i) =jacmi(i,e)*(uxyz(i,1)*rxm1(i,1,1,e)
     >                       + uxyz(i,2)*rym1(i,1,1,e) )
            us(i) =jacmi(i,e)*(uxyz(i,1)*sxm1(i,1,1,e)
     >                       + uxyz(i,2)*sym1(i,1,1,e) )
         enddo
         call local_grad2_t(ud,ur,us,N,1,dxm1,dxtm1,tmp)
      endif
      call cmult(ud,csgn,nxyz)
      call add2(grad,ud,nxyz)

      return
      end

!-----------------------------------------------------------------------

      subroutine gradm1_t(u,ux,uy,uz)
! JH082516 torn bleeding from Lu's dgf3.f. someday you are going to have
!          to vectorize cmt-nek properly.
c     source: . gradm1, from navier5.f 
c             . gradm1, from navier5.f 
c
c     Compute divergence of ux,uy,uz -- mesh 1 to mesh 1 (vel. to vel.)
c
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
c
      parameter (lxyz=lx1*ly1*lz1)
      real ux(lxyz,*),uy(lxyz,*),uz(lxyz,*),u(lxyz,*)

      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz),tmp(lxyz)
      real ur,us,ut,tmp

      integer e

      nxyz = lx1*ly1*lz1
      ntot = nxyz*nelt

      N = lx1-1
      do e=1,nelt
         if (if3d) then

            do i=1,lxyz
               ur(i) = jacmi(i,e)*(ux(i,e)*rxm1(i,1,1,e)
     $                           + uy(i,e)*rym1(i,1,1,e)
     $                           + uz(i,e)*rzm1(i,1,1,e) )
               us(i) = jacmi(i,e)*(ux(i,e)*sxm1(i,1,1,e)
     $                           + uy(i,e)*sym1(i,1,1,e)
     $                           + uz(i,e)*szm1(i,1,1,e) )
               ut(i) = jacmi(i,e)*(ux(i,e)*txm1(i,1,1,e)
     $                           + uy(i,e)*tym1(i,1,1,e)
     $                           + uz(i,e)*tzm1(i,1,1,e) )
            enddo
            call local_grad3_t(u,ur,us,ut,N,e,dxm1,dxtm1,tmp)
         else
            do i=1,lxyz
               ur(i) =jacmi(i,e)*(ux(i,e)*rxm1(i,1,1,e)
     $                          + uy(i,e)*rym1(i,1,1,e) )
               us(i) =jacmi(i,e)*(ux(i,e)*sxm1(i,1,1,e)
     $                          + uy(i,e)*sym1(i,1,1,e) )
            enddo
            call local_grad2_t(u,ur,us,N,e,dxm1,dxtm1,tmp)
         endif
      enddo
c
      return
      end
