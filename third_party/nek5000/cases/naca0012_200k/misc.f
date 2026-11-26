!!======================================================================
!!
!!   Bunch of miscelineous routines.
!!   misc.f made by Prabal Negi
!!   Routine are not mine...
!!
!!======================================================================
 
      real function step(x)
c     nima 
c     belongs to bla version 2.2
c     for more info see the bla.f file
c     
      real x
      if(x.le.0.02) then
         step=0.
      else
         if(x.le.0.98) then
            step=1./(1.+exp(1./(x-1.)+1./x))
         else
            step=1.
         endif
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine fix_mygll()

      implicit none

cc MA2      include 'SIZE_DEF'
      include 'SIZE'
c MA      include 'GEOM_DEF'
      include 'GEOM'
c MA      include 'INPUT_DEF'
      include 'INPUT'
c MA      include 'SOLN_DEF'
      include 'SOLN'

      real dum(3)

      CHARACTER CB*3
      integer Iel,NFACES,npts
      parameter(npts=1000001)
      real*8 xnew,ynew
      real*8 xexact(npts),yexact(npts)

      real*8 deltax(lx1,ly1,lz1,lelt),deltay(lx1,ly1,lz1,lelt)
      integer lt
      parameter (lt=lx1*ly1*lz1*lelt)

      integer iface,ifld
      integer i,n
      integer kx1,kx2,ky1,ky2,kz1,kz2
      integer ix,iy,iz

      real psx(lt),psy(lt)

      n=nx1*ny1*nz1*nelt

      call rzero(deltax,lt)
      call rzero(deltay,lt)

      NFACES=2*NDIM

      ifxyo = .true.
      call outpost (vx,vy,vz,pr,t,'gri')
      n = nx1*ny1*nz1*nelv

!     Read GLL points (2D) from file
      open(unit=19,file='naca4412.dat')
      do i=1,npts
        read(19,*) xexact(i),yexact(i)
      enddo
      close (19)

      ifld = 1
      do  Iel=1,NELV            !do ieg=1,nelgt
        do IFACE = 1,NFACES
          CB = CBC(IFACE,Iel,ifld)
          if (CB.EQ.'W  ') then
            CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1
     $           ,IFACE)
            do iz=KZ1,KZ2
              do iy=KY1,KY2
                do  ix=KX1,KX2
                  if (xm1(ix,iy,iz,iel).lt.0.99) then
                    call fix_naca_pts(xnew,ynew,xm1(ix,iy,iz,Iel),
     $              ym1(ix,iy,iz,Iel),xexact,yexact)
                    deltax(ix,iy,iz,Iel)=xnew-xm1(ix,iy,iz,iel)
                    deltay(ix,iy,iz,Iel)=ynew-ym1(ix,iy,iz,iel)
                  else  
                    deltax(ix,iy,iz,Iel)=0.
                    deltay(ix,iy,iz,Iel)=0.
                  endif
                enddo
              enddo
            enddo
          endif
        enddo
      enddo

      call my_fixmesh(psx,psy,deltax,deltay)
      call add2(xm1,psx,lt)
      call add2(ym1,psy,lt)
      call fix_geom

      call copy(vx,psx,lt)
      call copy(vy,psy,lt)

      call outpost (vx,vy,vz,pr,t,'gri')

      return
      end

!---------------------------------------------------------------------- 

      subroutine fix_naca_pts(xnew,ynew,xfoil,yfoil,xexact,yexact)

      real*8 xfoil,yfoil,xnew,ynew,dist_min
      integer npts,counter,i
      parameter(npts=1000001)

      real*8 dist

      real*8 xexact(1),yexact(1)

      dist_min = 1.
c     Computed the distance for each point

      do i=1,npts
        dist = sqrt(abs(xexact(i)-xfoil)**2+abs(yexact(i)-yfoil)**2)
        if (dist.lt.dist_min) then
          counter = i
          dist_min = dist
        endif
      enddo

      xnew = xexact(counter)
      ynew = yexact(counter)

      return
      end

c-----------------------------------------------------------------------
      subroutine my_fixmesh(psx,psy,usrfldx,usrfldy)

!     Initialize blending function for mesh motion. 
      implicit none

cc MA2      include 'SIZE_DEF'
      include 'SIZE'

      integer lt
      parameter (lt=lx1*ly1*lz1*lelv)
      
      common /scrns/  h1(lt),h2(lt),rhs(lt),msk(lt),tmp(lt)
      real h1,h2,rhs,msk,tmp

      real psx(lt),psy(lt)
      real usrfldx(lt),usrfldy(lt)

      call my_poss_soln(psx,psy,h1,h2,rhs,msk,tmp,'W  ',usrfldx,usrfldy)

      return
      end subroutine my_fixmesh

!---------------------------------------------------------------------- 

      subroutine my_poss_soln(psx,psy,h1,h2,rhs,msk,tmp,surface,
     $      usrfldx,usrfldy)

      implicit none

cc MA2      include 'SIZE_DEF'
      include 'SIZE'
c MA      include 'INPUT_DEF'
      include 'INPUT'
c MA      include 'GEOM_DEF'
      include 'GEOM'
c MA      include 'SOLN_DEF'
      include 'SOLN'          ! vmult

      real tmp(lx1,ly1,lz1,lelt)
      real h1(1),h2(1),rhs(1),msk(1)
      real h3(lx1*ly1*lz1*lelt)           ! diagnostics

      real m1
      integer i,e,f,n,imsh,ifield,ifld

      character(3) surface
      real rr,arg,delta,z0,tol,xavg,tolold

      integer nface

      real psx(1),psy(1)                ! output
      real usrfldx(1),usrfldy(1)        ! BCs

      n = nx1*ny1*nz1*nelv

      call rone (h1 ,n)  ! H*u = -div (h1 grad u) + h2 u = 0
      call rzero(h2 ,n)  ! h2  = 0
      call rone (msk,n)  ! Mask, for Dirichlet boundary values
      call rzero(tmp,n)  ! RHS for Laplace solver

      call rzero(h3,n)   ! temporary diagnostics. Remove from
                         ! declaration as well
c
c     Modify h1 to make blending function constant near the surface.
c     The idea here is to push the majority of the mesh deformation to the 
c     far field, away from where the fine, boundary-layer-resolving elements
c     are close to the cylinder.

      ifld = 1
      call cheap_dist(h1,ifld,surface)       ! calculate distance from defined "surface"
      delta = 0.5
      do i=1,n
        rr = h1(i)
        h3(i) = rr
        arg   = -rr/(delta**2)
        h1(i) = 1. + 9.0*exp(arg)
      enddo

      z0 =  0.

      nface = 2*ndim
      do e=1,nelv
        do f=1,nface
c         Set Dirichlet for mesh velocity on all non-interior boundaries
          if (cbc(f,e,1).ne.'E  '.and.cbc(f,e,1).ne.'P  ') 
     $         call facev(msk,e,f,z0,nx1,ny1,nz1)
        enddo
      enddo

cc      tol = 1.e-16
      tol=1.0e-12
      imsh   = 1
      ifield = 1
      
      tolold = param(22)
      param(22) = tol

!     deltax      
      call copy(tmp,usrfldx,n)
      call chsign(tmp,n)
      call axhelm (rhs,tmp,h1,h2,1,1)
      call hmholtz('mshv',psx,rhs,h1,h2,msk,vmult,imsh,tol,200000,1)
      call sub2(psx,tmp,n)

      call dsavg(psx)       ! This ensures periodic points have exactly the same deltax.
                            ! Should also remove tears from meshes  

!     deltay
      call copy(tmp,usrfldy,n)
      call chsign(tmp,n)
      call axhelm (rhs,tmp,h1,h2,1,1)
      call hmholtz('mshv',psy,rhs,h1,h2,msk,vmult,imsh,tol,200000,1)
      call sub2(psy,tmp,n)

      call dsavg(psy)       ! This ensures periodic points have exactly the same deltay.
                            ! Should also remove tears from meshes  

      param(22) = tolold
           
      return
      end subroutine my_poss_soln

c-----------------------------------------------------------------------

      subroutine get_spectra(coef,f)

      implicit none

cc MA2      include 'SIZE_DEF'
      include 'SIZE'

      integer i
      real coef(lx1,ly1,lz1,lelt)
      real f(lx1,ly1,lz1,lelt)

      do i=1,nelv
           call err_est_el_lget(coef(1,1,1,i),f(1,1,1,i))
      enddo

      return
      end subroutine get_spectra
c-----------------------------------------------------------------------
!!   Old userpar read routine.
!!   Keeping it just in case...
!!   Replaced by uprm_in...

!      subroutine user_param
!
!      include 'SIZE'            ! NID
!      include 'INPUT'           ! REAFLE
!      include 'PARALLEL'        ! ISIZE, WDSIZE
!
!c     Thir is user include file
!      include 'USER_PAR'        ! UNPARAM, UPARAM
!
!      integer i, len, ierr
!
!      character*132 fname
!      character*1 fnam1(132)
!      equivalence (fnam1,fname)
!
!      call rzero(UPARAM,unparam)
!
!c     Open parameter file and read contents
!      ierr=0
!      if (NID.eq.0) then
!         call blank(fname,132)
!         len = ltrunc(REAFLE,132)
!         call chcopy(fnam1(1),REAFLE,len)
!         call chcopy(fnam1(len+1),'.usr',4)
!         write(6,*) 'Openning uresr parameter file: ',fname
!         open (unit=59,file=fname,err=30, status='old')
!         read(59,*,err=30)      ! skip header
!         read(59,*,err=30) len  ! number of lines to read
!         goto 31
! 30      ierr=1
! 31      continue
!      endif
!      call err_chk(ierr,'Error reading .rea.usr file.$')
!
!c     send number of parameters
!      call bcast(len ,ISIZE)
!
!c     compare with array length
!      if (len.gt.unparam) then
!         if(NID.eq.0) write(6,*) 'ERROR: too many parameters in ',fname
!         call exitt
!      endif
!
!c     read and distribute user parameters
!      ierr=0
!      if (NID.eq.0) then
!         do i=1, len
!            read(59,*,err=40) UPARAM(i)
!            write(6,45) i,UPARAM(i)
!         enddo
!         close(59)
!         goto 41
! 40      ierr=1
! 41      continue     
!      endif
!      call err_chk(ierr,'Error reading .rea.usr file.$')
!
! 45   FORMAT('UPARAM(',I2,') = ',G13.5)
!
!      call bcast(UPARAM ,unparam*WDSIZE)
!
!      return
!      end
!c-----------------------------------------------------------------------

