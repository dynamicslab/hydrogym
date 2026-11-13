      subroutine setdtcmt
!--------------------------------------------------------------
! JH072914 now summed instead of maximized for compressible flow
! JH082015 why was I doing that again?
!     someday compute new timestep based on CFL-condition. follow
!     setdtc in nek/subs1.f very carefully.
! JH091616 now with diffusion number for naively explicit visc
! JH091616 consider migrating to compute_cfl
!--------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'MVGEOM'
      include 'MASS'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      COMMON /CMTGASPROP/  CSOUND(lx1,ly1,lz1,lelt)
      real                 csound

!--------------------------------------------------------------
! YOU REALLY PROBABLY WANT YOUR OWN SCRATCH SPACE FOR THIS
      common /scrsf/ u(lx1,ly1,lz1,lelt) ! mind if I borrow this?
     $ ,             v(lx1,ly1,lz1,lelt) ! as long as the mesh
     $ ,             w(lx1,ly1,lz1,lelt) ! doesn't move
! YOU REALLY PROBABLY WANT YOUR OWN SCRATCH SPACE FOR THAT
!--------------------------------------------------------------

      common /udxmax/ umax
      REAL VCOUR
      SAVE VCOUR
      REAL mdr
      SAVE mdr

      NTOT   = NX1*NY1*NZ1*NELV
      NTOTL  = LX1*LY1*LZ1*lelt
      NTOTD  = NTOTL*NDIM
      COLD   = COURNO
      CMAX   = 1.2*CTARG
      CMIN   = 0.8*CTARG
      do i=1,ntot
         u(i,1,1,1)=abs(vx(i,1,1,1))+csound(i,1,1,1)
         v(i,1,1,1)=abs(vy(i,1,1,1))+csound(i,1,1,1)
         w(i,1,1,1)=abs(vz(i,1,1,1))+csound(i,1,1,1)
      enddo
      CALL CUMAX (u,v,w,UMAX)
      COURNO = DT*UMAX
      VOLD   = VCOUR
      VCOUR  = UMAX

! diffusion number based on viscosity.

      call mindr(mdr,diffno2)
      if (nio.eq.0) WRITE(6,100)ISTEP,TIME,DT,COURNO,diffno2
 100  FORMAT('CMT ',I7,', t=',1pE14.7,', DT=',1pE14.7
     $,', C=',0pF7.3,', D=',1pE14.7)
!    $,', C=',0pF7.3,', D=',0pF7.3)
!     Zero DT
!
!     IF (DT .EQ. 0.0) THEN
!
!        IF (UMAX .NE. 0.0) THEN
!           DT = CTARG/UMAX
!           VCOUR = UMAX
      return
      end

      subroutine mindr(mdr,diffno)
c
c     Find minimum distance between grid points
c     and multiply it by viscosity to get diffusion number
c     Probably need to do this for energy equation too...
! JH091616 migrate to getdr 3d ASAP. Again, follow compute_cfl
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'

      real    mdr,dr1
      real    x0,x1,x2,x3,x4,y0,y1,y2,y3,y4
      integer e
c
      mdr=1.e5
      if(if3d) then
        write(6,*)'TODO:: mindr for 3D. getdr'
        call exitt
      else
        diffno=0.0
        do e=1,nelt
          do iy=2,ny1-1
          do ix=2,nx1-1
            dtmp=1.0e5
            x0 = xm1(ix  ,iy  ,1,e)
            x1 = xm1(ix  ,iy-1,1,e)
            x2 = xm1(ix+1,iy  ,1,e)
            x3 = xm1(ix  ,iy+1,1,e)
            x4 = xm1(ix-1,iy  ,1,e)
            y0 = ym1(ix  ,iy  ,1,e)
            y1 = ym1(ix  ,iy-1,1,e)
            y2 = ym1(ix+1,iy  ,1,e)
            y3 = ym1(ix  ,iy+1,1,e)
            y4 = ym1(ix-1,iy  ,1,e)
            dr1 = dist2(x0,y0,x1,y1)
            if(dr1.lt.mdr) mdr=dr1
            if(dr1.lt.dtmp) dtmp=dr1
            dr1 = dist2(x0,y0,x2,y2)
            if(dr1.lt.mdr) mdr=dr1
            if(dr1.lt.dtmp) dtmp=dr1
            dr1 = dist2(x0,y0,x3,y3)
            if(dr1.lt.mdr) mdr=dr1
            if(dr1.lt.dtmp) dtmp=dr1
            dr1 = dist2(x0,y0,x4,y4)
            if(dr1.lt.mdr) mdr=dr1
            if(dr1.lt.dtmp) dtmp=dr1
            diffno=max(diffno,dt*vdiff(ix,iy,1,e,imu)/dtmp/dtmp)
          enddo
          enddo
        enddo
      endif
      diffno=glmax(diffno,1)
      mdr = glmin(mdr,1)
      if(mdr.ge.1.e5) write(6,*) 'wrong mdr'

      return
      end

      real function dist2(x1,y1,x2,y2)
      real x1,y1,x2,y2,dx,dy
c
      dx = x1-x2
      dy = y1-y2
      dist2 = sqrt(dx*dx+dy*dy)
c
      return
      end
