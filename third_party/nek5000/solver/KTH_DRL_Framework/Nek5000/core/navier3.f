      SUBROUTINE EPREC2(Z2,R2)
C----------------------------------------------------------------
C
C     Precondition the explicit pressure operator (E) with
C     a Neumann type (H1) Laplace operator: JT*A*J.
C     Invert A by conjugate gradient iteration or multigrid.
C
C     NOTE: SCRNS is used.
C
C----------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'PARALLEL'
      INCLUDE 'TSTEP'
      REAL           Z2   (LX2,LY2,LZ2,LELV)
      REAL           R2   (LX2,LY2,LZ2,LELV)
      COMMON /SCRNS/ MASK (LX1,LY1,LZ1,LELV)
     $              ,R1   (LX1,LY1,LZ1,LELV)
     $              ,X1   (LX1,LY1,LZ1,LELV)
     $              ,W2   (LX2,LY2,LZ2,LELV)
     $              ,H1   (LX1,LY1,LZ1,LELV)
     $              ,H2   (LX1,LY1,LZ1,LELV)
      REAL    MASK
c
      integer icalld
      save    icalld
      data    icalld/0/
      icalld=icalld+1
c
      ntot2  = lx2*ly2*lz2*nelv
      call rzero(z2,ntot2)
c
c
c
c
c  Both local and global solver...
       call dd_solver (z2,r2)
c
c
c
c  Local solver only
c      call local_solves_fdm (z2,r2)
c
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dd_solver(u,v)
c
      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'CTIMER'
c
      real u(1),v(1)
      common /scrprc/ uc(lx1*ly1*lz1*lelt)
c
      if (icalld.eq.0) then
         tddsl=0.0
         tcrsl=0.0
         nddsl=0
         ncrsl=0
      endif
      icalld = icalld + 1
      nddsl  = nddsl  + 1
      ncrsl  = ncrsl  + 1

      ntot  = lx2*ly2*lz2*nelv
      call rzero(u,ntot)

      etime1=dnekclock()
      call local_solves_fdm    (u,v)
      tddsl=tddsl+dnekclock()-etime1

      etime1=dnekclock()
      call crs_solve_l2 (uc,v)
      tcrsl=tcrsl+dnekclock()-etime1

      alpha = 10.
c     if (param(89).ne.0.) alpha = abs(param(89))
      call add2s2(u,uc,alpha,ntot)

      return
      end
c-----------------------------------------------------------------------
      subroutine rar2_out(x,name13)
      include 'SIZE'
c
      real x(lx2,ly2,lz2,lelt)
      character*13 name13
c
      if (nelv.gt.20) return
      write(6,*) 
      write(6,1) name13
    1 format(a13)
      if (nelv.gt.2) then
         write(6,*) 
         do j=ly2,1,-1
            write(6,6) (x(k,j,1,3),k=1,lx2),(x(k,j,1,4),k=1,lx2)
         enddo
         write(6,*)
         write(6,*)
      endif
c
      do j=ly2,1,-1
         write(6,6) (x(k,j,1,1),k=1,lx2),(x(k,j,1,2),k=1,lx2)
      enddo
      write(6,*)
    6 format(3f8.4,5x,3f8.4)
      return
      end
c-----------------------------------------------------------------------
      subroutine rarr_out2(x,name13)
      include 'SIZE'
      include 'INPUT'
c
      real x(lx2,ly2,lz2,lelt)
      character*13 name13
c
      if (nelv.gt.20) return
      write(6,*) 
      write(6,1) name13
    1 format('rarr2',3x,a13)
c
c     3 D
c
      if (if3d) then
         do iz=1,lz1
            write(6,*) 
            do j=ly1,1,-1
               write(6,3) (x(k,j,iz,1),k=1,lx2),(x(k,j,iz,2),k=1,lx2)
            enddo
         enddo
         write(6,*)
         write(6,*)
         return
      endif
c
c     2 D
c
      if (nelv.gt.2) then
         write(6,*) 
         do j=ly2,1,-1
            write(6,6) (x(k,j,1,3),k=1,lx2),(x(k,j,1,4),k=1,lx2)
         enddo
         write(6,*)
         write(6,*)
      endif
c
      do j=ly2,1,-1
         write(6,6) (x(k,j,1,1),k=1,lx2),(x(k,j,1,2),k=1,lx2)
      enddo
      write(6,*)
    3 format(4f6.2,5x,4f6.2)
    6 format(4f8.5,5x,4f8.5)
      return
      end
c-----------------------------------------------------------------------
