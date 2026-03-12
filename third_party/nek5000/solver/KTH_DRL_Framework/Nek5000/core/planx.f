      SUBROUTINE PLAN3 (IGEOM)
C-----------------------------------------------------------------------
C
C     Compute pressure and velocity using consistent approximation spaces.     
C     Operator splitting technique.
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'EIGEN'
      include 'SOLN'
      include 'TSTEP'
C
      COMMON /SCRNS/  RESV1 (LX1,LY1,LZ1,LELV)
     $ ,              RESV2 (LX1,LY1,LZ1,LELV)
     $ ,              RESV3 (LX1,LY1,LZ1,LELV)
     $ ,              DV1   (LX1,LY1,LZ1,LELV)
     $ ,              DV2   (LX1,LY1,LZ1,LELV)
     $ ,              DV3   (LX1,LY1,LZ1,LELV)
      COMMON /SCRVH/  H1    (LX1,LY1,LZ1,LELV)
     $ ,              H2    (LX1,LY1,LZ1,LELV)
C
      IF (IGEOM.EQ.1) THEN
C
C        Old geometry
C
         CALL MAKEF
C
      ELSE
C
C        New geometry, new b.c.
C
         intype = -1
         call sethlm  (h1,h2,intype)
         call cresvif (resv1,resv2,resv3,h1,h2)

         call ophinv  (dv1,dv2,dv3,resv1,resv2,resv3,h1,h2,tolhv,nmxv)
         call opadd2  (vx,vy,vz,dv1,dv2,dv3)
c
         call incomprn(vx,vy,vz,pr)
C
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE LAGPRES 
C--------------------------------------------------------------------
C
C     Keep old pressure values
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'

      common /cgeom/ igeom

      IF (NBDINP.EQ.3.and.igeom.le.2) THEN
         NTOT2 = lx2*ly2*lz2*NELV
         CALL COPY (PRLAG,PR,NTOT2)
      ENDIF
      RETURN
      END
C
      subroutine cresvif (resv1,resv2,resv3,h1,h2)
C---------------------------------------------------------------------
C
C     Compute startresidual/right-hand-side in the velocity solver
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      REAL           RESV1 (LX1,LY1,LZ1,1)
      REAL           RESV2 (LX1,LY1,LZ1,1)
      REAL           RESV3 (LX1,LY1,LZ1,1)
      REAL           H1    (LX1,LY1,LZ1,1)
      REAL           H2    (LX1,LY1,LZ1,1)
      COMMON /SCRUZ/ W1    (LX1,LY1,LZ1,LELV)
     $ ,             W2    (LX1,LY1,LZ1,LELV)
     $ ,             W3    (LX1,LY1,LZ1,LELV)

      common /cgeom/ igeom

      NTOT1 = lx1*ly1*lz1*NELV
      NTOT2 = lx2*ly2*lz2*NELV
      if (igeom.eq.2) CALL LAGVEL 
      CALL BCDIRVC (VX,VY,VZ,v1mask,v2mask,v3mask)
      CALL BCNEUTR
C
      call extrapp (pr,prlag)
      call opgradt (resv1,resv2,resv3,pr)
      CALL OPADD2  (RESV1,RESV2,RESV3,BFX,BFY,BFZ)
      CALL OPHX    (W1,W2,W3,VX,VY,VZ,H1,H2)
      CALL OPSUB2  (RESV1,RESV2,RESV3,W1,W2,W3)
C
      RETURN
      END
c-----------------------------------------------------------------------
