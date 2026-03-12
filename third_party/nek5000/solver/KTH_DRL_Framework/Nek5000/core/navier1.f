      subroutine plan1 (igeom)
C-------------------------------------------------------------------------
C
C     Compute pressure and velocity using consistent approximation spaces.     
C
C-------------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'EIGEN'
      include 'SOLN'
      include 'TSTEP'
C
      COMMON /SCRHI/ H2INV (LX1,LY1,LZ1,LELV)
      COMMON /SCRNS/ RESV1 (LX1,LY1,LZ1,LELV)
     $ ,             RESV2 (LX1,LY1,LZ1,LELV)
     $ ,             RESV3 (LX1,LY1,LZ1,LELV)
     $ ,             DV1   (LX1,LY1,LZ1,LELV)
     $ ,             DV2   (LX1,LY1,LZ1,LELV)
     $ ,             DV3   (LX1,LY1,LZ1,LELV)
     $ ,             WP    (LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
      REAL           G1    (LX1,LY1,LZ1,LELV)
      REAL           G2    (LX1,LY1,LZ1,LELV)
      REAL           G3    (LX1,LY1,LZ1,LELV)
      EQUIVALENCE   (G1,RESV1), (G2,RESV2), (G3,RESV3)
      LOGICAL        IFSTUZ
C
      IF (IGEOM.EQ.1) THEN
C
C        Old geometry
C
         CALL MAKEF
         CALL LAGVEL
C
      ELSE
C
C        New geometry
C
         CALL BCDIRVC (VX,VY,VZ,v1mask,v2mask,v3mask)
         IF (IFSTRS) CALL BCNEUTR
C
C        Check if steady state
C
         IFSTUZ = .FALSE.
         CALL CONVUZ (IFSTUZ)
C... no steady state
         IFSTUZ = .FALSE.
         IF (IFSTUZ) THEN
            IF (NIO.EQ.0) WRITE (6,*) 
     $      'Steady state reached in the fluid solver'
            return
         ENDIF
C
C        Uzawa decoupling: First, compute pressure.....
C
         ntot1  = lx1*ly1*lz1*nelv
         ntot2  = lx2*ly2*lz2*nelv
         intype = 0
         if (iftran) intype = -1
         call sethlm  (h1,h2,intype)
         if (iftran)  call invers2 (h2inv,h2,ntot1)
         call makeg   (   g1,g2,g3,h1,h2,intype)
         call crespuz (wp,g1,g2,g3,h1,h2,h2inv,intype)
         call uzawa   (wp,h1,h2,h2inv,intype,icg)
         if (icg.gt.0) call add2 (pr,wp,ntot2)

C        .... then, compute velocity:

         call cresvuz (resv1,resv2,resv3)
         call ophinv  (dv1,dv2,dv3,resv1,resv2,resv3,h1,h2,tolhv,nmxv)
         call opadd2  (vx,vy,vz,dv1,dv2,dv3)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine crespuz (respr,g1,g2,g3,h1,h2,h2inv,intype)

c     Compute start-residual/right-hand-side in the pressure equation

      include 'SIZE'
      include 'TOTAL'
      REAL           RESPR  (LX2,LY2,LZ2,LELV)
      REAL           G1     (LX1,LY1,LZ1,LELV)
      REAL           G2     (LX1,LY1,LZ1,LELV)
      REAL           G3     (LX1,LY1,LZ1,LELV)
      REAL           H1     (LX1,LY1,LZ1,LELV)
      REAL           H2     (LX1,LY1,LZ1,LELV)
      REAL           H2INV  (LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/ TA1    (LX1,LY1,LZ1,LELV)
     $ ,             TA2    (LX1,LY1,LZ1,LELV)
     $ ,             TA3    (LX1,LY1,LZ1,LELV)
      COMMON /SCRMG/ VBDRY1 (LX1,LY1,LZ1,LELV)
     $ ,             VBDRY2 (LX1,LY1,LZ1,LELV)
     $ ,             VBDRY3 (LX1,LY1,LZ1,LELV)

      if ((intype.eq.0).or.(intype.eq.-1)) then
         call ophinv (ta1,ta2,ta3,g1,g2,g3,h1,h2,tolhr,nmxp)
      else
         call opbinv (ta1,ta2,ta3,g1,g2,g3,h2inv)
      endif
      call opamask   (vbdry1,vbdry2,vbdry3)
      call opsub2    (ta1,ta2,ta3,vbdry1,vbdry2,vbdry3)
      call opdiv     (respr,ta1,ta2,ta3)
      call ortho     (respr)

      return
      end
c-----------------------------------------------------------------------
      subroutine cresvuz (resv1,resv2,resv3)

c     Compute the residual for the velocity - UZAWA SCHEME.

      include 'SIZE'
      include 'GEOM'
      include 'SOLN'
      REAL           RESV1 (LX1,LY1,LZ1,1)
      REAL           RESV2 (LX1,LY1,LZ1,1)
      REAL           RESV3 (LX1,LY1,LZ1,1)
      COMMON /SCRMG/ TA1   (LX1,LY1,LZ1,LELV)
     $ ,             TA2   (LX1,LY1,LZ1,LELV)
     $ ,             TA3   (LX1,LY1,LZ1,LELV)
      COMMON /SCREV/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
C
      INLOC   = -1
      CALL SETHLM  (H1,H2,INLOC)
      CALL OPRZERO (RESV1,RESV2,RESV3)
      CALL OPGRADT (RESV1,RESV2,RESV3,PR)
      CALL OPADD2  (RESV1,RESV2,RESV3,BFX,BFY,BFZ)
      CALL OPHX    (TA1,TA2,TA3,VX,VY,VZ,H1,H2)
      CALL OPSUB2  (RESV1,RESV2,RESV3,TA1,TA2,TA3)
C
      return
      END
C
      subroutine makeg (out1,out2,out3,h1,h2,intype)
C----------------------------------------------------------------------
C
C     Compute inhomogeneities for the elliptic solver in the pressure
C     residual evaluation.
C     INTYPE =  0  steady state
C     INTYPE =  1  explicit, Euler forward
C     INTYPE = -1  implicit, Euler backward
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      REAL           OUT1  (LX1,LY1,LZ1,LELV)
      REAL           OUT2  (LX1,LY1,LZ1,LELV)
      REAL           OUT3  (LX1,LY1,LZ1,LELV)
      REAL           H1    (LX1,LY1,LZ1,LELV)
      REAL           H2    (LX1,LY1,LZ1,LELV)
      COMMON /SCRMG/ TA1   (LX1,LY1,LZ1,LELV)
     $              ,TA2   (LX1,LY1,LZ1,LELV)
     $              ,TA3   (LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/ TB1   (LX1,LY1,LZ1,LELV)
     $              ,TB2   (LX1,LY1,LZ1,LELV)
     $              ,TB3   (LX1,LY1,LZ1,LELV)
     $              ,HZERO (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
C
      CALL OPGRADT (OUT1,OUT2,OUT3,PR)
      CALL OPCHSGN (OUT1,OUT2,OUT3)
C
      IF ((INTYPE.EQ.0.).OR.(INTYPE.EQ.-1)) THEN
C
C        Steady state or implicit scheme
C
         CALL OPAMASK (TB1,TB2,TB3)
         CALL OPHX    (TA1,TA2,TA3,TB1,TB2,TB3,H1,H2)
         CALL OPADD2  (OUT1,OUT2,OUT3,TA1,TA2,TA3)
         CALL OPSUB2  (OUT1,OUT2,OUT3,BFX,BFY,BFZ)
C
      ELSEIF (INTYPE.EQ.1) THEN
C
C        Explicit scheme
C
         CALL RZERO   (HZERO,NTOT1)
         CALL OPHX    (TA1,TA2,TA3,VX,VY,VZ,H1,HZERO)
         CALL OPADD2  (OUT1,OUT2,OUT3,TA1,TA2,TA3)
         CALL OPSUB2  (OUT1,OUT2,OUT3,BFX,BFY,BFZ)
C
      ENDIF
C
      return
      END

c-----------------------------------------------------------------------
      subroutine ctolspl (tolspl,respr)
C
C     Compute the pressure tolerance
C
      include 'SIZE'
      include 'MASS'
      include 'TSTEP'
      REAL           RESPR (LX2,LY2,LZ2,LELV)
      COMMON /SCRMG/ WORK  (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
      CALL INVCOL3 (WORK,RESPR,BM1,NTOT1)
      CALL COL2    (WORK,RESPR,NTOT1)
      RINIT  = SQRT (GLSUM (WORK,NTOT1)/VOLVM1)
      IF (TOLPDF.GT.0.) THEN
         TOLSPL = TOLPDF
         TOLMIN = TOLPDF
      ELSE
         TOLSPL = TOLPS/DT
         TOLMIN = RINIT*PRELAX
      ENDIF
      IF (TOLSPL.LT.TOLMIN) THEN
         TOLOLD = TOLSPL
         TOLSPL = TOLMIN
         IF (NIO.EQ.0) 
     $   WRITE (6,*) 'Relax the pressure tolerance ',TOLSPL,TOLOLD
      ENDIF
      return
      end
c------------------------------------------------------------------------
      subroutine ortho (respr)

C     Orthogonalize the residual in the pressure solver with respect 
C     to (1,1,...,1)T  (only if all Dirichlet b.c.).

      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'TSTEP'
      real respr (lx2,ly2,lz2,lelv)
      integer*8 ntotg,nxyz2

      nxyz2 = lx2*ly2*lz2
      ntot  = nxyz2*nelv
      ntotg = nxyz2*nelgv

      if (ifield.eq.1) then
         if (ifvcor) then
            rlam  = glsum (respr,ntot)/ntotg
            call cadd (respr,-rlam,ntot)
         endif
       elseif (ifield.eq.ifldmhd) then
         if (ifbcor) then
            rlam = glsum (respr,ntot)/ntotg
            call cadd (respr,-rlam,ntot)
         endif
       else
         call exitti('ortho: unaccounted ifield = $',ifield)
      endif

      return
      end
c------------------------------------------------------------------------
      subroutine cdabdtp (ap,wp,h1,h2,h2inv,intype)

C     INTYPE= 0  Compute the matrix-vector product    DA(-1)DT*p
C     INTYPE= 1  Compute the matrix-vector product    D(B/DT)(-1)DT*p
C     INTYPE=-1  Compute the matrix-vector product    D(A+B/DT)(-1)DT*p

      include 'SIZE'
      include 'TOTAL'
      REAL           AP    (LX2,LY2,LZ2,1)
      REAL           WP    (LX2,LY2,LZ2,1)
      REAL           H1    (LX1,LY1,LZ1,1)
      REAL           H2    (LX1,LY1,LZ1,1)
      REAL           H2INV (LX1,LY1,LZ1,1)
C
      COMMON /SCRNS/ TA1 (LX1,LY1,LZ1,LELV)
     $ ,             TA2 (LX1,LY1,LZ1,LELV)
     $ ,             TA3 (LX1,LY1,LZ1,LELV)
     $ ,             TB1 (LX1,LY1,LZ1,LELV)
     $ ,             TB2 (LX1,LY1,LZ1,LELV)
     $ ,             TB3 (LX1,LY1,LZ1,LELV)

      call opgradt (ta1,ta2,ta3,wp)
      if ((intype.eq.0).or.(intype.eq.-1)) then
         tolhin=tolhs
         call ophinv (tb1,tb2,tb3,ta1,ta2,ta3,h1,h2,tolhin,nmxv)
      else
         if (ifanls) then
            dtbdi = dt/bd(1)   ! scale by dt*backwd-diff coefficient
            CALL OPBINV1(TB1,TB2,TB3,TA1,TA2,TA3,dtbdi)
         else
            CALL OPBINV (TB1,TB2,TB3,TA1,TA2,TA3,H2INV)
         endif
      endif
      call opdiv  (ap,tb1,tb2,tb3)

      return
      end
C
C-----------------------------------------------------------------------
      subroutine opgrad (out1,out2,out3,inp)
C---------------------------------------------------------------------
C
C     Compute OUTi = Di*INP, i=1,2,3. 
C     the gradient of the scalar field INP.
C     Note: OUTi is defined on the pressure mesh !!!
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'

      REAL OUT1 (LX2,LY2,LZ2,1)
      REAL OUT2 (LX2,LY2,LZ2,1)
      REAL OUT3 (LX2,LY2,LZ2,1)
      REAL INP  (LX1,LY1,LZ1,1)

      iflg = 0

      if (ifsplit .and. .not.ifaxis) then
         call wgradm1(out1,out2,out3,inp,nelv) ! weak grad on FLUID mesh
         return
      endif

      NTOT2 = lx2*ly2*lz2*NELV
      CALL MULTD (OUT1,INP,RXM2,SXM2,TXM2,1,iflg)
      CALL MULTD (OUT2,INP,RYM2,SYM2,TYM2,2,iflg)
      IF (ldim.EQ.3) 
     $CALL MULTD (OUT3,INP,RZM2,SZM2,TZM2,3,iflg)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine cdtp (dtx,x,rm2,sm2,tm2,isd)
C-------------------------------------------------------------
C
C     Compute DT*X (entire field)
C
C-------------------------------------------------------------
      include 'SIZE'
      include 'WZ'
      include 'DXYZ'
      include 'IXYZ'
      include 'GEOM'
      include 'MASS'
      include 'INPUT'
      include 'ESOLV'
C
      real dtx  (lx1*ly1*lz1,lelv)
      real x    (lx2*ly2*lz2,lelv)
      real rm2  (lx2*ly2*lz2,lelv)
      real sm2  (lx2*ly2*lz2,lelv)
      real tm2  (lx2*ly2*lz2,lelv)
C
      common /ctmp1/ wx  (lx1*ly1*lz1)
     $ ,             ta1 (lx1*ly1*lz1)
     $ ,             ta2 (lx1*ly1*lz1)
     $ ,             ta3 (lx1*ly1,lz1)

      REAL           DUAX(LX1)
c
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
      include 'CTIMER'

      integer e
C
#ifdef TIMER
      if (icalld.eq.0) tcdtp=0.0
      icalld=icalld+1
      ncdtp=icalld
      etime1=dnekclock()
#endif

      nxyz1 = lx1*ly1*lz1
      nxyz2 = lx2*ly2*lz2
      nyz2  = ly2*lz2
      nxy1  = lx1*ly1

      n1    = lx1*ly1
      n2    = lx1*ly2

      do e=1,nelv

C       Use the appropriate derivative- and interpolation operator in 
C       the y-direction (= radial direction if axisymmetric).
        if (ifaxis) then
         ly12   = ly1*ly2
         if (ifrzer(e)) then
            call copy (iym12,iam12,ly12)
            call copy (dym12,dam12,ly12)
            call copy (w3m2,w2am2,nxyz2)
         else
            call copy (iym12,icm12,ly12)
            call copy (dym12,dcm12,ly12)
            call copy (w3m2,w2cm2,nxyz2)
         endif
       endif
C
C      Collocate with weights
C
       if(ifsplit) then
         call col3 (wx,bm1(1,1,1,e),x(1,e),nxyz1)
         call invcol2(wx,jacm1(1,1,1,e),nxyz1)
       else
         if (.not.ifaxis) call col3 (wx,w3m2,x(1,e),nxyz2)
C
         if (ifaxis) then
            if (ifrzer(e)) then
                call col3    (wx,x(1,e),bm2(1,1,1,e),nxyz2)
                call invcol2 (wx,jacm2(1,1,1,e),nxyz2)
            else
                call col3    (wx,w3m2,x(1,e),nxyz2)
                call col2    (wx,ym2(1,1,1,e),nxyz2)
            endif
         endif
       endif
C
       if (ldim.eq.2) then
         if (.not.ifdfrm(e) .and. ifalgn(e)) then
C
            if (      ifrsxy(e).and.isd.eq.1  .or. 
     $           .not.ifrsxy(e).and.isd.eq.2) then
C
               call col3 (ta1,wx,rm2(1,e),nxyz2)
               call mxm  (dxtm12,lx1,ta1,lx2,ta2,nyz2)
               call mxm  (ta2,lx1,iym12,ly2,dtx(1,e),ly1)
            else
               call col3 (ta1,wx,sm2(1,e),nxyz2)
               call mxm  (ixtm12,lx1,ta1,lx2,ta2,nyz2)
               call mxm  (ta2,lx1,dym12,ly2,dtx(1,e),ly1)
            endif
         else
            call col3 (ta1,wx,rm2(1,e),nxyz2)
            call mxm  (dxtm12,lx1,ta1,lx2,ta2,nyz2)
            call mxm  (ta2,lx1,iym12,ly2,dtx(1,e),ly1)

            call col3 (ta1,wx,sm2(1,e),nxyz2)
            call mxm  (ixtm12,lx1,ta1,lx2,ta2,nyz2)
            call mxm  (ta2,lx1,dym12,ly2,ta1,ly1)

            call add2 (dtx(1,e),ta1,nxyz1)
         endif

       else
         if (ifsplit) then

            call col3 (ta1,wx,rm2(1,e),nxyz2)
            call mxm  (dxtm12,lx1,ta1,lx2,dtx(1,e),nyz2)
            call col3 (ta1,wx,sm2(1,e),nxyz2)
            i1 = 1
            i2 = 1
            do iz=1,lz2
               call mxm  (ta1(i2),lx1,dym12,ly2,ta2(i1),ly1)
               i1 = i1 + n1
               i2 = i2 + n2
            enddo
            call add2 (dtx(1,e),ta2,nxyz1)
            call col3 (ta1,wx,tm2(1,e),nxyz2)
            call mxm  (ta1,nxy1,dzm12,lz2,ta2,lz1)
            call add2 (dtx(1,e),ta2,nxyz1)

         else

            call col3 (ta1,wx,rm2(1,e),nxyz2)
            call mxm  (dxtm12,lx1,ta1,lx2,ta2,nyz2)
            i1 = 1
            i2 = 1
            do iz=1,lz2
               call mxm  (ta2(i2),lx1,iym12,ly2,ta1(i1),ly1)
               i1 = i1 + n1
               i2 = i2 + n2
            enddo
            call mxm  (ta1,nxy1,izm12,lz2,dtx(1,e),lz1)

            call col3 (ta1,wx,sm2(1,e),nxyz2)
            call mxm  (ixtm12,lx1,ta1,lx2,ta2,nyz2)
            i1 = 1
            i2 = 1
            do iz=1,lz2
               call mxm  (ta2(i2),lx1,dym12,ly2,ta1(i1),ly1)
               i1 = i1 + n1
               i2 = i2 + n2
            enddo
            call mxm  (ta1,nxy1,izm12,lz2,ta2,lz1)
            call add2 (dtx(1,e),ta2,nxyz1)

            call col3 (ta1,wx,tm2(1,e),nxyz2)
            call mxm  (ixtm12,lx1,ta1,lx2,ta2,nyz2)
            i1 = 1
            i2 = 1
            do iz=1,lz2
               call mxm  (ta2(i2),lx1,iym12,ly2,ta1(i1),ly1)
               i1 = i1 + n1
               i2 = i2 + n2
            enddo
            call mxm  (ta1,nxy1,dzm12,lz2,ta2,lz1)
            call add2 (dtx(1,e),ta2,nxyz1)

         endif

       endif         
C
C     If axisymmetric, add an extra diagonal term in the radial 
C     direction (only if solving the momentum equations and ISD=2)
C     NOTE: lz1=lz2=1
C
C
      if(ifsplit) then

       if (ifaxis.and.(isd.eq.4)) then
        call copy    (ta1,x(1,e),nxyz1)
        if (ifrzer(e)) THEN
           call rzero(ta1, lx1)
           call mxm  (x  (1,e),lx1,datm1,ly1,duax,1)
           call copy (ta1,duax,lx1)
        endif
        call col2    (ta1,baxm1(1,1,1,e),nxyz1)
        call add2    (dtx(1,e),ta1,nxyz1)
       endif

      else

       if (ifaxis.and.(isd.eq.2)) then
         call col3    (ta1,x(1,e),bm2(1,1,1,e),nxyz2)
         call invcol2 (ta1,ym2(1,1,1,e),nxyz2)
         call mxm     (ixtm12,lx1,ta1,lx2,ta2,ly2)
         call mxm     (ta2,lx1,iym12,ly2,ta1,ly1)
         call add2    (dtx(1,e),ta1,nxyz1)
       endif

      endif

      enddo
C
#ifdef TIMER
      tcdtp=tcdtp+(dnekclock()-etime1)
#endif
      return
      end
C
      subroutine multd (dx,x,rm2,sm2,tm2,isd,iflg)
C---------------------------------------------------------------------
C
C     Compute D*X
C     X    : input variable, defined on M1
C     DX   : output variable, defined on M2 (note: D is rectangular)   
C     RM2 : RXM2, RYM2 or RZM2
C     SM2 : SXM2, SYM2 or SZM2
C     TM2 : TXM2, TYM2 or TZM2
C     ISD : spatial direction (x=1,y=2,z=3)
C     IFLG: OPGRAD (iflg=0) or OPDIV (iflg=1)
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'WZ'
      include 'DXYZ'
      include 'IXYZ'
      include 'GEOM'
      include 'MASS'
      include 'INPUT'
      include 'ESOLV'

      real           dx   (lx2*ly2*lz2,lelv)
      real           x    (lx1*ly1*lz1,lelv)
      real           rm2  (lx2*ly2*lz2,lelv)
      real           sm2  (lx2*ly2*lz2,lelv)
      real           tm2  (lx2*ly2*lz2,lelv)

      common /ctmp1/ ta1 (lx1*ly1*lz1)
     $ ,             ta2 (lx1*ly1*lz1)
     $ ,             ta3 (lx1*ly1*lz1)

      real           duax(lx1)

      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
      include 'CTIMER'

      integer e
C
#ifdef TIMER
      if (icalld.eq.0) tmltd=0.0
      icalld=icalld+1
      nmltd=icalld
      etime1=dnekclock()
#endif

      nyz1  = ly1*lz1
      nxy2  = lx2*ly2
      nxyz1 = lx1*ly1*lz1
      nxyz2 = lx2*ly2*lz2

      n1    = lx2*ly1
      n2    = lx2*ly2

      do e=1,nelv

c        Use the appropriate derivative- and interpolation operator in 
c        the y-direction (= radial direction if axisymmetric).
         if (ifaxis) then
            ly12   = ly1*ly2
            if (ifrzer(e)) then
               call copy (iytm12,iatm12,ly12)
               call copy (dytm12,datm12,ly12)
               call copy (w3m2,w2am2,nxyz2)
            else
               call copy (iytm12,ictm12,ly12)
               call copy (dytm12,dctm12,ly12)
               call copy (w3m2,w2cm2,nxyz2)
            endif
         endif

         if (ldim.eq.2) then
            if (.not.ifdfrm(e) .and. ifalgn(e)) then
c
               if (      ifrsxy(e).and.isd.eq.1  .or. 
     $              .not.ifrsxy(e).and.isd.eq.2) then
                  call mxm     (dxm12,lx2,x(1,e),lx1,ta1,nyz1)
                  call mxm     (ta1,lx2,iytm12,ly1,dx(1,e),ly2)
                  call col2    (dx(1,e),rm2(1,e),nxyz2)
               else
                  call mxm     (ixm12,lx2,x(1,e),lx1,ta1,nyz1)
                  call mxm     (ta1,lx2,dytm12,ly1,dx(1,e),ly2)
                  call col2    (dx(1,e),sm2(1,e),nxyz2)
               endif
            else
               call mxm     (dxm12,lx2,x(1,e),lx1,ta1,nyz1)
               call mxm     (ta1,lx2,iytm12,ly1,dx(1,e),ly2)
               call col2    (dx(1,e),rm2(1,e),nxyz2)
               call mxm     (ixm12,lx2,x(1,e),lx1,ta1,nyz1)
               call mxm     (ta1,lx2,dytm12,ly1,ta3,ly2)
               call addcol3 (dx(1,e),ta3,sm2(1,e),nxyz2)
            endif

         else  ! 3D

c           if (ifsplit) then
c
c             call mxm  (dxm12,lx2,x(1,e),lx1,dx(1,e),nyz1)
c             call col2 (dx(1,e),rm2(1,e),nxyz2)
c             i1=1
c             i2=1
c             do iz=1,lz1
c                call mxm (x(1,e),lx2,dytm12,ly1,ta1(i2),ly2)
c                i1=i1+n1
c                i2=i2+n2
c             enddo
c             call addcol3 (dx(1,e),ta1,sm2(1,e),nxyz2)
c             call mxm (x(1,e),nxy2,dztm12,lz1,ta1,lz2)
c             call addcol3 (dx(1,e),ta1,tm2(1,e),nxyz2)

c           else ! PN - PN-2

             call mxm (dxm12,lx2,x(1,e),lx1,ta1,nyz1)
             i1=1
             i2=1
             do iz=1,lz1
               call mxm (ta1(i1),lx2,iytm12,ly1,ta2(i2),ly2)
               i1=i1+n1
               i2=i2+n2
             enddo
             call mxm  (ta2,nxy2,iztm12,lz1,dx(1,e),lz2)
             call col2 (dx(1,e),rm2(1,e),nxyz2)

             call mxm  (ixm12,lx2,x(1,e),lx1,ta3,nyz1) ! reuse ta3 below
             i1=1
             i2=1
             do iz=1,lz1
               call mxm (ta3(i1),lx2,dytm12,ly1,ta2(i2),ly2)
               i1=i1+n1
               i2=i2+n2
             enddo
             call mxm     (ta2,nxy2,iztm12,lz1,ta1,lz2)
             call addcol3 (dx(1,e),ta1,sm2(1,e),nxyz2)

c            call mxm (ixm12,lx2,x(1,e),lx1,ta1,nyz1) ! reuse ta3 from above
             i1=1
             i2=1
             do iz=1,lz1
               call mxm (ta3(i1),lx2,iytm12,ly1,ta2(i2),ly2)
               i1=i1+n1
               i2=i2+n2
             enddo
             call mxm (ta2,nxy2,dztm12,lz1,ta3,lz2)
             call addcol3 (dx(1,e),ta3,tm2(1,e),nxyz2)
c           endif
         endif
C
C        Collocate with the weights on the pressure mesh


       if(ifsplit) then
         call col2   (dx(1,e),bm1(1,1,1,e),nxyz1)
         call invcol2(dx(1,e),jacm1(1,1,1,e),nxyz1)
       else
         if (.not.ifaxis) call col2 (dx(1,e),w3m2,nxyz2)
         if (ifaxis) then
             if (ifrzer(e)) then
                 call col2    (dx(1,e),bm2(1,1,1,e),nxyz2)
                 call invcol2 (dx(1,e),jacm2(1,1,1,e),nxyz2)
             else
                 call col2    (dx(1,e),w3m2,nxyz2)
                 call col2    (dx(1,e),ym2(1,1,1,e),nxyz2)
             endif
         endif
       endif

c        If axisymmetric, add an extra diagonal term in the radial 
c        direction (ISD=2).
c        NOTE: lz1=lz2=1

      if(ifsplit) then

       if (ifaxis.and.(isd.eq.2).and.iflg.eq.1) then
        call copy    (ta3,x(1,e),nxyz1)
        if (ifrzer(e)) then
           call rzero(ta3, lx1)
           call mxm  (x(1,e),lx1,datm1,ly1,duax,1)
           call copy (ta3,duax,lx1)
        endif
        call col2    (ta3,baxm1(1,1,1,e),nxyz1)
        call add2    (dx(1,e),ta3,nxyz2)
       endif

      else

       if (ifaxis.and.(isd.eq.2)) then
            call mxm     (ixm12,lx2,x(1,e),lx1,ta1,ly1)
            call mxm     (ta1,lx2,iytm12,ly1,ta2,ly2)
            call col3    (ta3,bm2(1,1,1,e),ta2,nxyz2)
            call invcol2 (ta3,ym2(1,1,1,e),nxyz2)
            call add2    (dx(1,e),ta3,nxyz2)
       endif

      endif
C
      enddo
C
#ifdef TIMER
      tmltd=tmltd+(dnekclock()-etime1)
#endif
      return
      END
c-----------------------------------------------------------------------
      subroutine ophx (out1,out2,out3,inp1,inp2,inp3,h1,h2)
C----------------------------------------------------------------------
C
C     OUT = (H1*A+H2*B) * INP  
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      REAL OUT1 (LX1,LY1,LZ1,1)
      REAL OUT2 (LX1,LY1,LZ1,1)
      REAL OUT3 (LX1,LY1,LZ1,1)
      REAL INP1 (LX1,LY1,LZ1,1)
      REAL INP2 (LX1,LY1,LZ1,1)
      REAL INP3 (LX1,LY1,LZ1,1)
      REAL H1   (LX1,LY1,LZ1,1)
      REAL H2   (LX1,LY1,LZ1,1)
C
      IMESH = 1
C
      IF (IFSTRS) THEN
         MATMOD = 0
         CALL AXHMSF (OUT1,OUT2,OUT3,INP1,INP2,INP3,H1,H2,MATMOD)
      ELSE
         CALL AXHELM (OUT1,INP1,H1,H2,IMESH,1)
         CALL AXHELM (OUT2,INP2,H1,H2,IMESH,2)
         IF (ldim.EQ.3)
     $   CALL AXHELM (OUT3,INP3,H1,H2,IMESH,3)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine opbinv (out1,out2,out3,inp1,inp2,inp3,h2inv)
C--------------------------------------------------------------------
C
C     Compute OUT = (H2*B)-1 * INP   (explicit)
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
C
      REAL OUT1  (1)
      REAL OUT2  (1)
      REAL OUT3  (1)
      REAL INP1  (1)
      REAL INP2  (1)
      REAL INP3  (1)
      REAL H2INV (1)
C

      include 'OPCTR'
C
#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opbinv'
      endif
#endif
C
      call opmask  (inp1,inp2,inp3)
      call opdssum (inp1,inp2,inp3)
C
      NTOT=lx1*ly1*lz1*NELV
C
#ifdef TIMER
      isbcnt = ntot*(1+ldim)
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif

      call invcol3 (out1,bm1,h2inv,ntot)  ! this is expensive and should
      call dssum   (out1,lx1,ly1,lz1)     ! be changed (pff, 3/18/09)
      if (if3d) then
         do i=1,ntot
            tmp = 1./out1(i)
            out1(i)=inp1(i)*tmp
            out2(i)=inp2(i)*tmp
            out3(i)=inp3(i)*tmp
         enddo
      else
         do i=1,ntot
            tmp = 1./out1(i)
            out1(i)=inp1(i)*tmp
            out2(i)=inp2(i)*tmp
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine opbinv1(out1,out2,out3,inp1,inp2,inp3,SCALE)
C--------------------------------------------------------------------
C
C     Compute OUT = (B)-1 * INP   (explicit)
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
C
      REAL OUT1  (1)
      REAL OUT2  (1)
      REAL OUT3  (1)
      REAL INP1  (1)
      REAL INP2  (1)
      REAL INP3  (1)
C

      include 'OPCTR'
C
#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opbnv1'
      endif
#endif
C
      CALL OPMASK  (INP1,INP2,INP3)
      CALL OPDSSUM (INP1,INP2,INP3)
C
      NTOT=lx1*ly1*lz1*NELV
C
#ifdef TIMER
      isbcnt = ntot*(1+ldim)
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (IF3D) THEN
         DO 100 I=1,NTOT
            TMP    =BINVM1(I,1,1,1)*scale
            OUT1(I)=INP1(I)*TMP
            OUT2(I)=INP2(I)*TMP
            OUT3(I)=INP3(I)*TMP
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT
            TMP    =BINVM1(I,1,1,1)*scale
            OUT1(I)=INP1(I)*TMP
            OUT2(I)=INP2(I)*TMP
  200    CONTINUE
      ENDIF
C
      return
      END
C-----------------------------------------------------------------------
      subroutine uzprec (rpcg,rcg,h1m1,h2m1,intype,wp)
C--------------------------------------------------------------------
C
C     Uzawa preconditioner
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      include 'PARALLEL'
c
      REAL           RPCG (LX2,LY2,LZ2,LELV)
      REAL           RCG  (LX2,LY2,LZ2,LELV)
      REAL           WP   (LX2,LY2,LZ2,LELV)
      REAL           H1M1 (LX1,LY1,LZ1,LELV)
      REAL           H2M1 (LX1,LY1,LZ1,LELV)
      COMMON /SCRCH/ H1M2 (LX2,LY2,LZ2,LELV)
     $ ,             H2M2 (LX2,LY2,LZ2,LELV)
C
      integer kstep
      save    kstep
      data    kstep/-1/
c
      integer*8 ntotg,nxyz2
c
      NTOT2 = lx2*ly2*lz2*NELV
      if (istep.ne.kstep .and. .not.ifanls) then
         kstep=istep
         DO 100 IE=1,NELV
            CALL MAP12 (H1M2(1,1,1,IE),H1M1(1,1,1,IE),IE)
            CALL MAP12 (H2M2(1,1,1,IE),H2M1(1,1,1,IE),IE)
 100     CONTINUE
      endif
C
      IF (INTYPE.EQ.0) THEN
         CALL COL3        (WP,RCG,H1M2,NTOT2)
         CALL COL3        (RPCG,WP,BM2INV,NTOT2)
      ELSEIF (INTYPE.EQ.-1) THEN
         CALL EPREC       (WP,RCG)
         CALL COL2        (WP,H2M2,NTOT2)
         CALL COL3        (RPCG,RCG,BM2INV,NTOT2)
         CALL COL2        (RPCG,H1M2,NTOT2)
         CALL ADD2        (RPCG,WP,NTOT2)
      ELSEIF (INTYPE.EQ.1) THEN
         if (ifanls) then
            CALL EPREC2      (RPCG,RCG)
            DTBD = BD(1)/DT
            CALL cmult       (RPCG,DTBD,ntot2)
         else
            CALL EPREC2      (RPCG,RCG)
c           CALL COL2        (RPCG,H2M2,NTOT2)
         endif
      ELSE
         CALL COPY        (RPCG,RCG,NTOT2)
      ENDIF

      call ortho (rpcg)

      return
      end
C
      subroutine eprec (z2,r2)
C----------------------------------------------------------------
C
C     Precondition the explicit pressure operator (E) with
C     a Neumann type (H1) Laplace operator: JT*A*J.
C     Invert A by conjugate gradient iteration or multigrid.
C
C     NOTE: SCRNS is used.
C
C----------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'MASS'
      include 'PARALLEL'
      include 'TSTEP'
      REAL           Z2   (LX2,LY2,LZ2,LELV)
      REAL           R2   (LX2,LY2,LZ2,LELV)
      COMMON /SCRNS/ MASK (LX1,LY1,LZ1,LELV)
     $              ,R1   (LX1,LY1,LZ1,LELV)
     $              ,X1   (LX1,LY1,LZ1,LELV)
     $              ,W2   (LX2,LY2,LZ2,LELV)
     $              ,H1   (LX1,LY1,LZ1,LELV)
     $              ,H2   (LX1,LY1,LZ1,LELV)
      REAL    MASK
      COMMON /CPRINT/ IFPRINT, IFHZPC
      LOGICAL         IFPRINT, IFHZPC
      integer*8 ntotg,nxyz
 
      nxyz   = lx1*ly1*lz1
      ntotg  = nxyz*nelgv
      ntot1  = nxyz*nelv
      ntot2  = lx2*ly2*lz2*nelv
      nfaces = 2*ldim
C
C     Set the tolerance for the preconditioner
C
      CALL COL3 (W2,R2,BM2INV,NTOT2)
      RINIT  = SQRT(GLSC2(W2,R2,NTOT2)/VOLVM2)
      EPS    = 0.02
      TOL    = EPS*RINIT
c     if (param(91).gt.0) tol=param(91)*rinit
c     if (param(91).lt.0) tol=-param(91)
C
      DO 100 IEL=1,NELV
         CALL MAP21E (R1(1,1,1,IEL),R2(1,1,1,IEL),IEL)
 100  CONTINUE
C
      if (ifvcor) then
         otr1 = glsum (r1,ntot1)
         call rone  (x1,ntot1)
         c2   = -otr1/ntotg
         call add2s2 (r1,x1,c2,ntot1)
C
         OTR1 = GLSUM (R1,NTOT1)
         TOLMIN = 10.*ABS(OTR1)
         IF (TOL .LT. TOLMIN) THEN
             if (nid.eq.0) 
     $       write(6,*) 'Resetting tol in EPREC:(old,new)',tol,tolmin
             TOL = TOLMIN
         ENDIF
      ENDIF
C
      CALL RONE    (H1,NTOT1)
      CALL RZERO   (H2,NTOT1)
      IFHZPC = .TRUE.
      CALL HMHOLTZ ('PREC',X1,R1,H1,H2,PMASK,VMULT,IMESH,TOL,NMXP,1)
      IFHZPC = .FALSE.
C
      DO 200 IEL=1,NELV
         CALL MAP12 (Z2(1,1,1,IEL),X1(1,1,1,IEL),IEL)
 200  CONTINUE
C
      return
      END
C
      subroutine convprn (iconv,rbnorm,rrpt,res,z,tol)
C-----------------------------------------------------------------
C                                               T
C     Convergence test for the pressure step;  r z
C
C-----------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
      REAL           RES  (1)
      REAL           Z    (1)
      REAL           wrk1(2),wrk2(2)
c
      ntot2   = lx2*ly2*lz2*nelv
      wrk1(1) = vlsc21 (res,bm2inv,ntot2)  !  res*bm2inv*res
      wrk1(2) = vlsc2  (res,z     ,ntot2)  !  res*z
      call gop(wrk1,wrk2,'+  ',2)
      rbnorm  = sqrt(wrk1(1)/volvm2)
      rrpt    = wrk1(2)
c
c     CALL CONVPR (RCG,tolpss,ICONV,RNORM)
c     RRP1 = GLSC2 (RPCG,RCG,NTOT2)
c     RBNORM = SQRT (GLSC2 (BM2,TB,NTOT2)/VOLVM2)
c
      ICONV  = 0
      IF (RBNORM.LT.TOL) ICONV=1
      return
      END
C
      subroutine convpr (res,tol,iconv,rbnorm)
C-----------------------------------------------------------------
C
C     Convergence test for the pressure step
C
C-----------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
      REAL           RES  (LX2,LY2,LZ2,LELV)
      COMMON /SCRMG/ TA   (LX2,LY2,LZ2,LELV)
     $ ,             TB   (LX2,LY2,LZ2,LELV)
C
      RBNORM = 0.
      NTOT2  = lx2*ly2*lz2*NELV
      CALL COL3     (TA,RES,BM2INV,NTOT2)
      CALL COL3     (TB,TA,TA,NTOT2)
      RBNORM = SQRT (GLSC2 (BM2,TB,NTOT2)/VOLVM2)
c
      ICONV  = 0
      IF (RBNORM.LT.TOL) ICONV=1
      return
      END
C
      subroutine chktcg2 (tol,res,iconv)
C-------------------------------------------------------------------
C
C     Check that the tolerances are not too small for the CG-solver.
C     Important when calling the CG-solver (Gauss  mesh) with
C     all Dirichlet velocity b.c. (zero Neumann for the pressure).
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      REAL           RES (LX2,LY2,LZ2,LELV)
      COMMON /CTMP0/ TA  (LX2,LY2,LZ2,LELV)
     $ ,             TB  (LX2,LY2,LZ2,LELV)
      COMMON /CPRINT/ IFPRINT
      LOGICAL         IFPRINT
C
      ICONV = 0
C
C     Single or double precision???
C
      DELTA = 1.E-9
      X     = 1.+DELTA
      Y     = 1.
      DIFF  = ABS(X-Y)
      IF (DIFF.EQ.0.) EPS = 1.E-5
      IF (DIFF.GT.0.) EPS = 1.E-10
C
C     Relaxed pressure iteration; maximum decrease in the residual (ER)
C
      IF (PRELAX.NE.0.) EPS = PRELAX
C
      NTOT2 = lx2*ly2*lz2*NELV
      CALL COL3     (TA,RES,BM2INV,NTOT2)
      CALL COL3     (TB,TA,TA,NTOT2)
      CALL COL2     (TB,BM2,NTOT2)
      RINIT = SQRT( GLSUM (TB,NTOT2)/VOLVM2 )
      IF (RINIT.LT.TOL) THEN
         ICONV = 1
         return
      ENDIF
      IF (TOLPDF.GT.0.) THEN
         RMIN = TOLPDF
      ELSE
         RMIN  = EPS*RINIT
      ENDIF
      IF (TOL.LT.RMIN) THEN
         TOLOLD = TOL
         TOL = RMIN
         IF (NIO.EQ.0 .AND. IFPRINT) WRITE (6,*)
     $   'New CG2-tolerance (RINIT*10-5/10-10) = ',TOL,TOLOLD
      ENDIF
      IF (IFVCOR) THEN
         OTR = GLSUM (RES,NTOT2)
         TOLMIN = ABS(OTR)*100.
         IF (TOL .LT. TOLMIN) THEN
             TOLOLD = TOL
             TOL = TOLMIN
             IF (NIO.EQ.0 .AND. IFPRINT)
     $       WRITE (6,*) 'New CG2-tolerance (OTR) = ',TOLMIN,TOLOLD
         ENDIF
      ENDIF
      return
      END
C
      subroutine dudxyz (du,u,rm1,sm1,tm1,jm1,imsh,isd)
C--------------------------------------------------------------
C
C     DU   - dU/dx or dU/dy or dU/dz
C     U    - a field variable defined on mesh 1
C     RM1  - dr/dx or dr/dy or dr/dz  
C     SM1  - ds/dx or ds/dy or ds/dz
C     TM1  - dt/dx or dt/dy or dt/dz
C     JM1  - the Jacobian   
C     IMESH - topology: velocity (1) or temperature (2) mesh
C
C--------------------------------------------------------------
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
C
      REAL  DU  (LX1,LY1,LZ1,1)
      REAL  U   (LX1,LY1,LZ1,1)
      REAL  RM1 (LX1,LY1,LZ1,1)
      REAL  SM1 (LX1,LY1,LZ1,1)
      REAL  TM1 (LX1,LY1,LZ1,1)
      REAL  JM1 (LX1,LY1,LZ1,1)
C
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
C
      REAL  DRST(LX1,LY1,LZ1)
C
      IF (imsh.EQ.1) NEL = NELV
      IF (imsh.EQ.2) NEL = NELT
      NXY1  = lx1*ly1
      NYZ1  = ly1*lz1
      NXYZ1 = lx1*ly1*lz1
      NTOT  = NXYZ1*NEL

      DO 1000 IEL=1,NEL
C
      IF (IFAXIS) CALL SETAXDY (IFRZER(IEL) )
C
      IF (ldim.EQ.2) THEN
            CALL MXM     (DXM1,lx1,U(1,1,1,IEL),lx1,DU(1,1,1,IEL),NYZ1)
            CALL COL2    (DU(1,1,1,IEL),RM1(1,1,1,IEL),NXYZ1)
            CALL MXM     (U(1,1,1,IEL),lx1,DYTM1,ly1,DRST,ly1)
            CALL ADDCOL3 (DU(1,1,1,IEL),DRST,SM1(1,1,1,IEL),NXYZ1)
      ELSE
            CALL MXM   (DXM1,lx1,U(1,1,1,IEL),lx1,DU(1,1,1,IEL),NYZ1)
            CALL COL2  (DU(1,1,1,IEL),RM1(1,1,1,IEL),NXYZ1)
            DO 20 IZ=1,lz1
               CALL MXM  (U(1,1,IZ,IEL),lx1,DYTM1,ly1,DRST(1,1,IZ),ly1)
 20         CONTINUE
            CALL ADDCOL3 (DU(1,1,1,IEL),DRST,SM1(1,1,1,IEL),NXYZ1)
            CALL MXM     (U(1,1,1,IEL),NXY1,DZTM1,lz1,DRST,lz1)
            CALL ADDCOL3 (DU(1,1,1,IEL),DRST,TM1(1,1,1,IEL),NXYZ1)
      ENDIF
C
 1000 CONTINUE
C
c     CALL INVCOL2 (DU,JM1,NTOT)
      CALL COL2 (DU,JACMI,NTOT)
C
      return
      END
C
      subroutine convopo (conv,fi)
C--------------------------------------------------------------------
C
C     Compute the convective term CONV for a passive scalar field FI
C     using the skew-symmetric formulation.
C     The field variable FI is defined on mesh M1 (GLL) and
C     the velocity field is assumed given.
C
C     IMPORTANT NOTE: Use the scratch-arrays carefully!!!!!
C
C     The common-block SCRNS is used in CONV1 and CONV2.
C     The common-blocks CTMP0 and CTMP1 are also used as scratch-arrays
C     since there is no direct stiffness summation or Helmholtz-solves. 
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
C
C     Use the common blocks CTMP0 and CTMP1 as work space.
C
      COMMON /SCRCH/  CMASK1 (LX1,LY1,LZ1,LELV)
     $ ,              CMASK2 (LX1,LY1,LZ1,LELV)
      COMMON /CTMP1/  MFI    (LX1,LY1,LZ1,LELV)
     $ ,              DMFI   (LX1,LY1,LZ1,LELV)
     $ ,              MDMFI  (LX1,LY1,LZ1,LELV)
      REAL   MFI,DMFI,MDMFI
C
C     Arrays in parameter list
C
      REAL    CONV (LX1,LY1,LZ1,1) 
      REAL    FI   (LX1,LY1,LZ1,1)
C
C
      NXYZ1 = lx1*ly1*lz1
      NTOT1 = lx1*ly1*lz1*NELV
      NTOTZ = lx1*ly1*lz1*NELT
C
      CALL RZERO  (CONV,NTOTZ)
C
      IF (PARAM(86).EQ.0.0) THEN
C        Always use the convective form !!! (ER)
         CALL CONV1 (CONV,FI)
      ELSE
C        Use skew-symmetric form
C
C        Generate operator mask arrays CMASK1 and CMASK2
C
         CALL CMASK  (CMASK1,CMASK2)
C
         CALL COL3   (MFI,FI,CMASK1,NTOT1)
         CALL CONV1  (DMFI,MFI)
         CALL COL3   (MDMFI,DMFI,CMASK1,NTOT1)
         CALL ADD2S2 (CONV,MDMFI,0.5,NTOT1)
C
         CALL COL3   (MDMFI,DMFI,CMASK2,NTOT1)
         CALL ADD2   (CONV,MDMFI,NTOT1)      
C
         CALL CONV2  (DMFI,MFI)
         CALL COL3   (MDMFI,DMFI,CMASK1,NTOT1)
         CALL ADD2S2 (CONV,MDMFI,-0.5,NTOT1)
C
         CALL COL3   (MFI,FI,CMASK2,NTOT1)
         CALL CONV1  (DMFI,MFI)
         CALL COL3   (MDMFI,DMFI,CMASK2,NTOT1)
         CALL ADD2S2 (CONV,MDMFI,0.5,NTOT1)
C
         CALL CONV2  (DMFI,MFI)
         CALL COL3   (MDMFI,DMFI,CMASK1,NTOT1)
         CALL ADD2S2 (CONV,MDMFI,-1.,NTOT1)
C
         CALL COL3   (MDMFI,DMFI,CMASK2,NTOT1)
         CALL ADD2S2 (CONV,MDMFI,0.5,NTOT1)      
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine conv2 (dtfi,fi)
C--------------------------------------------------------------------
C
C     Compute DT*FI (part of the convection operator)
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      REAL           DTFI (LX1,LY1,LZ1,1) 
      REAL           FI   (LX1,LY1,LZ1,1)
      COMMON /SCRNS/ TA1  (LX1,LY1,LZ1,LELV)
     $ ,             TA2  (LX1,LY1,LZ1,LELV)
     $ ,             TA3  (LX1,LY1,LZ1,LELV)
     $ ,             TB1  (LX1,LY1,LZ1,LELV)
     $ ,             TB2  (LX1,LY1,LZ1,LELV)
     $ ,             TB3  (LX1,LY1,LZ1,LELV)
C
      NXY1  = lx1*ly1
      NYZ1  = ly1*lz1
      NXYZ1 = lx1*ly1*lz1
      NTOT1 = lx1*ly1*lz1*NELV
C
      CALL RZERO(DTFI,NTOT1)
C
      IF (ldim .EQ. 2) THEN
C
C     2-dimensional case
C
      CALL COL4 (TA1,RXM1,VX,FI,NTOT1)
      CALL COL4 (TA2,RYM1,VY,FI,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      DO 200 IEL=1,NELV
         CALL COL2    (TA1(1,1,1,IEL),W3M1,NXYZ1)
         CALL MXM     (DXTM1,lx1,TA1(1,1,1,IEL),lx1,TB1(1,1,1,IEL),ly1)
 200  CONTINUE
      CALL COPY(DTFI,TB1,NTOT1)
C
      CALL COL4 (TA1,SXM1,VX,FI,NTOT1)
      CALL COL4 (TA2,SYM1,VY,FI,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      DO 300 IEL=1,NELV
         CALL COL2    (TA1(1,1,1,IEL),W3M1,NXYZ1)
         CALL MXM     (TA1(1,1,1,IEL),lx1,DYM1,ly1,TB1(1,1,1,IEL),ly1)
 300  CONTINUE
      CALL ADD2    (DTFI,TB1,NTOT1)
      CALL INVCOL2 (DTFI,BM1,NTOT1)
C
      ELSE
C
C     3-dimensional case
C      
      CALL COL4 (TA1,RXM1,VX,FI,NTOT1)
      CALL COL4 (TA2,RYM1,VY,FI,NTOT1)
      CALL COL4 (TA3,RZM1,VZ,FI,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      CALL ADD2 (TA1,TA3,NTOT1)
      DO 600 IEL=1,NELV
         CALL COL2    (TA1(1,1,1,IEL),W3M1,NXYZ1)
         CALL MXM     (DXTM1,lx1,TA1(1,1,1,IEL),lx1,TB1(1,1,1,IEL),NYZ1)
 600  CONTINUE
      CALL COPY(DTFI,TB1,NTOT1)
C
      CALL COL4 (TA1,SXM1,VX,FI,NTOT1)
      CALL COL4 (TA2,SYM1,VY,FI,NTOT1)
      CALL COL4 (TA3,SZM1,VZ,FI,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      CALL ADD2 (TA1,TA3,NTOT1)
      DO 700 IEL=1,NELV
         CALL COL2    (TA1(1,1,1,IEL),W3M1,NXYZ1)
         DO 710 IZ=1,lz1
            CALL MXM (TA1(1,1,IZ,IEL),lx1,DYM1,ly1,TB1(1,1,IZ,IEL),ly1)
 710     CONTINUE
 700  CONTINUE
      CALL ADD2    (DTFI,TB1,NTOT1)
C
      CALL COL4 (TA1,TXM1,VX,FI,NTOT1)
      CALL COL4 (TA2,TYM1,VY,FI,NTOT1)
      CALL COL4 (TA3,TZM1,VZ,FI,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      CALL ADD2 (TA1,TA3,NTOT1)
      DO 800 IEL=1,NELV
         CALL COL2    (TA1(1,1,1,IEL),W3M1,NXYZ1)
         CALL MXM     (TA1(1,1,1,IEL),NXY1,DZM1,lz1,TB1(1,1,1,IEL),lz1)
 800  CONTINUE
      CALL ADD2    (DTFI,TB1,NTOT1)
      CALL INVCOL2 (DTFI,BM1,NTOT1)
C
      ENDIF
C
      return
      END
C
      subroutine cmask (cmask1,cmask2)
C---------------------------------------------------------------
C
C     Generate maskarrays for the convection operator
C
C       CMASK1   =  1.0 in the interior
C                =  0.0 at outflow
C       CMASK2   =  0.0 in the interior
C                =  1.0 at outflow
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
C
      REAL CMASK1 (LX1,LY1,LZ1,LELV)
      REAL CMASK2 (LX1,LY1,LZ1,LELV)
C
      CHARACTER CB*1
C
      NTOT1  = lx1*ly1*lz1*NELV
      NFACES = 2*ldim
      CALL CFILL (CMASK1,1.,NTOT1)
      CALL CFILL (CMASK2,0.,NTOT1)
      DO 100 IEL=1,NELV
      DO 100 IFACE=1,NFACES
         CB = CBC (IFACE,IEL,IFIELD)
         IF (CB.EQ.'O' .OR. CB.EQ.'o') THEN
            CALL FACEV (CMASK1,IEL,IFACE,0.,lx1,ly1,lz1)
            CALL FACEV (CMASK2,IEL,IFACE,1.,lx1,ly1,lz1)
         ENDIF
 100  CONTINUE
      return
      END
C
      subroutine makef
C---------------------------------------------------------------------
C
C     Compute and add: (1) user specified forcing function (FX,FY,FZ)
C                      (2) driving force due to natural convection
C                      (3) convection term
C
C     !! NOTE: Do not change the arrays BFX, BFY, BFZ until the
C              current time step is completed.
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'INPUT'
      include 'TSTEP'
      include 'CTIMER'
      include 'MVGEOM'

      etime1 = dnekclock()
                                                call makeuf
      if (filterType.eq.2)                      call make_hpf
      if (ifexplvis.and.ifsplit)                call makevis
      if (ifnav .and..not.ifchar)               call advab
      if (ifmvbd.and..not.ifchar)               call admeshv
      if (iftran)                               call makeabf
      if ((iftran.and..not.ifchar).or.
     $    (iftran.and..not.ifnav.and.ifchar))   call makebdf
      if (ifnav.and.ifchar)                     call advchar

c     Adding this call allows prescribed pressure bc for PnPn-2
c     if (.not.ifsplit.and..not.ifstrs)         call bcneutr
      
      tmakf=tmakf+(dnekclock()-etime1)

      return
      end
c
      subroutine makeuf
C---------------------------------------------------------------------
C
C     Compute and add: (1) user specified forcing function (FX,FY,FZ)
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      TIME = TIME-DT
      CALL NEKUF   (BFX,BFY,BFZ)
      CALL OPCOLV  (BFX,BFY,BFZ,BM1)
      TIME = TIME+DT
C
      return
      END
C
      subroutine nekuf (f1,f2,f3)
      include 'SIZE'
      include 'PARALLEL'
      include 'NEKUSE'
      include 'CTIMER'

      REAL F1 (LX1,LY1,LZ1,LELV)
      REAL F2 (LX1,LY1,LZ1,LELV)
      REAL F3 (LX1,LY1,LZ1,LELV)

#ifdef TIMER
      etime1=dnekclock_sync()
#endif

      CALL OPRZERO (F1,F2,F3)
      DO 100 IEL=1,NELV
         ielg = lglel(iel)
         DO 100 K=1,lz1
         DO 100 J=1,ly1
         DO 100 I=1,lx1
            if (optlevel.le.2) CALL NEKASGN (I,J,K,IEL)
            CALL USERF   (I,J,K,IELG)
            F1(I,J,K,IEL) = FFX
            F2(I,J,K,IEL) = FFY
            F3(I,J,K,IEL) = FFZ
 100  CONTINUE

#ifdef TIMER
      tusfq=tusfq+(dnekclock()-etime1)
#endif

      return
      END
C
      subroutine advab
C---------------------------------------------------------------
C
C     Eulerian scheme, add convection term to forcing function 
C     at current time step.
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      COMMON /SCRUZ/ TA1 (LX1,LY1,LZ1,LELV)
     $ ,             TA2 (LX1,LY1,LZ1,LELV)
     $ ,             TA3 (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
      CALL CONVOP  (TA1,VX)
      CALL CONVOP  (TA2,VY)
      CALL SUBCOL3 (BFX,BM1,TA1,NTOT1)
      CALL SUBCOL3 (BFY,BM1,TA2,NTOT1)
      IF (ldim.EQ.2) THEN
         CALL RZERO (TA3,NTOT1)
      ELSE
         CALL CONVOP  (TA3,VZ)
         CALL SUBCOL3 (BFZ,BM1,TA3,NTOT1)
      ENDIF
C

      return
      END
c-----------------------------------------------------------------------
      subroutine makebdf
C
C     Add contributions to F from lagged BD terms.
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
      NTOT1 = lx1*ly1*lz1*NELV
      CONST = 1./DT

      if(iflomach) then
        call cfill(h2,CONST,ntot1)
      else
        call cmult2(h2,vtrans(1,1,1,1,ifield),const,ntot1)
      endif

      CALL OPCOLV3c (TB1,TB2,TB3,VX,VY,VZ,BM1,bd(2))
C
      DO 100 ILAG=2,NBD
         IF (IFGEOM) THEN
            CALL OPCOLV3c(TA1,TA2,TA3,VXLAG (1,1,1,1,ILAG-1),
     $                                VYLAG (1,1,1,1,ILAG-1),
     $                                VZLAG (1,1,1,1,ILAG-1),
     $                                BM1LAG(1,1,1,1,ILAG-1),bd(ilag+1))
         ELSE
            CALL OPCOLV3c(TA1,TA2,TA3,VXLAG (1,1,1,1,ILAG-1),
     $                                VYLAG (1,1,1,1,ILAG-1),
     $                                VZLAG (1,1,1,1,ILAG-1),
     $                                BM1                   ,bd(ilag+1))
         ENDIF
         CALL OPADD2  (TB1,TB2,TB3,TA1,TA2,TA3)
 100  CONTINUE
      CALL OPADD2col (BFX,BFY,BFZ,TB1,TB2,TB3,h2)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine makeabf
C-----------------------------------------------------------------------
C
C     Sum up contributions to kth order extrapolation scheme.
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
C
      COMMON /SCRUZ/ TA1 (LX1,LY1,LZ1,LELV)
     $ ,             TA2 (LX1,LY1,LZ1,LELV)
     $ ,             TA3 (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
C
      AB0 = AB(1)
      AB1 = AB(2)
      AB2 = AB(3)
      CALL ADD3S2 (TA1,ABX1,ABX2,AB1,AB2,NTOT1)
      CALL ADD3S2 (TA2,ABY1,ABY2,AB1,AB2,NTOT1)
      CALL COPY   (ABX2,ABX1,NTOT1)
      CALL COPY   (ABY2,ABY1,NTOT1)
      CALL COPY   (ABX1,BFX,NTOT1)
      CALL COPY   (ABY1,BFY,NTOT1)
      CALL ADD2S1 (BFX,TA1,AB0,NTOT1)
      CALL ADD2S1 (BFY,TA2,AB0,NTOT1)
      if(.not.iflomach) CALL COL2   (BFX,VTRANS,NTOT1)          ! multiply by density
      if(.not.iflomach) CALL COL2   (BFY,VTRANS,NTOT1)
      IF (ldim.EQ.3) THEN
         CALL ADD3S2 (TA3,ABZ1,ABZ2,AB1,AB2,NTOT1)
         CALL COPY   (ABZ2,ABZ1,NTOT1)
         CALL COPY   (ABZ1,BFZ,NTOT1)
         CALL ADD2S1 (BFZ,TA3,AB0,NTOT1)
         if(.not.iflomach) CALL COL2   (BFZ,VTRANS,NTOT1)
      ENDIF
C
      return
      END
C
c-----------------------------------------------------------------------
      subroutine setab3 (ab0,ab1,ab2)
C
C     Set coefficients for 3rd order Adams-Bashforth scheme
C     (variable time step).
C
      include 'SIZE'
      include 'TSTEP'
C
      IF (ISTEP.LE.2) THEN
         AB0 = 1.
         AB1 = 0.
         AB2 = 0.
      ELSE
         DT0 = DTLAG(1)
         DT1 = DTLAG(2)
         DT2 = DTLAG(3)
         AB2  = (DT0/DT2)*((DT0/3.+DT1/2.)/(DT1+DT2))
         AB1  = -(DT0/DT1)*(0.5+(DT0/3.+DT1/2.)/DT2)
         AB0  = 1.-AB1-AB2 
      ENDIF
      return
      END
C
      subroutine setabbd (ab,dtlag,nab,nbd)
C-----------------------------------------------------------------------
C
C     Compute Adams-Bashforth coefficients (order NAB, less or equal to 3)
C     
C     NBD .EQ. 1
C     Standard Adams-Bashforth coefficients 
C
C     NBD .GT. 1
C     Modified Adams-Bashforth coefficients to be used in con-
C     junction with Backward Differentiation schemes (order NBD)
C
C-----------------------------------------------------------------------
      REAL AB(NAB),DTLAG(NAB)
C
      DT0 = DTLAG(1)
      DT1 = DTLAG(2)
      DT2 = DTLAG(3)
C
      IF ( NAB.EQ.1 ) THEN
C
          AB(1) = 1.0
C
      ELSEIF ( NAB.EQ.2 ) THEN
C
          DTA =  DT0/DT1
C
          IF ( NBD.EQ.1 ) THEN
C
          AB(2) = -0.5*DTA
          AB(1) =  1.0 - AB(2)
C
          ELSEIF ( NBD.EQ.2 ) THEN
C
          AB(2) = -DTA
          AB(1) =  1.0 - AB(2)
C
          ENDIF
C
      ELSEIF ( NAB.EQ.3 ) THEN
C
          DTS =  DT1 + DT2
          DTA =  DT0 / DT1
          DTB =  DT1 / DT2
          DTC =  DT0 / DT2
          DTD =  DTS / DT1
          DTE =  DT0 / DTS
C
          IF ( NBD.EQ.1 ) THEN
C
          AB(3) =  DTE*( 0.5*DTB + DTC/3. )
          AB(2) = -0.5*DTA - AB(3)*DTD
          AB(1) =  1.0 - AB(2) - AB(3)
C
          ELSEIF ( NBD.EQ.2 ) THEN
C
          AB(3) =  2./3.*DTC*(1./DTD + DTE)
          AB(2) = -DTA - AB(3)*DTD
          AB(1) =  1.0 - AB(2) - AB(3)
C
          ELSEIF ( NBD.EQ.3 ) THEN
C
          AB(3) =  DTE*(DTB + DTC)
          AB(2) = -DTA*(1.0 + DTB + DTC)
          AB(1) =  1.0 - AB(2) - AB(3)
C
          ENDIF
C
      ENDIF
C
      return
      END
C
      subroutine setbd (bd,dtbd,nbd)
C-----------------------------------------------------------------------
C
C     Compute bacward-differentiation coefficients of order NBD
C
C-----------------------------------------------------------------------
      PARAMETER (ldim = 10)
      REAL BDMAT(ldim,ldim),BDRHS(ldim)
      INTEGER IR(ldim),IC(ldim)
      REAL BD(1),DTBD(1)
C
      IF (NBD.EQ.1) THEN
         BD(1) = 1.
         BDF   = 1.
      ELSEIF (NBD.GE.2) THEN
         NSYS = NBD+1
         CALL BDSYS (BDMAT,BDRHS,DTBD,NBD,ldim)
         CALL LU    (BDMAT,NSYS,ldim,IR,IC)
         CALL SOLVE (BDRHS,BDMAT,1,NSYS,ldim,IR,IC)
         DO 30 I=1,NBD
            BD(I) = BDRHS(I)
 30      CONTINUE
         BDF = BDRHS(NBD+1)
      ENDIF
C
C     Normalize
C
      DO 100 IBD=NBD,1,-1
         BD(IBD+1) = BD(IBD)
 100  CONTINUE
      BD(1) = 1.
      DO 200 IBD=1,NBD+1
         BD(IBD) = BD(IBD)/BDF
 200  CONTINUE
c     write(6,1) (bd(k),k=1,nbd+1)
c   1 format('bd:',1p8e13.5)
C
      return
      END
C
      subroutine bdsys (a,b,dt,nbd,ldim)
      REAL A(ldim,9),B(9),DT(9)
      CALL RZERO (A,ldim**2)
      N = NBD+1
      DO 10 J=1,NBD
         A(1,J) = 1.
 10   CONTINUE
      A(1,NBD+1) = 0.
      B(1) = 1.
      DO 20 J=1,NBD
         SUMDT = 0.
         DO 25 K=1,J
            SUMDT = SUMDT+DT(K)
 25      CONTINUE
         A(2,J) = SUMDT
 20   CONTINUE
      A(2,NBD+1) = -DT(1)
      B(2) = 0.
      DO 40 I=3,NBD+1
      DO 30 J=1,NBD
         SUMDT = 0.
         DO 35 K=1,J
            SUMDT = SUMDT+DT(K)
 35      CONTINUE
         A(I,J) = SUMDT**(I-1)
 30   CONTINUE
      A(I,NBD+1) = 0.
      B(I) = 0.
 40   CONTINUE
      return
      END
C
      subroutine tauinit (tau,ilag)
C-------------------------------------------------------------------
C
C     Set initial time for subintegration 
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'TSTEP'
      TAU   = 0.
      DO 10 I=NBD,ILAG+1,-1
         TAU = TAU+DTLAG(I)
 10   CONTINUE
      return
      END
C
      subroutine velinit (vel1,vel2,vel3,ilag)
C-------------------------------------------------------------------
C
C     Set initial conditions for subintegration
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      REAL VEL1 (LX1,LY1,LZ1,LELV)
      REAL VEL2 (LX1,LY1,LZ1,LELV)
      REAL VEL3 (LX1,LY1,LZ1,LELV)
      IF (ILAG.EQ.1) THEN
         CALL OPCOPY (VEL1,VEL2,VEL3,VX,VY,VZ)
      ELSE
         CALL OPCOPY (VEL1,VEL2,VEL3,VXLAG(1,1,1,1,ILAG-1)
     $                              ,VYLAG(1,1,1,1,ILAG-1)
     $                              ,VZLAG(1,1,1,1,ILAG-1) )
      ENDIF
      return
      END
C
      subroutine velconv (vxn,vyn,vzn,tau)
C--------------------------------------------------------------------
C
C     Compute convecting velocity field (linearization)
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'
      REAL VXN (LX1,LY1,LZ1,LELV)
      REAL VYN (LX1,LY1,LZ1,LELV)
      REAL VZN (LX1,LY1,LZ1,LELV)
      CALL VELCHAR (VX,VXN,VXLAG,NBD,TAU,DTLAG)
      CALL VELCHAR (VY,VYN,VYLAG,NBD,TAU,DTLAG)
      IF (ldim.EQ.3) 
     $CALL VELCHAR (VZ,VZN,VZLAG,NBD,TAU,DTLAG)
      return
      END
C
      subroutine frkconv (y,x,mask)
C--------------------------------------------------------------------
C
C     Evaluate right-hand-side for Runge-Kutta scheme in the case of
C     pure convection.
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
      include 'TSTEP'
      REAL Y    (LX1,LY1,LZ1,1)
      REAL X    (LX1,LY1,LZ1,1)
      REAL MASK (LX1,LY1,LZ1,1)
C
      IF (IMESH.EQ.1) NEL=NELV
      IF (IMESH.EQ.2) NEL=NELT
      NTOT1 = lx1*ly1*lz1*NEL
      CALL CONVOP (Y,X)
      CALL COL2   (Y,BM1,NTOT1)
      CALL DSSUM  (Y,lx1,ly1,lz1)
      IF (IMESH.EQ.1) CALL COL2 (Y,BINVM1,NTOT1)
      IF (IMESH.EQ.2) CALL COL2 (Y,BINTM1,NTOT1)
      CALL COL2   (Y,MASK,NTOT1)
C
      return
      END
C
      subroutine velchar (vel,vn,vlag,nbd,tau,dtbd)
C-----------------------------------------------------------------------
C
C     Compute linearized velocity field.
C
C-----------------------------------------------------------------------
      include 'SIZE'
      REAL VEL  (LX1,LY1,LZ1,LELV)
      REAL VN   (LX1,LY1,LZ1,LELV)
      REAL VLAG (LX1,LY1,LZ1,LELV,9)
      REAL DTBD (NBD)
C
      NTOT1 = lx1*ly1*lz1*NELV
      IF (NBD.EQ.1) THEN
         CALL COPY (VEL,VN,NTOT1)
         return
      ELSEIF (NBD.EQ.2) THEN
         C1 = TAU/DTBD(2)
         C2 = 1.-C1
         CALL ADD3S2 (VEL,VN,VLAG,C1,C2,NTOT1)
      ELSEIF (NBD.EQ.3) THEN
         F1 = TAU**2-DTBD(3)*TAU
         F2 = TAU**2-(DTBD(2)+DTBD(3))*TAU
         F3 = DTBD(2)*DTBD(3)
         F4 = DTBD(2)*(DTBD(2)+DTBD(3))
         R1 = F2/F3
         R2 = F1/F4
         C1 = R2
         C2 = -R1
         C3 = 1+R1-R2
         CALL ADD3S2 (VEL,VLAG(1,1,1,1,1),VLAG(1,1,1,1,2),C2,C3,NTOT1)
         CALL ADD2S2 (VEL,VN,C1,NTOT1)
      ELSE
         WRITE (6,*) 'Need higher order expansion in VELCHAR'
         call exitt
      ENDIF
C
      return
      END
C
      subroutine lagvel
C-----------------------------------------------------------------------
C
C     Keep old velocity field(s) 
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
C
      NTOT1 = lx1*ly1*lz1*NELV
C
c      DO 100 ILAG=NBDINP-1,2,-1
      DO 100 ILAG=3-1,2,-1
         CALL COPY (VXLAG (1,1,1,1,ILAG),VXLAG (1,1,1,1,ILAG-1),NTOT1)
         CALL COPY (VYLAG (1,1,1,1,ILAG),VYLAG (1,1,1,1,ILAG-1),NTOT1)
         IF (ldim.EQ.3)
     $   CALL COPY (VZLAG (1,1,1,1,ILAG),VZLAG (1,1,1,1,ILAG-1),NTOT1)
 100  CONTINUE
C
      CALL OPCOPY (VXLAG,VYLAG,VZLAG,VX,VY,VZ)
C
      return
      END
C
      subroutine hypmsk3 (hv1msk,hv2msk,hv3msk)
C---------------------------------------------------------------------
C
C     Generate mask-array for the hyperbolic system (velocity).
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
      REAL           HV1MSK (LX1,LY1,LZ1,1)
      REAL           HV2MSK (LX1,LY1,LZ1,1)
      REAL           HV3MSK (LX1,LY1,LZ1,1)
      CHARACTER      CB*3
      PARAMETER (LXYZ1=LX1*LY1*LZ1)
      COMMON /CTMP1/ WORK   (LXYZ1,LELT)
C
      NFACES= 2*ldim
      NTOT1 = lx1*ly1*lz1*NELV
      CALL RZERO (WORK  ,NTOT1)
      CALL RONE  (HV1MSK,NTOT1)
C
      IF (IFIELD.EQ.1) THEN
      DO 100 IE=1,NELV
      DO 100 IFACE=1,NFACES
         CB=CBC(IFACE,IE,IFIELD)
         IF (CB(1:1).EQ.'V' .OR. CB(1:1).EQ.'v') THEN
           CALL FACCL3 (WORK(1,IE),VX(1,1,1,IE),UNX(1,1,IFACE,IE),IFACE)
           CALL FADDCL3(WORK(1,IE),VY(1,1,1,IE),UNY(1,1,IFACE,IE),IFACE)
           IF (IF3D) 
     $     CALL FADDCL3(WORK(1,IE),VZ(1,1,1,IE),UNZ(1,1,IFACE,IE),IFACE)
           CALL FCAVER (VAVER,WORK,IE,IFACE)
C
           IF (VAVER.LT.0.) CALL FACEV (HV1MSK,IE,IFACE,0.0,lx1,ly1,lz1)
         ENDIF
         IF (CB(1:2).EQ.'WS' .OR. CB(1:2).EQ.'ws') 
     $   CALL FACEV (HV1MSK,IE,IFACE,0.0,lx1,ly1,lz1)
 100   CONTINUE
      ENDIF
C
      CALL COPY (HV2MSK,HV1MSK,NTOT1)
      CALL COPY (HV3MSK,HV1MSK,NTOT1)
C
      return
      END
C
      subroutine setordbd
C----------------------------------------------------------------------
C
C     Set up parameters for backward differentiation scheme.
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
C
c     IF (IFSPLIT .OR. NBDINP.EQ.0) THEN     undid hardwire, 3/6/92 pff
      IF ( NBDINP.LT.1) THEN
          NBD = 1
      ELSE
         IF ((ISTEP.EQ.0).OR.(ISTEP.EQ.1))        NBD = 1
         IF ((ISTEP.GT.1).AND.(ISTEP.LE.NBDINP))  NBD = ISTEP
         IF (ISTEP.GT.NBDINP)                     NBD = NBDINP
      ENDIF
C
      return
      END
C
c-----------------------------------------------------------------------
      subroutine normsc (h1,semi,l2,linf,x,imesh)
C---------------------------------------------------------------
C
C     Compute error norms of a (scalar) field variable X 
C     defined on mesh 1 or mesh 2.
C     The error norms are normalized with respect to the volume
C     (except for Linf).
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
C
      REAL           X  (LX1,LY1,LZ1,1)
      COMMON /SCRNRM/ Y  (LX1,LY1,LZ1,LELT)
     $               ,TA1(LX1,LY1,LZ1,LELT)
     $               ,TA2(LX1,LY1,LZ1,LELT)
      REAL H1,SEMI,L2,LINF
      REAL LENGTH
C
      IF (IMESH.EQ.1) THEN
         NEL = NELV
         VOL = VOLVM1
      ELSEIF (IMESH.EQ.2) THEN
         NEL = NELT
         VOL = VOLTM1
      ENDIF
      LENGTH = VOL**(1./(ldim))
      NXYZ1  = lx1*ly1*lz1
      NTOT1  = NXYZ1*NEL
C
      H1     = 0.
      SEMI   = 0.
      L2     = 0.
      LINF   = 0.
C
      LINF = GLAMAX (X,NTOT1)
C
      CALL COL3   (TA1,X,X,NTOT1)
      CALL COL2   (TA1,BM1,NTOT1)
      L2   = GLSUM  (TA1,NTOT1)
      IF (L2.LT.0.0) L2 = 0.
C
      CALL RONE   (TA1,NTOT1)
      CALL RZERO  (TA2,NTOT1)
      CALL AXHELM (Y,X,TA1,TA2,IMESH,1)
      CALL COL3   (TA1,Y,X,NTOT1)
      SEMI = GLSUM  (TA1,NTOT1)
      IF (SEMI.LT.0.0) SEMI = 0.
C
      H1   = SQRT((SEMI*LENGTH**2+L2)/VOL)
      SEMI = SQRT(SEMI/VOL)
      L2   = SQRT(L2/VOL)
      IF (H1.LT.0.) H1 = 0.
C
      return
      END
C
      subroutine normvc (h1,semi,l2,linf,x1,x2,x3)
C---------------------------------------------------------------
C
C     Compute error norms of a (vector) field variable (X1,X2,X3)
C     defined on mesh 1 (velocity mesh only !)
C     The error norms are normalized with respect to the volume
C     (except for Linf).
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
C
      REAL           X1 (LX1,LY1,LZ1,1)
      REAL           X2 (LX1,LY1,LZ1,1)
      REAL           X3 (LX1,LY1,LZ1,1)
      COMMON /SCRMG/ Y1 (LX1,LY1,LZ1,LELT)
     $              ,Y2 (LX1,LY1,LZ1,LELT)
     $              ,Y3 (LX1,LY1,LZ1,LELT)
     $              ,TA1(LX1,LY1,LZ1,LELT)
      COMMON /SCRCH/ TA2(LX1,LY1,LZ1,LELT)
      REAL H1,SEMI,L2,LINF
      REAL LENGTH
C
      IMESH  = 1
      NEL    = NELV
      VOL    = VOLVM1
      LENGTH = VOL**(1./(ldim))
      NXYZ1  = lx1*ly1*lz1
      NTOT1  = NXYZ1*NEL
C
      H1     = 0.
      SEMI   = 0.
      L2     = 0.
      LINF   = 0.
C
      CALL COL3 (TA1,X1,X1,NTOT1)
      CALL COL3 (TA2,X2,X2,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      IF (ldim.EQ.3) THEN
      CALL COL3 (TA2,X3,X3,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      ENDIF
      LINF = GLAMAX (TA1,NTOT1)
      LINF = SQRT( LINF )
C
      CALL COL3 (TA1,X1,X1,NTOT1)
      CALL COL3 (TA2,X2,X2,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      IF (ldim.EQ.3) THEN
      CALL COL3 (TA2,X3,X3,NTOT1)
      CALL ADD2 (TA1,TA2,NTOT1)
      ENDIF
      CALL COL2 (TA1,BM1,NTOT1)
      L2 = GLSUM  (TA1,NTOT1)
      IF (L2.LT.0.0) L2 = 0.
C
      CALL RONE  (TA1,NTOT1)
      CALL RZERO (TA2,NTOT1)
      CALL OPHX  (Y1,Y2,Y3,X1,X2,X3,TA1,TA2)
      CALL COL3  (TA1,Y1,X1,NTOT1)
      CALL COL3  (TA2,Y2,X2,NTOT1)
      CALL ADD2  (TA1,TA2,NTOT1)
      IF (ldim.EQ.3) THEN
      CALL COL3  (TA2,Y3,X3,NTOT1)
      CALL ADD2  (TA1,TA2,NTOT1)
      ENDIF
      SEMI = GLSUM (TA1,NTOT1)
      IF (SEMI.LT.0.0) SEMI = 0.
C
      H1   = SQRT((SEMI*LENGTH**2+L2)/VOL)
      SEMI = SQRT(SEMI/VOL)
      L2   = SQRT(L2  /VOL)
      IF (H1.LT.0.) H1 = 0.
C
      return
      END
C
      subroutine genwp (wp,wm2,p)
C------------------------------------------------------------------
C
C     Collocate the weights on mesh M2 with the pressure or the
C     search direction in the cg-iteration.
C
C-----------------------------------------------------------------
      include 'SIZE'
C
      REAL   WP  (LX2,LY2,LZ2,LELV)
      REAL   P   (LX2,LY2,LZ2,LELV)
      REAL   WM2 (LX2,LY2,LZ2)
C
      NXYZ2 = lx2*ly2*lz2
      DO 100 IEL=1,NELV
         CALL COL3 (WP(1,1,1,IEL),WM2(1,1,1),P(1,1,1,IEL),NXYZ2)
 100  CONTINUE
      return
      END
C
      subroutine convuz (ifstuz)
C--------------------------------------------------------------------
C
C     Check convergence for the coupled form.
C     Consistent approximation spaces.
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      LOGICAL        IFSTUZ
      COMMON /SCRNS/ RESV1 (LX1,LY1,LZ1,LELV)
     $ ,             RESV2 (LX1,LY1,LZ1,LELV)
     $ ,             RESV3 (LX1,LY1,LZ1,LELV)
     $ ,             RESP  (LX2,LY2,LZ2,LELV)
     $ ,             TA1   (LX1,LY1,LZ1,LELV)
     $ ,             TA2   (LX1,LY1,LZ1,LELV)
     $ ,             TA3   (LX1,LY1,LZ1,LELV) 
      COMMON /SCRMG/ TB1   (LX1,LY1,LZ1,LELV)
     $ ,             TB2   (LX1,LY1,LZ1,LELV)
     $ ,             TB3   (LX1,LY1,LZ1,LELV)
     $ ,             WP    (LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
C
      IFSTUZ = .TRUE.
      TCRITV = TOLHV*1.5
      TCRITP = TOLPS*1.5
      NTOT1  = lx1*ly1*lz1*NELV
      NTOT2  = lx2*ly2*lz2*NELV
      INTYPE = 0
      CALL SETHLM (H1,H2,INTYPE)
C
C     Momentum
C
      CALL OPCOPY  (RESV1,RESV2,RESV3,BFX,BFY,BFZ)
      CALL OPGRADT (TA1,TA2,TA3,PR)
      CALL OPHX    (TB1,TB2,TB3,VX,VY,VZ,H1,H2)
      CALL OPADD2  (RESV1,RESV2,RESV3,TA1,TA2,TA3)
      CALL OPSUB2  (RESV1,RESV2,RESV3,TB1,TB2,TB3)
      CALL OPMASK  (RESV1,RESV2,RESV3)
      CALL OPDSSUM (RESV1,RESV2,RESV3)
      CALL OPCOLV3 (TA1,TA2,TA3,RESV1 ,RESV2 ,RESV3,BINVM1)
      RV1 = SQRT(GLSC3(TA1,RESV1,VMULT,NTOT1)/VOLVM1)
      RV2 = SQRT(GLSC3(TA2,RESV2,VMULT,NTOT1)/VOLVM1)
      IF (RV1 .GT. TCRITV) IFSTUZ = .FALSE.
      IF (RV2 .GT. TCRITV) IFSTUZ = .FALSE.
      IF (ldim.EQ.3) THEN
         RV3 = SQRT(GLSC3(TA3,RESV3,VMULT,NTOT1)/VOLVM1)
         IF (RV3 .GT. TCRITV) IFSTUZ = .FALSE.
      ENDIF
C
C     Continuity
C
      CALL OPDIV (RESP,VX,VY,VZ)
      CALL COL3  (WP,RESP,BM2INV,NTOT2)
      RP = SQRT  (GLSC2(WP,RESP,NTOT2)/VOLVM2)
      IF (RP .GT. TCRITP) IFSTUZ = .FALSE.
C
      return
      END
C
      subroutine convsp (ifstsp)
      LOGICAL IFSTSP
      IFSTSP = .FALSE.
      return
      END
C
      subroutine antimsk (y,x,xmask,n)
C------------------------------------------------------------------
C
C     Return Dirichlet boundary values of X in the array Y
C
C-------------------------------------------------------------------
      REAL  Y(1),X(1),XMASK(1)
      include 'OPCTR'
C
#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'antmsk'
      endif
      isbcnt = 2*n
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      DO 100 I=1,N
         Y(I) = X(I)*(1.-XMASK(I))
 100  CONTINUE
      return
      END
C
      subroutine opamask (vbdry1,vbdry2,vbdry3) 
C----------------------------------------------------------------------
C
C     Antimask the velocity arrays. 
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      REAL VBDRY1 (LX1,LY1,LZ1,1)
      REAL VBDRY2 (LX1,LY1,LZ1,1)
      REAL VBDRY3 (LX1,LY1,LZ1,1)
C
      NTOT1 = lx1*ly1*lz1*NELV
C
      IF (IFSTRS) THEN
         if (ifield.eq.ifldmhd) then
            CALL AMASK (VBDRY1,VBDRY2,VBDRY3,BX,BY,BZ,NELV)
         else
            CALL AMASK (VBDRY1,VBDRY2,VBDRY3,VX,VY,VZ,NELV)
         endif
      ELSE
         if (ifield.eq.ifldmhd) then
            CALL ANTIMSK (VBDRY1,BX,B1MASK,NTOT1)
            CALL ANTIMSK (VBDRY2,BY,B2MASK,NTOT1)
            IF (ldim.EQ.3) 
     $      CALL ANTIMSK (VBDRY3,BZ,B3MASK,NTOT1)
         else
            CALL ANTIMSK (VBDRY1,VX,V1MASK,NTOT1)
            CALL ANTIMSK (VBDRY2,VY,V2MASK,NTOT1)
            IF (ldim.EQ.3) 
     $      CALL ANTIMSK (VBDRY3,VZ,V3MASK,NTOT1)
         endif
      ENDIF
C
      return
      END
C
      subroutine opmask (res1,res2,res3)
C----------------------------------------------------------------------
C
C     Mask the residual arrays. 
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      REAL RES1(1),RES2(1),RES3(1)
C
      NTOT1 = lx1*ly1*lz1*NELV
C
c     sv=glsum(v3mask,ntot1)
c     sb=glsum(b3mask,ntot1)
c     write(6,*) istep,' ifld:',ifield,intype,sv,sb
      IF (IFSTRS) THEN
         CALL RMASK (RES1,RES2,RES3,NELV)
      ELSE
         if (ifield.eq.ifldmhd) then
            CALL COL2 (RES1,B1MASK,NTOT1)
            CALL COL2 (RES2,B2MASK,NTOT1)
            IF (ldim.EQ.3)
     $      CALL COL2 (RES3,B3MASK,NTOT1)
         else
            CALL COL2 (RES1,V1MASK,NTOT1)
            CALL COL2 (RES2,V2MASK,NTOT1)
            IF (ldim.EQ.3)
     $      CALL COL2 (RES3,V3MASK,NTOT1)
         endif
      ENDIF
C
      return
      END
C
      subroutine opadd2 (a1,a2,a3,b1,b2,b3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL ADD2(A1,B1,NTOT1)
      CALL ADD2(A2,B2,NTOT1)
      IF(ldim.EQ.3)CALL ADD2(A3,B3,NTOT1)
      return
      END
C
      subroutine opsub2 (a1,a2,a3,b1,b2,b3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL SUB2(A1,B1,NTOT1)
      CALL SUB2(A2,B2,NTOT1)
      IF(ldim.EQ.3)CALL SUB2(A3,B3,NTOT1)
      return
      END
C
      subroutine opsub3 (a1,a2,a3,b1,b2,b3,c1,c2,c3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1),C1(1),C2(1),C3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL SUB3(A1,B1,C1,NTOT1)
      CALL SUB3(A2,B2,C2,NTOT1)
      IF(ldim.EQ.3)CALL SUB3(A3,B3,C3,NTOT1)
      return
      END
C
      subroutine opcolv3(a1,a2,a3,b1,b2,b3,c)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1)
      REAL B1(1),B2(1),B3(1)
      REAL C (1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opcolv'
      endif
C
      isbcnt = ntot1*ldim
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            A1(I)=B1(I)*C(I)
            A2(I)=B2(I)*C(I)
            A3(I)=B3(I)*C(I)
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            A1(I)=B1(I)*C(I)
            A2(I)=B2(I)*C(I)
  200    CONTINUE
      ENDIF
      return
      END
C
      subroutine opcolv (a1,a2,a3,c)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),C(1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opcolv'
      endif
C
      isbcnt = ntot1*ldim
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            A1(I)=A1(I)*C(I)
            A2(I)=A2(I)*C(I)
            A3(I)=A3(I)*C(I)
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            A1(I)=A1(I)*C(I)
            A2(I)=A2(I)*C(I)
  200    CONTINUE
      ENDIF
      return
      END
C
      subroutine opcol2 (a1,a2,a3,b1,b2,b3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL COL2(A1,B1,NTOT1)
      CALL COL2(A2,B2,NTOT1)
      IF(ldim.EQ.3)CALL COL2(A3,B3,NTOT1)
      return
      END
C
      subroutine opchsgn (a,b,c)
      include 'SIZE'
      REAL A(1),B(1),C(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL CHSIGN(A,NTOT1)
      CALL CHSIGN(B,NTOT1)
      IF(ldim.EQ.3)CALL CHSIGN(C,NTOT1)
      return
      END
c
      subroutine opcopy (a1,a2,a3,b1,b2,b3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL COPY(A1,B1,NTOT1)
      CALL COPY(A2,B2,NTOT1)
      IF(ldim.EQ.3)CALL COPY(A3,B3,NTOT1)
      return
      END

c-----------------------------------------------------------------------
      subroutine rotate_cyc(r1,r2,r3,idir)

      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'

      real r1(lx1,ly1,lz1,1)
     $   , r2(lx1,ly1,lz1,1)
     $   , r3(lx1,ly1,lz1,1)

      integer e,f
      logical ifxy
      real length
 
c     (1) Face n-t transformation


      nface = 2*ldim
      do e=1,nelfld(ifield)
      do f=1,nface

         if(cbc(f,e,ifield) .eq. 'P  '.or.cbc(f,e,ifield).eq.'p  ')then

            call facind2 (js1,jf1,jskip1,js2,jf2,jskip2,f)
            if (idir.eq.1) then
              k=0
              do j2=js2,jf2,jskip2
              do j1=js1,jf1,jskip1
               k=k+1

               dotprod = unx(k,1,f,e)*ym1(j1,j2,1,e)
     $                  -uny(k,1,f,e)*xm1(j1,j2,1,e)
     
               ifxy = .true.
      
               length = unx(k,1,f,e)*unx(k,1,f,e) 
     $               + uny(k,1,f,e)*uny(k,1,f,e)
               length = sqrt(length)

               cost =  unx(k,1,f,e)/length
               sint =  uny(k,1,f,e)/length
               
               rnor = ( r1(j1,j2,1,e)*cost + r2(j1,j2,1,e)*sint )
               rtn1 = (-r1(j1,j2,1,e)*sint + r2(j1,j2,1,e)*cost )

               if (ifxy.and.dotprod .ge. 0.0) then 
                  r1(j1,j2,1,e) = rnor
                  r2(j1,j2,1,e) = rtn1
               elseif (ifxy) then
                  r1(j1,j2,1,e) =-rnor
                  r2(j1,j2,1,e) =-rtn1
               endif
              enddo
              enddo

            else    ! reverse rotate

              k=0
              do j2=js2,jf2,jskip2
              do j1=js1,jf1,jskip1
               k=k+1

               dotprod = unx(k,1,f,e)*ym1(j1,j2,1,e)
     $                  -uny(k,1,f,e)*xm1(j1,j2,1,e)
               ifxy = .true.
               
               length = unx(k,1,f,e)*unx(k,1,f,e) 
     $               + uny(k,1,f,e)*uny(k,1,f,e)
               length = sqrt(length)
               
               cost =  unx(k,1,f,e)/length
               sint =  uny(k,1,f,e)/length
               
               rnor = ( r1(j1,j2,1,e)*cost - r2(j1,j2,1,e)*sint )
               rtn1 = ( r1(j1,j2,1,e)*sint + r2(j1,j2,1,e)*cost )

               if(ifxy.and.dotprod .ge. 0.0) then 
                  r1(j1,j2,1,e) = rnor
                  r2(j1,j2,1,e) = rtn1
               elseif (ifxy) then
                  r1(j1,j2,1,e) =-rnor
                  r2(j1,j2,1,e) =-rtn1
               endif
              enddo
              enddo
            endif

         endif

      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine opdssum (a,b,c)! NOTE: opdssum works on FLUID/MHD arrays only!

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      include 'GEOM'

      real a(1),b(1),c(1)

      if (ifcyclic) then
         call rotate_cyc  (a,b,c,1)
         call vec_dssum   (a,b,c,lx1,ly1,lz1)
         call rotate_cyc  (a,b,c,0)
      else
         call vec_dssum   (a,b,c,lx1,ly1,lz1)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine opdsop (a,b,c,op)! opdsop works on FLUID/MHD arrays only!

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      include 'GEOM'

      real a(1),b(1),c(1)
      character*3 op

      if (ifcyclic) then

         if (op.eq.'*  ' .or. op.eq.'mul' .or. op.eq.'MUL') then
            call vec_dsop    (a,b,c,lx1,ly1,lz1,op)
         else
            call rotate_cyc  (a,b,c,1)
            call vec_dsop    (a,b,c,lx1,ly1,lz1,op)
            call rotate_cyc  (a,b,c,0)
         endif

      else

         call vec_dsop    (a,b,c,lx1,ly1,lz1,op)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine opicol2 (a1,a2,a3,b1,b2,b3)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL INVCOL2(A1,B1,NTOT1)
      CALL INVCOL2(A2,B2,NTOT1)
      IF(ldim.EQ.3)CALL INVCOL2(A3,B3,NTOT1)
      return
      END
C
      subroutine oprzero (a,b,c)
      include 'SIZE'
      REAL A(1),B(1),C(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL RZERO(A,NTOT1)
      CALL RZERO(B,NTOT1)
      IF(ldim.EQ.3) CALL RZERO(C,NTOT1)
      return
      END
C
      subroutine oprone (a,b,c)
      include 'SIZE'
      REAL A(1),B(1),C(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL RONE(A,NTOT1)
      CALL RONE(B,NTOT1)
      IF(ldim.EQ.3) CALL RONE(C,NTOT1)
      return
      END
C
      subroutine opcmult (a,b,c,const)
      include 'SIZE'
      REAL A(1),B(1),C(1)
      NTOT1=lx1*ly1*lz1*NELV
      CALL CMULT(A,CONST,NTOT1)
      CALL CMULT(B,CONST,NTOT1)
      IF(ldim.EQ.3) CALL CMULT(C,CONST,NTOT1)
      return
      END
c-----------------------------------------------------------------------
      subroutine opcolv2c(a1,a2,a3,b1,b2,c)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1)
      REAL B1(1),B2(1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opcv2c'
      endif
C
      isbcnt = ntot1*(ldim+2)
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            tmp = c*b1(i)*b2(i)
            A1(I)=A1(I)*tmp
            A2(I)=A2(I)*tmp
            A3(I)=A3(I)*tmp
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            tmp = c*b1(i)*b2(i)
            A1(I)=A1(I)*tmp
            A2(I)=A2(I)*tmp
  200    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine opcolv2(a1,a2,a3,b1,b2)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1)
      REAL B1(1),B2(1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opclv2'
      endif
C
      isbcnt = ntot1*(ldim+1)
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            tmp = b1(i)*b2(i)
            A1(I)=A1(I)*tmp
            A2(I)=A2(I)*tmp
            A3(I)=A3(I)*tmp
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            tmp = b1(i)*b2(i)
            A1(I)=A1(I)*tmp
            A2(I)=A2(I)*tmp
  200    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine opadd2col(a1,a2,a3,b1,b2,b3,c)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1)
      REAL B1(1),B2(1),B3(1),C(1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opa2cl'
      endif
C
      isbcnt = ntot1*(ldim*2)
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            A1(I)=A1(I)+b1(i)*c(i)
            A2(I)=A2(I)+b2(i)*c(i)
            A3(I)=A3(I)+b3(i)*c(i)
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            A1(I)=A1(I)+b1(i)*c(i)
            A2(I)=A2(I)+b2(i)*c(i)
  200    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine opcolv3c(a1,a2,a3,b1,b2,b3,c,d)
      include 'SIZE'
      REAL A1(1),A2(1),A3(1)
      REAL B1(1),B2(1),B3(1)
      REAL C (1)
      include 'OPCTR'
C
      NTOT1=lx1*ly1*lz1*NELV

#ifdef TIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'opcv3c'
      endif
C
      isbcnt = ntot1*ldim*2
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      IF (ldim.EQ.3) THEN
         DO 100 I=1,NTOT1
            A1(I)=B1(I)*C(I)*d
            A2(I)=B2(I)*C(I)*d
            A3(I)=B3(I)*C(I)*d
  100    CONTINUE
      ELSE
         DO 200 I=1,NTOT1
            A1(I)=B1(I)*C(I)*d
            A2(I)=B2(I)*C(I)*d
  200    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
C
      subroutine uzawa (rcg,h1,h2,h2inv,intype,iter)
C-----------------------------------------------------------------------
C
C     Solve the pressure equation by (nested) preconditioned 
C     conjugate gradient iteration.
C     INTYPE =  0  (steady)
C     INTYPE =  1  (explicit)
C     INTYPE = -1  (implicit)
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      COMMON  /CTOLPR/ DIVEX
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      REAL             RCG  (LX2,LY2,LZ2,LELV)
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/   WP   (LX2,LY2,LZ2,LELV)
     $ ,               XCG  (LX2,LY2,LZ2,LELV)
     $ ,               PCG  (LX2,LY2,LZ2,LELV) 
     $ ,               RPCG (LX2,LY2,LZ2,LELV)
 
      real*8 etime1,dnekclock
      integer*8 ntotg,nxyz2


      etime1 = dnekclock()
      DIVEX = 0.
      ITER  = 0
c
      CALL CHKTCG2 (TOLPS,RCG,ICONV)
      if (param(21).gt.0.and.tolps.gt.abs(param(21))) 
     $   TOLPS = abs(param(21))
C
c      IF (ICONV.EQ.1) THEN
c         IF (NID.EQ.0) WRITE(6,9999) ITER,DIVEX,TOLPS
c         return
c      ENDIF

      nxyz2 = lx2*ly2*lz2
      ntot2 = nxyz2*nelv
      ntotg = nxyz2*nelgv

      CALL UZPREC  (RPCG,RCG,H1,H2,INTYPE,WP)
      RRP1 = GLSC2 (RPCG,RCG,NTOT2)
      CALL COPY    (PCG,RPCG,NTOT2)
      CALL RZERO   (XCG,NTOT2)
      if (rrp1.eq.0) return
      BETA = 0.
      div0=0.
C
      tolpss = tolps
      DO 1000 ITER=1,NMXP
C
C        CALL CONVPR  (RCG,tolpss,ICONV,RNORM)
         call convprn (iconv,rnorm,rrp1,rcg,rpcg,tolpss)

         if (iter.eq.1)      div0   = rnorm
         if (param(21).lt.0) tolpss = abs(param(21))*div0

         ratio = rnorm/div0
         IF (IFPRINT.AND.NIO.EQ.0) 
     $   WRITE (6,66) iter,tolpss,rnorm,div0,ratio,istep
   66    format(i5,1p4e12.5,i8,' Divergence')
c
         IF (ICONV.EQ.1.and.iter.gt.1) GOTO 9000
c        IF (ICONV.EQ.1.and.(iter.gt.1.or.istep.le.2)) GOTO 9000
c        IF (ICONV.EQ.1) GOTO 9000
c        if (ratio.le.1.e-5) goto 9000


         IF (ITER .NE. 1) THEN
            BETA = RRP1/RRP2
            CALL ADD2S1 (PCG,RPCG,BETA,NTOT2)
         ENDIF

         CALL CDABDTP  (WP,PCG,H1,H2,H2INV,INTYPE)
         PAP   = GLSC2 (PCG,WP,NTOT2)

         IF (PAP.NE.0.) THEN
            ALPHA = RRP1/PAP
         ELSE
            pcgmx = glamax(pcg,ntot2)
            wp_mx = glamax(wp ,ntot2)
            ntot1 = lx1*ly1*lz1*nelv
            h1_mx = glamax(h1 ,ntot1)
            h2_mx = glamax(h2 ,ntot1)
            if (nid.eq.0) write(6,*) 'ERROR: pap=0 in uzawa.'
     $      ,iter,pcgmx,wp_mx,h1_mx,h2_mx
            call exitt
         ENDIF
         CALL ADD2S2 (XCG,PCG,ALPHA,NTOT2)
         CALL ADD2S2 (RCG,WP,-ALPHA,NTOT2)

         if (iter.eq.-1) then
            call convprn (iconv,rnrm1,rrpx,rcg,rpcg,tolpss)
            if (iconv.eq.1) then
               rnorm = rnrm1
               ratio = rnrm1/div0
               if (nio.eq.0) 
     $         write (6,66) iter,tolpss,rnrm1,div0,ratio,istep
               goto 9000
            endif
         endif

         call ortho(rcg)

         RRP2 = RRP1
         CALL UZPREC  (RPCG,RCG,H1,H2,INTYPE,WP)
c        RRP1 = GLSC2 (RPCG,RCG,NTOT2)

 1000 CONTINUE
      if (nid.eq.0) WRITE (6,3001) ITER,RNORM,tolpss
c     if (istep.gt.20) CALL EMERXIT
 3001 FORMAT(I6,' **ERROR**: Failed to converge in UZAWA:',6E13.4)
 9000 CONTINUE

      divex = rnorm
      iter  = iter-1

      if (iter.gt.0) call copy (rcg,xcg,ntot2)
      call ortho(rcg)

      etime1 = dnekclock()-etime1
      IF (NIO.EQ.0) WRITE(6,9999) ISTEP, '  U-Press std. ',
     &                            ITER,DIVEX,div0,tolpss,etime1
 9999 FORMAT(I11,a,I7,1p4E13.4)
19999 FORMAT(I11,'  U-Press 1.e-5: ',I7,1p4E13.4)
C
C
      return
      END
c-----------------------------------------------------------------------
      subroutine mapw(md,nd,m1,n1,mflg)
c
c     Interpolate from mesh "1" to "d" if mflg = 1
c
      include 'SIZE'
      include 'DEALIAS'
c
      real w(lxd*lxd*lx1)
      real md(lxd,lyd,lzd,lelv),m1(lx1,ly1,lz1,lelv)
c
      integer icalld
      save icalld
      data icalld /0/
c
      if (icalld .eq. 0) then
         call setmap(n1,nd)
         icalld = icalld + 1
      endif
c
      if (mflg .eq.1) then
         do ie = 1,nelv
            call specmp(md(1,1,1,ie),nd,m1(1,1,1,ie),n1,im1d,im1dt,w)
         enddo
      else
         do ie = 1,nelv
            call specmp(m1(1,1,1,ie),n1,md(1,1,1,ie),nd,imd1,imd1t,w)
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mapwp(md,nd,m1,n1,mflg)
c
c     Project from "d" to 1
c
      include 'SIZE'
      include 'DEALIAS'
c
      real w(lxd*lxd*lx1)
      real md(lxd,lyd,lzd,lelv),m1(lx1,ly1,lz1,lelv)
c
      integer icalld
      save icalld
      data icalld /0/
c
      if (icalld .eq. 0) then
          call setproj(n1,nd)
          icalld = icalld + 1
      endif
c
      do ie = 1,nelv
        call specmp(m1(1,1,1,ie),n1,md(1,1,1,ie),nd,pmd1,pmd1t,w)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine specmp(b,nb,a,na,ba,ab,w)
C
C     -  Spectral interpolation from A to B via tensor products
C     -  scratch arrays: w(na*na*nb)
C
C
      include 'SIZE'
      include 'INPUT'
      real b(nb,nb,nb),a(na,na,na)
      real w(1)
C
C
      if (if3d) then
         nab = na*nb
         nbb = nb*nb
         call mxm(ba,nb,a,na,b,na*na)
         k=1
         l=1
         do iz=1,na
            call mxm(b(k,1,1),nb,ab,na,w(l),nb)
            k=k+nab
            l=l+nbb
         enddo
         call mxm(w,nbb,ab,na,b,nb)
      else
         call mxm(ba,nb,a,na,w,na)
         call mxm(w,nb,ab,na,b,nb)
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine setmap(n1,nd)
c
      include 'SIZE'
      include 'DEALIAS'
c
      parameter(lx=80)
      real z1(lx),zd(lx),w(lx)
c
      if (n1.gt.lx.or.nd.gt.lx) then
         write(6,*)'ERROR: increase lx in setmap to max:',n1,nd
         call exitt
      endif
c
      call zwgll(z1,w,n1)
      call zwgll(zd,w,nd)
      call igllm(im1d,im1dt,z1,zd,n1,nd,n1,nd)
      call igllm(imd1,imd1t,zd,z1,nd,n1,nd,n1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_PND(P,LkD,LkNt,N,D)
c
      integer N,D
      real    P(N,D),LkD(0:N-1,D),LkNt(N,0:N-1)
c
      parameter(lx=80)
      real zN(lx),zD(lx),w(lx)
c
c     Compute Lagrangian interpolant points
c
      call zwgll(zN,w,N)
      call zwgll(zD,w,D)
c
c
c                 D           D
c     Compute L (x )  and L (x )
c              k  i        k  j
c
      do k=0,N-1
c
         do j=1,D
            LkD (k,j) = pnleg(zD(j),k)
         enddo
c
         do j=1,N
            LkNt(j,k) = pnleg(zN(j),k)
         enddo
c
      enddo
c
c     Find scale factors to normalize the first N-1 Legendre polynomials
c     such that  (L_i,L_j) = delta_ij
c
      do k=0,N-1
         s = 0
         do j=1,D
            s = s + LkD(k,j)*LkD(k,j)*w(j)
         enddo
         s = 1./sqrt(s)
c
c        Normalize polynomials
c
         do j=1,D
            LkD (k,j) = s * LkD (k,j)
         enddo
c
         do j=1,N
            LkNt(j,k) = s * LkNt(j,k)
         enddo
c
      enddo
c
c     Scale columns of LkD by w_j
c
      do j=1,D
         do k=0,N-1
            LkD(k,j) = LkD(k,j)*w(j)
         enddo
      enddo
c
c     Compute P = LkNt * LkD
c
      call mxm(LkNt,N,LkD,N,P,D)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine convop(conv,fi)
C
C     Compute the convective term CONV for a passive scalar field FI
C     using the skew-symmetric formulation.
C     The field variable FI is defined on mesh M1 (GLL) and
C     the velocity field is assumed given.
C
C     IMPORTANT NOTE: Use the scratch-arrays carefully!!!!!
C
C     The common-block SCRNS is used in CONV1 and CONV2.
C     The common-blocks CTMP0 and CTMP1 are also used as scratch-arrays
C     since there is no direct stiffness summation or Helmholtz-solves. 
C
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
C
C     Use the common blocks CTMP0 and CTMP1 as work space.
C
      COMMON /SCRCH/  CMASK1 (LX1,LY1,LZ1,LELV)
     $ ,              CMASK2 (LX1,LY1,LZ1,LELV)
      COMMON /CTMP1/  MFI    (LX1,LY1,LZ1,LELV)
     $ ,              DMFI   (LX1,LY1,LZ1,LELV)
     $ ,              MDMFI  (LX1,LY1,LZ1,LELV)
      REAL   MFI,DMFI,MDMFI
C
C     Arrays in parameter list
C
      REAL    CONV (LX1,LY1,LZ1,1) 
      REAL    FI   (LX1,LY1,LZ1,1)

      if (nio.eq.0.and.loglevel.gt.2)
     $   write(6,*) 'convop', ifield, ifdeal(ifield)

#ifdef TIMER
      if (icalld.eq.0) tadvc=0.0
      icalld=icalld+1
      nadvc=icalld
      etime1=dnekclock()
#endif
 
      nxyz1 = lx1*ly1*lz1
      ntot1 = lx1*ly1*lz1*nelv
      ntotz = lx1*ly1*lz1*nelfld(ifield)
      ntott = lx1*ly1*lz1*nelt
 
      call rzero  (conv,ntott)
 
      if (ifdgfld(ifield)) then
         call convect_dg (conv,fi,.false.,vxd,vyd,vzd,.true.)
         goto 100
      elseif (param(86).ne.0.0) then  ! skew-symmetric form
         call convopo(conv,fi)
         goto 100
      endif

      if (.not. ifdeal(ifield)) then
         call conv1 (conv,fi)
      elseif (param(99).eq.2.or.param(99).eq.3) then  
         call conv1d(conv,fi)
      elseif (param(99).eq.4) then
         if (ifpert) then
           call convect_new (conv,fi,.false.,vx,vy,vz,.false.)
         else
           call convect_new (conv,fi,.false.,vxd,vyd,vzd,.true.)
         endif
         call invcol2     (conv,bm1,ntot1)  ! local mass inverse
      elseif (param(99).eq.5) then
         call convect_cons(conv,fi,.false.,vx,vy,vz,.false.)
         call invcol2     (conv,bm1,ntot1)  ! local mass inverse
      else
         call conv1 (conv,fi)
      endif

 100  continue

#ifdef TIMER
      tadvc=tadvc+(dnekclock()-etime1)
#endif

      return
      END
c-----------------------------------------------------------------------
      subroutine conv1d (dfi,fi)
C--------------------------------------------------------------------
C
C     Compute D*FI (part of the convection operator)
C     De-aliased version 3/11/97
C
C--------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      REAL           DFI (LX1,LY1,LZ1,1) 
      REAL           FI  (LX1,LY1,LZ1,1) 
c
      COMMON /CTMP0/ TA1  (LX1,LY1,LZ1,LELV)
     $             , DFID (LXD,LYD,LZD,LELV) 
     $             , TA1D (LXD,LYD,LZD,lelv) 
C
      integer icalld
      save icalld
      data icalld /0/
c
      NTOTD = lxd*lyd*lzd*NELV
c
c
c     interpolate ta1 and vx onto larger mesh
c
      CALL DUDXYZ (TA1,FI,RXM1,SXM1,TXM1,JACM1,IMESH,1)
      call mapw   (ta1d,lxd,ta1,lx1,1)
      call mapw   (vxd ,lxd,vx ,lx1,1)
      CALL COL3   (DFID,TA1D,VXD,NTOTD)
c
c
c     interpolate ta1 and vy onto larger mesh
c
      CALL DUDXYZ  (TA1,FI,RYM1,SYM1,TYM1,JACM1,IMESH,2)
      call mapw    (ta1d,lxd,ta1,lx1,1)
      call mapw    (vyd ,lxd,vy ,lx1,1)
      CALL ADDCOL3 (DFID,TA1D,VYD,NTOTD)
c
      IF (if3d) THEN
c
c        interpolate ta1 and vy onto larger mesh
c
         CALL DUDXYZ  (TA1,FI,RZM1,SZM1,TZM1,JACM1,IMESH,3)
         call mapw    (ta1d,lxd,ta1,lx1,1)
         call mapw    (vzd ,lxd,vz ,lx1,1)
         CALL ADDCOL3 (DFID,TA1D,VZD,NTOTD)
c
      ENDIF
c
c     Now, *project* DFID onto mesh 1 using L2 projection
c
      call mapwp(dfid,lxd,dfi,lx1,-1)
      return
      END
C------------------------------------------------------------------------
      subroutine conv1(du,u)  ! used to be conv1n
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
c
      real  du  (lx1*ly1*lz1,1)
      real  u   (lx1,ly1,lz1,1)
c
      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
C
C     Store the inverse jacobian to speed this operation up
C
      common /ctmp0/ dudr(lx1,ly1,lz1)
     $             , duds(lx1,ly1,lz1)
     $             , dudt(lx1,ly1,lz1)

      nel = nelv
      if (imesh.eq.2) nel = nelt
      nxy1  = lx1*ly1
      nyz1  = ly1*lz1
      nxyz1 = lx1*ly1*lz1
      ntot  = nxyz1*nel
C
C     Compute vel.grad(u)
C
      do ie=1,nel
C
        if (if3d) then
c
           call mxm   (dxm1,lx1,u(1,1,1,ie),lx1,dudr,nyz1)
           do iz=1,lz1
             call mxm (u(1,1,iz,ie),lx1,dytm1,ly1,duds(1,1,iz),ly1)
           enddo
           call mxm   (u(1,1,1,ie),nxy1,dztm1,lz1,dudt,lz1)
c
           do i=1,nxyz1
              du(i,ie) = jacmi(i,ie)*(
     $                     vx(i,1,1,ie)*(
     $                          rxm1(i,1,1,ie)*dudr(i,1,1)
     $                        + sxm1(i,1,1,ie)*duds(i,1,1)
     $                        + txm1(i,1,1,ie)*dudt(i,1,1) )
     $                   + vy(i,1,1,ie)*(
     $                          rym1(i,1,1,ie)*dudr(i,1,1)
     $                        + sym1(i,1,1,ie)*duds(i,1,1)
     $                        + tym1(i,1,1,ie)*dudt(i,1,1) )
     $                   + vz(i,1,1,ie)*(
     $                          rzm1(i,1,1,ie)*dudr(i,1,1)
     $                        + szm1(i,1,1,ie)*duds(i,1,1)
     $                        + tzm1(i,1,1,ie)*dudt(i,1,1) ) )
           enddo
c
         else
c
c           2D
            call mxm (dxm1,lx1,u(1,1,1,ie),lx1,dudr,nyz1)
            call mxm (u(1,1,1,ie),lx1,dytm1,ly1,duds,ly1)
            do i=1,nxyz1
               du(i,ie) = jacmi(i,ie)*(
     $                      vx(i,1,1,ie)*(
     $                           rxm1(i,1,1,ie)*dudr(i,1,1)
     $                         + sxm1(i,1,1,ie)*duds(i,1,1) )
     $                    + vy(i,1,1,ie)*(
     $                           rym1(i,1,1,ie)*dudr(i,1,1)
     $                         + sym1(i,1,1,ie)*duds(i,1,1) ) )
            enddo
          endif

       enddo
c
       return
       end
c-----------------------------------------------------------------------
      subroutine conv1no(du,u)
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
c
      real  du  (lx1*ly1*lz1,1)
      real  u   (lx1,ly1,lz1,1)
c
      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
C
C     Store the inverse jacobian to speed this operation up
C
C
      common /ctmp0/ dudr(lx1,ly1,lz1)
     $             , duds(lx1,ly1,lz1)
     $             , dudt(lx1,ly1,lz1)
C
      nel = nelv
      if (imesh.eq.2) nel = nelt
      nxy1  = lx1*ly1
      nyz1  = ly1*lz1
      nxyz1 = lx1*ly1*lz1
      ntot  = nxyz1*nel
C
C     Compute vel.grad(u)
C
      do ie=1,nel
C
        if (if3d) then
c
           call mxm   (dxm1,lx1,u(1,1,1,ie),lx1,dudr,nyz1)
           do iz=1,lz1
             call mxm (u(1,1,iz,ie),lx1,dytm1,ly1,duds(1,1,iz),ly1)
           enddo
           call mxm   (u(1,1,1,ie),nxy1,dztm1,lz1,dudt,lz1)
c
           do i=1,nxyz1
              du(i,ie) = jacmi(i,ie)*(
     $                     vx(i,1,1,ie)*(
     $                          rxm1(i,1,1,ie)*dudr(i,1,1)
     $                        + sxm1(i,1,1,ie)*duds(i,1,1)
     $                        + txm1(i,1,1,ie)*dudt(i,1,1) )
     $                   + vy(i,1,1,ie)*(
     $                          rym1(i,1,1,ie)*dudr(i,1,1)
     $                        + sym1(i,1,1,ie)*duds(i,1,1)
     $                        + tym1(i,1,1,ie)*dudt(i,1,1) )
     $                   + vz(i,1,1,ie)*(
     $                          rzm1(i,1,1,ie)*dudr(i,1,1)
     $                        + szm1(i,1,1,ie)*duds(i,1,1)
     $                        + tzm1(i,1,1,ie)*dudt(i,1,1) ) )
           enddo
c
         else
c
c           2D
            call mxm (dxm1,lx1,u(1,1,1,ie),lx1,dudr,nyz1)
            call mxm (u(1,1,1,ie),lx1,dytm1,ly1,duds,ly1)
            do i=1,nxyz1
               du(i,ie) = jacmi(i,ie)*(
     $                      vx(i,1,1,ie)*(
     $                           rxm1(i,1,1,ie)*dudr(i,1,1)
     $                         + sxm1(i,1,1,ie)*duds(i,1,1) )
     $                    + vy(i,1,1,ie)*(
     $                           rym1(i,1,1,ie)*dudr(i,1,1)
     $                         + sym1(i,1,1,ie)*duds(i,1,1) ) )
            enddo
          endif

       enddo
c
       return
       end
c-----------------------------------------------------------------------
      subroutine conv1rk(du,dv,dw,u,v,w)
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
c
      real  du(lx1*ly1*lz1,1),dv(lx1*ly1*lz1,1),dw(lx1*ly1*lz1,1)
      real  u (lx1,ly1,lz1,1),v (lx1,ly1,lz1,1),w (lx1,ly1,lz1,1)
c
      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
C
      common /ctmp0/ duds(lx1,ly1,lz1)
     $             , dvds(lx1,ly1,lz1)
     $             , dwds(lx1,ly1,lz1)
C
      nel = nelv
      if (imesh.eq.2) nel = nelt
      nxy1  = lx1*ly1
      nyz1  = ly1*lz1
      nxyz1 = lx1*ly1*lz1
C
C     Compute vel.grad(u)
C
      do ie=1,nel
C
         if (if3d) then
            call mxm   (dxm1,lx1,u(1,1,1,ie),lx1,du(1,ie),nyz1)
            call mxm   (dxm1,lx1,v(1,1,1,ie),lx1,dv(1,ie),nyz1)
            call mxm   (dxm1,lx1,w(1,1,1,ie),lx1,dw(1,ie),nyz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie)*vx(i,1,1,ie)
               dv(i,ie) = dv(i,ie)*vx(i,1,1,ie)
               dw(i,ie) = dw(i,ie)*vx(i,1,1,ie)
            enddo
c
            do iz=1,lz1
              call mxm (u(1,1,iz,ie),lx1,dytm1,ly1,duds(1,1,iz),ly1)
            enddo
            do iz=1,lz1
              call mxm (v(1,1,iz,ie),lx1,dytm1,ly1,dvds(1,1,iz),ly1)
            enddo
            do iz=1,lz1
              call mxm (w(1,1,iz,ie),lx1,dytm1,ly1,dwds(1,1,iz),ly1)
            enddo
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vy(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vy(i,1,1,ie)
               dw(i,ie) = dw(i,ie) + dwds(i,1,1)*vy(i,1,1,ie)
            enddo
c
            call mxm   (u(1,1,1,ie),nxy1,dztm1,lz1,duds,lz1)
            call mxm   (v(1,1,1,ie),nxy1,dztm1,lz1,dvds,lz1)
            call mxm   (w(1,1,1,ie),nxy1,dztm1,lz1,dwds,lz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vz(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vz(i,1,1,ie)
               dw(i,ie) = dw(i,ie) + dwds(i,1,1)*vz(i,1,1,ie)
            enddo
         else
c           2D
            call mxm   (dxm1,lx1,u(1,1,1,ie),lx1,du(1,ie),nyz1)
            call mxm   (dxm1,lx1,v(1,1,1,ie),lx1,dv(1,ie),nyz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie)*vx(i,1,1,ie)
               dv(i,ie) = dv(i,ie)*vx(i,1,1,ie)
            enddo
c
            call mxm (u(1,1,1,ie),lx1,dytm1,ly1,duds,ly1)
            call mxm (v(1,1,1,ie),lx1,dytm1,ly1,dvds,ly1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vy(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vy(i,1,1,ie)
            enddo
c
         endif
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine velconvv(vxn,vyn,vzn,tau)
c
c     Compute convecting velocity field (linearization)
c
      include 'SIZE'
      include 'GEOM'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
      real vxn(1),vyn(1),vzn(1)
c
      include 'OPCTR'

c     Operation count
#ifdef TIMER
      integer opct
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'velcvv'
      endif
      ncall(myrout) = ncall(myrout) + 1
      opct = 0
#endif

      call velchar (vx,vxn,vxlag,nbd,tau,dtlag)
      call velchar (vy,vyn,vylag,nbd,tau,dtlag)
      if (ldim.eq.3) 
     $call velchar (vz,vzn,vzlag,nbd,tau,dtlag)
c
      ntot = lx1*ly1*lz1*nelv
      if (ldim.eq.3) then
         do i=1,ntot
c
            tx = vx(i,1,1,1)*bm1(i,1,1,1)*jacmi(i,1)
            ty = vy(i,1,1,1)*bm1(i,1,1,1)*jacmi(i,1)
            tz = vz(i,1,1,1)*bm1(i,1,1,1)*jacmi(i,1)
c
            vx(i,1,1,1) = tx*rxm1(i,1,1,1)
     $                  + ty*rym1(i,1,1,1)
     $                  + tz*rzm1(i,1,1,1)
            vy(i,1,1,1) = tx*sxm1(i,1,1,1)
     $                  + ty*sym1(i,1,1,1)
     $                  + tz*szm1(i,1,1,1)
            vz(i,1,1,1) = tx*txm1(i,1,1,1)
     $                  + ty*tym1(i,1,1,1)
     $                  + tz*tzm1(i,1,1,1)
         enddo
      else
         do i=1,ntot
c
            tx = vx(i,1,1,1)*bm1(i,1,1,1)*jacmi(i,1)
            ty = vy(i,1,1,1)*bm1(i,1,1,1)*jacmi(i,1)
c
            vx(i,1,1,1) = tx*rxm1(i,1,1,1)
     $                  + ty*rym1(i,1,1,1)
            vy(i,1,1,1) = tx*sxm1(i,1,1,1)
     $                  + ty*sym1(i,1,1,1)
         enddo
      endif
c
#ifdef TIMER
      if (ldim.eq.3) then
         opct = ntot*21
      else
         opct = ntot*10
      endif
      dct(myrout) = dct(myrout) + opct
      dcount      =      dcount + opct
#endif
C
c
      return
      end
c-----------------------------------------------------------------------
      subroutine frkconvv (du,dv,dw,u,v,w,mu)
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
c
      real  du(1),dv(1),dw(1)
      real  u (1),v (1),w (1)
      integer mu(0:1)
c
      include 'OPCTR'
c
c     Operation count
c
#ifdef TIMER
      integer opct
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'frkcvv'
      endif
      ncall(myrout) = ncall(myrout) + 1
#endif
c
c
c     Evaluate right-hand-side for Runge-Kutta scheme in the case of
c     pure convection.
c
      ntot = lx1*ly1*lz1*nelv
      call conv1rk   (du,dv,dw,u,v,w)
      CALL OPDSSUM   (du,dv,dw)
c
      if (ldim.eq.3) then
         do i=1,ntot
            du(i) = du(i)*binvm1(i,1,1,1)
            dv(i) = dv(i)*binvm1(i,1,1,1)
            dw(i) = dw(i)*binvm1(i,1,1,1)
         enddo
      else
         do i=1,ntot
            du(i) = du(i)*binvm1(i,1,1,1)
            dv(i) = dv(i)*binvm1(i,1,1,1)
         enddo
      endif
c
c     Mask
c
      nu = mu(0)
      if (ldim.eq.3) then
         do i=1,nu
            du(mu(i)) = 0.
            dv(mu(i)) = 0.
            dw(mu(i)) = 0.
         enddo
      else
         do i=1,nu
            du(mu(i)) = 0.
            dv(mu(i)) = 0.
         enddo
      endif
c
#ifdef TIMER
      opct = ldim*ntot
      dct(myrout) = dct(myrout) + opct
      dcount      =      dcount + opct
#endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine conv1rk2(du,dv,dw,u,v,w,cu,cv,cw,beta,wk)
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
c
      real  du(lx1*ly1*lz1,1),dv(lx1*ly1*lz1,1),dw(lx1*ly1*lz1,1)
      real  u (lx1,ly1,lz1,1),v (lx1,ly1,lz1,1),w (lx1,ly1,lz1,1)
      real  cu(lx1,ly1,lz1,1),cv(lx1,ly1,lz1,1),cw(lx1,ly1,lz1,1)
      real  wk(lx1,ly1,lz1,3)
c
      common /ctmp0/ duds(lx1,ly1,lz1)
     $             , dvds(lx1,ly1,lz1)
     $             , dwds(lx1,ly1,lz1)
C
      include 'OPCTR'
c
c     Operation count
c
#ifdef TIMER
      integer opct
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'cv1rk2'
      endif
      ncall(myrout) = ncall(myrout) + 1
      opct = 0
#endif
c
      nel = nelv
      if (imesh.eq.2) nel = nelt
      nxy1  = lx1*ly1
      nyz1  = ly1*lz1
      nxyz1 = lx1*ly1*lz1
      ntot  = nxyz1*nel
C
C     Compute vel.grad(u)
C
      do ie=1,nel
C
         if (if3d) then
            do i=1,nxyz1
               wk(i,1,1,1)=u(i,1,1,ie)+beta*cu(i,1,1,ie)
               wk(i,1,1,2)=v(i,1,1,ie)+beta*cv(i,1,1,ie)
               wk(i,1,1,3)=w(i,1,1,ie)+beta*cw(i,1,1,ie)
            enddo
c
            call mxm   (dxm1,lx1,wk(1,1,1,1),lx1,du(1,ie),nyz1)
            call mxm   (dxm1,lx1,wk(1,1,1,2),lx1,dv(1,ie),nyz1)
            call mxm   (dxm1,lx1,wk(1,1,1,3),lx1,dw(1,ie),nyz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie)*vx(i,1,1,ie)
               dv(i,ie) = dv(i,ie)*vx(i,1,1,ie)
               dw(i,ie) = dw(i,ie)*vx(i,1,1,ie)
            enddo
c
            do iz=1,lz1
              call mxm (wk(1,1,iz,1),lx1,dytm1,ly1,duds(1,1,iz),ly1)
            enddo
            do iz=1,lz1
              call mxm (wk(1,1,iz,2),lx1,dytm1,ly1,dvds(1,1,iz),ly1)
            enddo
            do iz=1,lz1
              call mxm (wk(1,1,iz,3),lx1,dytm1,ly1,dwds(1,1,iz),ly1)
            enddo
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vy(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vy(i,1,1,ie)
               dw(i,ie) = dw(i,ie) + dwds(i,1,1)*vy(i,1,1,ie)
            enddo
c
            call mxm   (wk(1,1,1,1),nxy1,dztm1,lz1,duds,lz1)
            call mxm   (wk(1,1,1,2),nxy1,dztm1,lz1,dvds,lz1)
            call mxm   (wk(1,1,1,3),nxy1,dztm1,lz1,dwds,lz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vz(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vz(i,1,1,ie)
               dw(i,ie) = dw(i,ie) + dwds(i,1,1)*vz(i,1,1,ie)
            enddo
         else
c           2D
            do i=1,nxyz1
               wk(i,1,1,1)=u(i,1,1,ie)+beta*cu(i,1,1,ie)
               wk(i,1,1,2)=v(i,1,1,ie)+beta*cv(i,1,1,ie)
            enddo
c
            call mxm   (dxm1,lx1,wk(1,1,1,1),lx1,du(1,ie),nyz1)
            call mxm   (dxm1,lx1,wk(1,1,1,2),lx1,dv(1,ie),nyz1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie)*vx(i,1,1,ie)
               dv(i,ie) = dv(i,ie)*vx(i,1,1,ie)
            enddo
c
            call mxm (wk(1,1,1,1),lx1,dytm1,ly1,duds,ly1)
            call mxm (wk(1,1,1,2),lx1,dytm1,ly1,dvds,ly1)
            do i=1,nxyz1
               du(i,ie) = du(i,ie) + duds(i,1,1)*vy(i,1,1,ie)
               dv(i,ie) = dv(i,ie) + dvds(i,1,1)*vy(i,1,1,ie)
            enddo
         endif
c
      enddo
c
#ifdef TIMER
      if (if3d) then
         opct = 21*ntot
      else
         opct = 10*ntot
      endif
c
      dct(myrout) = dct(myrout) + opct
      dcount      =      dcount + opct
#endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine frkconvv2(du,dv,dw,u,v,w,cu,cv,cw,beta,mu,wk)
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
c
      real  du(1),dv(1),dw(1)
      real  u (1),v (1),w (1)
      real  cu(1),cv(1),cw(1)
      real  wk(lx1*ly1*lz1,3)
      integer mu(0:1)
c
      include 'OPCTR'
c
c     Operation count
c
#ifdef TIMER
      integer opct
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'frkcv2'
      endif
      ncall(myrout) = ncall(myrout) + 1
#endif
c
c
c     Evaluate right-hand-side for Runge-Kutta scheme in the case of
c     pure convection.
c
C
      ntot = lx1*ly1*lz1*nelv
      call conv1rk2  (du,dv,dw,u,v,w,cu,cv,cw,beta,wk)
      CALL OPDSSUM   (du,dv,dw)
c
      if (ldim.eq.3) then
         do i=1,ntot
            du(i) = du(i)*binvm1(i,1,1,1)
            dv(i) = dv(i)*binvm1(i,1,1,1)
            dw(i) = dw(i)*binvm1(i,1,1,1)
         enddo
      else
         do i=1,ntot
            du(i) = du(i)*binvm1(i,1,1,1)
            dv(i) = dv(i)*binvm1(i,1,1,1)
         enddo
      endif
c
c     Mask
c
      nu = mu(0)
      if (ldim.eq.3) then
         do i=1,nu
            du(mu(i)) = 0.
            dv(mu(i)) = 0.
            dw(mu(i)) = 0.
         enddo
      else
         do i=1,nu
            du(mu(i)) = 0.
            dv(mu(i)) = 0.
         enddo
      endif
c
#ifdef TIMER
      opct = ldim*ntot
      dct(myrout) = dct(myrout) + opct
      dcount      =      dcount + opct
#endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine hypmsk3v(msk,mask)
C---------------------------------------------------------------------
C
C     Generate mask-array for the hyperbolic system (velocity).
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
      integer msk(0:1)
      CHARACTER      CB*3
      PARAMETER (LXYZ1=LX1*LY1*LZ1)
      COMMON /CTMP1/ WORK(LXYZ1,LELT)
      real mask(lxyz1,lelt)
C
      NFACES= 2*ldim
      NTOT1 = lx1*ly1*lz1*NELV
      CALL RZERO (WORK  ,NTOT1)
      CALL RONE  (mask,NTOT1)
C
      IF (IFIELD.EQ.1) THEN
      DO 100 IE=1,NELV
      DO 100 IFACE=1,NFACES
         CB=CBC(IFACE,IE,IFIELD)
         IF (CB(1:1).EQ.'V' .OR. CB(1:1).EQ.'v') THEN
           CALL FACCL3 (WORK(1,IE),VX(1,1,1,IE),UNX(1,1,IFACE,IE),IFACE)
           CALL FADDCL3(WORK(1,IE),VY(1,1,1,IE),UNY(1,1,IFACE,IE),IFACE)
           IF (IF3D) 
     $     CALL FADDCL3(WORK(1,IE),VZ(1,1,1,IE),UNZ(1,1,IFACE,IE),IFACE)
           CALL FCAVER (VAVER,WORK,IE,IFACE)
C
           IF (VAVER.LT.0.) CALL FACEV (mask,IE,IFACE,0.0,lx1,ly1,lz1)
         ENDIF
         IF (CB(1:2).EQ.'WS' .OR. CB(1:2).EQ.'ws') 
     $   CALL FACEV (mask,IE,IFACE,0.0,lx1,ly1,lz1)
 100   CONTINUE
      ENDIF
C
      nm = 0
      ntot = lx1*ly1*lz1*nelv
      do i=1,ntot
         if (mask(i,1).eq.0) then
            nm = nm+1
            msk(nm) = i
         endif
      enddo
      msk(0) = nm
C
      return
      END
c-----------------------------------------------------------------------
      subroutine ophyprk (vel1,vel2,vel3,ilag)
C---------------------------------------------------------------------------
C
C     Convection of all velocity components.
C     Runge-Kutta scheme.
C
C--------------------------------------------------------------------------
      include 'SIZE'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
C
      REAL           VEL1  (LX1,LY1,LZ1,1)
      REAL           VEL2  (LX1,LY1,LZ1,1)
      REAL           VEL3  (LX1,LY1,LZ1,1)
      COMMON /SCRNS/ VXN   (LX1,LY1,LZ1,LELV)
     $ ,             VYN   (LX1,LY1,LZ1,LELV)
     $ ,             VZN   (LX1,LY1,LZ1,LELV)
     $ ,             HV1MSK(LX1,LY1,LZ1,LELV)
     $ ,             HV2MSK(LX1,LY1,LZ1,LELV)
     $ ,             HV3MSK(LX1,LY1,LZ1,LELV)
     $ ,             WORK  (LX1,LY1,LZ1,LELV)
      COMMON /CTMP1/ RKX1  (LX1,LY1,LZ1,LELV)
     $ ,             RKX2  (LX1,LY1,LZ1,LELV)
     $ ,             RKX3  (LX1,LY1,LZ1,LELV)
     $ ,             RKX4  (LX1,LY1,LZ1,LELV)
      COMMON /SCRMG/ RKY1  (LX1,LY1,LZ1,LELV)
     $ ,             RKY2  (LX1,LY1,LZ1,LELV)
     $ ,             RKY3  (LX1,LY1,LZ1,LELV)
     $ ,             RKY4  (LX1,LY1,LZ1,LELV)
      COMMON /SCREV/ RKZ1  (LX1,LY1,LZ1,LELV)
     $ ,             RKZ2  (LX1,LY1,LZ1,LELV)
      COMMON /SCRCH/ RKZ3  (LX1,LY1,LZ1,LELV)
     $ ,             RKZ4  (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
C
      CALL OPCOPY  (VXN,VYN,VZN,VX,VY,VZ)
      CALL HYPMSK3 (HV1MSK,HV2MSK,HV3MSK)
      CALL TAUINIT (TAU,ILAG)
      CALL VELINIT (VEL1,VEL2,VEL3,ILAG)
      CALL VELCONV (VXN,VYN,VZN,TAU)
C
      DO 1000 JLAG=ILAG,1,-1
C
         DTAU   = DTLAG(JLAG)/(NTAUBD)
         DTHALF = DTAU/2.
         CRK1   = DTAU/6.
         CRK2   = DTAU/3.
C
         DO 900 ITAU=1,NTAUBD
C
C           Stage 1.
C
            CALL FRKCONV (RKX1,VEL1,HV1MSK)
            CALL FRKCONV (RKY1,VEL2,HV2MSK)
            IF (ldim.EQ.3) 
     $      CALL FRKCONV (RKZ1,VEL3,HV3MSK)
C
C
C           Stage 2.
C
            TAU = TAU + DTHALF
            CALL VELCONV (VXN,VYN,VZN,TAU)
C
            CALL COPY    (WORK,VEL1,NTOT1)
            CALL ADD2S2  (WORK,RKX1,-DTHALF,NTOT1)
            CALL FRKCONV (RKX2,WORK,HV1MSK)
C
            CALL COPY    (WORK,VEL2,NTOT1)
            CALL ADD2S2  (WORK,RKY1,-DTHALF,NTOT1)
            CALL FRKCONV (RKY2,WORK,HV2MSK)
C
            IF (ldim.EQ.3) THEN
               CALL COPY    (WORK,VEL3,NTOT1)
               CALL ADD2S2  (WORK,RKZ1,-DTHALF,NTOT1)
               CALL FRKCONV (RKZ2,WORK,HV3MSK)
            ENDIF
C
C
C           Stage 3.
C
            CALL COPY    (WORK,VEL1,NTOT1)
            CALL ADD2S2  (WORK,RKX2,-DTHALF,NTOT1)
            CALL FRKCONV (RKX3,WORK,HV1MSK)

            CALL COPY    (WORK,VEL2,NTOT1)
            CALL ADD2S2  (WORK,RKY2,-DTHALF,NTOT1)
            CALL FRKCONV (RKY3,WORK,HV2MSK)
C
            IF (ldim.EQ.3) THEN
               CALL COPY    (WORK,VEL3,NTOT1)
               CALL ADD2S2  (WORK,RKZ2,-DTHALF,NTOT1)
               CALL FRKCONV (RKZ3,WORK,HV3MSK)
            ENDIF
C
C
C           Stage 4.
C
            TAU = TAU + DTHALF
            CALL VELCONV (VXN,VYN,VZN,TAU)
C
            CALL COPY    (WORK,VEL1,NTOT1)
            CALL ADD2S2  (WORK,RKX3,-DTAU,NTOT1)
            CALL FRKCONV (RKX4,WORK,HV1MSK)

            CALL COPY    (WORK,VEL2,NTOT1)
            CALL ADD2S2  (WORK,RKY3,-DTAU,NTOT1)
            CALL FRKCONV (RKY4,WORK,HV2MSK)
C
            IF (ldim.EQ.3) THEN
               CALL COPY    (WORK,VEL3,NTOT1)
               CALL ADD2S2  (WORK,RKZ3,-DTAU,NTOT1)
               CALL FRKCONV (RKZ4,WORK,HV3MSK)
            ENDIF
C
C
C           Sum up contributions from 4 stages.
C
            CALL ADD2S2  (VEL1,RKX1,-CRK1,NTOT1)
            CALL ADD2S2  (VEL1,RKX2,-CRK2,NTOT1)
            CALL ADD2S2  (VEL1,RKX3,-CRK2,NTOT1)
            CALL ADD2S2  (VEL1,RKX4,-CRK1,NTOT1)
C
            CALL ADD2S2  (VEL2,RKY1,-CRK1,NTOT1)
            CALL ADD2S2  (VEL2,RKY2,-CRK2,NTOT1)
            CALL ADD2S2  (VEL2,RKY3,-CRK2,NTOT1)
            CALL ADD2S2  (VEL2,RKY4,-CRK1,NTOT1)
C
            IF (ldim.EQ.3) THEN
               CALL ADD2S2  (VEL3,RKZ1,-CRK1,NTOT1)
               CALL ADD2S2  (VEL3,RKZ2,-CRK2,NTOT1)
               CALL ADD2S2  (VEL3,RKZ3,-CRK2,NTOT1)
               CALL ADD2S2  (VEL3,RKZ4,-CRK1,NTOT1)
            ENDIF
C
  900    CONTINUE
 1000 CONTINUE
C
      CALL OPCOPY (VX,VY,VZ,VXN,VYN,VZN)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine opdiv(outfld,inpx,inpy,inpz)
C---------------------------------------------------------------------
C
C     Compute OUTFLD = SUMi Di*INPi, 
C     the divergence of the vector field (INPX,INPY,INPZ)
C
C---------------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      real outfld (lx2,ly2,lz2,1)
      real inpx   (lx1,ly1,lz1,1)
      real inpy   (lx1,ly1,lz1,1)
      real inpz   (lx1,ly1,lz1,1)
      common /ctmp0/ work (lx2,ly2,lz2,lelv)
C
      iflg = 1

      ntot2 = lx2*ly2*lz2*nelv
      call multd (work,inpx,rxm2,sxm2,txm2,1,iflg)
      call copy  (outfld,work,ntot2)
      call multd (work,inpy,rym2,sym2,tym2,2,iflg)
      call add2  (outfld,work,ntot2)
      if (ldim.eq.3) then
         call multd (work,inpz,rzm2,szm2,tzm2,3,iflg)
         call add2  (outfld,work,ntot2)
      endif
C
      return
      end
C
c-----------------------------------------------------------------------
      subroutine opgradt(outx,outy,outz,inpfld)
C------------------------------------------------------------------------
C
C     Compute DTx, DTy, DTz of an input field INPFLD 
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      real outx   (lx1,ly1,lz1,1)
      real outy   (lx1,ly1,lz1,1)
      real outz   (lx1,ly1,lz1,1)
      real inpfld (lx2,ly2,lz2,1)
C
      call cdtp (outx,inpfld,rxm2,sxm2,txm2,1)
      call cdtp (outy,inpfld,rym2,sym2,tym2,2)
      if (ldim.eq.3) 
     $   call cdtp (outz,inpfld,rzm2,szm2,tzm2,3)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine setproj(n1,nd)
c
      include 'SIZE'
      include 'DEALIAS'
      include 'INPUT'
c
      parameter(lx=80)
      real LkN(lx,lx),LkD(lx,lx),LkNt(lx,lx)
c
      if (n1.gt.lx.or.nd.gt.lx) then
         write(6,*)'ERROR: increase lx in setmap to max:',n1,nd
         call exitt
      endif
c
c
      if (param(99).eq.2) then
         call set_PND  (PmD1 ,LkD,LkNt,n1,nd)
      elseif (param(99).eq.3) then
         call set_PNDoi(PmD1 ,LkD,LkNt,n1,nd)
      endif
      call transpose(PmD1t,nd,PmD1,n1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_PNDoi(Pt,P,LkNt,N,D)

      include 'SIZE'   ! for write stmt

c
c     Set up operators for overintegration and interpolation
c
      integer N,D
      real    Pt(N,D),P(D,N),LkNt(N,0:N-1)
c
      parameter(lx=80)
      real zN(lx),zD(lx),wN(lx),wD(lx)
c
c     Compute Lagrangian interpolant points
c
      call zwgll(zN,wN,N)
      call zwgll(zD,wD,D)
c
      if (nio.eq.0) write(6,*) 'dealias, pndoi:',N,D
      call IGLLM (P,Pt,ZN,ZD,N,D,N,D)
c
      do j=1,D
      do i=1,N
         Pt(i,j) = wD(j)*Pt(i,j)/wN(i)
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine wgradm1(ux,uy,uz,u,nel) ! weak form of grad 
c
c     Compute gradient of T -- mesh 1 to mesh 1 (vel. to vel.)
c
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'WZ'
c
      parameter (lxyz=lx1*ly1*lz1)
      real ux(lxyz,1),uy(lxyz,1),uz(lxyz,1),u(lxyz,1)
c
      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz)

      integer e

      N = lx1-1
      do e=1,nel
         if (if3d) then
            call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
            do i=1,lxyz
               ux(i,e) = w3m1(i,1,1)*(ur(i)*rxm1(i,1,1,e)
     $                              + us(i)*sxm1(i,1,1,e)
     $                              + ut(i)*txm1(i,1,1,e) )
               uy(i,e) = w3m1(i,1,1)*(ur(i)*rym1(i,1,1,e)
     $                              + us(i)*sym1(i,1,1,e)
     $                              + ut(i)*tym1(i,1,1,e) )
               uz(i,e) = w3m1(i,1,1)*(ur(i)*rzm1(i,1,1,e)
     $                              + us(i)*szm1(i,1,1,e)
     $                              + ut(i)*tzm1(i,1,1,e) )
            enddo
         else

            if (ifaxis) then
               call setaxdy (ifrzer(e))  ! reset dytm1
               call setaxw1 (ifrzer(e))  ! reset w3m1
            endif

            call local_grad2(ur,us,u,N,e,dxm1,dytm1)

            do i=1,lxyz
               ux(i,e) =w3m1(i,1,1)*(ur(i)*rxm1(i,1,1,e)
     $                             + us(i)*sxm1(i,1,1,e) )
               uy(i,e) =w3m1(i,1,1)*(ur(i)*rym1(i,1,1,e)
     $                             + us(i)*sym1(i,1,1,e) )
            enddo
         endif

      enddo
c
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE MAKEVIS
C----------------------------------------------------------------------
C
C     construct viscous term:
c
c     DEL*[mue*(DEL V + (DEL V)^T - 2/3*DEL*V *I)]
c                 =  
c     2*DEL*[mue*(S - 1/3*QTL*I)]
c
c     where mue is the viscosity, S the strain rate tensor,
c     tr(S) the trace of S and I the indentitiy matrix. 
c
c     NOTE: for now only for incompressible flows
c           In the compressible case we need to compute S using 
c           a kth-order extrapolation scheme because we cannot 
c           use the existing MAKEABF extrapolater and we need
c           to use mue/QTL from the thermo-chemical subsystem.
c           CAUTION: we cannot use BFX,BFY,BFZ anymore. Some 
c                    extra handling in plan4 is needed to add the
c                    viscous term to the RHS!   
 
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
C
      COMMON /SCRUZ/ W1 (LX1,LY1,LZ1,LELV)
     $ ,             W2 (LX1,LY1,LZ1,LELV)
     $ ,             W3 (LX1,LY1,LZ1,LELV)
C
      COMMON /SCRNS/ SXZ(LX1,LY1,LZ1,LELT)
     $             , SYZ(LX1,LY1,LZ1,LELT)
     $             , SXX(LX1,LY1,LZ1,LELT)
     $             , SXY(LX1,LY1,LZ1,LELT)
     $             , SYY(LX1,LY1,LZ1,LELT)
     $             , SZZ(LX1,LY1,LZ1,LELT)

      REAL fac
C----------------------------------------------------------------------

      NTOT = lx1*ly1*lz1*NELV

      ! CONSTRUCT strain rate tensor S (SXX, ..., SZZ)
      ! CALL MAKEABS
      CALL COMP_SIEJ(VX,VY,VZ) 

      ! substract trace of S
c      CALL ADD4  (W1,SXX,SYY,SZZ,NTOT)
      CALL COPY  (W1,QTL,ntot)  
      fac = -1./3. 
      CALL CMULT (W1,fac,NTOT)
      CALL ADD2  (SXX,W1,NTOT)
      CALL ADD2  (SYY,W1,NTOT)
      CALL ADD2  (SZZ,W1,NTOT)

      CALL OPCOLV(SXX,SYY,SZZ,VDIFF_E)
      CALL OPCOLV(SXY,SXZ,SYZ,VDIFF_E)

      ! not sure if that is really needed
      CALL OPCOLV (SXX,SYY,SZZ,BM1)
      CALL OPCOLV (SXY,SXZ,SYZ,BM1)
      CALL OPDSSUM(SXX,SYY,SZZ)
      CALL OPDSSUM(SXY,SXZ,SYZ)
      CALL OPCOLV (SXX,SYY,SZZ,BINVM1)
      CALL OPCOLV (SXY,SXZ,SYZ,BINVM1)

      CALL RONE(W2,NTOT)
      fac = 2.0
      CALL CMULT  (W2,fac,NTOT)
      
c add to RHS (BFX,BFY,BFZ)
      CALL OPDIV (W1,SXX,SXY,SXZ)
      CALL COL2  (W1,W2,NTOT)
      CALL ADD2  (BFX,W1,NTOT)
      
      CALL OPDIV (W1,SXY,SYY,SYZ)
      CALL COL2  (W1,W2,NTOT)
      CALL ADD2  (BFY,W1,NTOT)

      IF (ldim.EQ.3) THEN
        CALL OPDIV (W1,SXZ,SYZ,SZZ)
        CALL COL2  (W1,W2,NTOT)
        CALL ADD2  (BFZ,W1,NTOT)
      ENDIF


      RETURN
      END
c-----------------------------------------------------------------------

      SUBROUTINE COMP_SIEJ (U1,U2,U3)
C
C     Compute strainrates
C
C     CAUTION : CB SCRNS is used for data change
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'

      COMMON /SCRNS/ EXZ(LX1,LY1,LZ1,LELT)
     $             , EYZ(LX1,LY1,LZ1,LELT)
     $             , EXX(LX1,LY1,LZ1,LELT)
     $             , EXY(LX1,LY1,LZ1,LELT)
     $             , EYY(LX1,LY1,LZ1,LELT)
     $             , EZZ(LX1,LY1,LZ1,LELT)

C
      DIMENSION U1(LX1,LY1,LZ1,1)
     $        , U2(LX1,LY1,LZ1,1)
     $        , U3(LX1,LY1,LZ1,1)


      NEL = NELV
      NTOT1 = lx1*ly1*lz1*NEL

      CALL RZERO3 (EXX,EYY,EZZ,NTOT1)
      CALL RZERO3 (EXY,EXZ,EYZ,NTOT1)
C
      CALL UXYZ  (U1,EXX,EXY,EXZ,NEL)
      CALL UXYZ  (U2,EXY,EYY,EYZ,NEL)
      IF (ldim.EQ.3) CALL UXYZ   (U3,EXZ,EYZ,EZZ,NEL)
C
      CALL COL2 (EXX,JACMi,NTOT1)
      CALL COL2 (EXY,JACMi,NTOT1)
      CALL COL2 (EYY,JACMi,NTOT1)
C
      IF (IFAXIS) CALL AXIEZZ (U2,EYY,EZZ,NEL)
C
      IF (ldim.EQ.3) THEN
         CALL COL2 (EXZ,JACMi,NTOT1)
         CALL COL2 (EYZ,JACMi,NTOT1)
         CALL COL2 (EZZ,JACMi,NTOT1)
      ENDIF
C
      fac = 0.5
      CALL CMULT (EXY,fac,NTOT1) 
      CALL CMULT (EXZ,fac,NTOT1)  
      CALL CMULT (EYZ,fac,NTOT1) 

      RETURN
      END
c-----------------------------------------------------------------------
      subroutine wlaplacian(out,a,diff,ifld)
c
c     compute weak form of the laplacian operator including the boundary
c     contribution
c
      include 'SIZE'
      include 'TOTAL'

      real out(1),a(1),diff(1)
      real wrk(lx1,ly1,lz1,lelt)
      real h2(lx1,ly1,lz1,lelt)

      ntot = lx1*ly1*lz1*nelfld(ifld)
      if (.not.iftmsh(ifld)) imesh = 1
      if (     iftmsh(ifld)) imesh = 2

      call rzero(h2,ntot)

      ifield_ = ifield
      ifield = ifld

      call bcneusc(out,1)
      call axhelm(wrk,a,diff,h2,imesh,1)
      call sub2 (out,wrk,ntot)  
 
      ifield = ifield_ 

      return
      end
c-----------------------------------------------------------------------
      subroutine explstrs ! Explicit stress tensor w/ variable viscosity

      include 'SIZE'
      include 'TOTAL'

      common /scruz/ u(lx1*ly1*lz1),v(lx1*ly1*lz1),w(lx1*ly1*lz1)

      integer e

      nxyz = lx1*ly1*lz1


      do e=1,nelv

        call expl_strs_e(u,v,w,vx(1,1,1,e),vy(1,1,1,e),vz(1,1,1,e),e)

        do i=1,nxyz
           bfx(i,1,1,e) = bfx(i,1,1,e) - u(i)
           bfy(i,1,1,e) = bfy(i,1,1,e) - v(i)
           bfz(i,1,1,e) = bfz(i,1,1,e) - w(i)
        enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine expl_strs(w1,w2,w3,u1,u2,u3)
      include 'SIZE'
      include 'TOTAL'

      real w1(1),w2(1),w3(1),u1(1),u2(1),u3(1)

      integer e

      nxyz = lx1*ly1*lz1
      k = 1
      do e=1,nelv

         call expl_strs_e(w1(k),w2(k),w3(k),u1(k),u2(k),u3(k),e)
         k = k+nxyz

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine expl_strs_e(w1,w2,w3,u1,u2,u3,e)
      include 'SIZE'
      include 'INPUT'  ! if3d
      include 'SOLN'   ! nu_star

      real w1(1),w2(1),w3(1),u1(1),u2(1),u3(1)
      integer e

      integer icalld
      save    icalld
      data    icalld /0/

      if (nio.eq.0.and.icalld.eq.0) write(6,*) 'nu_star:',nu_star
      icalld=1

      if (if3d) then
         call expl_strs_e_3d (w1,w2,w3,u1,u2,u3,e)
      else
         call expl_strs_e_2d (w1,w2,u1,u2,e)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine expl_strs_e_3d(w1,w2,w3,u1,u2,u3,e)

c     Evaluate, for element e,
c     
c     /dvi\T  /     dui       duj \
c     |---|   | dnu ---  + nu --- |    (no boundary terms at present)
c     \dxj/   \     dxj       dxi /


      real w1(1),w2(1),w3(1),u1(1),u2(1),u3(1)
      integer e

      include 'SIZE'
      include 'GEOM'    ! jacmi,rxm1, etc.
      include 'INPUT'   ! if3d
      include 'MASS'    ! bm1
      include 'SOLN'    ! vtrans,vdiff,nu_star
      include 'TSTEP'   ! dt
      include 'WZ'      ! w3m1

      real nu

      parameter (lxyz=lx1*ly1*lz1)
      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz)
     $             , vr(lxyz),vs(lxyz),vt(lxyz)
     $             , wr(lxyz),ws(lxyz),wt(lxyz)

      call gradl_rst(ur,us,ut,u1,lx1,if3d) ! Grad on GLL
      call gradl_rst(vr,vs,vt,u2,lx1,if3d)
      call gradl_rst(wr,ws,wt,u3,lx1,if3d)

      do i=1,lxyz

         nu  = vdiff(i,1,1,e,1)
         dnu = nu - nu_star  ! nu_star is the constant implicit part

c        uij := jac*( du_i / dx_j )

         u11=ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e)+ut(i)*txm1(i,1,1,e)
         u21=vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e)+vt(i)*txm1(i,1,1,e)
         u31=wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e)+wt(i)*txm1(i,1,1,e)
         u12=ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e)+ut(i)*tym1(i,1,1,e)
         u22=vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e)+vt(i)*tym1(i,1,1,e)
         u32=wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e)+wt(i)*tym1(i,1,1,e)
         u13=ur(i)*rzm1(i,1,1,e)+us(i)*szm1(i,1,1,e)+ut(i)*tzm1(i,1,1,e)
         u23=vr(i)*rzm1(i,1,1,e)+vs(i)*szm1(i,1,1,e)+vt(i)*tzm1(i,1,1,e)
         u33=wr(i)*rzm1(i,1,1,e)+ws(i)*szm1(i,1,1,e)+wt(i)*tzm1(i,1,1,e)

         w11 = dnu*u11 + nu*u11
         w12 = dnu*u12 + nu*u21
         w13 = dnu*u13 + nu*u31
         w21 = dnu*u21 + nu*u12
         w22 = dnu*u22 + nu*u22
         w23 = dnu*u23 + nu*u32
         w31 = dnu*u31 + nu*u13
         w32 = dnu*u32 + nu*u23
         w33 = dnu*u33 + nu*u33

         w   = w3m1(i,1,1)*jacmi(i,e)  ! note, ry has jac in it.

         ur(i)=(w11*rxm1(i,1,1,e)+w12*rym1(i,1,1,e)+w13*rzm1(i,1,1,e))*w
         us(i)=(w11*sxm1(i,1,1,e)+w12*sym1(i,1,1,e)+w13*szm1(i,1,1,e))*w
         ut(i)=(w11*txm1(i,1,1,e)+w12*tym1(i,1,1,e)+w13*tzm1(i,1,1,e))*w
         vr(i)=(w21*rxm1(i,1,1,e)+w22*rym1(i,1,1,e)+w23*rzm1(i,1,1,e))*w
         vs(i)=(w21*sxm1(i,1,1,e)+w22*sym1(i,1,1,e)+w23*szm1(i,1,1,e))*w
         vt(i)=(w21*txm1(i,1,1,e)+w22*tym1(i,1,1,e)+w23*tzm1(i,1,1,e))*w
         wr(i)=(w31*rxm1(i,1,1,e)+w32*rym1(i,1,1,e)+w33*rzm1(i,1,1,e))*w
         ws(i)=(w31*sxm1(i,1,1,e)+w32*sym1(i,1,1,e)+w33*szm1(i,1,1,e))*w
         wt(i)=(w31*txm1(i,1,1,e)+w32*tym1(i,1,1,e)+w33*tzm1(i,1,1,e))*w

      enddo

      call gradl_rst_t(w1,ur,us,ut,lx1,if3d)
      call gradl_rst_t(w2,vr,vs,vt,lx1,if3d)
      call gradl_rst_t(w3,wr,ws,wt,lx1,if3d)

      return
      end
c-----------------------------------------------------------------------
      subroutine expl_strs_e_2d(w1,w2,u1,u2,e)

c
c     Evaluate, for element e,
c     
c     /dvi\T  /     dui       duj \
c     |---|   | dnu ---  + nu --- |    (no boundary terms at present)
c     \dxj/   \     dxj       dxi /
c


      real w1(1),w2(1),u1(1),u2(1)
      integer e

      include 'SIZE'
      include 'GEOM'    ! jacmi,rxm1, etc.
      include 'INPUT'   ! if3d
      include 'MASS'    ! bm1
      include 'SOLN'    ! vtrans,vdiff,nu_star
      include 'TSTEP'   ! dt
      include 'WZ'      ! w3m1

      real nu

      parameter (lxyz=lx1*ly1*lz1)
      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz)
     $             , vr(lxyz),vs(lxyz),vt(lxyz)
     $             , wr(lxyz),ws(lxyz),wt(lxyz)

      call gradl_rst(ur,us,ut,u1,lx1,if3d) ! Grad on GLL
      call gradl_rst(vr,vs,vt,u2,lx1,if3d)

      do i=1,lxyz

         nu  = vdiff(i,1,1,e,1)
         dnu = nu - nu_star  ! nu_star is the constant implicit part

c        uij := jac*( du_i / dx_j )

         u11=ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e)
         u21=vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e)
         u12=ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e)
         u22=vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e)

         w11 = dnu*u11 + nu*u11
         w12 = dnu*u12 + nu*u21
         w21 = dnu*u21 + nu*u12
         w22 = dnu*u22 + nu*u22

         w   = w3m1(i,1,1)*jacmi(i,e)  ! note, ry has jac in it.

         ur(i)=(w11*rxm1(i,1,1,e)+w12*rym1(i,1,1,e))*w
         us(i)=(w11*sxm1(i,1,1,e)+w12*sym1(i,1,1,e))*w
         vr(i)=(w21*rxm1(i,1,1,e)+w22*rym1(i,1,1,e))*w
         vs(i)=(w21*sxm1(i,1,1,e)+w22*sym1(i,1,1,e))*w

      enddo

      call gradl_rst_t(w1,ur,us,ut,lx1,if3d)
      call gradl_rst_t(w2,vr,vs,vt,lx1,if3d)

      return
      end
c-----------------------------------------------------------------------
      subroutine gradl_rst_t(u,ur,us,ut,md,if3d)  ! GLL grad-transpose
c
c     Thus routine originally from fsi file: u5.usr (May 2010)
c
c
      include 'SIZE'
      include 'DXYZ'

      real    u(1),ur(1),us(1),ut(1)
      logical if3d

c     dgradl holds GLL-based derivative / interpolation operators

      parameter (ldg=lxd**3,lwkd=2*ldg)
      common /dgradl/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      m0 = md-1
      call get_dgll_ptr (ip,md,md)
      if (if3d) then
         call local_grad3_t(u,ur,us,ut,m0,1,d(ip),dt(ip),wkd)
      else
         call local_grad2_t(u,ur,us   ,m0,1,d(ip),dt(ip),wkd)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine gradl_rst(ur,us,ut,u,md,if3d)  ! GLL-based gradient
c
      include 'SIZE'
      include 'DXYZ'

      real    ur(1),us(1),ut(1),u(1)
      logical if3d

c     dgradl holds GLL-based derivative / interpolation operators

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgradl/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      m0 = md-1
      call get_dgll_ptr (ip,md,md)
      if (if3d) then
         call local_grad3(ur,us,ut,u,m0,1,d(ip),dt(ip))
      else
         call local_grad2(ur,us   ,u,m0,1,d(ip),dt(ip))
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad3_t(u,ur,us,ut,N,e,D,Dt,w)
c     Output: ur,us,ut         Input:u,N,e,D,Dt
      real u (0:N,0:N,0:N,1)
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1
      m3 = m1*m1*m1

      call mxm(Dt,m1,ur,m1,u(0,0,0,e),m2)

      do k=0,N
         call mxm(us(0,0,k),m1,D ,m1,w(0,0,k),m1)
      enddo
      call add2(u(0,0,0,e),w,m3)

      call mxm(ut,m2,D ,m1,w,m1)
      call add2(u(0,0,0,e),w,m3)

      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad2_t(u,ur,us,N,e,D,Dt,w)
c     Output: ur,us         Input:u,N,e,D,Dt
      real u (0:N,0:N,1)
      real ur(0:N,0:N),us(0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1

      call mxm(Dt,m1,ur,m1,u(0,0,e),m1)
      call mxm(us,m1,D ,m1,w         ,m1)
      call add2(u(0,0,e),w,m2)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_dgll_ptr (ip,mx,md)
c
c     Get pointer to GLL-GLL interpolation dgl() for pair (mx,md)
c
      include 'SIZE'

c     dgradl holds GLL-based derivative / interpolation operators

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgradl/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt
 
c     Pointers into GLL-based derivative / interpolation operators

      parameter (ld=2*lxd)
      common /jgradl/ pd    (0:ld*ld)
     $              , pdg   (0:ld*ld)
     $              , pjgl  (0:ld*ld)
      integer pd , pdg , pjgl

      ij = md + ld*(mx-1)
      ip = pdg (ij)

      if (ip.eq.0) then

         nstore   = pdg (0)
         pdg (ij) = nstore+1
         nstore   = nstore + md*mx
         pdg (0)  = nstore
         ip       = pdg (ij)

        if (nid.eq.985) write(6,*) nstore,ldg,ij,md,mx,ip,' NSTOR'
c
         nwrkd = mx + md
         call lim_chk(nstore,ldg ,'dg   ','ldg  ','get_dgl_pt')
         call lim_chk(nwrkd ,lwkd,'wkd  ','lwkd ','get_dgl_pt')
c
         call gen_dgll(d (ip),dt(ip),md,mx,wkd)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_dgll(dgl,dgt,mp,np,w)
c
c     Generate derivative from np GL points onto mp GL points
c
c        dgl  = derivative matrix, mapping from velocity nodes to pressure
c        dgt  = transpose of derivative matrix
c        w    = work array of size (3*np+mp)
c
c        np   = number of points on GLL grid
c        mp   = number of points on GL  grid
c
c
c
      real dgl(mp,np),dgt(np*mp),w(1)
c
c
      iz = 1
      id = iz + np
c
      call zwgll (w(iz),dgt,np)  ! GL points
      call zwgll (w(id),dgt,mp)  ! GL points
c
      ndgt = 2*np
      ldgt = mp*np
      call lim_chk(ndgt,ldgt,'ldgt ','dgt  ','gen_dgl   ')
c
      n  = np-1
      do i=1,mp
         call fd_weights_full(w(id+i-1),w(iz),n,1,dgt) ! 1=1st deriv.
         do j=1,np
            dgl(i,j) = dgt(np+j)                       ! Derivative matrix
         enddo
      enddo
c
      call transpose(dgt,np,dgl,mp)
c
      return
      end
c-----------------------------------------------------------------------
