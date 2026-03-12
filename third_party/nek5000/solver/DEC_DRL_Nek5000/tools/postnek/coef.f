c-----------------------------------------------------------------------
      SUBROUTINE COEF
C
C     GENERATE
C
C            - DERIVATIVE OPERATORS
C            - WEIGHTS
C            - COLLOCATION POINTS
C
C     ASSOCIATED WITH THE
C
C            - GAUSS-LEGENDRE LOBATTO MESH (SUFFIX M1)
C
C-----------------------------------------------------------------
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      REAL PN(nxm)
C
C     Compute derivative matrices
C
      CALL LEGEND (ZPTS,WGHT,NX)
      CALL GENPN1 (PN,ZPTS,NX)
      CALL DERMAT (DGDR,DGDRT,PN,ZPTS,NX,NX )
C
      DO 100 IE=1,NEL
         CALL GEOM1   (IE)
  100 CONTINUE
C
C     Set up box sizes associated with each element defined to be
C     the circumscribing parallelagram for the each element.
C
      NXYZ=NX*NY*NZ
      DO 200 IE=1,NEL
         IEOFF=NXYZ*(IE-1)+1
         XBMIN(IE) = ALMIN(XP(IEOFF),NXYZ)
         XBMAX(IE) = ALMAX(XP(IEOFF),NXYZ)
         YBMIN(IE) = ALMIN(YP(IEOFF),NXYZ)
         YBMAX(IE) = ALMAX(YP(IEOFF),NXYZ)
         IF (if3d) THEN
            ZBMIN(IE) = ALMIN(ZP(IEOFF),NXYZ)
            ZBMAX(IE) = ALMAX(ZP(IEOFF),NXYZ)
         ELSE
            ZBMIN(IE)=-2.0
            ZBMAX(IE)= 2.0
         ENDIF
C
C        Increase "box size" by a fraction to eliminate potential gaps between
C        elements
C
         DELTA =  ( XBMAX(IE) - XBMIN(IE) ) /20.
         XBMAX(IE) = XBMAX(IE) + DELTA
         XBMIN(IE) = XBMIN(IE) - DELTA
         DELTA =  ( YBMAX(IE) - YBMIN(IE) ) /20.
         YBMAX(IE) = YBMAX(IE) + DELTA
         YBMIN(IE) = YBMIN(IE) - DELTA
         DELTA =  ( ZBMAX(IE) - ZBMIN(IE) ) /20.
         ZBMAX(IE) = ZBMAX(IE) + DELTA
         ZBMIN(IE) = ZBMIN(IE) - DELTA
C
  200 CONTINUE
c
c     Set up weighted multiplicity --- Jacobian based
c
      ntot=nx*ny*nz*nel
      rmax1 = 0.
      rmax2 = 0.
      rmax3 = 0.
      do i=1,ntot
         mult(i) = abs(jacm1(i))
         rmax1 = max(rmax1,abs(mult(i)))
      enddo
      ifwkclobber = .true.
      call dssum(mult,wkv2)
      do i=1,ntot
         rmax2 = max(rmax2,abs(mult(i)))
         if (mult(i).ne.0) mult(i) = abs(jacm1(i))/mult(i)
         rmax3 = max(rmax3,abs(mult(i)))
      enddo
      write(6,*) 'THIS IS MULT MAX',rmax1,rmax2,rmax3,ntot
c
c     Find element connectivity
c
      call fcnnct
c
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GEOM1 (ie)
C-------------------------------------------------------------------
C
C     Routine to generate elemental geometry information on mesh M1,
C     (Gauss-Legendre Lobatto mesh).
C
C         RXM1,  RYM1,  RZM1   -   dr/dx, dr/dy, dr/dz
C         SXM1,  SYM1,  SZM1   -   ds/dx, ds/dy, ds/dz
C         TXM1,  TYM1,  TZM1   -   dt/dx, dt/dy, dt/dz
C         JACM1                -   Jacobian
C         BM1                  -   Mass matrix
C
C------------------------------------------------------------------
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      PARAMETER (NXM3=NXM*NXM*NXM)
      common /ctmp4/ xrm1,xsm1,xtm1,yrm1,ysm1,ytm1,zrm1,zsm1,ztm1
      REAL XRM1 (NXM3), XSM1(NXM3), XTM1(NXM3)
      REAL YRM1 (NXM3), YSM1(NXM3), YTM1(NXM3)
      REAL ZRM1 (NXM3), ZSM1(NXM3), ZTM1(NXM3)
c
c
      nxyz1=nx*ny*nz
      nyz1 =ny*nz
      ioff = nxyz1*(ie-1) + 1
C
C     Compute isoparametric partials.
C
C     XR=[DX]ip*[X]pjk
C
c
      CALL MXM(DGDR,NX,XP(ioff),NX,XRM1,NYZ1)
      CALL MXM(DGDR,NX,YP(ioff),NX,YRM1,NYZ1)
      CALL MXM(DGDR,NX,ZP(ioff),NX,ZRM1,NYZ1)
C
      DO 10 IZ=1,NZ
         IZZ=NX*NY*(IZ-1)
         IZ1=izz+1
         CALL MXM(XP(ioff+izz),NX,DGDRT,NY,XSM1(IZ1),NY)
         CALL MXM(YP(ioff+izz),NX,DGDRT,NY,YSM1(IZ1),NY)
         CALL MXM(ZP(ioff+izz),NX,DGDRT,NY,ZSM1(IZ1),NY)
   10 CONTINUE
C
      NXY1=NX*NY
      IF (if3d) THEN
         CALL MXM(XP(ioff),NXY1,DGDRT,NZ,XTM1,NZ)
         CALL MXM(YP(ioff),NXY1,DGDRT,NZ,YTM1,NZ)
         CALL MXM(ZP(ioff),NXY1,DGDRT,NZ,ZTM1,NZ)
      ELSE
         DO 20 I=1,NXY1
            XTM1(I)=0.
            YTM1(I)=0.
            ZTM1(I)=1.
  20     CONTINUE
      ENDIF
      IF (IFERROR) THEN
         DO I=1,NXY1
            RXM1(I)=XRM1(I)
            RYM1(I)=YRM1(I)
            RZM1(I)=ZRM1(I)
            SXM1(I)=XSM1(I)
            SYM1(I)=YSM1(I)
            SZM1(I)=ZSM1(I)
            TXM1(I)=XTM1(I)
            TYM1(I)=YTM1(I)
         ENDDO
         RETURN
      ENDIF
C
C     Compute Jacobian
C
      CALL RZERO(JACM1(ioff),NXYZ1)
C
      CALL ADDCOL4 (JACM1(ioff),XRM1,YSM1,ZTM1,NXYZ1)
      CALL ADDCOL4 (JACM1(ioff),XTM1,YRM1,ZSM1,NXYZ1)
      CALL ADDCOL4 (JACM1(ioff),XSM1,YTM1,ZRM1,NXYZ1)
      CALL SUBCOL4 (JACM1(ioff),XRM1,YTM1,ZSM1,NXYZ1)
      CALL SUBCOL4 (JACM1(ioff),XSM1,YRM1,ZTM1,NXYZ1)
      CALL SUBCOL4 (JACM1(ioff),XTM1,YSM1,ZRM1,NXYZ1)
C
C     Compute the inverse partials.
C
      DO 100 I=1,NXYZ1
         I11=IOFF+I
         RXM1(I11)=YSM1(I)*ZTM1(I)-YTM1(I)*ZSM1(I)
         RYM1(I11)=XTM1(I)*ZSM1(I)-XSM1(I)*ZTM1(I)
         RZM1(I11)=XSM1(I)*YTM1(I)-XTM1(I)*YSM1(I)
         SXM1(I11)=YTM1(I)*ZRM1(I)-YRM1(I)*ZTM1(I)
         SYM1(I11)=XRM1(I)*ZTM1(I)-XTM1(I)*ZRM1(I)
         SZM1(I11)=XTM1(I)*YRM1(I)-XRM1(I)*YTM1(I)
         TXM1(I11)=YRM1(I)*ZSM1(I)-YSM1(I)*ZRM1(I)
         TYM1(I11)=XSM1(I)*ZRM1(I)-XRM1(I)*ZSM1(I)
         TZM1(I11)=XRM1(I)*YSM1(I)-XSM1(I)*YRM1(I)
c        if (ie.eq.225) 
c    $      write(6,22) i,txm1(i11),zsm1(i),zrm1(i)
c  22       format(i5,1p3e14.5,' txzszr')
  100 CONTINUE
c
c     Compute unit normals and surface jacobians
c
      call setarea(ie)
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETAREA(ie)
C--------------------------------------------------------------------
C
C     Compute surface data: areas, normals and tangents
C
C--------------------------------------------------------------------
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      PARAMETER (NXM3=NXM*NXM*NXM)
      common /ctmp4/ xrm1,xsm1,xtm1,yrm1,ysm1,ytm1,zrm1,zsm1,ztm1
      REAL XRM1 (NXM3), XSM1(NXM3), XTM1(NXM3)
      REAL YRM1 (NXM3), YSM1(NXM3), YTM1(NXM3)
      REAL ZRM1 (NXM3), ZSM1(NXM3), ZTM1(NXM3)
c
c
      COMMON /CTMP1/ A  (NXM3)
     $ ,             B  (NXM3)
     $ ,             C  (NXM3)
     $ ,             DOT(NXM3)
c
C
      k = 1 + nx*nz*(2*ndim)*(ie-1)
c
      IF (NDIM.EQ.2) THEN
         CALL AREA2(area(k),unx(k),uny(k),xrm1,xsm1,yrm1,ysm1,ie)
      ELSE
         CALL AREA3(area(k),unx(k),uny(k),unz(k),a,b,c,dot)
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE AREA3(area,unx,uny,unz,a,b,c,dot)
C--------------------------------------------------------------------
C
C     Compute areas, normals and tangents (3D geom.)
C
C--------------------------------------------------------------------
      INCLUDE 'basics.inc'
C
      PARAMETER (NXM3=NXM*NXM*NXM)
      common /ctmp4/ xrm1,xsm1,xtm1,yrm1,ysm1,ytm1,zrm1,zsm1,ztm1
      REAL XRM1 (NXM3), XSM1(NXM3), XTM1(NXM3)
      REAL YRM1 (NXM3), YSM1(NXM3), YTM1(NXM3)
      REAL ZRM1 (NXM3), ZSM1(NXM3), ZTM1(NXM3)
c
      REAL  A  (nx,ny,nz)
      REAL  B  (nx,ny,nz)
      REAL  C  (nx,ny,nz)
      REAL  DOT(nx,ny,nz)
c
      real unx (nx,ny,6)
      real uny (nx,ny,6)
      real unz (nx,ny,6)
      real area(nx,ny,6)
C
      NXY1  = NX*NY
      NTOT  = NX*NY*NZ
      NSRF  = 6*NX*NY
C
C        "R"
C
      CALL VCROSS(A,B,C,XSM1,YSM1,ZSM1,XTM1,YTM1,ZTM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
C
      DO 100 IZ=1,NZ
      DO 100 IY=1,NY
         WEIGHT = wght(IY)*wght(IZ)
         AREA(IY,IZ,2) = SQRT(DOT(NX,IY,IZ))*WEIGHT
         AREA(IY,IZ,4) = SQRT(DOT( 1,IY,IZ))*WEIGHT
         UNX (IY,IZ,4) = -A( 1,IY,IZ)
         UNX (IY,IZ,2) =  A(NX,IY,IZ)
         UNY (IY,IZ,4) = -B( 1,IY,IZ)
         UNY (IY,IZ,2) =  B(NX,IY,IZ)
         UNZ (IY,IZ,4) = -C( 1,IY,IZ)
         UNZ (IY,IZ,2) =  C(NX,IY,IZ)
  100 CONTINUE
C
C        "S"
C
      CALL VCROSS(A,B,C,XRM1,YRM1,ZRM1,XTM1,YTM1,ZTM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
      DO 200 IZ=1,NZ
      DO 200 IX=1,NX
         WEIGHT=wght(IX)*wght(IZ)
         AREA(IX,IZ,1) = SQRT(DOT(IX, 1,IZ))*WEIGHT
         AREA(IX,IZ,3) = SQRT(DOT(IX,NY,IZ))*WEIGHT
         UNX (IX,IZ,1) =  A(IX, 1,IZ)
         UNX (IX,IZ,3) = -A(IX,NY,IZ)
         UNY (IX,IZ,1) =  B(IX, 1,IZ)
         UNY (IX,IZ,3) = -B(IX,NY,IZ)
         UNZ (IX,IZ,1) =  C(IX, 1,IZ)
         UNZ (IX,IZ,3) = -C(IX,NY,IZ)
  200 CONTINUE
C
C        "T"
C
      CALL VCROSS(A,B,C,XRM1,YRM1,ZRM1,XSM1,YSM1,ZSM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
      DO 300 IX=1,NX
      DO 300 IY=1,NY
         WEIGHT=wght(IX)*wght(IY)
         AREA(IX,IY,5) = SQRT(DOT(IX,IY, 1))*WEIGHT
         AREA(IX,IY,6) = SQRT(DOT(IX,IY,NZ))*WEIGHT
         UNX (IX,IY,5) = -A(IX,IY, 1)
         UNX (IX,IY,6) =  A(IX,IY,NZ)
         UNY (IX,IY,5) = -B(IX,IY, 1)
         UNY (IX,IY,6) =  B(IX,IY,NZ)
         UNZ (IX,IY,5) = -C(IX,IY, 1)
         UNZ (IX,IY,6) =  C(IX,IY,NZ)
  300 CONTINUE
C
      CALL UNITVEC (UNX,UNY,UNZ,NSRF)
c
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE AREA2(area,unx,uny,xrm1,xsm1,yrm1,ysm1,ie)
C--------------------------------------------------------------------
C
C     Compute areas, normals and tangents (2D and Axisymmetric geom.)
C
C--------------------------------------------------------------------
      INCLUDE 'basics.inc'
c
      real area(nx,4),unx(ny,4),uny(ny,4)
      real xrm1(nx,ny),xsm1(nx,ny),yrm1(ny,ny),ysm1(nx,ny)
c
C
C     This is for axisymmetric.  For now, just use wght.
C
c     CALL SETWGTR (WGTR1,WGTR2,WGTR3,WGTR4,ie)
C
C     "R"
C
      DO 100 IY=1,NY
         XS2  = XSM1(NX,IY)
         YS2  = YSM1(NX,IY)
         XS4  = XSM1( 1,IY)
         YS4  = YSM1( 1,IY)
         SS2  = SQRT( XS2**2 + YS2**2 )
         SS4  = SQRT( XS4**2 + YS4**2 )
         T2X  =  XS2 / SS2
         T2Y  =  YS2 / SS2
         T4X  = -XS4 / SS4
         T4Y  = -YS4 / SS4
         UNX (IY,2) =  T2Y
         UNY (IY,2) = -T2X
         UNX (IY,4) =  T4Y
         UNY (IY,4) = -T4X
c        AREA(IY,2) =  SS2 * WGTR2(IY)
c        AREA(IY,4) =  SS4 * WGTR4(IY)
         AREA(IY,2) =  SS2 * wght (IY)
         AREA(IY,4) =  SS4 * wght (IY)
  100 CONTINUE
C
C     "S"
C
      DO 200 IX=1,NX
         XR1  = XRM1(IX, 1)
         YR1  = YRM1(IX, 1)
         XR3  = XRM1(IX,NY)
         YR3  = YRM1(IX,NY)
         RR1  = SQRT( XR1**2 + YR1**2 )
         RR3  = SQRT( XR3**2 + YR3**2 )
         T1X  =  XR1 / RR1
         T1Y  =  YR1 / RR1
         T3X  = -XR3 / RR3
         T3Y  = -YR3 / RR3
         UNX (IX,1) =  T1Y
         UNY (IX,1) = -T1X
         UNX (IX,3) =  T3Y
         UNY (IX,3) = -T3X
c        AREA(IX,1) =  RR1 * WGTR1(IX)
c        AREA(IX,3) =  RR3 * WGTR3(IX)
         AREA(IX,1) =  RR1 * wght (IX)
         AREA(IX,3) =  RR3 * wght (IX)
  200 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETWGTR (WGTR1,WGTR2,WGTR3,WGTR4,ie)
C
      INCLUDE 'basics.inc'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      REAL WGTR1(nx),WGTR2(ny),WGTR3(nx),WGTR4(ny)
C
c     IF (IFAXIS) THEN
c        DO 120 IX=1,NX
c           WGTR1(IX) = YM1(IX, 1) * wght(IX)
c           WGTR3(IX) = YM1(IX,NY) * wght(IX)
c120    CONTINUE
c        IF ( IFRZER(ie) ) THEN
c           IY = 1
c           WGTR2(IY) = YSM1(NX,IY) * wght(IY)
c           WGTR4(IY) = YSM1( 1,IY) * wght(IY)
c           DO 160 IY=2,NY
c              DNR = 1. + ZAM1(IY)
c              WGTR2(IY) = YM1(NX,IY) * wght(IY) / DNR
c              WGTR4(IY) = YM1( 1,IY) * wght(IY) / DNR
c160       CONTINUE
c        ELSE
c           DO 180 IY=1,NY
c              WGTR2(IY) = YM1(NX,IY) * wght(IY)
c              WGTR4(IY) = YM1( 1,IY) * wght(IY)
c180       CONTINUE
c        ENDIF
c     ELSE
         CALL COPY (WGTR1,wght,NX)
         CALL COPY (WGTR2,wght,NY)
         CALL COPY (WGTR3,wght,NX)
         CALL COPY (WGTR4,wght,NY)
c     ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE UNITVEC (X,Y,Z,N)
      REAL X(1),Y(1),Z(1)
      DO 100 I=1,N
      XLNGTH = SQRT( X(I)**2 + Y(I)**2 + Z(I)**2 )
      IF (XLNGTH.NE.0.0) THEN
         X(I) = X(I)/XLNGTH
         Y(I) = Y(I)/XLNGTH
         Z(I) = Z(I)/XLNGTH
      ENDIF
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE VDOT3 (DOT,U1,U2,U3,V1,V2,V3,N)
C
C     Compute a Cartesian vector dot product. 3-d version
C
      REAL DOT(1)
      REAL U1(1),U2(1),U3(1)
      REAL V1(1),V2(1),V3(1)
C
C
      DO 100 I=1,N
         DOT(I) = U1(I)*V1(I) + U2(I)*V2(I) + U3(I)*V3(I)
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
