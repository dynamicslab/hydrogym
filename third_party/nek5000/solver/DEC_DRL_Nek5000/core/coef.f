      subroutine genwz
C-----------------------------------------------------------------
C
C     GENERATE
C
C            - DERIVATIVE OPERATORS
C            - INTERPOLATION OPERATORS
C            - WEIGHTS
C            - COLLOCATION POINTS
C
C     ASSOCIATED WITH THE
C
C            - GAUSS-LOBATTO LEGENDRE MESH (SUFFIX M1/M2/M3)
C            - GAUSS LEGENDRE         MESH (SUFFIX M2)
C            - GAUSS-LOBATTO JACOBI   MESH (SUFFIX M1/M2/M3)
C
C-----------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'WZ'
      INCLUDE 'DXYZ'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'

      REAL TMP(LY1,LY1),TMPT(LY1,LY1)
C
      IF (NDIM.EQ.2) THEN
C
C***  Two-dimensional case  **********************
C
C
C     Gauss-Lobatto Legendre mesh (suffix M1)
C     Generate collocation points and weights
C
      CALL ZWGLL (ZGM1(1,1),WXM1,NX1)
      CALL ZWGLL (ZGM1(1,2),WYM1,NY1)
      ZGM1(NZ1,3) = 0.
      WZM1(NZ1)   = 1.
      DO 100 IY=1,NY1
      DO 100 IX=1,NX1
      W3M1(IX,IY,1)=WXM1(IX)*WYM1(IY)
  100 CONTINUE
C
C     Compute derivative matrices
C
      CALL DGLL (DXM1,DXTM1,ZGM1(1,1),NX1,NX1)
      CALL DGLL (DYM1,DYTM1,ZGM1(1,2),NY1,NY1)
      CALL RZERO (DZM1 ,NZ1*NZ1)
      CALL RZERO (DZTM1,NZ1*NZ1)
C
C     Gauss Legendre mesh (suffix M2)
C     Generate collocation points and weights
C
      IF(IFSPLIT)THEN
         CALL ZWGLL (ZGM2(1,1),WXM2,NX2)
         CALL ZWGLL (ZGM2(1,2),WYM2,NY2)
      ELSE
         CALL ZWGL  (ZGM2(1,1),WXM2,NX2)
         CALL ZWGL  (ZGM2(1,2),WYM2,NY2)
      ENDIF
      ZGM2(NZ2,3) = 0.
      WZM2(NZ2)   = 1.
      DO 200 IY=1,NY2
      DO 200 IX=1,NX2
      W3M2(IX,IY,1)=WXM2(IX)*WYM2(IY)
  200 CONTINUE
C
C     Gauss-Lobatto Legendre mesh (suffix M3).
C     Generate collocation points and weights.
C
      CALL ZWGLL (ZGM3(1,1),WXM3,NX3)
      CALL ZWGLL (ZGM3(1,2),WYM3,NY3)
      ZGM3(NZ3,3) = 0.
      WZM3(NZ3)   = 1.
      DO 300 IY=1,NY3
      DO 300 IX=1,NX3
      W3M3(IX,IY,1)=WXM3(IX)*WYM3(IY)
  300 CONTINUE
C
C     Compute derivative matrices
C
      CALL DGLL (DXM3,DXTM3,ZGM3(1,1),NX3,NX3)
      CALL DGLL (DYM3,DYTM3,ZGM3(1,2),NY3,NY3)
      CALL RZERO (DZM3 ,NZ3*NZ3)
      CALL RZERO (DZTM3,NZ3*NZ3)
C
C     Generate interpolation operators for the staggered mesh
C
      CALL IGLLM (IXM12,IXTM12,ZGM1(1,1),ZGM2(1,1),NX1,NX2,NX1,NX2)
      CALL IGLLM (IYM12,IYTM12,ZGM1(1,2),ZGM2(1,2),NY1,NY2,NY1,NY2)
      IZM12 (NZ2,NZ1) = 1.
      IZTM12(NZ1,NZ2) = 1.
C
C     NOTE: The splitting scheme has only one mesh!!!!!
C
      IF (IFSPLIT) THEN
         CALL IGLLM (IXM21,IXTM21,ZGM1(1,1),ZGM2(1,1),NX1,NX2,NX1,NX2)
         CALL IGLLM (IYM21,IYTM21,ZGM1(1,2),ZGM2(1,2),NY1,NY2,NY1,NY2)
      ELSE
         CALL IGLM  (IXM21,IXTM21,ZGM2(1,1),ZGM1(1,1),NX2,NX1,NX2,NX1)
         CALL IGLM  (IYM21,IYTM21,ZGM2(1,2),ZGM1(1,2),NY2,NY1,NY2,NY1)
      ENDIF
      IZM21 (NZ1,NZ2) = 1.
      IZTM21(NZ2,NZ1) = 1.
C
C     Compute derivative operators for the staggered mesh
C
      IF(IFSPLIT)THEN
         CALL COPY (DXM12, DXM1, NX1*NX2)
         CALL COPY (DXTM12,DXTM1,NX1*NX2)
         CALL COPY (DYM12, DYM1, NY1*NY2)
         CALL COPY (DYTM12,DYTM1,NY1*NY2)
         CALL COPY (DZM12, DZM1, NZ1*NZ2)
         CALL COPY (DZTM12,DZTM1,NZ1*NZ2)
      ELSE
         CALL DGLLGL (DXM12,DXTM12,ZGM1(1,1),ZGM2(1,1),IXM12,
     $                                       NX1,NX2,NX1,NX2)
         CALL DGLLGL (DYM12,DYTM12,ZGM1(1,2),ZGM2(1,2),IYM12,
     $                                       NY1,NY2,NY1,NY2)
         DZM12 (NZ2,NZ1) = 0.
         DZTM12(NZ2,NZ1) = 0.
      ENDIF
C
C     Compute interpolation operators for the geometry mesh M3.
C
      CALL IGLLM (IXM13,IXTM13,ZGM1(1,1),ZGM3(1,1),NX1,NX3,NX1,NX3)
      CALL IGLLM (IYM13,IYTM13,ZGM1(1,2),ZGM3(1,2),NY1,NY3,NY1,NY3)
      CALL IGLLM (IXM31,IXTM31,ZGM3(1,1),ZGM1(1,1),NX3,NX1,NX3,NX1)
      CALL IGLLM (IYM31,IYTM31,ZGM3(1,2),ZGM1(1,2),NY3,NY1,NY3,NY1)
      IZM13 (NZ3,NZ1) = 1.
      IZTM13(NZ1,NZ3) = 1.
      IZM31 (NZ1,NZ3) = 1.
      IZTM31(NZ3,NZ1) = 1.
C
C
      IF (IFAXIS) THEN
C
C     Special treatment for the axisymmetric case
C     Generate additional points, weights, derivative operators and
C     interpolation operators required for elements close to the axis.
C
C
C     Gauss-Lobatto Jacobi mesh (suffix M1).
C     Generate collocation points and weights (alpha=0, beta=1).
C
      ALPHA = 0.
      BETA  = 1.
      CALL ZWGLJ (ZAM1,WAM1,NY1,ALPHA,BETA)
      DO 400 IY=1,NY1
      DO 400 IX=1,NX1
         W2AM1(IX,IY)=WXM1(IX)*WAM1(IY)
         W2CM1(IX,IY)=WXM1(IX)*WYM1(IY)
  400 CONTINUE
C
C     Compute derivative matrices
C
      CALL COPY (DCM1,DYM1,NY1*NY1)
      CALL COPY (DCTM1,DYTM1,NY1*NY1)
      CALL DGLJ (DAM1,DATM1,ZAM1,NY1,NY1,ALPHA,BETA)
C
C     Gauss Jacobi mesh (suffix M2)
C     Generate collocation points and weights
C
      IF(IFSPLIT)THEN
         CALL ZWGLJ (ZAM2,WAM2,NY2,ALPHA,BETA)
      ELSE
         CALL ZWGJ  (ZAM2,WAM2,NY2,ALPHA,BETA)
      ENDIF
      DO 500 IY=1,NY2
      DO 500 IX=1,NX2
         W2CM2(IX,IY)=WXM2(IX)*WYM2(IY)
         W2AM2(IX,IY)=WXM2(IX)*WAM2(IY)
  500 CONTINUE
C
C     Gauss-Lobatto Jacobi mesh (suffix M3).
C     Generate collocation points and weights.
C
      CALL ZWGLJ (ZAM3,WAM3,NY3,ALPHA,BETA)
      DO 600 IY=1,NY3
      DO 600 IX=1,NX3
         W2CM3(IX,IY)=WXM3(IX)*WYM3(IY)
         W2AM3(IX,IY)=WXM3(IX)*WAM3(IY)
  600 CONTINUE
C
C     Compute derivative matrices
C
      CALL COPY (DCM3,DYM3,NY3*NY3)
      CALL COPY (DCTM3,DYTM3,NY3*NY3)
      CALL DGLJ (DAM3,DATM3,ZAM3,NY3,NY3,ALPHA,BETA)
C
C     Generate interpolation operators for the staggered mesh
C
      CALL COPY  (ICM12,IYM12,NY2*NY1)
      CALL COPY  (ICTM12,IYTM12,NY1*NY2)
      CALL IGLJM (IAM12,IATM12,ZAM1,ZAM2,NY1,NY2,NY1,NY2,ALPHA,BETA)
      CALL COPY  (ICM21,IYM21,NY1*NY2)
      CALL COPY  (ICTM21,IYTM21,NY2*NY1)
      IF (IFSPLIT) THEN
      CALL IGLJM (IAM21,IATM21,ZAM2,ZAM1,NY1,NY2,NY1,NY2,ALPHA,BETA)
      ELSE
      CALL IGJM  (IAM21,IATM21,ZAM2,ZAM1,NY2,NY1,NY2,NY1,ALPHA,BETA)
      ENDIF
C
C     Compute derivative operators for the staggered mesh
C
      CALL COPY  (DCM12,DYM12,NY2*NY1)
      CALL COPY  (DCTM12,DYTM12,NY1*NY2)
      IF(IFSPLIT)THEN
         CALL COPY (DAM12, DAM1, NY1*NY2)
         CALL COPY (DATM12,DATM1,NY1*NY2)
      ELSE
         CALL DGLJGJ (DAM12,DATM12,ZAM1,ZAM2,IAM12,
     $                             NY1,NY2,NY1,NY2,ALPHA,BETA)
      ENDIF
C
C     Compute interpolation operators for the geometry mesh M3.
C
      CALL COPY  (ICM13,IYM13,NY3*NY1)
      CALL COPY  (ICTM13,IYTM13,NY1*NY3)
      CALL IGLJM (IAM13,IATM13,ZAM1,ZAM3,NY1,NY3,NY1,NY3,ALPHA,BETA)
      CALL COPY  (ICM31,IYM31,NY1*NY3)
      CALL COPY  (ICTM31,IYTM31,NY3*NY1)
      CALL IGLJM (IAM31,IATM31,ZAM3,ZAM1,NY3,NY1,NY3,NY1,ALPHA,BETA)
C
C     Compute interpolation operators between Gauss-Lobatto Jacobi
C     and Gauss-Lobatto Legendre (to be used in PREPOST).
C
      CALL IGLJM(IAJL1,IATJL1,ZAM1,ZGM1(1,2),NY1,NY1,NY1,NY1,ALPHA,BETA)
      IF (IFSPLIT) THEN
      CALL IGLJM(IAJL2,IATJL2,ZAM2,ZGM2(1,2),NY2,NY2,NY2,NY2,ALPHA,BETA)
      ELSE
      CALL IGJM (IAJL2,IATJL2,ZAM2,ZGM2(1,2),NY2,NY2,NY2,NY2,ALPHA,BETA)
      ENDIF

      CALL INVMT(IAJL1 ,IALJ1 ,TMP ,NY1)
      CALL INVMT(IATJL1,IATLJ1,TMPT,NY1)
      CALL MXM (IATJL1,NY1,IATLJ1,NY1,TMPT,NY1)
      CALL MXM (IAJL1 ,NY1,IALJ1 ,NY1,TMP ,NY1)

C
C     Compute interpolation operators between Gauss-Lobatto Legendre
C     and Gauss-Lobatto Jacobi (to be used in subr. genxyz IN postpre).
C
c
c     This call is not right, and these arrays are not used. 3/27/02. pff
c     CALL IGLLM(IALJ3,IATLJ3,ZGM3(1,2),ZAM3,NY3,NY3,NY3,NY3,ALPHA,BETA)
      CALL IGLJM(IALJ3,IATLJ3,ZGM3(1,2),ZAM3,NY3,NY3,NY3,NY3,ALPHA,BETA)
C
      ENDIF
C
C
      ELSE
C
C***  Three-dimensional case ************************************
C
C
C     Gauss-Lobatto Legendre mesh (suffix M1)
C     Generate collocation points and weights
C
      CALL ZWGLL (ZGM1(1,1),WXM1,NX1)
      CALL ZWGLL (ZGM1(1,2),WYM1,NY1)
      CALL ZWGLL (ZGM1(1,3),WZM1,NZ1)
      DO 700 IZ=1,NZ1
      DO 700 IY=1,NY1
      DO 700 IX=1,NX1
      W3M1(IX,IY,IZ)=WXM1(IX)*WYM1(IY)*WZM1(IZ)
  700 CONTINUE
C
C     Compute derivative matrices
C
      CALL DGLL (DXM1,DXTM1,ZGM1(1,1),NX1,NX1)
      CALL DGLL (DYM1,DYTM1,ZGM1(1,2),NY1,NY1)
      CALL DGLL (DZM1,DZTM1,ZGM1(1,3),NZ1,NZ1)
C
C     Gauss Legendre mesh (suffix M2)
C     Generate collocation points and weights
C
      IF(IFSPLIT)THEN
         CALL ZWGLL (ZGM2(1,1),WXM2,NX2)
         CALL ZWGLL (ZGM2(1,2),WYM2,NY2)
         CALL ZWGLL (ZGM2(1,3),WZM2,NZ2)
      ELSE
         CALL ZWGL  (ZGM2(1,1),WXM2,NX2)
         CALL ZWGL  (ZGM2(1,2),WYM2,NY2)
         CALL ZWGL  (ZGM2(1,3),WZM2,NZ2)
      ENDIF
      DO 800 IZ=1,NZ2
      DO 800 IY=1,NY2
      DO 800 IX=1,NX2
      W3M2(IX,IY,IZ)=WXM2(IX)*WYM2(IY)*WZM2(IZ)
  800 CONTINUE
C
C     Gauss-Loabtto Legendre mesh (suffix M3).
C     Generate collocation points and weights.
C
      CALL ZWGLL (ZGM3(1,1),WXM3,NX3)
      CALL ZWGLL (ZGM3(1,2),WYM3,NY3)
      CALL ZWGLL (ZGM3(1,3),WZM3,NZ3)
      DO 900 IZ=1,NZ3
      DO 900 IY=1,NY3
      DO 900 IX=1,NX3
      W3M3(IX,IY,IZ)=WXM3(IX)*WYM3(IY)*WZM3(IZ)
  900 CONTINUE
C
C     Compute derivative matrices
C
      CALL DGLL (DXM3,DXTM3,ZGM3(1,1),NX3,NX3)
      CALL DGLL (DYM3,DYTM3,ZGM3(1,2),NY3,NY3)
      CALL DGLL (DZM3,DZTM3,ZGM3(1,3),NZ3,NZ3)
C
C     Generate interpolation operators for the staggered mesh
C
      CALL IGLLM (IXM12,IXTM12,ZGM1(1,1),ZGM2(1,1),NX1,NX2,NX1,NX2)
      CALL IGLLM (IYM12,IYTM12,ZGM1(1,2),ZGM2(1,2),NY1,NY2,NY1,NY2)
      CALL IGLLM (IZM12,IZTM12,ZGM1(1,3),ZGM2(1,3),NZ1,NZ2,NZ1,NZ2)
C
C     NOTE: The splitting scheme has only one mesh!!!!!
C
      IF (IFSPLIT) THEN
         CALL IGLLM (IXM21,IXTM21,ZGM1(1,1),ZGM2(1,1),NX1,NX2,NX1,NX2)
         CALL IGLLM (IYM21,IYTM21,ZGM1(1,2),ZGM2(1,2),NY1,NY2,NY1,NY2)
         CALL IGLLM (IZM21,IZTM21,ZGM1(1,3),ZGM2(1,3),NZ1,NZ2,NZ1,NZ2)
      ELSE
         CALL IGLM  (IXM21,IXTM21,ZGM2(1,1),ZGM1(1,1),NX2,NX1,NX2,NX1)
         CALL IGLM  (IYM21,IYTM21,ZGM2(1,2),ZGM1(1,2),NY2,NY1,NY2,NY1)
         CALL IGLM  (IZM21,IZTM21,ZGM2(1,3),ZGM1(1,3),NZ2,NZ1,NZ2,NZ1)
      ENDIF
C
C     Compute derivative operators for the staggered mesh
C
      IF(IFSPLIT)THEN
         CALL COPY (DXM12, DXM1, NX1*NX2)
         CALL COPY (DXTM12,DXTM1,NX1*NX2)
         CALL COPY (DYM12, DYM1, NY1*NY2)
         CALL COPY (DYTM12,DYTM1,NY1*NY2)
         CALL COPY (DZM12, DZM1, NZ1*NZ2)
         CALL COPY (DZTM12,DZTM1,NZ1*NZ2)
      ELSE
         CALL DGLLGL (DXM12,DXTM12,ZGM1(1,1),ZGM2(1,1),IXM12,
     $                                       NX1,NX2,NX1,NX2)
         CALL DGLLGL (DYM12,DYTM12,ZGM1(1,2),ZGM2(1,2),IYM12,
     $                                       NY1,NY2,NY1,NY2)
         CALL DGLLGL (DZM12,DZTM12,ZGM1(1,3),ZGM2(1,3),IZM12,
     $                                       NZ1,NZ2,NZ1,NZ2)
      ENDIF
C
C     Compute interpolation operators for the geometry mesh M3.
C
      CALL IGLLM (IXM13,IXTM13,ZGM1(1,1),ZGM3(1,1),NX1,NX3,NX1,NX3)
      CALL IGLLM (IYM13,IYTM13,ZGM1(1,2),ZGM3(1,2),NY1,NY3,NY1,NY3)
      CALL IGLLM (IZM13,IZTM13,ZGM1(1,3),ZGM3(1,3),NZ1,NZ3,NZ1,NZ3)
      CALL IGLLM (IXM31,IXTM31,ZGM3(1,1),ZGM1(1,1),NX3,NX1,NX3,NX1)
      CALL IGLLM (IYM31,IYTM31,ZGM3(1,2),ZGM1(1,2),NY3,NY1,NY3,NY1)
      CALL IGLLM (IZM31,IZTM31,ZGM3(1,3),ZGM1(1,3),NZ3,NZ1,NZ3,NZ1)
C
      ENDIF
C
      RETURN
      END
      subroutine geom1 (xm3,ym3,zm3)
C-----------------------------------------------------------------------
C
C     Routine to generate all elemental geometric data for mesh 1.
C
C     Velocity formulation : global-to-local mapping based on mesh 3
C     Stress   formulation : global-to-local mapping based on mesh 1
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
C
C     Note : XM3,YM3,ZM3 should come from COMMON /SCRUZ/.
C
      DIMENSION XM3(LX3,LY3,LZ3,1)
     $        , YM3(LX3,LY3,LZ3,1)
     $        , ZM3(LX3,LY3,LZ3,1)
C
      IF (IFGMSH3 .AND. ISTEP.EQ.0) THEN
         CALL GLMAPM3 (XM3,YM3,ZM3)
      ELSE
         CALL GLMAPM1
      ENDIF
C
      CALL GEODAT1
C
      RETURN
      END
      subroutine glmapm3 (xm3,ym3,zm3)
C-------------------------------------------------------------------
C
C     Routine to generate mapping data based on mesh 3
C     (Gauss-Legendre Lobatto meshes).
C
C         XRM3,  YRM3,  ZRM3   -   dx/dr, dy/dr, dz/dr
C         XSM3,  YSM3,  ZSM3   -   dx/ds, dy/ds, dz/ds
C         XTM3,  YTM3,  ZTM3   -   dx/dt, dy/dt, dz/dt
C         RXM3,  RYM3,  RZM3   -   dr/dx, dr/dy, dr/dz
C         SXM3,  SYM3,  SZM3   -   ds/dx, ds/dy, ds/dz
C         TXM3,  TYM3,  TZM3   -   dt/dx, dt/dy, dt/dz
C         JACM3                -   Jacobian
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
C
C     Note : work arrays for mesh 3 in scratch commons will be 
C            changed after exit of routine.
C
      COMMON /SCRNS/ XRM3 (LX3,LY3,LZ3,LELT)
     $ ,             XSM3 (LX3,LY3,LZ3,LELT)
     $ ,             XTM3 (LX3,LY3,LZ3,LELT)
     $ ,             YRM3 (LX3,LY3,LZ3,LELT)
     $ ,             YSM3 (LX3,LY3,LZ3,LELT)
     $ ,             YTM3 (LX3,LY3,LZ3,LELT)
     $ ,             ZRM3 (LX3,LY3,LZ3,LELT)
      COMMON /CTMP0/ ZSM3 (LX3,LY3,LZ3,LELT)
     $ ,             ZTM3 (LX3,LY3,LZ3,LELT)
      COMMON /CTMP1/ RXM3 (LX3,LY3,LZ3,LELT)
     $ ,             RYM3 (LX3,LY3,LZ3,LELT)
     $ ,             RZM3 (LX3,LY3,LZ3,LELT)
     $ ,             SXM3 (LX3,LY3,LZ3,LELT)
      COMMON /SCRMG/ SYM3 (LX3,LY3,LZ3,LELT)
     $ ,             SZM3 (LX3,LY3,LZ3,LELT)
     $ ,             TXM3 (LX3,LY3,LZ3,LELT)
     $ ,             TYM3 (LX3,LY3,LZ3,LELT)
      COMMON /SCREV/ TZM3 (LX3,LY3,LZ3,LELT)
     $ ,             JACM3(LX3,LY3,LZ3,LELT)
      REAL           JACM3
      DIMENSION XM3(LX3,LY3,LZ3,1)
     $        , YM3(LX3,LY3,LZ3,1)
     $        , ZM3(LX3,LY3,LZ3,1)
C
C
      NXY3  = NX3*NY3
      NYZ3  = NY3*NZ3
      NXYZ3 = NX3*NY3*NZ3
      NTOT3 = NXYZ3*NELT
      NXYZ1 = NX1*NY1*NZ1
      NTOT1 = NXYZ1*NELT
C
C
C     Compute isoparametric partials.
C

      IF (NDIM.EQ.2) THEN
C
C     Two-dimensional case
C
      DO 200 IEL=1,NELT
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFAXIS) THEN
         NY33   = NY3*NY3
         IF (IFRZER(IEL)) THEN
            CALL COPY (DYTM3,DATM3,NY33)
         ELSE
            CALL COPY (DYTM3,DCTM3,NY33)
         ENDIF
      ENDIF
C
      CALL MXM(DXM3,NX3,XM3(1,1,1,IEL),NX3,XRM3(1,1,1,IEL),NY3)
      CALL MXM(DXM3,NX3,YM3(1,1,1,IEL),NX3,YRM3(1,1,1,IEL),NY3)
      CALL MXM(XM3(1,1,1,IEL),NX3,DYTM3,NY3,XSM3(1,1,1,IEL),NY3)
      CALL MXM(YM3(1,1,1,IEL),NX3,DYTM3,NY3,YSM3(1,1,1,IEL),NY3)
C
 200  CONTINUE
C
      CALL RZERO   (JACM3,NTOT3)
      CALL ADDCOL3 (JACM3,XRM3,YSM3,NTOT3)
      CALL SUBCOL3 (JACM3,XSM3,YRM3,NTOT3)
C
      CALL COPY    (RXM3,YSM3,NTOT3)
      CALL COPY    (RYM3,XSM3,NTOT3)
      CALL CHSIGN  (RYM3,NTOT3)
      CALL COPY    (SXM3,YRM3,NTOT3)
      CALL CHSIGN  (SXM3,NTOT3)
      CALL COPY    (SYM3,XRM3,NTOT3)
C
      ELSE
C
C     Three-dimensional case
C
      DO 300 IEL=1,NELT
C
      CALL MXM(DXM3,NX3,XM3(1,1,1,IEL),NX3,XRM3(1,1,1,IEL),NYZ3)
      CALL MXM(DXM3,NX3,YM3(1,1,1,IEL),NX3,YRM3(1,1,1,IEL),NYZ3)
      CALL MXM(DXM3,NX3,ZM3(1,1,1,IEL),NX3,ZRM3(1,1,1,IEL),NYZ3)
C
      DO 310 IZ=1,NZ3
      CALL MXM(XM3(1,1,IZ,IEL),NX3,DYTM3,NY3,XSM3(1,1,IZ,IEL),NY3)
      CALL MXM(YM3(1,1,IZ,IEL),NX3,DYTM3,NY3,YSM3(1,1,IZ,IEL),NY3)
      CALL MXM(ZM3(1,1,IZ,IEL),NX3,DYTM3,NY3,ZSM3(1,1,IZ,IEL),NY3)
 310  CONTINUE
C
      CALL MXM(XM3(1,1,1,IEL),NXY3,DZTM3,NZ3,XTM3(1,1,1,IEL),NZ3)
      CALL MXM(YM3(1,1,1,IEL),NXY3,DZTM3,NZ3,YTM3(1,1,1,IEL),NZ3)
      CALL MXM(ZM3(1,1,1,IEL),NXY3,DZTM3,NZ3,ZTM3(1,1,1,IEL),NZ3)
C
 300  CONTINUE
C
      CALL RZERO   (JACM3,NTOT3)
      CALL ADDCOL4 (JACM3,XRM3,YSM3,ZTM3,NTOT3)
      CALL ADDCOL4 (JACM3,XTM3,YRM3,ZSM3,NTOT3)
      CALL ADDCOL4 (JACM3,XSM3,YTM3,ZRM3,NTOT3)
      CALL SUBCOL4 (JACM3,XRM3,YTM3,ZSM3,NTOT3)
      CALL SUBCOL4 (JACM3,XSM3,YRM3,ZTM3,NTOT3)
      CALL SUBCOL4 (JACM3,XTM3,YSM3,ZRM3,NTOT3)
C
      CALL ASCOL5  (RXM3,YSM3,ZTM3,YTM3,ZSM3,NTOT3)
      CALL ASCOL5  (RYM3,XTM3,ZSM3,XSM3,ZTM3,NTOT3)
      CALL ASCOL5  (RZM3,XSM3,YTM3,XTM3,YSM3,NTOT3)
      CALL ASCOL5  (SXM3,YTM3,ZRM3,YRM3,ZTM3,NTOT3)
      CALL ASCOL5  (SYM3,XRM3,ZTM3,XTM3,ZRM3,NTOT3)
      CALL ASCOL5  (SZM3,XTM3,YRM3,XRM3,YTM3,NTOT3)
      CALL ASCOL5  (TXM3,YRM3,ZSM3,YSM3,ZRM3,NTOT3)
      CALL ASCOL5  (TYM3,XSM3,ZRM3,XRM3,ZSM3,NTOT3)
      CALL ASCOL5  (TZM3,XRM3,YSM3,XSM3,YRM3,NTOT3)
C
      ENDIF
C
C     Mapping from space P(n-2) to space P(n) (mesh M3 to mesh M1).
C
      IF (NDIM.EQ.2) THEN
         CALL RZERO (RZM1,NTOT1)
         CALL RZERO (SZM1,NTOT1)
         CALL RONE  (TZM1,NTOT1)
      ENDIF
C
      kerr = 0
      DO 400 ie=1,NELT

c        write(6,*) 'chkj1'
c        call outxm3j(xm3,ym3,jacm3)

         CALL CHKJAC(JACM3(1,1,1,ie),NXYZ3,ie,xm3,ym3,zm3,ndim,ierr)
         if (ierr.eq.1) kerr = kerr+1
         CALL MAP31 (RXM1(1,1,1,ie),RXM3(1,1,1,ie),ie)
         CALL MAP31 (RYM1(1,1,1,ie),RYM3(1,1,1,ie),ie)
         CALL MAP31 (SXM1(1,1,1,ie),SXM3(1,1,1,ie),ie)
         CALL MAP31 (SYM1(1,1,1,ie),SYM3(1,1,1,ie),ie)
         IF (NDIM.EQ.3) THEN
            CALL MAP31 (RZM1(1,1,1,ie),RZM3(1,1,1,ie),ie)
            CALL MAP31 (SZM1(1,1,1,ie),SZM3(1,1,1,ie),ie)
            CALL MAP31 (TXM1(1,1,1,ie),TXM3(1,1,1,ie),ie)
            CALL MAP31 (TYM1(1,1,1,ie),TYM3(1,1,1,ie),ie)
            CALL MAP31 (TZM1(1,1,1,ie),TZM3(1,1,1,ie),ie)
         ENDIF
         CALL MAP31 (JACM1(1,1,1,ie),JACM3(1,1,1,ie),ie)
         CALL MAP31 (XM1(1,1,1,ie),XM3(1,1,1,ie),ie)
         CALL MAP31 (YM1(1,1,1,ie),YM3(1,1,1,ie),ie)
         CALL MAP31 (ZM1(1,1,1,ie),ZM3(1,1,1,ie),ie)
 400  CONTINUE
      kerr = iglsum(kerr,1)
      if (kerr.gt.0) then
         ifxyo = .true.
         ifvo  = .false.
         ifpo  = .false.
         ifto  = .false.
         param(66) = 4
         call outpost(vx,vy,vz,pr,t,'xyz')
         if (nid.eq.0) write(6,*) 'Jac error 3, setting p66=4, ifxyo=t'
         call exitt
      endif

      call invers2(jacmi,jacm1,ntot1)

      RETURN
      END
      subroutine glmapm1
C-----------------------------------------------------------------------
C
C     Routine to generate mapping data based on mesh 1
C     (Gauss-Legendre Lobatto meshes).
C
C         XRM1,  YRM1,  ZRM1   -   dx/dr, dy/dr, dz/dr
C         XSM1,  YSM1,  ZSM1   -   dx/ds, dy/ds, dz/ds
C         XTM1,  YTM1,  ZTM1   -   dx/dt, dy/dt, dz/dt
C         RXM1,  RYM1,  RZM1   -   dr/dx, dr/dy, dr/dz
C         SXM1,  SYM1,  SZM1   -   ds/dx, ds/dy, ds/dz
C         TXM1,  TYM1,  TZM1   -   dt/dx, dt/dy, dt/dz
C         JACM1                -   Jacobian
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      COMMON /SCRNS/ XRM1(LX1,LY1,LZ1,LELT)
     $ ,             YRM1(LX1,LY1,LZ1,LELT)
     $ ,             XSM1(LX1,LY1,LZ1,LELT)
     $ ,             YSM1(LX1,LY1,LZ1,LELT)
     $ ,             XTM1(LX1,LY1,LZ1,LELT)
     $ ,             YTM1(LX1,LY1,LZ1,LELT)
     $ ,             ZRM1(LX1,LY1,LZ1,LELT)
      COMMON /CTMP1/ ZSM1(LX1,LY1,LZ1,LELT)
     $ ,             ZTM1(LX1,LY1,LZ1,LELT)
C
      NXY1  = NX1*NY1
      NYZ1  = NY1*NZ1
      NXYZ1 = NX1*NY1*NZ1
      NTOT1 = NXYZ1*NELT
C
      CALL XYZRST (XRM1,YRM1,ZRM1,XSM1,YSM1,ZSM1,XTM1,YTM1,ZTM1,
     $             IFAXIS)
C
      IF (NDIM.EQ.2) THEN
         CALL RZERO   (JACM1,NTOT1)
         CALL ADDCOL3 (JACM1,XRM1,YSM1,NTOT1)
         CALL SUBCOL3 (JACM1,XSM1,YRM1,NTOT1)
         CALL COPY    (RXM1,YSM1,NTOT1)
         CALL COPY    (RYM1,XSM1,NTOT1)
         CALL CHSIGN  (RYM1,NTOT1)
         CALL COPY    (SXM1,YRM1,NTOT1)
         CALL CHSIGN  (SXM1,NTOT1)
         CALL COPY    (SYM1,XRM1,NTOT1)
         CALL RZERO   (RZM1,NTOT1)
         CALL RZERO   (SZM1,NTOT1)
         CALL RONE    (TZM1,NTOT1)
      ELSE
         CALL RZERO   (JACM1,NTOT1)
         CALL ADDCOL4 (JACM1,XRM1,YSM1,ZTM1,NTOT1)
         CALL ADDCOL4 (JACM1,XTM1,YRM1,ZSM1,NTOT1)
         CALL ADDCOL4 (JACM1,XSM1,YTM1,ZRM1,NTOT1)
         CALL SUBCOL4 (JACM1,XRM1,YTM1,ZSM1,NTOT1)
         CALL SUBCOL4 (JACM1,XSM1,YRM1,ZTM1,NTOT1)
         CALL SUBCOL4 (JACM1,XTM1,YSM1,ZRM1,NTOT1)
         CALL ASCOL5  (RXM1,YSM1,ZTM1,YTM1,ZSM1,NTOT1)
         CALL ASCOL5  (RYM1,XTM1,ZSM1,XSM1,ZTM1,NTOT1)
         CALL ASCOL5  (RZM1,XSM1,YTM1,XTM1,YSM1,NTOT1)
         CALL ASCOL5  (SXM1,YTM1,ZRM1,YRM1,ZTM1,NTOT1)
         CALL ASCOL5  (SYM1,XRM1,ZTM1,XTM1,ZRM1,NTOT1)
         CALL ASCOL5  (SZM1,XTM1,YRM1,XRM1,YTM1,NTOT1)
         CALL ASCOL5  (TXM1,YRM1,ZSM1,YSM1,ZRM1,NTOT1)
         CALL ASCOL5  (TYM1,XSM1,ZRM1,XRM1,ZSM1,NTOT1)
         CALL ASCOL5  (TZM1,XRM1,YSM1,XSM1,YRM1,NTOT1)
      ENDIF
C
      kerr = 0
      DO 500 ie=1,NELT
         CALL CHKJAC(JACM1(1,1,1,ie),NXYZ1,ie,xm1,ym1,zm1,ndim,ierr)
         if (ierr.ne.0) kerr = kerr+1
  500 CONTINUE
      kerr = iglsum(kerr,1)
      if (kerr.gt.0) then
         ifxyo = .true.
         ifvo  = .false.
         ifpo  = .false.
         ifto  = .false.
         param(66) = 4
         call outpost(vx,vy,vz,pr,t,'xyz')
         if (nid.eq.0) write(6,*) 'Jac error 1, setting p66=4, ifxyo=t'
         call exitt
      endif

      call invers2(jacmi,jacm1,ntot1)

      RETURN
      END
      subroutine geodat1
C-----------------------------------------------------------------------
C
C     Routine to generate elemental geometric matrices on mesh 1
C     (Gauss-Legendre Lobatto mesh).
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      INCLUDE 'WZ'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      COMMON /SCRNS/ XRM1(LX1,LY1,LZ1,LELT)
     $ ,             YRM1(LX1,LY1,LZ1,LELT)
     $ ,             XSM1(LX1,LY1,LZ1,LELT)
     $ ,             YSM1(LX1,LY1,LZ1,LELT)
     $ ,             XTM1(LX1,LY1,LZ1,LELT)
     $ ,             YTM1(LX1,LY1,LZ1,LELT)
     $ ,             ZRM1(LX1,LY1,LZ1,LELT)
      COMMON /CTMP1/ ZSM1(LX1,LY1,LZ1,LELT)
     $ ,             ZTM1(LX1,LY1,LZ1,LELT)
     $ ,             WJ   (LX1,LY1,LZ1,LELT)
C
      NXYZ1 = NX1*NY1*NZ1
      NTOT1 = NXYZ1*NELT
C
      IF (IFGMSH3 .AND. ISTEP.EQ.0)
     $   CALL XYZRST (XRM1,YRM1,ZRM1,XSM1,YSM1,ZSM1,XTM1,YTM1,ZTM1,
     $                IFAXIS)
C
      IF (.NOT.IFAXIS) THEN
         CALL INVERS2 (WJ,JACM1,NTOT1)
      ELSE
         DO 500 IEL=1,NELT
           IF (IFRZER(IEL)) THEN
              DO 510 J=1,NY1
              DO 510 I=1,NX1
                IF (J.GT.1) THEN
                   WJ(I,J,1,IEL) = YM1(I,J,1,IEL)/
     $                            (JACM1(I,J,1,IEL)*(1.+ZAM1(J)))
                ELSE
                   WJ(I,J,1,IEL) = YSM1(I,J,1,IEL)/JACM1(I,J,1,IEL)
                ENDIF
 510          CONTINUE
           ELSE
              CALL INVCOL3 (WJ(1,1,1,IEL),YM1(1,1,1,IEL),
     $                      JACM1(1,1,1,IEL),NXYZ1)
           ENDIF
 500     CONTINUE
      ENDIF
C
C     Compute geometric factors for integrated del-squared operator.
C
      IF (NDIM.EQ.2) THEN
         CALL VDOT2 (G1M1,RXM1,RYM1,RXM1,RYM1,NTOT1)
         CALL VDOT2 (G2M1,SXM1,SYM1,SXM1,SYM1,NTOT1)
         CALL VDOT2 (G4M1,RXM1,RYM1,SXM1,SYM1,NTOT1)
         CALL COL2  (G1M1,WJ,NTOT1)
         CALL COL2  (G2M1,WJ,NTOT1)
         CALL COL2  (G4M1,WJ,NTOT1)
         CALL RZERO (G3M1,NTOT1)
         CALL RZERO (G5M1,NTOT1)
         CALL RZERO (G6M1,NTOT1)
      ELSE
         CALL VDOT3 (G1M1,RXM1,RYM1,RZM1,RXM1,RYM1,RZM1,NTOT1)
         CALL VDOT3 (G2M1,SXM1,SYM1,SZM1,SXM1,SYM1,SZM1,NTOT1)
         CALL VDOT3 (G3M1,TXM1,TYM1,TZM1,TXM1,TYM1,TZM1,NTOT1)
         CALL VDOT3 (G4M1,RXM1,RYM1,RZM1,SXM1,SYM1,SZM1,NTOT1)
         CALL VDOT3 (G5M1,RXM1,RYM1,RZM1,TXM1,TYM1,TZM1,NTOT1)
         CALL VDOT3 (G6M1,SXM1,SYM1,SZM1,TXM1,TYM1,TZM1,NTOT1)
         CALL COL2  (G1M1,WJ,NTOT1)
         CALL COL2  (G2M1,WJ,NTOT1)
         CALL COL2  (G3M1,WJ,NTOT1)
         CALL COL2  (G4M1,WJ,NTOT1)
         CALL COL2  (G5M1,WJ,NTOT1)
         CALL COL2  (G6M1,WJ,NTOT1)
      ENDIF
C
C     Multiply the geometric factors GiM1,i=1,5 with the
C     weights on mesh M1.
C
      DO 580 IEL=1,NELT
         IF (IFAXIS) CALL SETAXW1 ( IFRZER(IEL) )
            CALL COL2 (G1M1(1,1,1,IEL),W3M1,NXYZ1)
            CALL COL2 (G2M1(1,1,1,IEL),W3M1,NXYZ1)
            CALL COL2 (G4M1(1,1,1,IEL),W3M1,NXYZ1)
         IF (NDIM.EQ.3) THEN
            CALL COL2 (G3M1(1,1,1,IEL),W3M1,NXYZ1)
            CALL COL2 (G5M1(1,1,1,IEL),W3M1,NXYZ1)
            CALL COL2 (G6M1(1,1,1,IEL),W3M1,NXYZ1)
         ENDIF
  580 CONTINUE
C
C     Compute the mass matrix on mesh M1.
C
      DO 700 IEL=1,NELT
         IF (IFAXIS) CALL SETAXW1 ( IFRZER(IEL) )
            CALL COL3 (BM1  (1,1,1,IEL),JACM1(1,1,1,IEL),W3M1,NXYZ1)
         IF (IFAXIS) THEN 
             CALL COL3(BAXM1(1,1,1,IEL),JACM1(1,1,1,IEL),W3M1,NXYZ1)
          IF (IFRZER(IEL)) THEN
            DO 600 J=1,NY1
            IF (J.GT.1) THEN
               DO 610 I=1,NX1
                  BM1(I,J,1,IEL) = BM1(I,J,1,IEL)*YM1(I,J,1,IEL)
     $                                           /(1.+ZAM1(J))
                  BAXM1(I,J,1,IEL)=BAXM1(I,J,1,IEL)/(1.+ZAM1(J))
 610           CONTINUE
            ELSE
               DO 620 I=1,NX1
                  BM1(I,J,1,IEL) = BM1(I,J,1,IEL)*YSM1(I,J,1,IEL)
                  BAXM1(I,J,1,IEL)=BAXM1(I,J,1,IEL)
 620           CONTINUE
            ENDIF
 600        CONTINUE
          ELSE
            CALL COL2 (BM1(1,1,1,IEL),YM1(1,1,1,IEL),NXYZ1)
          ENDIF
         ENDIF
C
 700  CONTINUE

      IF(IFAXIS) THEN
        DO IEL=1,NELT
          IF(IFRZER(IEL)) THEN
            DO J=1,NY1
            DO I=1,NX1
              IF(J.EQ.1) THEN
                 YINVM1(I,J,1,IEL)=1.0D0/YSM1(I,J,1,IEL)
              ELSE
                 YINVM1(I,J,1,IEL)=1.0D0/YM1 (I,J,1,IEL)
              ENDIF
            ENDDO 
            ENDDO 
          ELSE
            CALL INVERS2(YINVM1(1,1,1,IEL),YM1(1,1,1,IEL),NXYZ1)
          ENDIF
        ENDDO
      ELSE
        CALL CFILL(YINVM1,1.0D0,NXYZ1*NELT)
      ENDIF
C
C     Compute normals, tangents, and areas on elemental surfaces
C
      CALL SETAREA
C
      RETURN
      END
      subroutine geom2
C-------------------------------------------------------------------
C
C     Routine to generate all elemental geometric data for mesh 2
C     (Gauss-Legendre mesh).
C
C         RXM2,  RYM2,  RZM2   -   dr/dx, dr/dy, dr/dz
C         SXM2,  SYM2,  SZM2   -   ds/dx, ds/dy, ds/dz
C         TXM2,  TYM2,  TZM2   -   dt/dx, dt/dy, dt/dz
C         JACM2                -   Jacobian
C         BM2                  -   Mass matrix
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
C
      NXYZ2 = NX2*NY2*NZ2
      NTOT2 = NXYZ2*NELV
C
      IF (IFSPLIT) THEN
C
C        Mesh 1 and 2 are identical
C
         CALL COPY (RXM2,RXM1,NTOT2)
         CALL COPY (RYM2,RYM1,NTOT2)
         CALL COPY (RZM2,RZM1,NTOT2)
         CALL COPY (SXM2,SXM1,NTOT2)
         CALL COPY (SYM2,SYM1,NTOT2)
         CALL COPY (SZM2,SZM1,NTOT2)
         CALL COPY (TXM2,TXM1,NTOT2)
         CALL COPY (TYM2,TYM1,NTOT2)
         CALL COPY (TZM2,TZM1,NTOT2)
         CALL COPY (JACM2,JACM1,NTOT2)
         CALL COPY (BM2,BM1,NTOT2)

         CALL COPY (XM2,XM1,NTOT2)
         CALL COPY (YM2,YM1,NTOT2)
         CALL COPY (ZM2,ZM1,NTOT2)

      ELSE
C
C     Consistent approximation spaces (UZAWA)
C
         IF (NDIM.EQ.2) THEN
            CALL RZERO (RZM2,NTOT2)
            CALL RZERO (SZM2,NTOT2)
            CALL RONE  (TZM2,NTOT2)
         ENDIF
C
         DO 1000 IEL=1,NELV
C
C        Mapping from mesh M1 to mesh M2
C
         CALL MAP12 (RXM2(1,1,1,IEL),RXM1(1,1,1,IEL),IEL)
         CALL MAP12 (RYM2(1,1,1,IEL),RYM1(1,1,1,IEL),IEL)
         CALL MAP12 (SXM2(1,1,1,IEL),SXM1(1,1,1,IEL),IEL)
         CALL MAP12 (SYM2(1,1,1,IEL),SYM1(1,1,1,IEL),IEL)
         IF (NDIM.EQ.3) THEN
            CALL MAP12 (RZM2(1,1,1,IEL),RZM1(1,1,1,IEL),IEL)
            CALL MAP12 (SZM2(1,1,1,IEL),SZM1(1,1,1,IEL),IEL)
            CALL MAP12 (TXM2(1,1,1,IEL),TXM1(1,1,1,IEL),IEL)
            CALL MAP12 (TYM2(1,1,1,IEL),TYM1(1,1,1,IEL),IEL)
            CALL MAP12 (TZM2(1,1,1,IEL),TZM1(1,1,1,IEL),IEL)
         ENDIF
         CALL MAP12 (JACM2(1,1,1,IEL),JACM1(1,1,1,IEL),IEL)
C
         CALL MAP12 (XM2(1,1,1,IEL),XM1(1,1,1,IEL),IEL)
         CALL MAP12 (YM2(1,1,1,IEL),YM1(1,1,1,IEL),IEL)
         CALL MAP12 (ZM2(1,1,1,IEL),ZM1(1,1,1,IEL),IEL)
C
C        Compute the mass matrix on mesh M2.
C
         IF (IFAXIS) CALL SETAXW2 ( IFRZER(IEL) )
         CALL COL3 (BM2(1,1,1,IEL),W3M2,JACM2(1,1,1,IEL),NXYZ2)
C
         IF (IFAXIS.AND.IFRZER(IEL)) THEN
            DO 300 J=1,NY2
            DO 300 I=1,NX2
               BM2(I,J,1,IEL) = BM2(I,J,1,IEL)*YM2(I,J,1,IEL)
     $                                        /(1.+ZAM2(J))
 300        CONTINUE
         ELSEIF (IFAXIS.AND.(.NOT.IFRZER(IEL))) THEN
            CALL COL2 (BM2(1,1,1,IEL),YM2(1,1,1,IEL),NXYZ2)
         ENDIF
 1000    CONTINUE
C
      ENDIF
C
C     Compute inverse of mesh 2 mass matrix, pff 3/5/92
      CALL INVERS2(BM2INV,BM2,NTOT2)
C
      RETURN
      END
      subroutine xyzrst (xrm1,yrm1,zrm1,xsm1,ysm1,zsm1,
     $                   XTM1,YTM1,ZTM1,IFAXIS)
C-----------------------------------------------------------------------
C
C     Compute global-to-local derivatives on mesh 1.
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'DXYZ'
C
      DIMENSION XRM1(LX1,LY1,LZ1,1),YRM1(LX1,LY1,LZ1,1)
     $        , ZRM1(LX1,LY1,LZ1,1),XSM1(LX1,LY1,LZ1,1)
     $        , YSM1(LX1,LY1,LZ1,1),ZSM1(LX1,LY1,LZ1,1)
     $        , XTM1(LX1,LY1,LZ1,1),YTM1(LX1,LY1,LZ1,1)
     $        , ZTM1(LX1,LY1,LZ1,1)
      LOGICAL IFAXIS
C
      NXY1=NX1*NY1
      NYZ1=NY1*NZ1
C
      DO 100 IEL=1,NELT
C
      IF (IFAXIS) CALL SETAXDY ( IFRZER(IEL) )
C
      CALL MXM (DXM1,NX1,XM1(1,1,1,IEL),NX1,XRM1(1,1,1,IEL),NYZ1)
      CALL MXM (DXM1,NX1,YM1(1,1,1,IEL),NX1,YRM1(1,1,1,IEL),NYZ1)
      CALL MXM (DXM1,NX1,ZM1(1,1,1,IEL),NX1,ZRM1(1,1,1,IEL),NYZ1)
C
      DO 10 IZ=1,NZ1
      CALL MXM (XM1(1,1,IZ,IEL),NX1,DYTM1,NY1,XSM1(1,1,IZ,IEL),NY1)
      CALL MXM (YM1(1,1,IZ,IEL),NX1,DYTM1,NY1,YSM1(1,1,IZ,IEL),NY1)
      CALL MXM (ZM1(1,1,IZ,IEL),NX1,DYTM1,NY1,ZSM1(1,1,IZ,IEL),NY1)
   10 CONTINUE
C
      IF (NDIM.EQ.3) THEN
         CALL MXM (XM1(1,1,1,IEL),NXY1,DZTM1,NZ1,XTM1(1,1,1,IEL),NZ1)
         CALL MXM (YM1(1,1,1,IEL),NXY1,DZTM1,NZ1,YTM1(1,1,1,IEL),NZ1)
         CALL MXM (ZM1(1,1,1,IEL),NXY1,DZTM1,NZ1,ZTM1(1,1,1,IEL),NZ1)
      ELSE
         CALL RZERO (XTM1(1,1,1,IEL),NXY1)
         CALL RZERO (YTM1(1,1,1,IEL),NXY1)
         CALL RONE  (ZTM1(1,1,1,IEL),NXY1)
      ENDIF
C
  100 CONTINUE
C
      RETURN
      END
      subroutine chkjac(jac,n,iel,X,Y,Z,ND,IERR)
c
      include 'SIZE'
      include 'PARALLEL'
C
C     Check the array JAC for a change in sign.
C
      REAL JAC(N),x(1),y(1),z(1)
c
      ierr = 1
      SIGN = JAC(1)
      DO 100 I=2,N
         IF (SIGN*JAC(I).LE.0.0) THEN
            ieg = lglel(iel)
            WRITE(6,101) nid,I,ieg
            write(6,*) jac(i-1),jac(i)
            if (ndim.eq.3) then
               write(6,7) nid,x(i-1),y(i-1),z(i-1)
               write(6,7) nid,x(i),y(i),z(i)
            else
               write(6,7) nid,x(i-1),y(i-1)
               write(6,7) nid,x(i),y(i)
            endif
    7       format(i5,' xyz:',1p3e14.5)
c           if (np.eq.1) call out_xyz_el(x,y,z,iel)
c           ierr=0
            return
         ENDIF
  100 CONTINUE
  101 FORMAT(//,i5,2x
     $ ,'ERROR:  Vanishing Jacobian near',i7,'th node of element'
     $ ,I10,'.')
c
c
      ierr = 0
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine volume
C
C     Compute the volume based on mesh M1 and mesh M2
C

      include 'SIZE'
      include 'ESOLV'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      integer e
C
      volvm1=glsum(bm1,nx1*ny1*nz1*nelv)
      volvm2=glsum(bm2,nx2*ny2*nz2*nelv)
      voltm1=glsum(bm1,nx1*ny1*nz1*nelt)
      voltm2=glsum(bm2,nx2*ny2*nz2*nelt)
      mfield=1
      if (ifmvbd) mfield=0
      nfldt = nfield
      if (ifmhd) nfldt = nfield+1

      do ifld=mfield,nfldt
         if (iftmsh(ifld)) then
             volfld(ifld) = voltm1
         else
             volfld(ifld) = volvm1
         endif
      enddo

c      if (nio.eq.0) write(6,*) 'vol_t,vol_v:',voltm1,volvm1


      nxyz = nx1*ny1*nz1
      do e=1,nelt
         volel(e) = vlsum(bm1(1,1,1,e),nxyz)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine setarea
C
C     Compute surface data: areas, normals and tangents
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
C
      NSRF  = 6*NX1*NZ1*NELT
C
      CALL RZERO  (AREA,NSRF)
      CALL RZERO3 (UNX,UNY,UNZ,NSRF)
      CALL RZERO3 (T1X,T1Y,T1Z,NSRF)      
      CALL RZERO3 (T2X,T2Y,T2Z,NSRF)      
C
      IF (NDIM.EQ.2) THEN
         CALL AREA2
      ELSE
         CALL AREA3
      ENDIF
C
      RETURN
      END
      subroutine area2
C--------------------------------------------------------------------
C
C     Compute areas, normals and tangents (2D and Axisymmetric geom.)
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      COMMON /SCRNS/ XRM1(LX1,LY1,LZ1,LELT)
     $ ,             YRM1(LX1,LY1,LZ1,LELT)
     $ ,             XSM1(LX1,LY1,LZ1,LELT)
     $ ,             YSM1(LX1,LY1,LZ1,LELT)
      COMMON /CTMP0/ WGTR1(LX1,LELT)
     $ ,             WGTR2(LY1,LELT)
     $ ,             WGTR3(LX1,LELT)
     $ ,             WGTR4(LY1,LELT)
C
      CALL SETWGTR (WGTR1,WGTR2,WGTR3,WGTR4)
C
C     "R"
C
      DO 100 IEL=1,NELT
      DO 100 IY=1,NY1
         XS2  = XSM1(NX1,IY,1,IEL)
         YS2  = YSM1(NX1,IY,1,IEL)
         XS4  = XSM1(  1,IY,1,IEL)
         YS4  = YSM1(  1,IY,1,IEL)
         SS2  = SQRT( XS2**2 + YS2**2 )
         SS4  = SQRT( XS4**2 + YS4**2 )
         T1X (IY,1,2,IEL) =  XS2 / SS2
         T1Y (IY,1,2,IEL) =  YS2 / SS2
         T1X (IY,1,4,IEL) = -XS4 / SS4
         T1Y (IY,1,4,IEL) = -YS4 / SS4
         UNX (IY,1,2,IEL) =  T1Y(IY,1,2,IEL)
         UNY (IY,1,2,IEL) = -T1X(IY,1,2,IEL)
         UNX (IY,1,4,IEL) =  T1Y(IY,1,4,IEL)
         UNY (IY,1,4,IEL) = -T1X(IY,1,4,IEL)
         AREA(IY,1,2,IEL) =  SS2 * WGTR2(IY,IEL)
         AREA(IY,1,4,IEL) =  SS4 * WGTR4(IY,IEL)
  100 CONTINUE
C
C     "S"
C
      DO 200 IEL=1,NELT
      DO 200 IX=1,NX1
         XR1  = XRM1(IX,  1,1,IEL)
         YR1  = YRM1(IX,  1,1,IEL)
         XR3  = XRM1(IX,NY1,1,IEL)
         YR3  = YRM1(IX,NY1,1,IEL)
         RR1  = SQRT( XR1**2 + YR1**2 )
         RR3  = SQRT( XR3**2 + YR3**2 )
         T1X (IX,1,1,IEL) =  XR1 / RR1
         T1Y (IX,1,1,IEL) =  YR1 / RR1
         T1X (IX,1,3,IEL) = -XR3 / RR3
         T1Y (IX,1,3,IEL) = -YR3 / RR3
         UNX (IX,1,1,IEL) =  T1Y(IX,1,1,IEL)
         UNY (IX,1,1,IEL) = -T1X(IX,1,1,IEL)
         UNX (IX,1,3,IEL) =  T1Y(IX,1,3,IEL)
         UNY (IX,1,3,IEL) = -T1X(IX,1,3,IEL)
         AREA(IX,1,1,IEL) =  RR1 * WGTR1(IX,IEL)
         AREA(IX,1,3,IEL) =  RR3 * WGTR3(IX,IEL)
  200 CONTINUE
C
      RETURN
      END
      subroutine setwgtr (wgtr1,wgtr2,wgtr3,wgtr4)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'WZ'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      COMMON /SCRNS/ XRM1(LX1,LY1,LZ1,LELT)
     $ ,             YRM1(LX1,LY1,LZ1,LELT)
     $ ,             XSM1(LX1,LY1,LZ1,LELT)
     $ ,             YSM1(LX1,LY1,LZ1,LELT)
C
      DIMENSION WGTR1(LX1,1)
     $ ,        WGTR2(LY1,1)
     $ ,        WGTR3(LX1,1)
     $ ,        WGTR4(LY1,1)
C
      IF (IFAXIS) THEN
         DO 100 IEL=1,NELT
            DO 120 IX=1,NX1
               WGTR1(IX,IEL) = YM1(IX,  1,1,IEL) * WXM1(IX)
               WGTR3(IX,IEL) = YM1(IX,NY1,1,IEL) * WXM1(IX)
  120       CONTINUE
            IF ( IFRZER(IEL) ) THEN
               IY = 1
               WGTR2(IY,IEL) = YSM1(NX1,IY,1,IEL) * WAM1(IY)
               WGTR4(IY,IEL) = YSM1(  1,IY,1,IEL) * WAM1(IY)
               DO 160 IY=2,NY1
                  DNR = 1. + ZAM1(IY)
                  WGTR2(IY,IEL) = YM1(NX1,IY,1,IEL) * WAM1(IY) / DNR
                  WGTR4(IY,IEL) = YM1(  1,IY,1,IEL) * WAM1(IY) / DNR
  160          CONTINUE
            ELSE
               DO 180 IY=1,NY1
                  WGTR2(IY,IEL) = YM1(NX1,IY,1,IEL) * WYM1(IY)
                  WGTR4(IY,IEL) = YM1(  1,IY,1,IEL) * WYM1(IY)
  180          CONTINUE
            ENDIF
  100    CONTINUE
      ELSE
         DO 200 IEL=1,NELT
            CALL COPY (WGTR1(1,IEL),WXM1,NX1)
            CALL COPY (WGTR2(1,IEL),WYM1,NY1)
            CALL COPY (WGTR3(1,IEL),WXM1,NX1)
            CALL COPY (WGTR4(1,IEL),WYM1,NY1)
  200    CONTINUE
      ENDIF
C
      RETURN
      END
      subroutine area3
C--------------------------------------------------------------------
C
C     Compute areas, normals and tangents (3D geom.)
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'WZ'
      INCLUDE 'GEOM'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
C
      COMMON /SCRNS/ XRM1(LX1,LY1,LZ1,LELT)
     $ ,             YRM1(LX1,LY1,LZ1,LELT)
     $ ,             XSM1(LX1,LY1,LZ1,LELT)
     $ ,             YSM1(LX1,LY1,LZ1,LELT)
     $ ,             XTM1(LX1,LY1,LZ1,LELT)
     $ ,             YTM1(LX1,LY1,LZ1,LELT)
     $ ,             ZRM1(LX1,LY1,LZ1,LELT)
      COMMON /CTMP1/ ZSM1(LX1,LY1,LZ1,LELT)
     $ ,             ZTM1(LX1,LY1,LZ1,LELT)
     $ ,             A  (LX1,LY1,LZ1,LELT)
     $ ,             B  (LX1,LY1,LZ1,LELT)
      COMMON /CTMP0/ C  (LX1,LY1,LZ1,LELT)
     $ ,             DOT(LX1,LY1,LZ1,LELT)
C
      NXY1  = NX1*NY1
      NFACE = 2*NDIM
      NTOT  = NX1*NY1*NZ1*NELT
      NSRF  = 6*NX1*NY1*NELT
C
C        "R"
C
      CALL VCROSS(A,B,C,XSM1,YSM1,ZSM1,XTM1,YTM1,ZTM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
C
      DO 100 IEL=1,NELT
      DO 100 IZ=1,NZ1
      DO 100 IY=1,NY1
         WEIGHT = WYM1(IY)*WZM1(IZ)
         AREA(IY,IZ,2,IEL) = SQRT(DOT(NX1,IY,IZ,IEL))*WEIGHT
         AREA(IY,IZ,4,IEL) = SQRT(DOT(  1,IY,IZ,IEL))*WEIGHT
         UNX (IY,IZ,4,IEL) = -A(  1,IY,IZ,IEL)
         UNX (IY,IZ,2,IEL) =  A(NX1,IY,IZ,IEL)
         UNY (IY,IZ,4,IEL) = -B(  1,IY,IZ,IEL)
         UNY (IY,IZ,2,IEL) =  B(NX1,IY,IZ,IEL)
         UNZ (IY,IZ,4,IEL) = -C(  1,IY,IZ,IEL)
         UNZ (IY,IZ,2,IEL) =  C(NX1,IY,IZ,IEL)
  100 CONTINUE
C
C        "S"
C
      CALL VCROSS(A,B,C,XRM1,YRM1,ZRM1,XTM1,YTM1,ZTM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
      DO 200 IEL=1,NELT
      DO 200 IZ=1,NZ1
      DO 200 IX=1,NX1
         WEIGHT=WXM1(IX)*WZM1(IZ)
         AREA(IX,IZ,1,IEL) = SQRT(DOT(IX,  1,IZ,IEL))*WEIGHT
         AREA(IX,IZ,3,IEL) = SQRT(DOT(IX,NY1,IZ,IEL))*WEIGHT
         UNX (IX,IZ,1,IEL) =  A(IX,  1,IZ,IEL)
         UNX (IX,IZ,3,IEL) = -A(IX,NY1,IZ,IEL)
         UNY (IX,IZ,1,IEL) =  B(IX,  1,IZ,IEL)
         UNY (IX,IZ,3,IEL) = -B(IX,NY1,IZ,IEL)
         UNZ (IX,IZ,1,IEL) =  C(IX,  1,IZ,IEL)
         UNZ (IX,IZ,3,IEL) = -C(IX,NY1,IZ,IEL)
  200 CONTINUE
C
C        "T"
C
      CALL VCROSS(A,B,C,XRM1,YRM1,ZRM1,XSM1,YSM1,ZSM1,NTOT)
      CALL VDOT3 (DOT,A,B,C,A,B,C,NTOT)
      DO 300 IEL=1,NELT
      DO 300 IX=1,NX1
      DO 300 IY=1,NY1
         WEIGHT=WXM1(IX)*WYM1(IY)
         AREA(IX,IY,5,IEL) = SQRT(DOT(IX,IY,  1,IEL))*WEIGHT
         AREA(IX,IY,6,IEL) = SQRT(DOT(IX,IY,NZ1,IEL))*WEIGHT
         UNX (IX,IY,5,IEL) = -A(IX,IY,  1,IEL)
         UNX (IX,IY,6,IEL) =  A(IX,IY,NZ1,IEL)
         UNY (IX,IY,5,IEL) = -B(IX,IY,  1,IEL)
         UNY (IX,IY,6,IEL) =  B(IX,IY,NZ1,IEL)
         UNZ (IX,IY,5,IEL) = -C(IX,IY,  1,IEL)
         UNZ (IX,IY,6,IEL) =  C(IX,IY,NZ1,IEL)
  300 CONTINUE
C
      CALL UNITVEC (UNX,UNY,UNZ,NSRF)
C
C     COMPUTE UNIT TANGENT T1
C
      DO 600 IEL=1,NELT
      DO 600 IFC=1,NFACE
      IF (IFC.EQ.1 .OR. IFC.EQ.6) THEN
         CALL FACEXV (T1X(1,1,IFC,IEL),T1Y(1,1,IFC,IEL),
     $                T1Z(1,1,IFC,IEL),
     $                XRM1(1,1,1,IEL),YRM1(1,1,1,IEL),
     $                ZRM1(1,1,1,IEL),IFC,0)
      ELSEIF (IFC.EQ.2 .OR. IFC.EQ.5) THEN
         CALL FACEXV (T1X(1,1,IFC,IEL),T1Y(1,1,IFC,IEL),
     $                T1Z(1,1,IFC,IEL),
     $                XSM1(1,1,1,IEL),YSM1(1,1,1,IEL),
     $                ZSM1(1,1,1,IEL),IFC,0)
      ELSE
         CALL FACEXV (T1X(1,1,IFC,IEL),T1Y(1,1,IFC,IEL),
     $                T1Z(1,1,IFC,IEL),
     $                XTM1(1,1,1,IEL),YTM1(1,1,1,IEL),
     $                ZTM1(1,1,1,IEL),IFC,0)
      ENDIF
  600 CONTINUE
C
      CALL UNITVEC (T1X,T1Y,T1Z,NSRF)
C
C     COMPUTE UNIT TANGENT T2  ( T2 = Normal X T1 )
C
      DO 700 IEL=1,NELT
      DO 700 IFC=1,NFACE
         CALL VCROSS (T2X(1,1,IFC,IEL),T2Y(1,1,IFC,IEL),
     $                T2Z(1,1,IFC,IEL),
     $                UNX(1,1,IFC,IEL),UNY(1,1,IFC,IEL),
     $                UNZ(1,1,IFC,IEL),
     $                T1X(1,1,IFC,IEL),T1Y(1,1,IFC,IEL),
     $                T1Z(1,1,IFC,IEL),NXY1)
  700 CONTINUE
C
      RETURN
      END
      subroutine lagmass
C--------------------------------------------------------------------
C
C     Lag the mass matrix (matrices)
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
C
      NTOT1 = NX1*NY1*NZ1*NELT
      DO 100 ILAG=NBDINP-1,2,-1
         CALL COPY (BM1LAG(1,1,1,1,ILAG),BM1LAG(1,1,1,1,ILAG-1),NTOT1)
 100  CONTINUE
      CALL COPY (BM1LAG(1,1,1,1,1),BM1,NTOT1)
C
      RETURN
      END
C
      subroutine setinvm
C--------------------------------------------------------------------
C
C     Invert the mass matrix.
C
C     1)  Copy BM1 to BINVM1
C     2)  Perform direct stiffness summation on BINVM1
C     3)  Compute BINVM1 = 1/BINVM1
C     4)  Two inverse mass matrices required because of difference
C         in DSSUM routine for IMESH=1 and IMESH=2.
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'MASS'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'WZ'

      nxyz1  = nx1*ny1*nz1

      ifld = ifield

      IF (IFFLOW) THEN ! Velocity mass matrix
         IFIELD = 1
         NTOT   = NXYZ1*NELV
         CALL COPY    (BINVM1,BM1,NTOT)
         CALL DSSUM   (BINVM1,NX1,NY1,NZ1)
         CALL INVCOL1 (BINVM1,NTOT)
      ENDIF


      IF (IFHEAT) THEN ! Temperature mass matrix
         IFIELD = 2
         NTOT   = NXYZ1*NELT
         CALL COPY    (BINTM1,BM1,NTOT)
         CALL DSSUM   (BINTM1,NX1,NY1,NZ1)
         CALL INVCOL1 (BINTM1,NTOT)
      ENDIF

      ifield = ifld

      return
      end

c-----------------------------------------------------------------------

      subroutine maprs(y,x,xa,nrest,iel)
C
C     Map the elemental array X from Restart mesh to Y on mesh M1
C     Conforming elements, i.e. NX1=NY1=NZ1.
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'WZ'
      INCLUDE 'INPUT'
C
      REAL X(NREST,NREST,NREST)
      REAL Y(LX1,LY1,LZ1)
C
      REAL XA(NX1,NREST,NREST)
      COMMON /CTMP0/ XB(LX1,LY1,LZ1)
C
      REAL IXRES(LX1,LX1),IXTRES(LX1,LX1)
      REAL IYRES(LY1,LY1),IYTRES(LY1,LY1)
      REAL IZRES(LZ1,LZ1),IZTRES(LZ1,LZ1)
      REAL ZCRES(20),WCRES(20)
      REAL ZARES(20),WARES(20)
C
      NZREST = NREST
      IF(NZ1.EQ.1) NZREST=1
      NYZRES = NREST*NZREST
      NXY1   = NX1 *NY1
C
      CALL ZWGLL   (ZCRES,WCRES,NREST)
      CALL IGLLM   (IXRES,IXTRES,ZCRES,ZGM1,NREST,NX1,NREST,NX1)
      IF (.NOT.IFAXIS) THEN
         CALL COPY (IYRES,IXRES,NX1*NREST)
         CALL COPY (IYTRES,IXTRES,NX1*NREST)
         CALL COPY (IZRES,IXRES,NX1*NREST)
         CALL COPY (IZTRES,IXTRES,NX1*NREST)
      ELSE
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
         IF (IFRZER(IEL)) THEN
           ALPHA = 0.
           BETA  = 1.
           CALL ZWGLJ   (ZARES,WARES,NREST,ALPHA,BETA)
           CALL IGLJM   (IYRES,IYTRES,ZARES,ZGM1,NREST,NY1,NREST,NY1,
     $                                                    ALPHA,BETA)
           NY1R   = NY1*NREST
         ELSE
           CALL COPY (IYRES,IXRES,NX1*NREST)
           CALL COPY (IYTRES,IXTRES,NX1*NREST)
         ENDIF
      ENDIF
C
      IF (NDIM.EQ.2) THEN
         CALL MXM (IXRES,NX1,X,NREST,XA,NREST)
         CALL MXM (XA,NX1,IYTRES,NREST,Y,NY1)
      ELSE
         CALL MXM (IXRES,NX1,X,NREST,XA,NYZRES)
         DO 100 IZ=1,NZREST
            CALL MXM (XA(1,1,IZ),NX1,IYTRES,NREST,XB(1,1,IZ),NY1)
 100     CONTINUE
         CALL MXM (XB,NXY1,IZTRES,NZREST,Y,NZ1)
      ENDIF
C
      RETURN
      END
C
      subroutine map31 (y,x,iel)
C---------------------------------------------------------------
C
C     Map the elemental array X from mesh M3 to mesh M1
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
C
      REAL X(LX3,LY3,LZ3)
      REAL Y(LX1,LY1,LZ1)
C
      COMMON /CTMP0/ XA(LX1,LY3,LZ3), XB(LX1,LY1,LZ3)
C
      NYZ3 = NY3*NZ3
      NXY1 = NX1*NY1
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFAXIS) THEN
         NY31   = NY1*NY3
         IF (IFRZER(IEL))      CALL COPY (IYTM31,IATM31,NY31)
         IF (.NOT.IFRZER(IEL)) CALL COPY (IYTM31,ICTM31,NY31)
      ENDIF
C
      IF (IF3D) THEN
         CALL MXM (IXM31,NX1,X,NX3,XA,NYZ3)
         DO 100 IZ=1,NZ3
            CALL MXM (XA(1,1,IZ),NX1,IYTM31,NY3,XB(1,1,IZ),NY1)
 100     CONTINUE
         CALL MXM (XB,NXY1,IZTM31,NZ3,Y,NZ1)
      ELSE
         CALL MXM (IXM31,NX1,X,NX3,XA,NYZ3)
         CALL MXM (XA,NX1,IYTM31,NY3,Y,NY1)
      ENDIF
C
      RETURN
      END
C
      subroutine map13 (y,x,iel)
C---------------------------------------------------------------
C
C     Map the elemental array X from mesh M1 to mesh M3
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
C
      REAL X(LX1,LY1,LZ1)
      REAL Y(LX3,LY3,LZ3)
C
      COMMON /CTMP0/ XA(LX3,LY1,LZ1),  XB(LX3,LY3,LZ1)
C
      NYZ1 = NY1*NZ1
      NXY3 = NX3*NY3
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFAXIS) THEN
         NY13   = NY1*NY3
         IF (IFRZER(IEL))      CALL COPY (IYTM13,IATM13,NY13)
         IF (.NOT.IFRZER(IEL)) CALL COPY (IYTM13,ICTM13,NY13)
      ENDIF
C
      CALL MXM (IXM13,NX3,X,NX1,XA,NYZ1)
      DO 100 IZ=1,NZ1
         CALL MXM (XA(1,1,IZ),NX3,IYTM13,NY1,XB(1,1,IZ),NY3)
 100  CONTINUE
      CALL MXM (XB,NXY3,IZTM13,NZ1,Y,NZ3)
C
      RETURN
      END
      subroutine map12 (y,x,iel)
C---------------------------------------------------------------
C
C     Map the elemental array X from mesh M1 to mesh M2
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
C
      REAL X(LX1,LY1,LZ1)
      REAL Y(LX2,LY2,LZ2)
C
      COMMON /CTMP0/ XA(LX2,LY1,LZ1), XB(LX2,LY2,LZ1)
C
      NYZ1 = NY1*NZ1
      NXY2 = NX2*NY2
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFAXIS) THEN
         NY12   = NY1*NY2
         IF (IFRZER(IEL))      CALL COPY (IYTM12,IATM12,NY12)
         IF (.NOT.IFRZER(IEL)) CALL COPY (IYTM12,ICTM12,NY12)
      ENDIF
C
      CALL MXM (IXM12,NX2,X,NX1,XA,NYZ1)
      DO 100 IZ=1,NZ1
         CALL MXM (XA(1,1,IZ),NX2,IYTM12,NY1,XB(1,1,IZ),NY2)
 100  CONTINUE
      CALL MXM (XB,NXY2,IZTM12,NZ1,Y,NZ2)
C
      RETURN
      END
C
      subroutine map21t (y,x,iel)
C---------------------------------------------------------------
C
C     Map the elemental array X from mesh M2 to mesh M1 (Y)
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
C
      REAL X(LX2,LY2,LZ2)
      REAL Y(LX1,LY1,LZ1)
C
      COMMON /CTMP0/ XA(LX1,LY2,LZ2), XB(LX1,LY1,LZ2)
C
      NYZ2 = NY2*NZ2
      NXY1 = NX1*NY1
      NXYZ = NX1*NY1*NZ1
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFSPLIT) THEN
         CALL COPY(Y,X,NXYZ)
         RETURN
      ENDIF
C
      IF (IF3D) THEN
         CALL MXM (IXM21,NX1,X,NX2,XA,NYZ2)
         DO 100 IZ=1,NZ2
            CALL MXM (XA(1,1,IZ),NX1,IYTM21,NY2,XB(1,1,IZ),NY1)
 100     CONTINUE
         CALL MXM (XB,NXY1,IZTM21,NZ2,Y,NZ1)
      ELSE
         CALL MXM (IXM21,NX1,X,NX2,XA,NYZ2)
         CALL MXM (XA,NX1,IYTM21,NY2,Y,NY1)
      ENDIF
      RETURN
      END
      subroutine map21e (y,x,iel)
C---------------------------------------------------------------
C
C     Map the elemental array X from mesh M2 to mesh M1
C
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
C
      REAL X(LX2,LY2,LZ2)
      REAL Y(LX1,LY1,LZ1)
C
      COMMON /CTMP0/ XA(LX1,LY2,LZ2), XB(LX1,LY1,LZ2)
C
      NYZ2 = NY2*NZ2
      NXY1 = NX1*NY1
C
C     Use the appropriate derivative- and interpolation operator in
C     the y-direction (= radial direction if axisymmetric).
C
      IF (IFAXIS) THEN
         NY21   = NY1*NY2
         IF (IFRZER(IEL))      CALL COPY (IYM12,IAM12,NY21)
         IF (.NOT.IFRZER(IEL)) CALL COPY (IYM12,ICM12,NY21)
      ENDIF
C
      CALL MXM (IXTM12,NX1,X,NX2,XA,NYZ2)
      DO 100 IZ=1,NZ2
         CALL MXM (XA(1,1,IZ),NX1,IYM12,NY2,XB(1,1,IZ),NY1)
 100  CONTINUE
      CALL MXM (XB,NXY1,IZM12,NZ2,Y,NZ1)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine out_xyz_el(x,y,z,e)
      include 'SIZE'
      integer e
      real x(1),y(1),z(1)
c
      call out_fld_el(x,e,'XQ')
      call out_fld_el(y,e,'YQ')
      call out_fld_el(z,e,'ZQ')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_fld_el(x,e,c2)
      include 'SIZE'
      real x(lx1,ly1,lz1,lelt)
      integer e
      character*2 c2
c
      write(6,1) c2,e
      nx8 = min(nx1,8)
      do k=1,nz1
      do j=1,ny1
         write(6,1) c2,e,(x(i,j,k,e),i=1,nx8)
      enddo
      enddo
    1 format(a2,i6,1p8e11.3)
      return
      end
c-----------------------------------------------------------------------
      subroutine outxm3j(xm3,ym3,jm3)
      include 'SIZE'
      include 'TOTAL'

      real xm3(lx1,ly1,lz1,lelv)
      real ym3(lx1,ly1,lz1,lelv)
      real jm3(lx1,ly1,lz1,lelv)

      integer e

      do e=1,nelt
         write(6,*) e,nelfld(e),iftmsh(e),' iftmsh'
         call outmat(xm3(1,1,1,e),nx3,ny3,' xm3  ',e)
         call outmat(ym3(1,1,1,e),nx3,ny3,' ym3  ',e)
         call outmat(jm3(1,1,1,e),nx3,ny3,' jm3  ',e)
      enddo

      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE INVMT(A,B,AA,N)
C
      REAL A(N,N),AA(N,N),B(N,N)
      INTEGER INDX(100)
C
      NN = N*N
      DO 12 I=1,N
       DO 11 J=1,N
        B(I,J) = 0.0
 11    CONTINUE
       B(I,I) = 1.0
 12   CONTINUE
C
      CALL COPY  (AA,A,NN)
      CALL LUDCMP(AA,N,N,INDX,D)
      DO 13 J=1,N
       CALL LUBKSB(AA,N,N,INDX,B(1,J))
 13   CONTINUE
C
      RETURN
      END

      SUBROUTINE LUBKSB(A,N,NP,INDX,B)
      REAL A(NP,NP),B(N)
      INTEGER INDX(N)
      II=0
      DO 12 I=1,N
        LL=INDX(I)
        SUM=B(LL)
        B(LL)=B(I)
        IF (II.NE.0)THEN
          DO 11 J=II,I-1
            SUM=SUM-A(I,J)*B(J)
11        CONTINUE
        ELSE IF (SUM.NE.0.0) THEN
          II=I
        ENDIF
        B(I)=SUM
12    CONTINUE
      DO 14 I=N,1,-1
        SUM=B(I)
        IF(I.LT.N)THEN
          DO 13 J=I+1,N
            SUM=SUM-A(I,J)*B(J)
13        CONTINUE
        ENDIF
        B(I)=SUM/A(I,I)
14    CONTINUE
      RETURN
      END
      SUBROUTINE LUDCMP(A,N,NP,INDX,D)
      PARAMETER (NMAX=100,TINY=1.0E-20)
      REAL A(NP,NP),VV(NMAX)
      INTEGER INDX(N)
      D=1.0
      DO 12 I=1,N
        AAMAX=0.0
        DO 11 J=1,N
          IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
11      CONTINUE
        IF (AAMAX.EQ.0.0) THEN
           write(6,*) 'Singular matrix.'
           call exitt
        ENDIF
        VV(I)=1.0/AAMAX
12    CONTINUE
      DO 19 J=1,N
        IF (J.GT.1) THEN
          DO 14 I=1,J-1
            SUM=A(I,J)
            IF (I.GT.1)THEN
              DO 13 K=1,I-1
                SUM=SUM-A(I,K)*A(K,J)
13            CONTINUE
              A(I,J)=SUM
            ENDIF
14        CONTINUE
        ENDIF
        AAMAX=0.0
        DO 16 I=J,N
          SUM=A(I,J)
          IF (J.GT.1)THEN
            DO 15 K=1,J-1
              SUM=SUM-A(I,K)*A(K,J)
15          CONTINUE
            A(I,J)=SUM
          ENDIF
          DUM=VV(I)*ABS(SUM)
          IF (DUM.GE.AAMAX) THEN
            IMAX=I
            AAMAX=DUM
          ENDIF
16      CONTINUE
        IF (J.NE.IMAX)THEN
          DO 17 K=1,N
            DUM=A(IMAX,K)
            A(IMAX,K)=A(J,K)
            A(J,K)=DUM
17        CONTINUE
          D=-D
          VV(IMAX)=VV(J)
        ENDIF
        INDX(J)=IMAX
        IF(J.NE.N)THEN
          IF(A(J,J).EQ.0.)A(J,J)=TINY
          DUM=1.0/A(J,J)
          DO 18 I=J+1,N
            A(I,J)=A(I,J)*DUM
18        CONTINUE
        ENDIF
19    CONTINUE
      IF(A(N,N).EQ.0.0)A(N,N)=TINY
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine set_unr
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'TOPOL'
c
      integer e,f,ef
c
      nface = 2*ndim
      call dsset(nx1,ny1,nz1)
c
      do e=1,nelt
      do ef=1,nface

         f      = eface1(ef)
         js1    = skpdat(1,f)
         jf1    = skpdat(2,f)
         jskip1 = skpdat(3,f)
         js2    = skpdat(4,f)
         jf2    = skpdat(5,f)
         jskip2 = skpdat(6,f)

         i = 0
         if (if3d) then
            do j2=js2,jf2,jskip2
            do j1=js1,jf1,jskip1
               i = i+1
               a = 1./jacm1(j1,j2,1,e)
               unr(i,ef,e) = a * ( rxm1(j1,j2,1,e)*unx(i,1,ef,e)
     $                           + rym1(j1,j2,1,e)*uny(i,1,ef,e)
     $                           + rzm1(j1,j2,1,e)*unz(i,1,ef,e) )
               uns(i,ef,e) = a * ( sxm1(j1,j2,1,e)*unx(i,1,ef,e)
     $                           + sym1(j1,j2,1,e)*uny(i,1,ef,e)
     $                           + szm1(j1,j2,1,e)*unz(i,1,ef,e) )
               unt(i,ef,e) = a * ( txm1(j1,j2,1,e)*unx(i,1,ef,e)
     $                           + tym1(j1,j2,1,e)*uny(i,1,ef,e)
     $                           + tzm1(j1,j2,1,e)*unz(i,1,ef,e) )
            enddo
            enddo
         else
            do j2=js2,jf2,jskip2
            do j1=js1,jf1,jskip1
               i = i+1
               a = 1./jacm1(j1,j2,1,e)
               unr(i,ef,e) = a * ( rxm1(j1,j2,1,e)*unx(i,1,ef,e)
     $                           + rym1(j1,j2,1,e)*uny(i,1,ef,e) )
               uns(i,ef,e) = a * ( sxm1(j1,j2,1,e)*unx(i,1,ef,e)
     $                           + sym1(j1,j2,1,e)*uny(i,1,ef,e) )
               unt(i,ef,e) = 0.
            enddo
            enddo
         endif
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
