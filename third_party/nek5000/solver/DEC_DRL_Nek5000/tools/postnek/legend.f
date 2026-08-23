c-----------------------------------------------------------------------
      subroutine legend(zpts,wght,n)
c
C     Fills vector ZPTS with Gauss-Legendre-Lobatto Collocation points.
C     -1 < ZPTS(I)< 1
c 
      call zwgll(zpts,wght,n)
c
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE DUDXYZ (DU,U,RM1,SM1,TM1,JM1,IEL)
C--------------------------------------------------------------
C
C     DU  - dU/dx or dU/dy or dU/dz
C     U   - a field variable defined on mesh 1
C     RM1 - dr/dx or dr/dy or dr/dz
C     SM1 - ds/dx or ds/dy or ds/dz
C     TM1 - dt/dx or dt/dy or dt/dz
C     JM1 - the Jacobian
C     IEL - element number
C
C--------------------------------------------------------------
C
      INCLUDE 'basics.inc'
      REAL  DU  (NX,NY,NZ)
      REAL  U   (NX,NY,NZ,1)
      REAL  RM1 (NX,NY,NZ,1)
      REAL  SM1 (NX,NY,NZ,1)
      REAL  TM1 (NX,NY,NZ,1)
      REAL  JM1 (NX,NY,NZ,1)
C
      PARAMETER (NXYZM=NXM*NYM*NZM)
      COMMON /CTMP3/ DRST(NXYZM)
C
      NXY1  = NX*NY
      NYZ1  = NY*NZ
      NXYZ1 = NX*NY*NZ
C
      CALL MXM     (DGDR,NX,U(1,1,1,IEL),NX,DRST,NYZ1)
      CALL COL3    (DU,RM1(1,1,1,IEL),DRST,NXYZ1)
      DO 10 IZ=1,NZ
      IZOFF=(IZ-1)*NXY1+1
         CALL MXM  (U(1,1,IZ,IEL),NX,DGDRT,NY,DRST(IZOFF),NY)
 10   CONTINUE
      CALL ADDCOL3 (DU,SM1(1,1,1,IEL),DRST,NXYZ1)
      IF (IF3D) THEN
         CALL MXM     (U(1,1,1,IEL),NXY1,DGDRT,NZ,DRST,NZ)
         CALL ADDCOL3 (DU,TM1(1,1,1,IEL),DRST,NXYZ1)
      ENDIF
C
C...NEW AXIS CODE..
       CALL INVCOL2 (DU,JM1(1,1,1,IEL),NXYZ1)
C...NEW AXIS CODE..     IF (.NOT.IFAXIS) THEN
C...NEW AXIS CODE..        CALL INVCOL2 (DU,JM1(1,1,1,IEL),NXYZ1)
C...NEW AXIS CODE..     ELSE
C...NEW AXIS CODE..        DO 100 I=1,NXYZ1
C...NEW AXIS CODE..           IF(JM1(I,1,1,IEL).NE.0.)
C...NEW AXIS CODE..    $      DU(I,1,1) = DU(I,1,1)/JM1(I,1,1,IEL)
C...NEW AXIS CODE..100     CONTINUE
C...NEW AXIS CODE..     ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE IGLLMT(IT12,Z1,Z2,NZ1,NZ2,ND1,ND2)
C----------------------------------------------------------------------
C
C     Compute the transpose (ONLY!) of the one-dimensional interpolation 
C     operator (matrix) IT12 for interpolating a variable from a
C     Gauss-Lobatto Legendre mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss-Lobatto Legendre points.
C     Z2 : NZ2 points on mesh M.
C
C--------------------------------------------------------------------
      REAL IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         CALL HLEGN(IT12(1,I),ZI,Z1,NZ1)
 10   CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      REAL FUNCTION HLEG (I,ZZ,ZLEG,NZZ)
      INCLUDE 'basics.inc'
      REAL ZLEG(NZZ)
      REAL    ALFAN,PNZZ(NXM)
      SAVE    ALFAN,PNZZ
      INTEGER NZ0
      SAVE    NZ0
      DATA    NZ0 /0/
C
      IF (NZZ.NE.NZ0) THEN
         NZ0 = NZZ
         NN = NZZ - 1
         ALFAN = FLOAT(NN*(NN+1))
         DO 10 IZ = 1,NZZ
            PNZZ(IZ) = ALFAN*PNZ(ZLEG(IZ),NZZ)
   10    CONTINUE
      ENDIF
C
      EPS = 1.E-7
      DZ = ZZ - ZLEG(I)
      IF (ABS(DZ) .LT. EPS) THEN
         HLEG = 1.
         RETURN
      ENDIF
      HLEG = - (1.-ZZ*ZZ)*PND(ZZ,NZZ)/(PNZZ(I)*(ZZ-ZLEG(I)))
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE HLEGN(H,ZH,ZLEG,NZZ)
      INCLUDE 'basics.inc'
      REAL H(1)
      REAL ZLEG(1)
      REAL    ALFAN,PNZZ(NXM)
      SAVE    ALFAN,PNZZ
      INTEGER NZ0
      SAVE    NZ0
      DATA    NZ0 /0/
C
      IF (NZZ.NE.NZ0) THEN
         NZ0 = NZZ
         ALFAN = FLOAT(NZZ*(NZZ-1))
         DO 10 I=1,NZZ
            PNZZ(I) = ALFAN*PNZ(ZPTS(I),NZZ)
   10    CONTINUE
      ENDIF
C
      IF (NZZ.EQ.1) THEN
         H(1)=1.0
         RETURN
      ENDIF
C
      EPS = 1.E-7
      PD  =  - (1.-ZH*ZH)*PND(ZH,NZZ)
C
      DO 20 I=1,NZZ
         DZ = ZH - ZPTS(I)
         IF (ABS(DZ) .LT. EPS) THEN
            H(I) = 1.0
         ELSE
            H(I) = PD/(PNZZ(I)*DZ)
         ENDIF
   20 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DERMAT (D,DT,PN,Z,NZ,NZD)
      REAL D(NZD,NZD),DT(NZD,NZD), PN(NZD), Z(NZD)
      IF (NZ.EQ.1) THEN
         D(1,1) = 0.
         RETURN
      ENDIF
      N  = NZ-1
      D0 = N*(N+1.)/4.
      DO 100 I = 1, NZ
         DO 100 J = 1, NZ
            D(I,J) = 0.0
            IF (I.NE.J) D(I,J) = PN(I)/(PN(J)*(Z(I)-Z(J)))
            IF ((I.EQ.J).AND.(I.EQ.1))  D(I,J) = - D0
            IF ((I.EQ.J).AND.(I.EQ.NZ)) D(I,J) = D0
 100  CONTINUE
      DO 200 J=1,NZ
      DO 200 I=1,NZ
      DT(J,I)=D(I,J)
  200 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      REAL FUNCTION PNZ (Z,NZ)
      P1   = 1.
      P2   = Z
      DO 10 N = 1, NZ-2
         P3  = ((2.*N+1.)*Z*P2 - N*P1)/(N+1.)
         P1  = P2
         P2  = P3
 10   CONTINUE
      PNZ = P3
      RETURN
      END
c-----------------------------------------------------------------------
      REAL FUNCTION PND (Z,NZ)
      P1   = 1.
      P2   = Z
      P1D  = 0.
      P2D  = 1.
      P3D  = 1.
      DO 10 N = 1, NZ-2
         P3  = ((2.*N+1.)*Z*P2 - N*P1)/(N+1.)
         P3D = ((2.*N+1.)*P2 + (2.*N+1.)*Z*P2D - N*P1D)/(N+1.)
         P1  = P2
         P2  = P3
         P1D = P2D
         P2D = P3D
 10   CONTINUE
      PND = P3D
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GENPN1(PN,Z,NZ)
      REAL PN(1), Z(1)
      DO 100 I = 1, NZ
         P1  = 1.
         P2  = Z(I)
         DO 10 N = 1, NZ-2
            P3  = ((2.*N+1.)*Z(I)*P2 - N*P1)/(N+1.)
            P1  = P2
            P2  = P3
 10      CONTINUE
         PN(I) = P3
 100  CONTINUE
      RETURN
      END
C
c-----------------------------------------------------------------------
