      program tst
      call get_scalar(s,e,nx,inew)
      stop
      end
c-----------------------------------------------------------------------
      subroutine get_velocity(u,v,w,e)
      return
      end
c-----------------------------------------------------------------------
      subroutine get_coords  (x,y,z,e)
      return
      end
c-----------------------------------------------------------------------
      subroutine get_scalar(s,ei,nx,inew)
      real s(nx*nx*nx)

      integer ei

c     inew = 0 -- a new call
c     inew = 1 -- not a new call


      parameter (lelx=34,lely=53,lelz=54)

      parameter(lx = 20)

      common /cb/ vb(lx*lelx),rho(lx)

      real u(lx*lx*lx),v(lx*lx*lx),w(lx*lx*lx)
      real x(lx*lx*lx),y(lx*lx*lx),z(lx*lx*lx)
      common /cu/ u,v,w,x,y,z

      integer e,ex,ey,ez
      
      if (nx.gt.lx) then
         write(6,*) 'error in get_scalar - increase lx:',lx,nx
         call exit
      endif

      if (inew.eq.0) then

         inew = 1

         call zwgll(vb,rho,nx)
         l = 0
         do ex =1,lelx
         do i =1,nx
            l = l+1
            vb(l) = 0
            db    = 0
            do ez=1,lelz
            do ey=1,lely
               e = ex + (ey-1)*lelx + (ez-1)*lelx*lely
               call get_velocity(u,v,w,e)
               call get_coords  (x,y,z,e)
               do k =1,nx
               do j =1,nx
                  ijk = i+(j-1)*nx + (k-1)*nx*nx
                  rr  = x(ijk)**2 + y(ijk)**2
                  rr  = sqrt(rr)
                  ct  = x(ijk)/rr
                  st  = y(ijk)/rr
                  vr  =  ct*u(ijk) + st*v(ijk)
                  vt  = -st*u(ijk) + ct*v(ijk)
                  vb(l)=vb(l)+rho(j)*rho(k)*vt
                  db   =db   +rho(j)*rho(k)
               enddo
               enddo
            enddo
            enddo
            vb(l) = vb(l) / db
         enddo
         enddo
      endif

      e = ei

      call get_velocity(u,v,w,e)
      call get_coords  (x,y,z,e)

      call get_exyz (ex,ey,ez,e,lelx,lely,lelz)

      l = 0
      do k=1,nx
      do j=1,nx
      do i=1,nx
         l   = l+1
         rr  = x(l)**2 + y(l)**2
         rr  = sqrt(rr)
         ct  = x(l)/rr
         st  = y(l)/rr
         vr  =  ct*u(l) + st*v(l)
         vt  = -st*u(l) + ct*v(l)

         iex  = i + nx*(ex-1)
         s(l) = vr*rr*(vt-vb(iex))

      enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine get_exyz(ex,ey,ez,eg,nelx,nely,nelz)
      integer ex,ey,ez,eg
c
      nelxy = nelx*nely
c
      ez = 1 +  (eg-1)/nelxy
      ey = mod1 (eg,nelxy)
      ey = 1 +  (ey-1)/nelx
      ex = mod1 (eg,nelx)
c
      return
      end
c-----------------------------------------------------------------------
C==============================================================================
C
C     LIBRARY ROUTINES FOR SPECTRAL METHODS
C
C     March 1989
C
C     For questions, comments or suggestions, please contact:
C
C     Einar Malvin Ronquist
C     Room 3-243
C     Department of Mechanical Engineering
C     Massachusetts Institute of Technology
C     77 Massachusetts Avenue
C     Cambridge, MA 02139
C     U.S.A.
C
C------------------------------------------------------------------------------
C
C     ABBRIVIATIONS:
C
C     M   - Set of mesh points
C     Z   - Set of collocation/quadrature points
C     W   - Set of quadrature weights
C     H   - Lagrangian interpolant
C     D   - Derivative operator
C     I   - Interpolation operator
C     GL  - Gauss Legendre
C     GLL - Gauss-Lobatto Legendre
C     GJ  - Gauss Jacobi
C     GLJ - Gauss-Lobatto Jacobi
C
C
C     MAIN ROUTINES:
C
C     Points and weights:
C
C     ZWGL      Compute Gauss Legendre points and weights
C     ZWGLL     Compute Gauss-Lobatto Legendre points and weights
C     ZWGJ      Compute Gauss Jacobi points and weights (general)
C     ZWGLJ     Compute Gauss-Lobatto Jacobi points and weights (general)
C
C     Lagrangian interpolants:
C
C     HGL       Compute Gauss Legendre Lagrangian interpolant
C     HGLL      Compute Gauss-Lobatto Legendre Lagrangian interpolant
C     HGJ       Compute Gauss Jacobi Lagrangian interpolant (general)
C     HGLJ      Compute Gauss-Lobatto Jacobi Lagrangian interpolant (general)
C
C     Derivative operators:
C
C     DGLL      Compute Gauss-Lobatto Legendre derivative matrix
C     DGLLGL    Compute derivative matrix for a staggered mesh (GLL->GL)
C     DGJ       Compute Gauss Jacobi derivative matrix (general)
C     DGLJ      Compute Gauss-Lobatto Jacobi derivative matrix (general)
C     DGLJGJ    Compute derivative matrix for a staggered mesh (GLJ->GJ) (general)
C
C     Interpolation operators:
C
C     IGLM      Compute interpolation operator GL  -> M
C     IGLLM     Compute interpolation operator GLL -> M
C     IGJM      Compute interpolation operator GJ  -> M  (general)
C     IGLJM     Compute interpolation operator GLJ -> M  (general)
C
C     Other:
C
C     PNLEG     Compute Legendre polynomial of degree N
C     PNDLEG    Compute derivative of Legendre polynomial of degree N
C
C     Comments:
C
C     Note that many of the above routines exist in both single and
C     double precision. If the name of the single precision routine is
C     SUB, the double precision version is called SUBD. In most cases
C     all the "low-level" arithmetic is done in double precision, even
C     for the single precsion versions.
C
C     Useful references:
C
C [1] Gabor Szego: Orthogonal Polynomials, American Mathematical Society,
C     Providence, Rhode Island, 1939.
C [2] Abramowitz & Stegun: Handbook of Mathematical Functions,
C     Dover, New York, 1972.
C [3] Canuto, Hussaini, Quarteroni & Zang: Spectral Methods in Fluid
C     Dynamics, Springer-Verlag, 1988.
C
C
C==============================================================================
C
C--------------------------------------------------------------------
      SUBROUTINE ZWGL (Z,W,NP)
C--------------------------------------------------------------------
C
C     Generate NP Gauss Legendre points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha=0,beta=0).
C     The polynomial degree N=NP-1.
C     Z and W are in single precision, but all the arithmetic
C     operations are done in double precision.
C
C--------------------------------------------------------------------
      REAL Z(1),W(1)
      ALPHA = 0.
      BETA  = 0.
      CALL ZWGJ (Z,W,NP,ALPHA,BETA)
      RETURN
      END
C
      SUBROUTINE ZWGLL (Z,W,NP)
C--------------------------------------------------------------------
C
C     Generate NP Gauss-Lobatto Legendre points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha=0,beta=0).
C     The polynomial degree N=NP-1.
C     Z and W are in single precision, but all the arithmetic
C     operations are done in double precision.
C
C--------------------------------------------------------------------
      REAL Z(1),W(1)
      ALPHA = 0.
      BETA  = 0.
      CALL ZWGLJ (Z,W,NP,ALPHA,BETA)
      RETURN
      END
C
      SUBROUTINE ZWGJ (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Single precision version.
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZD = 64)
      DOUBLE PRECISION ZD(NZD),WD(NZD)
      REAL Z(1),W(1),ALPHA,BETA
C
      IF (NP.GT.NZD) THEN
         WRITE (6,*) 'Too large polynomial degree in ZWGJ'
         WRITE (6,*) 'Maximum polynomial degree is',NZD,'Here NP=',NP
         STOP
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      CALL ZWGJD (ZD,WD,NP,ALPHAD,BETAD)
      DO 100 I=1,NP
         Z(I) = ZD(I)
         W(I) = WD(I)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGJD (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Double precision version.
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION Z(1),W(1),ALPHA,BETA
C
      N     = NP-1
      DN    = DBLE(FLOAT(N))
      ONE   = 1.D0
      TWO   = 2.D0
      APB   = ALPHA+BETA
C
      IF (NP.LE.0) THEN
         WRITE (6,*) 'Minimum number of Gauss points is 1'
         STOP
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      IF (NP.EQ.1) THEN
         Z(1) = (BETA-ALPHA)/(APB+TWO)
         W(1) = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+ONE)/GAMMAF(APB+TWO)
     $          * TWO**(APB+ONE)
         RETURN
      ENDIF
C
      CALL JACG (Z,NP,ALPHA,BETA)
C
      NP1   = N+1
      NP2   = N+2
      DNP1  = DBLE(FLOAT(NP1))
      DNP2  = DBLE(FLOAT(NP2))
      FAC1  = DNP1+ALPHA+BETA+ONE
      FAC2  = FAC1+DNP1
      FAC3  = FAC2+ONE
      FNORM = PNORMJ(NP1,ALPHA,BETA)
      RCOEF = (FNORM*FAC2*FAC3)/(TWO*FAC1*DNP2)
      DO 100 I=1,NP
         CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,NP2,ALPHA,BETA,Z(I))
         W(I) = -RCOEF/(P*PDM1)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGLJ (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS LOBATTO JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Single precision version.
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZD = 64)
      DOUBLE PRECISION ZD(NZD),WD(NZD)
      REAL Z(1),W(1),ALPHA,BETA
C
      IF (NP.GT.NZD) THEN
         WRITE (6,*) 'Too large polynomial degree in ZWGLJ'
         WRITE (6,*) 'Maximum polynomial degree is',NZD,'Here NP=',NP
         STOP
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      CALL ZWGLJD (ZD,WD,NP,ALPHAD,BETAD)
      DO 100 I=1,NP
         Z(I) = ZD(I)
         W(I) = WD(I)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGLJD (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS LOBATTO JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Double precision version.
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION Z(1),W(1),ALPHA,BETA
C
      N     = NP-1
      NM1   = N-1
      ONE   = 1.D0
      TWO   = 2.D0
C
      IF (NP.LE.1) THEN
         WRITE (6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      IF (NM1.GT.0) THEN
         ALPG  = ALPHA+ONE
         BETG  = BETA+ONE
         CALL ZWGJD (Z(2),W(2),NM1,ALPG,BETG)
      ENDIF
      Z(1)  = -ONE
      Z(NP) =  ONE
      DO 100  I=2,NP-1
         W(I) = W(I)/(ONE-Z(I)**2)
 100  CONTINUE
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(1))
      W(1)  = ENDW1 (N,ALPHA,BETA)/(TWO*PD)
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(NP))
      W(NP) = ENDW2 (N,ALPHA,BETA)/(TWO*PD)
C
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION ENDW1 (N,ALPHA,BETA)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION ALPHA,BETA
      ZERO  = 0.D0
      ONE   = 1.D0
      TWO   = 2.D0
      THREE = 3.D0
      FOUR  = 4.D0
      APB   = ALPHA+BETA
      IF (N.EQ.0) THEN
         ENDW1 = ZERO
         RETURN
      ENDIF
      F1   = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+ONE)/GAMMAF(APB+THREE)
      F1   = F1*(APB+TWO)*TWO**(APB+TWO)/TWO
      IF (N.EQ.1) THEN
         ENDW1 = F1
         RETURN
      ENDIF
      FINT1 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+ONE)/GAMMAF(APB+THREE)
      FINT1 = FINT1*TWO**(APB+TWO)
      FINT2 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+TWO)/GAMMAF(APB+FOUR)
      FINT2 = FINT2*TWO**(APB+THREE)
      F2    = (-TWO*(BETA+TWO)*FINT1 + (APB+FOUR)*FINT2)
     $        * (APB+THREE)/FOUR
      IF (N.EQ.2) THEN
         ENDW1 = F2
         RETURN
      ENDIF
      DO 100 I=3,N
         DI   = DBLE(FLOAT(I-1))
         ABN  = ALPHA+BETA+DI
         ABNN = ABN+DI
         A1   = -(TWO*(DI+ALPHA)*(DI+BETA))/(ABN*ABNN*(ABNN+ONE))
         A2   =  (TWO*(ALPHA-BETA))/(ABNN*(ABNN+TWO))
         A3   =  (TWO*(ABN+ONE))/((ABNN+TWO)*(ABNN+ONE))
         F3   =  -(A2*F2+A1*F1)/A3
         F1   = F2
         F2   = F3
 100  CONTINUE
      ENDW1  = F3
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION ENDW2 (N,ALPHA,BETA)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION ALPHA,BETA
      ZERO  = 0.D0
      ONE   = 1.D0
      TWO   = 2.D0
      THREE = 3.D0
      FOUR  = 4.D0
      APB   = ALPHA+BETA
      IF (N.EQ.0) THEN
         ENDW2 = ZERO
         RETURN
      ENDIF
      F1   = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+TWO)/GAMMAF(APB+THREE)
      F1   = F1*(APB+TWO)*TWO**(APB+TWO)/TWO
      IF (N.EQ.1) THEN
         ENDW2 = F1
         RETURN
      ENDIF
      FINT1 = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+TWO)/GAMMAF(APB+THREE)
      FINT1 = FINT1*TWO**(APB+TWO)
      FINT2 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+TWO)/GAMMAF(APB+FOUR)
      FINT2 = FINT2*TWO**(APB+THREE)
      F2    = (TWO*(ALPHA+TWO)*FINT1 - (APB+FOUR)*FINT2)
     $        * (APB+THREE)/FOUR
      IF (N.EQ.2) THEN
         ENDW2 = F2
         RETURN
      ENDIF
      DO 100 I=3,N
         DI   = DBLE(FLOAT(I-1))
         ABN  = ALPHA+BETA+DI
         ABNN = ABN+DI
         A1   =  -(TWO*(DI+ALPHA)*(DI+BETA))/(ABN*ABNN*(ABNN+ONE))
         A2   =  (TWO*(ALPHA-BETA))/(ABNN*(ABNN+TWO))
         A3   =  (TWO*(ABN+ONE))/((ABNN+TWO)*(ABNN+ONE))
         F3   =  -(A2*F2+A1*F1)/A3
         F1   = F2
         F2   = F3
 100  CONTINUE
      ENDW2  = F3
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION GAMMAF (X)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION X
      ZERO = 0.0D0
      HALF = 0.5D0
      ONE  = 1.0D0
      TWO  = 2.0D0
      FOUR = 4.0D0
      PI   = FOUR*DATAN(ONE)
      GAMMAF = ONE
      IF (X.EQ.-HALF) GAMMAF = -TWO*DSQRT(PI)
      IF (X.EQ. HALF) GAMMAF =  DSQRT(PI)
      IF (X.EQ. ONE ) GAMMAF =  ONE
      IF (X.EQ. TWO ) GAMMAF =  ONE
      IF (X.EQ. 1.5D0) GAMMAF =  DSQRT(PI)/2.D0
      IF (X.EQ. 2.5D0) GAMMAF =  1.5D0*DSQRT(PI)/2.D0
      IF (X.EQ. 3.5D0) GAMMAF =  2.5D0*1.5D0*DSQRT(PI)/2.D0
      IF (X.EQ. 3.D0 ) GAMMAF =  2.D0
      IF (X.EQ. 4.D0 ) GAMMAF = 6.D0
      IF (X.EQ. 5.D0 ) GAMMAF = 24.D0
      IF (X.EQ. 6.D0 ) GAMMAF = 120.D0
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION PNORMJ (N,ALPHA,BETA)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION ALPHA,BETA
      ONE   = 1.D0
      TWO   = 2.D0
      DN    = DBLE(FLOAT(N))
      CONST = ALPHA+BETA+ONE
      IF (N.LE.1) THEN
         PROD   = GAMMAF(DN+ALPHA)*GAMMAF(DN+BETA)
         PROD   = PROD/(GAMMAF(DN)*GAMMAF(DN+ALPHA+BETA))
         PNORMJ = PROD * TWO**CONST/(TWO*DN+CONST)
         RETURN
      ENDIF
      PROD  = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+ONE)
      PROD  = PROD/(TWO*(ONE+CONST)*GAMMAF(CONST+ONE))
      PROD  = PROD*(ONE+ALPHA)*(TWO+ALPHA)
      PROD  = PROD*(ONE+BETA)*(TWO+BETA)
      DO 100 I=3,N
         DINDX = DBLE(FLOAT(I))
         FRAC  = (DINDX+ALPHA)*(DINDX+BETA)/(DINDX*(DINDX+ALPHA+BETA))
         PROD  = PROD*FRAC
 100  CONTINUE
      PNORMJ = PROD * TWO**CONST/(TWO*DN+CONST)
      RETURN
      END
C
      SUBROUTINE JACG (XJAC,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Compute NP Gauss points XJAC, which are the zeros of the
C     Jacobi polynomial J(NP) with parameters ALPHA and BETA.
C     ALPHA and BETA determines the specific type of Gauss points.
C     Examples:
C     ALPHA = BETA =  0.0  ->  Legendre points
C     ALPHA = BETA = -0.5  ->  Chebyshev points
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION XJAC(1)
      DATA KSTOP/10/
      DATA EPS/1.0D-12/
      N   = NP-1
      DTH = 4.D0*DATAN(1.D0)/(2.D0*DBLE(FLOAT(N))+2.D0)
      DO 40 J=1,NP
         IF (J.EQ.1) THEN
            X = DCOS((2.D0*(DBLE(FLOAT(J))-1.D0)+1.D0)*DTH)
         ELSE
            X1 = DCOS((2.D0*(DBLE(FLOAT(J))-1.D0)+1.D0)*DTH)
            X2 = XLAST
            X  = (X1+X2)/2.D0
         ENDIF
         DO 30 K=1,KSTOP
            CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,X)
            RECSUM = 0.D0
            JM = J-1
            DO 29 I=1,JM
               RECSUM = RECSUM+1.D0/(X-XJAC(NP-I+1))
 29         CONTINUE
            DELX = -P/(PD-RECSUM*P)
            X    = X+DELX
            IF (ABS(DELX) .LT. EPS) GOTO 31
 30      CONTINUE
 31      CONTINUE
         XJAC(NP-J+1) = X
         XLAST        = X
 40   CONTINUE
      DO 200 I=1,NP
         XMIN = 2.D0
         DO 100 J=I,NP
            IF (XJAC(J).LT.XMIN) THEN
               XMIN = XJAC(J)
               JMIN = J
            ENDIF
 100     CONTINUE
         IF (JMIN.NE.I) THEN
            SWAP = XJAC(I)
            XJAC(I) = XJAC(JMIN)
            XJAC(JMIN) = SWAP
         ENDIF
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE JACOBF (POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,
     $                   N,ALP,BET,X)
C--------------------------------------------------------------------
C
C     Computes the Jacobi polynomial (POLY) and its derivative (PDER)
C     of degree N at X.
C
C--------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      APB  = ALP+BET
      POLY = 1.D0
      PDER = 0.D0
      IF (N .EQ. 0) RETURN
      POLYL = POLY
      PDERL = PDER
      POLY  = (ALP-BET+(APB+2.D0)*X)/2.D0
      PDER  = (APB+2.D0)/2.D0
      IF (N .EQ. 1) RETURN
      DO 20 K=2,N
         DK = DBLE(FLOAT(K))
         A1 = 2.D0*DK*(DK+APB)*(2.D0*DK+APB-2.D0)
         A2 = (2.D0*DK+APB-1.D0)*(ALP**2-BET**2)
         B3 = (2.D0*DK+APB-2.D0)
         A3 = B3*(B3+1.D0)*(B3+2.D0)
         A4 = 2.D0*(DK+ALP-1.D0)*(DK+BET-1.D0)*(2.D0*DK+APB)
         POLYN  = ((A2+A3*X)*POLY-A4*POLYL)/A1
         PDERN  = ((A2+A3*X)*PDER-A4*PDERL+A3*POLY)/A1
         PSAVE  = POLYL
         PDSAVE = PDERL
         POLYL  = POLY
         POLY   = POLYN
         PDERL  = PDER
         PDER   = PDERN
 20   CONTINUE
      POLYM1 = POLYL
      PDERM1 = PDERL
      POLYM2 = PSAVE
      PDERM2 = PDSAVE
      RETURN
      END
C
      REAL FUNCTION HGJ (II,Z,ZGJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGJ through
C     the NP Gauss Jacobi points ZGJ at the point Z.
C     Single precision version.
C
C---------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZD = 64)
      DOUBLE PRECISION ZD,ZGJD(NZD)
      REAL Z,ZGJ(1),ALPHA,BETA
      IF (NP.GT.NZD) THEN
         WRITE (6,*) 'Too large polynomial degree in HGJ'
         WRITE (6,*) 'Maximum polynomial degree is',NZD,'Here NP=',NP
         STOP
      ENDIF
      ZD = Z
      DO 100 I=1,NP
         ZGJD(I) = ZGJ(I)
 100  CONTINUE
      ALPHAD = ALPHA
      BETAD  = BETA
      HGJ    = HGJD (II,ZD,ZGJD,NP,ALPHAD,BETAD)
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION HGJD (II,Z,ZGJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGJD through
C     the NZ Gauss-Lobatto Jacobi points ZGJ at the point Z.
C     Double precision version.
C
C---------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION Z,ZGJ(1),ALPHA,BETA
      EPS = 1.D-5
      ONE = 1.D0
      ZI  = ZGJ(II)
      DZ  = Z-ZI
      IF (ABS(DZ).LT.EPS) THEN
         HGJD = ONE
         RETURN
      ENDIF
      CALL JACOBF (PZI,PDZI,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,ZI)
      CALL JACOBF (PZ,PDZ,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,Z)
      HGJD  = PZ/(PDZI*(Z-ZI))
      RETURN
      END
C
      REAL FUNCTION HGLJ (II,Z,ZGLJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGLJ through
C     the NZ Gauss-Lobatto Jacobi points ZGLJ at the point Z.
C     Single precision version.
C
C---------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZD = 64)
      DOUBLE PRECISION ZD,ZGLJD(NZD)
      REAL Z,ZGLJ(1),ALPHA,BETA
      IF (NP.GT.NZD) THEN
         WRITE (6,*) 'Too large polynomial degree in HGLJ'
         WRITE (6,*) 'Maximum polynomial degree is',NZD,'Here NP=',NP
         STOP
      ENDIF
      ZD = Z
      DO 100 I=1,NP
         ZGLJD(I) = ZGLJ(I)
 100  CONTINUE
      ALPHAD = ALPHA
      BETAD  = BETA
      HGLJ   = HGLJD (II,ZD,ZGLJD,NP,ALPHAD,BETAD)
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION HGLJD (I,Z,ZGLJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGLJD through
C     the NZ Gauss-Lobatto Jacobi points ZJACL at the point Z.
C     Double precision version.
C
C---------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION Z,ZGLJ(1),ALPHA,BETA
      EPS = 1.D-5
      ONE = 1.D0
      ZI  = ZGLJ(I)
      DZ  = Z-ZI
      IF (ABS(DZ).LT.EPS) THEN
         HGLJD = ONE
         RETURN
      ENDIF
      N      = NP-1
      DN     = DBLE(FLOAT(N))
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
      CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,ZI)
      CONST  = EIGVAL*PI+ALPHA*(ONE+ZI)*PDI-BETA*(ONE-ZI)*PDI
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z)
      HGLJD  = (ONE-Z**2)*PD/(CONST*(Z-ZI))
      RETURN
      END
C
      SUBROUTINE DGJ (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss Jacobi points Z.
C     Note: D and DT are square matrices.
C     Single precision version.
C
C-----------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZDD = 64)
      DOUBLE PRECISION DD(NZDD,NZDD),DTD(NZDD,NZDD),ZD(NZDD)
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
C
      IF (NZ.LE.0) THEN
         WRITE (6,*) 'Minimum number of Gauss points is 1'
         STOP
      ENDIF
      IF (NZ .GT. NZDD) THEN
         WRITE (6,*) 'Maximum polynomial degree is',NZDD,'Here NP=',NZ
         STOP
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NZ
         ZD(I) = Z(I)
 100  CONTINUE
      CALL DGJD (DD,DTD,ZD,NZ,NZDD,ALPHAD,BETAD)
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J)  = DD(I,J)
         DT(I,J) = DTD(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGJD (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss Jacobi points Z.
C     Note: D and DT are square matrices.
C     Double precision version.
C
C-----------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
      N    = NZ-1
      DN   = DBLE(FLOAT(N))
      ONE  = 1.D0
      TWO  = 2.D0
C
      IF (NZ.LE.1) THEN
         WRITE (6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,NZ,ALPHA,BETA,Z(I))
         CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,NZ,ALPHA,BETA,Z(J))
         IF (I.NE.J) D(I,J) = PDI/(PDJ*(Z(I)-Z(J)))
         IF (I.EQ.J) D(I,J) = ((ALPHA+BETA+TWO)*Z(I)+ALPHA-BETA)/
     $                        (TWO*(ONE-Z(I)**2))
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJ (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Jacobi points Z.
C     Note: D and DT are square matrices.
C     Single precision version.
C
C-----------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NZDD = 64)
      DOUBLE PRECISION DD(NZDD,NZDD),DTD(NZDD,NZDD),ZD(NZDD)
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
C
      IF (NZ.LE.1) THEN
         WRITE (6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF (NZ .GT. NZDD) THEN
         WRITE (6,*) 'Maximum polynomial degree is',NZDD,'Here NP=',NZ
         STOP
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NZ
         ZD(I) = Z(I)
 100  CONTINUE
      CALL DGLJD (DD,DTD,ZD,NZ,NZDD,ALPHAD,BETAD)
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J)  = DD(I,J)
         DT(I,J) = DTD(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJD (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Jacobi points Z.
C     Note: D and DT are square matrices.
C     Double precision version.
C
C-----------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
      N    = NZ-1
      DN   = DBLE(FLOAT(N))
      ONE  = 1.D0
      TWO  = 2.D0
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
C
      IF (NZ.LE.1) THEN
         WRITE (6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(I))
         CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(J))
         CI = EIGVAL*PI-(BETA*(ONE-Z(I))-ALPHA*(ONE+Z(I)))*PDI
         CJ = EIGVAL*PJ-(BETA*(ONE-Z(J))-ALPHA*(ONE+Z(J)))*PDJ
         IF (I.NE.J) D(I,J) = CI/(CJ*(Z(I)-Z(J)))
         IF ((I.EQ.J).AND.(I.NE.1).AND.(I.NE.NZ))
     $   D(I,J) = (ALPHA*(ONE+Z(I))-BETA*(ONE-Z(I)))/
     $            (TWO*(ONE-Z(I)**2))
         IF ((I.EQ.J).AND.(I.EQ.1))
     $   D(I,J) =  (EIGVAL+ALPHA)/(TWO*(BETA+TWO))
         IF ((I.EQ.J).AND.(I.EQ.NZ))
     $   D(I,J) = -(EIGVAL+BETA)/(TWO*(ALPHA+TWO))
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLL (D,DT,Z,NZ,NZD)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Legendre points Z.
C     Note: D and DT are square matrices.
C
C-----------------------------------------------------------------
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1)
      N  = NZ-1
      IF (NZ .GT. NZD) THEN
         WRITE (6,*) 'Subroutine DGLL'
         WRITE (6,*) 'Maximum polynomial degree is',NZD,'Here NP=',NZ
         WRITE (6,*) 'Polynomial degree         = ',N
      ENDIF
      IF (NZ .EQ. 1) THEN
         D(1,1) = 0.
         RETURN
      ENDIF
      FN = FLOAT(N)
      D0 = FN*(FN+1.)/4.
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J) = 0.
         IF  (I.NE.J) D(I,J) = PNLEG(Z(I),N)/
     $                        (PNLEG(Z(J),N)*(Z(I)-Z(J)))
         IF ((I.EQ.J).AND.(I.EQ.1))  D(I,J) = -D0
         IF ((I.EQ.J).AND.(I.EQ.NZ)) D(I,J) =  D0
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      REAL FUNCTION HGLL (I,Z,ZGLL,NZ)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant L through
C     the NZ Gauss-Lobatto Legendre points ZGLL at the point Z.
C
C---------------------------------------------------------------------
      REAL ZGLL(1)
      EPS = 1.E-5
      DZ = Z - ZGLL(I)
      IF (ABS(DZ) .LT. EPS) THEN
         HGLL = 1.
         RETURN
      ENDIF
      N = NZ - 1
      ALFAN = FLOAT(N)*(FLOAT(N)+1.)
      HGLL = - (1.-Z*Z)*PNDLEG(Z,N)/
     $         (ALFAN*PNLEG(ZGLL(I),N)*(Z-ZGLL(I)))
      RETURN
      END
C
      REAL FUNCTION HGL (I,Z,ZGL,NZ)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGL through
C     the NZ Gauss Legendre points ZGL at the point Z.
C
C---------------------------------------------------------------------
      REAL ZGL(1)
      EPS = 1.E-5
      DZ = Z - ZGL(I)
      IF (ABS(DZ) .LT. EPS) THEN
         HGL = 1.
         RETURN
      ENDIF
      N = NZ-1
      HGL = PNLEG(Z,NZ)/(PNDLEG(ZGL(I),NZ)*(Z-ZGL(I)))
      RETURN
      END
C
      REAL FUNCTION PNLEG (Z,N)
C---------------------------------------------------------------------
C
C     Compute the value of the Nth order Legendre polynomial at Z.
C     (Simpler than JACOBF)
C     Based on the recursion formula for the Legendre polynomials.
C
C---------------------------------------------------------------------
C
C     This next statement is to overcome the underflow bug in the i860.  
C     It can be removed at a later date.  11 Aug 1990   pff.
C
      IF(ABS(Z) .LT. 1.0E-25) Z = 0.0
C
      P1   = 1.
      IF (N.EQ.0) THEN
         PNLEG = P1
         RETURN
      ENDIF
      P2   = Z
      P3   = P2
      DO 10 K = 1, N-1
         FK  = FLOAT(K)
         P3  = ((2.*FK+1.)*Z*P2 - FK*P1)/(FK+1.)
         P1  = P2
         P2  = P3
 10   CONTINUE
      PNLEG = P3
      RETURN
      END
C
      REAL FUNCTION PNDLEG (Z,N)
C----------------------------------------------------------------------
C
C     Compute the derivative of the Nth order Legendre polynomial at Z.
C     (Simpler than JACOBF)
C     Based on the recursion formula for the Legendre polynomials.
C
C----------------------------------------------------------------------
      P1   = 1.
      P2   = Z
      P1D  = 0.
      P2D  = 1.
      P3D  = 1.
      DO 10 K = 1, N-1
         FK  = FLOAT(K)
         P3  = ((2.*FK+1.)*Z*P2 - FK*P1)/(FK+1.)
         P3D = ((2.*FK+1.)*P2 + (2.*FK+1.)*Z*P2D - FK*P1D)/(FK+1.)
         P1  = P2
         P2  = P3
         P1D = P2D
         P2D = P3D
 10   CONTINUE
      PNDLEG = P3D
      RETURN
      END
C
      SUBROUTINE DGLLGL (D,DT,ZM1,ZM2,IM12,NZM1,NZM2,ND1,ND2)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Legendre mesh (M1), and evaluate its
C     derivative on a Guass Legendre mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLLGL).
C     Note: D and DT are rectangular matrices.
C
C-----------------------------------------------------------------------
      REAL D(ND2,ND1), DT(ND1,ND2), ZM1(ND1), ZM2(ND2), IM12(ND2,ND1)
      IF (NZM1.EQ.1) THEN
        D (1,1) = 0.
        DT(1,1) = 0.
        RETURN
      ENDIF
      EPS = 1.E-6
      NM1 = NZM1-1
      DO 10 IP = 1, NZM2
         DO 10 JQ = 1, NZM1
            ZP = ZM2(IP)
            ZQ = ZM1(JQ)
            IF ((ABS(ZP) .LT. EPS).AND.(ABS(ZQ) .LT. EPS)) THEN
                D(IP,JQ) = 0.
            ELSE
                D(IP,JQ) = (PNLEG(ZP,NM1)/PNLEG(ZQ,NM1)
     $                     -IM12(IP,JQ))/(ZP-ZQ)
            ENDIF
            DT(JQ,IP) = D(IP,JQ)
 10   CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJGJ (D,DT,ZGL,ZG,IGLG,NPGL,NPG,ND1,ND2,ALPHA,BETA)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Jacobi mesh (M1), and evaluate its
C     derivative on a Guass Jacobi mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLJGJ).
C     Note: D and DT are rectangular matrices.
C     Single precision version.
C
C-----------------------------------------------------------------------
      REAL D(ND2,ND1), DT(ND1,ND2), ZGL(ND1), ZG(ND2), IGLG(ND2,ND1)
      PARAMETER (NDD = 64)
      DOUBLE PRECISION DD(NDD,NDD), DTD(NDD,NDD)
      DOUBLE PRECISION ZGD(NDD), ZGLD(NDD), IGLGD(NDD,NDD)
      DOUBLE PRECISION ALPHAD, BETAD
C
      IF (NPGL.LE.1) THEN
         WRITE(6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF (NPGL.GT.NDD) THEN
         WRITE(6,*) 'Maximum polynomial degree is',NDD,'Here NP=',NPGL
         STOP
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE(6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NPG
         ZGD(I) = ZG(I)
         DO 100 J=1,NPGL
            IGLGD(I,J) = IGLG(I,J)
 100  CONTINUE
      DO 200 I=1,NPGL
         ZGLD(I) = ZGL(I)
 200  CONTINUE
      CALL DGLJGJD (DD,DTD,ZGLD,ZGD,IGLGD,NPGL,NPG,NDD,NDD,ALPHAD,BETAD)
      DO 300 I=1,NPG
      DO 300 J=1,NPGL
         D(I,J)  = DD(I,J)
         DT(J,I) = DTD(J,I)
 300  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJGJD (D,DT,ZGL,ZG,IGLG,NPGL,NPG,ND1,ND2,ALPHA,BETA)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Jacobi mesh (M1), and evaluate its
C     derivative on a Guass Jacobi mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLJGJ).
C     Note: D and DT are rectangular matrices.
C     Double precision version.
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION D(ND2,ND1), DT(ND1,ND2), ZGL(ND1), ZG(ND2)
      DOUBLE PRECISION IGLG(ND2,ND1), ALPHA, BETA
C
      IF (NPGL.LE.1) THEN
         WRITE(6,*) 'Minimum number of Gauss-Lobatto points is 2'
         STOP
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE(6,*) 'Alpha and Beta must be greater than -1'
         STOP
      ENDIF
C
      EPS    = 1.D-6
      ONE    = 1.D0
      TWO    = 2.D0
      NGL    = NPGL-1
      DN     = DBLE(FLOAT(NGL))
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
C
      DO 100 I=1,NPG
      DO 100 J=1,NPGL
         DZ = ABS(ZG(I)-ZGL(J))
         IF (DZ.LT.EPS) THEN
            D(I,J) = (ALPHA*(ONE+ZG(I))-BETA*(ONE-ZG(I)))/
     $               (TWO*(ONE-ZG(I)**2))
         ELSE
            CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,NGL,ALPHA,BETA,ZG(I))
            CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,NGL,ALPHA,BETA,ZGL(J))
            FACI   = ALPHA*(ONE+ZG(I))-BETA*(ONE-ZG(I))
            FACJ   = ALPHA*(ONE+ZGL(J))-BETA*(ONE-ZGL(J))
            CONST  = EIGVAL*PJ+FACJ*PDJ
            D(I,J) = ((EIGVAL*PI+FACI*PDI)*(ZG(I)-ZGL(J))
     $               -(ONE-ZG(I)**2)*PDI)/(CONST*(ZG(I)-ZGL(J))**2)
         ENDIF
         DT(J,I) = D(I,J)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE IGLM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss Legendre mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss Legendre points.
C     Z2 : NZ2 points on mesh M.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGL(J,ZI,Z1,NZ1)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
c
      SUBROUTINE IGLLM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss-Lobatto Legendre mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss-Lobatto Legendre points.
C     Z2 : NZ2 points on mesh M.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGLL(J,ZI,Z1,NZ1)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
C
      SUBROUTINE IGJM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2,ALPHA,BETA)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss Jacobi mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss Jacobi points.
C     Z2 : NZ2 points on mesh M.
C     Single precision version.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGJ(J,ZI,Z1,NZ1,ALPHA,BETA)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
c
      SUBROUTINE IGLJM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2,ALPHA,BETA)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss-Lobatto Jacobi mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss-Lobatto Jacobi points.
C     Z2 : NZ2 points on mesh M.
C     Single precision version.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGLJ(J,ZI,Z1,NZ1,ALPHA,BETA)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      function mod1(i,n)
c
c     Yields MOD(I,N) with the exception that if I=K*N, result is N.
c
      mod1=0
      if (i.eq.0) return
c
      if (n.eq.0) then
         write(6,*) 
     $  'warning:  aTTEMPT TO TAKE mod(i,0) IN function mod1.'
         return
      endif
c
      ii = i+n-1
      mod1 = mod(ii,n)+1
c
      return
      end
c-----------------------------------------------------------------------
