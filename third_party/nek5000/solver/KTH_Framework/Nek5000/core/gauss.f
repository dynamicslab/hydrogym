      SUBROUTINE LU(A,N,ldim,IR,IC)
C  IT IS THE FIRST SUBROUTINE TO COMPUTE THE MX. INV.
      DIMENSION A(ldim,1),IR(1),IC(1)
      DO 10I=1,N
      IR(I)=I
      IC(I)=I
10      CONTINUE
      K=1
      L=K
      M=K
      XMAX=ABS(A(K,K))
      DO 100I=K,N
      DO 100J=K,N
      Y=ABS(A(I,J))
      IF(XMAX.GE.Y) GOTO 100
      XMAX=Y
      L=I
      M=J
100     CONTINUE
      DO 1000K=1,N
      IRL=IR(L)
      IR(L)=IR(K)
      IR(K)=IRL
      ICM=IC(M)
      IC(M)=IC(K)
      IC(K)=ICM
      IF(L.EQ.K) GOTO 300
      DO 200J=1,N
      B=A(K,J)
      A(K,J)=A(L,J)
      A(L,J)=B
200     CONTINUE
300     IF(M.EQ.K) GOTO 500
      DO 400I=1,N
      B=A(I,K)
      A(I,K)=A(I,M)
       A(I,M)=B
400    CONTINUE
500     C=1./A(K,K)
      A(K,K)=C
      IF(K.EQ.N) GOTO 1000
      K1=K+1
      XMAX=ABS(A(K1,K1))
      L=K1
      M=K1
      DO 600I=K1,N
       A(I,K)=C*A(I,K)
600     CONTINUE
      DO 800I=K1,N
      B=A(I,K)
      DO 800J=K1,N
      A(I,J)=A(I,J)-B*A(K,J)
      Y=ABS(A(I,J))
      IF(XMAX.GE.Y) GOTO 800
      XMAX=Y
      L=I
      M=J
800    CONTINUE
1000  CONTINUE
      RETURN
      END
      SUBROUTINE SOLVE(F,A,K,N,ldim,IR,IC)
C   IT IS THE SECOND PART OF THE MATRIX INVERSION
      DIMENSION A(ldim,1),F(ldim,1),IR(1),IC(1)
      COMMON /CTMPG/ G(2000)
C
C
      IF (N.GT.2000) THEN
         write(6,*) 'Abort IN Subrtouine SOLVE: N>2000, N=',N
         call exitt
      ENDIF
C
      N1=N+1
      DO 1000KK=1,K
      DO 100I=1,N
      IRI=IR(I)
        G(I)=F(IRI,KK)
100     CONTINUE
      DO 400I=2,N
      I1=I-1
      B=G(I)
      DO 300J=1,I1
        B=B-A(I,J)*G(J)
300     CONTINUE
        G(I)=B
400     CONTINUE
      DO 700IT=1,N
      I=N1-IT
      I1=I+1
      B=G(I)
      IF(I.EQ.N) GOTO 701
      DO 600J=I1,N
        B=B-A(I,J)*G(J)
600     CONTINUE
701     G(I)=B*A(I,I)
700     CONTINUE
      DO 900I=1,N
      ICI=IC(I)
        F(ICI,KK)=G(I)
900     CONTINUE
1000    CONTINUE
      RETURN
      END
