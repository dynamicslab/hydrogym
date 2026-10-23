C------------------------------------------------------------------------------
C
C                          NEKTON 2.6  2/8/90
C
C			Copyright (C) 1990, by the 
C
C		Massachusetts Institute of Technology  and Nektonics, Inc.
C
C All Rights Reserved
C
C This program is a licenced product of MIT and Nektonics, Inc.,  and it is 
C not to be disclosed to others, copied, distributed, or displayed 
C without prior authorization.
C
C------------------------------------------------------------------------------
c-----------------------------------------------------------------------
      SUBROUTINE SORT(A,IND,N)
C
C     Use Heap Sort (p 233 Num. Rec.), 5/26/93 pff.
C
      DIMENSION A(1),IND(1)
C
      if (n.le.1) return
      DO 10 J=1,N
         IND(j)=j
   10 continue
C
      L=n/2+1
      ir=n
  100 CONTINUE
         IF (l.gt.1) THEN
            l=l-1
            indx=ind(l)
            q=a(indx)
         ELSE
            indx=ind(ir)
            q=a(indx)
            ind(ir)=ind(1)
            ir=ir-1
            if (ir.eq.1) then
               ind(1)=indx
               return
            endif
         ENDIF
         i=l
         j=l+l
  200    CONTINUE
         IF (J.le.IR) THEN
            IF (J.lt.IR) THEN
               IF ( A(IND(j)).lt.A(IND(j+1)) ) j=j+1
            ENDIF
            IF (q.lt.A(IND(j))) THEN
               IND(I)=IND(J)
               I=J
               J=J+J
            ELSE
               J=IR+1
            ENDIF
         GOTO 200
         ENDIF
         IND(I)=INDX
      GOTO 100
      END
c-----------------------------------------------------------------------
      SUBROUTINE SWAP(A,W,IND,N)
C
C     Use IND to sort array A   (p 233 Num. Rec.), 5/26/93 pff.
C
      DIMENSION A(1),W(1),IND(1)
C
      if (n.le.1) return
      DO 10 J=1,N
         W(j)=A(j)
   10 continue
C
      DO 20 J=1,N
         A(j)=W(ind(j))
   20 continue
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE ISWAP(A,W,IND,N)
C
C     Use IND to sort array A
C
      INTEGER A(1),W(1),IND(1)
C
      if (n.le.1) return
      DO 10 J=1,N
         W(j)=A(j)
   10 continue
C
      DO 20 J=1,N
         A(j)=W(ind(j))
   20 continue
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHSWAP(A,W,L,IND,N)
C
C     Use IND to sort array A
C
      CHARACTER*1 A(L,1),W(L,1)
      INTEGER IND(1)
C
      if (n.le.1) return
      DO 10 J=1,L*N
         W(j,1)=A(j,1)
   10 continue
C
      DO 20 J=1,N
         ij=IND(J)
         DO 20 k=1,l
            A(l,j)=W(l,ij)
   20    continue
   30 continue
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE LJUST(STRING)
C     left justify string
      CHARACTER*1 STRING(80)
C
      IF (STRING(1).NE.' ') RETURN
C
      DO 100 I=2,80
C
         IF (STRING(I).NE.' ') THEN
            DO 20 J=1,81-I
               IJ=I+J-1
               STRING(J)=STRING(IJ)
   20       CONTINUE
            DO 30 J=82-I,80
               STRING(J)=' '
   30       CONTINUE
            RETURN
         ENDIF
C
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CAPIT(LETTRS,N)
C     Capitalizes string of length n
      CHARACTER LETTRS(N)
C
      DO 5 I=1,N
         INT=ICHAR(LETTRS(I))
         IF(INT.GE.97 .AND. INT.LE.122) THEN
            INT=INT-32
            LETTRS(I)=CHAR(INT)
         ENDIF
5     CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE PARSE(LINES,ARGS,IA)
C       Capitalizes LINE and splits it into 5 ARGuments of 10 Characters.
C
      CHARACTER ARGS(10,5),COMAND*10,LINES(70)
      LOGICAL GAP
      INTEGER IAS(80),IES(80)
C
      IA=0
      IE=0
        GAP=.TRUE.
      DO 3 I=1,10
         DO 3 J=1,5
            ARGS(I,J)= ' '
3     CONTINUE
C       Capitalize LINE
      CALL CAPIT(LINES,70)
C
      DO 10 I=1,70
         IF(LINES(I).NE.' ' .AND. LINES(I).NE.',') THEN
            IF(GAP) THEN
C! Found the start of new argument
               IA=IA+1
               IAS(IA)=I
            ELSE
            ENDIF
            GAP = .FALSE.
         ELSE
            IF(.NOT.(GAP)) THEN
C!Found End of argument
               IE=IE+1
               IES(IE)=I-1
            ELSE
            ENDIF
            GAP = .TRUE.
         ENDIF
10    CONTINUE
C       COMMAND
C     Max 5 Arguments
      IF(IA.GT.5)IA=5
      DO 50 I=1,IA
         DO 50 IC=IAS(I),IES(I)
            ARGS(IC-IAS(I)+1,I)= LINES(IC)
50    CONTINUE
      DO 60 I=1,10
60            COMAND(I:I)=ARGS(I,1)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BLANK(STRING,N)
      CHARACTER*1 STRING(N)
      CHARACTER*1   BLNK
      DATA BLNK/' '/
C
      DO 100 I=1,N
         STRING(I)=BLNK
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CADD2 (A,B,C,N)
      DIMENSION A(1),B(1)
      DO 100 I = 1, N
 100     A(I) = B(I) + C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CADD (A,B,N)
      DIMENSION A(1)
      DO 100 I = 1, N
 100     A(I) = A(I) + B
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CMULT(A,B,N)
      DIMENSION A(1)
      DO 100 I = 1, N
 100     A(I) = B*A(I)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHCOPY(A,B,N)
      CHARACTER*1 A(1), B(1)
      DO 100 I = 1, N
 100     A(I) = B(I)
      RETURN
      END
C
c-----------------------------------------------------------------------
      FUNCTION LTRUNC(STRING,L)
      CHARACTER*1 STRING(L)
      CHARACTER*1   BLNK
      DATA BLNK/' '/
      DO 100 I=L,1,-1
         L1=I
         IF (STRING(I).NE.BLNK) GOTO 200
  100 CONTINUE
      L1=0
  200 CONTINUE
      LTRUNC=L1
      if (l1.eq.l) then
         write(6,*) 'LTRUNC: string:',l
         write(6,80) (string(k),k=1,l)
   80    format(80a1)
      endif
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE ROOTS(XVAL,NROOTS,FTARGT,F,N)
C
C     Find all values of I which yield F(I) close to FTARGT
C     and give the additional increment DI such that F(I+DI)+FTARGT,
C     where F(I+DI) is interpreted as the linear interpolation of F.
C
      DIMENSION XVAL(2,N),F(0:N)
C
      NROOTS=0
      DO 100 I=1,N
         F1=F(I-1)
         F2=F(I)
         DF1=FTARGT-F1
         DF2=F2-FTARGT
         IF (DF1*DF2.GE.0.0) THEN
            NROOTS=NROOTS+1
            DF=F2-F1
            IF (DF.EQ.0) THEN
               DI=0.5
            ELSE
               DI=(FTARGT-F1)/DF
            ENDIF
            XVAL(1,NROOTS)=FLOAT(I-1)
            XVAL(2,NROOTS)=DI
         ENDIF
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE VSQRT(A,N)
      DIMENSION  A(1)
      do i=1,n
         if (a(i).gt.0) a(i) = sqrt(a(i))
      enddo
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE VCNVERT(A,N)
      REAL*4 A(1)
      DO 100 I=1,N
         CALL CONVERT(A(I))
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CONVERT(T)
C
      CHARACTER*12 VC
      CHARACTER*1  V1(12)
      EQUIVALENCE (V1,VC)
      CHARACTER*1  CSIGN
C
      REAL*4 W
      CHARACTER*4  WC
      CHARACTER*1  W1(4)
      EQUIVALENCE (W1,W)
      EQUIVALENCE (W1,WC)
C
      CHARACTER*1 ALPH64(0:63)
      SAVE        ALPH64
      DATA        ALPH64 
     $  /'1','2','3','4','5','6','7','8','9','0'
     $   ,'a','b','c','d','e','f','g','h','i','j'
     $   ,'k','l','m','n','o','p','q','r','s','t'
     $   ,'u','v','w','x','y','z'
     $   ,'A','B','C','D','E','F','G','H','I','J'
     $   ,'K','L','M','N','O','P','Q','R','S','T'
     $   ,'U','V','W','X','Y','Z','+','-'/
C
C     Find out the usual decimal format for T
C
      WRITE(VC,10) T
   10 FORMAT(E12.5)
C
C     Begin converting the mantissa to base 64
C
      READ(VC,11) MANTIS
   11 FORMAT(3X,I5)
C
C     Sign?
C
      READ(VC,12) CSIGN
   12 FORMAT(A1)
      IF (CSIGN.EQ.'-') MANTIS=-MANTIS
      MANTIS=MANTIS+131072
C
C     ONES,TENS, HUNDREDS
C
      IONE=MOD(MANTIS,64)
      ITMP=MANTIS/64
      ITEN=MOD(ITMP,64)
      ITMP=ITMP/64
      IHUN=MOD(ITMP,64)
C
C     Exponent
C
      READ(VC,21) IEXP
   21 FORMAT(9X,I3)
C     We assume that the exponent is bounded by 31.
      IEXP=IEXP+31
C
C     Compute alpha equivalent
C
      W1(1)=ALPH64(IHUN)
      W1(2)=ALPH64(ITEN)
      W1(3)=ALPH64(IONE)
      W1(4)=ALPH64(IEXP)
C
C     Convert the input value
      T=W
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DELTMP
      INCLUDE  'basics.inc'
      character*80 command
C     Remove all tmp.* files
C
      CALL BLANK(COMMAND,80)
      IF(IFVMS)THEN
C        VMS
         IERR=0
      ELSE
C        Ultrix
         write(command,10) 
   10    format('rm tmp.*')
         CALL SYSTEM(command)
      ENDIF
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION ILMIN(A,N)
      integer A(N),temp
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = min(TEMP,A(I))
   10 CONTINUE
      ILmin=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION ILMAX(A,N)
      integer A(N),temp
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MAX(TEMP,A(I))
   10 CONTINUE
      ILMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION VLMAX(A,N)
      DIMENSION A(N)
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MAX(TEMP,A(I))
   10 CONTINUE
      VLMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION VLAMAX(A,N)
      DIMENSION A(N)
      TEMP = abs(A(1))
      DO 10 I=1,N
         TEMP = MAX(TEMP,abs(A(I)))
   10 CONTINUE
      VLAMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION IGLSUM(A,N)
      integer A(N)
      ITEMP = 0
      DO 10 I=1,N
         ITEMP = ITEMP+A(I)
   10 CONTINUE
      IGLSUM=ITEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION IGLMIN(A,N)
      INTEGER A(N),TEMP
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MIN(TEMP,A(I))
   10 CONTINUE
      IGLMIN=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION IGLMAX(A,N)
      INTEGER A(N),TEMP
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MAX(TEMP,A(I))
   10 CONTINUE
      IGLMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION IVLMAX(A,N)
      INTEGER A(N),TEMP
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MAX(TEMP,A(I))
   10 CONTINUE
      IVLMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION GLMAX(A,N)
      DIMENSION A(N)
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MAX(TEMP,A(I))
   10 CONTINUE
      GLMAX=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION VLMIN(A,N)
      DIMENSION A(N)
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MIN(TEMP,A(I))
   10 CONTINUE
      VLMIN=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION GLMIN(A,N)
      DIMENSION A(N)
      TEMP = A(1)
      DO 10 I=1,N
         TEMP = MIN(TEMP,A(I))
   10 CONTINUE
      GLMIN=TEMP
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION MOD1(I,N)
C
C     Yields MOD(I,N) with the exception that if I=K*N, result is N.
C
      MOD1 = 1
      IF (N.EQ.0) THEN
         CALL PRS(
     $  'WARNING:  Attempt to take MOD(I,0) in FUNCTION MOD1.$')
         RETURN
      ENDIF
      II = I+N-1
      MOD1 = MOD(II,N)+1
      RETURN
      END
c-----------------------------------------------------------------------
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*80 S1,S2
C
      N1=80-L2+1
      INDX1=0
      IF (N1.LT.1) RETURN
C
      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            RETURN
         ENDIF
  100 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      INTEGER FUNCTION NINDX1(S1,S2,L2)
C
C     Return index of first character Not equal to S2
C
      CHARACTER*80 S1,S2
C
      N1=80-L2+1
      NINDX1=0
      IF (N1.LT.1) RETURN
C
      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).NE.S2(1:L2)) THEN
            NINDX1=I
            RETURN
         ENDIF
  100 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine isort(a,ind,n)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer ind(1)
      integer    a(1)
      integer    aa
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
            aa  = a  (l)
            ii  = ind(l)
         else
                 aa =   a(ir)
                 ii = ind(ir)
              a(ir) =   a( 1)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
                 a(1) = aa
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if ( a(j).lt.a(j+1) ) j=j+1
            endif
            if (aa.lt.a(j)) then
                 a(i) = a(j)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
           a(i) = aa
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      subroutine sortit(a,ind,n)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer ind(1)
      real    a(1)
      real    aa
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
            aa  = a  (l)
            ii  = ind(l)
         else
                 aa =   a(ir)
                 ii = ind(ir)
              a(ir) =   a( 1)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
                 a(1) = aa
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if ( a(j).lt.a(j+1) ) j=j+1
            endif
            if (aa.lt.a(j)) then
                 a(i) = a(j)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
           a(i) = aa
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      subroutine add3s12(x,y,cy,z,cz,n)
      real x(1),y(1),z(1),cy,cz
c
      do i=1,n
         x(i) = cy*y(i) + cz*z(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add3s2(x,y,z,c,n)
      real x(1),y(1),z(1),c
c
      do i=1,n
         x(i) = y(i) + c*z(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cmult2(x,y,c,n)
      real x(1),y(1),c
c
      do i=1,n
         x(i) = c*y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2s1(x,y,c,n)
      real x(1),y(1),c
c
      do i=1,n
         x(i) = c*x(i) + y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2s2(x,y,c,n)
      real x(1),y(1),c
c
      do i=1,n
         x(i) = x(i) + c*y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      function glsc2(x,y,n)
C
C     Perform inner-product
C
      real x(1), y(1)
c
      sum = 0.0
      do i=1,n
         sum=sum+x(i)*y(i)
      enddo
c
      glsc2 = sum
c
      return
      end
c-----------------------------------------------------------------------
      function vlsum(x,n)
      real x(1)
      sum = 0.0
      do i=1,n
         sum=sum+x(i)
      enddo
      vlsum = sum
      return
      end
c-----------------------------------------------------------------------
      function glsc1(x,n)
C     Perform inner-product
      real x(1)
c
      sum = 0.0
      do i=1,n
         sum=sum+x(i)*x(i)
      enddo
      glsc1 = sum
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE IRANK(A,IND,N)
C
C     Use Heap Sort (p 233 Num. Rec.)
C
      INTEGER A(1),IND(1)
      INTEGER Q
c
      if (n.le.1) return
      DO 10 J=1,N
         IND(j)=j
   10 continue
C
      L=n/2+1
      ir=n
  100 CONTINUE
         IF (l.gt.1) THEN
            l=l-1
            indx=ind(l)
            q=a(indx)
         ELSE
            indx=ind(ir)
            q=a(indx)
            ind(ir)=ind(1)
            ir=ir-1
            if (ir.eq.1) then
               ind(1)=indx
               return
            endif
         ENDIF
         i=l
         j=l+l
  200    CONTINUE
         IF (J.le.IR) THEN
            IF (J.lt.IR) THEN
               IF ( A(IND(j)).lt.A(IND(j+1)) ) j=j+1
            ENDIF
            IF (q.lt.A(IND(j))) THEN
               IND(I)=IND(J)
               I=J
               J=J+J
            ELSE
               J=IR+1
            ENDIF
         GOTO 200
         ENDIF
         IND(I)=INDX
      GOTO 100
      END
c-----------------------------------------------------------------------
      SUBROUTINE RANK(A,IND,N)
C
C     Use Heap Sort (p 233 Num. Rec.), 5/26/93 pff.
C
      DIMENSION A(1),IND(1)
C
      if (n.le.1) return
      DO 10 J=1,N
         IND(j)=j
   10 continue
C
      L=n/2+1
      ir=n
  100 CONTINUE
         IF (l.gt.1) THEN
            l=l-1
            indx=ind(l)
            q=a(indx)
         ELSE
            indx=ind(ir)
            q=a(indx)
            ind(ir)=ind(1)
            ir=ir-1
            if (ir.eq.1) then
               ind(1)=indx
               return
            endif
         ENDIF
         i=l
         j=l+l
  200    CONTINUE
         IF (J.le.IR) THEN
            IF (J.lt.IR) THEN
               IF ( A(IND(j)).lt.A(IND(j+1)) ) j=j+1
            ENDIF
            IF (q.lt.A(IND(j))) THEN
               IND(I)=IND(J)
               I=J
               J=J+J
            ELSE
               J=IR+1
            ENDIF
         GOTO 200
         ENDIF
         IND(I)=INDX
      GOTO 100
      END
c-----------------------------------------------------------------------
      SUBROUTINE COL3(A,B,C,N)
      REAL A(N),B(N),C(N)
      DO 100 I=1,N
         A(I)=B(I)*C(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE SUBCOL3(A,B,C,N)
      REAL A(N),B(N),C(N)
      DO 100 I=1,N
         A(I)=A(I)-B(I)*C(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE ADDCOL3(A,B,C,N)
      REAL A(N),B(N),C(N)
      DO 100 I=1,N
         A(I)=A(I)+B(I)*C(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE ADDCOL4(A,B,C,D,N)
      REAL A(N),B(N),C(N),D(N)
      DO 100 I=1,N
         A(I)=A(I)+B(I)*C(I)*D(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE SUBCOL4(A,B,C,D,N)
      REAL A(N),B(N),C(N),D(N)
      DO 100 I=1,N
         A(I)=A(I)-B(I)*C(I)*D(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE IZERO (IA,N)
      DIMENSION  IA(1)
      DO 100 I = 1, N
 100     IA(I) = 0
      RETURN
      END
      SUBROUTINE RZERO (A,N)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      RETURN
      END
      SUBROUTINE VDOT(DOT,U1,U2,U3,V1,V2,V3,N)
C
C     Compute an elemental based Cartesian vector dot product.
C
      DIMENSION DOT(N)
      DIMENSION U1(N),U2(N),U3(N)
      DIMENSION V1(N),V2(N),V3(N)
C
      DO 100 I=1,N
         DOT(I) = U1(I)*V1(I) + U2(I)*V2(I) + U3(I)*V3(I)
  100 CONTINUE
      RETURN
      END
      SUBROUTINE INVCOL2 (A,B,N)
      REAL A(1),B(1)
      DO 100 I=1,N
         A(I)=A(I)/B(I)
 100  CONTINUE
      RETURN
      END
      SUBROUTINE COPY(A,B,N)
      REAL A(N),B(N)
      DO 10 I=1,N
         A(I)=B(I)
   10 CONTINUE
      RETURN
      END
      SUBROUTINE ICOPY(A,B,N)
      INTEGER A(1),B(1)
      DO 10 I=1,N
         A(I)=B(I)
   10 CONTINUE
      RETURN
      END
      SUBROUTINE SUB2(A,B,N)
      REAL A(1),B(1)
      DO 10 I=1,N
         A(I)=A(I)-B(I)
   10 CONTINUE
      RETURN
      END
      SUBROUTINE SUB3(C,A,B,N)
      REAL C(1),A(1),B(1)
      DO 10 I=1,N
         C(I)=A(I)-B(I)
   10 CONTINUE
      RETURN
      END
      SUBROUTINE ADD3(C,A,B,N)
      REAL C(1),A(1),B(1)
      DO 10 I=1,N
         C(I)=A(I)+B(I)
   10 CONTINUE
      RETURN
      END
      SUBROUTINE ADD2(A,B,N)
      REAL A(N),B(N)
      DO 1 I=1,N
         A(I)=A(I)+B(I)
1     CONTINUE
      RETURN
      END
      FUNCTION ALMIN(A,N)
C
C     Compute the local minimum of quantity A
C
      REAL A(N)
C
C  Compute the local sum
C
      TMIN = A(1)
      DO 100 I=1,N
         TMIN=AMIN1(TMIN,A(I))
  100 CONTINUE
      ALMIN = TMIN
      RETURN
      END
      FUNCTION ALMAX(A,N)
C
C     Compute the local maximum of quantity A
C
      REAL A(N)
C
C  Compute the local sum
C
      TMAX = A(1)
      DO 100 I=1,N
         TMAX=AMAX1(TMAX,A(I))
  100 CONTINUE
      ALMAX = TMAX
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine iint(in,n)
      integer in(1)
      do i=1,n
         in(i) = i
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine irank_vec(ind,nn,a,m,n,key,nkey,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is destroyed
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),aa(m)
      logical iftuple_ianeb,a_ne_b
c
      if (m.eq.1) then
c
         write(6,*) 
     $        'WARNING: For single key, not clear that rank is unique!'
         call irank(a,ind,n)
         return
      endif
c
c
      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)
c
c     Find unique a's
c
      nn=1
c
      call icopy(aa,a,m)
      a(1,1) = nn
      a(2,1)=ind(1)
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
c        write(6,1) nn,i,a_ne_b,aa,(a(k,i),k=1,nk)
c  1     format(2i7,1x,l4,1x,12i6)
         if (a_ne_b) then
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         a(1,i) = nn
         a(2,i) = ind(i)
      enddo
c
c     Set ind() to rank
c
      do i=1,n
         iold=a(2,i)
         ind(iold) = a(1,i)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cfill(a,b,n)
      real a(1),b
      DO 100 I = 1, N
 100     A(I) = B
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine ifill(a,b,n)
      integer a(1),b
      DO 100 I = 1, N
 100     A(I) = B
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine semhat(a,b,c,d,dp,jp,z,n,w)
c
c     Generate matrices for single element, 1D operators:
c
c        a  = Laplacian
c        b  = diagonal mass matrix
c        c  = convection operator b*d
c        d  = derivative matrix
c        dp = derivative matrix, mapping from pressure nodes to velocity
c        jp = interpolation matrix, mapping from pressure nodes to velocity
c        z  = GLL points
c        n  = polynomial degree (velocity space)
c        w  = work array of size 2*n+2
c
c     Currently, this is set up for pressure nodes on the interior GLL pts.
c
c
      real a(0:n,0:n),b(0:n),c(0:n,0:n),d(0:n,0:n),z(0:n)
      real dp(0:n,1:n-1),jp(0:n,1:n-1)
      real w(0:1)
c
      np = n+1
      nm = n-1
      n2 = n-2
c
      call zwgll (z,b,np)
c
      do i=0,n
         call fd_weights_full(z(i),z,n,1,w)
         do j=0,n
            d(i,j) = w(j+np)                   !  Derivative matrix
         enddo
      enddo
c
      do i=0,n
         call fd_weights_full(z(i),z(1),n2,1,w(1))
         do j=1,nm
            jp(i,j) = w(j   )                  !  Interpolation matrix
            dp(i,j) = w(j+nm)                  !  Derivative    matrix
         enddo
      enddo
c
      call rzero(a,np*np)
      do j=0,n
      do i=0,n
         do k=0,n
            a(i,j) = a(i,j) + d(k,i)*b(k)*d(k,j)
         enddo
         c(i,j) = b(i)*d(i,j)
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine generalev(a,b,lam,n,w)
c
c     Solve the generalized eigenvalue problem  A x = lam B x
c
c     A -- symm.
c     B -- symm., pos. definite
c
      integer wdsize
      save    wdsize
      data    wdsize/0/
c
      real a(n,n),b(n,n),lam(n),w(n,n)
c
c
c     First call: determine word size for dgemm/sgemm discrimination, below.
      if (wdsize.eq.0) then
         one = 1.0
         eps = 1.e-12
         wdsize = 8
         if (one+eps.eq.1.0) wdsize = 4
      endif
c     write(6,*) 'in generalev, =',n,' wdsize',wdsize
c
c
c     call outmat(a,n,n,'Agn')
c     call outmat(b,n,n,'Bgn')
c
      lwork = max(n*n,3*n-1)
      call ssygv(1,'V','U',n,a,n,b,n,lam,w,lwork,info)
c
c
      if (info.ne.0) then
         ninf = n-info
         write(6,*) 'Error in generalev, info=',info,n,ninf
c
         call outmat(a,n,n,'Agn')
         call outmat(b,n,n,'Bgn')
c
         call exitt
      endif
c
c     call outmat(a,n,n,'Agn')
c
      return
      end
c-----------------------------------------------------------------------
