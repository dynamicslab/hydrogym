      FUNCTION REVERT(T)
C
      REAL*4 REVERT
      CHARACTER*4 T
      CHARACTER*12 VC
      CHARACTER*1  V1(12)
      EQUIVALENCE (V1,VC)
C
      INTEGER IC
      CHARACTER*1  C(4)
      EQUIVALENCE (IC,C)
C
      CHARACTER*4  WC
      CHARACTER*1  W1(4)
      EQUIVALENCE (W1,WC)
C
      CHARACTER*1 ALPSRT(4,0:63)
      CHARACTER*4 ALPSR4(0:63)
      SAVE        ALPH64,ALPSRT
      INTEGER INDEX(0:63),INTALP(0:63),LOG64
      SAVE    INDEX,INTALP,LOG64
      EQUIVALENCE (INTALP,ALPSRT)
      EQUIVALENCE (INTALP,ALPSR4)
C
      CHARACTER*1 ALPH64(0:63)
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
      DATA        ALPH64 
     $  /'1','2','3','4','5','6','7','8','9','0'
     $   ,'a','b','c','d','e','f','g','h','i','j'
     $   ,'k','l','m','n','o','p','q','r','s','t'
     $   ,'u','v','w','x','y','z'
     $   ,'A','B','C','D','E','F','G','H','I','J'
     $   ,'K','L','M','N','O','P','Q','R','S','T'
     $   ,'U','V','W','X','Y','Z','+','-'/
C
C     On first call, set up sorted list of Alpha values so
C     that we can use binary chop to evaluate the integer
C     values.
C
      IF (ICALLD.EQ.0) THEN
         ICALLD=1
         CALL BLANK(ALPSRT,256)
         DO 10 I=0,63
            ALPSRT(4,I)=ALPH64(I)
   10    CONTINUE
         CALL IRANK(INTALP,INDEX,64)
         DO 20 I=0,63
   20    INDEX(I)=INDEX(I)-1
         LOG64=7
      ENDIF
C
C     Copy T to W
C
      WC=T
      CALL BLANK(C,4)
C
C     Begin divide and conquer search
C
      ILO=0
      IHI=63
      C(4)=W1(1)
      DO 100 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IHUN=INDEX(I)
c           write(6,103) ihun,i,ilog,ilo,ihi
  103 format(' ihun',6i9)
            GOTO 101
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  100 CONTINUE
  101 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(2)
      DO 200 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            ITEN=INDEX(I)
c           write(6,203) iten,i,ilog,ilo,ihi
  203 format(' iten',6i9)
            GOTO 201
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  200 CONTINUE
  201 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(3)
      DO 300 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IONE=INDEX(I)
c           write(6,303) ione,i,ilog,ilo,ihi
  303 format(' ione',6i9)
            GOTO 301
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  300 CONTINUE
  301 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(4)
      DO 400 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IEXP=INDEX(I)
c           write(6,403) iexp,i,ilog,ilo,ihi
  403 format(' iexp',6i9)
            GOTO 401
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  400 CONTINUE
  401 CONTINUE
C
c     MANTIS=4096*IHUN+64*ITEN+IONE
      MANTIS=4096*IHUN+64*ITEN+IONE-131072
      IEXP  =IEXP-31
      IF (MANTIS.LT.0) THEN
         MANTIS=ABS(MANTIS)
         WRITE(VC,1001) MANTIS,IEXP
      ELSE
         WRITE(VC,1002) MANTIS,IEXP
      ENDIF
 1001 FORMAT('-0.',I5.5,'E',I3.2)
 1002 FORMAT(' 0.',I5.5,'E',I3.2)
C
      READ(VC,2001) W
 2001 FORMAT(E12.5)
 2002 FORMAT(2X,A12)
C
      REVERT=W
      RETURN
      END
