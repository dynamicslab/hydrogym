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
C
C
C     I/O Subroutine Library for reading & writing to screen device
C     There are two layers of I/O routines.  This upper layer has
C     about 10 routines for handling different formats.  Each of them
C     has one call to the lower layer.
C
C     Output Subroutine Convention:
C     PRS            Prints String                  CALL PRS   (S)
C     PRI            Prints Integer                 CALL PRI   (I)
C     PRR            Prints Real                    CALL PRR   (R)
C     PRRR           Prints Real,Real               CALL PRRR  (R,R)
C     PRII           Prints Integer,Integer         CALL PRII  (I,I)
C     PRIS           Prints Integer,String          CALL PRSI  (I,S)
C     PRSI           Prints String, Integer         CALL PRSI  (S,I)
C     PRSR           Prints String, Real            CALL PRSR  (S,R)
C     PRSIR          Prints String, Integer, Real   CALL PRSIR (S,I,R)
C     PRSIS          Prints String, Integer, String CALL PRSIS (S,I,S)
C     PRSRS          Prints String, Real,    String CALL PRSRS (S,R,S)
C
C     All strings must have $ in them to signal termination
C
C
C     Input Subroutine Convention:
C     RES           Reads String of Length N       CALL RES   (S,N)
C     REI           Reads Integer                  CALL REI   (I)
C     RER           Reads Real                     CALL RER   (R)
C     RERR          Reads Real, Real               CALL RERR  (R,R)
C     RERRR         Reads 3 Real                   CALL RERRR (R,R,R)
C     RERRRR        Reads 4 Real                   CALL RERRRR(R,R,R,R)
C
C
C     The lower layer (not seen by the applications code) contains two
C     routines, GETS and PUTS, which perform I/O with the device.
C
C     Driver Subroutine Calling Convention:
C     GETS         Reads  String of 80 chars        CALL GETS(S)
C     PUTS         Prints String of  N chars        CALL PUTS(S,N)
      SUBROUTINE PRS(S)
      CHARACTER S(*)
      CHARACTER*1 S1(80)
C
      DO 1 I=1,81
         IF(S(I).EQ.'$')THEN
            NCHARS=I-1
            GO TO 2
         ENDIF
 1    CONTINUE

c     CALL PUTSOLD('I/O Error: No String Terminator sent to PRS',43)
      write(6,*) 'I/O Error: No String Terminator sent to PRS'
      write(66,*) 'I/O Error: No String Terminator sent to PRS'

      nchars = 80
      
      CALL CHCOPY (S1,S,NCHARS)
      WRITE(66,3) (S1(J),J=1,NCHARS)

 2    CONTINUE
c     CALL PUTSOLD  (S   ,NCHARS)
      CALL CHCOPY(S1,S,NCHARS)
      WRITE(6,3) (S1(J),J=1,NCHARS)
c
 3    FORMAT(80A1)
      RETURN
      END
C
      SUBROUTINE PRI(I)
      CHARACTER S*80
C
      S=' '
      WRITE(S,'(I8)',ERR=1)I
      S(9:9)='$'
      CALL PRS(S)
 1    RETURN
      END
C
      SUBROUTINE PRR(R)
      CHARACTER S*80
C
      S=' '
      WRITE(S,'(G14.6)',ERR=1)R
      S(15:15)='$'
      CALL PRS(S)
 1    RETURN
      END
C
C
      SUBROUTINE PRII(I1,I2)
      CHARACTER S*80
C
      S=' '
      WRITE(S,'(2I8)',ERR=1)I1,I2
      S(17:17)='$'
      CALL PRS(S)
 1    RETURN
      END
C
      SUBROUTINE PRRRR(R1,R2,R3)
      CHARACTER S*80
C
      S=' '
      WRITE(S,'(3G18.9)',ERR=1)R1,R2,R3
      S(55:55)='$'
      CALL PRS(S)
 1    RETURN
      END
C
      SUBROUTINE PRRR(R1,R2)
      CHARACTER S*80
C
      S=' '
      WRITE(S,'(2G14.6)',ERR=1)R1,R2
      S(29:29)='$'
      CALL PRS(S)
 1    RETURN
      END
C
      SUBROUTINE PRSI(S,I)
      CHARACTER S(*),SS*80
C
      DO 1 J=1,70
         IF(S(J).EQ.'$')THEN
            NC=J-1
            GO TO 2
         ENDIF
         SS(J:J)=S(J)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSI',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
C
      WRITE(SS(NC+1:NC+8),'(I8)',ERR=13)I
      SS(NC+9:NC+9)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSI',39)
      RETURN
      END
C
      SUBROUTINE PRIS(I,S)
      CHARACTER S(*),SS*80
C
      call blank(ss,80)
      WRITE(SS(1:8),'(I8)',ERR=13)I
      DO 1 J=1,70
         IF(S(J).EQ.'$')THEN
            NC=J-1
            GO TO 2
         ENDIF
         SS(J+8:J+8)=S(J)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSI',44)
      CALL PUTSOLD(S,70)
      RETURN
 2    CONTINUE
C
      SS(NC+9:NC+9)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRIS',39)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSRRR(S,r1,r2,r3)
      CHARACTER S(*),SS*80
C
      DO 1 I=1,60
         IF(S(I).EQ.'$')THEN
            NC=I-1
            GO TO 2
         ENDIF
         SS(I:I)=S(I)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSRR',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+42),'(1p3e14.6)',ERR=13)r1,r2,r3
      SS(NC+43:NC+43)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSRR',39)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSRR(S,r1,r2)
      CHARACTER S(*),SS*80
C
      DO 1 I=1,60
         IF(S(I).EQ.'$')THEN
            NC=I-1
            GO TO 2
         ENDIF
         SS(I:I)=S(I)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSRR',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+28),'(1p2e14.6)',ERR=13)r1,r2
      SS(NC+29:NC+29)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSRR',39)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSR(S,R)
      CHARACTER S(*),SS*80
C
      DO 1 I=1,60
         IF(S(I).EQ.'$')THEN
            NC=I-1
            GO TO 2
         ENDIF
         SS(I:I)=S(I)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSR',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+14),'(G14.6)',ERR=13)R
      SS(NC+15:NC+15)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSR',39)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSII(S,I,J)
      CHARACTER S(*),SS*80
C
      DO 1 k=1,60
         IF(S(k).EQ.'$')THEN
            NC=k-1
            GO TO 2
         ENDIF
         SS(k:k)=S(k)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSii',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+20),'(2I10)',ERR=13)I,J
      SS(NC+21:NC+21)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSii',39)
      RETURN
      END
C
      SUBROUTINE PRSIR(S,I,R)
      CHARACTER S(*),SS*80
C
      DO 1 k=1,60
         IF(S(k).EQ.'$')THEN
            NC=k-1
            GO TO 2
         ENDIF
         SS(k:k)=S(k)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSir',44)
      CALL PUTSOLD(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+20),'(I6,G14.6)',ERR=13)I,R
      SS(NC+21:NC+21)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSir',39)
      RETURN
      END
C
      SUBROUTINE PRSIS(S1,I,S2)
      CHARACTER S1(*),S2(*),SS*80
C
      SS = ' '
      DO 1 J=1,70
         IF(S1(J).EQ.'$')THEN
            NC=J-1
            GO TO 2
         ENDIF
         SS(J:J)=S1(J)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSIS',45)
      CALL PUTSOLD(SS,70)
      RETURN
 2    CONTINUE
      NC = NC + 1
      WRITE(SS(NC+1:NC+8),'(I8)',ERR=13)I
C
      ISTART = NC+10
      IS2=0
      DO 11 J=ISTART,80
         IS2=IS2+1
         IF(S2(IS2).EQ.'$')THEN
            NC=J-1
            GO TO 12
         ENDIF
         SS(J:J)=S2(IS2)
 11   CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSIS',45)
      CALL PUTSOLD(SS,70)
      RETURN
 12   CONTINUE
C
      SS(NC+1:NC+1)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSIS',40)
      RETURN
      END
C
      SUBROUTINE PRSRS(S1,R,S2)
      CHARACTER S1(*),S2(*),SS*80
C
      SS = ' '
      DO 1 I=1,60
         IF(S1(I).EQ.'$')THEN
            NC=I-1
            GO TO 2
         ENDIF
         SS(I:I)=S1(I)
 1    CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSRS',45)
      CALL PUTSOLD(SS,70)
      RETURN
 2    CONTINUE
C
C     Pad a few blanks
      NC=NC+1
      WRITE(SS(NC+1:NC+14),'(G14.6)',ERR=13)R
C
      ISTART = NC + 16
      IS2 = 0
      DO 11 I=ISTART,80
         IS2 = IS2 + 1
         IF(S2(IS2).EQ.'$')THEN
            NC=I-1
            GO TO 12
         ENDIF
         SS(I:I)=S2(IS2)
 11   CONTINUE
      CALL PUTSOLD('I/O Error: No String Terminator sent to PRSRS',45)
      CALL PUTSOLD(SS,70)
      RETURN
 12   CONTINUE
      SS(NC+1:NC+1)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTSOLD('I/O Error: cant write to string in PRSRS',40)
      RETURN
      END
C
C
C
C                                          Input Routine Section
C
C
      SUBROUTINE RES(S1,N)
      include 'devices.inc'
      CHARACTER S1(*),S(80)
C
C     All text input comes thru here
C
C
      IF (.NOT.IFDEMO) THEN
         CALL GETSOLD(S)
      ELSE
         READ(55,80) S
   80    FORMAT(80A1)
      ENDIF
C
c     CALL GETSOLD(S)
      DO 1 I=1,N
         S1(I) = S(I)
 1    CONTINUE
      RETURN
      END
C
      SUBROUTINE REiii(i,j,k)
C     Read Integer
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13) i,j,k
      write (6,*) i,j,k
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter 3 integer Values$')
      GO TO 1
      END
C
      SUBROUTINE REIi(I,j)
C     Read Integer
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)I,j
      write (6,*) i,j
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter 2 integer Values$')
      GO TO 1
      END
c
      SUBROUTINE REI(I)
C     Read Integer
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)I
      write (6,*) i
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter Integer Value$')
      GO TO 1
      END
C
      SUBROUTINE RER(R1)
C     Read Real
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)R1
      write (6,*) r1
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter Real    Value$')
      GO TO 1
      END
C
      SUBROUTINE RERR(R1,R2)
C     Read Real
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)R1,R2
      write (6,*) r1,r2
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter 2 Real Values$')
      GO TO 1
      END
C
      SUBROUTINE RERRR(R1,R2,R3)
C     Read Real
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)R1,R2,R3
      write (6,*) r1,r2,r3
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter 3 Real Values$')
      GO TO 1
      END
C
      SUBROUTINE RERRRR(R1,R2,R3,R4)
C     Read Real
      CHARACTER*80 S
C
 1    CALL RES(S,80)
      REWIND(13)
      WRITE (13,'(A80)')S
      REWIND(13)
      READ  (13,*,ERR=13,END=13)R1,R2,R3,R4
      write (6,*) r1,r2,r3,r4
      REWIND(13)
      RETURN
 13   CALL PRS('Error reading input.  Enter 4 Real Values$')
      GO TO 1
      END
C
      SUBROUTINE PUTSOLD(S,NCHARS)
C     PUTSOLD is the one device-dependent output subroutine.
C     It Goes in Tekplot.f (Or Xinterface.f)  This is the Tek version
C     It displays a string on the output display device
      CHARACTER S(*)
      WRITE(6,'(1X,80A1)',ERR=1)(S(I),I=1,NCHARS)
 1    RETURN
      END
C
      SUBROUTINE GETSOLD(S)
C     GETSOLD is the one device-dependent input subroutine.
C     It Goes in Tekplot.f (Or Xinterface.f)  This is the Tek version
C     It returns an 80 character string entered from the input device
C
      CHARACTER S*80
C
      S=' '
      READ(5,'(A80)',ERR=1,END=1)S
 1    RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSIII(S,I1,i2,i3)
      CHARACTER S(*),SS*80
C
      DO 1 k=1,60
         IF(S(k).EQ.'$')THEN
            NC=k-1
            GO TO 2
         ENDIF
         SS(k:k)=S(k)
 1    CONTINUE
      CALL PUTS('I/O Error: No String Terminator sent to PRSiii',44)
      CALL PUTS(S,80)
      RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+30),'(3I10)',ERR=13)i1,i2,i3
      SS(NC+31:NC+31)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTS('I/O Error: cant write to string in PRSii',39)
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine prsiv(s,i1,i2,i3,i4)
      CHARACTER S(*),SS*80
C
c      write(6,*) (s(k),k=1,40)
c      write(6,*) i1,i2,i3,i4
c      NC=20
      DO 1 k=1,60
         IF(S(k).EQ.'$')THEN
            NC=k-1
            GO TO 2
         ENDIF
         SS(k:k)=S(k)
 1    CONTINUE
      CALL PUTS('I/O Error: No String Terminator sent to PRSiv ',44)
      CALL PUTS(S,80)
c     RETURN
 2    CONTINUE
      WRITE(SS(NC+1:NC+32),'(4I8)',ERR=13)i1,i2,i3,i4
      SS(NC+33:NC+33)='$'
      CALL PRS(SS)
      RETURN
 13   CALL PUTS('I/O Error: cant write to string in PRSiv ',39)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRIII(I1,I2,I3)
      CHARACTER S*80
      S=' '
      WRITE(S,'(3I8)',ERR=1)I1,I2,I3
      S(25:25)='$'
      CALL PRS(S)
 1    RETURN
      END
c-----------------------------------------------------------------------
      subroutine setgraph(ifgraf)
      logical ifgraf
      ifgraf = .false.
      return
      end
c-----------------------------------------------------------------------
