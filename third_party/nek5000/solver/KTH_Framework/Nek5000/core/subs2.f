      SUBROUTINE SETAXW1 (IFAXWG)
C
      INCLUDE 'SIZE'
      INCLUDE 'WZ'
C
      LOGICAL IFAXWG
C
      IF (IFAXWG) THEN
         CALL COPY (W3M1,W2AM1,lx1*ly1)
      ELSE
         CALL COPY (W3M1,W2CM1,lx1*ly1)
      ENDIF
C
      RETURN
      END
      SUBROUTINE SETAXW2 (IFAXWG)
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'WZ'
C
      LOGICAL IFAXWG
C
      IF (IFAXWG) THEN
         CALL COPY (W3M2,W2AM2,lx2*ly2)
      ELSE
         CALL COPY (W3M2,W2CM2,lx2*ly2)
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE STNRINV
C
C     Calculate 2nd and 3rd strain-rate invariants
C
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      common /screv/ ei2(lx1,ly1,lz1,lelt)
     $             , ei3(lx1,ly1,lz1,lelt)
      common /ctmp1/ exx(lx1,ly1,lz1,lelt)
     $             , exy(lx1,ly1,lz1,lelt)
     $             , eyy(lx1,ly1,lz1,lelt)
     $             , ezz(lx1,ly1,lz1,lelt)
      common /ctmp0/ exz(lx1,ly1,lz1,lelt)
     $             , eyz(lx1,ly1,lz1,lelt)
c
      NTOT1  = lx1*ly1*lz1*NELV
      CALL RZERO (EI2,NTOT1)
      CALL RZERO (EI3,NTOT1)
      IF (ISTEP.EQ.0) RETURN
C
      MATMOD = 0
      CALL STNRATE (VX,VY,VZ,NELV,MATMOD)
C
      IF (ldim.EQ.2) THEN
          CALL COL3    (EI2,EXX,EYY,NTOT1)
          CALL SUBCOL3 (EI2,EXY,EXY,NTOT1)
          CALL RZERO   (EI3,NTOT1)
      ELSE
          CONST = 2.0
          CALL COL4    (EI3,EXX,EYY,EZZ,NTOT1)          
          CALL COL4    (EI2,EXY,EXZ,EYZ,NTOT1)
          CALL ADD2S2  (EI3,EI2,CONST,NTOT1)
          CALL SUBCOL4 (EI3,EXX,EYZ,EYZ,NTOT1)
          CALL SUBCOL4 (EI3,EYY,EXZ,EXZ,NTOT1)
          CALL SUBCOL4 (EI3,EZZ,EXY,EXY,NTOT1)
          CALL COL3    (EI2,EXX,EYY,NTOT1)
          CALL ADDCOL3 (EI2,EXX,EZZ,NTOT1)
          CALL ADDCOL3 (EI2,EYY,EZZ,NTOT1)
          CALL SUBCOL3 (EI2,EXY,EXY,NTOT1)
          CALL SUBCOL3 (EI2,EXZ,EXZ,NTOT1)
          CALL SUBCOL3 (EI2,EYZ,EYZ,NTOT1)
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE OPDOT (DP,A1,A2,A3,B1,B2,B3,N)
C
      INCLUDE 'SIZE'
C
      DIMENSION DP(LX1,LY1,LZ1,1)
     $        , A1(LX1,LY1,LZ1,1)
     $        , A2(LX1,LY1,LZ1,1)
     $        , A3(LX1,LY1,LZ1,1)
     $        , B1(LX1,LY1,LZ1,1)
     $        , B2(LX1,LY1,LZ1,1)
     $        , B3(LX1,LY1,LZ1,1)
C
      IF (ldim.EQ.2) THEN
         CALL VDOT2 (DP,A1,A2,B1,B2,N)
      ELSE
         CALL VDOT3 (DP,A1,A2,A3,B1,B2,B3,N)
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE OPADDS (A1,A2,A3,B1,B2,B3,CONST,N,ISC)
C
      INCLUDE 'SIZE'
C
      DIMENSION A1(LX1,LY1,LZ1,1)
     $        , A2(LX1,LY1,LZ1,1)
     $        , A3(LX1,LY1,LZ1,1)
     $        , B1(LX1,LY1,LZ1,1)
     $        , B2(LX1,LY1,LZ1,1)
     $        , B3(LX1,LY1,LZ1,1)
C
      IF (ISC.EQ.1) THEN
         CALL ADD2S1 (A1,B1,CONST,N)
         CALL ADD2S1 (A2,B2,CONST,N)
         IF (ldim.EQ.3) CALL ADD2S1 (A3,B3,CONST,N)
      ELSEIF (ISC.EQ.2) THEN
         CALL ADD2S2 (A1,B1,CONST,N)
         CALL ADD2S2 (A2,B2,CONST,N)
         IF (ldim.EQ.3) CALL ADD2S2 (A3,B3,CONST,N)
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FACEXS (A,B,IFACE1,IOP)
C
C     IOP = 0
C     Extract scalar A from B on face IFACE1.
C
C     IOP = 1
C     Extract scalar B from A on face IFACE1.
C
C     A has the (NX,NY,NFACE) data structure
C     B has the (NX,NY,NZ)    data structure
C     IFACE1 is in the preprocessor notation 
C     IFACE  is the dssum notation.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
      DIMENSION A(LX1,LY1),B(LX1,LY1,LZ1)
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      IF (IOP.EQ.0) THEN
         DO 100 J2=JS2,JF2,JSKIP2
         DO 100 J1=JS1,JF1,JSKIP1
            I = I+1
            A(I,1) = B(J1,J2,1)
  100    CONTINUE
      ELSE
         DO 150 J2=JS2,JF2,JSKIP2
         DO 150 J1=JS1,JF1,JSKIP1
            I = I+1
            B(J1,J2,1) = A(I,1)
  150    CONTINUE
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FACEXV (A1,A2,A3,B1,B2,B3,IFACE1,IOP)
C
C     IOP = 0
C     Extract vector (A1,A2,A3) from (B1,B2,B3) on face IFACE1.
C
C     IOP = 1
C     Extract vector (B1,B2,B3) from (A1,A2,A3) on face IFACE1.
C
C     A1, A2, A3 have the (NX,NY,NFACE) data structure
C     B1, B2, B3 have the (NX,NY,NZ)    data structure
C     IFACE1 is in the preprocessor notation 
C     IFACE  is the dssum notation.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
      DIMENSION A1(LX1,LY1),A2(LX1,LY1),A3(LX1,LY1),
     $          B1(LX1,LY1,LZ1),B2(LX1,LY1,LZ1),B3(LX1,LY1,LZ1)
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      IF (IOP.EQ.0) THEN
         DO 100 J2=JS2,JF2,JSKIP2
         DO 100 J1=JS1,JF1,JSKIP1
            I = I+1
            A1(I,1) = B1(J1,J2,1)
            A2(I,1) = B2(J1,J2,1)
            A3(I,1) = B3(J1,J2,1)
  100    CONTINUE
      ELSE
         DO 150 J2=JS2,JF2,JSKIP2
         DO 150 J1=JS1,JF1,JSKIP1
            I = I+1
            B1(J1,J2,1) = A1(I,1)
            B2(J1,J2,1) = A2(I,1)
            B3(J1,J2,1) = A3(I,1)
  150    CONTINUE
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FACSUB2 (A1,A2,A3,B1,B2,B3,IFACE1)
C
C     Subtract B1,B2,B3 from A1,A2,A3 on surface IFACE1 of element IE.
C
C     A1, A2, A3 have the (NX,NY,NZ)    data structure
C     B1, B2, B3 have the (NX,NY,NFACE) data structure
C     IFACE1 is in the preprocessor notation 
C     IFACE  is the dssum notation.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
      DIMENSION A1(LX1,LY1,LZ1),A2(LX1,LY1,LZ1),A3(LX1,LY1,LZ1),
     $          B1(LX1,LY1),B2(LX1,LY1),B3(LX1,LY1)
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         A1(J1,J2,1) = A1(J1,J2,1) - B1(I,1)
         A2(J1,J2,1) = A2(J1,J2,1) - B2(I,1)
         A3(J1,J2,1) = A3(J1,J2,1) - B3(I,1)
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE GAMMASF (H1,H2)
C-----------------------------------------------------------------------
C
C     Compute lagrest eigenvalue of coupled Helmholtz operator
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'EIGEN'
      INCLUDE 'INPUT'
      INCLUDE 'MASS'
      INCLUDE 'MVGEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      INCLUDE 'WZ'
      common /scrmg/ ae1(lx1,ly1,lz1,lelv)
     $             , ae2(lx1,ly1,lz1,lelv)
     $             , ae3(lx1,ly1,lz1,lelv)
      common /scruz/ e1(lx1,ly1,lz1,lelv)
     $             , e2(lx1,ly1,lz1,lelv)
     $             , e3(lx1,ly1,lz1,lelv)
C
      DIMENSION H1(LX1,LY1,LZ1,1),H2(LX1,LY1,LZ1,1)
C
      NTOT1  = lx1*ly1*lz1*NELV
      IMESH  = 1
      MATMOD = 0
C
      IF (ISTEP.EQ.0) THEN
         EIGGA = 0.0
         CALL STX1SF
      ELSE
         CALL COPY (E1,EV1,NTOT1)
         CALL COPY (E2,EV2,NTOT1)
         IF (ldim.EQ.3) CALL COPY (E3,EV3,NTOT1)
      ENDIF
C
      EVNEW = EIGGA
C
      DO 1000 ITER=1,NMXE
C
      CALL AXHMSF  (AE1,AE2,AE3,E1,E2,E3,H1,H2,MATMOD)
      CALL RMASK   (AE1,AE2,AE3,NELV)
      CALL OPDSSUM (AE1,AE2,AE3)
C
      EVOLD = EVNEW
      EVNEW = GLSC3(E1,AE1,VMULT,NTOT1) + GLSC3(E2,AE2,VMULT,NTOT1)
      IF (ldim.EQ.3) EVNEW = EVNEW + GLSC3(E3,AE3,VMULT,NTOT1)
      CRIT = ABS( (EVNEW - EVOLD)/EVNEW )
      IF ( CRIT .LT. TOLEV ) GOTO 2000
C
      CALL COL3 (E1,BINVM1,AE1,NTOT1)
      CALL COL3 (E2,BINVM1,AE2,NTOT1)
      IF (ldim.EQ.3) CALL COL3 (E3,BINVM1,AE3,NTOT1)
      XX = GLSC3(E1,AE1,VMULT,NTOT1) + GLSC3(E2,AE2,VMULT,NTOT1)
      IF (ldim.EQ.3) XX = XX + GLSC3(E3,AE3,VMULT,NTOT1)
      IF (XX .LT. 0.0) GO TO 9000
C
      XNORM=1./SQRT( XX )
      CALL CMULT (E1,XNORM,NTOT1)
      CALL CMULT (E2,XNORM,NTOT1)
      IF (ldim.EQ.3) CALL CMULT (E3,XNORM,NTOT1)
C
 1000 CONTINUE
C
C     Save eigenvalue for all cases.
C     Save eigenvectors for deforming geometries.
C
 2000 EIGGA = EVNEW
      IF (IFMVBD) THEN
         CALL COPY (EV1,E1,NTOT1)
         CALL COPY (EV2,E2,NTOT1)
         IF (ldim.EQ.3) CALL COPY (EV3,E3,NTOT1)
      ENDIF
C
      RETURN
C
 9000 CONTINUE
      IF (NID.EQ.0) 
     $WRITE ( 6,*) ' Non +ve def. A-operator detected during eigenvalue 
     $computation : tran(x)Ax =',XX
      CALL EMERXIT
      call exitt
C
      END
C-----------------------------------------------------------------------
      SUBROUTINE CMULT2 (A,B,CONST,N)
      DIMENSION A(1),B(1)
      DO 100 I=1,N
         A(I)=B(I)*CONST
 100  CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE ADD3S (A,B,C,CONST,N)
      DIMENSION A(1),B(1),C(1)
      DO 100 I=1,N
        A(I)=B(I)+CONST*C(I)
  100 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE EMERXIT
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'PARALLEL'
C
C     Try to hang in there on the first few time steps (pff 8/92)
      IF (IFTRAN.AND.ISTEP.LT.9) RETURN
C
      LASTEP = 1
      CALL PREPOST(.true.,'   ')
C
      IF (NP.EQ.1) THEN
         WRITE (6,*) '       '
         WRITE (6,*) 
     $   ' Emergency exit:',ISTEP,'   time =',TIME
         WRITE (6,*) 
     $   ' Latest solution and data are dumped for post-processing.'
         WRITE (6,*) ' *** STOP ***'
      ELSE
         WRITE (6,*) '       '
         WRITE (6,*) NID,
     $   ' Emergency exit:',ISTEP,'   time =',TIME
         WRITE (6,*) 
     $   ' Latest solution and data are dumped for post-processing.'
         WRITE (6,*) ' *** STOP ***'
      ENDIF
C
      call runstat
c
      call exitt
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FACCVS (A1,A2,A3,B,IFACE1)
C
C     Collocate scalar B with vector A, components A1,A2,A3,
C     on the surface IFACE1 of an element.
C
C         A1,A2,A3 have the (NX,NY,NZ) data structure
C         B has the (NX,NY,IFACE) data structure
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
      DIMENSION A1(LX1,LY1,LZ1),A2(LX1,LY1,LZ1),A3(LX1,LY1,LZ1),
     $          B(LX1,LY1)
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      IF (ldim.EQ.2) THEN    
         DO 100 J2=JS2,JF2,JSKIP2
         DO 100 J1=JS1,JF1,JSKIP1
            I = I+1
            A1(J1,J2,1) = A1(J1,J2,1)*B(I,1)
            A2(J1,J2,1) = A2(J1,J2,1)*B(I,1)
  100    CONTINUE
      ELSE
         DO 200 J2=JS2,JF2,JSKIP2
         DO 200 J1=JS1,JF1,JSKIP1
            I = I+1
            A1(J1,J2,1) = A1(J1,J2,1)*B(I,1)
            A2(J1,J2,1) = A2(J1,J2,1)*B(I,1)
            A3(J1,J2,1) = A3(J1,J2,1)*B(I,1)
  200    CONTINUE
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE STX1SF
C------------------------------------------------------------------
C
C     Compute startvector for finding an eigenvalue on mesh 1.
C     Normalization: XT*B*X = 1
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'MASS'
      INCLUDE 'SOLN'
      common /scrmg/ ae1(lx1,ly1,lz1,lelv)
     $             , ae2(lx1,ly1,lz1,lelv)
     $             , ae3(lx1,ly1,lz1,lelv)
      common /scruz/ e1(lx1,ly1,lz1,lelv)
     $             , e2(lx1,ly1,lz1,lelv)
     $             , e3(lx1,ly1,lz1,lelv)
C
      NTOT1 = lx1*ly1*lz1*NELV
      CALL RZERO3 (E1 ,E2 ,E3 ,NTOT1)
      CALL RZERO3 (AE1,AE2,AE3,NTOT1)
C
      CALL COPY  (E1,BM1,NTOT1)
      CALL COPY  (E2,BM1,NTOT1)
      IF (ldim.EQ.3) CALL COPY  (E3,BM1,NTOT1)
C
      CALL RMASK (E1,E2,E3,NELV)
      CALL COL3  (AE1,BM1,E1,NTOT1)
      CALL COL3  (AE2,BM1,E2,NTOT1)
      IF (ldim.EQ.3) CALL COL3  (AE3,BM1,E3,NTOT1)
C
      CALL OPDSSUM (AE1,AE2,AE3)
C
      XX = GLSC3 (E1,AE1,VMULT,NTOT1) + GLSC3 (E2,AE2,VMULT,NTOT1)
      IF (ldim.EQ.3) XX = XX + GLSC3 (E3,AE3,VMULT,NTOT1)      
      XNORM  = 1./SQRT(XX)
      CALL CMULT (E1,XNORM,NTOT1)
      CALL CMULT (E2,XNORM,NTOT1)
      IF (ldim.EQ.3) CALL CMULT (E3,XNORM,NTOT1)

c     call exitti   ('quit in stx1sf$,',nel)

C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SOLVEL
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      DO 100 IEL=1,NELV
      DO 100 K=1,lz1
      DO 100 J=1,ly1
      DO 100 I=1,lx1
         CALL VSOLN (VX (I,J,K,IEL),VY (I,J,K,IEL),VZ (I,J,K,IEL),
     $               XM1(I,J,K,IEL),YM1(I,J,K,IEL),ZM1(I,J,K,IEL),PI)
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE VSOLN (UX,UY,UZ,X,Y,Z,PI)
C                                                                               
C       URR=(1.-0.75/SQRT(X**2+Y**2)+0.0625/(SQRT(X**2+Y**2)**3))*(X/
C     $     SQRT(X**2+Y**2))
C       UTETA=-(1.-0.375/SQRT(X**2+Y**2)-0.03125/(SQRT(X**2+Y**2)**3))*
C     $       (Y/SQRT(X**2+Y**2))
C       UX=URR*(X/SQRT(X**2+Y**2))-UTETA*(Y/SQRT(X**2+Y**2))
C       UY=URR*(Y/SQRT(X**2+Y**2))+UTETA*(X/SQRT(X**2+Y**2))
C
C       UX = 2.*COS( PI*X )
C       UY = PI*Y*SIN( PI*X )
C
        UX = 0.0
        UY = 0.0
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SOLPRES
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
C
      DO 100 IEL=1,NELV
      DO 100 K=1,lz2
      DO 100 J=1,ly2
      DO 100 I=1,lx2
         CALL PRSOLN (PR (I,J,K,IEL),XM2(I,J,K,IEL),YM2(I,J,K,IEL),
     *                ZM2(I,J,K,IEL),PI)
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRSOLN (P,X,Y,Z,PI)
C                                                                               
C      R  = SQRT( X**2 + Y**2 )
C      CS = X/R
C      P  = -0.75 * CS / R**2
C
C      P  = -SIN( PI*X )*COS( PI*Y )
C
      P = 0.0
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRINTEL (TA,A,IEL)
C
      INCLUDE 'SIZE'
      DIMENSION TA(LX1,LY1,LZ1,LELT)
      CHARACTER A*10
C
      lz1I   =  1
      lz1J   = lz1
      lz1INC =  1
C
      WRITE (21,*) 'ELEMENT NUMBER ',IEL
      DO 101 IPL=lz1I,lz1J,lz1INC
         CALL OUTM1 (TA,A,lz1,IEL,IPL)
 101  CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE PRINTV (TA,A,NEL)
C-----------------------------------------------------------------------
C
C     Store the results
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      DIMENSION TA(LX1,LY1,LZ1,LELT)
      CHARACTER A*10
C
      lz1I   =  1
      lz1J   = lz1
      lz1INC =  1
C
      DO 9001 IEL = 1,NEL
         WRITE (21,*) 'ELEMENT NUMBER ',IEL
         DO 101 IPL=lz1I,lz1J,lz1INC
            CALL OUTM1 (TA,A,lz1,IEL,IPL)
 101     CONTINUE
 9001 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE OUTF1 (X,TXT,IEL,IFC)
      INCLUDE 'SIZE'
      DIMENSION X(LX1,LZ1,6,LELT)
      CHARACTER*10 TXT
C
         NFACE = 2*ldim
         NZI   = lz1
         NZJ   =  1
         NZINC = -1
         NXI   =  1
         NXJ   = lx1
         NXINC =  1
C
         WRITE(21,106) TXT,IFC,NFACE
         DO 100 J=NZI,NZJ,NZINC
         WRITE(21,105) (X(I,J,IFC,IEL),I=NXI,NXJ,NXINC)
  100    CONTINUE
C
  105 FORMAT(5E15.6)
  106 FORMAT(///,5X,'     ^              ',/,
     $           5X,'   S |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','Plane = ',I2,'/',I2,/,
     $           5X,'       R            ',/)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE OUTM1 (X,TXT,NP,IEL,IP)
      INCLUDE 'SIZE'
      DIMENSION X(LX1,LY1,LZ1,LELT)
      CHARACTER*10 TXT
C
         NYI   = ly1
         NYJ   =  1
         NYINC = -1
         NXI   =  1
         NXJ   = lx1
         NXINC =  1
C
         WRITE(6,106) TXT,IP,NP
         DO 100 J=NYI,NYJ,NYINC
         WRITE(6,105) (X(I,J,IP,IEL),I=NXI,NXJ,NXINC)
  100    CONTINUE
C
c 105 FORMAT(1p8e10.3)
  105 FORMAT(8f10.3)
  106 FORMAT(///,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','Plane = ',I2,'/',I2,/,
     $           5X,'       X            ',/)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE OUTM2 (X,TXT,NP,IEL,IP)
      INCLUDE 'SIZE'
      DIMENSION X(LX2,LY2,LZ2,LELV)
      CHARACTER*10 TXT
C
         NYI   = ly2
         NYJ   =  1
         NYINC = -1
         NXI   =  1
         NXJ   = lx2
         NXINC =  1
C
         WRITE(21,106) TXT,IP,NP
         DO 100 J=NYI,NYJ,NYINC
         WRITE(21,105) (X(I,J,IP,IEL),I=NXI,NXJ,NXINC)
  100    CONTINUE
C
  105 FORMAT(5E15.6)
  106 FORMAT(///,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','Plane = ',I2,'/',I2,/,
     $           5X,'       X            ',/)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE STSMASK (C1MASK,C2MASK,C3MASK)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      include 'INPUT'
      common /screv/ hfmask(lx1,lz1,6,lelt)
     $             , hvmask(lx1,ly1,lz1,lelt)
C
      DIMENSION C1MASK(LX1,LY1,LZ1,1)
     $        , C2MASK(LX1,LY1,LZ1,1)
     $        , C3MASK(LX1,LY1,LZ1,1)
      INTEGER   IMDATA
      SAVE      IMDATA
      DATA      IMDATA /0/
C
      IFLD = IFIELD
      NEL  = NELFLD(IFIELD)
C
      IF (IMDATA.EQ.0) THEN
          CALL SETCDAT
          IMDATA=1
      ENDIF
C
      IF (IFLD.EQ.1) CALL SKIPCNR (NEL)
      CALL SETHMSK (HVMASK,HFMASK,IFLD,NEL)
      CALL SETMLOG (HVMASK,HFMASK,IFLD,NEL)
      CALL SETMASK (C1MASK,C2MASK,C3MASK,HVMASK,NEL)
      IF (IFLMSF(IFLD)) CALL SETCSYS (HVMASK,HFMASK,NEL)
      IF (IFLD.EQ.0)    CALL FIXWMSK (C2MASK,C3MASK,HVMASK,HFMASK,NEL)

      if (ifaxis.and.ifld.eq.1)  call fixmska (c1mask,c2mask,c3mask)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE UPDMSYS (IFLD)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      common /screv/ hfmask(lx1,lz1,6,lelt)
     $             , hvmask(lx1,ly1,lz1,lelt)
C
      IF (.NOT.IFLMSF(IFLD)) RETURN
C
      NEL  = NELFLD(IFLD)
      CALL SETHMSK (HVMASK,HFMASK,IFLD,NEL)
      CALL SETCSYS (HVMASK,HFMASK,NEL)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETHMSK (HVMASK,HFMASK,IFLD,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
C
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
      CHARACTER CB*3
C
      NTOT1 = lx1*ly1*lz1*NEL
      NXZ1  = lx1*lz1
      NTOTF = NXZ1*6*NEL
      NFACE = 2*ldim
      CONST = 5.0
      CALL CFILL (HVMASK,CONST,NTOT1)
      CALL CFILL (HFMASK,CONST,NTOTF)
C
      IF (IFLD.EQ.1) THEN
C
      DO 110 IEL=1,NEL
      DO 110 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'ON ' .OR. CB.EQ.'on ' .or.
     $       CB.EQ.'MM ' .OR. CB.EQ.'mm ' ) THEN
             CALL FACEV (HVMASK,IEL,IFC,3.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),3.0,NXZ1)
         ENDIF
  110 CONTINUE
C
      DO 120 IEL=1,NEL
      DO 120 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'SYM' .OR. CB.EQ.'A  ' .OR. CB.EQ.'WS ' .OR.
     $       CB.EQ.'ws ' .OR. CB.EQ.'WSL' .OR. CB.EQ.'wsl' .OR. 
     $       CB.EQ.'SH ' .OR. CB.EQ.'sh ' .OR. CB.EQ.'SHL' .OR. 
     $       CB.EQ.'shl')                                  THEN
             CALL FACEV (HVMASK,IEL,IFC,2.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),2.0,NXZ1)
         ENDIF
  120 CONTINUE
C
      DO 130 IEL=1,NEL
      DO 130 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'MF ' .OR. CB.EQ.'V  ' .OR. CB.EQ.'v  ' .OR.
     $       CB.EQ.'VL ' .OR. CB.EQ.'vl ' .OR. CB(1:2).EQ.'mv') THEN
             CALL FACEV (HVMASK,IEL,IFC,1.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),1.0,NXZ1)
         ENDIF
  130 CONTINUE
C
      DO 140 IEL=1,NEL
      DO 140 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'W  ') THEN
             CALL FACEV (HVMASK,IEL,IFC,0.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),0.0,NXZ1)
         ENDIF
  140 CONTINUE
C
      ELSE
C
      DO 210 IEL=1,NEL
      DO 210 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'SYM') THEN
             CALL FACEV (HVMASK,IEL,IFC,2.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),2.0,NXZ1)
         ENDIF
  210 CONTINUE
C
c     write(6,*) 'MASK this is ifield:',ifield
      DO 220 IEL=1,NEL
      DO 220 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB(1:1).EQ.'M' .OR. CB(1:1).EQ.'m') THEN
c            CALL FACEV (HVMASK,IEL,IFC,1.0,lx1,ly1,lz1)
c            CALL CFILL (HFMASK(1,1,IFC,IEL),1.0,NXZ1)
             CALL FACEV (HVMASK,IEL,IFC,2.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),2.0,NXZ1)
         ENDIF
  220 CONTINUE
C
      DO 230 IEL=1,NEL
      DO 230 IFC=1,NFACE
         CB=CBC(IFC,IEL,IFLD)
         IF (CB.EQ.'FIX') THEN
             CALL FACEV (HVMASK,IEL,IFC,0.0,lx1,ly1,lz1)
             CALL CFILL (HFMASK(1,1,IFC,IEL),0.0,NXZ1)
         ENDIF
  230 CONTINUE
C
      ENDIF
C
      CALL DSOP (HVMASK,'MNA',lx1,ly1,lz1)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SKIPCNR (NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      NFACE = 2*ldim
      NCRFC = NFACE - 2
      NMXCR = 8*NEL
      CALL LFALSE (IFNSKP,NMXCR)
C
      DO 100 IEL=1,NEL
      DO 100 IFC=1,NFACE
         IF (CDOF(IFC,IEL).EQ.'1') THEN
             ICR=MCRFC(1,IFC)
             IFNSKP(ICR,IEL)=.TRUE.
         ELSEIF (CDOF(IFC,IEL).EQ.'2') THEN
             ICR=MCRFC(2,IFC)
             IFNSKP(ICR,IEL)=.TRUE.
         ELSEIF (CDOF(IFC,IEL).EQ.'3') THEN
             ICR=MCRFC(3,IFC)
             IFNSKP(ICR,IEL)=.TRUE.
         ELSEIF (CDOF(IFC,IEL).EQ.'4') THEN
             ICR=MCRFC(4,IFC)
             IFNSKP(ICR,IEL)=.TRUE.
         ENDIF
         IF (CDOF(IFC,IEL).EQ.'*') THEN
         DO 160 ICRFC=1,NCRFC
            ICR=MCRFC(ICRFC,IFC)
            IFNSKP(ICR,IEL)=.TRUE.
  160    CONTINUE
         ENDIF
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETMASK (C1MASK,C2MASK,C3MASK,HVMASK,NEL) 
C 
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL' !XXXXX
C
      DIMENSION HVMASK (LX1,LY1,LZ1,1)
     $        , C1MASK(LX1,LY1,LZ1,1)
     $        , C2MASK(LX1,LY1,LZ1,1)
     $        , C3MASK(LX1,LY1,LZ1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL
      CALL RZERO3  (C1MASK,C2MASK,C3MASK,NTOT1)
C
      DO 100 IEL=1,NEL
      DO 100 IZ=1,lz1
      DO 100 IY=1,ly1
      DO 100 IX=1,lx1
         HMV=ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .GT. 2.9) THEN
             C1MASK(IX,IY,IZ,IEL) = 1.0
         ENDIF
         IF ((HMV.GT.1.9 .AND. HMV.LT.2.1) .OR. HMV.GT.4.9) THEN
             C2MASK(IX,IY,IZ,IEL) = 1.0
         ENDIF
  100 CONTINUE
C
      IF (ldim.EQ.3) CALL COPY (C3MASK,C2MASK,NTOT1)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETMLOG (HVMASK,HFMASK,IFLD,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NFACE = 2*ldim
      NEDGE = 12
      NCRNR = 2**ldim
      NTOTF = NFACE*NEL
      NTOTS = NEDGE*NEL
      NTOTC = NCRNR*NEL
      EPSA  = 1.E-6
C
      IFLMSF(IFLD) = .FALSE.
      IFLMSE(IFLD) = .FALSE.
      IFLMSC(IFLD) = .FALSE.
      CALL LFALSE (IFMSFC(1,1,IFLD),NTOTF)
      CALL LFALSE (IFMSEG(1,1,IFLD),NTOTS)
      CALL LFALSE (IFMSCR(1,1,IFLD),NTOTC)
C
      DO 100 IEL=1,NEL
      DO 100 IFC=1,NFACE
         HMF = ABS( HFMASK(1,1,IFC,IEL) )
         IF (HMF .GT. 1.9  .AND. HMF .LT. 3.1 ) THEN
             IFLMSF(IFLD)         = .TRUE.
             IFMSFC(IFC,IEL,IFLD) = .TRUE.
         ENDIF
 100  CONTINUE
      CALL GLLOG(IFLMSF(IFLD),.TRUE.)
C
      IF (ldim.EQ.3) THEN
         DO 200 IEL=1,NEL
         DO 200 ISD=1,NEDGE
            IX  = MIDRST(1,ISD)
            IY  = MIDRST(2,ISD)
            IZ  = MIDRST(3,ISD)
            HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
            IF (HMV .LT. 1.9  .OR.  HMV .GT. 3.1) GOTO 200
            IDIFF = 0
            DO 220 II=1,2
               IFC = MFCEG(II,ISD)
               HMF = ABS( HFMASK(1,1,IFC,IEL) )
               IF (ABS(HMV - HMF) .GT. EPSA) IDIFF=IDIFF + 1
 220        CONTINUE
            IF (IDIFF.EQ.2) THEN
               IFLMSE(IFLD)         = .TRUE.
               IFMSEG(ISD,IEL,IFLD) = .TRUE.
            ENDIF
 200     CONTINUE
         CALL GLLOG(IFLMSE(IFLD),.TRUE.)
      ENDIF
C
      DO 300 IEL=1,NEL
      DO 300 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)
         IY  = MCRRST(2,ICR)
         IZ  = MCRRST(3,ICR)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR.  HMV .GT. 3.1) GOTO 300
         IDIFF = 0
         DO 330 II=1,ldim
            IFC = MFCCR(II,ICR)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .GT. EPSA) IDIFF=IDIFF + 1
 330     CONTINUE
         IF (ldim.EQ.3) THEN
            DO 360 II=1,ldim
               ISD = MEGCR(II,ICR)
               IXS = MIDRST(1,ISD)
               IYS = MIDRST(2,ISD)
               IZS = MIDRST(3,ISD)
               HMS = ABS( HVMASK(IXS,IYS,IZS,IEL) )
               IF (ABS(HMV - HMS) .GT. EPSA) IDIFF=IDIFF + 1
 360        CONTINUE
         ENDIF
         IF ( (ldim.EQ.2 .AND. IDIFF.EQ.2)   .OR.
     $        (ldim.EQ.3 .AND. IDIFF.EQ.6) ) THEN
              IFLMSC(IFLD)         = .TRUE.
              IFMSCR(ICR,IEL,IFLD) = .TRUE.
         ENDIF
 300  CONTINUE
      CALL GLLOG(IFLMSC(IFLD),.TRUE.)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETCSYS (HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NFACE  = 2*ldim
      NTOT1  = lx1*ly1*lz1*NEL
C
      CALL RZERO3  (VNX,VNY,VNZ,NTOT1)
      CALL RZERO3  (V1X,V1Y,V1Z,NTOT1)
      CALL RZERO3  (V2X,V2Y,V2Z,NTOT1)
C
      DO 10 IEL=1,NEL
      DO 10 IFC=1,NFACE
         HMF = ABS( HFMASK(1,1,IFC,IEL) )
         IF (HMF .GT. 1.9  .AND.  HMF .LT. 3.1)
     $   CALL FACEXV(UNX(1,1,IFC,IEL),UNY(1,1,IFC,IEL),UNZ(1,1,IFC,IEL),
     $               VNX(1,1,1,IEL),VNY(1,1,1,IEL),VNZ(1,1,1,IEL),IFC,1)
   10 CONTINUE
C
      IF (ldim.EQ.2) THEN
          CALL COMAVN2 (HVMASK,HFMASK,NEL)
      ELSE
          CALL COMAVN3 (HVMASK,HFMASK,NEL)
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE COMAVN2 (HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NTOT1  = lx1*ly1*lz1*NEL
      NFACE  = 2*ldim
      NCRNR  = 2**ldim
      EPSA   = 1.0E-06
      CALL RZERO (VNZ,NTOT1)
C
      IZ = 1
      DO 100 IEL=1,NEL
      DO 100 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)
         IY  = MCRRST(2,ICR)           
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 100
         VNX(IX,IY,IZ,IEL) = 0.0
         VNY(IX,IY,IZ,IEL) = 0.0
  100 CONTINUE
C
      DO 200 IEL=1,NEL
      DO 200 ICR=1,NCRNR
         IF (IFNSKP(ICR,IEL)) GOTO 200
         IX  = MCRRST(1,ICR)           
         IY  = MCRRST(2,ICR)           
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 200
         DO 220 II=1,2
            IFC = MFCCR(II,ICR)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               IR = MCRRST(II,ICR)
               VNX(IX,IY,IZ,IEL)=VNX(IX,IY,IZ,IEL) + UNX(IR,IZ,IFC,IEL)
               VNY(IX,IY,IZ,IEL)=VNY(IX,IY,IZ,IEL) + UNY(IR,IZ,IFC,IEL)
            ENDIF
  220    CONTINUE
  200 CONTINUE
C
      CALL DSSUM   (VNX,lx1,ly1,lz1)
      CALL DSSUM   (VNY,lx1,ly1,lz1)
      CALL UNITVEC (VNX,VNY,VNZ,NTOT1)
C
      CALL COPY   (V1Y,VNX,NTOT1)
      CALL COPY   (V1X,VNY,NTOT1)
      CALL CHSIGN (V1X,NTOT1)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE COMAVN3 (HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      common /scrcg/  vnmag(lx1,ly1,lz1,lelt)
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NTOT1  = lx1*ly1*lz1*NEL
      NFACE  = 2*ldim
      NCRNR  = 2**ldim
      NEDGE  = 12
      NMID   =(lx1 + 1)/2
      NXM1   = lx1 - 1
      EPSA   = 1.0E-06
      EPSN   = 1.0E-03
C
      DO 100 IEL=1,NEL
      DO 100 ISD=1,NEDGE
         IX  = MIDRST(1,ISD)
         IY  = MIDRST(2,ISD)
         IZ  = MIDRST(3,ISD)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 100
         CALL EDGINDV (LV1,LV2,LVSKIP,ISD)
         DO 120 I=2,NXM1
            LV = LV1 + (I-1)*LVSKIP
            VNX(LV,1,1,IEL) = 0.0
            VNY(LV,1,1,IEL) = 0.0
            VNZ(LV,1,1,IEL) = 0.0
  120    CONTINUE
  100 CONTINUE
C
      DO 150 IEL=1,NEL
      DO 150 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)
         IY  = MCRRST(2,ICR)
         IZ  = MCRRST(3,ICR)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 150
         VNX(IX,IY,IZ,IEL) = 0.0
         VNY(IX,IY,IZ,IEL) = 0.0
         VNZ(IX,IY,IZ,IEL) = 0.0
 150  CONTINUE
C
C     (1) All Edges
C
      DO 200 IEL=1,NEL
      DO 200 ISD=1,NEDGE
         IX  = MIDRST(1,ISD)
         IY  = MIDRST(2,ISD)
         IZ  = MIDRST(3,ISD)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 200
         DO 220 II=1,2
            IFC = MFCEG(II,ISD)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               CALL EDGINDV (LV1,LV2,LVSKIP,ISD)
               CALL EDGINDF (LF1,LF2,LFSKIP,ISD,II)
               DO 240 I=2,NXM1
                  LV = LV1 + (I-1)*LVSKIP
                  LF = LF1 + (I-1)*LFSKIP
                  VNX(LV,1,1,IEL)=VNX(LV,1,1,IEL)+UNX(LF,1,IFC,IEL)
                  VNY(LV,1,1,IEL)=VNY(LV,1,1,IEL)+UNY(LF,1,IFC,IEL)
                  VNZ(LV,1,1,IEL)=VNZ(LV,1,1,IEL)+UNZ(LF,1,IFC,IEL)
  240          CONTINUE    
            ENDIF
  220    CONTINUE 
  200 CONTINUE
C
C     (2) All Corners
C
      DO 300 IEL=1,NEL
      DO 300 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)
         IY  = MCRRST(2,ICR)
         IZ  = MCRRST(3,ICR)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 3.1) GOTO 300
         DO 320 II=1,3
            IFC = MFCCR(II,ICR)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               IRA = NTCRF(1,II)
               ISA = NTCRF(2,II)
               IR  = MCRRST(IRA,ICR)
               IS  = MCRRST(ISA,ICR)
               VNX(IX,IY,IZ,IEL)=VNX(IX,IY,IZ,IEL)+UNX(IR,IS,IFC,IEL)
               VNY(IX,IY,IZ,IEL)=VNY(IX,IY,IZ,IEL)+UNY(IR,IS,IFC,IEL)
               VNZ(IX,IY,IZ,IEL)=VNZ(IX,IY,IZ,IEL)+UNZ(IR,IS,IFC,IEL)
            ENDIF
  320    CONTINUE
  300 CONTINUE
C
      CALL DSSUM   (VNX,lx1,ly1,lz1)
      CALL DSSUM   (VNY,lx1,ly1,lz1)
      CALL DSSUM   (VNZ,lx1,ly1,lz1)
      CALL UNITVEC (VNX,VNY,VNZ,NTOT1)
      CALL VDOT3   (VNMAG,VNX,VNY,VNZ,VNX,VNY,VNZ,NTOT1)
C
      DO 500 IEL=1,NEL
      DO 500 IFC=1,NFACE
         HMF = ABS( HFMASK(1,1,IFC,IEL) )
         IF (HMF .LT. 1.9  .OR. HMF .GT. 3.1) GOTO 500
         CALL FACIND2 (JS1,JF1,JSKIP1,JS2,JF2,JSKIP2,IFC)
         DO 520 J2=JS2,JF2,JSKIP2
         DO 520 J1=JS1,JF1,JSKIP1
            IF (VNMAG(J1,J2,1,IEL) .LT. EPSA) GOTO 520
            VlzdIF = ABS(VNZ(J1,J2,1,IEL)) - 1.0
            IF (ABS(VlzdIF) .LT. EPSN) THEN
                V1X(J1,J2,1,IEL) = 1.0
                V1Y(J1,J2,1,IEL) = 0.0
                V1Z(J1,J2,1,IEL) = 0.0
            ELSE
                SSN = SQRT(VNX(J1,J2,1,IEL)**2 + VNY(J1,J2,1,IEL)**2)
                V1X(J1,J2,1,IEL) = -VNY(J1,J2,1,IEL) / SSN
                V1Y(J1,J2,1,IEL) =  VNX(J1,J2,1,IEL) / SSN
                V1Z(J1,J2,1,IEL) =  0.0
            ENDIF
  520    CONTINUE
  500 CONTINUE
C
      CALL DSSUM   (V1X,lx1,ly1,lz1)
      CALL DSSUM   (V1Y,lx1,ly1,lz1)
      CALL DSSUM   (V1Z,lx1,ly1,lz1)
      CALL UNITVEC (V1X,V1Y,V1Z,NTOT1)
C
      CALL VCROSS (V2X,V2Y,V2Z,VNX,VNY,VNZ,V1X,V1Y,V1Z,NTOT1)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FIXWMSK (W2MASK,W3MASK,HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      DIMENSION HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      DIMENSION W2MASK(LX1,LY1,LZ1,1)
     $        , W3MASK(LX1,LY1,LZ1,1)
C
      IF (ldim.EQ.2) THEN
         CALL FXWMS2 (W2MASK,HVMASK,HFMASK,NEL)
      ELSE
         CALL FXWMS3 (W2MASK,W3MASK,HVMASK,HFMASK,NEL)
      ENDIF
C
      CALL DSOP(W2MASK,'MUL',lx1,ly1,lz1)
      IF (ldim.EQ.3) CALL DSOP(W3MASK,'MUL',lx1,ly1,lz1)
C 
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FXWMS2 (W2MASK,HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      DIMENSION W2MASK(LX1,LY1,LZ1,1)
     $        , HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NCRNR  = 2**ldim
      EPSA   = 1.0E-06
C
      IZ = 1
      DO 100 IEL=1,NEL
      DO 100 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)           
         IY  = MCRRST(2,ICR)           
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR.  HMV .GT. 2.1) GOTO 100
         DO 120 II=1,2
            IFC = MFCCR(II,ICR)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               IR  = MCRRST(II,ICR)
               DOT = VNX(IX,IY,IZ,IEL)*UNX(IR,IZ,IFC,IEL) + 
     $               VNY(IX,IY,IZ,IEL)*UNY(IR,IZ,IFC,IEL)
               IF (DOT .LT. 0.99) THEN
                  W2MASK(IX,IY,IZ,IEL) = 0.0
                  GOTO 100
               ENDIF
            ENDIF
  120    CONTINUE
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FXWMS3 (W2MASK,W3MASK,HVMASK,HFMASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      DIMENSION W2MASK(LX1,LY1,LZ1,1)
     $        , W3MASK(LX1,LY1,LZ1,1)
     $        , HVMASK(LX1,LY1,LZ1,1)
     $        , HFMASK(LX1,LZ1,6,1)
C
      NCRNR = 2**ldim
      NEDGE = 12
      NMID  = (lx1 + 1)/2
      EPSA  = 1.0E-06
C
      DO 100 IEL=1,NEL
      DO 100 ISD=1,12
         IX  = MIDRST(1,ISD)
         IY  = MIDRST(2,ISD)
         IZ  = MIDRST(3,ISD)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR.  HMV .GT. 2.1) GOTO 100
         DO 120 II=1,2
            IFC = MFCEG(II,ISD)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               CALL EDGINDF (LF1,LF2,LFSKIP,ISD,II)
               LF  = LF1 + (NMID-1)*LFSKIP
               DOT = VNX(IX,IY,IZ,IEL)*UNX(LF,1,IFC,IEL) + 
     $               VNY(IX,IY,IZ,IEL)*UNY(LF,1,IFC,IEL) +
     $               VNZ(IX,IY,IZ,IEL)*UNZ(LF,1,IFC,IEL)
               IF (DOT .LT. 0.99) THEN
                  CALL EDGINDV (LV1,LV2,LVSKIP,ISD)
                  DO 140 LV=LV1,LV2,LVSKIP
                     W3MASK(LV,1,1,IEL) = 0.0
  140             CONTINUE    
               ENDIF
            ENDIF
  120    CONTINUE
  100 CONTINUE
C
C     All Corners
C
      DO 300 IEL=1,NEL
      DO 300 ICR=1,NCRNR
         IX  = MCRRST(1,ICR)
         IY  = MCRRST(2,ICR)
         IZ  = MCRRST(3,ICR)
         HMV = ABS( HVMASK(IX,IY,IZ,IEL) )
         IF (HMV .LT. 1.9  .OR. HMV .GT. 2.1) GOTO 300
         DO 320 II=1,3
            IFC = MFCCR(II,ICR)
            HMF = ABS( HFMASK(1,1,IFC,IEL) )
            IF (ABS(HMV - HMF) .LT. EPSA) THEN
               IRA = NTCRF(1,II)
               ISA = NTCRF(2,II)
               IR  = MCRRST(IRA,ICR)
               IS  = MCRRST(ISA,ICR)
               DOT = VNX(IX,IY,IZ,IEL)*UNX(IR,IS,IFC,IEL) + 
     $               VNY(IX,IY,IZ,IEL)*UNY(IR,IS,IFC,IEL) +
     $               VNZ(IX,IY,IZ,IEL)*UNZ(IR,IS,IFC,IEL)
            ENDIF
            IF (DOT .LT. 0.99) THEN
                W2MASK(IX,IY,IZ,IEL) = 0.0
                GOTO 300
            ENDIF 
  320    CONTINUE
  300 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETCDAT 
C
      INCLUDE 'SIZE'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      NMID = (lx1 +1)/2
C
C     Corners on faces
C
      MCRFC(1,1) = 1
      MCRFC(2,1) = 2
      MCRFC(3,1) = 6
      MCRFC(4,1) = 5
      MCRFC(1,2) = 2
      MCRFC(2,2) = 3
      MCRFC(3,2) = 7
      MCRFC(4,2) = 6
      MCRFC(1,3) = 3
      MCRFC(2,3) = 4
      MCRFC(3,3) = 8
      MCRFC(4,3) = 7
      MCRFC(1,4) = 4
      MCRFC(2,4) = 1
      MCRFC(3,4) = 5
      MCRFC(4,4) = 8
      MCRFC(1,5) = 4
      MCRFC(2,5) = 3
      MCRFC(3,5) = 2
      MCRFC(4,5) = 1
      MCRFC(1,6) = 5
      MCRFC(2,6) = 6
      MCRFC(3,6) = 7
      MCRFC(4,6) = 8
C
C     Faces at corners
C
      MFCCR(1,1) = 4
      MFCCR(2,1) = 1
      MFCCR(3,1) = 5
      MFCCR(1,2) = 1
      MFCCR(2,2) = 2
      MFCCR(3,2) = 5
      MFCCR(1,3) = 2
      MFCCR(2,3) = 3
      MFCCR(3,3) = 5
      MFCCR(1,4) = 3
      MFCCR(2,4) = 4
      MFCCR(3,4) = 5
      MFCCR(1,5) = 4
      MFCCR(2,5) = 1
      MFCCR(3,5) = 6
      MFCCR(1,6) = 1
      MFCCR(2,6) = 2
      MFCCR(3,6) = 6
      MFCCR(1,7) = 2
      MFCCR(2,7) = 3
      MFCCR(3,7) = 6
      MFCCR(1,8) = 3
      MFCCR(2,8) = 4
      MFCCR(3,8) = 6
C
C     Edges at corners
C
      MEGCR(1,1) = 4
      MEGCR(2,1) = 1
      MEGCR(3,1) = 9
      MEGCR(1,2) = 1
      MEGCR(2,2) = 2
      MEGCR(3,2) = 10
      MEGCR(1,3) = 2
      MEGCR(2,3) = 3
      MEGCR(3,3) = 11
      MEGCR(1,4) = 3
      MEGCR(2,4) = 4
      MEGCR(3,4) = 12
      MEGCR(1,5) = 8
      MEGCR(2,5) = 5
      MEGCR(3,5) = 9
      MEGCR(1,6) = 5
      MEGCR(2,6) = 6
      MEGCR(3,6) = 10
      MEGCR(1,7) = 6
      MEGCR(2,7) = 7
      MEGCR(3,7) = 11
      MEGCR(1,8) = 7
      MEGCR(2,8) = 8
      MEGCR(3,8) = 12
C
C     Faces on edges
C
      MFCEG(1,1)  = 1
      MFCEG(2,1)  = 5
      MFCEG(1,2)  = 2
      MFCEG(2,2)  = 5
      MFCEG(1,3)  = 3
      MFCEG(2,3)  = 5
      MFCEG(1,4)  = 4
      MFCEG(2,4)  = 5
      MFCEG(1,5)  = 1
      MFCEG(2,5)  = 6
      MFCEG(1,6)  = 2
      MFCEG(2,6)  = 6
      MFCEG(1,7)  = 3
      MFCEG(2,7)  = 6
      MFCEG(1,8)  = 4
      MFCEG(2,8)  = 6
      MFCEG(1,9)  = 4
      MFCEG(2,9)  = 1
      MFCEG(1,10) = 1
      MFCEG(2,10) = 2
      MFCEG(1,11) = 2
      MFCEG(2,11) = 3
      MFCEG(1,12) = 3
      MFCEG(2,12) = 4
C
C     Corners at edges
C
      MCREG(1,1)  = 1
      MCREG(2,1)  = 2
      MCREG(1,2)  = 2
      MCREG(2,2)  = 3
      MCREG(1,3)  = 4
      MCREG(2,3)  = 3
      MCREG(1,4)  = 1
      MCREG(2,4)  = 4
      MCREG(1,5)  = 5
      MCREG(2,5)  = 6
      MCREG(1,6)  = 6
      MCREG(2,6)  = 7
      MCREG(1,7)  = 8
      MCREG(2,7)  = 7
      MCREG(1,8)  = 5
      MCREG(2,8)  = 8
      MCREG(1,9)  = 1
      MCREG(2,9)  = 5
      MCREG(1,10) = 2
      MCREG(2,10) = 6
      MCREG(1,11) = 3
      MCREG(2,11) = 7
      MCREG(1,12) = 4
      MCREG(2,12) = 8
C
C     Corner indices (Vol array)
C
      MCRRST(1,1) = 1
      MCRRST(2,1) = 1
      MCRRST(3,1) = 1
      MCRRST(1,2) = lx1
      MCRRST(2,2) = 1
      MCRRST(3,2) = 1
      MCRRST(1,3) = lx1
      MCRRST(2,3) = lx1
      MCRRST(3,3) = 1
      MCRRST(1,4) = 1
      MCRRST(2,4) = lx1
      MCRRST(3,4) = 1
      MCRRST(1,5) = 1
      MCRRST(2,5) = 1
      MCRRST(3,5) = lx1
      MCRRST(1,6) = lx1
      MCRRST(2,6) = 1
      MCRRST(3,6) = lx1
      MCRRST(1,7) = lx1
      MCRRST(2,7) = lx1
      MCRRST(3,7) = lx1
      MCRRST(1,8) = 1
      MCRRST(2,8) = lx1
      MCRRST(3,8) = lx1
C
C     Mid-edge indcies (Vol array)
C 
      MIDRST(1,1)  = NMID
      MIDRST(1,2)  = lx1
      MIDRST(1,3)  = NMID
      MIDRST(1,4)  = 1
      MIDRST(1,5)  = NMID
      MIDRST(1,6)  = lx1
      MIDRST(1,7)  = NMID
      MIDRST(1,8)  = 1
      MIDRST(1,9)  = 1
      MIDRST(1,10) = lx1
      MIDRST(1,11) = lx1
      MIDRST(1,12) = 1
      MIDRST(2,1)  = 1
      MIDRST(2,2)  = NMID
      MIDRST(2,3)  = lx1
      MIDRST(2,4)  = NMID
      MIDRST(2,5)  = 1
      MIDRST(2,6)  = NMID
      MIDRST(2,7)  = lx1
      MIDRST(2,8)  = NMID
      MIDRST(2,9)  = 1
      MIDRST(2,10) = 1
      MIDRST(2,11) = lx1
      MIDRST(2,12) = lx1
      MIDRST(3,1)  = 1
      MIDRST(3,2)  = 1
      MIDRST(3,3)  = 1
      MIDRST(3,4)  = 1
      MIDRST(3,5)  = lx1
      MIDRST(3,6)  = lx1
      MIDRST(3,7)  = lx1
      MIDRST(3,8)  = lx1
      MIDRST(3,9)  = NMID
      MIDRST(3,10) = NMID
      MIDRST(3,11) = NMID
      MIDRST(3,12) = NMID
C
C     1-D corners indices (Vol array)
C
      MCRIND(1) = 1
      MCRIND(2) = lx1
      MCRIND(3) = lx1**2
      MCRIND(7) = lx1**3
      MCRIND(4) = MCRIND(3) - lx1 + 1
      MCRIND(5) = MCRIND(7) - MCRIND(3) + 1
      MCRIND(6) = MCRIND(5) + lx1 - 1
      MCRIND(8) = MCRIND(7) - lx1 + 1
C
C     1-D  edge indices (Face array)
C
      MEDIND(1,1) = 1
      MEDIND(2,1) = lx1
      MEDIND(1,2) = lx1**2 - lx1 + 1
      MEDIND(2,2) = lx1**2
      MEDIND(1,3) = 1
      MEDIND(2,3) = MEDIND(1,2)
      MEDIND(1,4) = lx1
      MEDIND(2,4) = lx1**2
C
C     1-D edge index type (Face array)
C
      NTEFC(1,1)  = 1
      NTEFC(2,1)  = 1
      NTEFC(1,2)  = 1
      NTEFC(2,2)  = 4
      NTEFC(1,3)  = 1
      NTEFC(2,3)  = 2
      NTEFC(1,4)  = 1
      NTEFC(2,4)  = 3
      NTEFC(1,5)  = 2
      NTEFC(2,5)  = 1
      NTEFC(1,6)  = 2
      NTEFC(2,6)  = 4
      NTEFC(1,7)  = 2
      NTEFC(2,7)  = 2
      NTEFC(1,8)  = 2
      NTEFC(2,8)  = 3
      NTEFC(1,9)  = 3
      NTEFC(2,9)  = 3
      NTEFC(1,10) = 4
      NTEFC(2,10) = 3
      NTEFC(1,11) = 4
      NTEFC(2,11) = 4
      NTEFC(1,12) = 3
      NTEFC(2,12) = 4
C
C     Corner index address on face in MCRRST
C
      NTCRF(1,1) = 1
      NTCRF(2,1) = 3
      NTCRF(1,2) = 2
      NTCRF(2,2) = 3
      NTCRF(1,3) = 1
      NTCRF(2,3) = 2
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE EDGINDF (LF1,LF2,LFSKIP,ISD,IFCN)
C
      INCLUDE 'SIZE'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      ITYP = NTEFC(IFCN,ISD)
C
      LF1 = MEDIND(1,ITYP)
      LF2 = MEDIND(2,ITYP)
C
      LFSKIP = 1
      IF (ITYP .GE. 3) LFSKIP = lx1
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE EDGINDV (LV1,LV2,LVSKIP,ISD)
C
      INCLUDE 'SIZE'
      common /indxfc/ mcrfc(4,6)
     $              , mfccr(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
C
      IODD = ISD - ISD/2*2
      ICR1 = MCREG(1,ISD)
      ICR2 = MCREG(2,ISD)
C
      LV1  = MCRIND(ICR1)
      LV2  = MCRIND(ICR2)
C
      IF (ISD .GE. 9) THEN
         LVSKIP = lx1**2
      ELSE
         IF (IODD.EQ.0) THEN
            LVSKIP = lx1
         ELSE
            LVSKIP = 1
         ENDIF
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SETCDOF
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
C
      NFACE = 2*ldim
C
      DO 100 IEL=1,NELT
      DO 100 IFC=1,NFACE
         CDOF(IFC,IEL)=CBC(IFC,IEL,0)(1:1)
 100  CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE AMASK (VB1,VB2,VB3,V1,V2,V3,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      common /scrsf/ a1mask(lx1,ly1,lz1,lelt)
     $             , a2mask(lx1,ly1,lz1,lelt)
     $             , a3mask(lx1,ly1,lz1,lelt)
      common /ctmp0/ wa(lx1,ly1,lz1,lelt)
C
      DIMENSION VB1(LX1,LY1,LZ1,1)
     $        , VB2(LX1,LY1,LZ1,1)
     $        , VB3(LX1,LY1,LZ1,1)
     $        , V1(LX1,LY1,LZ1,1)
     $        , V2(LX1,LY1,LZ1,1)
     $        , V3(LX1,LY1,LZ1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL
      CALL RONE (WA,NTOT1)
      CALL COPY (VB1,V1,NTOT1)
      CALL COPY (VB2,V2,NTOT1)
      IF (ldim.EQ.3) CALL COPY (VB3,V3,NTOT1)
C
      IF (IFIELD.EQ.1) THEN
         CALL SUB3  (A1MASK,WA,V1MASK,NTOT1)
         CALL SUB3  (A2MASK,WA,V2MASK,NTOT1)
         IF (ldim.EQ.3) CALL SUB3 (A3MASK,WA,V3MASK,NTOT1)
      ELSEIF (IFIELD.EQ.ifldmhd) THEN
         CALL SUB3  (A1MASK,WA,B1MASK,NTOT1)
         CALL SUB3  (A2MASK,WA,B2MASK,NTOT1)
         IF (ldim.EQ.3) CALL SUB3 (A3MASK,WA,B3MASK,NTOT1)
      ELSE
         CALL SUB3  (A1MASK,WA,W1MASK,NTOT1)
         CALL SUB3  (A2MASK,WA,W2MASK,NTOT1)
         IF (ldim.EQ.3) CALL SUB3 (A3MASK,WA,W3MASK,NTOT1)
      ENDIF
C
      CALL QMASK (VB1,VB2,VB3,A1MASK,A2MASK,A3MASK,NEL)
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE RMASK (R1,R2,R3,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      INCLUDE 'MVGEOM'
C
      DIMENSION R1  (LX1,LY1,LZ1,1)
     $        , R2  (LX1,LY1,LZ1,1)
     $        , R3  (LX1,LY1,LZ1,1)
C
c      if (ifsplit) then
c         call opmask(r1,r2,r3)
c         return
c      endif

c     call outfldro (v1mask,'v1mask rmk',0)
c     call outfldro (v2mask,'v2mask rmk',1)

      IF (IFIELD.EQ.1) THEN
         CALL QMASK (R1,R2,R3,V1MASK,V2MASK,V3MASK,NEL)
      ELSEIF (ifield.eq.ifldmhd) then
         CALL QMASK (R1,R2,R3,B1MASK,B2MASK,B3MASK,NEL)
      ELSE
         CALL QMASK (R1,R2,R3,W1MASK,W2MASK,W3MASK,NEL)
      ENDIF

c     call outfldro (r1,'r1   rmask',0)
c     call outfldro (r2,'r2   rmask',1)
c     call exitti   ('quit in rmask$,',nel)

      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE QMASK (R1,R2,R3,R1MASK,R2MASK,R3MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      common /ctmp1/ s1(lx1,ly1,lz1,lelt)
     $             , s2(lx1,ly1,lz1,lelt)
     $             , s3(lx1,ly1,lz1,lelt)
C
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , R3(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
     $        , R3MASK(LX1,LY1,LZ1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL
C
C     (0) Collocate Volume Mask
C
      CALL COPY  (S1,R1,NTOT1)
      CALL COPY  (S2,R2,NTOT1)
      CALL COL2  (R1,R1MASK,NTOT1)
      CALL COL2  (R2,R2MASK,NTOT1)
      IF (ldim.EQ.3) THEN
         CALL COPY (S3,R3,NTOT1)
         CALL COL2 (R3,R3MASK,NTOT1)
      ENDIF
C
C     (1) Face Mask
C
      IF (IFLMSF(IFIELD)) THEN
         IF (ldim.EQ.2) THEN
            CALL FCMSK2 (R1,R2,S1,S2,R1MASK,R2MASK,NEL)
         ELSE
            CALL FCMSK3 (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
         ENDIF
      ENDIF
C
C     (2) Edge Mask  (3-D only)
C
      IF (ldim.EQ.3 .AND. IFLMSE(IFIELD)) 
     $   CALL EGMASK (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
C
C     (3) Corner Mask
C
      IF (IFLMSC(IFIELD)) THEN
         IF (ldim.EQ.2) THEN
            CALL CRMSK2 (R1,R2,S1,S2,R1MASK,R2MASK,NEL)
         ELSE
            CALL CRMSK3 (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
         ENDIF
      ENDIF
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FCMSK2 (R1,R2,S1,S2,R1MASK,R2MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , S1(LX1,LY1,LZ1,1)
     $        , S2(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
C
      NFACE = 2*ldim
C
      DO 100 IEL=1,NEL
      DO 100 IFC=1,NFACE
         IF (.NOT.IFMSFC(IFC,IEL,IFIELD)) GO TO 100
         CALL FACIND2 (JS1,JF1,JSKIP1,JS2,JF2,JSKIP2,IFC)
         DO 120 J2=JS2,JF2,JSKIP2
         DO 120 J1=JS1,JF1,JSKIP1
            RNOR = ( S1(J1,J2,1,IEL)*VNX(J1,J2,1,IEL) +
     $               S2(J1,J2,1,IEL)*VNY(J1,J2,1,IEL) ) *
     $               R1MASK(J1,J2,1,IEL)
            RTN1 = ( S1(J1,J2,1,IEL)*V1X(J1,J2,1,IEL) +
     $               S2(J1,J2,1,IEL)*V1Y(J1,J2,1,IEL) ) *
     $               R2MASK(J1,J2,1,IEL)
            R1(J1,J2,1,IEL) = RNOR*VNX(J1,J2,1,IEL) +
     $                        RTN1*V1X(J1,J2,1,IEL)
            R2(J1,J2,1,IEL) = RNOR*VNY(J1,J2,1,IEL) +
     $                        RTN1*V1Y(J1,J2,1,IEL)
  120       CONTINUE
  100    CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE FCMSK3 (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , R3(LX1,LY1,LZ1,1)
     $        , S1(LX1,LY1,LZ1,1)
     $        , S2(LX1,LY1,LZ1,1)
     $        , S3(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
     $        , R3MASK(LX1,LY1,LZ1,1)
C
      NFACE = 2*ldim
C
      DO 100 IEL=1,NEL
      DO 100 IFC=1,NFACE
         IF (.NOT.IFMSFC(IFC,IEL,IFIELD)) GO TO 100
         CALL FACIND2 (JS1,JF1,JSKIP1,JS2,JF2,JSKIP2,IFC)
         DO 120 J2=JS2,JF2,JSKIP2
         DO 120 J1=JS1,JF1,JSKIP1
            RNOR = ( S1(J1,J2,1,IEL)*VNX(J1,J2,1,IEL) +
     $               S2(J1,J2,1,IEL)*VNY(J1,J2,1,IEL) +
     $               S3(J1,J2,1,IEL)*VNZ(J1,J2,1,IEL) ) *
     $               R1MASK(J1,J2,1,IEL)
            RTN1 = ( S1(J1,J2,1,IEL)*V1X(J1,J2,1,IEL) +
     $               S2(J1,J2,1,IEL)*V1Y(J1,J2,1,IEL) +
     $               S3(J1,J2,1,IEL)*V1Z(J1,J2,1,IEL) ) *
     $               R2MASK(J1,J2,1,IEL)
            RTN2 = ( S1(J1,J2,1,IEL)*V2X(J1,J2,1,IEL) +
     $               S2(J1,J2,1,IEL)*V2Y(J1,J2,1,IEL) +
     $               S3(J1,J2,1,IEL)*V2Z(J1,J2,1,IEL) ) *
     $               R3MASK(J1,J2,1,IEL)
            R1(J1,J2,1,IEL) = RNOR*VNX(J1,J2,1,IEL) +
     $                        RTN1*V1X(J1,J2,1,IEL) +
     $                        RTN2*V2X(J1,J2,1,IEL)
            R2(J1,J2,1,IEL) = RNOR*VNY(J1,J2,1,IEL) +
     $                        RTN1*V1Y(J1,J2,1,IEL) +
     $                        RTN2*V2Y(J1,J2,1,IEL)
            R3(J1,J2,1,IEL) = RNOR*VNZ(J1,J2,1,IEL) +
     $                        RTN1*V1Z(J1,J2,1,IEL) +
     $                        RTN2*V2Z(J1,J2,1,IEL) 
  120       CONTINUE
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE EGMASK (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , R3(LX1,LY1,LZ1,1)
     $        , S1(LX1,LY1,LZ1,1)
     $        , S2(LX1,LY1,LZ1,1)
     $        , S3(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
     $        , R3MASK(LX1,LY1,LZ1,1)
C
      NEDGE = 12
C
      DO 100 IEL=1,NEL
      DO 100 ISD=1,NEDGE
         IF (.NOT.IFMSEG(ISD,IEL,IFIELD)) GOTO 100
         CALL EDGINDV (LV1,LV2,LVSKIP,ISD)
         DO 120 LV=LV1,LV2,LVSKIP
            RNOR = ( S1(LV,1,1,IEL)*VNX(LV,1,1,IEL) +
     $               S2(LV,1,1,IEL)*VNY(LV,1,1,IEL) +
     $               S3(LV,1,1,IEL)*VNZ(LV,1,1,IEL) ) *
     $               R1MASK(LV,1,1,IEL)
            RTN1 = ( S1(LV,1,1,IEL)*V1X(LV,1,1,IEL) +
     $               S2(LV,1,1,IEL)*V1Y(LV,1,1,IEL) +
     $               S3(LV,1,1,IEL)*V1Z(LV,1,1,IEL) ) *
     $               R2MASK(LV,1,1,IEL)
            RTN2 = ( S1(LV,1,1,IEL)*V2X(LV,1,1,IEL) +
     $               S2(LV,1,1,IEL)*V2Y(LV,1,1,IEL) +
     $               S3(LV,1,1,IEL)*V2Z(LV,1,1,IEL) ) *
     $               R3MASK(LV,1,1,IEL)
            R1(LV,1,1,IEL) = RNOR*VNX(LV,1,1,IEL) +
     $                       RTN1*V1X(LV,1,1,IEL) +
     $                       RTN2*V2X(LV,1,1,IEL)
            R2(LV,1,1,IEL) = RNOR*VNY(LV,1,1,IEL) +
     $                       RTN1*V1Y(LV,1,1,IEL) +
     $                       RTN2*V2Y(LV,1,1,IEL)
            R3(LV,1,1,IEL) = RNOR*VNZ(LV,1,1,IEL) +
     $                       RTN1*V1Z(LV,1,1,IEL) +
     $                       RTN2*V2Z(LV,1,1,IEL) 
  120    CONTINUE
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE CRMSK2 (R1,R2,S1,S2,R1MASK,R2MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , S1(LX1,LY1,LZ1,1)
     $        , S2(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
C
      NCRNR = 2**ldim
C
      DO 100 IEL=1,NEL
      DO 100 ICR=1,NCRNR
         IF (.NOT.IFMSCR(ICR,IEL,IFIELD)) GO TO 100
         IX = MCRRST(1,ICR)
         IY = MCRRST(2,ICR)
         IZ = 1
         RNOR = ( S1(IX,IY,IZ,IEL)*VNX(IX,IY,IZ,IEL) +
     $            S2(IX,IY,IZ,IEL)*VNY(IX,IY,IZ,IEL) ) *
     $            R1MASK(IX,IY,IZ,IEL)
         RTN1 = ( S1(IX,IY,IZ,IEL)*V1X(IX,IY,IZ,IEL) +
     $            S2(IX,IY,IZ,IEL)*V1Y(IX,IY,IZ,IEL) ) *
     $            R2MASK(IX,IY,IZ,IEL)
         R1(IX,IY,IZ,IEL) = RNOR*VNX(IX,IY,IZ,IEL) +
     $                      RTN1*V1X(IX,IY,IZ,IEL)
         R2(IX,IY,IZ,IEL) = RNOR*VNY(IX,IY,IZ,IEL) +
     $                      RTN1*V1Y(IX,IY,IZ,IEL)
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE CRMSK3 (R1,R2,R3,S1,S2,S3,R1MASK,R2MASK,R3MASK,NEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TSTEP'
      common /indxfc/ mcrfc(4,6)
     $              , MFCCR(3,8)
     $              , MEGCR(3,8)
     $              , MFCEG(2,12)
     $              , MCREG(2,12)
     $              , MCRRST(3,8)
     $              , MIDRST(3,12)
     $              , MCRIND(8)
     $              , MEDIND(2,4)
     $              , NTEFC(2,12)
     $              , NTCRF(2,3)     
      DIMENSION R1(LX1,LY1,LZ1,1)
     $        , R2(LX1,LY1,LZ1,1)
     $        , R3(LX1,LY1,LZ1,1)
     $        , S1(LX1,LY1,LZ1,1)
     $        , S2(LX1,LY1,LZ1,1)
     $        , S3(LX1,LY1,LZ1,1)
     $        , R1MASK(LX1,LY1,LZ1,1)
     $        , R2MASK(LX1,LY1,LZ1,1)
     $        , R3MASK(LX1,LY1,LZ1,1)
C
      NCRNR = 2**ldim
C
      DO 100 IEL=1,NEL
      DO 100 ICR=1,NCRNR
         IF (.NOT.IFMSCR(ICR,IEL,IFIELD)) GO TO 100
         IX = MCRRST(1,ICR)
         IY = MCRRST(2,ICR)
         IZ = MCRRST(3,ICR)
         RNOR = ( S1(IX,IY,IZ,IEL)*VNX(IX,IY,IZ,IEL) +
     $            S2(IX,IY,IZ,IEL)*VNY(IX,IY,IZ,IEL) +
     $            S3(IX,IY,IZ,IEL)*VNZ(IX,IY,IZ,IEL) ) *
     $            R1MASK(IX,IY,IZ,IEL)
         RTN1 = ( S1(IX,IY,IZ,IEL)*V1X(IX,IY,IZ,IEL) +
     $            S2(IX,IY,IZ,IEL)*V1Y(IX,IY,IZ,IEL) +
     $            S3(IX,IY,IZ,IEL)*V1Z(IX,IY,IZ,IEL) ) *
     $            R2MASK(IX,IY,IZ,IEL)
         RTN2 = ( S1(IX,IY,IZ,IEL)*V2X(IX,IY,IZ,IEL) +
     $            S2(IX,IY,IZ,IEL)*V2Y(IX,IY,IZ,IEL) +
     $            S3(IX,IY,IZ,IEL)*V2Z(IX,IY,IZ,IEL) ) *
     $            R3MASK(IX,IY,IZ,IEL)
         R1(IX,IY,IZ,IEL) = RNOR*VNX(IX,IY,IZ,IEL) +
     $                      RTN1*V1X(IX,IY,IZ,IEL) +
     $                      RTN2*V2X(IX,IY,IZ,IEL)
         R2(IX,IY,IZ,IEL) = RNOR*VNY(IX,IY,IZ,IEL) +
     $                      RTN1*V1Y(IX,IY,IZ,IEL) +
     $                      RTN2*V2Y(IX,IY,IZ,IEL)
         R3(IX,IY,IZ,IEL) = RNOR*VNZ(IX,IY,IZ,IEL) +
     $                      RTN1*V1Z(IX,IY,IZ,IEL) +
     $                      RTN2*V2Z(IX,IY,IZ,IEL) 
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine getSnormal(sn,ix,iy,iz,iside,e)

c     calculate surface normal

      include 'SIZE'
      include 'GEOM'
      include 'TOPOL'

      real sn(3)
      integer e,f

      f = eface1(iside)

      if (1.le.f.and.f.le.2) then     ! "r face"
         sn(1) = unx(iy,iz,iside,e)
         sn(2) = uny(iy,iz,iside,e)
         sn(3) = unz(iy,iz,iside,e)
      elseif (3.le.f.and.f.le.4) then ! "s face"
         sn(1) = unx(ix,iz,iside,e)
         sn(2) = uny(ix,iz,iside,e)
         sn(3) = unz(ix,iz,iside,e)
      elseif (5.le.f.and.f.le.6) then ! "t face"
         sn(1) = unx(ix,iy,iside,e)
         sn(2) = uny(ix,iy,iside,e)
         sn(3) = unz(ix,iy,iside,e)
      endif

      return
      end

      subroutine fixmska (c1mask,c2mask,c3mask)

c     fixes masks for A/SYM face corners

      include 'SIZE'
      include 'INPUT'
 
      real   c1mask(lx1,ly1,lz1,1)
     $      ,c2mask(lx1,ly1,lz1,1)
     $      ,c3mask(lx1,ly1,lz1,1)

      common /ctmp0/ im1(lx1,ly1,lz1),im2(lx1,ly1,lz1)
      integer e,f,val,im1,im2

      character*3 cb

      n = lx1*ly1*lz1

      nface = 2*ldim

      do e=1,nelv
         call izero (im1,n)
         call izero (im2,n)
         do f=1,nface
            cb  = cbc (f,e,1)
            if (cb.eq.'SYM')  call iface_e(im1,f,1,lx1,ly1,lz1)
            if (cb.eq.'A  ')  call iface_e(im2,f,2,lx1,ly1,lz1)
         enddo
         call icol2(im2,im1,n)

         k = 1
         do j=1,ly1,ly1-1
         do i=1,lx1,lx1-1
            if  ( im2(i,j,k) .eq. 2) then  ! corner of SYM & 'A  ' faces
               c1mask(i,j,k,e) = 0. 
               c2mask(i,j,k,e) = 0. 
            endif
         enddo
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine icol2(a,b,n)
      integer a(1),b(1)

      do i=1,n
         a(i)=a(i)*b(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine iface_e(a,iface,val,nx,ny,nz)

C     Assign the value VAL to face(IFACE,IE) of array A.
C     IFACE is the input in the pre-processor ordering scheme.

      include 'SIZE'
      integer a(nx,ny,nz),val
      call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
      do 100 iz=kz1,kz2
      do 100 iy=ky1,ky2
      do 100 ix=kx1,kx2
         a(ix,iy,iz)=val
  100 continue
      return
      end
c-----------------------------------------------------------------------
      function op_vlsc2_wt(b1,b2,b3,x1,x2,x3,wt)
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      real b1(1),b2(1),b3(1),x1(1),x2(1),x3(1),wt(1)

      nel = nelfld(ifield)
      n   = lx1*ly1*lz1*nel

      s = 0
      if (if3d) then
         do i=1,n
            s=s+wt(i)*(b1(i)*x1(i)+b2(i)*x2(i)+b3(i)*x3(i))
         enddo
      else
         do i=1,n
            s=s+wt(i)*(b1(i)*x1(i)+b2(i)*x2(i))
         enddo
      endif
      op_vlsc2_wt = s
      
      return
      end
c-----------------------------------------------------------------------
      function op_glsc2_wt(b1,b2,b3,x1,x2,x3,wt)
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      real b1(1),b2(1),b3(1),x1(1),x2(1),x3(1),wt(1)

      nel = nelfld(ifield)
      n   = lx1*ly1*lz1*nel

      s = 0
      if (if3d) then
         do i=1,n
            s=s+wt(i)*(b1(i)*x1(i)+b2(i)*x2(i)+b3(i)*x3(i))
         enddo
      else
         do i=1,n
            s=s+wt(i)*(b1(i)*x1(i)+b2(i)*x2(i))
         enddo
      endif
      op_glsc2_wt = glsum(s,1)
      
      return
      end
c-----------------------------------------------------------------------
