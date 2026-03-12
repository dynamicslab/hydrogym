      SUBROUTINE SSTEST (ISSS)
C------------------------------------------------------------------------
C
C     Test if Steady State Solver should be activated.
C
C------------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      ISSS = 0
      IADV = 0
      DO 100 IFIELD=1,NFIELD
         IF (IFADVC(IFIELD)) IADV = 1
 100  CONTINUE
      IF (.NOT.IFTRAN.AND.(IADV.EQ.1)) ISSS = 1
      IF (ISSS.EQ.1.AND.NFIELD.GT.4.AND.NID.EQ.0) THEN
         WRITE (6,*) ' '
         WRITE (6,*) 'Trying to activate the steady state solver'
         WRITE (6,*) 'using NFIELD =',NFIELD
         WRITE (6,*) 'Maximum number of fields is 4'
         call exitt
      ENDIF
      RETURN
      END
C
      SUBROUTINE SSINIT (KMAX)
C-----------------------------------------------------------
C
C     Initialize steady state solver
C
C-----------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'EIGEN'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
C
      IFTRAN    = .TRUE.
      IFCHAR    = .TRUE.
      CTARG     = 3.
      IF (lx1.GE.10) THEN
         CTARG = 5.
      ENDIF
      CALL SETPROP
      TAUMIN = 1.E20
      MFIELD = 1
      IF (.NOT.IFFLOW) MFIELD=2
      DO 10 IFIELD=MFIELD,NFIELD
         DIFFUS = AVDIFF(IFIELD)/AVTRAN(IFIELD)
         TAUSS(IFIELD)  = 1./(EIGAA*DIFFUS)
         TXNEXT(IFIELD) = TAUSS(IFIELD)
         IF (TAUSS(IFIELD).LT.TAUMIN) TAUMIN = TAUSS(IFIELD)
 10   CONTINUE
C
      NBDINP    = 1.
      TIME      = 0.
      DT        = 0.
      DTINIT    = TAUMIN/5.
      NSTEPS    = 10000
      IOSTEP    = 10000
C
                   IFMODP = .TRUE.
                   IFSKIP = .TRUE.
                   NSSKIP = 1
C
                   PRELAX = 1.E-1
      IF (IFSPLIT) PRELAX = 1.E-4
C
                   IFSSVT = .FALSE.
                   IFEXVT = .FALSE.
C
                   KMAX   = 5
C
      CALL SETCHAR
C
C
      RETURN
      END
C
      SUBROUTINE CHKEXT (IFACCX,Z,S)
C------------------------------------------------------------------
C
C     Accept extrapolation?
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'STEADY'
      LOGICAL IFACCX
      real*8 Z(1),S(1)
      REAL H1NRM1 (LDIMT1), H1NRM2(LDIMT1)
C
      CALL RZERO (H1NRM1,NFIELD)
      IF (IFFLOW) THEN
         IFIELD = 1
         CALL UNORM
         H1NRM1(IFIELD) = VNRMH1
      ENDIF
      DO 10 IFIELD=2,NFIELD
         CALL UNORM
         H1NRM1(IFIELD) = TNRMH1(IFIELD-1)
 10   CONTINUE
C
      CALL MKVEC (Z)
      CALL MKARR (S)
C
      CALL RZERO (H1NRM2,NFIELD)
      IF (IFFLOW) THEN
         IFIELD = 1
         CALL UNORM
         H1NRM2(IFIELD) = VNRMH1
      ENDIF
      DO 20 IFIELD=2,NFIELD
         CALL UNORM
         H1NRM2(IFIELD) = TNRMH1(IFIELD-1)
 20   CONTINUE
C
      XLIM   = .2
      IFACCX = .TRUE.
      RDMAX  = 0.
      RDLIM  = .5*TOLREL+1.E-4
      IF (IFFLOW) THEN
         IFIELD = 1
         RDIFF = ABS((H1NRM2(IFIELD)-H1NRM1(IFIELD))/H1NRM1(IFIELD))
         IF (RDIFF.GT.RDMAX) RDMAX = RDIFF
         IF (NIO.EQ.0) WRITE (6,*) ' ifield, rdiff ',ifield,rdiff
         IF (RDIFF.GT.XLIM) IFACCX = .FALSE.
      ENDIF
      DO 100 IFIELD=2,NFIELD
         RDIFF = ABS((H1NRM2(IFIELD)-H1NRM1(IFIELD))/H1NRM1(IFIELD))
         IF (RDIFF.GT.RDMAX) RDMAX = RDIFF
         IF (NIO.EQ.0) WRITE (6,*) ' ifield, rdiff ',ifield,rdiff
         IF (RDIFF.GT.XLIM) IFACCX = .FALSE.
 100  CONTINUE
C
      IF (.NOT.IFACCX) THEN
          IF (NIO.EQ.0) THEN
             WRITE (6,*) ' '
             write (6,*) 'Extrapolation attempt rejected'
             write (6,*) ' '
          ENDIF
          CALL MKARR (Z)
      ELSE
          IF (NIO.EQ.0) THEN
             write (6,*)  ' '
             write (6,*) 'Extrapolation accepted'
             write (6,*) ' '
          ENDIF
          IF (RDMAX.LT.RDLIM) IFSSVT = .TRUE.
          CALL FILLLAG
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE FILLLAG
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      NBDINP = 3
      IF (IFFLOW) THEN
         CALL LAGVEL
         CALL LAGVEL
      ENDIF
      IF (IFHEAT) THEN
         DO 100 IFIELD=2,NFIELD
            CALL LAGSCAL
            CALL LAGSCAL
 100     CONTINUE
      ENDIF
      RETURN
      END
C
      SUBROUTINE GONSTEP (N,ITEST)
C----------------------------------------------------------------
C
C     Do N steps; return if steady state
C
C----------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'STEADY'
      EXTERNAL GOSTEP
C
      DO 1000 JSTEP=1,N
         IF (ITEST.EQ.0.AND.IFSSVT) GOTO 1001
         IF (ITEST.EQ.1.AND.(IFSSVT.OR.IFEXVT)) GOTO 1001
         CALL GOSTEP
 1000 CONTINUE
 1001 CONTINUE
C
      RETURN
      END
C
      SUBROUTINE GO1STEP (X,Y,NVEC)
C----------------------------------------------------------------
C
C     Advance one (or more) time step(s)
C
C----------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      real*8 X(1), Y(1)
C
      CALL MKARR (X)
      IF (.NOT.IFSKIP) NJSTEP=1
      IF (     IFSKIP) NJSTEP=NSSKIP
C
      DO 9000 JSTEP=1,NJSTEP
C
      ISTEP = ISTEP+1
      CALL SETTIME
      CALL SETPROP
      IF (IFMODP) CALL MODPROP 
      CALL SETSOLV
      CALL COMMENT
      DO 100 IGEOM=1,2
         IF (IFGEOM) THEN
            CALL GENGEOM (IGEOM)
            CALL GENEIG  (IGEOM)
         ENDIF
         IF (IFFLOW) CALL FLUID (IGEOM)
         IF (IFHEAT) CALL HEAT  (IGEOM)
         IF (IFMVBD) CALL MESHV (IGEOM)
 100  CONTINUE
      CALL PREPOST(.false.)
      CALL USERCHK
C
 9000 CONTINUE
C
      IF (ISTEP.GT.1) CALL CHKSSVT
      CALL MKVEC (Y)
C
      RETURN
      END
C
      SUBROUTINE GOSTEP 
C----------------------------------------------------------------
C
C     Advance one (or more) time step(s)
C
C----------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
C
      IF (.NOT.IFSKIP) NJSTEP=1
      IF (     IFSKIP) NJSTEP=NSSKIP
C
      DO 9000 JSTEP=1,NJSTEP
C
      ISTEP = ISTEP+1
      CALL SETTIME
      CALL SETPROP
      IF (IFMODP) CALL MODPROP 
      CALL SETSOLV
      CALL COMMENT
      DO 100 IGEOM=1,2
         IF (IFGEOM) THEN
            CALL GENGEOM (IGEOM)
            CALL GENEIG  (IGEOM)
         ENDIF
         IF (IFFLOW) CALL FLUID (IGEOM)
         IF (IFHEAT) CALL HEAT  (IGEOM)
         IF (IFMVBD) CALL MESHV (IGEOM)
 100  CONTINUE
      CALL PREPOST(.false.)
      CALL USERCHK
C
 9000 CONTINUE
C
      IF (ISTEP.GT.1) CALL CHKSSVT
C
      RETURN
      END
C
      SUBROUTINE MODPROP
C------------------------------------------------------------------
C
C     Modify the properties
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
      INCLUDE 'INPUT'
C
      MFIELD=1
      IF (.NOT.IFFLOW) MFIELD=2
      DO 100 IFIELD=MFIELD,NFIELD
         NTOT  = lx1*ly1*lz1*NELFLD(IFIELD)
         TAU   = .02*TAUSS(IFIELD)
         DECAY = 1.+99.*EXP(-TIME/TAU)
         CALL CMULT (VDIFF(1,1,1,1,IFIELD),DECAY,NTOT)
c         if (nid.eq.0)
c     $   write (6,*) '.......... diff = ',IFIELD,vdiff(1,1,1,1,IFIELD)
 100  CONTINUE
C
      RETURN
      END
C
      SUBROUTINE MKVEC (X)
C-------------------------------------------------------------
C
C     Fill up the vector X with VX, VY, ....
C
C-------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      real*8 X(1)
C
      NTOTV = lx1*ly1*lz1*NELV
C
      IF (IFFLOW) THEN
         DO 100 I=1,NTOTV
            X(I) = VX(I,1,1,1)
 100     CONTINUE
         DO 200 I=1,NTOTV
            X(I+NTOTV) = VY(I,1,1,1)
 200     CONTINUE
         IF (ldim.EQ.3) THEN
            IOFF = 2*NTOTV
            DO 300 I=1,NTOTV
               X(I+IOFF) = VZ(I,1,1,1)
 300        CONTINUE
         ENDIF
      ENDIF
C
      IF (IFHEAT) THEN
         IOFF = ldim*NTOTV
         DO 401 IFIELD=2,NFIELD
            NTOT = lx1*ly1*lz1*NELFLD(IFIELD)
            DO 400 I=1,NTOT
               X(I+IOFF) = T(I,1,1,1,IFIELD-1)
 400        CONTINUE
            IOFF = IOFF+NTOT
 401     CONTINUE
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE MKARR (X)
C------------------------------------------------------------------
C
C     Split the vector X into VX, VY, .....
C
C------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      real*8 X(1)
C
      NTOTV = lx1*ly1*lz1*NELV
C
      IF (IFFLOW) THEN
         DO 10 I=1,NTOTV
            VX(I,1,1,1) = X(I)
 10      CONTINUE
         DO 20 I=1,NTOTV
            VY(I,1,1,1) = X(I+NTOTV)
 20      CONTINUE
         IF (ldim.EQ.3) THEN
            IOFF = 2*NTOTV
            DO 30 I=1,NTOTV
               VZ(I,1,1,1) = X(I+IOFF)
 30         CONTINUE
         ENDIF
      ENDIF
C
      IF (IFHEAT) THEN
         IOFF = ldim*NTOTV
         DO 41 IFIELD=2,NFIELD
            NTOT = lx1*ly1*lz1*NELFLD(IFIELD)
            DO 40 I=1,NTOT
               T(I,1,1,1,IFIELD-1) = X(I+IOFF)
 40         CONTINUE
            IOFF = IOFF+NTOT
 41      CONTINUE
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE SSPARAM (KMAX,L)
C------------------------------------------------------------------------------
C
C     Set steady state parameters
C
C------------------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'STEADY'
C
      IF (L.EQ.0) THEN
         CALL SSINIT (KMAX)
      ELSEIF (L.EQ.1) THEN
         ISTEP  = 0
C
         PRELAX = 1.E-2
         IF (IFSPLIT) PRELAX = 1.E-5
C
         CTARG  = 1.
         IF (IFSPLIT) CTARG  = 1.
         IF (lx1.GE.10) THEN
             CTARG  = 2.
             IF (IFSPLIT) CTARG = 2.
         ENDIF
C
         KMAX   = 5
         NBDINP = 3
         NSSKIP = 2
         IFSKIP = .TRUE.
         IFMODP = .FALSE.
C
      ELSEIF (L.EQ.2) THEN
C
         PRELAX = 1.E-3
         IF (IFSPLIT) PRELAX = 1.E-5
C
         CTARG = 1.
         IF (IFSPLIT) CTARG = 1.
         IF (lx1.GE.10) THEN
            CTARG = 2.
            IF (IFSPLIT) CTARG = 2.
         ENDIF
C
         KMAX   = 5
         NBDINP = 3
         NSSKIP = 2
         IFSKIP = .TRUE.
         IFMODP = .FALSE.
C
      ELSE
      ENDIF
      CALL SETCHAR
      RETURN
      END
C
      SUBROUTINE CHKSSVT
C-----------------------------------------------------------------------
C
C     Check for global steady state (velocity and temp/passive scalar)
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
C
      IF (IFFLOW) THEN
         IMESH  = 1
         IFIELD = 1
         CALL CHKSSV
      ENDIF
C
      IMESH = 2
      DO 100 IFIELD=2,NFIELD
         CALL CHKSST
 100  CONTINUE
C
      IFSSVT = .TRUE.
      IFEXVT = .TRUE.
      MFIELD = 1
      IF (.NOT.IFFLOW) MFIELD=2
      DO 200 IFIELD=MFIELD,NFIELD
         IF(.NOT.IFSTST(IFIELD)) IFSSVT = .FALSE.
         IF(.NOT.IFEXTR(IFIELD)) IFEXVT = .FALSE.
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE CHKSSV
C--------------------------------------------------------------------
C
C     Check steady state for velocity
C 
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'INPUT'
      INCLUDE 'EIGEN'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
      COMMON /CTOLPR/ DIVEX
      COMMON /CPRINT/ IFPRINT
      LOGICAL         IFPRINT
C
      COMMON /SCRSS2/ DV1 (LX1,LY1,LZ1,LELV)
     $ ,              DV2 (LX1,LY1,LZ1,LELV)
     $ ,              DV3 (LX1,LY1,LZ1,LELV)
      COMMON /SCRUZ/  W1  (LX1,LY1,LZ1,LELV)
     $ ,              W2  (LX1,LY1,LZ1,LELV)
     $ ,              W3  (LX1,LY1,LZ1,LELV)
     $ ,              BDIVV(LX2,LY2,LZ2,LELV)
      COMMON /SCRMG/  T1  (LX1,LY1,LZ1,LELV)
     $ ,              T2  (LX1,LY1,LZ1,LELV)
     $ ,              T3  (LX1,LY1,LZ1,LELV)
     $ ,              DIVV(LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/  H1  (LX1,LY1,LZ1,LELV)
     $ ,              H2  (LX1,LY1,LZ1,LELV)
C
      CALL OPSUB3 (DV1,DV2,DV3,VX,VY,VZ,VXLAG,VYLAG,VZLAG)
      CALL NORMVC (DVNNH1,DVNNSM,DVNNL2,DVNNL8,DV1,DV2,DV3)
      INTYPE = -1
      CALL SETHLM (H1,H2,INTYPE)
      CALL OPHX   (W1,W2,W3,DV1,DV2,DV3,H1,H2)
      CALL OPDSSUM(W1,W2,W3)
      CALL OPMASK (W1,W2,W3)
      CALL OPCOLV3(T1,T2,T3,W1,W2,W3,BINVM1)
      CALL OPHX   (W1,W2,W3,T1,T2,T3,H1,H2)
      CALL OPCOL2 (W1,W2,W3,DV1,DV2,DV3)
      NTOT1  = lx1*ly1*lz1*NELV
      USNRM1 = GLSUM(W1,NTOT1)
      USNRM2 = GLSUM(W2,NTOT1)
      USNRM3 = 0.
      IF (ldim.EQ.3) USNRM3 = GLSUM(W3,NTOT1)
      USNORM = SQRT( (USNRM1+USNRM2+USNRM3)/VOLVM1 )
C
      NTOT2 = lx2*ly2*lz2*NELV
      CALL OPDIV (BDIVV,VX,VY,VZ)
      CALL COL3 (DIVV,BDIVV,BM2INV,NTOT2)
      DNORM = SQRT(GLSC2(DIVV,BDIVV,NTOT2)/VOLVM2)
C
      TOLOLD = TOLPS
      CALL SETTOLV
      TOLHV3 = TOLHV*(ldim)
      IF (IFSTRS) TOLHV3 = TOLHV
      IF (NIO.EQ.0 .AND. IFPRINT) THEN
         WRITE (6,*) 'USNORM, TOLHV',USNORM,TOLHV3
         WRITE (6,*) 'DNORM, TOLPS',DNORM,TOLPS
      ENDIF
      IF (DNORM.GT.(1.1*DIVEX).AND.DIVEX.GT.0.
     $                        .AND.TOLPDF.EQ.0.) TOLPDF = 5.*DNORM
      USREL = USNORM/TOLHV3
      DREL  = DNORM/TOLPS
C
      IF (TOLREL.GT.0.) THEN
         EXFAC = .3/TOLREL
      ELSE
         WRITE (6,*) 'WARNING: TOLREL=0. Please modify *.rea'
         call exitt
      ENDIF
      IFEXTR(IFIELD) = .FALSE.
c      IF ((USREL.LT.EXFAC).OR.(TIME.GT.TXNEXT(IFIELD))) 
c     $                                       IFEXTR(IFIELD) = .TRUE.
      IF (USREL.LT.EXFAC)                    IFEXTR(IFIELD) = .TRUE.
      if (nio.eq.0 .and. ifprint)
     $WRITE (6,*) 'Tau, Txnext ',IFIELD,tauss(ifield),txnext(ifield)
C
      IFSTST(IFIELD) = .FALSE.
      USLIM = 2.*TOLHV3
      DLIM  = 2.*TOLPS
      IF (USNORM.LT.USLIM.AND.DNORM.LT.DLIM .AND. .NOT.IFSPLIT) 
     $                                       IFSTST(IFIELD) = .TRUE.
      IF (USNORM.LT.USLIM .AND. IFSPLIT) 
     $                                       IFSTST(IFIELD) = .TRUE.
C
      RETURN
      END
C
      SUBROUTINE CHKSST 
C----------------------------------------------------------------------
C
C     Check for steady state for temperature/passive scalar
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
      COMMON /SCRUZ/  DELTAT (LX1,LY1,LZ1,LELT)
     $ ,              WA     (LX1,LY1,LZ1,LELT)
     $ ,              WB     (LX1,LY1,LZ1,LELT)
      COMMON /SCRVH/  H1     (LX1,LY1,LZ1,LELT)
     $ ,              H2     (LX1,LY1,LZ1,LELT)
      COMMON /CPRINT/ IFPRINT
      LOGICAL         IFPRINT
C
      NTOT = lx1*ly1*lz1*NELT
      CALL SUB3 (DELTAT(1,1,1,1),T(1,1,1,1,IFIELD-1),
     $                           TLAG(1,1,1,1,1,IFIELD-1),NTOT)
      CALL NORMSC (DVNNH1,DVNNSM,DVNNL2,DVNNL8,DELTAT,IMESH)
      INTYPE = -1
      ISD    = 1
      CALL SETHLM (H1,H2,INTYPE)
      CALL AXHELM (WA,DELTAT,H1,H2,IMESH,ISD)
      CALL DSSUM  (WA,lx1,ly1,lz1)
      CALL COL2   (WA,TMASK(1,1,1,1,IFIELD-1),NTOT)
      CALL COL3   (WB,WA,BINTM1,NTOT)
      CALL AXHELM (WA,WB,H1,H2,IMESH,ISD)
      CALL COL2   (WA,DELTAT,NTOT)
      USNORM = SQRT(GLSUM(WA,NTOT)/VOLTM1)
C
      CALL SETTOLT
      IF (NIO.EQ.0 .AND. IFPRINT) 
     $WRITE (6,*) 'USNORM, TOLHT',USNORM,TOLHT(IFIELD)
      USREL = USNORM/TOLHT(IFIELD)
C
      IF (TOLREL.GT.0.) THEN
         EXFAC = .3/TOLREL
      ELSE
         WRITE (6,*) 'WARNING: TOLREL=0. Please modify *.rea'
         call exitt
      ENDIF
      IFEXTR(IFIELD) = .FALSE.
c      IF ((USREL.LT.EXFAC).OR.(TIME.GT.TXNEXT(IFIELD))) 
c     $                     IFEXTR(IFIELD) = .TRUE.
      IF (USREL.LT.EXFAC)  IFEXTR(IFIELD) = .TRUE.
      IF(NIO.EQ.0 .AND. IFPRINT) 
     $WRITE (6,*) 'Tau, Txnext ',IFIELD,tauss(ifield),txnext(ifield)
C
      IFSTST(IFIELD) = .FALSE.
      USLIM = 2.*TOLHT(IFIELD)
      IF (USNORM.LT.USLIM) IFSTST(IFIELD) = .TRUE.
C
      RETURN
      END
C
      SUBROUTINE SSNORMD (DV1,DV2,DV3)
      INCLUDE 'SIZE'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
      REAL DV1(1),DV2(1),DV3(1)
      CALL NORMVC (DVDFH1,DVDFSM,DVDFL2,DVDFL8,DV1,DV2,DV3)
      RETURN
      END
C
      SUBROUTINE SSNORMP (DV1,DV2,DV3)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'STEADY'
      REAL DV1(1),DV2(1),DV3(1)
      CALL NORMVC (DVPRH1,DVPRSM,DVPRL2,DVPRL8,DV1,DV2,DV3)
      RETURN
      END
C
      SUBROUTINE SETTOLV
C-------------------------------------------------------------------
C
C     Set tolerances for velocity solver
C
C-------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'EIGEN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      INCLUDE 'SOLN'
      REAL LENGTH
C
      NTOT   = lx1*ly1*lz1*NELFLD(IFIELD)
      AVVISC = GLMIN(VDIFF(1,1,1,1,IFIELD),NTOT)
      AVDENS = GLMAX(VTRANS(1,1,1,1,IFIELD),NTOT)
C
      IF (IFTRAN) THEN
         IF (ISTEP.EQ.1)  VNORM = VNRML8
         IF (ISTEP.GT.1)  VNORM = VNRMSM
         IF (VNORM.EQ.0.) VNORM = TOLABS
         FACTOR = 1.+(AVDENS/(EIGAA*AVVISC*DT))
      ELSE
         VNORM = VNRML8
         IF (VNORM.EQ.0.) VNORM = TOLABS
         FACTOR = 1.
      ENDIF
C
      TOLPS  = TOLREL*VNORM * SQRT(EIGAS)/(4.*FACTOR) 
      TOLHV  = TOLREL*VNORM * SQRT(EIGAA)*AVVISC/2.
      TOLHV  = TOLHV/3.
      IF (.NOT.IFTRAN .AND. .NOT.IFNAV) TOLHV = TOLHV/10.
      TOLHR  = TOLHV
      TOLHS  = TOLHV
C
C     Non-zero default pressure tolerance
C     NOTE: This tolerance may change due to precision problems.
C           See subroutine CHKSSV
C
      IF (TOLPDF.NE.0.) TOLPS = TOLPDF
C
      RETURN
      END
C
      SUBROUTINE SETTOLT
C-------------------------------------------------------------------
C
C     Set tolerances for temerature/passive scalar solver
C
C-------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'EIGEN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      INCLUDE 'SOLN'
      REAL LENGTH
C
      NTOT   = lx1*ly1*lz1*NELFLD(IFIELD)
      AVCOND = GLMIN (VDIFF(1,1,1,1,IFIELD),NTOT)
C
      IF (IFTRAN) THEN
         IF (ISTEP.EQ.1)  TNORM = TNRML8(IFIELD-1)
         IF (ISTEP.GT.1)  TNORM = TNRMSM(IFIELD-1)
         IF (TNORM.EQ.0.) TNORM = TOLABS
      ELSE
         TNORM = TNRML8(IFIELD-1)
         IF (TNORM.EQ.0.) TNORM = TOLABS
      ENDIF
C
      TOLHT(IFIELD) = TOLREL*TNORM * SQRT(EIGAA)*AVCOND
C
      RETURN
      END
C
      SUBROUTINE CHKTOLP (TOLMIN)
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      COMMON /SCRMG/ DIVFLD (LX2,LY2,LZ2,LELV)
     $ ,             WORK   (LX2,LY2,LZ2,LELV)
      NTOT2 = lx2*ly2*lz2*NELV
      CALL OPDIV   (DIVFLD,VX,VY,VZ)
      CALL COL3    (WORK,DIVFLD,BM2INV,NTOT2)
      CALL COL2    (WORK,DIVFLD,NTOT2)
      DIVV  = SQRT(GLSUM(WORK,NTOT2)/VOLVM2)
C
      IFIELD = 1
      CALL SETTOLV
      TOLMIN = DIVV/100.
      IF (TOLMIN.LT.TOLPS) TOLMIN = TOLPS
      RETURN
      END
C
      SUBROUTINE SETCHAR
C-----------------------------------------------------------------------
C
C     If characteristics, need number of sub-timesteps (DT/DS).
C     Current sub-timeintegration scheme: RK4.
C     If not characteristics, i.e. standard semi-implicit scheme,
C     check user-defined Courant number.
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'

      REAL MAX_CFL_RK4
      DATA MAX_CFL_RK4 /2.0/ ! stability limit for RK4 including safety factor
C
      IF (IFCHAR) THEN
         ICT    = MAX(INT(CTARG/MAX_CFL_RK4),1)
         NTAUBD = ICT
         DCT    = CTARG - ICT*MAX_CFL_RK4
         IF (DCT.GT.0.1) NTAUBD = NTAUBD+1
         IF (NIO.EQ.0) write(6,*) 'RK4 substeps:', ntaubd 
      ELSE
         NTAUBD = 0
         IF (CTARG.GT.0.5) THEN
            IF (NIO.EQ.0)
     $      WRITE (6,*) 'Reset the target Courant number to .5'
            CTARG = 0.5
         ENDIF
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE PROJECT
C--------------------------------------------------------------------
C
C     Project current solution onto the closest incompressible field
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      COMMON /SCRNS/ W1    (LX1,LY1,LZ1,LELV)
     $ ,             W2    (LX1,LY1,LZ1,LELV)
     $ ,             W3    (LX1,LY1,LZ1,LELV)
     $ ,             DV1   (LX1,LY1,LZ1,LELV)
     $ ,             DV2   (LX1,LY1,LZ1,LELV)
     $ ,             DV3   (LX1,LY1,LZ1,LELV)
     $ ,             RESPR (LX2,LY2,LZ2,LELV)
      COMMON /SCRVH/ H1    (LX1,LY1,LZ1,LELV)
     $ ,             H2    (LX1,LY1,LZ1,LELV)
C
      IF (NIO.EQ.0) WRITE(6,5)
    5 FORMAT(/,'  Project',/)
C
      NTOT1  = lx1*ly1*lz1*NELV
      NTOT2  = lx2*ly2*lz2*NELV
      INTYPE = 1
      CALL RZERO   (H1,NTOT1)
      CALL RONE    (H2,NTOT1)
      CALL OPDIV   (RESPR,VX,VY,VZ)
      CALL CHSIGN  (RESPR,NTOT2)
      CALL ORTHO   (RESPR)
      CALL UZAWA   (RESPR,H1,H2,INTYPE,ICG)
      CALL OPGRADT (W1,W2,W3,RESPR)
      CALL OPBINV  (DV1,DV2,DV3,W1,W2,W3,H2)
      CALL OPADD2  (VX,VY,VZ,DV1,DV2,DV3)
      RETURN
      END
