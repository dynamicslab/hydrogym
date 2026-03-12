      SUBROUTINE SRFPLOT
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INTEGER IERR
      INTEGER ICALLD,LOPEN
      SAVE    ICALLD,LOPEN
      DATA    ICALLD,LOPEN /0,0/
C
      NXOLD=NX
      if (icalld.eq.0) then
         NOPEN=0
         icalld=1
      endif
      NOPEN=NOPEN+1
      NOPEN=MAX(1,NOPEN)
C
      if (nopen.ne.lopen) then
         write(6,*) 'calling getsrf',nopen
         CALL GETSRF(IERR,NOPEN)
      endif
      lopen=nopen
C
      IF (IERR.EQ.2) RETURN
C
      CALL SETWRKS
C
C     Auto clear stuff
C
      CALL CLEAR
      CALL HARD
      CALL HEJECT
      CALL DRMESH
      WRITE(S,80) TIME
   80 FORMAT('TIME = ',F5.2,'$')
      CALL GWRITE(XPHY(0.5),YPHY(0.95),3.0,S)
C
      CALL TEMSRF
      CALL NOHARD
C
      RETURN
      END
      SUBROUTINE TEMSRF
C
C     Surface plotting routines for surface (not volume) based data
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      LOGICAL IFCRV(4)
      PARAMETER (NXM2=NXM*NYM)
      COMMON /SCRTCH/ TPLOT(NXM2),XPLOT(NXM2),YPLOT(NXM2)
     $               ,ZPLOT(NXM2)
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
C     Write plot labels
C
      IF (.NOT.IFDEMO) CALL SRFLABL(IERR)
      IF (IERR.EQ.1) RETURN
      CALL PENW(1)
C
C     3-D
C     Sort the selected list of planes according to visibility
C
      CALL SORTS(NLSTK)
C
      DO 5000 I=1,NLSTK
C
C        New plotting algorithms for color fill and fishnet - pff 7-1-90
C        (3D will come later - 10-10-90)
C
         LISTA=ABS(LIST(I))
         CALL DECOD(IPLANE,IPLN,IE,ILIST,LISTA,NX,6,NSRFM)
         I1=1
         I2=NX
         J1=1
         J2=NY
         K1=1
         K2=NZ
         IF (IPLN.LE.2) THEN
            I1=IPLANE
            I2=IPLANE
         ELSEIF (IPLN.LE.4) THEN
            J1=IPLANE
            J2=IPLANE
         ELSE
            K1=IPLANE
            K2=IPLANE
         ENDIF
         KTR=1
         IF (SCALPT.EQ.'CONTOUR LINES') KTR=0
         IFCRV(1)=.TRUE.
         IFCRV(2)=.TRUE.
         IFCRV(3)=.TRUE.
         IFCRV(4)=.TRUE.
         WSCALE=1.0
         IF (WKMAX.NE.WKMIN) WSCALE=1.0/(WKMAX-WKMIN)
         IF(IFUSCL.AND.USRMIN.NE.USRMAX)WSCALE=1.0/(USRMAX-USRMIN)
         CURMIN=WKMIN
         IF(IFUSCL) CURMIN=USRMIN
C
         II  =0
         IOFF=(ILIST-1)*NX*NZ
C
c        write(6,*) 'ioff1:',ioff,ilist,nx,nz
c        write(6,*) 'ioff2:',lista,nlstk,nlstp,ie
c        write(6,*) 'ioff3:',list(i),i
C
c        write(7,*) 'ioff1:',ioff,ilist,nx,nz
c        write(7,*) 'ioff2:',lista,nlstk,nlstp,ie
c        write(7,*) 'ioff3:',list(i),i
         DO 5001 IZ=K1,K2
         DO 5001 IY=J1,J2
         DO 5001 IX=I1,I2
            IPOINT=IND(IX,IY,IZ,IE)
            II=II+1
            IOFF=IOFF+1
            XPLOT(II)=XP(IPOINT)
            YPLOT(II)=YP(IPOINT)
            ZPLOT(II)=ZP(IPOINT)
c           write(6,*) 'problem:',ii,iplot,ipoint,ioff
c           write(7,*) 'problem:',ii,iplot,ipoint,ioff
            TPLOT(II)=WSCALE*(WORK(IOFF)-CURMIN)
            IF (IFUSCL) THEN
               TPLOT(II)=MIN(TPLOT(II),1.0)
               TPLOT(II)=MAX(TPLOT(II),0.0)
            ENDIF
 5001    CONTINUE
c
         IF (SCALPT.EQ.'FISHNET GRID') THEN
            INRM=1
            IF (LIST(I).LT.0) INRM=-1
            CALL HEATF(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,INRM)
         ELSE
            CALL HEATC(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,KTR)
         ENDIF
 5000 CONTINUE
C
      RETURN
      END
      SUBROUTINE SRFLABL(IERR)
C
C     Write plot labels
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
      CHARACTER LABEL1*10,LABEL2*10,LABEL3*6
      PARAMETER (NXM2=NXM*NYM)
      CHARACTER CMIN*20
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
      IF(ICALLD.NE.1.OR.IFNOSEG)THEN
         IF(IFNOSEG)CALL DRCOVR(13)
         ICALLD=1
C        DRBAR Draws the color bar onto segment 12
         CALL DRBAR(TX,15)
      ELSE
C        Disappear menu and cover; appear Color bar.
         CALL SGVIS(13,0)
         CALL SGVIS(14,0)
         CALL SGVIS(15,0)
         CALL SGVIS(15,1)
      ENDIF
C
      IERR=0
      IF (WKMAX.EQ.WKMIN) THEN
         IERR=1
c        write(6,*) 'in srflabl'
         IF(QUANTY.EQ.'PRESSURE') THEN
             CALL PRSRS
     $       ('*****Constant Field; P=$',WKMAX,'********$')
         ELSE IF(QUANTY(1:14).EQ.'PASSIVE SCALAR') THEN
             CALL PRSRS
     $       ('*****Constant Field; PS=$',WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'TEMPERATURE') THEN
             CALL PRSRS
     $       ('*****Constant Field; T=$',WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'STREAMFUNCTION') THEN
             CALL PRSRS
     $       ('*****Constant Field; PSI=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VORTEX')THEN
             CALL PRSRS
     $       ('*****Constant Field; VORTEX=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'V.W')THEN
             CALL PRSRS
     $       ('*****Constant Field; V.W=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VORTICITY')THEN
             CALL PRSRS
     $       ('*****Constant Field; VORTICITY=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VELOCITY') THEN
             CALL PRSRS(
     $       '*****Constant Field; '//COMPON//' VELOCITY=$'
     $       ,WKMAX,'********$')
         ENDIF
         RETURN
      ENDIF
C     Draw box to overwrite any old labels
      YTOP =0.10
      YBOT =0.06
      YTOP2=0.75
      YBOT2=0.71
      CALL FILLP(-2)
      CALL BEGINB(XPHY(1.01),YPHY(YTOP))
      CALL MOVESC(     1.29 ,     YTOP)
      CALL MOVESC(     1.29 ,     YBOT)
      CALL MOVESC(     1.01 ,     YBOT)
      CALL MOVESC(     1.01 ,     YTOP)
      CALL ENDP
      CALL FILLP(-14)
      CALL BEGINB(XPHY(1.01),YPHY(YTOP2))
      CALL MOVESC(     1.29 ,     YTOP2)
      CALL MOVESC(     1.29 ,     YBOT2)
      CALL MOVESC(     1.01 ,     YBOT2)
      CALL MOVESC(     1.01 ,     YTOP2)
      CALL ENDP
C
      IF(QUANTY.EQ.'VELOCITY') THEN
         WRITE(LABEL1,'(''V'',A1,''min$'')')COMPON
         WRITE(LABEL2,'(''V'',A1,''max$'')')COMPON
         IF(COMPON.EQ.'S')WRITE(LABEL1(1:2),'(''Sp'')')
         IF(COMPON.EQ.'S')WRITE(LABEL2(1:2),'(''Sp'')')
      ELSE IF(QUANTY.EQ.'PRESSURE') THEN
         LABEL1 = 'Pmin=$'
         LABEL2 = 'Pmax=$'
      ELSE IF(QUANTY.EQ.'VORTEX') THEN
         LABEL1 = 'VORTEXmn=$'
         LABEL2 = 'VORTEXmx=$'
      ELSE IF(QUANTY.EQ.'VORTICITY') THEN
      ELSE IF(QUANTY.EQ.'V.W') THEN
         LABEL1 = 'V.Wmn=$'
         LABEL2 = 'V.Wmx=$'
      ELSE IF(QUANTY.EQ.'VORTICITY') THEN
         LABEL1 = 'VRmin=$'
         LABEL2 = 'VRmax=$'
      ELSE IF(QUANTY.EQ.'TOTAL PRESSURE') THEN
         LABEL1 = 'T Pmin=$'
         LABEL2 = 'T Pmax=$'
      ELSE IF(QUANTY.EQ.'STREAMFUNCTION') THEN
         LABEL1 = 'PSImin=$'
         LABEL2 = 'PSImax=$'
      ELSE IF(QUANTY.EQ.'DIVERGENCE') THEN
         LABEL1 = 'DIVmin=$'
         LABEL2 = 'DIVmax=$'
      ELSE  IF(QUANTY.EQ.'TEMPERATURE') THEN
         LABEL1 = 'Tmin=$'
         LABEL2 = 'Tmax=$'
      ELSE  IF(QUANTY(1:14).EQ.'PASSIVE SCALAR') THEN
         LABEL1 = 'PSmin=$'
         LABEL2 = 'PSmax=$'
         LABEL3 = PSNAME(LPSDMP)
         LABEL3(6:6)='$'
         CALL GWRITE(XPHY(0.7),YPHY(0.95),1.0,LABEL3)
      ENDIF
C
      WRITE(CMIN,'(G11.4,''$'')')WKMIN
      CALL GWRITE(XPHY(1.10+.034),YPHY(0.07),1.0,CMIN)
      CALL GWRITE(XPHY(1.02),YPHY(0.07),1.0,LABEL1)
      WRITE(CMIN,'(G11.4,''$'')')WKMAX
      CALL GWRITE(XPHY(1.10+.034),YPHY(0.72),1.0,CMIN)
      CALL GWRITE(XPHY(1.02),YPHY(0.72),1.0,LABEL2)
      IF(DERIV.NE.'NO')THEN
         LABEL2=DERIV//'$'
         CALL GWRITE(XPHY(1.02),YPHY(0.76),1.0,LABEL2)
         CALL GWRITE(XPHY(1.02),YPHY(0.11),1.0,LABEL2)
      ENDIF
C
      RETURN
      END
      SUBROUTINE EXTPLN2(XX,XPLOT,NOUT,IPLN,Idum)
C
C     Extract a plane of data from array X and place into
C     output array work, with stride NOUT.
C
      INCLUDE 'basics.inc'
      DIMENSION XPLOT(NOUT,1),XX(1)
C
      II=0
      NXY=NX*NY
      DO 5001 IX=1,NXY
         II=II+1
         XX(IX)=XPLOT(1,II)
 5001 CONTINUE
      RETURN
      END
      SUBROUTINE SETWRKS
C
C     Set up WORK array according to current plotting state.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ WORK1(LXYZ),WORK2(LXYZ)
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
      NTOT = NLSTP*NX*NZ
C
c     write(6,*) 'in setwrks',icalld,quanty,compon
c     IF(QUANTY.EQ.'VELOCITY'.OR.ICALLD.EQ.0)THEN
      IF(QUANTY.EQ.'VELOCITY')THEN
         ICALLD=1
         CALL COPY(WKV1,U,NTOT)
         CALL COPY(WKV2,V,NTOT)
         IF (IF3D) THEN
            CALL COPY (WKV3,W,NTOT)
         ELSE
            CALL RZERO(WKV3,W,NTOT)
         ENDIF
      ENDIF
      IF(QUANTY.EQ.'VORTICITY')THEN
         IF (IF3D) THEN
c           CALL COPY(WKV1,WKV1,NTOT)
c           CALL COPY(WKV2,WKV2,NTOT)
c           CALL COPY(WKV3,WKV3,NTOT)
         ELSE
            CALL COPY(WORK,WKV1,NTOT)
         ENDIF
      ENDIF
      IF (QUANTY.EQ.'VELOCITY'.OR.
     $   (QUANTY.EQ.'VORTICITY'.AND.IF3D))THEN
C
C        Compute min speed and max speed for all choices
         SPMIN= 10.0E10
         SPMAX=-10.0E10
         DO 10 I=1,NTOT
            WORK(I)=SQRT(WKV1(I)**2+WKV2(I)**2+WKV3(I)**2)
            SPMIN=MIN(SPMIN,WORK(I))
            SPMAX=MAX(SPMAX,WORK(I))
   10    CONTINUE
         UVMAX=SPMAX
C
         IF(COMPON.EQ.'X')CALL COPY(WORK,WKV1,NTOT)
         IF(COMPON.EQ.'Y')CALL COPY(WORK,WKV2,NTOT)
         IF(COMPON.EQ.'Z')CALL COPY(WORK,WKV3,NTOT)
         IF(COMPON.EQ.'N')THEN
C        Normal to specified line
             AMPLIN = SQRT(  (XLINE(1)-XLINE(2))**2
     $              +        (YLINE(1)-YLINE(2))**2
     $              +        (ZLINE(1)-ZLINE(2))**2 )
             IF (AMPLIN.NE.0.0) THEN
                XTWDL = (XLINE(2)-XLINE(1)) / AMPLIN
                YTWDL = (YLINE(2)-YLINE(1)) / AMPLIN
                ZTWDL = (ZLINE(2)-ZLINE(1)) / AMPLIN
             ELSE
                CALL PRS
     $           ('Error: you must specify a line segment along $')
                CALL PRS('which to plot profile before plotting.$')
                RETURN
             ENDIF
             IF (.NOT.IF3D) THEN
c               SPMIN =  10.0E10
c               SPMAX = -10.0E10
                DO 20 I=1,NTOT
                   WORK(I)=(-WKV1(I)*YTWDL+WKV2(I)*XTWDL)
c                  SPEED  =(WKV1(I)**2+WKV2(I)**2)
c                  SPMIN  = MIN(SPEED,SPMIN)
c                  SPMAX  = MAX(SPEED,SPMAX)
   20           CONTINUE
c               SPMIN  = SQRT(SPMIN)
c               SPMAX  = SQRT(SPMAX)
             ELSE
C               Kludge: put the speed in here
                DO 30 I=1,NTOT
                   WORK(I)=SQRT(WKV1(I)**2+WKV2(I)**2+WKV3(I)**2)
   30           CONTINUE
c               DO 30 I=1,2
C                DO 163 I=1,NTOT
C                   WRK1 = (U(I)*YTWDL-V(I)*XTWDL)
C ?????                   WRK2 = (U(I)*YTWDL-V(I)*XTWDL)
C                   WRK1 =(U(I)*YTWDL-V(I)*XTWDL)
C                   WRK1 =(U(I)**2+V(I)**2+W(I)**2)
C                   WRK2 = (XTWDL*U(I)+YTWDL*V(I)+ZTWDL*W(I))**2
C                   WORK(I)=SQRT(ABS(WRK1-WRK2))
C                 IF (WRK2.GT.WRK1) WORK(I)=-WORK(I)
c  30           CONTINUE
             ENDIF
         ENDIF
      ENDIF
      IF(QUANTY .EQ.'STREAMFUNCTION' ) THEN
c       CALL STRFCT(WORK)
         WRITE(6,35)
   35    FORMAT(' STREAMFUNCTION not supported for surface viewing.$')
         CALL PRS(S)
      ENDIF
      IF(QUANTY.EQ.'DIVERGENCE')THEN
         WRITE(6,40)
   40    FORMAT(' DIVERGENCE not supported for surface viewing.$')
         CALL PRS(S)
C        WORK1=DV/DX ; WORK2=DU/DY
c        NXYZ  = NX*NY*NZ
c        IEOFF = 0
c        DO 50 IEL=1,NEL
c          CALL DUDXYZ(WORK1,U,RXM1,SXM1,TXM1,JACM1,IEL)
c          CALL DUDXYZ(WORK2,V,RYM1,SYM1,TYM1,JACM1,IEL)
c          DO 40 I=1,NXYZ
c             IX = IEOFF + I
c             WORK(IX) = ( WORK1(I)+WORK2(I) )
c  40      CONTINUE
c          IEOFF = IEOFF + NXYZ
c  50    CONTINUE
      ENDIF
      IF(QUANTY.EQ.'TEMPERATURE' )CALL COPY(WORK,T,NTOT)
      IF(QUANTY.EQ.'PRESSURE'    )CALL COPY(WORK,P,NTOT)
      IF(QUANTY.EQ.'TOTAL PRESSURE'    )THEN
         RHO2=PARAM(1)/2.0
         DO 75 I=1,NTOT
            WORK(I)=P(I)+RHO2*(U(I)**2+V(I)**2+W(I)**2)
   75    CONTINUE
      ENDIF
      IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')THEN
         IF(LPSDMP.EQ.1)CALL COPY(WORK,WKV1,NTOT)
         IF(LPSDMP.EQ.2)CALL COPY(WORK,WKV2,NTOT)
         IF(LPSDMP.EQ.3)CALL COPY(WORK,WKV3,NTOT)
      ENDIF
C
      NXYZ  = NX*NY*NZ
      DO 100 IEL=1,NEL
          IF(DERIV.EQ.'D/DX')
     $    CALL DUDXYZ(WORK1,WORK,RXM1,SXM1,TXM1,JACM1,IEL)
          IF(DERIV.EQ.'D/DY')
     $    CALL DUDXYZ(WORK1,WORK,RYM1,SYM1,TYM1,JACM1,IEL)
          IF(DERIV.EQ.'D/DZ')
     $    CALL DUDXYZ(WORK1,WORK,RZM1,SZM1,TZM1,JACM1,IEL)
C
          IEOFF = NXYZ*(IEL-1)+1
          IF(DERIV.EQ.'D/DX'.OR.DERIV.EQ.'D/DY'.OR.DERIV.EQ.'D/DZ')
     $    CALL COPY(WORK(IEOFF),WORK1,NXYZ)
  100 CONTINUE
C
C     Compute range for WORK array
C
c     write(6,*) 'this is ntot:',ntot,icalld
      IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
         WKMAX = SPMAX
         WKMIN = SPMIN
         WKDLT = WKMAX-WKMIN
      ELSE
         WKMAX = GLMAX(WORK,NTOT)
         WKMIN = GLMIN(WORK,NTOT)
         WKDLT = WKMAX-WKMIN
      ENDIF
      IF (WKDLT.EQ.0.0) THEN
         WRITE(S,1000) WKMAX,WKMIN
 1000    FORMAT('WORK - MAX,MIN:',2G14.4,'$')
         CALL PRS(S)
      ENDIF
C
      TTMIN=WKMIN
      TTMAX=WKMAX
      RETURN
      END
      SUBROUTINE DBLANK(STRING,N)
      CHARACTER*1 STRING(1)
      CHARACTER*1 BLNK
      SAVE        BLNK
      DATA        BLNK /' '/
C
C     Remove leading blanks from string
C
      DO 10 I=1,N
         IF (STRING(I).NE.BLNK) GOTO 11
   10 CONTINUE
   11 CONTINUE
      NBLANK=I-1
      IF (NBLANK.EQ.0) RETURN
C
C     Shift the string to the left
C
      NLEFT=N-NBLANK
      J=NBLANK
      DO 20 I=1,NLEFT
         J=J+1
         STRING(I)=STRING(J)
   20 CONTINUE
C
C     Fill the tail with blanks
C
      J=NLEFT
      DO 30 I=1,NBLANK
         J=J+1
         STRING(J)=BLNK
   30 CONTINUE
      RETURN
      END
      SUBROUTINE SORTS(NLSTK)
C
C     Sort plotting surfaces by zdepth
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /CTMPS/  SCRNAM(NELM,6) , IND(NELM,6)
      CHARACTER*16 SCRNAM
      CHARACTER*16 NAMBLK
      SAVE         NAMBLK
C
      CALL BLANK(NAMBLK,16)
      NXY =NX*NY
      NXYZ=NZ*NXY
      J=0
      K=0
      DO 100 I=1,NLSTP
         If (NAMSEL.EQ.NAMBLK.or.NAMSEL.EQ.NAMELS(i)) THEN
C
C           Name is in the selected name list
C
C           Get an XYZ point close to the center of the requested sub-plane.
            LISTA=ABS(LIST(I))
            CALL DECOD(IPLANE,IPLN,IE,ILIST,LISTA,NX,6,NSRFM)
            IX=(NX+1)/2
            IY=(NY+1)/2
            IZ=(NZ+1)/2
            IF (IPLN.LE.2) THEN
               IX=IPLANE
            ELSEIF (IPLN.LE.4) THEN
               IY=IPLANE
            ELSE
               IZ=IPLANE
            ENDIF
            IEOFF=NXYZ*(IE-1)+NXY*(IZ-1)+NX*(IY-1)+IX
            K=K+1
            ZDEPTH(k,1)=ZISO(XP(IEOFF),YP(IEOFF),ZP(IEOFF))
            IZDPTH(k,1)=LIST(I)
            if (k.gt.nelm*6) write(6,*) 'WARNING IN SUBR. SORTS, k=',k
            SCRNAM(k,1)  =NAMELS(I)
         ELSE
            IZDPTH(NLSTP-J,1)=LIST(I)
           nj=nlstp-j
           if (nj.gt.nelm*6) write(6,*) 'WARNING IN SUBR. SORTS, j=',nj
            SCRNAM(NLSTP-J,1)  =NAMELS(I)
            J=J+1
         ENDIF
100   CONTINUE
      NLSTK=K
      CALL ICOPY(LIST  ,IZDPTH,NLSTP)
      CALL CHCOPY(NAMELS,SCRNAM,NLSTP*16)
      CALL SORT  (ZDEPTH ,IND,NLSTK)
      CALL ISWAP (LIST  ,SCRNAM,IND,NLSTK)
      CALL CHSWAP(NAMELS,SCRNAM,16,IND,NLSTK)
C
      RETURN
      END
      SUBROUTINE GETSRF(IERR,mOPEN)
C
C     Get the surface data
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      INTEGER IERR
      CHARACTER*20  FRMAT
      CHARACTER*80  FILE1
      CHARACTER*6   STUFF
      CHARACTER*80  STUFF80
      CHARACTER*80 HEADER
      CHARACTER*1  HEADR1(80)
      CHARACTER*6   VFRMAT
      EQUIVALENCE (HEADER,HEADR1)
      EQUIVALENCE (STUFF,STUFF80)
      LOGICAL IFSRF
C
      integer icalld
      save    icalld
      data    icalld/0/
      icalld=icalld+1
C
C     Get the current surface data
C
      IUNIT=37
      ILIST=0
      IOFF=1
      NXY=NX*NY
      NPMAX=NEL
      IFSRF=.FALSE.
      DO 9000 JNID=0,NPMAX-1
C
C     First, do we have any files out there?
C
         CALL SRFILE(IERR,FILE1,JNID,IUNIT,mOPEN)
         IF (IERR.EQ.1) GOTO 9000
         IFSRF=.TRUE.
C
C        OK, We opened a file, let's get the data
C
         DO 8000 ISRF=1,10000
            PLTMOD='SURFACE'
C
            READ(IUNIT,80,END=9000) HEADER
   80       FORMAT(A80)
            READ(IUNIT,*,END=9000)  NXs,NYs,NZs
     $                             ,IPLN,IPLANE,IEG,NWORDS,NID
C
C           Reset NX according to the size in the .srf file
            NX=PARAM(20)
            IF (NX.NE.NXs) THEN
               NX=NXs
               NY=NXs
               NZ=NXs
               NXY=NX*NY
               PARAM(20)=NXs
               WRITE(S,81) NX,NY,NZ
               CALL PRS(S)
   81          FORMAT(' Resetting nx,ny,nz:',3I3,'.$')
            ENDIF
C
            ILIST=ILIST+1
            IOFF=1+(ILIST-1)*NXs*NZs
            LIST(ILIST) = NSRFM*6*NXs*(ILIST-1)
     $                   + 6*NXs*(IEG-1)+NXs*(IPLN-1)+IPLANE
            itwo31=2**30
C
            if (ilist.gt.nsrfm) then
               write(6,230) list,nsrfm
               stop
  230          format(' Error in getsrf, too many srfs:',2i9,' Exit.')
            endif
C
            if (list(ilist).gt.itwo31) then
               write(6,231) ilist,list(ilist)
               stop
  231          format(' Error in getsrf, list:',2i16,' Exit.')
            endif
C
c     write(6,7) list(ilist),ilist,NSRFM,nxs,ieg,ipln,iplane,ioff
c     write(7,7) list(ilist),ilist,NSRFM,nxs,ieg,ipln,iplane,ioff
c   7 format('listi',i14,7i8)
C
C
C-------------------------------------------------------
C           Extract header information:
C-------------------------------------------------------
C
C           Surface Format
C
            I1=INDX1(HEADER,'F=',2)+2
            I2=INDX1(HEADR1(I1),';',1)-1
            CALL CHCOPY(FRMAT,HEADR1(I1),I2)
C
C           Surface Data
C
            I1=INDX1(HEADER,'D=',2)+2
            I2=INDX1(HEADR1(I1),';',1)-1
            CALL BLANK(STUFF80,80)
            CALL CHCOPY(STUFF,HEADR1(I1),I2)
C
C           Surface Name
C
            I1=INDX1(HEADER,'N=',2)+2
            I2=INDX1(HEADR1(I1),';',1)-1
            CALL BLANK(NAMELS(ilist),16)
            CALL CHCOPY(NAMELS(ilist),HEADR1(I1),I2)
C
C           Time
C
            I1=INDX1(HEADER,'T=',2)+2
            I2=INDX1(HEADR1(I1),';',1)-1
            TIME=REALCHR(HEADR1(I1),I2)
C
C           Step number
C
            I1=INDX1(HEADER,'I=',2)+2
            I2=INDX1(HEADR1(I1),';',1)-1
            ISTEP=INTCHR(HEADR1(I1),I2)
C
            NOUT=0
            IOUT=0
            IF (INDX1(STUFF,'X',1).NE.0) NOUT=NOUT+NDIM
            IF (INDX1(STUFF,'V',1).NE.0) NOUT=NOUT+NDIM
            IF (INDX1(STUFF,'P',1).NE.0) NOUT=NOUT+1
            IF (INDX1(STUFF,'T',1).NE.0) NOUT=NOUT+1
            IF (INDX1(STUFF,'W',1).NE.0) NOUT=NOUT+3*(NDIM-2)+(3-NDIM)
C
C
C           Read the data!
C
C           (Note, it's ok to use WORK here, as setwrk is called later.
            IF (FRMAT.EQ.'UNIVERSAL') THEN
c              WRITE(VFRMAT,21) NOUT
c              READ(IUNIT,VFRMAT,END=9000,ERR=9000)
               READ(IUNIT,20,END=9000,ERR=9000)
     $             (WORK(I),I=1,NWORDS)
               CALL VRNVERT(WORK,NWORDS)
   20          FORMAT(20A4) 
   21          FORMAT('(',I2.2,'A4)') 
            ELSEIF (FRMAT.EQ.'UNFORMATTED') THEN
C              clearly, this needs work, because we won't 
C              have read any of the above in this case...
               READ(30) (WORK(I),I=1,NWORDS)
            ELSE
               READ(IUNIT,FRMAT,END=9000,ERR=9000)(WORK(I),I=1,NWORDS)
            ENDIF
c           write(6,*) 'work:',(work(j),j=1,30)
C
C           Extract
C
C        Geometry
         IF (INDX1(STUFF,'X',1).NE.0) THEN
            IOUT=IOUT+1
            CALL EXTPLN2(XP(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            IOUT=IOUT+1
            CALL EXTPLN2(YP(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            IF (IF3D) THEN
               IOUT=IOUT+1
               CALL EXTPLN2(ZP(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            ENDIF
         ENDIF
C
C        Velocity
         IF (INDX1(STUFF,'V',1).NE.0) THEN
            IOUT=IOUT+1
            CALL EXTPLN2(U(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            IOUT=IOUT+1
            CALL EXTPLN2(V(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            IF (IF3D) THEN
               IOUT=IOUT+1
               CALL EXTPLN2(W(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            ENDIF
         ELSE
            CALL RZERO(U(IOFF),NXY)
            CALL RZERO(V(IOFF),NXY)
            CALL RZERO(W(IOFF),NXY)
         ENDIF
C
C        Pressure
         IF (INDX1(STUFF,'P',1).NE.0) THEN
            IOUT=IOUT+1
            CALL EXTPLN2(P(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
         ELSE
            CALL RZERO(P(IOFF),NXY)
         ENDIF
C
C        Temperature
         IF (INDX1(STUFF,'T',1).NE.0) THEN
            IOUT=IOUT+1
            CALL EXTPLN2(T(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
         ELSE
            CALL RZERO(T(IOFF),NXY)
         ENDIF
C
C        Vorticity
         IF (INDX1(STUFF,'W',1).NE.0) THEN
            IOUT=IOUT+1
            CALL EXTPLN2(WKV1(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            IF (IF3D) THEN
               IOUT=IOUT+1
               CALL EXTPLN2(WKV2(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
               IOUT=IOUT+1
               CALL EXTPLN2(WKV3(IOFF),WORK(IOUT),NOUT,IPLN,IPLANE)
            ENDIF
         ENDIF
c        write(6,*) 'ilst:',ilist,ioff,wkv1(ioff),iout,list(ilist)
c        write(6,*) 'stuff:',stuff,namels(ilist)
 8000    CONTINUE
         CLOSE (UNIT=IUNIT)
 9000 CONTINUE
 9001 CONTINUE
C
C     Clean up
C
      IF (IFSRF) THEN
         NLSTP=ILIST
         IERR=0
         RETURN
      ENDIF
      if (icalld.eq.1) then
         IERR=2
         NLSTP=0
      else
         IERR=1
      endif
      RETURN
      END
c     SUBROUTINE SURFCHK(IERR,mOPEN)
C
C     Check the file srfdata
C
c     INCLUDE 'basics.inc'
c     INCLUDE 'basicsp.inc'
c     CHARACTER*80 HEADER
c     CHARACTER*1  HEADR1(80)
c     EQUIVALENCE (HEADER,HEADR1)
c     INTEGER ICALLD
c     SAVE    ICALLD
c     DATA    ICALLD /0/
c     CHARACTER*8 LCKFLE
c     CHARACTER*4 LOCK 
c     SAVE        LOCK,LCKFLE
c     DATA        LOCK /'Lock'/
c
c     IERR=1
c     NPMAX=1
C
c     CALL GETNAM(LCKFLE,LOCK,mOPEN)
c     OPEN(UNIT=39,FILE=LCKFLE,STATUS='OLD',ERR=2000)
C
C     Lock file was  found
c     IERR=0
c     ICALLD=ICALLD+1
c     CLOSE(UNIT=39)
c     CALL BLANK(S,80)
c     WRITE(6,100) LCKFLE
c     WRITE(S,100) LCKFLE
c 100 FORMAT('rm -f ',A8)
c     CALL SYSTEM(S)
c     RETURN
C     File wasn't found
c2000 CONTINUE
c     IERR=2
c     RETURN
c     END
      SUBROUTINE SRFILE(IERR,FILE1,JNID,IUNIT,mOPEN)
C
C     Try to see if we can open a surface data file
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
C
      CHARACTER*80  FILE1
c     write(6,14) sesion
c  14 format(2x,'session name:  ',a14)
C
C     Construct file name
C
      CALL GETNAM(FILE1,sesion,mOPEN,JNID)
c     write(6,*) 'Opening',FILE1
      OPEN(UNIT=IUNIT,FILE=FILE1,STATUS='OLD',ERR=2000)
C
C     Write the name of the .srf file to the logfile.
C
      WRITE(S,1000) FILE1
 1000 FORMAT(' Opening srf file: ',A20,'$')
      CALL PRS(S)
C
      IERR=0
      RETURN
C
 2000 CONTINUE
C
c     WRITE(S,2001) FILE1
c2001 FORMAT(' Unable to open srf file: ',A20,'$')
c     CALL PRS(S)
C
      IERR=1
      RETURN
      END
      FUNCTION REALCHR(S,L)
C
C     Read Real
      CHARACTER*1 S(1)
C
      REWIND(13)
      WRITE (13,'(80A1)') (S(j),j=1,L)
      REWIND(13)
      READ  (13,*,ERR=13,END=13)R1
      REWIND(13)
      REALCHR=R1
      RETURN
 13   CALL PRS('Error converting string to real, return 0.0.$')
      REALCHR=0.0
      RETURN
      END
      FUNCTION INTCHR(S,L)
C
C     Read Integer
      CHARACTER*1 S(1)
C
      REWIND(13)
      WRITE (13,'(80A1)') (S(j),j=1,L)
      REWIND(13)
      READ  (13,*,ERR=13,END=13) I1
      REWIND(13)
      INTCHR=I1
      RETURN
 13   CALL PRS('Error converting string to integer, return 0.$')
      INTCHR=0
      RETURN
      END
      SUBROUTINE GETNAM(FILENM,BASE,mOPEN,JNID)
      CHARACTER*80 FILENM
      CHARACTER*14 BASE
C
      CHARACTER*1  TMPFLE(80)
      CHARACTER*80 TMPFL8
      EQUIVALENCE (TMPFL8,TMPFLE)
      CHARACTER*1 SUFFIX(5)
C
      CHARACTER*1  NUMRL(0:9),DOT
      SAVE NUMRL
      DATA NUMRL          /'0','1','2','3','4','5','6','7','8','9'/
      SAVE DOT
      DATA DOT            /'.'/
      CHARACTER*3  SRF
      SAVE SRF
      DATA SRF            /'srf'/
C
      CALL BLANK(TMPFLE,80)
      CALL BLANK(FILENM,80)
C
C     Extract name from session
C
      call chcopy(tmpfle,base,14)
c     left adjust...
      do 10 j=1,14
         if (tmpfle(j).ne.' ') then
            i1=j
            goto 11
         endif
   10 continue
   11 continue
      i2=indx1(tmpfle(i1),' ',1)-1
      call chcopy(filenm,tmpfle(i1),i2)
cxx      write(6,118) i2,filenm
  118 format('i2 filenm:',i9,3x,a80)
C
C------------------------------------------------------------
C
C     add a prefix for an extra subdirectory
C
c....dir       CALL BLANK(TMPFLE,80)
c....dir       write(tmpfl8,15) mopen
c....dir    15 format(i5,'/')
c....dir       call chcopy(tmpfle(7),filenm,i2)
c
c     left adjust...
c....dir       do 20 j=1,20
c....dir          if (tmpfle(j).ne.' ') then
c....dir             i1=j
c....dir             goto 21
c....dir          endif
c....dir    20 continue
c....dir    21 continue
c....dir       i2=indx1(tmpfle(i1),' ',1)-1
c....dir       call chcopy(filenm,tmpfle(i1),i2)
C
C------------------------------------------------------------
c
      call chcopy(tmpfle,filenm,i2)
      i2=i2+1
      call chcopy(tmpfle(i2),srf,3)
      i2=i2+3
C
C     Append 'NOPEN' to file
C
C     less than 100000 dumps....
      ITTH=MOD(mOPEN,100000)/10000
                                        SUFFIX(1)=NUMRL(ITTH)
      ITHO=MOD(mOPEN,10000)/1000
                                        SUFFIX(2)=NUMRL(ITHO)
      IHUN=MOD(mOPEN,1000)/100
                                        SUFFIX(3)=NUMRL(IHUN)
      ITEN=MOD(mOPEN,100)/10
                                        SUFFIX(4)=NUMRL(ITEN)
      IONE=MOD(mOPEN,10)
                                        SUFFIX(5)=NUMRL(IONE)
      CALL CHCOPY(TMPFLE(i2),SUFFIX,5)
      i2=i2+5
      TMPFLE(i2)=DOT
C
C     Append 'NID' to file
C
C     less than 100000 dumps....
      ITTH=MOD(JNID,100000)/10000
                                        SUFFIX(1)=NUMRL(ITTH)
      ITHO=MOD(JNID,10000)/1000
                                        SUFFIX(2)=NUMRL(ITHO)
      IHUN=MOD(JNID,1000)/100
                                        SUFFIX(3)=NUMRL(IHUN)
      ITEN=MOD(JNID,100)/10
                                        SUFFIX(4)=NUMRL(ITEN)
      IONE=MOD(JNID,10)
                                        SUFFIX(5)=NUMRL(IONE)
      i2=i2+1
      CALL CHCOPY(TMPFLE(i2),SUFFIX,5)
      i2=LTRUNC(TMPFLE,80)
      CALL CHCOPY(FILENM,TMPFLE,i2)
      if (jnid.eq.0) write(6,90) (tmpfle(j),j=1,i2)
   90 format(' Trying to open file:',/,80a1)
C
      RETURN
      END
