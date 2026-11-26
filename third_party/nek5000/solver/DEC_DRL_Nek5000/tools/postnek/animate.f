c-----------------------------------------------------------------------
      SUBROUTINE ANIMATE
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
      CALL ANIVIEW
c     IF (ICALLD.EQ.0) CALL GETSTR
c     CALL GETBRST
c     CALL HBUBBLE
      ICALLD=1
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE HBUBBLE
C
C     This routine plots pre-drawn streamline information from file.st3
C     in a hydrogen bubble like fashion.  It is primarily intended for
C     animation of 3D streamline data.
C
C     The only argument is the three-term vector TCLOCK, which specifies
C     the start time, end time, and delta time step for the series of
C     plots.   In interactive mode, the user is prompted to advance each
C     time step.  In IFDEMO mode, DMPFRM is called between each time step.
C     It is assumed that the hydrogen bubble wire burst times have been
C     specified in TBURST prior to calling this routine.
C
C     pff  31 Jul 90 18:30:23 PDT
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 YESNO
      LOGICAL IFAUTH,IFMOVI
C
C     Auto mode?
C
      IFAUTH=.FALSE.
      CALL PRS(' Auto frame advance? (y/n)$')
      CALL RES(YESNO,1)
      IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFAUTH=.TRUE.
      IFMOVI=.FALSE.
      CALL PRS(' Dump frames automatically? (y/n)$')
      CALL RES(YESNO,1)
      IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFMOVI=.TRUE.
C
C     Check if tclock3 specifies DT or number of clock steps.
C
      IF (TCLOCK(3).GT.1.0) THEN
         DTC=(TCLOCK(2)-TCLOCK(1))/TCLOCK(3)
      ELSE
         DTC=TCLOCK(3)
      ENDIF
      IF (TCLOCK(1).EQ.TCLOCK(2).OR.TCLOCK(3).EQ.0.0) THEN
         NCLOCK=1
      ELSE
         NCLOCK=INT( 0.5+(TCLOCK(2)-TCLOCK(1))/DTC )+1
      ENDIF
C
C     Main loop:  Draw each BURST as it would be seen at time TCLK0
C
      DO 5000 ICLOCK=1,NCLOCK
C        Clear and set up the frame - viewpt, zoom, color, etc.
         CALL SETFRM
         CALL DRWIRE
C
C==============================================================
C        Draw the streamlines twice:
C==============================================================
C
         TCLK0=TCLOCK(1)+DTC*FLOAT(ICLOCK-1)
         DO 2000 IBURST=1,NBURST
            TAU1=TCLK0-TBURST(2,IBURST)
            TAU2=TCLK0-TBURST(1,IBURST)
            TAU1=MAX(TAU1,0.0)
            TAU2=MAX(TAU2,0.0)
            IF (TAU2.GT.0.0) THEN
               DO 1000 IP=1,NPART
                  CALL DRWSTL(IP,TAU1,TAU2,1)
 1000          CONTINUE
            ENDIF
 2000    CONTINUE
C
C        Redraw the object to hide lines behind it
C
         CALL DROBJT
         CALL DRWIRE
C
         DO 4000 IBURST=1,NBURST
            TAU1=TCLK0-TBURST(2,IBURST)
            TAU2=TCLK0-TBURST(1,IBURST)
            TAU1=MAX(TAU1,0.0)
            TAU2=MAX(TAU2,0.0)
            IF (TAU2.GT.0.0) THEN
               DO 3000 IP=1,NPART
                  CALL DRWSTL(IP,TAU1,TAU2,2)
 3000          CONTINUE
            ENDIF
 4000    CONTINUE
C
C==============================================================
C        Dump frame?
C==============================================================
C
         IF (IFDEMO) THEN
            CALL DMPFRM
         ELSEIF (.NOT.IFAUTH) THEN
            CALL PRS(' Advance frame? (y/n)$')
            CALL RES(YESNO,1)
            IF (YESNO.EQ.'N'.OR.YESNO.EQ.'n') RETURN
         ENDIF
         IF (IFMOVI.AND..NOT.IFDEMO) CALL DMPFRM
C==============================================================
C     Advance to next time step
C==============================================================
 5000 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRWSTL(IP,TAU1,TAU2,IPASS)
C
C     Draw streamline IP from time TAU1 to TAU2
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
C     Initial offset for streamline #IP is IP0
      IPMIN=IPOFF(IP)
      IPMAX=IPOFF(IP+1)-1
      IF (XYZ0(4,IP).GT.0) THEN
C        Streamline starts in front of the object
         IF (IPASS.EQ.1) THEN
            IPMIN=INT(XYZ0(4,IP))
         ELSE
            IPMAX=INT(XYZ0(4,IP))
         ENDIF
      ELSE
C        Streamline starts behind the object
         IF (IPASS.EQ.1) THEN
            IPMAX=-INT(XYZ0(4,IP))
         ELSE
            IPMIN=-INT(XYZ0(4,IP))
         ENDIF
      ENDIF
      if (ipmin.le.0.or.ipmin.gt.ipoff(ip+1)) then
         write(6,*) 'min problem: ',ip,ipoff(ip),ipoff(ip+1)
     $                             ,ipmin,ipmax,xyz0(4,ip)
         return
      endiF
      if (ipmax.le.0.or.ipmax.gt.ipoff(ip+1)) then
         write(6,*) 'max problem: ',ip,ipoff(ip),ipoff(ip+1)
     $                             ,ipmin,ipmax,xyz0(4,ip)
         return
      endiF
C
C     Find starting point, binary search
C      
      IF (TAU1.GE.XSTR(4,IPMAX)) RETURN
      IF (TAU2.LE.XSTR(4,IPMIN)) RETURN
      IF (TAU1.LE.XSTR(4,IPMIN)) THEN
         IPG=IPMIN
         IPG1=IPG+1
         FRAC=0.0
      ELSE
         IPGLO=IPMIN
         IPGUP=IPMAX
         DO 100 J=IPMIN,IPMAX
            IPG  = (IPGLO+IPGUP)/2
            IPG1=IPG+1
            IF (XSTR(4,IPG).LE.TAU1.AND.TAU1.LT.XSTR(4,IPG1)) THEN
               FRAC=(TAU1-XSTR(4,IPG))/(XSTR(4,IPG1)-XSTR(4,IPG))
               GOTO 101
            ENDIF
C           Else, we didn't find it yet
            IF (TAU1.LT.XSTR(4,IPG)) THEN 
               IPGUP=IPG
            ELSE
               IPGLO=IPG1
            ENDIF
  100    CONTINUE
  101    CONTINUE
      ENDIF
C
C     Start:
      XP0 = XSTR(1,IPG)+FRAC*(XSTR(1,IPG1)-XSTR(1,IPG))
      YP0 = XSTR(2,IPG)+FRAC*(XSTR(2,IPG1)-XSTR(2,IPG))
      ZP0 = XSTR(3,IPG)+FRAC*(XSTR(3,IPG1)-XSTR(3,IPG))
      WP0 = XSTR(5,IPG)+FRAC*(XSTR(5,IPG1)-XSTR(5,IPG))
      CALL  MOVEC(XISO(XP0,YP0,ZP0),YISO(XP0,YP0,ZP0))
C
C     Let's draw (while Tau < Tau2)
C
      DO 200 J=IPG1,IPMAX
         IF (XSTR(4,J).LE.TAU2) THEN
            XP1 = XSTR(1,J)
            YP1 = XSTR(2,J)
            ZP1 = XSTR(3,J)
            WP1 = XSTR(5,J)
            WPP = (WP0+WP1)/2.0
            ICC = ICOLOR1(WPP)
c           write(6,*) 'trouble:',icc,wpp,wp1,xp1,yp1,zp1,j,ipg1,ipmax
            CALL COLOR(ICC)
            CALL DRAWC(XISO(XP1,YP1,ZP1),YISO(XP1,YP1,ZP1))
            XP0 = XP1
            YP0 = YP1
            ZP0 = ZP1
            WP0 = WP1
         ELSE
            FRAC=(TAU2-XSTR(4,J-1))/(XSTR(4,J)-XSTR(4,J-1))
            XPP = XP0+FRAC*(XSTR(1,J)-XP0)
            YPP = YP0+FRAC*(XSTR(2,J)-YP0)
            ZPP = ZP0+FRAC*(XSTR(3,J)-ZP0)
            WPP = WP0+FRAC*(XSTR(5,J)-WP0)
            ICC = ICOLOR1(WPP)
            CALL COLOR(ICC)
            CALL DRAWC(XISO(XPP,YPP,ZPP),YISO(XPP,YPP,ZPP))
            GOTO 201
         ENDIF
  200 CONTINUE
  201 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GETSTR
C
C     This routine reads previously written streamline data from the 
C     file 'session'.st3
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*80 LSTRNG
      CHARACTER*1  PERIOD
      SAVE         PERIOD
      DATA         PERIOD/'.'/
C
C     Streamline Data for already written data.
      CALL BLANK(LSTRNG,80)
      CALL CHCOPY(LSTRNG,FILENM,14)
      M=INDX1(LSTRNG,PERIOD,1)
      IF (M.EQ.0) M=LTRUNC(LSTRNG,14)+1
      n=m+3
      filenm(m:n)='.st3'
      WRITE(6,*) 'Opening file ',FILENM
      CALL OPENF(16,FILENM,'OLD',1,IERR)
C
      IP=0
      NPART=0
      DO 1000 I=1,NSTOR
         IF (MOD(I,1000).eq.0) WRITE(6,*) 'Reading particle',IP
         READ(16,*,END=2000) (TDUMP(J),J=1,5)
         IF (TDUMP(1).NE.99.0.OR.TDUMP(2).NE.99.0.OR.TDUMP(3).NE.99.0
     $   .OR.TDUMP(4).NE.99.0.OR.TDUMP(5).NE.99.0)   THEN
C
C           We have another point
C
            IP=IP+1
            CALL COPY(XSTR(1,IP),TDUMP,5)
C
         ELSE
C
C           We have a new line
C
            NPART=NPART+1
            IPOFF(NPART)=IP+1
         ENDIF
 1000 CONTINUE
 2000 CONTINUE
C
      NPART=MIN(NPART,NPRT)
      NTMP=NPART+1
      IPOFF(NTMP)=IP+1
      WRITE(6,*) NPART,' paths',IP,' particles read.'
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GETBRST
C
C     Prompt the user for burst information
C
C     Three pieces of information required:
C
C       i) Window through which (previosly generated) streamlines
C          will be allowed to pass.
C
C      ii) Series of bursts, specified by t' and t", for K bursts.
C
C     iii) Specify run time, tclock1 - tclock2.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 YESNO
      COMMON /BUB2/XBUBW0,YBUBW0,ZBUBW0,XBUBW1,YBUBW1,ZBUBW1
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
      YESNO='y'
      IF (ICALLD.GT.0) THEN
         CALL PRS(' New burst sequence? (y/n)$')
         CALL RES(YESNO,1)
      ENDIF
C
      YYMIN=XSTR(2,1)
      YYMAX=XSTR(2,1)
      DO 200 I=1,NPART
         IP=IPOFF(I)
         CALL COPY(XYZ0(1,I),XSTR(1,IP),5)
         IF (XYZ0(2,I).LE.YYMIN) THEN
             XBUBW0=XYZ0(1,I)
             YBUBW0=XYZ0(2,I)
             ZBUBW0=XYZ0(3,I)
             YYMIN=YBUBW0
         ENDIF
         IF (XYZ0(2,I).GE.YYMAX) THEN
             XBUBW1=XYZ0(1,I)
             YBUBW1=XYZ0(2,I)
             ZBUBW1=XYZ0(3,I)
             YYMAX=YBUBW1
         ENDIF
C
C        Find transition point from behind to before cylinder
C
         NP=IPOFF(I+1)-IP
         ZVIEW1=ZISO(XSTR(1,IP),XSTR(2,IP),XSTR(3,IP))
         INFRNT=-1
         IF (ZVIEW1.GT.0.0) INFRNT=1
         DO 190 II=2,NP
            IJ=II+IP-1
            ZVIEW=ZISO(XSTR(1,IJ),XSTR(2,IJ),XSTR(3,IJ))
            IF (ZVIEW*ZVIEW1.LE.0.0) GOTO 191
            ZVIEW1=ZVIEW
  190    CONTINUE
  191    CONTINUE
         XYZ0(4,I)=INFRNT*IJ
         ip1=ipoff(i+1)-1
c        WRITE(6,*) 'transition:',i,ipoff(i),xyz0(4,i),ip1
  200 CONTINUE
C
      WRITE(6,*) YYMIN,YYMAX,XBUBW1,YBUBW1,ZBUBW1,XBUBW0,YBUBW0,ZBUBW0
  201 FORMAT(2X,'WIRE:',8F9.4)
C
C     Now prompt for burst times
C
      IF (ICALLD.EQ.0.OR.YESNO.EQ.'Y'.OR.YESNO.EQ.'y') THEN
         DO 400 IB=1,100
            CALL PRS( 'Input burst times t1,t2: (eg. t1=0 to t2=1)$')
            CALL PRS( '(negative to quit)$')
            CALL RERR(TBURST(1,IB),TBURST(2,IB))
            IF (TBURST(1,IB).LT.0.0.OR.TBURST(2,IB).LT.0.0) GOTO 401
  400    CONTINUE
  401    CONTINUE
         NBURST=IB-1
C
C        Prompt for clock time
C
         CALL PRS( 'Input clock times t1,t2 and dt:$')
         CALL RERRR(TCLOCK(1),TCLOCK(2),TCLOCK(3))
C
      ENDIF
      ICALLD=ICALLD+1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DMPFRM
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
      ICALLD=ICALLD+1
      ih=0
      istart=22
      open(unit=10,file='start',status='old')
      write(10,*) istart
      close(unit=10)
      do 2000 i=1,100000
      call sleep(1)
      open(unit=10,file='start',status='old')
      read(10,*) istart
      if (istart.eq.23) goto 9000
      close(unit=10)
 2000 continue
 9000 continue
      close(unit=10)
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETFRM
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /FILLFG/ IFILLM
C
      CALL CLEAR
      IFILLM=1
      CALL DRMESH
      IF (.NOT.IFFULL) CALL DRFRAM
      IFILLM=0
c     CALL DRWIRE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DROBJT
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /FILLFG/ IFILLM
      INTEGER FCORNS (4,6),FEDGES(4,6),FFLIP(4,6)
      DIMENSION XISOM(8),YISOM(8),CSPACE(100),XCRVED(100),YCRVED(100)
      CHARACTER CCUR
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
      DATA FEDGES / 9  ,1, 10 ,5,
     $              10 ,2, 11 ,6,
     $              11 ,3, 12 ,7,
     $              12 ,4, 9  ,8,
     $              4  ,1, 2  ,3,
     $              8  ,7, 6  ,5 /
      DATA FFLIP  / -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $               1, 1, 1,  1,
     $              -1,-1,-1, -1 /
C
C     If we're viewing from directly above, or directly at the side,
C     no need to redraw object.
C
      IF (THETA.EQ.0.0.OR.THETA.EQ.90.0) RETURN
C
      IFILLM=1
      NPOINT=NCSEGS
      DO 18 I=1,NPOINT
         CSPACE(I)=(I-1.0)/(NPOINT-1.0)
18    CONTINUE
c     IF (IF3D) CALL SORTZ
      CALL COLOR(1)
      CALL PENW(1)
C !!??
      NLOOPS=NEL
      IF(IF3D)NLOOPS=NEL*6
      DO 400 NUMBER=1,NLOOPS
         IF(NDIM.EQ.2)THEN
           IEL=NUMBER
           IFACE=5
         ELSE
           IEL  =IZDPTH(NUMBER,1)
           IFACE=IZDPTH(NUMBER,2)
         ENDIF
C        Draw and fill this face
         ICORN=4
	 if (if3d) then
	    zavg=0.0
	    do 917 iii=1,4
               zavg=zavg+z(iel,iii)
  917       continue
	 ELSE
	    zavg=1.0
         endif
         IF(IFILLM.EQ.1.and.
     $    ((zavg.gt.0.001.and.iface.eq.5).or.
     $     (zavg.gt.0.001.and.iface.eq.6).or.
     $      iface.lt.5) ) THEN
C           FILL Mesh with colored panels
C           Take outward normal of face, dot it with view angle.  If we are
C           Looking at the inside of an element, make color darker.
C           Check Here for points not counter-clockwise
            ic1=fcorns(1,iface)
            ic2=fcorns(2,iface)
            ic3=fcorns(3,iface)
            ic4=fcorns(4,iface)
            x1=xiso(x(iel,ic1),y(iel,ic1),z(iel,ic1))
            x2=xiso(x(iel,ic2),y(iel,ic2),z(iel,ic2))
            x3=xiso(x(iel,ic3),y(iel,ic3),z(iel,ic3))
            x4=xiso(x(iel,ic4),y(iel,ic4),z(iel,ic4))
            y1=yiso(x(iel,ic1),y(iel,ic1),z(iel,ic1))
            y2=yiso(x(iel,ic2),y(iel,ic2),z(iel,ic2))
            y3=yiso(x(iel,ic3),y(iel,ic3),z(iel,ic3))
            y4=yiso(x(iel,ic4),y(iel,ic4),z(iel,ic4))
            elarea = x1*y2-x2*y1 +
     $               x2*y3-x3*y2 +
     $               x3*y4-x4*y3 +
     $               x4*y1-x1*y4
            IF(ELAREA .GT. 0.0) THEN
C              Regular colors
c              IF(IFACE.EQ.6) CALL FILLP(-8)
c              IF(IFACE.EQ.5) CALL FILLP(-8)
               IF(IFACE.EQ.6) CALL FILLP(-14)
               IF(IFACE.EQ.5) CALL FILLP(-14)
               IF(IFACE.EQ.4) CALL FILLP(-14)
               IF(IFACE.EQ.2) CALL FILLP(-14)
               IF(IFACE.EQ.3) CALL FILLP(-12)
               IF(IFACE.EQ.1) CALL FILLP(-12)
            ELSE
C              Darker Colors
c              IF(IFACE.EQ.6) CALL FILLP(-6)
c              IF(IFACE.EQ.5) CALL FILLP(-6)
               IF(IFACE.EQ.6) CALL FILLP(-15)
               IF(IFACE.EQ.5) CALL FILLP(-15)
               IF(IFACE.EQ.4) CALL FILLP(-15)
               IF(IFACE.EQ.2) CALL FILLP(-15)
               IF(IFACE.EQ.3) CALL FILLP(-13)
               IF(IFACE.EQ.1) CALL FILLP(-13)
            ENDIF
         ELSE
C           Fill with black panels
            CALL FILLP(0)
            GOTO 400
         ENDIF
C        Don't draw or fill elemental boundaries; only physical boundaries
         IFLD=2
         IF(IFFLOW)IFLD=1
         IF( (CBC(IFACE,IEL,IFLD).NE.'E'.AND.IFDMSH) .OR.
     $       (.NOT.IF3D)                             .OR.
     $       (CBC(IFACE,IEL,IFLD).EQ.'W')           ) THEN
            IF( (IFDMSH.AND..NOT.IF3D.AND.CBC(IFACE,IEL,IFLD).NE.'P  ')
     $         .OR.(IF3D.AND.CBC(IFACE,IEL,IFLD).NE.'P  ')       ) THEN
               CALL BEGINB(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))),
     $                     YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))))
            ELSE
               CALL MOVEC (XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))),
     $                     YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))))
            ENDIF
C
C           Don't draw mesh on periodic boundaries if IFILLM=1
            IF(CBC(IFACE,IEL,IFLD).NE.'P'.OR.IFILLM.NE.1)THEN
               DO 370 ICORN=1,4
                  IICORN=ICORN-1
                  IF(IICORN.EQ.0)IICORN=IICORN+4
                  IF(.NOT.IF3D.AND.CBC(IICORN,IEL,1).EQ.'W')THEN
                     CALL COLOR(1)
                     CALL PENW(5)
                  ENDIF
C                 Find out which edge we are drawing
                  IEDGE=FEDGES(ICORN,IFACE)
                  IF(IEDGE.LE.8)CCUR=CCURVE(IEDGE,IEL)
                  IF(IEDGE.GT.8 .OR. CCUR.EQ.' ')THEN
C                    Straight side
                     IF(IFDMSH.OR.CBC(IICORN,IEL,1).EQ.'W'.OR.IF3D)THEN
                     CALL DRAWC(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                              Y(IEL,FCORNS(ICORN,IFACE)),
     $                              Z(IEL,FCORNS(ICORN,IFACE))),
     $                         YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                              Y(IEL,FCORNS(ICORN,IFACE)),
     $                              Z(IEL,FCORNS(ICORN,IFACE))))
                     ELSE
                     CALL MOVEC(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                              Y(IEL,FCORNS(ICORN,IFACE)),
     $                              Z(IEL,FCORNS(ICORN,IFACE))),
     $                         YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                              Y(IEL,FCORNS(ICORN,IFACE)),
     $                              Z(IEL,FCORNS(ICORN,IFACE))))
                     ENDIF
                  ELSE
C                    Draw curved side
c                    IF(IFMOVB.OR.IFXYO.OR.IFGNGM)THEN
C                    Use mesh data
c                    CALL GETPTM(NPOINT,CSPACE,IEL,IEDGE,XCRVED,YCRVED)
c                    ELSE
                     CALL GETPTS(NPOINT,CSPACE,IEL,IEDGE,XCRVED,YCRVED)
c                    ENDIF
                     IFLIP=FFLIP(ICORN,IFACE)
                     IF(IFLIP.EQ.1)THEN
                        IBEGIN=1
                        IEND  =NPOINT
                     ELSE
                        IBEGIN=NPOINT
                        IEND  =1
                     ENDIF
                     ZCRVED=Z(IEL,FCORNS(ICORN,IFACE))
C
C      Draw only if draw mesh is .TRUE.
                     IF(IFDMSH.OR.CBC(IICORN,IEL,1).EQ.'W'.OR.IF3D)THEN
                        DO 118 I=IBEGIN,IEND,IFLIP
                        CALL DRAWC( XISO(XCRVED(I),YCRVED(I),ZCRVED) ,
     $                             YISO(XCRVED(I),YCRVED(I),ZCRVED) )
  118                   CONTINUE
                     ELSE
                        I=IEND
                        CALL MOVEC( XISO(XCRVED(I),YCRVED(I),ZCRVED) ,
     $                             YISO(XCRVED(I),YCRVED(I),ZCRVED) )
                     ENDIF
                  ENDIF
                  IF(.NOT.IF3D.AND.CBC(IICORN,IEL,1).EQ.'W')THEN
                     CALL COLOR(1)
                     CALL PENW(1)
                  ENDIF
370            CONTINUE
            ENDIF
            IF(CBC(IFACE,IEL,IFLD).NE.'P  ') CALL ENDP
         ENDIF
400   CONTINUE
      CALL COLOR(1)
      CALL PENW(3)
      IFILLM=0
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRWIRE
      COMMON /BUB2/XBUBW0,YBUBW0,ZBUBW0,XBUBW1,YBUBW1,ZBUBW1
      CALL COLOR(1)
C
      XW=XISO(XBUBW0,YBUBW0,ZBUBW0)
      YW=YISO(XBUBW0,YBUBW0,ZBUBW0)
      CALL MOVEC(XW,YW)
      XW=XISO(XBUBW1,YBUBW1,ZBUBW1)
      YW=YISO(XBUBW1,YBUBW1,ZBUBW1)
      CALL DRAWC(XW,YW)
C
      XB8=.995*XBUBW0
      XB9=.995*XBUBW1
      XW=XISO(XB8,YBUBW0,ZBUBW0)
      YW=YISO(XB8,YBUBW0,ZBUBW0)
      CALL MOVEC(XW,YW)
      XW=XISO(XB9,YBUBW1,ZBUBW1)
      YW=YISO(XB9,YBUBW1,ZBUBW1)
      CALL DRAWC(XW,YW)
C
      ZB8=1.01*ZBUBW0
      ZB9=1.01*ZBUBW1
      XW=XISO(XB8,YBUBW0,ZB8)
      YW=YISO(XB8,YBUBW0,ZB8)
      CALL MOVEC(XW,YW)
      XW=XISO(XB9,YBUBW1,ZB9)
      YW=YISO(XB9,YBUBW1,ZB9)
      CALL DRAWC(XW,YW)
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE ANIVIEW
      INCLUDE 'basics.inc'
      CHARACTER*1 YESNO
      LOGICAL IFAUTH,IFMOVI,IFPROF,IFTMP
      COMMON /FILLFG/ IFILLM
      parameter (nmx=4000)
      DIMENSION THTAK(nmx),PHIIK(nmx),ZFACK(nmx),XCTRK(nmx),YCTRK(nmx)
      DIMENSION THTAF(nmx),PHIIF(nmx),ZFACF(nmx),XCTRF(nmx),YCTRF(nmx)
C
C     Animate view angles
C
C     User is prompted for a set of state information:
C
C     1)  Theta values - angle above the x-y plane
C
C     2)  Phi values - angle above the x-axis
C
C     3)  ZFAC - the zoom factor:  1 = std., as viewed from above, 10 = 10x
C
C     4)  Xctr,Yctr - the screen coordinates (0:1,0:1) for the origin (0,0,0).
C
      CALL PRS(' Input viewing data from a file? (y/n)$')
      CALL RES(YESNO,1)
      IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') THEN
         CALL OPENPF(IEXIT,11,'Enter input file name         ')
         DO 10 I=1,100
            READ(11,*,END=11) 
     $      THTAF(I),PHIIF(I),ZFACF(I),XCTRF(I),YCTRF(I)
            IF (ZFACF(I).LE.0.0) GOTO 11
   10    CONTINUE
   11    CONTINUE
         NFRAM=I-1
         CLOSE (UNIT=11)
         GOTO 200
      ENDIF
C
C     Else, we're inputting from the terminal
C
      XCTR=.5
      YCTR=.5
      NFRAM=0
  100 CONTINUE
      CALL PRS( 'Input theta,phi,zfac:$')
      CALL RERRR(THTA,PHII,ZFAC)
      IF (ZFAC.LE.0) GOTO 200
      CALL PRS( 'Input xctr,yctr:$')
      CALL RERR (xctr,yctr)
      CALL SETSTAT(THTA,PHII,ZFAC,XCTR,YCTR)
      CALL SETFRM
C
C     Check if we keep this view for animation
C
      CALL PRS(' Keep frame? (y/n)$')
      CALL RES(YESNO,1)
      IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') THEN
         NFRAM=NFRAM+1
         THTAF(NFRAM)=THTA
         PHIIF(NFRAM)=PHII
         ZFACF(NFRAM)=ZFAC
         XCTRF(NFRAM)=XCTR
         YCTRF(NFRAM)=YCTR
         WRITE(91,*) NFRAM,THTA,PHII,ZFAC,XCTR,YCTR
      ENDIF
      GOTO 100
  200 CONTINUE
C
C     Animate?
C
      CALL PRS(' Animate sequence? (y/n)$')
      CALL RES(YESNO,1)
      IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') THEN
C
         CALL PRS('Enter frame count (nrest,nsweep,nrest):$')
         CALL reiii(nrest1,NFRMK,nrest2)
         NFRMK=MIN(NFRMK,nmx)
C
         CALL SPFIT(THTAK,NFRMK,THTAF,NFRAM)
         CALL SPFIT(PHIIK,NFRMK,PHIIF,NFRAM)
         CALL SPFIT(ZFACK,NFRMK,ZFACF,NFRAM)
         CALL SPFIT(XCTRK,NFRMK,XCTRF,NFRAM)
         CALL SPFIT(YCTRK,NFRMK,YCTRF,NFRAM)
C
         IFAUTH=.FALSE.
         IFMOVI=.FALSE.
         IFPROF=.FALSE.
         CALL PRS(' Auto frame advance? (y/n)$')
         CALL RES(YESNO,1)
         IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFAUTH=.TRUE.
         IFMOVI=.FALSE.
         CALL PRS(' Dump frames automatically? (y/n)$')
         CALL RES(YESNO,1)
         IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFMOVI=.TRUE.
         CALL PRS(' Plot velocity profiles? (y/n)$')
         CALL RES(YESNO,1)
         IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFPROF=.TRUE.
c
         CALL PRS(' input reynolds number$')
         CALL rei(nre)
C
C        Perform the animation
C
         kframe = 0
c
         CALL SETSTAT(THTAK(1),PHIIK(1),ZFACK(1),XCTRK(1),YCTRK(1))
c
c        Rest
         i=1
         do j=1,nrest1
            CALL DRAW_STATE(kdump,ldump,ifprof
     $           ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I),nre)
            kframe = kframe+1
            IF (IFMOVI) CALL grab_window_pix('animate',7,kframe)
            call prsii('phase 1, frame:$',kframe,j)
         enddo
c
         DO 5000 I=1,NFRMK
            write(6,*) i,nfrmk,' thta:'
     $      ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I)
            CALL SETSTAT(THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I))
            CALL DRAW_STATE(kdump,ldump,ifprof
     $           ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I),nre)
            IF (.NOT.IFAUTH) THEN
               CALL PRS(' Advance frame? (y/n)$')
               CALL RES(YESNO,1)
               IF (YESNO.EQ.'N'.OR.YESNO.EQ.'n') RETURN
            ENDIF
            kframe = kframe+1
            IF (IFMOVI) CALL grab_window_pix('animate',7,kframe)
            call prsii('phase 2, frame:$',kframe,i)
 5000    CONTINUE
c
c
c        Rest
         i = nfrmk
         do j=1,nrest2
            CALL DRAW_STATE(kdump,ldump,ifprof
     $           ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I),nre)
            kframe = kframe+1
            IF (IFMOVI) CALL grab_window_pix('animate',7,kframe)
            call prsii('phase 3, frame:$',kframe,j)
         enddo
c
c
         DO I=NFRMK,1,-1
            write(6,*) 'thta:'
     $      ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I)
c           CALL SETSTAT(THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I))
            CALL DRAW_STATE(kdump,ldump,ifprof
     $           ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I),nre)
            IF (.NOT.IFAUTH) THEN
               CALL PRS(' Advance frame? (y/n)$')
               CALL RES(YESNO,1)
               IF (YESNO.EQ.'N'.OR.YESNO.EQ.'n') RETURN
            ENDIF
            kframe = kframe+1
            IF (IFMOVI) CALL grab_window_pix('animate',7,kframe)
            call prsii('phase 4, frame:$',kframe,i)
         ENDDO
c
c        Rest & exit
         i = 1
         do j=1,700
            CALL DRAW_STATE(kdump,ldump,ifprof
     $           ,THTAK(I),PHIIK(I),ZFACK(I),XCTRK(I),YCTRK(I),nre)
            kframe = kframe+1
            IF (IFMOVI) CALL grab_window_pix('animate',7,kframe)
            call prsii('phase 5, frame:$',kframe,kdump)
            if (kdump.eq.ldump) goto 1001
         enddo
c
c
      ENDIF
 1001 continue
c
      CALL PRS('WAKE UP!!  The movie is over!!$')
      IFCLRB=IFTMP
      PLFORM='MULTIPLOT'
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  CALL SCRDMP
c     ENDIF
      IFHARD=IFTMP
C
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE SETSTAT(THTA,PHII,ZFAC,XCTR,YCTR)
C
C     Sets all parameters requied to fix the view
C
      INCLUDE 'basics.inc'
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
C
C     Determine std. viewing coordinates on 1st pass
C
      IF (ICALLD.EQ.0) THEN
         THETA=90
         PHI =-90
         CALL SETVUE2
C
         XMAX = -1.0E20
         YMAX = -1.0E20
         XMIN =  1.0E20
         YMIN =  1.0E20
         DO 50 IEL=1,NEL
         DO 50 IC=1,8
            XMAX = MAX(XMAX,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
            XMIN = MIN(XMIN,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
            YMAX = MAX(YMAX,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
            YMIN = MIN(YMIN,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
50       CONTINUE
         XFACO=MAX( (XMAX-XMIN),(YMAX-YMIN) )*1.3
         IF (IFFULL) THEN
            WFR=1.3
            DELTX=(XMAX-XMIN)/WFR
            XFACO=MAX( DELTX , (YMAX-YMIN)) *1.025
         ENDIF
      ENDIF
C
C==========================================================
C     Preliminaries all set:
C==========================================================
C
C     Set viewing angle
C
      THETA=THTA
      PHI  =PHII
      CALL SETVUE2
C
C     Set scale factors
C
      XFAC=XFACO/ZFAC
      YFAC=XFAC
C
C     Set the origin: (0,0)=lower left corner (1,1)=upper right
C
      WF1=1.0
      IF (IFFULL) WF1=WFR
      ZERO=0.0
      X00=XISO(ZERO,ZERO,ZERO)
      Y00=YISO(ZERO,ZERO,ZERO)
      XZERO=WF1*(X00-XCTR*XFAC)
      YZERO=     Y00-YCTR*YFAC
C
C     Set clipping
C
      WT = YPHY(.995)
      WB = YPHY(.005)
      WL = XPHY(.005)
      WR = XPHY(.995)
      IF (IFFULL) WR = XPHY(WFR)
C
      ICALLD=ICALLD+1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETVUE2
      INCLUDE 'basics.inc'
C
C     Set viewing transformation
C
      THETAR=THETA*PI/180.0
      PHIR  =PHI  *PI/180.0
C
C     Direction toward eye (normal vector)
C
      VHOBS(1)=COS(THETAR)*COS(PHIR)
      VHOBS(2)=COS(THETAR)*SIN(PHIR)
      VHOBS(3)=SIN(THETAR)
C
C     Perpendicular on x-y plane
C
      XHOBS(1)=-1.0*SIN(PHIR)
      XHOBS(2)=COS(PHIR)
      XHOBS(3)=0.0
C
C     Perpendicular to above two
C
      YHOBS(1)=-1.0*SIN(THETAR)*COS(PHIR)
      YHOBS(2)=-1.0*SIN(THETAR)*SIN(PHIR)
      YHOBS(3)=COS(THETAR)
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SPFIT(YOUT,NOUT,YIN,NIN)
      DIMENSION YOUT(1),YIN(1)
      DIMENSION YWORK(100),XIN(100)
C
C     Returns an evenly distributed set of Y values based on XIN,YIN
C     data.  The XOUT range is assumed to span the end points
C     given by XIN.
C
      YP0=0.0
      DO 10 I=1,NIN
         XIN(I)=I
   10 CONTINUE
      CALL SPLINE(XIN,YIN,NIN,YP0,YP0,YWORK)
C
      X0=XIN(1)
      DX=(XIN(NIN)-X0)/FLOAT(NOUT-1)
      DO 100 I=1,NOUT
         XOUT=X0+DX*FLOAT(I-1)
         CALL SPLINT(XIN,YIN,YWORK,NIN,XOUT,YOUT(I))
c        WRITE(29,101) I,XOUT,YOUT(I)
  100 CONTINUE
  101 FORMAT(2X,I5,5G14.5)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SPLINT(XA,YA,Y2A,N,X,Y)
C     p. 88-89, numerical recipes
C
      DIMENSION XA(N),YA(N),Y2A(N)
C
      KLO=1
      KHI=N
    1   IF ((KHI-KLO).GT.1) THEN
           K=(KHI+KLO)/2
           IF (XA(K).GT.X) THEN
              KHI=K
           ELSE
              KLO=K
           ENDIF
           GOTO 1
        ENDIF
C
      H=XA(KHI)-XA(KLO)
      IF (H.EQ.0) THEN
         WRITE(6,*) XA(KHI), 'Hey buddy - you blew it.'
         RETURN
      ENDIF
      A=(XA(KHI)-X)/H
      B=(X-XA(KLO))/H
      Y=A*YA(KLO)+B*YA(KHI)+
     $  ((A**3-A)*Y2A(KLO)+(B**3-B)*Y2A(KHI))*(H**2)/6.
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SPLINE(X,Y,N,YP1,YPN,Y2)
C
      PARAMETER (NMAX=4000)
      DIMENSION X(N),Y(N),Y2(N),U(NMAX)
C
      Y2(1)=0.0
      U(1) =0.0
C
      DO 10 I=2,N-1
         IR=I+1
         IL=I-1
         SIG=(X(I)-X(IL))/(X(IR)-X(IL))
         P=SIG*Y2(IL)+2.
         Y2(I)=(SIG-1.)/P
         U(I)= ( 6.*
     $     ( (Y(IR)-Y(I))/(X(IR)-X(I))-(Y(I)-Y(IL))/ (X(I)-X(IL) ) )
     $            / (X(IR)-X(IL))
     $    - SIG*U(IL) )/P
   10 CONTINUE
C
      QN=0.0
      UN=0.0
C
      Y2(N)=(UN-QN*U(N-1))/(QN*Y2(N-1)+1.)
      DO 20 K=N-1,1,-1
         Y2(K)=Y2(K)*Y2(K+1)+U(K)
   20 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE OPENPF(IEXIT,IUNIT,TEXT)
      CHARACTER*30 TEXT
      CHARACTER*80 FILE1
      CHARACTER*4  FILE4
      IEXIT = 0
C ***
C ***  OPEN DATA FILE
C ***
      WRITE(6,90) TEXT
      CALL PUTS(TEXT,30)
90    FORMAT(2X,A30,':  ')
c     READ(5,91) FILE1
      CALL RES(FILE1,80)
      FILE4 = FILE1
      IF (FILE4.EQ.'EXIT' .OR. FILE4.EQ.'exit') THEN
         IEXIT = 1
         RETURN
      ENDIF
91    FORMAT(A40)
      OPEN (UNIT=IUNIT,FILE=FILE1,ACCESS='SEQUENTIAL', 
     *STATUS='UNKNOWN')
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine draw_state(kdump,ldump,ifprof,THT,phj,ZFAC,XCT,YCT,nre)
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
c
      logical ifdrm,ifprof
      integer iframe0,iframe1
      save    iframe0,iframe1
      data    iframe0,iframe1 /0,0/
c
      if (ldump.eq.0) then
         call prs  ('Input start and stop fld numbers:$')
         call reii (iframe0,iframe1)
         kdump = iframe0-1
      endif
      if (kdump.eq.ldump) kdump = iframe0-1
      kdump = kdump + 1
      ldump = iframe1
c
c     Get .fld_iframe .... 
c     (set quanty to PRESSURE to avoid expensive recalculations of work)
c
      QUANTY='PRESSURE'
      call getfld(kdump,ierr,.true.,.true.)
c
      CALL SETSTAT(THT,phj,ZFAC,XCT,YCT)
c
      CALL clear
      CALL DRMESH
      if (.not.ifprof) return
      call blank(line,70)
c
c     dt_run = .0098272
c     time_frame = kdump*dt_run
c     write(line,70) time_frame,kdump
c  70 format('Time = ',f12.7,'  Frame =',i5,'$')
c     CALL GSWRIT(.02,.565,1.0,line)
c
c     Plot all requested forms...
      write(6,*) 'nstate:',nstate,kdump,ldump
      DO I=1,nstate
c
         CALL SETSTATE(I,ifdrm,ir,ic,il,ii,'mlti')
c
c        Note, getfld does nothing if "ndumps" is unchanged between
c        successive calls
c
         CALL SETSTAT(THT,phj,ZFAC,XCT,YCT)
c
         CALL SETWRK(.false.)
         CALL PLOTIT(0)
      ENDDO
c     CALL GSWRIT(.02,.965,1.0,'Reynolds Number = 600,  N=11$')
      dt_run = .0098272
      time_frame = kdump*dt_run
      write(line,80) nre,time_frame,kdump
   80 format('Re=',i4,', N=11, t=',f9.4,', f=',i3,'$')
      CALL GSWRIT(.02,.965,1.0,line)
c
      return
      end
C-----------------------------------------------------------------------
