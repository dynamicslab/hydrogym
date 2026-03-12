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
c
      subroutine SETSCR
C
C     Set up screen characteristics, zoom, ifdmsh, etc.
C     16 Feb 1989 23:14:22  pff
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,inewt
      LOGICAL IFDSPL
      SAVE    IFDSPL
      DATA    IFDSPL /.FALSE./
c
      logical ifadjust
c
      integer mesh_toggle
      save    mesh_toggle
      data    mesh_toggle /1/
C
      NCHOIC=1
      ITEM(NCHOIC)                 = 'MAIN MENU'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'ELEMENT BOUNDARIES'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'SET DRAW AXIS'
c
c     NCHOIC=NCHOIC+1
c     ITEM(NCHOIC)                 = 'GENERAL GEOMETRY'
c     NCHOIC=NCHOIC+1
c     ITEM(NCHOIC)                 = 'GRID LATCH'
c     IF (IF3D) THEN
c        ITEM(NCHOIC)              = 'SAVE IMAGE' 
c        NCHOIC=NCHOIC+1
c     ENDIF
c
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'ADJUST VIEW'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'SET VIEW'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'ROTATE VIEW'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'ZOOM'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'TRANSLATE'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'FULL VIEW' 
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'COLOR FILL BORDER'
c     NCHOIC=NCHOIC+1
c     ITEM(NCHOIC)                 = 'NEWTON DIAG'
c     NCHOIC=NCHOIC+1
c     ITEM(NCHOIC)                 = 'DISPLAY OBJECT'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'REVERSE BACKGROUND'
      NCHOIC=NCHOIC+1
      ITEM(NCHOIC)                 = 'RESIZE WINDOW'
C
1     CONTINUE
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET SCREEN')
      IF(CHOICE.EQ.'MAIN MENU') THEN
         return
      ELSE IF(CHOICE.EQ.'TRANSLATE') THEN
         CALL SETRANS
         CALL CLEAR
         CALL HARD
         CALL HEJECT
         if (ifptplt) then
            call strplt(.false.)
         else
            CALL DRMESH
         endif
         CALL DRFRAM
         if (ifdspl)  call dspobj
         CALL NOHARD
      ELSE IF(CHOICE.EQ.'DRQUAD') THEN
         CALL SETQUAD
      ELSE IF(CHOICE.EQ.'ZOOM') THEN
         CALL SETZOOM
         CALL CLEAR
         CALL HARD
            CALL HEJECT
            CALL DRMESH
            CALL DRFRAM
            IF (IFDSPL) CALL DSPOBJ
         CALL NOHARD
         CALL RELITE
      ELSE IF(CHOICE.EQ.'SAVE IMAGE' ) THEN
         IF (IFSAVE) THEN
            IFSAVE = .FALSE.
            CALL PRS(' Turning the SAVE feature off.$')
         ELSE
            IFSAVE = .TRUE.
            CALL PRS(' Turning the SAVE feature on.$')
         ENDIF
      ELSE IF(CHOICE.EQ.'FULL VIEW') THEN
         IF (IFFULL) THEN
            IFFULL = .FALSE.
            CALL PRS(' Setting clipping window to standard size.$')
         ELSE
            IFFULL = .TRUE.
            CALL PRS(' Setting clipping window to full size.$')
         ENDIF
      ELSE IF(CHOICE.EQ.'NEWTON DIAG') THEN
         IF (inewt.eq.1) THEN
            inewt = 0
            CALL PRS(' Turning off Newton diagnostics.$')
         ELSE
            inewt = 1
            CALL PRS(' Turning on Newton diagnostics.$')
         ENDIF
      ELSE IF(CHOICE.EQ.'COLOR FILL BORDER') THEN
         IF (IFCFBD) THEN
            IFCFBD = .FALSE.
            CALL PRS(' Turning off color fill bordering.$')
         ELSE
            IFCFBD = .TRUE.
            CALL PRS(' Turning on color fill bordering.$')
         ENDIF
      ELSE IF(CHOICE.EQ.'ELEMENT BOUNDARIES') THEN
         mesh_toggle = mod(mesh_toggle+1,3)
         if (mesh_toggle.eq.0) then
            IFDMSH = .FALSE.
            IFMESH = .FALSE.
            CALL PRS(' No mesh boundaries.$')
         elseif (mesh_toggle.eq.1) then
            IFDMSH = .FALSE.
            IFMESH = .TRUE.
            CALL PRS(' Will draw mesh boundary upon refresh.$')
         else
            IFDMSH = .TRUE.
            IFMESH = .TRUE.
            CALL PRS(' Will draw all elements upon refresh.$')
         endif
      ELSE IF(CHOICE.EQ.'SET DRAW AXIS') THEN
        IF (IFDRAX) THEN
           CALL PRS(' Setting DRAW AXIS to .FALSE.$')
           IFDRAX=.FALSE.
        ELSE
           CALL PRS(' Setting DRAW AXIS to .TRUE.$')
           IFDRAX=.TRUE.
           CALL DRAXIS
        ENDIF
      ELSE IF (CHOICE.EQ.'ADJUST VIEW') THEN
777      CALL ADJVUE(ifadjust)
         if (ifadjust) then
            CALL CLEAR
            CALL HARD
            CALL HEJECT
            if (ifptplt) then
               call strplt(.false.)
            else
               CALL DRMESH
            endif
            CALL DRFRAM
            if (ifdspl)  call dspobj
            CALL NOHARD
            CALL RELITE
            goto 777
         endif
      ELSE IF (CHOICE.EQ.'SET VIEW') THEN
         CALL SETVUE
c
         CALL CLEAR
         CALL HARD
         CALL HEJECT
         if (ifptplt) then
            call strplt(.false.)
         else
            CALL DRMESH
         endif
         CALL DRFRAM
         if (ifdspl)  call dspobj
         CALL NOHARD
         CALL RELITE
      ELSE IF(CHOICE.EQ.'ROTATE VIEW') THEN
         IROT = IROT+1
         IROT = MOD(IROT,6)
         CALL RESCAL
         CALL CLEAR
         CALL HARD
         CALL HEJECT
         if (ifptplt) then
            call strplt(.false.)
         else
            CALL DRMESH
         endif
         CALL DRFRAM
         if (ifdspl)  call dspobj
         CALL NOHARD
         CALL RELITE
      ELSE IF(CHOICE.EQ.'GENERAL GEOMETRY') THEN
           IF (ifgngm) THEN
              ifgngm = .FALSE.
           ELSE
              ifgngm = .TRUE.
           ENDIF
           WRITE(S,20) ifgngm
20         FORMAT(2X,'Setting general geometry option to',L4)
           CALL PRS(S//'$')
      ELSE IF(CHOICE.EQ.'GRID LATCH') THEN
           IF (IFGRID) THEN
              IFGRID = .FALSE.
           ELSE
              IFGRID = .TRUE.
           ENDIF
           WRITE(S,30) IFGRID
30         FORMAT(2X,'Setting GRID LATCH to',L4)
           CALL PRS(S//'$')
      ELSE IF(CHOICE.EQ.'RESIZE WINDOW') THEN
           iww = windoww
           iwh = windowh
           write(line,70) iww,iwh
   70      format
     $     ('Input window width and height (curent',i5,',',i4,'):$')
           call prs(line)
           call reii(iww,iwh)
           call set_ww_wh(iww,iwh)
           call post_reset_window
      ELSE IF(CHOICE.EQ.'REVERSE BACKGROUND') THEN
           IF (ifrevbk) THEN
              ifrevbk = .FALSE.
           ELSE
              ifrevbk = .TRUE.
           ENDIF
           WRITE(S,31) ifrevbk
31         FORMAT(2X,'Setting BACKGROUND to',L4)
           CALL PRS(S//'$')
      ELSE IF(CHOICE.EQ.'DISPLAY OBJECT') THEN
         CALL CLEAR
         IFILLM=1
         CALL DRmesh
         IFILLM=0
         IF (IFFULL) THEN
            CALL PRS('Hit mouse button to continue$')
            CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         ENDIF
         CALL DRFRAM
         IFDSPL=.NOT.IFDSPL
         WRITE(S,32) IFDSPL
32       FORMAT(2X,'Setting DISPLAY to',L4)
         CALL PRS(S//'$')
         IF (IFDSPL) CALL DSPOBJ
      ELSE
         CALL PRS('Error reading input; plot parameter not changed$')
      ENDIF
      GOTO 1
      END
c-----------------------------------------------------------------------
C
      SUBROUTINE SETGRD
      INCLUDE 'basics.inc'
      CHARACTER*26 CCART,CPOLAR,CGRIDX,CGRIDY,CGRIDN,CGRIDR,BLNK
      CHARACTER*26 CGRDX,CGRDY,COBJCT
      CHARACTER*1  BLNK1(26)
      EQUIVALENCE (BLNK1,BLNK)
      LOGICAL IFCHNG
      DATA BLNK1 /26*' '/
C
C     Set up grids for polar or cartesian based input
C
      ONE = 1.0
      PI = 4.0*ATAN(ONE)
      IFCHNG=.FALSE.
 1000 CONTINUE
      CCART  = BLNK
      CPOLAR = BLNK
      CGRIDX = BLNK
      CGRIDY = BLNK
      CGRIDN = BLNK
      CGRIDR = BLNK
      CGRDX  = BLNK
      CGRDY  = BLNK
      COBJCT = BLNK
      WRITE(CCART , 2) IFGRDC
      WRITE(CPOLAR, 3) IFGRDP
      WRITE(CGRIDX, 4) GRIDDX
      WRITE(CGRIDY, 5) GRIDDY
      WRITE(CGRIDN, 6) NGRID
      WRITE(CGRIDR, 7) GRIDR
      WRITE(CGRDX , 8) GRIDXP
      WRITE(CGRDY , 9) GRIDYP
      IF (IFOBJS) WRITE(COBJCT,10) (IFOBJG(I),I=1,NOBJS)
    2 FORMAT('CARTESIAN',L4)
    3 FORMAT('POLAR    ',L4)
    4 FORMAT('DX =',G11.4)
    5 FORMAT('DY =',G11.4)
    6 FORMAT('# DIV =',I3)
    7 FORMAT('dR =',G11.4)
    8 FORMAT('Xo=',2G11.4)
    9 FORMAT('Yo=',2G11.4)
   10 FORMAT('OBJ: ',20L1)
C
      ITEM(1) ='MAIN MENU'
      ITEM(2) =CCART
      ITEM(3) =CPOLAR
      ITEM(4) =CGRIDX
      ITEM(5) =CGRIDY
      ITEM(6) =CGRIDN
      ITEM(7) =CGRIDR
      ITEM(8) =CGRDX
      ITEM(9) =CGRDY
      ITEM(10)=COBJCT
      NCHOIC = 9
      IF (IFOBJS) NCHOIC=NCHOIC+1
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET GRID')
C
      IF (CHOICE.EQ.'MAIN MENU') THEN
         IF (IFCHNG) THEN
            CALL REFRESH
            CALL DRGRID
            DO 50 IEL=1,NEL
               CALL DRAWEL(IEL)
   50       CONTINUE
         ENDIF
         RETURN
      ELSE
         IFCHNG=.TRUE.
      ENDIF
      IF (CHOICE.EQ.CCART)  IFGRDC=.NOT.IFGRDC
      IF (CHOICE.EQ.CPOLAR) IFGRDP=.NOT.IFGRDP
      IF (IFGRDP.AND.NGRID.EQ.0) THEN
         WRITE(S,401)
         CALL PRS(S//'$')
         CALL KEYPAD(RGRID)
         NGRID = IFIX(RGRID)
         NGRID = MAX(NGRID,3)
         GRIDA = 2.0*PI/FLOAT(NGRID)
      ENDIF
      IF (IFGRDP.AND.GRIDR.EQ.0.0) THEN
         WRITE(S,501)
         CALL PRS(S//'$')
         CALL KEYPAD(GRIDR)
      ENDIF
      IF (CHOICE.EQ.CGRIDX) THEN
         WRITE(S,301)
         CALL PRS(S//'$')
  301    FORMAT(2X,'Input new value of DX. (use keypad)')
         CALL KEYPAD(GRIDDX)
C        Don't change GRID !
         GRIDT =GRIDDX/XFAC
C        GRIDDX=XFAC*GRIDT  kludge for now
         GRIDDY=YFAC*GRIDT
         GRIDDY=GRIDDX
      ENDIF
      IF (CHOICE.EQ.CGRIDY) THEN
         WRITE(S,302)
         CALL PRS(S//'$')
  302    FORMAT(2X,'Input new value of DY. (use keypad)')
         CALL KEYPAD(GRIDDY)
C        Don't change GRID !
         GRIDT =GRIDDY/XFAC
         GRIDDX=XFAC*GRIDT
         GRIDDX=GRIDDY
C        GRIDDY=YFAC*GRIDT  kludge for now
      ENDIF
      IF (CHOICE.EQ.CGRIDN) THEN
         WRITE(S,401)
         CALL PRS(S//'$')
  401    FORMAT(2X,'Input number of radial divisions for polar grid.')
         CALL KEYPAD(RGRID)
         NGRID = IFIX(RGRID)
         NGRID = MAX(NGRID,3)
         GRIDA = PI/FLOAT(NGRID)
      ENDIF
      IF (CHOICE.EQ.CGRIDR) THEN
  500    WRITE(S,501)
         CALL PRS(S//'$')
  501    FORMAT(2X,'Input value of dR for polar grid.')
         CALL KEYPAD(GRIDR)
         IF (GRIDR.LE.0.0) GOTO 500
      ENDIF
      IF (CHOICE.EQ.CGRDX) THEN
         WRITE(S,601)
         CALL PRS(S//'$')
  601    FORMAT(2X,'Input X-ordinate for center of polar grid.')
         CALL KEYPAD(GRIDXP)
      ENDIF
      IF (CHOICE.EQ.CGRDY) THEN
         WRITE(S,602)
         CALL PRS(S//'$')
  602    FORMAT(2X,'Input Y-ordinate for center of polar grid.')
         CALL KEYPAD(GRIDYP)
      ENDIF
      IF (CHOICE.EQ.COBJCT) THEN
         IF (NOBJS.GT.1) THEN
            WRITE(S,701)
            CALL PRS(S//'$')
  701       FORMAT(2X,'Input object number you wish to change.')
            CALL KEYPAD(OBJECT)
            I=INT(OBJECT)
            IF (1.LE.I.AND.I.LE.NOBJS) IFOBJG(I)=.NOT.IFOBJG(I)
         ELSE
            IFOBJG(1)=.NOT.IFOBJG(1)
         ENDIF
      ENDIF
      GOTO 1000
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRGRID
      INCLUDE 'basics.inc'
      PI = 4.0*ATAN(1.0)
      CALL GINGRD(GRID)
      CALL COLOR(14)
c     write(6,*) 'ifgrdc,grid:',ifgrdc,grid
      IF (IFGRDC .AND. GRID .GT. .0099) THEN
C        PUT GRAPH PAPER ON SCREEN
         IF (GRIDDX.EQ.0.OR.GRIDDX.NE.GRIDDY) THEN
C           Then we're in Prenek, first time to draw grid
c             write(6,*) 'weve got the wrong one:',griddx,griddy
              DO 2 I=1,IFIX(1/GRID - 1)
                 XX=I*GRID * .9963
C                Don't write grid lines in bottom 1/10 of screen.
                 IF (XX.GT. 0.11) THEN
                    CALL MOVESC(0.01,XX)
                    CALL DRAWSC(0.99,XX)
                 ENDIF
                 CALL MOVESC(XX,0.11)
                 CALL DRAWSC(XX,0.99)
    2         CONTINUE
         ELSE
C           We use GRIDDX which measures the Grid in physical dimensions
C           and which can be shifted with the zero
            ZT = YPHY(.995)
            ZB = YPHY(.005)
            ZL = XPHY(.005)
            ZR = XPHY(.995)
            IXLEFT=INT(ZL/GRIDDX)
            IXRGHT=INT(ZR/GRIDDX)
            IF (ZL.GT.0) IXLEFT=IXLEFT+1
            IF (ZR.LT.0) IXRGHT=IXRGHT-1
c         write(6,*) 'weve got the right one:',griddx,griddy,iffull
            DO 3 I=IXLEFT,IXRGHT
               XX=FLOAT(I)*GRIDDX+GRIDXP
               CALL MOVEC(XX,ZT)
               CALL DRAWC(XX,ZB)
    3       CONTINUE
            IYBOTT=INT(ZB/GRIDDY)
            IYTOP =INT(ZT/GRIDDY)
            IF (ZB.GT.0) IYBOTT=IYBOTT+1
            IF (ZT.LT.0) IYTOP =IYTOP -1
            DO 4 I=IYBOTT,IYTOP
               YY=FLOAT(I)*GRIDDY+GRIDYP
               CALL MOVEC(ZL,YY)
               CALL DRAWC(ZR,YY)
    4       CONTINUE
         ENDIF
      ENDIF
C
      IF (IFGRDP) THEN
         WFR= 1.3
         WT = YPHY(.995)
         WB = YPHY(.005)
         WL = XPHY(.005)
         WR = XPHY(.995)
         IF (IFFULL) WR = XPHY(WFR)
C        set range for radius based on windows in physical coordinates
         RADMIN = 0.0
         RADMAX = 0.0
         RADIUS = SQRT(WL**2 + WB**2)
         RADMAX = MAX(RADMAX,RADIUS)
         RADIUS = SQRT(WR**2 + WB**2)
         RADMAX = MAX(RADMAX,RADIUS)
         RADIUS = SQRT(WR**2 + WT**2)
         RADMAX = MAX(RADMAX,RADIUS)
         RADIUS = SQRT(WL**2 + WT**2)
         RADMAX = MAX(RADMAX,RADIUS)
C        draw cirular segments at each radius
         NRAD  = IFIX(RADMAX/GRIDR) + 1
         NSEGS = 44
         DANG  = 2.0*PI/FLOAT(NSEGS)
         DO 220 IRAD = 1,NRAD
            RADIUS = GRIDR*FLOAT(IRAD)
            ANGLE = 0.0
            XPHY0 = RADIUS*COS(ANGLE)+GRIDXP
            YPHY0 = RADIUS*SIN(ANGLE)+GRIDYP
            CALL MOVEC(XPHY0,YPHY0)
            DO 200 ISEG = 1,NSEGS
               ANGLE = DANG*FLOAT(ISEG)
               XPHY1 = RADIUS*COS(ANGLE)+GRIDXP
               YPHY1 = RADIUS*SIN(ANGLE)+GRIDYP
               CALL DRAWC(XPHY1,YPHY1)
  200       CONTINUE
  220    CONTINUE
C        draw radial segments at each angle
         XPHY0 = GRIDXP
         YPHY0 = GRIDYP
         DANG  = 2.0*PI/FLOAT(NGRID)
         DO 320 IRAD = 1,NGRID
            ANGLE = DANG*FLOAT(IRAD)
            XPHY1 = RADMAX*COS(ANGLE)+GRIDXP
            YPHY1 = RADMAX*SIN(ANGLE)+GRIDYP
            CALL MOVEC(XPHY0,YPHY0)
            CALL DRAWC(XPHY1,YPHY1)
  320    CONTINUE
      ENDIF
C   DRAW AXIS
C! Axisymmetric
      IF(IFAXIS) THEN
              CENTRL=YSCR(0.0)
              CENTRx=XSCR(0.0)
              IF(CENTRL.LT.0.95 .AND. CENTRL .GT. 0.0) THEN
                CALL COLOR(10)
                CALL MOVESC(0.03,CENTRL)
                CALL DRAWSC(0.97,CENTRL)
                CALL GSWRIT(0.94,CENTRL+.01,1.0,'C/L$')
                IF(CENTRX.LT.0.95 .AND. CENTRX .GT. 0.0) THEN
                  CALL MOVESC(CENTRX,CENTRL-.02)
                  CALL DRAWSC(CENTRX,CENTRL+.02)
                ENDIF
              ENDIF
      ELSE
C! Planar 2-D
              CENTRX=XSCR(0.0)
              CENTRY=YSCR(0.0)
              IF(CENTRX.GT.0.0 .AND.CENTRY.GT.0.0 .AND.
     $           CENTRX.LE.0.85.AND.CENTRY.LE.0.85) THEN
C               Draw Axis
                CALL COLOR(10)
                CALL MOVE(0.0,0.0)
                CALL DRAW(0.0,0.1*YFAC)
                CALL MOVE(0.0,0.0)
                CALL DRAW(0.1*XFAC,0.0)
                CALL GWRITE(0.1 *XFAC ,0.01*YFAC,1.0,'X$')
                CALL GWRITE(0.01*XFAC,0.1 *YFAC,1.0,'Y$')
              ENDIF
      ENDIF
C
      write(6,*) 'nobjs',nobjs
      DO 400 I=1,NOBJS
         IF (IFOBJG(I)) CALL DRWOBJ(I)
  400 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE REFRESH
      INCLUDE 'basics.inc'
C
C     Draw big red box
c
      CALL CLEAR
      CALL CLSSEG
      CALL OPNSEG(9)
      XL=1.0
      XR=1.3
      YT=0.99
      YB=0.01
      CALL COLOR(1)
      CALL FILLP(-2)
      CALL BEGINB (XPHY(XL),YPHY(YB))
      CALL DRAWSC (XR,YB)
      CALL DRAWSC (XR,YT)
      CALL DRAWSC (XL,YT)
      CALL DRAWSC (XL,YB)
      CALL ENDP
C
      CALL NOHARD
      CALL CLSSEG
      CALL OPNSEG(10)
C     WRITE ON SCREEN
C     Draw Keypad
      CALL DRKEY
C     KEYPAD OFF; COVER ON
      CALL SGVIS(12,0)
      CALL SGVIS(11,1)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE MOVEC(X1,Y1)
C
C     Clip image denoted by vectors in { (QX,QY),i=1,NBUF } to be
C     within window specified by WT,WB,WL,WR.
C
      INCLUDE 'devices.inc'
      LOGICAL IFTMP
C
C     put this extra move command in because it doesn't hurt.
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     CALL MOVE(X1,Y1)
      IFHARD=IFTMP
      XMOVEC = X1
      YMOVEC = Y1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRAWC(X1,Y1)
C
C     Clip image denoted by vectors in { (QX,QY),i=1,NBUF } to be
C     within window specified by WT,WB,WL,WR.  Note number of
C     vectors will possibly be reduced if many of the vectors lie
C     completely outside the window.
C
      COMMON /CLIP/ XMOVEC,YMOVEC,WT,WB,WL,WR
      REAL NX(4),NY(4),FX(4),FY(4)
      LOGICAL IN,LASTIN
      REAL XLAST,YLAST
      SAVE XLAST,YLAST
      DATA XLAST,YLAST /10.0E+04,10.0E+04/
c
c     compute inward facing normals for CYBECK algorithm
c
      nx(1) =  0
      nx(2) = -1
      nx(3) =  0
      nx(4) =  1
      ny(1) =  1
      ny(2) =  0
      ny(3) = -1
      ny(4) =  0
c
      fx(1) = wl
      fx(2) = wr
      fx(3) = wr
      fx(4) = wl
      fy(1) = wb
      fy(2) = wb
      fy(3) = wt
      fy(4) = wt
C
C     Cyrus-Beck algorithm arrays are all set.
C
      X = XMOVEC
      Y = YMOVEC
      IN     = .FALSE.
      IF ((X.GE.WL.AND.X.LE.WR).AND.(Y.GE.WB.AND.Y.LE.WT)) IN=.TRUE.
      IF (IN.AND.(X.NE.XLAST.OR.Y.NE.YLAST)) CALL MOVE(X,Y)
C
      XO = XMOVEC
      YO = YMOVEC
      X  = X1
      Y  = Y1
C
C     Check if vector has zero length
C
      DOT = (X-XO)**2 + (Y-YO)**2
      IF (DOT.LT.1.0E-11) GOTO 1000
C
C     Check if both points are in
C
      LASTIN = IN
      IN     = .FALSE.
      IF ((X.GE.WL.AND.X.LE.WR).AND.(Y.GE.WB.AND.Y.LE.WT)) IN=.TRUE.
      IF (IN.AND.LASTIN) THEN
         CALL DRAW(X,Y)
         XLAST=X
         YLAST=Y
         GOTO 1000
      ENDIF
C
C     Check if both points are trivially out
C
      IF ( (X.LT.WL.AND.XO.LT.WL) .OR.
     $     (X.GT.WR.AND.XO.GT.WR) .OR.
     $     (Y.GT.WT.AND.YO.GT.WT) .OR.
     $     (Y.LT.WB.AND.YO.LT.WB) ) GOTO 1000
C
C     We may have a partial line segment.
C
C     Employ the Cyrus-Beck code from George Anagnostou
C
      dx = x-xo
      dy = y-yo
      tl = 0.0
      tu = 1.0
C
C     Loop over the 4 window edges
C
      do 505 ii = 1,4
         wx = xo - fx(ii)
         wy = yo - fy(ii)
         ddotn = dx*nx(ii)+dy*ny(ii)
         wdotn = wx*nx(ii)+wy*ny(ii)
         if (abs(ddotn).lt.1.0e-06) goto 502
         t = -wdotn/ddotn
         if (ddotn .gt. 0.0) goto 501
         if (t     .lt. 0.0) goto 505
         tu = min(t,tu)
         goto 505
c
c        looking for the lower limit
c
  501    if (t.gt.1.0) goto 505
         tl = max(t,tl)
         goto 505
  502    if (wdotn.lt.0.0) goto 1000
  505 continue
      if (  tl .gt. tu  ) goto 1000
c
c     If we arrived here, then we have clipped a segment.
c
      IF (tl.gt.0.0) THEN
C
C        We need an extra move (UP) command
C
         xp = xo + (x-xo)*tl
         yp = yo + (y-yo)*tl
         call move(xp,yp)
      ENDIF
C
C     New point is within box
C
      xp = xo + (x-xo)*tu
      yp = yo + (y-yo)*tu
      call draw(xp,yp)
      xlast=xp
      ylast=yp
c
 1000 CONTINUE
C
C     Clean up.
C
      XMOVEC = X1
      YMOVEC = Y1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE FILTER(XMOUSE,YMOUSE,XSCREEN,YSCREEN)
      INCLUDE 'basics.inc'
C
      XMSE0 = XPHY(XSCREEN)
      YMSE0 = YPHY(YSCREEN)
      ZMSE0 = ZPLANE
C     default return values
      XMOUSE=XMSE0
      YMOUSE=YMSE0
      ZMOUSE=ZMSE0
C
C     Check objects first, then background grids.
C
      IOBJCT = 0
      TOLOBJ = (GRIDDX**2+GRIDDY**2)/10.0
      DO 10 I=1,NOBJS
         IF (IFOBJG(I)) THEN
C           grab object?
            CALL LATCHOB(XMSE0,YMSE0,ZMSE0,DIST2,I)
            IF (DIST2.LT.TOLOBJ) THEN
               IOBJCT=I
               XMOUSE=XMSE0
               YMOUSE=YMSE0
               ZMOUSE=ZMSE0
               RETURN
            ENDIF
         ENDIF
   10 CONTINUE
      IF (IFGRDC.OR.IFGRDP) THEN
         IF (IFGRDC) THEN
C           cartesian grid
            IF (GRIDDX.EQ.0.0.OR.GRIDDY.EQ.0.0) THEN
               XSCRNC = ANINT(XSCREEN/GRID)*GRID
               YSCRNC = ANINT(YSCREEN/GRID)*GRID
               XPHYC  = XPHY(XSCRNC)
               YPHYC  = YPHY(YSCRNC)
               XMSE0  = XPHYC
               YMSE0  = YPHYC
            ELSE
               XPHYC  = XPHY(XSCREEN)-GRIDXP
               YPHYC  = YPHY(YSCREEN)-GRIDYP
               XPHYC  = ANINT(XPHYC/GRIDDX)*GRIDDX+GRIDXP
               YPHYC  = ANINT(YPHYC/GRIDDY)*GRIDDY+GRIDYP
               XMSE0  = XPHYC
               YMSE0  = YPHYC
            ENDIF
         ENDIF
         IF (IFGRDP) THEN
C           polar grid
            XPHYP  = XPHY(XSCREEN)-GRIDXP
            YPHYP  = YPHY(YSCREEN)-GRIDYP
            RADIUS = SQRT(XPHYP**2+YPHYP**2)
            RADIUS = ANINT(RADIUS/GRIDR)*GRIDR
            IF (RADIUS.GT.0) THEN
               ANGLE = ATAN2(YPHYP,XPHYP)
               ANGLE = ANINT(ANGLE/GRIDA)*GRIDA
            ELSE
               ANGLE = 0.0
            ENDIF
            XPHYP = RADIUS*COS(ANGLE)+GRIDXP
            YPHYP = RADIUS*SIN(ANGLE)+GRIDYP
            XMSE0 = XPHYP
            YMSE0 = YPHYP
         ENDIF
         IF (IFGRDC.AND.IFGRDP) THEN
C           check for the closest
            RADC = (XPHYC-XPHY0)**2 + (YPHYC-YPHY0)**2
            RADP = (XPHYP-XPHY0)**2 + (YPHYP-YPHY0)**2
            IF (RADC.LT.RADP) THEN
               XMSE0 = XPHYC
               YMSE0 = YPHYC
            ELSE
               XMSE0 = XPHYP
               YMSE0 = YPHYP
            ENDIF
         ENDIF
         XMOUSE=XMSE0
         YMOUSE=YMSE0
         ZMOUSE=ZMSE0
      ENDIF
C     grid filtering complete
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRWOBJ(IOBJ)
      INCLUDE 'basics.inc'
C
      IOFF=1
      DO 100 I=1,IOBJ-1
         IOFF=IOFF+NPTS(I)
  100 CONTINUE
C
      CALL COLOR(9)
      CALL VDRAW(XOBJ(IOFF),YOBJ(IOFF),NPTS(IOBJ))
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE LATCHOB(X0,Y0,Z0,DIST2,IOBJ)
      INCLUDE 'basics.inc'
C
      IF (.NOT.IFOBJS) RETURN
C
      DSTMIN=10.0E08
      IOFF=1
      DO 100 I=1,IOBJ-1
         IOFF=IOFF+NPTS(I)
  100 CONTINUE
C
      DO 200 I=IOFF,IOFF+NPTS(IOBJ)
         DIST=(X0-XOBJ(I))**2+(Y0-YOBJ(I))**2
         IF (DIST.LT.DSTMIN) THEN
            X1=XOBJ(I)
            Y1=YOBJ(I)
            DSTMIN=DIST
         ENDIF
  200 CONTINUE
      IF (DSTMIN.LT.9.9E08) THEN
         DIST2 = DSTMIN
         X0=X1
         Y0=Y1
      ENDIF
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETOBJ
      INCLUDE 'basics.inc'
      CHARACTER*80 OBJFLE
C
C     Set up object, first pass, just get an object from a file.
C     Later, we allow functions, repositioning, sizing, rotations.
C
C
      IOFF=1
      DO 200 I=1,NOBJS
         IOFF=IOFF+NPTS(I)
  200 CONTINUE
c
      write(6,*) 'call ellgen',nobjs,ioff,npts(nobjs)
      call ellgen(xobj(ioff),yobj(ioff),n,a,b)
      if (a.gt.0.and.b.gt.0) then
         nobjs=nobjs+1
         npts(nobjs)=n
         IFOBJG(NOBJS)  = .TRUE.
         IFOBJS         = .TRUE.
         write(6,*) 'this is nobjs'
         ccobjs(nobjs)  = 'O'
         cobjs(1,nobjs) =  nobjs
         cobjs(2,nobjs) =  a
         cobjs(3,nobjs) =  b
         call rzero(cobjs(4,nobjs),2)
         CALL DRWOBJ(NOBJS)
      endif
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETZOOM
      INCLUDE 'basics.inc'
      LOGICAL IFTMP
C
      IFTMP =IFGRID
      IFGRID=.FALSE.
      CALL PRS('Input with mouse 2 points on edge of new frame.$')
      CALL PRS('Menu area for keybd input.$')
c     CALL PRS('Menu area to restore to original size.$')
C
      CALL MOUSE(XMSE1,YMSE1,BUTTON)
   10 CONTINUE
      IF (XSCR(XMSE1).LT.1.0) THEN
         CALL COLOR(1)
         CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
         CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
         CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
         CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
         CALL MOUSE(XMSE2,YMSE2,BUTTON)
         IF (XSCR(XMSE2).LT.1.0) THEN
            CALL COLOR(1)
            CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
            CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
            CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
            CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
C           draw a box around frame area
            DELTX=ABS(XMSE1-XMSE2)
            DELTY=ABS(YMSE1-YMSE2)
            DELT2=MAX(DELTX,DELTY)/2.0
            XZ0=(XMSE1+XMSE2)/2.0
            YZ0=(YMSE1+YMSE2)/2.0
            CALL MOVEC(XZ0-DELT2,YZ0-DELT2)
            CALL DRAWC(XZ0+DELT2,YZ0-DELT2)
            CALL DRAWC(XZ0+DELT2,YZ0+DELT2)
            CALL DRAWC(XZ0-DELT2,YZ0+DELT2)
            CALL DRAWC(XZ0-DELT2,YZ0-DELT2)
C
c           CALL PRS('Click in menu area if OK, or re-enter points.$')
            CALL PRS('Click in menu area for keybd or re-enter pts.$')
            CALL MOUSE(XMSE3,YMSE3,BUTTON)
            IF (XSCR(XMSE3).GE.1.0) THEN
               IFZOOM=.TRUE.
               DELTX=ABS(XMSE1-XMSE2)
               DELTY=ABS(YMSE1-YMSE2)
               DELT =MAX(DELTX,DELTY)
               XFAC =DELT
               YFAC =XFAC
               write(6,*) 'z:xfyf',xfac,yfac
               XZERO=(XMSE1+XMSE2)/2 - XFAC/2
               YZERO=(YMSE1+YMSE2)/2 - YFAC/2
C              reset clipping window
               WFR=1.3
               WT = YPHY(.995)
               WB = YPHY(.005)
               WL = XPHY(.005)
               WR = XPHY(.995)
               IF (IFPOST.AND.IFFULL) THEN
                  WR = XPHY(WFR)
c                 XFAC=XFAC/WFR
c                 YFAC=YFAC/WFR
c                 XZERO=WFR*( (XMSE1+XMSE2)/2 - XFAC/2)
c                 YZERO=WFR*( (YMSE1+YMSE2)/2 - YFAC/2)
               ENDIF
c              write(6,*) 'zoom:',xfac,xzero,yzero,wt,wb,wl,wr
c              write(7,*) 'zoom:',xfac,xzero,yzero,wt,wb,wl,wr
            ELSE
               CALL COLOR(0)
               CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
               CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
               CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
               CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
               CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
               CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
               CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
               CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
C              un-draw a box around frame area
               CALL MOVEC(XZ0-DELT2,YZ0-DELT2)
               CALL DRAWC(XZ0+DELT2,YZ0-DELT2)
               CALL DRAWC(XZ0+DELT2,YZ0+DELT2)
               CALL DRAWC(XZ0-DELT2,YZ0+DELT2)
               CALL DRAWC(XZ0-DELT2,YZ0-DELT2)
               XMSE1=XMSE3
               YMSE1=YMSE3
               CALL PRS(
     $         'Click 2nd point on edge of new frame, or in$')
               CALL PRS(
     $         'menu area to abort and restore to original size.$')
               GOTO 10
            ENDIF
         ELSE
            IFZOOM=.FALSE.
            CALL RESCAL
         ENDIF
      ELSE
c
c        1st click was in menu area... therefore use keybd input
c
         CALL PRS('Input two X,Y pairs (x1,y1,x2,y2)$')
         CALL PRS(' to bracket zoom area.$')
         CALL PRS('Input all zeros to restore screen.$')
         CALL PRS('Input 1 1 * * to cancel operation.$')
         CALL PRS('Input 2 2 s * to to set scale zoom by s.$')
         CALL PRS('Input 2 -2 s * to to set scale y zoom by s.$')
         CALL PRS('Input 3 3 sx sy to to scale x by sx, y by sy.$')
         CALL RERRRR(x1,y1,x2,y2)
c  
         if (x1.eq.x2.and.y1.eq.y2) then
            if (x1.eq.0.0.and.y1.eq.0.0) then
c             Restore
               IFZOOM=.FALSE.
               CALL RESCAL
            elseif (x1.eq.1.0.and.y1.eq.1.0) then
               CALL PRS('Aborting zoom request.$')
            endif
         else if (x1.eq.2. .and. y1.eq.2.) then
c           Scale current coordinates by scale
            if (x2.eq.0) x2=1.
            scale = 1./x2
            IFZOOM=.TRUE.
            delt =xfac*scale
            xzero=xzero + xfac/2 - delt/2
            yzero=yzero + yfac/2 - delt/2
            xfac =delt
            yfac =xfac
            write(6,*) 'z:xfyf',xfac,yfac
c
c           reset clipping window
            WFR=1.3
            WT = YPHY(.995)
            WB = YPHY(.005)
            WL = XPHY(.005)
            WR = XPHY(.995)
            IF (IFPOST.AND.IFFULL) THEN
               WR = XPHY(WFR)
            ENDIF
         else if (x1.eq.2. .and. y1.eq.-2.) then
c           Scale current y coordinates by scale
            if (x2.eq.0) x2=1.
            scale = 1./x2
            IFZOOM=.TRUE.
            delt =yfac*scale
            yzero=yzero + yfac/2 - delt/2
            yfac =delt
            write(6,*) 'z:xfyf',xfac,yfac
c
c           reset clipping window
            WFR=1.3
            WT = YPHY(.995)
            WB = YPHY(.005)
            WL = XPHY(.005)
            WR = XPHY(.995)
            IF (IFPOST.AND.IFFULL) THEN
               WR = XPHY(WFR)
            ENDIF
         else if (x1.eq.3. .and. y1.eq.3.) then
c           Scale x coordinates by x2, y by y2
            if (x2.eq.0) x2=1.
            if (y2.eq.0) y2=1.
            scalx = 1./x2
            scaly = 1./y2
            IFZOOM=.TRUE.
            deltx =xfac*scalx
            delty =yfac*scaly
            xzero=xzero + xfac/2 - deltx/2
            yzero=yzero + yfac/2 - delty/2
            xfac =deltx
            yfac =delty
            write(6,*) 'z:xfyf',xfac,yfac
c
c           reset clipping window
            WFR=1.3
            WT = YPHY(.995)
            WB = YPHY(.005)
            WL = XPHY(.005)
            WR = XPHY(.995)
            IF (IFPOST.AND.IFFULL) THEN
               WR = XPHY(WFR)
            ENDIF
         else
c
c        Got two (x,y) pairs
c
            xmse1 = x1
            ymse1 = y1
            xmse2 = x2
            ymse2 = y2
c
            CALL COLOR(1)
            CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
            CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
            CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
            CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
c           CALL MOUSE(XMSE2,YMSE2,BUTTON)
            CALL COLOR(1)
            CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
            CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
            CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
            CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
C           draw a box around frame area
            DELTX=ABS(XMSE1-XMSE2)
            DELTY=ABS(YMSE1-YMSE2)
            DELT2=MAX(DELTX,DELTY)/2.0
            XZ0=(XMSE1+XMSE2)/2.0
            YZ0=(YMSE1+YMSE2)/2.0
            CALL MOVEC(XZ0-DELT2,YZ0-DELT2)
            CALL DRAWC(XZ0+DELT2,YZ0-DELT2)
            CALL DRAWC(XZ0+DELT2,YZ0+DELT2)
            CALL DRAWC(XZ0-DELT2,YZ0+DELT2)
            CALL DRAWC(XZ0-DELT2,YZ0-DELT2)
C
c           CALL PRS('Click in menu area if OK, or re-enter points.$')
c           CALL MOUSE(XMSE3,YMSE3,BUTTON)
cccccc      IF (XSCR(XMSE3).GE.1.0) THEN
               IFZOOM=.TRUE.
               DELTX=ABS(XMSE1-XMSE2)
               DELTY=ABS(YMSE1-YMSE2)
               DELT =MAX(DELTX,DELTY)
               XFAC =DELT
               YFAC =XFAC
               write(6,*) 'z:xfyf',xfac,yfac
               XZERO=(XMSE1+XMSE2)/2 - XFAC/2
               YZERO=(YMSE1+YMSE2)/2 - YFAC/2
C              reset clipping window
               WFR=1.3
               WT = YPHY(.995)
               WB = YPHY(.005)
               WL = XPHY(.005)
               WR = XPHY(.995)
               IF (IFPOST.AND.IFFULL) THEN
                  WR = XPHY(WFR)
               ENDIF
ccccccc     endif
         endif
c.old    IFZOOM=.FALSE.
c.old    CALL RESCAL
      ENDIF
      IFGRID=IFTMP
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETRANS
      INCLUDE 'basics.inc'
      LOGICAL IFTMP
C
      IFTMP =IFGRID
      IFGRID=.TRUE.
      CALL DRGRID
      CALL PRS('Input with mouse 2 points on edge of new frame.$')
      CALL PRS('Menu area to exit.$')
C
      CALL MOUSE(XMSE1,YMSE1,BUTTON)
   10 CONTINUE
      IF (XSCR(XMSE1).LT.1.0) THEN
         CALL COLOR(1)
         CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
         CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
         CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
         CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
         CALL MOUSE(XMSE2,YMSE2,BUTTON)
         IF (XSCR(XMSE2).LT.1.0) THEN
            CALL COLOR(1)
            CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
            CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
            CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
            CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
C
            CALL PRS('Click in menu area if OK, or re-enter points.$')
            CALL MOUSE(XMSE3,YMSE3,BUTTON)
            IF (XSCR(XMSE3).GE.1.0) THEN
               IFZOOM=.TRUE.
               DELTX=ABS(XMSE1-XMSE2)
               DELTY=ABS(YMSE1-YMSE2)
               XZERO=XZERO+XMSE1-XMSE2
               YZERO=YZERO+YMSE1-YMSE2
C              reset clipping window
               WFR=1.3
               WT = YPHY(.995)
               WB = YPHY(.005)
               WL = XPHY(.005)
               WR = XPHY(.995)
               IF (IFFULL) THEN
                  WR = XPHY(WFR)
               ENDIF
            ELSE
               CALL COLOR(0)
               CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
               CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
               CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
               CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
               CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
               CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
               CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
               CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
               XMSE1=XMSE3
               YMSE1=YMSE3
               CALL PRS(
     $         'Click 2nd point in the domain, or in$')
               CALL PRS(
     $         'menu area to return.$')
               GOTO 10
            ENDIF
         ENDIF
      ENDIF
      IFGRID=IFTMP
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE RESCAL
      INCLUDE 'basics.inc'
c
      if (ifzoom) return
c
      XMAX = -1.0E20
      YMAX = -1.0E20
      XMIN =  1.0E20
      YMIN =  1.0E20
      nc = 4
      if (if3d) nc = 8
      DO 50 IEL=1,NEL
         DO 50 IC=1,nc
         XMAX = MAX(XMAX,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         XMIN = MIN(XMIN,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         YMAX = MAX(YMAX,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         YMIN = MIN(YMIN,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
50    CONTINUE
      IF(IFPOST)THEN
         IF(IF3D)THEN
C           Fill screen with object
            XFAC = MAX( (XMAX-XMIN) , (YMAX-YMIN)) *1.3
            YFAC = XFAC
C           Center on Screen
            XZERO = ((XMAX+XMIN)/2) - XFAC/2
            YZERO = ((YMAX+YMIN)/2) - YFAC/2
            IF (IFFULL) THEN
C              Fill screen with object
               DELTX=(XMAX-XMIN)/WFR
               XFAC = MAX( DELTX , (YMAX-YMIN)) *1.025
               YFAC = XFAC
C              Center on Screen
               XZERO = WFR*( ((XMAX+XMIN)/2) - XFAC/2)
C
C              quickie fix 12-5-91 pff
C
               XFAC=XFACO*5.0
               YFAC=YFACO*5.0
               XZERO=XZEROO
               YZERO=YZEROO
               YZERO =       ((YMAX+YMIN)/2) - YFAC/2
               write(6,*) 'r1xfyf',xfac,yfac
C
C
C
            ENDIF
         ELSE
C        For 2-d, leave xfac,yfac alone if THETA=90 and PHI=-90
c           IF (THETA.EQ.90.0.AND.PHI.EQ.-90.0.AND.IROT.EQ.0) THEN
            IF (THETA.EQ.99.0.AND.PHI.EQ.-99.0.AND.IROT.EQ.9) THEN
               XFAC=XFACO
               YFAC=YFACO
               XZERO=XZEROO
               YZERO=YZEROO
            ELSE
C              Fill screen with object
               XFAC = MAX( (XMAX-XMIN) , (YMAX-YMIN)) *1.3
               YFAC = XFAC
C              Center on Screen
               XZERO = ((XMAX+XMIN)/2) - XFAC/2
               YZERO = ((YMAX+YMIN)/2) - YFAC/2
            ENDIF
         ENDIF
      ELSE
         XFAC=XFACO
         YFAC=YFACO
c              XFAC=XFACO*5.0
c              YFAC=YFACO*5.0
c              write(6,*) 'r2xfyf',xfac,yfac
         XZERO=XZEROO
         YZERO=YZEROO
C        Put up in right upper corner  (obsolete, draw icon seperately)
c        XFAC = MAX( (XMAX-XMIN) , (YMAX-YMIN)) * 10.0
c        YFAC = XFAC
C        Center on Screen
c        XZERO = ((XMAX+XMIN)/2) - XFAC*.9
c        YZERO = ((YMAX+YMIN)/2) - YFAC*.9
      ENDIF
C     Reset clipping window
      WFR= 1.3
      WT = YPHY(.995)
      WB = YPHY(.005)
      WL = XPHY(.005)
      WR = XPHY(.995)
      IF (IFFULL) WR = XPHY(WFR)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine ellgen(x,y,n,a,b)
c
c     Generate a set of points on an ellipse (a,b)
c
      real x(0:1),y(0:1)
      real t(0:800)
c
      call prs('Input a,b (0 or <0 = quit) :$')
      call rerr(a,b)
      if (a.le.0 .or. b.le.0) then
         call prs('returning from ellgen, nothing done$')
         return
      endif
c
c     Recursively bisect the ellipse to get constant relative curvature
c
      b2i = 1.0/(b*b)
      a2i = 1.0/(a*a)
      x(0) = -a
      y(0) = 0.0
c
      dx  = .1
      dx4 = dx/4.
      do i=1,12
         x(i) = x(i-1)+dx4
         y(i) = b*sqrt(1.0-a2i*x(i)**2)
c        write(36,*) i,x(i),y(i)
      enddo
      do i=11,1000
         x(i) = x(i-1)+dx
         if (x(i).ge.0.0) goto 10
         y(i) = b*sqrt(1.0-a2i*x(i)**2)
c        write(36,*) i,x(i),y(i)
      enddo
   10 continue
      x(i) = 0.0
      y(i) =  b
c
      nnew = i
      do irecur = 1,7
         j = 0
         do i = 1,nnew
            call genpair(t(j),x(i-1),y(i-1),x(i),y(i),a,b)
c        write(39,6) j,t(j),t(j+1),y(i-1),y(i),x(i-1),x(i),i,irecur,a,b
            j=j+2
         enddo
    6    format(i3,6f9.5,2i5,2f5.2)
         nold = nnew
         nnew = j
         t(nnew) = y(nold)
c        write(38,*) 'this is nnew',nnew,nold,irecur,j
         call copy(y,t,nnew+1)
         do j=1,nnew-1
            disc = 1.0 - b2i*y(j)**2
            if (disc.le.0.0) then
               x(j) = 0.0
            else
               x(j) = -a*sqrt(1.0-b2i*y(j)**2)
            endif
c           write(38,8) j,x(j),y(j),b2i,disc,t(j)
    8    format(i5,5g13.5)
         enddo
         x(0) =    -a
         y(0) =    0.0
         x(nnew) = 0.0
         y(nnew) =  b
            j=0
c           write(38,8) j,x(j),y(j),b2i,disc
            j=nnew
c           write(38,8) j,x(j),y(j),b2i,disc
c
         if (nnew.gt.200) goto 20
      enddo
   20 continue
c
c     Now copy to  rh plane
c
      n = nnew
c     n = 2*nnew
c     do i=0,nnew-1
c        write(6,*) i,x(i),y(i)
c        x(n-i) = -x(i)
c        y(n-i) =  y(i)
c     enddo
c
      n = n+1
      write(6,*) 'Num obj segments:',n
      return
      end
c-----------------------------------------------------------------------
      subroutine genpair(ynew,x0,y0,x1,y1,a,b)
c
c     Generate new endpoints for bisected secants
c
      real ynew(0:1)
c
      ynew(0) = y0
c
c     Slope of perpendicular bisector
c
      if (y1.ne.y0) then
         d = -(x1-x0)/(y1-y0)
      else
         d = -9.e9
      endif
      e = 0.5*(y0+y1 - d*(x0+x1))
c
      ab2 = (a/b)**2
      AA = (1.+ab2*d**2)
      BB = 2.*d*e*ab2
      CC = ab2*(e**2 - (b**2))
c
c     write(60,*) 'a,b',a,b,aa,bb,cc
c     write(60,*) 'x,y',x0,y0,x1,y1
c     write(60,*) 'd,e',d,e,ab2
      if (BB.gt.0) then
        xt = 0.5*(-BB-sqrt(BB**2-4.*AA*CC))/AA
      else
        xt = 2.0*CC/(-BB+sqrt(BB**2-4.*AA*CC))
      endif
      xt = -abs(xt)
      ynew(1) = abs(d*xt+e)
c     write(60,*) 'x,t',xt,ynew(1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine refresh2
      include 'basics.inc'
c
      call clear
      call hard
      call heject
      call drmesh
      call drfram
      call nohard
c
      return
      end
c-----------------------------------------------------------------------
