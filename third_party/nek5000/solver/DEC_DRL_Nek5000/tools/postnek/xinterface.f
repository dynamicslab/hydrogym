      PROGRAM MAINF
c
c     Default window size
c
c     call set_ww_wh(ww,wh)
c     call set_ww_wh(850,800)
      call set_ww_wh(950,920)  ! std
c     call set_ww_wh(850,825)
c     call set_ww_wh(730,710)
c
c     Start nekton
c
      CALL MAINC
      STOP
      END
c-----------------------------------------------------------------------
      SUBROUTINE READER(VALUE,IERR)
      INCLUDE 'basics.inc'
      IERR=0
      REWIND(13)
      WRITE(13,'(A70)')LINE        
      REWIND(13)
      IF(LINE.EQ.' ')THEN
         VALUE=0
         RETURN
      ENDIF
      READ(13,*,ERR=2,END=2)VALUE
      REWIND(13)
      RETURN
2     IERR=1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE OPENF(IUNIT,FILE,STATUS,ITYPE,IERR)
      CHARACTER*(*) FILE,STATUS
      CHARACTER*30 NEWFILE,LFILE
      CHARACTER NULL
C
C     NULL out local filename character strings
      NULL = CHAR(0)
      DO 5 I=1,30
         LFILE  (I:I) = NULL
         NEWFILE(I:I) = NULL
5     CONTINUE
C
      LFILE = FILE
      IF(STATUS.EQ.'NEW' .OR. STATUS.EQ.'new') THEN
           NEWFILE=FILE
           do 10 i=30,1,-1
              if(ichar(newfile(I:I)).ne.0.and.newfile(I:I).ne.' ') then
                 newfile(I+1:I+1) = '~'
                 GO TO 20
              endif
 10           continue
 20           continue
           print*,'renaming ',file,' ',newfile
c          MVERR=RENAME(FILE,NEWFILE)
        ENDIF
        OPEN(UNIT=IUNIT,FILE=LFILE,STATUS=STATUS,ERR=1)
        IERR=0
        RETURN
1       IERR=1
        RETURN
        END
c-----------------------------------------------------------------------
        SUBROUTINE FINDFL(SESION,OLDVER)
        RETURN
        END
c-----------------------------------------------------------------------
        SUBROUTINE MOUSE(XMOUSE,YMOUSE,BUTTON)
        INCLUDE 'basics.inc'
	INTEGER*4 RECORD(3),SIZE,PET
	INTEGER ERRIND,MODE,ESW,XFORM,LDR,RETSIZ,ISTAT,DEVNO
	REAL LOCX,LOCY,EAREA(4)
        COMMON/MOUSEFAC/ WINDOW
C
C                                                Universal
C       BUTTON only recorded in interactive mode, not from journal
C       BUTTON=' ' for typed in input or when reading from journal
C       BUTTON='LEFT','RIGHT', or 'MIDDLE'
C	NEW! IMPROVED!!! Green button means "EXIT" end, etc
C       Return XMOUSE,YMOUSE, BUTTON; backspace and write x,y in journal file
20      CONTINUE
c
C       Look for button input on SUN
97      CONTINUE
        if (.not.ifgraf) then
           call prs('lmr? (left, middle, right)$')
           call res(ans,1)
           button = 'LEFT'   ! default 
c          if (ans.eq.'l' .or. ans.eq.'L') button = 'LEFT'
           if (ans.eq.'m' .or. ans.eq.'M') button = 'MIDDLE'
           if (ans.eq.'r' .or. ans.eq.'R') button = 'RIGHT'
           return
        endif

        IF(.NOT.IFDEMO)call XWAITM(IBUTTON,xm,ym)
C       IF(     IFDEMO)CALL XWAITM(IBUTTON,XM,YM)
        IF(     IFDEMO)READ(55,*)IBUTTON,XM,YM
c       IF(     IFDEMO)READ(5,*)IBUTTON,XM,YM

        IF(IFLEARN)WRITE(3,*)IBUTTON,XM,YM
        if(ibutton.eq.0) then
           CALL PRS('Still waiting for mouse input$')
           go to 97
	endif	
        IF(IBUTTON.EQ.1)BUTTON='LEFT'
        IF(IBUTTON.EQ.2)BUTTON='MIDDLE'
        IF(IBUTTON.EQ.3)BUTTON='RIGHT'
	window=1.3
        XSCREEN=(XM/WINDOWW*WINDOW) 
        YSCREEN=(YM/(WINDOWH-180.))-0.05
C     OLD VERSION -- coordinate transformation is almost
C                    device dependent!
C        YSCREEN=((YM-12)/1000.0*WINDOW) + 0.015
        IF(XSCREEN .EQ. OLDXSCREEN .AND.
     $     YSCREEN .EQ. OLDYSCREEN .AND. 
     $     CHOICE  .EQ.'ACCEPT CURRENT SWITHCES') THEN
           CALL PRS('Error reading mouse input; please move mouse'//
     $              'slightly and re-enter$')
           GO TO 97
        ENDIF
        OLDXSCREEN = XSCREEN
        OLDYSCREEN = YSCREEN
        IF(XSCREEN .LT. 0.0 .OR. XSCREEN .GT. 1.3  .OR.
     $     YSCREEN .LT. 0.0 .OR. YSCREEN .GT. 1.0) THEN
           CALL PRS('Error reading mouse input; please re-enter$')
           GO TO 97
        ENDIF
        IF(XSCREEN .EQ. 0.0 .AND. YSCREEN .EQ. 1.5e-2) THEN
           CALL PRS('Mouse not activated; please move mouse slightly'//
     $          'and re-enter$')
           GO TO 97
        ENDIF
C       FILTER X,Y THRU GRID
        GRID=PARAM(18)
        IF (GRID.LT. .0099) IFGRDC=.FALSE.
C       Filter Only if building and outside build menu area.
        IF(XSCREEN.LT.1.0 .AND.IFGRID.AND.BUTTON.NE.'RIGHT') THEN
           CALL FILTER(XMOUSE,YMOUSE,XSCREEN,YSCREEN)
        ELSE
           XMOUSE=XPHY(XSCREEN)
           YMOUSE=YPHY(YSCREEN)
        ENDIF
        IF(BUTTON.EQ.'MIDDLE')THEN
           CALL PRSR('Cursor Coordinates: x: $',XMOUSE)
           CALL PRSR('Cursor Coordinates: y: $',YMOUSE)
           GO TO 20
        ELSE IF(BUTTON.EQ.'LEFT'.OR.BUTTON.EQ.'RIGHT')then
        ENDIF

        RETURN
        END
c-----------------------------------------------------------------------
      SUBROUTINE MOVE(X,Y)
      CALL MOVESC(XSCR(X),YSCR(Y))
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRAW(X,Y)
      CALL DRAWSC(XSCR(X),YSCR(Y))
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE MOVESC(X,Y)
C     Moves with "Pen up", no line Drawn  IN SCREEN COORDINATES
      CHARACTER STRING*5
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)

      IF(IFILL.NE.0) THEN
C          Currently Filling region defined by moves and draws
           IF(NFILL.GT.210)THEN
C             Forgot somewhere to shut off fill
              IFILL=0
              NFILL=0
              RETURN
           ENDIF
           NFILL = NFILL+1
           XFILL(NFILL) = SCOORDX(X)
           YFILL(NFILL) = SCOORDY(Y)
      ENDIF
      CALL XMOVE(SCOORDX(X),SCOORDY(Y))
c     write(6,*) ifposts,ifhard,ifhard0
      IF(IFPOSTS.AND.IFHARD.AND.IFHARD0)
     $WRITE(45,'(2F10.3,A3)')REAL(550-Y*500),REAL(X*500+50),' m '
c12/23/92/pff $WRITE(45,'(2F10.3,A6)')REAL(550-Y*500),REAL(X*500+50),' move '
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DRAWSC(X,Y)
C
C     Draws lines for various devices with "Pen down" IN SCREEN COORDINATES
      CHARACTER STRING*5
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)
C
      IF(IFILL.NE.0) THEN
C          Currently Filling region defined by moves and draws
           IF(NFILL.GT.210)THEN
C             Forgot somewhere to shut off fill
              IFILL=0
              NFILL=0
              RETURN
           ENDIF
           NFILL = NFILL+1
           XFILL(NFILL) = SCOORDX(X)
           YFILL(NFILL) = SCOORDY(Y)
      ENDIF
      CALL XDRAWTO(SCOORDX(X),SCOORDY(Y))
      IF(IFPOSTS.AND.IFHARD.AND.IFHARD0)
     $WRITE(45,'(2F9.3,A3)')REAL(550-Y*500),REAL(X*500+50),' d '
c12/23/92/pff $WRITE(45,'(2F9.3,A6)')REAL(550-Y*500),REAL(X*500+50),' draw '
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BEGINP(X1,Y1)
C     Begins panel for fill  NO BOUNDARY DISPLAY
      LOGICAL IFTMP 
      CHARACTER STRING*5
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)
C
      X=(X1-XZERO)/XFAC
      Y=(Y1-YZERO)/YFAC
C
      ICFILL=0
      IFILL = 1
      NFILL = 0
      CALL MOVEC(X1,Y1)
      IFTMP=IFHARD
      IFHARD=.FALSE.
      CALL MOVE(X1,Y1)
      IFHARD=IFTMP
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BEGINB(X1,Y1)
C     Begins panel for fill WITH BOUNDARY DISPLAY
      CHARACTER STRING*5
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)
      LOGICAL IFTMP 
      X=(X1-XZERO)/XFAC
      Y=(Y1-YZERO)/YFAC
      ICFILL=1
      IFILL = 1
      NFILL = 0
      CALL MOVEC(X1,Y1)
      IFTMP=IFHARD
      IFHARD=.FALSE.
      CALL MOVE(X1,Y1)
      IFHARD=IFTMP
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine gnable
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE GINDIS
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine sgvis(iseg,ivisibility)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE HEJECT
c
C     EJECTS page from hard copy laser, and closes/opens new plot file
c
      INCLUDE 'devices.inc'
c
      character*6  s6
      character*7  s7
      character*40 fname
      character*1  fname1(40)
      equivalence (fname,fname1)
c
      IF(IFHARD.AND.IFHARD0) THEN
c
c        IF(IFPOSTS)WRITE(45,*)' showpage'
c        IF(IFPOSTS)CALL PENW(3)
c
         IF (IFPOSTS) then
            WRITE(45,*)' showpage'
            close(unit=45)
c
c           Update filename stuff for successive plot files
c
            fname    = fplotnam
            m        = iplotnam
            iplotnum = iplotnum+1
c
            if (iplotnum.le.99) then
               write(s6,6) iplotnum
    6          format('.plt',i2.2)
c              fname(m:m+5)=s6
               call chcopy(fname1(m),s6,6)
            else
               write(s7,7) iplotnum
    7          format('.plt',i3.3)
c              fname(m:m+6)=s7
               call chcopy(fname1(m),s7,7)
            endif
            CALL OPENF(45,fname,'NEW',3,IERR)
            CALL INITPS
            CALL PENW(3)
         endif
c
         IF(IFLASE)WRITE(45,*)' ^,'
c
      ENDIF
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CLSSEG
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE OPNSEG(ISEG)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DELSEG(ISEG)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE PENW(IWIDTH)
C     Sets Pen width of laser. IWIDTH should be an odd integer 1 to 31
C     Default is 3.
      INCLUDE 'devices.inc'
      INTEGER IWDOLD
      SAVE    IWDOLD
      DATA    IWDOLD /-99999/ 
      IF (IWIDTH.EQ.IWDOLD) RETURN
C
      IWDOLD=IWIDTH
      IF(IFHARD.AND.IFHARD0) THEN
        IF(IFPOSTS)
     $    WRITE(45,'(f7.2,a14)')REAL(IWIDTH/15.),' setlinewidth '
        IF(IFLASE)WRITE(45,100) IWIDTH
      ENDIF
100   FORMAT(' ^PW',I2.2)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE SCROLL(ILINES)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE RUBBER
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE OFFRUB
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GWRITE(X,Y,SIZE,STRING)
      CHARACTER STRING(100)
      COMMON /CLIP/ XMOVEC,YMOVEC,WT,WB,WL,WR
      CALL GSWRIT(XSCR(X),YSCR(Y),SIZE,STRING)     
c     changed 2/27/00 pff
c     if ((wl.lt.x.and.x.lt.wr) .and. (wb.lt.y.and.y.lt.wt)) 
c    $   CALL GSWRIT(XSCR(X),YSCR(Y),SIZE,STRING)     
      RETURN
      END      
c-----------------------------------------------------------------------
      SUBROUTINE GSWRIT(X,Y,SIZE,STRING)
      CHARACTER STRING(100),DIR,STRING2(100)
      COMMON /FONT/ IFONT,IDIR
      INCLUDE 'devices.inc'
      DATA IFONT,IDIR /204,1/
      NULL=CHAR(0)
      DO 1 I=1,100
          STRING2(i)=STRING(I)
          IF(STRING(I).EQ.'$') THEN
                STRING2(i)=NULL
                NCHARS=I-1
                GO TO 2
          ENDIF
1     CONTINUE
      CALL PRS('PUT DOLLAR SIGN AFTER LAST CHARACTER!$')
2     CONTINUE
      CALL XPUTXT(SCOORDX(X),SCOORDY(Y),SIZE,STRING2,NCHARS)
      IF(IFPOSTS.AND.IFHARD.AND.IFHARD0) THEN
          CALL MOVESC(x,y)
          write(45,*)' 90 rotate'
          write(45,'(80A1)') '(',(STRING(I),I=1,NCHARS),')',' ','s',
     $    'h','o','w'
          write(45,*)'-90 rotate'
      ENDIF
5     RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DEVINI(DEVICE)
      INCLUDE 'basics.inc'
      
      IFNOSEG= .TRUE.

      IFXWIN = .TRUE.

      CALL GETWIN(XSIZE,YSIZE)
      WINDOWW=XSIZE
      WINDOWH=YSIZE
c
c
c     Pixel map default size
c
      xpmn = 0.
      ypmn = 0.
      xpmx = xsize
      ypmx = ysize
c
      CALL XPLANES(NPLANES)
      IFBWGKS = .FALSE.
      IF(NPLANES .EQ. 1) IFBWGKS = .TRUE.
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE DEVEX
      INCLUDE 'devices.inc'
C
C     Set Screen to scroll
      IF(IFPOSTS)WRITE(45,*)' showpage'
      IF(IFPOSTS)       CLOSE(UNIT=45)
      CALL CLOSEW
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE ENDP
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)
      IF (NFILL.GT.210) THEN
         NFILL=210 
         XFILL(NFILL)=XFILL(1)
         YFILL(NFILL)=YFILL(1)
      ENDIF
      CALL XPOLY(XFILL,YFILL,NFILL,ICFILL)
      IFILL = 0
      NDRAW = NFILL
      NFILL = 0
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE FILLP(ICOLOR)
C     SELECTS FILL panel
      CHARACTER STRING*5
      INCLUDE 'devices.inc'
      CALL XSETFILL(IABS(ICOLOR))
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CLEAR
C     clears screen (graphics portion)
      INCLUDE 'devices.inc'
c
      if (ifrevbk) then
         iback = 1
      else
         iback = 0
      endif
c     write(6,*) 'call xclear:',iback,ifrevbk
      CALL XCLEAR(iback)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CLEARA
C     clears screen
      CALL XCLEARA
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE COLOR(Incolor)
C     SETS COLOR OF LINES TO BE
C     WHICHEVER DRAW IS TO BE THIS COLOR
      EQUIVALENCE(CCOLOR,ICDUM)
      include 'devices.inc'

c     character*4 colors(0:200),ccolor
      CHARACTER*3 COLORS(0:200),CCOLOR

      logical lrevbk
C
C     Save color for use with IFSAVE DRAW option.
C
      IF (InCOLOR.EQ.LSTCLR) return
c     IF (InCOLOR.EQ.LSTCLR  .and.
c    $   ((ifrevbk.and.lrevbk).or..not.(ifrevbk.or.lrevbk)) ) RETURN
c     write(6,*) 'this is incolor',incolor,ifrevbk
c     lrevbk = ifrevbk
      LSTCLR=InCOLOR
C
      ICOLOR = Incolor
      NCOLOR = 15
      icolor = max(icolor,0)
      icolor = min(icolor,ncolor)
      ICDUM  = ICOLOR
      INDEX=ICOLOR
      IF(IFBWGKS .AND. ICOLOR .NE.0) INDEX=1
C
      if (ifrevbk) then
c
c        Reverse for white background, black lines
c
         COLORS(1)='BLACK'
         COLORS(0)='WHITE'
         if (incolor.eq.0) icolor = 1
         if (incolor.eq.1) icolor = 0
         INDEX=ICOLOR
c
      else
c
c        Standard
c
         COLORS(0)='BLACK'
         COLORS(1)='WHITE'
c
      endif
c
c
      COLORS(2)='RED'
      COLORS(3)='GREEN'
      COLORS(4)='BLUE'
      COLORS(5)='LTBLUE'
      COLORS(6)='PURPLE'
      COLORS(7)='YELLOW'
      COLORS(8)='ORANGE'
      COLORS(9)='GREEN'
      COLORS(10)='LTGREEN'
      COLORS(11)='MEDBLUE'
      COLORS(12)='VIOLET'
      COLORS(13)='PINK'
      COLORS(14)='GRAY'
      COLORS(15)='LGRAY'
C
      
      DO 1 I=0,NCOLOR
          IF(CCOLOR.EQ.COLORS(I)) INDEX=I
1     CONTINUE

      CALL XSETPEN(INDEX)
c     write(6,*) index,incolor,' color'
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CLRMAP(TYPE)
      CHARACTER*4 TYPE
      REAL RED(0:15),GREEN(0:15),BLUE(0:15)
      REAL MESHMP(4,0:15),HEATMP(4,0:15)
C
C      DATA MESHMP / 0,      0,       0,       0,
C     $            1,      61440,   61440,   61440,
C     $            2,      61440,   0,       0,
C     $            3,      0,       61440,   0,
C     $            4,      0,       0,       61440,
C     $            5,      12288,   40960,   53248,
C     $            6,      40960,   12288,   53248,
C    $            7,      61440,   61440,   0,
C     $            8,      61440,   32748,   0,
C     $            9,      0,       61440,   0,
C     $            10,     0,       61440,   32768,
C     $            11,     12288,   12288,   53248,
C     $            12,     20480,   12288,   20480,
C     $            13,     61440,   0,       61440,
C     $            14,     20480,   20480,   20480,
C     $            15,     45056,   45056,   45056  /
C                         RED      GREEN    BLUE

C      DATA HEATMP / 0,      0,       0,       0,
C     $            1,      61440,   61440,   61440,
C     $            2,      61440,   0,       0,
C     $            3,      0,       61440,   0,
C     $            4,      0,       0,       61440,
C     $            5,      12288,   40960,   53248,
C     $            6,      40960,   12288,   53248,
C     $            7,      61440,   61440,   0,
C     $            8,      61440,   32748,   0,
C     $            9,      0,       61440,   0,
C     $            10,     0,       61440,   32768,
C     $            11,     12288,   12288,   53248,
C     $            12,     20480,   12288,   20480,
C     $            13,     61440,   0,       61440,
C     $            14,     20480,   20480,   20480,
C     $            15,     45056,   45056,   45056  /
C                         RED      GREEN    BLUE

      
      DATA MESHMP /    0.,      .1,    .1,   .1,
     $                 1.,      1.0,    1.0,   1.0,
     $                  2.,      1.0,    0.0,   0.0,
     $                  3.,      1.0,    1.0,	0.0,
     $                  4.,      0.5,    0.5,	0.5,
     $                  5.,      0.0,    .0,     1.0,
     $                  6.,      1.0,    0.0,	0.7,
     $                  7.,      1.0,    1.0,	0.0,
     $                  8.,      0.7,    1.0,	0.0,
     $                  9.,      0.2,    1.0,	0.2,
     $                  10.,     0.0,    1.0,	1.0,
     $                  11.,     0.0,    0.3,	1.0,
     $                  12.,     1.0,    0.0,	1.0,
     $                  13.,     1.0,    0.0,	0.2,
     $                  14.,     0.6,    0.6,	0.6,
     $                  15.,     0.5,    0.5,	0.7 /
C                               RED     Green   BLUE
C
C
      DATA HEATMP /     0.0,     0.0,    0.0,	0.0,
     $                  1.,      1.0,    1.0,   1.0,
     $                  2.,      0.0,    0.0,   1.0,
     $                  3.,      0.0,    0.0,	1.0,
     $                  4.,      0.0,    0.1,	1.0,
     $                  5.,      0.1,    0.2,	0.9,
     $                  6.,      0.2,    0.7,	0.8,
     $                  7.,      0.4,    0.9,	0.6,
     $                  8.,      0.6,    1.0,	0.4,
     $                  9.,      0.7,    1.0,	0.3,
     $                  10.,     0.8,    0.9,	0.2,
     $                  11.,     0.9,    0.7,	0.1,
     $                  12.,     1.0,    0.2,	0.0,
     $                  13.,     1.0,    0.1,	0.0,
     $                  14.,     1.0,    0.0,	0.0,
     $                  15.,     1.0,    0.0,	0.0 /
C                               RED     Green   BLUE
C

      IF(TYPE.EQ.'HEAT') THEN
          DO 10 i=0,15
             RED   (i)= HEATMP(2,I)
             GREEN (i)= HEATMP(3,I)
             BLUE  (i)= HEATMP(4,I)
10        CONTINUE
          DO 11 i=2,15
             RED(i)  = (i -2)/13.
             blue(i) = (15-i)/13.
             green(i)= amin1(blue(i),red(i)) *2.0
 11          CONTINUE
      ENDIF
      IF(TYPE.EQ.'MESH') THEN
          DO 20 i=0,15
             RED   (i)= MESHMP(2,I)
             GREEN (i)= MESHMP(3,I)
             BLUE  (i)= MESHMP(4,I)
20        CONTINUE
      ENDIF
C     scale colormap up by 61440
      DO 30 I=0,15
         RED(I)=RED(I)*61440.0
         GREEN(I)=GREEN(I)*61440.0
         BLUE(I)=BLUE(I)*61440.0
 30   CONTINUE
C     white text
      RED(1)=61440.0
      GREEN(1)=61440.0
      BLUE(1)=61440.0
      CALL XCMAP(RED,GREEN,BLUE,15)
      CALL COLOR(1)
      RETURN
      END
c-----------------------------------------------------------------------
      function scoordx(worldx)
      include 'basics.inc'
      scoordx=worldx/1.3*windoww
      return
      end
c-----------------------------------------------------------------------
      function scoordy(worldy)
      include 'basics.inc'
      scoordy=abs(worldy-1.0)*(windowh - 180.0)
      return
      end
c-----------------------------------------------------------------------
      function scoordx_inv(pix)
      include 'basics.inc'
c     scoordx=worldx/1.3*windoww
      scoordx_inv=0.
      if (windoww.ne.0) scoordx_inv=1.3*pix/windoww
      return
      end
c-----------------------------------------------------------------------
      function scoordy_inv(pix)
      include 'basics.inc'
c     scoordy=abs(worldy-1.0)*(windowh - 180.0)
      scoordy_inv=1. +   pix/(windowh - 180.0)
      return
      end
c-----------------------------------------------------------------------
