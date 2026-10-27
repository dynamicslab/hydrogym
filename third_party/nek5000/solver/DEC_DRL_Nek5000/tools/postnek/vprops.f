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
      SUBROUTINE VPROPS
C
      INCLUDE 'basics.inc'
      CHARACTER*26 OLDCHOICE
      CHARACTER*80 LINE80,ITEMD
C       Sets the elements that are used for conduction only
C       Must be done AFTER mesh is drawn
C
      LINE80=' '
      IF1=2
      IF(IFFLOW)IF1=1
1     NCHOIC =  5
      ITEM(1)='ACCEPT MATL,QVOL'
      ITEM(2)='SELECT ELEMENT'
      ITEMD  ='SELECT MATL [   ]'
      WRITE(ITEMD(14:16),'(I3)')IGRP
      ITEM(3)=ITEMD
      ITEM(4)='SET  MATL PROPS,Q'
      ITEM(5)='SHOW MATL PROPS,Q'
      ITEM(6)='ZOOM'
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'VARIABLE PROPERTY')
C
      OLDCHOICE=CHOICE
      IF(CHOICE(1:11).EQ.'SELECT MATL')THEN
         CALL PRSI('CURRENT MATERIAL:$',IGRP)
         CALL PRS('Please SELECT MATERIAL with mouse on keypad$')
111      CALL PRS(' Enter Material Group Number:$')
         CALL KEYPAD(RDUMMY)
         IGRPx=RDUMMY
         IF(IGRPx.LE.0)THEN
            CALL PRS(' Fluid elements$')
         ELSE
            CALL PRS(' Conduction only (no fluid) elements$')
         ENDIF
         IF(IGRPx.LT.0 .AND. .NOT. IFFLOW) THEN
            CALL PRS
     $      ('ERROR: GROUP MUST BE GE 0 FOR CONDUCTION ELEMENTS$')
            CALL BEEP
            GOTO 1
         ELSE IF(IGRPx.GT.0 .AND. .NOT. IFHEAT) THEN
            CALL PRS('ERROR: GROUP MUST BE LE 0 FOR FLUID ELEMENTS$')
            CALL BEEP
            GOTO 1
         ELSE IF(IGRPx.GT.10 .OR. IGRPx .LT. -5) THEN
            CALL PRS
     $      ('Material Group number must be between -5 and 10.$')
            CALL BEEP
            GOTO 1
         ENDIF
C        Really change group only here
         IGRP = IGRPx
      ELSE IF(CHOICE.EQ.'SET  MATL PROPS,Q')THEN
C        specify properties
C        EQUIVALENCE FOR CONDUCT,...??!!
C        IF GROUP .le.0 then fluids stuff too
         IIF1=IF1
         IF(IGRP .GE.1) IIF1=2
         DO 122 IF=IIF1,NFLDS
            IF(NKTONV.EQ.1)THEN
               CHOICE='CONSTANT'
            ELSE
               CALL PRS('Enter form properties are to be entered$')
               WRITE(S,'(1X,A10,I3,A15,I4)') 'For field ',IF,
     $         ' and Matl Group ',IGRP
               CALL PRS(S//'$')
               ITEM(1)='CONSTANT'
               ITEM(2)='FORTRAN FUNCTION'
               ITEM(3)='KEEP DEFAULTS'
               NCHOIC=3
               CALL MENU(XMOUSE,YMOUSE,BUTTON,'FORTRAN FUNCTION')
            ENDIF
            IF(CHOICE.EQ.'KEEP DEFAULTS')THEN
               MATYPE(IGRP,IF) = 0
               CALL PRS('Using DEFAULT values from PARAMETER Menu$')
               GOTO 121
            ENDIF
            IF(IF.EQ.1)CALL PRS(' Enter Variable Dynamic Viscosity$')
            IF(IF.EQ.2)CALL PRSI(
     $      ' Enter CONDUCT, RHOCP, QVOL for Material $',IGRP)
            IPSCAL=IF-2
            IF(IF.GT.2)WRITE(S,'(1X,A27,I4,A34,I4)')
     $      ' Enter (FOR PASSIVE SCALAR #',ipscal,
     $      'CONDUCT, RHOCP, QVOL for Material ',IGRP
            CALL PRS(S//'$')
            IF(CHOICE.EQ.'CONSTANT')THEN
               IF(IGRP.EQ.0)THEN
                  CALL PRS('*** ERROR ***$')
                  CALL PRS
     $            (' If properties in group zero are constants,$')
                  CALL PRS
     $            (' they can be set in the SET PARAMETER menu.$')
                  CALL PRS(' You cannot override those values here '//
     $            'except with a FORTRAN FUNCTION.$')
                  CALL PRS(' Please change them in the SET PARAMETER '//
     $            'menu in a new session.$')
                  GOTO 1
               ENDIF
               MATYPE(IGRP,IF)=1
C              Make sure Fortran functions get turned off
               VPROP(IGRP,IF,1)='C'
               VPROP(IGRP,IF,2)='C'
               VPROP(IGRP,IF,3)='C'
               IF(IF.EQ.1)THEN
                  CALL PRS(' Enter Dynamic Viscosity $')
                  CALL KEYPAD(CPROP(IGRP,IF,1))
                  CALL PRS(' Enter Density $')
                  CALL KEYPAD(CPROP(IGRP,IF,2))
               ELSE
                  CALL PRS(' Enter CONDUCT $')
                  IF(IF.GT.2)CALL PRSI
     $            ('FOR PASSIVE SCALAR FIELD$',IPSCAL)
                  CALL KEYPAD(CPROP(IGRP,IF,1))
                  CALL PRS(' Now enter RHOCP $')
                  CALL KEYPAD(CPROP(IGRP,IF,2))
                  CALL PRS(' Now enter QVOL $')
                  CALL KEYPAD(CPROP(IGRP,IF,3))
               ENDIF
            ELSE IF(CHOICE.EQ.'FORTRAN FUNCTION')THEN
               MATYPE(IGRP,IF) = 2
               IP=IF-2
               DO 225 IPROP=1,3
221            CONTINUE
C              Make sure Constants are zero
               CPROP(IGRP,IF,IPROP)=0.0
               CALL PRS('Enter fortran statement for quantity$')
C
               IF(IPROP.EQ.1)THEN
C                UDIFF
                 IF(IF.EQ.1) THEN
                   CALL PRS(
     $             'Variable Dynamic viscosity (mu, not nu) EXAMPLE:$')
                   CALL PRS
     $             ('VISCOS=2.0+.01*TEMP + X**2 + Y + EXP(-TIME)$')
                 ELSE IF (IF.EQ.2) THEN
                   CALL PRS('Thermal conductivity CONDUCT.  EXAMPLE:$')
                   CALL PRS
     $             ('CONDUCT=2.0+.01*TEMP + X**2 + Y + EXP(-TIME)$')
                 ELSE IF (IF.GT.2)THEN
                   CALL PRS(
     $           'Variable Diffusivity for passive scalar field.  EX:$')
                   CALL PRS
     $             ('DIFF=2.0+.01*TEMP + X**2 + Y + EXP(-TIME)$')
                 ENDIF
               ENDIF
               IF(IPROP.EQ.2)THEN
C                UTRANS
                 IF(IF.EQ.1)THEN
                   CALL PRS(
     $             'DENSITY.  Density should be a constant.  EX:$')
                   CALL PRS('DENSITY = 4.52$')
                 ELSE IF (IF.EQ.2)THEN
                   CALL PRS('Variable RHOCP  Ex:$')
                   CALL PRS('RHOCP = 2.0 + 0.01*TEMP +0.01*Y$')
                 ELSE IF (IF.GT.2)THEN
                   CALL PRS('CONST1 for passive Scalar field.$')
                   CALL PRS
     $             ('This is analogous to RHOCP in heat problem.$')
                   CALL PRS
     $             ('For diffusion problems, CONST1 always = 1.$')
                   CALL PRS('Ex:$')
                   CALL PRS('CONST1 = 1.0$')
                 ENDIF
               ENDIF
               IF(IPROP.EQ.3)THEN
                  IF(IF.EQ.1)THEN
C                    Forcing function not done here
                     LINE80=' '
                     GO TO 224
                  ELSE IF(IF.EQ.2)THEN
                     CALL PRS('Variable QVOL$')
                     CALL PRS('Enter fortran statement  EXAMPLE:$')
                     CALL PRS
     $               ('QVOL=2.0+.01*TEMP + X**2 + Y + EXP(-TIME)$')
                  ELSE IF(IF.GT.2)THEN
                     CALL PRSI(
     $               'Variable SOURCE for passive scalar field$',ip)
                     CALL PRS('Enter fortran statement  EXAMPLE:$')
                    CALL PRS
     $              ('SOURCE=2.0+.01*TEMP + X**2 + Y + EXP(-TIME)$')
                  ENDIF
               ENDIF
               LINE80=' '
               CALL RES(LINE80(8:80),73)
               IF(IFLEARN)WRITE(3,'(A73)')LINE80(8:80)
               IF(LINE80.EQ.' ')THEN
                  CALL PRS('ERROR, BLANK LINES NOT ALLOWED.  '//
     $            'MUST SPECIFY FORTRAN STATEMENT.$')
                  VPROP(IGRP,IF,IPROP)='C'
                  GOTO 221
               ELSE
                  LINE80(1:7)='       '
                  VPROP(IGRP,IF,IPROP)=LINE80
               ENDIF
225            CONTINUE
224            CONTINUE
            ENDIF
121      CONTINUE
122      CONTINUE
         GOTO 1
      ELSE IF(CHOICE.EQ.'SHOW MATL PROPS,Q')THEN
         DO 123 I=-5,10
            DO 124 IF=IF1,NFLDS
               IF(IF.EQ.IGRPX .AND. MATYPE(I,IF).EQ.0)THEN
                  IF(IF.EQ.1)CALL PRSI
     $            ('Default Parameters for Group 0 field$',IF)
               ELSE IF(MATYPE(I,IF).EQ.1)THEN
                  IF(IF.EQ.1)CALL PRSR('Constant Viscosity of $',
     $            CPROP(I,IF,1))
                  IF(IF.EQ.1)CALL PRSR('Constant Density of $',
     $            CPROP(I,IF,2))
                  IF(IF.GT.1)THEN
                     WRITE(S,'(1X,A31,I3,A7,I3)')
     $               'Constants for elements in group',I,
     $               ' field ',IF
                     CALL PRS(S//'$')
                     WRITE(S,'(1X,A7,G11.3,A6,G11.3,A5,G11.3)')
     $               'CONDUCT',CPROP(I,IF,1),
     $              ' RHOCP',CPROP(I,IF,2),' QVOL',CPROP(I,IF,3)
                     CALL PRS(S//'$')
                  ENDIF
               ELSE IF(MATYPE(I,IF).EQ.2)THEN
                  WRITE(S,'(1X,A27,I3,A7,I3)')
     $            'Fortran functions for group',I,' field ',IF
                  CALL PRS(S//'$')
                  DO 1123 IPROP=1,3
                     CALL PRS(VPROP(I,IF,IPROP)//'$')
1123              CONTINUE
               ENDIF
124         CONTINUE
123      CONTINUE
      ELSE IF(CHOICE.EQ.'ACCEPT MATL,QVOL')THEN
C        First check that all groups have req'd parameters set
C        Can't really check it if it's a fortran function
         DO 89 IEL=1,NEL
            DO 89 IF=IF1,NFLDS
               IGRP=IGROUP(IEL)
C              IF(MATYPE(IGRP,IF) .EQ.0 .AND.IGRP .NE.0)THEN
C                  CALL PRSI('ERROR: Properties not set for material'
C     $            //' group$',IGRP)
C                  GOTO 1
C              ENDIF
               IF(IGRP.NE.0)THEN
                  IF(MATYPE(IGRP,IF) .EQ. 1)THEN
C                    Can't check anything about vicosity.
                     IF(IF.GT.1.AND.CPROP(IGRP,IF,1) .LE.0.0)THEN
C     $                 .or. CPROP(IGRP,IF,2) .LE.0.0 .AND. IFTRAN)THEN
                        WRITE(S,'(1X,A38,A6,I4,A7,I4)')
     $                  'ERROR: Properties not set for material'
     $                  ,' group',IGRP,' IFIELD',IF
                        CALL PRS(S//'$')
                        GOTO 1
                     ENDIF
                  ENDIF
               ENDIF
89       CONTINUE
C        recalculate NCOND
         NCOND=0
         do 82 I=1,NEL
            IF(IGROUP(I).GT.0)NCOND=NCOND+1
82       CONTINUE
         NELF=NEL-NCOND
C        SET mask of all elements
C        Change mask of all elements that have FLUID mesh.
         DO 350 IELC=1,NEL
          DO 350 IFLD=1,NFLDS
            IF(.NOT. IFTMSH(IFLD))THEN
               IF(IGROUP(IELC).LE.0 )MASKEL(IELC,IFLD) = 1
               IF(IGROUP(IELC).GT.0 )MASKEL(IELC,IFLD) = 0
            ENDIF
350       CONTINUE
C        SHUFFLE ELEMENT NUMBERS
         IF(NCOND.GT.0)THEN
C           REQUIRES HEAT TRANSFER FIELD
C           If an element is a conduction element and it's current number
C           is <= NELF, Swap it with the highest # fluid element
            DO 11 IEL=1,NELF
               IF(MASKEL(IEL,1).EQ.0)THEN
C                 Find fluid element to swap it with
                  DO 8 JEL=NELF+1,NEL
                     IF(MASKEL(JEL,1).EQ.1)THEN
C                       Do the swap; first copy fluid to last
                        CALL COPYEL(JEL ,NELM-1)
                        CALL COPYEL(IEL   ,JEL)
                        CALL COPYEL(NELM-1,IEL)
                        CALL DRAWEL(IEL)
                        CALL DRAWEL(JEL)
                        GOTO 9
                     endif
8                 CONTINUE
                  CALL PRS(' Error: Conduction elements not last$')
9                 CONTINUE
               ENDIF
11          CONTINUE
         ENDIF
         RETURN
      ELSE IF(CHOICE.EQ.'SELECT ELEMENT'.AND.NLEVEL.GT.1)THEN
2        NCHOIC =  4
C        NEED TO DRAW LEVEL FIRST TIME AROUND?
         ITEM(1)='UP   LEVEL'
         ITEM(2)='DOWN LEVEL'
         ITEM(3)='CHOOSE ELEMENT'
         ITEM(4)='MAIN MENU'
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'CHOOSE ELEMENT')
         IF(NCOND.GT.0)CALL PRSIS(' CURRENTLY$',NCOND,
     $   ' CONDUCTION ELEMENTS SPECIFIED$')
         IF(CHOICE.EQ.'UP   LEVEL' .OR.CHOICE.EQ.'DOWN LEVEL')THEN
C           Erase old mesh& draw new
            IF(CHOICE.EQ. 'UP   LEVEL'.AND.ILEVEL.EQ.NLEVEL)THEN
               CALL PRS(' ERROR: AT TOP LEVEL ALREADY$')
               GOTO 2
            ENDIF
            IF(CHOICE.EQ. 'DOWN LEVEL'.AND.ILEVEL.EQ.1)THEN
               CALL PRS(' ERROR: AT FIRST LEVEL ALREADY$')
               GOTO 2
            ENDIF
            IF(CHOICE.EQ.'UP   LEVEL')
     $      CALL DRELEV(ILEVEL+1,ILEVEL,'     ')
            IF(CHOICE.EQ.'DOWN LEVEL')
     $      CALL DRELEV(ILEVEL-1,ILEVEL,'     ')
            DO 500 I=1,NEL
                IF(NUMAPT(I).EQ.ILEVEL) CALL DRAWEL(-I)
500         CONTINUE
            IF(CHOICE.EQ. 'UP   LEVEL')ILEVEL=ILEVEL+1
            IF(CHOICE.EQ. 'DOWN LEVEL')ILEVEL=ILEVEL-1
            DO 510 I=1,NEL
               IF(NUMAPT(I).EQ.ILEVEL)THEN
               CALL DRAWEL( I)
               ENDIF
510         CONTINUE
            GOTO 2
         ELSE IF(CHOICE.EQ.'CHOOSE ELEMENT')THEN
            CALL PRS
     $      ('       *** TOGGLE CONDUCTION ELEMENT WITH MOUSE ***$')
            CALL MOUSE(XMOUSE,YMOUSE,BUTTON,'TITLE$')
C           Find conduction element
            RMIN=1.0E6
            DO 100 IEL=1,NEL
                R=SQRT((XCEN(IEL)-XMOUSE)**2+(YCEN(IEL)-YMOUSE)**2)
                IF(R.LT.RMIN.AND. NUMAPT(IEL).EQ.ILEVEL) THEN
                   RMIN=R
                   IELC=IEL
                ENDIF
100         CONTINUE
            GOTO 2
         ELSE IF(CHOICE.EQ.'MAIN MENU'     )THEN
            GOTO 1
         ENDIF
      ELSE IF(CHOICE.EQ.'SELECT ELEMENT'.AND.NLEVEL.LE.1)THEN
C        2-dimensional way
         CALL PRSIS('*** SELECT ELEMENT FOR GROUP$',IGRP,
     $   ' WITH MOUSE ***$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON,'TITLE$')
C        Find conduction element
         RMIN=1.0E6
         DO 200 IEL=1,NEL
             R=SQRT((XCEN(IEL)-XMOUSE)**2+(YCEN(IEL)-YMOUSE)**2)
             IF(R.LT.RMIN) THEN
                RMIN=R
                IELC=IEL
             ENDIF
200      CONTINUE
      ELSE IF(CHOICE.EQ.'ZOOM')THEN
         CALL SETZOOM
         CALL REFRESH
         CALL DRMENU('NOCOVER')
         CALL DRGRID
         DO 300 IEL=1,NEL
            CALL DRAWEL(IEL)
  300    CONTINUE
      ENDIF
      IF(OLDCHOICE.EQ.'SELECT ELEMENT')THEN
         IGROUP(IELC)=IGRP
         CALL DRAWEL(IELC)
      ENDIF
      GOTO 1
C      IF(NLEVEL.EQ.1)GOTO 1
C      IF(NLEVEL.GT.1)GOTO 2
      END
