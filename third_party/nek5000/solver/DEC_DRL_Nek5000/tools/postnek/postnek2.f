c-----------------------------------------------------------------------
      subroutine interp(val,xex,yex,zex,ieo,tt,ierr)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      common /findri/ ixyzmin,ixyzemn,iexyzmn  ! location of closest point
      common /findrr/ dxyzmin
      logical ifstd
c
      REAL TT(1)
c
      REAL RRL(3),XXL(3)
      SAVE RRL   ,XXL
      DATA RRL   ,XXL /6*9.e10/
C
      INTEGER ICALLD,IE,NXYZ
      SAVE    ICALLD,IE,NXYZ
      DATA    ICALLD,IE,NXYZ /3*0/
c
      ifstd = .true.                      ! usual error handler, else
      if (ierr.eq.99) ifstd = .false.     ! return closest point
      dxyzmin = 1.e20
C
      IF (ICALLD.lt.2) THEN
         call locglob
         call coef
         nxyz=nx*ny*nz
         xxl(1)=1.0E14
         xxl(2)=1.0E14
         xxl(3)=1.0E14
         ie    = 0
      ENDIF
      ICALLD = ICALLD+1
C
C     Check to see if point is same as last point:
C
      if (xxl(1).eq.xex.and.xxl(2).eq.yex.and.xxl(3).eq.zex
     $   .and. 1.le.ie .and. ie.le.nel) then
         ieoff=nxyz*(ie-1)+1
         call evalsc( val , tt(ieoff) ,rrl , 0 )
         ierr = 0
         ieo  = ie
         return
      endif
C
      XXL(1) = XEX
      XXL(2) = YEX
      XXL(3) = ZEX
C
C     Check old element first
C
      IF (1.le.ie .and. ie.le.nel) then
         IF(   XBMIN(IE).LE.XXL(1) .AND. XXL(1).LE.XBMAX(IE)
     $   .AND. YBMIN(IE).LE.XXL(2) .AND. XXL(2).LE.YBMAX(IE)
     $   .AND. ZBMIN(IE).LE.XXL(3) .AND. XXL(3).LE.ZBMAX(IE) ) THEN
C           Potentially in this element, check to be sure.
            call findr(rrl(1),xxl(1),ie,1,ierr)
            IF (ABS(RRL(1)).LE.1.00001 .AND.
     $          ABS(RRL(2)).LE.1.00001 .AND.
     $          ABS(RRL(3)).LE.1.00001       ) then
                ieoff=nxyz*(ie-1)+1
                call evalsc( val , tt(ieoff) ,rrl , 0 )
c               write(6,*) 'ie ok1',ie,val,(rrl(k),k=1,2)
                ierr = 0
                ieo  = ie
                return
            endif
         endif
      endif
C
C
C
C     Not found yet, check all elements
C
      call findre(ie,rrl,xxl,rminmax,ierr)
c     write(6,*) ie,rminmax,ierr,' rminmax'
c
      if (rminmax.le.1.002) then
         ieoff=nxyz*(ie-1)+1
         call evalsc( val , tt(ieoff) ,rrl , 0 )
         ierr = 0
         ieo  = ie
c        write(6,*) 'ie ok2',ie,val,(rrl(k),k=1,2)
         return
      endif
c
c
c     Not found at all....
c
      if (.not.ifauto) then
         WRITE(S,2001) IELAST,xex,yex,zex
         CALL PRS(S)
         call diamd3   (xex,yex,zex,15)
 2001    FORMAT(' Point is not inside an element. (',I5,3F8.3,')$')
      endif
c
c     write(6,*) 'ie bad',ie
      if (ifstd) then
         IE     = 0
         VAL    = 0.
         IENTRP = IE
         IDUM   = IENTRP
         ierr = 1
         ieo  = ie
         return
      else                             ! use closest point
         IE     = iexyzmn
         VAL    = work(ixyzemn)
         IENTRP = IE
         IDUM   = IENTRP
         ierr = 0
         ieo  = ie
         return
      endif
C
      end
c-----------------------------------------------------------------------
      subroutine evalsc( x0 , scal , rrl , inew )
C
C     Evaluate a scalar, SCAL, at position RRL and return the result in X0.
C
      INCLUDE 'basics.inc'
      DIMENSION SCAL(1)
      DIMENSION RRL(3)
      COMMON  /ceval/ HR(NXM),HS(NXM),HT(NXM)
     $               ,HHH(NXM,NXM,NXM)
c
      common  /ceval2/ reval(3)
C
      REAL    ICALLD
      SAVE    ICALLD
      REAL    RLXOLD,RLYOLD,RLZOLD
      SAVE    RLXOLD,RLYOLD,RLZOLD
      DATA    ICALLD /0/
      DATA    RLXOLD,RLYOLD,RLZOLD/3*99.0/
C
      call copy(reval,rrl,3) ! Save most recent rrl, for diagnostics
      SUM = 0.0
C
      IF (RRL(1).NE.RLXOLD.OR.RRL(2).NE.RLYOLD.OR.RRL(3).NE.RLZOLD
     $   .OR.ICALLD.EQ.0 .or. inew.eq.1 ) THEN
         ICALLD = 1
         RLXOLD = RRL(1)
         RLYOLD = RRL(2)
         RLZOLD = RRL(3)
         CALL HLEGN(HR,RRL(1),ZPTS,NX)
         CALL HLEGN(HS,RRL(2),ZPTS,NY)
         IF (IF3D) THEN
            CALL HLEGN(HT,RRL(3),ZPTS,NZ)
            IJK=0
            DO 100 K=1,NZ
            DO 100 J=1,NY
            DO 100 I=1,NX
               IJK=IJK+1
               HHH(IJK,1,1)=HR(I)*HS(J)*HT(K)
c              write(6,*) sum,scal(ijk),ijk,' sum'
               SUM=SUM+SCAL(IJK)*HHH(IJK,1,1)
  100       CONTINUE
         ELSE
            IJK=0
            DO 200 J=1,NY
            DO 200 I=1,NX
               IJK=IJK+1
               HHH(IJK,1,1)=HR(I)*HS(J)
               SUM=SUM+SCAL(IJK)*HHH(IJK,1,1)
 200       CONTINUE
         ENDIF
      ELSE
C
C     Reevaluate a new scalar at an old point.
C
         nxyz=nx*ny*nz
         do 300 i=1,nxyz
            sum=sum+scal(i)*hhh(i,1,1)
  300    continue
      endif
      x0 = sum
      return
      end
c-----------------------------------------------------------------------
      subroutine findl(txy,level,tmin,tmax)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
C
      TMPXY = TXY
      IF (QUANTY.EQ.'T CONTOUR')
     $   TMPXY=CONTRS*AMOD(TXY,CNTRLV)
      DO 1 L=1,14
              IF(TMPXY .GE. TX(L) .AND. TMPXY .LE. TX(L+1)) THEN
                LEVEL = L
                TMAX = TX(L+1)
                TMIN = TX(L)
                return
              ENDIF
1     CONTINUE
      IF(TMPXY.LE.0.0)    LEVEL= 1
      IF(TMPXY.GE.1.0)    LEVEL= 14
      return
      END
c-----------------------------------------------------------------------
      subroutine histry
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      REAL SPEEDH(MAXS)
      CHARACTER KEY,STRING*5,ANS2*2,LABEL*40,CVEL*20
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C       Displays Histories
C
      IF(XYATTR.EQ.'HISTORY')THEN
         CALL PRS('                  *** HISTORIES ***$')
         IF(NHIS.EQ.0) THEN
              CALL PRS('No history data had been saved.$')
              CALL CLEAR
              CALL DRMESH
              CALL DRFRAM
              return
C             ??!! Courant??
         ELSE
C           Use Find historical point nearest to current point.
            IPH=LOCHIS(1,IHISPT)
            JPH=LOCHIS(2,IHISPT)
            KPH=LOCHIS(3,IHISPT)
            IEK=LOCHIS(4,IHISPT)
            IH =IHISPT
         ENDIF
         CALL PRS('COORDINATES OF CURRENT POINT:$')
         WRITE(s,'(3G14.3)')XP(IND(IPH,JPH,KPH,IEK))
     $                     ,YP(IND(IPH,JPH,KPH,IEK))
     $                     ,ZP(IND(IPH,JPH,KPH,IEK))
         CALL PRS(S//'$')
      ELSE
C        Integral Quantity
         CALL PRSI('Integral Quantity $',IINTEG)
         IH=NHIS+IINTEG
      ENDIF
C     Error Checking
      IF(XYATTR.EQ.'HISTORY')THEN
         IF( (QUANTY .EQ. 'TEMPERATURE' .AND. HCODE(5,IH).EQ.' ') .OR.
     $       (QUANTY .EQ. 'PRESSURE   ' .AND. HCODE(4,IH).NE.'P'))THEN
             CALL PRSIS('For History point #$',IH,' At Coordinates$')
             WRITE(s,'(3G14.3)')XP(IND(IPH,JPH,KPH,IEK))
     $                         ,YP(IND(IPH,JPH,KPH,IEK))
     $                         ,ZP(IND(IPH,JPH,KPH,IEK))
             CALL PRS(S//'$')
             CALL PRS('The Quantity '//QUANTY//' Was not saved.$')
             CALL CLEAR
             CALL DRMESH
             CALL DRFRAM
             return
         ENDIF
      ELSE
C        Integral Quantity
         IF( (QUANTY .EQ. 'TEMPERATURE'.AND.HCODE(5,IH).NE.'Q'))THEN
            CALL PRS('The Integral of FLUX was not saved for$')
            CALL PRS('the object$')
            CALL PRS(SOBJ(ISOBJ)//'$')
         ENDIF
         IF( (QUANTY .EQ. 'FORCE'.AND.HCODE(1,IH).NE.'F'))THEN
            CALL PRS('The Integral of FLUX was not saved for$')
            CALL PRS('the object$')
            CALL PRS(SOBJ(ISOBJ)//'$')
         ENDIF
         IF( QUANTY .NE. 'FORCE'.AND.QUANTY.NE.'TEMPERATURE')THEN
            CALL PRS('Integral Plot require Quantity to be set$')
            CALL PRS('To either FORCE or TEMPERATURE (for Flux)$')
         ENDIF
      ENDIF
      IF(QUANTY.EQ.'COORDINATE' )THEN
         IF( (COMPON.EQ.'X' .AND. HCODE(1,IH).NE.'X') .OR.
     $       (COMPON.EQ.'Y' .AND. HCODE(2,IH).NE.'Y') .OR.
     $       (COMPON.EQ.'Z' .AND. HCODE(3,IH).NE.'Z'))THEN
             CALL PRSIS('For History point #$',IH,' At Coordinates$')
             WRITE(s,'(3G14.3)')XP(IND(IPH,JPH,KPH,IEK))
     $                         ,YP(IND(IPH,JPH,KPH,IEK))
     $                         ,ZP(IND(IPH,JPH,KPH,IEK))
             CALL PRS(S//'$')
             CALL PRS('The Quantity '//QUANTY//' Was not saved.$')
             CALL PRS('Try SET HISTORY POINT menu.$')
             CALL CLEAR
             CALL DRMESH
             CALL DRFRAM
             return
         ENDIF
      ENDIF
      IF(QUANTY.EQ.'VELOCITY' )THEN
         IF( (COMPON.EQ.'X' .AND. HCODE(1,IH).NE.'U') .OR.
     $       (COMPON.EQ.'Y' .AND. HCODE(2,IH).NE.'V') .OR.
     $       (COMPON.EQ.'Z' .AND. HCODE(3,IH).NE.'W'))THEN
             CALL PRSIS('For History point #$',IH,' At Coordinates$')
             WRITE(s,'(3G14.3)')XP(IND(IPH,JPH,KPH,IEK))
     $                         ,YP(IND(IPH,JPH,KPH,IEK))
     $                         ,ZP(IND(IPH,JPH,KPH,IEK))
             CALL PRS(S//'$')
             CALL PRS('The Quantity '//QUANTY//' Was not saved.$')
             CALL PRS('Try SET HISTORY POINT menu.$')
             CALL CLEAR
             CALL DRMESH
             CALL DRFRAM
             return
         ENDIF
      ENDIF
      CALL CLEAR
      IF(QUANTY.EQ.'VELOCITY' )THEN
         IF(XYATTR.EQ.'HISTORY')THEN
            IF(COMPON.EQ.'X')CALL EZXY(Timeh,Uh(1,IH),NHISPT,1)
            IF(COMPON.EQ.'Y')CALL EZXY(Timeh,Vh(1,IH),NHISPT,2)
            IF(COMPON.EQ.'Z')CALL EZXY(Timeh,Wh(1,IH),NHISPT,3)
            IF(COMPON.eq.'S')THEN
               DO 121 I=1,NHISPT
                  SPEEDH(I)=SQRT(UH(I,IH)**2+VH(I,IH)**2+WH(I,IH)**2)
121            CONTINUE
               CALL EZXY(timeh,SPEEDH,NHISPT,15)
            ENDIF
         ENDIF
      ENDIF
      IF(QUANTY.EQ.'FORCE' .AND. XYATTR .EQ.'INTEGRAL QUANTITY')THEN
C           INtegrated Quantity
            IF(COMPON.EQ.'X')CALL EZXY(Timeh,Uh(1,IH),NHISPT,10)
            IF(COMPON.EQ.'Y')CALL EZXY(Timeh,Vh(1,IH),NHISPT,11)
            IF(COMPON.EQ.'Z')CALL EZXY(Timeh,Wh(1,IH),NHISPT,12)
            IF(COMPON.eq.'S')CALL PRS('Can''t print integral of Speed.'
     $      //'Choose X,Y, or Z component$')
      ENDIF
      IF(QUANTY.EQ.'COORDINATE' )THEN
         IF(XYATTR.EQ.'HISTORY')THEN
            IF(COMPON.EQ.'X')CALL EZXY(Timeh,Uh(1,IH),NHISPT,16)
            IF(COMPON.EQ.'Y')CALL EZXY(Timeh,Vh(1,IH),NHISPT,17)
            IF(COMPON.EQ.'Z')CALL EZXY(Timeh,Wh(1,IH),NHISPT,18)
            IF(COMPON.eq.'S')THEN
               CALL PRS('CAN''T PLOT SPEED OF MESH POINT$')
               CALL PRS('Choose X, Y (Z) Component$')
            ENDIF
         ELSE
C           INtegrated Quantity
               CALL PRS('CAN''T INTEGRATE COORDINATE$')
         ENDIF
      ENDIF
      IF(XYATTR.EQ.'HISTORY')THEN
         IF(QUANTY.EQ.'TEMPERATURE')CALL EZXY(Timeh,Th(1,IH),NHISPT,4)
         IF(QUANTY.EQ.'PRESSURE'   )CALL EZXY(Timeh,PH(1,IH),NHISPT,7)
         IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')
     $   CALL EZXY(Timeh,PSh(1,IH),NHISPT,9)
      ELSE
C?         IF(QUANTY.EQ.'PRESSURE'   )CALL EZXY(Timeh,PH(1,IH),NHISPT,7)
         IF(QUANTY.EQ.'TEMPERATURE')CALL EZXY(Timeh,Th(1,IH),NHISPT,13)
         IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')
     $   CALL EZXY(Timeh,PSh(1,IH),NHISPT,14)
      ENDIF
      IF(QUANTY.EQ.'COURANT'    )CALL EZXY(Timeh,COURNT    ,NHISPT,5)
      IF(QUANTY.EQ.'FLOWRATE'   )CALL EZXY(Timeh,FLOW      ,NHISPT,6)
C     PUT LABELS ON PLOT
      call hard
      SESION(11:11)='$'
      CALL GWRITE(XPHY(1.12),YPHY(0.87),1.3,SESION)
      ANS2='C '
      IF(ANS2.NE.'C '.AND.ANS2.NE.'F '.AND.ANS2.NE.'PG')THEN
C               Write X and Y Location
              WRITE(CVEL,'(G11.3,''$'')')XP(IND(IPH,JPH,KPH,IEK))
              CALL GWRITE(XPHY(0.1),YPHY(0.95),1.3,CVEL)
              CALL GWRITE(XPHY(0.05),YPHY(0.95),1.3,'X=$')
              WRITE(CVEL,'(G11.3,''$'')')YP(IND(IPH,JPH,KPH,IEK))
              CALL GWRITE(XPHY(0.45),YPHY(0.95),1.3,CVEL)
              CALL GWRITE(XPHY(0.4),YPHY(0.95),1.3,'Y=$')
              IF(NDIM.EQ.3)THEN
                 WRITE(CVEL,'(G11.3,''$'')')ZP(IND(IPH,JPH,KPH,IEK))
                 CALL GWRITE(XPHY(0.75),YPHY(0.95),1.3,CVEL)
                 CALL GWRITE(XPHY(0.7),YPHY(0.95),1.3,'Z=$')
              ENDIF
      ENDIF
      CALL NOHARD
C
      DO 90 I=1,4
90    CALL PRS(' $')
      ans=' '
95    CALL PRS(' <cr> to continue; A<cr> to annotate plot>$')
      CALL RES(ANS,1)
      IF(IFLEARN)WRITE(3,101)ANS
      IF(ANS.EQ.'A'.OR.ANS.EQ.'a') THEN
         CALL PRS('Input location of beginning of label.$')
         CALL MOUSE(XH,YH,BUTTON)
         CALL PRS('Type in label:$')
         CALL RES(LABEL,40)
         call hard
         CALL GWRITE(XH,YH,1.0,LABEL//'$')
         call nohard
         go to 95
      ELSE
         CALL PRS('End of labels$')
         call Heject
         CALL CLEAR
         CALL DRMESH
         CALL DRFRAM
         return
      ENDIF
100   FORMAT(4I7,'   I,J,IEL, and ICODE of Point for History')
101   FORMAT(A1,A5)
102   FORMAT(A2)
300   FORMAT(G11.3)
      END
c-----------------------------------------------------------------------
      subroutine setann
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*40 LABEL
C
      DO 90 I=1,10
90    CALL PRS(' $')
      ans=' '
      CALL PRS('Input (with mouse) location of beginning of label.$')
      CALL MOUSE(XH,YH,BUTTON)
      CALL PRS('Type in label:$')
      CALL RES(LABEL,40)
      call hard
      CALL GWRITE(XH,YH,1.0,LABEL//'$')
      call nohard
      return
      END
c-----------------------------------------------------------------------
      subroutine ezxy(x,y,n,icode)
      CHARACTER*10 TITLEX,TITLEY(18)
      character*12 CHARS
      REAL X(N),Y(N)
      DATA TITLEX,TITLEY/'Time$','X-Vel$','Y-Vel$','Z-Vel$','Temp$'
     $,'Courant$','Flow$','Pressure$','Pgradx$','Ps Scalar$'
     $,'X-Force$','Y-Force$','Z-Force$','Flux$','PS Flux$','Speed$'
     $,'X-Coord$','Y-Coord$','Z-Coord$'/
      XSC(XD)= 0.2 + (XD-XMIN)/(XMAX-XMIN) * .9
      YSC(YD)= 0.2 + (YD-YMIN)/(YMAX-YMIN) * .7
      CALL HARD
      IF(.FALSE.)THEN
         CALL OPENF(19,'bt.txt','NEW',1,ierr)
         CALL PRS('Enter title for file$')
         CALL RES(LINE,80)
         WRITE(19,'(a80)')LINE
c        NPTDUMP=nptpro+1
         NPTDUMP=n+1
         WRITE(19,*)N,'     Points follow. x,y or z, then data 2g16.9'
         WRITE(19,*)'0          0             0           0        '
     $   ,'   Above for consistency with other plot files'
      ENDIF
      XMIN= X(1)
      XMAX= X(1)
      YMIN= Y(1)
      YMAX= Y(1)
      DO 16 I=1,N
         XMIN=AMIN1(XMIN,X(I))
         XMAX=AMAX1(XMAX,X(I))
         YMIN=AMIN1(YMIN,Y(I))
         YMAX=AMAX1(YMAX,Y(I))
         IF(.FALSE.)WRITE(19,'(2G16.9)')X(I),Y(I)
   16 CONTINUE
      IF(YMIN.EQ.YMAX) THEN
              CALL GSWRIT(0.2,0.5,2.0,'CONSTANT FIELD$')
              CALL GSWRIT(0.2,0.3,2.0,TITLEY(ICODE))
              WRITE(CHARS,'(G11.4,''$'')')YMIN
              CALL GSWRIT(0.6,0.3,2.0,CHARS)
              return
      ENDIF
      IF(XMIN.EQ.XMAX) THEN
         CALL PRS('Data for 1 Time slice only; '//titley(icode)//'=$')
         CALL PRR(Y(N))
         return
      ENDIF
      CALL GSWRIT(0.6 ,0.14,1.0,TITLEX)
      CALL GSWRIT(0.06,0.55,1.0,TITLEY(ICODE))
C       Numeric Labels
      WRITE(CHARS,'(G11.4,''$'')')XMIN
      CALL GSWRIT(XSC(XMIN)-.04,YSC(YMIN)-.03,1.0,CHARS)
      WRITE(CHARS,'(G11.4,''$'')')XMAX
      CALL GSWRIT(XSC(XMAX)-.04,YSC(YMIN)-.03,1.0,CHARS)
      WRITE(CHARS,'(G11.4,''$'')')YMIN
      CALL GSWRIT(XSC(XMIN)-.15,YSC(YMIN),1.0,CHARS)
      WRITE(CHARS,'(G11.4,''$'')')YMAX
      CALL GSWRIT(XSC(XMIN)-.15,YSC(YMAX),1.0,CHARS)
200   FORMAT(G11.4)
C       Draw Axes
      CALL MOVESC(XSC(XMIN),YSC(YMIN))
      CALL DRAWSC(XSC(XMIN),YSC(YMAX))
      CALL MOVESC(XSC(XMIN),YSC(YMIN))
      CALL DRAWSC(XSC(XMAX),YSC(YMIN))
C
      CALL MOVESC(XSC(X(1)),YSC(Y(1)))
      DO 1 I=2,N
              CALL DRAWSC(XSC(X(I)),YSC(Y(I)))
1     CONTINUE
      CALL NOHARD
      return
      END
c-----------------------------------------------------------------------
      subroutine value
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      IF(QUANTY .EQ.'VORTICITY' )THEN
         CALL INTERP(VAL,XPOINT,YPOINT,ZPOINT,IDUM,WORK,ierr)
         IF(DERIV.NE.'NO')CALL PRS('sorry, cant give derivative$')
         IF(.NOT.IF3D)WRITE(s,'(1X,A10,2G14.3)')
     $   'AT X,Y,  =',XPOINT,YPOINT
         CALL PRS(s//'$')
         CALL PRSRS(QUANTY//' = $',VAL,'$')
      ELSE IF(QUANTY .EQ.'VELOCITY' )THEN
         CALL INTERP(VAL1,XPOINT,YPOINT,ZPOINT,IDUM,U,ierr)
         CALL INTERP(VAL2,XPOINT,YPOINT,ZPOINT,IDUM,V,ierr)
         IF(DERIV.NE.'NO')CALL PRS('sorry, cant give derivative$')
         IF(IF3D)THEN
            CALL INTERP(VAL3,XPOINT,YPOINT,ZPOINT,IDUM,W,ierr)
            WRITE(S,'(1X,A10,3G14.3)')'AT X,Y,Z =',XPOINT,YPOINT,ZPOINT
            CALL PRS(s//'$')
            WRITE(S,'(1X,A10,3G14.3)')'U,V,W    =',VAL1  ,VAL2  ,VAL3
            CALL PRS(s//'$')
         ELSE
            WRITE(S,'(1X,A8,2G14.3)')'AT X,Y =',XPOINT,YPOINT
            CALL PRS(s//'$')
            WRITE(S,'(1X,A8,2G14.3)')'U,V    =',VAL1  ,VAL2
            CALL PRS(s//'$')
         ENDIF
      ELSE
C        SCALAR
         CALL INTERP(VAL,XPOINT,YPOINT,ZPOINT,IDUM,work,ierr)
         IF(     IF3D)THEN
            WRITE(S,'(1X,A10,3G14.3)')'AT X,Y,Z =',XPOINT,YPOINT,ZPOINT
            CALL PRS(s//'$')
         ELSE IF(.NOT.IF3D)THEN
            WRITE(S,'(1X,A8,2G14.3)')'AT X,Y,  =',XPOINT,YPOINT
            CALL PRS(s//'$')
         ENDIF
         IF(DERIV.NE.'NO')CALL PRS(DERIV//' OF$')
         CALL PRSR(QUANTY//' = $',VAL)
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine values_std  ! Changed to dump only WORK() in routine at end
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      CHARACTER FILE*40
      CHARACTER*1 FILE1(40)
      EQUIVALENCE (FILE,FILE1)
      DATA FILE /'                                        '/
      REAL XXSCR(3)
c
      integer isel
      save    isel
      data    isel /-1/
c
      character*3 c3
C
      NDIM = 2
      IF (IF3D) NDIM = 3
C
      if (ifauto .and. isel .eq. 4) goto 7
         CALL PRS
     $('  Input: 0=File 1=Keyboard  2=Mouse 3=.fld file$')
         CALL REI(ISEL)
    7 continue
C
C     Read from file:
C
      write(6,*) 'ISEL = ',isel
      if (isel.eq.3) then
         write(6,*) 'call u_fld_values_ 2',isel
         call u_fld_values
         return
      endif
c
      IF (ISEL.EQ.0) THEN
         CALL PRS('  Input data file:$')
         CALL RES(FILE,40)
         if (file1(1).eq.'u') then
            write(6,*) 'Attempting to open unformatted file ',file
            open(unit=17,file=file ,form='unformatted',err=914)
            open(unit=89,file='v.dat',form='unformatted',err=914)
            goto 915
  914       continue
            write(6,*) 'Couldn''t open file ',file
            return
  915       continue
         else
            CALL OPENF(17,FILE,'OLD',1,ierr)
            open(unit=89,file='v.dat',form='formatted')
            IF (ierr.NE.0) THEN
               CALL PRS('Can''t find file'//FILE//'$')
               return
            ENDIF
         ENDIF
c
         CALL PRS(' Mapping from 1 file to another? (0=no,1=yes): $ ')
C
         CALL REI(IFMAT)
C
C        IF IFMAT is 1, then produce a new field file from an old one.
C
         IF (IFMAT.EQ.1) THEN
            READ(17,*,END=111,ERR=111) NELNEW,NXNEW
            NZNEW = 1
            IF (IF3D) NZNEW = NXNEW
            ISTEP=IDSTEP(IDUMP)
            TIME=DMPTIM(IDUMP)
            WRITE(89,'(4I4,1X,G13.4,I5,1X,10A2,1X,A20)')
     $         NELNEW,NXNEW,NXNEW,NZNEW,TIME,ISTEP
C    $         NELNEW,NXNEW,NXNEW,NZNEW,TIME,ISTEP,(EXCODE(I),I=1,10)
            READ(17,10006,END=111,ERR=111)
     $         (WORK(IELNEW),IELNEW=1,NELNEW)
            WRITE(89,10006)
     $         (WORK(IELNEW),IELNEW=1,NELNEW)
10006       FORMAT(6G11.4)
         ENDIF
         DO 110 IPT=1,999999
            IF (MOD(IPT,20).EQ.0) CALL PRSI('Interpolating point$',IPT)
            IF (IFMAT.EQ.1) THEN
               READ(17,10007,END=111,ERR=111) (XXSCR(I),I=1,NDIM)
10007          FORMAT(3G14.6)
            ELSEif (file1(1).eq.'u') then
               READ(17,END=111,ERR=111) (XXSCR(I),I=1,NDIM)
            ELSE
               READ(17,*,END=111,ERR=111) (XXSCR(I),I=1,NDIM)
            ENDIF
c
            if (ipt.eq.1) then 
c              call prs('Input scale size:$')
c              call rer(sc)
c              call prs('Input g3writ size:$')
c              call rei(igwsz)
               igwsz = 1
               ntot = nel*nx*ny*nz
               dx = glmax(xp,ntot)-glmin(xp,ntot)
               dy = glmax(yp,ntot)-glmin(yp,ntot)
               dz = glmax(zp,ntot)-glmin(zp,ntot)
               dr = (dx+dy+dz)/300
            endif
            if (ndim.eq.2) xxscr(3) = 0.
            write(c3,133) ipt
  133       format(i2,'$')
c           call draw_circ(xxscr(1),xxscr(2),xxscr(3),dr)
            call hard
            write(6,*) 'ifhard:',ifhard
            call g3writ   (xxscr(1),xxscr(2),xxscr(3),igwsz,c3)
            call diamd3   (xxscr(1),xxscr(2),xxscr(3),9)
c           ript = ipt
c           call draw_circu(xxscr(1),xxscr(2),xxscr(3),dr,ript,sc,ipt)
C
            IF (IFFLOW) THEN
               CALL INTERP(UVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,U,ierr)
               CALL INTERP(VVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,V,ierr)
               IF (IF3D)
     $         CALL INTERP(WVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,W,ierr)
               CALL INTERP(PVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,P,ierr)
            ENDIF
            IF (IFHEAT) THEN
               CALL INTERP(TVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,T,ierr)
            ENDIF
C
C       special kludge....
C
            IF (IFMAT.EQ.1) THEN
               ICNT=0
               IF (IFFLOW) THEN
                  ICNT = ICNT+1
                  TDUMP(ICNT) = UVAL
                  ICNT = ICNT+1
                  TDUMP(ICNT) = VVAL
                  IF (IF3D) THEN
                     ICNT = ICNT+1
                     TDUMP(ICNT) = WVAL
                  ENDIF
                  ICNT = ICNT+1
                  TDUMP(ICNT) = PVAL
               ENDIF
               IF (IFHEAT) THEN
                  ICNT = ICNT+1
                  TDUMP(ICNT) = TVAL
               ENDIF
               WRITE(89,112) (TDUMP(IICNT),IICNT=1,ICNT)
               GOTO 110
            ENDIF
C
            if (file1(1).eq.'u') then
               IF (IF3D) THEN
                  IF (IFHEAT.AND.IFFLOW) write(89)
     $            (XXSCR(I),I=1,NDIM),UVAL,VVAL,WVAL,PVAL,TVAL
                  IF (IFHEAT.AND..NOT.IFFLOW) write(89)
     $            (XXSCR(I),I=1,NDIM),TVAL
                  IF (.NOT.IFHEAT.AND.IFFLOW) write(89)
     $            (XXSCR(I),I=1,NDIM),UVAL,VVAL,WVAL,PVAL
               ELSE
                  IF (IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,PVAL,TVAL
                  IF (IFHEAT.AND..NOT.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),TVAL
                  IF (.NOT.IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,PVAL
               ENDIF
            ELSE
               IF (IF3D) THEN
                  IF (IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,WVAL,PVAL,TVAL
                  IF (IFHEAT.AND..NOT.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),TVAL
                  IF (.NOT.IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,WVAL,PVAL
               ELSE
                  IF (IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,PVAL,TVAL
                  IF (IFHEAT.AND..NOT.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),TVAL
                  IF (.NOT.IFHEAT.AND.IFFLOW) write(89,112)
     $             (XXSCR(I),I=1,NDIM),UVAL,VVAL,PVAL
               ENDIF
            ENDIF
c
         call find_vertexa(i1,j1,k1,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
         call find_vertex(i,j,k,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
            ijke = i+nx*(j-1 + ny*(k-1 + nz*(ie-1) ) )
            xq=xp(ijke)
            yq=yp(ijke)
            zq=zp(ijke)
            CALL PRSI('  Element number is:$',ie )
            if (if3d) then
              CALL PRSIII('  Nearest (mod NX) vertex number is:$',i,j,k)
              CALL prsrrr('  corresponding to xyz:$',xq,yq,zq)
            else
              CALL PRSII ('  Nearest (mod NX) vertex number is:$',i,j  )
              CALL prsrr ('  corresponding to xyz:$',xq,yq   )
            endif
            write(6,109) i,j,k,ie
  109       format(' UVWPT    H',2x,4i5)
c
  110    CONTINUE
  111    CONTINUE
         CLOSE (UNIT=17)
         CLOSE (UNIT=89)
C 112    FORMAT(8E12.4)
  112    FORMAT(9G14.6)
      ENDIF
      IF (ISEL.EQ.1.OR.ISEL.EQ.2) THEN
  210    CONTINUE
C
C     MOUSE INPUT
C
      IF (ISEL.EQ.2.AND..NOT.IF3D) THEN
         CALL PRS('  Click with mouse point of interest: (right=end)$')
         CALL MOUSE(XXSCR(1),XXSCR(2),BUTTON)
         CALL COLOR(1)
         CALL XDOT(XXSCR(1),XXSCR(2))
         IF (BUTTON.EQ.'RIGHT') GOTO 220
      ELSE
C
C     KEYBOARD INPUT
C
         CALL PRS('  Input x,y,(z): (-13,-13=end)$')
         NDIM = 2
         IF (IF3D) NDIM=3
         IF(NDIM.EQ.2)CALL RERR (XXSCR(1),XXSCR(2)         )
         IF(NDIM.EQ.3)CALL RERRR(XXSCR(1),XXSCR(2),XXSCR(3))
         IF(XXSCR(1).EQ.-13.0 .AND. XXSCR(2).EQ.-13.0)GO TO 220
      ENDIF
C
C
C
C
C        Quick kludge for different function values. 2 Feb 1989 14:01:27
C
C
C
C
         IF(QUANTY .EQ.'STREAMFUNCTION' .OR.
     $      QUANTY .EQ.'VORTICITY'      .OR.
     $      QUANTY .EQ.'DIVERGENCE'     .OR.
     $      DERIV  .NE.'NO'                  ) THEN
            XPOINT = XXSCR(1)
            YPOINT = XXSCR(2)
            ZPOINT = XXSCR(3)
            CALL VALUE
         ENDIF
c
         IF (.NOT.IF3D) XXSCR(3)=0.0
         IF (IFFLOW) THEN
            CALL INTERP(UVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,U,ierr)
            CALL INTERP(VVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,V,ierr)
            IF (IF3D)
     $      CALL INTERP(WVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,W,ierr)
            CALL INTERP(PVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,P,ierr)
         ENDIF
         IF (IFHEAT) THEN
            INEWTX=1
            CALL INTERP(TVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,T,ierr)
            INEWTX=0
         ELSE
            TVAL=0.0
         ENDIF
         WRITE(S,212) (XXSCR(I),I=1,NDIM)
  212    FORMAT(1X,'x,y:',3G12.4)
         CALL PRS(S//'$')
         IF (.NOT.IF3D) WRITE(S,'(4e12.4)') UVAL,VVAL,PVAL,TVAL
         IF (     IF3D) WRITE(S,'(5e12.4)') UVAL,VVAL,WVAL,PVAL,TVAL
         CALL PRS(S//'$')
c        CALL PRSI('  Element number is:$',IENTRP )
c
         call find_vertexa(i1,j1,k1,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
         call find_vertex (i,j,k,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
         ijke = i+nx*(j-1 + ny*(k-1 + nz*(ie-1) ) )
         xq=xp(ijke)
         yq=yp(ijke)
         zq=zp(ijke)
         if (if3d) then
            CALL PRSIII('  Nearest (mod NX) vertex number is:$',i,j,k )
            CALL prsrrr('  corresponding to xyz:$',xq,yq,zq)
         else
            CALL PRSII ('  Nearest (mod NX) vertex number is:$',i,j   )
            CALL prsrr ('  corresponding to xyz:$',xq,yq   )
         endif
         write(6,109) i,j,k,ie
         CALL PRSI('  Element number is:$',ie )
c
         GOTO 210
  220    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine nekprofil
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ANS2*2,LABEL*40,CVEL*20
C
C     Check if line has been input
      IF(XLINE(1).EQ.XLINE(2) .AND. YLINE(1).EQ.YLINE(2) .AND.
     $   ZLINE(1).EQ.ZLINE(2) ) THEN
         CALL PRS('Error: you must specify a line segment along$')
         CALL PRS('which to plot profile before plotting.$')
         return
      ENDIF
      CALL NOHARD
         IF(IF3D)THEN
            CALL MOVE3(XLINE(1),YLINE(1),ZLINE(1))
            CALL DRAW3(XLINE(2),YLINE(2),ZLINE(2))
         ELSE
            CALL MOVEC(XLINE(1),YLINE(1))
            CALL DRAWC(XLINE(2),YLINE(2))
         ENDIF
      CALL HARD
      IF(QUANTY.EQ.'VELOCITY' )THEN
         IF(COMPON.EQ.'X')CALL EZXYP(U   ,   PINGRL,1)
         IF(COMPON.EQ.'Y')CALL EZXYP(V   ,   PINGRL,2)
         IF(COMPON.EQ.'Z')CALL EZXYP(W   ,   PINGRL,3)
         IF(COMPON.EQ.'S')CALL EZXYP(WORK,   PINGRL,9)
         IF(COMPON.EQ.'N')THEN
C           Normal to specified line
                          CALL SETWRK(.false.)
                          CALL EZXYP(WORK,   PINGRL,10)
         ENDIF
      ENDIF
      IF(QUANTY.EQ.'TEMPERATURE' )    CALL EZXYP(T,   PINGRL,4)
      IF(QUANTY.EQ.'PRESSURE')        CALL EZXYP(P,   PINGRL,5)
      IF(QUANTY.EQ.'VORTICITY')       CALL EZXYP(WORK,PINGRL,8)
      IF(QUANTY.EQ.'DIVERGENCE')      CALL EZXYP(WORK,PINGRL,11)
      IF(QUANTY.EQ.'STREAMFUNCTION')  CALL EZXYP(WORK,PINGRL,12)
      IF(QUANTY.EQ.'USER FUNC')       CALL EZXYP(WORK,PINGRL,13)
      IF(QUANTY.EQ.'VORTEX')          CALL EZXYP(WORK,PINGRL,14)
      IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')THEN
         IF(LPSDMP.EQ.1)CALL EZXYP(PSA,PINGRL,7)
         IF(LPSDMP.EQ.2)CALL EZXYP(PSB,PINGRL,7)
         IF(LPSDMP.EQ.3)CALL EZXYP(PSC,PINGRL,7)
         IF(LPSDMP.EQ.4)CALL EZXYP(PSD,PINGRL,7)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine ezxyp(dum,pingrl,icode)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      LOGICAL IFTMP
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP0/ WORK1(LXYZ)
      CHARACTER*10 TITLEX(3),TITLEY(14)
      CHARACTER FILE*40
      character*12 chars
      character label*40,cvel*20,CSTRING*10
      REAL DUM(NX,NY,NZ,NEL)

      DATA TITLEX /'X$','Y$','Z$'/

      DATA TITLEY /'X-Vel$','Y-Vel$','Z-Vel$'
     $,'Temp$','Pres$','Tgradnor$','Ps SCALAR$','VORTICITY$'
     $,'Speed$','U-Normal$','Divergenc$','PSI$','User Func$','Vortex$'/
c
      XSC(XD) = 0.2 + XD * .9
      PSCC(PD) = 0.2 + (PD-PMIN)/(PMAX-PMIN) * .7
c     Avoid array overflow
      NPTPRO=MIN(NPTPRO,mxpro)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine drbox(iel,k)
C     Draws box around element IEL at plane K
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
      CALL COLOR(1)
      DO 20 IC=1,5
         IF(IC.EQ.1.OR.IC.EQ.5)THEN
            I=1
            J=1
         ELSE IF(IC.EQ.2)THEN
            I=NX
            J=1
         ELSE IF(IC.EQ.3)THEN
            I=NX
            J=NY
         ELSE IF(IC.EQ.4)THEN
            I=1
            J=NY
         ENDIF
         IF(IC.EQ.1)THEN
            CALL MOVE3
     $      (XP(IND(I,J,K,IEL)),YP(IND(I,J,K,IEL)),ZP(IND(I,J,K,IEL)))
         ELSE
            CALL DRAW3
     $      (XP(IND(I,J,K,IEL)),YP(IND(I,J,K,IEL)),ZP(IND(I,J,K,IEL)))
         ENDIF
20    CONTINUE
      return
      END
C**************************************************************end of postnek
c-----------------------------------------------------------------------
      subroutine stream
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      common /ws_comm/ wscale,wmin
c
      common /s_mirror/ imirror
c
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      LOGICAL IFSTRIO,IFCIRC,IFRST,iflast,IFTMP,IFOUTS,IFMOVI
      CHARACTER*1 YESNO
      CHARACTER*5 TYPPT(3)
      SAVE        TYPPT
      INTEGER ICALLD,IDUMPO
      SAVE    ICALLD,IDUMPO,ISTR,NSTR
      DATA    ICALLD,IDUMPO,ISTR,NSTR/0,99,0,0/
      DATA    TYPPT/'0 0 0','0 0 1','0 1 0'/
C
      IFTMP = IFGRID
      IFGRID= .FALSE.
      IFOUTS= .FALSE.
      IFMOVI= .FALSE.
      ISCND = 1
c     IFSCND = .TRUE.
      IFSCND = .FALSE.
      IF (IFAXIS) IFSCND = .FALSE.
C
      BUTTON  = 'LEFT'
      IFSTRIO = .FALSE.
      IFCIRC  = .FALSE.
      iflast  = .FALSE.
      STRTOL  = .001
      STPTOL  = .81
      ISTPT   = 0
      nxyz=nx*ny*nz
      wscale=1.0
      IF (TTMAX.NE.TTMIN) wscale = 1.0/(TTMAX-TTMIN)
      wmin = ttmin
      IF (IFUSCL.AND.USRMIN.NE.USRMAX) then
         wscale=1.0/(USRMAX-USRMIN)
         wmin = usrmin
      ENDIF
C
C     Set up Lagrangian tracers
C
      IF (ICALLD.EQ.0)                       CALL locglob
      IF (ICALLD.EQ.0)                       CALL COEF
      IF (ICALLD.EQ.0)                       CALL SETWRK(.true.)
C moved to below, after ISEL
c     IF (IDUMP.NE.IDUMPO.OR.ILGRNG.NE.1)    CALL SETLGR
      ICALLD=1
      IDUMPO=IDUMP
C
C     1st MENU:  Request streamline integration option:
C
   10 CONTINUE
      IF (ISTR.EQ.0) THEN
         MSEL=2
         CALL PRS
     $(' 0 - return, 1 - Read file, 2 - New Line, 3 - Set Parameters$')
         CALL PRS('(negative for reflect about y-plane)$')
      ELSE
c        MSEL=4
c        CALL PRS(' 0 - return, 1 - Read file,'//
c    $   ' 2 - New Line, 3 - Set Parameters, 4 - Replot$')
         MSEL=3
         CALL PRS(' 0 - return, 1 - Read file,'//
     $   ' 2 - New Line, 3 - Set Parameters$')
         CALL PRS('(negative for reflect about y-plane)$')
      ENDIF
      CALL PRS(' Enter selection: $')
      CALL REI(ISEL)
      IF (IFLEARN) WRITE(3,*) ISEL
      IF (ISEL.LT.-msel.OR.ISEL.GT.MSEL) GOTO 200
      GOTO 210
  200 CONTINUE
      CALL PRSIS(' Selection must be an integer between -4 and$',
     $ MSEL,'.$')
      GOTO 10
  210 CONTINUE
      isel_sign = isel
      isel = abs(isel)
c
      if (isel.eq.0) return
c
      imirror = 0
      if (isel_sign.lt.0) imirror = 1
c
C
C     If we're not replotting, then we may have to regenerate some stuff
C
      IF (ISEL.NE.4.AND.(IDUMP.NE.IDUMPO.OR.ILGRNG.NE.1)) CALL SETLGR
C
C     Read data from file SESION.STR
C
      IF (ISEL.EQ.1) THEN
         CALL OPENF(18,STRFLE,'OLD',1,ierr)
         NSTR=0
         ISTR=0
         IFILE=0
         NPRT1=NPRT+1
         DO 1000 ILINE=1,1000
            ISCND = 1
            iflast= .FALSE.
            IS=ISTR+1
            IS=MOD(IS,NPRT1)
            IS=MAX(1,IS)
            IF (IF3D) THEN
               READ(18,*,END=1001,ERR=1002)
     $         (XLG(J,IS),J=1,3),DPT(IS),NSTPPS,IOSTP(IS),ICLR
c              WRITE(6,*) 'xyz:',(XLG(J,IS),j=1,3),is
            ELSE
               READ(18,*,END=1001,ERR=1002)
     $         (XLG(J,IS),J=1,2),DPT(IS),NSTPPS,IOSTP(IS),ICLR
            ENDIF
            IFILE=1

            INTDIR=1
            IF (DPT(ISTR).lt.0.or.NSTPPS.LT.0) INTDIR=-INTDIR

            if (iclr.eq.-8.and.iline.eq.1) open(unit=88,file='str.out')
            if (iclr.eq.-8) write(88,*)
            if (iclr.eq.-8) ifile = 2

            ISTR=IS
            NSTR=NSTR+1
            NSTR=MIN(NPRT,NSTR)
            ICST(NSTR) = ICLR
            ISTP0=ISTPT+1
            ilost(ISTR)=ISTR
            call findre(ielgr(istr),rr(1,istr) ,xlg(1,istr)
     $                                         ,rminmax,ierr)
            write(6,*) 'reset ilost 1:'
     $                ,ilost(istr),istr,ielgr(istr),nstr,rminmax
C
C           Variable time stepping and streamlines
C
            IF (DPT(ISTR).EQ.0.)                INTDIR= IOSTP(ISTR)
            IF (DPT(ISTR).EQ.0.AND.NSTPPS.LT.0) INTDIR=-INTDIR
C
C           Move cursor to starting point
C
            XXIS=XISOm(XLG(1,ISTR),XLG(2,ISTR),XLG(3,ISTR))
            YYIS=YISOm(XLG(1,ISTR),XLG(2,ISTR),XLG(3,ISTR))
            CALL MOVEC(XXIS,YYIS)
c
            if (ndim.eq.2) then
               call streamline_draw_2d_close
     $         (xlg(1,istr),iostp(istr),dpt(istr)
     $         ,intdir,nstpps,iclr,ndim,iflast)
            else
               call streamline_draw
     $         (xlg(1,istr),iostp(istr),dpt(istr)
     $         ,intdir,nstpps,iclr,ndim)
            endif
            xlast = xlg(1,istr)
            ylast = xlg(2,istr)
            zlast = xlg(3,istr)
c
 1000   CONTINUE
        ISCND = 1

 1001   CONTINUE
        IF (IFILE.EQ.0) THEN
           CALL PRS('** ERROR **  Empty file for streamline data.$')
        ENDIF
        CLOSE(UNIT=18)
        if (ifile.eq.2) close(88)
        GOTO 10
 1002   CONTINUE
        CALL PRS('** ERROR **  Error reading for streamline data.$')
        CLOSE(UNIT=18)
        GOTO 10
      ENDIF
C
C     2nd menu option:  input from terminal
C
      IF (ISEL.EQ.2) THEN
         NPRT1=NPRT+1
         DO 2000 ILINE=1,1000
            ISCND = 1
            iflast    = .FALSE.
            IS=ISTR+1
            IS=MOD(IS,NPRT1)
            IS=MAX(1,IS)
            IF (IF3D.AND..NOT.IFPLAN) THEN
               CALL PRS('Input X,Y,Z,DT,NSTEPS (0=quit),acc,ICOLOR: $')
               CALL PRS('(e.g.  0 0 0  0  400 12  9)$')
               CALL RES(S,80)
               REWIND(13)
               WRITE (13,'(A80)')S
               REWIND(13)
               READ (13,*,END=1020,ERR=1850)
     $         (XLG(J,IS),J=1,3),DPT(IS),NSTPPS,IOSTP(IS),ICLR
               REWIND(13)
               IF (IFLEARN) WRITE(3,*)
     $         (XLG(J,IS),J=1,3),DPT(IS),NSTPPS,IOSTP(IS),ICLR
c              IF (IOSTP(IS).LE.0) IOSTP(IS)=1
               iflast = .FALSE.
               GOTO 1050
 1020          CONTINUE
                  IS1=IS-1
                  IF (IS1.EQ.0) IS1=NPRT
                  DPT(IS)=DPT(IS1)
                  IOSTP(IS)=IOSTP(IS1)
                  iflast = .TRUE.
 1050          CONTINUE
            ELSE
C              2-D
               IF (ILINE.GT.1.AND.IFGRAF) THEN
                  CALL PRS('  Input X,Y with mouse '//
     $               '(outside domain to exit, right to continue)$')
                  IF (IF3D) THEN
                     CALL MOUSE(XXIS,YYIS,BUTTON)
                     CALL PROJECT(XLG(1,IS),XXIS,YYIS,2)
                     IF (BUTTON.EQ.'LEFT')
     $                  WRITE(s,1991) XLG(1,IS),XLG(2,IS),XLG(3,IS)
 1991                   FORMAT(2X,'XYZ:',3G11.3)
                        CALL PRS(S//'$')
                  ELSE
                     CALL MOUSE(XLG(1,IS),XLG(2,IS),BUTTON)
                     IF (BUTTON.EQ.'LEFT')
     $                  WRITE(S,1990) XLG(1,IS),XLG(2,IS)
 1990                   FORMAT(2X,'X,Y:',2G11.3)
                        CALL PRS(S//'$')
                  ENDIF
C
C                 Check to see if button clicked within domain.
C                 If not, prompt user for keyboard input.
C
                  DTZERO = 0.0
                  ilost(IS)=IS
                  write(6,*) 'reset ilost 3:',ilost(is),is,ielgr(is)
                  CALL LAGRNGE(DTZERO,IS,INTDIR,DTPRIM,DCIRC,0)
                  IF(IELGR(IS).EQ.0.AND.BUTTON.NE.'RIGHT') BUTTON='LOST'
               ENDIF
C
               IF (ILINE.EQ.1.OR.BUTTON.EQ.'LOST'.OR..NOT.IFGRAF) THEN
                  IF (IF3D) THEN
                     CALL PRS(
     $               'Input X,Y,Z,DT,NSTEPS (0=quit),acc,ICOLOR: $')
                     CALL PRS( '(e.g.  0 0 0  0  400 12  9)$')
                     CALL RES(S,80)
                     REWIND(13)
                     WRITE (13,'(A80)')S
                     REWIND(13)
                     READ  (13,*,END=10,ERR=1850)
     $               (XLG(J,IS),J=1,3),DPT(IS),NSTPPS,IOSTP(IS),ICLR
                     REWIND(13)
                     IF (IFLEARN) WRITE(3,*)
     $                  (XLG(J,IS),J=1,3),DPT(IS),NSTPPS,IOSTP(IS),ICLR
                     IF (IFPROJ)
     $                  CALL PROJECT(XLG(1,IS),XLG(1,IS),XLG(1,IS),3)
                  ELSE
                     CALL PRS(
     $               'Input X,Y,DT,NSTEPS (0=quit),IO,ICOLOR: $')
                     CALL PRS( '(e.g.  0 0  0  400  2  9)$')
                     CALL RES(S,80)
                     REWIND(13)
                     WRITE (13,'(A80)')S
                     REWIND(13)
                     READ(13,*,END=10,ERR=1850)
     $               (XLG(J,IS),J=1,2),DPT(IS),NSTPPS,IOSTP(IS),ICLR
                     REWIND(13)
                     IF (IFLEARN) WRITE(3,*)
     $               (XLG(J,IS),J=1,2),DPT(IS),NSTPPS,IOSTP(IS),ICLR
                  ENDIF
C
c                 IF (IOSTP(IS).LE.0) IOSTP(IS)=1
                  iflast = .FALSE.
               ELSE
C                 Using mouse
                  IS1=IS-1
                  IF (IS1.EQ.0) IS1=NPRT
                  DPT(IS)=DPT(IS1)
                  IOSTP(IS)=IOSTP(IS1)
                  IF (BUTTON.EQ.'RIGHT') iflast=.TRUE.
               ENDIF
            ENDIF
            IF (NSTPPS.EQ.0) GOTO 10
            INTDIR=1
            ISTR=IS
            NSTR=NSTR+1
            NSTR=MIN(NPRT,NSTR)
            ICST(NSTR) = ICLR
            ISTP0=ISTPT+1
            ilost(ISTR)=ISTR
            write(6,*) 'reset ilost 4:',ilost(istr),istr,ielgr(istr)
C
C           Interpret input results according to the following options:
C
            NSTP1=IABS(NSTPPS)
            IF (DPT(ISTR).EQ.0.)                INTDIR= IOSTP(ISTR)
            IF (DPT(ISTR).EQ.0.AND.NSTPPS.LT.0) INTDIR=-INTDIR
C
C           Open file for diagnostics if requested.
C
            IF (IFSTRIO) OPEN(UNIT=59,FILE='STR.LOG'
     $          ,STATUS='UNKNOWN',ACCESS='SEQUENTIAL')
C
C           Move cursor to starting point
C
            IF (DPT(ISTR).EQ.0.AND..NOT.iflast) THEN
               XXIS=XISOm(XLG(1,ISTR),XLG(2,ISTR),XLG(3,ISTR))
               YYIS=YISOm(XLG(1,ISTR),XLG(2,ISTR),XLG(3,ISTR))
               CALL MOVEC(XXIS,YYIS)
C              Circulation integration stuff....
               XX00 = XXIS
               YY00 = YYIS
               X00 = XLG(1,ISTR)
               Y00 = XLG(2,ISTR)
               Z00 = XLG(3,ISTR)
               RSTRMX = 0.0
               RSTR01 = 0.0
               IFRST  = .TRUE.
               CIRC   = 0.0
               TOTDT  = 0.0
               NSTCNT = 0
            ENDIF
            IF (iflast) THEN
               XLG(1,IS) = XLAST
               XLG(2,IS) = YLAST
               XLG(3,IS) = ZLAST
               IELGR(IS) = 0
               ilost(IS) = IS
            write(6,*) 'reset ilost 5:',ilost(is),is,ielgr(is)
C               IELGR(IS) = IELAS
            ENDIF
C
C           Start integration of streamline
C
            IF (IFSTRIO) WRITE(59,*) 'ISTR:',ISTR
c
            if (ndim.eq.2) then
               call streamline_draw_2d_close
     $         (xlg(1,istr),iostp(istr),dpt(istr)
     $         ,intdir,nstpps,iclr,ndim,iflast)
            else
               call streamline_draw
     $         (xlg(1,istr),iostp(istr),dpt(istr)
     $         ,intdir,nstpps,iclr,ndim)
            endif
            xlast = xlg(1,istr)
            ylast = xlg(2,istr)
            zlast = xlg(3,istr)
c
 1800       CONTINUE
 1801 FORMAT(1X,I4,1p9E12.4)
 1803 FORMAT(3i5,1p9E12.4)
            GOTO 2000
 1850       CONTINUE
            CALL PRS('** ERROR **  Error reading input, try again.$')
 2000    CONTINUE
            ISCND = 1
         IF (IFSTRIO) CLOSE(UNIT=59)
         GOTO 10
      ENDIF
C
C     Set streamline plotting parameters
C
      IF (ISEL.EQ.3) THEN
C
         CALL PRS(' Output streamlines to a file?$')
         CALL RES(YESNO,1)
         IFOUTS=.FALSE.
         IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFOUTS=.TRUE.
C
         CALL PRS(' Animate precomputed data?$')
         CALL RES(YESNO,1)
         IFMOVI=.FALSE.
         IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFMOVI=.TRUE.
C
1212     CALL PRS(' Input tolerance (%) for closing streamlines,$')
         CALL PRS(' ISTRIO (1=out,0=noout), and ICIRC:$')
         CALL PRS(' (Typical values ".85 .03 0 0").$')
C
         CALL RES(S,80)
         REWIND(13)
         WRITE (13,'(A80)')S
         REWIND(13)
         READ  (13,*,ERR=1212,END=1212)STPTOL,STRTOL,ISTRIO,ICIRC
         REWIND(13)
         IF (IFLEARN) WRITE(3,*) STPTOL,STRTOL,ISTRIO,ICIRC
         IF (ISTRIO.EQ.1) IFSTRIO=.TRUE.
         IF (ISTRIO.NE.1) IFSTRIO=.FALSE.
         IF (ICIRC.EQ.1) IFCIRC=.TRUE.
         IF (ICIRC.NE.1) IFCIRC=.FALSE.
         STPTOL = STPTOL**2
         STRTOL = STRTOL**2
         IF (IF3D) THEN
C
C           Query if mouse plane is to be established.
C
            CALL PRS(' Establish a plane for mouse input? (y/n)$')
            CALL RES(YESNO,1)
C            READ(5,2211,ERR=10)    YESNO
            IFPLAN=.FALSE.
            IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') THEN
               DO 2150 IV=1,3
                  WRITE(S,2202) IV,TYPPT(IV)
 2202             FORMAT(1X,'Input X,Y,Z(',i1,'):  (Typ. ',A5,')')
                  CALL PRS(S//'$')
                  CALL RES(S,80)
                  REWIND(13)
                  WRITE (13,'(A80)')S
                  REWIND(13)
                  READ  (13,*,ERR=10) (XV(J,IV),J=1,3)
                  REWIND(13)
 2150          CONTINUE
               IFPLAN=.TRUE.
               DO 2151 J=1,3
                  XV(J,2) = XV(J,2) - XV(J,1)
                  XV(J,3) = XV(J,3) - XV(J,1)
 2151          CONTINUE
               CALL CROSS(VN,XV(1,2),XV(1,3))
               CALL NORM3D(VN)
C
C              Set up input interpretation...
C
               CALL PROJECT(XV(1,1),XV(1,2),XV(1,3),1)
C
C              Query if streamlines are to be projected onto plane
C
               CALL PRS('Project streamlines onto plane? (y/n)$')
               CALL RES(YESNO,1)
               IFPROJ=.FALSE.
               IF (YESNO.EQ.'Y'.OR.YESNO.EQ.'y') IFPROJ=.TRUE.
            ENDIF
         ENDIF
         GOTO 10
      ENDIF
C
C     Replot existing data
C
      IF (ISEL.EQ.4) THEN
c        write(6,*) 'isel=3:',nstr,nstr0(1),nstrp(1)
         DO 3001 IS=1,NSTR
            IF (NSTR0(IS).GT.0.AND.NSTRP(IS).GT.0) THEN
            DO 3000 IP=NSTR0(IS),NSTRP(IS)
               XXIS=XISOm(XSTR(1,IP),XSTR(2,IP),XSTR(3,IP))
               YYIS=YISOm(XSTR(1,IP),XSTR(2,IP),XSTR(3,IP))
               ICSTR = XSTR(4,IP)
               CALL COLOR(ICSTR)
               IF (DPT(IS).EQ.0) THEN
                  IF (IP.EQ.NSTR0(IS)) THEN
                     CALL MOVEC(XXIS,YYIS)
                  ELSE
                     CALL DRAWC(XXIS,YYIS)
                  ENDIF
               ELSE
                     CALL DRAWC(XXIS,YYIS)
c                 CALL DIAMDA(XXIS,YYIS)
               ENDIF
 3000       CONTINUE
            ENDIF
 3001    CONTINUE
      GOTO 10
      ENDIF
      IFGRID=IFTMP
      return
      END
c-----------------------------------------------------------------------
      subroutine lagrnge(dtl,il,intdir,dtprim,dcirc,KS)
C
C     Compute Lagrangian movement of particles for the current time
C     step.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      LOGICAL IFCOND
C
      ie = ielgr(il)
      CALL UPDATEL(XLG(1,IL),RR(1,IL),DTL,IL,IE,INTDIR,DTPRIM,DCIRC,ks)
      call findre(ie,rr(1,il),xlg(1,il),rminmax,ierr)
c
      IF ( rminmax.lt.1.002 ) then
c        found an acceptable fit
         ielgr(il)=ie
         ilost(il)=0
         return
      endif
c     write(6,*) 'rminmax:'
c    $           ,( rr(k,il),k=1,3)
c    $           ,(xlg(k,il),k=1,3)
c    $           ,ie,il
C
C     Zero out remaining lost list to ensure that lost particles go away.
C
      ielgr(il)=0
      ilost(IL)=0
C
      return
      END
c-----------------------------------------------------------------------
      subroutine setlgr
C
C     Set up some basic arrays for the tracking of Lagrangian particles
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
C
      ILGRNG=1
      INEWTX=0
      IFOUTL=.TRUE.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine updatel(xxl,rrl,dt0,il,ie,intdir,dtprim,dcirc,KS)
C
C     Update XXL and RRL according to new time step.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      REAL  DTL,DTOLD,UOLD,VOLD,WOLD,DTLMIN,DTLMAX,DXCHAR,Fintinv
      SAVE  DTL,DTOLD,UOLD,VOLD,WOLD,DTLMIN,DTLMAX,DXCHAR,Fintinv
      INTEGER ICALLD,IEOLD
      SAVE  ICALLD,IEOLD
      DATA  ICALLD,IEOLD /0,0/
      SAVE  nstp1,nstp2
      DATA  nstp1,nstp2 /0,0/
      DATA  DTLMIN /1000.0/
      DATA  DTLMAX /-1000.0/
c
C
C     Arguments
C
      DIMENSION RRL(3),XXL(3)
C
      nxyz=nx*ny*nz
      ieoff=nxyz*(ie-1)+1
      if (ie.ne.ieold) dzchar = (zp(ieoff+1)-zp(ieoff))**2
      if (.not.if3d)   dzchar = 0.
      if (ie.ne.ieold) dxchar = (xp(ieoff+1)-xp(ieoff))**2 +
     $                          (yp(ieoff+1)-yp(ieoff))**2 +
     $                          dzchar
      if (ie.ne.ieold) then
         f1 = 0.
         f2 = 0.
         nstps = nstp1+nstp2
         if (nstps.ne.0) then
            f1 = nstp1
            f1 = f1/nstps
            f2 = nstp2
            f2 = f2/nstps
         endif
         write(6,*) '1st/2nd ord:',f1,f2,nstps,ieold
         nstp1 = 0
         nstp2 = 0
      endif
      ieold = ie
C
C    Compute XXL
C
      CALL EVALSC( U0 , WKV1(IEOFF) ,RRL , 1 )
      CALL EVALSC( V0 , WKV2(IEOFF) ,RRL , 0 )
      IF (IF3D) THEN
         CALL EVALSC( W0 , WKV3(IEOFF) ,RRL , 0 )
      ELSE
         W0=0.0
      ENDIF
C
C     Choose DT according to local velocity if DT0 is unspecified.
C
C     Set up AB2 coefficients
C
      IF (ISCND.Le.2) THEN
         ALPH2  = 0.0
         UOLD   = 0.00
         VOLD   = 0.00
         WOLD   = 0.00
         DTOLD  = 0.00
         DTOLDi = 0.00
         Fintinv = 1.0 / FLOAT(INTDIR)
      ELSE
         ALPH2 = 0.5
      ENDIF
C
C     Subtract off normal to projection plane:
C
      IF (IFPROJ) THEN
         CALL PROJECT(U0,V0,W0,4)
      ENDIF
C
C
c     write(6,*) 'this is ie!!:',ie,nstp1,nstp2,dt0
      IF (DT0.EQ.0) THEN
         umag   = u0**2 + v0**2 + w0**2
         IF (umag.NE.0) then
             DT1 = 2.0*SQRT( DXCHAR/umag )
         ELSE
             DT1 = 2.0*SQRT( DXCHAR )
         ENDIF
         DTL   = DT1*Fintinv
C
C        2nd Order A-B
C
         DX = DTL*U0
         DY = DTL*V0
         DZ = DTL*W0
         CX = DTL*DTL*ALPH2*(U0-UOLD)*DTOLDi
         CY = DTL*DTL*ALPH2*(V0-VOLD)*DTOLDi
         CZ = DTL*DTL*ALPH2*(W0-WOLD)*DTOLDi
         ad = abs(dx)+abs(dy)+abs(dz)
         ac = abs(cx)+abs(cy)+abs(cz)
         if (ac.lt.0.2*ad) then
            dx = dx+cx
            dy = dy+cy
            dz = dz+cz
            nstp2 = nstp2+1
         else
            nstp1 = nstp1+1
         endif
      ELSE 
         DX = DT0*(U0 + ALPH2*(U0-UOLD))
         DY = DT0*(V0 + ALPH2*(V0-VOLD))
         DZ = DT0*(W0 + ALPH2*(W0-WOLD))
      ENDIF 
c     if (ie.eq.21) write(6,44) nstp1,nstp2
c    $              ,u0,v0,w0,rrl(1),rrl(2),rrl(3),dx,dy,dz
c    $              ,xxl(1),xxl(2),xxl(3)
c  44  format('st:',2i5,4(1x,1p3e13.6))
C
      UOLD   = U0
      VOLD   = V0
      WOLD   = W0
      DTOLD  = DTL
      if (dtl.ne.0.0) DTOLDi = 1.0/DTL
C
C
      XXL(1) = XXL(1) + DX
      XXL(2) = XXL(2) + DY
      XXL(3) = XXL(3) + DZ
c     SCLOCK=SCLOCK+DTL
c     DCIRC  = (DX**2+DY**2+DZ**2)/ABS(DTL)
      IF (IFPROJ) CALL PROJECT(XXL,XXL,XXL,3)
C
C
C     Use Newton-Raphson iteration to ensure that RRL doesn't stray from XXL
C
c     call findr(RRL,XXL,IE,1,ierr)
C
      DTPRIM = DTL
      return
      END
c-----------------------------------------------------------------------
      subroutine diamda(xx,yy)
      INCLUDE 'basics.inc'
c     draws a diamond around XX,YY
      CALL MOVEC(XX+.0005*XFAC ,YY+.0005*YFAC)
      CALL DRAWC(XX-.0005*XFAC ,YY+.0005*YFAC)
      CALL DRAWC(XX-.0005*XFAC ,YY-.0005*YFAC)
      CALL DRAWC(XX+.0005*XFAC ,YY-.0005*YFAC)
      CALL DRAWC(XX+.0005*XFAC ,YY+.0005*YFAC)
      return
      END
c-----------------------------------------------------------------------
      subroutine intcrc(totdt,nstcnt,x00,y00,z00,stptol,strtol)
C
C     Compute the integral of U*ds around a (supposedly) closed curve
C     starting at (x0,y0)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DATA XLMAX,YLMAX,ZLMAX,XLMIN,YLMIN,ZLMIN,ULAST,ULMAX,
     $VLAST,VLMAX,WLAST,WLMAX,ULMIN,VLMIN,WLMIN,SLMAX,SLMIN /17*0.0/
      SAVE XLMAX,YLMAX,ZLMAX,XLMIN,YLMIN,ZLMIN,ULAST,ULMAX,
     $VLAST,VLMAX,WLAST,WLMAX,ULMIN,VLMIN,WLMIN,SLMAX,SLMIN
C
      NXYZ = NX*NY*NZ
C
C     Use number of profile points to determine resolution of integration
C
      NPTCRC = NPTPRO
      NPTCRC = MAX(NPTCRC,NSTCNT)
      DT2 = 2.0 * TOTDT/FLOAT(NPTCRC)
      NSTP4 = 4*NPTPRO
C
      DO 2000 IRICH = 1,2
         DT2 = DT2/2.0
         INTDIR=1
         ilost(1)=1
         IELGR(1)=0
         XLG(1,1) = X00
         XLG(2,1) = Y00
         XLG(3,1) = Z00
         DT0 = 0.000001
         CALL LAGRNGE(DT0,1,INTDIR,DTPRIM,DCIRC,0)
C
         XLG(1,1) = X00
         XLG(2,1) = Y00
         XLG(3,1) = Z00
C
         XNEW = XLG(1,1)
         YNEW = XLG(2,1)
         ZNEW = XLG(3,1)
C
         XLMAX = AMAX1(XLMAX,XNEW)
         YLMAX = AMAX1(YLMAX,YNEW)
         ZLMAX = AMAX1(ZLMAX,ZNEW)
         XLMIN = AMIN1(XLMIN,XNEW)
         YLMIN = AMIN1(YLMIN,YNEW)
         ZLMIN = AMIN1(ZLMIN,ZNEW)
C
         ULMAX = AMAX1(ULMAX,ULAST)
         VLMAX = AMAX1(VLMAX,VLAST)
         WLMAX = AMAX1(WLMAX,WLAST)
         ULMIN = AMIN1(ULMIN,ULAST)
         VLMIN = AMIN1(VLMIN,VLAST)
         WLMIN = AMIN1(WLMIN,WLAST)
C
         RSTRMX = 0.0
         RSTR01 = 0.0
C
         CIRC = 0.0
         DO 1000 ISTPP=1,NSTP4
            XLAST = XNEW
            YLAST = YNEW
            ZLAST = ZNEW
            IE = IELGR(1)
            IEOFF=NXYZ*(IE-1)+1
            CALL EVALSC( U0 , U(IEOFF) ,RR , 1 )
            CALL EVALSC( V0 , V(IEOFF) ,RR , 0 )
            CALL EVALSC( W0 , W(IEOFF) ,RR , 0 )
C
            XLMAX = AMAX1(XLMAX,XLAST)
            YLMAX = AMAX1(YLMAX,YLAST)
            ZLMAX = AMAX1(ZLMAX,ZLAST)
            XLMIN = AMIN1(XLMIN,XLAST)
            YLMIN = AMIN1(YLMIN,YLAST)
            ZLMIN = AMIN1(ZLMIN,ZLAST)
C
            ULMAX = AMAX1(ULMAX,U0)
            VLMAX = AMAX1(VLMAX,V0)
            WLMAX = AMAX1(WLMAX,W0)
            ULMIN = AMIN1(ULMIN,U0)
            VLMIN = AMIN1(VLMIN,V0)
            WLMIN = AMIN1(WLMIN,W0)
            SPEED = U0**2 + V0**2 + W0**2
            SLMAX = AMAX1(SLMAX,SPEED)
            SLMIN = AMIN1(SLMIN,SPEED)
C
            CALL LAGRNGE(DT2,1,INTDIR,DTPRIM,DCIRC,0)
C
            XNEW = XLG(1,1)
            YNEW = XLG(2,1)
            ZNEW = XLG(3,1)
C
            DX = XNEW - XLAST
            DY = YNEW - YLAST
            DZ = ZNEW - ZLAST
            CIRC = CIRC + U0*DX + V0*DY + W0*DZ
C
C           Check for closing of streamlines
C
            RSTR1  = (XNEW-X00)**2 + (YNEW-Y00)**2 + (ZNEW-Z00)**2
            IF (ISTPP.EQ.1) RSTEP = RSTR1*STPTOL
            IF (RSTR1.LT.RSTR01.OR.RSTR1.LT.RSTEP) THEN
C
               IE = IELGR(1)
               IEOFF=NXYZ*(IE-1)+1
               CALL EVALSC( U0 , U(IEOFF) ,RR , 1 )
               CALL EVALSC( V0 , V(IEOFF) ,RR , 0 )
               CALL EVALSC( W0 , W(IEOFF) ,RR , 0 )
C
               DX = X00 - XNEW
               DY = Y00 - YNEW
               DZ = Z00 - ZNEW
               CIRC = CIRC + U0*DX + V0*DY + W0*DZ
               GOTO 1500
            ENDIF
            RSTRMX = AMAX1(RSTRMX,RSTR1)
            RSTR01 = STRTOL*RSTRMX
 1000    CONTINUE
 1500    CONTINUE
C
         IF (IRICH.EQ.1) THEN
            CIRC1 = CIRC
         ELSE
            CIRC2 = 2.0*CIRC - CIRC1
            WRITE(9,*) 'The circulation is:',CIRC2,CIRC,CIRC1
            write(9,*) 'umax:',ulmax,vlmax,wlmax,slmax
            write(9,*) 'umin:',ulmin,vlmin,wlmin,slmin
            write(9,*) 'xmax:',xlmax,ylmax,zlmax
            write(9,*) 'xmin:',xlmin,ylmin,zlmin
            write(s,'(1X,A19,3G14.3)')
     $      'The circulation is:',CIRC2,CIRC,CIRC1
            call prs(s//'$')
         ENDIF
 2000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine project(v0,v1,v2,iset)
      INCLUDE 'basics.inc'
      DIMENSION V0(3),V1(3),V2(3)
      DIMENSION VT(3),VT1(3),VT2(3)
      SAVE VT,VT1,VT2
C
      IF (ISET.EQ.1) THEN
C
C        Set up projection vectors
C
         CALL COPY(VT,V1,3)
         CALL NORM3D(VT)
         CALL CROSS(VT2,VT,VN)
         CALL ADD2(VT,V0,3)
         X1IS = XISO(VT(1),VT(2),VT(3))
         Y1IS = YISO(VT(1),VT(2),VT(3))
         CALL ADD2(VT2,V0,3)
         X2IS = XISO(VT2(1),VT2(2),VT2(3))
         Y2IS = YISO(VT2(1),VT2(2),VT2(3))
C
         RJAC = X1IS*Y2IS-X2IS*Y1IS
         IF (ABS(RJAC).LT.1.0E-05) THEN
            CALL PRS('Please choose a different plane or rotate view.$')
 
            IFPLAN=.FALSE.
            IFPROJ=.FALSE.
            return
         ENDIF
         ATWID =  Y2IS/RJAC
         BTWID = -X2IS/RJAC
         CTWID = -Y1IS/RJAC
         DTWID =  X1IS/RJAC
C
         DO 200 J=1,3
            VT1(J) = ATWID*VT(J) + CTWID*VT2(J)
            VT2(J) = BTWID*VT(J) + DTWID*VT2(J)
            VT (J) = V0(J)
  200    CONTINUE
         return
      ENDIF
      IF (ISET.EQ.2) THEN
C
C        Return xyz vector, given xmouse,ymouse
C
         DO 300 J=1,3
            V0(J) = VT(J) +  V1(1)*VT1(J) + V2(1)*VT2(J)
  300    CONTINUE
         return
      ENDIF
      IF (ISET.EQ.3) THEN
C
C        Return projection of V1 onto plane
C
         RLNGTH = DOTPROD(VN,V1)
         DO 400 J=1,3
            V0(J) = V1(J) - RLNGTH*VN(J)
  400    CONTINUE
         return
      ENDIF
      IF (ISET.EQ.4) THEN
C
C        Return projection of V0,V1,V2 onto plane
C
         RLNGTH = VN(1)*V0(1)+VN(2)*V1(1)+VN(3)*V2(1)
         V0(1) = V0(1) - RLNGTH*VN(1)
         V1(1) = V1(1) - RLNGTH*VN(2)
         V2(1) = V2(1) - RLNGTH*VN(3)
         return
      ENDIF
      END
c-----------------------------------------------------------------------
      subroutine cross(v1,v2,v3)
C
C     Compute Cartesian vector cross product.
C
      DIMENSION V1(3),V2(3),V3(3)
C
      V1(1) = V2(2)*V3(3) - V2(3)*V3(2)
      V1(2) = V2(3)*V3(1) - V2(1)*V3(3)
      V1(3) = V2(1)*V3(2) - V2(2)*V3(1)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine norm3d(v1)

C     Compute Cartesian vector dot product.

      DIMENSION V1(3)

      VLNGTH = DOTPROD(V1,V1)
      if (vlngth.le.0) return

      vlngth = sqrt(vlngth)
      v1(1) = v1(1) / vlngth
      v1(2) = v1(2) / vlngth
      v1(3) = v1(3) / vlngth

      return
      END
C----------------------------------------------------
C
      subroutine calcerr(tt,iferr)
C
C----------------------------------------------------
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      return
      END
C----------------------------------------------------------------------
      subroutine extrap(est,e,nx,ny,nz,ie)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DIMENSION E(1),X(4),Y(4)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine strfct(psi)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      REAL PSI(1)
      real psimxx,psimnn
      save psimxx,psimnn
      data psimxx,psimnn / -1.e20, 1.e20 /
C
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      COMMON /CTMP1/  ARCR(NXM,NYM),ARCS(NXM,NYM)
     $               ,PINTX(NXM,NXM),PINTY(NYM,NYM),FLAG(NELM)
     $ ,XRM1(NXM,NYM),YRM1(NXM,NYM),XSM1(NXM,NYM),YSM1(NXM,NYM)
      DIMENSION II1(4),JJ1(4),II2(4),JJ2(4)
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
C     Set ILGRNG=0 so that we know that the Lagrangian (streamline) data
C     in COMMON /CTMP1/ has been clobbered and will need to be recomputed.
C
      ILGRNG=0
C
C     Set matching side indices to set streamfunctions equivalent.
C
C              3
C          +------+
C          |      |
C        4 |      | 2
C          |      |
C          +------+
C             1
C
      II1(1)=1
      JJ1(1)=1
      II1(2)=NX
      JJ1(2)=1
      II1(3)=NX
      JJ1(3)=NY
      II1(4)=1
      JJ1(4)=NY
C
      II2(1)=NX
      JJ2(1)=1
      II2(2)=NX
      JJ2(2)=NY
      II2(3)=1
      JJ2(3)=NY
      II2(4)=1
      JJ2(4)=1
C
C     Set up partial integral operators, PINTX, PINTY:
C
      CALL SPINT(PINTX,ZPTS,WGHT,NX)
      CALL SPINT(PINTY,ZPTS,WGHT,NY)
C
C     Compute PSI locally for each element
C
      DO 100 IE=1,NEL
         CALL STREM2(PSI,IE)
  100 CONTINUE
C
C     Adjust the streamfunction level for all elements, starting with IE=1.
C
C     Flag = 0     -     not set
C     Flag = 1     -     set
C     Flag = 2     -     saturated, i.e. - all neighbors are set.
C
      CALL RZERO(FLAG,NEL)
      if (ifaxis) then
         call find_y_flag(k)
         flag(k) = 1.
      else
         FLAG(1)=1.0
      endif
      DO 1000 IEE=1,NEL
      DO 1000 IE=1,NEL
         IF (FLAG(IE).EQ.1.) THEN
         DO 800 ISIDE=1,4
            IF (CBC(ISIDE,IE,1).EQ.'E') THEN
c              IEN=IFIX(BC(1,ISIDE,IE,1))
               ien=bc(1,iside,ie,1)+0.25   ! just round up, slightly, + floor
               IF (FLAG(IEN).EQ.0.) THEN
C
C                 Adjust neighboring element.
C
c                 ISN=IFIX(BC(2,ISIDE,IE,1))
                  isn=bc(2,iside,ie,1)+0.25   ! round up, slightly, + floor
                  I1=II1(ISIDE)
                  J1=JJ1(ISIDE)
                  I2=II2(ISN)
                  J2=JJ2(ISN)
                  DIF=PSI(ind(I1,J1,1,IE))-PSI(ind(I2,J2,1,IEN))
                  DO 600 I=1,NX
                  DO 600 J=1,NY
                     PSI(ind(I,J,1,IEN))=PSI(ind(I,J,1,IEN))+DIF
  600             CONTINUE
                  FLAG(IEN)=1.
               ENDIF
            ENDIF
  800       CONTINUE
            FLAG(IE)=2.
         ENDIF
 1000 CONTINUE
c
c     Adjust constant so that min_strf = -max_strf     4/23/01 pff
c
      ntot = nx*ny*nz*nel
      psimin = glmin(psi,ntot)
      psimax = glmax(psi,ntot)
      vxmn = glmin(u,ntot)
      vxmx = glmax(u,ntot)
      vymn = glmin(v,ntot)
      vymx = glmax(v,ntot)
      write(6,1) psimin,psimax,vxmn,vxmx,vymn,vymx
    1 format('psiuv:',1p6e12.4)
      psimid = -(psimax+psimin)/2.
c
c     write(6,*) 'this is psimid:',psimid,ntot
      call cadd(psi,psimid,ntot)
      psimin = glmin(psi,ntot)
      psimax = glmax(psi,ntot)
c     write(6,*) 'this is psimnx:',psimin,psimax
c
c     Quick hack 4/21/01
c
c     psimxx = max(psimxx,psimax)
c     psimnn = min(psimnn,psimin)
c     dsi = 1.e-4*(psimxx-psimnn)
c     psimxx = psimxx + dsi
c
c     dsi = 1.e-4*(psimax-psimin)
c     psimax = psimax + dsi
c     do i=1,ntot
c        psi(i) = psimax-abs(psi(i))
c        psi(i) = log(psi(i))
c     enddo
c     call prs('Plotting log(psi).$')
c
      return
      END
c-----------------------------------------------------------------------
      subroutine strem2(psi,ie)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      REAL PSI(1)
      COMMON /CTMP1/  ARCR(NXM,NYM),ARCS(NXM,NYM)
     $               ,PINTX(NXM,NXM),PINTY(NYM,NYM),FLAG(NELM)
     $ ,XRM1(NXM,NYM),YRM1(NXM,NYM),XSM1(NXM,NYM),YSM1(NXM,NYM)
C
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
      CALL MXM(DGDR,NX,XP(IND(1,1,1,IE)),NX,XRM1,NY)
      CALL MXM(DGDR,NX,YP(IND(1,1,1,IE)),NX,YRM1,NY)
C
      CALL MXM(XP(IND(1,1,1,IE)),NX,DGDRT,NY,XSM1,NY)
      CALL MXM(YP(IND(1,1,1,IE)),NX,DGDRT,NY,YSM1,NY)
C
      NXY=NX*NY
C
      CALL RZERO(PSI(IND(1,1,1,IE)),NXY)
C
C        Begin by computing psi along the left border:
C
c     write(6,*) 'this is ifaxis:',ifaxis,ie
      if (.not.ifaxis) then
         DO 130 IP=1,NY
            IPP = NX*(IP-1) + 1
            FLUX=U(IND(1,IP,1,IE))*YSM1(IPP,1)
     $          -V(IND(1,IP,1,IE))*XSM1(IPP,1)
C
            DO 120 IY=2,NY
               IYP = IY + NX*(IP-1)
               PSI(IND(1,IY,1,IE))=PSI(IND(1,IY,1,IE))
     $                            +PINTY(IYP,1)*FLUX
  120       CONTINUE
  130    CONTINUE
C
C        Integrate in R direction for each row, IY.
C
         DO 300 IY=1,NY
            DO 200 IX=2,NX
               PSI(IND(IX,IY,1,IE))=PSI(IND(1,IY,1,IE))
  200       CONTINUE
            DO 230 IP=1,NX
               IPY = IP + NX*(IY-1)
               FLUX=U(IND(IP,IY,1,IE))*YRM1(IPY,1)
     $             -V(IND(IP,IY,1,IE))*XRM1(IPY,1)
C
               DO 220 IX=2,NX
                  IXP = IX + NX*(IP-1)
                  PSI(IND(IX,IY,1,IE))=PSI(IND(IX,IY,1,IE))
     $                                +PINTX(IXP,1)*FLUX
  220          CONTINUE
  230       CONTINUE
  300    CONTINUE
c
      else
c
C        axisymmetric case        (pff 11/29/98)
c
         one   = 1.
         twopi = 8.*atan(one)
         DO 135 IP=1,NY
            IPP = NX*(IP-1) + 1
            FLUX=U(IND(1,IP,1,IE))*YSM1(IPP,1)
     $          -V(IND(1,IP,1,IE))*XSM1(IPP,1)
            FLUX=flux*twopi*yp(ind(1,ip,1,ie))
C
            DO 125 IY=2,NY
               IYP = IY + NX*(IP-1)
               PSI(IND(1,IY,1,IE))=PSI(IND(1,IY,1,IE))
     $                            +PINTY(IYP,1)*FLUX
  125       CONTINUE
  135    CONTINUE
C
C        Integrate in R direction for each row, IY.
C
         DO 305 IY=1,NY
            DO 205 IX=2,NX
               PSI(IND(IX,IY,1,IE))=PSI(IND(1,IY,1,IE))
  205       CONTINUE
            DO 235 IP=1,NX
               IPY = IP + NX*(IY-1)
               FLUX=U(IND(IP,IY,1,IE))*YRM1(IPY,1)
     $             -V(IND(IP,IY,1,IE))*XRM1(IPY,1)
               FLUX=flux*twopi*yp(ind(ip,iy,1,ie))
C
               DO 225 IX=2,NX
                  IXP = IX + NX*(IP-1)
                  PSI(IND(IX,IY,1,IE))=PSI(IND(IX,IY,1,IE))
     $                                +PINTX(IXP,1)*FLUX
  225          CONTINUE
  235       CONTINUE
  305    CONTINUE
      endif
C
      return
      END
c-----------------------------------------------------------------------
      subroutine spint(p,s,wght,n)
C
C     Compute the indefinite integral of j Lagrangian interpolants
C     at i collocation points:
C
      DIMENSION P(N,N),S(N),WGHT(N)
C
      N2=N**2
      CALL RZERO(P,N2)
C
      DO 400 I=2,N-1
      DO 400 J=1,N
      SUM=0.0
C
C        Stretch coordinate:
C
         DO 300 K=1,N
            XSI=(S(I)+1.0)*(S(K)+1.0)/2.0 - 1.0
            SUM=SUM+HLEG(J,XSI,S,N)*WGHT(K)
  300    CONTINUE
         P(I,J) = SUM * (S(I)+1.0)/2.0
  400 CONTINUE
C
      DO 500 J=1,N
         P(N,J)=WGHT(J)
  500 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      logical function ifcond(xyzlg)
C
      include 'basics.inc'
      include 'basicsp.inc'
      DIMENSION XYZLG(3)
C
C     Example, check if |Y| > delta.  If so, IFCOND is true.
C
      IFCOND=.FALSE.
c     IF (PARAM(51).GT.0.AND.ABS(XYZLG(2)).GT.PARAM(51)) IFCOND=.TRUE.
      return
      END
c-----------------------------------------------------------------------
      subroutine factor_solve(x,a,lda,b,n)
      real a(lda,n)
      real x(n),b(n)
c
      call sgefa(a,lda,n,x,info)
c
      if (info.ne.0) then
         write(6,*) 'info in sgefa',info
      endif
c
      call sgesl(a,lda,n,x,b,0)
      call copy (x,b,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine sgefa(a,lda,n,ipvt,info)
      integer lda,n,ipvt(1),info
      real a(lda,1)
c
c     sgefa factors a real matrix by gaussian elimination.
c
c     sgefa is usually called by sgeco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for sgeco) = (1 + 9/n)*(time for sgefa) .
c
c     on entry
c
c        a       real(lda, n)
c                the matrix to be factored.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix and the multipliers
c                which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if  u(k,k) .eq. 0.0 .  this is not an error
c                     condition for this subroutine, but it does
c                     indicate that sgesl or sgedi will divide by zero
c                     if called.  use  rcond  in sgeco for a reliable
c                     indication of singularity.
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas saxpy,sscal,isamax
c
c     internal variables
c
      real t
      integer isamax,j,k,kp1,l,nm1
c
c
c     gaussian elimination with partial pivoting
c
      info = 0
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 k = 1, nm1
         kp1 = k + 1
c
c        find l = pivot index
c
         l = isamax(n-k+1,a(k,k),1) + k - 1
         ipvt(k) = l
c
c        zero pivot implies this column already triangularized
c
         if (a(l,k) .eq. 0.0e0) go to 40
c
c           interchange if necessary
c
            if (l .eq. k) go to 10
               t = a(l,k)
               a(l,k) = a(k,k)
               a(k,k) = t
   10       continue
c
c           compute multipliers
c
            t = -1.0e0/a(k,k)
            call sscal(n-k,t,a(k+1,k),1)
c
c           row elimination with column indexing
c
            do 30 j = kp1, n
               t = a(l,j)
               if (l .eq. k) go to 20
                  a(l,j) = a(k,j)
                  a(k,j) = t
   20          continue
               call saxpy(n-k,t,a(k+1,k),1,a(k+1,j),1)
   30       continue
         go to 50
   40    continue
            info = k
   50    continue
   60 continue
   70 continue
      ipvt(n) = n
      if (a(n,n) .eq. 0.0e0) info = n
      return
      end
c-----------------------------------------------------------------------
      subroutine sgesl(a,lda,n,ipvt,b,job)
      integer lda,n,ipvt(1),job
      real a(lda,1),b(1)
c
c     sgesl solves the real system
c     a * x = b  or  trans(a) * x = b
c     using the factors computed by sgeco or sgefa.
c
c     on entry
c
c        a       real(lda, n)
c                the output from sgeco or sgefa.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c        ipvt    integer(n)
c                the pivot vector from sgeco or sgefa.
c
c        b       real(n)
c                the right hand side vector.
c
c        job     integer
c                = 0         to solve  a*x = b ,
c                = nonzero   to solve  trans(a)*x = b  where
c                            trans(a)  is the transpose.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains a
c        zero on the diagonal.  technically this indicates singularity
c        but it is often caused by improper arguments or improper
c        setting of lda .  it will not occur if the subroutines are
c        called correctly and if sgeco has set rcond .gt. 0.0
c        or sgefa has set info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call sgeco(a,lda,n,ipvt,rcond,z)
c           if (rcond is too small) go to ...
c           do 10 j = 1, p
c              call sgesl(a,lda,n,ipvt,c(1,j),0)
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas saxpy,sdot
c
c     internal variables
c
      real sdot,t
      integer k,kb,l,nm1
c
      nm1 = n - 1
      if (job .ne. 0) go to 50
c
c        job = 0 , solve  a * x = b
c        first solve  l*y = b
c
         if (nm1 .lt. 1) go to 30
         do 20 k = 1, nm1
            l = ipvt(k)
            t = b(l)
            if (l .eq. k) go to 10
               b(l) = b(k)
               b(k) = t
   10       continue
            call saxpy(n-k,t,a(k+1,k),1,b(k+1),1)
   20    continue
   30    continue
c
c        now solve  u*x = y
c
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/a(k,k)
            t = -b(k)
            call saxpy(k-1,t,a(1,k),1,b(1),1)
   40    continue
      go to 100
   50 continue
c
c        job = nonzero, solve  trans(a) * x = b
c        first solve  trans(u)*y = b
c
         do 60 k = 1, n
            t = sdot(k-1,a(1,k),1,b(1),1)
            b(k) = (b(k) - t)/a(k,k)
   60    continue
c
c        now solve trans(l)*x = y
c
         if (nm1 .lt. 1) go to 90
         do 80 kb = 1, nm1
            k = n - kb
            b(k) = b(k) + sdot(n-k,a(k+1,k),1,b(k+1),1)
            l = ipvt(k)
            if (l .eq. k) go to 70
               t = b(l)
               b(l) = b(k)
               b(k) = t
   70       continue
   80    continue
   90    continue
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine blas_saxpy(n,sa,sx,incx,sy,incy)
c
c     constant times a vector plus a vector.
c     uses unrolled loop for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      real sx(1),sy(1),sa
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if (sa .eq. 0.0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        sy(iy) = sy(iy) + sa*sx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,4)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        sy(i) = sy(i) + sa*sx(i)
   30 continue
      if( n .lt. 4 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,4
        sy(i) = sy(i) + sa*sx(i)
        sy(i + 1) = sy(i + 1) + sa*sx(i + 1)
        sy(i + 2) = sy(i + 2) + sa*sx(i + 2)
        sy(i + 3) = sy(i + 3) + sa*sx(i + 3)
   50 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine streamline_draw (w,iostep,dti,intdir,nstep,iclr,n)
c
c     Program to compute path or streamlines by RK4
c
      real w(3),wo(3),w1(3),w2(3),wh(3),eps,dt,dt2,t,t2,tt
      real r(3)
c
c     w(i) -- state vector:   w(1-3) = (x,y,z)
c                             w(4-6) = (u,v,w)
c                             w(7)   = scalar
c
c
c     n = length of vector
c     m = order of scheme, i.e. RK4 ==> m=4
c
c
      m=4
c
c
c     eps = 10.0**(-abs(iostep))
      eps = 2.0**(-iostep)
c
      dt0 = dti
      dt  = dti
      if (dti.eq.0) call set_dt(dt0,w2,w1,0,m,n,eps,dti)
c
      dt  = dt0/10.
c
c     Take one step to get initial dt based upon eps
c
      dt2 = 0.5*dt
      t2  = t+dt2
      tt  = t+dt
c
      call rk4(w1,w ,t ,dt2,n)
      call rk4(w2,w1,t2,dt2,n)
c
      call rk4(w1,w ,t ,dt ,n)
C
      if (dti.eq.0) call set_dt(dt,w2,w1,1,m,n,eps,dti)
      call prsrrr('This is dt:$',dt,dt0,eps)
c     call prs ('Input new dt:$')
c     call rer(dt)
c
c
c     TIME STEPPING:
c
      t = 0.
      iopt = 1
      do k=1,nstep
         call out_w (w,t,dtn,dto,k,nstep,iclr)
c
c        RK4 w/ variable dt
c
         dto = dt
         dt2 = 0.5*dto
         t2  = t+dt2
         tt  = t+dto
c
         call rk4(wh,w ,t ,dt2,n)
         call rk4(w2,wh,t2,dt2,n)
c
         call rk4(w1,w ,t ,dto,n)
c
         if (dti.eq.0) call set_dt(dt,w2,w1,k,m,n,eps,dti)
         dte = dt
         dt  = min(dt,dto*1.1)
         dt  = max(dt,dto/10.)
         dt  = min(dt,dt0)
         dt  = max(dt,dt0/(25*abs(iostep)))
c
c        Save current w-vector for "close_w" routine.
         call copy(wo,w,n)
c
c        Choose updating option
c
         if (iopt.eq.1) then
c           Take "real" RK4 step, using computed "dt"
            call rk4 (w1,w ,t ,dt ,n)
            call copy(w,w1,n)
            t   = t+dt
            dtn = dt
         elseif (iopt.eq.2) then
c           Use the dt/2 value computed above.
            call copy(w,w2,n)
            t   = tt
            dtn = dto
         elseif (iopt.eq.3) then
c           Use the dt & dt/2 values computed above w/ Richardson ext.
            c2 = 16./15.
            c1 = -1./15.
            call add3s12(w,w1,c1,w2,c2,n)
            t   = tt
            dtn = dto
         elseif (iopt.eq.4) then
c           Use the dt/2 value if dt < dt*/2
            if (dt2.lt.dt.and.dt.lt.dto) then
c              Use the dt/2 value computed above.
               call copy(w,wh,n)
               dtn = dt2
            else
c              Take "real" RK4 step, using computed "dt"
               call rk4 (w1,w ,t ,dt ,n)
               call copy(w,w1,n)
               dtn = dt
            endif
         elseif (iopt.eq.5) then
c           Use the two dt/2 values if dt < dt*/2
            if (dt2.lt.dt.and.dt.lt.dto) then
c              Use the dt/2 value computed above.
               call copy(w,w2,n)
               dtn = dto
            else
c              Take "real" RK4 step, using computed "dt"
               call rk4 (w1,w ,t ,dt ,n)
               call copy(w,w1,n)
               dtn = dt
            endif
         endif
c
c
         tnew = t+dtn
         t    = t+dtn
         dt   = dtn
c
c        Check current position for possible termination
c        (i.e., if point is not inside any subdomain).
c
         call findre(ie,r,w,rminmax,ierr)
         if (rminmax.ge.1.01) goto 1001
c
      enddo
 1000 continue
      call out_w     (w,t,dtn,dto,-1,k,iclr)
 1001 continue
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_dt(dt_sti,w0,w1,kstep,m,n,eps,dti)
      real w0(1),w1(1)
      real d(4)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real est_fintime
      save est_fintime
c
c     First step, request dt.   Subsequently, use stability criteria....
c
      dt_str = abs(dt_sti)
      if (kstep.eq.0) then
         ntot = nx*ny*nz*nel
         if (if3d) then
            uchar2 = glsc1(wkv1,ntot)/ntot
            vchar2 = glsc1(wkv2,ntot)/ntot
            wchar2 = glsc1(wkv3,ntot)/ntot
            uchar  = sqrt ( uchar2 + vchar2 + wchar2 )
            dxchar = glmax (xp,ntot)-glmin(xp,ntot)
     $             + glmax (yp,ntot)-glmin(yp,ntot)
     $             + glmax (zp,ntot)-glmin(zp,ntot)
         else
               uchar2 = glsc1(wkv1,ntot)/ntot
               vchar2 = glsc1(wkv2,ntot)/ntot
               uchar  = sqrt ( uchar2 + vchar2 )
               dxchar = glmax (xp,ntot)-glmin(xp,ntot)
     $                + glmax (yp,ntot)-glmin(yp,ntot)
         endif
         est_fintime = dxchar/uchar
         dt_str      = est_fintime/10.
         write(6,*) dti,'est_fin:',est_fintime,dxchar,uchar,ntot,dt_str
c        call prsrrr('Input est_fin:$',est_fintime,dxchar,uchar)
c        call rer(est_fintime)
c
      else
c
c        Set dt for mth order scheme
c
c        Max error:
         delta = 0.0
         call sub3(d,w0,w1,n)
         delta = some_norm(d,w0,w1,n)
c
         if (delta.lt.1.e-6) then
            dt_str = 1.1*dt_str
c           write(6,*) dti,'double dt_str:',dt_str,delta
            if (dt_sti.lt.0) dt_str = -abs(dt_str)
            dt_sti = dt_str
            return
         endif
c
         ek   = (2**m)*delta/(2**m-1.)
         dt_stro = dt_str
         ekdt = ek/dt_str
         epsp = eps/est_fintime
         dtnew = ( epsp*dt_str**(m+1)*(2**m-1)/ (delta*2**m) )**(1./m)
c        write(6,*) dti,'delta:',delta,dt_str,dtnew,epsp
         dt_str= dtnew
      endif
   6  format(i6,1x,a6,1p9e14.6)
c
      if (dt_sti.lt.0) dt_str = -abs(dt_str)
      dt_sti = dt_str
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_f(f,xpos,t_str)
c
c     Compute RHS of ODE:
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c
      real f(3),xpos(3),r(3),rminmax
      integer ie,ierr
c
c     f = velocity at point
c
      if (ndim.eq.2) xpos(3)=0.
      call findre(ie,r,xpos,rminmax,ierr)
      if (ierr.ne.0) return
c
      nxyz = nx*ny*nz
      ieoff = nxyz*(ie-1) + 1
      call evalsc( f(1) , wkv1(ieoff) , r , 1 )
      call evalsc( f(2) , wkv2(ieoff) , r , 1 )
      if (if3d) call evalsc( f(3) , wkv3(ieoff) , r , 1 )
c     write(6,1) 'fxr:',(f(j),j=1,3),(xpos(j),j=1,3),(r(j),j=1,3)
c   1 format(a4,3(2x,3f12.7))
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_w(xpos,t_str,dt_str,dto,i,nstep,iclr)
      real xpos(3),r(3)
c
      include 'basics.inc'
      include 'basicsp.inc'
      common /ws_comm/ wscale,wmin
c
c     Output results
c
      if (iclr.ge.0.or.wscale.eq.0.0) then
         icstr=abs(iclr)
      else
C        color streamline according to current scalar variable
         call findre(ie,r,xpos,rminmax,ierr)
         nxyz = nx*ny*nz
         ieoff = nxyz*(ie-1) + 1
         call evalsc( val , work(ieoff) , r , 1 )
         tmp=wscale*(val-wmin)
         icstr=icolor1(tmp)
      endif
c
      call color(icstr)
      xxis=xisom(xpos(1),xpos(2),xpos(3))
      yyis=yisom(xpos(1),xpos(2),xpos(3))
      if (i.eq.1) call movec(xxis,yyis)
      call drawc(xxis,yyis)

      if (iclr.eq.-8) write(88,8) (xpos(k),k=1,ndim),val
    8 format(1p4e16.6)

      return
      end
c-----------------------------------------------------------------------
      subroutine rk4(wnew,w,t,dt,n)
      real wnew(1),w(1),t,dt
      real wh(4),f1(4),f2(4),f3(4),f4(4)
c
c     RK4:
c
      dt2 = dt/2.
      dt3 = dt/3.
      dt6 = dt/6.
c
      t2 = t+dt2
      tt = t+dt
c
      call compute_f (f1,w ,t )
      call add3s2    (wh,w,f1,dt2,n)
c
      call compute_f (f2,wh,t2)
      call add3s2    (wh,w,f2,dt2,n)
c
      call compute_f (f3,wh,t2)
      call add3s2    (wh,w,f3,dt ,n)
c
      call compute_f (f4,wh,tt)
c
      call copy      (wnew,w,n)
      call add2s2    (wnew,f1,dt6,n)
      call add2s2    (wnew,f2,dt3,n)
      call add2s2    (wnew,f3,dt3,n)
      call add2s2    (wnew,f4,dt6,n)
c
      return
      end
c-----------------------------------------------------------------------
      function some_norm(e,u,v,n)
      real e(1),u(1),v(1)
      real e22,u22,v22
c
      e22 = 0.
      u22 = 0.
      v22 = 0.
      do k=1,n
         e22  = e22 + e(k)*e(k)
         u22  = u22 + u(k)*u(k)
         v22  = v22 + v(k)*v(k)
      enddo
      e22 = 2.*e22/(u22+v22)
      if (e22.gt.0.) then
         some_norm = sqrt( e22 )
      elseif (e22.lt.0.) then
         write(6,*) 'NORM LT ZERO?'
     $          ,(e(j),j=1,n),(u(j),j=1,n),(v(j),j=1,n)
     $          ,e22,u22,v22
         some_norm = 0.
      else
         some_norm = 0.
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine u_fld_values_in(x_in,y_in,z_in,ie,ierr)
c
c     Read one element at a time
c
c     Take as input unformatted (param(66)=4) .fld file and put it as 
c     output
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      character*40 fname
      character*80 s80
      character*1  s81(80)
      equivalence (s81,s80)
c
      real*4  bytetest
      logical if_byte_sw
      save    if_byte_sw
c
      logical if_byte_swap_test, if_byte_swap_test8
c
      common /value_fld_i/ nxr,nyr,nzr,neltr
      common /value_fld_c/ excode_i(10)
      character*2 excode_i
c
      real x_in(1) , y_in(1) , z_in(1)
c
      real    xmin,xmax,xdel,ymin,ymax,ydel,zmin,zmax,zdel
      save    xmin,xmax,xdel,ymin,ymax,ydel,zmin,zmax,zdel
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
c
      write(6,*) 'inside u_fld_values_in'
      if (ie.eq.1) then
         call blank(fname,40)
         call prs(' Input name of .fld file w/ xyz data.$')
         call res(fname,40)
c
         len = ltrunc(fname,40)
         call chcopy(s81,fname,len)
         s81(len+1) = '$'
         call prs(s81)
c
         call terminate_string(fname,40)
         call byte_open(fname)
c
c        Get header info from new file, so we know how many elements, etc.
c
         call byte_read(s80,20)
         write(6,*) s80
         READ(s80,'(4i4,1x,g13.4,i5,1x,10a2,i2)',err=1500,end=1500)
     $    neltr,nxr,nyr,nzr,rstime,istepr,(excode_i(i),i=1,10),idsize
c
c        Get swap_test info
c
c        read test pattern to determine if byte-swapping is req'd
         call byte_read(bytetest,1)
         if_byte_sw = if_byte_swap_test(bytetest)
c
c        We assume that the data is just xyz (or just xy for 2d)
c
         nxyzr = nxr*nyr*nzr
         k=1
         write(6,*) 'this is nxyzr:',nxyzr,nxr,nyr,nzr,neltr
      endif
c
      nxyzr = nxr*nyr*nzr
      k = 1
c
      write(6,*) 'read:',ie,k,if3d,nxyzr,if_byte_sw
      call byte_read(x_in(k),nxyzr)
      call byte_read(y_in(k),nxyzr)
      if (nzr.eq.nxr) call byte_read(z_in(k),nxyzr)
c
      if (if_byte_sw) then
         call byte_reverse(x_in(k),nxyzr)
         call byte_reverse(y_in(k),nxyzr)
         if (nzr.eq.nxr) call byte_reverse(z_in(k),nxyzr)
      endif
c
c     Close input file
c
      if (ie.eq.neltr) call byte_close()
c
c
c
c     Are we using a reflected mesh to compute new values in a full mesh?
      if (icalld.eq.0) then
         call prs('Flush y values to < 0 for reflected mesh?$')
         call res(ans,1)
         icalld=1
         if (ans.eq.'y' .or. ans.eq.'Y') then
            icalld=2
         else
            call prs('Shift values? (enter n=no or x, or y, or z):$')
            call res(ans,1)
            ntot = nx*ny*nz*nel
            if (ans.eq.'x' .or. ans.eq.'X') then
               xmin = glmin(xp,ntot)
               xmax = glmax(xp,ntot)
               xdel = xmax-xmin
               icalld = 3
            elseif (ans.eq.'y' .or. ans.eq.'Y') then
               ymin = glmin(yp,ntot)
               ymax = glmax(yp,ntot)
               ydel = ymax-ymin
               icalld = 4
            elseif (ans.eq.'z' .or. ans.eq.'Z') then
               zmin = glmin(zp,ntot)
               zmax = glmax(zp,ntot)
               zdel = zmax-zmin
               icalld = 5
            else
               icalld = 1
            endif
         endif
      endif
c
      if (icalld.eq.2) then
         do i=1,nxyzr
            y_in(i) = -abs(y_in(i))
         enddo
      elseif (icalld.eq.3) then   ! shift x
         do i=1,nxyzr
            if (x_in(i).gt.xmax) then
               nfold = (x_in(i)-xmin)/xdel
               x_in(i) = x_in(i) - nfold*xdel
            endif
         enddo
      elseif (icalld.eq.4) then   ! shift y
         do i=1,nxyzr
            if (y_in(i).gt.ymax) then
               nfold = (y_in(i)-ymin)/ydel
               y_in(i) = y_in(i) - nfold*ydel
            endif
         enddo
      elseif (icalld.eq.5) then   ! shift z
         do i=1,nxyzr
            if (z_in(i).gt.zmax) then
               nfold = (z_in(i)-zmin)/zdel
               z_in(i) = z_in(i) - nfold*zdel
            endif
         enddo
      endif
c
      ierr = 0
      return
c
c
 1500 continue
      s81(70) = '$'
      call prs(s81)
      call prs('Unable to open file.  Returning.$')
      call byte_close()
c
      ierr = 1
      return
      end
c-----------------------------------------------------------------------
      subroutine set_u_fld_values_out
c
c     Take as input unformatted (param(66)=4) .fld file and put it as 
c     output
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      character*40 fname
      character*80 s80
c
      real*4  bytetest
      logical if_byte_sw, if_byte_swap_test, if_byte_swap_test8
c
      common /value_fld_i/ nxr,nyr,nzr,neltr
      common /value_fld_c/ excode_i(10)
      character*2 excode_i
c
c
c     Set up output file
c
      call blank(fname,40)
      write(fname,1)
    1 format('out.fld00')
      call terminate_string(fname,40)
      call byte_open(fname)
      call prs('Writing data to out.fld00$')
c
c     Write header info from new file, so we know how many elements, etc.
c
      call blank(excode_i,20)
      if (ifflow) excode_i(1) = 'U '
      if (ifflow) excode_i(2) = 'P '
      if (ifheat) excode_i(3) = 'T '
      idsize = 4
      call blank(s80,80)
      write(6,'(4i4,1x,g13.4,i5,1x,10a2,i2)')
     $ neltr,nxr,nyr,nzr,time,istep,(excode_i(i),i=1,10),idsize
      write(s80,'(4i4,1x,g13.4,i5,1x,10a2,i2)')
     $ neltr,nxr,nyr,nzr,time,istep,(excode_i(i),i=1,10),idsize
      call byte_write(s80,20)
c
c     write test pattern to determine if byte-swapping is req'd
      test_pattern = 6.54321
      call byte_write(test_pattern,1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine u_fld_values
c
c     Take as input unformatted (param(66)=4) .fld file and put it as 
c     output
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      common /findri/ ixyzmin,ixyzemn,iexyzmn  ! location of closest point
      common /findrr/ dxyzmin
c
      character*40 fname
      character*80 s80
c
      real*4  bytetest
      logical if_byte_sw, if_byte_swap_test, if_byte_swap_test8
c
      common /value_fld_i/ nxr,nyr,nzr,neltr
      common /value_fld_c/ excode_i(10)
      character*2 excode_i
c
      real x_in(1) , y_in(1) , z_in(1)
      equivalence (x_in,wkv1)
      equivalence (y_in,wkv2)
      equivalence (z_in,wkv3)
c
c     These should be equivalenced to something, but there's no good
c     option since the metrics are used by INTERP.
c
      parameter (npte=nxm*nym*nzm)
      real uo(npte),vo(npte),wo(npte),po(npte),to(npte)
      equivalence (uo,work(1))
      equivalence (vo,work(1*npte+1))
      equivalence (wo,work(2*npte+1))
      equivalence (po,work(3*npte+1))
      equivalence (to,work(4*npte+1))
c
c
      write(6,*) 'inside u_fld_values, call in'
      ie = 1
      call u_fld_values_in(x_in,y_in,z_in,ie,ierr)
      if (ierr.ne.0) return
c
      ntot = nxr*nyr*nzr*neltr
      nmod = max(1,ntot/1000)
      nmod = max(nmod,100)
c
      uxshift = 0.0             !  Jan 8 2002 - allow velocity shift
      l = 0
      nxyzr = nxr*nyr*nzr
      do ie=1,neltr
         if (ie.gt.1)
     $   call u_fld_values_in(x_in,y_in,z_in,ie,ierr)
         do i=1,nxyzr
            l = l+1
            if (mod(l,nmod).eq.0) call prsii('Number done: $',l,ntot)
            ierr = 0
c
            if (l.eq.1) then
               call prs('Input constant shift for vx:$')
               call rer(uxshift)
            endif
c
            if (nzr.eq.nxr.and..not.if3d) z_in(i)=0.
            if (ifflow.and.ifheat) then
              ierr = 99
              call interp(uo(i),x_in(i),y_in(i),z_in(i),i0,u,ierr)
              if (ierr.ne.0) then
c                point not inside domain
                 uo(i) = user_flow(x_in(i),y_in(i),z_in(i))
                 vo(i) = 0.
                 wo(i) = 0.
                 po(i) = 0.
                 to(i) = user_temp(x_in(i),y_in(i),z_in(i))
c                write(6,6) i,' bad:',x_in(i),y_in(i),z_in(i)
c    $                               ,uo(i),to(i),
   6             format(i9,a5,1p5e12.4)
                 write(6,*) 'ixyzmin:',ixyzmin,iexyzmn
                 nxy   = nx*ny
                 nyz   = ny*nz
                 ixm   = mod1(ixyzmin,nyz)
                 izm   = (ixyzmin-1)/(nxy)+1
                 iym   = mod1(ixyzmin,nz)
                 iym   = (iym-1)/nx+1
                 ixyze = ixyzmin + nx*ny*nz*(iexyzmn-1)
                 xmn = xp(ixyze)
                 ymn = xp(ixyze)
                 zmn = xp(ixyze)
                 write(6,8) i,' bad:',x_in(i),y_in(i),z_in(i)
     $                               ,xmn,ymn,zmn,ixm,iym,izm,iexyzmn
   8             format(i9,a5,1p6e12.4,3i3,i6)
              else
                 ierr = 99
                 call interp(vo(i),x_in(i),y_in(i),z_in(i),i0,v,ierr)
                 ierr = 99
                 if (if3d) 
     $           call interp(wo(i),x_in(i),y_in(i),z_in(i),i0,w,ierr)
                 ierr = 99
                 call interp(po(i),x_in(i),y_in(i),z_in(i),i0,t,ierr)
                 ierr = 99
                 call interp(to(i),x_in(i),y_in(i),z_in(i),i0,t,ierr)
              endif
            elseif (ifflow) then
              ierr = 99
              call interp(uo(i),x_in(i),y_in(i),z_in(i),i0,u,ierr)
              if (ierr.ne.0) then
c                point not inside domain
c                uo(i) = user_flow(x_in(i),y_in(i),z_in(i))
                 uo(i) = 0.
                 vo(i) = 0.
                 wo(i) = 0.
                 po(i) = 0.
c                write(6,6) i,' bad:',
c    $               x_in(i),y_in(i),z_in(i),uo(i),to(i)
                 write(6,*) 'ixyzmin:',ixyzmin,iexyzmn
                 nxy   = nx*ny
                 nyz   = ny*nz
                 ixm   = mod1(ixyzmin,nyz)
                 izm   = (ixyzmin-1)/(nxy)+1
                 iym   = mod1(ixyzmin,nz)
                 iym   = (iym-1)/nx+1
                 ixyze = ixyzmin + nx*ny*nz*(iexyzmn-1)
                 xmn = xp(ixyze)
                 ymn = xp(ixyze)
                 zmn = xp(ixyze)
                 write(6,8) i,' bad:',x_in(i),y_in(i),z_in(i)
     $                               ,xmn,ymn,zmn,ixm,iym,izm,iexyzmn
              else
                 ierr = 99
                 call interp(vo(i),x_in(i),y_in(i),z_in(i),i0,v,ierr)
                 ierr = 99
                 if (if3d) 
     $           call interp(wo(i),x_in(i),y_in(i),z_in(i),i0,w,ierr)
                 ierr = 99
                 call interp(po(i),x_in(i),y_in(i),z_in(i),i0,t,ierr)
              endif
            elseif (ifheat) then
              ierr = 99
              call interp(to(i),x_in(i),y_in(i),z_in(i),i0,t,ierr)
              if (ierr.ne.0) then
                 to(i) = user_temp(x_in(i),y_in(i),z_in(i))
              endif
            endif
c
            uo(i) = uo(i)+uxshift        !  SHIFT ux
c
         enddo
         if (ie.eq.1) 
     $    open(unit=8,file='pf.tmp',status='unknown',form='unformatted')
         if (ifflow) then
           call pf_write(uo,nxyzr)
           call pf_write(vo,nxyzr)
           if (nzr.eq.nxr) call pf_write(wo,nxyzr)
           call pf_write(po,nxyzr)
         endif
         if (ifheat) call pf_write(to,nxyzr)
         write(6,*) 'Done write',ie,nxyzr
      enddo
c
c     Now, rewind data set, re-read, and dump using byte_write.
c
      call byte_close
      rewind(unit=8)
      call set_u_fld_values_out
      do ie=1,neltr
         if (ifflow) then
           call pf_read   (uo,nxyzr)
           call byte_write(uo,nxyzr)
           call pf_read   (vo,nxyzr)
           call byte_write(vo,nxyzr)
           if (nzr.eq.nxr) call pf_read   (wo,nxyzr)
           if (nzr.eq.nxr) call byte_write(wo,nxyzr)
           call pf_read   (po,nxyzr)
           call byte_write(po,nxyzr)
         endif
         if (ifheat) call pf_read   (to,nxyzr)
         if (ifheat) call byte_write(to,nxyzr)
         write(6,*) 'Done re-write',ie,nxyzr
      enddo
      close(unit=8)
      call byte_close
c
c
      call prs('NOTE: work must be reset.$')
c
      return
      end
c-----------------------------------------------------------------------
      function user_flow(xi,yi,zi)
c
      include 'basics.inc'
C
      user_flow=0.
c
C     Compute the Blasius profile
C
c
c     UMAX  =  1.0
c     delta=param(71)
c     if (delta.eq.0.0) DELTA= 0.58
c     U0   = 1.0
c     user_flow=UMAX*BLASIUS(zi,DELTA,U0)
c
      return
      end
c-----------------------------------------------------------------------
      function user_temp(x,y,z)
C
      user_temp = 1.
c
      return
      end
c-----------------------------------------------------------------------
      FUNCTION BLASIUS(Y,DELTA,U)
C
C     Return the velocity at a given y value for specified delta and U.
C
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
      DIMENSION U0(45),Y0(45),WORK(45)
      SAVE      U0    ,Y0    ,WORK
      DATA      U0 / 0.00000 , 0.06641 , 0.13277 , 0.19894 , 0.26471
     $             , 0.32979 , 0.39378 , 0.45627 , 0.51676 , 0.57477
     $             , 0.62977 , 0.68132 , 0.72899 , 0.77246 , 0.81152 
     $             , 0.84605 , 0.87609 , 0.90177 , 0.92333 , 0.94112
     $             , 0.95552 , 0.96696 , 0.97587 , 0.98269 , 0.98779
     $             , 0.99155 , 0.99425 , 0.99616 , 0.99748 , 0.99838
     $             , 0.99898 , 0.99937 , 0.99961 , 0.99977 , 0.99987
     $             , 0.99992 , 0.99996 , 0.99998 , 0.99999 , 1.00000
     $             , 1.00000 , 1.00000 , 1.00000 , 1.00000 , 1.00000 /
C
      IF (ICALLD.EQ.0) THEN
C        Initialize Blasius profile and spline fitting routine.
         ICALLD=1
         DO 10 I=1,45
            Y0(I)=FLOAT(I-1)/5.0
   10    CONTINUE
         CALL SPLINE(Y0,U0,45,WORK)
      ENDIF
C
      ETA=5.0*y/delta
C
      IF (ETA.GT.8.5) THEN
         BLASIUS=U
      ELSE
         CALL SPLINT(Y0,U0,WORK,45,ETA,VEL)
         BLASIUS=VEL*U
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine dump_xyz_fld
c
c     Take as input unformatted (param(66)=4) .fld file and put it as 
c     output
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      character*40 fname
      character*80 s80
c
      real*4  bytetest
      logical if_byte_sw, if_byte_swap_test, if_byte_swap_test8
c
      common /value_fld_i/ nxr,nyr,nzr,neltr
      common /value_fld_c/ excode_i(10)
      character*2 excode_i
c
c     Set up output file
c
      call blank(fname,40)
      write(fname,1)
    1 format('xyz.fld00')
      call terminate_string(fname,40)
      call byte_open(fname)
      call prs('Writing data to xyz.fld00$')
c
c     Write header info from new file, so we know how many elements, etc.
c
      call blank(excode_i,20)
      excode_i(1) = 'X '
      idsize = 4
      write(s80,'(4i4,1x,g13.4,i5,1x,10a2,i2)')
     $ nel,nx,ny,nz,time,istep,(excode_i(i),i=1,10),idsize
      call byte_write(s80,20)
c
c     write test pattern to determine if byte-swapping is req'd
      test_pattern = 6.54321
      call byte_write(test_pattern,1)
c
c     Dump data, scalar fld by scalar fld
c
      k=1
      nxyzr = nx*ny*nz
      do ie=1,nel
c
         call byte_write(xp(k),nxyzr)
         call byte_write(yp(k),nxyzr)
         if (if3d) call byte_write(zp(k),nxyzr)
c
         k=k+nxyzr
      enddo
c
c     Close output file
c
      call byte_close()
c
      return
      end
c-----------------------------------------------------------------------
      subroutine terminate_string(s,n)
c
c     Place a null character at the end of the string
c
      character*1 s(n)
      character*4 null
      integer     nill
      equivalence (nill,null)
c
      nill = 0
c
      iend = ltrunc(s,n)
      call chcopy(s(iend+1),null,4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine streamline_draw_2d_close
     $              (w,iostep,dti,intdir,nstep,iclr,n,iflast)
c
c     Program to compute path or streamlines by RK4
c
c     This version checks for "closing" streamlines
c
c
      real    w(3)
      logical iflast
c
      real    r(3),wo(3),w1(3),w2(3),wh(3),eps,dt,dt2,t,t2,tt
      logical if_close,if_closed
c
      real w0(3)
      save w0
c
c     w(i) -- state vector:   w(1-3) = (x,y,z)
c                             w(4-6) = (u,v,w)
c                             w(7)   = scalar
c
c     Save initial condition
      if (.not.iflast) call copy(w0,w,3)
c
c
c     n = length of vector
c     m = order of scheme, i.e. RK4 ==> m=4
c
      m=4
c
c
c     write(6,*) 'dti',dti,(w(k),k=1,3),nstep,n,iflast,iostep
c     eps = 10.0**(-abs(iostep))
      eps = 2.0**(-iostep)
c
      dt0 = dti
      dt  = dti
      if (dti.eq.0) call set_dt(dt0,w2,w1,0,m,n,eps,dti)
c
      nstepa = abs(nstep)
      if (nstep.lt.0) then
         dt  = -dt
         dt0 = -dt0
      endif
c
      dt  = dt0/10.
c
c     Take one step to get initial dt based upon eps
c
      dt2 = 0.5*dt
      t2  = t+dt2
      tt  = t+dt
c
      call rk4(w1,w ,t ,dt2,n)
      call rk4(w2,w1,t2,dt2,n)
c
      call rk4(w1,w ,t ,dt ,n)
C
      if (dti.eq.0) call set_dt(dt,w2,w1,1,m,n,eps,dti)
      call prsrrr('This is 2dt:$',dt,dt0,eps)
c     call prs ('Input new dt:$')
c     call rer(dt)
c
c
c     TIME STEPPING:
c
      t = 0.
      iopt = 1
      do k=1,nstepa
         call out_w (w,t,dtn,dto,k,nstep,iclr)
c
c        RK4 w/ variable dt
c
         dto = dt
         dt2 = 0.5*dto
         t2  = t+dt2
         tt  = t+dto
c
         call rk4(wh,w ,t ,dt2,n)
         call rk4(w2,wh,t2,dt2,n)
c
         call rk4(w1,w ,t ,dto,n)
c
         if (dti.eq.0) call set_dt(dt,w2,w1,k,m,n,eps,dti)
         dte = dt
         adt  = min(abs(dt),abs(dto)*1.1)
         adt  = max(abs(dt),abs(dto)/10.)
         if (dt.lt.0) adt = -adt
         dt = adt
c        adt  = min(abs(dt),abs(dt0))
c        adt  = max(abs(dt),abs(dt0)/(25*abs(iostep)))
c
c        Save current w-vector for "close_w" routine.
         call copy(wo,w,n)
c
c        Choose updating option
c
         if (iopt.eq.1) then
c           Take "real" RK4 step, using computed "dt"
            call rk4 (w1,w ,t ,dt ,n)
            call copy(w,w1,n)
            t   = t+dt
            dtn = dt
         elseif (iopt.eq.2) then
c           Use the dt/2 value computed above.
            call copy(w,w2,n)
            t   = tt
            dtn = dto
         elseif (iopt.eq.3) then
c           Use the dt & dt/2 values computed above w/ Richardson ext.
            c2 = 16./15.
            c1 = -1./15.
            call add3s12(w,w1,c1,w2,c2,n)
            t   = tt
            dtn = dto
         elseif (iopt.eq.4) then
c           Use the dt/2 value if dt < dt*/2
            if (abs(dt2).lt.abs(dt).and.abs(dt).lt.abs(dto)) then
c              Use the dt/2 value computed above.
               call copy(w,wh,n)
               dtn = dt2
            else
c              Take "real" RK4 step, using computed "dt"
               call rk4 (w1,w ,t ,dt ,n)
               call copy(w,w1,n)
               dtn = dt
            endif
         elseif (iopt.eq.5) then
c           Use the two dt/2 values if dt < dt*/2
            if (abs(dt2).lt.abs(dt).and.abs(dt).lt.abs(dto)) then
c              Use the dt/2 value computed above.
               call copy(w,w2,n)
               dtn = dto
            else
c              Take "real" RK4 step, using computed "dt"
               call rk4 (w1,w ,t ,dt ,n)
               call copy(w,w1,n)
               dtn = dt
            endif
         endif
c
c
         tnew = t+dtn
         if_closed = if_close (w,w0,tnew,dtn,k,nstep,iflast)
         if (if_closed) then
            call out_w  (w0,t,dtn,dto,-1,k,iclr)
            return
c           call close_w(w1,w,dtn,wo,to,k,nstep,n)
c           call copy(w,w1,n)
c           t = t+dtn
c           goto 1000
         endif
         t  = t+dtn
         dt = dtn
c
c        Check current position for possible termination
c        (i.e., if point is not inside any subdomain).
c
         call findre(ie,r,w,rminmax,ierr)
         if (rminmax.ge.1.01) goto 1001
c
      enddo
 1000 continue
      call out_w     (w,t,dtn,dto,-1,k,iclr)
 1001 continue
c
      return
      end
c-----------------------------------------------------------------------
      subroutine  close_w(wc,w1,dt1,wo,to,k,nsteps,n)
c
c     Use one secant step to find dt_c which will close the loop
c
      real wc(4),w1(4),wo(4)
c
      theta1 = atan2(w1(2),w1(1))
c
      dt2 = dt1/2.
      call rk4 (wc,wo,to,dt2,n)
      theta2 = atan2(wc(2),wc(1))
c
c     Secant extrapolation
c
      do k=1,4
         dt3 = dt2 - theta2*(dt1-dt2)/(theta1-theta2)
         call rk4(wc,wo,to,dt3,n)
         theta3 = atan2(wc(2),wc(1))
c        write(6,*) theta1,theta2,theta3,'close'
         dt1 = dt2
         dt2 = dt3
         theta1 = theta2
         theta2 = theta3
      enddo
c
c     write(6,*) theta1,theta2,thetac,'close'
c
      return
      end
c-----------------------------------------------------------------------
      function if_close(w,w0,t,dt,i,nsteps,iflast)
c
c     Check to see if radius and distance from start point are decreasing
c
      logical if_close,iflast
      real w(4),w0(4),t,dt
      real w10,w20,dmax
      save w10,w20,dmax
c
      eps = .001
c
      if (i.eq.1.and..not.iflast) then
         w10 = w0(1)
         w20 = w0(2)
         dmax = 0.
      endif
c
      d = (w(1)-w10)**2 + (w(2)-w20)**2
c
      if (dmax.gt.0  .and. (d/dmax).lt.eps) then
         if_close = .true.
         return
      endif
c
      if_close = .false.
      dmax = max(dmax,d)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine find_y_flag(iemin)
c
c     Find an element with minimal "y" value
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      n    = nx*ny*nz*nel
      nxyz = nx*ny*nz
c
      ymin = glmax(yp,n)
c
      l = 1
      do ie=1,nel
         yy = vlmin(yp(l),nxyz)
         if (yy.lt.ymin) then
            ymin  = yy
            iemin = ie
         endif
         l = l + nxyz
      enddo
c
c     write(6,*) 'this is iemin,yy:',iemin,yy
c
      return
      end
c-----------------------------------------------------------------------
      subroutine find_vertex(ix,iy,iz,e,xx,x,y,z,nel,nx,ny,nz,if3d)
c
      real xx(3)
      real x(nx,ny,nz,nel)
      real y(nx,ny,nz,nel)
      real z(nx,ny,nz,nel)
      logical if3d
      integer e
c
      dmin = 1.e20
      if (if3d) then
         do k=1,nz,nz-1
         do j=1,ny,ny-1
         do i=1,nx,nx-1
            d = (xx(1)-x(i,j,k,e))**2 
     $        + (xx(2)-y(i,j,k,e))**2
     $        + (xx(3)-z(i,j,k,e))**2
            if (d.lt.dmin) then
               ix = i
               iy = j
               iz = k
               dmin = d
            endif
         enddo
         enddo
         enddo
      else
         iz=1
         k =1
         do j=1,ny,ny-1
         do i=1,nx,nx-1
            d = (xx(1)-x(i,j,k,e))**2 
     $        + (xx(2)-y(i,j,k,e))**2
            if (d.lt.dmin) then
               ix = i
               iy = j
               dmin = d
            endif
         enddo
         enddo
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine find_vertexa(ix,iy,iz,ie,xx,x,y,z,nel,nx,ny,nz,if3d)
c
      real xx(3)
      real x(nx,ny,nz,nel)
      real y(nx,ny,nz,nel)
      real z(nx,ny,nz,nel)
      logical if3d
      integer e
c
      dmin = 1.e20
      if (if3d) then
         do e=1,nel
            do k=1,nz
            do j=1,ny
            do i=1,nx
               d = (xx(1)-x(i,j,k,e))**2 
     $           + (xx(2)-y(i,j,k,e))**2
     $           + (xx(3)-z(i,j,k,e))**2
               if (d.lt.dmin) then
                  ix = i
                  iy = j
                  iz = k
                  ie = e
                  dmin = d
               endif
            enddo
            enddo
            enddo
         enddo
      else
         iz=1
         k =1
         do e=1,nel
            do j=1,ny
            do i=1,nx
               d = (xx(1)-x(i,j,k,e))**2 
     $           + (xx(2)-y(i,j,k,e))**2
               if (d.lt.dmin) then
                  ix = i
                  iy = j
                  ie = e
                  dmin = d
               endif
            enddo
            enddo
         enddo
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine set_curve(ifcurv)
C
C     Find (r,s,t) given (x,y,z)
C
      INCLUDE 'basics.inc'
      logical ifcurv(nelm)
c
      do ie=1,nelm
         ifcurv(ie) = .false.
      enddo
c
      do ie=1,nelm
         do i=1,8
            if (ccurve(i,ie).ne.' ') ifcurv(ie)=.true.
         enddo
      enddo
c
c     Quick check for artery problems
c
      nfaces = 2*ndim
      if (ifheat) ifld=2
      if (ifflow) ifld=1
c
      if (ifxyo) then
         do ie=1,nelm
         do jf=1,nfaces
            if (cbc(jf,ie,ifld).eq.'W  ') ifcurv(ie)=.true.
         enddo
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine findr_new(rrl,xxl,ie,iguess,ierr)
C
C     Find (r,s,t) given (x,y,z)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      DIMENSION RRL(3),XXL(3)
      REAL JAC
c
      real rrd(3),drvec(3),dxvec(3),dxdr_mat(3,3)
c
      logical ifcurv(nelm)
      save    ifcurv

      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld=1
         call set_curve(ifcurv)
      endif
c
      if (ifcurv(ie)) then
c        call findr_full(rrl,xxl,ie,iguess,ierr)
         return
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine pf_read(v,n)
      real v(1)
c
      read(8) (v(k),k=1,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine pf_write(v,n)
      real v(1)
c
      write(8) (v(k),k=1,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine findre(ie_min,rl,xl,rminmax,ierr)
C
C     Find (ie,rl) pair which minimizes the distance from x(r,ie) to xl
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      integer ie_last
      save    ie_last
      data    ie_last /0/
c
      real    r_last(3)
      save    r_last
      data    r_last /0.,0.,0./
c
      real rl(3),xl(3)
C
c     quick check w/ last element
c
      ie_min = 0
c
c     write(6,1) 'iemin0',ie_last,r_last(1),r_last(2),r_last(3)
c    $            ,xl(1),xl(2),xl(3)
    1 format(a6,i9,2(2x,3f13.8))
c
c
      if (1.le.ie_last .and. ie_last.le.nel
     $   .AND. XBMIN(ie_last).LE.XL(1) .AND. XL(1).LE.XBMAX(ie_last)
     $   .AND. YBMIN(ie_last).LE.XL(2) .AND. XL(2).LE.YBMAX(ie_last)
     $   .AND. ZBMIN(ie_last).LE.XL(3) .AND. XL(3).LE.ZBMAX(ie_last) )
     $   THEN
c
         call findr(r_last,XL,IE_LAST,1,ierr)
         rmax = vlamax(r_last,ndim)
c        write(6,1) 'iemin1',ie_min,r_last(1),r_last(2),r_last(3)
c    $               ,xl(1),xl(2),xl(3)
         if (rmax.le.1.0) then
            rl(1) = r_last(1)
            rl(2) = r_last(2)
            rl(3) = r_last(3)
            rminmax = rmax
            ie_min = ie_last
c           write(6,1) 'iemin:',ie_min
c    $                 ,rl(1),rl(2),rl(3),xl(1),xl(2),xl(3)
            ierr = 0
            return
         endif
      endif
C
      rminmax = 1.e20
      rl1_min  = 5.0
      rl2_min  = 5.0
      rl3_min  = 5.0
      DO 2000 IE=1,NEL
         IF(   XBMIN(IE).LE.XL(1) .AND. XL(1).LE.XBMAX(IE)
     $   .AND. YBMIN(IE).LE.XL(2) .AND. XL(2).LE.YBMAX(IE)
     $   .AND. ZBMIN(IE).LE.XL(3) .AND. XL(3).LE.ZBMAX(IE) )
     $   THEN
C
C           Potentially in this element, check to be sure.
C
            call findr(RL,XL,IE,0,ierf)
            rmax = vlamax(rl,ndim)
c
            if ( rmax .le. rminmax ) then
               ie_min    = ie
               rminmax   = rmax
               rl1_min   = rl(1)
               rl2_min   = rl(2)
               rl3_min   = rl(3)
               ierr      = ierf
            endif
         ENDIF
c
c        NOTE:  At present, nekton will not accept mouse input
c        unless the geometry is ***not*** rotated from default position.
c        pff 6/21/00
c
c
c
c        write(6,1) 'ie   p',ie_min,rl(1),rl(2),rl(3)
c    $               ,xl(1),xl(2),xl(3)
c        write(6,1) 'xbmin ',ie     
c    $               ,xbmin(ie),xbmax(ie)
c    $               ,ybmin(ie),ybmax(ie)
c    $               ,zbmin(ie),zbmax(ie)
 2000 CONTINUE
c
      rl(1) = rl1_min
      rl(2) = rl2_min
      rl(3) = rl3_min
c
      r_last(1) = rl1_min
      r_last(2) = rl2_min
      r_last(3) = rl3_min
      ie_last   = ie_min
c
c     write(6,1) 'iemin2',ie_min,rl(1),rl(2),rl(3),xl(1),xl(2),xl(3)
c
      if (ierr.ne.0) then
         ie_min  = 1
         ie_last = 0
      endif
      return
      END
c-----------------------------------------------------------------------
      subroutine findr(rrl,xxl,ie,iguess,ierr)
c     subroutine findr_full(rrl,xxl,ie,iguess,ierr)
C
C     Find (r,s,t) given (x,y,z)
c
      common /findri/ ixyzmin,ixyzemn,iexyzmn  ! location of closest point
      common /findrr/ dxyzmin
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      DIMENSION RRL(3),XXL(3)
      REAL JAC
c
      real rrd(3),drvec(3),dxvec(3),dxdr_mat(3,3)
c
      CHARACTER*3 CNUMB3
      CHARACTER*2 CNUMBR
      EQUIVALENCE (CNUMBR,CNUMB3)
      INTEGER IEOLD
      SAVE    IEOLD
      DATA    IEOLD /0/
C
      ipass = 0
    1 continue
      ipass = ipass+1
c
      NXY =NX*NY
      nxyz=nz*nxy
      ieoff=nxyz*(ie-1)+1
      RR1=9.0
      RR2=9.0
      RR3=9.0
      dstmin = 1.e20
c
      dxchar = xbmax(ie) - xbmin(ie)
      dychar = ybmax(ie) - ybmin(ie)
      dzchar = 0.
      if (if3d) dzchar = zbmax(ie) - zbmin(ie)
      eps    = ((dxchar + dychar + dzchar)*2.e-6)**2
c
C
C        Initialze guess.
C
      if (iguess.eq.0 .or.ipass.gt.1) then
         ii = ieoff-1
         if (if3d) then
            do iz = 1,nz
            do iy = 1,ny
            do ix = 1,nx
               ii = ii+1
               DIST2 = (XXL(1)-XP(ii))**2
     $               + (XXL(2)-YP(ii))**2
     $               + (XXL(3)-ZP(ii))**2
               if (dist2.lt.dxyzmin) then
                  dxyzmin = dist2
                  ixyzemn = ii
                  ixyzmin = ii-ieoff+1
                  iexyzmn = ie
c                 write(6,*) dist2,ii,ixyzmin,ie,eps,' dist2,ii,ie'
               endif
               IF (DIST2.LT.eps) THEN
                  RRL(1) = ZPTS(IX)
                  RRL(2) = ZPTS(IY)
                  RRL(3) = ZPTS(IZ)
                  ierr = 0
               elseIF (DIST2.LT.DSTMIN) THEN
                  DSTMIN = DIST2
                  RRL(1) = ZPTS(IX)
                  RRL(2) = ZPTS(IY)
                  RRL(3) = ZPTS(IZ)
              ENDIF
            enddo
            enddo
            enddo
         else
c        2D
            do iy = 1,ny
            do ix = 1,nx
               ii = ii+1
               DIST2 = (XXL(1)-XP(ii))**2
     $               + (XXL(2)-YP(ii))**2
               if (dist2.lt.dxyzmin) then
                  dxyzmin = dist2
                  ixyzmin = ii
                  iexyzmn = ie
               endif
               IF (DIST2.LT.DSTMIN) THEN
                  DSTMIN = DIST2
                  RRL(1) = ZPTS(IX)
                  RRL(2) = ZPTS(IY)
                  RRL(3) = 0.
               ENDIF
            enddo
            enddo
         endif
c
         ieold=ie
         if (dstmin.lt.eps) then
c           write(6,*) 'dstmin?',dstmin,eps
            ierr=0
            return
         endif
c
c        Check to see if the minimum distance is at all reasonable
c        before proceeding further
c
         dstmin_char = 2.*(dxchar+dychar+dzchar)/NX
         dstmin      = sqrt(dstmin)
         if (dstmin.gt.dstmin_char) then
c           write(6,*) 'distmin:',dstmin,dstmin_char
c           write(6,*) 'distminr',(rrl(k),k=1,3)
c           write(6,*) 'distminx',(xxl(k),k=1,3)
            RRL(1)=3.0
            RRL(2)=dstmin
            RRL(3)=dstmin_char
            ierr = 1
            return
         endif
      endif
c
      IF (.NOT.IF3D) RRL(3)=0.0
      r_min_1 = rrl(1)
      r_min_2 = rrl(2)
      r_min_3 = rrl(3)
C
C     Set tolerance according to given element size  (actually... position)
C
      tollg2 = abs(xbmax(ie))+abs(xbmin(ie))
     $       + abs(ybmax(ie))+abs(ybmin(ie))
      if (if3d) tollg2 = tollg2 + abs(zbmax(ie))+abs(zbmin(ie))
c
      tollg  = 1.0e-5
c     if (if3d) tollg = 1.e-4
c
      tollg2 = (tollg2*tollg)**2
C
C============================================
C     Begin Newton-Raphson search
C============================================
C
      eps2    = 1.e+10
      eps2min = eps2
      MXNWT = 20
      DO 1000 INWT = 1,MXNWT
         CALL EVALSC( XI , XP(IEOFF) , RRL , 1 )
         CALL EVALSC( YI , YP(IEOFF) , RRL , 0 )
         IF (IF3D) THEN
            CALL EVALSC( ZI , ZP(IEOFF) , RRL , 0 )
         ELSE
            ZI=0.0
         ENDIF
C
C
C        Convergence check
C
         DX = XXL(1)-XI
         DY = XXL(2)-YI
         DZ = XXL(3)-ZI
         epso = eps2
         EPS2 = DX**2 + DY**2 + DZ**2
C
c
c        if (ipass.eq.2) 
c        write(6,6) 'nt',ie,inwt,eps2,tollg2
c    $                  ,rrl(1),rrl(2),rrl(3)
c    $                  ,xi,yi,zi
c    $                  ,xxl(1),xxl(2),xxl(3)
c   6    format(a2,2i3,2e11.3,3(1x,3f10.7))
c        if (ipass.eq.2) 
c        write(6,*)    'inwt',INWT,mxnwt,ipass,iguess
c
c
C        Convergence check
         IF (EPS2.LT.TOLLG2.and.inwt.gt.1) THEN
c           write(6,*) 'eps2,tol:',eps2,tollg2,xxl(1),xxl(2),xxl(3)
c    $                 ,rrl(1),rrl(2),rrl(3),ie,inwt
            ierr = 0
            return
         ENDIF
         if (eps2.lt.eps2min) then
            eps2min = eps2
            r_min_1 = rrl(1)
            r_min_2 = rrl(2)
            r_min_3 = rrl(3)
         endif
C
C        Update RRL
         if (inwt.lt.14.or.mod(inwt,3).eq.0.or.mod(inwt,3).eq.1) then
            CALL EVALSC( DRDX , RXM1(IEOFF) , RRL , 0 )
            CALL EVALSC( DRDY , RYM1(IEOFF) , RRL , 0 )
            CALL EVALSC( DSDX , SXM1(IEOFF) , RRL , 0 )
            CALL EVALSC( DSDY , SYM1(IEOFF) , RRL , 0 )
            CALL EVALSC( JAC  ,JACM1(IEOFF) , RRL , 0 )
C           Kludge for axisymmetric case w/vanishing Jacobian
            IF (JAC.EQ.0.0) JAC=1.0
c
            IF (IF3D) THEN
               CALL EVALSC( DRDZ , RZM1(IEOFF) , RRL , 0 )
               CALL EVALSC( DSDZ , SZM1(IEOFF) , RRL , 0 )
               CALL EVALSC( DTDX , TXM1(IEOFF) , RRL , 0 )
               CALL EVALSC( DTDY , TYM1(IEOFF) , RRL , 0 )
               CALL EVALSC( DTDZ , TZM1(IEOFF) , RRL , 0 )
               dr = ( DRDX*DX + DRDY*DY + DRDZ*DZ ) / JAC
               ds = ( DSDX*DX + DSDY*DY + DSDZ*DZ ) / JAC
               dq = ( DTDX*DX + DTDY*DY + DTDZ*DZ ) / JAC
            ELSE
               dr = ( DRDX*DX + DRDY*DY ) / JAC
               ds = ( DSDX*DX + DSDY*DY ) / JAC
               dq = 0.0
            ENDIF
         else
c
c           try finite differences
c
            if (if3d) then
               do k=1,3
                  call copy(rrd,rrl,3)
c                 rrd(k) = rrl(k)*.9995
                  rrd(k) = rrl(k)*.9
                  if (rrd(k).eq.0) rrd(k) = .01
                  CALL EVALSC( xd , XP(IEOFF) , RRD , 0 )
                  CALL EVALSC( yd , YP(IEOFF) , RRD , 0 )
                  CALL EVALSC( zd , ZP(IEOFF) , RRD , 0 )
                  h = rrd(k) - rrl(k)
                  dxdr_mat(1,k) = (xd-xxl(1))/h
                  dxdr_mat(2,k) = (yd-xxl(2))/h
                  dxdr_mat(3,k) = (zd-xxl(3))/h
               enddo
c
               dxvec(1) = dx
               dxvec(2) = dy
               dxvec(3) = dz
c
c              call outmat(dxdr_mat,3,3,'X_R')
               call factor_solve(drvec,dxdr_mat,3,dxvec,3)
               dr = drvec(1)
               ds = drvec(2)
               dt = drvec(3)
c
            else
               do k=1,2
                  call copy(rrd,rrl,2)
                  rrd(k) = rrl(k)*.9995
                  if (rrd(k).eq.0) rrd(k) = .0001
                  CALL EVALSC( xd , XP(IEOFF) , RRD , 0 )
                  CALL EVALSC( yd , YP(IEOFF) , RRD , 0 )
                  h = rrd(k) - rrl(k)
                  dxdr_mat(1,k) = (xd-xxl(1))/h
                  dxdr_mat(2,k) = (yd-xxl(2))/h
               enddo
c
               dxvec(1) = dx
               dxvec(2) = dy
c
               call factor_solve(drvec,dxdr_mat,3,dxvec,2)
               dr = drvec(1)
               ds = drvec(2)
            endif
         endif
c
         relax = 1.0
         if (inwt.gt.5.and.mod(inwt,2).eq.0) relax = 0.5
c
c        write(6,*) 'rrl1:',(rrl(kk),kk=1,3)
         RRL(1) = RRL(1) + dr*relax
         RRL(2) = RRL(2) + ds*relax
         RRL(3) = RRL(3) + dq*relax
c        write(6,*) 'rrl2:',(rrl(kk),kk=1,3)
         if (inwt.le.10) then
c          do irl=1,3
           irl = mod1(inwt,3)
             if (1.0.le.rrl(irl).and.rrl(irl).lt.1.005)  rrl(irl)= 1.0
             if (-1.005.le.rrl(irl).and.rrl(irl).lt.-1.) rrl(irl)=-1.0
c          enddo
         endif
c        write(6,*) 'rrl3:',(rrl(kk),kk=1,3)
c
C        If RRL has strayed too far, exit to avoid floating point overflow.
         IF (ABS(RRL(1)).GT.1.5  .OR.
     $       ABS(RRL(2)).GT.1.5  .OR.
     $       ABS(RRL(3)).GT.1.5) THEN
             goto 1001
c            ierr = 1
c            return
         ENDIF
C
C        If RRL is outside of element, but reasonable, try to fix it.
C
c        IF (ABS(RRL(1)).GT.1.25) RRL(1)=SIGN(1.0,RRL(1))
c        IF (ABS(RRL(2)).GT.1.25) RRL(2)=SIGN(1.0,RRL(2))
c        IF (ABS(RRL(3)).GT.1.25) RRL(3)=SIGN(1.0,RRL(3))
C
 1000 CONTINUE
 1001 CONTINUE
c
c     See if we can improve upon our initial guess with min search
c
      if (ipass.eq.1.and.iguess.eq.1) goto 1
C
C     If we've reached the end of Newton iteration, set RRL to best guess
c
c        write(6,6) 'nx',ie,inwt,eps2,tollg2
c    $                  ,rrl(1),rrl(2),rrl(3)
c    $                  ,xi,yi,zi
c    $                  ,xxl(1),xxl(2),xxl(3)
c
      RRL(1)=r_min_1
      RRL(2)=r_min_2
      RRL(3)=r_min_3
c
      ierr = 1
      return
      END
c-----------------------------------------------------------------------
      subroutine values
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      CHARACTER FILE*40
      CHARACTER*1 FILE1(40)
      EQUIVALENCE (FILE,FILE1)
      DATA FILE /'                                        '/
      REAL XXSCR(3)
c
      integer isel
      save    isel
      data    isel /-1/
c
      character*3 c3
C
      NDIM = 2
      IF (IF3D) NDIM = 3
C
      if (ifauto .and. isel .eq. 4) goto 7
         CALL PRS
     $('  Input: 0=File 1=Keyboard  2=Mouse 3=.fld file$')
         CALL REI(ISEL)
    7 continue
C
C     Read from file:
C
      write(6,*) 'ISEL = ',isel
      if (isel.eq.3) then
         write(6,*) 'call u_fld_values_ 2',isel
         call u_fld_values
         return
      endif
c
      IF (ISEL.EQ.0) THEN
         CALL PRS('  Input data file:$')
         CALL RES(FILE,40)
         if (file1(1).eq.'u') then
            write(6,*) 'Attempting to open unformatted file ',file
            open(unit=17,file=file ,form='unformatted',err=914)
            open(unit=89,file='v.dat',form='unformatted',err=914)
            goto 915
  914       continue
            write(6,*) 'Couldn''t open file ',file
            return
  915       continue
         else
            CALL OPENF(17,FILE,'OLD',1,ierr)
            open(unit=89,file='v.dat',form='formatted')
            IF (ierr.NE.0) THEN
               CALL PRS('Can''t find file'//FILE//'$')
               return
            ENDIF
         ENDIF
c
         xxscr(3) = 0. ! for ndim=2 case
         DO 110 IPT=1,9999999
            IF (MOD(IPT,20).EQ.0) CALL PRSI('Interpolating point$',IPT)
            READ(17,*,END=111,ERR=111) (XXSCR(I),I=1,NDIM)
            call interp(valw,xxscr(1),xxscr(2),xxscr(3),idum,work,ierr)
            write(89,*) valw
  110    continue
  111    continue
         close (unit=17)
         close (unit=89)
C 112    FORMAT(8E12.4)
  112    FORMAT(9G14.6)
      ENDIF
      IF (ISEL.EQ.1.OR.ISEL.EQ.2) THEN
  210    CONTINUE
C
C     MOUSE INPUT
C
      IF (ISEL.EQ.2.AND..NOT.IF3D) THEN
         CALL PRS('  Click with mouse point of interest: (right=end)$')
         CALL MOUSE(XXSCR(1),XXSCR(2),BUTTON)
         CALL COLOR(1)
         CALL XDOT(XXSCR(1),XXSCR(2))
         IF (BUTTON.EQ.'RIGHT') GOTO 220
      ELSE
C
C     KEYBOARD INPUT
C
         CALL PRS('  Input x,y,(z): (-13,-13=end)$')
         NDIM = 2
         IF (IF3D) NDIM=3
         IF(NDIM.EQ.2)CALL RERR (XXSCR(1),XXSCR(2)         )
         IF(NDIM.EQ.3)CALL RERRR(XXSCR(1),XXSCR(2),XXSCR(3))
         IF(XXSCR(1).EQ.-13.0 .AND. XXSCR(2).EQ.-13.0)GO TO 220
      ENDIF
C
C
C
C
C        Quick kludge for different function values. 2 Feb 1989 14:01:27
C
C
C
C
         IF(QUANTY .EQ.'STREAMFUNCTION' .OR.
     $      QUANTY .EQ.'VORTICITY'      .OR.
     $      QUANTY .EQ.'DIVERGENCE'     .OR.
     $      DERIV  .NE.'NO'                  ) THEN
            XPOINT = XXSCR(1)
            YPOINT = XXSCR(2)
            ZPOINT = XXSCR(3)
            CALL VALUE
         ENDIF
c
         IF (.NOT.IF3D) XXSCR(3)=0.0
         IF (IFFLOW) THEN
            CALL INTERP(UVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,U,ierr)
            CALL INTERP(VVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,V,ierr)
            IF (IF3D)
     $      CALL INTERP(WVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,W,ierr)
            CALL INTERP(PVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,P,ierr)
         ENDIF
         IF (IFHEAT) THEN
            INEWTX=1
            CALL INTERP(TVAL,XXSCR(1),XXSCR(2),XXSCR(3),IDUM,T,ierr)
            INEWTX=0
         ELSE
            TVAL=0.0
         ENDIF
         WRITE(S,212) (XXSCR(I),I=1,NDIM)
  212    FORMAT(1X,'x,y:',3G12.4)
         CALL PRS(S//'$')
         IF (.NOT.IF3D) WRITE(S,'(4e12.4)') UVAL,VVAL,PVAL,TVAL
         IF (     IF3D) WRITE(S,'(5e12.4)') UVAL,VVAL,WVAL,PVAL,TVAL
         CALL PRS(S//'$')
c        CALL PRSI('  Element number is:$',IENTRP )
c
         call find_vertexa(i1,j1,k1,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
         call find_vertex (i,j,k,ie,xxscr,xp,yp,zp,nel,nx,ny,nz,if3d)
         ijke = i+nx*(j-1 + ny*(k-1 + nz*(ie-1) ) )
         xq=xp(ijke)
         yq=yp(ijke)
         zq=zp(ijke)
         if (if3d) then
            CALL PRSIII('  Nearest (mod NX) vertex number is:$',i,j,k )
            CALL prsrrr('  corresponding to xyz:$',xq,yq,zq)
         else
            CALL PRSII ('  Nearest (mod NX) vertex number is:$',i,j   )
            CALL prsrr ('  corresponding to xyz:$',xq,yq   )
         endif
         write(6,109) i,j,k,ie
  109    format(' UVWPT    H',2x,4i5)
         CALL PRSI('  Element number is:$',ie )
c
         GOTO 210
  220    CONTINUE
      ENDIF
      return
      END
c-----------------------------------------------------------------------
