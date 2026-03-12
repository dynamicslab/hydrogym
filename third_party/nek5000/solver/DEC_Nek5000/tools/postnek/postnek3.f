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
c-----------------------------------------------------------------------
      subroutine pfiles
C
C     Draws multiple plots on the domain
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
      common /inputc/ input
c
      pi=4.0*atan(1.0)
c
      write(6,*) 'IFAUTO:',ifauto,nlines
c
      if (ifauto) then
c        we're in movie making mode
         call pfiles2
         return
      endif
C
C     Get end points
C
      CALL PRS
     $('Enter -1 return, 0 repeat, 1 file, 2 keybd, 3 auto-box :$')
c-----------------------------------------------------------------------
      CALL rei(input)
      if (input.eq.-1) return
      if (input.eq.0) then
         call pfiles2
         return
      endif
c
      if (input.eq.1) then
         call gtlines
         call get_prof_vars
         if (choice.eq.'PLOT') call pfiles2
         return
      endif
c
      if_auto_box = .false.
      if (input.eq.3) then
         if_auto_box = .true.
         call gtlines_box_z
         call get_prof_vars
         if (choice.eq.'PLOT') call pfiles2
         return
      endif
c
      call get_prof_vars
c
      if (ifmtpf) then
          open (unit=46,file='profile.out')
      else
          open (unit=46,file='profile.out',form='unformatted')
      endif
c
      nlines = 0
      DO 1000 IPROF=1,NLNMAX
C
         CALL PRS
     $   ('Enter location for plot (IDENTICAL points to EXIT).$')
         CALL PRS('Enter 1st Endpoint:$')
         CALL GETPT(XLIN1,YLIN1,ZLIN1,' ')
         CALL PRS('Enter 2nd Endpoint:$')
         CALL GETPT(XLIN2,YLIN2,ZLIN2,' ')
C
C        Exit Condtion: identical points for endpoints of line.
C
         IF (XLIN1.EQ.XLIN2 .AND.
     $       YLIN1.EQ.YLIN2 .AND.
     $       ZLIN1.EQ.ZLIN2 ) THEN
             GOTO 9000
         ENDIF
C
C        We use temporary LINE points so that the last location
C        plotted can be used in the normal profile routine w/o having
C        to call SET LOCATION.
C
         XLINE(1) = XLIN1
         YLINE(1) = YLIN1
         ZLINE(1) = ZLIN1
         XLINE(2) = XLIN2
         YLINE(2) = YLIN2
         ZLINE(2) = ZLIN2
         nlines=nlines+1
         xlines(1,nlines) = xlin1
         ylines(1,nlines) = ylin1
         zlines(1,nlines) = zlin1
         xlines(2,nlines) = xlin2
         ylines(2,nlines) = ylin2
         zlines(2,nlines) = zlin2
C
C        Set up scalar plot data
C
         IF (IPROF.EQ.1.OR.
     $      (.NOT.if3d.AND.QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N')) 
     $       call setwrk(.true.)
C
C        Plot on physical screen
C
         call plotph(46)
 1000 CONTINUE
 9000 CONTINUE
      close(46)
      return
      END
c-----------------------------------------------------------------------
      subroutine setwrk(ifdoin)
C
C     Set up WORK array according to current plotting state.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ WORK1(LXYZ),WORK2(LXYZ)
      INTEGER ICALLD,LDUMP
      DATA    ICALLD,LDUMP /0,-33/
c
      CHARACTER*5  derivo
      SAVE         derivo
      DATA         derivo /'     '/
c
      CHARACTER*20 LASTQT
      SAVE         LASTQT
      DATA         LASTQT /'                    '/
c
      logical ifdoin
      logical ifdoit
c
c     from getfld                    pff 2/28/98
      common /sfldwrk/ odumpw
      integer odumpw
c
      iwmin = 1
      iwmax = 1
C
      write(6,66) ifdoin,compon,deriv,quanty
   66 format('setwrk:',l4,2x,a1,2x,a1,2x,a40)
      IF (PLTMOD.EQ.'SURFACE') THEN
         call setwrks
         return
      ENDIF
      NTOT = NX*NY*NZ*NEL
c
c     Copy current work field into wrk2, for contour surface coloring
c
      NTOT = NX*NY*NZ*NEL
      call copy(wrk2,work,ntot)
C
      ifdoit = .false.
      if (ldump.ne.odumpw  .or.
     $   quanty.ne.lastqt  .or.
     $    deriv.ne.derivo  .or.
     $   ifdoin                 )    ifdoit = .true.
      if (QUANTY .EQ. 'USER FUNC')   ifdoit = .true.
c
c     If we've clobbered work arrays, recompute, unless quant=vortex
c
      if (quanty.ne.'VORTEX') then
         if (ifwkclobber)               ifdoit = .true.
      endif
      if (ifdoit) ifwkclobber = .false.
c
c
      write(6,*) 'ifnew_work:',ifdoit,ifdoin,lastqt,ldump,odumpw
c
      write(6,*) ifdoit,' quant: ',quanty
      if (ifdoit) then
        IF ( QUANTY .EQ.'STREAMFUNCTION' ) CALL STRFCT (WORK)
        IF ( QUANTY .EQ.'NORMAL VEC' )     CALL GENSTRV
c
        IF ( QUANTY .EQ.'STRESS VEC' ) then
           call comp_vorticity ! to get vorticity magnitude into work()
           call magnitude(work,wkv1,wkv2,wkv3)
           call genstrv
        endif
c
        IF ( QUANTY.EQ.'DIVERGENCE')THEN
C          WORK1=DV/DX ; WORK2=DU/DY
           NXYZ  = NX*NY*NZ
           IEOFF = 0
           DO 50 IEL=1,NEL
             CALL DUDXYZ(WORK1,U,RXM1,SXM1,TXM1,JACM1,IEL)
             CALL DUDXYZ(WORK2,V,RYM1,SYM1,TYM1,JACM1,IEL)
             IF (if3d) THEN
                CALL ADD2(WORK1,WORK2,NXYZ)
                CALL DUDXYZ(WORK2,W,RZM1,SZM1,TZM1,JACM1,IEL)
             ENDIF
             DO 45 I=1,NXYZ
                IX = IEOFF + I
                WORK(IX) = ( WORK1(I)+WORK2(I) )
   45        CONTINUE
             IEOFF = IEOFF + NXYZ
   50      CONTINUE
        ENDIF
        IF(QUANTY.EQ.'TEMPERATURE' )CALL COPY(WORK,T,NTOT)
        IF(QUANTY.EQ.'PRESSURE'    )CALL COPY(WORK,P,NTOT)
c
        IF(QUANTY.EQ.'TOTAL PRESSURE'    )THEN
           RHO2=PARAM(1)/2.0
           DO 75 I=1,NTOT
              WORK(I)=P(I)+RHO2*(U(I)**2+V(I)**2+W(I)**2)
   75      CONTINUE
        ENDIF
        IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')THEN
           IF(LPSDMP.EQ.1)CALL COPY(WORK,PSA,NTOT)
           IF(LPSDMP.EQ.2)CALL COPY(WORK,PSB,NTOT)
           IF(LPSDMP.EQ.3)CALL COPY(WORK,PSC,NTOT)
           IF(LPSDMP.EQ.4)CALL COPY(WORK,PSD,NTOT)
        ENDIF
c
        IF(QUANTY .EQ.'USER FUNC' ) THEN
c
          CALL USERF
c
        ENDIF
C
        IF(QUANTY .EQ.'Proc. Map' ) THEN
          CALL procmap
        ENDIF
        IF(QUANTY.EQ.'COORDINATE')THEN
           ICALLD=1
           CALL COPY(WKV1,XP,NTOT)
           CALL COPY(WKV2,YP,NTOT)
           IF (if3d) THEN
              CALL COPY (WKV3,ZP,NTOT)
           ELSE
              CALL RZERO(WKV3,NTOT)
           ENDIF
        ENDIF
        IF(QUANTY.EQ.'VELOCITY')THEN
           ICALLD=1
           CALL COPY(WKV1,U,NTOT)
           CALL COPY(WKV2,V,NTOT)
           IF (if3d) THEN
              CALL COPY (WKV3,W,NTOT)
           ELSE
              CALL RZERO(WKV3,NTOT)
           ENDIF
        ENDIF
        IF(QUANTY.EQ.'VORTICITY'.or.QUANTY.EQ.'V.W') call comp_vorticity
        IF(QUANTY.EQ.'VORTEX') call new_vortex
        IF(QUANTY.EQ.'FILTER') call filter_fld
c
c       !   These are now set in postnek
c       IF(QUANTY.EQ.'Linear Combination') call lin_combo
c       IF(QUANTY.EQ.'rms-avg') call rms_avg
c
        IF(QUANTY.EQ.'V.W')THEN
          ntot = nx*ny*nz*nel
          call    col3(work,wkv1,u,ntot)
          call addcol3(work,wkv2,v,ntot)
          call addcol3(work,wkv3,w,ntot)
          call copy   (wkv1,u,ntot)
          call copy   (wkv2,v,ntot)
          call copy   (wkv3,w,ntot)
        endif
C_______________________________________________________________________
C       Derivatives............................................
C-----------------------------------------------------------------------
        NXYZ  = NX*NY*NZ
        write(s,99) deriv,nxyz
   99   format(2x,'derivative:',a4,' (',i8,')$')
        call prs(s)
c
        if (deriv.ne.'NO') then
           do ie=1,nel
               IF(DERIV.EQ.'D/DX')
     $         CALL DUDXYZ(WORK1,WORK,RXM1,SXM1,TXM1,JACM1,ie)
               IF(DERIV.EQ.'D/DY')
     $         CALL DUDXYZ(WORK1,WORK,RYM1,SYM1,TYM1,JACM1,ie)
               IF(DERIV.EQ.'D/DZ')
     $         CALL DUDXYZ(WORK1,WORK,RZM1,SZM1,TZM1,JACM1,ie)
c
               IEOFF = NXYZ*(ie-1)+1
               IF(DERIV.EQ.'D/DX'.OR.DERIV.EQ.'D/DY'.OR.DERIV.EQ.'D/DZ')
     $         CALL COPY(WORK(IEOFF),WORK1,NXYZ)
c
               IF(DERIV.EQ.'GRAD') THEN
                  CALL DUDXYZ(WKV1(ieoff),WORK,RXM1,SXM1,TXM1,JACM1,ie)
                  CALL DUDXYZ(WKV2(ieoff),WORK,RYM1,SYM1,TYM1,JACM1,ie)
                  CALL DUDXYZ(WKV3(ieoff),WORK,RZM1,SZM1,TZM1,JACM1,ie)
               ENDIF
           enddo
           if (ifdssum) then
              call dsavg(wkv1,work)
              call dsavg(wkv2,work)
              call dsavg(wkv3,work)
           endif
           IF (DERIV.EQ.'GRAD') THEN
              if (if3d) then
                 do i=1,ntot
                    work(i) = sqrt(wkv1(i)*wkv1(i)
     $                      +      wkv2(i)*wkv2(i)
     $                      +      wkv3(i)*wkv3(i))
                 enddo
              else
                 do i=1,ntot
                    work(i) = sqrt(wkv1(i)*wkv1(i)
     $                      +      wkv2(i)*wkv2(i))
                 enddo
              endif
           endif
        endif
      endif
C
      IF (QUANTY.EQ.'VELOCITY'             .OR.
     $   (QUANTY.EQ.'VORTICITY'.AND.if3d)  .OR.
     $     DERIV.EQ.'GRAD'              )  THEN
C
         if (deriv.eq.'NO' .or. deriv.eq.'GRAD') then
c
C           Compute min speed and max speed for all choices
            SPMIN= 10.0E10
            SPMAX=-10.0E10
            DO 10 I=1,NTOT
               WORK(I)=SQRT(WKV1(I)**2+WKV2(I)**2+WKV3(I)**2)
               SPMIN=MIN(SPMIN,WORK(I))
               SPMAX=MAX(SPMAX,WORK(I))
   10       CONTINUE
            UVMAX=SPMAX
C
            IF (COMPON.EQ.'X') CALL COPY(WORK,WKV1,NTOT)
            IF (COMPON.EQ.'Y') CALL COPY(WORK,WKV2,NTOT)
            IF (COMPON.EQ.'Z') CALL COPY(WORK,WKV3,NTOT)
            if (compon.eq.'H') then
              call col3(work,u,wkv1,ntot)
              call addcol3(work,v,wkv2,ntot)
              if (if3d) call addcol3(work,w,wkv3,ntot)
            endif
            IF (COMPON.EQ.'N') THEN
C           Normal to specified line
                AMPLIN = SQRT(  (XLINE(1)-XLINE(2))**2
     $                 +        (YLINE(1)-YLINE(2))**2
     $                 +        (ZLINE(1)-ZLINE(2))**2 )
                IF (AMPLIN.NE.0.0) THEN
                   XTWDL = (XLINE(2)-XLINE(1)) / AMPLIN
                   YTWDL = (YLINE(2)-YLINE(1)) / AMPLIN
                   ZTWDL = (ZLINE(2)-ZLINE(1)) / AMPLIN
                ELSE
                   CALL PRS
     $              ('Error: you must specify a line segment along $')
                   CALL PRS('which to plot profile before plotting.$')
                   return
                ENDIF
                IF (.NOT.if3d) THEN
c                  SPMIN =  10.0E10
c                  SPMAX = -10.0E10
                   DO 20 I=1,NTOT
                      WORK(I)=(-WKV1(I)*YTWDL+WKV2(I)*XTWDL)
c                     SPEED  =(WKV1(I)**2+WKV2(I)**2)
c                     SPMIN  = MIN(SPEED,SPMIN)
c                     SPMAX  = MAX(SPEED,SPMAX)
   20              CONTINUE
c                  SPMIN  = SQRT(SPMIN)
c                  SPMAX  = SQRT(SPMAX)
                ELSE
C                  Kludge: put the speed in here
                   DO 30 I=1,NTOT
                      WORK(I)=SQRT(WKV1(I)**2+WKV2(I)**2+WKV3(I)**2)
   30              CONTINUE
                ENDIF
            ENDIF
         else
C           Compute min speed and max speed for all choices
            SPMIN= 10.0E10
            SPMAX=-10.0E10
            DO 40 I=1,NTOT
               SPMIN=MIN(SPMIN,WORK(I))
               SPMAX=MAX(SPMAX,WORK(I))
   40       CONTINUE
            UVMAX=SPMAX
C
         endif
      ENDIF
C
C     Compute range for WORK array
C
      IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
         WKMAX = SPMAX
         WKMIN = SPMIN
         WKDLT = WKMAX-WKMIN
      ELSE
c
c        WKMAX = GLMAX(WORK,NTOT)
c        WKMIN = GLMIN(WORK,NTOT)
c

         iwmin = 1
         iwmax = 1
         wkmin = work(iwmin)
         wkmax = work(iwmax)

         do i=1,ntot
c           write(6,*) i,t(i),work(i),' ',quanty
            if (work(i).lt.wkmin) then
               iwmin = i
               wkmin = work(i)
            endif
            if (work(i).gt.wkmax) then
               iwmax = i
               wkmax = work(i)
            endif
         enddo
         WKDLT = WKMAX-WKMIN
         write(6,*) 'wkmax,delt:',quanty,wkmax,wkmin,wkdlt
      ENDIF
      WRITE(S,1000) WKMAX,WKMIN
 1000 FORMAT('WORK - MAX,MIN:',2g14.5,'$')
      CALL PRS(S)
C
      WRITE(S,1001) 'wkmin:',wkmin,xp(iwmin),yp(iwmin),zp(iwmin),iwmin
 1001 FORMAT(a6,1p4e12.4,i9,'$')
      CALL PRS(S)
c
      WRITE(S,1001) 'wkmax:',wkmax,xp(iwmax),yp(iwmax),zp(iwmax),iwmax
      CALL PRS(S)
C
      TTMIN=WKMIN
      TTMAX=WKMAX
      if (ifuscl) then
         TTMIN=usrMIN
         TTMAX=usrMAX
      endif
c
      derivo=deriv
      LASTQT=QUANTY
      LDUMP =odumpw
c
c     compute volume integral,  just for kicks
c
      call vol_int(dum1,vol1,work)
c
      return
      END
c-----------------------------------------------------------------------
      subroutine plotph(io)
C
C     Plot a profile in the physical domain
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common /inputc/ input
c
      character*1 dollar
      DIMENSION UL(3),VP(3)
      common  /ceval2/ reval(3)
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld = icalld + 1
         call interp(v1,xp,yp,zp,idum,work,ierr)
      endif
c
      dollar = '$'
c
C     White
      CALL COLOR(1)
c
      ukdlt = wkdlt
      IF(IFUSCL.AND.USRMIN.NE.USRMAX) ukdlt=(USRMAX-USRMIN)
C     For movies, use PLTSCL=10*UKDLT/XFACO
C     Otherwise, use PLTSCL=10*UKDLT/XFAC (Problem if XFAC different from YFAC)
c     PLTSCL = 10.* UKDLT/XFACO
      PLTSCL = 10.* UKDLT/XFAC
      SPAROW = SARROW/PLTSCL
c
      IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
         ifuprf = .true.
         ifvprf = .true.
         if (if3d) ifwprf = .true.
      ENDIF
C
C     Abcissa
C
      ABCLNG = (XLINE(2)-XLINE(1))**2
     $       + (YLINE(2)-YLINE(1))**2
     $       + (ZLINE(2)-ZLINE(1))**2
      if (abclng.gt.0) abclng = sqrt(abclng)
      dabc   = abclng/nptpro
c
      XCOS = (XLINE(2)-XLINE(1)) / ABCLNG
      YCOS = (YLINE(2)-YLINE(1)) / ABCLNG
      ZCOS = (ZLINE(2)-ZLINE(1)) / ABCLNG
      DXABC= XCOS*DABC
      DYABC= YCOS*DABC
      DZABC= ZCOS*DABC
      X0 = XLINE(1)
      Y0 = YLINE(1)
      Z0 = ZLINE(1)
c
      UL(1)=XLINE(2)-XLINE(1)
      UL(2)=YLINE(2)-YLINE(1)
      UL(3)=ZLINE(2)-ZLINE(1)
      CALL NORM3D(UL)
c
      call get_profile(x0,y0,z0,dxabc,dyabc,dzabc,dabc)
C
      IF (if3d) THEN
         if (input.ne.3) call move3s( xline(1),yline(1),zline(1) )
         if (input.ne.3) call draw3s( xline(2),yline(2),zline(2) )
C
         DO 1000 i=0,NPTPRO
            IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
               vp(1)=upr(i)
               vp(2)=vpr(i)
               vp(3)=wpr(i)
               costht=dotprod(ul,vp)
               vp1=vp(1)-costht*ul(1)
               vp2=vp(2)-costht*ul(2)
               vp3=vp(3)-costht*ul(3)
               dxord = vp1*sparow
               dyord = vp2*sparow
               dzord = vp3*sparow
               amp   = vp(1)**2 + vp(2)**2 + vp(3)**2
               if (amp.gt.0) amp = sqrt(amp)
c              write(io,46) x0,y0,z0,amp,(vp(k),k=1,3)!,(reval(k),k=1,3)
            else
               if (ifmean) then
                  wk1=v1p(i)
                  wk2=v2p(i)
c                 write(io,46) x0,y0,z0,wk1,wk2!,(reval(k),k=1,3)
                  AMP   = WK1
               elseif (ifmftp) then
                  wk1=v1p(i)
                  wk2=v2p(i)
                  wk3=v3p(i)
c                 write(io,46) x0,y0,z0,wk1,wk2,wk3!,(reval(k),k=1,3)
                  AMP   = WK1
               else
                  amp = wkp(i)
c                 write(6,*) ierr,amp,z0,' amp'
c                 write(io,46) x0,y0,z0,amp!,(reval(k),k=1,3)
               endif
               wkval = amp
               DXORD = -ZCOS*WKVAL*SPAROW
               DYORD =  0.0
               DZORD =  XCOS*WKVAL*SPAROW
            endif
c
c
            X1    = X0+DXORD
            Y1    = Y0+DYORD
            Z1    = Z0+DZORD
            IF (i.EQ.1.and.input.ne.3) CALL MOVE3S(X1,Y1,Z1)
            IF (i.GT.1.and.input.ne.3) CALL DRAW3S(X1,Y1,Z1)
            if (input.ne.3.and..not.ifauto) 
     $         write(6,*) ierr,amp,x1,y1,z1,' amp'
            X0    = X0 + DXABC
            Y0    = Y0 + DYABC
            Z0    = Z0 + DZABC
C
C           Gather statistics for this line
C
            IF (i.EQ.1) THEN
               AMPMIN=AMP
               AMPMAX=AMP
               AMPAVG=AMP
            ELSE
               AMPMIN=MIN(AMPMIN,AMP)
               AMPMAX=MAX(AMPMAX,AMP)
               AMPAVG=AMPAVG+AMP
            ENDIF
 1000    CONTINUE
         AMPAVG=AMPAVG/FLOAT(NPTPRO)
         WRITE(S,1001) AMPMIN,AMPMAX,AMPAVG,dollar
 1001    FORMAT(2X,'Min, Max, Avg:',3G12.4,a1)
         if (input.ne.3) CALL PRS(S)
C
C        If plotting velocity profiles, and NPTPRO>450, we'll plot arrows too.
C
         IF (QUANTY.EQ.'VELOCITY'.AND.NPTPRO.GT.150) THEN
            NARROW=5+MOD(NPTPRO,25)
            X0 = XLINE(1)
            Y0 = YLINE(1)
            Z0 = ZLINE(1)
            dabc=abclng/narrow
            DXABC= XCOS*DABC
            DYABC= YCOS*DABC
            DZABC= ZCOS*DABC
            DO 1500 IPT=0,NARROW
               IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
                  CALL INTERP(VP(1),X0,Y0,Z0,IDUM,U,ierr)
                  CALL INTERP(VP(2),X0,Y0,Z0,IDUM,V,ierr)
                  CALL INTERP(VP(3),X0,Y0,Z0,IDUM,W,ierr)
                  COSTHT=DOTPROD(UL,VP)
                  VP1=VP(1)-COSTHT*UL(1)
                  VP2=VP(2)-COSTHT*UL(2)
                  VP3=VP(3)-COSTHT*UL(3)
                  DXORD = VP1*SPAROW
                  DYORD = VP2*SPAROW
                  DZORD = VP3*SPAROW
               ELSE
                  CALL INTERP(WKVAL,X0,Y0,Z0,IDUM,WORK,ierr)
                  WKVAL = WKVAL
                  DXORD = -ZCOS*WKVAL*SPAROW
                  DYORD =  0.0
                  DZORD =  XCOS*WKVAL*SPAROW
               ENDIF
               if (input.ne.3) CALL ARROW4S(X0,Y0,Z0,DXORD,DYORD,DZORD)
               X0 = X0 + DXABC
               Y0 = Y0 + DYABC
               Z0 = Z0 + DZABC
 1500       CONTINUE
         ENDIF
      ELSE
c
c        2D
c
         if (input.ne.3) CALL MOVEC(XLINE(1),YLINE(1))
         if (input.ne.3) CALL DRAWC(XLINE(2),YLINE(2))
         UL(1)=XLINE(2)-XLINE(1)
         UL(2)=YLINE(2)-YLINE(1)
         UL(3)=0.
         CALL NORM3D(UL)
C
c        Output results
c
         do 1002 i=0,nptpro ! Note change in start from 1 to 0, pff 10/30/03
c
            x0 = xpt(i)
            y0 = ypt(i)
c
            IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
               vp(1) = upr(i)
               vp(2) = vpr(i)
               vp(3) = 0.
               COSTHT=DOTPROD(UL,VP)
               VP1=VP(1)-COSTHT*UL(1)
               VP2=VP(2)-COSTHT*UL(2)
               VP3=0.
               DXORD = VP1*SPAROW
               DYORD = VP2*SPAROW
               DZORD = 0.
               AMP   = VP(1)**2 + VP(2)**2
               if (amp.gt.0) AMP   = SQRT(AMP)
c              write(io,46) x0,y0,amp,(vp(k),k=1,2)!,(reval(k),k=1,3)
            elseif (ifuvtp) then       !  M.Paul: flow and tp profiles
               velx = upr(i)
               vely = vpr(i)
               pres = ppr(i)
               psi  = wkp(i)
               tper = v1p(i)
               amp   = tper
            else                      ! STANDARD CASE
               wkval = wkp(i)
               amp   = wkval
               dxord = -ycos*wkval*sparow
               dyord =  xcos*wkval*sparow
c              write(io,46) x0,y0,amp!,(reval(k),k=1,3)
            endif
            X1    = X0+DXORD
            Y1    = Y0+DYORD
            Z1    = 0.
            IF (i.EQ.1.and.input.ne.3) CALL MOVEC(X1,Y1)
            IF (i.GT.1.and.input.ne.3) CALL DRAWC(X1,Y1)
            X0    = X0 + DXABC
            Y0    = Y0 + DYABC
            Z0    = 0.
C
C           Gather statistics for this line
C
            IF (i.EQ.1) THEN
               AMPMIN=AMP
               AMPMAX=AMP
               AMPAVG=AMP
            ELSE
               AMPMIN=MIN(AMPMIN,AMP)
               AMPMAX=MAX(AMPMAX,AMP)
               AMPAVG=AMPAVG+AMP
            ENDIF
 1002    CONTINUE
         AMPAVG=AMPAVG/FLOAT(NPTPRO)
         WRITE(S,1001) AMPMIN,AMPMAX,AMPAVG,dollar
         CALL PRS(S)
C
C        If plotting velocity profiles, and NPTPRO>450, we'll plot arrows too.
C
         IF (QUANTY.EQ.'VELOCITY'.AND.NPTPRO.GT.150) THEN
            NARROW=5+MOD(NPTPRO,25)
            X0 = XLINE(1)
            Y0 = YLINE(1)
            Z0 = 0.
            dabc=abclng/narrow
            DXABC= XCOS*DABC
            DYABC= YCOS*DABC
            DZABC= 0.
            DO 1502 i=0,NARROW
               IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
                  CALL INTERP(VP(1),X0,Y0,Z0,IDUM,U,ierr)
                  CALL INTERP(VP(2),X0,Y0,Z0,IDUM,V,ierr)
                  VP(3)=0.
                  COSTHT=DOTPROD(UL,VP)
                  VP1=VP(1)-COSTHT*UL(1)
                  VP2=VP(2)-COSTHT*UL(2)
                  VP3=0.
                  DXORD = VP1*SPAROW
                  DYORD = VP2*SPAROW
                  DZORD = 0.
               ELSE
                  CALL INTERP(WKVAL,X0,Y0,Z0,IDUM,WORK,ierr)
                  WKVAL = WKVAL
                  DXORD = -YCOS*WKVAL*SPAROW
                  DYORD =  XCOS*WKVAL*SPAROW
                  DZORD =  0.
               ENDIF
               CALL ARROW4S(X0,Y0,Z0,DXORD,DYORD,DZORD)
               X0 = X0 + DXABC
               Y0 = Y0 + DYABC
               Z0 = 0.
 1502       CONTINUE
         ENDIF
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine drawel(iel)
C     IF ELEMENT NUMBER IS NEGATIVE, ERASE ELEMENT
      include 'basics.inc'
      real xx(3,9)

      CHARACTER STRING*6
      DIMENSION CSPACE(100),XCRVED(100),YCRVED(100)
C     Now, draw new elements for elements that were modified
      IIEL=IABS(IEL)
      npoint=10
      DO 8 I=1,NPOINT
         CSPACE(I)=(I-1.0)/(NPOINT-1.0)
8     CONTINUE
      IF(IEL.GT.0)THEN
C        Normal Draw, unless it is a conduction element
C           Draw conduction elements red
            call color(10)
            IF(IGROUP(IEL).LT.0)THEN
               call fillp(-14)
            ELSE IF(IGROUP(IEL).EQ.0)THEN
C              NORMAL
               call fillp(-15)
            ELSE IF(IGROUP(IEL).EQ.1)THEN
               call fillp(-2)
            ELSE IF(IGROUP(IEL).EQ.2)THEN
               CALL FILLP( -(IGROUP(IEL)+2) )
            ELSE
               CALL FILLP( -(IGROUP(IEL)+3) )
            ENDIF
      ELSE
C        fill black (i.e., erase)
         CALL COLOR(0)
         CALL FILLP(0)
      ENDIF
C     One more Kludge: if IEL is .GT. 10,000 then draw outline only
      IC=4
      IF(IFCEIL)IC=8
      IF(IEL.GT.10000)THEN
         IIEL=IIEL-10000
         CALL MOVEC(x(Iiel,IC),y(Iiel,IC))
      ELSE
         CALL BEGINB(x(Iiel,IC),y(Iiel,IC))
      ENDIF
      XCENTER=0.0
      YCENTER=0.0
      IF(IFCEIL)THEN
         ICBEG=5
         ICEND=8
      ELSE
         ICBEG=1
         ICEND=4
      ENDIF
      DO 6 IC=ICBEG,ICEND
          YCENTER=YCENTER+Y(IIEL,IC)/4.
          XCENTER=XCENTER+X(IIEL,IC)/4.
          IEDGE=IC-1
          IF(IC.EQ.1)IEDGE=4
          IF(IC.EQ.5)IEDGE=8
 
          IF(CCURVE(IEDGE,IIEL).EQ.' ')THEN
             CALL DRAWC(X(IIEL,IC),Y(IIEL,IC))
          ELSE
C            Draw curved side
             CALL GETPTS(NPOINT,CSPACE,IIEL,IEDGE,XCRVED,YCRVED)
             DO 118 I=1,NPOINT
                CALL DRAWC(XCRVED(I),YCRVED(I))
118          CONTINUE
          ENDIF
6     CONTINUE
      IF(IEL.LT.10000)CALL ENDP
C
      IF(IEL.GT.0)THEN
C        LABEL Element Center
         IF(if3d)     WRITE(STRING,'(I3,A1)')NUMAPT(IEL),LETAPT(IEL)
         IF(.NOT.if3d)WRITE(STRING,'(I3)')IEL
         IF(IFCEIL)WRITE(STRING(5:5),'(A1)')'C'
         STRING(6:6)='$'
         IF(CWRITE.NE.0.)CALL GWRITE(XCENTER-.055*XFAC,
     $   YCENTER-.02*YFAC,1.0,STRING)
      ENDIF
      call color(1)

      call rzero(xx,9)
      do j=5,8
         xx(1,1)=xx(1,1)+.25*x(iel,j)
         xx(2,1)=xx(2,1)+.25*y(iel,j)
         xx(3,1)=xx(3,1)+.25*z(iel,j)
      stop
      enddo
      do j=5,8
         xx(1,2)=max(xx(1,2),abs(xx(1,1)-x(iel,j)))
         xx(2,2)=max(xx(2,2),abs(xx(2,1)-y(iel,j)))
         xx(3,2)=max(xx(3,2),abs(xx(3,1)-z(iel,j)))
      enddo
c     do j=5,8
c        xxm=x(iel,j) + .12*(xx(1,1)-x(iel,j))**2 / xx(1,2)
c        yym=y(iel,j) + .12*(xx(2,1)-x(iel,j))**2 / xx(2,2)
c        zzm=z(iel,j) + .12*(xx(3,1)-x(iel,j))**2 / xx(3,2)
c        xm=xiso(xxm,yym,zzm)
c        ym=yiso(xxm,yym,zzm)
c        if (j.eq.5) call gwrite(xm,ym,1.0,'5$')
c        if (j.eq.6) call gwrite(xm,ym,1.0,'6$')
c        if (j.eq.7) call gwrite(xm,ym,1.0,'7$')
c        if (j.eq.8) call gwrite(xm,ym,1.0,'8$')
c     enddo

      return
      END
c-----------------------------------------------------------------------
      subroutine pfiles2
C
C     Draws multiple plots on the domain
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common /inputc/ input
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
C
C     For now, we wipe out the old redraw information
C
      CALL RESETDW
      IFSAVE=.TRUE.
      if (.not.ifauto) then
         if (ifmtpf) then
            open (unit=46,file='profile.out')
         else
            open (unit=46,file='profile.out',form='unformatted')
         endif
      endif
C
C     Get end points
C
      DO 1000 IPROF=1,NLNMAX
         XLIN1 = XLINES(1,IPROF)
         YLIN1 = YLINES(1,IPROF)
         ZLIN1 = ZLINES(1,IPROF)
         XLIN2 = XLINES(2,IPROF)
         YLIN2 = YLINES(2,IPROF)
         ZLIN2 = ZLINES(2,IPROF)
         write(6,*) iprof,' lin1:',xlin1,ylin1,zlin1
         write(6,*) iprof,' lin2:',xlin2,ylin2,zlin2
C 
C        Exit Condtion: identical points for endpoints of line.
C
         IF (XLIN1.EQ.XLIN2 .AND.
     $       YLIN1.EQ.YLIN2 .AND.
     $       ZLIN1.EQ.ZLIN2 ) THEN
             GOTO 9000
         ENDIF
C
C        We use temporary LINE points so that the last location
C        plotted can be used in the normal profile routine w/o having
C        to call SET LOCATION.
C
         XLINE(1) = XLIN1
         YLINE(1) = YLIN1
         ZLINE(1) = ZLIN1
         XLINE(2) = XLIN2
         YLINE(2) = YLIN2
         ZLINE(2) = ZLIN2
C
C        Set up scalar plot data
C
         IF (IPROF.EQ.1.OR.
     $      (.NOT.if3d.AND.QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N')) 
     $       call setwrk(.true.)
C
C        Plot on physical screen
C
         call plotph(46)
 1000 CONTINUE
 9000 CONTINUE
      if (.not.ifauto) close(46)
      return
      END
c-----------------------------------------------------------------------
      subroutine gtlines
C
C     Gets a set of lines
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
C
C     Get end points
C
      CALL PRS
     $   ('Enter file containing coordinate pairs:$')
      call blank(line,70)
      call res  (line,70)
      open (unit=28,file=line,status='old',err=9999)
c
      zlin1 = 0.
      zlin2 = 0.
      nlines=0
      do 1000 ilines=1,nlnmax
C
         if (if3d) then
            read (28,*,err=9000,end=9000) xlin1,ylin1,zlin1
            read (28,*,err=9000,end=9000) xlin2,ylin2,zlin2
         else
            read (28,*,err=9000,end=9000) xlin1,ylin1
            read (28,*,err=9000,end=9000) xlin2,ylin2
         endif
c
c        CALL PRS
c    $   ('Enter endpoints of line (IDENTICAL points to EXIT).$')
c        CALL PRS('Enter 1st Endpoint:$')
c        CALL GETPT(XLIN1,YLIN1,ZLIN1,' ')
c        CALL PRS('Enter 2nd Endpoint:$')
c        CALL GETPT(XLIN2,YLIN2,ZLIN2,' ')
C
C        Exit Condtion: identical points for endpoints of line.
C
         IF (XLIN1.EQ.XLIN2 .AND.
     $       YLIN1.EQ.YLIN2 .AND.
     $       ZLIN1.EQ.ZLIN2 ) THEN
             GOTO 9000
         ENDIF
C
C        We use temporary LINE points so that the last location
C        plotted can be used in the normal profile routine w/o having
C        to call SET LOCATION.
C
         XLINE(1) = XLIN1
         YLINE(1) = YLIN1
         ZLINE(1) = ZLIN1
         XLINE(2) = XLIN2
         YLINE(2) = YLIN2
         ZLINE(2) = ZLIN2
         nlines=nlines+1
         xlines(1,nlines) = xlin1
         ylines(1,nlines) = ylin1
         zlines(1,nlines) = zlin1
         xlines(2,nlines) = xlin2
         ylines(2,nlines) = ylin2
         zlines(2,nlines) = zlin2
 1000 CONTINUE
 9000 CONTINUE
      xlines(1,nlines+1) = 0.
      ylines(1,nlines+1) = 0.
      zlines(1,nlines+1) = 0.
      xlines(2,nlines+1) = 0.
      ylines(2,nlines+1) = 0.
      zlines(2,nlines+1) = 0.
      close (unit=28)
      return
 9999 CONTINUE
      call prs('unable to open file$')
      return
      END
c-----------------------------------------------------------------------
      subroutine arrow4s(x,y,z,dx,dy,dz)
      COMMON/SCALE/XFAC,YFAC,XZERO,YZERO
C     Plot a X pointed arrow
      CALL COLOR(12)
      IF(DX.EQ.0.0 .AND. DY.EQ.0.0 .AND.DZ.EQ.0.0) return
      CALL PENW(1)
      CALL COLOR(1)
      CALL MOVE3S(X,Y,Z)
      X1=X+DX
      Y1=Y+DY
      Z1=Z+DZ
      CALL DRAW3S(X1,Y1,Z1)
C     Wierd arrowheads
      CALL DRAW3S(X+0.85*DX-.1*DY ,Y+0.85*DY+0.1*DX*yfac/xfac,Z+DZ*0.85)
      CALL MOVE3S(X+DX,Y+DY,Z+DZ)
      CALL DRAW3S(X+0.85*DX+0.1*DY,Y+0.85*DY-0.1*DX*yfac/xfac,Z+DZ*0.85)
      IF(DX.EQ.0.0 .AND. DY.EQ.0.0) THEN
C        Kludge to get arrowhead on vertical arrow
         CALL MOVE3S(X,Y,Z+DZ)
         CALL DRAW3S(X,Y+0.1*Dz,Z+DZ*0.85)
         CALL MOVE3S(X,Y,Z+DZ)
         CALL DRAW3S(X,Y-0.1*Dz,Z+DZ*0.85)
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine draw3s(x3,y3,z3)
      INCLUDE 'devices.inc'
C
C     Draw and save in buffer for rapid redraw, if IFSAVE.
C
      IF (IFSAVE) THEN
         IDRCNT=IDRCNT+1
         IF (IDRCNT.GT.NDRMAX) IDRCNT=1
         XYZD(1,IDRCNT)=X3
         XYZD(2,IDRCNT)=Y3
         XYZD(3,IDRCNT)=Z3
         XYZD(4,IDRCNT)=1.0
         XYZD(5,IDRCNT)=LSTCLR
      ENDIF
C
      X=XISO(X3,Y3,Z3)
      Y=YISO(X3,Y3,Z3)
      CALL DRAWC(X,Y)
      return
      END
c-----------------------------------------------------------------------
      subroutine move3s(x3,y3,z3)
      INCLUDE 'devices.inc'
C
C     Move and save in buffer for rapid redraw, if IFSAVE.
C
      IF (IFSAVE) THEN
         IDRCNT=IDRCNT+1
         IF (IDRCNT.GT.NDRMAX) IDRCNT=1
         IDRAWD=MAX(IDRCNT,IDRAWD)
         XYZD(1,IDRCNT)=X3
         XYZD(2,IDRCNT)=Y3
         XYZD(3,IDRCNT)=Z3
         XYZD(4,IDRCNT)=0.0
         XYZD(5,IDRCNT)=LSTCLR
      ENDIF
      X=XISO(X3,Y3,Z3)
      Y=YISO(X3,Y3,Z3)
      CALL MOVEC(X,Y)
      return
      END
c-----------------------------------------------------------------------
      subroutine redraw
      INCLUDE 'devices.inc'
      common /iscdmp/ nsdump
C
      IF (NSDUMP.GT.0) THEN
         CALL SCRUDMP
      ELSE
C
C        Rapid redraw of saved move and draw commands.
C
         OPEN (unit=77,file='t.q',status='old',err=77)
C
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            call color(9)
            CALL MOVEC(X-.015,Y)
            CALL DRAWC(X+.015,Y)
            CALL MOVEC(X,Y-.015)
            CALL DRAWC(X,Y+.015)
C
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            CALL MOVEC(X-.015,Y)
            CALL DRAWC(X+.015,Y)
            CALL MOVEC(X,Y-.015)
            CALL DRAWC(X,Y+.015)
C
         DO 100 I=1,10000
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            ic = int(12.0*s)
            ic = mod(ic,12)+2
            if (s.eq.0) then
               CALL MOVEC(X,Y)
            else
               CALL COLOR(IC)
               CALL DRAWC(X,Y)
            endif
  100    CONTINUE
  101    CONTINUE
         close (unit=77)
   77    CONTINUE
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine resetdw
      INCLUDE 'devices.inc'
C
C     Reset the redraw counters, effectively eliminating the redraw information.
C
      IDRAWD=0
      IDRCNT=0
      return
      END
c-----------------------------------------------------------------------
      subroutine modwrk
C
C     Modify WORK array according to current plotting state.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ WORK1(LXYZ),WORK2(LXYZ)
      INTEGER ICALLD
      DATA    ICALLD /0/
      CHARACTER*20 LASTQT
      SAVE         LASTQT
      DATA         LASTQT /'                    '/
C
      NTOT = NX*NY*NZ*NEL
C
      IF(DERIV.EQ.'LOG ') THEN
        CALL VLOGA(WORK,NTOT)
        CALL VLOGA(WKV1,NTOT)
        CALL VLOGA(WKV2,NTOT)
        IF (if3d) CALL VLOGA(WKV3,NTOT)
      ENDIF
C
C
C     Compute range for WORK array
C
      IF (QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N') THEN
         WKMAX = SPMAX
         WKMIN = SPMIN
         WKDLT = WKMAX-WKMIN
      ELSE
         WKMAX = GLMAX(WORK,NTOT)
         WKMIN = GLMIN(WORK,NTOT)
         WKDLT = WKMAX-WKMIN
         write(6,*) 'wkmax:',quanty,wkmax,wkmin,wkdlt
      ENDIF
      IF (WKDLT.EQ.0.0) THEN
         WRITE(S,1000) WKMAX,WKMIN
 1000    FORMAT('WORK - MAX,MIN:',2G14.4,'$')
         CALL PRS(S)
      ENDIF
C
      TTMIN=WKMIN
      TTMAX=WKMAX
c
      if (ifuscl) then
         TTMIN=usrMIN
         TTMAX=usrMAX
      endif
c
      LASTQT=QUANTY
      return
      END
c-----------------------------------------------------------------------
      subroutine vloga(x,n)
      DIMENSION X(1)
C
      do 10 i=1,n
         if (x(i).ne.0) then
            xmin = abs(x(i))
            goto 11
         endif
   10 continue
   11 continue
C
      do 20 i=1,n
         if (x(i).ne.0) xmin = min( xmin,abs(x(i)) )
   20 continue
C
      do 30 i=1,n
         x(i) = max(xmin,abs(x(i)))
         x(i) = log10(x(i))
   30 continue
C
      return
      end
c-----------------------------------------------------------------------
      subroutine coef2
C-----------------------------------------------------------------
C
C     GENERATE Derivative operators
C
C-----------------------------------------------------------------
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      REAL PN(20)
C
      DO 100 IE=1,NEL
         CALL GEOM1(IE)
  100 CONTINUE
C
C     Set up box sizes associated with each element defined to be
C     the circumscibing parallelagram for the each element.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine find_lam2(lam2,lam,aa,w,miter,ndim,ierr)
      real aa(3,3),lam(3),w(3,3),lam2
c
      common /ecmnr/ a,b,c,d,e,f,f2,ef,df,r0,r1
      common /ecmni/ nr
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
c
c
      iffout = .false.
      ierr = 0
c
c     2D case....
c
c
      if (ndim.eq.2) then
         a = aa(1,1)
         b = aa(1,2)
         c = aa(2,1)
         d = aa(2,2)
         aq = 1.
         bq = -(a+d)
         cq = a*d-c*b
c
         call quadratic(x1,x2,aq,bq,cq,ierr)
c 
         lam2 = min(x1,x2)
c
         return
      endif
c
c
c
c
c     Else ...  3D case....
c
c                                    a d e
c     Get symmetric 3x3 matrix       d b f
c                                    e f c
c
      a = aa(1,1)
      b = aa(2,2)
      c = aa(3,3)
      d = 0.5*(aa(1,2)+aa(2,1))
      e = 0.5*(aa(1,3)+aa(3,1))
      f = 0.5*(aa(2,3)+aa(3,2))
      ef = e*f
      df = d*f
      f2 = f*f
c
c     Use Gershgorin to get bounds on eigenvalues
c
      x0  =         a - abs(d) - abs(e)
      x0  = min(x0,(b - abs(d) - abs(f)))
      x0  = min(x0,(c - abs(e) - abs(f)))
      x1  =         a + abs(d) + abs(e)
      x1  = max(x1,(b + abs(d) + abs(f)))
      x1  = max(x1,(c + abs(e) + abs(f)))
c
      dx  = x1-x0
      x2  = 0.5*(x0+x1)
      tol = 1.e-4*(abs(x0)+abs(x1)+abs(x2))
c
      if (dx.le.tol) then
         lam2 = x2
         return
      endif
c
c     Increase interval width to guarantee bracketing
      x0 = x0-dx*1.e-3
      x1 = x1+dx*1.e-3
c
c     number of known roots is zero
      nr = 0
c
      tolf = 1.e-5
      miter = 0
c
      do i=1,3
         if (i.eq.1) then
c           call muller(lam(i),x2,x1,x0,tol,tolf,kiter,ierr)
            lam(i) = zbrent(x1,x0,tol,tolf,kiter,ierr)
         else
            call muller2(lam(i),x2,x1,x0,tol,tolf,kiter,i,ierr)
         endif
         if (ierr.ne.0) return
         miter = max(kiter,miter)
         lam2=lam(i)
         if (i.eq.2 .and. lam(1).gt.lam(2)) then
            lam(2)=lam(1)
            lam(1)=lam2
         endif
c
c        Shift initial guesses away from previous solution
c
         if (abs(x0-lam2).lt.tol) then
            x0 = x0 - 0.1*(x2-x0)
         elseif (abs(x1-lam2).lt.tol) then
            x1 = x1 + 0.1*(x2-x0)
         elseif (abs(x2-lam2).lt.tol) then
            x2 = x2 + 0.5*(x1-x2)
         endif
c
         r0 = lam(1)
         r1 = lam(2)
         nr = i
      enddo
c
c     write(6,1) (aa(k,1),k=1,9),(lam(l),l=1,3)
c   1 format(1p12e10.2)
c
c
c     Find middle eigenvalue
c
c
      if (lam(3).ge.lam(2)) then
         lam2=lam(2)
         return
      elseif (lam(3).lt.lam(1)) then
         lam2=lam(1)
         return
      endif
c
      lam2=lam(3)
      return
      end
c-----------------------------------------------------------------------
      subroutine muller(x2,x2i,x1i,x0i,tol,tolf,kiter,ierr)
c
c     Use muller's method to find (known real) roots of f(x)
c     Burden and Faires p. 79
c
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
c
      iffout = .false.
      ifdefl = .true.
      ierr   = 0
c
      x0=x0i
      x1=x1i
      x2=x2i
      kiter=1
c
    1 continue
      h1=x1-x0
      h2=x2-x1
c
      f0 = ff(x0)
      if (abs(f0).lt.tolf) then
         x2 = x0
         return
      endif
c
      f1 = ff(x1)
      if (abs(f1).lt.tolf) then
         x2 = x1
         return
      endif
c
      f2 = ff(x2)
      if (abs(f2).lt.tolf) return
      d1 = (f1-f0)/h1
      d2 = (f2-f1)/h2
      d  = (d2-d1)/(h1+h2)
c
c     Check discriminate
c
      b = d2+h2*d
      disc = b*b-4.*f2*d
      if (disc.le.0) then
         x2 = x2+0.5*min(abs(x1-x2),abs(x0-x2))
         goto 1
      endif
c
      do i=1,9
         disc = sqrt(disc)
         bm = b-disc
         bp = b+disc
         if (abs(bm).lt.abs(bp)) then
            e=bp
         else
            e=bm
         endif
         h = -2*f2/e
         x0 = x1
         x1 = x2
         x2 = x2+h
c
         f0 = f1
         f1 = f2
c
    3    continue
c
         if (abs(h).lt.tol) then
            return
         endif
         kiter = kiter+1
         if (kiter.gt.9) ifdefl=.false.
c
         h1=x1-x0
         h2=x2-x1
         f2 = ff(x2)
         if (abs(f2).lt.tolf) return
         d1 = (f1-f0)/h1
         d2 = (f2-f1)/h2
         d  = (d2-d1)/(h1+h2)
c
c        Check discriminate
c
         b = d2+h2*d
         disc = b*b-4.*f2*d
         if (abs(disc).lt.tol) disc = tol*tol
         if (disc.eq.0.) disc = 1.e-15
         if (disc.lt.0) then
            write(6,4) i,kiter,disc,x0,x1,x2,f0,f1,f2
    4       format(2i2,1p7e10.2)
            x2 = x2+1.0*max(abs(x1-x2),abs(x0-x2))
            iffout = .true.
            if (kiter.gt.15) ierr = 1
            if (kiter.gt.15) return
            goto 3
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine muller2(x2,x2i,x1i,x0i,tol,tolf,kiter,l,ierr)
c
c     Use muller's method to find (known real) roots of f(x)
c     Burden and Faires p. 79
c
      integer ibad,igood
      save    ibad,igood
      data    ibad,igood /0,0/
c
      real xsave
      save xsave
c
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
c
      iffout = .false.
      ifdefl = .true.
      ierr   = 0
c
c
      if (l.eq.3) then
         x2=xsave
         return
      endif
c
      x0=x0i
      x1=x1i
      x2=x2i
c
      x2    = 0.5*(x0+x1)
      xsave = x2
      if (abs(x1-x0).lt.tol) return
c
      x2 = x1 + 0.5*(x1-x0)
      kiter=1
c
    1 continue
      h1=x1-x0
      h2=x2-x1
c
      f0 = ff(x0)
      f1 = ff(x1)
      f2 = ff(x2)
c
      d1 = (f1-f0)/h1
      d2 = (f2-f1)/h2
      d  = (d2-d1)/(h1+h2)
c
c     Check discriminate
c
      b = d2+h2*d
      disc = b*b-4.*f2*d
      if (disc.lt.tolf) then
         ibad = ibad+1
         ib10 = ibad/10000
         if (mod(ibad,10000).eq.0)
     $   write(6,4) 'm2',ib10,disc,x0,x1,x2,f0,f1,f2
    4    format(a2,i8,1p7e10.2)
         ierr = 1
         return
      endif
c
      c = f2
      a = d
c
      call quadratic(x2,xsave,a,b,c,ierr)
c
c     igood = igood+1
c     ig10  = igood/10000
c     if (mod(igood,10000).eq.0)
c    $   write(6,5) 'ok',ig10,ierr,x2,xsave,x0,x1,f2
c   5    format(a2,i8,i2,1p7e10.2)
c
      return
      end
c-----------------------------------------------------------------------
      function ff(x)
c
      common /ecmnr/ a,b,c,d,e,f,f2,ef,df,r0,r1
      common /ecmni/ nr
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
c
      eps = 1.e-5
      ax = a-x
      bx = b-x
      cx = c-x
      ff = ax*(bx*cx-f2) - d*(d*cx-ef) + e*(df-e*bx)
      if (abs(ff).lt.eps) return
      if (.not.ifdefl) return
c
c     Deflate, if possible
c
      f1 = ff
      if (nr.ge.1.and.x.ne.r0) ff = ff/(x-r0)
      f2 = ff
      if (nr.ge.2.and.x.ne.r1) ff = ff/(x-r1)
      f3 = ff
      if (iffout) write(6,6) 'ff: ',f1,f2,f3,x,r0,r1
    6 format(a4,1p6e12.4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lin_combo
c
c     Set uvwpt to be linear combination of current and new .fld files
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ wk(LXYZ,3,3)
      real vv(3,3),ss(3,3),oo(3,3),lam(3)
c
      common /sfldwrk/ odumpw
      integer odumpw
c
      integer idumvtx
      save    idumvtx
      data    idumvtx  /-1234567/
c
      integer lalg
      save    lalg
      data    lalg  /-1234567/
c
      call prs('Input c0 and c1 (e.g., 1 -1) for linear combination$')
      call prs('of current (0) and new (1) .fld files:$')
      call rerr(c0,c1)
c
      ntot = nx*ny*nz*nel
      call cmult2(wkv1,u,c0,ntot)
      call cmult2(wkv2,v,c0,ntot)
      call cmult2(wkv3,w,c0,ntot)
      call cmult2(work,p,c0,ntot)
      call cmult2(wrk2,t,c0,ntot)
c
      u1max = glmax(u,ntot)
      u2max = glmax(v,ntot)
      u3max = glmax(w,ntot)
      u1min = glmin(u,ntot)
      u2min = glmin(v,ntot)
      u3min = glmin(w,ntot)
      write(6,11) ntot,c0,u1min,u2min,u3min,u1max,u2max,u3max,'U12MX'
c
c
      w1max = glmax(wkv1,ntot)
      w2max = glmax(wkv2,ntot)
      w3max = glmax(wkv3,ntot)
      w1min = glmin(wkv1,ntot)
      w2min = glmin(wkv2,ntot)
      w3min = glmin(wkv3,ntot)
      write(6,11) ntot,c0,w1min,w2min,w3min,w1max,w2max,w3max,'W12MX'
  11  format(i6,1p7e12.5,1x,a5)
c
      call prs('Input new dump number for .fld (1):$')
      call rei(kdump)
c
c     Get new field w/o calling SETWRK and w/o calling COEF
c
      call getfld(kdump,ierr,.false.,.false.)
c
      u1max = glmax(u,ntot)
      u2max = glmax(v,ntot)
      u3max = glmax(w,ntot)
      u1min = glmin(u,ntot)
      u2min = glmin(v,ntot)
      u3min = glmin(w,ntot)
      write(6,11) ntot,c0,u1min,u2min,u3min,u1max,u2max,u3max,'U12MB'
c
      w1max = glmax(wkv1,ntot)
      w2max = glmax(wkv2,ntot)
      w3max = glmax(wkv3,ntot)
      w1min = glmin(wkv1,ntot)
      w2min = glmin(wkv2,ntot)
      w3min = glmin(wkv3,ntot)
      write(6,11) ntot,c1,w1min,w2min,w3min,w1max,w2max,w3max,'W12MB'
c
      if (ierr.ne.0) then
         CALL PRSI('Unable to open dump number:$',newdump)
         return
      endif
c
      call add2s1(u,wkv1,c1,ntot)
      call add2s1(v,wkv2,c1,ntot)
      call add2s1(w,wkv3,c1,ntot)
      call add2s1(p,work,c1,ntot)
      call add2s1(t,wrk2,c1,ntot)
c
      u1max = glmax(u,ntot)
      u2max = glmax(v,ntot)
      u3max = glmax(w,ntot)
      u1min = glmin(u,ntot)
      u2min = glmin(v,ntot)
      u3min = glmin(w,ntot)
      write(6,11) ntot,c1,u1min,u2min,u3min,u1max,u2max,u3max,'U12MC'
c
      w1max = glmax(wkv1,ntot)
      w2max = glmax(wkv2,ntot)
      w3max = glmax(wkv3,ntot)
      w1min = glmin(wkv1,ntot)
      w2min = glmin(wkv2,ntot)
      w3min = glmin(wkv3,ntot)
      write(6,11) ntot,c1,w1min,w2min,w3min,w1max,w2max,w3max,'W12MC'
c
      return
      end
c-----------------------------------------------------------------------
      subroutine new_vortex
c
c     Compute vortices as recommended by Jeong & Hussai JFM 95 [285]
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ wk(LXYZ,3,3)
      real vv(3,3),ss(3,3),oo(3,3),lam(3)
c
      common /sfldwrk/ odumpw
      integer odumpw
c
      integer idumvtx
      save    idumvtx
      data    idumvtx  /-1234567/
c
      integer lalg
      save    lalg
      data    lalg  /-1234567/
c
      common /lfilt/ ifstdfilt
      logical        ifstdfilt

      common /ctmp1/ wf(0:3*nxm*nym*nzm)
      real filt(nxm*nxm)
      save filt

      integer nro
      save    nro
      data    nro  / -1 /

      nr = nx-1
      if (nr.ne.nro) then
         nro   = nr
         ncut  = 3
         alpha = 0.3
         nid   = 0
         call build_new_filter(filt,zpts,nx,ncut,alpha,nid)
      endif
c
c
      ialg = 2
c     call prs('algorithm? (0=old) (1=new) (2=cubic!)$')
c     call rei(ialg)
      write(6,*) 'in vortex routine ifds, ialg',ifdssum,ialg
      if (ialg.ne.lalg) then
c        force recomputation
         lalg    = ialg
         idumvtx = -999
      endif
c
      if (idumvtx.eq.odumpw) then
         ntot = nx*ny*nz*nel
         call copy(work,vortex,ntot)
         return
      endif
      idumvtx = odumpw
c
      write(6,*) 'start computing vortex',ifdssum
      NXYZ  = NX*NY*NZ
      lk = 0
      nssyv = 0
      DO IEL=1,NEL
         IEOFF = (IEL-1)*NXYZ+1
c
c        Compute velocity gradient tensor:
c
c
          call rzero(wk,lxyz*9)
          call vel_grad_tens(u,1,iel)
          call vel_grad_tens(v,2,iel)
          if (if3d) call vel_grad_tens(w,3,iel)
c
c        Compute symm. and skew-symm. forms
c
         miter = 0
c        write(6,*) 'vortex:',iel
         do l=1,nxyz
c
            do j=1,ndim
            do i=1,ndim
               ss(i,j) = 0.5*(wk(l,i,j)+wk(l,j,i))
               oo(i,j) = 0.5*(wk(l,i,j)-wk(l,j,i))
            enddo
            enddo
c
            call rzero(vv,9)
            do j=1,ndim
            do i=1,ndim
               do k=1,ndim
                  vv(i,j) = vv(i,j)
     $                    + ss(i,k)*ss(k,j) + oo(i,k)*oo(k,j)
               enddo
            enddo
            enddo
c
c           Solve eigenvalue problem using LAPACK ssyev; eigenvalues
c           returned in ascending order.
c
c           write(6,*) iel,l,ialg,' vortex?'
            ialg = 0
            if (ialg.eq.0) then
               call ssyev('N','U',3,vv,3,lam,ss,9,info)
            elseif (ialg.eq.1) then
               call find_lam2(xl2,lam,vv,w,niter,ndim,ierr)
               if (ierr.ne.0) then
                  call ssyev('N','U',3,vv,3,lam,ss,9,info)
                  nssyv = nssyv+1
               else
                  miter = max(miter,niter)
               endif
               lam(2) = xl2
               info =0
            else
               call find_lam3(xl2,lam,vv,w,ndim,ierr)
            endif
c
            if(info.ne.0) call prsi('Info nonzero in ssyev!:$',info)
c
            lk = lk+1
            vortex(lk) = lam(2)
         enddo
c
      enddo
c
      ntot = nx*ny*nz*nel
c
      write(6,*) 'done vortex, call dsavg.  maxiter:',miter,nssyv,ntot
      if (ifdssum) call dsavg(vortex,wkv3)
      write(6,*) 'done dsavg',ifdssum
c
      m  = nx*ny*nz
      m2 = nx*ny*nz*2
      ifstdfilt = .true.
      call filterq(vortex,filt,nx,nz,nel,wf,wf(m),wf(m2),if3d,fmx)
c
      ntot = nx*ny*nz*nel
      call copy(work,vortex,ntot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vel_grad_tens(vel,i,ie)
c
c     Compute dvel/dx_j  j=1,3
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      parameter (lxyz=nxm*nym*nzm)
      COMMON /CTMP3/ vrst(lxyz,3)
      COMMON /CTMP4/ wk  (lxyz,3,3)
c
      real vel(1)
c
      NXY1  = NX*NY
      NYZ1  = NY*NZ
      NXYZ1 = NX*NY*NZ
C
      jel = nxyz1*(ie-1) + 1
c
      if (if3d) then
         call    mxm  (dgdr,nx,vel(jel),nx,vrst(1,1),nyz1)
         do iz=1,nz
            k=jel + nxy1*(iz-1)
            l=  1 + nxy1*(iz-1)
            call mxm  (vel(k),nx,dgdrt,ny,vrst(l,2),ny)
         ENDDO
         call    mxm  (vel(jel),nxy1,dgdrt,nz,vrst(1,3),nz)
c
c        Now, compute du/dx
c
         call    col3(wk(1,i,1),vrst(1,1),rxm1(jel),nxyz1)
         call addcol3(wk(1,i,1),vrst(1,2),sxm1(jel),nxyz1)
         call addcol3(wk(1,i,1),vrst(1,3),txm1(jel),nxyz1)
         call invcol2(wk(1,i,1),jacm1(jel) ,nxyz1)
c
c        Now, compute du/dy
c
         call    col3(wk(1,i,2),vrst(1,1),rym1(jel),nxyz1)
         call addcol3(wk(1,i,2),vrst(1,2),sym1(jel),nxyz1)
         call addcol3(wk(1,i,2),vrst(1,3),tym1(jel),nxyz1)
         call invcol2(wk(1,i,2),jacm1(jel) ,nxyz1)
c
c        Now, compute du/dz
c
         call    col3(wk(1,i,3),vrst(1,1),rzm1(jel),nxyz1)
         call addcol3(wk(1,i,3),vrst(1,2),szm1(jel),nxyz1)
         call addcol3(wk(1,i,3),vrst(1,3),tzm1(jel),nxyz1)
         call invcol2(wk(1,i,3),jacm1(jel) ,nxyz1)
c
      else
c
c        2D
c
         call    mxm  (dgdr,nx,vel(jel),nx,vrst(1,1),nyz1)
         do iz=1,nz
            k=jel + nxy1*(iz-1)
            l=  1 + nxy1*(iz-1)
            call mxm  (vel(k),nx,dgdrt,ny,vrst(l,2),ny)
         ENDDO
c
c        Now, compute du/dx
c
         call    col3(wk(1,i,1),vrst(1,1),rxm1(jel),nxyz1)
         call addcol3(wk(1,i,1),vrst(1,2),sxm1(jel),nxyz1)
         call invcol2(wk(1,i,1),jacm1(jel) ,nxyz1)
c
c        Now, compute du/dy
c
         call    col3(wk(1,i,2),vrst(1,1),rym1(jel),nxyz1)
         call addcol3(wk(1,i,2),vrst(1,2),sym1(jel),nxyz1)
         call invcol2(wk(1,i,2),jacm1(jel) ,nxyz1)
c
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine quadratic(x1,x2,a,b,c,ierr)
c
c     Stable routine for computation of real roots of quadratic
c
      ierr = 0
      x1 = 0.
      x2 = 0.
c
      if (a.eq.0.) then
         if (b.eq.0) then
            if (c.ne.0) then
c              write(6,10) x1,x2,a,b,c
               ierr = 1
            endif
            return
         endif
         ierr = 2
         x1 = -c/b
c        write(6,11) x1,a,b,c
         return
      endif
c
      d = b*b - 4.*a*c
      if (d.eq.0) then
         x1 = -b/(2*a)
         x2 = -b/(2*a)
         return
      elseif (d.lt.0) then
         ierr = 1
c        write(6,12) a,b,c,d
         return
      endif
      if (d.gt.0) d = sqrt(d)
c
      if (b.gt.0) then
         x1 = -2.*c / ( d+b )
         x2 = -( d+b ) / (2.*a)
      else
         x1 =  ( d-b ) / (2.*a)
         x2 = -2.*c / ( d-b )
      endif
         x1 = -b/2*a
c
   10 format('ERROR: Both a & b zero in routine quadratic NO ROOTS.'
     $      ,1p5e12.4)
   11 format('ERROR: a = 0 in routine quadratic, only one root.'
     $      ,1p5e12.4)
   12 format('ERROR: negative discriminate in routine quadratic.'
     $      ,1p5e12.4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vol_int(sum,vol,g)
c
      include 'basics.inc'
      include 'basicsp.inc'
      real g(1)
c
      sum = 0.
      vol = 0.
      l   = 0
      if (ndim.eq.3) then
         do ie = 1,nel
            do iz = 1,nz
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               sum = sum + wght(ix)*wght(iy)*wght(iz)*jacm1(l)*g(l)
               vol = vol + wght(ix)*wght(iy)*wght(iz)*jacm1(l)
            enddo
            enddo
            enddo
         enddo
      elseif (ifaxis) then
         one = 1.
         pi  = 4.*atan(one)
         do ie = 1,nel
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               tpr = 2.*pi*yp(l)
               sum = sum + tpr*wght(ix)*wght(iy)*jacm1(l)*g(l)
               vol = vol + tpr*wght(ix)*wght(iy)*jacm1(l)
            enddo
            enddo
         enddo
      else
         do ie = 1,nel
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               sum = sum + wght(ix)*wght(iy)*jacm1(l)*g(l)
               vol = vol + wght(ix)*wght(iy)*jacm1(l)
            enddo
            enddo
         enddo
      endif
c
      call prsr('Volume   of   domain:$',vol)
      call prsr('Integral of function:$',sum)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vol_int2(sum,vol,g,h)
c
      include 'basics.inc'
      include 'basicsp.inc'
      real g(1),h(1)
c
      sum = 0.
      vol = 0.
      l   = 0
      if (ndim.eq.3) then
         do ie = 1,nel
            do iz = 1,nz
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               gh  = g(l)*h(l)
               sum = sum + wght(ix)*wght(iy)*wght(iz)*jacm1(l)*gh
               vol = vol + wght(ix)*wght(iy)*wght(iz)*jacm1(l)
            enddo
            enddo
            enddo
         enddo
      elseif (ifaxis) then
         one = 1.
         pi  = 4.*atan(one)
         do ie = 1,nel
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               tpr = 2.*pi*yp(l)
               gh  = g(l)*h(l)
               sum = sum + tpr*wght(ix)*wght(iy)*jacm1(l)*gh
               vol = vol + tpr*wght(ix)*wght(iy)*jacm1(l)
            enddo
            enddo
         enddo
      else
         do ie = 1,nel
            do iy = 1,ny
            do ix = 1,nx
               l = l+1
               gh  = g(l)*h(l)
               sum = sum + wght(ix)*wght(iy)*jacm1(l)*gh
               vol = vol + wght(ix)*wght(iy)*jacm1(l)
            enddo
            enddo
         enddo
      endif
c
      call prsr('Volume   of   domain:$',vol)
      call prsr('Integral of function:$',sum)
c
      return
      end
c-----------------------------------------------------------------------
      FUNCTION ZBRENT(X1,X2,TOL,TOLF,iter)
C
C     Using the Van Wijngaarden-Dekker-Brent Method, find the root
C     of a function FUNC known to lie between X1 and X2.  The root,
C     returned as ZBRENT, will be refined until its accuracy is TOL.
C
      PARAMETER (ITMAX=100,EPS=3.0E-8)
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
C
      iffout = .false.
      ifdefl = .true.
c
      iter=0
      A = X1
      B = X2
      FA = ff(A)
      FB = ff(B)
c
      if (abs(fa).le.tolf) then
         zbrent = a
         return
      endif
      if (abs(fb).le.tolf) then
         zbrent = b
         return
      endif
c
      IF (FB*FA.GT.0.0) GOTO 9000
      FC=FB
C
      DO 1000 ITER=1,ITMAX
         IF (FB*FC.GT.0.0) THEN
            C  = A
            FC = FA
            D  = B-A
            E  = D
         ENDIF
         IF (ABS(FC).LT.ABS(FB)) THEN
            A = B
            B = C
            C = A
            FA = FB
            FB = FC
            FC = FA
         ENDIF
         TOL1 = 2.0*EPS*ABS(B)+0.5*TOL
         XM = 0.5*(C-B)
         IF (ABS(XM).LE.TOL1.OR.FB.EQ.0.0) THEN
            ZBRENT = B
            return
         ENDIF
         IF (ABS(E).GT.TOL1.AND. ABS(FA).GT.ABS(FB)) THEN
C
C           Attempt inverse quadratic interpolation
C
            S=FB/FA
            IF (A.EQ.C) THEN
               P=2.0*XM*S
               Q=1.0-S
            ELSE
               Q=FA/FC
               R=FB/FC
               P=S*( 2.0*XM*Q*(Q-R) - (B-A)*(R-1.0) )
               Q=(Q-1.0)*(R-1.0)*(S-1.0)
            ENDIF
C
C           Check whether in bounds...
C
            IF (P.GT.0.0) Q = -Q
            P = ABS(P)
            IF (2.0*P.LT.MIN(3.0*XM*Q-ABS(TOL1*Q),ABS(E*Q))) THEN
C              Accept interpolation.
               E=D
               D=P/Q
            ELSE
C              Interpolation failed, us bisection.
               D=XM
               E=D
            ENDIF
         ELSE
C           Bounds decreasing too slowly, use bisection.
            D=XM
            E=D
         ENDIF
         A=B
         FA=FB
         IF (ABS(D).GT.TOL1) THEN
C           Evaluate new trial root
            B=B+D
         ELSE
            B=B+SIGN(TOL1,XM)
         ENDIF
         FB=ff(B)
 1000 CONTINUE
C
C
 9000 CONTINUE
c
      ZBRENT=B
      if (abs(fa).lt.abs(fb)) zbrent = a
c
c     iffout = .true.
c     FA = ff(A)
c     FB = ff(B)
c     write(6,6) 'zbrent',iter,a,b,fa,fb
c     if (iffout) write(6,6) 'ff: ',f1,f2,f3,x,r0,r1
    6 format(a6,i5,1p6e12.4)
c
      return
      END
c-----------------------------------------------------------------------
      subroutine find_lam3(lam2,lam,aa,w,ndim,ierr)
      real aa(3,3),lam(3),w(3,3),lam2
c
c     Use cubic eqn. to compute roots
c
c
      common /ecmnr/ a,b,c,d,e,f,f2,ef,df,r0,r1
      common /ecmni/ nr
      common /ecmnl/ iffout,ifdefl
      logical        iffout,ifdefl
c
c
      iffout = .false.
      ierr = 0
c
c     2D case....
c
c
      if (ndim.eq.2) then
         a = aa(1,1)
         b = aa(1,2)
         c = aa(2,1)
         d = aa(2,2)
         aq = 1.
         bq = -(a+d)
         cq = a*d-c*b
c
         call quadratic(x1,x2,aq,bq,cq,ierr)
c 
         lam2 = min(x1,x2)
c
         return
      endif
c
c
c
c
c     Else ...  3D case....
c
c                                    a d e
c     Get symmetric 3x3 matrix       d b f
c                                    e f c
c
      a = aa(1,1)
      b = aa(2,2)
      c = aa(3,3)
      d = 0.5*(aa(1,2)+aa(2,1))
      e = 0.5*(aa(1,3)+aa(3,1))
      f = 0.5*(aa(2,3)+aa(3,2))
      ef = e*f
      df = d*f
      f2 = f*f
c
c
c     Use cubic eqn. to compute roots
c
c     ax = a-x
c     bx = b-x
c     cx = c-x
c     y = ax*(bx*cx-f2) - d*(d*cx-ef) + e*(df-e*bx)
c
      a1 = -(a+b+c)
      a2 =  (a*b+b*c+a*c) - (d*d+e*e+f*f)
      a3 =  a*f*f + b*e*e + c*d*d - a*b*c - 2*d*e*f
c
      call cubic  (lam,a1,a2,a3,ierr)
      call sortit (lam,w,3)
      lam2 = lam(2)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cubic(xo,ai1,ai2,ai3,ierr)
      real xo(3),ai1,ai2,ai3
      complex*16 x(3),a1,a2,a3,q,r,d,arg,t1,t2,t3,theta,sq,a13
c
c     Compute real solutions to cubic root eqn. (Num. Rec. v. 1, p. 146)
c     pff/Sang-Wook Lee  Jan 19 , 2004
c
c     Assumption is that all x's are *real*
c
      real*8 twopi
      save   twopi
      data   twopi /6.283185307179586476925286766/
c
      ierr = 0
c
      zero = 0.
      a1   = cmplx(ai1,zero)
      a2   = cmplx(ai2,zero)
      a3   = cmplx(ai3,zero)
c
      q = (a1*a1 - 3*a2)/9.
      if (q.eq.0) goto 999
c
      r = (2*a1*a1*a1 - 9*a1*a2 + 27*a3)/54.
c
      d = q*q*q - r*r
c
c     if (d.lt.0) goto 999
c
      arg   = q*q*q
      arg   = sqrt(arg)
      arg   = r/arg
c
      if (abs(arg).gt.1) goto 999
      theta = acos(abs(arg))
c
      t1    = theta / 3.
      t2    = (theta + twopi) / 3.
      t3    = (theta + 2.*twopi) / 3.
c
      sq  = -2.*sqrt(q)
      a13 = a1/3.
      x(1) = sq*cos(t1) - a13
      x(2) = sq*cos(t2) - a13
      x(3) = sq*cos(t3) - a13
c
      xo(1) = real(x(1))
      xo(2) = real(x(2))
      xo(3) = real(x(3))
c
      return
c
  999 continue   ! failed
      ierr = 1
      call rzero(x,3)
      return
c
      end
c-----------------------------------------------------------------------
      subroutine vfiles
C
C     Draws multiple vector plots on the domain  (pff 2/25/00)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
      PI=4.0*ATAN(1.0)
C
C     Get end points
C
      CALL PRS
     $   ('Enter 0 for file input, 1 for keyboard input:$')
      CALL rei(input)
      if (input.eq.0) then
         call gtlines
         call vfiles2
         return
      endif
c
      DO 1000 IPROF=1,NLNMAX
C
         CALL PRS
     $   ('Enter location for plot (IDENTICAL points to EXIT).$')
         CALL PRS('Enter 1st Endpoint:$')
         CALL GETPT(XLIN1,YLIN1,ZLIN1,' ')
         CALL PRS('Enter 2nd Endpoint:$')
         CALL GETPT(XLIN2,YLIN2,ZLIN2,' ')
C
C        Exit Condtion: identical points for endpoints of line.
C
         IF (XLIN1.EQ.XLIN2 .AND.
     $       YLIN1.EQ.YLIN2 .AND.
     $       ZLIN1.EQ.ZLIN2 ) THEN
             GOTO 9000
         ENDIF
C
C        We use temporary LINE points so that the last location
C        plotted can be used in the normal profile routine w/o having
C        to call SET LOCATION.
C
         XLINE(1) = XLIN1
         YLINE(1) = YLIN1
         ZLINE(1) = ZLIN1
         XLINE(2) = XLIN2
         YLINE(2) = YLIN2
         ZLINE(2) = ZLIN2
C
C        Set up scalar plot data
C
         IF (IPROF.EQ.1.OR.
     $      (.NOT.if3d.AND.QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N')) 
     $       call setwrk(.true.)
C
C        Plot on physical screen
C
         call plotphv
 1000 CONTINUE
 9000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine vfiles2
C
C     Draws multiple vector plots on the domain  2/25/00 pff
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
C
C     For now, we wipe out the old redraw information
C
      CALL RESETDW
      IFSAVE=.TRUE.
C
C     Get end points
C
      DO 1000 IPROF=1,NLNMAX
         XLIN1 = XLINES(1,IPROF)
         YLIN1 = YLINES(1,IPROF)
         ZLIN1 = ZLINES(1,IPROF)
         XLIN2 = XLINES(2,IPROF)
         YLIN2 = YLINES(2,IPROF)
         ZLIN2 = ZLINES(2,IPROF)
C 
C        Exit Condtion: identical points for endpoints of line.
C
         IF (XLIN1.EQ.XLIN2 .AND.
     $       YLIN1.EQ.YLIN2 .AND.
     $       ZLIN1.EQ.ZLIN2 ) THEN
             GOTO 9000
         ENDIF
C
C        We use temporary LINE points so that the last location
C        plotted can be used in the normal profile routine w/o having
C        to call SET LOCATION.
C
         XLINE(1) = XLIN1
         YLINE(1) = YLIN1
         ZLINE(1) = ZLIN1
         XLINE(2) = XLIN2
         YLINE(2) = YLIN2
         ZLINE(2) = ZLIN2
C
C        Set up scalar plot data
C
         IF (IPROF.EQ.1.OR.
     $      (.NOT.if3d.AND.QUANTY.EQ.'VELOCITY'.AND.COMPON.EQ.'N')) 
     $       call setwrk(.true.)
C
C        Plot on physical screen
C
         call plotphv
 1000 CONTINUE
 9000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine plotphv
C
C     Plot a vector in the physical domain
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      character*1 dollar
      DIMENSION UL(3),VP(3)
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld = icalld + 1
         x0 = xp(1)
         y0 = yp(1)
         z0 = zp(1)
         call interp(v1,x0,y0,z0,idum,work,ierr)
      endif
c
      dollar = '$'
c
C     White
      CALL COLOR(1)
C
      ukdlt = wkdlt
      IF(IFUSCL.AND.USRMIN.NE.USRMAX) ukdlt=(USRMAX-USRMIN)
C     For movies, use PLTSCL=10*UKDLT/XFACO
C     Otherwise, use PLTSCL=10*UKDLT/XFAC (Problem if XFAC different from YFAC)
c     PLTSCL = 10.* UKDLT/XFACO
      PLTSCL = 10.* UKDLT/XFAC
      SPAROW = SARROW/PLTSCL
C
C     Abcissa
C
      ABCLNG = (XLINE(2)-XLINE(1))**2
     $       + (YLINE(2)-YLINE(1))**2
     $       + (ZLINE(2)-ZLINE(1))**2
      ABCLNG = SQRT(ABCLNG)
      DABC   = ABCLNG/FLOAT(NPTPRO-1)
C
      IF (if3d) THEN
         CALL MOVE3S( XLINE(1),YLINE(1),ZLINE(1) )
         CALL DRAW3S( XLINE(2),YLINE(2),ZLINE(2) )
         XCOS = (XLINE(2)-XLINE(1)) / ABCLNG
         YCOS = (YLINE(2)-YLINE(1)) / ABCLNG
         ZCOS = (ZLINE(2)-ZLINE(1)) / ABCLNG
         DXABC= XCOS*DABC
         DYABC= YCOS*DABC
         DZABC= ZCOS*DABC
C
         X0 = XLINE(1)
         Y0 = YLINE(1)
         Z0 = ZLINE(1)
         UL(1)=XLINE(2)-XLINE(1)
         UL(2)=YLINE(2)-YLINE(1)
         UL(3)=ZLINE(2)-ZLINE(1)
         CALL NORM3D(UL)
C
         do ipt=0,nptpro
            CALL INTERP(V1,X0,Y0,Z0,IDUM,wkv1,ierr)
            CALL INTERP(V2,X0,Y0,Z0,IDUM,wkv2,ierr)
            CALL INTERP(V3,X0,Y0,Z0,IDUM,wkv3,ierr)
            call arow3np(x0,y0,z0,v1,v2,v3,sparow)
            X0    = X0 + DXABC
            Y0    = Y0 + DYABC
            Z0    = Z0 + DZABC
        enddo
      ELSE
c
c        2D
c
C
         z0 = 0.
         do ipt=0,nptpro
            CALL INTERP(V1,X0,Y0,Z0,IDUM,wkv1,ierr)
            CALL INTERP(V2,X0,Y0,Z0,IDUM,wkv2,ierr)
            call arow3np(x0,y0,z0,v1,v2,v3,sparow)
            X0    = X0 + DXABC
            Y0    = Y0 + DYABC
        enddo
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine rms_avg
c
c     Set uvwpt to be combination of rms and avg .fld files
c
c     Assumes current data set is u_av, v_av, etc.
c
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ wk(LXYZ,3,3)
      real vv(3,3),ss(3,3),oo(3,3),lam(3)
c
      common /sfldwrk/ odumpw
      integer odumpw
c
c
c     rms^2 = rms_in - avg^2
c
      call prs('Input new dump number for rms.fld (1):$')
c
      ntot = nx*ny*nz*nel
      call col3(wkv1,u,u,ntot)
      call col3(wkv2,v,v,ntot)
      call col3(wkv3,w,w,ntot)
c     call col3(work,v,w,ntot)
      call col3(work,p,p,ntot)
c
      call rei(kdump)
c
c     Get new field w/o calling SETWRK!
c
      call getfld(kdump,ierr,.false.,.false.)
      if (ierr.ne.0) then
         CALL PRSI('Unable to open dump number:$',newdump)
         return
      endif
c
      c1 = -1
      call add2s2(u,wkv1,c1,ntot)
      call add2s2(v,wkv2,c1,ntot)
      call add2s2(w,wkv3,c1,ntot)
      call add2s2(p,work,c1,ntot)
c
      call vsqrt (u,ntot)
      call vsqrt (v,ntot)
      call vsqrt (w,ntot)
      call vsqrt (p,ntot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rms_uvw
c
c     Set uvwpt to be combination of rm2 and avg .fld files
c
c     Assumes current data set is u_av, v_av, etc.
c
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ wk(LXYZ,3,3)
      real vv(3,3),ss(3,3),oo(3,3),lam(3)
c
      common /sfldwrk/ odumpw
      integer odumpw
c
c
c     <u'v'> = <uv> - <u><v>
c
      call prs('Input new dump number for rm2.fld (1):$')
c
      ntot = nx*ny*nz*nel
      call col3(wkv1,u,v,ntot)
      call col3(wkv2,v,w,ntot)
      call col3(wkv3,w,u,ntot)
      call col3(work,p,w,ntot)
c
      call rei(kdump)
c
c     Get new field w/o calling SETWRK!
c
      call getfld(kdump,ierr,.false.,.false.)
      if (ierr.ne.0) then
         CALL PRSI('Unable to open dump number:$',newdump)
         return
      endif
c
      c1 = -1
      call add2s2(u,wkv1,c1,ntot)
      call add2s2(v,wkv2,c1,ntot)
      call add2s2(w,wkv3,c1,ntot)
      call add2s2(p,work,c1,ntot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine filter_fld
c
c     filter vx,vy,vz, and T by projecting in the L_k - L_k-2 basis
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /lfilt/ ifstdfilt
      logical        ifstdfilt
c
      common /ctmp1/ wk(0:3*nxm*nym*nzm)
c
      real dmax(50) ! Hopefully, few than 50 fields
c
      real filt(nxm*nxm)
c
      integer nro
      save    nro
      data    nro  / -1 /
c
      nr = nx-1
      if (nr.ne.nro) then
         nro  = nr
c
         ncut  = 1                                          ! Nek5000 default
c        alpha = 0.05                                       ! Nek5000 default
         alpha = 1.0
         nid   = 0                                          ! Nek5000 default
c
         call build_new_filter(filt,zpts,nx,ncut,alpha,nid)
      endif
c
      m  = nx*ny*nz
      m2 = nx*ny*nz*2
c
      ifstdfilt = .false.
c
      write(6,*) 'fq:',nx,nz,m,if3d,u(1),filt(1),zpts(1)
      if (ifflow) then
         call filterq(u,filt,nx,nz,nel,wk,wk(m),wk(m2),if3d,dmax(1))
         call filterq(v,filt,nx,nz,nel,wk,wk(m),wk(m2),if3d,dmax(2))
         if (if3d)
     $   call filterq(w,filt,nx,nz,nel,wk,wk(m),wk(m2),if3d,dmax(3))
      endif
c
      dmax(1) = glmax(dmax(1),1)
      dmax(2) = glmax(dmax(2),1)
      if (if3d) dmax(3) = glmax(dmax(3),1)
c
      nfldd = ndim
      nfldt = 1+npscal
      if (ifheat) then
         do ifld=1,nfldt
            nfldd = nfldd + 1
            ifldp = 1 + ntot*(ifld-1)
            call filterq(t(ifldp),filt
     $                ,nx,nz,nel,wk,wk(m),wk(m2),if3d,dmax(nfldd))
         enddo
         dmax(nfldd) = glmax(dmax(nfldd),1)
      endif
c
c
c     if (nid.eq.0) write(6,1) istep,time,(dmax(k),k=1,nfldd)
c   1 format(i8,' qfilt:',1p10e12.4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine filterqdif(v,f,nx,nz,nel,w1,w2,ft,if3d,dmax)
c
      real v(nx*nx*nz,nel),w1(1),w2(1)
      logical if3d
c
      real f(nx,nx),ft(nx,nx)
c
      integer e
c
      call transpose(ft,nx,f,nx)
c
      nxyz=nx*nx*nz
      dmax = 0.
c
      if (if3d) then
         do e=1,nel
c           Filter
            call copy(w2,v(1,e),nxyz)
            call mxm(f,nx,w2,nx,w1,nx*nx)
            i=1
            j=1
            do k=1,nx
               call mxm(w1(i),nx,ft,nx,w2(j),nx)
               i = i+nx*nx
               j = j+nx*nx
            enddo
            call mxm (w2,nx*nx,ft,nx,w1,nx)
c           call sub3(w2,v(1,e),w1,nxyz)     ! Standard way
c           call copy(v(1,e),w1,nxyz)                            _
            call sub2(v(1,e),w1,nxyz)        ! In post, return v-v
            smax = vlamax(v(1,e),nxyz)
            dmax = max(dmax,abs(smax))
         enddo
c
      else
         do e=1,nel
c           Filter
            call copy(w1,v(1,e),nxyz)
            call mxm(f ,nx,w1,nx,w2,nx)
            call mxm(w2,nx,ft,nx,w1,nx)
c
            call sub2(v(1,e),w1,nxyz)
            smax = vlamax(v(1,e),nxyz)
            dmax = max(dmax,abs(smax))
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine legendre_poly(L,x,N)
c
c     Evaluate Legendre polynomials of degrees 0-N at point x
c
      real L(0:N)
c
      L(0) = 1.
      L(1) = x
c
      do j=2,N
         L(j) = ( (2*j-1) * x * L(j-1) - (j-1) * L(j-2) ) / j 
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine build_new_filter(filt,zpts,nx,kut,wght,nid)
c
c     This routing builds a 1D filter with a transfer function that
c     looks like:
c
c
c        ^
c    d_k |
c        |                 |
c     1  |__________      _v_
c        |          -_     
c        |            \  wght
c        |             \  ___
c        |             |   ^
c     0  |-------------|---|>
c
c        0         c   N   k-->
c
c        Where c := N-kut is the point below which d_k = 1.
c
c
c
c      Here, nx = number of points
c
      real filt(nx,nx),zpts(nx)
c
      parameter (lm=90)
      parameter (lm2=lm*lm)
      real      phi(lm2),pht(lm2),diag(lm2),rmult(lm),Lj(lm)
      integer   indr(lm),indc(lm),ipiv(lm)
c
      if (nx.gt.lm) then
         write(6,*) 'ABORT in build_new_filter:',nx,lm
         call exitt
      endif
c
      kj = 0
      n  = nx-1
      do j=1,nx
         z = zpts(j)
         call legendre_poly(Lj,z,n)
         kj = kj+1
         pht(kj) = Lj(1)
         kj = kj+1
         pht(kj) = Lj(2)
         do k=3,nx
            kj = kj+1
            pht(kj) = Lj(k)-Lj(k-2)
         enddo
      enddo
      call transpose (phi,nx,pht,nx)
      call copy      (pht,phi,nx*nx)
      call gaujordf  (pht,nx,nx,indr,indc,ipiv,ierr,rmult)
c
c     Set up transfer function
c
      call ident   (diag,nx)
c
      k0 = nx-kut
      do k=k0+1,nx
         kk = k+nx*(k-1)
         amp = wght*(k-k0)*(k-k0)/(kut*kut)   ! quadratic growth
         diag(kk) = 1.-amp
      enddo
c
      call mxm  (diag,nx,pht,nx,filt,nx)      !          -1
      call mxm  (phi ,nx,filt,nx,pht,nx)      !     V D V
      call copy (filt,pht,nx*nx)
c
      do k=1,nx*nx
         pht(k) = 1.-diag(k)
      enddo
      np1 = nx+1
      if (nid.eq.0) then
         write(6,6) 'filt amp',(pht (k),k=1,nx*nx,np1)
         write(6,6) 'filt trn',(diag(k),k=1,nx*nx,np1)
   6     format(a8,16f7.4,6(/,8x,16f7.4))
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine ident(a,n)
      real a(n,n)
c
      do i=1,n*n
         a(i,1) = 0.
      enddo
c
      do i=1,n
         a(i,i) = 1.
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine ctr_intp3(u)  ! 3 pt. ctr'd interpolation
      real u(0:1)
c
      k=1
      u(0) = 0.5*(u(-k)+u(k))
c
      return
      end
c-----------------------------------------------------------------------
      subroutine ctr_intp5(u)  ! 5 pt. ctr'd interpolation
      real u(0:1)
c
      k=1
      u(0) = 0.5*(u(-k)+u(k))
c
      k=2
      u(0) = (4*u(0) - 0.5*(u(-k)+u(k)))/3.
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_prof_vars
c
c     Get list of variables for profile output
c
c     10/25/03 pff
C
      include 'basics.inc'
      include 'basicsp.inc'
c
      character*26 items(11)
      save         items
      data         items /              'PLOT                      '
     $ , 'RETURN (no plot)          '
     $ , '   coordinates            ' , '   velocity               '
     $ , '   pressure               ' , '   temperature            '
     $ , '   vortex                 '
     $ , '   scalar work            ' , '   vector work            '  
     $ , '   formatted              ' , '   distance along line    ' /
c    $ , '123456789 123456789 123456' , '123456789 123456789 123456' /
c

      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld = icalld + 1
         ifxprf = .false.
         ifyprf = .false.
         ifzprf = .false.
         ifuprf = .false.
         ifvprf = .false.
         ifwprf = .false.
         ifpprf = .false.
         iftprf = .false.
         ifwkpf = .true.
         ifv1pf = .false.
         ifv2pf = .false.
         ifv3pf = .false.
         ifvtpf = .false.
         ifmtpf = .true.
         ifdprf = .false.  ! distance along line
c        if (if3d) ifzprf = .true.
c        if (if3d) ifwprf = .true.
      endif
c
      nchoic=11
c
    1 continue
         do i=3,nchoic
            call blank(items(i),3)
         enddo
         call chcopy(items(10),'F',1)     ! Unformatted
c
         if (ifxprf) call chcopy(items(3),'T',1)
         if (ifyprf) call chcopy(items(3),'T',1)
         if (ifzprf) call chcopy(items(3),'T',1)
         if (ifuprf) call chcopy(items(4),'T',1)
         if (ifvprf) call chcopy(items(4),'T',1)
         if (ifwprf) call chcopy(items(4),'T',1)
         if (ifpprf) call chcopy(items(5),'T',1)
         if (iftprf) call chcopy(items(6),'T',1)
         if (ifvtpf) call chcopy(items(7),'T',1)
         if (ifwkpf) call chcopy(items(8),'T',1)
         if (ifv1pf) call chcopy(items(9),'T',1)
         if (ifv2pf) call chcopy(items(9),'T',1)
         if (ifv3pf) call chcopy(items(9),'T',1)
         if (ifmtpf) call chcopy(items(10),'T',1)
         if (ifdprf) call chcopy(items(11),'T',1)
c
         do i=1,nchoic
            call chcopy (item(i),items(i),26)
         enddo
c
         call menu(xmouse,ymouse,button,'Select Profile Output')
c

         if (choice.eq.'PLOT'.or.choice.eq.'RETURN (no plot)') then
            nprof_outs = 0
            do k=1,14
               if (ifprofs(k)) then
                   nprof_outs = nprof_outs+1
                   indx_prof(nprof_outs) = k
               endif
            enddo
            return
         elseif (choice.eq.'T  coordinates') then
            ifxprf = .false.
            ifyprf = .false.
            ifzprf = .false.
         elseif (choice.eq.'   coordinates') then
            ifxprf = .true.
            ifyprf = .true.
            if (if3d) ifzprf = .true.
         elseif (choice.eq.'   velocity') then
            ifuprf = .true.
            ifvprf = .true.
            if (if3d) ifwprf = .true.
         elseif (choice.eq.'T  velocity') then
            ifuprf = .false.
            ifvprf = .false.
            ifwprf = .false.
         elseif (choice.eq.'   pressure') then
            ifpprf = .true.
         elseif (choice.eq.'T  pressure') then
            ifpprf = .false.
         elseif (choice.eq.'   temperature') then
            iftprf = .true.
         elseif (choice.eq.'T  temperature') then
            iftprf = .false.
         elseif (choice.eq.'   vortex') then
            ifvtpf = .true.
         elseif (choice.eq.'T  vortex') then
            ifvtpf = .false.
         elseif (choice.eq.'   scalar work') then
            ifwkpf = .true.
         elseif (choice.eq.'T  scalar work') then
            ifwkpf = .false.
         elseif (choice.eq.'   vector work') then
            ifv1pf = .true.
            ifv2pf = .true.
            if (if3d) ifv3pf = .true.
         elseif (choice.eq.'T  vector work') then
            ifv1pf = .false.
            ifv2pf = .false.
            ifv3pf = .false.
         elseif (choice.eq.'F  formatted') then
            ifmtpf = .true.
         elseif (choice.eq.'T  formatted') then
            ifmtpf = .false.
         elseif (choice.eq.'   distance along line    ') then
            ifdprf = .true.
         elseif (choice.eq.'T  distance along line    ') then
            ifdprf = .false.
         endif
      goto 1
      return
      end
c-----------------------------------------------------------------------
      subroutine get_profile(xo,yo,zo,dxabc,dyabc,dzabc,dabc)
c
c     Get profile output
c
c     10/25/03 pff
C
      include 'basics.inc'
      include 'basicsp.inc'
c
      x0 = xo
      y0 = yo
      z0 = zo
      dtest = (0.1*dabc)**2
c
      call izero(itst,nptpro+1)
                  call rzero ( xpt ,nptpro+1)
                  call rzero ( ypt ,nptpro+1)
      if (if3d)   call rzero ( zpt ,nptpro+1)
      if (ifuprf) call rzero ( upr ,nptpro+1)
      if (ifvprf) call rzero ( vpr ,nptpro+1)
      if (ifwprf) call rzero ( wpr ,nptpro+1)
      if (ifpprf) call rzero ( ppr ,nptpro+1)
      if (iftprf) call rzero ( tep ,nptpro+1)
      if (ifvtpf) call rzero ( vtx ,nptpro+1)
      if (ifv1pf) call rzero ( v1p ,nptpro+1)
      if (ifv2pf) call rzero ( v2p ,nptpro+1)
      if (ifv3pf) call rzero ( v3p ,nptpro+1)
      if (ifwkpf) call rzero ( wkp ,nptpro+1)
      if (ifdprf) call rzero ( dpf ,nptpro+1)
c
c     write(6,*) 'get.prf:',nptpro,' CONTINUE??'
c     read (5,*) ans
c
      do i=0,nptpro ! Note change in start from 1 to 0, pff 10/30/03
c
         call interp(xpt(i),x0,y0,z0,idum,xp,ierr)
         if (ierr.eq.0) then ! Brute force verification
            if (if3d) then
               call interp(ypt(i),x0,y0,z0,idum,yp,ierr)
               call interp(zpt(i),x0,y0,z0,idum,zp,ierr)
               zdiff=(xpt(i)-x0)**2+(ypt(i)-y0)**2+(zpt(i)-z0)**2
            else
               call interp(ypt(i),x0,y0,z0,idum,yp,ierr)
               zdiff=(xpt(i)-x0)**2+(ypt(i)-y0)**2
            endif
            if (zdiff.gt.dtest) ierr=2
         endif
c        write(6,66) i,ierr,zdiff,xpt(i),ypt(i),zpt(i),x0,y0,z0
   66    format(i9,i3,1p7e11.3,' xyz')
         itst(i) = ierr
c
         if (ierr.eq.0) then
            if (ifuprf) call interp(upr(i),x0,y0,z0,idum,u   ,ierr)
            if (ifvprf) call interp(vpr(i),x0,y0,z0,idum,v   ,ierr)
            if (ifwprf) call interp(wpr(i),x0,y0,z0,idum,w   ,ierr)
            if (ifpprf) call interp(ppr(i),x0,y0,z0,idum,p   ,ierr)
            if (iftprf) call interp(tep(i),x0,y0,z0,idum,t   ,ierr)
            if (ifvtpf) call interp(vtx(i),x0,y0,z0,idum,vortex,ierr)
            if (ifv1pf) call interp(v1p(i),x0,y0,z0,idum,wkv1,ierr)
            if (ifv2pf) call interp(v2p(i),x0,y0,z0,idum,wkv2,ierr)
            if (ifv3pf) call interp(v3p(i),x0,y0,z0,idum,wkv3,ierr)
            if (ifwkpf) call interp(wkp(i),x0,y0,z0,idum,work,ierr)
         endif
         X0    = X0 + DXABC
         Y0    = Y0 + DYABC
         if (if3d) Z0    = Z0 + DZABC
         if (ifdprf) then
            ddist = dxabc**2 + dyabc**2 + dzabc**2
            if (ddist.gt.0) ddist = sqrt(ddist)
            dpf(i+1)=dpf(i) + ddist
         endif
      enddo
c
c     Clean-up any isolated drop-outs via ctrd interpolation
c
      do i=2,nptpro-2
        if (itst(i).ne.0) then
          if ( itst(i-2).eq.0.and.itst(i-1).eq.0 .and.
     $         itst(i+1).eq.0.and.itst(i+2).eq.0 ) then
                        call ctr_intp5(xpt(i))   ! 5 pt
                        call ctr_intp5(ypt(i))
            if (if3d)   call ctr_intp5(zpt(i))
            if (ifuprf) call ctr_intp5(upr(i))
            if (ifvprf) call ctr_intp5(vpr(i))
            if (ifwprf) call ctr_intp5(wpr(i))
            if (ifpprf) call ctr_intp5(ppr(i))
            if (iftprf) call ctr_intp5(tep(i))
            if (ifvtpf) call ctr_intp5(vtx(i))
            if (ifv1pf) call ctr_intp5(v1p(i))
            if (ifv2pf) call ctr_intp5(v2p(i))
            if (ifv3pf) call ctr_intp5(v3p(i))
            if (ifwkpf) call ctr_intp5(wkp(i))
            itst(i) = -itst(i)
c
            write(6,*) i,' clean up:',xpt(i),ypt(i),zpt(i),itst(i)
            do k=i-2,i+2
               write(6,*) k,xpt(k),ypt(k),zpt(k),itst(k),wkp(k)
            enddo
          endif
c
        endif
      enddo
c
      nptpr2 = max(nptpro-2,1)
      do i=1,nptpro-1,nptpr2
        if (itst(i).ne.0) then
          if (itst(i-2).eq.0.and.itst(i-1).eq.0 .and.
     $        itst(i+1).eq.0.and.itst(i+2).eq.0 ) then
                        call ctr_intp3(xpt(i))   ! 3 pt
                        call ctr_intp3(ypt(i))
            if (if3d)   call ctr_intp3(zpt(i))
            if (ifuprf) call ctr_intp3(upr(i))
            if (ifvprf) call ctr_intp3(vpr(i))
            if (ifwprf) call ctr_intp3(wpr(i))
            if (ifpprf) call ctr_intp3(ppr(i))
            if (iftprf) call ctr_intp3(tep(i))
            if (ifvtpf) call ctr_intp3(vtx(i))
            if (ifv1pf) call ctr_intp3(v1p(i))
            if (ifv2pf) call ctr_intp3(v2p(i))
            if (ifv3pf) call ctr_intp3(v3p(i))
            if (ifwkpf) call ctr_intp3(wkp(i))
          endif
        endif
      enddo
c
c     Output data
c
      do i=0,nptpro
         xpt(i) = xo + i*dxabc
         ypt(i) = yo + i*dyabc
         zpt(i) = zo + i*dzabc
         do k=1,nprof_outs
            outprof(k) = prodat(i,indx_prof(k))
         enddo
         if (ifmtpf) then
            write(46,46) (outprof(k),k=1,nprof_outs)
         else
            write(46) (outprof(k),k=1,nprof_outs)
         endif
c
   46    format(1p13e13.5)
      enddo
      if (.not.if_auto_box) write(46,46) ! blank line to 'profile.dat'
c
      return
      end
c-----------------------------------------------------------------------
      subroutine comp_vorticity
C
C     Set up WORK array according to current plotting state.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ WORK1(LXYZ),WORK2(LXYZ)
c
      write(6,*) 'start computing vorticity',ifdssum
      IF (if3d) THEN
         NXYZ  = NX*NY*NZ
         DO 3 IEL=1,NEL
           IEOFF = (IEL-1)*NXYZ+1
C        WORK1=DW/DY ; WORK2=DV/DZ
           CALL DUDXYZ(WORK1,W,RYM1,SYM1,TYM1,JACM1,IEL)
           CALL DUDXYZ(WORK2,V,RZM1,SZM1,TZM1,JACM1,IEL)
           CALL SUB3(WKV1(IEOFF),WORK1,WORK2,NXYZ)
C        WORK1=DU/DZ ; WORK2=DW/DX
           CALL DUDXYZ(WORK1,U,RZM1,SZM1,TZM1,JACM1,IEL)
           CALL DUDXYZ(WORK2,W,RXM1,SXM1,TXM1,JACM1,IEL)
           CALL SUB3(WKV2(IEOFF),WORK1,WORK2,NXYZ)
C        WORK1=DV/DX ; WORK2=DU/DY
           CALL DUDXYZ(WORK1,V,RXM1,SXM1,TXM1,JACM1,IEL)
           CALL DUDXYZ(WORK2,U,RYM1,SYM1,TYM1,JACM1,IEL)
           CALL SUB3(WKV3(IEOFF),WORK1,WORK2,NXYZ)
    3    CONTINUE
         if (ifdssum) then
            call dsavg(wkv1,work)
            call dsavg(wkv2,work)
            call dsavg(wkv3,work)
         endif
         if (lvrt.gt.ntot) then
            call copy(vrt1,wkv1,ntot)
            call copy(vrt2,wkv2,ntot)
            call copy(vrt3,wkv3,ntot)
         endif
      ELSE
C        WORK1=DV/DX ; WORK2=DU/DY
         NXYZ  = NX*NY*NZ
         DO 5 IEL=1,NEL
           IEOFF = (IEL-1)*NXYZ+1
           CALL DUDXYZ(WORK1,V,RXM1,SXM1,TXM1,JACM1,IEL)
           CALL DUDXYZ(WORK2,U,RYM1,SYM1,TYM1,JACM1,IEL)
           CALL SUB3(WORK(IEOFF),WORK1,WORK2,NXYZ)
    5    CONTINUE
         if (ifdssum) call dsavg(work,wkv3)
      ENDIF
      write(6,*) ' done computing vorticity',ifdssum
      return
      end
c-----------------------------------------------------------------------
      subroutine magnitude(wk,v1,v2,v3)
C
      include 'basics.inc'
c
      real wk(1),v1(1),v2(1),v3(1)
c
      ntot = nx*ny*nz*nel
c
      if (if3d) then
         do i=1,ntot
            wk(i) = v1(i)*v1(i) + v2(i)*v2(i) + v3(i)*v3(i)
            if (wk(i).gt.0) wk(i) = sqrt(wk(i))
         enddo
      else
         do i=1,ntot
            wk(i) = v1(i)*v1(i) + v2(i)*v2(i)
            if (wk(i).gt.0) wk(i) = sqrt(wk(i))
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gtlines_box_x
C
C     Gets a set of lines corresponding to a box 
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
c
C
C     Get end points
C
    1 call prs('Enter number of points in x-direction ( < 0, abort) :$')
c
      call rei(mx)
      if (mx.le.0) then
         call prs('Number of points < 1, returning.$')
         return
      endif
c
c
c     Now, determine grid spacing in x, y, and z
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
c
      dx = (xmax-xmin)/mx
      dy = dx  ! Uniform grid spacing
      dz = dx
      my = 1 + (ymax-ymin)/dy
      mz = 1 + (zmax-zmin)/dz
      write(s,2) mx,my,mz
    2 format('Number of points will be',i4,' x',i4,' x',i4,'. OK?$')
      call prs(s)
      call res(ans,1)
      if (ans.eq.'n'.or.ans.eq.'N') goto 1
c
      nlines = (my+1)*(mz+1)
      if (nlines.gt.nlnmax) then
         write(s,3) nlines,nlnmax
    3    format
     $   ('Number of lines (ny*nz)=',i5,' exceeds max. (',i5,').$') 
         call prs(s)
         call prs
     $   ('Increase NLNMAX in basicsp.inc or decrease resolution.$')
         goto 1
      endif
c
      if (mx.gt.mxpro) then
         write(s,4) nptpro,mxpro
    4    format
     $   ('Number of points (nx)=',i5,' exceeds max. (',i5,').$') 
         call prs(s)
         call prs
     $   ('Increase mxpro in basicsp.inc or decrease resolution.$')
         goto 1
      endif
c
      k = 0
      do kz=0,mz
      do ky=0,my
         k = k+1
c
         ylin1 = ymin + ky*dy
         zlin1 = zmin + kz*dz
c
         xlines(1,k) = xmin
         ylines(1,k) = ylin1
         zlines(1,k) = zlin1
c
         xlines(2,k) = xmax
         ylines(2,k) = ylin1
         zlines(2,k) = zlin1
c
      enddo
      enddo
      nlines = k
c
      xlines(1,nlines+1) = 0.
      ylines(1,nlines+1) = 0.
      zlines(1,nlines+1) = 0.
      xlines(2,nlines+1) = 0.
      ylines(2,nlines+1) = 0.
      zlines(2,nlines+1) = 0.
c
      nptpro     = mx
c
      return
      END
c-----------------------------------------------------------------------
      subroutine gtlines_box_z
C
C     Gets a set of lines corresponding to a box 
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
c
C
C     Get end points
C
    1 call prs('Enter number of points in z-direction ( < 0, abort) :$')
c
      call rei(mz1)
      mz = mz1-1
      if (mz.le.0) then
         call prs('Number of points < 1, returning.$')
         return
      endif
c
c
c     Now, determine grid spacing in x, y, and z
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
c
      dx = (xmax-xmin)
      dy = (ymax-ymin)
      dz = (zmax-zmin)
c
      xmin = xmin + .0001*dx  ! make all points interior to box
      xmax = xmax - .0001*dx
      ymin = ymin + .0001*dy
      ymax = ymax - .0001*dy
      zmin = zmin + .0001*dz
      zmax = zmax - .0001*dz
c
      dx = (xmax-xmin)
      dy = (ymax-ymin)
      dz = (zmax-zmin)
c
      dz = dz/mz
      dy = dz  ! Uniform grid spacing
      dx = dz
c
      my = 1 + (ymax-ymin)/dy
      mx = 1 + (xmax-xmin)/dx
c
      mx1 = mx+1
      my1 = my+1
      mz1 = mz+1
c
      write(s,2) mx1,my1,mz1
    2 format('Number of points will be',i4,' x',i4,' x',i4,'. OK?$')
      call prs(s)
      call res(ans,1)
      if (ans.eq.'n'.or.ans.eq.'N') goto 1
c
      nlines = mx1*my1
      if (nlines.gt.nlnmax) then
         write(s,3) nlines,nlnmax
    3    format
     $   ('Number of lines (nx*ny)=',i5,' exceeds max. (',i5,').$') 
         call prs(s)
         call prs
     $   ('Increase NLNMAX in basicsp.inc or decrease resolution.$')
         goto 1
      endif
c
      if (mz.gt.mxpro) then
         write(s,4) nptpro,mxpro
    4    format
     $   ('Number of points (nx)=',i5,' exceeds max. (',i5,').$') 
         call prs(s)
         call prs
     $   ('Increase mxpro in basicsp.inc or decrease resolution.$')
         goto 1
      endif
c
      k = 0
      do ky=0,my
      do kx=0,mx
         k = k+1
c
         xlin1 = xmin + kx*dx
         ylin1 = ymin + ky*dy
c
         xlines(1,k) = xlin1
         ylines(1,k) = ylin1
         zlines(1,k) = zmin
c
         xlines(2,k) = xlin1
         ylines(2,k) = ylin1
         zlines(2,k) = zmax
c
      enddo
      enddo
      nlines = k
c
      xlines(1,nlines+1) = 0.
      ylines(1,nlines+1) = 0.
      zlines(1,nlines+1) = 0.
      xlines(2,nlines+1) = 0.
      ylines(2,nlines+1) = 0.
      zlines(2,nlines+1) = 0.
c
      nptpro     = mz
c
      return
      end
c-----------------------------------------------------------------------
      subroutine filterqstd(v,f,nx,nz,nel,w1,w2,ft,if3d,dmax)
c
      real v(nx*nx*nz,nel),w1(1),w2(1)
      logical if3d
c
      real f(nx,nx),ft(nx,nx)
c
      integer e
c
      call transpose(ft,nx,f,nx)
c
      nxyz=nx*nx*nz
      dmax = 0.
c
      if (if3d) then
         do e=1,nel
c           Filter
            call copy(w2,v(1,e),nxyz)
            call mxm(f,nx,w2,nx,w1,nx*nx)
            i=1
            j=1
            do k=1,nx
               call mxm(w1(i),nx,ft,nx,w2(j),nx)
               i = i+nx*nx
               j = j+nx*nx
            enddo
            call mxm (w2,nx*nx,ft,nx,w1,nx)
            call sub3(w2,v(1,e),w1,nxyz)     ! Standard way
            call copy(v(1,e),w1,nxyz)
            smax = vlamax(w2,nxyz)
            dmax = max(dmax,abs(smax))
         enddo
c
      else
         do e=1,nel
c           Filter
            call copy(w1,v(1,e),nxyz)
            call mxm(f ,nx,w1,nx,w2,nx)
            call mxm(w2,nx,ft,nx,w1,nx)
c
            call sub3(w2,v(1,e),w1,nxyz)
            call copy(v(1,e),w1,nxyz)
            smax = vlamax(w2,nxyz)
            dmax = max(dmax,abs(smax))
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine filterq(v,f,nx,nz,nel,w1,w2,ft,if3d,dmax)
c
      common /lfilt/ ifstdfilt
      logical        ifstdfilt
c
      real v(nx*nx*nz,nel),w1(1),w2(1)
      logical if3d
c
      real f(nx,nx),ft(nx,nx)
c
      if (ifstdfilt) then
         call filterqstd(v,f,nx,nz,nel,w1,w2,ft,if3d,dmax)
      else
         call filterqdif(v,f,nx,nz,nel,w1,w2,ft,if3d,dmax)
      endif
c
      return
      end
c-----------------------------------------------------------------------
