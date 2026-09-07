c 
c  SET RANGE
c Autoscale for color fills? (y/n).
c Enter lower value of scalar range:
c Enter upper value of scalar range:
c 
c  PLOT
c 
c Floating point exception [fndrr:709 +0x18,0x42cc18]
c         RC = R0 + ( (V-V0)/(V1-V0) ) * (R1-R0)
c (dbx) ^C
c 
C------------------------------------------------------------------------------
C
C                          NEKTON 2.6  2/8/90
C
C                       Copyright (C) 1990, by the 
C
C               Massachusetts Institute of Technology  and Nektonics, Inc.
C
C			     All Rights Reserved
C
C This program is a licenced product of MIT and Nektonics, Inc.,  and it is 
C not to be disclosed to others, copied, distributed, or displayed 
C without prior authorization.
C
C------------------------------------------------------------------------------
C
c-----------------------------------------------------------------------
      subroutine heatc(tpp,xpp,ypp,zpp,ifcrv,ktr)
C
C     This routine does what HEATP used to do, as regards plotting
C     color fills.  In addition, it will produce contour plots.
C
C     It is assumed (for no particularly good reason) that TWRK ranges
C     from 0 to 1.
C 
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION TPP(NX,NY),XPP(NX,NY),YPP(NX,NY),ZPP(NX,NY)
      LOGICAL   IFCRV(4)
C
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
C
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      REAL INTRP
      COMMON /ccdiag/ iee
C
C     Two common blocks below are used in several subsequent routines.
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
      LOGICAL IFOK
      INTEGER ICALLD,NXXLST
      SAVE    ICALLD,NXXLST
      DATA    ICALLD,NXXLST /0,0/
C
C ===================================================================
C     THIS IS THE SETUP STUFF
C ===================================================================
C
      IF (NXBAND.GT.NXXM) THEN

        NXBAND=NXXM
        NXBAND=NXXM / 2

        WRITE(S,10) nxband
   10    FORMAT(2X,'Resetting resolution on color fill to',I3,'.')
        CALL PRS(S//'$')
      ENDIF
C
C     Set the contour levels, etc.
C
      NXX=NXBAND
      NCLLEV=14
      IF (KTR.EQ.1) THEN
        nlevs=NCLLEV
        if (cont_lev.ge.0) cont_lev = (wkmax-wkmin)/nlevs
      ELSE
c       Contours
        if (cont_lev.ge.0) then
           nlevs=abs(NXCONT)
           cont_lev = (wkmax-wkmin)/nlevs
        else
           nlevs  = (wkmax-wkmin)/abs(cont_lev)
           nxcont = nlevs
        endif
      ENDIF
C
      IF (NXX.NE.NXXLST.OR.ILGRNG.NE.2) THEN 
        NXXLST=NXX
        DO 30 I=1,NXX
           R1(I)=2.0*FLOAT(I-1)/FLOAT(NXX-1) - 1.0
           S1(I)=R1(I)
   30    CONTINUE
      ENDIF
C
C     Set ILGRNG=2 so that we know that the Lagrangian (streamline) data
C     in COMMON /CTMP1x/ has been clobbered and will need to be recomputed.
C
C     Force reevaluation of INTRP by changing NXX,NX for this call.
      IF (ILGRNG.NE.2) THEN
         ILGRNG=2
         CALL SETIX(INTRP,NXX,NX,TWRK)
         CALL RZERO(RXVR,NXX2M)
         CALL IZERO(INDX,NXXM)
      ENDIF
C
      NXY=NX*NY
      CALL MINMAX(TWMIN,TWMAX,TPP,NXY)
C
C
C     Map TPP from Gauss-Lobatto to Regular array, TWRK.
C
      CALL MAPGLR(TWRK,TPP,INTRP,NXX,NX,WKK)
C
C     Scale according to user's specs:
C
      IF (IFUSCL) THEN
         DO 35 I=1,NXX*NXX
            TWRK(I)=MAX(TWRK(I),0.0)
            TWRK(I)=MIN(TWRK(I),1.0)
   35    CONTINUE
      ENDIF

C
C     Quick kludge to avoid black holes, we truncate stuff which is
C     below TWMIN, to be slightly above TWMIN.  This will have to be 
C     modified when thresholding and set range are introduced.
C
      EPS=1.0E-04
      NXX2=NXX*NXX
      DO 40 I=1,NXX2
         IF (TWRK(I).LT.TWMIN) TWRK(I)=TWMIN+EPS
   40 CONTINUE
C
C ===================================================================
C     March through the colors/contour levels
C ===================================================================
C
C     The first break point is just slightly below zero. 
C     (Recall, data is normalized from 0 to 1 at this point.)
      EPS=1.0E-4
      BREAK1=-EPS
      IFOK=.TRUE.
C
      DBREAK=1.0
      IF (nlevs.GT.1) DBREAK=1.0/FLOAT(nlevs)
C
      IF (IFCVOL) THEN
C        If we're doing contour volumes....
         BREAK1=THRMIN
         IF (nlevs.GT.1) DBREAK=(THRMAX-THRMIN)/FLOAT(nlevs)
      ENDIF
c
c     Set contour level to specific value (sorry, no fixed level set yet)
      if (ktr.ne.1.and.cont_lev.lt.0) 
     $   dbreak = abs(cont_lev)/(wkmax-wkmin)
c
c
c     Specifying contour levels explicitly?
      if (if_spec_cont) then
         nlevs = spec_cont_nlev+1
         BREAK1=(spec_cont(0)-wkmin)/(wkmax-wkmin)
         spec_cont(nlevs)=1.1
c        write(6,*) 'if_spc:',nlevs,break1,spec_cont(nlevs)
      endif
c
c
c     write(6,*) 'Inside HEATC, nlevs:',nlevs,break1,dbreak
      DO 8000 ILEV=1,nlevs
       BREAK0=BREAK1
       BREAK1=BREAK0+DBREAK
       if (if_spec_cont) BREAK1=(spec_cont(ilev)-wkmin)/(wkmax-wkmin)
C
C      If tmin < br1  and br0 < tmax
C
C      If  br0 < tmax and tmin < br1  
C
c      write(6,*) 'if_spc2',ilev,break0,break1,twmin,twmax
       IF ( (TWMIN.LE.BREAK1.AND.TWMAX.GT.BREAK0)   .or.
     $      (ILEV.EQ.nlevs .and. TWMIN.GE.BREAK1) ) THEN
c
c      Use this option if you don't want black regions
c      (for Meelan,  pff 02/03/00)
c
cc     IF ( (nlevs.ge.1)                            .or.
cc   $      (ILEV.EQ.nlevs .and. TWMIN.GE.BREAK1) ) THEN
C
C        VAL is the current contour level.
C
         VAL=BREAK0
C
C        Establish contour array, CRS:  C(i,point) = r or s for each cross point
C        KTR=0 -- Line  contours
C        KTR=1 -- Solid contours
C
        JROW1=1
        JROW2=NXX
  100    CONTINUE
        CALL CONTOUR(VAL,TWRK,IFCRV,NXX,JROW1,JROW2,KTR)
C
C        Check for internal, undetected contours.
C
         IFOK=.TRUE.
        DO 200 JROW=JROW1+1,NXX-1
          JROFF=NXX*(JROW-1)+1
          IF (NXVR(JROW).GT.1) THEN
             CALL SORT(RXVR(1,JROW),    INDX,NXVR(JROW))
             CALL SWAP(RXVR(1,JROW),WKK,INDX,NXVR(JROW))
          ENDIF
          CALL CTRSCAN(IFOK,TWRK(JROFF),R1,RXVR(1,JROW),NXVR(JROW),NXX)
          IF (.NOT.IFOK) GOTO 201
  200    CONTINUE
  201    CONTINUE
C
        IF (IFOK) THEN 
C
           CALL PLTCNTR(XPP,YPP,ZPP,VAL,KTR)
        ELSE
C
C           Plot the stuff that's OK up to this point.
C
           CALL CONTOUR(VAL,TWRK,IFCRV,NXX,JROW1,JROW,KTR)
           CALL PLTCNTR(XPP,YPP,ZPP,VAL,KTR)
C
C           Shift pointer to the "new" starting column, and start over.
C
           JROW1=JROW
           GOTO 100
        ENDIF
       ENDIF
 8000 CONTINUE


C     Draw boundary if color fill boundary is requested:

c     write(6,*) ifcfbd,ifgngm,' ifcfbd,ifgngm' 
c     call prs('continue?$')
c     call res(ans,1)

      IF (IFCFBD) THEN
         K=1
c
c        white color-fill border always       pff   3/6/98
         if (ifrevbk) then
            CALL COLOR(0)
         else
            CALL COLOR(1)
         endif
c
         CALL MOVE3(XPP(K,1),YPP(K,1),ZPP(K,1))
         DO 8200 IEDG=1,4
            IF (IFCRV(IEDG).or.ifgngm) THEN
               I1=1
               IF (IEDG.EQ.1) KSTEP =  1
               IF (IEDG.EQ.2) KSTEP =  NX
               IF (IEDG.EQ.3) KSTEP = -1
               IF (IEDG.EQ.4) KSTEP = -NX
            ELSE
               I1=NX-1
               IF (IEDG.EQ.1) KSTEP =  NX-1
               IF (IEDG.EQ.2) KSTEP =  NX*(NX-1)
               IF (IEDG.EQ.3) KSTEP = -(NX-1)
               IF (IEDG.EQ.4) KSTEP = -NX*(NX-1)
            ENDIF
            DO 8100 I=2,NX,I1
               K=K+KSTEP
               CALL DRAW3(XPP(K,1),YPP(K,1),ZPP(K,1))
 8100    CONTINUE
 8200 CONTINUE
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine contour(val,wrk,ifcrv,ncol,jrow1,jrow2,ktr)
C
C     Returns a set of NCRS points in CRS(r,s)
C     accompanied by move or draw instructions.
C
      DIMENSION WRK(NCOL,NCOL)
      LOGICAL   IFCRV(4)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C     Two common blocks below are used in several subsequent routines.
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
C
      DIMENSION NEDGE(4)
      PARAMETER (NXXM4=4*NXXM)
      LOGICAL   IFMRKD(NXXM4)
      LOGICAL   IFEDGB,IFCRN,IFLINE
      LOGICAL   IFIN
C
      IFLINE=.TRUE.
      IF (KTR.EQ.0) IFLINE=.FALSE.
      EPS2=1.0E-8
      JCEL0=JROW1-1
      JCELMN=1
      ICELMN=1
      JCELMX=JROW2-JROW1
      ICELMX=NCOL-1
      NEDGE(1)=ICELMX
      NEDGE(2)=JCELMX
      NEDGE(3)=ICELMX
      NEDGE(4)=JCELMX
C
      CALL LFALSE(IFMRKD,NXXM4)
      CALL IZERO(NXVR,NXXM)
C
C     Edge march
C
      NCTR=0
      NCRS=1
      IMRK=0
      DO 8000 IEDGE=1,4
      DO 8000 IEE=1,NEDGE(IEDGE)
        IF (IEDGE.EQ.1) THEN
           ICEL   =  IEE
           I1     =  ICEL
           I2     =  I1+1
           JCEL   =  1
           J1     =  JCEL            +  JCEL0
           J2     =  J1
        ELSEIF (IEDGE.EQ.2) THEN
           ICEL   =  ICELMX
           I1     =  ICEL+1
           I2     =  I1
           JCEL   =  IEE
           J1     =  JCEL            +  JCEL0
           J2     =  J1+1
        ELSEIF (IEDGE.EQ.3) THEN
           ICEL   =  ICELMX-IEE+1
           I1     =  ICEL+1
           I2     =  I1-1
           JCEL   =  JCELMX
           J1     =  JCELMX+1        +  JCEL0
           J2     =  J1
        ELSEIF (IEDGE.EQ.4) THEN
           ICEL   =  1
           I1     =  ICEL
           I2     =  I1
           JCEL   =  JCELMX-IEE+1
           J1     =  JCEL+1          +  JCEL0
           J2     =  J1-1
        ENDIF
        IMRK=IMRK+1
C
C        Mark off the edge points which have been visited
C
        IF ((IFIN(WRK(I2,J2)).AND..NOT.IFMRKD(IMRK)) .OR.
     $      (IFIN(WRK(I1,J1)).AND.IEE.EQ.1.AND.IEDGE.EQ.1) ) THEN
C
C           A new contour has been found
C
           IFEDGB=.TRUE.
           IF (IEE.EQ.1.AND.IEDGE.EQ.1.AND.IFIN(WRK(I1,J1))) THEN
C              Point (1,1) is anomolous, if it's in.
              RC=R1(I1)
              SC=S1(J1)
              IFCRN=.TRUE.
           ELSE
              IF (IEDGE.EQ.1.OR.IEDGE.EQ.3) THEN
c                write(77,*) 'call 1'
                 CALL FNDRR(RC,R1(I1),R1(I2),WRK(I1,J1),WRK(I2,J2),VAL)
                 SC=S1(J1)
              ELSE
                 RC=R1(I1)
c                write(77,*) 'call 2'
                 CALL FNDRR(SC,S1(J1),S1(J2),WRK(I1,J1),WRK(I2,J2),VAL)
              ENDIF
              IFCRN=.FALSE.
           ENDIF
C
           RSTART=RC
           SSTART=SC
           IF (KTR.EQ.1) THEN
C              For C.Lines, we update counter later.
              IF (NCTR.GE.1) NCRS=NCRS+1
              CRS(1,NCRS)=RC
              CRS(2,NCRS)=SC
              CRS(3,NCRS)=KTR
              NCTR=NCTR+1
              ICTR(NCTR)=NCRS
              NCRS=NCRS+1
           ENDIF
C
C           Begin contouring
C                
           IEDGC=IEDGE
C
           NCMAX=6*ICELMX*JCELMX
C
           DO 1000 ICT=1,NCMAX
              ICEL1=ICEL
              JCEL1=JCEL
              IEDGC1=IEDGC
              CALL ADVCELL
     $         (RC,SC,IFEDGB,IFCRN,IEDGC,VAL,WRK,NCOL,ICEL,JCEL,JCEL0)
              DIST=(RSTART-RC)**2+(SSTART-SC)**2
C              WAS the last move an edge move?
              IF (IFEDGB) THEN
C
                 IF (IEDGC1.EQ.1) IMRC=ICEL1
                 IF (IEDGC1.EQ.2) IMRC=ICELMX+JCEL1
                 IF (IEDGC1.EQ.3) IMRC=2*ICELMX+  JCELMX-ICEL1+1
                 IF (IEDGC1.EQ.4) IMRC=2*ICELMX+2*JCELMX-JCEL1+1
                 IFMRKD(IMRC)=.TRUE.
C
C                 This is the section where we distinguish between
C                 curved elements, color fills, and contours.
C
                 IF (KTR.EQ.1) THEN
C                    Color fill
                    IF ( IFCRV(IEDGC1)
     $               .OR.(.NOT.IFCRN)
     $               .OR.(IEDGC.NE.IEDGC1)
     $               .OR.(DIST.LT.EPS2)     )  THEN
C
C                        Standard case, we advance with a move, if KTR=1
C
                       CRS(1,NCRS) = RC
                       CRS(2,NCRS) = SC
                       CRS(3,NCRS) = -KTR
                       NCRS=NCRS+1
                    ENDIF
                 ELSE
C                    Contour line, do not draw on element edges. Instead,
C                    update pointer (r,s) for 1st move of (next) new line.
                    IFLINE=.FALSE.
                    CRS(1,NCRS) = RC
                    CRS(2,NCRS) = SC
                    CRS(3,NCRS) = -KTR
                 ENDIF
              ELSE
C
C              We are in the middle of the domain, proceed as normal.
C
                 IF (.NOT.IFLINE) THEN
C                    IFLINE=false implies that we aren't currently
C                    drawing a line (for contour lines) so, we need
C                    to start a new one.
                    NCTR=NCTR+1
                    ICTR(NCTR)=NCRS
                    NCRS=NCRS+1
                    IFLINE=.TRUE.
                 ENDIF
                 CRS(1,NCRS) = RC
                 CRS(2,NCRS) = SC
                 CRS(3,NCRS) = KTR
                 NCRS=NCRS+1
C
C                 If the new edge is a 1 or a 3 (horizontal) then
C                 we need to mark it so we can check for undetected
C                 interior contours later:
C
                 IF (IEDGC.EQ.1) THEN
                    JROW=JCEL+JCEL0
                    NXVR(JROW)=NXVR(JROW)+1
                    RXVR(NXVR(JROW),JROW)=RC
                 ENDIF
                 IF (IEDGC.EQ.3) THEN
                    JROW=JCEL+1+JCEL0
                    NXVR(JROW)=NXVR(JROW)+1
                    RXVR(NXVR(JROW),JROW)=RC
                 ENDIF
C
              ENDIF
C
C              Check for loop closing
C
              IF (DIST.LT.EPS2) THEN
                 IF (KTR.EQ.1) THEN
C                    Enforce closing to be eXact, only for color fills!
                    NCRS=NCRS-1
                    CRS(1,NCRS)=RSTART
                    CRS(2,NCRS)=SSTART
                 ENDIF
                 GOTO 1001
              ENDIF
 1000       CONTINUE
 1001       CONTINUE
        ENDIF
C
C     Resume edge march
C
      IFMRKD(IMRK)=.TRUE.
 8000 CONTINUE
C
      IF (KTR.EQ.0) NCRS=NCRS-1
      ICTR(NCTR+1)=NCRS
      return
      END
c-----------------------------------------------------------------------
      subroutine advcell
     $  (RC,SC,IFEDGB,IFCRN,IEDGC,VAL,WRK,NCOL,ICEL,JCEL,JCEL0)
C
C     Given a cell number, denoted by (ICEL,JCEL), and an inbound
C     edge (IEDGC=1,2,3,4) where contour crosses into CELL, this
C     routine finds the exiting point, denoted by (RC,SC).  In addition,
C     (IEDGC,ICEL,JCEL) is updated to represent the new edge.
C
C     If the associated "move" (ie, the line connecting (RC,SC) with the
C     returned (RC',SC') ), is along the edge of the BOX of CELLS, then
C     IFEDGB is returned as true.
C
C     If the associated move is an edge move which includes the corner
C     or vertex of the new cell, IFCRN is returned as true.  This is 
C     useful in discriminating edge cell sides which require 2 moves, eg,
C     as in the cell denoted by "o" below:
C                     /
C         +    +    +/   +
C                   /
C                o /
C     ----+----+--/ +    +
C
      DIMENSION WRK(NCOL,NCOL)
      LOGICAL   IFEDGB,IFCRN
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
C     Two common blocks below are used in several subsequent routines.
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
      LOGICAL   IFIN
      DIMENSION IVEC(4),JVEC(4),NRMOUT(2,4)
      SAVE IVEC,JVEC,NRMOUT
      DATA IVEC / 0 , 1 , 1 , 0 /
      DATA JVEC / 0 , 0 , 1 , 1 /
      DATA NRMOUT / 0,-1,1,0,0,1,-1,0/
C
C
      IFEDGB=.FALSE.
      NEWEDG=1
C
      ICRN=IEDGC
      ICRN=MOD1(ICRN,4)
      I0 = ICEL + IVEC(ICRN)
      J0 = JCEL + JVEC(ICRN) + JCEL0
C
C===================================================
C     MAIN FOUR-EDGE LOOP
C===================================================
C
      DO 1000 ICORNR=1,4
C
C        Now we check the 4 corners of this cell 
C        until we find one that is IN.
C
        ICRN=IEDGC+ICORNR
        ICRN=MOD1(ICRN,4)
        I = ICEL + IVEC(ICRN)
        J = JCEL + JVEC(ICRN) + JCEL0
        IF (IFCRN.AND.ICORNR.EQ.1.AND..NOT.IFIN(WRK(I,J))) THEN
C
C           We've got a partial edge.
C
           IF (IEDGC.EQ.1.OR.IEDGC.EQ.3) THEN
c                write(77,*) 'call 3'
              CALL FNDRR(RC,R1(I0),R1(I),WRK(I0,J),WRK(I,J),VAL)
              SC=S1(J)
           ELSE
              RC=R1(I)
c                write(77,*) 'call 4'
              CALL FNDRR(SC,S1(J0),S1(J),WRK(I,J0),WRK(I,J),VAL)
           ENDIF
           IFEDGB=.TRUE.
           IFCRN =.FALSE.
           return
        ENDIF
C
        IF (IFIN(WRK(I,J))) THEN
           IF (ICORNR.EQ.1) THEN
C
C              We're on an edge, advance to next cell.
C
              RC=R1(I)
              SC=S1(J)
              NXTEDG=ICRN
              ICEL=ICEL+NRMOUT(1,NXTEDG)
              JCEL=JCEL+NRMOUT(2,NXTEDG)
C              Check cell advance past boundaries.
              IF (ICEL.LT.ICELMN.OR.ICEL.GT.ICELMX.OR.
     $             JCEL.LT.JCELMN.OR.JCEL.GT.JCELMX ) THEN
                 IEDGC=NXTEDG
                 ICEL = MAX(ICEL,ICELMN)
                 ICEL = MIN(ICEL,ICELMX)
                 JCEL = MAX(JCEL,JCELMN)
                 JCEL = MIN(JCEL,JCELMX)
              ENDIF
              IFEDGB=.TRUE.
              IFCRN =.TRUE.
C
           ELSE
C           Standard case:
C
C              Find cross point
C
              IF (NEWEDG.EQ.1.OR.NEWEDG.EQ.3) THEN
c                write(77,*) 'call 5'
                 CALL FNDRR(RC,R1(I0),R1(I),WRK(I0,J),WRK(I,J),VAL)
                 SC=S1(J)
              ELSE
                 RC=R1(I)
c                write(77,*) 'call 6'
                 CALL FNDRR(SC,S1(J0),S1(J),WRK(I,J0),WRK(I,J),VAL)
              ENDIF
C
C              Advance the cell
C              
              ICEL=ICEL+NRMOUT(1,NEWEDG)
              JCEL=JCEL+NRMOUT(2,NEWEDG)
              IEDGC=NEWEDG+2
              IEDGC=MOD1(IEDGC,4)
C              Check cell advance past boundaries.
              IF (ICEL.LT.ICELMN.OR.ICEL.GT.ICELMX.OR.
     $             JCEL.LT.JCELMN.OR.JCEL.GT.JCELMX ) THEN
                 IEDGC=NEWEDG
                 ICEL = MAX(ICEL,ICELMN)
                 ICEL = MIN(ICEL,ICELMX)
                 JCEL = MAX(JCEL,JCELMN)
                 JCEL = MIN(JCEL,JCELMX)
              ENDIF
              IFEDGB=.FALSE.
              IFCRN =.FALSE.
           ENDIF
           return
C
C           Else, check if next corner is in.
C
        ENDIF
        NEWEDG=ICRN
        I0=I
        J0=J
 1000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine ctrscan(ifok,wrk,r,rxvr,nxvr,n)
C
C     Scan the contoured domain for any new interior contours not 
C     previously detected.
C
      DIMENSION WRK(N),R(N),RXVR(N)
      LOGICAL IFOK,IFIN,IFSTAT
C
      IFOK= .FALSE.
      IFSTAT=IFIN(WRK(1))
C
      IF (NXVR.EQ.0) THEN
         DO 10 I=2,N-1
c          IF (IFSTAT.NE.IFIN(WRK(I))) return
c          f90 compatible...
           IF ( (ifstat.and..not.ifin(wrk(i))) .or.
     $          (.not.ifstat.and.ifin(wrk(i))) ) return
   10    CONTINUE
      ELSE
C
        IXVR=1
        DO 20 I=2,N-1
C           Crossover?
           IF (IXVR.LE.NXVR.AND.R(I).GT.RXVR(IXVR)) THEN
               IFSTAT=.NOT.IFSTAT
               IXVR=IXVR+1
           ENDIF
c          IF (IFSTAT.NE.IFIN(WRK(I))) return
c          f90 compatible...
           IF ( (ifstat.and..not.ifin(wrk(i))) .or.
     $          (.not.ifstat.and.ifin(wrk(i))) ) return
   20    CONTINUE
      ENDIF
C
C     If exited ok, then no new internal contours detected.
      IFOK = .TRUE.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine pltcntr(xpp,ypp,zpp,val,ktr)
C
C     Plot a contour given by CRS which denotes the Contour points
C     in the (R,S) plane.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION XPP(NX,NY),YPP(NX,NY),ZPP(NX,NY)
C
C     Two common blocks below are used in several subsequent routines.
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
      COMMON /ccdiag/ iee,iii
C
C     Begin panel fill if KTR=1 for color fills
      IF (KTR.EQ.1) THEN
        ICLR=ICOLOR1(VAL)
        CALL FILLP(-ICLR)
        CALL COLOR(ICLR)
      ENDIF
C
      DO 1000 ICT=1,NCTR
        II =0
        ICFRST=ICTR(ICT)
        ICLAST=ICTR(ICT+1)-1
        IF (ICT.EQ.NCTR) ICLAST=ICTR(ICT+1)
        IF (KTR.EQ.1) CALL BEGINPP
        DO 200 ICRS=ICFRST,ICLAST
           II=II+1
C           More sophisticated (1D) interpolation will go here later.
           iii=ii
           CALL EVALSC2(XYZ(1,II),XPP,CRS(1,ICRS))
           CALL EVALSC2(XYZ(2,II),YPP,CRS(1,ICRS))
           IF (IF3D) THEN
              CALL EVALSC2(XYZ(3,II),ZPP,CRS(1,ICRS))
           ELSE
              XYZ(3,II)=0.0
           ENDIF
C
           XYZ(4,II)=VAL
  200    CONTINUE
c
c       write(6,6) iee,iii,(xyz(1,j),xyz(2,j),xyz(3,j),j=1,2)
c   6   format(2i8,4x,1p3e12.4,/,20x,3e12.4)
c
        IF (KTR.EQ.1) CALL MOVE(xiso(xyz(1,1),xyz(2,1),xyz(3,1))
     $                         ,yiso(xyz(1,1),xyz(2,1),xyz(3,1)))
        CALL VDRAW3(XYZ,II)
        IF (KTR.EQ.1) CALL ENDP
 1000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine evalsc2( x0 , scal , rrl )
C
C     Evaluate a scalar, SCAL, at position RRL and return the result in X0.
C     RRL is assumed to be a 2 dimensional (R,S) point, and SCAL is assumed
C     to be an (NX,NY) field.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION SCAL(1)
      DIMENSION RRL(3)
      COMMON  /CTMP5/ HR(NXM),HS(NXM)
     $               ,HHH(NXM,NXM)
      COMMON /ccdiag/ iee,iii
C
      REAL    ICALLD
      SAVE    ICALLD
      REAL    RLXOLD,RLYOLD,RLZOLD
      SAVE    RLXOLD,RLYOLD,RLZOLD
      DATA    ICALLD /0/
      DATA    RLXOLD,RLYOLD,RLZOLD/3*0.0/
C
      SUM = 0.0
C
      IF (RRL(1).NE.RLXOLD.OR.RRL(2).NE.RLYOLD
     $   .OR.ICALLD.EQ.0) THEN
        ICALLD = 1
        RLXOLD = RRL(1)
        RLYOLD = RRL(2)
        CALL HLEGN(HR,RRL(1),ZPTS,NX)
        CALL HLEGN(HS,RRL(2),ZPTS,NY)
        IJK=0
        DO 200 J=1,NY
        DO 200 I=1,NX
           IJK=IJK+1
           HHH(IJK,1)=HR(I)*HS(J)
           SUM=SUM+SCAL(IJK)*HHH(IJK,1)
 200     CONTINUE
      ELSE
C
C     Reevaluate a new scalar at an old point.
C
        NXY=NX*NY
        DO 300 I=1,NXY
           SUM=SUM+SCAL(I)*HHH(I,1)
  300    CONTINUE
      ENDIF
      X0 = SUM
      return
      END
c-----------------------------------------------------------------------
      subroutine fndrr(rc,r0,r1,v0,v1,v)
C
        RC = R0 + ( (V-V0)/(V1-V0) ) * (R1-R0)
C
        RMX= MAX(R0,R1)
        RMN= MIN(R0,R1)
        RC = MIN(RMX,RC)
        RC = MAX(RMN,RC)
C
      return
      END
c-----------------------------------------------------------------------
      LOGICAL FUNCTION IFIN(VAL)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
C
      IF (VAL.GT.BREAK0) THEN
        IFIN=.TRUE.
      ELSE
        IFIN=.FALSE.
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine beginpp
C     Begins panel for fill  NO BOUNDARY DISPLAY
      INCLUDE 'devices.inc'
      COMMON /FILL/ ICFILL,IFILL,NFILL,XFILL(210),YFILL(210)
        ICFILL=0
        IFILL = 1
        NFILL = 0
      return
      END
c-----------------------------------------------------------------------
      subroutine minmax(xmn,xmx,x,n)
      DIMENSION X(1)
      XMN=X(1)
      XMX=X(1)
      DO 100 I=1,N
        XMN=MIN(XMN,X(I))
        XMX=MAX(XMX,X(I))
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine lfalse (ifa,n)
      LOGICAL IFA(1)
      DO 100 I=1,N
      IFA(I)=.FALSE.
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine heatf(tpp,xpp,ypp,zpp,ifcrv,inrm)
C
C     This is the new fishnet routine.
C     10-10-90 pff (upgraded to include 3D, 1-12-91)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION TPP(NX,NY),XPP(NX,NY),YPP(NX,NY),ZPP(NX,NY)
      LOGICAL   IFCRV(4)
C     Note: This CTMP1x has nothing to do with CTMP1 in HEATC
      PARAMETER (NXG2=NXXM*NXXM)
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      COMMON /CTMP1x/ TWORK(NXG2),XWORK(NXG2),YWORK(NXG2),ZWORK(NXG2)
     $              ,WRK(NXG2),INTRP(NXXM,NXM,2),INTRP2(NXXM,NXM,2)
     $              ,VEC(3),VEC1(3),VEC2(3),VEC3(3)
      REAL INTRP,INTRP2
C
c     call pris(inrm,' enter new inrm:$')
c     call rei (inrm)
c
      NXX=abs(NXGRID)
      NYY=NXX
      IF (NXX.GT.NXXM) THEN
         WRITE(S,10) NXXM
   10    FORMAT(2X,'Resetting resolution on fishnet to',I3,'.')
         CALL PRS(S//'$')
         if (nxgrid.gt.0) then
            NXGRID=NXXM
         else
            NXGRID=-NXXM
         endif
      ENDIF
C
C     Force reevaluation of INTRP by changing NXX,NX for this call.
      IF (ILGRNG.NE.3) CALL SETIX(INTRP,NXX,NX,WRK)
C     Set ILGRNG=3 so that we know that the Lagrangian (streamline) data
C     in COMMON /CTMP1x/ has been clobbered and will need to be recomputed.
      ILGRNG=3
C
C     2D fishnet grid.
C
      IF (TTMAX*TTMIN.LT.0.0) THEN
         SHIFT=TTMIN/(TTMAX-TTMIN)
      ELSE
         SHIFT=0.0
      ENDIF
      if (nxgrid.gt.0) then
         CALL MAPGLR(XWORK,XPP,INTRP,NXX,NX,WRK)
         CALL MAPGLR(YWORK,YPP,INTRP,NXX,NX,WRK)
         CALL MAPGLR(ZWORK,ZPP,INTRP,NXX,NX,WRK)
         CALL MAPGLR(TWORK,TPP,INTRP,NXX,NX,WRK)
      else
         CALL MAPGLG(XWORK,XPP,INTRP,NXX,NX,WRK)
         CALL MAPGLG(YWORK,YPP,INTRP,NXX,NX,WRK)
         CALL MAPGLG(ZWORK,ZPP,INTRP,NXX,NX,WRK)
         CALL MAPGLG(TWORK,TPP,INTRP,NXX,NX,WRK)
      endif
C
C     Scale according to user's specs:
C
      IF (IFUSCL) THEN
         DO 35 I=1,NXX*NXX
            TWORK(I)=MAX(TWORK(I),0.0)
            TWORK(I)=MIN(TWORK(I),1.0)
   35    CONTINUE
      ENDIF
C
      DO 20 I=1,NXX
      DO 20 J=1,NYY
         IJ = NXX*(J-1)+I
         XGB=XWORK(IJ)
         YGB=YWORK(IJ)
         ZGB=ZWORK(IJ)
         TGB=TWORK(IJ)
         IF (IF3D) THEN
            IM=MAX(I-1,1)
            IP=MIN(I+1,NXX)
            JM=MAX(J-1,1)
            JP=MIN(J+1,NYY)
            I1 = NXX*(J-1)+IM
            I2 = NXX*(J-1)+IP
            I3 = NXX*(JM-1)+I
            I4 = NXX*(JP-1)+I
            VEC1(1)=XWORK(I1)
            VEC1(2)=YWORK(I1)
            VEC1(3)=ZWORK(I1)
            VEC2(1)=XWORK(I2)
            VEC2(2)=YWORK(I2)
            VEC2(3)=ZWORK(I2)
            CALL SUB3(VEC1,VEC2,VEC1,3)
            VEC2(1)=XWORK(I3)
            VEC2(2)=YWORK(I3)
            VEC2(3)=ZWORK(I3)
            VEC3(1)=XWORK(I4)
            VEC3(2)=YWORK(I4)
            VEC3(3)=ZWORK(I4)
            CALL SUB3(VEC2,VEC3,VEC2,3)
            CALL CROSS (VEC3,VEC1,VEC2)
            CALL NORM3D(VEC3)
            XGB = XGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(1))
            YGB = YGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(2))
            ZGB = ZGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(3))
         ELSE
            ZGB = ZGB+(TGB+SHIFT) * HFISH
         ENDIF

         call findl_new(tgb,levb,tmin,tmax)
c        write(6,*) levb,tgb,tmin,tmax,' levb'

         CALL COLOR(LEVB+1)
         IF (J.EQ.1) THEN
            CALL MOVE3(XGB,YGB,ZGB)
         ELSE
            call draw3(xgb,ygb,zgb)
c           if (ziso(xgb,ygb,zgb).gt.0) call draw3(xgb,ygb,zgb) ! QUICK BCKF CULL
         ENDIF
   20 CONTINUE
      DO 30 J=1,NYY
      DO 30 I=1,NXX
         IJ = NXX*(J-1)+I
         XGB=XWORK(IJ)
         YGB=YWORK(IJ)
         ZGB=ZWORK(IJ)
         TGB=TWORK(IJ)
         IF (IF3D) THEN
            IM=MAX(I-1,1)
            IP=MIN(I+1,NXX)
            JM=MAX(J-1,1)
            JP=MIN(J+1,NYY)
            I1 = NXX*(J-1)+IM
            I2 = NXX*(J-1)+IP
            I3 = NXX*(JM-1)+I
            I4 = NXX*(JP-1)+I
            VEC1(1)=XWORK(I1)
            VEC1(2)=YWORK(I1)
            VEC1(3)=ZWORK(I1)
            VEC2(1)=XWORK(I2)
            VEC2(2)=YWORK(I2)
            VEC2(3)=ZWORK(I2)
            CALL SUB3(VEC1,VEC2,VEC1,3)
            VEC2(1)=XWORK(I3)
            VEC2(2)=YWORK(I3)
            VEC2(3)=ZWORK(I3)
            VEC3(1)=XWORK(I4)
            VEC3(2)=YWORK(I4)
            VEC3(3)=ZWORK(I4)
            CALL SUB3(VEC2,VEC3,VEC2,3)
            CALL CROSS (VEC3,VEC1,VEC2)
            CALL NORM3D(VEC3)
            XGB = XGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(1))
            YGB = YGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(2))
            ZGB = ZGB+FLOAT(INRM)*( (TGB+SHIFT) * HFISH * VEC3(3))
         ELSE
            ZGB = ZGB+(TGB+SHIFT) * HFISH
         ENDIF
         CALL FINDL(TGB,LEVB,TMIN,TMAX)
         CALL COLOR(LEVB+1)
         IF (I.EQ.1) THEN
            CALL MOVE3(XGB,YGB,ZGB)
         ELSE
            call draw3(xgb,ygb,zgb)
c           if (ziso(xgb,ygb,zgb).gt.0) call draw3(xgb,ygb,zgb) ! QUICK BCKF CULL
         ENDIF
   30 CONTINUE
C
C     Draw boundary if color fill boundary is requested:
C
      IF (IFCFBD) THEN
         K=1
         CALL COLOR(1)
         CALL MOVE3(XWORK(K),YWORK(K),ZWORK(K))
         DO 200 IEDG=1,4
            IF (IFCRV(IEDG)) THEN
               I1=1
               IF (IEDG.EQ.1) KSTEP =  1
               IF (IEDG.EQ.2) KSTEP =  NXX
               IF (IEDG.EQ.3) KSTEP = -1
               IF (IEDG.EQ.4) KSTEP = -NXX
            ELSE
               I1=NXX-1
               IF (IEDG.EQ.1) KSTEP =  NXX-1
               IF (IEDG.EQ.2) KSTEP =  NXX*(NXX-1)
               IF (IEDG.EQ.3) KSTEP = -(NXX-1)
               IF (IEDG.EQ.4) KSTEP = -NXX*(NXX-1)
            ENDIF
            DO 100 I=2,NXX,I1
               K=K+KSTEP
               CALL DRAW3(XWORK(K),YWORK(K),ZWORK(K))
 100     CONTINUE
 200  CONTINUE
      ENDIF
C
C     End of 2D fishnet grid.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine setrstp(ivec)
C     Set up list of candidate RST planes
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      DIMENSION VEC(3),VEC1(3),VEC2(3),VEC3(3)
      CHARACTER*1 YESNO
      LOGICAL IFANY,IFTMP
      character*70 fname
C
      CALL PRS
     $('Enter xyz point near surface to be plotted (-1,-1,-1=file):$') 
      CALL RERRR(XPT1,YPT1,ZPT1)
      if (xpt1.eq.-1.0.and.ypt1.eq.-1.0.and.zpt1.eq.-1.0) then
         call prs('Name of file containing points?$')
         call blank(fname,70)
         call res(fname,70)
         open (unit=33,file=fname,err=999)
         do k=1,99999
            read(33,*,err=99,end=99) xpt1,ypt1,zpt1,xnrm1,ynrm1,znrm1
            ivv = 0
            if (xnrm1.eq.0.and.ynrm1.eq.0.and.znrm1.eq.0) ivv=ivec
            call setrstp_act(ivv,xpt1,ypt1,zpt1,xnrm1,ynrm1,znrm1)
         enddo
   99    continue
         close(unit=33)
         return
      else
         xnrm1 = 0.
         ynrm1 = 0.
         znrm1 = 0.
         call setrstp_act(ivec,xpt1,ypt1,zpt1,xnrm1,ynrm1,znrm1)
         return
      endif
c
  999 continue
      call prs('Could not open file.$')
      close(unit=33)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine setrstp_ACT(ivec,XPT1,YPT1,ZPT1,XNRM1,YNRM1,ZNRM1)
C     Set up list of candidate RST planes
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      DIMENSION VEC(3),VEC1(3),VEC2(3),VEC3(3)
      CHARACTER*1 YESNO
      LOGICAL IFANY,IFTMP
C
C     Find the element containing this point
C
      CALL INTERP(VAL1,XPT1,YPT1,ZPT1,IE,WORK,ierr)
C
C     Exception handling:
      IF (IE.EQ.0) THEN
         WRITE(S,20) XPT1,YPT1,ZPT1
         CALL PRS(S)
   20    FORMAT(2X,'Point (',G10.3,',',G10.3,',',G10.3,') is not in '
     $            ,'the domain.$')
         return
      ENDIF
C
C     Now that element is found, find what type of plane is closest: R,S, or T?
C
      CALL RZERO(VEC,3)
      IF (IVEC.GE.1.AND.IVEC.LE.3) THEN
         VEC(IVEC)=1.0
      ELSEif (xnrm1.eq.0.and.ynrm1.eq.0.and.znrm1.eq.0) then
         CALL PRS
     $('Enter approximate normal vector of the surface at this point.$')
         CALL PRS
     $('  (x_n,y_n,z_n):$') 
         CALL RERRR(VEC(1),VEC(2),VEC(3))
         CALL NORM3D(VEC)
      ELSE
         vec(1) = xnrm1
         vec(2) = ynrm1
         vec(3) = znrm1
         CALL NORM3D(VEC)
         write(6,44) xpt1,ypt1,zpt1,(vec(k),k=1,3)
   44    format(3e12.4,2x,3f9.4,'  VEC')
      ENDIF
C
C     Does this correspond most closely to an R, S, or T plane?
C
      RLMAX=0.0
      DO 30 IPLN=1,6 
         CALL SUB3(VEC1,XYZQ(1,2,IPLN,IE),XYZQ(1,1,IPLN,IE),3)
         CALL SUB3(VEC2,XYZQ(1,4,IPLN,IE),XYZQ(1,3,IPLN,IE),3)
         CALL CROSS(VEC3,VEC1,VEC2)
         CALL NORM3D(VEC3)
         RLNGTH = DOTPROD(VEC,VEC3)
         RLNGTH = ABS(RLNGTH)
         IF (RLNGTH.GT.RLMAX) THEN
            IJKPLN  = IPLN
            RLMAX   = RLNGTH
            DXORD=VEC3(1) 
            DYORD=VEC3(2) 
            DZORD=VEC3(3) 
         ENDIF
   30 CONTINUE
      VEC3(1)=DXORD
      VEC3(2)=DYORD
      VEC3(3)=DZORD
C
C     Diagnostics
c     IFTMP=IFHARD
c     IFHARD=.FALSE.
c     CALL COLOR(5)
c     CALL ARROW4S(XPT1,YPT1,ZPT1,DXORD,DYORD,DZORD)
c     CALL COLOR(1)
C
      DXORD=VEC(1)
      DYORD=VEC(2)
      DZORD=VEC(3)
c     CALL ARROW4S(XPT1,YPT1,ZPT1,DXORD,DYORD,DZORD)
c     IFHARD=IFTMP
c     write(S,40) IE,IJKPLN
c     CALL PRS(S)
c  40 FORMAT(2X,'IE=',I5,' PLANE=',I1,'.$')
C
C     Find the plane corresponding to the chosen point and normal.
C
      NXY =NX*NY
      NXYZ=NX*NY*NZ
      DMIN=10.0E15
      do 100 ie=1,nel
      DO 100 I=1,NXYZ
         IEOFF=NXYZ*(IE-1)+I
         DIST=(XP(IEOFF)-XPT1)**2+(YP(IEOFF)-YPT1)**2+
     $        (ZP(IEOFF)-ZPT1)**2
         IF (DIST.LT.DMIN) THEN
            IEMIN =IE
            IJKMIN=I
            DMIN=DIST
         ENDIF
  100 CONTINUE

      IE = IEMIN
      CALL DECOD(IXP,IYP,IZP,IDUM,IJKMIN,NX,NY,NZ)

c     i  = ijkmin
c     ieoff=nxyz*(ie-1)+i
c     write(s,101) ixp,iyp,izp,ie,ijkmin
c    4             ,xp(ieoff),yp(ieoff),zp(ieoff)
c 101 format(' Point is at:',3I4,i6,i8,1p3e12.5,'$')
c     call prs(s)

      IF (IJKPLN.LE.2) IPLAN=IXP
      IF (IJKPLN.GT.2) IPLAN=IYP
      IF (IJKPLN.GT.4) IPLAN=IZP
C     Set IJKPLN according to actual value of IPLAN
      NXT=2*IPLAN-1
      IF (NXT.LT.NX) IJKPLN=2*((IJKPLN-1)/2)+1
      IF (NXT.GT.NX) IJKPLN=2*((IJKPLN-1)/2)+2

c     write(7,40) IE,IJKPLN,IPLAN
c     write(S,40) IE,IJKPLN,IPLAN
c     CALL PRS(S)
c  40 format(2x,'ie=',i5,' plane=',2i2,'. Continue?$')
c     call res(S,1)

C
C===============================================
C     Start generating LIST of plotting planes
C===============================================
C
C     Initialize pointers to zero
      DO 105 I=1,24*NELM
         XYZQ(5,I,1,1)=0.0
  105 CONTINUE
C
c     NLSTP=1                    pff 12/21/96
      NLSTP=NLSTP+1
      LIST(NLSTP)=6*NX*(IE-1)+NX*(IJKPLN-1)+IPLAN
      IMD=MOD(IJKPLN,2)
C
C     Set the sign of this plane according to requested normal vector:
C
      RLNGTH = DOTPROD(VEC,VEC3)
      INRM=1
      IF (RLNGTH.LT.0.0) INRM=-1
      LIST(NLSTP)=INRM*LIST(NLSTP)
C
c     nid=9
      DO 110 I=1,4
         NEIGHI=INT(XYZQ(4,I,IJKPLN,IE))
c            write(6,*) nid,'s neigh:',i,ijkpln,ie,neighi
         IF (NEIGHI.EQ.0) THEN
            XYZQ(5,I,IJKPLN,IE)=2.0
         ELSE
            RN=FLOAT(NEIGHI)
            XYZQ(5,I,IJKPLN,IE)=SIGN(1.0,RN)
         ENDIF
  110 CONTINUE
C
C=======================================
C     Find all adjoining planes
C=======================================
C
      ICOUNT=0
  200 CONTINUE
      IFANY=.FALSE.
      ICOUNT=ICOUNT+1
      DO 500 IE=1,NEL
      DO 500 IPLN=1,6
      DO 500 IQ=1,4
         IF (ABS(XYZQ(5,IQ,IPLN,IE)).EQ.1.0) THEN
            IFANY=.TRUE.
            NEIGHI=INT(XYZQ(4,IQ,IPLN,IE))
            NEIGHA=ABS(NEIGHI)
c     CALL DECOD(JQ,JPLN,JEg,IDUM,NEIGHA,4,6,NEL)
c     write(6,201) nid,iq,ipln,ie,neighi,jq,jpln,jeg,je,neigha
c 201 format(i2,'xyq',9i6)
            IF (XYZQ(5,NEIGHA,1,1).EQ.0.0) THEN
C              The neighboring plane has Not been set.
               CALL DECOD(JQ,JPLN,JE,IDUM,NEIGHA,4,6,NEL)
               DO 400 J=1,3
                  JJ=JQ+J
                  JJ=MOD1(JJ,4)
                  NEIGHJ=INT(XYZQ(4,JJ,JPLN,JE))
                  IF (NEIGHJ.EQ.0) THEN
                     XYZQ(5,JJ,JPLN,JE)=2.0
                  ELSE
                     RN=FLOAT(NEIGHJ)
                     XYZQ(5,JJ,JPLN,JE)=
     $                   XYZQ(5,IQ,IPLN,IE)*SIGN(1.0,RN)

                  ENDIF
  400          CONTINUE
C
C              New neighbor points all set, now add neighbor plane to LIST.
               JMD=MOD(JPLN,2)
               IF (IMD.EQ.JMD) THEN
                  JPLAN=IPLAN
               ELSE
                  JPLAN=NX+1-IPLAN
               ENDIF
               NLSTP=NLSTP+1
               LIST(NLSTP)=6*NX*(JE-1)+NX*(JPLN-1)+JPLAN
c     write(6,*) nlstp,list(nlstp),je,jpln,jplan,nx
               LIST(NLSTP)=LIST(NLSTP)*SIGN(1.0,XYZQ(5,IQ,IPLN,IE))
C              Flip according to initial plane
               LIST(NLSTP)=INRM*LIST(NLSTP)
c              write(S,401) NLSTP,JE,JPLN,JPLAN,LIST(NLSTP)
c 401          format(' Found',i3,'th plane: JE,JP,PLANE:',4I6,'$')
c              CALL PRS(S)
C
C              Remove pointers and exit
               XYZQ(5,NEIGHA,1,1)=2.0
            ENDIF
            XYZQ(5,IQ,IPLN,IE)=2.0
         ENDIF
  500 CONTINUE
      IF (IFANY) GOTO 200
C
C     { LIST is complete, sort by zbuff. (? or perhaps after rotation?)}
C     LIST is complete, set normals according to the following:
C
C     If plane is a Y-plane, i.e. 3 or 4, then it must be flipped (?).
C
      write(s,501) nlstp
      call prs(s)
  501 format(2x,'num planes:',I5,'$')
c     DO 9000 I=1,NLSTP
c        LISTA=ABS(LIST(I))
c        CALL DECOD(IPLANE,IPLN,IEG,IDUM,LISTA,NX,6,NELM)
c        INID=9
c         NID=9
c        INID=GLLNID(IEG)
c        IE  =GLLEL (IEG)
c        write(s,9001) NID,INID,IEG,IPLN,IPLANE,IE,LISTA
c        write(7,9001) NID,INID,IEG,IPLN,IPLANE,IE,LISTA
c        call prs(s)
c9001    FORMAT(I3,'found planes',6i8,'$')
c9000 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine aarrow(xpp,ypp,zpp,xvec,yvec,zvec,nxx)
C
C     An array of arrows, with plane projections
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /CTMP1x/ VEC(3),VEC1(3),VEC2(3),VEC3(3)
      DIMENSION XPP (NXX,NXX),YPP (NXX,NXX),ZPP (NXX,NXX)
      DIMENSION XVEC(NXX,NXX),YVEC(NXX,NXX),ZVEC(NXX,NXX)
C
      NYY=NXX
      DO 30 J=1,NYY
      DO 30 I=1,NXX
         IF (IF3D) THEN
            IM=MAX(I-1,1)
            IP=MIN(I+1,NXX)
            JM=MAX(J-1,1)
            JP=MIN(J+1,NYY)
C           Find unit normal at this (i,j) point:
c
            VEC1(1)=XPP(IP,J)-XPP(IM,J)
            VEC1(2)=YPP(IP,J)-YPP(IM,J)
            VEC1(3)=ZPP(IP,J)-ZPP(IM,J)
c
            VEC2(1)=XPP(I,JP)-XPP(I,JM)
            VEC2(2)=YPP(I,JP)-YPP(I,JM)
            VEC2(3)=ZPP(I,JP)-ZPP(I,JM)
c
            CALL CROSS (VEC3,VEC1,VEC2)
            CALL NORM3D(VEC3)
c
c           IF (.NOT.IFCFBD) THEN
C              quickie backface culling of arrows
c              IF (ZISO(VEC3(1),VEC3(2),VEC3(3)).LT.0.0) GOTO 29
c              doesn't quite work yet....
c           ENDIF
            VEC1(1)=XVEC(I,J)
            VEC1(2)=YVEC(I,J)
            VEC1(3)=ZVEC(I,J)
            ALPHA=DOTPROD(VEC1,VEC3)
            CALL CMULT(VEC3,ALPHA,3)
            CALL SUB2(VEC1,VEC3,3)
C           We'll call a new Arrow routine which plots a vector
C           and the projection onto the plane determined by the
C           unit normal at that point
            CALL AROW3P(XPP(I,J),YPP(I,J),ZPP(I,J)
     $        ,XVEC(I,J),YVEC(I,J),ZVEC(I,J),VEC1,SARROW)
c        write(6,6) i,j,xpp(i,j),ypp(i,j),xvec(i,j),yvec(i,j)
c   6    format(2i5,4g12.4)
         ENDIF
   29 CONTINUE
   30 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine arow3p(x,y,z,dxbar,dybar,dzbar,vecp,sarrow)
      COMMON/SCALE/XFAC,YFAC,XZERO,YZERO
      DIMENSION VECP(3)
c
      real uu(3),d(3),nhat(3,3)
      save nhat
      data nhat / 1,0,0  ,  0,1,0  ,  0,0,1  /
c
C     Now draw shadow on plane at Zbase (without +DZ)
C     Sort of pale orange in heat map
C     FIX COLOR(WHITE USUALLY, YELLOOW FOR AXIS.  <FIX 3-D ARROWHEAD)!!??
C
      IF (DXBAR.EQ.0.0.AND.DYBAR.EQ.0.0.AND.DZBAR.EQ.0.0) return
C
C     Draw base at Z=Zplane (or whichever plane you're plotting on)
      CALL PENW(1)
      CALL COLOR(12)
      DX = X+VECP(1)*SARROW
      DY = Y+VECP(2)*SARROW
      DZ = Z+VECP(3)*SARROW
      CALL MOVE3(X,Y,Z)
      CALL DRAW3(DX,DY,DZ)
C
C     Draw 3-dimensional arrow
      CALL PENW(3)
      CALL COLOR(1)
      DX = DXBAR*SARROW
      DY = DYBAR*SARROW
      DZ = DZBAR*SARROW
      CALL MOVE3(X,Y,Z)
c
      xh = x+dx
      yh = y+dy
      zh = z+dz
      CALL DRAW3(xh,yh,zh)
c
c     New arrowheads  (pff 11/24/99)
c
      dn = dx*dx + dy*dy + dz*dz
      if (dn.gt.0) dn = sqrt(dn)
      dn = 0.04*dn
c
      dxn = -.14*dx
      dyn = -.14*dy
      dzn = -.14*dz
c
c     Origin of plane intersecting arrow
c
      o1 = xh+dxn
      o2 = yh+dyn
      o3 = zh+dzn
c
c     Find projection of arrow onto x y and z planes
c
      d(1) = dx
      d(2) = dy
      d(3) = dz
c
      imax = 1
      vmax = abs(dx)
      if (abs(dy).gt.vmax) imax = 2
      if (abs(dy).gt.vmax) vmax = abs(dy)
      if (abs(dz).gt.vmax) imax = 3
      if (abs(dz).gt.vmax) vmax = abs(dz)
c
c
      do i=1,3
         if (i.ne.imax) then
            call cross (uu,nhat(1,i),d)
            call norm3d(uu)
            call cmult (uu,dn,3)
c
            v1 = o1+uu(1)
            v2 = o2+uu(2)
            v3 = o3+uu(3)
c
            w1 = o1-uu(1)
            w2 = o2-uu(2)
            w3 = o3-uu(3)
c
            call move3(v1,v2,v3)
            call draw3(xh,yh,zh)
            call draw3(w1,w2,w3)
c
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine border(xpp,ypp,zpp,ifcrv,nxx_IN,ifnx)
C
C     Draw a border around the requested element (or sub-plane in 3D)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      DIMENSION XPP(1),YPP(1),ZPP(1)
      LOGICAL   IFCRV(4),IFNX
C
C     Arrays for mapping from coarse to fine 1-D arrays
      PARAMETER (LXF=nxm+nxxm)
c     DIMENSION XPT(LXF),YPT(LXF),ZPT(LXF)
      DIMENSION XPF(LXF),YPF(LXF),ZPF(LXF)
C
      nxx = min(nxx_in,nxxm)
c
      CALL COLOR(1)
C

      K=1
      J=1
      XPT(J)=XPP(K)
      YPT(J)=YPP(K)
      ZPT(J)=ZPP(K)
      CALL MOVE3(XPP(K),YPP(K),ZPP(K))
      DO 500 IEDG=1,4

         IF (IFCRV(IEDG)) THEN
            I1=1
            IF (IEDG.EQ.1) KSTEP =  1
            IF (IEDG.EQ.2) KSTEP =  NXX
            IF (IEDG.EQ.3) KSTEP = -1
            IF (IEDG.EQ.4) KSTEP = -NXX
         ELSE
            I1=NXX-1
            IF (IEDG.EQ.1) KSTEP =  NXX-1
            IF (IEDG.EQ.2) KSTEP =  NXX*(NXX-1)
            IF (IEDG.EQ.3) KSTEP = -(NXX-1)
            IF (IEDG.EQ.4) KSTEP = -NXX*(NXX-1)
         ENDIF
C
         iii=0
         if (iii.eq.5) then
c        IF (IFNX.AND.IFCRV(IEDG)) THEN
C           IFNX implies that the points are on the GL mesh.
            J=1
            DO 100 I=2,NXX,I1
               J=J+1
               K=K+KSTEP
               XPT(J)=XPP(K)
               YPT(J)=YPP(K)
               ZPT(J)=ZPP(K)
  100       CONTINUE
C
C           Map the G.L points to a finer, regular grid.
C
            CALL MAPGL1(XPF,XPT,LXF,NXX)
            CALL MAPGL1(YPF,YPT,LXF,NXX)
            IF (IF3D) THEN
               CALL MAPGL1(ZPF,ZPT,LXF,NXX)
            ELSE
               CALL RZERO(ZPF,LXF)
            ENDIF
            DO 110 I=2,LXF
               CALL DRAW3(XPF(I),YPF(I),ZPF(I))
  110       CONTINUE
C
         ELSE
C
C           The points have already been mapped to a fine mesh. Plot.
            DO 200 I=2,NXX,I1
               K=K+KSTEP
               CALL DRAW3(XPP(K),YPP(K),ZPP(K))
  200       CONTINUE
         ENDIF
C
C        Reset 1st XPT of the next side to last (Nth) XPT of this side.
         XPT(1)=XPP(K)
         YPT(1)=YPP(K)
         ZPT(1)=ZPP(K)
C
  500 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine mapgl1(xf,xgl,mxx,mx)
C
C     Map from DIMENSIONAL Gauss-Lobatto to Regular 1D array 
C
      INCLUDE 'basics.inc'
      DIMENSION XF(1)
      DIMENSION XGL(1)
      PARAMETER (LXX=200*NXM)
      REAL INTRP(LXX)
      SAVE INTRP
C
      INTEGER MXXOLD,MXOLD
      SAVE    MXXOLD,MXOLD
      DATA    MXXOLD,MXOLD /0,0/
C
      IF (2*MX*MXX.GT.LXX) THEN
         WRITE(S,10) 
   10    FORMAT(
     $ ' Warning, attempt to call MAPGL1 w/o sufficient workspace.'
     $,' Returning.$')
         CALL PRS(S)
         write(6,*) 'mx,mxx,lxx:',mx,mxx,lxx
         return
      ENDIF
C
C     Set up the interpolation matrix, if necessary
C
C     Note: we use the new result, XF, as a work array in this call.
      IF (MXX.NE.MXXOLD.OR.MX.NE.MXOLD) CALL SETIX(INTRP,MXX,MX,XF)
      MXXOLD=MXX
      MXOLD =MX
C
C     Interpolate
      CALL MXM (INTRP,MXX,XGL,MX,XF,1)
      return
      END
c-----------------------------------------------------------------------
      subroutine mapglr(xmap,ymap,intrp,mxx,mx,wrk)
C
C     Map from Gauss-Lobatto to Regular mesh
C     using interpolation matrix INTRP
C
      INCLUDE 'basics.inc'
      DIMENSION XMAP(MXX,MXX)
      DIMENSION YMAP(MX,MX)
      REAL INTRP(MXX,MX,2)
      DIMENSION WRK(1)
C
      INTEGER MXXOLD
      SAVE    MXXOLD
      DATA    MXXOLD /0/
C
      IF (MXX.NE.MXXOLD) CALL SETIX(INTRP,MXX,MX,WRK)
      MXXOLD=MXX
C
c     IF (IF3D) THEN
c        DUM=1.0
c     ELSE
         CALL MXM (INTRP(1,1,1),MXX,YMAP,MX,WRK,MX)
         CALL MXM (WRK,MXX,INTRP(1,1,2),MX,XMAP,MXX)
c     ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine setix(intrp,mxx,mx,wrk)
C
C     Set up interpolation matrix for 1-D interpolation operator
C     from G-L mesh (MX) to regularly spaced grid (MXX).
C
      REAL INTRP(MXX,MX,2),WRK(MXX)
      include 'basics.inc'
C
      DX=2.0/FLOAT(MXX-1)
      DO 10 I=1,MXX
         WRK(I)=FLOAT(I-1)*DX - 1.0
   10 CONTINUE
      CALL IGLLM (INTRP(1,1,1),INTRP(1,1,2),XXPTS,WRK,MX,MXX,MX,MXX)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine mapglr2(xmap,ymap,intrp,mxx,mx,r1,r2,s1,s2,wrk)
C
C     Map from Gauss-Lobatto to Regular mesh
C     using interpolation matrix INTRP
C
      INCLUDE 'basics.inc'
      DIMENSION XMAP(MXX,MXX)
      DIMENSION YMAP(MX,MX)
      REAL INTRP(MXX,MX,2)
      DIMENSION WRK(1)
C
      REAL    R1OLD,R2OLD,S1OLD,S2OLD
      SAVE    R1OLD,R2OLD,S1OLD,S2OLD
      INTEGER MXXOLD
      SAVE    MXXOLD
      DATA    MXXOLD /0/
      DATA    R1OLD,R2OLD,S1OLD,S2OLD /4*0.0/
C
      IF (R1.NE.R1OLD .OR. R2.NE.R2OLD .OR.
     $    S1.NE.S1OLD .OR. S2.NE.S2OLD .OR.
     $    MXX.NE.MXXOLD ) THEN
          CALL SETIX2(INTRP,MXX,MX,R1,R2,S1,S2,WRK)
          MXXOLD=MXX
          R1OLD =R1
          R2OLD =R2
          S1OLD =S1
          S2OLD =S2
      ENDIF
C
      CALL MXM (INTRP(1,1,1),MXX,YMAP,MX,WRK,MX)
      CALL MXM (WRK,MXX,INTRP(1,1,2),MX,XMAP,MXX)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine setix2(intrp,mxx,mx,r1,r2,s1,s2,wrk)
C
C     Set up interpolation matrix for 1-D interpolation operator
C     from G-L mesh (MX) to regularly spaced grid (MXX)
C     with ranges (R1:R2) and (S1:S2).
C
      REAL INTRP(MXX,MX,2),WRK(MXX)
      include 'basics.inc'
C
      DX=(R2-R1)/FLOAT(MXX-1)
      DO 10 I=1,MXX
         WRK(I)=R1+FLOAT(I-1)*DX
   10 CONTINUE
      CALL IGLLMT(INTRP(1,1,2),XXPTS,WRK,MX,MXX,MX,MXX)
c     CALL TRNSPS2(INTRP(1,1,1),INTRP(1,1,2),MX,MXX)
      call transpose(intrp(1,1,1),mxx,intrp(1,1,2),mx)
C
      DY=(S2-S1)/FLOAT(MXX-1)
      DO 20 I=1,MXX
         WRK(I)=S1+FLOAT(I-1)*DY
   20 CONTINUE
      CALL IGLLMT(INTRP(1,1,2),XXPTS,WRK,MX,MXX,MX,MXX)
      return
      END
c-----------------------------------------------------------------------
      subroutine drwmsh
c
      include 'basics.inc'
      include 'basicsp.inc'
c
C     Arrays for mapping from coarse to fine 1-D arrays
      PARAMETER (LXF=nxm+nxxm)
c     DIMENSION XPT(LXF),YPT(LXF),ZPT(LXF)
      DIMENSION XPF(LXF),YPF(LXF),ZPF(LXF)
C
      CALL COLOR(4)
C
      nxyz = nx*ny*nz
      mxx = min(5*nx,lxf)
      do ie=1,nel
C
c        constant s lines
         do j=1,ny
            ioff = nxyz*(ie-1) + ny*(j-1) + 1
C           Map the G.L points to a finer, regular grid.
            CALL MAPGL1(XPF,XP(ioff),mxx,Nx)
            CALL MAPGL1(YPF,YP(ioff),mxx,Nx)
            IF (IF3D) THEN
               CALL MAPGL1(ZPF,ZP(ioff),mxx,Nx)
            ELSE
               CALL RZERO(ZPF,mxx)
            ENDIF
            CALL MOVE3(XPF(1),YPF(1),ZPF(1))
            DO I=2,mxx
               CALL DRAW3(XPF(I),YPF(I),ZPF(I))
            enddo
         enddo
c
c        constant r lines
         do i=1,nx
c
            if (if3d) then
               do j=1,ny
                  ioff = nxyz*(ie-1) + ny*(j-1) + i
                  xpt(j) = xp(ioff)
                  ypt(j) = yp(ioff)
                  zpt(j) = zp(ioff)
               enddo
            else
               do j=1,ny
                  ioff = nxyz*(ie-1) + ny*(j-1) + i
                  xpt(j) = xp(ioff)
                  ypt(j) = yp(ioff)
               enddo
            endif
c
C           Map the G.L points to a finer, regular grid.
            CALL MAPGL1(XPF,XPT,mxx,Ny)
            CALL MAPGL1(YPF,YPT,mxx,Ny)
            IF (IF3D) THEN
               CALL MAPGL1(ZPF,ZPT,mxx,Ny)
            ELSE
               CALL RZERO(ZPF,mxx)
            ENDIF
            CALL MOVE3(XPF(1),YPF(1),ZPF(1))
            DO j=2,mxx
               CALL DRAW3(XPF(j),YPF(j),ZPF(j))
            enddo
         enddo
      enddo
C
      return
      END
c-----------------------------------------------------------------------
      subroutine drwmsh2
c
      include 'basics.inc'
      include 'basicsp.inc'
      parameter (mx2=nxm-2,my2=nym-2)
      COMMON /CTMP1x/ xpr(mx2*my2*nzm)
     $              , ypr(mx2*my2*nzm)
     $              , zpr(mx2*my2*nzm)
c
      logical ifbdpts
c
      CALL COLOR(4)
      call prs('Input desired scale factor (1=std, neg. for int pts).$')
      call rer(user_scale)
c
      ifbdpts = .true.
      if (user_scale.lt.0) ifbdpts = .false.
C
      circ_scale = abs(user_scale)*min(.1,0.5/nx)/2.0
      nxyz = nx*ny*nz
      if (rcen(1).eq.0) call gencen
      if (ifbdpts) then
         do ie=1,nel
            rad_circ = rcen(ie)*circ_scale
            l = nxyz*(ie-1)
            do i=1,nxyz
               call draw_circ1(xp(i+l),yp(i+l),zp(i+l),rad_circ)
            enddo
         enddo
      else
         do ie=1,nel
            m = nxyz*(ie-1) + 1
            rad_circ = rcen(ie)*circ_scale
            if (if3d) then
               call map_gll_2_gl(xpr,nx-2,xp(m),nx,if3d)
               call map_gll_2_gl(ypr,nx-2,yp(m),nx,if3d)
               call map_gll_2_gl(zpr,nx-2,zp(m),nx,if3d)
               do k=1,nz-2
               do j=1,ny-2
               do i=1,nx-2
                  l = i+(nx-2)*(j-1) + (nx-2)*(ny-2)*(k-1)
                  call draw_circ1(xpr(l),ypr(l),zpr(l),rad_circ)
               enddo
               enddo
               enddo
            else
               call map_gll_2_gl(xpr,nx-2,xp(m),nx,if3d)
               call map_gll_2_gl(ypr,nx-2,yp(m),nx,if3d)
               do j=1,ny-2
               do i=1,nx-2
                  l = i+(nx-2)*(j-1)
                  call draw_circ1(xpr(l),ypr(l),zpr(l),rad_circ)
               enddo
               enddo
            endif
         enddo
      endif
      return
      end
c-----------------------------------------------------------------------
C    03-08-99
      subroutine nekplanar
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C--------------------------
C     INPUT
C     ngrdx : # of interior evaluation  points in X  
C     ngrdy : # of interior evaluation  points in Y  
C     ngrdz : # of interior evaluation  planes in Z  
C     OUTPUT
C     enstrphy_mx(k) , k = 1,ngrdz ; Max enstrphy on kth plane.
c
      REAL wk_pln(0:500),z_pln(0:500)
      INTEGER ngrdx,ngrdy,ngrdz
c     LOCAL
      INTEGER ix,iy,iz,ie,m,m1,m2,i,j,k,idum,ierr
      REAL zup,zlw,xup,xlw,yup,ylw,dx,dy,dz
      REAL xx,yy,zz,v1,v2,v3,trm,ensmx
C-----------------------------------------
C     Determine (uniform) grid spacing
      call prs('Input number of points in x-, y- and z-directions:$')
      call reiii(ngrdx,ngrdy,ngrdz)
      ngrdz = min(500,ngrdz)

C     Determine Domain Size
      ntot = nel*nx*ny*nz
      xlw = glmin(xp,ntot)
      ylw = glmin(yp,ntot)
      zlw = glmin(zp,ntot)
      xup = glmax(xp,ntot)
      yup = glmax(yp,ntot)
      zup = glmax(zp,ntot)

c     Spacing
      dx = (xup - xlw)/ngrdx
      dy = (yup - ylw)/ngrdy
      dz = (zup - zlw)/ngrdz

C     For each horizontal plane,we calc the MAX enstrphy over an 
C     interior grid .
c
      call prs  ('Input start,stop, skip fld numbers:$')
      call reiii (istart_fld,iend_fld,iskip)
      iskip = max(iskip,1)
c
c
      iframe = 0
      do ifld=istart_fld,iend_fld,iskip
c
         iframe = iframe+1
         ndumps = ifld
         call getfld(ndumps,ierr,.true.,.true.)
         time_3 = time
c
         write(9 ,99)
         zz = zlw
         do k = 0,ngrdz
c
C           For each horizontal plane,we calc the MAX enstrphy over an 
C           interior grid .
c
            l=0
            dzk = dz*(1.02)**k
            zz = zlw + k*dzk
            if (zz.le.zup/2.) then 
               do j = 0,ngrdy
                 yy = ylw + j*dy
                 do i = 0,ngrdx
                   xx = xlw + i*dx
                   CALL interp(v1,xx,yy,zz,idum,work,ierr)
                   if (l.eq.0.or.v1.gt.wkmax) then
                      l=1
                      wkmax = v1
                      xwmax = xx
                      ywmax = yy
                   endif
                 enddo
                enddo
               wk_pln(k) = wkmax
               z_pln (k) = zz
               write(25,25) zz,wkmax,xwmax,ywmax,time_3
               write(6 ,26) k,zz,wkmax,xwmax,ywmax,time_3
   25          format(1p5e13.4)
   26          format(i5,1p5e13.4)
c
c              Output for "stacked" gnuplot
c
               wwt = wkmax + 10.*(ifld-1)
               write(9 ,29) zz,wwt
   29          format(1p5e13.4)
            endif
         enddo
c
         wwt = 10.*(ifld-1)
         write(9 ,99)
         write(9 ,20) zlw,wwt
         write(9 ,20) zup,wwt
   20    format(1p5e13.4)
   99    format(/)
c
      enddo
c
      return
      end
C-----------------------------------------------------------------------
      subroutine setrstp_SET(ivec,XPT1,YPT1,ZPT1)
C     Set up list of candidate RST planes
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      DIMENSION VEC(3),VEC1(3),VEC2(3),VEC3(3)
      CHARACTER*1 YESNO
      LOGICAL IFANY,IFTMP
C
C
C     Find the element containing this point
C
      CALL INTERP(VAL1,XPT1,YPT1,ZPT1,IE,WORK,ierr)
C
C     Exception handling:
      IF (IE.EQ.0) THEN
         WRITE(S,20) XPT1,YPT1,ZPT1
         CALL PRS(S)
   20    FORMAT(2X,'Point (',G10.3,',',G10.3,',',G10.3,') is not in '
     $            ,'the domain.$')
         CALL PRS('  Continue (y/n)?$')
         CALL RES(YESNO,1)
         return
      ENDIF
C
C     Now that element is found, find what type of plane is closest: R,S, or T?
C
      CALL RZERO(VEC,3)
      IF (IVEC.GE.1.AND.IVEC.LE.3) THEN
         VEC(IVEC)=1.0
      ELSE
         CALL PRS
     $('Enter approximate normal vector of the surface at this point.$')
         CALL PRS
     $('  (x_n,y_n,z_n):$') 
         CALL RERRR(VEC(1),VEC(2),VEC(3))
         CALL NORM3D(VEC)
      ENDIF
C
C     Does this correspond most closely to an R, S, or T plane?
C
      RLMAX=0.0
      DO 30 IPLN=1,6 
         CALL SUB3(VEC1,XYZQ(1,2,IPLN,IE),XYZQ(1,1,IPLN,IE),3)
         CALL SUB3(VEC2,XYZQ(1,4,IPLN,IE),XYZQ(1,3,IPLN,IE),3)
         CALL CROSS(VEC3,VEC1,VEC2)
         CALL NORM3D(VEC3)
         RLNGTH = DOTPROD(VEC,VEC3)
         RLNGTH = ABS(RLNGTH)
         IF (RLNGTH.GT.RLMAX) THEN
            IJKPLN  = IPLN
            RLMAX   = RLNGTH
            DXORD=VEC3(1) 
            DYORD=VEC3(2) 
            DZORD=VEC3(3) 
         ENDIF
   30 CONTINUE
      VEC3(1)=DXORD
      VEC3(2)=DYORD
      VEC3(3)=DZORD
C
      DXORD=VEC(1)
      DYORD=VEC(2)
      DZORD=VEC(3)
C
C     Find the plane corresponding to the chosen point and normal.
C
      NXY =NX*NY
      NXYZ=NX*NY*NZ
      DMIN=10.0E15
      DO 100 I=1,NXYZ
         IEOFF=NXYZ*(IE-1)+I
         DIST=(XP(IEOFF)-XPT1)**2+(YP(IEOFF)-YPT1)**2+
     $        (ZP(IEOFF)-ZPT1)**2
         IF (DIST.LT.DMIN) THEN
            IJKMIN=I
            DMIN=DIST
         ENDIF
  100 CONTINUE
      CALL DECOD(IXP,IYP,IZP,IDUM,IJKMIN,NX,NY,NZ)
      IF (IJKPLN.LE.2) IPLAN=IXP
      IF (IJKPLN.GT.2) IPLAN=IYP
      IF (IJKPLN.GT.4) IPLAN=IZP
C     Set IJKPLN according to actual value of IPLAN
      NXT=2*IPLAN-1
      IF (NXT.LT.NX) IJKPLN=2*((IJKPLN-1)/2)+1
      IF (NXT.GT.NX) IJKPLN=2*((IJKPLN-1)/2)+2
C
C===============================================
C     Start generating LIST of plotting planes
C===============================================
C
C     Initialize pointers to zero
      DO 105 I=1,24*NELM
         XYZQ(5,I,1,1)=0.0
  105 CONTINUE
C
c     NLSTP=1                    pff 12/21/96
      NLSTP=NLSTP+1
      LIST(NLSTP)=6*NX*(IE-1)+NX*(IJKPLN-1)+IPLAN
      IMD=MOD(IJKPLN,2)
C
C     Set the sign of this plane according to requested normal vector:
C
      RLNGTH = DOTPROD(VEC,VEC3)
      INRM=1
      IF (RLNGTH.LT.0.0) INRM=-1
      LIST(NLSTP)=INRM*LIST(NLSTP)
C
      DO 110 I=1,4
         NEIGHI=INT(XYZQ(4,I,IJKPLN,IE))
         IF (NEIGHI.EQ.0) THEN
            XYZQ(5,I,IJKPLN,IE)=2.0
         ELSE
            RN=FLOAT(NEIGHI)
            XYZQ(5,I,IJKPLN,IE)=SIGN(1.0,RN)
         ENDIF
  110 CONTINUE
C
C=======================================
C     Find all adjoining planes
C=======================================
C
      ICOUNT=0
  200 CONTINUE
      IFANY=.FALSE.
      ICOUNT=ICOUNT+1
      DO 500 IE=1,NEL
      DO 500 IPLN=1,6
      DO 500 IQ=1,4
         IF (ABS(XYZQ(5,IQ,IPLN,IE)).EQ.1.0) THEN
            IFANY=.TRUE.
            NEIGHI=INT(XYZQ(4,IQ,IPLN,IE))
            NEIGHA=ABS(NEIGHI)
            IF (XYZQ(5,NEIGHA,1,1).EQ.0.0) THEN
C              The neighboring plane has Not been set.
               CALL DECOD(JQ,JPLN,JE,IDUM,NEIGHA,4,6,NEL)
               DO 400 J=1,3
                  JJ=JQ+J
                  JJ=MOD1(JJ,4)
                  NEIGHJ=INT(XYZQ(4,JJ,JPLN,JE))
                  IF (NEIGHJ.EQ.0) THEN
                     XYZQ(5,JJ,JPLN,JE)=2.0
                  ELSE
                     RN=FLOAT(NEIGHJ)
                     XYZQ(5,JJ,JPLN,JE)=
     $                   XYZQ(5,IQ,IPLN,IE)*SIGN(1.0,RN)

                  ENDIF
  400          CONTINUE
C
C              New neighbor points all set, now add neighbor plane to LIST.
               JMD=MOD(JPLN,2)
               IF (IMD.EQ.JMD) THEN
                  JPLAN=IPLAN
               ELSE
                  JPLAN=NX+1-IPLAN
               ENDIF
               NLSTP=NLSTP+1
               LIST(NLSTP)=6*NX*(JE-1)+NX*(JPLN-1)+JPLAN
               LIST(NLSTP)=LIST(NLSTP)*SIGN(1.0,XYZQ(5,IQ,IPLN,IE))
C              Flip according to initial plane
               LIST(NLSTP)=INRM*LIST(NLSTP)
C
C              Remove pointers and exit
               XYZQ(5,NEIGHA,1,1)=2.0
            ENDIF
            XYZQ(5,IQ,IPLN,IE)=2.0
         ENDIF
  500 CONTINUE
      IF (IFANY) GOTO 200
C
C     { LIST is complete, sort by zbuff. (? or perhaps after rotation?)}
C     LIST is complete, set normals according to the following:
C
C     If plane is a Y-plane, i.e. 3 or 4, then it must be flipped (?).
C
      write(s,501) nlstp
      call prs(s)
  501 format(2x,'num planes:',I5,'$')
C
      return
      END
c-----------------------------------------------------------------------
      subroutine mapglg(xmap,ymap,intrp,mxx,mx,wrk)
C
C     Map from Gauss-Lobatto to Regular mesh
C     using interpolation matrix INTRP
C
      INCLUDE 'basics.inc'
      DIMENSION XMAP(MXX,MXX)
      DIMENSION YMAP(MX,MX)
      REAL INTRP(MXX,MX,2)
      DIMENSION WRK(1)
C
      INTEGER MXXOLD
      SAVE    MXXOLD
      DATA    MXXOLD /0/
C
      IF (MXX.NE.MXXOLD) CALL SETIXG(INTRP,MXX,MX,WRK)
      MXXOLD=MXX
C
c     IF (IF3D) THEN
c        DUM=1.0
c     ELSE
         CALL MXM (INTRP(1,1,1),MXX,YMAP,MX,WRK,MX)
         CALL MXM (WRK,MXX,INTRP(1,1,2),MX,XMAP,MXX)
c     ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine setixg(intrp,mxx,mx,wrk)
C
C     Set up interpolation matrix for 1-D interpolation operator
C     from G-L mesh (MX) to regularly spaced grid (MXX).
C
      REAL INTRP(MXX,MX,2),WRK(MXX)
      include 'basics.inc'
C
      call zwgll(wrk,intrp,mxx)
      CALL IGLLM (INTRP(1,1,1),INTRP(1,1,2),XXPTS,WRK,MX,MXX,MX,MXX)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine drwmsh3
c
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifbdpts
c
      CALL COLOR(4)
      call prs('Input desired scale factor (1=std, neg. for int pts).$')
      call rer(sc)
c
      ifbdpts = .true.
      if (sc.lt.0) ifbdpts = .false.
      sc = abs(sc)
C
      circ_scale = abs(sc)*min(.1,0.5/nx)/2.0
C
C     For clarity, pull points toward cg of element by scale factor pull
C
      pull = sc
c
      nxyz = nx*ny*nz
      if (rcen(1).eq.0) call gencen
      if (ifbdpts) then
         do ie=1,nel
            rc = rcen(ie)*circ_scale
            l = nxyz*(ie-1)
            do i=1,nxyz
               xc = xcen(ie) + pull*(xp(i+l)-xcen(ie))
               yc = ycen(ie) + pull*(yp(i+l)-ycen(ie))
               zc = zcen(ie) + pull*(zp(i+l)-zcen(ie))
               call draw_circu(xc,yc,zc,rc,u(i+l),sc,i)
            enddo
         enddo
      else
         do ie=1,nel
            rc = rcen(ie)*circ_scale
            m = nxyz*(ie-1)
            if (if3d) then
               do k=2,nz-1
               do j=2,ny-1
               do i=2,nx-1
                  l = i+nx*(j-1) + nx*ny*(k-1)
                  xc = xcen(ie) + pull*(xp(i+l)-xcen(ie))
                  yc = ycen(ie) + pull*(yp(i+l)-ycen(ie))
                  zc = zcen(ie) + pull*(zp(i+l)-zcen(ie))
                  call draw_circu(xc,yc,zc,rc,u(i+l),sc,i)
               enddo
               enddo
               enddo
            else
               zc = 0.
               do j=2,ny-1
               do i=2,nx-1
                  l = i+nx*(j-1)
                  xc = xcen(ie) + pull*(xp(i+l)-xcen(ie))
                  yc = ycen(ie) + pull*(yp(i+l)-ycen(ie))
                  call draw_circu(xc,yc,zc,rc,u(i+l),sc,i)
               enddo
               enddo
            endif
         enddo
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine draw_circu(x3,y3,z3,r,u,sc,l)
      include 'devices.inc'
c
      character*10 au
c
c     draws a circle around XX,YY,ZZ, places int(u) next to it
c
      xm=xiso(x3-r,y3,z3)
      ym=yiso(x3-r,y3,z3)
      xp=xiso(x3+r,y3,z3)
      yp=yiso(x3+r,y3,z3)
      dx = (xp-xm)**2 + (yp-ym)**2
c
      xm=xiso(x3,y3-r,z3)
      ym=yiso(x3,y3-r,z3)
      xp=xiso(x3,y3+r,z3)
      yp=yiso(x3,y3+r,z3)
      dy = (xp-xm)**2 + (yp-ym)**2
      dr = max(dx,dy)
c
      if (if3d) then
         xm=xiso(x3,y3,z3-r)
         ym=yiso(x3,y3,z3-r)
         xp=xiso(x3,y3,z3+r)
         yp=yiso(x3,y3,z3+r)
         dz = (xp-xm)**2 + (yp-ym)**2
         dr = max(dr,dz)
      endif
c
      dr = 0.5*sqrt(dr)
c
      xc=xiso(x3,y3,z3)
      yc=yiso(x3,y3,z3)
c     write(6,11) xc,yc,x3,y3,z3,dr,r
c  11 format(8f10.5)
c
      x = xc+dr
      y = yc
      call movec(x,y)
c
      n = 100
      pi2n = 8.*atan(1.0)/n
      do i=1,n
         theta = i*pi2n
         x = xc + dr*cos(theta)
         y = yc + dr*sin(theta)
         call drawc(x,y)
      enddo
c
c     Output iu
c
      iu = u
      call blank(au,10)
      if (0.le.iu .and. iu.le.9) then
         write(au,1) iu
      elseif (-9.le.iu .and. iu.le.99) then
         write(au,2) iu
      elseif (-99.le.iu .and. iu.le.999) then
         write(au,3) iu
      elseif (-999.le.iu .and. iu.le.9999) then
         write(au,4) iu
      elseif (-9999.le.iu .and. iu.le.99999) then
         write(au,5) iu
      elseif (-99999.le.iu .and. iu.le.999999) then
         write(au,6) iu
      elseif (-999999.le.iu .and. iu.le.9999999) then
         write(au,7) iu
      elseif (-9999999.le.iu .and. iu.le.99999999) then
         write(au,8) iu
      else
         write(au,9) iu
      endif
    1 format(i1,'$')
    2 format(i2,'$')
    3 format(i3,'$')
    4 format(i4,'$')
    5 format(i5,'$')
    6 format(i6,'$')
    7 format(i7,'$')
    8 format(i8,'$')
    9 format(i9,'$')
      yshift = 0.3*dr
      if (mod(l,2).eq.0) yshift = -.7*dr
      ys = yc+yshift
      call gwrite(xc,ys,sc,au)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine arow3np(x,y,z,dxbar,dybar,dzbar,sarrow)
c
c     New arrow routine, but no planar projection.    2/25/00   pff
c
      COMMON/SCALE/XFAC,YFAC,XZERO,YZERO
c
      real uu(3),d(3),nhat(3,3)
      save nhat
      data nhat / 1,0,0  ,  0,1,0  ,  0,0,1  /
c
C     Now draw shadow on plane at Zbase (without +DZ)
C     Sort of pale orange in heat map
C     FIX COLOR(WHITE USUALLY, YELLOOW FOR AXIS.  <FIX 3-D ARROWHEAD)!!??
C
      IF (DXBAR.EQ.0.0.AND.DYBAR.EQ.0.0.AND.DZBAR.EQ.0.0) return
      IF (sarrow.eq.0) return
c
      write(6,1) 'move3',x,y,z
    1 format(a5,2x,3f12.5)
c
      CALL MOVE3(X,Y,Z)
C
C
C     Draw 3-dimensional arrow
      CALL PENW(3)
      CALL COLOR(1)
      DX = DXBAR*SARROW
      DY = DYBAR*SARROW
      DZ = DZBAR*SARROW
c
      xh = x+dx
      yh = y+dy
      zh = z+dz
      CALL DRAW3(xh,yh,zh)
c     write(6,1) 'dmove',dx,dy,dz
c     write(6,1) 'draw3',xy,yh,zh
c
c     New arrowheads  (pff 11/24/99)
c
      dn = dx*dx + dy*dy + dz*dz
      if (dn.gt.0) dn = sqrt(dn)
      dn = 0.04*dn
c
      dxn = -.14*dx
      dyn = -.14*dy
      dzn = -.14*dz
c
c     Origin of plane intersecting arrow
c
      o1 = xh+dxn
      o2 = yh+dyn
      o3 = zh+dzn
c
c     Find projection of arrow onto x y and z planes
c
      d(1) = dx
      d(2) = dy
      d(3) = dz
c
      imax = 1
      vmax = abs(dx)
      if (abs(dy).gt.vmax) imax = 2
      if (abs(dy).gt.vmax) vmax = abs(dy)
      if (abs(dz).gt.vmax) imax = 3
      if (abs(dz).gt.vmax) vmax = abs(dz)
c
c
      do i=1,3
         if (i.ne.imax) then
            call cross (uu,nhat(1,i),d)
            call norm3d(uu)
            call cmult (uu,dn,3)
c
            v1 = o1+uu(1)
            v2 = o2+uu(2)
            v3 = o3+uu(3)
c
            w1 = o1-uu(1)
            w2 = o2-uu(2)
            w3 = o3-uu(3)
c
            call move3(v1,v2,v3)
            call draw3(xh,yh,zh)
            call draw3(w1,w2,w3)
c
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine draw_circ1(x3,y3,z3,r)
c     draws a circle around XX,YY,ZZ
c
      include 'devices.inc'
      logical iftmp
c
      xm=xiso(x3-r,y3,z3)
      ym=yiso(x3-r,y3,z3)
      xp=xiso(x3+r,y3,z3)
      yp=yiso(x3+r,y3,z3)
      dx = (xp-xm)**2 + (yp-ym)**2
c
      xm=xiso(x3,y3-r,z3)
      ym=yiso(x3,y3-r,z3)
      xp=xiso(x3,y3+r,z3)
      yp=yiso(x3,y3+r,z3)
      dy = (xp-xm)**2 + (yp-ym)**2
      dr = max(dx,dy)
c
      if (if3d) then
         xm=xiso(x3,y3,z3-r)
         ym=yiso(x3,y3,z3-r)
         xp=xiso(x3,y3,z3+r)
         yp=yiso(x3,y3,z3+r)
         dz = (xp-xm)**2 + (yp-ym)**2
         dr = max(dr,dz)
      endif
c
      dr = 0.5*sqrt(dr)
c
      xc=xiso(x3,y3,z3)
      yc=yiso(x3,y3,z3)
c     write(6,1) xc,yc,x3,y3,z3,dr,r
    1 format(8f10.5)
c
      x = xc+dr
      y = yc
c
      n = 100
      pi2n = 8.*atan(1.0)/n
c
c     Turn off hardcopy so we can use postscript circles  (pff 4/15/00)
      iftmp=ifhard
      ifhard=.false.
c
      call movec(x,y)
      do i=1,n
         theta = i*pi2n
         x = xc + dr*cos(theta)
         y = yc + dr*sin(theta)
         call drawc(x,y)
      enddo
c
c     if hardcopy, use postscript circles
c
      if (iftmp .and. ifhard0 .and. ifposts) then
         xpc = 550-500* yscr(yc)
         ypc =  50+500* xscr(xc)
         rc  =     500*(xscr(dr)-xscr(0))
         write(45,2) xpc,ypc,rc
    2    format('newpath',3f12.4,' 0 360 arc closepath stroke')
c   2    format('newpath',3f12.4,' 0 360 arc fill closepath stroke')
      endif
c
      ifhard=iftmp
      return
      end
c-----------------------------------------------------------------------
      subroutine map_gll_2_gl(ugl,mx,ugll,nx,if3d)
c
      parameter (nxm=30)
      real gll_pts(nxm),gll_wts(nxm),gl_pts (nxm),gl_wts (nxm)
      real i (3*nxm*nxm),it(3*nxm*nxm),w(nxm*nxm*nxm,2)
      save gll_pts,gll_wts,gl_pts,gl_wts,i,it
c
      integer mx_last,nx_last
      save    mx_last,nx_last
      data    mx_last,nx_last  /0,0/
c
c
      if (nx.gt.nxm .or. nx*mx.gt. 3*nxm*nxm) then
         write(6,*) 'increase array dimensions in mapreg3d1',mx,nx,nxm
         call copy(w_map,w_in,nx*nx*nx)
         return
      endif
c
      if (nx.ne.nx_last .or. mx.ne.mx_last) then
c
c        Recompute interpolation matrices
c
         call zwgll(gll_pts,gll_wts,nx)
         call zwgl (gl_pts ,gl_wts ,mx)
c        Use Fornberg's routine to compute interpolation coefficients
         nx1 = nx-1
         l  = 1
         do k=1,mx
            call fd_weights_full(gl_pts(k),gll_pts,nx1,0,it(l))
            l  = l+nx
         enddo
         call transpose(i,mx,it,nx)
c
         nx_last = nx
         mx_last = mx
c
      endif
c
      call map_tnsr(ugl,mx,ugll,nx,i,it,w(1,1),w(1,2),if3d)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine findl_new(txy,level,tmin,tmax)

      level = 14*txy
      level = min(level,14)
      level = max(level,1)

      return
      end
c-----------------------------------------------------------------------
