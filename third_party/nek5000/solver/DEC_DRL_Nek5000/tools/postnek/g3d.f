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
       SUBROUTINE DRISOG
C
C      Draw ISOMETRIC view of general 3D object.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 CB1
      INTEGER IFIRST,NCSOLD,NCSEG1,NCSGM1
      INTEGER ICEDG(3,16),IEDGFC(4,6),ICFACE(4,10)
      SAVE    ICEDG,IEDGFC,ICFACE
      SAVE    IFIRST,NCSOLD,NCSEG1,NCSGM1
C
      COMMON /HLINE/ HIK(NXM,50),XXIS(50),YYIS(50),ZZIS(50),
     $               XK(50),YK(50),ZK(50)
      COMMON /ILINE/ IC(8),NSKIPX(3)
      COMMON /LLINE/ IFEDG(12,2,NELM)
      LOGICAL IFEDG,IFCURV
C
      DATA    IFIRST,NCSOLD,NCSEG1,NCSGM1 /4*0/
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
c      -2D-  (NOTE:  Edges default to faces in 2D! Not consistent, but...)
     $                1,3,2,   2,4,2,   1,2,1,   3,4,1 /
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
c      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C
C
C     Sort the faces according to Z-location
C     for the color fill plotting routines
c     IF (IF3D) CALL SORTZ
C
      IF(IFIRST.EQ.0)THEN
         NFACES=2*NDIM
         IFIRST=1
         IF (IF3D) CALL RESCAL
C
C        Set up logicals determining whether edge is to be
C        displayed or not.
C
C        Assign EB's numbering scheme to PF's scheme.
C
         EFACE(1)=4
         EFACE(2)=2
         EFACE(3)=1
         EFACE(4)=3
         EFACE(5)=5
         EFACE(6)=6
C
C        Assign inverse of EB's numbering scheme to PF's scheme.
C
         EFACE1(1)=3
         EFACE1(2)=2
         EFACE1(3)=4
         EFACE1(4)=1
         EFACE1(5)=5
         EFACE1(6)=6
C
         NEDGE=4
         IF (IF3D) NEDGE=12
         DO 100 I=1,NEDGE
         DO 100 IE=1,NEL
            IFEDG(I,1,IE) = .FALSE.
            IFEDG(I,2,IE) = .TRUE.
  100    CONTINUE
         IFLD=1
         IF (.NOT.IFFLOW) IFLD=2
         DO 220 IFACE=1,NFACES
         DO 220 IE=1,NEL
            IFACE1=EFACE(IFACE)
            CB1 = CBC(IFACE1,IE,IFLD)
            IF (CB1.EQ.'W' .OR. CB1.EQ.'T'   .or.
     $          CB1.EQ.'V' .or. cb1.eq.'v'  ) THEN
               IF (IF3D) THEN
                  DO 210 I=1,4
                     II = IEDGFC(I,IFACE)
                     IFEDG(II,1,IE)=.TRUE.
  210             CONTINUE
               ELSE
                  IFEDG(IFACE,1,IE)=.TRUE.
c                 IFEDG(IFACE1,1,IE)=.TRUE.
               ENDIF
            ENDIF
  220    CONTINUE
      ENDIF
      CALL COLOR(13)
      CALL PENW(1)
      NXYZ = NX*NY*NZ
      NSKIPX(1) = 1
      NSKIPX(2) = NX
      NSKIPX(3) = NX*NY
C
C     Fill corners - 1 through 8.
C
      IC(1) =                                 0
      IC(2) =                             (NX-1)
      IC(3) =                 NX*(NY-1)
      IC(4) =                 NX*(NY-1) + (NX-1)
      IC(5) =  NX*NY*(NZ-1)
      IC(6) =  NX*NY*(NZ-1)             + (NX-1)
      IC(7) =  NX*NY*(NZ-1) + NX*(NY-1)
      IC(8) =  NX*NY*(NZ-1) + NX*(NY-1) + (NX-1)
C
      IF (NCSOLD.NE.NCSEGS) THEN
C
C        Set up curve line segment evaluation array (exclude endpoints) :
C
         NCSOLD = NCSEGS
         NCSEG1 = NCSEGS + 1
         NCSGM1 = NCSEGS - 1
         DR1 = 2.0 / FLOAT(NCSEGS)
         DO 400 K=1,NCSGM1
            RR1 = -1.0 + DR1*FLOAT(K)
            CALL HLEGN(HIK(1,K),RR1,ZPTS,NX)
  400    CONTINUE
      ENDIF
C
C     Draw element edges (for now, we do the 4X redundant work)
C
      NEDGE=4
      IF (IF3D) NEDGE=12
      DO 1000 IE=1,NEL
         IEOFF=NXYZ*(IE-1)+1
C
C        Check if we can take a short cut in plotting the element bdry.
         IFCURV=.FALSE.
         DO 500 IEDG = 1,8
            IF (CCURVE(IEDG,IE).NE.' ') IFCURV=.TRUE.
  500    CONTINUE
C
C        Edges
C
         DO 800 IEDG = 1,NEDGE
         IF ( IFEDG(IEDG,1,IE)                      .OR.
     $        IFDMSH                                .OR.
     $        PLANE2.EQ.'ALL BOUNDARIES'                ) THEN
C
C           There is a bug in this; I will make all edges red
c           IF (CBC(IEDG,IE,1).EQ.'W  '.AND..NOT.IF3D) THEN
c              CALL COLOR(1)
c           ELSE
c              CALL COLOR(13)
c           ENDIF
               CALL COLOR(13)
            IEDG0 = IEDG + 12
            IF (IF3D) IEDG0 = IEDG
            ICE1 = ICEDG(1,IEDG0)
            ICE2 = ICEDG(2,IEDG0)
            NSKIP = NSKIPX(ICEDG(3,IEDG0))
            IC0 = IEOFF+IC(ICE1)
            IC1 = IEOFF+IC(ICE2)
            XK(     1) = XP(IC0)
            YK(     1) = YP(IC0)
            ZK(     1) = ZP(IC0)
            XK(NCSEG1) = XP(IC1)
            YK(NCSEG1) = YP(IC1)
            ZK(NCSEG1) = ZP(IC1)
            IF (IFCURV) THEN
               CALL EVLINE(XK(2),XP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(YK(2),YP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(ZK(2),ZP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL VDRAW3A(XK,YK,ZK,NCSEG1)
            ELSE
               XK(2) = XP(IC1)
               YK(2) = YP(IC1)
               ZK(2) = ZP(IC1)
               CALL VDRAW3A(XK,YK,ZK,2)
            ENDIF
         ENDIF
  800    CONTINUE
 1000 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE EVLINE(UV,UVP,HIK,N1,N1D,NPT,NSKIP)
      INCLUDE 'basics.inc'
C
C     Evaluate vector uv <== uvp, @ k=1,npts
C
      REAL UV(NPT),UVP(1)
      REAL HIK(N1D,NPT)
      IF (NPT.LE.N1) THEN
         DO 100 K=1,NPT
            UV(K) = 0.0
            ISKIP = 1 - NSKIP
            DO 100 I=1,N1
               ISKIP = ISKIP + NSKIP
               UV(K) = UV(K) + HIK(I,K)*UVP(ISKIP)
  100    CONTINUE
      ELSE
         DO 200 K=1,NPT
            UV(K)=0.0
  200    CONTINUE
         ISKIP = 1 - NSKIP
         DO 300 I=1,N1
            ISKIP = ISKIP + NSKIP
            DO 300 K=1,NPT
               UV(K) = UV(K) + HIK(I,K)*UVP(ISKIP)
  300    CONTINUE
      ENDIF
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE HILIGHT(IE,icolr)
C
C      Draw ISOMETRIC view of general 3D element number IE.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 CB1
      INTEGER IFIRST,NCSOLD,NCSEG1,NCSGM1
      INTEGER ICEDG(3,16),IEDGFC(4,6),ICFACE(4,10)
      SAVE    ICEDG,IEDGFC,ICFACE
      SAVE    IFIRST,NCSOLD,NCSEG1,NCSGM1
C
      COMMON /HLINE/ HIK(NXM,50),XXIS(50),YYIS(50),ZZIS(50),
     $               XK(50),YK(50),ZK(50)
      COMMON /ILINE/ IC(8),NSKIPX(3)
      COMMON /LLINE/ IFEDG(12,2,NELM)
      LOGICAL IFEDG,IFCURV
C
      DATA    IFIRST,NCSOLD,NCSEG1,NCSGM1 /4*0/
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
c      -2D-  (NOTE:  Edges default to faces in 2D! Not consistent, but...)
     $                1,3,2,   2,4,2,   1,2,1,   3,4,1 /
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
c      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C
C
C
      IF(IFIRST.EQ.0)THEN
         NFACES=2*NDIM
         IFIRST=1
C
C        Set up logicals determining whether edge is to be
C        displayed or not.
C
C        Assign EB's numbering scheme to PF's scheme.
C
         EFACE(1)=4
         EFACE(2)=2
         EFACE(3)=1
         EFACE(4)=3
         EFACE(5)=5
         EFACE(6)=6
C
C        Assign inverse of EB's numbering scheme to PF's scheme.
C
         EFACE1(1)=3
         EFACE1(2)=2
         EFACE1(3)=4
         EFACE1(4)=1
         EFACE1(5)=5
         EFACE1(6)=6
C
      ENDIF
      CALL COLOR(icolr)
      CALL PENW(1)
      NXYZ = NX*NY*NZ
      NSKIPX(1) = 1
      NSKIPX(2) = NX
      NSKIPX(3) = NX*NY
C
C     Fill corners - 1 through 8.
C
      IC(1) =                                 0
      IC(2) =                             (NX-1)
      IC(3) =                 NX*(NY-1)
      IC(4) =                 NX*(NY-1) + (NX-1)
      IC(5) =  NX*NY*(NZ-1)
      IC(6) =  NX*NY*(NZ-1)             + (NX-1)
      IC(7) =  NX*NY*(NZ-1) + NX*(NY-1)
      IC(8) =  NX*NY*(NZ-1) + NX*(NY-1) + (NX-1)
C
      IF (NCSOLD.NE.NCSEGS) THEN
C
C        Set up curve line segment evaluation array (exclude endpoints) :
C
         NCSOLD = NCSEGS
         NCSEG1 = NCSEGS + 1
         NCSGM1 = NCSEGS - 1
         DR1 = 2.0 / FLOAT(NCSEGS)
         DO 400 K=1,NCSGM1
            RR1 = -1.0 + DR1*FLOAT(K)
            CALL HLEGN(HIK(1,K),RR1,ZPTS,NX)
  400    CONTINUE
      ENDIF
C
C     Draw element edges (for now, we do the 4X redundant work)
C
      NEDGE=4
      IF (IF3D) NEDGE=12
         IEOFF=NXYZ*(IE-1)+1
         call minmax(xmn,xmx,xp(ieoff),nxyz)
         call minmax(ymn,ymx,yp(ieoff),nxyz)
         if (if3d) then
            call minmax(zmn,zmx,zp(ieoff),nxyz)
            write(S,'(A8,3g12.4,A1)') 'XYZ min:',xmn,ymn,zmn,'$'
            CALL PRS (S)
            write(S,'(A8,3g12.4,A1)') 'XYZ max:',xmx,ymx,zmx,'$'
            CALL PRS (S)
         else
            write(S,'(A7,2g12.4,A1)') 'XY min:',xmn,ymn,'$'
            CALL PRS (S)
            write(S,'(A7,2g12.4,A1)') 'XY max:',xmx,ymx,'$'
            CALL PRS (S)
         endif
C
C        Check if we can take a short cut in plotting the element bdry.
         IFCURV=.FALSE.
         DO 500 IEDG = 1,8
            IF (CCURVE(IEDG,IE).NE.' ') IFCURV=.TRUE.
  500    CONTINUE
C
C        Edges
C
         DO 800 IEDG = 1,NEDGE
C
C           There is a bug in this; I will make all edges red
c           IF (CBC(IEDG,IE,1).EQ.'W  '.AND..NOT.IF3D) THEN
c              CALL COLOR(1)
c           ELSE
c              CALL COLOR(9)
c           ENDIF
               CALL COLOR(icolr)
            IEDG0 = IEDG + 12
            IF (IF3D) IEDG0 = IEDG
            ICE1 = ICEDG(1,IEDG0)
            ICE2 = ICEDG(2,IEDG0)
            NSKIP = NSKIPX(ICEDG(3,IEDG0))
            IC0 = IEOFF+IC(ICE1)
            IC1 = IEOFF+IC(ICE2)
            XK(     1) = XP(IC0)
            YK(     1) = YP(IC0)
            ZK(     1) = ZP(IC0)
            XK(NCSEG1) = XP(IC1)
            YK(NCSEG1) = YP(IC1)
            ZK(NCSEG1) = ZP(IC1)
            IF (IFCURV) THEN
               CALL EVLINE(XK(2),XP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(YK(2),YP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(ZK(2),ZP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL VDRAW3A(XK,YK,ZK,NCSEG1)
            ELSE
               XK(2) = XP(IC1)
               YK(2) = YP(IC1)
               ZK(2) = ZP(IC1)
               CALL VDRAW3A(XK,YK,ZK,2)
            ENDIF
  800    CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE HILIGHT2(IE,icolr)
C
C      Draw ISOMETRIC view of general 3D element number IE.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 CB1
      INTEGER IFIRST,NCSOLD,NCSEG1,NCSGM1
      INTEGER ICEDG(3,16),IEDGFC(4,6),ICFACE(4,10)
      SAVE    ICEDG,IEDGFC,ICFACE
      SAVE    IFIRST,NCSOLD,NCSEG1,NCSGM1
C
      COMMON /HLINE/ HIK(NXM,50),XXIS(50),YYIS(50),ZZIS(50),
     $               XK(50),YK(50),ZK(50)
      COMMON /ILINE/ IC(8),NSKIPX(3)
      COMMON /LLINE/ IFEDG(12,2,NELM)
      LOGICAL IFEDG,IFCURV
C
      DATA    IFIRST,NCSOLD,NCSEG1,NCSGM1 /4*0/
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
c      -2D-  (NOTE:  Edges default to faces in 2D! Not consistent, but...)
     $                1,3,2,   2,4,2,   1,2,1,   3,4,1 /
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
c      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C
C
C
      IF(IFIRST.EQ.0)THEN
         NFACES=2*NDIM
         IFIRST=1
C
C        Set up logicals determining whether edge is to be
C        displayed or not.
C
C        Assign EB's numbering scheme to PF's scheme.
C
         EFACE(1)=4
         EFACE(2)=2
         EFACE(3)=1
         EFACE(4)=3
         EFACE(5)=5
         EFACE(6)=6
C
C        Assign inverse of EB's numbering scheme to PF's scheme.
C
         EFACE1(1)=3
         EFACE1(2)=2
         EFACE1(3)=4
         EFACE1(4)=1
         EFACE1(5)=5
         EFACE1(6)=6
C
      ENDIF
      CALL COLOR(icolr)
      CALL PENW(1)
      NXYZ = NX*NY*NZ
      NSKIPX(1) = 1
      NSKIPX(2) = NX
      NSKIPX(3) = NX*NY
C
C     Fill corners - 1 through 8.
C
      IC(1) =                                 0
      IC(2) =                             (NX-1)
      IC(3) =                 NX*(NY-1)
      IC(4) =                 NX*(NY-1) + (NX-1)
      IC(5) =  NX*NY*(NZ-1)
      IC(6) =  NX*NY*(NZ-1)             + (NX-1)
      IC(7) =  NX*NY*(NZ-1) + NX*(NY-1)
      IC(8) =  NX*NY*(NZ-1) + NX*(NY-1) + (NX-1)
C
      IF (NCSOLD.NE.NCSEGS) THEN
C
C        Set up curve line segment evaluation array (exclude endpoints) :
C
         NCSOLD = NCSEGS
         NCSEG1 = NCSEGS + 1
         NCSGM1 = NCSEGS - 1
         DR1 = 2.0 / FLOAT(NCSEGS)
         DO 400 K=1,NCSGM1
            RR1 = -1.0 + DR1*FLOAT(K)
            CALL HLEGN(HIK(1,K),RR1,ZPTS,NX)
  400    CONTINUE
      ENDIF
C
C     Draw element edges (for now, we do the 4X redundant work)
C
      NEDGE=4
      IF (IF3D) NEDGE=12
         IEOFF=NXYZ*(IE-1)+1
         call minmax(xmn,xmx,xp(ieoff),nxyz)
         call minmax(ymn,ymx,yp(ieoff),nxyz)
c        if (if3d) then
c           call minmax(zmn,zmx,zp(ieoff),nxyz)
c           write(S,'(A8,3g12.4,A1)') 'XYZ min:',xmn,ymn,zmn,'$'
c           CALL PRS (S)
c           write(S,'(A8,3g12.4,A1)') 'XYZ max:',xmx,ymx,zmx,'$'
c           CALL PRS (S)
c        else
c           write(S,'(A7,2g12.4,A1)') 'XY min:',xmn,ymn,'$'
c           CALL PRS (S)
c           write(S,'(A7,2g12.4,A1)') 'XY max:',xmx,ymx,'$'
c           CALL PRS (S)
c        endif
C
C        Check if we can take a short cut in plotting the element bdry.
         IFCURV=.FALSE.
         DO 500 IEDG = 1,8
            IF (CCURVE(IEDG,IE).NE.' ') IFCURV=.TRUE.
  500    CONTINUE
C
C        Edges
C
         DO 800 IEDG = 1,NEDGE
C
C           There is a bug in this; I will make all edges red
c           IF (CBC(IEDG,IE,1).EQ.'W  '.AND..NOT.IF3D) THEN
c              CALL COLOR(1)
c           ELSE
c              CALL COLOR(9)
c           ENDIF
               CALL COLOR(icolr)
            IEDG0 = IEDG + 12
            IF (IF3D) IEDG0 = IEDG
            ICE1 = ICEDG(1,IEDG0)
            ICE2 = ICEDG(2,IEDG0)
            NSKIP = NSKIPX(ICEDG(3,IEDG0))
            IC0 = IEOFF+IC(ICE1)
            IC1 = IEOFF+IC(ICE2)
            XK(     1) = XP(IC0)
            YK(     1) = YP(IC0)
            ZK(     1) = ZP(IC0)
            XK(NCSEG1) = XP(IC1)
            YK(NCSEG1) = YP(IC1)
            ZK(NCSEG1) = ZP(IC1)
            IF (IFCURV) THEN
               CALL EVLINE(XK(2),XP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(YK(2),YP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(ZK(2),ZP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL VDRAW3A(XK,YK,ZK,NCSEG1)
            ELSE
               XK(2) = XP(IC1)
               YK(2) = YP(IC1)
               ZK(2) = ZP(IC1)
               CALL VDRAW3A(XK,YK,ZK,2)
            ENDIF
  800    CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
