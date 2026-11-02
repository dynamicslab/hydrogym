c-----------------------------------------------------------------------
      subroutine setquad 
C 
C     Set up the quadrant points and associated element-element links
C     for plotting r-s-t surfaces in 3D and for establishing general
C     zipper planes.        pff 11-20-90  21:30 and  1-8-91  
C
C     (Note: this algorithm is inherently CONFORMING, as constant r-s-t
C            planes have little significance in a non-conforming
C            discretization where the continuum assumption is not honored.)
C
      include 'basics.inc'
      include 'basicsp.inc'
C
C     Algorithm:  Set up indices, e.g:
C                                                     ^ s
C         These points define planes                  |          
C         which propagate from one         +----------|---------+
C         element to the next, as          |          |         |
C         shown near pt. 3:                |         4+         |
C                                          |          |         |
C                                          |          |         |
C                                          +--1+------------+2--+--> r
C                                          |          |         |
C                                          |          |         |
C                                          | - - - - 3+ - - - - |
C                                          |          |         |
C                                          +--------------------+
C
C     Set up the connectivity list by performing the K^2 loop which
C     searches for coincident nodes, thus identifying an element-element
C     interaction.
C
CXXX   
CXXX   recall, that normals will always be defined positive if they point
CXXX   away from the element center...,,, thus, once the topology of the 
CXXX   normal (in or out) is established for one element, it will carry through
CXXX   to the rest of the elements.
CXXX   
CXXX   2)  we can have a pseudo XYZ plane for now which takes as the default
CXXX   normal to be x y or z at the given point.
CXXX   
CXXX   3)  when determining which  element contains the originating point, just
CXXX       use the first found point, and choose the closest quadpoint with an
CXXX       appropriately oriented normal vector (in some distance/normal norm).
CXXX   
C
      DIMENSION XYZ(3,8) 
      DIMENSION INEIGH(6,6)
      DATA      INEIGH/  1, -1, -1,  1,  1, -1
     $                , -1,  1,  1, -1, -1,  1
     $                , -1,  1,  1, -1, -1,  1
     $                ,  1, -1, -1,  1,  1, -1
     $                ,  1, -1, -1,  1,  1, -1
     $                , -1,  1,  1, -1, -1,  1 /
C
      IF (.NOT.IF3D) return
      CALL GENCEN
      NFACES=6
      EPSM = 1.0E-10
      NEL5=120*NELM
      CALL RZERO(XYZQ,NEL5)

c     if (nel.gt.20000) then
c        call prsi('Will not set quad for nel > 20K. NEL=$',nel)
c        return
c     endif

C     Define quadpoints:
      DO 100 IE=1,NEL
         CALL MAPXC(XYZ,IE)
C
C     Plane 1
         XYZQ(1,1,1,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,5))+XYZ(1,2)+XYZ(1,6))
         XYZQ(2,1,1,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,5))+XYZ(2,2)+XYZ(2,6))
         XYZQ(3,1,1,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,5))+XYZ(3,2)+XYZ(3,6))
C
         XYZQ(1,2,1,IE)=.125*(3.0*(XYZ(1,3)+XYZ(1,7))+XYZ(1,4)+XYZ(1,8))
         XYZQ(2,2,1,IE)=.125*(3.0*(XYZ(2,3)+XYZ(2,7))+XYZ(2,4)+XYZ(2,8))
         XYZQ(3,2,1,IE)=.125*(3.0*(XYZ(3,3)+XYZ(3,7))+XYZ(3,4)+XYZ(3,8))
C
         XYZQ(1,3,1,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,3))+XYZ(1,2)+XYZ(1,4))
         XYZQ(2,3,1,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,3))+XYZ(2,2)+XYZ(2,4))
         XYZQ(3,3,1,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,3))+XYZ(3,2)+XYZ(3,4))
C
         XYZQ(1,4,1,IE)=.125*(3.0*(XYZ(1,5)+XYZ(1,7))+XYZ(1,6)+XYZ(1,8))
         XYZQ(2,4,1,IE)=.125*(3.0*(XYZ(2,5)+XYZ(2,7))+XYZ(2,6)+XYZ(2,8))
         XYZQ(3,4,1,IE)=.125*(3.0*(XYZ(3,5)+XYZ(3,7))+XYZ(3,6)+XYZ(3,8))
C
C     Plane 2
         XYZQ(1,1,2,IE)=.125*(3.0*(XYZ(1,2)+XYZ(1,6))+XYZ(1,1)+XYZ(1,5))
         XYZQ(2,1,2,IE)=.125*(3.0*(XYZ(2,2)+XYZ(2,6))+XYZ(2,1)+XYZ(2,5))
         XYZQ(3,1,2,IE)=.125*(3.0*(XYZ(3,2)+XYZ(3,6))+XYZ(3,1)+XYZ(3,5))
C
         XYZQ(1,2,2,IE)=.125*(3.0*(XYZ(1,4)+XYZ(1,8))+XYZ(1,3)+XYZ(1,7))
         XYZQ(2,2,2,IE)=.125*(3.0*(XYZ(2,4)+XYZ(2,8))+XYZ(2,3)+XYZ(2,7))
         XYZQ(3,2,2,IE)=.125*(3.0*(XYZ(3,4)+XYZ(3,8))+XYZ(3,3)+XYZ(3,7))
C
         XYZQ(1,3,2,IE)=.125*(3.0*(XYZ(1,2)+XYZ(1,4))+XYZ(1,1)+XYZ(1,3))
         XYZQ(2,3,2,IE)=.125*(3.0*(XYZ(2,2)+XYZ(2,4))+XYZ(2,1)+XYZ(2,3))
         XYZQ(3,3,2,IE)=.125*(3.0*(XYZ(3,2)+XYZ(3,4))+XYZ(3,1)+XYZ(3,3))
C
         XYZQ(1,4,2,IE)=.125*(3.0*(XYZ(1,6)+XYZ(1,8))+XYZ(1,5)+XYZ(1,7))
         XYZQ(2,4,2,IE)=.125*(3.0*(XYZ(2,6)+XYZ(2,8))+XYZ(2,5)+XYZ(2,7))
         XYZQ(3,4,2,IE)=.125*(3.0*(XYZ(3,6)+XYZ(3,8))+XYZ(3,5)+XYZ(3,7))
C
C     Plane 5
         XYZQ(1,1,5,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,3))+XYZ(1,5)+XYZ(1,7))
         XYZQ(2,1,5,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,3))+XYZ(2,5)+XYZ(2,7))
         XYZQ(3,1,5,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,3))+XYZ(3,5)+XYZ(3,7))
C
         XYZQ(1,2,5,IE)=.125*(3.0*(XYZ(1,2)+XYZ(1,4))+XYZ(1,6)+XYZ(1,8))
         XYZQ(2,2,5,IE)=.125*(3.0*(XYZ(2,2)+XYZ(2,4))+XYZ(2,6)+XYZ(2,8))
         XYZQ(3,2,5,IE)=.125*(3.0*(XYZ(3,2)+XYZ(3,4))+XYZ(3,6)+XYZ(3,8))
C
         XYZQ(1,3,5,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,2))+XYZ(1,5)+XYZ(1,6))
         XYZQ(2,3,5,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,2))+XYZ(2,5)+XYZ(2,6))
         XYZQ(3,3,5,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,2))+XYZ(3,5)+XYZ(3,6))
C
         XYZQ(1,4,5,IE)=.125*(3.0*(XYZ(1,3)+XYZ(1,4))+XYZ(1,7)+XYZ(1,8))
         XYZQ(2,4,5,IE)=.125*(3.0*(XYZ(2,3)+XYZ(2,4))+XYZ(2,7)+XYZ(2,8))
         XYZQ(3,4,5,IE)=.125*(3.0*(XYZ(3,3)+XYZ(3,4))+XYZ(3,7)+XYZ(3,8))
C
C     Plane 6
         XYZQ(1,1,6,IE)=.125*(3.0*(XYZ(1,5)+XYZ(1,7))+XYZ(1,1)+XYZ(1,3))
         XYZQ(2,1,6,IE)=.125*(3.0*(XYZ(2,5)+XYZ(2,7))+XYZ(2,1)+XYZ(2,3))
         XYZQ(3,1,6,IE)=.125*(3.0*(XYZ(3,5)+XYZ(3,7))+XYZ(3,1)+XYZ(3,3))
C
         XYZQ(1,2,6,IE)=.125*(3.0*(XYZ(1,6)+XYZ(1,8))+XYZ(1,2)+XYZ(1,4))
         XYZQ(2,2,6,IE)=.125*(3.0*(XYZ(2,6)+XYZ(2,8))+XYZ(2,2)+XYZ(2,4))
         XYZQ(3,2,6,IE)=.125*(3.0*(XYZ(3,6)+XYZ(3,8))+XYZ(3,2)+XYZ(3,4))
C
         XYZQ(1,3,6,IE)=.125*(3.0*(XYZ(1,5)+XYZ(1,6))+XYZ(1,1)+XYZ(1,2))
         XYZQ(2,3,6,IE)=.125*(3.0*(XYZ(2,5)+XYZ(2,6))+XYZ(2,1)+XYZ(2,2))
         XYZQ(3,3,6,IE)=.125*(3.0*(XYZ(3,5)+XYZ(3,6))+XYZ(3,1)+XYZ(3,2))
C
         XYZQ(1,4,6,IE)=.125*(3.0*(XYZ(1,7)+XYZ(1,8))+XYZ(1,3)+XYZ(1,4))
         XYZQ(2,4,6,IE)=.125*(3.0*(XYZ(2,7)+XYZ(2,8))+XYZ(2,3)+XYZ(2,4))
         XYZQ(3,4,6,IE)=.125*(3.0*(XYZ(3,7)+XYZ(3,8))+XYZ(3,3)+XYZ(3,4))
C
C     Plane 3
         XYZQ(1,1,3,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,5))+XYZ(1,3)+XYZ(1,7))
         XYZQ(2,1,3,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,5))+XYZ(2,3)+XYZ(2,7))
         XYZQ(3,1,3,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,5))+XYZ(3,3)+XYZ(3,7))
C
         XYZQ(1,2,3,IE)=.125*(3.0*(XYZ(1,2)+XYZ(1,6))+XYZ(1,4)+XYZ(1,8))
         XYZQ(2,2,3,IE)=.125*(3.0*(XYZ(2,2)+XYZ(2,6))+XYZ(2,4)+XYZ(2,8))
         XYZQ(3,2,3,IE)=.125*(3.0*(XYZ(3,2)+XYZ(3,6))+XYZ(3,4)+XYZ(3,8))
C
         XYZQ(1,3,3,IE)=.125*(3.0*(XYZ(1,1)+XYZ(1,2))+XYZ(1,3)+XYZ(1,4))
         XYZQ(2,3,3,IE)=.125*(3.0*(XYZ(2,1)+XYZ(2,2))+XYZ(2,3)+XYZ(2,4))
         XYZQ(3,3,3,IE)=.125*(3.0*(XYZ(3,1)+XYZ(3,2))+XYZ(3,3)+XYZ(3,4))
C
         XYZQ(1,4,3,IE)=.125*(3.0*(XYZ(1,5)+XYZ(1,6))+XYZ(1,7)+XYZ(1,8))
         XYZQ(2,4,3,IE)=.125*(3.0*(XYZ(2,5)+XYZ(2,6))+XYZ(2,7)+XYZ(2,8))
         XYZQ(3,4,3,IE)=.125*(3.0*(XYZ(3,5)+XYZ(3,6))+XYZ(3,7)+XYZ(3,8))
C
C     Plane 4
         XYZQ(1,1,4,IE)=.125*(3.0*(XYZ(1,3)+XYZ(1,7))+XYZ(1,1)+XYZ(1,5))
         XYZQ(2,1,4,IE)=.125*(3.0*(XYZ(2,3)+XYZ(2,7))+XYZ(2,1)+XYZ(2,5))
         XYZQ(3,1,4,IE)=.125*(3.0*(XYZ(3,3)+XYZ(3,7))+XYZ(3,1)+XYZ(3,5))
C
         XYZQ(1,2,4,IE)=.125*(3.0*(XYZ(1,4)+XYZ(1,8))+XYZ(1,2)+XYZ(1,6))
         XYZQ(2,2,4,IE)=.125*(3.0*(XYZ(2,4)+XYZ(2,8))+XYZ(2,2)+XYZ(2,6))
         XYZQ(3,2,4,IE)=.125*(3.0*(XYZ(3,4)+XYZ(3,8))+XYZ(3,2)+XYZ(3,6))
C
         XYZQ(1,3,4,IE)=.125*(3.0*(XYZ(1,3)+XYZ(1,4))+XYZ(1,1)+XYZ(1,2))
         XYZQ(2,3,4,IE)=.125*(3.0*(XYZ(2,3)+XYZ(2,4))+XYZ(2,1)+XYZ(2,2))
         XYZQ(3,3,4,IE)=.125*(3.0*(XYZ(3,3)+XYZ(3,4))+XYZ(3,1)+XYZ(3,2))
C
         XYZQ(1,4,4,IE)=.125*(3.0*(XYZ(1,7)+XYZ(1,8))+XYZ(1,5)+XYZ(1,6))
         XYZQ(2,4,4,IE)=.125*(3.0*(XYZ(2,7)+XYZ(2,8))+XYZ(2,5)+XYZ(2,6))
         XYZQ(3,4,4,IE)=.125*(3.0*(XYZ(3,7)+XYZ(3,8))+XYZ(3,5)+XYZ(3,6))
C
  100 CONTINUE
C
C     Find Neighbor-Neighbor connections:
C
      DO 500 IE=1,NEL
      DO 500 IFACE=1,NFACES
      DO 500 IQPT=1,4
C
         NEIGHI=INT(XYZQ(4,IQPT,IFACE,IE))
         IF (NEIGHI.EQ.0) THEN
            IQP1=IQPT+1
            IQP1=MOD1(IQP1,4)
            EPSI=(XYZQ(1,IQPT,IFACE,IE)-XYZQ(1,IQP1,IFACE,IE))**2
     $          +(XYZQ(2,IQPT,IFACE,IE)-XYZQ(2,IQP1,IFACE,IE))**2
     $          +(XYZQ(3,IQPT,IFACE,IE)-XYZQ(3,IQP1,IFACE,IE))**2
C
c           DO 204 JE=1,NEL
c   pff: 4/13/93
            DO 204 JE=ie,Nel
            IF (JE.EQ.IE) GOTO 202
            DIST=(XCEN(IE)-XCEN(JE))**2+(YCEN(IE)-YCEN(JE))**2
     $          +(ZCEN(IE)-ZCEN(JE))**2
            DST2=(RCEN(IE)+RCEN(JE))**2
c           write(6,101) ie,je,dst2,dist,rcen(ie),rcen(je)
  101       format(2x,'dist:',2i3,4f12.7)
            IF (DST2.LT.DIST) GOTO 202
            DO 200 JFACE=1,NFACES
            DO 200 JQPT=1,4
c   pff: 4/13/93
            NEIGHJ=INT(XYZQ(4,IQPT,IFACE,IE))
            IF (NEIGHJ.ne.0) GOTO 200
c   pff: 4/13/93
               JQP1=JQPT+1
               JQP1=MOD1(JQP1,4)
               EPSJ=(XYZQ(1,JQPT,JFACE,JE)-XYZQ(1,JQP1,JFACE,JE))**2
     $             +(XYZQ(2,JQPT,JFACE,JE)-XYZQ(2,JQP1,JFACE,JE))**2
     $             +(XYZQ(3,JQPT,JFACE,JE)-XYZQ(3,JQP1,JFACE,JE))**2
               EPS=EPSM*(EPSJ+EPSI)
               EPS=MAX(EPS,EPSM)
               DIST=(XYZQ(1,IQPT,IFACE,IE)-XYZQ(1,JQPT,JFACE,JE))**2
     $             +(XYZQ(2,IQPT,IFACE,IE)-XYZQ(2,JQPT,JFACE,JE))**2
     $             +(XYZQ(3,IQPT,IFACE,IE)-XYZQ(3,JQPT,JFACE,JE))**2
               IF (DIST.LE.EPS) THEN
C                 We found our match, make a note of it 
C                 and move to the next point.
                  NEIGHI=24*(JE-1)+4*(JFACE-1)+JQPT
                  NEIGHJ=24*(IE-1)+4*(IFACE-1)+IQPT
                  INRM=INEIGH(IFACE,JFACE)
                  XYZQ(4,NEIGHJ,1,1)=FLOAT(INRM*NEIGHI)
                  XYZQ(4,NEIGHI,1,1)=FLOAT(INRM*NEIGHJ)
                  GOTO 300
               endif
  200       CONTINUE
  202       CONTINUE
  204       CONTINUE
  300       CONTINUE
         endif
  500 CONTINUE
c
c     Initialize interp routine ---
c
      CALL INTERP(VAL1,xp(1),yp(1),zp(1),idum,wkv1,ierr)
C
C     Diagnostic (1-8-91)
C
c     CALL DRQUAD
C
      return
      end
c-----------------------------------------------------------------------
      subroutine drquad
      include 'basics.inc'
C
      NFACES=6
      DO 600 IE=1,NEL
      DO 600 IFACE=1,NFACES
      DO 600 IQPT=1,4
         ICLR=2*IFACE
c        ICLR=MOD1(ICLR,14)
c        ICLR=5
         CALL COLOR(ICLR)
         CALL XDIAM3(XYZQ(1,IQPT,IFACE,IE))
  600 CONTINUE
C
      return
      end
c-----------------------------------------------------------------------
      subroutine xdiam3(xyz)
c     draws a diamond around XX,YY
      include 'basics.inc'
C
      DIMENSION XYZ(3)
      XX=XISO(XYZ(1),XYZ(2),XYZ(3))
      YY=YISO(XYZ(1),XYZ(2),XYZ(3))
      CALL MOVEC(XX+.0025*XFAC ,YY+.0025*YFAC)
      CALL DRAWC(XX-.0025*XFAC ,YY+.0025*YFAC)
      CALL DRAWC(XX-.0025*XFAC ,YY-.0025*YFAC)
      CALL DRAWC(XX+.0025*XFAC ,YY-.0025*YFAC)
      CALL DRAWC(XX+.0025*XFAC ,YY+.0025*YFAC)
      return
      end
c-----------------------------------------------------------------------
      subroutine mapxc(xyz,ie)
C     Map the element corner points (x,y,z) to standard vector.
      include 'basics.inc'
      DIMENSION INDX(8)
      DIMENSION XYZ(3,8)
      INDX(1)=1
      INDX(2)=2
      INDX(3)=4
      INDX(4)=3
      INDX(5)=5
      INDX(6)=6
      INDX(7)=8
      INDX(8)=7
      NDIM2 = 2**NDIM
C
      DO 50 IX=1,NDIM2
         I=INDX(IX)
         XYZ(1,IX)=X(IE,I)
         XYZ(2,IX)=Y(IE,I)
         XYZ(3,IX)=Z(IE,I)
   50 CONTINUE
      return
      end
c-----------------------------------------------------------------------
      subroutine decod(ix,iy,iz,ie,Jxyz,nx,ny,nz)
C
      ixyz = abs(jxyz)
      NXY  = NX*NY
      NXYZ = NXY*NZ
C
      IE  = (IXYZ + NXYZ - 1)/NXYZ
      IXY = IXYZ-(IE-1)*NXYZ
      IZ  = (IXY + NXY - 1)/NXY
      IXX = IXY-(IZ-1)*NXY
      IY  = (IXX + NX - 1)/NX
      IX  = IXX-(IY-1)*NX
C
C     Check
C
      ICHECK = (IE-1)*NXYZ + (IZ-1)*NXY + (IY-1)*NX + IX
      IF (ICHECK.NE.IXYZ) THEN
         WRITE(6,*) 'hey, what gives here?  icheck,ixyz:',icheck,ixyz
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine setsrfp(srfqal,nqual,ifld)
      CHARACTER*1 SRFQAL(1)
      CHARACTER*1 CB1,CB
C
      include 'basics.inc'
C
C     Select a list of planes according to the specified boundary
C     qualifications given in SRFQAL.  If the number of qualifications,
C     NQUAL, is negative, select all boundaries NOT meeting the 
C     qualifications.
C
C     Note:  Modification 10/2/92 -
C            Set INRM such that the "+" direction associated with the
C            plane has the same direction as the closest fundamental
C            unit vector (e_1,e_2, or e_3).    pff.
C
C
c     NLSTP=0                      pff 12/21/96
      NFACES=2*NDIM
C
      if (nqual.gt.0) then
c
         do iq=1,nqual
            cb1=srfqal(iq)
            if (cb1.eq.'1'.or.cb1.eq.'2'.or.cb1.eq.'3'
     $      .or.cb1.eq.'4'.or.cb1.eq.'5'.or.cb1.eq.'6') then
               read(cb1,1) iface
    1          format(i1)
               do ie=1,nel
                  nlstp=nlstp+1
                  ijkpln=eface1(iface)
                  ix=1
                  if (mod(ijkpln,2).eq.0) ix=nx
                  list(nlstp)=6*nx*(ie-1)+nx*(ijkpln-1)+ix
                  call signpl(list(nlstp))
               enddo
            else
               do ie=1,nel
               do iface=1,nfaces
                  cb=cbc(iface,ie,ifld)
                  if (cb.eq.cb1) then
                     nlstp=nlstp+1
                     ijkpln=eface1(iface)
                     ix=1
                     if (mod(ijkpln,2).eq.0) ix=nx
                     list(nlstp)=6*nx*(ie-1)+nx*(ijkpln-1)+ix
                     call signpl(list(nlstp))
                  endif
               enddo
               enddo
            endif
         enddo
C
      ELSE
C
c        open(unit=55,file='bdry')
         NQ=-NQUAL
         DO 300 IE=1,NEL
         DO 300 IFACE=1,NFACES
            CB=CBC(IFACE,IE,IFLD)
            DO 200 IQ=1,NQ
               CB1=SRFQAL(IQ)
C              if it's one of these, don't plot.
               IF (CB1.EQ.CB) GOTO 300
  200       CONTINUE
C
C           Since the b.c. isn't one of the one's specified, we plot
C           this surface....
            NLSTP=NLSTP+1
            IJKPLN=EFACE1(IFACE)
c           call outvtx(ie,iface,55)
            IX=1
            IF (MOD(IJKPLN,2).EQ.0) IX=NX
            LIST(NLSTP)=6*NX*(IE-1)+NX*(IJKPLN-1)+IX
            CALL SIGNPL(LIST(NLSTP))
  300    CONTINUE
c        close(unit=55)
C
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine sortl
      include 'basics.inc'
      include 'basicsp.inc'
      COMMON /CTMPS1/ IND(NELM,6),WK(NELM,6)
C
      NXY =NX*NY
      NXYZ=NZ*NXY
c
c     March over all mirror configurations, all planes
c
      i = 0
      do im  =1,nmirror
         do ipl =1,nlstp
            i = i+1
            LISTA=ABS(LIST(ipl))
            CALL DECOD(IPLANE,IPLN,IE,IDUM,LISTA,NX,6,NELM)
c
c           Z-sort based on centerpoint of selected plane
c
            IX=(NX+1)/2
            IY=(NY+1)/2
            IZ=(NZ+1)/2
            IF (IPLN.LE.2) THEN
               IX=IPLANE
            ELSEIF (IPLN.LE.4) THEN
               IY=IPLANE
            ELSE
               IZ=IPLANE
            endif
c
            IEOFF=NXYZ*(IE-1)+NXY*(IZ-1)+NX*(IY-1)+IX
            if (im.eq.1) then
               ZDEPTH(I,1)=ZISO(XP(IEOFF),YP(IEOFF),ZP(IEOFF))
            else
               call mirror(xmi,ymi,zmi,xp(ieoff),yp(ieoff),zp(ieoff),im)
               ZDEPTH(I,1)=ZISO(xmi,ymi,zmi)
            endif
            lmir(i) = im
         enddo
         call icopy(lisw(1+(im-1)*nlstp),list,nlstp)
      enddo
      CALL SORT(ZDEPTH,   IND,NLSTP*nmirror)
      CALL ISWAP(LISW ,WK,IND,NLSTP*nmirror)
      CALL ISWAP(lmir ,WK,IND,NLSTP*nmirror)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine signpl(lplane)
C
C     Find the closest component of (1,1,1) to the normal associated
C     with this plane, and assign the sign of the dot product of that
C     component with the normal to Lplane.
C     11/92 pff.
C
      include 'basics.inc'
      include 'basicsp.inc'
      DIMENSION VEC0(3),VEC1(3),VEC2(3),VEC3(3)
C
      CALL DECOD(IPLANE,IPLN,IE,IDUM,Lplane,NX,6,NELM)
C
      I=(1+NX)/2
      J=(1+NY)/2
      K=(1+NZ)/2
C
      Nxyz=Nx*Ny*Nz
      Nxy =Nx*Ny
C
      IF (IPLN.LE.2) THEN
         I =IPLANE
         ieoff1 = nxyz*(ie-1)+nxy*(k-1)+nx*(j  )+i
         ieoff2 = nxyz*(ie-1)+nxy*(k  )+nx*(j-1)+i
      ELSEIF (IPLN.LE.4) THEN
         J =IPLANE
         ieoff1 = nxyz*(ie-1)+nxy*(k-1)+nx*(j-1)+i+1
         ieoff2 = nxyz*(ie-1)+nxy*(k  )+nx*(j-1)+i
      ELSE
         K =IPLANE
         ieoff1 = nxyz*(ie-1)+nxy*(k-1)+nx*(j-1)+i+1
         ieoff2 = nxyz*(ie-1)+nxy*(k-1)+nx*(j  )+i
      endif
      ieoff  = nxyz*(ie-1)+nxy*(k-1)+nx*(j-1)+i
C
      VEC0(1)=Xp(ieoff)
      VEC0(2)=Yp(ieoff)
      VEC0(3)=Zp(ieoff)
C
      VEC1(1)=Xp(ieoff1)
      VEC1(2)=Yp(ieoff1)
      VEC1(3)=Zp(ieoff1)
C
      VEC2(1)=Xp(ieoff2)
      VEC2(2)=Yp(ieoff2)
      VEC2(3)=Zp(ieoff2)
C
      CALL SUB3  (VEC1,VEC1,VEC0,3)
      CALL SUB3  (VEC2,VEC2,VEC0,3)
      CALL CROSS (VEC0,VEC1,VEC2)
      CALL NORM3D(VEC0)
C
      Vmax=0.0
      DO 10 I=1,3
         IF (ABS(VEC0(i)).GT.Vmax) THEN
             Vmax=ABS(VEC0(i))
             Imax=i
         endif
   10 CONTINUE
      IF (VEC0(Imax).LT.0.0) Lplane=-abs(Lplane)
c     write(6,*) ie,vec0(imax),lplane,' signpl'
      return
      end
c-----------------------------------------------------------------------
      subroutine gencen
      include 'basics.inc'
C
C     Generate the element centers
C
      IF (IF3D) THEN
         DO 100 IEL=1,NEL
            XCEN(IEL)=( X(IEL,1)+X(IEL,2)+X(IEL,3)+X(IEL,4)
     $               +  X(IEL,5)+X(IEL,6)+X(IEL,7)+X(IEL,8))/8.
            YCEN(IEL)=( Y(IEL,1)+Y(IEL,2)+Y(IEL,3)+Y(IEL,4)
     $               +  Y(IEL,5)+Y(IEL,6)+Y(IEL,7)+Y(IEL,8))/8.
            ZCEN(IEL)=( Z(IEL,1)+Z(IEL,2)+Z(IEL,3)+Z(IEL,4)
     $               +  Z(IEL,5)+Z(IEL,6)+Z(IEL,7)+Z(IEL,8))/8.
  100    CONTINUE
      ELSE
         DO 200 IEL=1,NEL
            XCEN(IEL)=(X(IEL,1)+X(IEL,2)+X(IEL,3)+X(IEL,4))/4.
            YCEN(IEL)=(Y(IEL,1)+Y(IEL,2)+Y(IEL,3)+Y(IEL,4))/4.
            ZCEN(IEL)=0.0
  200    CONTINUE
      endif
C
C     Compute the maximum radius from the center
C
      CALL RZERO(RCEN,NEL) 
      IF (IF3D) THEN
         DO 300 IEL=1,NEL
         DO 300 J=1,8
            RAD=(X(IEL,J)-XCEN(IEL))**2 + (Y(IEL,J)-YCEN(IEL))**2
     $         +(Z(IEL,J)-ZCEN(IEL))**2
            RCEN(IEL)=MAX(RCEN(IEL),RAD)
  300    CONTINUE
      ELSE
         DO 400 IEL=1,NEL
         DO 400 J=1,4
            RAD=(X(IEL,J)-XCEN(IEL))**2 + (Y(IEL,J)-YCEN(IEL))**2
            RCEN(IEL)=MAX(RCEN(IEL),RAD)
  400    CONTINUE
      endif
      CALL VSQRT(RCEN,NEL)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine clip_plane
C     Set up list of candidate RST planes
      include 'basics.inc'
      include 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      DIMENSION VEC(3),VEC1(3),VEC2(3),VEC3(3)
      CHARACTER*1 cplane,dplane
      LOGICAL IFANY,IFTMP
c
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
      CALL PRS('Clip in X, Y, or Z ?$')
      CALL RES(cplane,1)
      call capit(cplane,1)
      write(6,*)'cplane : ',cplane
c
      CALL PRS('Enter location of clipping plane:$')
      CALL RER(clipping_plane)
c
      CALL PRS
     $('Enter "<" or ">" if you wish to remove below or above plane.$')
      CALL RES(dplane,1)
c
C     3-D
C
      inew = 0
      DO 6000 I=1,NLSTP
C
C           New plotting algorithms for color fill and fishnet - pff 7-1-90
C           (3D will come later - 10-10-90)
C
         LISTA=ABS(LIST(I))
         CALL DECOD(IPLANE,IPLN,IE,IDUM,LISTA,NX,6,NELM)
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
         endif
c
         DO 5001 IZ=K1,K2
         DO 5001 IY=J1,J2
         DO 5001 IX=I1,I2
            IPOINT=IND(IX,IY,IZ,IE)
c
            if (cplane.eq.'X') qq = xp(ipoint)
            if (cplane.eq.'Y') qq = yp(ipoint)
            if (cplane.eq.'Z') qq = zp(ipoint)
c
c           See if we delete this plane from the list
            if (dplane.eq.'<'.and.qq.lt.clipping_plane) goto 5002
            if (dplane.eq.'>'.and.qq.gt.clipping_plane) goto 5002
c
 5001    CONTINUE
c        If we get here, then all points are included
         inew = inew+1
         list(inew) = list(i)
c
 5002    CONTINUE
 6000 CONTINUE
      nlstp = inew
      return
      end
c-----------------------------------------------------------------------
      subroutine mirror(xm,ym,zm,xi,yi,zi,m)
c
c     Based on mirror specification number, m,  compute the reflected
c     image of xi,yi,zi
c
c     For now, we support only one mirror -- that is the y=0 plane
c
      xm =  xi
      ym = -yi
      zm =  zi
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vmirror(tm,xm,ym,zm,n,m)
      real tm(n,n),xm(n,n),ym(n,n),zm(n,n)
c
c     Based on mirror specification number, m,  compute the reflected
c     image of xi,yi,zi
c
c     For now, we support only one mirror -- that is the y=0 plane
c
      if (m.eq.2) then
         do i=1,n*n
            ym(i,1) = -ym(i,1)
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outvtx(ie,ifce,io)
c
      include 'basics.inc'
c
      integer fv3(4,6)
      save    fv3
      data    fv3 /  1,2,5,6
     $            ,  2,3,6,7
     $            ,  3,4,7,8
     $            ,  1,4,5,8
     $            ,  1,2,3,4
     $            ,  5,6,7,8  /
c
c     Output coordinates of this face
c
      xa = 0
      ya = 0
      za = 0
c
      do iv=1,4
         xa = xa+0.25*x(ie,fv3(iv,ifce))
         ya = ya+0.25*y(ie,fv3(iv,ifce))
         za = za+0.25*z(ie,fv3(iv,ifce))
      enddo
      write(io,1) xa,ya,za
    1 format(3f14.4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine assign_bc
c     
c     prompt user for new bc to be applied to currently selected planes
c     
      include 'basics.inc'
      include 'basicsp.inc'
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gaujordf(a,m,n,indr,indc,ipiv,ierr)
C
C     Gauss-Jordan matrix inversion with full pivoting
c
c     Num. Rec. p. 30, 2nd Ed., Fortran
c
C
C     A is an nxn matrix, assumed contiguous in memory
C     W is a  work array of dimension n
C
      real    a(m,n)
      integer indr(m),indc(n),ipiv(n)
c
      real rmult
      integer wdsize
c
      ierr = 0
c
      eps = 1.e-12
      one = 1.
      wdsize = 8
      if (one+eps.eq.1.) wdsize = 4
c
      call izero(ipiv,n)
c
c
      do j=1,n
         amx=0.
c
c        Pivot search
         do i=1,m
            if (ipiv(i).ne.1) then
               do k=1,n
                  if (ipiv(k).eq.0) then
                    if (abs(a(i,k)).ge.amx) then
                       amx = abs(a(i,k))
                       ir  = i
                       jc  = k
                    endif
                 elseif (ipiv(k).gt.1) then
                    ierr = -ipiv(k)
                    return
                 endif
              enddo
           endif
        enddo
        ipiv(jc) = ipiv(jc) + 1
c
c       Swap rows
        if (ir.ne.jc) then
           do k=1,n
              tmp     = a(ir,k)
              a(ir,k) = a(jc,k)
              a(jc,k) = tmp
           enddo
        endif
        indr(j)=ir
        indc(j)=jc
        if (abs(a(jc,jc)).lt.eps) then
           write(6,*) 'small Gauss Jordan Piv:',jc,a(jc,jc)
           ierr = jc
           return
        endif
        piv = 1./a(jc,jc)
        a(jc,jc)=1.
        do k=1,n
           a(jc,k) = a(jc,k)*piv
        enddo
c
        do i=1,m
           if (i.ne.jc) then
              rmult   = a(i,jc)
              a(i,jc) = 0.
              do k=1,n
                 a(i,k) = a(i,k) - rmult*a(jc,k)
              enddo
           endif
        enddo
      enddo
c
c     Unscramble matrix
      do k=n,1,-1
         if (indr(k).ne.indc(k)) then
            do i=1,m
               tmp=a(i,indr(k))
               a(i,indr(k))=a(i,indc(k))
               a(i,indc(k))=tmp
            enddo
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine shiftcg
c     
c     prompt user for new cg and translate geometry to this location
c     
      include 'basics.inc'
      include 'basicsp.inc'
c
      real cgx,cgy,cgz
      real ngx,ngy,ngz
      character*80 fnm
c
      ntot= nx*ny*nz*nel
      mtot= nelm*8
      if (if3d) then
         call vol_int(cgx,vol,xp)
         call vol_int(cgy,vol,yp)
         call vol_int(cgz,vol,zp)
         cgx = cgx/vol
         cgy = cgy/vol
         cgz = cgz/vol
         call prsrrr('Old CG: $',cgx,cgy,cgz)
c
c        call prs   ('Input new CG: $')
c        call rerrr (ngx,ngy,ngz)
c
         call prs ('Input file containing new matrix:$')
         call blank(fnm,80)
         call res (fnm,80)
         open (unit=88,file=fnm,status='old',err=999)
         read (88,*,err=9999,end=9999) ngx,ngy,ngz
         close (unit=88)
c
         dx = ngx-cgx
         dy = ngy-cgy
         dz = ngz-cgz
         call cadd(xp,dx,ntot)
         call cadd(yp,dy,ntot)
         call cadd(zp,dz,ntot)
         call cadd(x ,dx,mtot)
         call cadd(y ,dy,mtot)
         call cadd(z ,dz,mtot)
      else
         call vol_int(cgx,vol,xp)
         call vol_int(cgy,vol,yp)
         cgx = cgx/vol
         cgy = cgy/vol
         call prsrr ('Old CG: $',cgx,cgy)
         call prs   ('Input new CG: $')
         call rerr  (ngx,ngy)
         dx = ngx-cgx
         dy = ngy-cgy
         call cadd(xp,dx,ntot)
         call cadd(yp,dy,ntot)
         call cadd(x ,dx,mtot)
         call cadd(y ,dy,mtot)
      endif
c
      call gencen
      call coef
      return
c
  999 continue
      call prs('Problem opening file.$')
      return
c
 9999 continue
      call prs('Problem reading file.$')
      close (unit=88)
      return
c
      end
c-----------------------------------------------------------------------
      subroutine shiftmom
c     
c     prompt user for new cg and translate geometry to this location
c     
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ixx,iyy,izz,ixy,ixz,iyz
      real m(3,3),lam(3),wrk(9)
      real n(3,3),r(3,3)
      integer indr(3),indc(3),ipiv(3)
      character*80 fnm
c
      ntot= nx*ny*nz*nel
      if (if3d) then
         call vol_int(cgx,vol,xp)
         call vol_int(cgy,vol,yp)
         call vol_int(cgz,vol,zp)
         cgx = cgx/vol
         cgy = cgy/vol
         cgz = cgz/vol
         cgxm=-cgx
         cgym=-cgy
         cgzm=-cgz
c
         call cadd2(wkv1,xp,cgxm,ntot)
         call cadd2(wkv2,yp,cgym,ntot)
         call cadd2(wkv3,zp,cgzm,ntot)
         call vol_int2(ixx,vol,wkv1,wkv1)
         call vol_int2(iyy,vol,wkv2,wkv2)
         call vol_int2(izz,vol,wkv3,wkv3)
         call vol_int2(ixy,vol,wkv1,wkv2)
         call vol_int2(ixz,vol,wkv1,wkv3)
         call vol_int2(iyz,vol,wkv2,wkv3)
c
         ixx = ixx/vol
         iyy = iyy/vol
         izz = izz/vol
         ixy = ixy/vol
         ixz = ixz/vol
         iyz = iyz/vol
c
         a = iyy+izz
         b = izz+ixx
         c = ixx+iyy
c
         f = iyz
         g = ixz
         h = ixy
c
         m(1,1) =  a
         m(1,2) = -h
         m(1,3) = -g
c
         m(2,1) = -h
         m(2,2) =  b
         m(2,3) = -f
c
         m(3,1) = -g
         m(3,2) = -f
         m(3,3) =  c
c

         call ssyev('V','U',3,M,3,lam,wrk,9,info)
c
         if (info.ne.0) then
            call prsi('Returning. ssyev info not 0:$',info)
            return
         endif
c
         call prsrrr('Vec 1:$',m(1,1),m(2,1),m(3,1))
         call prsrrr('Vec 2:$',m(1,2),m(2,2),m(3,2))
         call prsrrr('Vec 3:$',m(1,3),m(2,3),m(3,3))
c
         call outmat(m,3,3,'Mom')
c
c        call prs   ('Input new matrix $')
c        call prs   ('Vec 1:$')
c        call rerrr (n(1,1),n(2,1),n(3,1))
c        call prs   ('Vec 2:$')
c        call rerrr (n(1,2),n(2,2),n(3,2))
c        call prs   ('Vec 3:$')
c        call rerrr (n(1,3),n(2,3),n(3,3))
c
         call prs ('Input file containing new matrix:$')
         call blank(fnm,80)
         call res (fnm,80)
         open (unit=88,file=fnm,status='old',err=999)
         read (88,*,err=9999,end=9999) ((n(i,j),j=1,3),i=1,3)
         close (unit=88)
c
c
c        X" = R X'  ==>    R  =  X" (X')^-1
c
         do k=1,3
            call norm3d(m(1,k))
            call norm3d(n(1,k))
         enddo
c
         call outmat   (m,3,3,'mb4')
c        call gaujord  (m,3,3,wrk,info)
         call gaujordf (m,3,3,indr,indc,ipiv,info)
         call outmat   (m,3,3,'maf')
         call outmat   (n,3,3,'nb4')
         call mxm      (n,3,m,3,r,3)
         call outmat   (r,3,3,'Rot')
c
         if (info.ne.0) then
            call prsi('Returning. gaujord info not 0:$',info)
            return
         endif
c
c        Rotation matrix set, apply to every point on the planet
c
         do i=1,ntot
            xn = r(1,1)*wkv1(i)+r(1,2)*wkv2(i)+r(1,3)*wkv3(i)+cgx
            yn = r(2,1)*wkv1(i)+r(2,2)*wkv2(i)+r(2,3)*wkv3(i)+cgy
            zn = r(3,1)*wkv1(i)+r(3,2)*wkv2(i)+r(3,3)*wkv3(i)+cgz
            xp(i) = xn
            yp(i) = yn
            zp(i) = zn
         enddo
c
         do ie=1,nel
         do ic=1,8
            xo = x(ie,ic)-cgx
            yo = y(ie,ic)-cgy
            zo = z(ie,ic)-cgz
            xn = r(1,1)*xo+r(1,2)*yo+r(1,3)*zo
            yn = r(2,1)*xo+r(2,2)*yo+r(2,3)*zo
            zn = r(3,1)*xo+r(3,2)*yo+r(3,3)*zo
            x(ie,ic) = xn + cgx
            y(ie,ic) = yn + cgy
            z(ie,ic) = zn + cgz
         enddo
         enddo
      endif
c
      call gencen
      call coef
      return
c
  999 continue
      call prs('Problem opening file.$')
      return
c
 9999 continue
      call prs('Problem reading file.$')
      close (unit=88)
      return
c
      end
c-----------------------------------------------------------------------
      subroutine mapz(f,regdir)
      include 'basics.inc'
c
      real f(nxm*nym*nzm*nelm)
      real gll_pts(nxm),gll_wts(nxm),i(nxm*nxm),it(nxm*nxm)
      real w(nxm*nxm*nxm)
      character*1 regdir
c
      integer e
c
c     This routine maps data from GLL points to uniform points in
c     the z direction only.   This is really a special purpose routine...
c
      call legend(gll_pts,gll_wts,nx)
c
c     Use Fornberg's routine to compute interpolation coefficients
c
      nx1 = nx-1
      l   = 1
      mx  = nx
      do k=1,mx
         z0 = (2.*(k-1))/(mx-1) - 1.
         call fd_weights_full(z0,gll_pts,nx1,0,it(l))
         l  = l+nx
      enddo
      call transpose(i,mx,it,nx)
c
      if (regdir.eq.'z'.or.regdir.eq.'Z') then
         nxy  = nx*ny
         nxyz = nx*ny*nz
         l = 1
         do e=1,nel
            call mxm (f(l),nxy,it,nz,w,nz)
            call copy(f(l),w,nxyz)
            l = l+nxyz
         enddo
      elseif (regdir.eq.'x'.or.regdir.eq.'X') then
         nyz  = ny*nz
         nxyz = nx*ny*nz
         l = 1
         do e=1,nel
            call mxm (i,nx,f(l),nx,w,nyz)
            call copy(f(l),w,nxyz)
            l = l+nxyz
         enddo
      else
         call prs('Sorry, direction is not currently supported$')
         write(6,*) 'Direction: ',regdir
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine setsrfp_periodic(e,f,e1,f1,ifld)
 
      include 'basics.inc'

      integer e,f,e1,f1,e0,f0
c
c     Select a pair of planes w / periodic bcs
c
      nlstp=0
      nfaces=2*ndim
c
      nlstp=nlstp+1
      ijkpln=eface1(f)
      ix=1
      if (mod(ijkpln,2).eq.0) ix=nx
      list(nlstp)=6*nx*(e-1)+nx*(ijkpln-1)+ix
      call signpl(list(nlstp))
c
      e1 = bc(1,f,e,ifld)
      f1 = bc(2,f,e,ifld)
c
      e0 = bc(1,f1,e1,ifld)
      f0 = bc(2,f1,e1,ifld)
c
      if (e0.ne.e .or. f0.ne.f) then
         call prsiii('Element RECIPROCITY FAIL:$',e,e0,e1)
         call prsiii('Face    RECIPROCITY FAIL:$',f,f0,f1)
      else
         nlstp=nlstp+1
         ijkpln=eface1(f1)
         ix=1
         if (mod(ijkpln,2).eq.0) ix=nx
         list(nlstp)=6*nx*(e1-1)+nx*(ijkpln-1)+ix
         call signpl(list(nlstp))
         call prsiv('Reciprocity OK:$',e,f,e1,f1)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine periodic_check
 
      include 'basics.inc'

      integer e,f,e1,f1,e0,f0
c
c     Select a pair of planes w / periodic bcs
c
      ifld = 2
      if (ifflow) ifld = 1
c
      nfaces = 2*ndim
      do e=1,nel
      do f=1,nfaces
         write(6,*) e,f,cbc(f,e,ifld),' CBC'
         if (cbc(f,e,ifld).eq.'P  ') then
            call setsrfp_periodic(e,f,e1,f1,ifld)
            call tem
            call prsiv('Plotted e,f:$',e,f,e1,f1)
            call prs  ('Continue (y/n) ?$')
            call res  (ans,1)
            if (ans.eq.'n'.or.ans.eq.'N') return
         endif
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
