c-----------------------------------------------------------------------
      subroutine cutting_plane
C
c     This subroutine is called when cutting_plane is chosen from the 
c     main menu.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     dimension xxyy(4,4000)
      dimension xxyy(4,40)
c 
      integer nsect,stepval
      save    nsect,stepval
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
c
      character*80 cutchoice
      character*1 ans1
      call prs('IN CUTTING PLANE$')
                               nchoic=1
      item(nchoic)='UP MENU'
                               nchoic=nchoic+1
      item(nchoic)='SET PLANE'
c                              nchoic=nchoic+1
c     item(nchoic)='SET LOCATION'
                               nchoic=nchoic+1
      item(nchoic)='PLOT CONTOUR INFO'
                               nchoic=nchoic+1
      item(nchoic)='SHOW CONTOUR'
c
      call menu(xmouse,ymouse,button,'CUTTING PLANE')
      cutchoice=CHOICE
      IF(cutchoice.EQ.'SET PLANE')THEN
         call getplane(nsect,stepval)
         do is=1,nsect
           call contour_setup
           call new_contour(nsect,stepval,is)
           call out_contour(nsect,stepval,is)
         enddo
      ELSE IF(cutchoice.EQ.'SHOW CONTOUR')THEN
         do is=1,nsect
           call contour_setup
           call new_contour(nsect,stepval,is)
           call out_contour(nsect,stepval,is)
         enddo
      ELSE IF(cutchoice.EQ.'PLOT CONTOUR INFO')THEN
         do i=1,ncntval
           do j=1,4
             xxyy(j,i)=cntval(j,i)
           enddo
         enddo
         call vdraw3(xxyy,ncntval)
c        call dumpfile
      ELSE IF(cutchoice.EQ.'UP MENU')THEN
         return
      ENDIF
c
      return
      end
c-----------------------------------------------------------------------
      subroutine getplane(nsect,stepval)
C
c     This routine prompts the user for the xzero point and coordinates
c     of x_twidle, which define the cutting plane.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     real ptx,pty,ptz,xnor,ynor,znor,dx,dy,dz,vnrm
      character*1 yesno
      character*1 morecont
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
c
      nsect = 1
      call prs('IN CUTTING PLANE$')
   15 continue
c
c     call clear
c     call hard
c     call heject
c     call drmesh
c     call drfram
c     call nohard
c
      call prs('Input the coord. of xzero (x,y,z):$')
      call rerrr(ptx,pty,ptz)
      call prs('Input the coord. of xnormal FROM THE$')
      call prs('ORIGIN(xnor,ynor,znor):$')
      call prs('Normal vector points from xzero to xnormal coord.!!$')
      call rerrr(xnor,ynor,znor)
      dx = xnor-ptx
      dy = ynor-pty
      dz = znor-ptz
      vnrm = dx*dx + dy*dy + dz*dz
      if (vnrm.gt.0) then
         vnrm = sqrt(vnrm)
         xnor = dx/vnrm
         ynor = dy/vnrm
         znor = dz/vnrm
      endif
      call arrow4(ptx,pty,ptz,xnor,ynor,znor)
      call prs('Do you want this arrow to define your plane? (y/n):$')
      call res(yesno,1)
      if (yesno.eq.'n'.or.yesno.eq.'N') goto 15
      call prs('Do you want more than one contour?$')
      call prs('Type y for yes or just ENTER for no.$')
      call res(morecont,1)
c
      if (morecont.eq.' ') goto 222
      call prs('Enter number of sections:$')
      call rei(nsect)
      call prs('Enter the step value$')
      call rer(stepval)
  222 continue
c
c     call prs('LEAVE CUTTING PLANE$')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine new_contour(nsect,stepval,is)
c
c     This routine normalize the unit normal vector, which defines the 
c     cutting plane.  Also, finds contour through the selected planes
c     and use them to find the interpolated values of x,y,z,t,p....etc.
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      LOGICAL IFCRV(4)
c     common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
c     COMMON /ccdiag/ iee
      PARAMETER (NXM2=NXM*NYM)
      CHARACTER C*20
      SAVE IFIRST
      DATA IFIRST /0/
c
      parameter (nx2m=nxm*nxm)
      real w_gll(nx2m,0:15)
c
      parameter (nxx2m=nxxm*nxxm)
      real w_interp(nxx2m,0:15)
c
      call hard
      write(6,*) 'this is jhard:',jhard,ifhard
      if (jhard.eq.3) ifhard = .true.
      if (is.eq.1) then
        val = 0
      else
        val = 0 + (is-1)*stepval
      endif
c
      write(6,*) 'val = ',val
      write(6,*) 'nx2m=',nx2m,'   nxx2m=',nxx2m,'  nx=',nx
C
C     Plot face by face
      IF (.NOT.IF3D) THEN
        write(6,*) 'hey, no 2D yet'
      ELSE
c
C       3-D
C       Sort the selected list of planes according to visibility
C
        call sortl
C
        i = 0
        DO 500 Ipla=1,NLSTP
        i = i+1
C
C       New plotting algorithms for color fill and fishnet - pff 7-1-90
C       (3D will come later - 10-10-90)
C
        LISTA=ABS(LISW(I))
        im   =    lmir(i)
        call decod(iplane,ipln,ie,idum,lista,nx,6,nelm)
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
        KTR=0
c       write(48,48) ie,i1,i2,j1,j2,k1,k2
c  48   format(i7,6i3)
c
        II=0
        DO IZ=K1,K2
        DO IY=J1,J2
        DO IX=I1,I2
          ipoint = ix + nx*(iy-1) + nx*ny*(iz-1) + nx*ny*nz*(ie-1)
c         write(6,*)  ipoint,'  ', ix,iy,iz,ie
          ii=ii+1
          w_gll(ii, 1) =   xp(ipoint)
          w_gll(ii, 2) =   yp(ipoint)
          w_gll(ii, 3) =   zp(ipoint)
          w_gll(ii, 4) =    u(ipoint)
          w_gll(ii, 5) =    v(ipoint)
          w_gll(ii, 6) =    w(ipoint)
          w_gll(ii, 7) =    p(ipoint)
          w_gll(ii, 8) =    t(ipoint)
          w_gll(ii, 9) = work(ipoint)
          w_gll(ii,10) = wkv1(ipoint)
          w_gll(ii,11) = wkv2(ipoint)
          w_gll(ii,12) = wkv3(ipoint)
          n_gll = 12
        enddo
        enddo
        enddo
c
c       call out_mat(w_gll(1,1),nx,nx,' x1')
c       call out_mat(w_gll(1,2),nx,nx,' y1')
c       call out_mat(w_gll(1,3),nx,nx,' z1')
c
c       compute contour variable, and see if "0" is contained

        call compute_contour_variable(w_gll,nx2m,nx*nx)
        do nn=1,nx*nx
          w_gll(nn,0) = w_gll(nn,0) - val
        enddo
        contour_min = glmin (w_gll,nx*nx)
        contour_max = glmax (w_gll,nx*nx)
c
        if (contour_min*contour_max .le.0) then
c         nxx  = 30
          nxx  = nxband
          call map_interp  (w_interp,nxx2m,w_gll,nx2m,n_gll,nxx)
          call find_contour(w_interp,nxx2m,w_gll,nx2m,n_gll,KTR,nxx)
        endif
 500    enddo
      endif
c
      write(6,*) 'this is jhard2:',jhard,ifhard
      if (jhard.eq.3) ifhard = .false.
      call nohard
c
      return
      end

c-----------------------------------------------------------------------
      subroutine compute_contour_variable(w,ldw,n)
c
c     Normalizing the unit vector and putting it into w.
c
      real w(ldw,0:15)
C
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor
c
      do i = 1, n
         xtw=w(i,1)-ptx
         ytw=w(i,2)-pty
         ztw=w(i,3)-ptz
         w(i,0) = xnor*xtw + ynor*ytw + znor*ztw
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine map_interp(w_interp,ldwi,w_gll,ldwg,n_gll,nxx)
c
c     This routine maps and interpolate gauss labatto info into regular
c     grid of size nxx.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real w_gll   (ldwg,0:n_gll)
      real w_interp(ldwi,0:n_gll)
C
      common /c_map_i/ intrp(nxxm,nxxm,2),wrk(nxxm,nxm,2)
      real intrp
c     
      integer nxxlst
      save    nxxlst
      data    nxxlst /0/
c
      if (nxx.ne.nxxlst) call setix(intrp,nxx,nx,w_interp)
      nxxlst = nxx
c
      do i=0,n_gll
         call mapglr(w_interp(1,i),w_gll(1,i),intrp,nxx,nx,wrk)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine find_contour(wpp,ldw,w_gll,ldw_gll,n_gll,KTR,nxx)
C
C     This routine finds the contour on the regular grid (w_interp).
C     NOTE: There might be some internal contours; this routine checks 
c     for such contours.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      LOGICAL   IFCRV(4)
c
      real wpp  (ldw    ,0:n_gll)
      real w_gll(ldw_gll,0:n_gll)
C
C     Two common blocks below are used in several subsequent routines.
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
      LOGICAL IFOK
      ifcrv(1)=.false.
      ifcrv(2)=.false.
      ifcrv(3)=.false.
      ifcrv(4)=.false.
C
C     VAL is the current contour level.
C
      VAL=0
      BREAK0 = VAL
C
C     Establish contour array, CRS:  C(i,point) = r or s for each cross point
C     KTR=0 -- Line  contours
C     KTR=1 -- Solid contours
C
      jrow1=1
      jrow2=nxx
c
      wmax=glmax(wpp,nxx*nxx)
      wmin=glmin(wpp,nxx*nxx)
      CALL CONTOUR(VAL,wpp,IFCRV,nxx,JROW1,JROW2,KTR)
c
c     call out_mat(w_gll(1,1),nx,nx,' x ')
c     call out_mat(w_gll(1,2),nx,nx,' y ')
c     call out_mat(w_gll(1,3),nx,nx,' z ')
c
c
c     CALL PLTCNTR(w_gll(1,1),w_gll(1,2),w_gll(1,3),VAL,KTR)
c
c     call prs('continue?$')
c     call res(ans1)
c
      call pltcntr2(w_gll,ldw_gll,n_gll,VAL,KTR)
c
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE dumpfile
C
C     Plot a contour given by CRS which denotes the Contour points
C     in the (R,S) plane.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
c
C
      return
      end
c-----------------------------------------------------------------------
      subroutine contour_setup
C
C 
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
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
C     THIS IS THE SETUP STUFF
      IF (NXBAND.GT.NXXM) THEN
        WRITE(S,10) NXXM
   10    FORMAT(2X,'Resetting resolution on color fill to',I3,'.')
        CALL PRS(S//'$')
        NXBAND=NXXM
      ENDIF
      NXX=NXBAND
      IF (NXX.NE.NXXLST.OR.ILGRNG.NE.2) THEN 
        NXXLST=NXX
        DO 30 I=1,NXX
           R1(I)=2.0*FLOAT(I-1)/FLOAT(NXX-1) - 1.0
           S1(I)=R1(I)
   30    CONTINUE
      ENDIF
c
      ncntval = 0
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat(a,m,n,ch3)
      real a(m,n)
      character*3 ch3
c
      write(6,*) 'MATRIX: ',ch3,m,n
c
      if (n.le.4) then
         do i=1,m
            write(6,2) (a(i,j),j=1,n)
         enddo
         return
      endif
c
      do i=1,m
         write(6,1) (a(i,j),j=1,n)
      enddo
    1 format(9f8.3)
    2 format(1p4e15.6)
      return
      end
c-----------------------------------------------------------------------
      subroutine out_mat(a,m,n,ch3)
      real a(m,n)
      character*3 ch3
c
      write(6,*) 'MATRIX: ',ch3,m,n
c
      do j=m,1,-1
         write(6,1) (a(i,j),i=1,m)
      enddo
    1 format(9f8.3)
      return
      end
c-----------------------------------------------------------------------
      subroutine pltcntr2(wpp,ldw,n_gll,val,ktr) 
C 
C     Plot a contour given by CRS which denotes the Contour points
C     in the (R,S) plane.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      dimension wpp(ldw,0:15)
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
C
C     Two common blocks below are used in several subsequent routines.
      PARAMETER (NXX2M=NXXM*NXXM)
      PARAMETER (NXX22=2*NXX2M)
      COMMON /CTMP1x/ CRS(4,NXX2M),R1(NXXM),S1(NXXM),RXVR(NXXM,NXXM)
     $              ,TWRK(NXX2M),XYZ(4,NXX22),BREAK0,BREAK1
     $              , WKK(NXX2M),INTRP(NXXM,NXM,2)
      COMMON /CTMP2/ NCRS,NCTR,ICTR(50),NXVR(NXXM),INDX(NXXM)
     $              ,ICELMN,ICELMX,JCELMN,JCELMX
C
      DO 1000 ICT=1,NCTR
        II =0
        ICFRST=ICTR(ICT)
        ICLAST=ICTR(ICT+1)-1
        IF (ICT.EQ.NCTR) ICLAST=ICTR(ICT+1)
        IF (KTR.EQ.1) CALL BEGINPP
        DO 200 ICRS=ICFRST,ICLAST
           II=II+1
C          More sophisticated (1D) interpolation will go here later.
           CALL EVALSC2(XYZ(1,II),wpp(1,1),CRS(1,ICRS))
           CALL EVALSC2(XYZ(2,II),wpp(1,2),CRS(1,ICRS))
           IF (IF3D) THEN
              CALL EVALSC2(xyz(3,II),wpp(1,3),CRS(1,ICRS))
           ELSE
              xyz   (3,II)=0.
           ENDIF
c
           ncntval = ncntval+1
           cntval(1,ncntval) = xyz(1,ii)
           cntval(2,ncntval) = xyz(2,ii)
           cntval(3,ncntval) = xyz(3,ii)
           do nn = 4,n_gll
              CALL EVALSC2(cntval(nn,ncntval),wpp(1,nn),CRS(1,ICRS))
           enddo
c
  200   CONTINUE
c       write(6,1) (cntval(k,ii),k=1,n_gll)
c   1   format(1p12e12.4)
c
        CALL VDRAW3(xyz,II)
c
 1000 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine out_contour(nsect,stepval,is) 
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      dimension empty(0:15)
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
c
      open(unit=33,file='data')
c
c     do i=1,ncntval
c        write(33,1) (cntval(j,i),j=0,8)
c     enddo
c   1 format(9f8.3)
c       write(33,*) 'end'
c
      call set_progress_variable(is)
      do i=0,15
        empty(i)=0.
      enddo
      empty(0)=ncntval 
c
      write(33,1) (empty(j),j=0,12)
      do i=1,ncntval
         write(33,1) (cntval(j,i),j=0,12)
c        write(33,2) (cntval(0,i),cntval(9,i))
      enddo
    1 format(4f8.3,1p10e12.4)
c   2 format(f8.3,1pe12.4)
c     write(33,*) ' '
c
      if (is.eq.nsect) close(unit=33)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_progress_variable(is)
c
c     This routine set and put values into the cntval array, which is
c     then sorted either by theta, x, y, or z.
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
      common/Cuttin2/uxtw,uytw,uztw,vxtw,vytw,vztw
      common/Cuttinc/progvar
      character progvar*1
c
      if (is.ne.1) goto 111
  110 continue
      call prs('Enter the type of progress variable to use:$')
      call prs('(x, y, z, or (t)heta)$')
      call res(progvar,1)
      if ((progvar.ne.'x') .and. (progvar.ne.'y') .and.
     $   (progvar.ne.'z') .and. (progvar.ne.'t')) goto 110
      write(6,*) 'ncntval ',ncntval
  111 continue
c
      if (progvar .eq. 't') then
        call prog_theta(is)
      elseif (progvar .eq. 'x') then
        do i=1,ncntval
          cntval(0,i)=cntval(1,i)
        enddo
      elseif (progvar .eq. 'y') then
        do i=1,ncntval
          cntval(0,i)=cntval(2,i)
        enddo
      elseif (progvar .eq. 'z') then
        do i=1,ncntval
          cntval(0,i)=cntval(3,i)
        enddo
      endif
c     
      call sort_prog
c
c     do i=1,ncntval
c        write(6,1) (cntval(j,i),j=0,8)
c     enddo
c   1 format(9f8.3)
c     
      return
      end
c-----------------------------------------------------------------------
      subroutine prog_theta(is)
c
c     This routine calculates the progress variable theta, choosing 
c     the center of gravity as the origin.
c
      include 'basics.inc'
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
      common/Cuttin2/uxtw,uytw,uztw,vxtw,vytw,vztw
      common/Cuttinc/progvar
      character progvar*1
c
      DIMENSION V1(3),V2(3),V3(3),xtwid(3)
      character*80 prin
c
      if (ncntval.eq.0) return
c
c     do i=1,50
c        write(6,1) (cntval(j,i),j=0,8)
c     enddo
c   1 format(9f8.3)
      sumx = 0.
      sumy = 0.
      sumz = 0.
      do i=1,ncntval
        sumx = sumx+cntval(1,i)
        sumy = sumy+cntval(2,i)
        sumz = sumz+cntval(3,i)
      enddo
      centx = sumx/ncntval
      centy = sumy/ncntval
      centz = sumz/ncntval
c
      if (is.ne.1) goto 120
      xtwid(1) = 0
      xtwid(2) = 0
      xtwid(3) = 0
c
      call prs('Choose from the menu the x principle axis!$')
                               nchoic=1
      item(nchoic)='POSITIVE X'
                               nchoic=nchoic+1
      item(nchoic)='NEGATIVE X'
                               nchoic=nchoic+1
      item(nchoic)='POSITIVE Y'
                               nchoic=nchoic+1
      item(nchoic)='NEGATIVE Y'
                               nchoic=nchoic+1
      item(nchoic)='POSITIVE Z'
                               nchoic=nchoic+1
      item(nchoic)='NEGATIVE Z'
      call menu(xmouse,ymouse,button,'PRINCIPLE AXIS')
      prin=CHOICE
      IF(prin.EQ.'POSITIVE X')THEN
         xtwid(1)=1
      ELSE IF(prin.EQ.'NEGATIVE X')THEN
         xtwid(1)=-1
      ELSE IF(prin.EQ.'POSITIVE Y')THEN
         xtwid(2)=1
      ELSE IF(prin.EQ.'NEGATIVE Y')THEN
         xtwid(2)=-1
      ELSE IF(prin.EQ.'POSITIVE Z')THEN
         xtwid(3)=1
      ELSE IF(prin.EQ.'NEGATIVE Z')THEN
         xtwid(3)=-1
      ENDIF
c
      xnn=dot_prod(xtwid(1),xtwid(2),xtwid(3),xnor,ynor,znor)
      write(6,*) 'xnn ',xnn
      uxtw = xtwid(1)-xnn*xnor
      uytw = xtwid(2)-xnn*ynor
      uztw = xtwid(3)-xnn*znor
c
      dx=cntval(1,i)-centx
      dy=cntval(2,i)-centy
      dz=cntval(3,i)-centz
      dist= dx*dx + dy*dy + dz*dz
      far  = sqrt(dist)
c
      dist= uxtw*uxtw + uytw*uytw + uztw*uztw
      dist  = sqrt(dist)
      uxtw = uxtw/dist
      uytw = uytw/dist
      uztw = uztw/dist
      far10 = far/10.
      call draw_circ1(centx,centy,centz,far10)
c
      vxtw = ynor*uztw - uytw*znor
      vytw = xnor*uztw - uxtw*znor
      vztw = xnor*uytw - uxtw*ynor
  120 continue
c
      write(6,*) 'normal  ',xnor,ynor,znor
      write(6,*) 'udir  ',uxtw,uytw,uztw
      write(6,*) 'vdir  ',vxtw,vytw,vztw
      call arrow4(centx,centy,centz,vxtw,vytw,vztw)
      call arrow4(centx,centy,centz,uxtw,uytw,uztw)
c
      do i=1,ncntval
        xi = cntval(1,i)-centx
        yi = cntval(2,i)-centy
        zi = cntval(3,i)-centz
        uu = dot_prod(uxtw,uytw,uztw,xi,yi,zi)
        vv = dot_prod(vxtw,vytw,vztw,xi,yi,zi)
        angle = atan2(vv,uu)*180/pi
        cntval(0,i) = angle
      enddo
c
c     do i=1,ncntval
c        write(6,1) (cntval(j,i),j=0,8)
c     enddo
c   1 format(9f8.3)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine sort_prog
c
c     This routine sorts the contour informations by the progress 
c     variable stored in cntval(0,:).
c
      real temp(0:15)
c
      common/Cutting/ptx,pty,ptz,xnor,ynor,znor,cntval(0:15,40)
      common/CuttinI/ncntval
      common/ctmp1/ ind(4000),aa(16)
c
c
c
cubroutine tuple_sort(a    ,lda,n      ,key,nkey,ind,aa)
c     call tuple_sort(cntval,16,ncntval,1  ,1   ,ind,aa)
c
c
      do i=1,ncntval-1
        ncompare = ncntval - i
        do j=1,ncompare
          if (cntval(0,j) .gt. cntval(0,j+1)) then 
            do ii=0,12
              temp(ii)=cntval(ii,j)
            enddo
            do ii=0,12
              cntval(ii,j)=cntval(ii,j+1)
            enddo
            do ii=0,12
              cntval(ii,j+1)=temp(ii)
            enddo
          endif
        enddo
      enddo
c     
      return
      end
c-----------------------------------------------------------------------
      function dot_prod(x1,y1,z1,x2,y2,z2)
c
c     This routine calculates the dot product of two vectors.
c
      dot_prod = x1*x2 + y1*y2 + z1*z2
c     write(6,*) 'sadf ',dot_prod,x1,y1,z1,x2,y2,z2
c
      return
      end
c-----------------------------------------------------------------------
      subroutine SETCVOL
C
C     This routine will generate a volume on X - Y - or Z planes
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
C     Select what type of plane
      nchoic=1
      ITEM(nchoic)='X-PLANE'
      nchoic=nchoic+1
      ITEM(nchoic)='Y-PLANE'
      nchoic=nchoic+1
      ITEM(nchoic)='Z-PLANE'
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET LOCATION')
      IF (CHOICE.EQ.'X-PLANE' .OR.
     $    CHOICE.EQ.'Y-PLANE' .OR.
     $    CHOICE.EQ.'Z-PLANE' ) THEN
          VPLANE=CHOICE
      ENDIF
C
C     Set range on points
C
      CALL PRS
     $('Enter a point near surface to be plotted (X,Y,Z):$') 
      CALL RERRR(VXPT1(1),VXPT1(2),VXPT1(3))
C
      CALL PRS
     $('Enter far point near surface to be plotted (X,Y,Z):$') 
      CALL RERRR(VXPT2(1),VXPT2(2),VXPT2(3))
C
      CALL PRS
     $('Enter number of slices to be plotted:$') 
      CALL REI(NSLICE)
C
C     Set thresholds (0,1)
C
      CALL PRS('Enter upper and lower thresholds (0,1):$')
      CALL RERR(THRMIN,THRMAX)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine CVOLUME
C
C     This routine will generate a volume on X - Y - or Z planes
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      LOGICAL  IFSAME,IFTMP1,IFTMP2
C
      IF (NSLICE.LT.2) CALL SETCVOL
      IF (NSLICE.LT.2) RETURN
C
      X0 = VXPT1(1)
      Y0 = VXPT1(2)
      Z0 = VXPT1(3)
      DX = (VXPT2(1)-VXPT1(1)) / FLOAT(NSLICE-1)
      DY = (VXPT2(2)-VXPT1(2)) / FLOAT(NSLICE-1)
      DZ = (VXPT2(3)-VXPT1(3)) / FLOAT(NSLICE-1)
C
      IF (VPLANE.EQ.'X-PLANE') IVEC=1
      IF (VPLANE.EQ.'Y-PLANE') IVEC=2
      IF (VPLANE.EQ.'Z-PLANE') IVEC=3
C
      IFCVOL=.TRUE.
      IFTMP1=IFCFBD
      IFCFBD=.FALSE.
      SCALPT='CONTOUR LINES'
      IFTMP2=IFCLRB
      DO 100 I=1,NSLICE
         CALL SETRSTV(X0,Y0,Z0,IVEC,IFSAME)
         IF (.NOT.IFSAME) CALL TEM
c        IF (.NOT.IFSAME) THEN
c           write(56,*) VPLANE
c           write(56,*) X0,Y0,Z0
c        ENDIF
C        turn off color bar
         IF (.NOT.IFSAME) IFCLRB=.FALSE.
         X0 = X0 + DX
         Y0 = Y0 + DY
         Z0 = Z0 + DZ
  100 CONTINUE
      IFCVOL=.FALSE.
      IFCFBD=IFTMP1
      IFCLRB=IFTMP2
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine SETRSTV(XPT1,YPT1,ZPT1,IVEC,IFSAME)
C     Set up list of candidate RST planes
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX
      DIMENSION VEC(3),VEC1(3),VEC2(3),VEC3(3)
      CHARACTER*1 YESNO
      LOGICAL IFANY,IFTMP,IFSAME
      INTEGER ICALLD,IPLANL,IELAST,IVECL,IJKLST
      SAVE    ICALLD,IPLANL,IELAST,IVECL,IJKLST
      DATA    ICALLD /0/
C
C
C     Find the element containing this point
C
      CALL INTERP(VAL1,XPT1,YPT1,ZPT1,IE,WORK,ierr)
c     write(6,*) ' ie:',ie,ivec
C
C     Exception handling:
      IF (IE.EQ.0) RETURN
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
C     Check to see if this plane is the same as the previous
C
      IF (ICALLD.GT.0       .AND.
     $    IPLAN.EQ.IPLANL   .AND.
     $    IE.EQ.IELAST      .AND.
     $    IVEC.EQ.IVECL     .AND.
     $    IJKPLN.EQ.IJKLST) THEN
          IFSAME=.TRUE.
          RETURN
       ELSE
          ICALLD=1
          IPLANL=IPLAN
          IELAST=IE
          IVECL=IVEC
          IJKLST=IJKPLN
          IFSAME=.FALSE.
       ENDIF
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
      NLSTP=1
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
      CALL DECOD(JQ,JPLN,JEg,IDUM,NEIGHA,4,6,NEL)
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
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine strplt(ifnewplt)
C
C     Plot a diamond on each point read in from file
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      character*40 infile
      logical ifnewplt
c
      integer j
      save    j
      data    j /0/
c
      real xst(3,10)
      save xst
C
      if (ifnewplt.or.j.eq.0) then
         call prs('Input file name:$')
         call blank(infile,40)
         call res(infile,40)
         lstr = ltrunc(infile,40)
c
         if (lstr.gt.0) then
            open (unit=44,file=infile,err=1001,status='old')
            read(44,*,err=100,end=1000) xst(1,1),xst(2,1),xst(3,1)
            call move3(xst(1,1),xst(2,1),xst(3,1))
            do i=1,99999
               j=min(i,100000)
               read(44,*,err=100,end=1000) xst(1,j),xst(2,j),xst(3,j)
               call draw3(xst(1,j),xst(2,j),xst(3,j))
            enddo
            close (unit=44)
         else
            call move3(xst(1,1),xst(2,1),xst(3,1))
            do i=1,j
               call draw3(xst(1,i),xst(2,i),xst(3,i))
            enddo
         endif
c
      else
         call move3(xst(1,1),xst(2,1),xst(3,1))
         do i=1,j
            call draw3(xst(1,i),xst(2,i),xst(3,i))
         enddo
      endif
c
  100 continue
 1000 continue
      return
 1001 continue
      call prs('couldnt open file, try again$')
      return
      end
c-----------------------------------------------------------------------
      subroutine genfem
c
c     Generate an FEM input file based on Nekton .rea file
c
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      integer iwork(maxpts),loc(maxpts),ind(maxpts)
      equivalence (iwork,work)
      equivalence (loc  ,txm1)
      equivalence (ind  ,tym1)
      integer ic(8)
c
      ntot=nx*ny*nz*nel
      nxyz=nx*ny*nz
      nxy =nx*ny
C
c     Establish local to global node numbering for GLL points
c
      call locglob
      call icopy(loc,iglob,ntot)
      call copy (wkv1,xp,ntot)
      call copy (wkv2,yp,ntot)
      call copy (wkv3,zp,ntot)
c
      call irank(loc,ind,ntot)
      call iswap(loc,work,ind,ntot)
      call swap (wkv1,work,ind,ntot)
      call swap (wkv2,work,ind,ntot)
      call swap (wkv3,work,ind,ntot)
      write(37,10) nglob
   10 format(i9,'  #  this is number of vertices ')
      i=1
      write(37,20) wkv1(i),wkv2(i),wkv3(i),loc(i)
      do i=2,ntot
         if (loc(i).ne.loc(i-1)) 
     $      write(37,20) wkv1(i),wkv2(i),wkv3(i),loc(i)
   20       format(3g15.8,' # ',i9)
      enddo
c
      nbrick = nel*(nz-1)*(ny-1)*(nx-1)
      write(37,30) nbrick,nel,nx,ny,nz
   30 format(i9,' # this is nbrick ',4i8)
      if (if3d) then
         do ie=1,nel
            ieoff = nxyz*(ie-1)
            do iz=1,nz-1
            do iy=1,ny-1
            do ix=1,nx-1
               i = 0
               do jz=0,1
               do jy=0,1
               do jx=0,1
                  i  = i+1
                  ii = ieoff+nxy*(iz+jz-1)+nx*(iy+jy-1)+ix+jx
                  ic(i) = loc(ii)
               enddo
               enddo
               enddo
               if (nglob.lt.100000) then
                  write(37,41) (ic(i),i=1,8)
               else
                  write(37,42) (ic(i),i=1,8)
               endif
   41          format(8i5)
   42          format(8i9)
            enddo
            enddo
            enddo
         enddo
      endif
c
c     Boundary faces
c
      ifld=1
      if (ifheat) ifld=2
      nfaces = 2*ndim
      nb=0
      do ie=1,nel
      do jface=1,nfaces
         if (cbc(jface,ie,ifld).ne.'E') nb=nb+(ny-1)*(nz-1)
      enddo
      enddo
      write(37,50) nb
   50 format(i9,'   #  this is number of boundary faces ')
c
      do ie=1,nel
      do jface=1,nfaces
         if (cbc(jface,ie,ifld).ne.'E') then
            iface=eface1(jface)
            i1=1
            i2=nx
            j1=1
            j2=ny
            k1=1
            k2=nz
            if (iface.eq.1) i2=1
            if (iface.eq.2) i1=nx
            if (iface.eq.3) j2=1
            if (iface.eq.4) j1=ny
            if (iface.eq.5) k2=1
            if (iface.eq.6) k1=nz
c
            ieoff = nxyz*(ie-1)
            ii=0
            do k=k1,k2
            do j=j1,j2
            do i=i1,i2
               ii=ii+1
               iwork(ii) = iglob(ieoff+nxy*(k-1)+nx*(j-1)+i)
            enddo
            enddo
            enddo
c
            do iz=1,nz-1
            do ix=1,nx-1
               i=0
               do jz=0,1
               do jx=0,1
                  i=i+1
                  ii = nx*(iz+jz-1) + (ix+jx)
                  ic(i)=iwork(ii)
               enddo
               enddo
               write(37,60) (ic(i),i=1,4), ie,iface
            enddo
            enddo
         endif
      enddo
      enddo
   60 format(4i9,'   # bd ',2i8)
c
      call prs('End of FEM generation$')
      return
      end
c-----------------------------------------------------------------------
