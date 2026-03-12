c-----------------------------------------------------------------------
      subroutine readat
C
C     Read in data from preprocessor input file (.rea)
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
 
      logical ifbswap,ifre2,parfound
      character*132 string
      integer idum(3*numsts+3)

      ierr = 0
      call flush_io

      ! check if new rea file version exists
      if(nid.eq.0) inquire(file=parfle, exist=parfound)
      call bcast(parfound,lsize)
      if (parfound) then
         if(nio.eq.0) write(6,'(A,A)') ' Reading ', parfle
         call readat_par
         goto 99
      endif  

      etime0 = dnekclock_sync()

      if(nid.eq.0) then
        write(6,'(A,A)') ' Reading ', reafle
        open (unit=9,file=reafle,status='old', iostat=ierr)
      endif

      call bcast(ierr,isize)
      if (ierr .gt. 0) call exitti('Cannot open rea file!$',1)

C     Read parameters and logical flags
      call rdparam
      meshPartitioner=3 ! HYBRID (RSB+RCB)

C     Read Mesh Info 
      if(nid.eq.0) then
        read(9,*)    ! xfac,yfac,xzero,yzero
        read(9,*)    ! dummy
        read(9,*)  nelgs,ldimr,nelgv
        nelgt = abs(nelgs)
      endif
      call bcast(ldimr,ISIZE)
      call bcast(nelgs,ISIZE)
      call bcast(nelgv,ISIZE)
      call bcast(nelgt,ISIZE)
      ifre2 = .false.
      if (nelgs.lt.0) ifre2 = .true.

      call usrdat0

      if (nelgt.gt.350000 .and. .not.ifre2) 
     $   call exitti('Problem size requires .re2!$',1)

      if (ifre2) call read_re2_hdr(ifbswap, .true.) ! rank0 will open and read
      call chk_nel  ! make certain sufficient array sizes

      call mapelpr

      if (ifre2) then
        call read_re2_data(ifbswap)
      else
        maxrd = 32               ! max # procs to read at once
        mread = (np-1)/maxrd+1   ! mod param
        iread = 0                ! mod param
        x     = 0
        do i=0,np-1,maxrd
           call nekgsync()
           if (mod(nid,mread).eq.iread) then
              if (nid.ne.0) then
                open(UNIT=9,FILE=REAFLE,STATUS='OLD')
                call cscan(string,'MESH DATA',9)
                read(9,*) string
              endif 
              call rdmesh
              call rdcurve !  Curved side data
              call rdbdry  !  Boundary Conditions
              if (nid.ne.0) close(unit=9)
           endif
           iread = iread + 1
        enddo
      endif

C     Read Restart options / Initial Conditions / Drive Force
      CALL RDICDF
C     Read materials property data
      CALL RDMATP
C     Read history data
      CALL RDHIST
C     Read output specs
      CALL RDOUT
C     Read objects
      CALL RDOBJ

      call nekgsync()

C     End of input data, close read file.
      if(nid.eq.0) then
        close(unit=9)
        call echopar
        write(6,'(A,g13.5,A,/)')  ' done :: read .rea file ',
     $                             dnekclock()-etime0,' sec'
      endif

 99   call izero(boundaryID, size(boundaryID))
      call izero(boundaryIDt, size(boundaryIDt))

      ifld = 2 
      if(ifflow) ifld = 1
      do iel = 1,nelv
      do ifc = 1,2*ndim   
         boundaryID(ifc,iel) = bc(5,ifc,iel,ifld)
      enddo
      enddo

      ntmsh = 0
      do i=1,ldimt
         if(iftmsh(1+i)) ntmsh = ntmsh + 1 
      enddo

      if (ntmsh.gt.0) then
        do iel = 1,nelt
        do ifc = 1,2*ndim   
           boundaryIDt(ifc,iel) = bc(5,ifc,iel,2)
        enddo
        enddo
      endif 

      return
      END
c-----------------------------------------------------------------------
      subroutine vrdsmsh
C
C=====================================================================
C     Verify that mesh and dssum are properly defined by performing
C        a direct stiffness operation on the X,Y and Z coordinates.
C     Note that periodic faces are not checked here.
C=====================================================================
C
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      COMMON /SCRNS/ TA(LX1,LY1,LZ1,LELT),TB(LX1,LY1,LZ1,LELT)
     $           ,QMASK(LX1,LY1,LZ1,LELT),tmp(2)
      CHARACTER*3 CB

c      call  vrdsmshx  ! verify mesh topology

      if(nio.eq.0) write(*,*) 'verify mesh topology'

      IERR      = 0
      EPS       = 1.0e-04
      EPS       = 1.0e-03
      IFIELD    = 1
      if (nelgv.ne.nelgt .or. .not.ifflow) ifield = 2
      NXYZ1     = lx1*ly1*lz1
      NTOT      = lx1*ly1*lz1*NELT
      NFACES    = 2*ldim

      xmx = glmax(xm1,ntot)
      xmn = glmin(xm1,ntot)
      ymx = glmax(ym1,ntot)
      ymn = glmin(ym1,ntot)
      zmx = glmax(zm1,ntot)
      zmn = glmin(zm1,ntot)
      if (nio.eq.0) write(6,*) xmn,xmx,' Xrange'
      if (nio.eq.0) write(6,*) ymn,ymx,' Yrange'
      if (nio.eq.0) write(6,*) zmn,zmx,' Zrange'
c     return
C
C     First check - use 1/Multiplicity
C
      IF (IFHEAT) THEN 
         CALL COPY(TA,TMULT,NTOT)
      ELSE
         CALL COPY(TA,VMULT,NTOT)
      ENDIF
c
c     write(6,1) 
c    $(nid,'tab4',lglel(ie),(ta(k,1,1,ie),k=1,lx1*ly1),ie=1,nelt)
c   1 format(i3,a4,i3,16f5.2)
c
      CALL DSSUM(TA,lx1,ly1,lz1)
c
c     write(6,1) 
c    $(nid,'taaf',lglel(ie),(ta(k,1,1,ie),k=1,lx1*ly1),ie=1,nelt)
c
      CALL RONE (TB,NTOT)
      CALL SUB2 (TB,TA,NTOT)
      DO 1000 IE=1,NELT
      ieg=lglel(ie)
      DO 1000 IZ=1,lz1
      DO 1000 IY=1,ly1
      DO 1000 IX=1,lx1
         IF (ABS(TB(IX,IY,IZ,IE)).GT.EPS ) THEN
            WRITE(6,1005) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TA(IX,IY,IZ,IE),eps
c           WRITE(7,1005) IX,IY,IZ,IEG
c    $      ,XM1(IX,IY,IZ,IE),TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE)
c    $      ,QMASK(IX,IY,IZ,IE)
 1005       FORMAT(2X,'WARNING: DSSUM problem at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,2X,'Near X =',3G16.8,', d:',2G16.8)
            IERR=4
         ENDIF
 1000 CONTINUE
C
C     Set up QMASK quickly to annihilate checks on periodic bc's
C
      CALL RONE(QMASK,NTOT)
      DO 100 IEL=1,NELT
      DO 100 IFACE=1,NFACES
         CB =CBC(IFACE,IEL,IFIELD)
         IF (CB.EQ.'P  '.or.cb.eq.'p  ') 
     $         CALL FACEV(QMASK,IEL,IFACE,0.0,lx1,ly1,lz1)
  100 CONTINUE
      CALL DSOP(QMASK,'MUL',lx1,ly1,lz1)

c      xxmin = glmin(xm1,ntot)
c      yymin = glmin(ym1,ntot)
c      zzmin = glmin(zm1,ntot)
c      xxmax = glmax(xm1,ntot)
c      yymax = glmax(ym1,ntot)
c      zzmax = glmax(zm1,ntot)
c      if (nio.eq.0) write(6,7) xxmin,yymin,zzmin,xxmax,yymax,zzmax
c    7 format('xyz minmx2:',6g13.5)



C
C     X-component
C
      CALL COPY(TA,XM1,NTOT)
      CALL COPY(TB,XM1,NTOT)
      CALL DSOP(TA,'MIN',lx1,ly1,lz1)
      CALL DSOP(TB,'MAX',lx1,ly1,lz1)
      CALL SUB2(TA,XM1,NTOT)
      CALL SUB2(TB,XM1,NTOT)
      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)
      DO 1100 IE=1,NELT
         XSCMAX = VLMAX(XM1(1,1,1,IE),NXYZ1)
         XSCMIN = VLMIN(XM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(XSCMAX-XSCMIN)
         SCAL2=ABS(XSCMAX)
         SCAL3=ABS(XSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         XSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1100 IZ=1,lz1
         DO 1100 IY=1,ly1
         DO 1100 IX=1,lx1
         if (abs(ta(ix,iy,iz,ie)*xscale).gt.eps .or.
     $       abs(tb(ix,iy,iz,ie)*xscale).gt.eps ) then
            write(6,1105) ix,iy,iz,ieg
     $      ,xm1(ix,iy,iz,ie),ym1(ix,iy,iz,ie),zm1(ix,iy,iz,ie)
     $      ,tb(ix,iy,iz,ie),ta(ix,iy,iz,ie),XSCALE
 1105       format(1x,'WARNING1 Element mesh mismatch at:',/
     $            ,1x,'i,j,k,ie:',3i5,I12,/
     $            ,1X,'Near X =',3G16.8,', d:',3G16.8)
            ierr=1
         endif
 1100 CONTINUE
C
C     Y-component
C
      CALL COPY(TA,YM1,NTOT)
      CALL COPY(TB,YM1,NTOT)
      CALL DSOP(TA,'MIN',lx1,ly1,lz1)
      CALL DSOP(TB,'MAX',lx1,ly1,lz1)
      CALL SUB2(TA,YM1,NTOT)
      CALL SUB2(TB,YM1,NTOT)
      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)
      DO 1200 IE=1,NELT
         YSCMAX = VLMAX(YM1(1,1,1,IE),NXYZ1)
         YSCMIN = VLMIN(YM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(YSCMAX-YSCMIN)
         SCAL2=ABS(YSCMAX)
         SCAL3=ABS(YSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         YSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1200 IZ=1,lz1
         DO 1200 IY=1,ly1
         DO 1200 IX=1,lx1
         IF (ABS(TA(IX,IY,IZ,IE)*YSCALE).GT.EPS .OR.
     $       ABS(TB(IX,IY,IZ,IE)*YSCALE).GT.EPS ) THEN
            WRITE(6,1205) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE),yscale
 1205       FORMAT(1X,'WARNING2 Element mesh mismatch at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,1X,'Near Y =',3G16.8,', d:',3G16.8)
            IERR=2
         ENDIF
 1200 CONTINUE
C
C     Z-component
C
      IF (IF3D) THEN
       CALL COPY(TA,ZM1,NTOT)
       CALL COPY(TB,ZM1,NTOT)
       CALL DSOP(TA,'MIN',lx1,ly1,lz1)
       CALL DSOP(TB,'MAX',lx1,ly1,lz1)
       CALL SUB2(TA,ZM1,NTOT)
       CALL SUB2(TB,ZM1,NTOT)
       CALL COL2(TA,QMASK,NTOT)
       CALL COL2(TB,QMASK,NTOT)
       DO 1300 IE=1,NELT
          ZSCMAX = VLMAX(ZM1(1,1,1,IE),NXYZ1)
          ZSCMIN = VLMIN(ZM1(1,1,1,IE),NXYZ1)
          SCAL1=ABS(ZSCMAX-ZSCMIN)
          SCAL2=ABS(ZSCMAX)
          SCAL3=ABS(ZSCMIN)
          SCAL1=MAX(SCAL1,SCAL2)
          SCAL1=MAX(SCAL1,SCAL3)
          ZSCALE = 1./SCAL1
          ieg=lglel(ie)
          DO 1300 IZ=1,lz1
          DO 1300 IY=1,ly1
          DO 1300 IX=1,lx1
          IF (ABS(TA(IX,IY,IZ,IE)*ZSCALE).GT.EPS .OR.
     $        ABS(TB(IX,IY,IZ,IE)*ZSCALE).GT.EPS ) THEN
           WRITE(6,1305) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE),zscale
 1305       FORMAT(1X,'WARNING3 Element mesh mismatch at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,1X,'Near Z =',3G16.8,', d:',3G16.8)
            IERR=3
          ENDIF
 1300  CONTINUE
      ENDIF

      ierr = iglsum(ierr,1)
      IF (IERR.gt.0) THEN
         if(nid.eq.0) WRITE(6,1400) 
 1400    FORMAT
     $   (' Mesh consistency check failed.  EXITING in VRDSMSH.')
            call exitt
      ENDIF
 
      tmp(1)=ierr
      CALL GOP(tmp,tmp(2),'M  ',1)
      IF (tmp(1).ge.4.0) THEN
         WRITE(6,1400) 
     $   (' Mesh consistency check failed.  EXITING in VRDSMSH.')
         call exitt
      ENDIF
 
      if(nio.eq.0) then
        write(6,*) 'done :: verify mesh topology'
        write(6,*) ' '
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine vrdsmshx  ! verify mesh topology
C
C=====================================================================
C     Verify that mesh and dssum are properly defined by performing
C        a direct stiffness operation on the X,Y and Z coordinates.
C     Note that periodic faces are not checked here.
C=====================================================================
C
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      common /scrns/ tc(lx1,ly1,lz1,lelt),td(lx1,ly1,lz1,lelt)
     $             , ta(lx1,ly1,lz1,lelt),tb(lx1,ly1,lz1,lelt)
     $             , qmask(lx1,ly1,lz1,lelt)
      CHARACTER*3 CB
C
      IERR      = 0
      EPS       = 1.0e-04
      EPS       = 1.0e-03
      IFIELD    = 1
      IF (IFHEAT) IFIELD = 2
      NXYZ1     = lx1*ly1*lz1
      NTOT      = lx1*ly1*lz1*NELT
      NFACES    = 2*ldim

      xmx = glmax(xm1,ntot)
      xmn = glmin(xm1,ntot)
      ymx = glmax(ym1,ntot)
      ymn = glmin(ym1,ntot)
      zmx = glmax(zm1,ntot)
      zmn = glmin(zm1,ntot)
      if (nio.eq.0) write(6,*) xmn,xmx,' Xrange'
      if (nio.eq.0) write(6,*) ymn,ymx,' Yrange'
      if (nio.eq.0) write(6,*) zmn,zmx,' Zrange'
c     return
C
C     First check - use 1/Multiplicity
C
      IF (IFHEAT) THEN 
         CALL COPY(TA,TMULT,NTOT)
      ELSE
         CALL COPY(TA,VMULT,NTOT)
      ENDIF
c
c     write(6,1) 
c    $(nid,'tab4',lglel(ie),(ta(k,1,1,ie),k=1,lx1*ly1),ie=1,nelt)
c   1 format(i3,a4,i3,16f5.2)
c
      CALL DSSUM(TA,lx1,ly1,lz1)
c
c     write(6,1) 
c    $(nid,'taaf',lglel(ie),(ta(k,1,1,ie),k=1,lx1*ly1),ie=1,nelt)
c
      CALL RONE (TB,NTOT)
      CALL SUB2 (TB,TA,NTOT)
      DO 1000 IE=1,NELT
      ieg=lglel(ie)
      DO 1000 IZ=1,lz1
      DO 1000 IY=1,ly1
      DO 1000 IX=1,lx1
         IF (ABS(TB(IX,IY,IZ,IE)).GT.EPS ) THEN
            WRITE(6,1005) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TA(IX,IY,IZ,IE),eps
c           WRITE(7,1005) IX,IY,IZ,IEG
c    $      ,XM1(IX,IY,IZ,IE),TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE)
c    $      ,QMASK(IX,IY,IZ,IE)
 1005       FORMAT(2X,'WARNING: DSSUM problem at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,2X,'Near X =',3G16.8,', d:',2G16.8)
            IERR=4
         ENDIF
 1000 CONTINUE
C
C     Set up QMASK quickly to annihilate checks on periodic bc's
C
      CALL RONE(QMASK,NTOT)
      DO 100 IEL=1,NELT
      DO 100 IFACE=1,NFACES
         CB =CBC(IFACE,IEL,IFIELD)
         IF (CB.EQ.'P  '.or.cb.eq.'p  ') 
     $         CALL FACEV(QMASK,IEL,IFACE,0.0,lx1,ly1,lz1)
  100 CONTINUE
      call dsop(QMASK,'MUL',lx1,ly1,lz1)

      xxmin = glmin(xm1,ntot)
      yymin = glmin(ym1,ntot)
      zzmin = glmin(zm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ym1,ntot)
      zzmax = glmax(zm1,ntot)
      if (nio.eq.0) write(6,7) xxmin,yymin,zzmin,xxmax,yymax,zzmax
    7 format('xyz minmx2:',6g13.5)


C
C     X-component
C
      call copy(ta,xm1,ntot)
      call copy(tb,xm1,ntot)
      call dsop(ta,'min',lx1,ly1,lz1)
      call dsop(tb,'max',lx1,ly1,lz1)

      call copy(tc,xm1,ntot)
      call copy(td,xm1,ntot)
      call dsop(tc,'min',lx1,ly1,lz1)
      call dsop(td,'max',lx1,ly1,lz1)

      xxmin = glmin(xm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,9) xxmin,yymin,zzmin,xxmax,yymax,zzmax
    9 format('xyz minmx3:',6g13.5)

      CALL SUB2(TA,XM1,NTOT)
      CALL SUB2(TB,XM1,NTOT)

      xxmin = glmin(qmask,ntot)
      xxmax = glmax(qmask,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,19) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   19 format('xyz minmx4:',6g13.5)

      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)

      xxmin = glmin(qmask,ntot)
      xxmax = glmax(qmask,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,29) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   29 format('xyz minmx5:',6g13.5)

      DO 1100 IE=1,NELT
         XSCMAX = VLMAX(XM1(1,1,1,IE),NXYZ1)
         XSCMIN = VLMIN(XM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(XSCMAX-XSCMIN)
         SCAL2=ABS(XSCMAX)
         SCAL3=ABS(XSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         XSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1100 IZ=1,lz1
         DO 1100 IY=1,ly1
         DO 1100 IX=1,lx1
         if (abs(ta(ix,iy,iz,ie)*xscale).gt.eps .or.
     $       abs(tb(ix,iy,iz,ie)*xscale).gt.eps ) then
            write(6,1105) nid,ix,iy,iz,ie,ieg
     $      ,xm1(ix,iy,iz,ie),tc(ix,iy,iz,ie),td(ix,iy,iz,ie)
     $      ,ym1(ix,iy,iz,ie),zm1(ix,iy,iz,ie)
     $      ,ta(ix,iy,iz,ie),tb(ix,iy,iz,ie),xscale
     $      ,qmask(ix,iy,iz,ie)
 1105       format(i4.4,1x,'ie:',3i3,i10,i10,1p9e11.3)
c1105       format(i4.4,1x,'ie:',3i3,i6,1p9e11.3)
            ierr=1
            goto 1101
         endif
 1100 CONTINUE
 1101 CONTINUE

      xxmin = glmin(xm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,39) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   39 format('xyz minmx5:',6g13.5)

c     ifvo = .true.
c     ifpo = .false.
c     ifto = .true.
c     call outpost(xm1,ta,tb,pr,qmask,'   ')
c     call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine rotat2(xyz,angle,npts)
C
C     Rotate NPTS through ANGLE (in two directions IF3D).
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZ(3,1)
      COMMON /CTMP0/ RMTRX(3,3),RX(3,3),RZ(3,3),XYZN(3,10)
C
      SINA=SIN(ANGLE)
      COSA=COS(ANGLE)
      CALL RZERO(RX,9)
      CALL RZERO(RZ,9)
      RX(1,1)=COSA
      RX(2,2)=COSA
      RX(1,2)=SINA
      RX(2,1)=-SINA
      RX(3,3)=1.0
      IF (IF3D) THEN
         RZ(1,1)=COSA
         RZ(3,3)=COSA
         RZ(1,3)=SINA
         RZ(3,1)=-SINA
         RZ(2,2)=1.0
      ELSE
         RZ(1,1)=1.0
         RZ(2,2)=1.0
         RZ(3,3)=1.0
      ENDIF
      CALL MXM(RX,3,RZ,3,RMTRX,3)
C
C     Strip mine mxms in chunks of 10:
      DO 100 I=1,NPTS-10,10
         CALL MXM(RMTRX,3,XYZ(1,I),3,XYZN,10)
         CALL COPY(XYZ(1,I),XYZN,30)
  100 CONTINUE
      N10=MOD1(NPTS,10)
      I=NPTS-N10+1
      CALL RZERO(XYZN,30)
      IF (N10.GT.0) THEN
         CALL MXM(RMTRX,3,XYZ(1,I),3,XYZN,N10)
         CALL COPY(XYZ(1,I),XYZN,3*N10)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine scale(xyzl,nl)
C
C     Rescale XYZL such that the mean value of IXX=IYY=IZZ for each element.
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZL(3,8,LELT)
      COMMON /CTMP0/ VO(LELT),XYZI(3,LELT),CG(3,LELT)
     $              ,TI(6),WORK(6)
C
C     Compute volumes -
C
      CALL VOLUME2(VO,XYZL,NL)
      VTOT=GLSUM (VO,NL)
C
C     Compute (weighted) average inertia for each element.
C
      NCRNR=2**ldim
      CALL RZERO(TI,6)
      DO 100 IL=1,NL
         VO0 = VO(IL)/VTOT
         CALL INRTIA(XYZI(1,IL),CG(1,IL),XYZL(1,1,IL),NCRNR,1)
         TI(1)=TI(1)+XYZI(1,IL)*VO0
         TI(2)=TI(2)+XYZI(2,IL)*VO0
         TI(3)=TI(3)+XYZI(3,IL)*VO0
         TI(4)=TI(4)+CG(1,IL)  *VO0
         TI(5)=TI(5)+CG(2,IL)  *VO0
         TI(6)=TI(6)+CG(3,IL)  *VO0
  100 CONTINUE
      CALL GOP(TI,WORK,'+  ',6)
      XI  =SQRT(TI(1))
      YI  =SQRT(TI(2))
      ZI  =1.0
      IF (IF3D) ZI=SQRT(TI(3))
C
C     Rescale ( & shift to a nearly mean zero )
C
      DO 200 IL=1,NL
      DO 200 IC=1,NCRNR
         XYZL(1,IC,IL)=(XYZL(1,IC,IL)-TI(4))/XI
         XYZL(2,IC,IL)=(XYZL(2,IC,IL)-TI(5))/YI
         XYZL(3,IC,IL)=(XYZL(3,IC,IL)-TI(6))/ZI
  200 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine inrtia(xyzi,cg,xyzl,n,itype)
C
C     Compute cg and inertia for a collection of unit point masses.
C     This is a global (multiprocessor) operation, only IF itype=2.
C
      DIMENSION XYZI(3),CG(3),XYZL(3,1)
      DIMENSION TI(4),WORK(4)
C
      TI(1)=0.0
      TI(2)=0.0
      TI(3)=0.0
      TI(4)=N
      DO 100 I=1,N
         TI(1)=TI(1)+XYZL(1,I)
         TI(2)=TI(2)+XYZL(2,I)
         TI(3)=TI(3)+XYZL(3,I)
  100 CONTINUE
      IF (ITYPE.EQ.2) CALL GOP(TI,WORK,'+  ',4)
      IF (TI(4).EQ.0.0) TI(4)=1.0
      CG(1)=TI(1)/TI(4)
      CG(2)=TI(2)/TI(4)
      CG(3)=TI(3)/TI(4)
C
      TI(1)=0.0
      TI(2)=0.0
      TI(3)=0.0
      DO 200 I=1,N
         TI(1)=TI(1)+( XYZL(1,I)-CG(1) )**2
         TI(2)=TI(2)+( XYZL(2,I)-CG(2) )**2
         TI(3)=TI(3)+( XYZL(3,I)-CG(3) )**2
  200 CONTINUE
      IF (ITYPE.EQ.2) CALL GOP(TI,WORK,'+  ',3)
      TI(1)=TI(1)/TI(4)
      TI(2)=TI(2)/TI(4)
      TI(3)=TI(3)/TI(4)
      IF (ITYPE.EQ.2) THEN
C        std. def'n of inertia.
         XYZI(1)=TI(2)+TI(3)
         XYZI(2)=TI(3)+TI(1)
         XYZI(3)=TI(1)+TI(2)
      ELSE
         XYZI(1)=TI(1)
         XYZI(2)=TI(2)
         XYZI(3)=TI(3)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine volume2(vol,xyz,n)
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZ(3,2,2,2,1)
      DIMENSION VOL(1)
C
      DO 1000 IE=1,N
         VOL(IE)=0.0
         IF (IF3D) THEN        
           DO 20 K=1,2
           DO 20 J=1,2
           DO 20 I=1,2
              VOL1 = (XYZ(1,2,J,K,IE)-XYZ(1,1,J,K,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,I,J,2,IE)-XYZ(3,I,J,1,IE))
              VOL2 = (XYZ(1,2,J,K,IE)-XYZ(1,1,J,K,IE))
     $             * (XYZ(2,I,J,2,IE)-XYZ(2,I,J,1,IE))
     $             * (XYZ(3,I,2,K,IE)-XYZ(3,I,1,K,IE))
              VOL3 = (XYZ(1,I,2,K,IE)-XYZ(1,I,1,K,IE))
     $             * (XYZ(2,2,J,K,IE)-XYZ(2,1,J,K,IE))
     $             * (XYZ(3,I,J,2,IE)-XYZ(3,I,J,1,IE))
              VOL4 = (XYZ(1,I,J,2,IE)-XYZ(1,I,J,1,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,I,2,K,IE)-XYZ(3,I,1,K,IE))
              VOL5 = (XYZ(1,I,2,K,IE)-XYZ(1,I,1,K,IE))
     $             * (XYZ(2,I,J,2,IE)-XYZ(2,I,J,1,IE))
     $             * (XYZ(3,2,J,K,IE)-XYZ(3,1,J,K,IE))
              VOL6 = (XYZ(1,I,J,2,IE)-XYZ(1,I,J,1,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,2,J,K,IE)-XYZ(3,1,J,K,IE))
              VOL(IE) = VOL(IE)+VOL1+VOL2+VOL3+VOL4+VOL5+VOL6
   20      CONTINUE
           VOL(IE)=VOL(IE)/8.0
         ELSE
C     2-D:
            DO 40 J=1,2
            DO 40 I=1,2
              VOL1 = (XYZ(1,2,J,1,IE)-XYZ(1,1,J,1,IE))
     $             * (XYZ(2,I,2,1,IE)-XYZ(2,I,1,1,IE))
              VOL3 = (XYZ(1,I,2,1,IE)-XYZ(1,I,1,1,IE))
     $             * (XYZ(2,2,J,1,IE)-XYZ(2,1,J,1,IE))
              VOL(IE)=VOL(IE)+VOL1+VOL3
   40      CONTINUE
           VOL(IE)=VOL(IE)/4.0
         ENDIF
         VOL(IE)=ABS(VOL(IE))
 1000 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine findcg(cg,xyz,n)
C
C     Compute cg for N elements.
C
      INCLUDE 'SIZE'
      DIMENSION CG(3,1),XYZ(3,8,1)
C
      NCRNR=2**ldim
      CALL RZERO(CG,3*N)
      DO 100 I =1,N
      DO 100 IC=1,NCRNR
         CG(1,I)=CG(1,I)+XYZ(1,IC,I)
         CG(2,I)=CG(2,I)+XYZ(2,IC,I)
         CG(3,I)=CG(3,I)+XYZ(3,IC,I)
  100 CONTINUE
      TMP=1.0/(NCRNR)
      CALL CMULT(CG,TMP,3*N)
      return
      END
c-----------------------------------------------------------------------
      subroutine divide(list1,list2,nl1,nl2,ifok,list,nl,xyzi,cg,WGT)
C
C     Divide the elements associated with this subdomain according to
C     the direction having the smallest moment of inertia (the "long"
C     direction).
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'TSTEP'
C
      DIMENSION LIST(LELT),LIST1(LELT),LIST2(LELT)
      DIMENSION XYZI(3),CG(3,LELT),wgt(1)
      COMMON /CTMP0/ XCG(LELT),YCG(LELT),ZCG(LELT)
      REAL IXX,IYY,IZZ
      INTEGER WORK(2),WRK2(2)
      LOGICAL IFOK
C
C     Choose "long" direction:
C
      IXX=XYZI(1)
      IYY=XYZI(2)
      IZZ=XYZI(3)
      IF (IF3D) THEN
         IF (IXX.LE.IYY.AND.IXX.LE.IZZ) THEN
            DO 104 IE=1,NL
               XCG(IE)=CG(1,IE)
               YCG(IE)=CG(2,IE)
               ZCG(IE)=CG(3,IE)
  104       CONTINUE
         ELSEIF (IYY.LE.IXX.AND.IYY.LE.IZZ) THEN
            DO 106 IE=1,NL
               XCG(IE)=CG(2,IE)
               YCG(IE)=CG(3,IE)
               ZCG(IE)=CG(1,IE)
  106       CONTINUE
         ELSEIF (IZZ.LE.IXX.AND.IZZ.LE.IYY) THEN
            DO 108 IE=1,NL
               XCG(IE)=CG(3,IE)
               YCG(IE)=CG(1,IE)
               ZCG(IE)=CG(2,IE)
  108       CONTINUE
         ENDIF
      ELSE
C     2-D:
         IF (IXX.LE.IYY) THEN
            DO 114 IE=1,NL
               XCG(IE)=CG(1,IE)
               YCG(IE)=CG(2,IE)
  114       CONTINUE
         ELSE
            DO 116 IE=1,NL
               XCG(IE)=CG(2,IE)
               YCG(IE)=CG(1,IE)
  116       CONTINUE
         ENDIF
      ENDIF
      call col2(xcg,wgt,nl)
      call col2(ycg,wgt,nl)
      call col2(zcg,wgt,nl)
C
C     Find median value of CG to determine dividing point:
C
      XM=FMDIAN(XCG,NL,IFOK)
      YM=FMDIAN(YCG,NL,IFOK)
      ZM=0.0
      IF (IF3D) ZM=FMDIAN(ZCG,NL,IFOK)
C
C     Diagnostics
C
      IF (.NOT.IFOK) THEN
         WRITE(6,130) NID,NL,XM,YM,ZM
         DO 120 IL=1,NL
            WRITE(6,135) NID,IL,XCG(IL),YCG(IL),ZCG(IL)
  120    CONTINUE
  130    FORMAT(I10,'DIVIDE: NL,XM,YM,ZM',I3,3F12.5)
  135    FORMAT(I10,'DIVIDE: NID,IL,XC,YC,ZCG',I4,3F12.5)
      ENDIF
C
C=============================================================
C     Divide LIST into LIST1 (XCG < XM) and LIST2 (XCG>XM).
C=============================================================
C
      NL1=0
      NL2=0
      DO 200 IE=1,NL
         IF (XCG(IE).LT.XM) THEN
            NL1=NL1+1
            LIST1(NL1)=LIST(IE)
         ENDIF
         IF (XCG(IE).GT.XM) THEN
            NL2=NL2+1
            LIST2(NL2)=LIST(IE)
         ENDIF
         IF (XCG(IE).EQ.XM) THEN
C
C           We have to look at the other directions to arrive at
C           a unique subdivision algortithm.
C
C
C           More Diagnostics
C
            IF (.NOT.IFOK) WRITE(6,201) NID,IE,XCG(IE),XM
  201    FORMAT(I10,'DIVIDE: IE,XCG,XM:',I4,3F12.5)
C
            IF (YCG(IE).LT.YM) THEN
               NL1=NL1+1
               LIST1(NL1)=LIST(IE)
            ENDIF
            IF (YCG(IE).GT.YM) THEN
               NL2=NL2+1
               LIST2(NL2)=LIST(IE)
            ENDIF
            IF (YCG(IE).EQ.YM) THEN
C              look at 3rd direction.  
               IF (IF3D .AND. ZCG(IE).LT.ZM) THEN
                  NL1=NL1+1
                  LIST1(NL1)=LIST(IE)
               ELSE IF (IF3D .AND. ZCG(IE).GT.ZM) THEN
                  NL2=NL2+1
                  LIST2(NL2)=LIST(IE)
               ELSE 
C                 for 2- or 3-D intdeterminate case:
                  NL1=NL1+1
                  LIST1(NL1)=LIST(IE)
               ENDIF
            ENDIF
C
         ENDIF
  200 CONTINUE
C
C     Check for an even distribution (i.e. - not different by
C     more than 1):
C
      IFOK=.TRUE.  
      WORK(1)=NL1 
      WORK(2)=NL2
      CALL IGOP(WORK,WRK2,'+  ',2)
      IF (ABS(WORK(1)-WORK(2)).GT.1) IFOK=.FALSE.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine bufchk(buf,n)
      integer n
      real buf(n)
      do i=1,n
         write(6,*) buf(i), ' whhhh'
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine chk_xyz
      include 'SIZE'
      include 'TOTAL'
      integer e,f,eg

      do e=1,nelt
         eg = lglel(e)
         write(6,1) nid,eg,e,(cbc(f,e,1),f=1,6)
      enddo
    1 format(3i12,6(1x,a3),'  cbc')

      return
      end
c-----------------------------------------------------------------------
      subroutine chk_nel
      include 'SIZE'
      include 'TOTAL'

      neltmx=np*lelt
      nelvmx=np*lelv

      neltmx=min(neltmx,lelg)
      nelvmx=min(nelvmx,lelg)

      nelgt = iglmax(nelgt,1)
      nelgv = iglmax(nelgv,1)

c     write(6,*) nid,' inside chk_nel',nelgt,neltmx,nelvmx

      if (nelgt.gt.neltmx.or.nelgv.gt.nelvmx) then
         if (nid.eq.0) then
          lelt_needed = nelgt/np
          if (mod(nelgt,np).ne.0) lelt_needed = lelt_needed + 1 
          write(6,12) lelt,lelg,lelt_needed,np,nelgt
 12         format(//,2X,'ABORT: Problem size too large!'
     $         ,/,2X
     $         ,/,2X,'This solver has been compiled for:'
     $         ,/,2X,'   number of elements/proc  (lelt):',i12
     $         ,/,2X,'   total number of elements (lelg):',i12
     $         ,/,2X
     $         ,/,2X,'Recompile with the following SIZE  parameters:'
     $         ,/,2X,'   lelt >= ',i12,'  for np = ',i12
     $         ,/,2X,'   lelg >= ',i12,/)
c           write(6,*)'help:',lp,np,nelvmx,nelgv,neltmx,nelgt
c           write(6,*)'help:',lelt,lelv,lelgv
         endif
         call exitt
      endif

      if(nelgt.gt.nelgt_max) then
        if(nid.eq.0) write(6,*)
     $               'ABORT: Total number of elements too large!',
     $               '       nel_max = ', nelgt_max 
        call exitt
      endif

      if (nelt.gt.lelt) then
        write(6,'(A,3I12)') 'ABORT: nelt>lelt!', nid, nelt, lelt
        call exitt
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cscan(sout,key,nk)

      character*132 sout,key
      character*132 string
      character*1  string1(132)
      equivalence (string1,string)
c
      do i=1,100000000
         call blank(string,132)
         read (nk,80,end=100,err=100) string
         call chcopy(sout,string,132)
c        write (6,*) string
         if (indx1(string,key,nk).ne.0) return
      enddo
  100 continue
c
   80 format(a132)
      return

      end
c-----------------------------------------------------------------------
