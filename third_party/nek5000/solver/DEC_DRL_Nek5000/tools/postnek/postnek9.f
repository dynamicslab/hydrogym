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
      subroutine mltiplt2
c     For hmt's movies.... pff 2/28/98.
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp,iftmh
      save    ifdrm,iftmp,iftmh
c
      common /c46/ ifopen46
      logical      ifopen46
c
      character*20 plform_tmp
      character*8  boxname
c
      ifauto   = .true.
      ifopen46 = .false.
      plform_tmp=plform

      if (if_output_ijke) open(unit=33,file='ijke.out')
c
c     call prsi ('Current number of states:$',nstate)
c     call prs  ('Save state (1=yes, 0=no)?$')
c     call rei  (isave)
      isave = 0
c
      if (isave.eq.1) then
         nstat=nstate+1
         jstat=nstat
         call savstate(jstat,nstat,ir,ic,il,ii,'save')
      endif
c
      IFTMP  = IFCLRB
      IFCLRB = .FALSE.
      IFTMH  = IFHARD
      write(6,*) 'ifhard:',ifhard
c
      nframe = 1
c     write(6,*) 'nstate:',nstate,nframe
c     call prs('Input number of frames following this one:$')
c     call rei(nframe)
c
      call prs  ('Input start,stop, skip fld numbers:$')
      call reiii (iframe0,iframe1,iskip)
c
      call prs  ('Trim the ouput frame (1=yes, 0=no, <0=no dump)?$')
      call rei  (itrim)
c
      call prs  ('Silent movie (1=yes, 0=no)?$')
      call rei  (isilent)
c
      if (itrim.gt.0) call auto_size_pixel_window
c
      kframe = 0
      do iframe = iframe0,iframe1,iskip
         if (if_auto_box) then
             write(boxname,33) iframe
   33        format('box.',i4.4)
             if (ifmtpf) then
                open (unit=46,file=boxname)
             else
                open (unit=46,file=boxname,form='unformatted')
             endif
             ifopen46 = .true.
         endif

         IFHARD = IFTMH
c
c        Turn window on or off.
c
         if (isilent.eq.1.and.iframe.gt.iframe0) call htw_t()
         if (isilent.eq.1.and.iframe.ge.iframe1) call htw_f()
c
c        Get .fld_iframe ....
         ndumps = iframe
         call getfld(ndumps,ierr,.true.,.true.)
         call userchk
c
         write(6,*) 'ifhard:',ifhard,iframe
         CALL CLEAR
c        CALL DRMESH
         CALL HARD
         CALL HEJECT
         CALL DRMESH
         CALL DRFRAM
         CALL NOHARD
c
         visc = param(2)
         if (visc.le.0) then
            rey = -visc
         else
            rey = 1./visc
         endif
         call blank(line,70)
         write(line,8) rey,nx
    8    format('Reynolds Number =',f7.0,'  N=',i2,'$')
         CALL GSWRIT(.02,.965,1.0,line)
         call blank(line,70)
c
c        dt_run = .0098272
c        time_frame = kframe*dt_run
         time_frame = time
         write(line,70) time_frame,kframe
   70    format('Time = ',f12.7,'  Frame =',i5,'$')
c        CALL GSWRIT(.02,.565,1.0,line)
         CALL HARD

c        Plot all requested forms...

         do i=1,nstate
            call setstate(i,ifdrm,ir,ic,il,ii,'mlti')
            ifhard = iftmh

            write(6,*) 
            if (plform.eq.'X-Y PLOT'.and.xyattr.eq.'SET OF PROFILES'
     $      .and..not.ifopen46) then
                open (unit=46,file='profile.out')
                ifopen46 = .true.
            endif
c
c           Note, getfld does nothing if "ndumps" is unchanged between
c           successive calls
c
            CALL SETWRK(.false.)
            if (ifdrm) CALL DRMESH
            CALL PLOTIT(0)
         ENDDO
c
         kframe = kframe+1
c        call grab_window('x12_movie\0',kframe)
c        if (itrim.ge.0) call grab_window_pix('movie',5,kframe)
         if (itrim.ge.0) call grab_window_pix('movie',5,iframe)
         CALL NOHARD
c
      ENDDO

      i=1
      call setstate(i,ifdrm,ir,ic,il,ii,'mlti') ! reset state
      ifhard = iftmh

      CALL PRS('WAKE UP!!  The movie is over!!$')
      if (isilent.eq.1) call htw_f()
      IFCLRB=IFTMP
      plform=plform_tmp
c
c
      ifauto = .false.
c
c     Check to close any open files, etc.
c
      if (if_output_ijke) close(unit=33)
      if (ifopen46) then  ! close profile.out 
         ifopen46 = .false.
         close(unit=46)
      endif
c
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
c
      IFHARD=IFTMP
c
      return
      end
C-----------------------------------------------------------------------
      subroutine mltiplt
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp,iftmh
C
      CALL CLEAR
      IFTMP  = IFCLRB
      IFCLRB = .FALSE.
      IFTMH  = IFHARD
      write(6,*) 'nstate:',nstate
      do 1000 i=1,nstate
         ifhard = iftmh
         call setstate(i,ifdrm,ir,ic,il,ii,'mlti')
         call getfld(ndumps,ierr,.true.,.true.)
         call setwrk(.false.)
         if (ifdrm) call drmesh
         call plotit(0)
 1000 continue
      IFCLRB = IFTMP
      IFHARD = iftmh
      PLFORM='MULTIPLOT'
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
      CALL PRS('Hit left button for menu, right for dump$')
      CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
c
      IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)

      i=1
      call setstate(i,ifdrm,ir,ic,il,ii,'mlti') ! reset state

      ifhard=iftmp

C
      return
      end
C-----------------------------------------------------------------------
      subroutine setstate(is,ifdrm,ir,ic,il,ii,name)
C
C     Set up the currnet plotting state
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm
      character*4 name
C
      ir=0
      ic=0
      il=0
      ii=0
C
C     State of the screen
C
                        ir=ir+1
      XFAC=RSTAT(IR,is)
                        ir=ir+1
      YFAC=RSTAT(IR,is)
                        ir=ir+1
      XZERO=RSTAT(IR,is)
                        ir=ir+1
      YZERO=RSTAT(IR,is)
                        ir=ir+1
      CALL COPY(XHOBS,RSTAT(ir,is),3)
                        ir=ir+3
      CALL COPY(YHOBS,RSTAT(ir,is),3)
                        ir=ir+3
      CALL COPY(VHOBS,RSTAT(ir,is),3)
                        ir=ir+3
      THETA=RSTAT(IR,is)
                        ir=ir+1
      PHI=RSTAT(IR,is)
                        ir=ir+1
      XPHY0=RSTAT(IR,is)
                        ir=ir+1
      YPHY0=RSTAT(IR,is)
                        ir=ir+1
      wt    = RSTAT(IR,is)
                        ir=ir+1
      wb    = RSTAT(IR,is)
                        ir=ir+1
      wl    = RSTAT(IR,is)
                        ir=ir+1
      wr    = RSTAT(IR,is)
                        ir=ir+1
      xpmn  = RSTAT(IR,is)
                        ir=ir+1
      ypmn  = RSTAT(IR,is)
                        ir=ir+1
      xpmx  = RSTAT(IR,is)
                        ir=ir+1
      ypmx  = RSTAT(IR,is)
c
c     write(6,*) 'this is wt-wr:',wt,wb,wl,wr
c     write(6,*) 'this is jt-jr:',(      jr    ,jr=ir,ir-4,-1)
c
                        ii=ii+1
      IROT=ISTAT(Ii,is)
C
C     Reals
C
                        ir=ir+1
      thrmin = RSTAT(IR,is)
                        ir=ir+1
      thrmax = RSTAT(IR,is)
                        ir=ir+1
      usrmin = RSTAT(IR,is)
                        ir=ir+1
      usrmax = RSTAT(IR,is)
                        ir=ir+1
      cont_lev = RSTAT(IR,is)
C
C     Logicals
C
                        il=il+1
      IFSELE=LSTAT(il,is)
                        il=il+1
      IFDRAX=LSTAT(il,is)
                        il=il+1
c     IFGNGM=LSTAT(il,is)
                        il=il+1
c     IFGRAF=LSTAT(il,is)
      call setgraph(ifgraf)
                        il=il+1
c     IFHARD=LSTAT(il,is)
c                       il=il+1
      IFPOST=LSTAT(il,is)
                        il=il+1
      IFPLAN=LSTAT(il,is)
                        il=il+1
      IFPROJ=LSTAT(il,is)
                        il=il+1
      IFDMSH=LSTAT(il,is)
                        il=il+1
      IFCFBD=LSTAT(il,is)
                        il=il+1
      IFZOOM=LSTAT(il,is)
c     write(6,*) 'ifzoom:',ifzoom,il,is,lstat(il,is)
                        il=il+1
      ifportrait=LSTAT(il,is)
                        il=il+1
      IFFULL=LSTAT(il,is)
                        il=il+1
      IFUSCL=LSTAT(il,is)
c                       il=il+1
c     IFDSSUM=LSTAT(il,is)
C
C     Characters
C
                        ic=ic+1
      SCALPT= cSTAT(ic,IS)
                        ic=ic+1
      VECTPT= cSTAT(ic,IS)
                        ic=ic+1
      XYATTR= cSTAT(ic,IS)
                        ic=ic+1
      PLFORM= cSTAT(ic,IS)
                        ic=ic+1
      DERIV = cSTAT(ic,IS)
                        ic=ic+1
      QUANTY= cSTAT(ic,IS)
                        ic=ic+1
      SUATTR= cSTAT(ic,IS)
                        ic=ic+1
      COMPON= cSTAT(ic,IS)
C
C     Integers
C
                        ii=ii+1
      Ndumps=ISTAT(II,IS)
c     write(6,*) 'this is ndumps in setstat',ndumps
c
c     draw mesh?
                        ii=ii+1
      idrm=ISTAT(II,IS)
      ifdrm=.false.
      if (idrm.eq.1) ifdrm=.true.
      write(6,*) 'set: ii,is,idrm:',ii,is,idrm,ifdrm
c
c     Plot planes
                        ii=ii+1
      Nmirror=ISTAT(II,IS)
                        ii=ii+1
      NLSTP=ISTAT(II,IS)
                        ii=ii+1
      CALL ICOPY(LIST,ISTAT(II,IS),NLSTP)
                        ii=ii+nlstp
c
c     Now save resolution always (pff 4/6/98)
c     if (name.eq.'strt') then
c
c        Save resolution
c
         nxcont = ISTAT(II,IS)
                                 ii=ii+1
         nxgrid = ISTAT(II,IS)
                                 ii=ii+1
         nxband = ISTAT(II,IS)
                                 ii=ii+1
         nptpro = ISTAT(II,IS)
                                 ii=ii+1
ccc   endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine savstate(is,ns,ir,ic,il,ii,name)
C
C     SAVE the currnet plotting state
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      character*4 name
C
      nstate=ns
C
      ir=0
      ic=0
      il=0
      ii=0
C
C     State of the screen
C
                              ir=ir+1
      RSTAT(IR,is)=XFAC
                              ir=ir+1
      RSTAT(IR,is)=YFAC
                              ir=ir+1
      RSTAT(IR,is)=XZERO
                              ir=ir+1
      RSTAT(IR,is)=YZERO
                              ir=ir+1
      CALL COPY(RSTAT(ir,is),XHOBS,3)
                              ir=ir+3
      CALL COPY(RSTAT(ir,is),YHOBS,3)
                              ir=ir+3
      CALL COPY(RSTAT(ir,is),VHOBS,3)
                              ir=ir+3
      RSTAT(IR,is)=THETA
                              ir=ir+1
      RSTAT(IR,is)=PHI
                              ir=ir+1
      RSTAT(IR,is)=XPHY0
                              ir=ir+1
      RSTAT(IR,is)=YPHY0
                              ir=ir+1
      RSTAT(IR,is)=wt    
                              ir=ir+1
      RSTAT(IR,is)=wb    
                              ir=ir+1
      RSTAT(IR,is)=wl    
                              ir=ir+1
      RSTAT(IR,is)=wr    
                              ir=ir+1
      RSTAT(IR,is)=xpmn    
                              ir=ir+1
      RSTAT(IR,is)=ypmn    
                              ir=ir+1
      RSTAT(IR,is)=xpmx    
                              ir=ir+1
      RSTAT(IR,is)=ypmx    
c
c     write(6,*) 'this is wt-wr:',(rstat(jr,is),jr=ir,ir-4,-1)
c     write(6,*) 'this is jt-jr:',(      jr    ,jr=ir,ir-4,-1)
c
                              ii=ii+1
      ISTAT(Ii,is)=IROT
C
C     Reals
C
                        ir=ir+1
      RSTAT(IR,is) =  thrmin
                        ir=ir+1
      RSTAT(IR,is) =  thrmax
                        ir=ir+1
      RSTAT(IR,is) =  usrmin 
                        ir=ir+1
      RSTAT(IR,is) =  usrmax 
                        ir=ir+1
      RSTAT(IR,is) =  cont_lev 
C
C     Logicals
C
                              il=il+1
      LSTAT(il,is)=IFSELE
                              il=il+1
      LSTAT(il,is)=IFDRAX
                              il=il+1
      LSTAT(il,is)=IFGNGM
                              il=il+1
      LSTAT(il,is)=IFGRAF
                              il=il+1
c     LSTAT(il,is)=IFHARD
c                             il=il+1
      LSTAT(il,is)=IFPOST
                              il=il+1
      LSTAT(il,is)=IFPLAN
                              il=il+1
      LSTAT(il,is)=IFPROJ
                              il=il+1
      LSTAT(il,is)=IFDMSH
                              il=il+1
      LSTAT(il,is)=IFCFBD
                              il=il+1
c     write(6,*) 'IFSOOM:',ifzoom,il,is,lstat(il,is)
      LSTAT(il,is)=IFZOOM
                              il=il+1
      LSTAT(il,is)=ifportrait
                              il=il+1
      LSTAT(il,is)=IFFULL
                              il=il+1
      LSTAT(il,is)=IFUSCL
c                             il=il+1
c     LSTAT(il,is)=IFDSSUM
C
C     Characters
C
                              ic=ic+1
      cSTAT(ic,IS)=SCALPT
                              ic=ic+1
      cSTAT(ic,IS)=VECTPT
                              ic=ic+1
      cSTAT(ic,IS)=XYATTR
                              ic=ic+1
      cSTAT(ic,IS)=PLFORM
                              ic=ic+1
      cSTAT(ic,IS)=DERIV
                              ic=ic+1
      cSTAT(ic,IS)=QUANTY
                              ic=ic+1
      cSTAT(ic,IS)=SUATTR
                              ic=ic+1
      cSTAT(ic,IS)=COMPON
C
C     Integers
C
                              ii=ii+1
      ISTAT(II,IS)=Ndumps
c     write(6,*) 'this is ndumps in savestat',ndumps
c     draw mesh?
                              ii=ii+1
      istat(ii,is)            = 1
      if (name.eq.'save') then
         call prs('Redraw mesh?? (1=yes,0=no)$')
         call rei(istat(ii,is))
      elseif (name.eq.'nmsh') then
         istat(ii,is)         = 0
      endif
      istat(ii,is)            = 0
      write(6,*) 'sav: ii,is,idrm:',ii,is,istat(ii,is)


c
c     plot planes
                              ii=ii+1
      ISTAT(II,IS)=Nmirror
                              ii=ii+1
      ISTAT(II,IS)=NLSTP
                              ii=ii+1
      CALL ICOPY(ISTAT(II,IS),LIST,NLSTP)
                              ii=ii+nlstp
c     Now save resolution always (pff 4/6/98)
ccc   if (name.eq.'quit') then
c
c        Save resolution
c
         ISTAT(II,IS)= NXCONT
                                 ii=ii+1
         ISTAT(II,IS)= NXGRID
                                 ii=ii+1
         ISTAT(II,IS)= NXBAND
                                 ii=ii+1
         ISTAT(II,IS)= NPTPRO
                                 ii=ii+1
ccc   endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine genstrv
C
C     Generate Stress Vectors on selected planes
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
      PARAMETER (NXM2=NXM*NYM)
      PARAMETER (NXM3=NXM*NYM*NZM)
      COMMON /SCRTCH/ XPLOT(NXM2),YPLOT(NXM2),ZPLOT(NXM2)
     $               ,XVEC(NXM2) ,YVEC(NXM2) ,ZVEC(NXM2)
     $               ,dudr(NXM3) ,duds(NXM3) ,dudt(NXM3)
     $               ,dxds(NXM2) ,dyds(NXM2) ,dzds(NXM2)
      DIMENSION       dxdr(NXM2) ,dydr(NXM2) ,dzdr(NXM2)
     $               ,drdn(NXM2) ,dsdn(NXM2) ,dtdn(NXM2)
      equivalence (dxdr,dudr),(dydr,duds),(dzdr,dudt)
      equivalence (dxds,drdn),(dyds,dsdn),(dzds,dtdn)
c
      LOGICAL IFCRV(4),IFNX
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
      nxyz = nx*ny*nz
      uvmax=0.0
      IF (IF3D) THEN
C
C        Sort the selected list of planes according to visibility
         CALL SORTL
C
C        Plot the selected planes
         DO 9000 IP=1,NLSTP
C
            LISTA=ABS(LIST(IP))
            CALL DECOD(IPLANE,IPLN,IEL,IDUM,LISTA,NX,6,NELM)
c           write(6,*) 'n',nlstp,iel,ipln,iplane
c
            I1=1
            I2=NX
            J1=1
            J2=NY
            K1=1
            K2=NZ
c
C           Plot them all
            I3=1
            J3=1
            K3=1
            NXSKIP=NX 
c
            IF (IPLN.LE.2) THEN
               I1=IPLANE
               I2=IPLANE
               I3=1
            ELSEIF (IPLN.LE.4) THEN
               J1=IPLANE
               J2=IPLANE
               J3=1
            ELSE
               K1=IPLANE
               K2=IPLANE
               K3=1
            ENDIF
            II=0
            DO 100 K=K1,K2,K3
            DO 100 J=J1,J2,J3
            DO 100 I=I1,I2,I3
               IPOINT=IND(I,J,K,IEL)
               II=II+1
               XPLOT(II)=XP(IPOINT)
               YPLOT(II)=YP(IPOINT)
               ZPLOT(II)=ZP(IPOINT)
  100       CONTINUE
            nxy = ii
C
C           Compute normal vector
C
            call mxm(dgdr,nx,xplot ,nx,dxdr,nx)
            call mxm(dgdr,nx,yplot ,nx,dydr,nx)
            call mxm(dgdr,nx,zplot ,nx,dzdr,nx)
            call mxm(xplot,nx,dgdrt,nx,dxds,nx)
            call mxm(yplot,nx,dgdrt,nx,dyds,nx)
            call mxm(zplot,nx,dgdrt,nx,dzds,nx)
            call vcross
     $           (xvec,yvec,zvec,dxdr,dydr,dzdr,dxds,dyds,dzds,nxy)
            alpha=1.0
            if(ipln.eq.2.or.ipln.eq.3.or.ipln.eq.6) then
c              left-handed
               alpha = -alpha
            endif
            call vnorms(xvec,yvec,zvec,alpha,nxy)
c
            IF(QUANTY.EQ.'NORMAL VEC') THEN
               ii = 0
               call dudrst(dudr,duds,dudt,u(ieoff))
               DO 190 K=K1,K2,K3
               DO 190 J=J1,J2,J3
               DO 190 I=I1,I2,I3
                  IPOINT=IND(I,J,K,IEL)
                  II=II+1
                  wkv1(ipoint) = xvec(ii)
                  wkv2(ipoint) = yvec(ii)
                  wkv3(ipoint) = zvec(ii)
  190          CONTINUE
            ELSE
c
c              Normals are computed... now compute du/dn along normal
c              NOTE:  We rely upon incompressibility to annihilate the
c                     derivative of the normal velocity wrt the normal
c                     direction.
c
c              Compute du/dn near surface with spectral formulation
c
               IEOFF=NXYZ*(IEL-1)+1
c
               ii = 0
               DO 200 K=K1,K2,K3
               DO 200 J=J1,J2,J3
               DO 200 I=I1,I2,I3
                  II=II+1
                  IPOINT=IND(I,J,K,IEL)
                  drdn(ii) = rxm1(ipoint)*xvec(ii)
     $                     + rym1(ipoint)*yvec(ii)
     $                     + rzm1(ipoint)*zvec(ii)
                  dsdn(ii) = sxm1(ipoint)*xvec(ii)
     $                     + sym1(ipoint)*yvec(ii)
     $                     + szm1(ipoint)*zvec(ii)
                  dtdn(ii) = txm1(ipoint)*xvec(ii)
     $                     + tym1(ipoint)*yvec(ii)
     $                     + tzm1(ipoint)*zvec(ii)
  200          CONTINUE
c
               ii = 0
               call dudrst(dudr,duds,dudt,u(ieoff))
               DO 210 K=K1,K2,K3
               DO 210 J=J1,J2,J3
               DO 210 I=I1,I2,I3
                  II     = II+1
                  ipp    = IND(I,J,K,1)
                  IPOINT = IND(I,J,K,iel)
                  wkv1(ipoint) = (dudr(ipp)*drdn(ii)
     $                         +  duds(ipp)*dsdn(ii)
     $                         +  dudt(ipp)*dtdn(ii))/jacm1(ipoint)
c
  210          CONTINUE
               ii = 0
               call dudrst(dudr,duds,dudt,v(ieoff))
               DO 220 K=K1,K2,K3
               DO 220 J=J1,J2,J3
               DO 220 I=I1,I2,I3
                  II     = II+1
                  ipp    = IND(I,J,K,1)
                  IPOINT = IND(I,J,K,iel)
                  wkv2(ipoint) = (dudr(ipp)*drdn(ii)
     $                         +  duds(ipp)*dsdn(ii)
     $                         +  dudt(ipp)*dtdn(ii))/jacm1(ipoint)
  220          CONTINUE
c
               ii = 0
               call dudrst(dudr,duds,dudt,w(ieoff))
               DO 230 K=K1,K2,K3
               DO 230 J=J1,J2,J3
               DO 230 I=I1,I2,I3
                  II     = II+1
                  ipp    = IND(I,J,K,1)
                  IPOINT = IND(I,J,K,iel)
                  wkv3(ipoint) = (dudr(ipp)*drdn(ii)
     $                         +  duds(ipp)*dsdn(ii)
     $                         +  dudt(ipp)*dtdn(ii))/jacm1(ipoint)
  230          CONTINUE
            ENDIF
c
c           set scale
            DO 240 K=K1,K2,K3
            DO 240 J=J1,J2,J3
            DO 240 I=I1,I2,I3
               IPOINT=IND(I,J,K,iel)
               uvamp=wkv1(ipoint)**2+wkv2(ipoint)**2+wkv3(ipoint)**2
               uvmax=max(uvmax,uvamp)
  240       CONTINUE
c
c           write(6,*) 'this is iel',iel,rcen(iel),alpha,uvmax
 9000    CONTINUE
c        write(6,*) 'n2',nlstp,iel,ipln,iplane
         uvmax = sqrt(uvmax)
      ENDIF
      return
      end
c-----------------------------------------------------------------------
      subroutine vcross
     $           (x1,x2,x3,y1,y2,y3,z1,z2,z3,n)
      dimension x1(1),x2(1),x3(1)
      dimension y1(1),y2(1),y3(1)
      dimension z1(1),z2(1),z3(1)
c
      do 10 i=1,n
         X1(i) = Y2(i)*Z3(i) - Y3(i)*Z2(i)
         X2(i) = Y3(i)*Z1(i) - Y1(i)*Z3(i)
         X3(i) = Y1(i)*Z2(i) - Y2(i)*Z1(i)
   10 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine vnorms(x1,x2,x3,alpha,n)
      dimension x1(1),x2(1),x3(1)
c
      do 10 i=1,n
         vnrm=sqrt(x1(i)**2+x2(i)**2+x3(i)**2)
         if (vnrm.ne.0) vnrm=alpha/vnrm
         x1(i) = x1(i)*vnrm
         x2(i) = x2(i)*vnrm
         x3(i) = x3(i)*vnrm
   10 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine dudrst(dudr,duds,dudt,U)
      include 'basics.inc'
c
      dimension    u(nx,ny,nz),dudr(nx,ny,nz)
      dimension duds(nx,ny,nz),dudt(nx,ny,nz)
c
      nyz = ny*nz
      nxy = nx*ny
      CALL MXM(DGDR,NX,U,NX,dudr,NYZ)
      DO 10 IZ=1,NZ
         CALL MXM(U(1,1,iz),NX,DGDRT,NY,duds(1,1,iz),NY)
 10   CONTINUE
      if (if3d) CALL MXM (U,NXY,DGDRT,NZ,dudt,NZ)
      return
      end
c-----------------------------------------------------------------------
      subroutine mltiplt3
c     For creating 100 vtk files    pff 10/12/98
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp,iftmh
      common /c46/ ifopen46
      logical      ifopen46
c
      character*80  mv_command,rm_command
      character*112 xx_command
      character*56  xx_commnd1(2)
      equivalence  (xx_command,xx_commnd1)
c
      character quanty1*20,compon1*1,deriv1*5
C
      IFTMP=IFCLRB
      IFCLRB=.FALSE.
      nframe = 1
c     write(6,*) 'nstate:',nstate,nframe
c     call prs('Input number of frames following this one:$')
c     call rei(nframe)
c
      call prs  ('Input start and stop fld numbers:$')
      call reii (iframe0,iframe1)
c
      call prs  ('Input stride:$')
      call rei  (istride)
c
      call prs  ('Input resolution:$')
      call rei  (nppcell)
c
      call blank(xx_command,112)
      call blank(mv_command,80)
      call blank(rm_command,80)
c
      write(xx_commnd1(1),11)
      write(xx_commnd1(2),12)
   11 format('/quicksand/papka/surf -ipath vt1.vtk -start 0 -stop 0   ')
   12 format('-step 1 -ofile  tt1.vtk -contourvalue -0.9 -unstructured')
c             123456789 123456789 123456789 123456789 123456789 123456 
c
      write(rm_command,2)
    2 format('rm vt1.vtk')
c
      quanty1 = quanty
      deriv1  = deriv
      compon1 = compon
c
      kframe = 0
      do iframe = iframe0,iframe1,istride
         write(6,*) rm_command
         call system(rm_command)
c
         QUANTY= quanty1
         DERIV = deriv1
c        QUANTY= 'VORTEX'
c        QUANTY= 'PRESSURE'
c
c        Get .fld_iframe ....
         ndumps = iframe
         call getfld(ndumps,ierr,.true.,.true.)
         call setwrk(.true.)
c
         call vtk_out_xyz_sc(nppcell,.true.)
         close(unit=33)
c
         write(6,*) xx_command
         call system(xx_command)
c
         write(mv_command,3) kframe
         write(6,*) mv_command
         call system(mv_command)
    3    format('mv tt1.vtk  srf',i5.5,'.vtk')
c
         kframe = kframe+1
c
         if (if_auto_box) then
             close (unit=46)
             ifopen46 = .false.
         endif
c
      enddo
      CALL PRS('WAKE UP!!  The file creation is over!!$')
      IFHARD = IFTMH
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
      IFHARD=IFTMP
C
      return
      end
c-----------------------------------------------------------------------
      subroutine grab_window_pix(iname,l,iframe)
c
c     Dump subwindow and convert it to gif file
c
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      character*1 iname(1)
      character*1 fname(80)
c
      character*80 s80
      character*1  s81(80)
      equivalence (s81,s80)
c
c     logical if_img_convert,if_img_imcopy,if_img_tif,if_img_rgb
c
      if_img_convert = .true.
      if_img_imcopy  = .false.
      if_img_tif     = .false.
      if_img_rgb     = .false.
c
c
      kframe=iframe
      if (iframe.lt.0) then
         icalld = icalld + 1
         kframe = icalld
      endif
c
      call blank            (fname,80)
      call chcopy           (fname,iname,l)
      call terminate_string (fname,l)
c
      write(6,*) 'xpmn grw:',xpmn,ypmn,xpmx,ypmx
      call grab_window_raw(xpmn,ypmn,xpmx,ypmx,'raw\0')
c
      call blank(s80,80)    ! REMOVE existing .gif file
      write(s80,11) kframe
      write(6,*) s80
      call system(s80)
   11 format('rm -f movie',i5.5,'.gif')
c
c
c     Now convert raw to ppm to gif
c
      iw   = xpmx-xpmn
      ih   = ypmx-ypmn
c
      if (if_img_imcopy) then
         call blank(s80,80)
         write(s80,1) iw,ih
    1    format('rawtoppm ',2i6,' raw.rgb > raw.ppm')
         write(6,*) s80
         call system(s80)
c
         call blank(s80,80)
         write(s80,21) kframe
   21    format('imcopy -infile raw.ppm -outfile movie',i5.5,'.gif')
         write(6,*) s80
         call system(s80)
         if (if_img_tif) then    ! Save tif files too
            write(s80,31) kframe
   31       format('imcopy -infile raw.ppm -outfile movie',i5.5,'.tif')
            write(6,*) s80
            call system(s80)
         endif
c
      elseif (if_img_convert) then
         write(s80,22) iw,ih,kframe
   22    format('convert -depth 8 -size ',i4.4,'x',i4.4,
     $          ' raw.rgb movie',i5.5,'.gif')
         write(6,*) s80
         call system(s80)
c
         if (if_img_tif) then    ! Save tif files too
            write(s80,32) iw,ih,kframe
   32       format('convert -depth 8 -size ',i4.4,'x',i4.4,
     $          ' raw.rgb movie',i5.5,'.tif')
            write(s80,31) kframe
            write(6,*) s80
            call system(s80)
         endif
      endif
c
c
c     Save raw rgb for anil
c
      if (if_img_rgb) then
         call blank(s80,80)
         write(s80,4) iw,ih,kframe
    4    format('mv raw.rgb ',2i5.5,'m',i5.5,'.rgb')
         write(6,*) s80
         call system(s80)
      endif
c
      return
      end
C-----------------------------------------------------------------------
      subroutine auto_size_pixel_window
c
      include 'basics.inc'
c
      call quick_get_box(xmse1,ymse1,xmse2,ymse2)
c
      xscr1 = xscr(xmse1)
      yscr1 = yscr(ymse1)
      xscr2 = xscr(xmse2)
      yscr2 = yscr(ymse2)
c
      if (xscr1.lt.0.02 .or. xscr2.gt.0.98 .or.
     $    yscr1.lt.0.02 .or. yscr2.gt.0.98 ) then
          xscr1 = 0.02
          yscr1 = 0.02
          xscr2 = 0.98
          yscr2 = 0.98
      endif
c
      xpix1 = scoordx(xscr1)
      ypix1 = scoordy(yscr1)
      xpix2 = scoordx(xscr2)
      ypix2 = scoordy(yscr2)
c
      xpmn = min(xpix1,xpix2)
      xpmx = max(xpix1,xpix2)
      ypmn = min(ypix1,ypix2)
      ypmx = max(ypix1,ypix2)
      write(6,*) 'xp:',xpmn,xpmx,ypmn,ypmx
c
      i=xpmn
      xpmn=i
      j=xpmx
      xpmx=j
      k=ypmn
      ypmn=k
      l=ypmx
      ypmx=l
      write(6,*) 'xp:',xpmn,xpmx,ypmn,ypmx
c
      return
      end
c-----------------------------------------------------------------------
      subroutine quick_get_box(xmse1,ymse1,xmse2,ymse2)
c
c     This routine specifies a rectangular box based on object size
c
c     xmsei,ymsei  are returned in screen coordinates  (0,1)
c
      INCLUDE 'basics.inc'
C
      call quickscan_xy(x1,y1,x2,y2)
c  
c     Got two (x,y) pairs
c
      xmse1 = x1
      ymse1 = y1
      xmse2 = x2
      ymse2 = y2
c
c     Re-order xmsei and ymsei
c
      xmn = min(xmse1,xmse2)
      ymn = min(ymse1,ymse2)
      xmx = max(xmse1,xmse2)
      ymx = max(ymse1,ymse2)
c
      xmse1 = xmn
      ymse1 = ymn
      xmse2 = xmx
      ymse2 = ymx
c
c     draw a box around frame area
c
      call movec(xmn,ymn)
      call drawc(xmx,ymn)
      call drawc(xmx,ymx)
      call drawc(xmn,ymx)
      call drawc(xmn,ymn)
c
c
      write(line,70) xmse1,ymse1,xmse2,ymse2
   70 format
     $  ('Got (x1,y1)=(',f8.4,',',f8.4,') (x2,y2)=(',f8.4,',',f8.4,')$')
      call prs(line)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mltiplt4
c
c     For creating 100 vtk files    pff 10/12/98;  Updated 12/4/98 to
c     dump full vtk files
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp
c
      character*80  mv_command,rm_command
      character*112 xx_command
      character*56  xx_commnd1(2)
      equivalence  (xx_command,xx_commnd1)
c
      character quanty1*20,compon1*1,deriv1*5
C
      IFTMP=IFCLRB
      IFCLRB=.FALSE.
      nframe = 1
c     write(6,*) 'nstate:',nstate,nframe
c     call prs('Input number of frames following this one:$')
c     call rei(nframe)
c
      call prs  ('Input start and stop fld numbers:$')
      call reii (iframe0,iframe1)
c
      call prs  ('Input stride:$')
      call rei  (istride)
c
      call prs  ('Input resolution:$')
      call rei  (nppcell)
c
      call blank(xx_command,112)
      call blank(mv_command,80)
      call blank(rm_command,80)
c
      write(xx_commnd1(1),11)
      write(xx_commnd1(2),12)
   11 format('/quicksand/papka/surf -ipath vt1.vtk -start 0 -stop 0   ')
   12 format('-step 1 -ofile  tt1.vtk -contourvalue -0.9 -unstructured')
c  11 format('xxm -ifile vt -start 1 -stop 1 -step 1    ')
c  12 format('-ofile tt -contourvalue .750 -unstructured')
c  12 format('-ofile tt -contourvalue .500 -unstructured')
c  12 format('-ofile tt -contourvalue -1.3 -unstructured')
c             123456789 123456789 123456789 123456789 1
c
      write(rm_command,20)
   20 format('rm vt1.vtk')
c
      quanty1 = quanty
      deriv1  = deriv
      compon1 = compon
c
      kframe = 0
      do iframe = iframe0,iframe1,istride
         call system(rm_command)
c
c        QUANTY= quanty1
c        DERIV = deriv1
         DERIV = 'NO'
         QUANTY= 'VORTEX'
c        QUANTY= 'PRESSURE'
c
c        Get .fld_iframe ....
         ndumps = iframe
         call getfld(ndumps,ierr,.true.,.true.)
         call setwrk(.true.)
c
         call vtk_out_xyz_sc(nppcell,.true.)
         close(unit=33)
c
c        This is the command which runs Matt Szymanski's stripper
c        call system(xx_command)
c
c        This is the command which renames the file

c        if (kframe.lt.10000) write(mv_command,4) kframe
c        if (kframe.lt.1000)  write(mv_command,3) kframe
c        if (kframe.lt.100)   write(mv_command,2) kframe
c        if (kframe.lt.10)    write(mv_command,1) kframe

         if (iframe.lt.100000) write(mv_command,5) iframe
         if (iframe.lt.10000)  write(mv_command,4) iframe
         if (iframe.lt.1000)   write(mv_command,3) iframe
         if (iframe.lt.100)    write(mv_command,2) iframe
         if (iframe.lt.10)     write(mv_command,1) iframe

         call system(mv_command)
         write(6,*) mv_command

    1    format('mv vt1.vtk  srf',i1.1,'.vtk')
    2    format('mv vt1.vtk  srf',i2.2,'.vtk')
    3    format('mv vt1.vtk  srf',i3.3,'.vtk')
    4    format('mv vt1.vtk  srf',i4.4,'.vtk')
    5    format('mv vt1.vtk  srf',i5.5,'.vtk')
c
         kframe = kframe+1
      enddo
      CALL PRS('WAKE UP!!  The file creation is over!!$')
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
      IFHARD=IFTMP
C
      return
      end
c-----------------------------------------------------------------------
      subroutine particle_paths
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cpathr/ time_0,time_1,time_2,time_3
     $              , x_max,x_min,x_del
      common /cpatha/ v0x(lvrt),v1x(lvrt),v2x(lvrt)
     $              , v0y(lvrt),v1y(lvrt),v2y(lvrt) 
     $              , v0z(lvrt),v1z(lvrt),v2z(lvrt)
      common /cpathl/ if_periodic_x
      logical         if_periodic_x
c
      parameter (maxpart=500)
      common /cparta/ xpart  (3,maxpart)
      common /cparti/ iclr_part(maxpart)
c
      common /crk4pp/ f1(4),f2(4),f3(4),f4(4)
c
      integer icalld
      save    icalld
      data    icalld /0/
c
      real xnew(3)
c
c
      if_periodic_x = .true.
c
      ntot  = nx*ny*nz*nel
c
      if (icalld.eq.0) then
         icalld = 1
         x_min = glmin(xp,ntot)
         x_max = glmax(xp,ntot)
         x_del = x_max - x_min
      endif
c
      call get_part(ierr)
      if (ierr.eq.0) return
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
c
         call copy(v0x,v1x,ntot)
         call copy(v0y,v1y,ntot)
         call copy(v0z,v1z,ntot)
         time_0 = time_1
c
         call copy(v1x,v2x,ntot)
         call copy(v1y,v2y,ntot)
         call copy(v1z,v2z,ntot)
         time_1 = time_2
c
         call copy(v2x,u ,ntot)
         call copy(v2y,v ,ntot)
         call copy(v2z,w ,ntot)
         time_2 = time_3
c
         ndumps = ifld
         call getfld(ndumps,ierr,.true.,.true.)
         time_3 = time
c
         call prsii('after getfld$',ierr,iframe)
         call prsii('after getfld$',ntot,npart)
c
         if (iframe.ge.4) then
c
c           Advance particles from time1 to time2
c
            nstp = 50
            dtp = (time_2-time_1)/nstp
            do ipart=1,npart
c
c              color and start particle path
c
               call color    (iclr_part(ipart))
c
c
c              Integrate path from time1 to time2
c
               call out_xpos (xpart(1,ipart),0)
               do istp=1,nstp
c                 write(6,*) 'istep:',istp,timep,ipart
                  timep = time_1 + dtp*(istp-1)
                  call rk4pp     (xnew,xpart(1,ipart),timep,dtp,ndim)
c
c
c             ----DIAGNOSTICS----
c                 if (istp.eq.1) 
c    $               write(6,2) 'x',ipart,istp,iframe,timep
c    $               ,(f1(j),j=1,3),(xpart(j,ipart),j=1,3)
c                 if (istp.eq.1) 
c    $               write(9,2) 'x',ipart,istp,iframe,timep
c    $               ,(f1(j),j=1,3),(xpart(j,ipart),j=1,3)
c                 if (istp.eq.nstp) 
c    $               write(6,2) 'x',ipart,istp,iframe,timep
c    $               ,(f4(j),j=1,3),(xnew(j),j=1,3)
c                 if (istp.eq.nstp) 
c    $               write(9,2) 'x',ipart,istp,iframe,timep
c    $               ,(f4(j),j=1,3),(xnew(j),j=1,3)
c   2             format(a1,3i3,1p7e12.4)
c             ----DIAGNOSTICS----
c
c
                  call out_xpos  (xnew,istp)
c
c                 For periodic case, we reset x if it's outside the box
                  if (if_periodic_x) then
                     if (xnew(1).lt.x_min) then
                        xnew(1) = xnew(1) + x_del
                        call out_xpos  (xnew,0)
                     elseif (xnew(1).gt.x_max) then
                        xnew(1) = xnew(1) - x_del
                        call out_xpos  (xnew,0)
                     endif
                  endif
c
                  call copy      (xpart(1,ipart),xnew,ndim)
               enddo
            enddo
         endif
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_xpos(xpos,i)
      real xpos(3)
c
      xxis=xisom(xpos(1),xpos(2),xpos(3))
      yyis=yisom(xpos(1),xpos(2),xpos(3))
      if (i.eq.0) then
         call movec(xxis,yyis)
      else
         call drawc(xxis,yyis)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rk4pp(wnew,w,t,dt,n)
      real wnew(1),w(1),t,dt
      real wh(4)
c
      common /crk4pp/ f1(4),f2(4),f3(4),f4(4)
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
      call compute_fp(f1,w ,t )
      call add3s2    (wh,w,f1,dt2,n)
c
      call compute_fp(f2,wh,t2)
      call add3s2    (wh,w,f2,dt2,n)
c
      call compute_fp(f3,wh,t2)
      call add3s2    (wh,w,f3,dt ,n)
c
      call compute_fp(f4,wh,tt)
c
      call copy      (wnew,w,n)
      call add2s2    (wnew,f1,dt6,n)
      call add2s2    (wnew,f2,dt3,n)
      call add2s2    (wnew,f3,dt3,n)
      call add2s2    (wnew,f4,dt6,n)
c
c     write(9,1) 'f1:',(f1(j),j=1,3)
c     write(9,1) 'f2:',(f2(j),j=1,3)
c     write(9,1) 'f3:',(f3(j),j=1,3)
c     write(9,1) 'f4:',(f4(j),j=1,3)
c     write(9,1) 'dt:',dt3,dt6,dt,t
c     write(9,1) 'wo:',(w(j),j=1,n)
c     write(9,1) 'wn:',(wnew(j),j=1,n)
c   1 format(a3,1p4e14.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_fp(vel,xpos,t_str)
c
c     Compute RHS of ODE for particle path
c
c        INPUT:   xpos  -- particle position
c                 t_str -- time of interest
c
c        OUTPUT:  vel   -- velocity at point
c
c
c     The velocity field is known at time0,time1,time2, and time, with
c
c         time0  <  time1  =<  t_str  =<  time2  <  time
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cpathr/ time_0,time_1,time_2,time_3
     $              , x_max,x_min,x_del
      common /cpatha/ v0x(lvrt),v1x(lvrt),v2x(lvrt)
     $              , v0y(lvrt),v1y(lvrt),v2y(lvrt)
     $              , v0z(lvrt),v1z(lvrt),v2z(lvrt)
      common /cpathl/ if_periodic_x
      logical         if_periodic_x
c
      real vel(3),xpos(3),r(3),xpos_t(3)
      real vtmp(3,0:3),coef(0:3)
c
c     Check for possible translation, due to periodicity
c
      xpos_t(1) = xpos(1)
      xpos_t(2) = xpos(2)
      xpos_t(3) = xpos(3)
      if (if_periodic_x) then
         if (xpos_t(1).lt.x_min) then
            xpos_t(1) = xpos_t(1) + x_del
         elseif (xpos_t(1).gt.x_max) then
            xpos_t(1) = xpos_t(1) - x_del
         endif
      endif
c
      call get_coef(coef,t_str,time_0,time_1,time_2,time_3)
c
      call findre(ie,r,xpos_t,rminmax,ierr)
      nxyz  = nx*ny*nz
      ieoff = nxyz*(ie-1) + 1
c
      if (if3d) then
c
         call evalsc( vtmp(1,0) , v0x(ieoff) , r , 0 )
         call evalsc( vtmp(2,0) , v0y(ieoff) , r , 0 )
         call evalsc( vtmp(3,0) , v0z(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,1) , v1x(ieoff) , r , 0 )
         call evalsc( vtmp(2,1) , v1y(ieoff) , r , 0 )
         call evalsc( vtmp(3,1) , v1z(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,2) , v2x(ieoff) , r , 0 )
         call evalsc( vtmp(2,2) , v2y(ieoff) , r , 0 )
         call evalsc( vtmp(3,2) , v2z(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,3) , u  (ieoff) , r , 0 )
         call evalsc( vtmp(2,3) , v  (ieoff) , r , 0 )
         call evalsc( vtmp(3,3) , w  (ieoff) , r , 0 )
         do j=1,3
            vel(j) = 0.
            do i=0,3
               vel(j) = vel(j) + coef(i)*vtmp(j,i)
            enddo
         enddo
c
      else
c
         call evalsc( vtmp(1,0) , v0x(ieoff) , r , 0 )
         call evalsc( vtmp(2,0) , v0y(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,1) , v1x(ieoff) , r , 0 )
         call evalsc( vtmp(2,1) , v1y(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,2) , v2x(ieoff) , r , 0 )
         call evalsc( vtmp(2,2) , v2y(ieoff) , r , 0 )
c
         call evalsc( vtmp(1,3) , u  (ieoff) , r , 0 )
         call evalsc( vtmp(2,3) , v  (ieoff) , r , 0 )
c
         do j=1,2
            vel(j) = 0.
            do i=0,3
               vel(j) = vel(j) + coef(i)*vtmp(j,i)
            enddo
         enddo
c
c
      endif
c
c     write(9,1) 'vel ',(vel(j)   ,j=1,3),(xpos_t(j),j=1,3)
c     write(9,1) 'vel0',(vtmp(j,0),j=1,3),(r   (j),j=1,3)
c     write(9,1) 'vel1',(vtmp(j,1),j=1,3),time_2,time_3,t_str
c     write(9,1) 'vel2',(vtmp(j,2),j=1,3),(coef(j),j=0,1),time_0
c     write(9,1) 'vel3',(vtmp(j,3),j=1,3),(coef(j),j=2,3),time_1
c   1 format(a4,3(2x,3f12.7))
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_coef(coef,t,t0,t1,t2,t3)
c
c     Compute coefficients for lagrangian interpolat at time t
c     with knots at t0--t3.
c
      real coef(0:3),tk(0:3)
c
      tk(0) = t0
      tk(1) = t1
      tk(2) = t2
      tk(3) = t3
c
      do j=0,3
         coef(j) = 1.
         do i=0,3
            if (i.ne.j) coef(j) = coef(j)*(t-tk(i))/(tk(j)-tk(i))
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine get_part(isel)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (maxpart=500)
      common /cparta/ xpart  (3,maxpart)
      common /cparti/ iclr_part(maxpart)
c
c     if (if3d) then
c        call prs('Input format: (0) cancel (1) Keyboard (2) File?$')
c     else
c        call prs
c    $   ('Input format: (0) cancel (1) Keyboard (2) File (3) Mouse?$')
c     endif
c     call rei(isel)
c
      isel = 2
      if (isel.eq.2) then
         open (unit=68,file='part.path',err=991)
         npart = 0
         do i=1,maxpart
c           Note:  Colors should be between 2 and 15, (0=white,1=black)
            read(68,*,end=100) (xpart(j,i),j=1,ndim),iclr_part(i)
            iclr_part(i) = min(15,iclr_part(i))
            iclr_part(i) = max( 0,iclr_part(i))
            npart = npart+1
         enddo
  100    continue
         close(68)
         return
      endif
c
  991 continue
      call prs('Could not open file, "part.path"')
      isel=0
      return
      end
c-----------------------------------------------------------------------
      subroutine mltiplt_all
c
c     For creating 100 vtk files    pff 10/12/98;  Updated 12/4/98 to
c     dump full vtk files
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp
c
      character*80  mv_command,rm_command
      character*112 xx_command
      character*56  xx_commnd1(2)
      equivalence  (xx_command,xx_commnd1)
c
      character quanty1*20,compon1*1,deriv1*5
C
      IFTMP=IFCLRB
      IFCLRB=.FALSE.
      nframe = 1
c     write(6,*) 'nstate:',nstate,nframe
c     call prs('Input number of frames following this one:$')
c     call rei(nframe)
c
      call prs  ('Input start and stop fld numbers:$')
      call reii (iframe0,iframe1)
c
      call prs  ('Input stride:$')
      call rei  (istride)
c
      call prs  ('Input resolution:$')
      call rei  (nppcell)
c
      call blank(xx_command,112)
      call blank(mv_command,80)
      call blank(rm_command,80)
c
      write(xx_commnd1(1),11)
      write(xx_commnd1(2),12)
   11 format('/quicksand/papka/surf -ipath vt1.vtk -start 0 -stop 0   ')
   12 format('-step 1 -ofile  tt1.vtk -contourvalue -0.9 -unstructured')
c  11 format('xxm -ifile vt -start 1 -stop 1 -step 1    ')
c  12 format('-ofile tt -contourvalue .750 -unstructured')
c  12 format('-ofile tt -contourvalue .500 -unstructured')
c  12 format('-ofile tt -contourvalue -1.3 -unstructured')
c             123456789 123456789 123456789 123456789 1
c
      write(rm_command,20)
   20 format('rm vt1.vtk')
c
      quanty1 = quanty
      deriv1  = deriv
      compon1 = compon
c
      kframe = 0
      do iframe = iframe0,iframe1,istride
         call system(rm_command)
c
         QUANTY= quanty1
         DERIV = deriv1
c        QUANTY= 'VORTEX'
c        QUANTY= 'PRESSURE'
c
c        Get .fld_iframe ....
         ndumps = iframe
         call getfld(ndumps,ierr,.true.,.true.)
         call setwrk(.true.)
c
         call vtk_out_all(nppcell,.true.)
         close(unit=33)
c
c        This is the command which runs Matt Szymanski's stripper
c        call system(xx_command)
c
c        This is the command which renames the file
         write(mv_command,5) kframe
         if (kframe.lt.10000) write(mv_command,4) kframe
         if (kframe.lt.1000)  write(mv_command,3) kframe
         if (kframe.lt.100)   write(mv_command,2) kframe
         if (kframe.lt.10)    write(mv_command,1) kframe
         call system(mv_command)
    1    format('mv vt1.vtk  srf',i1.1,'.vtk')
    2    format('mv vt1.vtk  srf',i2.2,'.vtk')
    3    format('mv vt1.vtk  srf',i3.3,'.vtk')
    4    format('mv vt1.vtk  srf',i4.4,'.vtk')
    5    format('mv vt1.vtk  srf',i5.5,'.vtk')
c
         kframe = kframe+1
      enddo
      CALL PRS('WAKE UP!!  The file creation is over!!$')
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
      IFHARD=IFTMP
c
      return
      end
c-----------------------------------------------------------------------
      subroutine scrn_out
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifbigwindow
c
      ifbigwindow = .false.
c
      call prs('Input s for standard h for high res image.$')
      call res(ans,1)
      if (ans.eq.'h'.or.ans.eq.'H') ifbigwindow = .true.
c
      if (ifbigwindow) then
         iww = windoww
         iwh = windowh
         write(line,70) iww,iwh
   70    format('Curent window size',i5,',',i4,'):$')
         call prs(line)
         nww = 2550
         nwh = 2475
         call set_ww_wh(nww,nwh)
         call post_reset_window
c
c        Plot
c
         IF (PLFORM.NE.'MULTIPLOT') THEN
            IFMLTI=.FALSE.
            CALL HARD
            if (ifportrait) then
               CALL CLEAR
               CALL DRMESH
               CALL PLOTIT(1)
            else
               CALL PLOTIT(1)
            endif
            CALL NOHARD
         ELSE
            IFMLTI=.TRUE.
            CALL HARD
            CALL MLTIPLT
            CALL NOHARD
         ENDIF
      endif
c
c     Dump image
c
      call auto_size_pixel_window
      call grab_window_pix('frame',5,-1)
c
      if (ifbigwindow) then   ! Reset window
         call set_ww_wh(iww,iwh)
         call post_reset_window
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'basics.inc'
      include 'basicsp.inc'

      l   = 0
      rs3 = 0.
      rc3 = 0.
      ts3 = 0.
      tc3 = 0.
      zs3 = 0.
      zc3 = 0.
      vol = 0.

      do ie = 1,nel
      do iz = 1,nz
      do iy = 1,ny
      do ix = 1,nx
         l = l+1

         tt = atan2(yp(l),xp(l))
         st = sin(tt)
         ct = cos(tt)

         z3 = 3.*tt
         s3 = sin(z3)
         c3 = cos(z3)


         ur =  ct*u(l) + st*v(l)
         ut = -st*u(l) + ct*v(l)
         uz =                      w(l)


         wt  = wght(ix)*wght(iy)*wght(iz)*jacm1(l)

         rs3 = rs3 + ur*s3
         rc3 = rc3 + ur*c3

         ts3 = ts3 + ut*s3
         tc3 = tc3 + ut*c3

         zs3 = zs3 + uz*s3
         zc3 = zc3 + uz*c3

         vol = vol + wt

      enddo
      enddo
      enddo
      enddo

      rs3 = rs3/vol
      rc3 = rc3/vol
      ts3 = ts3/vol
      tc3 = tc3/vol
      zs3 = zs3/vol
      zc3 = zc3/vol

      write(6 ,1) time,rs3,rc3,ts3,tc3,zs3,zc3
      write(36,1) time,rs3,rc3,ts3,tc3,zs3,zc3
    1 format(1p7e12.4,' phs')

      return
      end
c-----------------------------------------------------------------------
