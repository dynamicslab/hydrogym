c-----------------------------------------------------------------------
      subroutine q_filter(wght)
c
c     filter vx,vy,vz, and p by simple interpolation
c
      include 'SIZE'
      include 'TOTAL'
c
c
c     These are the dimensions that we interpolate onto for v and p:
      parameter(lxv=lx1-1)
      parameter(lxp=lx2-1)
c
      real intdv(lx1,lx1)
      real intuv(lx1,lx1)
      real intdp(lx1,lx1)
      real intup(lx1,lx1)
      real intv(lx1,lx1)
      real intp(lx1,lx1)
c
      save intdv
      save intuv
      save intdp
      save intup
      save intv
      save intp

      common /ctmp0/ intw,intt
     $             , wk1,wk2
     $             , zgmv,wgtv,zgmp,wgtp,tmax(100),omax(103)

      real intw(lx1,lx1)
      real intt(lx1,lx1)
      real wk1  (lx1,lx1,lx1,lelt)
      real wk2  (lx1,lx1,lx1)
      real zgmv(lx1),wgtv(lx1),zgmp(lx1),wgtp(lx1)
c
c     outpost arrays
      parameter (lt=lx1*ly1*lz1*lelv)
      common /scruz/ w1(lt),w2(lt),w3(lt),wt(lt)

      character*18 sfmt

      integer icalld
      save    icalld
      data    icalld /0/

      logical if_fltv

      imax = nid
      imax = iglmax(imax,1)
      jmax = iglmax(imax,1)
      if (icalld.eq.0) then
         icalld = 1
         ncut = param(101)+1
         call build_new_filter(intv,zgm1,nx1,ncut,wght,nio)
      elseif (icalld.lt.0) then   ! old (std.) filter
         icalld = 1
         call zwgll(zgmv,wgtv,lxv)
         call igllm(intuv,intw,zgmv,zgm1,lxv,nx1,lxv,nx1)
         call igllm(intdv,intw,zgm1,zgmv,nx1,lxv,nx1,lxv)
c
         call zwgl (zgmp,wgtp,lxp)
         call iglm (intup,intw,zgmp,zgm2,lxp,nx2,lxp,nx2)
         call iglm (intdp,intw,zgm2,zgmp,nx2,lxp,nx2,lxp)
c
c        Multiply up and down interpolation into single operator
c
         call mxm(intup,nx2,intdp,lxp,intp,nx2)
         call mxm(intuv,nx1,intdv,lxv,intv,nx1)
c
c        Weight the filter to make it a smooth (as opposed to truncated)
c        decay in wave space

         w0 = 1.-wght
         call ident(intup,nx2)
         call add2sxy(intp,wght,intup,w0,nx2*nx2)

         call ident   (intuv,nx1)
         call add2sxy (intv ,wght,intuv,w0,nx1*nx1)

      endif

      ifldt  = ifield
c     ifield = 1

      if_fltv = .false.
      if ( ifflow .and. .not. ifmhd ) if_fltv = .true.
      if ( ifield.eq.1  .and. ifmhd ) if_fltv = .true.

c     Adam Peplinski; to take into account freezing of base flow
      if ( .not.ifbase             ) if_fltv = .false. ! base-flow frozen

      if ( if_fltv ) then
         call filterq(vx,intv,nx1,nz1,wk1,wk2,intt,if3d,umax)
         call filterq(vy,intv,nx1,nz1,wk1,wk2,intt,if3d,vmax)
         if (if3d)
     $      call filterq(vz,intv,nx1,nz1,wk1,wk2,intt,if3d,wmax)
         if (ifsplit.and..not.iflomach) 
     $      call filterq(pr,intv,nx1,nz1,wk1,wk2,intt,if3d,pmax)
      endif
c
      if (ifmhd.and.ifield.eq.ifldmhd) then
         call filterq(bx,intv,nx1,nz1,wk1,wk2,intt,if3d,umax)
         call filterq(by,intv,nx1,nz1,wk1,wk2,intt,if3d,vmax)
         if (if3d)
     $   call filterq(bz,intv,nx1,nz1,wk1,wk2,intt,if3d,wmax)
      endif
c
      if (ifpert) then
        do j=1,npert

         ifield = 1
         call filterq(vxp(1,j),intv,nx1,nz1,wk1,wk2,intt,if3d,umax)
         call filterq(vyp(1,j),intv,nx1,nz1,wk1,wk2,intt,if3d,vmax)
         if (if3d)
     $   call filterq(vzp(1,j),intv,nx1,nz1,wk1,wk2,intt,if3d,wmax)

         ifield = 2
         if (ifheat .and. .not.ifcvfld(ifield)) 
     $   call filterq(tp(1,j,1),intv,nx1,nz1,wk1,wk2,intt,if3d,wmax)

        enddo
      endif
c
      mmax = 0
      if (ifflow) then
c        pmax    = glmax(pmax,1)
         omax(1) = glmax(umax,1)
         omax(2) = glmax(vmax,1)
         omax(3) = glmax(wmax,1)
         mmax = ndim
      endif
         
c
      nfldt = 1+npscal
      if (ifheat .and. .not.ifcvode) then
         do ifld=1,nfldt
            ifield = ifld + 1
            call filterq(t(1,1,1,1,ifld),intv
     $                  ,nx1,nz1,wk1,wk2,intt,if3d,tmax(ifld))
            mmax = mmax+1
            omax(mmax) = glmax(tmax(ifld),1)
         enddo
      endif

      if (nio.eq.0) then
         if (npscal.eq.0) then
c           write(6,101) mmax
c           write(sfmt,101) mmax
c 101       format('''(i8,1p',i1,'e12.4,a6)''')
c           write(6,sfmt) istep,(omax(k),k=1,mmax),' qfilt'
c         write(6,'(i8,1p4e12.4,a6)') istep,(omax(k),k=1,mmax),' qfilt'
         else
            if (if3d) then
               write(6,1) istep,ifield,umax,vmax,wmax
            else
               write(6,1) istep,ifield,umax,vmax
            endif
    1       format(4x,i7,i3,' qfilt:',1p3e12.4)
            if(ifheat .and. .not.ifcvode) 
     &            write(6,'(1p50e12.4)') (tmax(k),k=1,nfldt)
         endif
      endif

      ifield = ifldt   ! RESTORE ifield


      return
      end
c-----------------------------------------------------------------------
      subroutine filterq(v,f,nx,nz,w1,w2,ft,if3d,dmax)
c
      include 'SIZE'
      include 'TSTEP'

      real v(nx*nx*nz,nelt),w1(1),w2(1)
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


      nel = nelfld(ifield)


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
            call sub3(w2,v(1,e),w1,nxyz)
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
      subroutine outmatx(a,m,n,io,name)
      real a(m*n)
      character*4 name
c
      open(unit=io,file=name)
      do i=1,m*n
         write(io,1) a(i)
      enddo
    1 format(1p1e22.13)
      close(unit=io)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine drag_calc(scale)
c
      INCLUDE 'SIZE'  
      INCLUDE 'TOTAL' 
c
      common /scrns/         pm1(lx1,ly1,lz1,lelv)
     $,vxx(lx1,ly1,lz1,lelv),vxy(lx1,ly1,lz1,lelv),vxz(lx1,ly1,lz1,lelv)
     $,vyx(lx1,ly1,lz1,lelv),vyy(lx1,ly1,lz1,lelv),vyz(lx1,ly1,lz1,lelv)
      common /scruz/ 
     $ vzx(lx1,ly1,lz1,lelv),vzy(lx1,ly1,lz1,lelv),vzz(lx1,ly1,lz1,lelv)
     $,one(lx1,ly1,lz1,lelv)
       real work(1)
       equivalence (work,one)
c
      common /cdrag/ dragx(0:maxobj),dragy(0:maxobj),dragz(0:maxobj)
     $             ,  momx(0:maxobj), momy(0:maxobj), momz(0:maxobj)
     $             ,  dpdx_mean,dpdy_mean,dpdz_mean
      real momx ,momy ,momz
c
      common /tdrag/ drag(11)
      real dragpx,dragpy,dragpz,dragvx,dragvy,dragvz
      real momvx ,momvy ,momvz
      real check1,check2
c
      equivalence (dragpx,drag(1)),(dragpy,drag(2)),(dragpz,drag(3))
      equivalence (dragvx,drag(4)),(dragvy,drag(5)),(dragvz,drag(6))
      equivalence (momvx ,drag(7)),(momvy ,drag(8)),(momvz ,drag(9)) 
      equivalence (check1,drag(10)),(check2,drag(11))

      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec
     $                , scale_vf(3)

      ntot1  = nx1*ny1*nz1*nelv

c    Map pressure onto mesh 1   (vxx and vyy are used as work arrays)
      call mappr(pm1,pr,vxx,vyy)
      call rone (one,ntot1)
c
c    Add mean_pressure_gradient.X to p:

      if (param(55).ne.0) then
         dpdx_mean = -scale_vf(1)
         dpdy_mean = -scale_vf(2)
         dpdz_mean = -scale_vf(3)
      endif

      call add2s2(pm1,xm1,dpdx_mean,ntot1)  ! Doesn't work if object is cut by 
      call add2s2(pm1,ym1,dpdy_mean,ntot1)  ! periodicboundary.  In this case,
      call add2s2(pm1,zm1,dpdz_mean,ntot1)  ! set ._mean=0 and compensate in
c                                           ! usrchk()  [ pff 10/21/04 ].

c    Compute du/dn
                CALL DUDXYZ (vxx,vx,RXM1,SXM1,TXM1,JACM1,1,1)
                CALL DUDXYZ (vxy,vx,RYM1,SYM1,TYM1,JACM1,1,1)
      if (if3d) CALL DUDXYZ (vxz,vx,RZM1,SZM1,TZM1,JACM1,1,1)
c
                CALL DUDXYZ (vyx,vy,RXM1,SXM1,TXM1,JACM1,1,1)
                CALL DUDXYZ (vyy,vy,RYM1,SYM1,TYM1,JACM1,1,1)
      if (if3d) CALL DUDXYZ (vyz,vy,RZM1,SZM1,TZM1,JACM1,1,1)
c
      if (if3d) then
                CALL DUDXYZ (vzx,vz,RXM1,SXM1,TXM1,JACM1,1,1)
                CALL DUDXYZ (vzy,vz,RYM1,SYM1,TYM1,JACM1,1,1)
                CALL DUDXYZ (vzz,vz,RZM1,SZM1,TZM1,JACM1,1,1)
      endif
c
c     Fill up viscous array w/ default
c
      if (istep.lt.1) call cfill(vdiff,param(2),ntot1)
c
      call col2(vxx,vdiff,ntot1)
      call col2(vxy,vdiff,ntot1)
      call col2(vxz,vdiff,ntot1)
      call col2(vyx,vdiff,ntot1)
      call col2(vyy,vdiff,ntot1)
      call col2(vyz,vdiff,ntot1)
      call col2(vzx,vdiff,ntot1)
      call col2(vzy,vdiff,ntot1)
      call col2(vzz,vdiff,ntot1)
c
      dragxt=0.0
      dragyt=0.0
      dragzt=0.0
c
      DO 100 II=1,NHIS
         IF (HCODE(10,II).NE.'I') GOTO 100
         IOBJ   = LOCHIS(1,II)
         MEMTOT = NMEMBER(IOBJ)
C
c
         IF (HCODE(1,II).NE.' ' .OR. HCODE(2,II).NE.' ' .OR.
     $       HCODE(3,II).NE.' ' ) THEN
            IFIELD = 1
c
c---------------------------------------------------------------------------
c           Compute drag for this object
c---------------------------------------------------------------------------
c
            dragvx=0.0
            dragvy=0.0
            dragvz=0.0
            dragpx=0.0
            dragpy=0.0
            dragpz=0.0
c
            momvx=0.0
            momvy=0.0
            momvz=0.0
c
            check1=0.0
            check2=0.0
            DO 50 MEM=1,MEMTOT
               ISK   = 0
               IEG   = OBJECT(IOBJ,MEM,1)
               IFC   = OBJECT(IOBJ,MEM,2)
               IF (GLLNID(IEG).EQ.NID) THEN
C                 This processor has a contribution
                  IE = GLLEL(IEG)
c
c                 Pressure drag
                  check1=check1+facint(one,one,area,ifc,ie)
                  check2=check2+facint(one,uny,area,ifc,ie)
c
                  dragpx=dragpx+facint(pm1,unx,area,ifc,ie)
                  dragpy=dragpy+facint(pm1,uny,area,ifc,ie)
                  if (if3d) dragpz=dragpz+facint(pm1,unz,area,ifc,ie)
c
c                 Viscous drag
                  if (if3d) then
                     dragvx=dragvx+facint(vxx,unx,area,ifc,ie)
     $                            +facint(vxy,uny,area,ifc,ie)
     $                            +facint(vxz,unz,area,ifc,ie)
     $                            +facint(vxx,unx,area,ifc,ie)
     $                            +facint(vyx,uny,area,ifc,ie)
     $                            +facint(vzx,unz,area,ifc,ie)
                     dragvy=dragvy+facint(vyx,unx,area,ifc,ie)
     $                            +facint(vyy,uny,area,ifc,ie)
     $                            +facint(vyz,unz,area,ifc,ie)
     $                            +facint(vxy,unx,area,ifc,ie)
     $                            +facint(vyy,uny,area,ifc,ie)
     $                            +facint(vzy,unz,area,ifc,ie)
                     dragvz=dragvz+facint(vzx,unx,area,ifc,ie)
     $                            +facint(vzy,uny,area,ifc,ie)
     $                            +facint(vzz,unz,area,ifc,ie)
     $                            +facint(vxz,unx,area,ifc,ie)
     $                            +facint(vyz,uny,area,ifc,ie)
     $                            +facint(vzz,unz,area,ifc,ie)
c
                     momvx=momvx-facint2(vxy,unx,unz,area,ifc,ie)
     $                        -facint2(vyx,unx,unz,area,ifc,ie)
     $                        -facint2(vyy,uny,unz,area,ifc,ie)
     $                        -facint2(vyy,uny,unz,area,ifc,ie)
     $                        -facint2(vzy,unz,unz,area,ifc,ie)
     $                        -facint2(vyz,unz,unz,area,ifc,ie)
     $                        +facint2(vxz,unx,uny,area,ifc,ie)
     $                        +facint2(vzx,unx,uny,area,ifc,ie)
     $                        +facint2(vyz,uny,uny,area,ifc,ie)
     $                        +facint2(vzy,uny,uny,area,ifc,ie)
     $                        +facint2(vzz,unz,uny,area,ifc,ie)
     $                        +facint2(vzz,unz,uny,area,ifc,ie)
                     momvy=momvy+facint2(vxx,unx,unz,area,ifc,ie)
     $                        +facint2(vxx,unx,unz,area,ifc,ie)
     $                        +facint2(vyx,uny,unz,area,ifc,ie)
     $                        +facint2(vxy,uny,unz,area,ifc,ie)
     $                        +facint2(vzx,unz,unz,area,ifc,ie)
     $                        +facint2(vxz,unz,unz,area,ifc,ie)
     $                        -facint2(vxz,unx,unx,area,ifc,ie)
     $                        -facint2(vzx,unx,unx,area,ifc,ie)
     $                        -facint2(vyz,uny,unx,area,ifc,ie)
     $                        -facint2(vzy,uny,unx,area,ifc,ie)
     $                        -facint2(vzz,unz,unx,area,ifc,ie)
     $                        -facint2(vzz,unz,unx,area,ifc,ie)
                     momvz=momvz-facint2(vxx,unx,uny,area,ifc,ie)
     $                        -facint2(vxx,unx,uny,area,ifc,ie)
     $                        -facint2(vyx,uny,uny,area,ifc,ie)
     $                        -facint2(vxy,uny,uny,area,ifc,ie)
     $                        -facint2(vzx,unz,uny,area,ifc,ie)
     $                        -facint2(vxz,unz,uny,area,ifc,ie)
     $                        +facint2(vxy,unx,unx,area,ifc,ie)
     $                        +facint2(vyx,unx,unx,area,ifc,ie)
     $                        +facint2(vyy,uny,unx,area,ifc,ie)
     $                        +facint2(vyy,uny,unx,area,ifc,ie)
     $                        +facint2(vzy,unz,unx,area,ifc,ie)
     $                        +facint2(vyz,unz,unx,area,ifc,ie) 
c
                  else
                     dragvx=dragvx+facint(vxx,unx,area,ifc,ie)
     $                            +facint(vxy,uny,area,ifc,ie)
                     dragvy=dragvy+facint(vyx,unx,area,ifc,ie)
     $                            +facint(vyy,uny,area,ifc,ie)
                  endif
               ENDIF
   50       CONTINUE
c
c          Sum contributions from all processors
            call gop(drag,work,'+  ',11)
            dragvx = -dragvx
            dragvy = -dragvy
            dragvz = -dragvz
         ENDIF
c
c        Scale by user specified scale factor (for convenience)
c
         dragvx = scale*dragvx
         dragvy = scale*dragvy
         dragvz = scale*dragvz
c
         dragpx = scale*dragpx
         dragpy = scale*dragpy
         dragpz = scale*dragpz
c
         dragx(iobj) = dragvx+dragpx
         dragy(iobj) = dragvy+dragpy
         dragz(iobj) = dragvz+dragpz
c
c
         momx(iobj)  = 0.5*momvx
         momy(iobj)  = 0.5*momvy
         momz(iobj)  = 0.5*momvz
c
         dragxt = dragxt + dragx(iobj)
         dragyt = dragyt + dragy(iobj)
         dragzt = dragzt + dragz(iobj)
c
         if (nio.eq.0.and.istep.eq.1) 
     $      write(6,*) 'drag_calc: scale=',scale
         if (nio.eq.0) then
            write(6,6) istep,time,dragx(iobj),dragpx,dragvx,'dragx',iobj
            write(6,6) istep,time,dragy(iobj),dragpy,dragvy,'dragy',iobj
            if (if3d) 
     $      write(6,6) istep,time,dragz(iobj),dragpz,dragvz,'dragz',iobj
c
c done by zly (3/17/03)
c          if(if3d) then
c             write(6,113) istep,time,momx,momy,momz
c          else
c             write(6,112) istep,time,momx,momy
c          endif
c         
         endif
    6    format(i8,1p4e15.7,2x,a5,i5)
  112    format(i6,1p3e15.7,'  momx')
  113    format(i6,1p4e15.7,'  momx')
         if (istep.lt.10.and.nio.eq.0) 
     $      write(6,9) 'check:',check1,check2,dpdx_mean,istep
    9    format(a6,1p3e16.8,i9)
c        if (time.gt.1.0.and.dragx.gt.10.) call emerxit
  100 continue
c
      if (nio.eq.0) then
         write(6,6) istep,time,dragxt,dragpx,dragvx,'drgxt',iobj
         write(6,6) istep,time,dragyt,dragpy,dragvy,'drgyt',iobj
         if (if3d) 
     $   write(6,6) istep,time,dragzt,dragpz,dragvz,'drgzt',iobj
      endif
c
      dragx(0) = dragxt
      dragy(0) = dragyt
      dragz(0) = dragzt
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mappr(pm1,pm2,pa,pb)
c
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      real pm1(lx1,ly1,lz1,lelv),pm2(lx2,ly2,lz2,lelv)
     $    ,pa (lx1,ly2,lz2)     ,pb (lx1,ly1,lz2)
c
C     Map the pressure onto the velocity mesh
C
      NGLOB1 = NX1*NY1*NZ1*NELV
      NYZ2   = NY2*NZ2
      NXY1   = NX1*NY1
      NXYZ   = NX1*NY1*NZ1
C
      IF (IFSPLIT) THEN
         CALL COPY(PM1,PM2,NGLOB1)
      ELSE
         DO 1000 IEL=1,NELV
            CALL MXM (IXM21,NX1,PM2(1,1,1,IEL),NX2,pa (1,1,1),NYZ2)
            DO 100 IZ=1,NZ2
               CALL MXM (PA(1,1,IZ),NX1,IYTM21,NY2,PB(1,1,IZ),NY1)
 100        CONTINUE
            CALL MXM (PB(1,1,1),NXY1,IZTM21,NZ2,PM1(1,1,1,IEL),NZ1)
 1000    CONTINUE

C     Average the pressure on elemental boundaries
       IFIELD=1
       CALL DSSUM (PM1,NX1,NY1,NZ1)
       CALL COL2  (PM1,VMULT,NGLOB1)

      ENDIF
C
C
      return
      end
c
c-----------------------------------------------------------------------
      function facint_a(a,area,f,e)
c     Integrate areal array a() on face f of element e.  27 June, 2012 pff

c     f  is in the preprocessor notation

      include 'SIZE'
      include 'TOPOL'
      real a(lx1,lz1,6,lelt),area(lx1,lz1,6,lelt)

      integer e,f

      sum=0.0
      do i=1,lx1*lz1
         sum = sum + a(i,1,f,e)*area(i,1,f,e)
      enddo

      facint_a = sum

      return
      end
c-----------------------------------------------------------------------
      function facint_v(a,area,f,e)
c     Integrate volumetric array a() on face f of element e

c        f  is in the preprocessor notation
c        fd  is the dssum notation.
c        27 June, 2012            PFF

      include 'SIZE'
      include 'TOPOL'
      real a(lx1,ly1,lz1,lelt),area(lx1,lz1,6,lelt)

      integer e,f,fd

      call dsset(nx1,ny1,nz1) ! set counters
      fd     = eface1(f)
      js1    = skpdat(1,fd)
      jf1    = skpdat(2,fd)
      jskip1 = skpdat(3,fd)
      js2    = skpdat(4,fd)
      jf2    = skpdat(5,fd)
      jskip2 = skpdat(6,fd)

      sum=0.0
      i = 0
      do 100 j2=js2,jf2,jskip2
      do 100 j1=js1,jf1,jskip1
         i = i+1
         sum = sum + a(j1,j2,1,e)*area(i,1,f,e)
  100 continue

      facint_v = sum

      return
      end
c-----------------------------------------------------------------------
      function facint(a,b,area,ifc,ie)
c
C
C     Take the dot product of A and B on the surface IFACE1 of element IE.
C
C         IFACE1 is in the preprocessor notation
C         IFACE  is the dssum notation.
C         5 Jan 1989 15:12:22      PFF
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
      DIMENSION A    (LX1,LY1,LZ1,lelv)
     $         ,B    (lx1,lz1,6,lelv)
     $         ,area (lx1,lz1,6,lelv)
C
C     Set up counters
C
      CALL DSSET(NX1,NY1,NZ1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      SUM=0.0
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         SUM = SUM + A(J1,J2,1,ie)*B(I,1,ifc,ie)*area(I,1,ifc,ie)
c        SUM = SUM + A(J1,J2,1,ie)*B(J1,J2,1,ie)*area(I,1,ifc,ie)
  100 CONTINUE
C
      facint = SUM
c
      return
      end
c-----------------------------------------------------------------------
      function facint2(a,b,c,area,ifc,ie)
      include 'SIZE'
      include 'TOPOL'
      dimension a    (lx1,ly1,lz1,lelv)
     $        , b    (lx1,lz1,6,lelv)
     $        , c    (lx1,lz1,6,lelv)
     $        , area (lx1,lz1,6,lelv) 
      call dsset(nx1,ny1,nz1)
      iface  = eface1(ifc)
      js1    = skpdat(1,iface)
      jf1    = skpdat(2,iface)
      jskip1 = skpdat(3,iface)
      js2    = skpdat(4,iface)
      jf2    = skpdat(5,iface)
      jskip2 = skpdat(6,iface)
      sum=0.0
      i=0
      do j2=js2,jf2,jskip2
      do j1=js1,jf1,jskip1
         i=i+1
         sum=sum+a(j1,j2,1,ie)*b(i,1,ifc,ie)*c(i,1,ifc,ie)
     $          *area(i,1,ifc,ie)
      end do
      end do 
      facint2=sum
      return
      end 
c-----------------------------------------------------------------------
      subroutine out_csrmats(acsr,ia,ja,n,name9)
      real    acsr(1)
      integer ia(1),ja(1)
c
      character*9 name9
      character*9 s(16)
c
      nnz = ia(n+1)-ia(1)
c
      write(6,1) name9,n,nnz
    1 format(/,'CSR Mat:',a9,3x,'n =',i5,3x,'nnz =',i5,/)
c
      n16 = min(n,16)
      n29 = min(n,29)
      do i=1,n29
         call blank(s,9*16)
         n1 = ia(i)
         n2 = ia(i+1)-1
         do jj=n1,n2
            j = ja  (jj)
            a = acsr(jj)
            if (a.ne.0..and.j.le.n16) write(s(j),9) a
         enddo
         write(6,16) (s(k),k=1,n16)
      enddo
    9 format(f9.4)
   16 format(16a9)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad3(ur,us,ut,u,N,e,D,Dt)
c     Output: ur,us,ut         Input:u,N,e,D,Dt
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real u (0:N,0:N,0:N,1)
      real D (0:N,0:N),Dt(0:N,0:N)
      integer e
c
      m1 = N+1
      m2 = m1*m1
c
      call mxm(D ,m1,u(0,0,0,e),m1,ur,m2)
      do k=0,N
         call mxm(u(0,0,k,e),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u(0,0,0,e),m2,Dt,m1,ut,m1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad2(ur,us,u,N,e,D,Dt)
c     Output: ur,us         Input:u,N,e,D,Dt
      real ur(0:N,0:N),us(0:N,0:N)
      real u (0:N,0:N,1)
      real D (0:N,0:N),Dt(0:N,0:N)
      integer e
c
      m1 = N+1
c
      call mxm(D ,m1,u(0,0,e),m1,ur,m1)
      call mxm(u(0,0,e),m1,Dt,m1,us,m1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gradm1(ux,uy,uz,u)
c
c     Compute gradient of T -- mesh 1 to mesh 1 (vel. to vel.)
c
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
c
      parameter (lxyz=lx1*ly1*lz1)
      real ux(lxyz,1),uy(lxyz,1),uz(lxyz,1),u(lxyz,1)

      common /ctmp1/ ur(lxyz),us(lxyz),ut(lxyz)

      integer e

      nxyz = nx1*ny1*nz1
      ntot = nxyz*nelt

      N = nx1-1
      do e=1,nelt
         if (if3d) then
            call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
            do i=1,lxyz
               ux(i,e) = jacmi(i,e)*(ur(i)*rxm1(i,1,1,e)
     $                             + us(i)*sxm1(i,1,1,e)
     $                             + ut(i)*txm1(i,1,1,e) )
               uy(i,e) = jacmi(i,e)*(ur(i)*rym1(i,1,1,e)
     $                             + us(i)*sym1(i,1,1,e)
     $                             + ut(i)*tym1(i,1,1,e) )
               uz(i,e) = jacmi(i,e)*(ur(i)*rzm1(i,1,1,e)
     $                             + us(i)*szm1(i,1,1,e)
     $                             + ut(i)*tzm1(i,1,1,e) )
            enddo
         else
            if (ifaxis) call setaxdy (ifrzer(e))
            call local_grad2(ur,us,u,N,e,dxm1,dytm1)
            do i=1,lxyz
               ux(i,e) =jacmi(i,e)*(ur(i)*rxm1(i,1,1,e)
     $                            + us(i)*sxm1(i,1,1,e) )
               uy(i,e) =jacmi(i,e)*(ur(i)*rym1(i,1,1,e)
     $                            + us(i)*sym1(i,1,1,e) )
            enddo
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine comp_vort3(vort,work1,work2,u,v,w)
c
      include 'SIZE'
      include 'TOTAL'
c
      parameter(lt=lx1*ly1*lz1*lelv)
      real vort(lt,3),work1(1),work2(1),u(1),v(1),w(1)
c
      ntot  = nx1*ny1*nz1*nelv
      if (if3d) then
c        work1=dw/dy ; work2=dv/dz
           call dudxyz(work1,w,rym1,sym1,tym1,jacm1,1,2)
           call dudxyz(work2,v,rzm1,szm1,tzm1,jacm1,1,3)
           call sub3(vort(1,1),work1,work2,ntot)
c        work1=du/dz ; work2=dw/dx
           call dudxyz(work1,u,rzm1,szm1,tzm1,jacm1,1,3)
           call dudxyz(work2,w,rxm1,sxm1,txm1,jacm1,1,1)
           call sub3(vort(1,2),work1,work2,ntot)
c        work1=dv/dx ; work2=du/dy
           call dudxyz(work1,v,rxm1,sxm1,txm1,jacm1,1,1)
           call dudxyz(work2,u,rym1,sym1,tym1,jacm1,1,2)
           call sub3(vort(1,3),work1,work2,ntot)
      else
c        work1=dv/dx ; work2=du/dy
           call dudxyz(work1,v,rxm1,sxm1,txm1,jacm1,1,1)
           call dudxyz(work2,u,rym1,sym1,tym1,jacm1,1,2)
           call sub3(vort,work1,work2,ntot)
      endif
c
c    Avg at bndry
c
      ifielt = ifield
      ifield = 1
      if (if3d) then
         do idim=1,ndim
            call col2  (vort(1,idim),bm1,ntot)
            call dssum (vort(1,idim),nx1,ny1,nz1)
            call col2  (vort(1,idim),binvm1,ntot)
         enddo
      else
         call col2  (vort,bm1,ntot)
         call dssum (vort,nx1,ny1,nz1)
         call col2  (vort,binvm1,ntot)
      endif
      ifield = ifielt
c
      return
      end
c-----------------------------------------------------------------------
      subroutine surface_int(sint,sarea,a,e,f)
C
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'TOPOL'
      real a(lx1,ly1,lz1,1)

      integer e,f

      call dsset(nx1,ny1,nz1)

      iface  = eface1(f)
      js1    = skpdat(1,iface)
      jf1    = skpdat(2,iface)
      jskip1 = skpdat(3,iface)
      js2    = skpdat(4,iface)
      jf2    = skpdat(5,iface)
      jskip2 = skpdat(6,iface)

      sarea = 0.
      sint  = 0.
      i     = 0

      do 100 j2=js2,jf2,jskip2
      do 100 j1=js1,jf1,jskip1
         i = i+1
         sarea = sarea+area(i,1,f,e)
         sint  = sint +area(i,1,f,e)*a(j1,j2,1,e)
  100 continue

      return
      end
c-----------------------------------------------------------------------
      subroutine surface_flux(dq,qx,qy,qz,e,f,w)

      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'PARALLEL'
      include 'TOPOL'
      parameter (l=lx1*ly1*lz1)

      real qx(l,1),qy(l,1),qz(l,1),w(lx1,ly1,lz1)
      integer e,f

      call           faccl3  (w,qx(1,e),unx(1,1,f,e),f)
      call           faddcl3 (w,qy(1,e),uny(1,1,f,e),f)
      if (if3d) call faddcl3 (w,qz(1,e),unz(1,1,f,e),f)

      call dsset(nx1,ny1,nz1)
      iface  = eface1(f)
      js1    = skpdat(1,iface)
      jf1    = skpdat(2,iface)
      jskip1 = skpdat(3,iface)
      js2    = skpdat(4,iface)
      jf2    = skpdat(5,iface)
      jskip2 = skpdat(6,iface)

      dq = 0
      i  = 0
      do 100 j2=js2,jf2,jskip2
      do 100 j1=js1,jf1,jskip1
         i = i+1
         dq    = dq + area(i,1,f,e)*w(j1,j2,1)
  100 continue

      return
      end
c-----------------------------------------------------------------------
      subroutine gaujordf(a,m,n,indr,indc,ipiv,ierr,rmult)
C
C     Gauss-Jordan matrix inversion with full pivoting
c
c     Num. Rec. p. 30, 2nd Ed., Fortran
c
C
C     a     is an m x n matrix
C     rmult is a  work array of dimension m
C
c
      real a(m,n),rmult(m)
      integer indr(m),indc(n),ipiv(n)

c     call outmat(a,m,n,'ab4',n)
c     do i=1,m
c        write(6,1) (a(i,j),j=1,n)
c     enddo
c 1   format('mat: ',1p6e12.4)

      ierr = 0
      eps = 1.e-9
      call izero(ipiv,m)

      do k=1,m
         amx=0.
         do i=1,m                    ! Pivot search
            if (ipiv(i).ne.1) then
               do j=1,m
                  if (ipiv(j).eq.0) then
                    if (abs(a(i,j)).ge.amx) then
                       amx = abs(a(i,j))
                       ir  = i
                       jc  = j
                    endif
                 elseif (ipiv(j).gt.1) then
                    ierr = -ipiv(j)
                    return
                 endif
              enddo
           endif
        enddo
        ipiv(jc) = ipiv(jc) + 1
c
c       Swap rows
        if (ir.ne.jc) then
           do j=1,n
              tmp     = a(ir,j)
              a(ir,j) = a(jc,j)
              a(jc,j) = tmp
           enddo
        endif
        indr(k)=ir
        indc(k)=jc
c       write(6 ,*) k,' Piv:',jc,a(jc,jc)
c       write(28,*) k,' Piv:',jc,a(jc,jc)
        if (abs(a(jc,jc)).lt.eps) then
           write(6 ,*) 'small Gauss Jordan Piv:',jc,a(jc,jc)
           write(28,*) 'small Gauss Jordan Piv:',jc,a(jc,jc)
           ierr = jc
           call exitt
           return
        endif
        piv = 1./a(jc,jc)
        a(jc,jc)=1.
        do j=1,n
           a(jc,j) = a(jc,j)*piv
        enddo
c
        do j=1,n
           work    = a(jc,j)
           a(jc,j) = a(1 ,j)
           a(1 ,j) = work
        enddo
        do i=2,m
           rmult(i) = a(i,jc)
           a(i,jc)  = 0.
        enddo
c
        do j=1,n
        do i=2,m
           a(i,j) = a(i,j) - rmult(i)*a(1,j)
        enddo
        enddo
c
        do j=1,n
           work    = a(jc,j)
           a(jc,j) = a(1 ,j)
           a(1 ,j) = work
        enddo
c
c       do i=1,m
c          if (i.ne.jc) then
c             rmult   = a(i,jc)
c             a(i,jc) = 0.
c             do j=1,n
c                a(i,j) = a(i,j) - rmult*a(jc,j)
c             enddo
c          endif
c       enddo
c
      enddo
c
c     Unscramble matrix
      do j=m,1,-1
         if (indr(j).ne.indc(j)) then
            do i=1,m
               tmp=a(i,indr(j))
               a(i,indr(j))=a(i,indc(j))
               a(i,indc(j))=tmp
            enddo
         endif
      enddo
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
      subroutine build_new_filter(intv,zpts,nx,kut,wght,nio)
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
      real intv(nx,nx),zpts(nx)
c
      parameter (lm=40)
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
      call mxm  (diag,nx,pht,nx,intv,nx)      !          -1
      call mxm  (phi ,nx,intv,nx,pht,nx)      !     V D V
      call copy (intv,pht,nx*nx)
c
      do k=1,nx*nx
         pht(k) = 1.-diag(k)
      enddo
      np1 = nx+1
      if (nio.eq.0) then
         write(6,6) 'filt amp',(pht (k),k=1,nx*nx,np1)
         write(6,6) 'filt trn',(diag(k),k=1,nx*nx,np1)
   6     format(a8,16f7.4,6(/,8x,16f7.4))
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_all
c
c     This routine computes running averages E(X),E(X^2),E(X*Y)
c     and outputs to avg*.fld*, rms*.fld*, and rm2*.fld* for all
c     fields.
c
c     E denotes the expected value operator and X,Y two
c     real valued random variables.
c
c     variances and covariances can be computed in a post-processing step:
c
c        var(X)   := E(X^X) - E(X)*E(X) 
c        cov(X,Y) := E(X*Y) - E(X)*E(Y)  
c
c     Note: The E-operator is linear, in the sense that the expected
c           value is given by E(X) = 1/N * sum[ E(X)_i ], where E(X)_i
c           is the expected value of the sub-ensemble i (i=1...N).
c
      include 'SIZE'  
      include 'TOTAL' 
      include 'AVG'

      logical ifverbose
      integer icalld
      save    icalld
      data    icalld  /0/

      if (ax1.ne.lx1 .or. ay1.ne.ly1 .or. az1.ne.lz1) then
         if(nid.eq.0) write(6,*)
     $     'ABORT: wrong size of ax1,ay1,az1 in avg_all(), check SIZE!'
         call exitt
      endif
      if (ax2.ne.lx2 .or. ay2.ne.ay2 .or. az2.ne.lz2) then
         if(nid.eq.0) write(6,*)
     $     'ABORT: wrong size of ax2,ay2,az2 in avg_all(), check SIZE!'
         call exitt
      endif

      ntot  = nx1*ny1*nz1*nelv
      ntott = nx1*ny1*nz1*nelt
      nto2  = nx2*ny2*nz2*nelv

      ! initialization
      if (icalld.eq.0) then
         icalld = icalld + 1
         atime  = 0.
         timel  = time

         call rzero(uavg,ntot)
         call rzero(vavg,ntot)
         call rzero(wavg,ntot)
         call rzero(pavg,nto2)
         do i = 1,ldimt
            call rzero(tavg(1,1,1,1,i),ntott)
         enddo

         call rzero(urms,ntot)
         call rzero(vrms,ntot)
         call rzero(wrms,ntot)
         call rzero(prms,nto2)
         do i = 1,ldimt
            call rzero(trms(1,1,1,1,i),ntott)
         enddo

         call rzero(vwms,ntot)
         call rzero(wums,ntot)
         call rzero(uvms,ntot)
      endif

      dtime = time  - timel
      atime = atime + dtime

      ! dump freq
      iastep = param(68)
      if  (iastep.eq.0) iastep=param(15)   ! same as iostep
      if  (iastep.eq.0) iastep=500

      ifverbose=.false.
      if (istep.le.10) ifverbose=.true.
      if  (mod(istep,iastep).eq.0) ifverbose=.true.

      if (atime.ne.0..and.dtime.ne.0.) then
         if(nio.eq.0) write(6,*) 'Compute statistics ...'
         beta  = dtime/atime
         alpha = 1.-beta
         ! compute averages E(X)
         call avg1    (uavg,vx,alpha,beta,ntot ,'um  ',ifverbose)
         call avg1    (vavg,vy,alpha,beta,ntot ,'vm  ',ifverbose)
         call avg1    (wavg,vz,alpha,beta,ntot ,'wm  ',ifverbose)
         call avg1    (pavg,pr,alpha,beta,nto2 ,'prm ',ifverbose)
         call avg1    (tavg,t ,alpha,beta,ntott,'tm  ',ifverbose)
         do i = 2,ldimt
            call avg1 (tavg(1,1,1,1,i),t(1,1,1,1,i),alpha,beta,
     &                 ntott,'psav',ifverbose)
         enddo

         ! compute averages E(X^2) 
         call avg2    (urms,vx,alpha,beta,ntot ,'ums ',ifverbose)
         call avg2    (vrms,vy,alpha,beta,ntot ,'vms ',ifverbose)
         call avg2    (wrms,vz,alpha,beta,ntot ,'wms ',ifverbose)
         call avg2    (prms,pr,alpha,beta,nto2 ,'prms',ifverbose)
         call avg2    (trms,t ,alpha,beta,ntott,'tms ',ifverbose)
         do i = 2,ldimt
            call avg2 (trms(1,1,1,1,i),t(1,1,1,1,i),alpha,beta,
     &                 ntott,'psms',ifverbose)
         enddo

         ! compute averages E(X*Y) (for now just for the velocities)
         call avg3    (uvms,vx,vy,alpha,beta,ntot,'uvm ',ifverbose)
         call avg3    (vwms,vy,vz,alpha,beta,ntot,'vwm ',ifverbose)
         call avg3    (wums,vz,vx,alpha,beta,ntot,'wum ',ifverbose)
      endif
c
c-----------------------------------------------------------------------
      if ( (mod(istep,iastep).eq.0.and.istep.gt.1) .or.lastep.eq.1) then

         time_temp = time
         time      = atime   ! Output the duration of this avg

         call outpost2(uavg,vavg,wavg,pavg,tavg,ldimt,'avg')
         call outpost2(urms,vrms,wrms,prms,trms,ldimt,'rms')
         call outpost (uvms,vwms,wums,prms,trms,      'rm2')

         atime = 0.
         time  = time_temp  ! Restore clock

      endif
c
      timel = time
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg1(avg,f,alpha,beta,n,name,ifverbose)
      include 'SIZE'
      include 'TSTEP'
c
      real avg(n),f(n)
      character*4 name
      logical ifverbose
c
      do k=1,n
         avg(k) = alpha*avg(k) + beta*f(k)
      enddo
c
      if (ifverbose) then
         avgmax = glmax(avg,n)
         avgmin = glmin(avg,n)
         if (nio.eq.0) write(6,1) istep,time,avgmin,avgmax
     $                           ,alpha,beta,name
    1    format(i9,1p5e13.5,1x,a4,' av1mnx')
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg2(avg,f,alpha,beta,n,name,ifverbose)
      include 'SIZE'
      include 'TSTEP'
c
      real avg(n),f(n)
      character*4 name
      logical ifverbose
c
      do k=1,n
         avg(k) = alpha*avg(k) + beta*f(k)*f(k)
      enddo
c
      if (ifverbose) then
         avgmax = glmax(avg,n)
         avgmin = glmin(avg,n)
         if (nio.eq.0) write(6,1) istep,time,avgmin,avgmax
     $                           ,alpha,beta,name
    1    format(i9,1p5e13.5,1x,a4,' av2mnx')
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg3(avg,f,g,alpha,beta,n,name,ifverbose)
      include 'SIZE'
      include 'TSTEP'
c
      real avg(n),f(n),g(n)
      character*4 name
      logical ifverbose
c
      do k=1,n
         avg(k) = alpha*avg(k) + beta*f(k)*g(k)
      enddo
c
      if (ifverbose) then
         avgmax = glmax(avg,n)
         avgmin = glmin(avg,n)
         if (nio.eq.0) write(6,1) istep,time,avgmin,avgmax
     $                           ,alpha,beta,name
    1    format(i9,1p5e13.5,1x,a4,' av3mnx')
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine build_legend_transform(Lj,Ljt,zpts,nx)
c
      real Lj(nx*nx),Ljt(nx*nx),zpts(nx)
c
      parameter (lm=90)
      integer   indr(lm),indc(lm),ipiv(lm)
c
      if (nx.gt.lm) then
         write(6,*) 'ABORT in build_legend_transform:',nx,lm
         call exitt
      endif
c
      j = 1
      n = nx-1
      do i=1,nx
         z = zpts(i)
         call legendre_poly(Lj(j),z,n)  ! Return Lk(z), k=0,...,n
         j = j+nx
      enddo
      call transpose1(Lj,nx)
c     call outmat(Lj,nx,nx,'Lj ',n)
c     call exitt
      call gaujordf  (Lj,nx,nx,indr,indc,ipiv,ierr,rmult)
      call transpose (Ljt,nx,Lj,nx)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine local_err_est(err,u,nx,Lj,Ljt,uh,w,if3d)
c
c     Local error estimates for u_e
c
      include 'SIZE'
      real err(5,2),u(1),uh(nx,nx,nx),w(1),Lj(1),Ljt(1)
      logical if3d
c
      call rzero(err,10)
c
      nxyz = nx**ndim
      utot = vlsc2(u,u,nxyz)
      if (utot.eq.0) return
c
      call tensr3(uh,nx,u,nx,Lj,Ljt,Ljt,w)    !  Go to Legendre space
c
c
c     Get energy in low modes
c
      m = nx-2
c
      if (if3d) then
         amp2_l = 0.
         do k=1,m
         do j=1,m
         do i=1,m
            amp2_l = amp2_l + uh(i,j,k)**2
         enddo
         enddo
         enddo
c
c        Energy in each spatial direction
c        
         amp2_t = 0
         do k=m+1,nx
         do j=1,m
         do i=1,m
            amp2_t = amp2_t + uh(i,j,k)**2
         enddo
         enddo
         enddo
c        
         amp2_s = 0
         do k=1,m
         do j=m+1,nx
         do i=1,m
            amp2_s = amp2_s + uh(i,j,k)**2
         enddo
         enddo
         enddo
c        
         amp2_r = 0
         do k=1,m
         do j=1,m
         do i=m+1,nx
            amp2_r = amp2_r + uh(i,j,k)**2
         enddo
         enddo
         enddo
c
         amp2_h = 0
         do k=m+1,nx
         do j=m+1,nx
         do i=m+1,nx
            amp2_h = amp2_h + uh(i,j,k)**2
         enddo
         enddo
         enddo
c
         etot = amp2_l + amp2_r + amp2_s + amp2_t + amp2_h
c
         relr = amp2_r / (amp2_r + amp2_l)
         rels = amp2_s / (amp2_s + amp2_l)
         relt = amp2_t / (amp2_t + amp2_l)
         relh = (amp2_r + amp2_s + amp2_t + amp2_h) / etot
c
      else
         k = 1
         amp2_l = 0.
         do j=1,m
         do i=1,m
            amp2_l = amp2_l + uh(i,j,k)**2
         enddo
         enddo
         if (amp2_l.eq.0) return
c
c        Energy in each spatial direction
c        
         amp2_t = 0
c        
         amp2_s = 0
         do j=m+1,nx
         do i=1,m
            amp2_s = amp2_s + uh(i,j,k)**2
         enddo
         enddo
c        
         amp2_r = 0
         do j=1,m
         do i=m+1,nx
            amp2_r = amp2_r + uh(i,j,k)**2
         enddo
         enddo
c
         amp2_h = 0
         do j=m+1,nx
         do i=m+1,nx
            amp2_h = amp2_h + uh(i,j,k)**2
         enddo
         enddo
c
         etot = amp2_l + amp2_r + amp2_s + amp2_h
c
         relr = amp2_r / (amp2_r + amp2_l)
         rels = amp2_s / (amp2_s + amp2_l)
         relt = 0
         relh = (amp2_r + amp2_s + amp2_h) / etot
c
      endif
c
      err (1,1) = sqrt(relr)
      err (2,1) = sqrt(rels)
      if (if3d) err (3,1) = sqrt(relt)
      err (4,1) = sqrt(relh)
      err (5,1) = sqrt(etot)
c
      err (1,2) = sqrt(amp2_r)
      err (2,2) = sqrt(amp2_s)
      if (if3d) err (3,2) = sqrt(amp2_t)
      err (4,2) = sqrt(amp2_h)
      err (5,2) = sqrt(utot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_exyz(ex,ey,ez,eg,nelx,nely,nelz)
      integer ex,ey,ez,eg
c
      nelxy = nelx*nely
c
      ez = 1 +  (eg-1)/nelxy
      ey = mod1 (eg,nelxy)
      ey = 1 +  (ey-1)/nelx
      ex = mod1 (eg,nelx)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dump_header2d(excode,nx,ny,nlx,nly,ierr)

      include 'SIZE'
      include 'TOTAL'

      character*2 excode(15)

      real*4         test_pattern

      character*1 fhdfle1(132)
      character*132 fhdfle
      equivalence (fhdfle,fhdfle1)

      jstep = istep
      if (jstep.gt.10000) jstep = jstep / 10
      if (jstep.gt.10000) jstep = jstep / 10
      if (jstep.gt.10000) jstep = jstep / 10
      if (jstep.gt.10000) jstep = jstep / 10
      if (jstep.gt.10000) jstep = jstep / 10

      nlxy = nlx*nly
      nzz  = 1

c     write(6,'(4i4,1PE14.7,i5,1x,15a2,1x,a12)')
c    $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15),
c    $  'NELT,NX,NY,N'
c
      p66 = 0.
      ierr= 0
      IF (p66.EQ.1.0) THEN
C       unformatted i/o
        WRITE(24)
     $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15)
      ELSEIF (p66.EQ.3.0) THEN
C       formatted i/o to header file
        WRITE(27,'(4I4,1pe14.7,I5,1X,15A2,1X,A12)')
     $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15),
     $  'NELT,NX,NY,N'
      ELSEIF (p66.eq.4.0) THEN
C       formatted i/o to header file
        WRITE(fhdfle,'(4I4,1pe14.7,I5,1X,15A2,1X,A12)')
     $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15),
     $  ' 4 NELT,NX,NY,N'
        call byte_write(fhdfle,20,ierr)
      ELSEIF (p66.eq.5.0) THEN
C       formatted i/o to header file
        WRITE(fhdfle,'(4I4,1pe14.7,I5,1X,15A2,1X,A12)')
     $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15),
     $  ' 8 NELT,NX,NY,N'
        call byte_write(fhdfle,20,ierr)
      ELSE
C       formatted i/o
        WRITE(24,'(4I4,1pe14.7,I5,1X,15A2,1X,A12)')
     $  nlxy,nx,ny,nzz,TIME,jstep,(excode(i),i=1,15),
     $  'NELT,NX,NY,N'
      ENDIF
C     cdrror is a dummy cerror value for now.
      CDRROR=0.0
      IF (p66.EQ.1.0) THEN
C       unformatted i/o
        WRITE(24)(CDRROR,I=1,nlxy)
      ELSEIF (p66.eq.3. .or. p66.eq.4.0) then
C       write byte-ordering test pattern to byte file...
        test_pattern = 6.54321
        call byte_write(test_pattern,1,ierr)
      ELSEIF (p66.eq.5.) then
        test_pattern8 = 6.54321
        call byte_write(test_pattern8,2,ierr)
      ELSE
C       formatted i/o
        WRITE(24,'(6G11.4)')(CDRROR,I=1,nlxy)
      ENDIF
 
      return
      end
c-----------------------------------------------------------------------
      subroutine outfld2d_p(u,v,w,nx,ny,nlx,nly,name,ifld,jid,npido,ir)

      include 'SIZE'
      include 'TOTAL'

      integer icalld
      save    icalld
      data    icalld /0/

      real u(nx*ny*nlx*nly)
      real v(nx*ny*nlx*nly)
      real w(nx*ny*nlx*nly)
      character*4 name

      character*2  excode(15)
      character*12 fm
      character*20 outfile

      character*1 slash,dot
      save        slash,dot
      data        slash,dot  / '/' , '.' /

      icalld = icalld+1

      call blank(excode,30)
      excode(4) = 'U '
      excode(5) = '  '
      excode(6) = 'T '
      nthings   =  3
      ir = 0 !error code for dump_header2d

      call blank(outfile,20)
      if (npido.lt.100) then
         if (ifld.lt.100) then
            write(outfile,22) jid,slash,name,ifld
   22       format('B',i2.2,a1,a4,'.fld',i2.2)
         elseif (ifld.lt.1000) then
            write(outfile,23) jid,slash,name,ifld
   23       format('B',i2.2,a1,a4,'.fld',i3)
         elseif (ifld.lt.10000) then
            write(outfile,24) jid,slash,name,ifld
   24       format('B',i2.2,a1,a4,'.fld',i4)
         elseif (ifld.lt.100000) then
            write(outfile,25) jid,slash,name,ifld
   25       format('B',i2.2,a1,a4,'.fld',i5)
         elseif (ifld.lt.1000000) then
            write(outfile,26) jid,slash,name,ifld
   26       format('B',i2.2,a1,a4,'.fld',i6)
         endif
      else
         if (ifld.lt.100) then
            write(outfile,32) jid,slash,name,ifld
   32       format('B',i3.3,a1,a4,'.fld',i2.2)
         elseif (ifld.lt.1000) then
            write(outfile,33) jid,slash,name,ifld
   33       format('B',i3.3,a1,a4,'.fld',i3)
         elseif (ifld.lt.10000) then
            write(outfile,34) jid,slash,name,ifld
   34       format('B',i3.3,a1,a4,'.fld',i4)
         elseif (ifld.lt.100000) then
            write(outfile,35) jid,slash,name,ifld
   35       format('B',i3.3,a1,a4,'.fld',i5)
         elseif (ifld.lt.1000000) then
            write(outfile,36) jid,slash,name,ifld
   36       format('B',i3.3,a1,a4,'.fld',i6)
         endif
      endif

      if (icalld.le.4) write(6,*) nid,outfile,' OPEN',nlx,nly
      open(unit=24,file=outfile,status='unknown')
      call dump_header2d(excode,nx,ny,nlx,nly,ir)

      n = nx*ny*nlx*nly
      write(fm,10) nthings
      write(24,fm) (u(i),v(i),w(i),i=1,n)
   10 format('(1p',i1,'e14.6)')
      close(24)

      return
      end
c-----------------------------------------------------------------------
      subroutine outfld2d(u,v,w,nx,ny,nlx,nly,name,ifld)

      include 'SIZE'
      include 'TOTAL'

      real u(nx*ny*nlx*nly)
      real v(nx*ny*nlx*nly)
      real w(nx*ny*nlx*nly)
      character*3 name

      character*2  excode(15)
      character*12 fm
      character*20 outfile

c     if (istep.le.10) write(6,*) nid,' in out2d:',iz

      call blank(excode,30)
c
c     excode(1) = 'X '
c     excode(2) = 'Y '
c     excode(3) = 'U '
c     excode(4) = 'V '
c     excode(5) = 'P '
c     excode(6) = 'T '
c
      excode(4) = 'U '
      excode(5) = '  '
      excode(6) = 'T '
      nthings   =  3

      ierr = 0 
      if (nid.eq.0) then
         call blank(outfile,20)
         if (ifld.lt.100) then
            write(outfile,2) name,ifld
    2       format(a3,'2d.fld',i2.2)
         elseif (ifld.lt.1000) then
            write(outfile,3) name,ifld
    3       format(a3,'2d.fld',i3)
         elseif (ifld.lt.10000) then
            write(outfile,4) name,ifld
    4       format(a3,'2d.fld',i4)
         elseif (ifld.lt.100000) then
            write(outfile,5) name,ifld
    5       format(a3,'2d.fld',i5)
         elseif (ifld.lt.1000000) then
            write(outfile,6) name,ifld
    6       format(a3,'2d.fld',i6)
         endif
         open(unit=24,file=outfile,status='unknown')
         call dump_header2d(excode,nx,ny,nlx,nly,ierr)

         n = nx*ny*nlx*nly
         write(fm,10) nthings
c        write(6,*) fm
c        call exitt
         write(24,fm) (u(i),v(i),w(i),i=1,n)
   10    format('(1p',i1,'e14.6)')
c  10    format('''(1p',i1,'e15.7)''')
c  10    format(1p7e15.7)
c
         close(24)
      endif
      call err_chk(ierr,'Error using byte_write,outfld2d. Abort.$')

      return
      end
c-----------------------------------------------------------------------
      subroutine planar_average_z(ua,u,w1,w2)
c
c     Compute r-s planar average of quantity u()
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'
c
      real ua(nz1,nelz),u(nx1*ny1,nz1,nelv),w1(nz1,nelz),w2(nz1,nelz)
      integer e,eg,ez
c
      melxy = nelx*nely
c
      nz = nz1*nelz
      call rzero(ua,nz)
      call rzero(w1,nz)
c
      do e=1,nelt
c
         eg = lglel(e)
         ez = 1 + (eg-1)/melxy
c
         do k=1,nz1
         do i=1,nx1*ny1
            zz = (1.-zgm1(k,3))/2.  ! = 1 for k=1, = 0 for k=nz1
            aa = zz*area(i,1,5,e) + (1-zz)*area(i,1,6,e)  ! wgtd jacobian
            w1(k,ez) = w1(k,ez) + aa
            ua(k,ez) = ua(k,ez) + aa*u(i,k,e)
         enddo
         enddo
      enddo
c
      call gop(ua,w2,'+  ',nz)
      call gop(w1,w2,'+  ',nz)
c
      do i=1,nz
         ua(i,1) = ua(i,1) / w1(i,1)   ! Normalize
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine drgtrq(dgtq,xm0,ym0,zm0,sij,pm1,visc,f,e)
c
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TOPOL'
      INCLUDE 'TSTEP'
c
      real dgtq(3,4)
      real xm0 (lx1,ly1,lz1,lelt)
      real ym0 (lx1,ly1,lz1,lelt)
      real zm0 (lx1,ly1,lz1,lelt)
      real sij (lx1,ly1,lz1,3*ldim-3,lelv)
      real pm1 (lx1,ly1,lz1,lelv)
      real visc(lx1,ly1,lz1,lelv)
c
      real dg(3,2)
c
      integer f,e,pf
      real    n1,n2,n3
c
      call dsset(nx1,ny1,nz1)    ! set up counters
      pf     = eface1(f)         ! convert from preproc. notation
      js1    = skpdat(1,pf)
      jf1    = skpdat(2,pf)
      jskip1 = skpdat(3,pf)
      js2    = skpdat(4,pf)
      jf2    = skpdat(5,pf)
      jskip2 = skpdat(6,pf)
C
      call rzero(dgtq,12)
c
      if (if3d.or.ifaxis) then
       i = 0
       a = 0
       do j2=js2,jf2,jskip2
       do j1=js1,jf1,jskip1
         i = i+1
         n1 = unx(i,1,f,e)*area(i,1,f,e)
         n2 = uny(i,1,f,e)*area(i,1,f,e)
         n3 = unz(i,1,f,e)*area(i,1,f,e)
         a  = a +          area(i,1,f,e)
c
         v  = visc(j1,j2,1,e)
c
         s11 = sij(j1,j2,1,1,e)
         s21 = sij(j1,j2,1,4,e)
         s31 = sij(j1,j2,1,6,e)
c
         s12 = sij(j1,j2,1,4,e)
         s22 = sij(j1,j2,1,2,e)
         s32 = sij(j1,j2,1,5,e)
c
         s13 = sij(j1,j2,1,6,e)
         s23 = sij(j1,j2,1,5,e)
         s33 = sij(j1,j2,1,3,e)
c
         dg(1,1) = pm1(j1,j2,1,e)*n1     ! pressure drag
         dg(2,1) = pm1(j1,j2,1,e)*n2
         dg(3,1) = pm1(j1,j2,1,e)*n3
c
         dg(1,2) = -v*(s11*n1 + s12*n2 + s13*n3) ! viscous drag
         dg(2,2) = -v*(s21*n1 + s22*n2 + s23*n3)
         dg(3,2) = -v*(s31*n1 + s32*n2 + s33*n3)
c
         r1 = xm0(j1,j2,1,e)
         r2 = ym0(j1,j2,1,e)
         r3 = zm0(j1,j2,1,e)
c
         do l=1,2
         do k=1,3
            dgtq(k,l) = dgtq(k,l) + dg(k,l)
         enddo
         enddo
c
         dgtq(1,3) = dgtq(1,3) + (r2*dg(3,1)-r3*dg(2,1)) ! pressure
         dgtq(2,3) = dgtq(2,3) + (r3*dg(1,1)-r1*dg(3,1)) ! torque
         dgtq(3,3) = dgtq(3,3) + (r1*dg(2,1)-r2*dg(1,1))
c
         dgtq(1,4) = dgtq(1,4) + (r2*dg(3,2)-r3*dg(2,2)) ! viscous
         dgtq(2,4) = dgtq(2,4) + (r3*dg(1,2)-r1*dg(3,2)) ! torque
         dgtq(3,4) = dgtq(3,4) + (r1*dg(2,2)-r2*dg(1,2))
       enddo
       enddo

      else ! 2D

       i = 0
       a = 0
       do j2=js2,jf2,jskip2
       do j1=js1,jf1,jskip1
         i = i+1
         n1 = unx(i,1,f,e)*area(i,1,f,e)
         n2 = uny(i,1,f,e)*area(i,1,f,e)
         a  = a +          area(i,1,f,e)
         v  = visc(j1,j2,1,e)

         s11 = sij(j1,j2,1,1,e)
         s12 = sij(j1,j2,1,3,e)
         s21 = sij(j1,j2,1,3,e)
         s22 = sij(j1,j2,1,2,e)

         dg(1,1) = pm1(j1,j2,1,e)*n1     ! pressure drag
         dg(2,1) = pm1(j1,j2,1,e)*n2
         dg(3,1) = 0

         dg(1,2) = -v*(s11*n1 + s12*n2) ! viscous drag
         dg(2,2) = -v*(s21*n1 + s22*n2)
         dg(3,2) = 0.

         r1 = xm0(j1,j2,1,e)
         r2 = ym0(j1,j2,1,e)
         r3 = 0.

         do l=1,2
         do k=1,3
            dgtq(k,l) = dgtq(k,l) + dg(k,l)
         enddo
         enddo

         dgtq(1,3) = 0! dgtq(1,3) + (r2*dg(3,1)-r3*dg(2,1)) ! pressure
         dgtq(2,3) = 0! dgtq(2,3) + (r3*dg(1,1)-r1*dg(3,1)) ! torque
         dgtq(3,3) = dgtq(3,3) + (r1*dg(2,1)-r2*dg(1,1))

         dgtq(1,4) = 0! dgtq(1,4) + (r2*dg(3,2)-r3*dg(2,2)) ! viscous
         dgtq(2,4) = 0! dgtq(2,4) + (r3*dg(1,2)-r1*dg(3,2)) ! torque
         dgtq(3,4) = dgtq(3,4) + (r1*dg(2,2)-r2*dg(1,2))
       enddo
       enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine torque_calc(scale,x0,ifdout,iftout)
c
c     Compute torque about point x0
c
c     Scale is a user-supplied multiplier so that results may be
c     scaled to any convenient non-dimensionalization.
c
c
      INCLUDE 'SIZE'  
      INCLUDE 'TOTAL' 

      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec
     $                , scale_vf(3)

c
      real x0(3),w1(0:maxobj)
      logical ifdout,iftout
c
      common /scrns/         sij (lx1*ly1*lz1*6*lelv)
      common /scrcg/         pm1 (lx1,ly1,lz1,lelv)
      common /scrsf/         xm0(lx1,ly1,lz1,lelt)
     $,                      ym0(lx1,ly1,lz1,lelt)
     $,                      zm0(lx1,ly1,lz1,lelt)
c
      parameter (lr=lx1*ly1*lz1)
      common /scruz/         ur(lr),us(lr),ut(lr)
     $                     , vr(lr),vs(lr),vt(lr)
     $                     , wr(lr),ws(lr),wt(lr)
c
      common /ctorq/ dragx(0:maxobj),dragpx(0:maxobj),dragvx(0:maxobj)
     $             , dragy(0:maxobj),dragpy(0:maxobj),dragvy(0:maxobj)
     $             , dragz(0:maxobj),dragpz(0:maxobj),dragvz(0:maxobj)
c
     $             , torqx(0:maxobj),torqpx(0:maxobj),torqvx(0:maxobj)
     $             , torqy(0:maxobj),torqpy(0:maxobj),torqvy(0:maxobj)
     $             , torqz(0:maxobj),torqpz(0:maxobj),torqvz(0:maxobj)
c
     $             , dpdx_mean,dpdy_mean,dpdz_mean
     $             , dgtq(3,4)
c
c
      n = nx1*ny1*nz1*nelv
c
      call mappr(pm1,pr,xm0,ym0) ! map pressure onto Mesh 1
c
c    Add mean_pressure_gradient.X to p:

      if (param(55).ne.0) then
         dpdx_mean = -scale_vf(1)
         dpdy_mean = -scale_vf(2)
         dpdz_mean = -scale_vf(3)
      endif

      call add2s2(pm1,xm1,dpdx_mean,n)  ! Doesn't work if object is cut by 
      call add2s2(pm1,ym1,dpdy_mean,n)  ! periodicboundary.  In this case,
      call add2s2(pm1,zm1,dpdz_mean,n)  ! set ._mean=0 and compensate in
c
c    Compute sij
c
      nij = 3
      if (if3d.or.ifaxis) nij=6
      call comp_sij(sij,nij,vx,vy,vz,ur,us,ut,vr,vs,vt,wr,ws,wt)
c
c
c     Fill up viscous array w/ default
c
      if (istep.lt.1) call cfill(vdiff,param(2),n)
c
      call cadd2(xm0,xm1,-x0(1),n)
      call cadd2(ym0,ym1,-x0(2),n)
      call cadd2(zm0,zm1,-x0(3),n)
c
      x1min=glmin(xm0(1,1,1,1),n)
      x2min=glmin(ym0(1,1,1,1),n)
      x3min=glmin(zm0(1,1,1,1),n)
c
      x1max=glmax(xm0(1,1,1,1),n)
      x2max=glmax(ym0(1,1,1,1),n)
      x3max=glmax(zm0(1,1,1,1),n)
c
      do i=0,maxobj
         dragpx(i) = 0   ! BIG CODE  :}
         dragvx(i) = 0
         dragx (i) = 0
         dragpy(i) = 0
         dragvy(i) = 0
         dragy (i) = 0
         dragpz(i) = 0
         dragvz(i) = 0
         dragz (i) = 0
         torqpx(i) = 0
         torqvx(i) = 0
         torqx (i) = 0
         torqpy(i) = 0
         torqvy(i) = 0
         torqy (i) = 0
         torqpz(i) = 0
         torqvz(i) = 0
         torqz (i) = 0
      enddo
c
c
      nobj = 0
      do ii=1,nhis
        if (hcode(10,ii).EQ.'I') then
          iobj   = lochis(1,ii)
          memtot = nmember(iobj)
          nobj   = max(iobj,nobj)
c
          if (hcode(1,ii).ne.' ' .or. hcode(2,ii).ne.' ' .or.
     $      hcode(3,ii).ne.' ' ) then
            ifield = 1
c
c           Compute drag for this object
c
            do mem=1,memtot
               ieg   = object(iobj,mem,1)
               ifc   = object(iobj,mem,2)
               if (gllnid(ieg).eq.nid) then ! this processor has a contribution
                  ie = gllel(ieg)
                  call drgtrq(dgtq,xm0,ym0,zm0,sij,pm1,vdiff,ifc,ie)
c
                  call cmult(dgtq,scale,12)
c
                  dragpx(iobj) = dragpx(iobj) + dgtq(1,1)  ! pressure 
                  dragpy(iobj) = dragpy(iobj) + dgtq(2,1)
                  dragpz(iobj) = dragpz(iobj) + dgtq(3,1)
c
                  dragvx(iobj) = dragvx(iobj) + dgtq(1,2)  ! viscous
                  dragvy(iobj) = dragvy(iobj) + dgtq(2,2)
                  dragvz(iobj) = dragvz(iobj) + dgtq(3,2)
c
                  torqpx(iobj) = torqpx(iobj) + dgtq(1,3)  ! pressure 
                  torqpy(iobj) = torqpy(iobj) + dgtq(2,3)
                  torqpz(iobj) = torqpz(iobj) + dgtq(3,3)
c
                  torqvx(iobj) = torqvx(iobj) + dgtq(1,4)  ! viscous
                  torqvy(iobj) = torqvy(iobj) + dgtq(2,4)
                  torqvz(iobj) = torqvz(iobj) + dgtq(3,4)
c
               endif
            enddo
          endif
        endif
      enddo
c
c     Sum contributions from all processors
c
      call gop(dragpx,w1,'+  ',maxobj+1)
      call gop(dragpy,w1,'+  ',maxobj+1)
      call gop(dragpz,w1,'+  ',maxobj+1)
      call gop(dragvx,w1,'+  ',maxobj+1)
      call gop(dragvy,w1,'+  ',maxobj+1)
      call gop(dragvz,w1,'+  ',maxobj+1)
c
      call gop(torqpx,w1,'+  ',maxobj+1)
      call gop(torqpy,w1,'+  ',maxobj+1)
      call gop(torqpz,w1,'+  ',maxobj+1)
      call gop(torqvx,w1,'+  ',maxobj+1)
      call gop(torqvy,w1,'+  ',maxobj+1)
      call gop(torqvz,w1,'+  ',maxobj+1)
c
      nobj = iglmax(nobj,1)
c
      do i=1,nobj
         dragx(i) = dragpx(i) + dragvx(i)
         dragy(i) = dragpy(i) + dragvy(i)
         dragz(i) = dragpz(i) + dragvz(i)
c
         torqx(i) = torqpx(i) + torqvx(i)
         torqy(i) = torqpy(i) + torqvy(i)
         torqz(i) = torqpz(i) + torqvz(i)
c
         dragpx(0) = dragpx (0) + dragpx (i)
         dragvx(0) = dragvx (0) + dragvx (i)
         dragx (0) = dragx  (0) + dragx  (i)
c
         dragpy(0) = dragpy (0) + dragpy (i)
         dragvy(0) = dragvy (0) + dragvy (i)
         dragy (0) = dragy  (0) + dragy  (i)
c
         dragpz(0) = dragpz (0) + dragpz (i)
         dragvz(0) = dragvz (0) + dragvz (i)
         dragz (0) = dragz  (0) + dragz  (i)
c
         torqpx(0) = torqpx (0) + torqpx (i)
         torqvx(0) = torqvx (0) + torqvx (i)
         torqx (0) = torqx  (0) + torqx  (i)
c
         torqpy(0) = torqpy (0) + torqpy (i)
         torqvy(0) = torqvy (0) + torqvy (i)
         torqy (0) = torqy  (0) + torqy  (i)
c
         torqpz(0) = torqpz (0) + torqpz (i)
         torqvz(0) = torqvz (0) + torqvz (i)
         torqz (0) = torqz  (0) + torqz  (i)
c
      enddo
c
      i0 = 0
      if (nobj.le.1) i0 = 1  ! one output for single-object case
c
      do i=i0,nobj
        if (nio.eq.0) then
          if (if3d.or.ifaxis) then
           if (ifdout) then
            write(6,6) istep,time,dragx(i),dragpx(i),dragvx(i),i,'dragx'
            write(6,6) istep,time,dragy(i),dragpy(i),dragvy(i),i,'dragy'
            write(6,6) istep,time,dragz(i),dragpz(i),dragvz(i),i,'dragz'
           endif
           if (iftout) then
            write(6,6) istep,time,torqx(i),torqpx(i),torqvx(i),i,'torqx'
            write(6,6) istep,time,torqy(i),torqpy(i),torqvy(i),i,'torqy'
            write(6,6) istep,time,torqz(i),torqpz(i),torqvz(i),i,'torqz'
           endif
          else
           if (ifdout) then
            write(6,6) istep,time,dragx(i),dragpx(i),dragvx(i),i,'dragx'
            write(6,6) istep,time,dragy(i),dragpy(i),dragvy(i),i,'dragy'
           endif
           if (iftout) then
            write(6,6) istep,time,torqz(i),torqpz(i),torqvz(i),i,'torqz'
           endif
          endif
        endif
    6   format(i8,1p4e19.11,2x,i5,a5)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine comp_sij(sij,nij,u,v,w,ur,us,ut,vr,vs,vt,wr,ws,wt)
c                                       du_i       du_j
c     Compute the stress tensor S_ij := ----   +   ----
c                                       du_j       du_i
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e
c
      real sij(lx1*ly1*lz1,nij,lelv)
      real u  (lx1*ly1*lz1,lelv)
      real v  (lx1*ly1*lz1,lelv)
      real w  (lx1*ly1*lz1,lelv)
      real ur (1) , us (1) , ut (1)
     $   , vr (1) , vs (1) , vt (1)
     $   , wr (1) , ws (1) , wt (1)

      real j ! Inverse Jacobian

      n    = nx1-1      ! Polynomial degree
      nxyz = nx1*ny1*nz1

      if (if3d) then     ! 3D CASE
       do e=1,nelv
        call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
        call local_grad3(vr,vs,vt,v,N,e,dxm1,dxtm1)
        call local_grad3(wr,ws,wt,w,N,e,dxm1,dxtm1)

        do i=1,nxyz

         j = jacmi(i,e)

         sij(i,1,e) = j*  ! du/dx + du/dx
     $   2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e)+ut(i)*txm1(i,1,1,e))

         sij(i,2,e) = j*  ! dv/dy + dv/dy
     $   2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e)+vt(i)*tym1(i,1,1,e))

         sij(i,3,e) = j*  ! dw/dz + dw/dz
     $   2*(wr(i)*rzm1(i,1,1,e)+ws(i)*szm1(i,1,1,e)+wt(i)*tzm1(i,1,1,e))

         sij(i,4,e) = j*  ! du/dy + dv/dx
     $   (ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e)+ut(i)*tym1(i,1,1,e) +
     $    vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e)+vt(i)*txm1(i,1,1,e) )

         sij(i,5,e) = j*  ! dv/dz + dw/dy
     $   (wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e)+wt(i)*tym1(i,1,1,e) +
     $    vr(i)*rzm1(i,1,1,e)+vs(i)*szm1(i,1,1,e)+vt(i)*tzm1(i,1,1,e) )

         sij(i,6,e) = j*  ! du/dz + dw/dx
     $   (ur(i)*rzm1(i,1,1,e)+us(i)*szm1(i,1,1,e)+ut(i)*tzm1(i,1,1,e) +
     $    wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e)+wt(i)*txm1(i,1,1,e) )

        enddo
       enddo

      elseif (ifaxis) then  ! AXISYMMETRIC CASE  

c
c        Notation:                       ( 2  x  Acheson, p. 353)
c                     Cylindrical
c            Nek5k    Coordinates
c
c              x          z
c              y          r
c              z          theta
c

         do e=1,nelv
            call setaxdy ( ifrzer(e) )  ! change dytm1 if on-axis
            call local_grad2(ur,us,u,N,e,dxm1,dytm1)
            call local_grad2(vr,vs,v,N,e,dxm1,dytm1)
            call local_grad2(wr,ws,w,N,e,dxm1,dytm1)

            do i=1,nxyz
               j = jacmi(i,e)
               r = ym1(i,1,1,e)                              ! Cyl. Coord:

               sij(i,1,e) = j*  ! du/dx + du/dx              ! e_zz
     $           2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e))

               sij(i,2,e) = j*  ! dv/dy + dv/dy              ! e_rr
     $           2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))

               if (r.gt.0) then                              ! e_@@
                  sij(i,3,e) = v(i,e)/r  ! v / r 
               else
                  sij(i,3,e) = j*  ! L'Hopital's rule: e_@@ = dv/dr
     $            2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))
               endif

               sij(i,4,e) = j*  ! du/dy + dv/dx             ! e_zr
     $            ( ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) +
     $              vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) )

               if (r.gt.0) then                             ! e_r@
                  sij(i,5,e) = j*  ! dw/dy 
     $              ( wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e) )
     $              - w(i,e) / r
               else
                  sij(i,5,e) = 0
               endif

               sij(i,6,e) = j*  ! dw/dx                     ! e_@z
     $            ( wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e) )

            enddo
         enddo

      else              ! 2D CASE

         do e=1,nelv
            call local_grad2(ur,us,u,N,e,dxm1,dxtm1)
            call local_grad2(vr,vs,v,N,e,dxm1,dxtm1)

            do i=1,nxyz
               j = jacmi(i,e)

               sij(i,1,e) = j*  ! du/dx + du/dx
     $           2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e))

               sij(i,2,e) = j*  ! dv/dy + dv/dy
     $           2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))

               sij(i,3,e) = j*  ! du/dy + dv/dx
     $           (ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) +
     $            vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) )

            enddo
         enddo
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine auto_averager(fname127) ! simple average of files

c     This routine reads files specificed of file.list and averages
c     them with uniform weight
c
c     Note that it relies on scravg and scruz common blocks. pff 12/7/14
c

      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      character*127 fname127
      character*1   f1(127)

      parameter (lt=lx1*ly1*lz1*lelt)
      common /scravg/ ua(lt),va(lt),wa(lt),pa(lt)
      common /scrsf/  ta(lt,ldimt)

      character*1 s1(127)
      equivalence (s1,initc) ! equivalence to initial condition

      if (nid.eq.0) then
         ib=indx1(fname127,' ',1)-1
         call chcopy(f1,fname127,ib)
         write(6,2) (f1(k),k=1,ib)
    2    format('Open file: ',127a1)
      endif

      ierr = 0
      if (nid.eq.0) open(77,file=fname127,status='old',err=199)
      ierr = iglmax(ierr,1)
      if (ierr.gt.0) goto 199
      n = nx1*ny1*nz1*nelt
      n2= nx2*ny2*nz2*nelt

      call rzero (ua,n)
      call rzero (va,n)
      call rzero (wa,n)
      call rzero (pa,n2)
      do k=1,npscal+1
         call rzero (ta(1,k),n)
      enddo

      icount = 0
      do ipass=1,9999999

         call blank(initc,127)
         initc(1) = 'done '
         if (nid.eq.0) read(77,127,end=998) initc(1)
  998    call bcast(initc,127)
  127    format(a127)

         iblank = indx1(initc,' ',1)-1
         if (nio.eq.0) write(6,1) ipass,(s1(k),k=1,iblank)
    1    format(i8,'Reading: ',127a1)

         if (indx1(initc,'done ',5).eq.0) then ! We're not done

            nfiles = 1
            call restart(nfiles)  ! Note -- time is reset.

            call opadd2 (ua,va,wa,vx,vy,vz)
            call add2   (pa,pr,n2)
            do k=1,npscal+1
               call add2(ta(1,k),t(1,1,1,1,k),n)
            enddo
            icount = icount+1

         else
            goto 999
         endif

      enddo

  999 continue  ! clean up averages
      if (nid.eq.0) close(77)

      scale = 1./icount
      call cmult2(vx,ua,scale,n)
      call cmult2(vy,va,scale,n)
      call cmult2(vz,wa,scale,n)
      call cmult2(pr,pa,scale,n2)
      do k=1,npscal+1
         call cmult2(t(1,1,1,1,k),ta(1,k),scale,n)
      enddo
      return

  199 continue ! exception handle for file not found
      ierr = 1
      if (nid.eq.0) ierr = iglmax(ierr,1)
      call exitti('Auto averager did not find list file.$',ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine x_average(ua,u,w1,w2)
c
c     Compute the x average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real ua(ny1,nz1,nely,nelz),u (nx1,ny1,nz1,nelv)
     $    ,w1(ny1,nz1,nely,nelz),w2(ny1,nz1,nely,nelz)
      integer e,eg,ex,ey,ez
      real dy2

      nelyz = nely*nelz
      if (nelyz.gt.lely*lelz) call exitti
     $  ('ABORT IN x_average. Increase lely*lelz in SIZE:$',nelyz)

      myz = nely*nelz*ny1*nz1
      call rzero(ua,myz)
      call rzero(w1,myz)

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do j=1,ny1
            do i=1,nx1
               dx2 = 1.0  !  Assuming uniform element size in "x" direction
               ua(j,k,ey,ez) = ua(j,k,ey,ez)+dx2*wzm1(i)*u(i,j,k,e)
               w1(j,k,ey,ez) = w1(j,k,ey,ez)+dx2*wzm1(i) ! redundant but clear
            enddo
         enddo
         enddo
      enddo

      call gop(ua,w2,'+  ',myz)
      call gop(w1,w2,'+  ',myz)

      do i=1,myz
         ua(i,1,1,1) = ua(i,1,1,1) / w1(i,1,1,1)   ! Normalize
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine x_average_transpose(u,ua) ! distribute ua to each z-plane

      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real u(nx1,ny1,nz1,nelv),ua(ny1,nz1,nely,nelz)

      integer e,eg,ex,ey,ez


      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do j=1,ny1
            do i=1,nx1
               u(i,j,k,e) = ua(j,k,ey,ez)
            enddo
         enddo
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine x_distribute(u)
c
c     Compute the x average of quantity u() and redistribute
c
c     Assumes you have nelx*nely elements, in the same order,
c     within each x plane
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      real u(1)

      parameter (lyavg = ly1*lz1*lely*lelz)
      common /scravg/ ua(lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)

      call x_average          (ua,u,w1,w2)
      call x_average_transpose(u,ua) ! distribute ua to each z-plane

      return
      end
c-----------------------------------------------------------------------
      subroutine x_distribute2(ua,u)
c
c     Compute the x average of quantity u() and redistribute
c
c     Assumes you have nelx*nely elements, in the same order,
c     within each x plane
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      real ua(1),u(1)

      parameter (lyavg = ly1*lz1*lely*lelz)
      common /scravg/ w1(lyavg)
     $              , w2(lyavg)

      call x_average          (ua,u,w1,w2)
      call x_average_transpose(u,ua) ! distribute ua to each z-plane

      return
      end
c-----------------------------------------------------------------------
      subroutine y_slice (ua,u,w1,w2)
c
c     Extract a y slice of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'
c
      real ua(nx1,nz1,nelx,nelz),u (nx1,ny1,nz1,nelv)
     $    ,w1(nx1,nz1,nelx,nelz),w2(nx1,nz1,nelx,nelz)
      integer e,eg,ex,ey,ez
      real dy2
c
      mxz = nelx*nelz*nx1*nz1
      call rzero(ua,mxz)
c
      do e=1,nelt
c
         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         j = 1
         if (ey.eq.1) then
            do k=1,nz1
            do i=1,nx1
               ua(i,k,ex,ez) = u(i,j,k,e)
            enddo
            enddo
         endif
      enddo

      call gop(ua,w2,'+  ',mxz)

      return
      end
c-----------------------------------------------------------------------
      subroutine z_slice_g (uz,u,w1,kz,ezi,nx,ny,nz,nlxy)
c
c     Extract a z slice of quantity u() 
c
c     ASSUMES data is in a global tensor-product structure, nlxy x nelz,
c             as would be produced by n2to3 or genbox.
c
c     Arguments:
c
c     uz(nx1,ny1,nlxy):       extracted z-slice data
c     u (nx1,ny1,nz1,nelt):   input data
c     w1(nx1,ny1,nlxy):       work array
c     kz:                     z-plane within element ezi to be extracted
c     ezi:                    elemental z-slab to be extracted
c     nx,ny,nz:               dimensions for 3D spectral element input
c     nlxy:                   global number of elements in x-y plane.

      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'

      real uz(nx,ny,nlxy),u (nx,ny,nz,nelv),w1(nx,ny,nlxy)
      integer e,eg,ex,ey,ez,ezi
      real dy2

      nxy = nx*ny
      mxy = nxy*nlxy

      call rzero(uz,mxy) ! zero out the entire plane

      do e=1,nelt
         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nlxy,1,1)

         if (ez.eq.ezi) 
     $      call copy(uz(1,1,ex),u(1,1,kz,e),nxy) ! u(zlice) --> uz()

      enddo

      call gop(uz,w1,'+  ',mxy) ! Collect partial contributions from all procs

      return
      end
c-----------------------------------------------------------------------
      subroutine z_slice (ua,u,w1,w2)
c
c     Extract a z slice of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'
c
      real ua(nx1,ny1,nelx,nely),u (nx1,ny1,nz1,nelv)
     $    ,w1(nx1,ny1,nelx,nely),w2(nx1,ny1,nelx,nely)
      integer e,eg,ex,ey,ez
      real dy2
c
      mxy = nelx*nely*nx1*ny1
      call rzero(ua,mxy)
c
      do e=1,nelt
c
         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         k = 1
         if (ez.eq.1) then
            do j=1,ny1
            do i=1,nx1
               ua(i,j,ex,ey) = u(i,j,k,e)
            enddo
            enddo
         endif
      enddo

      call gop(ua,w2,'+  ',mxy)

      return
      end
c-----------------------------------------------------------------------
      subroutine y_average(ua,u,w1,w2)
c
c     Compute the y average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'
c
      real ua(nx1,nz1,nelx,nelz),u (nx1,ny1,nz1,nelv)
     $    ,w1(nx1,nz1,nelx,nelz),w2(nx1,nz1,nelx,nelz)
      integer e,eg,ex,ey,ez
      real dy2
c
      mxz = nelx*nelz*nx1*nz1
      call rzero(ua,mxz)
      call rzero(w1,mxz)
c
      do e=1,nelt
c
         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)
c
         do k=1,nz1
         do i=1,nx1
c           dy2 = .5*( ym1(i,ny1,k,e) - ym1(i,1,k,e) )
            dy2 = 1.0  !  Assuming uniform in "y" direction
            do j=1,ny1
               ua(i,k,ex,ez) = ua(i,k,ex,ez)+dy2*wym1(j)*u(i,j,k,e)
               w1(i,k,ex,ez) = w1(i,k,ex,ez)+dy2*wym1(j) ! redundant but clear
            enddo
         enddo
         enddo
      enddo
c
      call gop(ua,w2,'+  ',mxz)
      call gop(w1,w2,'+  ',mxz)
c
      do i=1,mxz
         ua(i,1,1,1) = ua(i,1,1,1) / w1(i,1,1,1)   ! Normalize
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine y_avg_buff(ux,uy,uz,c2,name,icount)
c
c     Compute the y average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      real ux(1),uy(1),uz(1)
      character*2 c2,name

      parameter (lyavg = lx1*lz1*lelx*lelz)
      common /scravg/ u (lyavg)
     $              , v (lyavg)
     $              , w (lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)

      call y_average(u,ux,w1,w2)
      call y_average(v,uy,w1,w2)
      call y_average(w,uz,w1,w2)

      call buff_2d_out(u,v,w,nx1,nz1,nelx,nelz,c2,name,icount)

      return
      end
c-----------------------------------------------------------------------
      subroutine z_avg_buff(ux,uy,uz,c2,name,icount)
c
c     Compute the z average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'
c
      real ux(1),uy(1),uz(1)
      character*2 c2,name
c
      parameter (lyavg = lx1*ly1*lelx*lely)
      common /scravg/ u (lyavg)
     $              , v (lyavg)
     $              , w (lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)
c
      call z_average(u,ux,w1,w2)
      call z_average(v,uy,w1,w2)
      call z_average(w,uz,w1,w2)

      call buff_2d_out(u,v,w,nx1,ny1,nelx,nely,c2,name,icount)

      return
      end
c-----------------------------------------------------------------------
      subroutine y_ins_buff(ux,uy,uz,c2,name,icount)
c
c     Compute the z average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'
c
      real ux(1),uy(1),uz(1)
      character*2 c2,name
c
      parameter (lyavg = lx1*lz1*lelx*lelz)
      common /scravg/ u (lyavg)
     $              , v (lyavg)
     $              , w (lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)
c
      call y_slice  (u,ux,w1,w2)
      call y_slice  (v,uy,w1,w2)
      call y_slice  (w,uz,w1,w2)
c
      call buff_2d_out(u,v,w,nx1,nz1,nelx,nelz,c2,name,icount)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine z_ins_buff(ux,uy,uz,c2,name,icount)
c
c     Compute the z average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'
c
      real ux(1),uy(1),uz(1)
      character*2 c2,name
c
      parameter (lyavg = lx1*ly1*lelx*lely)
      common /scravg/ u (lyavg)
     $              , v (lyavg)
     $              , w (lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)
c
      call z_slice  (u,ux,w1,w2)
      call z_slice  (v,uy,w1,w2)
      call z_slice  (w,uz,w1,w2)
c
      call buff_2d_out(u,v,w,nx1,ny1,nelx,nely,c2,name,icount)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine buff_2d_out(u,v,w,nx,ny,nex,ney,c2,name,ifld)
c
      include 'SIZE'
      include 'TOTAL'

      real u(1),v(1),w(1)
      character*2 c2,name
      character*4  bname
      save         bname

      parameter (lyzm = lelx*max(lely,lelz))
      common /scrav2/ ub(lx1,lz1,lyzm),vb(lx1,lz1,lyzm),wb(lx1,lz1,lyzm)

      integer ibfld,icalld,nxf,nyf,nexf,neyf
      save    ibfld,icalld,nxf,nyf,nexf,neyf
      data    ibfld,icalld,nxf,nyf,nexf,neyf  / 6*0 /

c     npido = 64             !  64 files buffered
      npido = 128            !  64 files buffered
      npido =  min(npido,np) !  P  files buffered

      mpido = np/npido     ! stride between processors (e.g., 128/64 = 2)

      jcalld = mod(icalld,npido)       ! call # 0,1,...,63,0,1,...
      if (mod(nid,mpido) .eq. 0) then  ! this is a buffering/writing proc

         jid = nid/mpido
         if (jid.eq.jcalld) then       ! save this buffer on this proc
c           write(6,1) nid,jid,istep,icalld,jcalld,c2,name,nex,ney,ifld
c   1       format(5i7,' buffering: ',2a2,3i7)
            write(bname,4) c2,name
    4       format(2a2)
            n = nx*ny*nex*ney
            ibfld = ifld
            call copy(ub,u,n)
            call copy(vb,v,n)
            call copy(wb,w,n)
            nxf  = nx
            nyf  = ny
            nexf = nex
            neyf = ney
         endif

         if (jcalld .eq. npido-1) call  ! output buffer
     $  outfld2d_p(ub,vb,wb,nxf,nyf,nexf,neyf,bname,ibfld,jid,npido,ir)

      endif
      call err_chk(ir,'Error with byte_write, buff_2d_out $')
      icalld = icalld+1
      return
      end
c-----------------------------------------------------------------------
      subroutine y2d(u,v,w,p,c1,icount)
c
c     Compute the y average of quantity u() - assumes global tens.prod.
c

      include 'SIZE'
      include 'TOTAL'
      real u(1),v(1),w(1),p(1)
      character*1 c1,c2(2)

      common /scrns/ ur(lx1*ly1*lz1*lelv)
     $             , ut(lx1*ly1*lz1*lelv)
     $             , wr(lx1*ly1*lz1*lelv)
     $             , wt(lx1*ly1*lz1*lelv)
     $             , wp(lx1*ly1*lz1*lelv)
c
c     Convert velocities to poloidal-toroidal
c
      n = nx1*ny1*nz1*nelv
      do i=1,n
         rr = xm1(i,1,1,1)*xm1(i,1,1,1)+ym1(i,1,1,1)*ym1(i,1,1,1)
         rr = sqrt(rr)
         ct = xm1(i,1,1,1)/rr
         st = ym1(i,1,1,1)/rr
         ur(i) = ct*u(i)+st*v(i)
         ut(i) = ct*v(i)-st*u(i)
         wr(i) = ur(i)**2
         wt(i) = ut(i)**2
         wp(i) = w (i)**2
      enddo

      c2(1) = c1
      c2(2) = 'y'

      call y_avg_buff(ur,w ,ut,c2,'ub',icount)
      call y_avg_buff(wr,wp,wt,c2,'u2',icount)

      do i=1,n
         wr(i) = ur(i)*ut(i)
         wt(i) = ut(i)*w (i)
         wp(i) = w (i)*ur(i)
      enddo
      call y_avg_buff(wr,wt,wp,c2,'uv',icount)

      call y_ins_buff(ur,w ,ut,c2,'ui',icount)

      return
      end
c-----------------------------------------------------------------------
      subroutine z2d(u,v,w,p,c1,icount)
c
c     Compute the y average of quantity u() - assumes global tens.prod.
c

      include 'SIZE'
      include 'TOTAL'
      real u(1),v(1),w(1),p(1)
      character*1 c1,c2(2)

      common /scrns/ ur(lx1*ly1*lz1*lelv)
     $             , ut(lx1*ly1*lz1*lelv)
     $             , wr(lx1*ly1*lz1*lelv)
     $             , wt(lx1*ly1*lz1*lelv)
     $             , wp(lx1*ly1*lz1*lelv)
c
c
c     Convert velocities to poloidal-toroidal
c
      n = nx1*ny1*nz1*nelv
      do i=1,n
         wr(i) = u (i)**2
         wt(i) = v (i)**2
         wp(i) = w (i)**2
      enddo

      c2(1) = c1
      c2(2) = 'z'

      call z_avg_buff(u ,v ,w ,c2,'ub',icount)
      call z_avg_buff(wr,wt,wp,c2,'u2',icount)

      do i=1,n
         wr(i) = u(i)*v(i)
         wt(i) = v(i)*w(i)
         wp(i) = w(i)*u(i)
      enddo
      call z_avg_buff(wr,wt,wp,c2,'uv',icount)

      call z_ins_buff(u ,v ,w ,c2,'ui',icount)

      return
      end
c-----------------------------------------------------------------------
      subroutine anal_2d

      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      integer icount
      save    icount

      if (nelx.gt.lelx .or.
     $    nely.gt.lely .or.
     $    nelz.gt.lelz ) then
         if (nio.eq.0) write(6,1) nelx,nely,nelz,lelx,lely,lelz
    1    format('anal_2d fail:',6i6)
         return
      endif

      if (istep.eq.0) then   ! dump four times, just to keep phase

         icount = 0
         call z2d(xm1,ym1,zm1,pr,'u',icount)
         if (ifmhd) call z2d(xm1,ym1,zm1,pm,'b',icount)

         call y2d(xm1,ym1,zm1,pr,'u',icount)
         if (ifmhd) call y2d(xm1,ym1,zm1,pm,'b',icount)

      endif

      icount = icount + 1

      call z2d(vx,vy,vz,pr,'u',icount)
      if (ifmhd) call z2d(bx,by,bz,pm,'b',icount)

      call y2d(vx,vy,vz,pr,'u',icount)
      if (ifmhd) call y2d(bx,by,bz,pm,'b',icount)

      return
      end
c-----------------------------------------------------------------------
      subroutine chkit(u,name4,n)

      include 'SIZE'
      include 'TOTAL'

      character*4 name4
      real u(1)


      integer icalld
      save    icalld
      data    icalld /0/
      
      icalld = icalld + 1

      u2   = vlsc2(u,u,n)
      umin = vlmin(u,n)
      umax = vlmax(u,n)
      ulst = u(n)
      if (nio.eq.0)
     $write(6,1) nid,icalld,istep,n,umin,umax,ulst,name4,' chkit',nid
    1 format(4i7,1p3e12.4,a4,a6,i1)

      return
      end
c-----------------------------------------------------------------------
      subroutine outmesh
      include 'SIZE'
      include 'TOTAL'
      integer e,eg

      common /cmesh/ xt(2**ldim,ldim)

      len = wdsize*ndim*(2**ndim)

      if (nid.eq.0) open(unit=29,file='rea.new')

      do eg=1,nelgt
         call nekgsync()          !  belt
         jnid = gllnid(eg)
         e    = gllel (eg)
c     tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
         mtype = e
         if (jnid.eq.0 .and. nid.eq.0) then
            call get_el(xt,xm1(1,1,1,e),ym1(1,1,1,e),zm1(1,1,1,e))
            call out_el(xt,eg)
         elseif (nid.eq.0) then
            call crecv2(mtype,xt,len,jnid)
            call out_el(xt,eg)
         elseif (jnid.eq.nid) then
            call get_el(xt,xm1(1,1,1,e),ym1(1,1,1,e),zm1(1,1,1,e))
            call csend(mtype,xt,len,0,0)
         endif
         call nekgsync()          !  suspenders
      enddo

      if (nid.eq.0) close(29)
      call nekgsync()
      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine out_el(xt,e)
      include 'SIZE'
      include 'TOTAL'

      real xt(2**ldim,ldim)
      integer e

      integer ed(8)
      save    ed
      data    ed  / 1,2,4,3 , 5,6,8,7 /

      write(29,1) e
      write(29,2) ((xt(ed(k),j),k=1,4),j=1,ndim)
      write(29,2) ((xt(ed(k),j),k=5,8),j=1,ndim)

    1 format(12x,'ELEMENT',i6,' [    1 ]    GROUP     0')
    2 format(1p4e18.10)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_el(xt,x,y,z)
      include 'SIZE'
      include 'TOTAL'

      real xt(2**ldim,ldim)
      real x(nx1,ny1,nz1),y(nx1,ny1,nz1),z(nx1,ny1,nz1)

      l = 0
      do k=1,nz1,nz1-1
      do j=1,ny1,ny1-1
      do i=1,nx1,nx1-1
         l = l+1
         xt(l,1) = x(i,j,k)
         xt(l,2) = y(i,j,k)
         xt(l,3) = z(i,j,k)
      enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine shear_calc_max(strsmx,scale,x0,ifdout,iftout)
c
c     Compute maximum shear stress on objects
c
c     Scale is a user-supplied multiplier so that results may be
c     scaled to any convenient non-dimensionalization.
c
c
      INCLUDE 'SIZE'  
      INCLUDE 'TOTAL' 

      real    strsmx(maxobj),x0(3),w1(0:maxobj)
      logical ifdout,iftout

      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec
     $                , scale_vf(3)


      common /scrns/         sij (lx1*ly1*lz1*6*lelv)
      common /scrcg/         pm1 (lx1,ly1,lz1,lelv)
      common /scrsf/         xm0(lx1,ly1,lz1,lelt)
     $,                      ym0(lx1,ly1,lz1,lelt)
     $,                      zm0(lx1,ly1,lz1,lelt)

      parameter (lr=lx1*ly1*lz1)
      common /scruz/         ur(lr),us(lr),ut(lr)
     $                     , vr(lr),vs(lr),vt(lr)
     $                     , wr(lr),ws(lr),wt(lr)


      n = nx1*ny1*nz1*nelv

      call mappr(pm1,pr,xm0,ym0) ! map pressure onto Mesh 1

c    Add mean_pressure_gradient.X to p:

      if (param(55).ne.0) then
         dpdx_mean = -scale_vf(1)
         dpdy_mean = -scale_vf(2)
         dpdz_mean = -scale_vf(3)
      endif

      call add2s2(pm1,xm1,dpdx_mean,n)  ! Doesn't work if object is cut by 
      call add2s2(pm1,ym1,dpdy_mean,n)  ! periodicboundary.  In this case,
      call add2s2(pm1,zm1,dpdz_mean,n)  ! set ._mean=0 and compensate in
c
c    Compute sij
c
      nij = 3
      if (if3d.or.ifaxis) nij=6
      call comp_sij(sij,nij,vx,vy,vz,ur,us,ut,vr,vs,vt,wr,ws,wt)
c
c
c     Fill up viscous array w/ default
c
      if (istep.lt.1) call cfill(vdiff,param(2),n)
c
      call cadd2(xm0,xm1,-x0(1),n)
      call cadd2(ym0,ym1,-x0(2),n)
      call cadd2(zm0,zm1,-x0(3),n)
c
      x1min=glmin(xm0(1,1,1,1),n)
      x2min=glmin(ym0(1,1,1,1),n)
      x3min=glmin(zm0(1,1,1,1),n)
c
      x1max=glmax(xm0(1,1,1,1),n)
      x2max=glmax(ym0(1,1,1,1),n)
      x3max=glmax(zm0(1,1,1,1),n)
c
      call rzero(strsmx,maxobj)
c
c
      nobj = 0
      do ii=1,nhis
        if (hcode(10,ii).EQ.'I') then
          iobj   = lochis(1,ii)
          memtot = nmember(iobj)
          nobj   = max(iobj,nobj)
c
          if (hcode(1,ii).ne.' ' .or. hcode(2,ii).ne.' ' .or.
     $      hcode(3,ii).ne.' ' ) then
            ifield = 1
c
c           Compute max stress for this object
c
            strsmx(ii) = 0.
            do mem=1,memtot
               ieg   = object(iobj,mem,1)
               ifc   = object(iobj,mem,2)
               if (gllnid(ieg).eq.nid) then ! this processor has a contribution

                  ie = gllel(ieg)
                  call get_strsmax
     $                    (strsmxl,xm0,ym0,zm0,sij,pm1,vdiff,ifc,ie)

                  call cmult(strsmxl,scale,1)
                  strsmx(ii)=max(strsmx(ii),strsmxl)

               endif
            enddo
          endif
        endif
      enddo
c
c     Max contributions over all processors
c
      call gop(strsmx,w1,'M  ',maxobj)


      return
      end
c-----------------------------------------------------------------------
      subroutine get_strsmax(strsmax,xm0,ym0,zm0,sij,pm1,visc,f,e)
c
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TOPOL'
      INCLUDE 'TSTEP'
c
      real dgtq(3,4)
      real xm0 (lx1,ly1,lz1,lelt)
      real ym0 (lx1,ly1,lz1,lelt)
      real zm0 (lx1,ly1,lz1,lelt)
      real sij (lx1,ly1,lz1,3*ldim-3,lelv)
      real pm1 (lx1,ly1,lz1,lelv)
      real visc(lx1,ly1,lz1,lelv)

      integer f,e,pf
      real    n1,n2,n3

      call dsset(nx1,ny1,nz1)    ! set up counters
      pf     = eface1(f)         ! convert from preproc. notation
      js1    = skpdat(1,pf)
      jf1    = skpdat(2,pf)
      jskip1 = skpdat(3,pf)
      js2    = skpdat(4,pf)
      jf2    = skpdat(5,pf)
      jskip2 = skpdat(6,pf)

      if (if3d.or.ifaxis) then
         i       = 0
         strsmax = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1
            i = i+1
            n1 = unx(i,1,f,e)
            n2 = uny(i,1,f,e)
            n3 = unz(i,1,f,e)
c
            v  = visc(j1,j2,1,e)
c
            s11 = sij(j1,j2,1,1,e)
            s21 = sij(j1,j2,1,4,e)
            s31 = sij(j1,j2,1,6,e)
c
            s12 = sij(j1,j2,1,4,e)
            s22 = sij(j1,j2,1,2,e)
            s32 = sij(j1,j2,1,5,e)

            s13 = sij(j1,j2,1,6,e)
            s23 = sij(j1,j2,1,5,e)
            s33 = sij(j1,j2,1,3,e)

            stress1 = -v*(s11*n1 + s12*n2 + s13*n3) ! viscous drag
            stress2 = -v*(s21*n1 + s22*n2 + s23*n3)
            stress3 = -v*(s31*n1 + s32*n2 + s33*n3)

            strsnrm = stress1*stress1+stress2*stress2+stress3*stress3
            strsmax = max(strsmax,strsnrm)

         enddo
         enddo

      else ! 2D

         i       = 0
         strsmax = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1
            i = i+1
            n1 = unx(i,1,f,e)*area(i,1,f,e)
            n2 = uny(i,1,f,e)*area(i,1,f,e)
            v  = visc(j1,j2,1,e)

            s11 = sij(j1,j2,1,1,e)
            s12 = sij(j1,j2,1,3,e)
            s21 = sij(j1,j2,1,3,e)
            s22 = sij(j1,j2,1,2,e)

            stress1 = -v*(s11*n1 + s12*n2) ! viscous drag
            stress2 = -v*(s21*n1 + s22*n2)

            strsnrm = stress1*stress1+stress2*stress2
            strsmax = max(strsmax,strsnrm)

       enddo
       enddo

      endif

      if (strsmax.gt.0) strsmax = sqrt(strsmax)

      return
      end
c-----------------------------------------------------------------------
      subroutine fix_geom ! fix up geometry irregularities

      include 'SIZE'
      include 'TOTAL'

      parameter (lt = lx1*ly1*lz1)
      common /scrns/ xb(lt,lelt),yb(lt,lelt),zb(lt,lelt)
      common /scruz/ tmsk(lt,lelt),tmlt(lt,lelt),w1(lt),w2(lt)

      integer e,f
      character*3 cb

      n      = nx1*ny1*nz1*nelt
      nxyz   = nx1*ny1*nz1
      nfaces = 2*ndim
      ifield = 1                   ! velocity field
      if (ifheat) ifield = 2       ! temperature field


      call rone  (tmlt,n)
      call dssum (tmlt,nx1,ny1,nz1)  ! denominator

      call rone  (tmsk,n)
      do e=1,nelfld(ifield)      ! fill mask where bc is periodic
      do f=1,nfaces              ! so we don't translate periodic bcs (z only)
         cb =cbc(f,e,ifield)
         if (cb.eq.'P  ') call facev (tmsk,e,f,0.0,nx1,ny1,nz1)
      enddo
      enddo

      do kpass = 1,ndim+1   ! This doesn't work for 2D, yet.
                            ! Extra pass is just to test convergence

c        call opcopy (xb,yb,zb,xm1,ym1,zm1) ! Must use WHOLE field,
c        call opdssum(xb,yb,zb)             ! not just fluid domain.
         call copy   (xb,xm1,n)
         call copy   (yb,ym1,n)
         call copy   (zb,zm1,n)
         call dssum  (xb,nx1,ny1,nz1)
         call dssum  (yb,nx1,ny1,nz1)
         call dssum  (zb,nx1,ny1,nz1)

         xm = 0.
         ym = 0.
         zm = 0.

         do e=1,nelfld(ifield)
            do i=1,nxyz                       ! compute averages of geometry
               s     = 1./tmlt(i,e)
               xb(i,e) = s*xb(i,e)
               yb(i,e) = s*yb(i,e)
               zb(i,e) = s*zb(i,e)

               xb(i,e) = xb(i,e) - xm1(i,1,1,e)   ! local displacements
               yb(i,e) = yb(i,e) - ym1(i,1,1,e)
               zb(i,e) = zb(i,e) - zm1(i,1,1,e)
               xb(i,e) = xb(i,e)*tmsk(i,e)
               yb(i,e) = yb(i,e)*tmsk(i,e)
               zb(i,e) = zb(i,e)*tmsk(i,e)

               xm = max(xm,abs(xb(i,e)))
               ym = max(ym,abs(yb(i,e)))
               zm = max(zm,abs(zb(i,e)))
            enddo

            if (kpass.le.ndim) then
               call gh_face_extend(xb(1,e),zgm1,nx1,kpass,w1,w2)
               call gh_face_extend(yb(1,e),zgm1,nx1,kpass,w1,w2)
               call gh_face_extend(zb(1,e),zgm1,nx1,kpass,w1,w2)
            endif

         enddo

         if (kpass.le.ndim) then
            call add2(xm1,xb,n)
            call add2(ym1,yb,n)
            call add2(zm1,zb,n)
         endif
         
         xx = glamax(xb,n)
         yx = glamax(yb,n)
         zx = glamax(zb,n)

         xm = glmax(xm,1)
         ym = glmax(ym,1)
         zm = glmax(zm,1)

         if (nio.eq.0) write(6,1) xm,ym,zm,xx,yx,zx,kpass
    1    format(1p6e12.4,' xyz repair',i2)

      enddo

      param(59) = 1.       ! ifdef = .true.
      call geom_reset(1)   ! reset metrics, etc.
      
      return
      end
c-----------------------------------------------------------------------
      subroutine gh_face_extend(x,zg,n,gh_type,e,v)
      include 'SIZE'

      real x(1),zg(1),e(1),v(1)
      integer gh_type

      if (ndim.eq.2) then
         call gh_face_extend_2d(x,zg,n,gh_type,e,v)
      else
         call gh_face_extend_3d(x,zg,n,gh_type,e,v)
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine gh_face_extend_2d(x,zg,n,gh_type,e,v)
c
c     Extend 2D faces into interior via gordon hall
c
c     gh_type:  1 - vertex only
c               2 - vertex and faces
c
c
      real x(n,n)
      real zg(n)
      real e(n,n)
      real v(n,n)
      integer gh_type
c
c     Build vertex interpolant
c
      ntot=n*n
      call rzero(v,ntot)
      do jj=1,n,n-1
      do ii=1,n,n-1
         do j=1,n
         do i=1,n
            si     = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            sj     = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            v(i,j) = v(i,j) + si*sj*x(ii,jj)
         enddo
         enddo
      enddo
      enddo
      if (gh_type.eq.1) then
         call copy(x,v,ntot)
         return
      endif


c     Extend 4 edges
      call rzero(e,ntot)
c
c     x-edges
c
      do jj=1,n,n-1
         do j=1,n
         do i=1,n
            hj     = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            e(i,j) = e(i,j) + hj*(x(i,jj)-v(i,jj))
         enddo
         enddo
      enddo
c
c     y-edges
c
      do ii=1,n,n-1
         do j=1,n
         do i=1,n
            hi     = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            e(i,j) = e(i,j) + hi*(x(ii,j)-v(ii,j))
         enddo
         enddo
      enddo

      call add3(x,e,v,ntot)

      return
      end
c-----------------------------------------------------------------------
      subroutine gh_face_extend_3d(x,zg,n,gh_type,e,v)
c
c     Extend faces into interior via gordon hall
c
c     gh_type:  1 - vertex only
c               2 - vertex and edges
c               3 - vertex, edges, and faces
c
c
      real x(n,n,n)
      real zg(n)
      real e(n,n,n)
      real v(n,n,n)
      integer gh_type
c
c     Build vertex interpolant
c
      ntot=n*n*n
      call rzero(v,ntot)
      do kk=1,n,n-1
      do jj=1,n,n-1
      do ii=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            si       = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            sj       = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            sk       = 0.5*((n-kk)*(1-zg(k))+(kk-1)*(1+zg(k)))/(n-1)
            v(i,j,k) = v(i,j,k) + si*sj*sk*x(ii,jj,kk)
         enddo
         enddo
         enddo
      enddo
      enddo
      enddo
      if (gh_type.eq.1) then
         call copy(x,v,ntot)
         return
      endif
c
c
c     Extend 12 edges
      call rzero(e,ntot)
c
c     x-edges
c
      do kk=1,n,n-1
      do jj=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hj       = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            hk       = 0.5*((n-kk)*(1-zg(k))+(kk-1)*(1+zg(k)))/(n-1)
            e(i,j,k) = e(i,j,k) + hj*hk*(x(i,jj,kk)-v(i,jj,kk))
         enddo
         enddo
         enddo
      enddo
      enddo
c
c     y-edges
c
      do kk=1,n,n-1
      do ii=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hi       = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            hk       = 0.5*((n-kk)*(1-zg(k))+(kk-1)*(1+zg(k)))/(n-1)
            e(i,j,k) = e(i,j,k) + hi*hk*(x(ii,j,kk)-v(ii,j,kk))
         enddo
         enddo
         enddo
      enddo
      enddo
c
c     z-edges
c
      do jj=1,n,n-1
      do ii=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hi       = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            hj       = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            e(i,j,k) = e(i,j,k) + hi*hj*(x(ii,jj,k)-v(ii,jj,k))
         enddo
         enddo
         enddo
      enddo
      enddo
c
      call add2(e,v,ntot)
c
      if (gh_type.eq.2) then
         call copy(x,e,ntot)
         return
      endif
c
c     Extend faces
c
      call rzero(v,ntot)
c
c     x-edges
c
      do ii=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hi       = 0.5*((n-ii)*(1-zg(i))+(ii-1)*(1+zg(i)))/(n-1)
            v(i,j,k) = v(i,j,k) + hi*(x(ii,j,k)-e(ii,j,k))
         enddo
         enddo
         enddo
      enddo
c
c     y-edges
c
      do jj=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hj       = 0.5*((n-jj)*(1-zg(j))+(jj-1)*(1+zg(j)))/(n-1)
            v(i,j,k) = v(i,j,k) + hj*(x(i,jj,k)-e(i,jj,k))
         enddo
         enddo
         enddo
      enddo
c
c     z-edges
c
      do kk=1,n,n-1
         do k=1,n
         do j=1,n
         do i=1,n
            hk       = 0.5*((n-kk)*(1-zg(k))+(kk-1)*(1+zg(k)))/(n-1)
            v(i,j,k) = v(i,j,k) + hk*(x(i,j,kk)-e(i,j,kk))
         enddo
         enddo
         enddo
      enddo
c
      call add2(v,e,ntot)
      call copy(x,v,ntot)

      return
      end
c-----------------------------------------------------------------------
      function ran1(idum)
c
      integer idum,ia,im,iq,ir,ntab,ndiv
      real    ran1,am,eps,rnmx
c
      parameter (ia=16807,im=2147483647,am=1./im,iq=127773,ir=2836)
      parameter (ntab=32,ndiv=1+(im-1)/ntab,eps=1.2e-7,rnmx=1.-eps)
c
c     Numerical Rec. in Fortran, 2nd eD.  P. 271
c
      integer j,k
      integer iv(ntab),iy
      save    iv,iy
      data    iv,iy /ntab*0,0/
c
      if (idum.le.0.or.iy.eq.0) then
         idum=max(-idum,1)
         do j=ntab+8,1,-1
            k    = idum/iq
            idum = ia*(idum-k*iq)-ir*k
            if(idum.lt.0) idum = idum+im
            if (j.le.ntab) iv(j) = idum
         enddo
         iy = iv(1)
      endif
      k    = idum/iq
      idum = ia*(idum-k*iq)-ir*k
      if(idum.lt.0) idum = idum+im
      j     = 1+iy/ndiv
      iy    = iv(j)
      iv(j) = idum
      ran1  = min(am*iy,rnmx)
c     ran1  = cos(ran1*1.e8)

      return
      end
c-----------------------------------------------------------------------
      subroutine rand_fld_h1(x)

      include 'SIZE'
      real x(1)

      n=nx1*ny1*nz1*nelt
      id = n
      do i=1,n
         x(i) = ran1(id)
      enddo
      call dsavg(x)

      return
      end
c-----------------------------------------------------------------------
      subroutine rescale_x (x,x0,x1)
      include 'SIZE'
      real x(1)

      n = nx1*ny1*nz1*nelt
      xmin = glmin(x,n)
      xmax = glmax(x,n)

      if (xmax.le.xmin) return

      scale = (x1-x0)/(xmax-xmin)
      do i=1,n
         x(i) = x0 + scale*(x(i)-xmin)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine z_distribute(u)
c
c     Compute the z average of quantity u() and redistribute
c
c     Assumes you have nelx*nely elements, in the same order,
c     within each z plane
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      real ux(1),uy(1),uz(1)
      character*2 c2,name

      parameter (lyavg = lx1*ly1*lelx*lely)
      common /scravg/ ua(lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)

      call z_average          (ua,u,w1,w2)
      call z_average_transpose(u,ua) ! distribute ua to each z-plane

      return
      end
c-----------------------------------------------------------------------
      subroutine z_average(ua,u,w1,w2)
c
c     Compute the z average of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real ua(nx1,ny1,nelx,nely),u (nx1,ny1,nz1,nelv)
     $    ,w1(nx1,ny1,nelx,nely),w2(nx1,ny1,nelx,nely)
      integer e,eg,ex,ey,ez
      real dy2

      nelxy = nelx*nely
      if (nelxy.gt.lelx*lely) call exitti
     $  ('ABORT IN z_average. Increase lelx*lely in SIZE:$',nelxy)

      mxy = nelx*nely*nx1*ny1
      call rzero(ua,mxy)
      call rzero(w1,mxy)

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do j=1,ny1
         do i=1,nx1
            dz2 = 1.0  !  Assuming uniform in "z" direction
            do k=1,nz1
               ua(i,j,ex,ey) = ua(i,j,ex,ey)+dz2*wzm1(k)*u(i,j,k,e)
               w1(i,j,ex,ey) = w1(i,j,ex,ey)+dz2*wzm1(k) ! redundant but clear
            enddo
         enddo
         enddo
      enddo

      call gop(ua,w2,'+  ',mxy)
      call gop(w1,w2,'+  ',mxy)

      do i=1,mxy
         ua(i,1,1,1) = ua(i,1,1,1) / w1(i,1,1,1)   ! Normalize
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine z_average_transpose(u,ua) ! distribute ua to each z-plane

      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real u(nx1,ny1,nz1,nelv),ua(nx1,ny1,nelx,nely)

      integer e,eg,ex,ey,ez


      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do j=1,ny1
         do i=1,nx1
            do k=1,nz1
               u(i,j,k,e) = ua(i,j,ex,ey)
            enddo
         enddo
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine no_z_profile(u)

c     Subtract the z_profile from u for a tensor-product array of elements

c     Assumes you have nelx*nely*nelz elements, in the same order,
c     and that lelx,lely,lelz are defined to be >= nelx,nely,nelz


      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'         ! nelx,nely,nelz

      real u(1)

      parameter (lyavg = ly1*lely)
      common /scravg/ ua(lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)
      common /scrmg/  ub(lx1*ly1*lz1*lelt)

      call z_profile          (ua,u,w1,w2)
      call z_profile_transpose(ub,ua) ! distribute ua to each z-plane

      n = nx1*ny1*nz1*nelv
      call sub2(u,ub,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine z_profile(ua,u,w1,w2)
c
c     Compute the z profile of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real ua(lz1,lelz),u (nx1,ny1,nz1,nelv)
     $    ,w1(lz1,lelz),w2(lz1,lelz)
      integer e,eg,ex,ey,ez

      mz = nz1*nelz
      call rzero(ua,mz)
      call rzero(w1,mz)

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do i=1,nx1*ny1
            ua(k,ez) = ua(k,ez) + area(i,1,5,e)*u(i,1,k,e)
            w1(k,ez) = w1(k,ez) + area(i,1,5,e)
         enddo
         enddo

      enddo

      call gop(ua,w2,'+  ',mz)
      call gop(w1,w2,'+  ',mz)

      do i=1,mz
         ua(i,1) = ua(i,1) / w1(i,1)   ! Normalize
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine z_profile_transpose(u,ua) ! distribute ua to each z-plane

      include 'SIZE'
      include 'PARALLEL'
      include 'ZPER'

      real u(nx1,ny1,nz1,nelv),ua(lz1,lelz)
      integer e,eg,ex,ey,ez

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do i=1,nx1*ny1
            u(i,1,k,e) = ua(k,ez)
         enddo
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine no_y_profile(u)

c     Subtract the y_profile from u for a tensor-product array of elements

c     Assumes you have nelx*nely*nelz elements, in the same order,
c     and that lelx,lely,lelz are defined to be >= nelx,nely,nelz


      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'         ! nelx,nely,nelz

      real u(1)

      parameter (lyavg = ly1*lely)
      common /scravg/ ua(lyavg)
     $              , w1(lyavg)
     $              , w2(lyavg)
      common /scrmg/  ub(lx1*ly1*lz1*lelt)

      call y_profile          (ua,u,w1,w2)
      call y_profile_transpose(ub,ua) ! distribute ua to each y-plane

      n = nx1*ny1*nz1*nelv
      call sub2(u,ub,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine y_profile(ua,u,w1,w2)
c
c     Compute the z profile of quantity u() - assumes global tens.prod.
c
      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'WZ'
      include 'ZPER'

      real ua(lz1,lelz),u (nx1,ny1,nz1,nelv)
     $    ,w1(lz1,lelz),w2(lz1,lelz)
      integer e,eg,ex,ey,ez

      my = ny1*nely
      call rzero(ua,my)
      call rzero(w1,my)

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do j=1,ny1
         do i=1,nx1
            ua(j,ey) = ua(j,ey) + area(i,k,1,e)*u(i,j,k,e)
            w1(j,ey) = w1(j,ey) + area(i,k,1,e)
         enddo
         enddo
         enddo

      enddo

      call gop(ua,w2,'+  ',my)
      call gop(w1,w2,'+  ',my)

      do i=1,my
         ua(i,1) = ua(i,1) / w1(i,1)   ! Normalize
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine y_profile_transpose(u,ua) ! distribute ua to each z-plane

      include 'SIZE'
      include 'PARALLEL'
      include 'ZPER'

      real u(nx1,ny1,nz1,nelv),ua(lz1,lelz)
      integer e,eg,ex,ey,ez

      do e=1,nelt

         eg = lglel(e)
         call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)

         do k=1,nz1
         do j=1,ny1
         do i=1,nx1
            u(i,j,k,e) = ua(j,ey)
         enddo
         enddo
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine build_filter(f,diag,nx)
      include 'SIZE'

      real f(nx,nx),diag(nx),zpts(nx)

      parameter (lm=4*lx1) ! Totally arbitrary
      parameter (lm2=lm*lm)

      common /cfilt1/ phi,pht,ft,rmult,Lj,gpts,indr,indc,ipiv
      real      phi(lm2),pht(lm2),ft(lm2),rmult(lm),Lj(lm),gpts(lm)
      integer   indr(lm),indc(lm),ipiv(lm)

      integer nxl
      save    nxl
      data    nxl / -9 /

      if (nx.gt.lm) call exitti('ABORT in build_filter:$',nx)

      if (nx.ne.nxl) then

        nxl = nx

        call zwgll (gpts,f,nx)  ! Get nx GLL points

        kj = 0
        n  = nx-1
        do j=1,nx
         z = gpts(j)
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

      endif ! End of save section

      ij=0
      do j=1,nx
      do i=1,nx
         ij = ij+1
         ft(ij) = diag(i)*pht(ij)
      enddo
      enddo
                                          !          -1
      call mxm  (phi,nx,ft,nx,f,nx)       !     V D V

      return
      end
c-----------------------------------------------------------------------
      subroutine g_filter(u,diag,ifld)
c
c     Generalized filter: F(u) with F = J^T D J, where D=diag(diag)
c
      include 'SIZE'
      include 'TOTAL'

      real u(1),diag(1)

      parameter (lxx=lx1*lx1,lxyz=lx1*ly1*lz1)
      common /ctmp0/ f(lxx),wk1(lxyz),wk2(lxyz),wk3(lxyz)

      ifldt = ifield
      ifield = ifld

      call build_filter(f,diag,nx1)
      call filterq(u,f,nx1,nz1,wk1,wk2,wk3,if3d,umax)

      ifield = ifldt

      return
      end
c-----------------------------------------------------------------------
      subroutine cut_off_filter(u,mx,ifld) ! mx=max saved mode
c
c     Generalized filter: F(u) with F = J^T D J, where D=diag(diag)
c
      include 'SIZE'
      include 'TOTAL'

      real u(1)

      parameter (lxx=lx1*lx1,lxyz=lx1*ly1*lz1)
      common /ctmp0/ f(lxx),wk1(lxyz),wk2(lxyz),wk3(lxyz),diag(lx1)

      ifldt = ifield
      ifield = ifld

      call rone(diag,nx1)
      do i=mx+1,nx1
         diag(i)=0.
      enddo

      call build_filter(f,diag,nx1)
      call filterq(u,f,nx1,nz1,wk1,wk2,wk3,if3d,umax)

      ifield = ifldt

      return
      end
c-----------------------------------------------------------------------
      subroutine filter_d2(v,nx,nz,wgt,ifd4)

      include 'SIZE'
      include 'TOTAL'

      parameter (lt=lx1*ly1*lz1)
      real v(lt,nelt)
      logical ifd4

      common /ctmp1/ w(lt,lelt),ur(lt),us(lt),ut(lt),w1(2*lt)

      integer e

      n   = nx1*ny1*nz1*nelt
      nn  = nx-1
      nel = nelfld(ifield)

      bmax = glamax(v,n)

      if (if3d) then
        do e=1,nel
          call local_grad3(ur,us,ut,v(1,e),nn,1,dxm1,dxtm1)
          do i=1,lt
            ur(i) = ur(i)*w3m1(i,1,1)
            us(i) = us(i)*w3m1(i,1,1)
            ut(i) = ut(i)*w3m1(i,1,1)
          enddo
          call local_grad3_t(w(1,e),ur,us,ut,nn,1,dxm1,dxtm1,w1)
        enddo
        call dsavg(w)  !NOTE STILL NEED BC TREATMENT !

        if (ifd4) then
           wght = 20./(nx1**4)
           do e=1,nel
             do i=1,lt
               w(i,e)  = wght*w(i,e)/w3m1(i,1,1)
             enddo
             call local_grad3(ur,us,ut,w(1,e),nn,1,dxm1,dxtm1)
             do i=1,lt
               ur(i) = ur(i)*w3m1(i,1,1)
               us(i) = us(i)*w3m1(i,1,1)
               ut(i) = ut(i)*w3m1(i,1,1)
             enddo
             call local_grad3_t(w(1,e),ur,us,ut,nn,1,dxm1,dxtm1,w1)
           enddo
           call dsavg(w)  !NOTE STILL NEED BC TREATMENT !
        endif

        wght = wgt/(nx1**4)
        do e=1,nel
          do i=1,lt
            v(i,e)  = v(i,e) - wght*w(i,e)/w3m1(i,1,1)
          enddo
        enddo

      else  ! 2D

        do e=1,nel
          call local_grad2(ur,us,v(1,e),nn,1,dxm1,dxtm1)
          do i=1,lt
            ur(i) = ur(i)*w3m1(i,1,1)
            us(i) = us(i)*w3m1(i,1,1)
          enddo
          call local_grad2_t(w(1,e),ur,us,nn,1,dxm1,dxtm1,w1)
        enddo
        call dsavg(w)  !NOTE STILL NEED BC TREATMENT !

        if (ifd4) then
           wght = 200./(nx1**4)
           do e=1,nel
             do i=1,lt
               w(i,e)  = wght*w(i,e)/w3m1(i,1,1)
             enddo
             call local_grad2(ur,us,w(1,e),nn,1,dxm1,dxtm1)
             do i=1,lt
               ur(i) = ur(i)*w3m1(i,1,1)
               us(i) = us(i)*w3m1(i,1,1)
             enddo
             call local_grad2_t(w(1,e),ur,us,nn,1,dxm1,dxtm1,w1)
           enddo
           call dsavg(w)  !NOTE STILL NEED BC TREATMENT !
        endif

        wght = wgt/(nx1**4)
        do e=1,nel
          do i=1,lt
            v(i,e)  = v(i,e) - wght*w(i,e)/w3m1(i,1,1)
          enddo
        enddo

      endif

      vmax = glamax(v,n)
      if (nio.eq.0) write(6,1) istep,time,vmax,bmax,' filter max'
    1 format(i9,1p3e12.4,a11)

      return
      end
c-------------------------------------------------------------------------
      function dist3d(a,b,c,x,y,z)

      d = (a-x)**2 + (b-y)**2 + (c-z)**2

      dist3d = 0.
      if (d.gt.0) dist3d = sqrt(d)

      return
      end
c-----------------------------------------------------------------------
      function dist2d(a,b,x,y)

      d = (a-x)**2 + (b-y)**2

      dist2d = 0.
      if (d.gt.0) dist2d = sqrt(d)

      return
      end
c-----------------------------------------------------------------------
      subroutine domain_size(xmin,xmax,ymin,ymax,zmin,zmax)

      include 'SIZE'
      include 'TOTAL'

      n = nx1*ny1*nz1*nelt

      xmin = glmin(xm1,n)
      xmax = glmax(xm1,n)
      ymin = glmin(ym1,n)
      ymax = glmax(ym1,n)
      if (if3d) then
         zmin = glmin(zm1,n)
         zmax = glmax(zm1,n)
      else
         zmin = 0.
         zmax = 0.
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cheap_dist(d,ifld,b)

c     Finds a pseudo-distance function.

c     INPUT:  ifld - field type for which distance function is to be found.
c             ifld = 1 for velocity
c             ifld = 2 for temperature, etc.

c     OUTPUT: d = "path" distance to nearest wall

c     This approach has a significant advantage that it works for
c     periodict boundary conditions, whereas most other approaches
c     will not.

      include 'SIZE'
      include 'GEOM'       ! Coordinates
      include 'INPUT'      ! cbc()
      include 'TSTEP'      ! nelfld
      include 'PARALLEL'   ! gather-scatter handle for field "ifld"

      real d(lx1,ly1,lz1,lelt)

      character*3 b  ! Boundary condition of interest

      integer e,eg,f

      nel = nelfld(ifld)
      n = nx1*ny1*nz1*nel

      call domain_size(xmin,xmax,ymin,ymax,zmin,zmax)

      xmn = min(xmin,ymin)
      xmx = max(xmax,ymax)
      if (if3d) xmn = min(xmn ,zmin)
      if (if3d) xmx = max(xmx ,zmax)

      big = 10*(xmx-xmn)
      call cfill(d,big,n)


      nface = 2*ndim
      do e=1,nel     ! Set d=0 on walls
      do f=1,nface
         if (cbc(f,e,ifld).eq.b) call facev(d,e,f,0.,nx1,ny1,nz1)
      enddo
      enddo

      do ipass=1,10000
         dmax    = 0
         nchange = 0
         do e=1,nel
           do k=1,nz1
           do j=1,ny1
           do i=1,nx1
             i0=max(  1,i-1)
             j0=max(  1,j-1)
             k0=max(  1,k-1)
             i1=min(nx1,i+1)
             j1=min(ny1,j+1)
             k1=min(nz1,k+1)
             do kk=k0,k1
             do jj=j0,j1
             do ii=i0,i1

              if (if3d) then
               dtmp = d(ii,jj,kk,e) + dist3d(
     $           xm1(ii,jj,kk,e),ym1(ii,jj,kk,e),zm1(ii,jj,kk,e)
     $          ,xm1(i ,j ,k ,e),ym1(i ,j ,k ,e),zm1(i ,j ,k ,e))
              else
               dtmp = d(ii,jj,kk,e) + dist2d(
     $           xm1(ii,jj,kk,e),ym1(ii,jj,kk,e)
     $          ,xm1(i ,j ,k ,e),ym1(i ,j ,k ,e))
              endif

              if (dtmp.lt.d(i,j,k,e)) then
                d(i,j,k,e) = dtmp
                nchange = nchange+1
                dmax = max(dmax,d(i,j,k,e))
              endif
             enddo
             enddo
             enddo

           enddo
           enddo
           enddo
         enddo
         call gs_op(gsh_fld(ifld),d,1,3,0) ! min over all elements
         nchange = iglsum(nchange,1)
         dmax = glmax(dmax,1)
         if (nio.eq.0) write(6,1) ipass,nchange,dmax,b
    1    format(i9,i12,1pe12.4,' max distance b: ',a3)
         if (nchange.eq.0) goto 1000
      enddo
 1000 return
      end
c-----------------------------------------------------------------------
      subroutine distf(d,ifld,b,dmin,emin,xn,yn,zn)

c     Generate a distance function to boundary with bc "b".
c     This approach does not yet work with periodic boundary conditions.

c     INPUT:  ifld - field type for which distance function is to be found.
c             ifld = 1 for velocity
c             ifld = 2 for temperature, etc.

c     OUTPUT: d = distance to nearest boundary with boundary condition "b"

c     Work arrays:  dmin,emin,xn,yn,zn

      include 'SIZE'
      include 'GEOM'       ! Coordinates
      include 'INPUT'      ! cbc()
      include 'TSTEP'      ! nelfld
      include 'PARALLEL'   ! gather-scatter handle for field "ifld"

      real d(lx1,ly1,lz1,lelt)
      character*3 b

      real dmin(lx1,ly1,lz1,lelt),emin(lx1,ly1,lz1,lelt)
      real xn(lx1,ly1,lz1,lelt),yn(lx1,ly1,lz1,lelt)
      real zn(lx1,ly1,lz1,lelt)


      integer e,eg,f

      nel = nelfld(ifld)
      n = nx1*ny1*nz1*nel

      call domain_size(xmin,xmax,ymin,ymax,zmin,zmax)

      xmn = min(xmin,ymin)
      xmx = max(xmax,ymax)
      if (if3d) xmn = min(xmn ,zmin)
      if (if3d) xmx = max(xmx ,zmax)

      big = 10*(xmx-xmn)
      call cfill (d,big,n)

      call opcopy(xn,yn,zn,xm1,ym1,zm1)

      nface = 2*ndim
      do e=1,nel     ! Set d=0 on walls
      do f=1,nface
         if (cbc(f,e,1).eq.b) call facev(d,e,f,0.,nx1,ny1,nz1)
      enddo
      enddo

      nxyz = nx1*ny1*nz1

      do ipass=1,10000
         dmax    = 0
         nchange = 0
         do e=1,nel
            do k=1,nz1
            do j=1,ny1
            do i=1,nx1
              i0=max(  1,i-1)
              j0=max(  1,j-1)
              k0=max(  1,k-1)
              i1=min(nx1,i+1)
              j1=min(ny1,j+1)
              k1=min(nz1,k+1)
              do kk=k0,k1
              do jj=j0,j1
              do ii=i0,i1

               dself  = d(i,j,k,e)
               dneigh = d(ii,jj,kk,e)
               if (dneigh.lt.dself) then  ! check neighbor's nearest point
                  d2 = (xm1(i,j,k,e)-xn(ii,jj,kk,e))**2
     $               + (ym1(i,j,k,e)-yn(ii,jj,kk,e))**2
                  if (if3d) d2 = d2 + (zm1(i,j,k,e)-zn(ii,jj,kk,e))**2
                  if (d2.gt.0) d2 = sqrt(d2)
                  if (d2.lt.dself) then
                    nchange = nchange+1
                    d (i,j,k,e) = d2
                    xn(i,j,k,e) = xn(ii,jj,kk,e)
                    yn(i,j,k,e) = yn(ii,jj,kk,e)
                    zn(i,j,k,e) = zn(ii,jj,kk,e)
                    dmax = max(dmax,d(i,j,k,e))
                  endif
               endif
              enddo
              enddo
              enddo

            enddo
            enddo
            enddo

            re = lglel(e)
            call cfill(emin(1,1,1,e),re,nxyz)
            call copy (dmin(1,1,1,e),d(1,1,1,e),nxyz)

         enddo
         nchange = iglsum(nchange,1)

         call gs_op(gsh_fld(ifld),dmin,1,3,0) ! min over all elements


         nchange2=0
         do e=1,nel
         do i=1,nxyz
          if (dmin(i,1,1,e).ne.d(i,1,1,e)) then
             nchange2 = nchange2+1
             emin(i,1,1,e) = 0  ! Flag
          endif
         enddo
         enddo
         call copy(d,dmin,n)                !   Ensure updated distance
         nchange2 = iglsum(nchange2,1)
         nchange  = nchange + nchange2
         call gs_op(gsh_fld(ifld),emin,1,4,0) ! max over all elements

         do e=1,nel    ! Propagate nearest wall points
         do i=1,nxyz
          eg = emin(i,1,1,e)
          if (eg.ne.lglel(e)) then
             xn(i,1,1,e) = 0
             yn(i,1,1,e) = 0
             zn(i,1,1,e) = 0
          endif
         enddo
         enddo
         call gs_op(gsh_fld(ifld),xn,1,1,0) !   Sum over all elements to
         call gs_op(gsh_fld(ifld),yn,1,1,0) !   convey nearest point
         call gs_op(gsh_fld(ifld),zn,1,1,0) !   to shared neighbor.

         dmax = glmax(dmax,1)
         if (nio.eq.0) write(6,1) ipass,nchange,dmax
    1    format(i9,i12,1pe12.4,' max wall distance 2')
         if (nchange.eq.0) goto 1000
      enddo
 1000 continue

c     wgt = 0.3
c     call filter_d2(d,nx1,nz1,wgt,.true.)

      return
      end
c-----------------------------------------------------------------------
      subroutine turb_outflow(d,m1,rq,uin)

c     . Set div U > 0 in elements with 'O  ' bc.
c
c     . rq is nominally the ratio of Qout/Qin and is typically 1.5
c
c     . uin is normally zero, unless your flow is zero everywhere 
c
c     . d and m1 are work arrays of size (lx1,ly1,lz1,lelt), assumed persistant
c
c     This routine may or may not work with multiple outlets --- it has
c     not been tested for this case.
c
c
c     TYPICAL USAGE -- ADD THESE LINES TO userchk() in your .usr file:
c                      (uncommented)
c
c     common /myoutflow/ d(lx1,ly1,lz1,lelt),m1(lx1*ly1*lz1,lelt)
c     real m1
c     rq  = 2.
c     uin = 0.
c     call turb_outflow(d,m1,rq,uin)
c

      include 'SIZE'
      include 'TOTAL'

      real d(lx2,ly2,lz2,lelt),m1(lx1*ly1*lz1,lelt)

      parameter (lw = 3*lx1*ly1*lz1)
      common /ctmp1/ w(lw)

      integer icalld,noutf,e,f
      save    icalld,noutf
      data    icalld,noutf /0,0/

      real ddmax,cso
      save ddmax,cso
      logical ifout

      character*3 b

      n     = nx1*ny1*nz1*nelv
      n2    = nx2*ny2*nz2*nelv
      nxyz  = nx1*ny1*nz1
      nxyz2 = nx2*ny2*nz2

      if (icalld.eq.0) then
         icalld = 1

         b = 'O  '
         call cheap_dist(m1,1,b)

         call rzero (d,n2)

         ddmax = 0.
         noutf = 0

         do e=1,nelv
           ifout = .false.
           do f=1,2*ndim
             if (cbc(f,e,1).eq.b) ifout = .true. ! outflow
             if (cbc(f,e,1).eq.b) noutf = noutf+1
           enddo
           if (ifout) then
            if (lx2.lt.lx1) then ! Map distance function to Gauss
             call maph1_to_l2(d(1,1,1,e),nx2,m1(1,e),nx1,if3d,w,lw)
            else
             call copy(d(1,1,1,e),m1(1,e),nxyz)
            endif
            dmax  = vlmax(m1(1,e),nxyz)
            ddmax = max(ddmax,dmax)
            call rzero(m1(1,e),nxyz) ! mask points at outflow
           else
             call rone (m1(1,e),nxyz)
           endif
         enddo

         ddmax = glamax(ddmax,1)

         do e=1,nelv
           ifout = .false.
           do f=1,2*ndim
             if (cbc(f,e,1).eq.b) ifout = .true. ! outflow
           enddo
           if (ifout) then
              do i=1,nxyz2
                d(i,1,1,e) = (ddmax - d(i,1,1,e))/ddmax
              enddo
           endif
         enddo
         noutf = iglsum(noutf,1)
      endif

      if (noutf.eq.0) return

      if (uin.ne.0) then ! Use user-supplied characteristic velocity
         ubar = uin
      else
         ubar = glsc3(vx,bm1,m1,n)   ! Masked average
         vbar = glsc3(vy,bm1,m1,n)
         wbar = glsc3(vz,bm1,m1,n)
         volu = glsc2(bm1,m1,n)
         ubar = abs(ubar)+abs(vbar)
         if (if3d) ubar = abs(ubar)+abs(wbar)
         ubar = ubar/volu
      endif

      cs = 3*(rq-1.)*(ubar/ddmax)
      if (.not.ifsplit) cs = cs*bd(1)/dt

      do i=1,n2
         usrdiv(i,1,1,1) = cs*(d(i,1,1,1)**2)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine add_temp(f2tbc,nbc,npass)

c     add multiple passive scalar fields (npass new ones)

      include 'SIZE'
      include 'TOTAL'

      character*3 f2tbc(2,nbc)

      do i=1,npass
         call add_temp_1(f2tbc,nbc)
      enddo

      igeom = 2
      call setup_topo  ! Set gs handles and multiplicity

      return
      end
c-----------------------------------------------------------------------
      subroutine add_temp_1(f2tbc,nbc)

c
c     TYPICAL USAGE:  Add the below to usrdat().
c
c     parameter (lbc=10) ! Maximum number of bc types
c     character*3 f2tbc(2,lbc)
c
c     f2tbc(1,1) = 'W  '   ! Any 'W  ' bc is swapped to ft2bc(2,1)
c     f2tbc(2,1) = 'I  '
c
c     f2tbc(1,2) = 'v  '   ! Any 'v  ' bc is swapped to ft2bc(2,2)
c     f2tbc(2,2) = 't  '
c
c     nbc = 2      ! Number of boundary condition pairings (e.g., W-->t)
c     do i=1,ldimt-1
c        call add_temp(f2tbc,nbc)
c     enddo

      include 'SIZE'
      include 'TOTAL'
      character*3 f2tbc(2,nbc)

      integer e,f

      nfld=nfield+1

      if (nio.eq.0) write(6,*) 'add temp: ',nfld,nfield,istep

      nelfld(nfld) = nelfld(nfield)
      nel = nelfld(nfield)
      call copy  (bc(1,1,1,nfld),bc(1,1,1,nfield),30*nel)
      call chcopy(cbc(1,1,nfld),cbc(1,1,nfield),3*6*nel)

      do k=1,3
         cpfld(nfld,k)=cpfld(nfield,k)
         call copy (cpgrp(-5,nfld,k),cpgrp(-5,nfield,k),16)
      enddo
      call icopy(matype(-5,nfld),matype(-5,nfield),16)

      param(7) = param(1)  ! rhoCP   = rho
      param(8) = param(2)  ! conduct = dyn. visc

      ifheat       = .true.
      ifadvc(nfld) = .true.
      iftmsh(nfld) = .true.
      ifvarp(nfld) = ifvarp(nfield)
      if (nfld.eq.2) ifto = .true.
      if (nfld.gt.2) ifpsco(nfld-2) = .true.
      if (nfld.gt.2) npscal = npscal+1


      nfield = nfld

      nface = 2*ndim
      do k=1,nbc               ! BC conversion
        do e=1,nelfld(nfld)
        do f=1,nface
          if (cbc(f,e,nfld).eq.f2tbc(1,k)) cbc(f,e,nfld)=f2tbc(2,k)
        enddo
        enddo
      enddo


      return
      end
c-----------------------------------------------------------------------
