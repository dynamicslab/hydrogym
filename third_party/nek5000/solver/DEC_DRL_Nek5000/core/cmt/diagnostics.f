      subroutine printalot(e,f,lu,pileoffaces)
      include 'SIZE'
      include 'DG'
      include 'CMTDATA'
      include 'GEOM'
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ xface(lx1*lz1,2*ldim),yface(lx1*lz1,2*ldim),
     >               supapad(6*ldd-4*ldim*lx1*lz1)
      real   pileoffaces(nx1*nz1,2*ndim,nelt,toteq)!,ndim)
      integer   e,f,lu
      real ghere, betahere,pi,kond,h
      pi=4.0*atan(1.0)
      ghere=1.4
      betahere=5.0

      call full2face_cmt(1,nx1,ny1,nz1,iface_flux(1,e),xface,
     >                   xm1(1,1,1,e))
      call full2face_cmt(1,nx1,ny1,nz1,iface_flux(1,e),yface,
     >                   ym1(1,1,1,e))
      nxz=nx1*nz1
      write(lu,*) '# e=',e
      do i=1,nxz
!        x=xface(i,f)-5.0
!        y=yface(i,f)
!        r2=x**2+y**2
!        zeu=-betahere*y*0.5/pi*exp(1.0-r2)
!        zev= betahere*x*0.5/pi*exp(1.0-r2)
!        tau11=2.0*muref*betahere*x*y/pi*exp(1.0-r2)
!        tau12=betahere/pi*(y**2-x**2)*exp(1.0-r2)*muref
!        tau22=-tau11
!        zework1=zeu*tau11+zev*tau12
!        zework2=zeu*tau12+zev*tau22
!        kond=cpgref*muref/prlam
!        tx=0.25*betahere**2/pi/pi*(ghere-1.0)/ghere*exp(2.*(1.0-r2))*x
!        ty=0.25*betahere**2/pi/pi*(ghere-1.0)/ghere*exp(2.*(1.0-r2))*y
!        fluxx=zework1-kond*tx
!        fluxy=zework2-kond*ty
         y=yface(i,f)
         u2=1.0
         h=1.0
         tau12=u2*muref/h
!        write(lu,'(5e17.5)') xface(i,f),yface(i,f),-tau12,
!    >                        pileoffaces(i,f,e,3,1),
!    >                        pileoffaces(i,f,e,3,2)
!    >                        pileoffaces(i,f,e,3,1),-tau11,
!        write(lu,'(4e17.5)') xface(i,f),yface(i,f),
!    >                        pileoffaces(i,f,e,3,1),
!    >                        pileoffaces(i,f,e,3,2)
         write(lu,*) xface(i,f),yface(i,f), pileoffaces(i,f,e,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
! A bunch of diagostic routines
c-----------------------------------------------------------------------
      subroutine matout_rowsum(ab,na,nb)
      integer   na,nb
      real   ab(na,nb)
      character*100 zefmt
      write(zefmt,'(a1,i2,a6)') '(',nb,'e15.7)'
      do i=1,na
         write(37,zefmt) (ab(i,j),j=1,nb)
!        write(6,*) i,'rowsum=',sum(ab(i,1:nb))
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine out_fld_nek(wfnav)
      include 'SIZE'
      include 'SOLN'
      COMMON /solnconsvar/ U(LX1,LY1,LZ1,TOTEQ,lelt) 
      COMMON /SCRNS/      OTVAR(LX1,LY1,LZ1,lelt,7)
      real                OTVAR
      integer e,f

      n = nx1*ny1*nz1
      do e=1,nelt
         call copy(otvar(1,1,1,e,1),u(1,1,1,1,e),n)
         call copy(otvar(1,1,1,e,2),u(1,1,1,2,e),n)
         call copy(otvar(1,1,1,e,3),u(1,1,1,3,e),n)
         call copy(otvar(1,1,1,e,4),u(1,1,1,4,e),n)
         call copy(otvar(1,1,1,e,5),u(1,1,1,5,e),n)
      enddo

c     ifxyo=.true.
      if (lx2.ne.lx1) call exitti('Set LX1=LX2 for I/O$',lx2)

      itmp = 1
      call outpost2(otvar(1,1,1,1,2),otvar(1,1,1,1,3),otvar(1,1,1,1,4)
     $             ,otvar(1,1,1,1,1),otvar(1,1,1,1,5),itmp,'fld')
      return
      end
c----------------------------------------------------------------------
      subroutine out_pvar_nek(wfnav)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA'
      include 'PERFECTGAS'
      COMMON /SCRNS/      OTVAR(LX1,LY1,LZ1,lelt,6)
      real                OTVAR

      integer e,f
      n = nx1*ny1*nz1*nelt
      itmp = 1
      if (lx2.ne.lx1) call exitti('Set LX1=LX2 for I/O$',lx2)

      call outpost2(vx,vy,vz,vtrans(1,1,1,1,irho),t,itmp,'vdt')
      do i=1,n
         ux = vx(i,1,1,1)
         uy = vy(i,1,1,1)
         uz = vz(i,1,1,1)
         cp = vtrans(i,1,1,1,icp)/vtrans(i,1,1,1,irho)
c        cv = vtrans(i,1,1,1,icv)/vtrans(i,1,1,1,irho)
         gma = MixtPerf_G_CpR(cp,rgasref) 
         otvar(i,1,1,1,2) = sqrt(ux*ux+uy*uy+uz*uz)/csound(i,1,1,1)
         otvar(i,1,1,1,3) = phig(i,1,1,1)
         otvar(i,1,1,1,4) = MixtPerf_To_CpTUVW(cp,t(i,1,1,1,1),ux
     $                                    ,uy,uz)
         otvar(i,1,1,1,5) = MixtPerf_Po_GPTTo(gma,pr(i,1,1,1)
     $                            ,t(i,1,1,1,1),otvar(i,1,1,1,4))
      enddo
      call copy(otvar(1,1,1,1,1),vtrans(1,1,1,1,irho),n)
      call outpost2(otvar(1,1,1,1,1),otvar(1,1,1,1,2),otvar(1,1,1,1,3)
     $             ,otvar(1,1,1,1,5),otvar(1,1,1,1,4),itmp,'dmt')
      return
      end
c----------------------------------------------------------------------
      subroutine dumpresidue(wfnav,inmbr)
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'MASS'
      include 'CMTDATA'

      character*32  wfnav
      character*32  citer,citer2
      integer e,i1,is,il,i,inmbr,length,eq,is2,il2
      real    rhseqs(toteq)

c     write(6,*)wfnav,inmbr
      i1 =index(wfnav,' ')-1
      write(citer,*)inmbr
      write(citer2,*)nid
      length = len(wfnav)
      do i=1,length
         if(citer(i:i).ne.' ')then
         is=i
c        exit ! Not valid w/ pgf77
         endif
      enddo
      do i=length,1,-1 
         if(citer(i:i).ne.' ')then
         il=i
c        exit ! Not valid w/ pgf77
         endif
      enddo
!     get the string that contains character nid
      do i=1,length
         if(citer2(i:i).ne.' ')then
         is2=i
c        exit ! Not valid w/ pgf77
         endif
      enddo
      do i=length,1,-1 
         if(citer2(i:i).ne.' ')then
         il2=i
c        exit ! Not valid w/ pgf77
         endif
      enddo
      nxyz1 = nx1*ny1*nz1
      nxy1  = nx1*ny1
      open(unit=11,file=wfnav(1:i1)//'.'//'it='//citer(is:il)//
     $                     '.'//'p='//citer2(is2:il2))
      write(11,*)'Title="',wfnav(1:i1),'"'
c      write(6,*)wfnav(1:i1),'.',citer(is:il)
      do i=1,length
        wfnav(i:i)=' '
      enddo
      do i=1,length
        citer(i:i)=' '
      enddo
      if(if3d)then
        write(11,*)'Variables=x,y,z,e1,e2,e3,e4,e5'
        do e = 1,nelt
          write(11,*)'zone T="',e,'",i=',nx1,',j=',ny1,',k=',nz1
          do i=1,nxyz1
             do eq=1,toteq
                rhseqs(eq) = res1(i,1,1,e,eq)/bm1(i,1,1,e)
             enddo
          write(11,101)xm1(i,1,1,e),ym1(i,1,1,e),zm1(i,1,1,e)
     $         ,rhseqs(1),rhseqs(2),rhseqs(3),rhseqs(4)
     $         ,rhseqs(5)
          enddo
        enddo
      else
        write(11,*)'Variables=x,y,e1,e2,e3,e4,e5'
        do e = 1,nelt
          write(11,*)'zone T="',e,'",i=',nx1,',j=',ny1
          do i=1,nxy1
             do eq=1,toteq
                rhseqs(eq) = res1(i,1,1,e,eq)/bm1(i,1,1,e)
             enddo
          write(11,102)xm1(i,1,1,e),ym1(i,1,1,e)
     $         ,rhseqs(1),rhseqs(2),rhseqs(3),rhseqs(4)
     $         ,rhseqs(5)
          enddo
        enddo
      endif
      close(11)
101   format(8(3x,e12.5))
102   format(7(3x,e14.7))
      return
      end
c----------------------------------------------------------------------
      subroutine mass_balance(if3d)
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'MASS'
      INCLUDE 'TSTEP'
      COMMON /solnconsvar/ U(LX1,LY1,LZ1,TOTEQ,lelt) 
      logical if3d
      integer e
      real msum,total_mass
c     First get the mass in the local domain. Then use
c     Global sum to get the total mass
c     mass = \rho_g \phi_g * Wts(i,j,k)
c     Note -  this is only for single processor runs
c          need to add glsum to make it parallel
      if (if3d)then
         msum = 0.0
         do e=1,nelt
            il = 0
            do k=1,nz1
               do j=1,ny1
                  do i=1,nx1
                     il = il +1
                     msum = msum + (u(i,j,k,1,e)*bm1(i,j,k,e))
                  enddo
               enddo
            enddo
         enddo
      else
         msum = 0.0
         do e=1,nelt
            il = 0
            do j=1,ny1
            do i=1,nx1
               il = il +1
               msum = msum + (u(i,j,1,1,e)*bm1(i,j,1,e))
            enddo
            enddo
         enddo
      endif
      total_mass = glsum(msum,1) 
      if(nio.eq.0)
c    $   write(6,*)'Total mass in the domain ',total_mass
     $   write(144,*)'Time ',time,'Mass ',total_mass
      return
      end
c-----------------------------------------------------------------------
! That was a bunch of diagostic routines
c-----------------------------------------------------------------------
      subroutine torque_calc_cmt(scale,x0,ifdout,iftout)
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
      common /scrns/         sij (lx1*ly1*lz1*6*lelt)
      common /scrcg/         pm1 (lx1,ly1,lz1,lelt)
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
      common /ICPVARS/ pvars(lx1,ly1,lz1,7)
      real             pvars
c
      n = nx1*ny1*nz1*nelv
      ntot1 = nx1*ny1*nz1
c
c     call mappr(pm1,pr,xm0,ym0) ! map pressure onto Mesh 1
c     IN CMT-Nek Pressure is defined on the velocity grid
      do ie=1,nelv 
         call copy(pm1(1,1,1,ie),pr(1,1,1,ie),ntot1)
      enddo
c
c    Add mean_pressure_gradient.X to p:

c    In CMT-Nek we do not fix the volume flow rate
c     if (param(55).ne.0) then
c        dpdx_mean = -scale_vf(1)
c        dpdy_mean = -scale_vf(2)
c        dpdz_mean = -scale_vf(3)
c     endif

c     call add2s2(pm1,xm1,dpdx_mean,n)  ! Doesn't work if object is cut by 
c     call add2s2(pm1,ym1,dpdy_mean,n)  ! periodicboundary.  In this case,
c     call add2s2(pm1,zm1,dpdz_mean,n)  ! set ._mean=0 and compensate in
c
c    Compute sij
c
      nij = 3
      if (if3d.or.ifaxis) nij=6
c     we store primivtive variables only on dealiased grid. We will modify
c     comp_sij such that primvars are stored on the grid element
c     by element in a scratch array. 
      call comp_sij_cmt(sij,nij,ur,us,ut,vr,vs,vt,wr,ws,wt)
c
c
c     Fill up viscous array w/ default
      param(2) = 0.0
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
      subroutine comp_sij_cmt(sij,nij,ur,us,ut,vr,vs,vt,wr,ws,wt)
c                                       du_i       du_j
c     Compute the stress tensor S_ij := ----   +   ----
c                                       du_j       du_i
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e
c
      real sij(lx1*ly1*lz1,nij,lelt)
      real ur (1) , us (1) , ut (1)
     $   , vr (1) , vs (1) , vt (1)
     $   , wr (1) , ws (1) , wt (1)

      real j ! Inverse Jacobian

      common /ICPVARS/ pvars(lx1,ly1,lz1,7)
      real             pvars

      n    = nx1-1      ! Polynomial degree
      nxyz = nx1*ny1*nz1

      if (if3d) then     ! 3D CASE
       do e=1,nelv
          call copy(pvars(1,1,1,1),vx(1,1,1,e),nxyz)
          call copy(pvars(1,1,1,2),vy(1,1,1,e),nxyz)
          call copy(pvars(1,1,1,3),vz(1,1,1,e),nxyz)
        call local_grad3(ur,us,ut,pvars,N,1,dxm1,dxtm1)
        call local_grad3(vr,vs,vt,pvars,N,2,dxm1,dxtm1)
        call local_grad3(wr,ws,wt,pvars,N,3,dxm1,dxtm1)

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
            call copy(pvars(1,1,1,1),vx(1,1,1,e),nxyz)
            call copy(pvars(1,1,1,2),vy(1,1,1,e),nxyz)
            call copy(pvars(1,1,1,3),vz(1,1,1,e),nxyz)
            call setaxdy ( ifrzer(e) )  ! change dytm1 if on-axis
            call local_grad2(ur,us,pvars,N,1,dxm1,dytm1)
            call local_grad2(vr,vs,pvars,N,2,dxm1,dytm1)
            call local_grad2(wr,ws,pvars,N,3,dxm1,dytm1)

            do i=1,nxyz
               j = jacmi(i,e)
               r = ym1(i,1,1,e)                              ! Cyl. Coord:

               sij(i,1,e) = j*  ! du/dx + du/dx              ! e_zz
     $           2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e))

               sij(i,2,e) = j*  ! dv/dy + dv/dy              ! e_rr
     $           2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))

               if (r.gt.0) then                              ! e_@@
                  sij(i,3,e) = pvars(i,1,1,1)/r  ! v / r 
               else
                  sij(i,3,e) = j*  ! L'Hopital's rule: e_@@ = dv/dr
     $            2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))
               endif

               sij(i,4,e) = j*  ! du/dy + dv/dx             ! e_zr
     $            ( ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) +
     $              vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) )

               if (yyyr.gt.0) then                             ! e_r@
                  sij(i,5,e) = j*  ! dw/dy 
     $              ( wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e) )
     $              - pvars(i,1,1,3) / r
               else
                  sij(i,5,e) = 0
               endif

               sij(i,6,e) = j*  ! dw/dx                     ! e_@z
     $            ( wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e) )

            enddo
         enddo

      else              ! 2D CASE

         do e=1,nelv
            call copy(pvars(1,1,1,1),vx(1,1,1,e),nxyz)
            call copy(pvars(1,1,1,2),vy(1,1,1,e),nxyz)
            call local_grad2(ur,us,pvars,N,1,dxm1,dxtm1)
            call local_grad2(vr,vs,pvars,N,2,dxm1,dxtm1)

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
