      subroutine compute_entropy(s)
! computes entropy at istep and pushes the stack down for previous
! steps needed to compute ds/dt via finite difference (for now).
! hardcoded for Burgers equation. More later when folded into CMT-nek
! for Burgers, s=energy=1/2 U^2
      include 'SIZE'
      include 'TOTAL'  ! tlag is lurking. be careful
      include 'CMTDATA'
! I've always seen lorder=3, but I still need three steps
!          s(:,               1,       1)  ! entropy at current step
!          s(:,               2,       1)  ! entropy at step n-1
!          s(:,               1,       2)  ! entropy at step n-2
      real s(lx1*ly1*lz1*lelt,lorder-1,*)
      real ntol
      integer e
      data icalld /0/
      save icalld

      n=lx1*ly1*lz1
      ntot=n*nelt
      ntol=1.0e-10

      if (icalld .eq. 0) then
         if (nio .eq. 0) write(6,*) 'zeroing out entropy stack',istep
         icalld=1
         call rzero(s,ntot)
         call rzero(s(1,1,2),ntot) ! s_{n-1}
         call rzero(s(1,2,1),ntot) ! s_n
      endif

! compute the current entropy. This actually needs to go back in the
! usr file because it's EOS-dependent
      rgam=rgasref/(gmaref-1.0)
      do i=1,ntot
         rho=max(vtrans(i,1,1,1,irho),ntol)
         s(i,1,1)=rgam*rho*log(pr(i,1,1,1)/(rho**gmaref))
      enddo

      if (stage .eq. 1) then
! push the stack
         call copy(s(1,1,2),s(1,2,1),ntot) ! s_{n-1}=s_n
         call copy(s(1,2,1),s(1,1,1),ntot) ! fill s_n
      endif

      return
      end

!-----------------------------------------------------------------------

      subroutine entropy_viscosity
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ scrent(lxyz,lelt)
      integer e
      character*132 deathmessage

      pi=4.0*atan(1.0)

      n=lx1*ly1*lz1
      ntot=n*nelt

! entropy at this and lorder prior steps
      call compute_entropy(tlag)
! compute maxval(|S-<S>|)
      savg    =    glsc2(tlag,bm1,ntot)
      savg    = -savg/volvm1
      call cadd2(scrent,tlag,savg,ntot)
      maxdiff =     glamax(scrent,ntot)
      if (maxdiff.le.0.0) then
         write(6,*) 'zero maxdiff usually means NAN$'
!        write(deathmessage,*) 'zero maxdiff usually means NAN$'
!        call exittr(deathmessage,maxdiff,istep)
      endif
      call entropy_residual(tlag) ! fill res2
      call copy(res2(1,1,1,1,2),res2,ntot) ! raw residual in res2
      call wavevisc(t(1,1,1,1,3))
      call resvisc(res2) ! overwrite res2
      call evmsmooth(res2,t(1,1,1,1,3),.true.) ! endpoints=.false.
                                               ! is intended to
                                               ! preserve face states,
                                               ! but this is easier to
                                               ! test 1D
!     call evmsmooth(res2,t(1,1,1,1,3),.true.) ! And again.
      call dsavg(res2) ! you DEFINITELY don't want a min here


      return
      end

!-----------------------------------------------------------------------

      subroutine piecewiseAV(shock_detector)
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      integer e
      character*132 deathmessage
      common /scrvh/ avmask(lelt)
      real avmask
      external shock_detector

      pi=4.0*atan(1.0)

      nxyz=lx1*ly1*lz1
      ntot=nxyz*nelt

! toggle shock detector with AV application Lv, See & Ihme (2016) JCP 322

      if (time4av) then
         call shock_detector(avmask)

! old nu_max
         call wavevisc(t(1,1,1,1,3))
! diagnostic
!        if (stage.gt.0)then
!        do e=1,nelt
!!           do i=1,nxyz
!              write(stage*100+nid,*)xm1(i,1,1,e),ym1(i,1,1,e),t(i,1,1,e,3)
!           enddo
!        enddo
!        endif
         do e=1,nelt
            call cmult(t(1,1,1,e,3),avmask(e),nxyz)
            if (avmask(e).ne.1.0) write(6,*) 'duh sir'
         enddo

      else
         call rzero(t(1,1,1,1,3),ntot)
!        call shock_detector(avmask)
      endif

! diagnostic
!      if (stage.gt.0)then
!      do e=1,nelt
!         do i=1,nxyz
!            write(stage*1000+nid,'(5e15.7)') xm1(i,1,1,e),ym1(i,1,1,e),
!!     >        t(i,1,1,e,3),avmask(e),epsebdg(e)
!         enddo
!      enddo
!      endif

!     call max_to_trilin(t(1,1,1,1,3))

      return
      end

c-----------------------------------------------------------------------

      subroutine entropy_residual(s)
! COMPUTE R=ds/dt + div F(s) for entropy pair s and (hardcoded) F
! and store its norm functional thing |R| in res2 where it will
! provide artificial viscosity according to the code in
! entropy_viscosity and the method of Guermond
      include 'SIZE'
      include 'TOTAL' ! tlag lurks
      include 'CMTDATA'
      integer e
      real dsdtcoef(3) ! put this in /TIMESTEPCOEF/ someday
      data dsdtcoef /1.0,1.0,0.5/
      real s(lx1*ly1*lz1*lelt,lorder-1,ldimt) ! because it's really tlag

      n=lx1*ly1*lz1
      ntot=n*nelt

      if (istep .eq. 1) return
      rdt=1.0/(dsdtcoef(stage)*dt)
      if (stage .eq. 1) then ! THE MOST SABOTAGEABLE PART OF THE CODE
         call sub3(res2,s(1,1,1),s(1,1,2),ntot) ! EVALUATE s_n-s_{n-1}
      else
         call sub3(res2,s(1,1,1),s(1,2,1),ntot) ! EVALUATE s^{(stage)}-s_n
      endif
      call cmult(res2,rdt,ntot)
! res2=ds/dt. now,

!-----------------------------------------------------------------------
! cons approach: strong-conservation form of flux divergence in entropy residual
!-----------------------------------------------------------------------
! get around to expanding totalh to store fluxes for whole fields and
! properly vectorize evaluate_*_h and flux_div_integral
      do e=1,nelt
         call evaluate_entropy_flux(e) ! diffh. zero it before diffusion 
         call flux_div_mini(e) ! into res2 it goes.
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine evaluate_entropy_flux(e)
! entropy flux function for entropy residual.
! just vel*s for now
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'CMTDATA'
      integer e

      call rzero(totalh,3*lxd*lyd*lzd)
      n=lx1*ly1*lz1

      call col3(totalh(1,1),vx(1,1,1,e),tlag(1,1,1,e,1,1),n)
      call col3(totalh(1,2),vy(1,1,1,e),tlag(1,1,1,e,1,1),n)
      if (if3d) call col3(totalh(1,3),vz(1,1,1,e),tlag(1,1,1,e,1,1),n)

      return
      end

c-----------------------------------------------------------------------

      subroutine flux_div_mini(e)
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'DXYZ'
      include 'SOLN'
      include 'CMTDATA'
      parameter (ldd=lx1*ly1*lz1)
      parameter (ldg=lx1**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju

      integer e

      nrstd=ldd
      nxyz=lx1*ly1*lz1
      mdm1=lx1-1

      call rzero(ur,nrstd)
      call rzero(us,nrstd)
      call rzero(ut,nrstd)
      call rzero(ud,nrstd)
      call rzero(tu,nrstd)

      if (if3d) then
         call local_grad3(ur,us,ut,totalh(1,1),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)
     $                        + sxm1(i,1,1,e)*us(i)
     $                        + txm1(i,1,1,e)*ut(i))
         enddo
         call local_grad3(ur,us,ut,totalh(1,2),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz ! confirmed to have no effect in 1D
            ud(i)=ud(i)+jacmi(i,e)*(rym1(i,1,1,e)*ur(i)
     $                            + sym1(i,1,1,e)*us(i)
     $                            + tym1(i,1,1,e)*ut(i))
         enddo
         call local_grad3(ur,us,ut,totalh(1,3),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz ! confirmed to have no effect in 1D
            ud(i)=ud(i)+jacmi(i,e)*(rzm1(i,1,1,e)*ur(i)
     $                            + szm1(i,1,1,e)*us(i)
     $                            + tzm1(i,1,1,e)*ut(i))
         enddo
      else
         call local_grad2(ur,us,totalh(1,1),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)
     $                        + sxm1(i,1,1,e)*us(i))
         enddo
         call local_grad2(ur,us,totalh(1,2),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i)=ud(i)+jacmi(i,e)*(rym1(i,1,1,e)*ur(i)
     $                            + sym1(i,1,1,e)*us(i))
         enddo
      endif
      call add2(res2(1,1,1,e,1),ud,nxyz)

      return
      end

!-----------------------------------------------------------------------

      subroutine resvisc(residual)
! smooth residual-based entropy visc, defined by Guermond, Popov, whoever
! DSAVG assumes IFFLOW to work correctly. IFHEAT still doesn't work with CMT-nek
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'DG'
      integer lfq,heresize,hdsize
      parameter (lxyz=lx1*ly1*lz1)
      real residual(lxyz,nelt)
      integer e,f

      nxyz =lx1*ly1*lz1
      nxz  =lx1*lz1
      nface=2*ldim
      nxzf =nxz*nface
      nfq  =nxzf*nelt

! ensure continuity at faces. doing this before |abs| causes some cancellation that,
! so far, appears to beneficially reduce spikiness at faces.

      do e=1,nelt
         do i=1,nxyz
            residual(i,e)=residual(i,e)*meshh(e)**2
         enddo
      enddo

      call dsavg(residual) ! signed, can cancel at faces. Hope it does

      do e=1,nelt
         do i=1,nxyz
            residual(i,e)=abs(residual(i,e))
         enddo
      enddo

      call cmult(residual,c_sub_e,nxyz*nelt)

      if (maxdiff .ne. 0) then
         const=1.0/maxdiff
         call cmult(residual,const,nxyz*nelt)
      endif

      return
      end

!-----------------------------------------------------------------------

      subroutine evmsmooth(resvisc,wavevisc,endpoints)
      include 'SIZE'
      include 'INPUT'
      real resvisc(lx1,ly1,lz1,nelt),wavevisc(lx1,ly1,lz1,nelt)
      real rtmp
      common /ctmp1/ rtmp(lx1,ly1,lz1)
! are faces included in smoothing? if not (say they're fixed by dsavg) then
! endpoints should be .false.
      logical endpoints
      integer e
! clip residual viscosity and smooth it.
! actually just smooth first and then clip it. Just average-with-my-neighbors for now
! weight the neighbors according to GLL-spacing instead of uniform-grid
! at a later date.

      nxyz=lx1*ly1*lz1
      kstart=1
      kend=lz1
      rldim=1.0/ldim

      if (endpoints) then
         istart=1
         jstart=1
         iend=lx1
         jend=ly1
      else
         istart=2
         jstart=2
         iend=lx1-1
         jend=ly1-1
         if (if3d) then
            kstart=2
            kend=lz1-1
         endif
      endif

! yes I know this loop sucks and I need to write a matrix or something
      do e=1,nelt
         do i=1,nxyz
            if (wavevisc(i,1,1,e).le. resvisc(i,1,1,e))
     >          resvisc(i,1,1,e)= wavevisc(i,1,1,e)
         enddo
         call copy(rtmp,resvisc(1,1,1,e),nxyz) ! really only for .false.
         do iz=kstart,kend
            if (if3d) then
               km1=iz-1
               kp1=iz+1
               izm=km1
!              if (km1 .lt. 1) izm=iz ! bias towards {{face point}}
               if (km1 .lt. 1) izm=kp1 ! Guermond symmetry
               izp=kp1
!              if (kp1 .gt. lz1) izp=iz ! bias towards {{face point}}
               if (kp1 .gt. lz1) izp=km1 ! Guermond symmetry
            else
               izm=iz
               izp=iz
            endif
            do iy=jstart,jend
               jm1=iy-1
               jp1=iy+1
               iym=jm1
!              if (jm1 .lt. 1) iym=iy ! bias towards {{face point}}
               if (jm1 .lt. 1) iym=jp1 ! Guermond symmetry
               iyp=jp1
!              if (jp1 .gt. ly1) iyp=iy ! bias toward {{face point}}
               if (jp1 .gt. ly1) iyp=jm1 ! Guermond symmetry
               do ix=istart,iend
                  im1=ix-1
                  ip1=ix+1
                  ixm=im1
!                 if (im1 .lt. 1) ixm=ix ! bias towards {{face point}}
                  if (im1 .lt. 1) ixm=ip1 ! Guermond symmetry
                  ixp=ip1
!                 if (ip1 .gt. lx1) ixp=ix ! bias towards {{face point}}
                  if (ip1 .gt. lx1) ixp=im1 ! Guermond symmetry
                  x0 = resvisc(ix ,iy ,iz ,e)
                  x1 = resvisc(ixm,iy ,iz ,e)
                  x2 = resvisc(ixp,iy ,iz ,e)
                  x3 = resvisc(ix ,iym,iz ,e)
                  x4 = resvisc(ix ,iyp,iz ,e)
                  if (if3d) then
                     x5 = resvisc(ix ,iy ,izm,e)
                     x6 = resvisc(ix ,iy ,izp,e)
                  else
                     x5=0.0
                     x6=0.0
                  endif
                  rtmp(ix,iy,iz)=0.25*(2.0*ldim*x0+x1+x2+x3+x4+x5+x6)
     >                               *rldim
               enddo
            enddo
         enddo
         call copy(resvisc(1,1,1,e),rtmp,nxyz)
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine wavevisc(numax)
! max entropy visc, defined by Guermond, Popov, whoever (chapter-verse)
! as 
! numax = c_max*h*max(dH/dU)
! in a given element
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ wavespeed(lxyz)
      real wavespeed
      real maxeig
      real numax(lxyz,nelt)
      integer e

      nxyz=lx1*ly1*lz1

      do e=1,nelt
         do i=1,nxyz
            wavespeed(i)=csound(i,1,1,e)+
     >      sqrt(vx(i,1,1,e)**2+vy(i,1,1,e)**2+vz(i,1,1,e)**2)
         enddo
         maxeig=vlamax(wavespeed,nxyz)
! Zingan (2015) only. not long for this world
!        rhomax(e)=vlamax(vtrans(1,1,1,e,irho),nxyz)
         do i=1,nxyz
            numax(i,e)=c_max*maxeig*meshh(e)
         enddo
      enddo

      call max_to_trilin(t(1,1,1,1,3))

      return
      end

!-----------------------------------------------------------------------

      subroutine max_to_trilin(field)
! stupid subroutine to take a stupid uniform field and compute a trilinear
! tent between maximum shared values at the vertices.
      include 'SIZE'
      include 'TOTAL'
      real field(lx1,ly1,lz1,nelt)
      integer e

      nxyz=lx1*ly1*lz1

! get maxima on faces
      call dsop(field,'MAX',lx1,ly1,lz1)

! trilinear interpolation. you should adapt xyzlin to your needs instead
      do e=1,nelt
         p000=field(1,  1,  1,  e)
         p100=field(lx1,1,  1,  e)
         p010=field(1,  ly1,1,  e)
         p110=field(lx1,ly1,1,  e)
         p001=field(1,  1,  lz1,e)
         p101=field(lx1,1,  lz1,e)
         p011=field(1,  ly1,lz1,e)
         p111=field(lx1,ly1,lz1,e)
         c1=p100-p000
         c2=p010-p000
         c3=p001-p000
         c4=p110-p010-p100+p000
         c5=p011-p001-p010+p000
         c6=p101-p001-p100+p000
         c7=p111-p011-p101-p110+p100+p001+p010-p000
         rdx=1.0/(xm1(lx1,1,1,e)-xm1(1,1,1,e)) ! cubes only!!!
         rdy=1.0/(ym1(1,ly1,1,e)-ym1(1,1,1,e))
         rdz=0.0
         if(if3d) rdz=1.0/(zm1(1,1,lz1,e)-zm1(1,1,1,e))
         do i=1,nxyz
            deltax=rdx*(xm1(i,1,1,e)-xm1(1,1,1,e)) ! cubes only!!!
            deltay=rdy*(ym1(i,1,1,e)-ym1(1,1,1,e))
            deltaz=0.0
            if (if3d) deltaz=rdz*(zm1(i,1,1,e)-zm1(1,1,1,e))
            field(i,1,1,e)=p000+c1*deltax+c2*deltay+c3*deltaz+
     >                          c4*deltax*deltay+c5*deltay*deltaz+
     >                       c6*deltaz*deltax+c7*deltay*deltaz*deltax
         enddo
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine perssonperaire(shkdet,var,shtmp)
! Peraire & Persson (2006) Eq. 7
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ fvar (lxyz,lelt)
!    $ ,             ytm1 (lx1,ly1,lz1,lelv)
      real shkdet(lxyz,nelt),var(lxyz,*),shtmp(lxyz,*)
      integer e
! this whole common block is atrocious
      common /CMTFILTERS/  intv(lx1,lx1),intt(lx1,lx1)
     $                  ,  intvd(lxd,lxd),inttd(lxd,lxd)
     $                  ,  wk1(lx1,lx1,lz1),wk2(lx1,lx1,lz1)
     $                  ,  wkd1(lxd,lxd,lzd),wkd2(lxd,lxd,lzd)
      real                 intv, intt, intvd, inttd
      real kappa
      integer icalld
      save    icalld
      data    icalld/0/

      if (icalld.eq.0) then
         ncut=-110
         icalld=1
         call userfilt(intv,zgm1,lx1,nio)
      endif

      n=lx1*ly1*lz1
      ntot=n*nelt

! JH082418 Hardcoded threshold
      cpp=0.001

      call copy(fvar,var,ntot)
      call filterq(fvar,intv,lx1,lz1,wk1,wk2,intt,if3d,dmax)

      do e=1,nelt
         call sub3(shkdet(1,e),var(1,e),fvar(1,e),n) ! store u-u^ in shkdet
         shock_or_not=vlsc3(shkdet(1,e),shkdet(1,e),bm1(1,1,1,e),n)
         denom=vlsc3(var(1,e),var(1,e),bm1(1,1,1,e),n)
         if (denom .gt. 0.0) then ! mask this somehow. ifs are bad
            shock_or_not=(shock_or_not/denom)*(lx1-1)**4
!           shock_or_not=log10(shock_or_not)
            if (shock_or_not .gt. cpp) then
!              call cfill(shkdet(1,e),shock_or_not,n)
               call rone(shkdet(1,e),n)
            else
               call rzero(shkdet(1,e),n)
            endif
         else
            write(6,*) 'nid,iel=',nid,e
            write(6,*) 'variables that change sign are bad for shox'
            call exitt
         endif
      enddo

      return
      end
