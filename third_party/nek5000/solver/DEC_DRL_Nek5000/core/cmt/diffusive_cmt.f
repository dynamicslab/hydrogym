      subroutine imqqtu(ummcu,uminus,uplus)
! Computes (I-0.5*QQT)U for all five conserved variables.
! See call in compute_rhs_and_dt for important documentation
!                                     -
! Spoiler: for SEM this comes out to U -{{U}}, which when
!          spoken is "UMinus Minus the Central flux of U" which I
!          then abbreviate as ummcu
      include 'SIZE'

      real ummcu (nx1*nz1*2*ndim*nelt,toteq) ! intent(out)
      real uminus(nx1*nz1*2*ndim*nelt,toteq) ! intent(in)
      real uplus (nx1*nz1*2*ndim*nelt,toteq) ! intent(in)
      integer ivar

      nf = nx1*nz1*2*ndim*nelt
      const=-0.5
      do ivar=1,toteq
         call add3(ummcu(1,ivar),uminus(1,ivar),uplus(1,ivar),nf)
         call cmult(ummcu(1,ivar),const,nf)        !         -
         call add2(ummcu(1,ivar),uminus(1,ivar),nf)!ummcu = U -{{U}}
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine fluxj_ns_vol(flux,gradu,e,eq)
! viscous flux jacobian for compressible Navier-Stokes equations (NS)
! SOLN and CMTDATA are indexed, assuming vdiff has been filled by uservp
! somehow. In serious need of debugging and replacement.
      include 'SIZE'
      include 'GEOM' ! diagnostic

      parameter (ldd=lx1*ly1*lz1)
      common /ctmp1/ viscscr(lx1,ly1,lz1)
      real viscscr

      integer e,eq
      real flux(nx1*ny1*nz1,ndim),gradu(nx1*ny1*nz1,toteq,ndim)
      integer eijk3(3,3,3)
!     data eijk2 / 0, -1, 1, 0/
      data eijk3
     >/0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0/

      n=nx1*ny1*nz1
      call rzero(flux,n)

! This is a disaster that I might want to program less cleverly
      do j=1,ndim
         do k=1,ndim
            ieijk=0
            if (eq .lt. toteq) ieijk=eijk3(eq-1,j,k) ! does this work in 2D?

            if (ieijk .eq. 0) then
              call agradu_ns_vol(flux(1,j),gradu(1,1,k),viscscr,e,
     >                           eq,j,k)
            endif
         enddo
      enddo

      if (eq.eq.2) then
      pi=4.0*atan(1.0)
      do i=1,nxyz
         x=xm1(i,1,1,e)-5.0
         y=ym1(i,1,1,e)
         r2=x**2+y**2
         write(100,'(6e17.8)') x,y,flux(i,1),10.0*x*y/pi*exp(1-r2),
     >                 flux(i,2),5.0*(y**2-x**2)/pi*exp(1-r2)
      enddo
      elseif (eq .eq. 3) then
      pi=4.0*atan(1.0)
      do i=1,nxyz
         x=xm1(i,1,1,e)-5.0
         y=ym1(i,1,1,e)
         r2=x**2+y**2
         write(200,'(6e17.8)') x,y,
     >                 flux(i,1),5.0*(y**2-x**2)/pi*exp(1-r2),
     >   flux(i,2),-10.0*x*y/pi*exp(1-r2)
      enddo
      endif

      return
      end

!-----------------------------------------------------------------------

      subroutine fluxj_vol(flux,gradu,e,eq)
      include 'SIZE'
      include 'GEOM' ! diagnostic

      integer e,eq
      real flux(nx1*ny1*nz1,ndim),gradu(nx1*ny1*nz1,toteq,ndim)

      n=nx1*ny1*nz1
      do j=1,ndim
         call agradu_vol(flux(1,j),gradu(1,1,j),e,eq,j)
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine agradu_sfc(gijklu,gvar,du,eq)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA'
! subroutine for computing flux of a conserved variable by higher-order
! differential operators.
! This one is classic Navier-Stokes, that is, it computes viscous fluxes
! for everything except gas density.
! eq         index i; LHS equation
! jflux      index j; flux direction
! kdir       index k; direction of derivative or jump in U
      parameter (lxyz=lx1*lz1*2*ldim)
      integer  eq,jflux
      real    du(nx1*nz1*2*ndim*nelt,toteq)
      real gvar(nx1*nz1*2*ndim*nelt,*)    ! intent(in)
! variables making up Gjkil terms, viscous stress tensor and total energy
! equation, compressible Navier-Stokes equations
! assume the following ordering remains in CMTDATA
!     gvar(:,1)  rho ! especially here
!     gvar(:,2)  u   ! especially here
!     gvar(:,3)  v   ! especially here
!     gvar(:,4)  w   ! especially here
!     gvar(:,5)  p
!     gvar(:,6)  T
!     gvar(:,7)  a
!     gvar(:,8)  phi_g
!     gvar(:,9)  rho*cv
!     gvar(:,10) rho*cp
!     gvar(:,11) mu
!     du(1,:)  rho
!     du(2,:)  rho u
!     du(3,:)  rho v
!     du(4,:)  rho w
!     du(5,:)  rho E

      real gijklu(nx1*nz1*2*ndim*nelt)

      npt=lxyz*nelt ! lazy
      call col3(gijklu,du(1,eq),gvar(1,imuf),npt)

      return
      end

!-----------------------------------------------------------------------

      subroutine agradu_vol(gijklu,dut,e,eq,jflux)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA'
! monolithic viscous flux jacobian
! eq         index i; LHS equation and dut variable
! jflux      index j; flux direction
      parameter (lxyz=lx1*ly1*lz1)
      integer  e,eq,jflux
      real   dut(lxyz,toteq)
! derivatives of conserved variables gradu
!     dut(:,1) rho
!     dut(:,2) rho u
!     dut(:,3) rho v
!     dut(:,4) rho w
!     dut(:,5) rho E

      real gijklu(lxyz)

      call col3(gijklu,dut(1,eq),vdiff(1,1,1,e,imu),lxyz)

      return
      end

!-----------------------------------------------------------------------

      subroutine agradu_ns_sfc(gijklu,gvar,du,visco,eq,jflux,kdir)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA'
! subroutine for computing flux of a conserved variable by higher-order
! differential operators.
! This one is classic Navier-Stokes, that is, it computes viscous fluxes
! for everything except gas density.
! eq         index i; LHS equation
! jflux      index j; flux direction
! kdir       index k; direction of derivative or jump in U
      parameter (lxyz=lx1*lz1*2*ldim)
      integer  eq,jflux,kdir
      real    du(lxyz*nelt,toteq)
      real visco(lxyz*nelt) ! you know, you should probably just
                         ! pass mu+lambda and mu-k/cv when eq=5
                         ! so you don't have to recompute them
                         ! so many times
      real gvar(lxyz*nelt,*)    ! intent(in)
! variables making up Gjkil terms, viscous stress tensor and total energy
! equation, compressible Navier-Stokes equations
! assume the following ordering remains in CMTDATA
!     gvar(:,1)  rho ! especially here
!     gvar(:,2)  u   ! especially here
!     gvar(:,3)  v   ! especially here
!     gvar(:,4)  w   ! especially here
!     gvar(:,5)  p
!     gvar(:,6)  T
!     gvar(:,7)  a
!     gvar(:,8)  phi_g
!     gvar(:,9)  rho*cv
!     gvar(:,10) rho*cp
!     gvar(:,11) mu
!     gvar(:,12) thermal conductivity
!     gvar(:,13) lambda
!     gvar(:,18) U5 ! FIX THE DAMN ENERGY COMPUTATION I'M SIGHING ABOUT
! derivatives or jumps, conserved variables, compressible Navier-Stokes
! equations
!     du(1,:)  rho
!     du(2,:)  rho u
!     du(3,:)  rho v
!     du(4,:)  rho w
!     du(5,:)  rho E
      real gijklu(lxyz*nelt) !incremented. never give this exclusive intent
      integer eijk3(3,3,3)
!     data eijk2 / 0, -1, 1, 0/
      data eijk3
     >/0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0/

      if (eq .eq. 1) return

      npt=lxyz*nelt ! lazy

      call rzero(visco,npt)

      if (eq .lt. 5) then

         if (jflux .eq. eq-1) then
            m=kdir
            call copy(visco,gvar(1,ilamf),npt)
            if (kdir .eq. jflux) then
               call add2s2(visco,gvar(1,imuf),2.0,npt)
            endif
         else
            call copy(visco,gvar(1,imuf),npt)
            if (kdir .eq. jflux) then
               m=eq-1
            else
               m=jflux
            endif
         endif

         m=m+1 ! skip density

         call invcol2(visco,gvar(1,irho),npt)
         call subcol4(gijklu,visco,gvar(1,m),du(1,1),npt)
         call addcol3(gijklu,visco,du(1,m),npt)

      else ! energy equation is very different. and could use a rewrite

         if (jflux .eq. kdir) then
            kp1=kdir+1

            l=1 ! sigh
            call vdot3(visco,gvar(1,iux),gvar(1,iux), ! now twoke
     >                       gvar(1,iuy),gvar(1,iuy),
     >                       gvar(1,iuz),gvar(1,iuz),npt)
            do ipt=1,npt ! someone else can make this loop more clever
               gdu=(gvar(ipt,imuf)-gvar(ipt,ikndf))*twoke
               energy=gvar(ipt,icvf)*gvar(ipt,ithm)+0.5*twoke ! sigh. iu5?/phi?
               gdu=gdu+(gvar(ipt,imuf)+gvar(ipt,ilamf))*gvar(ipt,kp1)**2
               gijklu(ipt)=gijklu(ipt)-(gvar(ipt,ikndf)*energy-gdu)*
     >                                  du(ipt,l)/gvar(ipt,irho)
            enddo

            call sub3(visco,gvar(1,imuf),gvar(1,ikndf),npt) ! form mu-K/cv
            do ipt=1,npt
               visco(ipt)=visco(ipt)/gvar(ipt,1)
               gdu=0.0
               do l=2,ldim+1 ! both gvar and du are indexed by l
                  gdu=gdu+gvar(ipt,l)*du(ipt,l)
               enddo
               gijklu(ipt)=gijklu(ipt)+gdu*visco(ipt)
            enddo
            call add3(visco,gvar(1,imuf),gvar(1,ilamf),npt)
            l=jflux+1
            do ipt=1,npt
               gijklu(ipt)=gijklu(ipt)+visco(ipt)*gvar(ipt,l)*du(ipt,l)/
     >                                                      gvar(ipt,1)
            enddo

            l=5
            call copy(visco,gvar(1,ikndf),npt)
            call invcol2(visco,gvar(1,1),npt)
            call add2col2(gijklu,visco,du(1,l),npt)

         else ! dU is off-diagonal

            call add3(visco,gvar(1,imuf),gvar(1,ilamf),npt)
            jp1=jflux+1
            kp1=kdir+1
            call invcol2(visco,gvar(1,1),npt)
            do ipt=1,npt
               gijklu(ipt)=gijklu(ipt)-
     >                  visco(ipt)*gvar(ipt,jp1)*gvar(ipt,kp1)*du(ipt,1)
            enddo

            do l=2,ldim+1
               lm1=l-1
               if (eijk3(jflux,kdir,lm1) .eq. 0) then
                  if (lm1 .eq. kdir) then
                     call copy(visco,gvar(1,ilamf),npt)
                     m=jflux+1
                  else
                     call copy(visco,gvar(1,imuf),npt)
                     m=kdir+1
                  endif
                  call invcol2(visco,gvar(1,1),npt)
                  call addcol4(gijklu,visco,gvar(1,m),du(1,l),npt)
               endif
            enddo ! l

         endif ! diagonal?

      endif ! energy equation

      return
      end

!-----------------------------------------------------------------------

      subroutine agradu_ns_vol(gijklu,dut,visco,e,eq,jflux,kdir)
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'CMTDATA'
! classic Navier-Stokes flux jacobian that computes viscous fluxes
! for everything except gas density.
! eq         index i; LHS equation
! jflux      index j; flux direction
! kdir       index k; direction of derivative or jump in U
      parameter (lxyz=lx1*ly1*lz1)
      integer  e,eq,jflux,kdir
      real   dut(lxyz,toteq)
      real visco(lxyz) ! you know, you should probably just
                         ! pass mu+lambda and mu-k/cv when eq=5
                         ! so you don't have to recompute them
                         ! so many times
! derivatives or jumps, conserved variables, compressible Navier-Stokes
! equations
!     du(1,:) or dut(:,1) rho
!     du(2,:) or dut(:,2) rho u
!     du(3,:) or dut(:,3) rho v
!     du(4,:) or dut(:,4) rho w
!     du(5,:) or dut(:,5) rho E
      real gijklu(lxyz) !incremented. never give this exclusive intent
      integer eijk3(3,3,3)
!     data eijk2 / 0, -1, 1, 0/
      data eijk3
     >/0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0/

      if (eq .eq. 1) return

      npt=lxyz !%s/npt/lxyz/g is hard

      call rzero(visco,npt)

      if (eq .lt. 5) then

         if (jflux .eq. eq-1) then
            m=kdir
            call copy(visco,vdiff(1,1,1,e,ilam),npt)
            if (kdir .eq. jflux) then
               call add2s2(visco,vdiff(1,1,1,e,imu),2.0,npt)
            endif
         else
            call copy(visco,vdiff(1,1,1,e,imu),npt)
            if (kdir .eq. jflux) then
               m=eq-1
            else
               m=jflux
            endif
         endif

         m=m+1 ! skip density

         call invcol2(visco,vtrans(1,1,1,e,irho),npt)
         call addcol3(gijklu,visco,dut(1,m),npt)
! sigh. times like this I hate fortran
         if (m .eq. 2) call subcol4(gijklu,visco,dut,vx(1,1,1,e),npt)
         if (m .eq. 3) call subcol4(gijklu,visco,dut,vy(1,1,1,e),npt)
         if (m .eq. 4) call subcol4(gijklu,visco,dut,vz(1,1,1,e),npt)

      else ! energy equation is very different. and could use a rewrite

         if (jflux .eq. kdir) then
            kp1=kdir+1

            call copy(visco,vx(1,1,1,e),npt)
            call vsq(visco,npt)
            call addcol3(visco,vy(1,1,1,e),vy(1,1,1,e),npt)
            if(if3d) call addcol3(visco,vz(1,1,1,e),vy(1,1,1,e),npt)
! visco now contains uiui=2KE. only happens here
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
            call addcol4(gijklu,visco,vdiff(1,1,1,e,iknd),dut(1,1),npt)
! gdu-=(mu-k/cv)uiui*drho
            call subcol4(gijklu,visco,vdiff(1,1,1,e,imu),dut(1,1),npt)
            call cmult(visco,0.5,npt)
            do ipt=1,npt
                  visco(ipt)=visco(ipt)+
     >            vtrans(ipt,1,1,e,icv)*t(ipt,1,1,e,1)/
     >                             vtrans(ipt,1,1,e,irho) ! sigh
            enddo
! visco now contains E =(cvg*T+0.5*(ux**2+uy**2+uz**2))/rho
! gdu-=k/cv*E*drho/rho
            call subcol4(gijklu,visco,vdiff(1,1,1,e,iknd),dut(1,1),npt)

            call add3(visco,vdiff(1,1,1,e,imu),vdiff(1,1,1,e,ilam),npt)
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
! visco now contains mu+lambda. Much better
! but I've completely given up on doing this through math.f
            if (kp1 .eq. 2) then
               do i=1,npt
                  gijklu(i)=gijklu(i)-visco(i)*vx(i,1,1,e)**2*dut(i,1)
               enddo
            elseif(kp1 .eq. 3) then
               do i=1,npt
                  gijklu(i)=gijklu(i)-visco(i)*vy(i,1,1,e)**2*dut(i,1)
               enddo
            elseif(kp1 .eq. 4) then
               do i=1,npt
                  gijklu(i)=gijklu(i)-visco(i)*vz(i,1,1,e)**2*dut(i,1)
               enddo
            endif

! form mu-K/cv
            call sub3(visco,vdiff(1,1,1,e,imu),vdiff(1,1,1,e,iknd),npt)
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
            call addcol4(gijklu,visco,vx(1,1,1,e),dut(1,2),npt)
            call addcol4(gijklu,visco,vy(1,1,1,e),dut(1,3),npt)
            if(if3d) call addcol4(gijklu,visco,vz(1,1,1,e),dut(1,4),npt)
            call add3(visco,vdiff(1,1,1,e,imu),vdiff(1,1,1,e,ilam),npt)
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
            l=jflux+1
            if (l.eq.2)
     >         call addcol4(gijklu,visco,vx(1,1,1,e),dut(1,l),npt)
            if (l.eq.3)
     >         call addcol4(gijklu,visco,vy(1,1,1,e),dut(1,l),npt)
            if (l.eq.4)
     >         call addcol4(gijklu,visco,vz(1,1,1,e),dut(1,l),npt)

            l=toteq
            call copy(visco,vdiff(1,1,1,e,iknd),npt)
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
            call addcol3(gijklu,visco,dut(1,l),npt)

         else ! dU is off-diagonal

            call add3(visco,vdiff(1,1,1,e,iknd),vdiff(1,1,1,e,ilam),npt)
            jp1=jflux+1
            kp1=kdir+1
            call invcol2(visco,vtrans(1,1,1,e,irho),npt)
! just clobber visco relentlessly
            if (jp1 .eq. 2) call col2(visco,vx(1,1,1,e),npt)
            if (jp1 .eq. 3) call col2(visco,vy(1,1,1,e),npt)
            if (jp1 .eq. 4) call col2(visco,vz(1,1,1,e),npt)
            if (kp1 .eq. 2)
     >         call subcol4(gijklu,visco,vx(1,1,1,e),dut(1,1),npt)
            if (kp1 .eq. 3)
     >         call subcol4(gijklu,visco,vy(1,1,1,e),dut(1,1),npt)
            if (kp1 .eq. 4)
     >         call subcol4(gijklu,visco,vz(1,1,1,e),dut(1,1),npt)

            do l=2,ldim+1
               lm1=l-1
               if (eijk3(jflux,kdir,lm1) .eq. 0) then

                  if (lm1 .eq. kdir) then
                     call copy(visco,vdiff(1,1,1,e,ilam),npt)
                     m=jflux+1
                  else
                     call copy(visco,vdiff(1,1,1,e,imu),npt)
                     m=kdir+1
                  endif

                  call invcol2(visco,vtrans(1,1,1,e,irho),npt)
                  if (m .eq. 2)
     >            call addcol4(gijklu,visco,dut(1,l),vx(1,1,1,e),npt)
                  if (m .eq. 3)
     >            call addcol4(gijklu,visco,dut(1,l),vy(1,1,1,e),npt)
                  if (m .eq. 4)
     >            call addcol4(gijklu,visco,dut(1,l),vz(1,1,1,e),npt)
                endif
            enddo ! l
         endif ! diagonal?

      endif ! energy equation

      return
      end

!-----------------------------------------------------------------------

      subroutine half_iku_cmt(res,diffh,e)
      include 'SIZE'
      include 'MASS'
! diffh has D AgradU. half_iku_cmt applies D^T BM1 to it and increments
! the residual res with the result
      integer e ! lopsided. routine for one element must reference bm1
      real res(nx1,ny1,nz1),diffh(nx1*ny1*nz1,ndim)

      n=nx1*ny1*nz1

      do j=1,ndim
         call col2(diffh(1,j),bm1(1,1,1,e),n)
      enddo

!     const=-1.0 ! I0
      const=1.0  ! *-1 in time march
      call gradm11_t(res,diffh,const,e)

      return
      end

!-----------------------------------------------------------------------

      subroutine compute_transport_props
! get vdiff props (viscosity in imu, second viscosity in ilam, and
! thermal conductivity in iknd; second viscosity is usually -2/3
! visc, but we refuse to assume Stokes' hypothesis for the user)
! via nekasn
! JH082216 Guermond eddy viscosity method (EVM) regularization starts
!          here
      include 'SIZE'
      include 'PARALLEL'
      include 'NEKUSE'
      include 'SOLN'
      include 'CMTDATA'

      integer   e

      do e=1,nelt
         ieg=lglel(e)
         do k=1,nz1
         do j=1,ny1
         do i=1,nx1
            call nekasgn(i,j,k,e)
            call cmtasgn(i,j,k,e)
            call uservp(i,j,k,ieg)
            vdiff(i,j,k,e,imu) = mu
            vdiff(i,j,k,e,ilam) = lambda
            vdiff(i,j,k,e,iknd) = udiff
         enddo
         enddo
         enddo
      enddo
      return
      end
