      subroutine compute_primitive_vars
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'CMTDATA'
      include 'SOLN'
      include 'DEALIAS' ! until we are comfortable with setup_convect

      parameter (lxyz=lx1*ly1*lz1)
      common /ctmp1/ energy(lx1,ly1,lz1),scr(lx1,ly1,lz1)
      integer e, eq

      nxyz= nx1*ny1*nz1
      ntot=nxyz*nelt

      do e=1,nelt
         call invcol3(vx(1,1,1,e),u(1,1,1,irpu,e),u(1,1,1,irg,e),nxyz)
         call invcol3(vy(1,1,1,e),u(1,1,1,irpv,e),u(1,1,1,irg,e),nxyz)
!        if (if3d)
         call invcol3(vz(1,1,1,e),u(1,1,1,irpw,e),u(1,1,1,irg,e),nxyz)
! first kinetic energy
         if (if3d) then
            call vdot3(scr,
     >             u(1,1,1,irpu,e),u(1,1,1,irpv,e),u(1,1,1,irpw,e),
     >             u(1,1,1,irpu,e),u(1,1,1,irpv,e),u(1,1,1,irpw,e),nxyz)
         else
            call vdot2(scr,u(1,1,1,irpu,e),u(1,1,1,irpv,e),
     >                     u(1,1,1,irpu,e),u(1,1,1,irpv,e),nxyz)
         endif
         call invcol2(scr,u(1,1,1,irg,e),nxyz)
         call cmult(scr,0.5,nxyz)
! then to internal energy
         call sub3(energy,u(1,1,1,iret,e),scr,nxyz)
! now mass-specific
         call invcol2(energy,u(1,1,1,irg,e),nxyz)
! don't forget to get density where it belongs
         call invcol3(vtrans(1,1,1,e,irho),u(1,1,1,irg,e),phig(1,1,1,e),
     >                nxyz)
         call tdstate(e,energy)
      enddo

! setup_convect has the desired effect
! if IFPART=F
! if IFCHAR=F
! if IFCONS=T
! if igeom .ne. 1
! if param(99) .ge. 0
!-----------------------------------------------------------------------
!     call setup_convect(0)
!-----------------------------------------------------------------------
! to make life easier until we master this stuff and harmonize even better with nek,
! I'm including 'DEALIAS' and calling set_convect_cons here
      if (nxd.gt.nx1) then
         call set_convect_cons (vxd,vyd,vzd,vx,vy,vz)
      else
         call copy(vxd,vx,ntot) 
         call copy(vyd,vy,ntot) 
         call copy(vzd,vz,ntot) 
      endif

      return
      end
!-----------------------------------------------------------------------

      subroutine tdstate(e,energy)!,energy)
c compute the gas properties. We will have option to add real gas models
c We have perfect gas law. Cvg is stored full field
      include 'SIZE'
      include 'CMTDATA'
      include 'SOLN'
      include 'PARALLEL'
      include 'NEKUSE'
      integer   e,eg
      real energy(nx1,ny1,nz1)

      eg = lglel(e)
      do k=1,nz1
      do j=1,ny1
      do i=1,nx1
         call nekasgn(i,j,k,e)
         call cmtasgn(i,j,k,e)
         e_internal=energy(i,j,k) !nekasgn should do this, but can't
         call cmt_userEOS(i,j,k,eg)
         vtrans(i,j,k,e,icp)= cp*rho
         vtrans(i,j,k,e,icv)= cv*rho
         t(i,j,k,e,1)       = temp
         pr(i,j,k,e)        = pres
         csound(i,j,k,e)    = asnd
      enddo
      enddo
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine cmtasgn (ix,iy,iz,e)
      include 'SIZE'
      include 'SOLN'
      include 'CMTDATA'
      include 'NEKUSE'

      integer e,eqnum

      do eqnum=1,toteq
         varsic(eqnum)=u(ix,iy,iz,eqnum,e)  
      enddo
      phi  = phig  (ix,iy,iz,e)
      rho  = vtrans(ix,iy,iz,e,irho)
      pres = pr    (ix,iy,iz,e)
      if (rho.ne.0) then
         cv   = vtrans(ix,iy,iz,e,icv)/rho
         cp   = vtrans(ix,iy,iz,e,icp)/rho
      endif
      asnd = csound(ix,iy,iz,e)
      mu     = vdiff(ix,iy,iz,e,imu)
      udiff  = vdiff(ix,iy,iz,e,iknd)
! MAKE SURE WE''RE NOT USING UTRANS FOR ANYTHING IN pre-v16 code!!
      lambda = vdiff(ix,iy,iz,e,ilam)

      return
      end

!-----------------------------------------------------------------------

      subroutine cmt_ics
! overlaps with setics. -DCMT will require IFDG as well
      include 'SIZE'
      include 'SOLN'
      include 'GEOM'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'NEKUSE'
      nxyz2=nx2*ny2*nz2       ! Initialize all fields:
      ntot2=nxyz2*nelv
      nxyz1=nx1*ny1*nz1
      ntott=nelt*nxyz1
      ntotv=nelv*nxyz1
      ltott=lelt*nxyz1
      ntotcv=lelt*nxyz1*toteq
      call rzero(phig,ltott)
      call rzero(csound,ltott)
      call rzero(vtrans,ltott*ldimt1)
      call rzero(vdiff ,ltott*ldimt1)
      call rzero(u,ntotcv)
      call cmtuic
!     call nekgsync()
!     call restart(nfiles) !  Check restart files. soon...
!     call nekgsync()
C print min values
      xxmax = glmin(xm1,ntott)
      yymax = glmin(ym1,ntott)
      zzmax = glmin(zm1,ntott)

      vxmax = glmin(vx,ntotv)
      vymax = glmin(vy,ntotv)
      vzmax = glmin(vz,ntotv)
      prmax = glmin(pr,ntot2)

      ntot = nxyz1*nelt
      ttmax = glmin(t ,ntott)

      if (nio.eq.0) then
         write(6,19) xxmax,yymax,zzmax
   19    format('Cxyz min  ',5g13.5)
      endif
      if (nio.eq.0) then
         write(6,20) vxmax,vymax,vzmax,prmax,ttmax
   20    format('Cuvwpt min',5g13.5)
      endif

c print max values
      xxmax = glmax(xm1,ntott)
      yymax = glmax(ym1,ntott)
      zzmax = glmax(zm1,ntott)

      vxmax = glmax(vx,ntotv)
      vymax = glmax(vy,ntotv)
      vzmax = glmax(vz,ntotv)
      prmax = glmax(pr,ntot2)

      ntot = nxyz1*nelt
      ttmax = glmax(t ,ntott)

      if (nio.eq.0) then
         write(6,16) xxmax,yymax,zzmax
   16    format('Cxyz max  ',5g13.5)
      endif

      if (nio.eq.0) then
         write(6,17) vxmax,vymax,vzmax,prmax,ttmax
   17    format('Cuvwpt max',5g13.5)
      endif

c     ! save velocity on fine mesh for dealiasing
!     call setup_convect(2) ! check what this does again. might be a good
!                           ! idea, or it might be counterproductive
      if(nio.eq.0) then
        write(6,*) 'done :: set initial conditions, CMT-nek'
        write(6,*) ' '
      endif
      return
      end

!-----------------------------------------------------------------------

      subroutine cmtuic
! overlaps with setics. -DCMT will require IFDG as well
      include 'SIZE'
      include 'SOLN'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'NEKUSE'
      integer e,eg
      do e=1,nelt
         eg = lglel(e)
         do k=1,nz1
         do j=1,ny1
         do i=1,nx1           
            call nekasgn (i,j,k,e)
            call cmtasgn (i,j,k,e)
            call useric  (i,j,k,eg)
            vx(i,j,k,e) = ux
            vy(i,j,k,e) = uy
            vz(i,j,k,e) = uz
            vtrans(i,j,k,e,irho)  = rho
            vtrans(i,j,k,e,icv)= rho*cv
            vtrans(i,j,k,e,icp)= rho*cp
            phig(i,j,k,e)  = phi
            pr(i,j,k,e)    = pres
            u(i,j,k,irg,e) = phi*rho
            u(i,j,k,irpu,e)= phi*rho*ux
            u(i,j,k,irpv,e)= phi*rho*uy
            u(i,j,k,irpw,e)= phi*rho*uz
            u(i,j,k,iret,e)= phi*rho*(cv*temp+0.5*(ux**2+uy**2+uz**2))
            vdiff(i,j,k,e,imu) = mu
            vdiff(i,j,k,e,iknd)= udiff
            vdiff(i,j,k,e,ilam)= lambda
            t(i,j,k,e,1) = temp
         enddo
         enddo
         enddo
      enddo
      return
      end
