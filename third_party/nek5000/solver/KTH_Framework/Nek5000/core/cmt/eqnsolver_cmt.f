C> @file eqnsolver_cmt.f Routines for entire terms on RHS. Mostly volume integrals

C> \ingroup diffhvol
C> @{
C> Volume integral for diffusive terms. Compute \f$\mathbf{H}^d\f$
C> and store it for one element. Store faces of \f$\mathbf{H}^d\f$ for IGU. 
      subroutine viscous_cmt(e,eq)
      include  'SIZE'
      include  'CMTDATA'
      include  'DG'
      include  'INPUT'

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ fatface(heresize),graduf(hdsize)
      real fatface,graduf

      integer e,eq

      if (eq .lt. toteq) then ! not energy
         if (eq .gt. ldim+1) return ! not if3d
      endif

      nxyz=lx1*ly1*lz1
      nfq=lx1*lz1*2*ldim*nelt
      nstate = nqq
! where different things live
      iqm =1
      iqp =iqm+nstate*nfq
      iuj =iqp+nstate*nfq

      call rzero(diffh,3*nxyz)

      call agradu(diffh,gradu,e,eq)

      call diffh2graduf(e,eq,graduf) ! on faces for QQ^T and igu_cmt

! volume integral involving "DG-ish" stiffness operator K
      call half_iku_cmt(res1(1,1,1,e,eq),diffh,e)

C> @}
      return
      end

!-----------------------------------------------------------------------

C> \ingroup vsurf
C> @{
C> \f$G^T U\f$
      subroutine igtu_cmt(qminus,ummcu,hface)

!     Vol integral [[u]].{{gradv}}. adapted from Lu's dgf3.f;
! interior penalty stuff could go
! here too. Debug subroutine ihu in heat.usr and try to fold it in here.

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'DG'      ! iface
      include 'CMTDATA'
      include 'SOLN' ! for vz. goes away when agradu_ns works

! arguments
      real qminus(lx1*lz1,2*ldim,nelt,*)    ! intent(in)
      real ummcu(lx1*lz1*2*ldim,nelt,toteq) ! intent(in)
      real hface(lx1*lz1*2*ldim*nelt,toteq,3) ! intent(out) scratch

! commons and scratch
      common /scrns/ superhugeh(lx1*ly1*lz1*lelt,3) ! like totalh, but super-huge
      common /scruz/ gradm1_t_overwrites(lx1*ly1*lz1*lelt) ! sigh
      real superhugeh,gradm1_t_overwrites
!     common /ctmp0/ viscscr(lx1,ly1,lz1)
!     real viscscr
!     parameter (lfq=lx1*lz1*2*ldim*lelt)
!     common /ctmp0/ ftmp1(lfq),ftmp2(lfq)
!     real ftmp1,ftmp2

      integer eijk3(3,3,3)
!     data eijk2 / 0, -1, 1, 0/
      data eijk3
     >/0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0/

      integer e, eq, n, npl, nf, f, i, k, eq2

      nxz    = lx1*lz1
      nfaces = 2*ldim
      nf     = nxz*nfaces ! 1 element's face points
      nfq    = nf*nelt ! all points in a pile of faces
!     if (ifsip) then
!        const=-1.0 ! SIP
!     else
         const=1.0 ! Baumann-Oden
!     endif

! compute (U-{{U}})_i * n_k
! OK FOLKS GIANT BUG UMMCU IS BAD AT INFLOW
      l=1
      do e=1,nelt
         do eq=1,toteq
            call col3(hface(l,eq,3),ummcu(1,e,eq),area(1,1,1,e),nf)
            call col3(hface(l,eq,1),hface(l,eq,3),unx(1,1,1,e), nf)
            call col3(hface(l,eq,2),hface(l,eq,3),uny(1,1,1,e), nf)
            if(if3d) call col2(hface(l,eq,3),unz(1,1,1,e),nf)
         enddo
         l=l+nf
      enddo

      nxyz  =lx1*ly1*lz1
      nvol  =nxyz*nelt
      ngradu=nxyz*toteq*3
      do eq=1,toteq
         call rzero(superhugeh,3*lx1*ly1*lz1*lelt)
         if (eq .eq. 4 .and. .not. if3d) goto 133
         l=1
         m=1
         do e=1,nelt
            call rzero(diffh,3*nxyz)
            call rzero(gradu,ngradu) ! this too goes away when gradu is global
            do j=1,ldim
               do eq2=1,toteq ! sigh
                  call add_face2full_cmt(1,lx1,ly1,lz1,iface_flux(1,e),
     >                                gradu(1,eq2,j),hface(l,eq2,j))
               enddo
            enddo

            l=l+nf ! index for hface, which is global. this all goes away
                ! once you get rid of that execrable "element loop" in
                ! compute_rhs_and_dt
!           call fluxj_ns(superhugeh,... THIS will be correctly strided as well
! JH110716 AND someday it will work
!!            do j=1,ldim    ! flux direction
!!               do k=1,ldim ! dU   direction
!!                  ieijk=0
!!                  if (eq .lt. toteq) ieijk=eijk3(eq-1,j,k) ! does this work in 2D?
!!                  if (ieijk .eq. 0) then
!!                     call agradu_ns(superhugeh(m,j),gradu(1,1,k),viscscr
!!     >                             ,e,eq,j,k) ! the worst stride ever
!!                  endif
!!               enddo
!!            enddo
! JH110716 but not today. for now, let maxima do my thinking in the fluxj* routines

            call agradu(diffh,gradu,e,eq)

            do j=1,ldim
               call copy(superhugeh(m,j),diffh(1,j),nxyz)
            enddo

            m=m+nxyz

         enddo ! element loop

! gradm1_t uses /ctmp1/
         call gradm1_t(gradm1_t_overwrites,superhugeh(1,1),
     >                        superhugeh(1,2),superhugeh(1,3))
         call cmult(gradm1_t_overwrites,const,nvol)
         call add2(res1(1,1,1,1,eq),gradm1_t_overwrites,nvol)
133      continue
      enddo ! equation loop

C> @}

      return
      end

!-----------------------------------------------------------------------

C> \ingroup convhvol
C> @{
C> Convective volume terms formed and differentiated^T here
      subroutine convective_cmt(e,eq)
! JH081916 convective flux divergence integrated in weak form and
!          placed in res1.
      include 'SIZE'
      include 'CMTDATA'
      integer e,eq

      n=3*lxd*lyd*lzd

      call rzero(convh,n)
      if (lxd.gt.lx1) then
         call evaluate_dealiased_conv_h(e,eq)
         call copy(totalh,convh,n)
         call flux_div_integral_dealiased(e,eq)
      else
         call evaluate_aliased_conv_h(e,eq)
         call copy(totalh,convh,n)
         call flux_div_integral_aliased(e,eq)
      endif

      return
      end
C> @}

C> \ingroup convhvol
C> @{
C> Convective pointwise flux function \f$\mathbf{H}^c\f$ on fine grid.
      subroutine evaluate_dealiased_conv_h(e,eq)
! computed as products between primitive variables and conserved variables.
! if you want to write rho u_i u_j as (rho u_i) (rho u_j) (rho^{-1}), this
! is the place to do it
      include  'SIZE'
      include  'SOLN'
      include  'DEALIAS'
      include  'CMTDATA'
      include  'INPUT'
     
      integer  e,eq

      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ju1(ldd),ju2(ldd)!,ur(ldd),us(ldd),ud(ldd),tu(ldd)
      real ju1,ju2

      n=lxd*lyd*lzd

      if (eq .eq. 1) then ! convective flux of mass=rho u_j=U_{j+1}

         do j=1,ldim
            call intp_rstd(convh(1,j),u(1,1,1,eq+j,e),lx1,lxd,if3d,0)
         enddo

      else

c To be consistent with momentum equation, for mass balance flux vector is 
c computed by multiplying rho by u_j
         call intp_rstd(ju1,phig(1,1,1,e),lx1,lxd,if3d,0)
         call intp_rstd(ju2,pr(1,1,1,e),lx1,lxd,if3d,0)

         if (eq .lt. 5) then ! self-advection of rho u_i by rho u_i u_j

            call intp_rstd(convh(1,1),u(1,1,1,eq,e),lx1,lxd,if3d,0)
            do j=2,ldim
               call copy(convh(1,j),convh(1,1),n)
            enddo
            call col2(convh(1,1),vxd(1,1,1,e),n)
            call col2(convh(1,2),vyd(1,1,1,e),n)
            if (if3d) call col2(convh(1,3),vzd(1,1,1,e),n)
            call add2col2(convh(1,eq-1),ju1,ju2,n)

         elseif (eq .eq. 5) then

            call intp_rstd(convh(1,1),u(1,1,1,eq,e),lx1,lxd,if3d,0)
            call add2col2(convh(1,1),ju1,ju2,n)
            do j=2,ldim
               call copy(convh(1,j),convh(1,1),n)
            enddo
            call col2(convh(1,1),vxd(1,1,1,e),n)
            call col2(convh(1,2),vyd(1,1,1,e),n)
            call col2(convh(1,3),vzd(1,1,1,e),n)

         else
            if(nio.eq.0) write(6,*) 'eq=',eq,'really must be <= 5'
            if(nio.eq.0) write(6,*) 'aborting in evaluate_conv_h'
            call exitt
         endif

      endif
     
      return
      end

      subroutine evaluate_aliased_conv_h(e,eq)
! computed as products between primitive variables and conserved variables.
! if you want to write rho u_i u_j as (rho u_i) (rho u_j) (rho^{-1}), this
! is the place to do it
      include  'SIZE'
      include  'SOLN'
      include  'DEALIAS'
      include  'CMTDATA'
      include  'INPUT'

      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ju1(ldd),ju2(ldd)!,ur(ldd),us(ldd),ud(ldd),tu(ldd)
      real ju1,ju2
      integer  e,eq

      n=lxd*lyd*lzd

      call copy(ju1,phig(1,1,1,e),n)
      call copy(ju2,pr(1,1,1,e),n)

      if (eq .lt. 5) then ! self-advection of rho u_i by rho u_i u_j

         call copy(convh(1,1),u(1,1,1,eq,e),n)
         do j=2,ldim
            call copy(convh(1,j),convh(1,1),n)
         enddo
         call col2(convh(1,1),vxd(1,1,1,e),n)
         call col2(convh(1,2),vyd(1,1,1,e),n)
         if (if3d) call col2(convh(1,3),vzd(1,1,1,e),n)
         if(eq. gt. 1) call add2col2(convh(1,eq-1),ju1,ju2,n)

      elseif (eq .eq. 5) then

         call copy(convh(1,1),u(1,1,1,eq,e),n)
         call add2col2(convh(1,1),ju1,ju2,n)
         do j=2,ldim
            call copy(convh(1,j),convh(1,1),n)
         enddo
         call col2(convh(1,1),vxd(1,1,1,e),n)
         call col2(convh(1,2),vyd(1,1,1,e),n)
         call col2(convh(1,3),vzd(1,1,1,e),n)

      else
         if(nio.eq.0) write(6,*) 'eq=',eq,'really must be <= 5'
         if(nio.eq.0) write(6,*) 'aborting in evaluate_conv_h'
         call exitt
      endif

      return
      end
C> @}

C> \ingroup convhvol
C> @{
C> \f$(\nabla v)\cdot \mathbf{H}^c=\mathcal{I}^{\intercal}\mathbf{D}^{\intercal}\cdots\f$ for equation eq, element e
      subroutine flux_div_integral_dealiased(e,eq)
      include  'SIZE'
      include  'INPUT'
      include  'GEOM'
      include  'MASS'
      include  'CMTDATA'

      integer  e,eq
      integer  dir
      parameter (ldd=lxd*lyd*lzd)
      parameter (ldg=lxd**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      nrstd=ldd
      nxyz=lx1*ly1*lz1
      call get_dgl_ptr(ip,lxd,lxd) ! fills dg, dgt
      mdm1=lxd-1

      call rzero(ur,nrstd)
      call rzero(us,nrstd)
      call rzero(ut,nrstd)
      call rzero(ud,nrstd)
      call rzero(tu,nrstd)

      j0=0
      do j=1,ldim
         j0=j0+1
         call add2col2(ur,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      do j=1,ldim
         j0=j0+1
         call add2col2(us,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      if (if3d) then
         do j=1,ldim
            j0=j0+1
            call add2col2(ut,totalh(1,j),rx(1,j0,e),nrstd)
         enddo
         call local_grad3_t(ud,ur,us,ut,mdm1,1,dg(ip),dgt(ip),wkd)
      else
         call local_grad2_t(ud,ur,us,   mdm1,1,dg(ip),dgt(ip),wkd)
      endif

      call intp_rstd(tu,ud,lx1,lxd,if3d,1)

! multiply the residue by mass matrix. Non diagonal B should still be
! one block per element
!     call col2(ud,bm1(1,1,1,e),nxyz)  ! res = B*res  --- B=mass matrix
!     call add2(res1(1,1,1,e,eq),tu,nxyz)
! weak?
      call sub2(res1(1,1,1,e,eq),tu,nxyz)

      return
      end

C> @}

      subroutine flux_div_integral_aliased(e,eq)
      include  'SIZE'
      include  'INPUT'
      include  'GEOM'
      include  'MASS'
      include  'CMTDATA'

      integer  e,eq
      integer  dir
      parameter (ldd=lxd*lyd*lzd)
      parameter (ldg=lxd**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju
      common /dgradl/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt
      character*32 cname

      nrstd=ldd
      nxyz=lx1*ly1*lz1
      call get_dgll_ptr(ip,lxd,lxd) ! fills dg, dgt
      mdm1=lxd-1

      call rzero(ur,nrstd)
      call rzero(us,nrstd)
      call rzero(ut,nrstd)
      call rzero(ud,nrstd)
      call rzero(tu,nrstd)

      j0=0
      do j=1,ldim
         j0=j0+1
         call add2col2(ur,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      do j=1,ldim
         j0=j0+1
         call add2col2(us,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      if (if3d) then
         do j=1,ldim
            j0=j0+1
            call add2col2(ut,totalh(1,j),rx(1,j0,e),nrstd)
         enddo
         call local_grad3_t(ud,ur,us,ut,mdm1,1,d(ip),dt(ip),wkd)
      else
         call local_grad2_t(ud,ur,us,   mdm1,1,d(ip),dt(ip),wkd)
      endif

      call copy(tu,ud,nxyz)

! needs fleg or removal altogether. not good modularity
      call sub2(res1(1,1,1,e,eq),tu,nxyz)

      return
      end

!-----------------------------------------------------------------------
      subroutine compute_forcing(e,eq_num)
      include  'SIZE'
      include  'INPUT'
      include  'GEOM'
      include  'PARALLEL'
      include  'MASS'
      include  'SOLN'
      include  'CMTDATA'
      include  'DEALIAS'
      
      integer e,eq_num
      parameter (ldd=lxd*lyd*lzd)

      common /lpm_fix/ phigdum,phigvdum
      real phigdum(lx1,ly1,lz1,lelt,3),phigvdum(lx1,ly1,lz1,lelt)

      nxyz=lx1*ly1*lz1
      if(eq_num.ne.1.and.eq_num.ne.5)then


        if (eq_num.eq.4.and.ldim.eq.2)then

        else
#ifdef LPM
           call subcol3(res1(1,1,1,e,eq_num),phigdum(1,1,1,e,eq_num-1)
     >                  ,bm1(1,1,1,e),nxyz)
#endif
           call subcol3(res1(1,1,1,e,eq_num),usrf(1,1,1,eq_num)
     $                  ,bm1(1,1,1,e),nxyz) 
        endif
      elseif(eq_num.eq.5)then

#ifdef LPM
           call subcol3(res1(1,1,1,e,eq_num),phigvdum(1,1,1,e)
     >                  ,bm1(1,1,1,e),nxyz)
#endif
c          call subcol3(res1(1,1,1,e,eq_num),usrf(1,1,1,eq_num)
c    $                  ,bm1(1,1,1,e),nxyz) 

      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine cmtusrf(e)
      include 'SIZE'
      include 'NEKUSE'
      include 'CMTDATA'
      include 'TSTEP'
      include 'PARALLEL'

      integer e,eg

      if(istep.eq.1)then
        n = lx1*ly1*lz1*5
        call rzero(usrf,n)
      endif
      eg = lglel(e)
      do k=1,lz1
         do j=1,ly1
            do i=1,lx1
               call NEKASGN(i,j,k,e)
               call userf(i,j,k,eg)
               rdum4 = 0.
#ifdef LPM
               call lpm_userf(I,J,K,e,rdum1,rdum2,rdum3,rdum4)
               FFX  = FFX + rdum1
               FFY  = FFY + rdum2
               FFZ  = FFZ + rdum3
#endif
               ! note fx,fy,fz multiply by density to stay 
               ! consistent with nek5000 units. Same for phig (cancels)
               usrf(i,j,k,2) = FFX*u(i,j,k,1,e)*phig(i,j,k,e)
               usrf(i,j,k,3) = FFY*u(i,j,k,1,e)*phig(i,j,k,e)
               usrf(i,j,k,4) = FFZ*u(i,j,k,1,e)*phig(i,j,k,e)
               usrf(i,j,k,5) = 0.0
c              usrf(i,j,k,5) = (U(i,j,k,2,e)*FFX + U(i,j,k,3,e)*FFY
c    &                       +  U(i,j,k,4,e)*FFZ)/ U(i,j,k,1,e)
            enddo
         enddo
      enddo

      return
      end 
