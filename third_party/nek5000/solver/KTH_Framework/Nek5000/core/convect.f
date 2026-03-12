c-----------------------------------------------------------------------
      subroutine char_conv(p0,u,ulag,bm,bmlag,msk,c,cs,gsl)
c
c
c     Convect over last NBD steps using characteristics scheme
c
c     NOTE:  Here, we assume that ulag is stored by time-slice first,
c            then by field number (this is opposite to prior Nek5000)
c
c
      include 'SIZE'
      include 'TOTAL'
      real    p0(1),u(1),ulag(1),bm(1),bmlag(1),msk(1),c(1),cs(0:1)
      integer gsl

      common /scrns/ ct  (lxd*lyd*lzd*lelv*ldim)

      common /scrvh/ bmsk(lx1*ly1*lz1*lelv)
     $             , bdwt(lx1*ly1*lz1*lelv)
     $             , bmst(lx1*ly1*lz1*lelv)
     $             , u1  (lx1*ly1*lz1*lelv)

      common /scrmg/ r1  (lx1*ly1*lz1*lelv)
     $             , r2  (lx1*ly1*lz1*lelv)
     $             , r3  (lx1*ly1*lz1*lelv)
     $             , r4  (lx1*ly1*lz1*lelv)
      
      nelc = nelv            ! number of elements in convecting field
      if (ifield.eq.ifldmhd) nelc = nelfld(ifield)

      nc  = cs(0)            ! number of stored convecting fields

      ln  = lx1*ly1*lz1*lelt
      n   = lx1*ly1*lz1*nelfld(ifield)
      m   = lxd*lyd*lzd*nelc*ldim

c      if(nid.eq.0) write(*,*) 'going into char_conv1 '
      call char_conv1 (p0,u,bmnv,n,ulag,ln,gsl,c,m,cs(1),nc,ct
     $  ,u1,r1,r2,r3,r4,bmsk,bdivw,bdwt,bmass,bmst,bm,bmlag)

      return
      end
c-----------------------------------------------------------------------
      subroutine char_conv1 (p0,u,bmnv,n,ulag,ln,gsl,c,m,cs,nc,ct
     $  ,u1,r1,r2,r3,r4,bmsk,bdivw,bdwt,bmass,bmst,bm,bmlag)

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'

      real p0(n),u(n),bmnv(n,1),ulag(ln,1),c(m,0:nc),cs(0:nc),bdivw(n,1)
     $          ,bm(n), bmlag(ln,1)

      real ct(m),u1(n),r1(n),r2(n),r3(n),r4(n),bmsk(n),bdwt(n) ! work arrays

      integer gsl


!     Convect over last NBD steps using characteristics scheme

!              n-q                                      n-1
!     Given u(t    , X ),  q = 1,2,...,nbd, compute  phi     ( := p0 )

!        n-1       nbd   ~n-q
!     phi     :=  sum    u
!                  q=1

!          ~n-q             ~n-q
!     each u     satisfies  u    := v  such that


!     dv
!     -- + C.grad v = 0  t \in [t^n-q,t^n],   v(t^n-q,X) = u(t^n-q,X)
!     dt

!     n = lx1*ly1*lz1*nelv
!     m = lxd*lyd*lzd*nelv

      tau = time-vlsum(dtlag,nbd)              ! initialize time for u^n-k
      call int_vel (ct  ,tau,c    ,m,nc,cs,nid) ! ct(t) = sum w_k c(.,k)
      call int_vel (bmsk,tau,bmnv ,n,nc,cs,nid) ! B^-1(t^n-1)
      call int_vel (bmst,tau,bmass,n,nc,cs,nid) ! B(t^n-1)
      call int_vel (bdwt,tau,bdivw,n,nc,cs,nid) ! BdivW(t^n-1)

      call rzero(p0,n)

      do ilag = nbd,1,-1

         um = 0
         if (ilag.eq.1) then
            do i=1,n
               p0(i) = p0(i)+bd(ilag+1)*u(i)*bm(i)
               um=max(um,u(i))
            enddo
         else
           if(ifmvbd) then
            do i=1,n
               p0(i) = p0(i)+bd(ilag+1)*ulag(i,ilag-1)*bmlag(i,ilag-1)
               um=max(um,ulag(i,ilag-1))
            enddo
           else
            do i=1,n
               p0(i) = p0(i)+bd(ilag+1)*ulag(i,ilag-1)*bm(i)
               um=max(um,ulag(i,ilag-1))
            enddo
           endif
         endif

         dtau = dtlag(ilag)/ntaubd
         do itau = 1,ntaubd ! ntaubd=number of RK4 substeps (typ. 1 or 2)

            tau1 = tau + dtau

            c1 = 1.
            c2 = -dtau/2.
            c3 = -dtau
            th = tau+dtau/2.

            call invcol3 (u1,p0,bmst,n)
            call conv_rhs(r1,u1,ct,bmsk,bmst,bdwt,gsl)          ! STAGE 1
            call col2    (r1,bmst,n)     !             ! r1 = B(n-1)* r1

            call add3s12 (u1,p0,r1,c1,c2,n)
            call int_vel (bmst,th,bmass,n,nc,cs,nid)   ! B(n-1/2)
            call invcol2 (u1,bmst,n)                   ! u2=B(n-1/2)

            call int_vel (ct  ,th,c    ,m,nc,cs,nid)   ! STAGE 2
            call int_vel (bmsk,th,bmnv ,n,nc,cs,nid)   ! B^-1(n-1/2)
            call int_vel (bdwt,th,bdivw,n,nc,cs,nid)   ! BdivW(n-1/2)
            call conv_rhs(r2,u1,ct,bmsk,bmst,bdwt,gsl)
            call col2    (r2,bmst,n)     !  du = B          * du

            call add3s12 (u1,p0,r2,c1,c2,n)           ! STAGE 3
            call invcol2 (u1,bmst,n)
            call conv_rhs(r3,u1,ct,bmsk,bmst,bdwt,gsl)          ! B(n-1/2) (still)
            call col2    (r3,bmst,n)     !  du = B          * du

            call add3s12 (u1,p0,r3,c1,c3,n)
            call int_vel (bmst,tau1,bmass,n,nc,cs,nid) ! B^-1(n)
            call invcol2 (u1,bmst,n)                   ! u2=B(n-1/2)

            call int_vel (ct  ,tau1,c    ,m,nc,cs,nid) ! STAGE 4
            call int_vel (bmsk,tau1,bmnv ,n,nc,cs,nid) ! B^-1(n)
            call int_vel (bdwt,tau1,bdivw,n,nc,cs,nid) ! BdivW(n)
            call conv_rhs(r4,u1,ct,bmsk,bmst,bdwt,gsl)
            call col2    (r4,bmst,n)     !  du = B          * du

            c1 = -dtau/6.
            c2 = -dtau/3.
            do i=1,n
               p0(i) = p0(i)+c1*(r1(i)+r4(i))+c2*(r2(i)+r3(i))
            enddo
            tau = tau1
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine int_vel(c_t,t0,c,n,nc,ct,nid)

c     Interpolate convecting velocity field c(t_1,...,t_nconv) to
c     time t0 and return result in c_t.

c     Ouput:   c_t = sum wt_k * ct_i(k) 

c     Here, t0 is the time of interest

      real c_t(n),c(n,0:nc),ct(0:nc)

      parameter (lwtmax=10)
      real wt(0:lwtmax)

      if (nc.gt.lwtmax) then
         write(6,*) nid,'ERROR int_vel: lwtmax too small',lwtmax,nc
         call exitt
      endif

      no = nc-1
      call fd_weights_full(t0,ct(0),no,0,wt)  ! interpolation weights

      call rzero(c_t,n)
      do j=1,n
      do i=0,no
         c_t(j) = c_t(j) + wt(i)*c(j,i)
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine conv_rhs (du,u,c,bmsk,bmst,bdwt,gsl)
c
      include 'SIZE'
      include 'TOTAL'
c
c     apply convecting field c(1,ldim) to scalar field u(1)
c
      real du(1),u(1),c(1),bmsk(1),bdwt(1)
      integer gsl
c
      logical ifconv
c
c     ifconv = .false.
      ifconv = .true.
c
      n = lx1*ly1*lz1*nelv

      if (ifdgfld(ifield)) then

       if (param(99).eq.1) call conv_rhs_dg         (du,u,c)
       if (param(99).eq.0) call conv_rhs_dg_aliased (du,u,c)

      elseif (ifconv) then

         if (ifcons) then
           if (if3d     ) call convop_cons_3d (du,u,c,lx1,lxd,nelv)
           if (.not.if3d) call convop_cons_2d (du,u,c,lx1,lxd,nelv)
         else
           if (if3d     ) call convop_fst_3d  (du,u,c,lx1,lxd,nelv)
           if (.not.if3d) call convop_fst_2d  (du,u,c,lx1,lxd,nelv)
         endif

         call subcol3(du,bdwt,u,n)
         call fgslib_gs_op (gsl,du,1,1,0)  !  +

         call col2 (du,bmsk,n)     !  du = Binv * msk * du

      else
         call rzero   (du,n)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine convop_fst_3d(du,u,c,mx,md,nel)
c
      include 'SIZE'
c
c     apply convecting field c to scalar field u
c
      real du(mx*mx*mx,nel)
      real  u(mx*mx*mx,nel)
      real  c(md*md*md,nel,3)
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ud(ldd)
c
      logical if3d,ifd
      integer e
c
      if3d = .true.
      ifd  = .false.
      if (md.ne.mx) ifd=.true.
      ifd=.true.
c
      nrstd = md**3
      call lim_chk(nrstd,ldd,'urus ','ldd  ','convop_fst')
c
      do e=1,nel
         call grad_rstd(ur,us,ut,u(1,e),mx,md,if3d,ud)
         if (ifd) then    ! dealiased
            do i=1,nrstd
c              C has the mass matrix factored in per (4.8.5), p. 227, DFM.
               ud(i) = c(i,e,1)*ur(i)+c(i,e,2)*us(i)+c(i,e,3)*ut(i)
            enddo
            call intp_rstd(du(1,e),ud,mx,md,if3d,1) ! 1 --> transpose
         else
            do i=1,nrstd
c              C has the mass matrix factored in per (4.8.5), p. 227, DFM.
               du(i,e) = c(i,e,1)*ur(i)+c(i,e,2)*us(i)+c(i,e,3)*ut(i)
            enddo
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine convop_fst_2d(du,u,c,mx,md,nel)
c
      include 'SIZE'
c
c     apply convecting field c to scalar field u
c
      real du(mx*mx,nel)
      real  u(mx*mx,nel)
      real  c(md*md,nel,2)
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ud(ldd)
c
      logical if3d,ifd
      integer e
c
      if3d = .false.
      ifd  = .false.
      if (md.ne.mx) ifd=.true.
      ifd=.true.
c
      nrstd = md**2
      call lim_chk(nrstd,ldd,'urus ','ldd  ','convop_fst')
c
      do e=1,nel
         call grad_rstd(ur,us,ut,u(1,e),mx,md,if3d,ud)
         if (ifd) then    ! dealiased
            do i=1,nrstd
c              C has the mass matrix factored in per (4.8.5), p. 227, DFM.
               ud(i) = c(i,e,1)*ur(i)+c(i,e,2)*us(i)
            enddo
            call intp_rstd(du(1,e),ud,mx,md,if3d,1) ! 1 --> transpose
         else
            do i=1,nrstd
c              C has the mass matrix factored in per (4.8.5), p. 227, DFM.
               du(i,e) = c(i,e,1)*ur(i)+c(i,e,2)*us(i)
            enddo
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine grad_rstd(ur,us,ut,u,mx,md,if3d,ju) ! GLL->GL grad

      include 'SIZE'
      include 'DXYZ'

      real    ur(1),us(1),ut(1),u(1),ju(1)
      logical if3d

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      call intp_rstd(ju,u,mx,md,if3d,0) ! 0 = forward

      m0 = md-1
      call get_dgl_ptr (ip,md,md)
      if (if3d) then
         call local_grad3(ur,us,ut,ju,m0,1,dg(ip),dgt(ip))
      else
         call local_grad2(ur,us   ,ju,m0,1,dg(ip),dgt(ip))
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine intp_rstd(ju,u,mx,md,if3d,idir) ! GLL->GL interpolation

c     GLL interpolation from mx to md.

c     If idir ^= 0, then apply transpose operator  (md to mx)

      include 'SIZE'

      real    ju(1),u(1)
      logical if3d

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      parameter (ld=2*lxd)
      common /ctmp0/ w(ld**ldim,2)

      call lim_chk(md,ld,'md   ','ld   ','grad_rstd ')
      call lim_chk(mx,ld,'mx   ','ld   ','grad_rstd ')

      ldw = 2*(ld**ldim)

      call get_int_ptr (i,mx,md)
c
      if (idir.eq.0) then
         call specmpn(ju,md,u,mx,jgl(i),jgt(i),if3d,w,ldw)
      else
         call specmpn(ju,mx,u,md,jgt(i),jgl(i),if3d,w,ldw)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_int(jgl,jgt,mp,np,w)
c
c     Generate interpolation from np GLL points to mp GL points
c
c        jgl  = interpolation matrix, mapping from velocity nodes to pressure
c        jgt  = transpose of interpolation matrix
c        w    = work array of size (np+mp)
c
c        np   = number of points on GLL grid
c        mp   = number of points on GL  grid
c
c
      real jgl(mp,np),jgt(np*mp),w(1)
c
      iz = 1
      id = iz + np
c
      call zwgll (w(iz),jgt,np)
      call zwgl  (w(id),jgt,mp)
c
      n  = np-1
      do i=1,mp
         call fd_weights_full(w(id+i-1),w(iz),n,0,jgt)
         do j=1,np
            jgl(i,j) = jgt(j)                  !  Interpolation matrix
         enddo
      enddo
c
      call transpose(jgt,np,jgl,mp)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_dgl(dgl,dgt,mp,np,w)
c
c     Generate derivative from np GL points onto mp GL points
c
c        dgl  = interpolation matrix, mapping from velocity nodes to pressure
c        dgt  = transpose of interpolation matrix
c        w    = work array of size (3*np+mp)
c
c        np   = number of points on GLL grid
c        mp   = number of points on GL  grid
c
c
c
      real dgl(mp,np),dgt(np*mp),w(1)
c
c
      iz = 1
      id = iz + np
c
      call zwgl  (w(iz),dgt,np)  ! GL points
      call zwgl  (w(id),dgt,mp)  ! GL points
c
      ndgt = 2*np
      ldgt = mp*np
      call lim_chk(ndgt,ldgt,'ldgt ','dgt  ','gen_dgl   ')
c
      n  = np-1
      do i=1,mp
         call fd_weights_full(w(id+i-1),w(iz),n,1,dgt) ! 1=1st deriv.
         do j=1,np
            dgl(i,j) = dgt(np+j)                       ! Derivative matrix
         enddo
      enddo
c
      call transpose(dgt,np,dgl,mp)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lim_chk(n,m,avar5,lvar5,sub_name10)
      include 'SIZE'            ! need nid
      character*5  avar5,lvar5
      character*10 sub_name10

      if (n.gt.m) then
         write(6,1) nid,n,m,avar5,lvar5,sub_name10
    1    format(i8,' ERROR: :',2i12,2(1x,a5),1x,a10)
         call exitti('lim_chk problem. $',n)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine get_int_ptr (ip,mx,md) ! GLL-->GL pointer

c     Get pointer to jgl() for interpolation pair (mx,md)

      include 'SIZE'

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt
c
      parameter (ld=2*lxd)
      common /igrad/ pd    (0:ld*ld)
     $             , pdg   (0:ld*ld)
     $             , pjgl  (0:ld*ld)
      integer pd , pdg , pjgl
c
      ij = md + ld*(mx-1)
      ip = pjgl(ij)
c
      if (ip.eq.0) then
c
         nstore   = pjgl(0)
         pjgl(ij) = nstore+1
         nstore   = nstore + md*mx
         pjgl(0)  = nstore
         ip       = pjgl(ij)
c
         nwrkd = mx + md
         call lim_chk(nstore,ldg ,'jgl  ','ldg  ','get_int_pt')
         call lim_chk(nwrkd ,lwkd,'wkd  ','lwkd ','get_int_pt')
c
         call gen_int(jgl(ip),jgt(ip),md,mx,wkd)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_dgl_ptr (ip,mx,md)
c
c     Get pointer to GL-GL interpolation dgl() for pair (mx,md)
c
      include 'SIZE'
c
      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt
c
      parameter (ld=2*lxd)
      common /jgrad/ pd    (0:ld*ld)
     $             , pdg   (0:ld*ld)
     $             , pjgl  (0:ld*ld)
      integer pd , pdg , pjgl
c
      ij = md + ld*(mx-1)
      ip = pdg (ij)

      if (ip.eq.0) then

         nstore   = pdg (0)
         pdg (ij) = nstore+1
         nstore   = nstore + md*mx
         pdg (0)  = nstore
         ip       = pdg (ij)
c
         nwrkd = mx + md
         call lim_chk(nstore,ldg ,'dg   ','ldg  ','get_dgl_pt')
         call lim_chk(nwrkd ,lwkd,'wkd  ','lwkd ','get_dgl_pt')
c
         call gen_dgl(dg (ip),dgt(ip),md,mx,wkd)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_conv_char(ct,c,ux,uy,uz,nelc,tau,ifnew)
      include 'SIZE'
      include 'TSTEP'

      real ct(0:1)               ! time stamps for saved field (0=#flds)
      real c(1)                  ! saved vel. fields, dealiased etc.
      real ux(1),uy(1),uz(1)     ! input vel. field
      integer nelc               ! number of elements in conv. field
      logical ifnew              ! =true if shifting stack of fields

      numr      = lxd*lyd*lzd*lelv*ldim*(lorder+1)
      denr      = lxd*lyd*lzd*nelv*ldim
      nconv_max = numr/denr
      if (nconv_max.lt.nbdinp+1) 
     $   call exitti(
     $     'ABORT: not enough memory for characteristics scheme!$',
     $     nconv_max)

      nc = ct(0)

      m  = lxd*lyd*lzd*nelc*ldim

      call set_ct_cvx
     $    (ct,c,m,ux,uy,uz,tau,nc,nconv_max,nelc,ifnew)

      nc = min (nc,nbdinp)
      ct(0) = nc  ! store current count

      return
      end
c-----------------------------------------------------------------------
      subroutine set_ct_cvx(ct,c,m,u,v,w,tau,nc,mc,nelc,ifnew)
      include 'SIZE'
      include 'INPUT'  ! ifcons

      real ct(0:1),c(m,1)
      real u(1),v(1),w(1)
      logical ifnew

      if (ifnew) then

c        Shift existing convecting fields
c        Note:  "1" entry is most recent

         nc = nc+1
         nc = min(nc,mc)
         ct(0) = nc

         do i=nc,2,-1
            call copy(c(1,i),c(1,i-1),m)
            ct(i) = ct(i-1)
         enddo
      endif

c     Save time and map the current velocity to rst coordinates.

      ix = 1
      iy = ix + lxd*lyd*lzd*nelc
      iz = iy + lxd*lyd*lzd*nelc

      if (ifcons) then
         call set_convect_cons(c(ix,1),c(iy,1),c(iz,1),u,v,w)
      else
         call set_convect_new (c(ix,1),c(iy,1),c(iz,1),u,v,w)
      endif

      ct(1) = tau

      return
      end
c-----------------------------------------------------------------------
      subroutine grad_rst(ur,us,ut,u,md,if3d) ! Gauss-->Gauss grad

      include 'SIZE'
      include 'DXYZ'

      real    ur(1),us(1),ut(1),u(1)
      logical if3d

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      m0 = md-1
      call get_dgl_ptr (ip,md,md)
      if (if3d) then
         call local_grad3(ur,us,ut,u,m0,1,dg(ip),dgt(ip))
      else
         call local_grad2(ur,us   ,u,m0,1,dg(ip),dgt(ip))
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine convect_new(bdu,u,ifuf,cx,cy,cz,ifcf)

C     Compute dealiased form:  J^T Bf *JC .grad Ju w/ correct Jacobians
C
      include 'SIZE'
      include 'TOTAL'

      real bdu(1),u(1),cx(1),cy(1),cz(1)
      logical ifuf,ifcf            ! u and/or c already on fine mesh?

      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)
      common /scrcv/ fx(ltd),fy(ltd),fz(ltd)
     $             , ur(ltd),us(ltd),ut(ltd)
     $             , tr(ltd,3),uf(ltd)

      integer e

      call set_dealias_rx

      nxyz1 = lx1*ly1*lz1
      nxyzd = lxd*lyd*lzd

      nxyzu = nxyz1
      if (ifuf) nxyzu = nxyzd

      nxyzc = nxyz1
      if (ifcf) nxyzc = nxyzd

      iu = 1    ! pointer to scalar field u
      ic = 1    ! pointer to vector field C
      ib = 1    ! pointer to scalar field Bdu


      do e=1,nelv

         if (ifcf) then

            call copy(tr(1,1),cx(ic),nxyzd)  ! already in rst form
            call copy(tr(1,2),cy(ic),nxyzd)
            if (if3d) call copy(tr(1,3),cz(ic),nxyzd)

         else  ! map coarse velocity to fine mesh (C-->F)

           call intp_rstd(fx,cx(ic),lx1,lxd,if3d,0) ! 0 --> forward
           call intp_rstd(fy,cy(ic),lx1,lxd,if3d,0) ! 0 --> forward
           if (if3d) call intp_rstd(fz,cz(ic),lx1,lxd,if3d,0) ! 0 --> forward

           if (if3d) then  ! Convert convector F to r-s-t coordinates

             do i=1,nxyzd
               tr(i,1)=rx(i,1,e)*fx(i)+rx(i,2,e)*fy(i)+rx(i,3,e)*fz(i)
               tr(i,2)=rx(i,4,e)*fx(i)+rx(i,5,e)*fy(i)+rx(i,6,e)*fz(i)
               tr(i,3)=rx(i,7,e)*fx(i)+rx(i,8,e)*fy(i)+rx(i,9,e)*fz(i)
             enddo

           else

             do i=1,nxyzd
               tr(i,1)=rx(i,1,e)*fx(i)+rx(i,2,e)*fy(i)
               tr(i,2)=rx(i,3,e)*fx(i)+rx(i,4,e)*fy(i)
             enddo

           endif

         endif

         if (ifuf) then
            call grad_rst(ur,us,ut,u(iu),lxd,if3d)
         else
            call intp_rstd(uf,u(iu),lx1,lxd,if3d,0) ! 0 --> forward
            call grad_rst(ur,us,ut,uf,lxd,if3d)
         endif

         if (if3d) then
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               uf(i) = tr(i,1)*ur(i)+tr(i,2)*us(i)+tr(i,3)*ut(i)
            enddo
         else
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               uf(i) = tr(i,1)*ur(i)+tr(i,2)*us(i)
            enddo
         endif
         call intp_rstd(bdu(ib),uf,lx1,lxd,if3d,1) ! Project back to coarse

         ic = ic + nxyzc
         iu = iu + nxyzu
         ib = ib + nxyz1

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine convect_cons(bdu,u,ifuf,cx,cy,cz,ifcf)

c     Compute dealiased form:  J^T Bf *div. JC Ju w/ correct Jacobians

c     conservative form


      include 'SIZE'
      include 'TOTAL'

      real bdu(1),u(1),cx(1),cy(1),cz(1)

      logical ifuf,ifcf            ! u and/or c already on fine mesh?

      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)
      common /scrcv/ uf(ltd),cf(ltd),cu(ltd)
     $             , cr(ltd),cs(ltd),ct(ltd)


      integer e

      call set_dealias_rx

      nxyz1 = lx1*ly1*lz1
      nxyzd = lxd*lyd*lzd

      nxyzu = nxyz1
      if (ifuf) nxyzu = nxyzd

      nxyzc = nxyz1
      if (ifcf) nxyzc = nxyzd

      iu = 1    ! pointer to scalar field u
      ic = 1    ! pointer to vector field C
      ib = 1    ! pointer to scalar field Bdu

      do e=1,nelv

        call intp_rstd(uf,u(iu),lx1,lxd,if3d,0) ! 0 --> forward

        call rzero(cu,nxyzd)
        do i=1,ldim

         if (ifcf) then  ! C is already on fine mesh

           call exitt  ! exit for now

         else  ! map coarse velocity to fine mesh (C-->F)

           if (i.eq.1) call intp_rstd(cf,cx(ic),lx1,lxd,if3d,0) ! 0 --> forward
           if (i.eq.2) call intp_rstd(cf,cy(ic),lx1,lxd,if3d,0) ! 0 --> forward
           if (i.eq.3) call intp_rstd(cf,cz(ic),lx1,lxd,if3d,0) ! 0 --> forward

           call col2(cf,uf,nxyzd)   !  collocate C and u on fine mesh

           call grad_rst(cr,cs,ct,cf,lxd,if3d)  ! d/dr (C_i*u)

           if (if3d) then

             do j=1,nxyzd
               cu(j)=cu(j)
     $              +cr(j)*rx(j,i,e)+cs(j)*rx(j,i+3,e)+ct(j)*rx(j,i+6,e)
             enddo

           else  ! 2D

             do j=1,nxyzd
               cu(j)=cu(j)
     $              +cr(j)*rx(j,i,e)+cs(j)*rx(j,i+2,e)
             enddo

           endif
         endif
        enddo

        call intp_rstd(bdu(ib),cu,lx1,lxd,if3d,1) ! Project back to coarse

        ic = ic + nxyzc
        iu = iu + nxyzu
        ib = ib + nxyz1

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine set_convect_cons(cx,cy,cz,ux,uy,uz)

c     Put vx,vy,vz on fine mesh (for conservation form)


      include 'SIZE'
      include 'TOTAL'

      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)

      real cx(ltd,1),cy(ltd,1),cz(ltd,1)
      real ux(lxy,1),uy(lxy,1),uz(lxy,1)

      integer e

      call set_dealias_rx

      do e=1,nelv    ! Map coarse velocity to fine mesh (C-->F)

         call intp_rstd(cx(1,e),ux(1,e),lx1,lxd,if3d,0) ! 0 --> forward
         call intp_rstd(cy(1,e),uy(1,e),lx1,lxd,if3d,0) ! 0 --> forward
         if (if3d) call intp_rstd(cz(1,e),uz(1,e),lx1,lxd,if3d,0) ! 0 --> forward

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine set_convect_new(cr,cs,ct,ux,uy,uz)
C
C     Put vxd,vyd,vzd into rst form on fine mesh
C
C     For rst form, see eq. (4.8.5) in Deville, Fischer, Mund (2002).
C
      include 'SIZE'
      include 'TOTAL'

      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)

      real cr(ltd,1),cs(ltd,1),ct(ltd,1)
      real ux(lxy,1),uy(lxy,1),uz(lxy,1)

      common /scrcv/ fx(ltd),fy(ltd),fz(ltd)
     $             , ur(ltd),us(ltd),ut(ltd)
     $             , tr(ltd,3),uf(ltd)

      integer e

      call set_dealias_rx

      nxyz1 = lx1*ly1*lz1
      nxyzd = lxd*lyd*lzd

      ic = 1    ! pointer to vector field C

      do e=1,nelv 

c        Map coarse velocity to fine mesh (C-->F)

         call intp_rstd(fx,ux(1,e),lx1,lxd,if3d,0) ! 0 --> forward
         call intp_rstd(fy,uy(1,e),lx1,lxd,if3d,0) ! 0 --> forward
         if (if3d) call intp_rstd(fz,uz(1,e),lx1,lxd,if3d,0) ! 0 --> forward

c        Convert convector F to r-s-t coordinates

         if (if3d) then

           do i=1,nxyzd
              cr(i,e)=rx(i,1,e)*fx(i)+rx(i,2,e)*fy(i)+rx(i,3,e)*fz(i)
              cs(i,e)=rx(i,4,e)*fx(i)+rx(i,5,e)*fy(i)+rx(i,6,e)*fz(i)
              ct(i,e)=rx(i,7,e)*fx(i)+rx(i,8,e)*fy(i)+rx(i,9,e)*fz(i)
           enddo

         else

           do i=1,nxyzd
              cr(i,e)=rx(i,1,e)*fx(i)+rx(i,2,e)*fy(i)
              cs(i,e)=rx(i,3,e)*fx(i)+rx(i,4,e)*fy(i)
           enddo

         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine advchar
c
c     Compute convective contribution using 
c     operator-integrator-factor method (characteristics).
c
      include 'SIZE'
      include 'MASS'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      include 'PARALLEL'

      include 'CTIMER'

      common /cchar/ ct_vx(0:lorder) ! time for each slice in c_vx()

      common /scruz/ phx  (lx1*ly1*lz1*lelt)
     $ ,             phy  (lx1*ly1*lz1*lelt)
     $ ,             phz  (lx1*ly1*lz1*lelt)
     $ ,             hmsk (lx1*ly1*lz1*lelt)

      if (icalld.eq.0) tadvc=0.0
      icalld=icalld+1
      nadvc=icalld
      etime1=dnekclock()

      dti = 1./dt
      n   = lx1*ly1*lz1*nelv

      call char_conv(phx,vx,vxlag,bm1,bm1lag,hmsk,c_vx,ct_vx,gsh_fld(1))
      call char_conv(phy,vy,vylag,bm1,bm1lag,hmsk,c_vx,ct_vx,gsh_fld(1))
      if (if3d) call char_conv
     $              (phz,vz,vzlag,bm1,bm1lag,hmsk,c_vx,ct_vx,gsh_fld(1))

      call cfill(hmsk,dti,n)
      if(.not. iflomach) call col2(hmsk,vtrans,n) 

      if (if3d) then

        do i=1,n
           h2i = hmsk(i)
           bfx(i,1,1,1) = bfx(i,1,1,1)+phx(i)*h2i
           bfy(i,1,1,1) = bfy(i,1,1,1)+phy(i)*h2i
           bfz(i,1,1,1) = bfz(i,1,1,1)+phz(i)*h2i
        enddo

      else
        
        do i=1,n
           h2i = hmsk(i)
           bfx(i,1,1,1) = bfx(i,1,1,1)+phx(i)*h2i
           bfy(i,1,1,1) = bfy(i,1,1,1)+phy(i)*h2i
        enddo

      endif

      tadvc=tadvc+(dnekclock()-etime1)

      return
      end
c-----------------------------------------------------------------------
      subroutine convch

c     Compute convective contribution using 
c     operator-integrator-factor method (characteristics).

      include 'SIZE'
      include 'MASS'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      include 'PARALLEL'
      include 'CTIMER'

      common /cchar/ ct_vx(0:lorder) ! time for each slice in c_vx()

      common /scruz/ phi  (lx1*ly1*lz1*lelt)
     $ ,             hmsk (lx1*ly1*lz1*lelt)

      if (icalld.eq.0) tadvc=0.0
      icalld=icalld+1
      nadvc=icalld
      etime1=dnekclock()

      n   = lx1*ly1*lz1*nelv
      dti = 1./dt

      if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'convch', ifield
      call char_conv(phi,t(1,1,1,1,ifield-1),tlag(1,1,1,1,1,ifield-1)
     $        ,bm1,bm1lag,hmsk,c_vx,ct_vx,gsh_fld(1))

      do i=1,n
         bq(i,1,1,1,ifield-1) = bq(i,1,1,1,ifield-1)
     $          + phi(i)*vtrans(i,1,1,1,ifield)*dti
      enddo

      tadvc=tadvc+(dnekclock()-etime1)

      return
      end
c-----------------------------------------------------------------------
      subroutine convop_cons_3d(du,u,c,mx,md,nel) ! Conservation form

c     Apply convecting field c to scalar field u, conservation form d/dxj cj phi

c     Assumes that current convecting field is on dealias mesh, in c()

      include 'SIZE'
      include 'DEALIAS'
      include 'GEOM'

      real du(mx*mx*mx,nel)
      real  u(mx*mx*mx,nel)
      real  c(md*md*md,nel,3)
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju

      logical if3d,ifd
      integer e

      if3d = .true.
      ifd  = .false.
      if (md.ne.mx) ifd=.true.
      ifd=.true.

      nrstd = md**3
      call lim_chk(nrstd,ldd,'urus ','ldd  ','convp_cons')

      do e=1,nel

         call intp_rstd (ju,u(1,e),mx,md,if3d,0) ! 0 = forward; on Gauss points!
         call rzero     (ud,nrstd)

         do j=1,ldim
            do i=1,nrstd
               tu(i)=c(i,e,j)*ju(i)   ! C_j*T
            enddo
            call grad_rst(ur,us,ut,tu,md,if3d)  ! Already on fine (Gauss) mesh

            j0 = j+0
            j3 = j+3
            j6 = j+6
            do i=1,nrstd   ! rx has mass matrix and Jacobian on fine mesh
               ud(i)=ud(i)
     $              +rx(i,j0,e)*ur(i)+rx(i,j3,e)*us(i)+rx(i,j6,e)*ut(i)
            enddo
         enddo

         call intp_rstd(du(1,e),ud,mx,md,if3d,1) ! 1 --> transpose

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine convop_cons_2d(du,u,c,mx,md,nel) ! Conservation form

c     Apply convecting field c to scalar field u, conservation form d/dxj cj phi

c     Assumes that current convecting field is on dealias mesh, in c()

      include 'SIZE'
      include 'GEOM'
      include 'TSTEP'


      real du(mx*mx,nel)
      real  u(mx*mx,nel)
      real  c(md*md,nel,2)
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju

      logical if3d,ifd
      integer e

      if3d = .false.
      ifd  = .false.
      if (md.ne.mx) ifd=.true.

      nrstd = md**2
      call lim_chk(nrstd,ldd,'urus ','ldd  ','convp_cons')

      if (nio.eq.0.and.istep.lt.3) write(6,*) 'convp_cons',istep

      do e=1,nel

         call intp_rstd (ju,u(1,e),mx,md,if3d,0) ! 0 = forward; on Gauss points!
         call rzero     (ud,nrstd)

c        call outmat(c(1,e,1),md,md,'fine u',e)
c        call outmat(c(1,e,2),md,md,'fine v',e)
c        call outmat(ju      ,md,md,'fine T',e)

         do j=1,ldim
            do i=1,nrstd
               tu(i)=c(i,e,j)*ju(i)   ! C_j*T
            enddo
            call grad_rst(ur,us,ut,tu,md,if3d)  ! Already on fine (Gauss) mesh

            j0 = j+0
            j2 = j+2
            do i=1,nrstd   ! rx has mass matrix and Jacobian on fine mesh
               ud(i)=ud(i)+rx(i,j0,e)*ur(i)+rx(i,j2,e)*us(i)
            enddo
         enddo

         call intp_rstd(du(1,e),ud,mx,md,if3d,1) ! 1 --> transpose

      enddo

c     call exitti('convop_cons_2d$',istep)

      return
      end
c-----------------------------------------------------------------------
      subroutine iface_vert_int8(fa,va,jz0,jz1,nel)
      include 'SIZE'
      integer*8 fa(lx1*lz1,2*ldim,nel),va(0:lx1+1,0:ly1+1,jz0:jz1,nel)
      integer e,f

      n = lx1*lz1*2*ldim*nel
      call i8zero(fa,n)

      mx1 = lx1+2
      my1 = ly1+2
      mz1 = lz1+2
      if (ldim.eq.2) mz1=1

      nface = 2*ldim
      do e=1,nel
      do f=1,nface
         call facind (kx1,kx2,ky1,ky2,kz1,kz2,lx1,ly1,lz1,f)

         if     (f.eq.1) then ! EB notation
            ky1=ky1-1
            ky2=ky1
         elseif (f.eq.2) then
            kx1=kx1+1
            kx2=kx1
         elseif (f.eq.3) then
            ky1=ky1+1
            ky2=ky1
         elseif (f.eq.4) then
            kx1=kx1-1
            kx2=kx1
         elseif (f.eq.5) then
            kz1=kz1-1
            kz2=kz1
         elseif (f.eq.6) then
            kz1=kz1+1
            kz2=kz1
         endif

         i = 0
         do iz=kz1,kz2
         do iy=ky1,ky2
         do ix=kx1,kx2
            i = i+1
            fa(i,f,e)=va(ix,iy,iz,e)
c           write(6,*) 'fa:',fa(i,f,e),i,f,e
         enddo
         enddo
         enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine set_binv(bmnv,hmsk,n) ! Store binvm1*(hyperbolic mask)

      include 'SIZE'
      include 'PARALLEL'
      include 'MASS'

      real bmnv(n,lorder),hmsk(n)

      do i=lorder,2,-1
         call copy(bmnv(1,i),bmnv(1,i-1),n)
      enddo

      call copy (bmnv,bm1,n)  ! Fill bmnv(1,1)

      call fgslib_gs_op(gsh_fld(1),bmnv,1,1,0)  ! 1 ==> +; gsh_fld(1) is velocity

      do i=1,n
         bmnv(i,1)=hmsk(i)/bmnv(i,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine set_bdivw(bdivw,hmsk,n) ! Store binvm1*(hyperbolic mask)

      include 'SIZE'
      include 'PARALLEL'
      include 'MASS'
      include 'INPUT'
      include 'MVGEOM'
      common /scruz/ cx  (lx1*ly1*lz1*lelt)
     $ ,             cy  (lx1*ly1*lz1*lelt)
     $ ,             cz  (lx1*ly1*lz1*lelt)

      real bdivw(n,lorder),hmsk(n)

      do i=lorder,2,-1
         call copy(bdivw(1,i),bdivw(1,i-1),n)
      enddo

      call gradm1 (bdivw,cy    ,cz   , wx   )
      call gradm1 (cx   ,cy    ,cz   , wy   )
      call add2   (bdivw,cy    ,       n    )
      if (if3d) then
         call gradm1 (cx   ,cy    ,cz   , wz   )
         call add2   (bdivw,cz    ,       n    )
      endif
      call col2(bdivw,bm1,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_bmass(bmass,hmsk,n) ! Store bmass*(hyperbolic mask)

      include 'SIZE'
      include 'PARALLEL'
      include 'MASS'

      real bmass(n,lorder),hmsk(n)

      do i=lorder,2,-1
         call copy(bmass(1,i),bmass(1,i-1),n)
      enddo

      call copy (bmass,bm1,n)  ! Fill bmass(1,1)

      return
      end
c-----------------------------------------------------------------------
      subroutine setup_dg_gs(dgh,nx,ny,nz,nel,melg,vertex)

c     Global-to-local mapping for gs

      include 'SIZE'
      include 'TOTAL'

      integer   dgh,vertex(1)

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /c_is1/ glo_num_face(lf)
     $             , glo_num_vol((lx1+2)*(ly1+2)*(lz1+2)*lelt)
      integer*8 glo_num_face,glo_num_vol,ngv

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      mx = nx+2
      call set_vert(glo_num_vol,ngv,mx,nel,vertex,.false.)

      mz0 = 1
      mz1 = 1
      if (if3d) mz0 = 0
      if (if3d) mz1 = lz1+1
      call iface_vert_int8 (glo_num_face,glo_num_vol,mz0,mz1,nelt) 

      nf = lx1*lz1*2*ldim*nelt !total number of points on faces
      call fgslib_gs_setup(dgh,glo_num_face,nf,nekcomm,np)

      return
      end
c-----------------------------------------------------------------------
      subroutine dg_set_fc_ptr

c     Set up pointer to restrict u to faces ! NOTE: compact

      include 'SIZE'
      include 'TOTAL'

      integer e,f,ef

      call dsset(lx1,ly1,lz1) ! set skpdat

      nxyz  = lx1*ly1*lz1
      nxz   = lx1*lz1
      nface = 2*ldim
      nxzf  = lx1*lz1*nface ! red'd mod to area, unx, etc.

      k = 0

      do e=1,nelv
      do ef=1,nface  ! EB notation

         f      = eface1(ef)
         js1    = skpdat(1,f)
         jf1    = skpdat(2,f)
         jskip1 = skpdat(3,f)
         js2    = skpdat(4,f)
         jf2    = skpdat(5,f)
         jskip2 = skpdat(6,f)

         i = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1

            i = i+1
            k = i+nxz*(ef-1)+nxzf*(e-1)           ! face   numbering
            dg_face(k) = j1+lx1*(j2-1)+nxyz*(e-1) ! global numbering

         enddo
         enddo

      enddo
      enddo
      ndg_facex = nxzf*nelv

      return
      end
c-----------------------------------------------------------------------
      subroutine full2face(faceary, vol_ary)

      include 'SIZE'
      include 'TOTAL'

      real     faceary(lx1*lz1,2*ldim,lelt)
      real     vol_ary(lx1,ly1,lz1,lelt)
      integer  i,j

      do j=1,ndg_facex
         i=dg_face(j)
         faceary(j,1,1) = vol_ary(i,1,1,1)
      enddo


      return
      end
c-----------------------------------------------------------------------
      subroutine face2full(vol_ary, faceary)

      include 'SIZE'
      include 'TOTAL'

      real     faceary(lx1*lz1,2*ldim,lelt)
      real     vol_ary(lx1,ly1,lz1,lelt)
      integer  i,j

      n=lx1*ly1*lz1*nelfld(ifield)
      call rzero(vol_ary,n)

      do j=1,ndg_facex
         i=dg_face(j)
         vol_ary(i,1,1,1) = vol_ary(i,1,1,1)+faceary(j,1,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine add_face2full(vol_ary, faceary)

      include 'SIZE'
      include 'TOTAL'

      real     faceary(lx1*lz1,2*ldim,lelt)
      real     vol_ary(lx1,ly1,lz1,lelt)
      integer  i,j

      do j=1,ndg_facex
         i=dg_face(j)
         vol_ary(i,1,1,1) = vol_ary(i,1,1,1)+faceary(j,1,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine conv_rhs_dg_aliased (du,u,c)
c
      include 'SIZE'
      include 'TOTAL'

c     Apply convecting field c(1,ldim) to scalar field u(1).

      real du(1),u(1),c(1)

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /scrdg/ uf(lf),uxf(lf),uyf(lf),uzf(lf),upwind_wgt(lf)

      integer e,f

      n  = lx1*ly1*lz1*nelv
      nf = lx1*lz1*2*ldim*nelt


      if (ifcons) then
        if (if3d     ) call convop_cons_3d (du,u,c,lx1,lxd,nelv)
        if (.not.if3d) call convop_cons_2d (du,u,c,lx1,lxd,nelv)
      else
        if (if3d     ) call convop_fst_3d  (du,u,c,lx1,lxd,nelv)
        if (.not.if3d) call convop_fst_2d  (du,u,c,lx1,lxd,nelv)
      endif

      call full2face(uf ,u )
      call full2face(uxf,vx)
      call full2face(uyf,vy)
      call full2face(uzf,vz)
      if (.not.if3d) call rzero(uzf,nf)

      beta_u = 0.00 ! 1=full upwind; 0=central flux
      beta_u = 0.25 ! 1=full upwind; 0=central flux
      beta_u = 1.00 ! 1=full upwind; 0=central flux
      beta_u = param(98)
      if (istep.le.5.and.nio.eq.0) write(6,*) beta_u,' dg upwind'

      nface = 2*ldim
      nxz   = lx1*lz1
      k     = 0
      do e=1,nelt
      do f=1,nface
      do i=1,nxz

         k=k+1

         beta   = ( unx (i,1,f,e)*uxf(k)
     $            + uny (i,1,f,e)*uyf(k)
     $            + unz (i,1,f,e)*uzf(k))

         uf(k)  = -beta*area(i,1,f,e)*uf(k)

         upwind_wgt(k) = 1.0
         if (beta.gt.0) upwind_wgt(k) = 0.0
         upwind_wgt(k) = 0.5*(1-beta_u) + upwind_wgt(k)*beta_u
         if (beta.eq.0) upwind_wgt(k) = 0.5

      enddo
      enddo
      enddo
      
      call fgslib_gs_op(dg_hndlx,uf,1,1,0)  ! 1 ==> +
      call col2 (uf,upwind_wgt,nf)  ! Inefficient, but ok for now.
                                    ! Should be combined with
      call add_face2full( du , uf)  ! <--- this stmt.

      call fbinvert     ( du )      ! Right now works only for undeformed

      return
      end
c-----------------------------------------------------------------------
      subroutine map_faced(ju,u,mx,md,fdim,idir) ! GLL->GL interpolation

c     GLL interpolation from mx to md for a face array of size (nx,nz)

c     If idir ^= 0, then apply transpose operator  (md to mx)

      include 'SIZE'

      real    ju(1),u(1)
      integer fdim

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      parameter (ld=2*lxd)
      common /ctmp0/ w(ld**ldim,2)

      call lim_chk(md,ld,'md   ','ld   ','map_faced ')
      call lim_chk(mx,ld,'mx   ','ld   ','map_faced ')

c     call copy(ju,u,mx)
c     return

      call get_int_ptr (i,mx,md)

      if (idir.eq.0) then
         if (fdim.eq.2) then
            call mxm(jgl(i),md,u,mx,wkd,mx)
            call mxm(wkd,md,jgt(i),mx,ju,md)
         else
            call mxm(jgl(i),md,u,mx,ju,1)
         endif
      else
         if (fdim.eq.2) then
            call mxm(jgt(i),mx,u,md,wkd,md)
            call mxm(wkd,mx,jgl(i),md,ju,mx)
         else
            call mxm(jgt(i),mx,u,md,ju,1)
         endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine fbinvert(rhs) ! Still in development.  10/10/15, pff.

      include 'SIZE'
      include 'TOTAL'

      real     rhs(lx1,ly1,lz1,lelt)

      common /cfbinv/ qn(lx1),alpha_n,beta_n
     $               ,s1(ly1,lz1),bnv(lx1)
     $               ,tmp(lx1*ly1*lz1*lelt)
      integer icalld
      save    icalld
      data    icalld /0/

      integer e

      n = lx1*ly1*lz1*nelfld(ifield)

      do i=1,n                            ! FOR NOW, USE DIAGONAL MASS
         rhs(i,1,1,1)=rhs(i,1,1,1)/bm1(i,1,1,1)   ! MATRIX.   pff, 10/10/15
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine grad_rstd_ta(du,ur,us,ut,md,if3d) ! GL->GL gradt

      include 'SIZE'
      include 'DXYZ'

      real    ur(1),us(1),ut(1),u(1)

      logical if3d

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      call get_dgl_ptr (ip,md,md)
      call gradrta     (du,ur,us,ut,dgt(ip),dg(ip),dg(ip),md,md,md,if3d)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_dg_wgts

      include 'SIZE'
      include 'TOTAL'

      integer icalld
      save    icalld
      data    icalld /0/

      common /finewts/ zptf(lxd),wgtf(lxd),wghtf(lxd*lzd),wghtc(lx1*lz1)

      if (icalld.eq.0) then    !  Set fine-scale surface weights

         icalld = 1

         call zwgl(zptf,wgtf,lxd)
         if (if3d) then
            k=0
            do j=1,ly1
            do i=1,lx1
               k=k+1
               wghtc(k)=wxm1(i)*wzm1(j)
            enddo
            enddo
            k=0
            do j=1,lyd
            do i=1,lxd
               k=k+1
               wghtf(k)=wgtf(i)*wgtf(j)
            enddo
            enddo
         else
            call copy(wghtc,wxm1,lx1)
            call copy(wghtf,wgtf,lxd)
         endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine conv_rhs_dg (du,u,c)
c
      include 'SIZE'
      include 'TOTAL'

c     Apply convecting field c(1,ldim) to scalar field u(1).

      parameter(ldd=lxd*lyd*lzd)
      real du(1),u(1),c(ldd*lelv,3)

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /scrdg/ uf(lf),uxf(lf),uyf(lf),uzf(lf),upwind_wgt(lf)
     $             , beta_c(lx1*lz1),jaco_c(lx1*lz1)
     $             , beta_f(lxd*lzd),jaco_f(lxd*lzd)
     $             , ufine (lxd*lzd)
      real jaco_c,jaco_f
      common /finewts/ zptf(lxd),wgtf(lxd),wghtf(lxd*lzd),wghtc(lx1*lz1)

      integer e,f,fdim


      n  = lx1*ly1*lz1*lelv
      nf = lx1*lz1*2*ldim*lelt

      call conv_rhs_dg_weak (du,u,c(1,1),c(1,2),c(1,3))

      call full2face(uf ,u )
      call full2face(uxf,vx)
      call full2face(uyf,vy)
      call full2face(uzf,vz)
      if (.not.if3d) call rzero(uzf,nf)

      beta_u = 0.00 ! 1=full upwind; 0=central flux
      beta_u = 0.25 ! 1=full upwind; 0=central flux
      beta_u = 1.00 ! 1=full upwind; 0=central flux
      beta_u = param(98)
      if (istep.le.5.and.nio.eq.0) write(6,*) beta_u,' dg upwind'

      nface = 2*ldim
      nxz   = lx1*lz1
      nxzd  = lxd*lzd
      k     = 0
      do e=1,nelt         ! This formula for upwind weights appears to
      do f=1,nface        ! assume that U is continuous, or at least of

        kface = k+1
        do i=1,nxz        ! the same sign on either side of the interface.

         k=k+1

         beta   = ( unx (i,1,f,e)*uxf(k)
     $            + uny (i,1,f,e)*uyf(k)
     $            + unz (i,1,f,e)*uzf(k))

         upwind_wgt(k) = 1.0
         if (beta.gt.0) upwind_wgt(k) = 0.0
         upwind_wgt(k) = 0.5*(1-beta_u) + upwind_wgt(k)*beta_u

         beta_c(i)   = -beta
         jaco_c(i)   = area(i,1,f,e)/wghtc(i)

        enddo

        fdim = ldim-1 ! Dimension of face
        call map_faced(beta_f,beta_c  ,lx1,lxd,fdim,0) ! Dealiased quadrature,
        call map_faced(jaco_f,jaco_c  ,lx1,lxd,fdim,0) ! 0 --> coarse to fine,
        call map_faced(ufine,uf(kface),lx1,lxd,fdim,0) !   ufine = J uf

        do i=1,nxzd
           ufine(i)=wghtf(i)*jaco_f(i)*beta_f(i)*ufine(i)
        enddo
        call map_faced(uf(kface),ufine,lx1,lxd,fdim,1)  ! 1 --> uf = J^T ufine

      enddo
      enddo

      call fgslib_gs_op(dg_hndlx,uf,1,1,0)  ! 1 ==> +

      call col2 (uf,upwind_wgt,nf)  ! Inefficient, but ok for now.
                                    ! Should be combined with
      call add_face2full( du , uf)  ! <--- this stmt.

      call fbinvert     ( du )      ! Right now works only for undeformed

      return
      end
c-----------------------------------------------------------------------
      subroutine convect_dg(du,u,ifuf,cr,cs,ct,ifcf)
      include 'SIZE'
      include 'TOTAL'
      real du(1),u(1),cr(1),cs(1),ct(1)
      logical ifuf,ifcf

      call conv_rhs_dg_weak (du,u,cr,cs,ct)

      return
      end
c-----------------------------------------------------------------------
      subroutine convop_weak(du,u,cr,cs,ct,mx,md,nel) ! Weak Conservation form

c     Apply convecting field c to scalar field u, conservation form d/dxj cj phi

c     Assumes that current convecting field is on dealias mesh, in c()

      include 'SIZE'
      include 'TOTAL'

      parameter (lxx=lx1*ly1*lz1,ldd=lxd*lyd*lzd)
      real du(lxx,nel)
      real  u(lxx,nel)
      real  cr(ldd,nel),cs(ldd,nel),ct(ldd,nel)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju

      integer e

      nxyz  = lx1*ly1*lz1
      nrstd = md**ldim

      call lim_chk(nrstd,ldd,'urus5','ldd  ','convp_cons')

      do e=1,nel

         call intp_rstd (ju,u(1,e),mx,md,if3d,0) ! 0 = forward; on Gauss points!
         if (ldim.eq.3) then
          do i=1,ldd
            ur(i)=ju(i)*cr(i,e) ! Already in r-s-t coordinates
            us(i)=ju(i)*cs(i,e)
            ut(i)=ju(i)*ct(i,e)
            ud(i)=0
          enddo
         else
          do i=1,ldd
            ur(i)=ju(i)*cr(i,e) ! Already in r-s-t coordinates
            us(i)=ju(i)*cs(i,e)
            ud(i)=0
          enddo
         endif
         call grad_rstd_ta (ud,ur,us,ut,lxd,if3d) ! GL->GL gradt
         call intp_rstd    (du(1,e),ud,mx,md,if3d,1) ! 1 = backward; on Gauss points!
         do i=1,nxyz
            du(i,e) = -du(i,e)*binvdg(i,e)
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine conv_bdry_dg_weak (du,u) ! THIS SHOULD HAVE: ,cr,cs,ct)
c
c     Implement  Cu = Div (cu) in weak form using DG
c
      include 'SIZE'
      include 'TOTAL'

c     Apply convecting field c(1,ldim) to scalar field u(1).

      real du(1),u(1)

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /scrdg/uf(lf),uxf(lf),uyf(lf),uzf(lf),upwind_wgt(lf),us(lf)
     $             ,beta_c(lx1*lz1),jaco_c(lx1*lz1)
     $             ,beta_f(lxd*lzd),jaco_f(lxd*lzd)
     $             ,ufine (lxd*lzd)
      real jaco_c,jaco_f

      common /finewts/ zptf(lxd),wgtf(lxd),wghtf(lxd*lzd),wghtc(lx1*lz1)

      integer e,f,fdim

      n  = lx1*ly1*lz1*nelv
      nf = lx1*lz1*2*ldim*nelt

      call full2face(uf ,u )
      call full2face(uxf,vx)
      call full2face(uyf,vy)
      call full2face(uzf,vz)
      if (.not.if3d) call rzero(uzf,nf)

      beta_u = 1.00 ! 1=full upwind; 0=central flux

      nface = 2*ldim
      nxz   = lx1*lz1
      nxzd  = lxd*lzd
      k     = 0
      do e=1,nelt         ! This formula for upwind weights appears to
      do f=1,nface        ! assume that U is continuous, or at least of

       kface = k+1
       if (fw(f,e).gt.0.6) then

        do i=1,nxz        ! the same sign on either side of the interface.

         k=k+1

         beta   = ( unx (i,1,f,e)*uxf(k)
     $            + uny (i,1,f,e)*uyf(k)
     $            + unz (i,1,f,e)*uzf(k))

         upwind_wgt(k) = 0.0
         if (beta.lt.0) upwind_wgt(k) = 1.0

         beta_c(i)   = beta
         jaco_c(i)   = area(i,1,f,e)/wghtc(i)

        enddo

        fdim = ldim-1 ! Dimension of face
        call map_faced(beta_f,beta_c  ,lx1,lxd,fdim,0) ! Dealiased quadrature,
        call map_faced(jaco_f,jaco_c  ,lx1,lxd,fdim,0) ! 0 --> coarse to fine,
        call map_faced(ufine,uf(kface),lx1,lxd,fdim,0) !   ufine = J uf

        do i=1,nxzd
           ufine(i)=wghtf(i)*jaco_f(i)*beta_f(i)*ufine(i)
        enddo
        call map_faced(uf(kface),ufine,lx1,lxd,fdim,1)  ! 1 --> uf = J^T ufine

       else
        do i=1,nxz
         k=k+1
         upwind_wgt(k) = 0.0
        enddo
       endif

      enddo
      enddo

      do j=1,ndg_facex
         i=dg_face(j)
         du(i) =  du(i) - ( upwind_wgt(j)*uf(j) )
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine conv_rhs_dg_weak (du,u,cr,cs,ct)
c
c     Implement  Cu = Div (cu) in weak form using DG
c
      include 'SIZE'
      include 'TOTAL'

c     Apply convecting field c(1,ldim) to scalar field u(1).

      real du(1),u(1),cr(1),cs(1),ct(1)

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /scrdg/uf(lf),uxf(lf),uyf(lf),uzf(lf),upwind_wgt(lf),us(lf)
     $             ,beta_c(lx1*lz1),jaco_c(lx1*lz1)
     $             ,beta_f(lxd*lzd),jaco_f(lxd*lzd)
     $             ,ufine (lxd*lzd)
      real jaco_c,jaco_f

      common /finewts/ zptf(lxd),wgtf(lxd),wghtf(lxd*lzd),wghtc(lx1*lz1)

      integer e,f,fdim

      n  = lx1*ly1*lz1*lelv
      nf = lx1*lz1*2*ldim*lelt


      call convop_weak (du,u,cr,cs,ct,lx1,lxd,nelv)  ! Volumetric term

      call full2face(uf ,u )
      call full2face(uxf,vx)
      call full2face(uyf,vy)
      call full2face(uzf,vz)
      if (.not.if3d) call rzero(uzf,nf)

      beta_u = 0.25 ! 1=full upwind; 0=central flux
      beta_u = 0.00 ! 1=full upwind; 0=central flux
      beta_u = 1.00 ! 1=full upwind; 0=central flux
      if (istep.le.5.and.nio.eq.0) write(6,*) beta_u,' dg upwind'

      nface = 2*ldim
      nxz   = lx1*lz1
      nxzd  = lxd*lzd
      k     = 0
      do e=1,nelt         ! This formula for upwind weights appears to
      do f=1,nface        ! assume that U is continuous, or at least of

        kface = k+1
        do i=1,nxz        ! the same sign on either side of the interface.
            k=k+1
            beta   = ( unx (i,1,f,e)*uxf(k)
     $               + uny (i,1,f,e)*uyf(k)
     $               + unz (i,1,f,e)*uzf(k) )

            upwind_wgt(k) = 0.0
            if (beta.gt.0) upwind_wgt(k) = 1.0
            upwind_wgt(k) = 0.5*(1-beta_u) + beta_u*(1-upwind_wgt(k))

            if (fw(f,e).gt.0.6 .and. beta.lt.0) upwind_wgt(k)=1.
            if (fw(f,e).gt.0.6 .and. beta.gt.0) upwind_wgt(k)=0.

            beta_c(i)   = beta
            jaco_c(i)   = area(i,1,f,e)/wghtc(i)
        enddo

        fdim = ldim-1 ! Dimension of face
        call map_faced(beta_f,beta_c  ,lx1,lxd,fdim,0) ! Dealiased quadrature,
        call map_faced(jaco_f,jaco_c  ,lx1,lxd,fdim,0) ! 0 --> coarse to fine,
        call map_faced(ufine,uf(kface),lx1,lxd,fdim,0) !   ufine = J uf

        do i=1,nxzd
           ufine(i)=wghtf(i)*jaco_f(i)*beta_f(i)*ufine(i)
        enddo
        call map_faced(uf(kface),ufine,lx1,lxd,fdim,1)  ! 1 --> uf = J^T ufine
        call copy     (us(kface),uf(kface),lx1*lz1)     ! Save uf for later recombination

      enddo
      enddo

      call fgslib_gs_op(dg_hndlx,uf,1,1,0)  ! 1 ==> +  :   uf <-- uf^- + uf^+

      do j=1,ndg_facex
         i=dg_face(j)
         du(i) = du(i) + ( us(j)-upwind_wgt(j)*uf(j) )*binvdg(i,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
