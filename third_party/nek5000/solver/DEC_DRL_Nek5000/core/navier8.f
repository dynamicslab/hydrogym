c-----------------------------------------------------------------------
c
      subroutine set_vert(glo_num,ngv,nx,nel,vertex,ifcenter)
c
c     Given global array, vertex, pointing to hex vertices, set up
c     a new array of global pointers for an nx^ndim set of elements.
c
      include 'SIZE'
      include 'INPUT'
c
      integer*8 glo_num(1),ngv
      integer vertex(1),nx
      logical ifcenter

      if (if3d) then
         call setvert3d(glo_num,ngv,nx,nel,vertex,ifcenter)
      else
         call setvert2d(glo_num,ngv,nx,nel,vertex,ifcenter)
      endif

c     Check for single-element periodicity 'p' bc
      nz = 1
      if (if3d) nz = nx
      call check_p_bc(glo_num,nx,nx,nz,nel)

      if(nio.eq.0) write(6,*) 'call usrsetvert'
      call usrsetvert(glo_num,nel,nx,nx,nz)
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrsetvert'

      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine crs_solve_l2(uf,vf)
c
c     Given an input vector v, this generates the H1 coarse-grid solution
c
      include 'SIZE'
      include 'DOMAIN'
      include 'ESOLV'
      include 'GEOM'
      include 'PARALLEL'
      include 'SOLN'
      include 'INPUT'
      include 'TSTEP'
      real uf(1),vf(1)
      common /scrpre/ uc(lcr*lelt),w(2*lx1*ly1*lz1)



      call map_f_to_c_l2_bilin(uf,vf,w)
      call crs_solve(xxth(ifield),uc,uf)
      call map_c_to_f_l2_bilin(uf,uc,w)

      return
      end
c
c-----------------------------------------------------------------------
c      subroutine test_h1_crs
c      include 'SIZE'
c      include 'DOMAIN'
c      common /scrxxt/ x(lcr*lelv),b(lcr*lelv)
c      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
c      real x,b
c      ntot=nelv*nxyz_c
c      do i=1,12
c         call rzero(b,ntot)
c         if(mp.eq.1) then
c            b(i)=1
c         else if(mid.eq.0) then
c            if(i.gt.8) b(i-8)=1
c         else
c            if(i.le.8) b(i)=1
c         endif
c         call hsmg_coarse_solve(x,b)
c         print *, 'Column ',i,':',(x(j),j=1,ntot)
c      enddo
c      return
c      end
c-----------------------------------------------------------------------
c
      subroutine set_up_h1_crs

      include 'SIZE'
      include 'GEOM'
      include 'DOMAIN'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      common /ivrtx/ vertex ((2**ldim)*lelt)
      integer vertex

      integer gs_handle
      integer null_space,e

      character*3 cb
      common /scrxxt/ cmlt(lcr,lelv),mask(lcr,lelv)
      common /scrxxti/ ia(lcr,lcr,lelv), ja(lcr,lcr,lelv)
      real mask
      integer ia,ja
      real z

      integer key(2),aa(2)
      common /scrch/ iwork(2,lx1*ly1*lz1*lelv)
      common /scrns/ w(7*lx1*ly1*lz1*lelv)
      common /vptsol/ a(27*lx1*ly1*lz1*lelv)
      integer w
      real wr(1)
      equivalence (wr,w)

      common /scrvhx/ h1(lx1*ly1*lz1*lelv),h2(lx1*ly1*lz1*lelv)
      common /scrmgx/ w1(lx1*ly1*lz1*lelv),w2(lx1*ly1*lz1*lelv)

      integer*8 ngv

      t0 = dnekclock()

c     nxc is order of coarse grid space + 1, nxc=2, linear, 3=quad,etc.
c     nxc=param(82)
c     if (nxc.gt.lxc) then
c        nxc=lxc
c        write(6,*) 'WARNING :: coarse grid space too large',nxc,lxc 
c     endif
c     if (nxc.lt.2) nxc=2

      nxc     = 2
      nx_crs  = nxc

      if(nio.eq.0) write(6,*) 'setup h1 coarse grid, nx_crs=', nx_crs

      ncr     = nxc**ndim
      nxyz_c  = ncr
c
c     Set SEM_to_GLOB
c
      call get_vertex
      call set_vert(se_to_gcrs,ngv,nxc,nelv,vertex,.true.)

c     Set mask
      z=0
      ntot=nelv*nxyz_c
      nzc=1
      if (if3d) nzc=nxc
      call rone(mask,ntot)
      call rone(cmlt,ntot)
      nfaces=2*ndim
c     ifield=1			!c? avo: set in set_overlap through 'TSTEP'?

      if (ifield.eq.1) then
         do ie=1,nelv
         do iface=1,nfaces
            cb=cbc(iface,ie,ifield)
            if (cb.eq.'o  '  .or.  cb.eq.'on '  .or. 
     $          cb.eq.'O  '  .or.  cb.eq.'ON '  .or.  cb.eq.'MM '  .or.
     $          cb.eq.'mm '  .or.  cb.eq.'ms '  .or.  cb.eq.'MS ')
     $           call facev(mask,ie,iface,z,nxc,nxc,nzc) ! 'S* ' & 's* ' ?avo?
         enddo
         enddo
      elseif (ifield.eq.ifldmhd) then   ! no ifmhd ?avo?
         do ie=1,nelv
         do iface=1,nfaces
            cb=cbc(iface,ie,ifield)
            if (cb.eq.'ndd'  .or.  cb.eq.'dnd'  .or.  cb.eq.'ddn')
     $          call facev(mask,ie,iface,z,nxc,nxc,nzc)
         enddo
         enddo
      endif

c     Set global index of dirichlet nodes to zero; xxt will ignore them

      call gs_setup(gs_handle,se_to_gcrs,ntot,nekcomm,mp)
      call gs_op   (gs_handle,mask,1,2,0)  !  "*"
      call gs_op   (gs_handle,cmlt,1,1,0)  !  "+"
      call gs_free (gs_handle)
      call set_jl_crs_mask(ntot,mask,se_to_gcrs)

      call invcol1(cmlt,ntot)

c     Setup local SEM-based Neumann operators (for now, just full...)

c      if (param(51).eq.1) then     ! old coarse grid
c         nxyz1=nx1*ny1*nz1
c         lda = 27*nxyz1*lelt
c         ldw =  7*nxyz1*lelt
c         call get_local_crs(a,lda,nxc,h1,h2,w,ldw)
c      else
c        NOTE: a(),h1,...,w2() must all be large enough
         n = nx1*ny1*nz1*nelv
         call rone (h1,n)
         call rzero(h2,n)
         call get_local_crs_galerkin(a,ncr,nxc,h1,h2,w1,w2)
c      endif

      call set_mat_ij(ia,ja,ncr,nelv)
      null_space=0
      if (ifield.eq.1) then
         if (ifvcor)  null_space=1
      elseif (ifield.eq.ifldmhd) then
         if (ifbcor)  null_space=1
      endif

      nz=ncr*ncr*nelv
      call crs_setup(xxth(ifield),nekcomm,mp, ntot,se_to_gcrs,
     $               nz,ia,ja,a, null_space)
c     call crs_stats(xxth(ifield))

      t0 = dnekclock()-t0
      if (nio.eq.0) then
         write(6,*) 'done :: setup h1 coarse grid ',t0, ' sec'
         write(6,*) ' '
      endif

      return
      end
c
c-----------------------------------------------------------------------
      subroutine set_jl_crs_mask(n, mask, se_to_gcrs)
      real mask(1)
      integer*8 se_to_gcrs(1)
      do i=1,n
         if(mask(i).lt.0.1) se_to_gcrs(i)=0
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine set_mat_ij(ia,ja,n,ne)
      integer n,ne
      integer ia(n,n,ne), ja(n,n,ne)
c
      integer i,j,ie
      do ie=1,ne
      do j=1,n
      do i=1,n
         ia(i,j,ie)=(ie-1)*n+i-1
         ja(i,j,ie)=(ie-1)*n+j-1
      enddo
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine irank_vec(ind,nn,a,m,n,key,nkey,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is destroyed
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),aa(m)
      logical iftuple_ianeb,a_ne_b
c
      if (m.eq.1) then
c
         write(6,*) 
     $        'WARNING: For single key, not clear that rank is unique!'
         call irank(a,ind,n)
         return
      endif
c
c
      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)
c
c     Find unique a's
c
      nn=1
c
      call icopy(aa,a,m)
      a(1,1) = nn
      a(2,1)=ind(1)
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
         if (a_ne_b) then
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         a(1,i) = nn
         a(2,i) = ind(i)
      enddo
c
c     Set ind() to rank
c
      do i=1,n
         iold=a(2,i)
         ind(iold) = a(1,i)
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine ituple_sort(a,lda,n,key,nkey,ind,aa)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer a(lda,n),aa(lda)
      integer ind(1),key(nkey)
      logical iftuple_ialtb
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
c           aa  = a  (l)
            call icopy(aa,a(1,l),lda)
            ii  = ind(l)
         else
c           aa =   a(ir)
            call icopy(aa,a(1,ir),lda)
            ii = ind(ir)
c           a(ir) =   a( 1)
            call icopy(a(1,ir),a(1,1),lda)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
c              a(1) = aa
               call icopy(a(1,1),aa,lda)
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if (iftuple_ialtb(a(1,j),a(1,j+1),key,nkey)) j=j+1
            endif
            if (iftuple_ialtb(aa,a(1,j),key,nkey)) then
c              a(i) = a(j)
               call icopy(a(1,i),a(1,j),lda)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
c        a(i) = aa
         call icopy(a(1,i),aa,lda)
         ind(i) = ii
      GOTO 100
      end
c
c-----------------------------------------------------------------------
c
      subroutine tuple_sort(a,lda,n,key,nkey,ind,aa)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      real a(lda,n),aa(lda)
      integer ind(1),key(nkey)
      logical iftuple_altb
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
c           aa  = a  (l)
            call copy(aa,a(1,l),lda)
            ii  = ind(l)
         else
c           aa =   a(ir)
            call copy(aa,a(1,ir),lda)
            ii = ind(ir)
c           a(ir) =   a( 1)
            call copy(a(1,ir),a(1,1),lda)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
c              a(1) = aa
               call copy(a(1,1),aa,lda)
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
c              if ( a(j).lt.a(j+1) ) j=j+1
               if (iftuple_altb(a(1,j),a(1,j+1),key,nkey)) j=j+1
            endif
c           if (aa.lt.a(j)) then
            if (iftuple_altb(aa,a(1,j),key,nkey)) then
c              a(i) = a(j)
               call copy(a(1,i),a(1,j),lda)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
c        a(i) = aa
         call copy(a(1,i),aa,lda)
         ind(i) = ii
      GOTO 100
      end
c
c-----------------------------------------------------------------------
c
      logical function iftuple_ialtb(a,b,key,nkey)
      integer a(1),b(1)
      integer key(nkey)
c
      do i=1,nkey
         k=key(i)
         if (a(k).lt.b(k)) then
            iftuple_ialtb = .true.
            return
         elseif (a(k).gt.b(k)) then
            iftuple_ialtb = .false.
            return
         endif
      enddo
      iftuple_ialtb = .false.
      return
      end
c
c-----------------------------------------------------------------------
c
      logical function iftuple_altb(a,b,key,nkey)
      real a(1),b(1)
      integer key(nkey)
c
      do i=1,nkey
         k=key(i)
         if (a(k).lt.b(k)) then
            iftuple_altb = .true.
            return
         elseif (a(k).gt.b(k)) then
            iftuple_altb = .false.
            return
         endif
      enddo
      iftuple_altb = .false.
      return
      end
c
c-----------------------------------------------------------------------
c
      logical function iftuple_ianeb(a,b,key,nkey)
      integer a(1),b(1)
      integer key(nkey)
c
      do i=1,nkey
         k=key(i)
         if (a(k).ne.b(k)) then
            iftuple_ianeb = .true.
            return
         endif
      enddo
      iftuple_ianeb = .false.
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine get_local_crs(a,lda,nxc,h1,h2,w,ldw)
c
c     This routine generates Nelv submatrices of order nxc^ndim.
c
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'DOMAIN'
      include 'PARALLEL'
c
c
c     Generate local triangular matrix
c
      real   a(1),h1(1),h2(1),w(ldw)
c
      parameter (lcrd=lx1**ldim)
      common /ctmp1/ x(lcrd),y(lcrd),z(lcrd)
c
c
      ncrs_loc = nxc**ndim
      n2       = ncrs_loc*ncrs_loc
c
c     Required storage for a:
      nda = n2*nelv
      if (nda.gt.lda) then
         write(6,*)nid,'ERROR: increase storage get_local_crs:',nda,lda
         call exitt
      endif
c
c
      l = 1
      do ie=1,nelv
c
         call map_m_to_n(x,nxc,xm1(1,1,1,ie),nx1,if3d,w,ldw)
         call map_m_to_n(y,nxc,ym1(1,1,1,ie),nx1,if3d,w,ldw)
         if (if3d) call map_m_to_n(z,nxc,zm1(1,1,1,ie),nx1,if3d,w,ldw)
c.later. call map_m_to_n(hl1,nxc,h1(1,1,1,ie),nx1,if3d,w,ldw)
c.later. call map_m_to_n(hl2,nxc,h2(1,1,1,ie),nx1,if3d,w,ldw)
c
         call a_crs_enriched(a(l),h1,h2,x,y,z,nxc,if3d,ie)
         l=l+n2
c
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine a_crs_enriched(a,h1,h2,x1,y1,z1,nxc,if3d,ie)
c
c     This sets up a matrix for a single array of tensor-product
c     gridpoints (e.g., an array defined by SEM-GLL vertices)
c
c         For example, suppose ndim=3.
c
c         Then, there would be ncrs_loc := nxc^3 dofs for this matrix,
c
c         and the matrix size would be (ncrs_loc x ncrs_loc).
c
c
c
      include 'SIZE'
c
      real a(1),h1(1),h2(1)
      real x1(nxc,nxc,1),y1(nxc,nxc,1),z1(nxc,nxc,1)
      logical if3d
c
      parameter (ldm2=2**ldim)
      real a_loc(ldm2,ldm2)
      real x(8),y(8),z(8)
c
      ncrs_loc = nxc**ndim
      n2       = ncrs_loc*ncrs_loc
      call rzero(a,n2)
c
      nyc=nxc
      nzc=2
      if (if3d) nzc=nxc
      nz =0
      if (if3d) nz=1
c
c     Here, we march across sub-cubes
c
      do kz=1,nzc-1
      do ky=1,nyc-1
      do kx=1,nxc-1
         k = 0
         do iz=0,nz
         do iy=0,1
         do ix=0,1
            k = k+1
            x(k) = x1(kx+ix,ky+iy,kz+iz)
            y(k) = y1(kx+ix,ky+iy,kz+iz)
            z(k) = z1(kx+ix,ky+iy,kz+iz)
         enddo
         enddo
         enddo
         if (if3d) then
            call a_crs_3d(a_loc,h1,h2,x,y,z,ie)
         else
            call a_crs_2d(a_loc,h1,h2,x,y,ie)
         endif
c        call outmat(a_loc,ldm2,ldm2,'A_loc ',ie)
c
c        Assemble:
c
         j = 0
         do jz=0,nz
         do jy=0,1
         do jx=0,1
            j = j+1
            ja = (kx+jx) + nxc*(ky+jy-1) + nxc*nyc*(kz+jz-1)
c
            i = 0
            do iz=0,nz
            do iy=0,1
            do ix=0,1
               i   = i+1
               ia  = (kx+ix) + nxc*(ky+iy-1) + nxc*nyc*(kz+iz-1)
c
               ija = ia + ncrs_loc*(ja-1)
               a(ija) = a(ija) + a_loc(i,j)
c
            enddo
            enddo
            enddo
c
         enddo
         enddo
         enddo
      enddo
      enddo
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine a_crs_3d(a,h1,h2,xc,yc,zc,ie)
c
c     Generate stiffness matrix for 3D coarse grid problem.
c     
c     This is done by using two tetrahedrizations of each
c     hexahedral subdomain (element) such that each of the
c     6 panels (faces) on the sides of an element has a big X.
c
c
      real a(0:7,0:7),h1(0:7),h2(0:7)
      real xc(0:7),yc(0:7),zc(0:7)
c
      real a_loc(4,4)
      real xt(4),yt(4),zt(4)
c
      integer vertex(4,5,2)
      save    vertex
      data    vertex / 000 ,  001 , 010 , 100
     $               , 000 ,  001 , 011 , 101 
     $               , 011 ,  010 , 000 , 110 
     $               , 011 ,  010 , 001 , 111 
     $               , 000 ,  110 , 101 , 011
c
     $               , 101 ,  100 , 110 , 000
     $               , 101 ,  100 , 111 , 001 
     $               , 110 ,  111 , 100 , 010 
     $               , 110 ,  111 , 101 , 011 
     $               , 111 ,  001 , 100 , 010  /
c
      integer icalld
      save    icalld
      data    icalld/0/
c
      if (icalld.eq.0) then
         do i=1,40
            call bindec(vertex(i,1,1))
         enddo
      endif
      icalld=icalld+1
c
      call rzero(a,64)
      do k=1,10
         do iv=1,4
            xt(iv) = xc(vertex(iv,k,1))
            yt(iv) = yc(vertex(iv,k,1))
            zt(iv) = zc(vertex(iv,k,1))
         enddo
         call get_local_A_tet(a_loc,xt,yt,zt,k,ie)
         do j=1,4
            jj = vertex(j,k,1)
            do i=1,4
               ii = vertex(i,k,1)
               a(ii,jj) = a(ii,jj) + 0.5*a_loc(i,j)
            enddo
         enddo
      enddo
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine bindec(bin_in)
      integer bin_in,d,b,b2
c
      keep  = bin_in
      d  = bin_in
      b2 = 1
      b  = 0
      do l=1,12
         b  = b + b2*mod(d,10)
         d  = d/10
         b2 = b2*2
         if (d.eq.0) goto 1
      enddo
    1 continue
      bin_in = b
      return
      end
c
c-----------------------------------------------------------------------
      subroutine get_local_A_tet(a,x,y,z,kt,ie)
c
c     Generate local tetrahedral matrix
c
c
      real a(4,4), g(4,4)
      real x(4),y(4),z(4)
c
   11 continue
      x23 = x(2) - x(3)
      y23 = y(2) - y(3)
      z23 = z(2) - z(3)
      x34 = x(3) - x(4)
      y34 = y(3) - y(4)
      z34 = z(3) - z(4)
      x41 = x(4) - x(1)
      y41 = y(4) - y(1)
      z41 = z(4) - z(1)
      x12 = x(1) - x(2)
      y12 = y(1) - y(2)
      z12 = z(1) - z(2)
c
      xy234 = x34*y23 - x23*y34
      xy341 = x34*y41 - x41*y34
      xy412 = x12*y41 - x41*y12
      xy123 = x12*y23 - x23*y12
      xz234 = x23*z34 - x34*z23
      xz341 = x41*z34 - x34*z41
      xz412 = x41*z12 - x12*z41
      xz123 = x23*z12 - x12*z23
      yz234 = y34*z23 - y23*z34
      yz341 = y34*z41 - y41*z34
      yz412 = y12*z41 - y41*z12
      yz123 = y12*z23 - y23*z12
c
      g(1,1) = -(x(2)*yz234 + y(2)*xz234 + z(2)*xy234)
      g(2,1) = -(x(3)*yz341 + y(3)*xz341 + z(3)*xy341)
      g(3,1) = -(x(4)*yz412 + y(4)*xz412 + z(4)*xy412)
      g(4,1) = -(x(1)*yz123 + y(1)*xz123 + z(1)*xy123)
      g(1,2) = yz234
      g(2,2) = yz341
      g(3,2) = yz412
      g(4,2) = yz123
      g(1,3) = xz234
      g(2,3) = xz341
      g(3,3) = xz412
      g(4,3) = xz123
      g(1,4) = xy234
      g(2,4) = xy341
      g(3,4) = xy412
      g(4,4) = xy123
c
c        vol36 = 1/(36*volume) = 1/(6*determinant)
c
      det = x(1)*yz234 + x(2)*yz341 + x(3)*yz412 + x(4)*yz123
      vol36 = 1.0/(6.0*det)
      if (vol36.lt.0) then
         write(6,*) 'Error: tetrahedron not right-handed',ie
         write(6,1) 'x',(x(k),k=1,4)
         write(6,1) 'y',(y(k),k=1,4)
         write(6,1) 'z',(z(k),k=1,4)
 1       format(a1,1p4e15.5)

c        call exitt                 ! Option 1

         xx = x(1)                  ! Option 2
         x(1) = x(2)                !  -- this is the option that 
         x(2) = xx                  !     actually works. 11/25/07

         xx = y(1)
         y(1) = y(2)
         y(2) = xx

         xx = z(1)
         z(1) = z(2)
         z(2) = xx

         goto 11

c        call rzero(a,16)           ! Option 3
c        return

c        vol36 = abs(vol36)         ! Option 4

      endif
c
      do j=1,4
         do i=1,4
            a(i,j)=vol36*(g(i,2)*g(j,2)+g(i,3)*g(j,3)+g(i,4)*g(j,4))
         enddo
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine a_crs_2d(a,h1,h2,x,y,ie)
c
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'DOMAIN'
      include 'PARALLEL'
c
c     Generate local triangle-based stiffnes matrix for quad
c
      real a(4,4),h1(1),h2(1)
      real x(1),y(1)
c
c     Triangle to Square pointers
c
      integer elem(3,2)
      save    elem
      data    elem / 1,2,4  ,  1,4,3 /
c
      real a_loc(3,3)
c
c
      call rzero(a,16)
c
      do i=1,2
         j1 = elem(1,i)
         j2 = elem(2,i)
         j3 = elem(3,i)
         x1=x(j1)
         y1=y(j1)
         x2=x(j2)
         y2=y(j2)
         x3=x(j3)
         y3=y(j3)
c
         y23=y2-y3
         y31=y3-y1
         y12=y1-y2
c
         x32=x3-x2
         x13=x1-x3
         x21=x2-x1
c
c        area4 = 1/(4*area)
         area4 = 0.50/(x21*y31 - y12*x13)
c
         a_loc(1, 1) = area4*( y23*y23+x32*x32 )
         a_loc(1, 2) = area4*( y23*y31+x32*x13 )
         a_loc(1, 3) = area4*( y23*y12+x32*x21 )
c
         a_loc(2, 1) = area4*( y31*y23+x13*x32 )
         a_loc(2, 2) = area4*( y31*y31+x13*x13 )
         a_loc(2, 3) = area4*( y31*y12+x13*x21 )
c
         a_loc(3, 1) = area4*( y12*y23+x21*x32 )
         a_loc(3, 2) = area4*( y12*y31+x21*x13 )
         a_loc(3, 3) = area4*( y12*y12+x21*x21 )
c
c        Store in "4 x 4" format
c
         do il=1,3
            iv = elem(il,i)
            do jl=1,3
               jv = elem(jl,i)
               a(iv,jv) = a(iv,jv) + a_loc(il,jl)
            enddo
         enddo
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine map_m_to_n(a,na,b,nb,if3d,w,ldw)
c
c     Input:   b
c     Output:  a
c
      real a(1),b(1),w(1)
      logical if3d
c
      parameter(lx=50)
      real za(lx),zb(lx)
c
      real iba(lx*lx),ibat(lx*lx)
      save iba,ibat
c
      integer nao,nbo
      save    nao,nbo
      data    nao,nbo  / -9, -9/
c
      if (na.gt.lx.or.nb.gt.lx) then
         write(6,*)'ERROR: increase lx in map_m_to_n to max:',na,nb
         call exitt
      endif
c
      if (na.ne.nao  .or.   nb.ne.nbo) then
         nao = na
         nbo = nb
         call zwgll(za,w,na)
         call zwgll(zb,w,nb)
         call igllm(iba,ibat,zb,za,nb,na,nb,na)
      endif
c
      call specmpn(a,na,b,nb,iba,ibat,if3d,w,ldw)
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine specmpn(b,nb,a,na,ba,ab,if3d,w,ldw)
C
C     -  Spectral interpolation from A to B via tensor products
C     -  scratch arrays: w(na*na*nb + nb*nb*na)
C
C     5/3/00  -- this routine replaces specmp in navier1.f, which
c                has a potential memory problem
C
C
      logical if3d
c
      real b(nb,nb,nb),a(na,na,na)
      real w(ldw)
c
      ltest = na*nb
      if (if3d) ltest = na*na*nb + nb*na*na
      if (ldw.lt.ltest) then
         write(6,*) 'ERROR specmp:',ldw,ltest,if3d
         call exitt
      endif
c
      if (if3d) then
         nab = na*nb
         nbb = nb*nb
         call mxm(ba,nb,a,na,w,na*na)
         k=1
         l=na*na*nb + 1
         do iz=1,na
            call mxm(w(k),nb,ab,na,w(l),nb)
            k=k+nab
            l=l+nbb
         enddo
         l=na*na*nb + 1
         call mxm(w(l),nbb,ab,na,b,nb)
      else
         call mxm(ba,nb,a,na,w,na)
         call mxm(w,nb,ab,na,b,nb)
      endif
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine irank(A,IND,N)
C
C     Use Heap Sort (p 233 Num. Rec.), 5/26/93 pff.
C
      integer A(1),IND(1)
C
      if (n.le.1) return
      DO 10 J=1,N
         IND(j)=j
   10 continue
C
      if (n.eq.1) return
      L=n/2+1
      ir=n
  100 continue
         IF (l.gt.1) THEN
            l=l-1
            indx=ind(l)
            q=a(indx)
         ELSE
            indx=ind(ir)
            q=a(indx)
            ind(ir)=ind(1)
            ir=ir-1
            if (ir.eq.1) then
               ind(1)=indx
               return
            endif
         ENDIF
         i=l
         j=l+l
  200    continue
         IF (J.le.IR) THEN
            IF (J.lt.IR) THEN
               IF ( A(IND(j)).lt.A(IND(j+1)) ) j=j+1
            ENDIF
            IF (q.lt.A(IND(j))) THEN
               IND(I)=IND(J)
               I=J
               J=J+J
            ELSE
               J=IR+1
            ENDIF
         GOTO 200
         ENDIF
         IND(I)=INDX
      GOTO 100
      END
c-----------------------------------------------------------------------
      subroutine iranku(r,input,n,w,ind)
c
c     Return the rank of each input value, and the maximum rank.
c
c     OUTPUT:    r(k) = rank of each entry,  k=1,..,n
c                maxr = max( r )
c                w(i) = sorted & compressed list of input values
c
      integer r(1),input(1),ind(1),w(1)
c
      call icopy(r,input,n)
      call isort(r,ind,n)
c
      maxr  = 1
      rlast = r(1) 
      do i=1,n
c        Bump rank only when r_i changes
         if (r(i).ne.rlast) then
            rlast = r(i)
            maxr  = maxr + 1
         endif
         r(i) = maxr
      enddo
      call iunswap(r,ind,n,w)
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine ifacev_redef(a,ie,iface,val,nx,ny,nz)
C
C     Assign the value VAL to face(IFACE,IE) of array A.
C     IFACE is the input in the pre-processor ordering scheme.
C
      include 'SIZE'
      integer a(nx,ny,nz,lelt),val
      call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
      do 100 iz=kz1,kz2
      do 100 iy=ky1,ky2
      do 100 ix=kx1,kx2
         a(ix,iy,iz,ie)=val
  100 continue
      return
      end
c
c-----------------------------------------------------------------------
      subroutine map_c_to_f_l2_bilin(uf,uc,w)
c
c     H1 Iterpolation operator:  linear --> spectral GLL mesh
c
      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'
c
      parameter (lxyz = lx2*ly2*lz2)
      real uc(nxyz_c,lelt),uf(lxyz,lelt),w(1)

      ltot22 = 2*lx2*ly2*lz2
      nx_crs = 2   ! bilinear only

      do ie=1,nelv
         call maph1_to_l2(uf(1,ie),nx2,uc(1,ie),nx_crs,if3d,w,ltot22)
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine map_f_to_c_l2_bilin(uc,uf,w)

c     TRANSPOSE of L2 Iterpolation operator:                    T
c                                 (linear --> spectral GLL mesh)

      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'

      parameter (lxyz = lx2*ly2*lz2)
      real uc(nxyz_c,lelt),uf(lxyz,lelt),w(1)

      ltot22 = 2*lx2*ly2*lz2
      nx_crs = 2   ! bilinear only

      do ie=1,nelv
         call maph1_to_l2t(uc(1,ie),nx_crs,uf(1,ie),nx2,if3d,w,ltot22)
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine maph1_to_l2(a,na,b,nb,if3d,w,ldw)
c
c     Input:   b
c     Output:  a
c
      real a(1),b(1),w(1)
      logical if3d
c
      parameter(lx=50)
      real za(lx),zb(lx)
c
      real iba(lx*lx),ibat(lx*lx)
      save iba,ibat
c
      integer nao,nbo
      save    nao,nbo
      data    nao,nbo  / -9, -9/
c
c
      if (na.gt.lx.or.nb.gt.lx) then
         write(6,*)'ERROR: increase lx in maph1_to_l2 to max:',na,nb
         call exitt
      endif
c
      if (na.ne.nao  .or.   nb.ne.nbo) then
         nao = na
         nbo = nb
         call zwgl (za,w,na)
         call zwgll(zb,w,nb)
         call igllm(iba,ibat,zb,za,nb,na,nb,na)
      endif
c
      call specmpn(a,na,b,nb,iba,ibat,if3d,w,ldw)
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine maph1_to_l2t(b,nb,a,na,if3d,w,ldw)
c
c     Input:   a
c     Output:  b
c
      real a(1),b(1),w(1)
      logical if3d
c
      parameter(lx=50)
      real za(lx),zb(lx)
c
      real iba(lx*lx),ibat(lx*lx)
      save iba,ibat
c
      integer nao,nbo
      save    nao,nbo
      data    nao,nbo  / -9, -9/
c
c
      if (na.gt.lx.or.nb.gt.lx) then
         write(6,*)'ERROR: increase lx in maph1_to_l2 to max:',na,nb
         call exitt
      endif
c
      if (na.ne.nao  .or.   nb.ne.nbo) then
         nao = na
         nbo = nb
         call zwgl (za,w,na)
         call zwgll(zb,w,nb)
         call igllm(iba,ibat,zb,za,nb,na,nb,na)
      endif
c
      call specmpn(b,nb,a,na,ibat,iba,if3d,w,ldw)
c
      return
      end
c
c-----------------------------------------------------------------------
      subroutine irank_vec_tally(ind,nn,a,m,n,key,nkey,key2,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is destroyed
c               a(1,i) tally of preceding structure values
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),key2(0:3),aa(m)
      logical iftuple_ianeb,a_ne_b
c
c
      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)
c     do i=1,n
c        write(6,*) i,' sort:',(a(k,i),k=1,3)
c     enddo
c
c
c     Find unique a's
c
      call icopy(aa,a,m)
      nn=1
      mm=0
c
      a(1,1) = nn
      a(2,1)=ind(1)
      a(3,1)=mm
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
         if (a_ne_b) then              ! new structure
            ms = aa(3)                 ! structure type
            if (aa(2).eq.0) ms = aa(2) ! structure type
            mm = mm+key2(ms)           ! n dofs
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         a(1,i) = nn
         a(2,i) = ind(i)
         a(3,i) = mm
      enddo
      ms = aa(3)
      if (aa(2).eq.0) ms = aa(2) ! structure type
      nn = mm+key2(ms)
c
c     Set ind() to rank
c
      do i=1,n
         iold=a(2,i)
         ind(iold) = a(1,i)
      enddo
c
c     Set a1() to number of preceding dofs
c
      do i=1,n
         iold=a(2,i)
         a(1,iold) = a(3,i)
      enddo
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine out_se1(se2crs,nx,name)
c
      include 'SIZE'
      integer se2crs(nx,nx,1)
      character*4 name
c
      write(6,*) 
      write(6,*) 'out_se',nx,name
      do ie=nelv-1,1,-2
         write(6,*)
         do j=nx,1,-1
            if(nx.eq.4) then
               write(6,4) name,((se2crs(i,j,k+ie),i=1,nx),k=0,1)
            elseif(nx.eq.3) then
               write(6,3) name,((se2crs(i,j,k+ie),i=1,nx),k=0,1)
            else
               write(6,2) name,((se2crs(i,j,k+ie),i=1,nx),k=0,1)
            endif
         enddo
      enddo
c
    4 format(a4,5x,2(4i5,3x))
    3 format(a4,5x,2(3i5,3x))
    2 format(a4,5x,2(2i5,3x))
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine out_se0(se2crs,nx,nel,name)
c
      include 'SIZE'
      integer se2crs(nx,nx,1)
      character*4 name
c
      write(6,*) 
      write(6,*) 'out_se',nx,name,nel
      do ie=nel-3,1,-4
         write(6,*)
         do j=nx,1,-1
            if(nx.eq.4) then
               write(6,4) name,((se2crs(i,j,k+ie),i=1,nx),k=0,3)
            elseif(nx.eq.3) then
               write(6,3) name,((se2crs(i,j,k+ie),i=1,nx),k=0,3)
            else
               write(6,2) name,((se2crs(i,j,k+ie),i=1,nx),k=0,3)
            endif
         enddo
      enddo
c
    4 format(a4,5x,4(4i5,3x))
    3 format(a4,5x,4(3i5,3x))
    2 format(a4,5x,4(2i5,3x))
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine crs_solve_h1(uf,vf)
c
c     Given an input vector v, this generates the H1 coarse-grid solution
c
      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'PARALLEL'
      include 'CTIMER'
      include 'TSTEP'

      real uf(1),vf(1)
      common /scrpre/ uc(lcr*lelt)
      common /scrpr2/ vc(lcr*lelt)
      common /scrxxt/ cmlt(lcr,lelv),mask(lcr,lelv)

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      
      if (icalld1.eq.0) then ! timer info
         ncrsl=0
         tcrsl=0.0
         icalld1=1
      endif
      ncrsl  = ncrsl  + 1

      ntot = nelv*nx1*ny1*nz1
      call col3(uf,vf,vmult,ntot)

      call map_f_to_c_h1_bilin(vc,uf)   ! additive Schwarz

#ifdef TIMER
      etime1=dnekclock()
#endif
      call crs_solve(xxth(ifield),uc,vc)
#ifdef TIMER
      tcrsl=tcrsl+dnekclock()-etime1
#endif

      call map_c_to_f_h1_bilin(uf,uc)


      return
      end
c-----------------------------------------------------------------------
      subroutine set_h1_basis_bilin
c
      include 'SIZE'
      include 'DOMAIN'
      include 'WZ'
c
      do ix=1,nx1
         h1_basis(ix) = 0.5*(1.0-zgm1(ix,1))
         h1_basis(ix+nx1) = 0.5*(1.0+zgm1(ix,1))
      enddo
      call transpose(h1_basist,2,h1_basis,lx1)
c
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine map_c_to_f_h1_bilin(uf,uc)
c
c     H1 Iterpolation operator:  linear --> spectral GLL mesh
c
      include 'SIZE'
      include 'INPUT'
      include 'DOMAIN'
c
      parameter (lxyz = lx1*ly1*lz1)
      real uc(2,2,ldim-1,lelt),uf(lxyz,lelt)
      parameter (l2 = ldim-1)
      common /ctmp0/ w(lx1,lx1,2),v(lx1,2,l2,lelt)
c
      integer icalld
      save    icalld
      data    icalld/0/
      if (icalld.eq.0) then
         icalld=icalld+1
         call set_h1_basis_bilin
      endif
c
c
      n2 = 2
      if (if3d) then
c
         n31 = n2*n2*nelv
         n13 = nx1*nx1
c
         call mxm(h1_basis,nx1,uc,n2,v,n31)
         do ie=1,nelv
            do iz=1,n2
               call mxm(v(1,1,iz,ie),nx1,h1_basist,n2,w(1,1,iz),nx1)
            enddo
            call mxm(w,n13,h1_basist,n2,uf(1,ie),nx1)
         enddo
c
      else
c
         n31 = 2*nelv
         call mxm(h1_basis,nx1,uc,n2,v,n31)
         do ie=1,nelv
            call mxm(v(1,1,1,ie),nx1,h1_basist,n2,uf(1,ie),nx1)
         enddo
      endif
      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine map_f_to_c_h1_bilin(uc,uf)
c
c     TRANSPOSE of H1 Iterpolation operator:                    T
c                                 (linear --> spectral GLL mesh)
c
      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'
c
      parameter (lxyz = lx1*ly1*lz1)
      real uc(lcr,lelt),uf(lx1,ly1,lz1,lelt)
      common /ctmp0/ w(2,2,lx1),v(2,ly1,lz1,lelt)
c
      integer icalld
      save    icalld
      data    icalld/0/
      if (icalld.eq.0) then
         icalld=icalld+1
         call set_h1_basis_bilin
      endif
c
      n2 = 2
      if (if3d) then
         n31 = ny1*nz1*nelv
         n13 = n2*n2
         call mxm(h1_basist,n2,uf,nx1,v,n31)
         do ie=1,nelv
            do iz=1,nz1
               call mxm(v(1,1,iz,ie),n2,h1_basis,nx1,w(1,1,iz),n2)
            enddo
            call mxm(w,n13,h1_basis,nx1,uc(1,ie),n2)
         enddo
      else
         n31 = ny1*nelv
         call mxm(h1_basist,n2,uf,nx1,v,n31)
         do ie=1,nelv
               call mxm(v(1,1,1,ie),n2,h1_basis,nx1,uc(1,ie),n2)
         enddo
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_local_crs_galerkin(a,ncl,nxc,h1,h2,w1,w2)

c     This routine generates Nelv submatrices of order ncl using
c     Galerkin projection

      include 'SIZE'

      real    a(ncl,ncl,1),h1(1),h2(1)
      real    w1(nx1*ny1*nz1,nelv),w2(nx1*ny1*nz1,nelv)

      parameter (lcrd=lx1**ldim)
      common /ctmp1z/ b(lcrd,8)

      integer e

      do j=1,ncl
         call gen_crs_basis(b(1,j),j) ! bi- or tri-linear interpolant
      enddo

      isd  = 1
      imsh = 1

      nxyz = nx1*ny1*nz1
      do j = 1,ncl
         do e = 1,nelv
            call copy(w1(1,e),b(1,j),nxyz)
         enddo

         call axhelm (w2,w1,h1,h2,imsh,isd)        ! A^e * bj

         do e = 1,nelv
         do i = 1,ncl
            a(i,j,e) = vlsc2(b(1,i),w2(1,e),nxyz)  ! bi^T * A^e * bj
         enddo
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_crs_basis(b,j) ! bi- tri-linear

      include 'SIZE'
      real b(nx1,ny1,nz1)

      real z0(lx1),z1(lx1)
      real zr(lx1),zs(lx1),zt(lx1)

      integer p,q,r

      call zwgll(zr,zs,nx1)

      do i=1,nx1
         z0(i) = .5*(1-zr(i))  ! 1-->0
         z1(i) = .5*(1+zr(i))  ! 0-->1
      enddo

      call copy(zr,z0,nx1)
      call copy(zs,z0,nx1)
      call copy(zt,z0,nx1)

      if (mod(j,2).eq.0)                        call copy(zr,z1,nx1)
      if (j.eq.3.or.j.eq.4.or.j.eq.7.or.j.eq.8) call copy(zs,z1,nx1)
      if (j.gt.4)                               call copy(zt,z1,nx1)

      if (ndim.eq.3) then
         do r=1,nx1
         do q=1,nx1
         do p=1,nx1
            b(p,q,r) = zr(p)*zs(q)*zt(r)
         enddo
         enddo
         enddo
      else
         do q=1,nx1
         do p=1,nx1
            b(p,q,1) = zr(p)*zs(q)
         enddo
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_crs_basis2(b,j) ! bi- tri-quadratic

      include 'SIZE'
      real b(nx1,ny1,nz1)

      real z0(lx1),z1(lx1),z2(lx1)
      real zr(lx1),zs(lx1),zt(lx1)

      integer p,q,r

      call zwgll(zr,zs,nx1)

      do i=1,nx1
         z0(i) = .5*(zr(i)-1)*zr(i)  ! 1-->0   ! Lagrangian, ordered
         z1(i) = 4.*(1+zr(i))*(1-zr(i))        ! lexicographically
         z2(i) = .5*(zr(i)+1)*zr(i)  ! 0-->1   !
      enddo

      call copy(zr,z0,nx1)
      call copy(zs,z0,nx1)
      call copy(zt,z0,nx1)

      if (mod(j,2).eq.0)                        call copy(zr,z1,nx1)
      if (j.eq.3.or.j.eq.4.or.j.eq.7.or.j.eq.8) call copy(zs,z1,nx1)
      if (j.gt.4)                               call copy(zt,z1,nx1)

      if (ndim.eq.3) then
         do r=1,nx1
         do q=1,nx1
         do p=1,nx1
            b(p,q,r) = zr(p)*zs(q)*zt(r)
         enddo
         enddo
         enddo
      else
         do q=1,nx1
         do p=1,nx1
            b(p,q,1) = zr(p)*zs(q)
         enddo
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine get_vertex
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      common /ivrtx/ vertex ((2**ldim)*lelt)
      integer vertex

      integer icalld
      save    icalld
      data    icalld  /0/

      if (icalld.gt.0) return
      icalld = 1

      if (ifgtp) then
         call gen_gtp_vertex    (vertex, ncrnr)
      else
         call get_vert
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine assign_gllnid(gllnid,iunsort,nelgt,nelgv,np)
c
      integer gllnid(1),iunsort(1),nelgt,np 
      integer e,eg


      log2p = log2(np)
      np2   = 2**log2p
      if (np2.eq.np.and.nelgv.eq.nelgt) then   ! std power of 2 case

         npstar = ivlmax(gllnid,nelgt)+1
         nnpstr = npstar/np
         do eg=1,nelgt
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

         return

      elseif (np2.eq.np) then   ! std power of 2 case, conjugate heat xfer

c        Assign fluid elements
         npstar = max(np,ivlmax(gllnid,nelgv)+1)
         nnpstr = npstar/np
         do eg=1,nelgv
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

c        Assign solid elements
         nelgs  = nelgt-nelgv  ! number of solid elements
         npstar = max(np,ivlmax(gllnid(nelgv+1),nelgs)+1)
         nnpstr = npstar/np
         do eg=nelgv+1,nelgt
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

         return

      elseif (nelgv.ne.nelgt) then
         call exitti
     $       ('Conjugate heat transfer requires P=power of 2.$',np)
      endif


c  Below is the code for P a non-power of two:

c  Split the sorted gllnid array (read from .map file) 
c  into np contiguous partitions. 

c  To load balance the partitions in case of mod(nelgt,np)>0 
c  add 1 contiguous entry out of the sorted list to NODE_i 
c  where i = np-mod(nelgt,np) ... np


      nel   = nelgt/np       ! number of elements per processor
      nmod  = mod(nelgt,np)  ! bounded between 1 ... np-1
      npp   = np - nmod      ! how many paritions of size nel 
 
      ! sort gllnid  
      call isort(gllnid,iunsort,nelgt)

      ! setup partitions of size nel 
      k   = 0
      do ip = 0,npp-1
         do e = 1,nel  
            k = k + 1 
            gllnid(k) = ip
         enddo
      enddo
      ! setup partitions of size nel+1
      if(nmod.gt.0) then 
        do ip = npp,np-1
           do e = 1,nel+1  
              k = k + 1 
              gllnid(k) = ip
           enddo
        enddo 
      endif

      ! unddo sorting to restore initial ordering by
      ! global element number
      call iswapt_ip(gllnid,iunsort,nelgt)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_vert
      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'

      common /ivrtx/ vertex ((2**ldim),lelt)
      integer vertex

      integer e,eg

      integer icalld
      save    icalld
      data    icalld  /0/
      if (icalld.gt.0) return
      icalld = 1

      ncrnr = 2**ndim

      if (ifmoab) then
#ifdef MOAB
         call nekMOAB_loadConn (vertex, nelgt, ncrnr)
#endif
      else
         call get_vert_map(vertex, ncrnr, nelgt, '.map', ifgfdm)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine get_vert_map(vertex, nlv, nel, suffix, ifgfdm)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      logical ifgfdm
      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
      integer vertex(nlv,1)
      character*4 suffix

      parameter(mdw=2+2**ldim)
      parameter(ndw=7*lx1*ly1*lz1*lelv/mdw)
      common /scrns/ wk(mdw,ndw)   ! room for long ints, if desired
      integer wk,e,eg,eg0,eg1

      character*132 mapfle
      character*1   mapfle1(132)
      equivalence  (mapfle,mapfle1)

      iok = 0
      if (nid.eq.0) then
         lfname = ltrunc(reafle,132) - 4
         call blank (mapfle,132)
         call chcopy(mapfle,reafle,lfname)
         call chcopy(mapfle1(lfname+1),suffix,4)
         if(nio.eq.0) write(6,'(A,A)') ' Reading ', mapfle
         open(unit=80,file=mapfle,status='old',err=99)
         read(80,*,err=99) neli,nnzi
         iok = 1
      endif
   99 continue
      iok = iglmax(iok,1)
      if (iok.eq.0) goto 999     ! Mapfile not found

      if (nid.eq.0) then
         neli = iglmax(neli,1)   ! communicate to all procs
      else
         neli = 0
         neli = iglmax(neli,1)   ! communicate neli to all procs
      endif

      npass = 1 + (neli/ndw)
      if (npass.gt.np) then
         if (nid.eq.0) write(6,*) npass,np,neli,ndw,'Error get_vert_map'
         call exitt
      endif 

      len = 4*mdw*ndw
      if (nid.gt.0.and.nid.lt.npass) msg_id=irecv(nid,wk,len)
      call nekgsync

      if (nid.eq.0) then
         eg0 = 0
         do ipass=1,npass
            eg1 = min(eg0+ndw,neli)
            m   = 0
            do eg=eg0+1,eg1
               m = m+1
               read(80,*,end=998) (wk(k,m),k=2,mdw)
               if(.not.ifgfdm)  gllnid(eg) = wk(2,m)  !proc map,  must still be divided
               wk(1,m)    = eg
            enddo
            if (ipass.lt.npass) call csend(ipass,wk,len,ipass,0) !send to ipass
            eg0 = eg1
         enddo
         close(80)
         ntuple = m
      elseif (nid.lt.npass) then
         call msgwait(msg_id)
         ntuple = ndw
      else
         ntuple = 0
      endif

c     Distribute and assign partitions
      if (.not.ifgfdm) then             ! gllnid is already assigned for gfdm
        lng = isize*neli
        call bcast(gllnid,lng)
        call assign_gllnid(gllnid,gllel,nelgt,nelgv,np) ! gllel is used as scratch

c       if(nid.eq.0) then
c         write(99,*) (gllnid(i),i=1,nelgt)
c       endif
c       call exitt
      endif

      nelt=0 !     Count number of elements on this processor
      nelv=0
      do eg=1,neli
         if (gllnid(eg).eq.nid) then
            if (eg.le.nelgv) nelv=nelv+1
            if (eg.le.nelgt) nelt=nelt+1
         endif
      enddo
      if (np.le.64) write(6,*) nid,nelv,nelt,nelgv,nelgt,' NELV'

c     NOW: crystal route vertex by processor id

      do i=1,ntuple
         eg=wk(1,i)
         wk(2,i)=gllnid(eg)        ! processor id for element eg
      enddo

      key = 2  ! processor id is in wk(2,:)
      call crystal_ituple_transfer(cr_h,wk,mdw,ntuple,ndw,key)

      if (.not.ifgfdm) then            ! no sorting for gfdm?
         key = 1  ! Sort tuple list by eg := wk(1,:)
         nkey = 1
         call crystal_ituple_sort(cr_h,wk,mdw,nelt,key,nkey)
      endif

      iflag = 0
      if (ntuple.ne.nelt) then
         write(6,*) nid,ntuple,nelv,nelt,nelgt,' NELT FAIL'
         write(6,*) 'Check that .map file and .rea file agree'
         iflag=1
      else
         nv = 2**ndim
         do e=1,nelt
            call icopy(vertex(1,e),wk(3,e),nv)
         enddo
      endif

      iflag = iglmax(iflag,1)
      if (iflag.gt.0) then
         do mid=0,np-1
            call nekgsync
            if (mid.eq.nid)
     $      write(6,*) nid,ntuple,nelv,nelt,nelgt,' NELT FB'
            call nekgsync
         enddo
         call nekgsync
         call exitt
      endif

      return

  999 continue
      if (nid.eq.0) write(6,*) 'ABORT: Could not find map file ',mapfle
      call exitt

  998 continue
      if (nid.eq.0) write(6,*)ipass,npass,eg0,eg1,mdw,m,eg,'get v fail'
      call exitt0  ! Emergency exit

      return
      end
c-----------------------------------------------------------------------
      subroutine irank_vecn(ind,nn,a,m,n,key,nkey,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is permuted
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),aa(m)
      logical iftuple_ianeb,a_ne_b

      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)

c     Find unique a's
      call icopy(aa,a,m)
      nn     = 1
      ind(1) = nn
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
         if (a_ne_b) then
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         ind(i) = nn ! set ind() to rank
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine gbtuple_rank(tuple,m,n,nmax,cr_h,nid,np,ind)
c
c     Return a unique rank for each matched tuple set. Global.  Balanced.
c
c     tuple is destroyed.
c
c     By "balanced" we mean that none of the tuple entries is likely to
c     be much more uniquely populated than any other, so that any of
c     the tuples can serve as an initial (parallel) sort key
c
c     First two slots in tuple(:,i) assumed empty
c
      integer ind(nmax),tuple(m,nmax),cr_h

      parameter (mmax=40)
      integer key(mmax),wtuple(mmax)

      if (m.gt.mmax) then
         write(6,*) nid,m,mmax,' gbtuple_rank fail'
         call exitt
      endif

      do i=1,n
         tuple(1,i) = mod(tuple(3,i),np) ! destination processor
         tuple(2,i) = i                  ! return location
      enddo

      ni= n
      ky=1  ! Assumes crystal_new already called
      call crystal_ituple_transfer(cr_h, tuple,m,ni,nmax, ky)

      nimx = iglmax(ni,1)
      if (ni.gt.nmax)   write(6,*) ni,nmax,n,'cr_xfer problem, A'
      if (nimx.gt.nmax) call exitt

      nkey = m-2
      do k=1,nkey
         key(k) = k+2
      enddo

      call irank_vecn(ind,nu,tuple,m,ni,key,nkey,wtuple)! tuple re-ordered,
                                                        ! but contents same

      nu_tot   = igl_running_sum(nu) ! running sum over P processors
      nu_prior = nu_tot - nu

      do i=1,ni
         tuple(3,i) = ind(i) + nu_prior  ! global ranking
      enddo

      call crystal_ituple_transfer(cr_h, tuple,m,ni,nmax, ky)

      nk = 1  ! restore to original order, local rank: 2; global: 3
      ky = 2
      call ituple_sort(tuple,m,n,ky,nk,ind,wtuple)


      return
      end
c-----------------------------------------------------------------------
      subroutine setvert3d(glo_num,ngv,nx,nel,vertex,ifcenter)
c
c     setup unique ids for dssum  
c     note:
c     total number of unique vertices, edges and faces has to be smaller 
c     than 2**31 (integer-4 limit).
c     if nelgt < 2**31/12 we're ok for sure (independent of N)! 
c
      include 'SIZE'
      include 'CTIMER'
      include 'PARALLEL'
      include 'TOPOL'
      include 'GEOM'

      integer*8 glo_num(1),ngv
      integer vertex(0:1,0:1,0:1,1),nx
      logical ifcenter

      integer  edge(0:1,0:1,0:1,3,lelt),enum(12,lelt),fnum(6,lelt)
      common  /scrmg/ edge,enum,fnum

      parameter (nsafe=8)  ! OFTEN, nsafe=2 suffices
      integer etuple(4,12*lelt*nsafe),ftuple(5,6,lelt*nsafe)
      integer ind(12*lelt*nsafe)
      common  /scrns/ ind,etuple
      equivalence  (etuple,ftuple)

      integer gvf(4),facet(4),aa(3),key(3),e
      logical ifij
      
      integer*8 igv,ig0
      integer*8 ngvv,ngve,ngvs,ngvi,ngvm
      integer*8 n_on_edge,n_on_face,n_in_interior
      integer*8 i8glmax
c
      ny   = nx
      nz   = nx
      nxyz = nx*ny*nz
c
      key(1)=1
      key(2)=2
      key(3)=3
c
c     Assign hypercube ordering of vertices
c     -------------------------------------
c
c     Count number of unique vertices
      nlv  = 2**ndim
      ngvv = iglmax(vertex,nlv*nel)
c
      do e=1,nel
         do k=0,1
         do j=0,1
         do i=0,1
c           Local to global node number (vertex)
            il  = 1 + (nx-1)*i + nx*(nx-1)*j + nx*nx*(nx-1)*k
            ile = il + nx*ny*nz*(e-1)
            glo_num(ile)   = vertex(i,j,k,e)
         enddo
         enddo
         enddo
      enddo
      ngv  = ngvv
c
      if (nx.eq.2) return
c
c     Assign global vertex numbers to SEM nodes on each edge
c     ------------------------------------------------------
c
c     Assign edge labels by bounding vertices.  
      do e=1,nel
         do k=0,1
         do j=0,1
         do i=0,1
            edge(i,j,k,1,e) = vertex(i,j,k,e)  ! r-edge
            edge(j,i,k,2,e) = vertex(i,j,k,e)  ! s-edge
            edge(k,i,j,3,e) = vertex(i,j,k,e)  ! t-edge
         enddo
         enddo
         enddo
      enddo
c
c     Sort edges by bounding vertices.
      do i=0,12*nel-1
         if (edge(0,i,0,1,1).gt.edge(1,i,0,1,1)) then
            kswap = edge(0,i,0,1,1)
            edge(0,i,0,1,1) = edge(1,i,0,1,1)
            edge(1,i,0,1,1) = kswap
         endif
         etuple(3,i+1) = edge(0,i,0,1,1)
         etuple(4,i+1) = edge(1,i,0,1,1)
      enddo
c
c     Assign a number (rank) to each unique edge
      m    = 4
      n    = 12*nel
      nmax = 12*lelt*nsafe  ! nsafe for crystal router factor of safety
      call gbtuple_rank(etuple,m,n,nmax,cr_h,nid,np,ind)
      do i=1,12*nel
         enum(i,1) = etuple(3,i)
      enddo
      n_unique_edges = iglmax(enum,12*nel)
c
      n_on_edge = nx-2
      ngve      = n_unique_edges*n_on_edge
      do e=1,nel
         iedg_loc = 0
c
c        Edges 1-4
         do k=0,1
         do j=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,e)-1)
            i0  = nx*(nx-1)*j + nx*nx*(nx-1)*k
            i0e = i0 + nxyz*(e-1)
            if (glo_num(i0e+1).lt.glo_num(i0e+nx)) then
               do i=2,nx-1                                   ! std forward case
                  glo_num(i0e+i) = igv + i-1
               enddo
            else
               do i=2,nx-1                                   ! backward case
                  glo_num(i0e+i) = igv + 1 + n_on_edge-(i-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
c
c        Edges 5-8
         do k=0,1
         do i=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,e)-1)
            i0  = 1+(nx-1)*i + nx*nx*(nx-1)*k
            i0e = i0 + nxyz*(e-1)
            if (glo_num(i0e).lt.glo_num(i0e+nx*(nx-1))) then
               do j=2,nx-1                                   ! std forward case
                  glo_num(i0e+(j-1)*nx) = igv + j-1
               enddo
            else
               do j=2,nx-1                                   ! backward case
                  glo_num(i0e+(j-1)*nx) = igv + 1 + n_on_edge-(j-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
c
c        Edges 9-12
         do j=0,1
         do i=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,e)-1)
            i0  = 1 + (nx-1)*i + nx*(nx-1)*j
            i0e = i0 + nxyz*(e-1)
            if (glo_num(i0e).lt.glo_num(i0e+nx*nx*(nx-1))) then
               do k=2,nx-1                                   ! std forward case
                  glo_num(i0e+(k-1)*nx*nx) = igv + k-1
               enddo
            else
               do k=2,nx-1                                   ! backward case
                  glo_num(i0e+(k-1)*nx*nx) = igv + 1 + n_on_edge-(k-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
      enddo
      ngv   = ngv + ngve
c
c     Asign global node numbers on the interior of each face
c     ------------------------------------------------------ 
c
c     Assign faces by 3-tuples 
c
c     (The following variables all take the symmetric 
c     notation of IFACE as arguments:)
c
c     ICFACE(i,IFACE) -   Gives the 4 vertices which reside on face IFACE
c                         as depicted below, e.g. ICFACE(i,2)=2,4,6,8.
c
c                        3+-----+4    ^ Y
c                        /  2  /|     |
c     Edge 1 extends    /     / |     |
c       from vertex   7+-----+8 +2    +----> X
c       1 to 2.        |  4  | /     /
c                      |     |/     /
c                     5+-----+6    Z
c                         3
c
      nfaces=ndim*2
      ncrnr =2**(ndim-1)
      do e=1,nel
         do ifac=1,nfaces
            do icrn=1,ncrnr
               i                  = icface(icrn,ifac)-1
               facet(icrn)        = vertex(i,0,0,e)
            enddo
            call isort(facet,ind,ncrnr)
            call icopy(ftuple(3,ifac,e),facet,ncrnr-1)
         enddo
      enddo

c     Assign a number (rank) to each unique face
      m    = 5
      n    = 6*nel
      nmax = 6*lelt*nsafe  ! nsafe for crystal router factor of safety
      call gbtuple_rank(ftuple,m,n,nmax,cr_h,nid,np,ind)
      do i=1,6*nel
         fnum(i,1) = ftuple(3,i,1)
      enddo
      n_unique_faces = iglmax(fnum,6*nel)
c
      call dsset (nx,ny,nz)
      do e=1,nel
       do iface=1,nfaces
         i0 = skpdat(1,iface)
         i1 = skpdat(2,iface)
         is = skpdat(3,iface)
         j0 = skpdat(4,iface)
         j1 = skpdat(5,iface)
         js = skpdat(6,iface)
c
c        On each face, count from minimum global vertex number,
c        towards smallest adjacent vertex number.  e.g., suppose
c        the face is defined by the following global vertex numbers:
c
c
c                    11+--------+81
c                      |c      d|
c                      |        |
c                      |        |
c                      |a      b|
c                    15+--------+62
c                          
c        We would count from c-->a, then towards d.
c
         gvf(1) = glo_num(i0+nx*(j0-1)+nxyz*(e-1))
         gvf(2) = glo_num(i1+nx*(j0-1)+nxyz*(e-1))
         gvf(3) = glo_num(i0+nx*(j1-1)+nxyz*(e-1))
         gvf(4) = glo_num(i1+nx*(j1-1)+nxyz*(e-1))
c
         call irank(gvf,ind,4)
c
c        ind(1) tells which element of gvf() is smallest.
c
         ifij = .false.
         if (ind(1).eq.1) then
            idir =  1
            jdir =  1
            if (gvf(2).lt.gvf(3)) ifij = .true.
         elseif (ind(1).eq.2) then
            idir = -1
            jdir =  1
            if (gvf(1).lt.gvf(4)) ifij = .true.
         elseif (ind(1).eq.3) then
            idir =  1
            jdir = -1
            if (gvf(4).lt.gvf(1)) ifij = .true.
         elseif (ind(1).eq.4) then
            idir = -1
            jdir = -1
            if (gvf(3).lt.gvf(2)) ifij = .true.
         endif
c
         if (idir.lt.0) then
            it=i0
            i0=i1
            i1=it
            is=-is
         endif
c
         if (jdir.lt.0) then
            jt=j0
            j0=j1
            j1=jt
            js=-js
         endif
c
         nxx = nx*nx
         n_on_face = (nx-2)*(ny-2)
         ngvs  = n_unique_faces*n_on_face
         ig0 = ngv + n_on_face*(fnum(iface,e)-1)
         if (ifij) then
            k=0
            l=0
            do j=j0,j1,js
            do i=i0,i1,is
               k=k+1
c              this is a serious kludge to stay on the face interior
               if (k.gt.nx.and.k.lt.nxx-nx .and.
     $            mod(k,nx).ne.1.and.mod(k,nx).ne.0) then
c                 interior
                  l = l+1
                  glo_num(i+nx*(j-1)+nxyz*(e-1)) = l + ig0
               endif
            enddo
            enddo
         else
            k=0
            l=0
            do i=i0,i1,is
            do j=j0,j1,js
               k=k+1
c              this is a serious kludge to stay on the face interior
               if (k.gt.nx.and.k.lt.nxx-nx .and.
     $            mod(k,nx).ne.1.and.mod(k,nx).ne.0) then
c                 interior
                  l = l+1
                  glo_num(i+nx*(j-1)+nxyz*(e-1)) = l + ig0
               endif
            enddo
            enddo
         endif
       enddo
      enddo
      ngv   = ngv + ngvs
c
c     Finally,  number interiors (only ifcenter=.true.)
c     -------------------------------------------------
c
      n_in_interior = (nx-2)*(ny-2)*(nz-2)
      ngvi = n_in_interior*nelgt
      if (ifcenter) then
         do e=1,nel
            ig0 = ngv + n_in_interior*(lglel(e)-1)
            l = 0
            do k=2,nz-1
            do j=2,ny-1
            do i=2,nx-1
               l = l+1
               glo_num(i+nx*(j-1)+nx*ny*(k-1)+nxyz*(e-1)) = ig0+l
            enddo
            enddo
            enddo
         enddo
         ngv = ngv + ngvi
      else
         do e=1,nel
            l = 0
            do k=2,nz-1
            do j=2,ny-1
            do i=2,nx-1
               l = l+1
               glo_num(i+nx*(j-1)+nx*ny*(k-1)+nxyz*(e-1)) = 0
            enddo
            enddo
            enddo
         enddo
      endif
c
c     Quick check on maximum #dofs:
      m    = nxyz*nelt
      ngvm = i8glmax(glo_num,m)
      ngvv = ngvv + ngve + ngvs  ! number of unique ids w/o interior 
      ngvi = ngvi + ngvv         ! total number of unique ids 
      if (nio.eq.0) write(6,1) nx,ngvv,ngvi,ngv,ngvm
    1 format('   setvert3d:',i4,4i12)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine setvert2d(glo_num,ngv,nx,nel,vertex,ifcenter)
c
c     setup unique ids for dssum  
c
      include 'SIZE'
      include 'CTIMER'
      include 'PARALLEL'
      include 'TOPOL'
      include 'GEOM'

      integer*8 glo_num(1),ngv
      integer vertex(0:1,0:1,1),nx
      logical ifcenter

      integer  edge(0:1,0:1,2,lelt),enum(4,lelt)
      common  /scrmg/ edge,enum

      parameter (nsafe=8)  ! OFTEN, nsafe=2 suffices
      integer etuple(4,4*lelt*nsafe),ind(4*lelt*nsafe)
      common  /scrns/ ind,etuple

      integer gvf(4),aa(3),key(3),e,eg
      logical ifij

      integer*8 igv,ig0
      integer*8 ngvv,ngve,ngvs,ngvi,ngvm
      integer*8 n_on_edge,n_on_face,n_in_interior
      integer*8 i8glmax
c
c
c     memory check...
c
      ny   = nx
      nz   = 1
      nxyz = nx*ny*nz
c
      key(1)=1
      key(2)=2
      key(3)=3
c
c     Count number of unique vertices
      nlv  = 2**ndim
      ngvv = iglmax(vertex,nlv*nel)
      ngv  = ngvv
c
c     Assign hypercube ordering of vertices.
      do e=1,nel
         do j=0,1
         do i=0,1
c           Local to global node number (vertex)
            il  = 1 + (nx-1)*i + nx*(nx-1)*j
            ile = il + nx*ny*(e-1)
            glo_num(ile)   = vertex(i,j,e)
         enddo
         enddo
      enddo
      if (nx.eq.2) return
c
c     Assign edge labels by bounding vertices.  
      do e=1,nel
         do j=0,1
         do i=0,1
            edge(i,j,1,e) = vertex(i,j,e)  ! r-edge
            edge(j,i,2,e) = vertex(i,j,e)  ! s-edge
         enddo
         enddo
      enddo

c     Sort edges by bounding vertices.
      do i=0,4*nel-1
         if (edge(0,i,1,1).gt.edge(1,i,1,1)) then
            kswap = edge(0,i,1,1)
            edge(0,i,1,1) = edge(1,i,1,1)
            edge(1,i,1,1) = kswap
         endif
         etuple(3,i+1) = edge(0,i,1,1)
         etuple(4,i+1) = edge(1,i,1,1)
      enddo

c     Assign a number (rank) to each unique edge
      m    = 4
      n    = 4*nel
      nmax = 4*lelt*nsafe  ! nsafe for crystal router factor of safety

      call gbtuple_rank(etuple,m,n,nmax,cr_h,nid,np,ind)
      do i=1,4*nel
         enum(i,1) = etuple(3,i)
      enddo
      n_unique_edges = iglmax(enum,4*nel)

c= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
c     Assign global vertex numbers to SEM nodes on each edge
      n_on_edge = nx-2
      do e=1,nel

         iedg_loc = 0

c        Edges 1-2
         do j=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,e)-1)
            i0  = nx*(nx-1)*j
            i0e = i0 + nxyz*(e-1)
            if (glo_num(i0e+1).lt.glo_num(i0e+nx)) then
               do i=2,nx-1                                   ! std forward case
                  glo_num(i0e+i) = igv + i-1
               enddo
            else
               do i=2,nx-1                                   ! backward case
                  glo_num(i0e+i) = igv + 1 + n_on_edge-(i-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
c
c        Edges 3-4
         do i=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,e)-1)
            i0  = 1+(nx-1)*i
            i0e = i0 + nxyz*(e-1)
            if (glo_num(i0e).lt.glo_num(i0e+nx*(nx-1))) then
               do j=2,nx-1                                   ! std forward case
                  glo_num(i0e+(j-1)*nx) = igv + j-1
               enddo
            else
               do j=2,nx-1                                   ! backward case
                  glo_num(i0e+(j-1)*nx) = igv + 1 + n_on_edge-(j-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
      enddo
 
      ngve = n_unique_edges*n_on_edge
      ngv  = ngv + ngve    
c
c     Finally,  number interiors  
c
      n_in_interior = (nx-2)*(ny-2)
      ngvi          = n_in_interior*nelgt
      if (ifcenter) then
         do e=1,nel
            ig0 = ngv + n_in_interior*(lglel(e)-1)
            l = 0
            do j=2,ny-1
            do i=2,nx-1
               l = l+1
               glo_num(i+nx*(j-1)+nxyz*(e-1)) = ig0+l
            enddo
            enddo
         enddo
         ngv = ngv + ngvi
      else
         do e=1,nel
            l = 0
            do j=2,ny-1
            do i=2,nx-1
               l = l+1
               glo_num(i+nx*(j-1)+nxyz*(e-1)) = 0
            enddo
            enddo
         enddo
      endif

c
c     Quick check on maximum #dofs:
      m    = nxyz*nelt
      ngvm = i8glmax(glo_num,m)
      ngvv = ngvv + ngve         ! number of unique ids w/o interior 
      ngvi = ngvi + ngvv         ! total number of unique ids 
      if (nio.eq.0) write(6,1) nx,ngvv,ngvi,ngv,ngvm
    1 format('   setvert2d:',i4,4i12)
c
      return
      end
c-----------------------------------------------------------------------
c
      subroutine map_to_crs(a,na,b,nb,if3d,w,ldw)
c
c     Input:   b
c     Output:  a
c
      real a(1),b(1),w(1)
      logical if3d
c
      parameter(lx=40)
      real za(lx),zb(lx)
c
      real iba(lx*lx),ibat(lx*lx)
      save iba,ibat
c
      integer nao,nbo
      save    nao,nbo
      data    nao,nbo  / -9, -9/
c
      if (na.gt.lx.or.nb.gt.lx) then
         write(6,*)'ERROR: increase lx in map_to_crs to max:',na,nb
         call exitt
      endif
c
      if (na.ne.nao  .or.   nb.ne.nbo) then
         nao = na
         nbo = nb
         call zwgll(za,w,na)
         call zwgll(zb,w,nb)
         call igllm(iba,ibat,zb,za,nb,na,nb,na)
      endif
c
      call specmpn(a,na,b,nb,iba,ibat,if3d,w,ldw)

      return
      end
c-----------------------------------------------------------------------
      subroutine check_p_bc(glo_num,nx,ny,nz,nel)

      include 'SIZE'
      include 'TOTAL'

      integer*8 glo_num(nx,ny,nz,nel)
      integer*8 gmn

      integer e,f,fo,ef,efo
      integer eface0(6)
      save    eface0
      data    eface0 / 4,2,1,3,5,6 /

      ifld = 2
      if (ifflow) ifld = 1

      nface=2*ndim
      do e=1,nelt
      do f=1,nface,2
         fo  = f+1
         ef  = eface0(f)
         efo = eface0(fo)
         if (cbc(ef,e,ifld).eq.'p  '.and.cbc(efo,e,ifld).eq.'p  ') then
            if (f.eq.1) then  ! r=-/+1
               do k=1,nz
               do j=1,ny
                  gmn = min(glo_num(1,j,k,e),glo_num(nx,j,k,e))
                  glo_num(1 ,j,k,e) = gmn
                  glo_num(nx,j,k,e) = gmn
               enddo
               enddo
            elseif (f.eq.3) then  ! s=-/+1
               do k=1,nz
               do i=1,nx
                  gmn = min(glo_num(i,1,k,e),glo_num(i,ny,k,e))
                  glo_num(i,1 ,k,e) = gmn
                  glo_num(i,ny,k,e) = gmn
               enddo
               enddo
            else
               do j=1,ny
               do i=1,nx
                  gmn = min(glo_num(i,j,1,e),glo_num(i,j,nz,e))
                  glo_num(i,j,1 ,e) = gmn
                  glo_num(i,j,nz,e) = gmn
               enddo
               enddo
            endif
         endif
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
