c-----------------------------------------------------------------------
c
c Some relevant parameters
c
c param(41):
c     0 - use additive SEMG
c     1 - use hybrid SEMG (not yet working... but coming soon!)
c
c param(42):   navier0.f, fasts.f
c     0 - use GMRES for iterative solver, use non-symmetric weighting
c     1 - use PCG for iterative solver, do not use weighting
c
c param(43):   uzawa_gmres.f, navier6.f
c     0 - use additive multilevel scheme (requires param(42).eq.0)
c     1 - use original 2 level scheme
c
c param(44):   fast3d.f, navier6.f
c     0 - base top-level additive Schwarz on restrictions of E
c     1 - base top-level additive Schwarz on restrictions of A
c
c----------------------------------------------------------------------
      subroutine hsmg_setup()
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'HSMG'
      include 'SEMHAT'
      include 'TSTEP'

      integer nf,nc,nr
      integer nx,ny,nz

      mg_fld = 1
      if (ifield.gt.1) mg_fld = 2
      if (ifield.eq.1) call hsmg_index_0 ! initialize index sets

      call hsmg_setup_mg_nx  ! set nx values for each level of multigrid
      call hsmg_setup_semhat ! set spectral element hat matrices
      call hsmg_setup_intp
      call hsmg_setup_dssum  ! set direct stiffness summation handles
      call hsmg_setup_wtmask ! set restriction weight matrices and bc masks
      call hsmg_setup_fdm    ! set up fast diagonalization method
      call hsmg_setup_schwarz_wt(.false.)
      call hsmg_setup_solve  ! set up the solver
c     call hsmg_setup_dbg

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_semhat
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      include 'SEMHAT'
      integer n,l
c     generate the SEM hat matrices for each level
c     top level
      n = mg_nx(mg_lmax)
      call semhat(ah,bh,ch,dh,zh,dph,jph,bgl,zglhat,dgl,jgl,n,wh)
      call copy(mg_zh(1,mg_lmax),zglhat,n-1) !top level based on gl points
      mg_nh(mg_lmax)=n-1
      mg_nhz(mg_lmax)=n-1
      if(.not.if3d) mg_nhz(mg_lmax)=1
c     lower levels
      do l=1,mg_lmax-1
         n = mg_nx(l)
         if(n.gt.1) then
            call semhat(ah,bh,ch,dh,zh,dph,jph,bgl,zglhat,dgl,jgl,n,wh)
            call copy(mg_ah(1,l),ah,(n+1)*(n+1))
            call copy(mg_bh(1,l),bh,n+1)
            call copy(mg_dh(1,l),dh,(n+1)*(n+1))
            call transpose(mg_dht(1,l),n+1,dh,n+1)
            call copy(mg_zh(1,l),zh,n+1)
         else
            mg_zh(1,l) = -1.
            mg_zh(2,l) =  1.
         endif
         mg_nh(l)=n+1
         mg_nhz(l)=mg_nz(l)+1
      enddo
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_intp
      include 'SIZE'
      include 'HSMG'
      include 'SEMHAT'
      integer l,nf,nc

      do l=1,mg_lmax-1

         nf=mg_nh(l+1)
         nc=mg_nh(l)

!        Standard multigrid coarse-to-fine interpolation
         call hsmg_setup_intpm(
     $           mg_jh(1,l),mg_zh(1,l+1),mg_zh(1,l),nf,nc)
         call transpose(mg_jht(1,l),nc,mg_jh(1,l),nf)

!        Fine-to-coarse interpolation for variable-coefficient operators
         call hsmg_setup_intpm(
     $           mg_jhfc(1,l),mg_zh(1,l),mg_zh(1,l+1),nc,nf)
         call transpose(mg_jhfct(1,l),nf,mg_jhfc(1,l),nc)
c        call outmat(mg_jhfc(1,l),nc,nf,'MG_JHFC',l)

      enddo
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_intpm(jh,zf,zc,nf,nc)
      integer nf,nc
      real jh(nf,nc),zf(1),zc(1)
      include 'SIZE'
      real w(2*lx1+2)
      do i=1,nf
         call fd_weights_full(zf(i),zc,nc-1,1,w)
         do j=1,nc
            jh(i,j)=w(j)
         enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_dssum
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'HSMG'
      parameter (lxyz=(lx1+2)*(ly1+2)*(lz1+2))
      common /c_is1/ glo_num(lxyz*lelv)
      common /ivrtx/ vertex ((2**ldim)*lelt)

      integer*8 glo_num
      integer vertex
      integer nx,ny,nz
      integer l
      
c     set up direct stiffness summation for each level
      ncrnr = 2**ldim
      call get_vert

c++   write(6,*) mg_fld,' mgfld in hsmg_setup_dssum'

      do l=1,mg_lmax-1
         nx=mg_nh(l)
         ny=mg_nh(l)
         nz=mg_nhz(l)
         call setupds(mg_gsh_handle(l,mg_fld),nx,ny,nz
     $                ,nelv,nelgv,vertex,glo_num)
         nx=nx+2
         ny=ny+2
         nz=nz+2
         if(.not.if3d) nz=1
         call setupds(mg_gsh_schwarz_handle(l,mg_fld),nx,ny,nz
     $                ,nelv,nelgv,vertex,glo_num)
      enddo
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_wtmask
      include 'SIZE'
      include 'HSMG'
      integer i,l
      i = mg_mask_index(mg_lmax,mg_fld-1)
      do l=1,mg_lmax
         mg_rstr_wt_index(l,mg_fld)=i
         mg_mask_index   (l,mg_fld)=i
         i=i+mg_nh(l)*mg_nhz(l)*2*ldim*nelv
         if(i .gt. lmgs*lmg_rwt*2*ldim*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'parameter lmg_rwt too small',i,itmp,lmg_rwt
            call exitt
         endif
         call hsmg_setup_rstr_wt(
     $           mg_rstr_wt(mg_rstr_wt_index(l,mg_fld))
     $          ,mg_nh(l),mg_nh(l),mg_nhz(l),l,mg_work)
         call hsmg_setup_mask(
     $           mg_mask(mg_mask_index(l,mg_fld))
     $          ,mg_nh(l),mg_nh(l),mg_nhz(l),l,mg_work)
      enddo
      mg_mask_index(l,mg_fld)=i
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_wtmask
      include 'SIZE'
      include 'HSMG'
      integer i,l
      i = mg_mask_index(mg_lmax,mg_fld-1)
      do l=1,mg_lmax-1
         mg_rstr_wt_index(l,mg_fld)=i
         mg_mask_index   (l,mg_fld)=i
         i=i+mg_nh(l)*mg_nhz(l)*2*ldim*nelv
         if(i .gt. lmgs*lmg_rwt*2*ldim*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'parameter lmg_rwt too small',i,itmp,lmg_rwt
            call exitt
         endif
         call hsmg_setup_rstr_wt(
     $           mg_rstr_wt(mg_rstr_wt_index(l,mg_fld))
     $          ,mg_nh(l),mg_nh(l),mg_nhz(l),l,mg_work)
         call hsmg_setup_mask(
     $           mg_mask(mg_mask_index(l,mg_fld))
     $          ,mg_nh(l),mg_nh(l),mg_nhz(l),l,mg_work)
      enddo
      mg_mask_index(l,mg_fld)=i
      end
c----------------------------------------------------------------------
      subroutine hsmg_intp(uf,uc,l) ! l is coarse level
      real uf(1),uc(1)
      integer l
      include 'SIZE'
      include 'HSMG'
      call hsmg_tnsr(uf,mg_nh(l+1),uc,mg_nh(l),mg_jh(1,l),mg_jht(1,l))
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_rstr(uc,uf,l) ! l is coarse level
      real uf(1),uc(1)
      integer l
      include 'SIZE'
      include 'HSMG'
      if(l.ne.mg_lmax-1)
     $   call hsmg_do_wt(uf,mg_rstr_wt(mg_rstr_wt_index(l+1,mg_fld))
     $                     ,mg_nh(l+1),mg_nh(l+1),mg_nhz(l+1))
      call hsmg_tnsr(uc,mg_nh(l),uf,mg_nh(l+1),mg_jht(1,l),mg_jh(1,l))
      call hsmg_dssum(uc,l)
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_rstr_no_dssum(uc,uf,l) ! l is coarse level
      real uf(1),uc(1)
      integer l
      include 'SIZE'
      include 'HSMG'
      if(l.ne.mg_lmax-1)
     $   call hsmg_do_wt(uf,mg_rstr_wt(mg_rstr_wt_index(l+1,mg_fld))
     $                     ,mg_nh(l+1),mg_nh(l+1),mg_nhz(l+1))
      call hsmg_tnsr(uc,mg_nh(l),uf,mg_nh(l+1),mg_jht(1,l),mg_jh(1,l))
      return
      end
c----------------------------------------------------------------------
c     computes
c     v = [A (x) A] u      or
c     v = [A (x) A (x) A] u 
      subroutine hsmg_tnsr(v,nv,u,nu,A,At)
      integer nv,nu
      real v(1),u(1),A(1),At(1)
      include 'SIZE'
      include 'INPUT'
      if (.not. if3d) then
         call hsmg_tnsr2d(v,nv,u,nu,A,At)
      else
         call hsmg_tnsr3d(v,nv,u,nu,A,At,At)
      endif
      return
      end
c----------------------------------------------------------------------
c     computes
c              T
c     v = A u B
      subroutine hsmg_tnsr2d(v,nv,u,nu,A,Bt)
      integer nv,nu
      real v(nv*nv,nelv),u(nu*nu,nelv),A(1),Bt(1)
      include 'SIZE'
      common /hsmgw/ work((lx1+2)*(lx1+2))
      integer ie
      do ie=1,nelv
         call mxm(A,nv,u(1,ie),nu,work,nu)
         call mxm(work,nv,Bt,nu,v(1,ie),nv)
      enddo
      return
      end
c----------------------------------------------------------------------
c     computes
c              
c     v = [C (x) B (x) A] u
      subroutine hsmg_tnsr3d(v,nv,u,nu,A,Bt,Ct)
      integer nv,nu
      real v(nv*nv*nv,nelv),u(nu*nu*nu,nelv),A(1),Bt(1),Ct(1)
      include 'SIZE'
      parameter (lwk=(lx1+2)*(ly1+2)*(lz1+2))
      common /hsmgw/ work(0:lwk-1),work2(0:lwk-1)
      integer ie, i
      do ie=1,nelv
         call mxm(A,nv,u(1,ie),nu,work,nu*nu)
         do i=0,nu-1
            call mxm(work(nv*nu*i),nv,Bt,nu,work2(nv*nv*i),nv)
         enddo
         call mxm(work2,nv*nv,Ct,nu,v(1,ie),nv)
      enddo
      return
      end
c----------------------------------------------------------------------
c     computes
c              T
c     v = A u B
      subroutine hsmg_tnsr2d_el(v,nv,u,nu,A,Bt)
      integer nv,nu
      real v(nv*nv),u(nu*nu),A(1),Bt(1)
      include 'SIZE'
      common /hsmgw/ work((lx1+2)*(lx1+2))
c
      call mxm(A,nv,u,nu,work,nu)
      call mxm(work,nv,Bt,nu,v,nv)
c
      return
      end
c----------------------------------------------------------------------
c     computes
c              
c     v = [C (x) B (x) A] u
      subroutine hsmg_tnsr3d_el(v,nv,u,nu,A,Bt,Ct)
      integer nv,nu
      real v(nv*nv*nv),u(nu*nu*nu),A(1),Bt(1),Ct(1)
      include 'SIZE'
      parameter (lwk=(lx1+2)*(ly1+2)*(lz1+2))
      common /hsmgw/ work(0:lwk-1),work2(0:lwk-1)
      integer i
c
      call mxm(A,nv,u,nu,work,nu*nu)
      do i=0,nu-1
         call mxm(work(nv*nu*i),nv,Bt,nu,work2(nv*nv*i),nv)
      enddo
      call mxm(work2,nv*nv,Ct,nu,v,nv)
c
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_dssum(u,l)
      include 'SIZE'
      include 'HSMG'
      include 'CTIMER'
      real u(1)

      if (ifsync) call nekgsync()
      etime1=dnekclock()

      call fgslib_gs_op(mg_gsh_handle(l,mg_fld),u,1,1,0)
      tdadd =tdadd + dnekclock()-etime1


      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_dsprod(u,l)
      include 'SIZE'
      include 'HSMG'
      include 'CTIMER'
      real u(1)

      if (ifsync) call nekgsync()

      call fgslib_gs_op(mg_gsh_handle(l,mg_fld),u,1,2,0)
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_dssum(u,l)
      include 'SIZE'
      include 'HSMG'
      include 'CTIMER'

      if (ifsync) call nekgsync()
      etime1=dnekclock()

      call fgslib_gs_op(mg_gsh_schwarz_handle(l,mg_fld),u,1,1,0)
      tdadd =tdadd + dnekclock()-etime1

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_extrude(arr1,l1,f1,arr2,l2,f2,nx,ny,nz)
      include 'SIZE'
      include 'INPUT'
      integer l1,l2,nx,ny,nz
      real arr1(nx,ny,nz,nelv),arr2(nx,ny,nz,nelv)
      real f1,f2
      
      integer i,j,k,ie,i0,i1
      i0=2
      i1=nx-1
      
      if(.not.if3d) then
         do ie=1,nelv
            do j=i0,i1
               arr1(l1+1 ,j,1,ie) = f1*arr1(l1+1 ,j,1,ie)
     $                             +f2*arr2(l2+1 ,j,1,ie)
               arr1(nx-l1,j,1,ie) = f1*arr1(nx-l1,j,1,ie)
     $                             +f2*arr2(nx-l2,j,1,ie)
            enddo
            do i=i0,i1
               arr1(i,l1+1 ,1,ie) = f1*arr1(i,l1+1 ,1,ie)
     $                             +f2*arr2(i,l2+1 ,1,ie)
               arr1(i,ny-l1,1,ie) = f1*arr1(i,ny-l1,1,ie)
     $                             +f2*arr2(i,nx-l2,1,ie)
            enddo
         enddo
      else
         do ie=1,nelv
            do k=i0,i1
            do j=i0,i1
               arr1(l1+1 ,j,k,ie) = f1*arr1(l1+1 ,j,k,ie)
     $                             +f2*arr2(l2+1 ,j,k,ie)
               arr1(nx-l1,j,k,ie) = f1*arr1(nx-l1,j,k,ie)
     $                             +f2*arr2(nx-l2,j,k,ie)
            enddo
            enddo
            do k=i0,i1
            do i=i0,i1
               arr1(i,l1+1 ,k,ie) = f1*arr1(i,l1+1 ,k,ie)
     $                             +f2*arr2(i,l2+1 ,k,ie)
               arr1(i,nx-l1,k,ie) = f1*arr1(i,nx-l1,k,ie)
     $                             +f2*arr2(i,nx-l2,k,ie)
            enddo
            enddo
            do j=i0,i1
            do i=i0,i1
               arr1(i,j,l1+1 ,ie) = f1*arr1(i,j,l1+1 ,ie)
     $                             +f2*arr2(i,j,l2+1 ,ie)
               arr1(i,j,nx-l1,ie) = f1*arr1(i,j,nx-l1,ie)
     $                             +f2*arr2(i,j,nx-l2,ie)
            enddo
            enddo
         enddo
      endif
      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_schwarz(e,r,sigma,l)
      include 'SIZE'
      include 'HSMG'

      real e(1),r(1)

      n = mg_h1_n(l,mg_fld)

      call h1mg_schwarz_part1 (e,r,l)
      call hsmg_schwarz_wt    (e,l)          ! e  := W e
      call cmult              (e,sigma,n)    !  l       l

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_schwarz_part1 (e,r,l)
      include 'SIZE'
      include 'INPUT'  ! if3d
      include 'TSTEP'  ! ifield
      include 'HSMG'

      real e(1),r(1)

      integer enx,eny,enz,pm

      zero =  0
      one  =  1
      onem = -1

      n  = mg_h1_n (l,mg_fld)
      pm = p_mg_msk(l,mg_fld)

      call h1mg_mask  (r,mg_imask(pm),nelfld(ifield))  ! Zero Dirichlet nodes

      if (if3d) then ! extended array 
         call hsmg_schwarz_toext3d(mg_work,r,mg_nh(l))
      else
         call hsmg_schwarz_toext2d(mg_work,r,mg_nh(l))
      endif

      enx=mg_nh(l)+2
      eny=mg_nh(l)+2
      enz=mg_nh(l)+2
      if(.not.if3d) enz=1
      i = enx*eny*enz*nelv+1
 
c     exchange interior nodes
      call hsmg_extrude(mg_work,0,zero,mg_work,2,one,enx,eny,enz)
      call hsmg_schwarz_dssum(mg_work,l)
      call hsmg_extrude(mg_work,0,one ,mg_work,2,onem,enx,eny,enz)

      call hsmg_fdm(mg_work(i),mg_work,l) ! Do the local solves

c     Sum overlap region (border excluded)
      call hsmg_extrude(mg_work,0,zero,mg_work(i),0,one ,enx,eny,enz)
      call hsmg_schwarz_dssum(mg_work(i),l)
      call hsmg_extrude(mg_work(i),0,one ,mg_work,0,onem,enx,eny,enz)
      call hsmg_extrude(mg_work(i),2,one,mg_work(i),0,one,enx,eny,enz)

      if(.not.if3d) then ! Go back to regular size array
         call hsmg_schwarz_toreg2d(e,mg_work(i),mg_nh(l))
      else
         call hsmg_schwarz_toreg3d(e,mg_work(i),mg_nh(l))
      endif

      call hsmg_dssum(e,l)                           ! sum border nodes
      call h1mg_mask (e,mg_imask(pm),nelfld(ifield)) ! apply mask 

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz(e,r,l)
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      real e(1),r(1)
      integer l
      integer enx,eny,enz
      integer i

      real zero,one,onem
      zero =  0
      one  =  1
      onem = -1

c     apply mask (zeros Dirichlet nodes)
      !!!!! uncommenting
      call hsmg_do_wt(r,mg_mask(mg_mask_index(l,mg_fld)),
     $                mg_nh(l),mg_nh(l),mg_nhz(l))
      
c     go to extended size array (room for overlap)      
      if (if3d) then
         call hsmg_schwarz_toext3d(mg_work,r,mg_nh(l))
      else
         call hsmg_schwarz_toext2d(mg_work,r,mg_nh(l))
      endif
      
      enx=mg_nh(l)+2
      eny=mg_nh(l)+2
      enz=mg_nh(l)+2
      if(.not.if3d) enz=1
      i = enx*eny*enz*nelv+1

c     exchange interior nodes
      call hsmg_extrude(mg_work,0,zero,mg_work,2,one,enx,eny,enz)
      call hsmg_schwarz_dssum(mg_work,l)
      call hsmg_extrude(mg_work,0,one ,mg_work,2,onem,enx,eny,enz)

c     do the local solves
      call hsmg_fdm(mg_work(i),mg_work,l)
c     sum overlap region (border excluded)
      call hsmg_extrude(mg_work,0,zero,mg_work(i),0,one ,enx,eny,enz)
      call hsmg_schwarz_dssum(mg_work(i),l)
      call hsmg_extrude(mg_work(i),0,one ,mg_work,0,onem,enx,eny,enz)
      call hsmg_extrude(mg_work(i),2,one,mg_work(i),0,one,enx,eny,enz)
c     go back to regular size array
      if(.not.if3d) then
         call hsmg_schwarz_toreg2d(e,mg_work(i),mg_nh(l))
      else
         call hsmg_schwarz_toreg3d(e,mg_work(i),mg_nh(l))
      endif
c     sum border nodes
      call hsmg_dssum(e,l)
c     apply mask (zeros Dirichlet nodes)
      !!!!!! changing r to e
      call hsmg_do_wt(e,mg_mask(mg_mask_index(l,mg_fld)),
     $                mg_nh(l),mg_nh(l),mg_nhz(l))
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_toext2d(a,b,n)
      include 'SIZE'
      integer n
      real a(0:n+1,0:n+1,nelv),b(n,n,nelv)
      
      integer i,j,ie
c      call rzero(a,(n+2)*(n+2)*nelv)
      do ie=1,nelv
         do i=0,n+1
            a(i,0,ie)=0.
         enddo
         do j=1,n
            a(0  ,j,ie)=0.
            do i=1,n
               a(i,j,ie)=b(i,j,ie)
            enddo
            a(n+1,j,ie)=0.
         enddo
         do i=0,n+1
            a(i,n+1,ie)=0.
         enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_toext3d(a,b,n)
      include 'SIZE'
      integer n
      real a(0:n+1,0:n+1,0:n+1,nelv),b(n,n,n,nelv)
      
      integer i,j,k,ie
      call rzero(a,(n+2)*(n+2)*(n+2)*nelv)
      do ie=1,nelv
      do k=1,n
      do j=1,n
      do i=1,n
         a(i,j,k,ie)=b(i,j,k,ie)
      enddo
      enddo
      enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_toreg2d(b,a,n)
      include 'SIZE'
      integer n
      real a(0:n+1,0:n+1,nelv),b(n,n,nelv)
      
      integer i,j,ie
      do ie=1,nelv
      do j=1,n
      do i=1,n
         b(i,j,ie)=a(i,j,ie)
      enddo
      enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_toreg3d(b,a,n)
      include 'SIZE'
      integer n
      real a(0:n+1,0:n+1,0:n+1,nelv),b(n,n,n,nelv)
      
      integer i,j,k,ie
      do ie=1,nelv
      do k=1,n
      do j=1,n
      do i=1,n
         b(i,j,k,ie)=a(i,j,k,ie)
      enddo
      enddo
      enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_fdm()
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      
      integer l,i,j,nl
      i = mg_fast_s_index(mg_lmax,mg_fld-1)
      j = mg_fast_d_index(mg_lmax,mg_fld-1)
      do l=2,mg_lmax
         mg_fast_s_index(l,mg_fld)=i
         nl = mg_nh(l)+2
         i=i+nl*nl*2*ldim*nelv
         if(i .gt. lmg_fasts*2*ldim*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'lmg_fasts too small',i,itmp,lmg_fasts,l
            call exitt
         endif
         mg_fast_d_index(l,mg_fld)=j
         j=j+(nl**ldim)*nelv
         if(j .gt. lmg_fastd*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'lmg_fastd too small',i,itmp,lmg_fastd,l
            call exitt
         endif
         call hsmg_setup_fast(
     $             mg_fast_s(mg_fast_s_index(l,mg_fld))
     $            ,mg_fast_d(mg_fast_d_index(l,mg_fld))
     $            ,mg_nh(l)+2,mg_ah(1,l),mg_bh(1,l),mg_nx(l))
      enddo
      mg_fast_s_index(l,mg_fld)=i
      mg_fast_d_index(l,mg_fld)=j
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_fdm()
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      
      integer l,i,j,nl
      i = mg_fast_s_index(mg_lmax,mg_fld-1)
      j = mg_fast_d_index(mg_lmax,mg_fld-1)
      do l=2,mg_lmax-1
         mg_fast_s_index(l,mg_fld)=i
         nl = mg_nh(l)+2
         i=i+nl*nl*2*ldim*nelv
         if(i .gt. lmg_fasts*2*ldim*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'lmg_fasts too small',i,itmp,lmg_fasts,l
            call exitt
         endif
         mg_fast_d_index(l,mg_fld)=j
         j=j+(nl**ldim)*nelv
         if(j .gt. lmg_fastd*lelv) then
            itmp = i/(2*ldim*lelv)
            write(6,*) 'lmg_fastd too small',i,itmp,lmg_fastd,l
            call exitt
         endif
         call hsmg_setup_fast(
     $             mg_fast_s(mg_fast_s_index(l,mg_fld))
     $            ,mg_fast_d(mg_fast_d_index(l,mg_fld))
     $            ,mg_nh(l)+2,mg_ah(1,l),mg_bh(1,l),mg_nx(l))
      enddo
      mg_fast_s_index(l,mg_fld)=i
      mg_fast_d_index(l,mg_fld)=j
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_fast(s,d,nl,ah,bh,n)
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      real s(nl*nl,2,ldim,nelv)
      real d(nl**ldim,nelv)
      real ah(1),bh(1)
      common /ctmpf/  lr(2*lx1+4),ls(2*lx1+4),lt(2*lx1+4)
     $              , llr(lelt),lls(lelt),llt(lelt)
     $              , lmr(lelt),lms(lelt),lmt(lelt)
     $              , lrr(lelt),lrs(lelt),lrt(lelt)
      real lr ,ls ,lt
      real llr,lls,llt
      real lmr,lms,lmt
      real lrr,lrs,lrt
      
      integer i,j,k
      integer ie,il,nr,ns,nt
      integer lbr,rbr,lbs,rbs,lbt,rbt,two
      real eps,diag
      
      two  = 2
      ierr = 0
      do ie=1,nelv
         call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,ie,two,ierr)
         nr=nl
         ns=nl
         nt=nl
         call hsmg_setup_fast1d(s(1,1,1,ie),lr,nr,lbr,rbr
     $            ,llr(ie),lmr(ie),lrr(ie),ah,bh,n,ie)
         call hsmg_setup_fast1d(s(1,1,2,ie),ls,ns,lbs,rbs
     $            ,lls(ie),lms(ie),lrs(ie),ah,bh,n,ie)
         if(if3d) call hsmg_setup_fast1d(s(1,1,3,ie),lt,nt,lbt,rbt
     $                     ,llt(ie),lmt(ie),lrt(ie),ah,bh,n,ie)
         il=1
         if(.not.if3d) then
            eps = 1.e-5*(vlmax(lr(2),nr-2) + vlmax(ls(2),ns-2))
            do j=1,ns
            do i=1,nr
               diag = lr(i)+ls(j)
               if (diag.gt.eps) then
                  d(il,ie) = 1.0/diag
               else
c                 write(6,2) ie,'Reset Eig in hsmg setup fast:',i,j,l
c    $                         ,eps,diag,lr(i),ls(j)
    2             format(i6,1x,a21,3i5,1p4e12.4)
                  d(il,ie) = 0.0
               endif
               il=il+1
            enddo
            enddo
         else
            eps = 1.e-5 * (vlmax(lr(2),nr-2)
     $                  + vlmax(ls(2),ns-2) + vlmax(lt(2),nt-2))
            do k=1,nt
            do j=1,ns
            do i=1,nr
               diag = lr(i)+ls(j)+lt(k)
               if (diag.gt.eps) then
                  d(il,ie) = 1.0/diag
               else
c                 write(6,3) ie,'Reset Eig in hsmg setup fast:',i,j,k,l
c    $                         ,eps,diag,lr(i),ls(j),lt(k)
    3             format(i6,1x,a21,4i5,1p5e12.4)
                  d(il,ie) = 0.0
               endif
               il=il+1
            enddo
            enddo
            enddo
         endif
      enddo

      ierrmx = iglmax(ierr,1)
      if (ierrmx.gt.0) then
         if (ierr.gt.0) write(6,*) nid,ierr,' BC FAIL'
         call exitti('A INVALID BC FOUND in genfast$',ierrmx)
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_fast1d(s,lam,nl,lbc,rbc,ll,lm,lr,ah,bh,n,ie)
      integer nl,lbc,rbc,n
      real s(nl,nl,2),lam(nl),ll,lm,lr
      real ah(0:n,0:n),bh(0:n)
      
      include 'SIZE'
      parameter(lxm=lx1+2)
      common /ctmp0/ b(2*lxm*lxm),w(2*lxm*lxm)
      
      call hsmg_setup_fast1d_a(s,lbc,rbc,ll,lm,lr,ah,n)
      call hsmg_setup_fast1d_b(b,lbc,rbc,ll,lm,lr,bh,n)
      
c     if (nid.eq.0) write(6,*) 'THIS is generalev call',nl,lbc
      call generalev(s,b,lam,nl,w)
      if(lbc.gt.0) call row_zero(s,nl,nl,1)
      if(lbc.eq.1) call row_zero(s,nl,nl,2)
      if(rbc.gt.0) call row_zero(s,nl,nl,nl)
      if(rbc.eq.1) call row_zero(s,nl,nl,nl-1)
      
      call transpose(s(1,1,2),nl,s,nl)
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_fast1d_a(a,lbc,rbc,ll,lm,lr,ah,n)
      integer lbc,rbc,n
      real a(0:n+2,0:n+2),ll,lm,lr
      real ah(0:n,0:n)
      
      real fac
      integer i,j,i0,i1
      i0=0
      if(lbc.eq.1) i0=1
      i1=n
      if(rbc.eq.1) i1=n-1
      
      call rzero(a,(n+3)*(n+3))
      fac = 2.0/lm
      a(1,1)=1.0
      a(n+1,n+1)=1.0
      do j=i0,i1
         do i=i0,i1
            a(i+1,j+1)=fac*ah(i,j)
         enddo
      enddo
      if(lbc.eq.0) then
         fac = 2.0/ll
         a(0,0)=fac*ah(n-1,n-1)
         a(1,0)=fac*ah(n  ,n-1)
         a(0,1)=fac*ah(n-1,n  )
         a(1,1)=a(1,1)+fac*ah(n  ,n  )
      else
         a(0,0)=1.0
      endif
      if(rbc.eq.0) then
         fac = 2.0/lr
         a(n+1,n+1)=a(n+1,n+1)+fac*ah(0,0)
         a(n+2,n+1)=fac*ah(1,0)
         a(n+1,n+2)=fac*ah(0,1)
         a(n+2,n+2)=fac*ah(1,1)
      else
         a(n+2,n+2)=1.0
      endif
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_fast1d_b(b,lbc,rbc,ll,lm,lr,bh,n)
      integer lbc,rbc,n
      real b(0:n+2,0:n+2),ll,lm,lr
      real bh(0:n)
      
      real fac
      integer i,j,i0,i1
      i0=0
      if(lbc.eq.1) i0=1
      i1=n
      if(rbc.eq.1) i1=n-1
      
      call rzero(b,(n+3)*(n+3))
      fac = 0.5*lm
      b(1,1)=1.0
      b(n+1,n+1)=1.0
      do i=i0,i1
         b(i+1,i+1)=fac*bh(i)
      enddo
      if(lbc.eq.0) then
         fac = 0.5*ll
         b(0,0)=fac*bh(n-1)
         b(1,1)=b(1,1)+fac*bh(n  )
      else
         b(0,0)=1.0
      endif
      if(rbc.eq.0) then
         fac = 0.5*lr
         b(n+1,n+1)=b(n+1,n+1)+fac*bh(0)
         b(n+2,n+2)=fac*bh(1)
      else
         b(n+2,n+2)=1.0
      endif
      return
      end
c----------------------------------------------------------------------
c     clobbers r
      subroutine hsmg_fdm(e,r,l)
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      call hsmg_do_fast(e,r,
     $      mg_fast_s(mg_fast_s_index(l,mg_fld)),
     $      mg_fast_d(mg_fast_d_index(l,mg_fld)),
     $      mg_nh(l)+2)
      return
      end
c----------------------------------------------------------------------
c     clobbers r
      subroutine hsmg_do_fast(e,r,s,d,nl)
      include 'SIZE'
      include 'INPUT'
      real e(nl**ldim,nelv)
      real r(nl**ldim,nelv)
      real s(nl*nl,2,ldim,nelv)
      real d(nl**ldim,nelv)
      
      integer ie,nn,i
      nn=nl**ldim
      if(.not.if3d) then
         do ie=1,nelv
            call hsmg_tnsr2d_el(e(1,ie),nl,r(1,ie),nl
     $                         ,s(1,2,1,ie),s(1,1,2,ie))
            do i=1,nn
               r(i,ie)=d(i,ie)*e(i,ie)
            enddo
            call hsmg_tnsr2d_el(e(1,ie),nl,r(1,ie),nl
     $                         ,s(1,1,1,ie),s(1,2,2,ie))
         enddo
      else
         do ie=1,nelv
            call hsmg_tnsr3d_el(e(1,ie),nl,r(1,ie),nl
     $                         ,s(1,2,1,ie),s(1,1,2,ie),s(1,1,3,ie))
            do i=1,nn
               r(i,ie)=d(i,ie)*e(i,ie)
            enddo
            call hsmg_tnsr3d_el(e(1,ie),nl,r(1,ie),nl
     $                         ,s(1,1,1,ie),s(1,2,2,ie),s(1,2,3,ie))
         enddo
      endif
      return
      end
c----------------------------------------------------------------------
c     u = wt .* u
      subroutine hsmg_do_wt(u,wt,nx,ny,nz)
      include 'SIZE'
      include 'INPUT'
      integer nx,ny,nz
      real u(nx,ny,nz,nelv)
      real wt(nx,nz,2,ldim,nelv)
      
      integer e

c     if (nx.eq.2) then
c        do e=1,nelv
c           call outmat(wt(1,1,1,1,e),nx,nz,'wt 1-1',e)
c           call outmat(wt(1,1,2,1,e),nx,nz,'wt 2-1',e)
c           call outmat(wt(1,1,1,2,e),nx,nz,'wt 1-2',e)
c           call outmat(wt(1,1,2,2,e),nx,nz,'wt 2-2',e)
c        enddo
c        call exitti('hsmg_do_wt quit$',nelv)
c     endif

      if (.not. if3d) then
         do ie=1,nelv
            do j=1,ny
               u( 1,j,1,ie)=u( 1,j,1,ie)*wt(j,1,1,1,ie)
               u(nx,j,1,ie)=u(nx,j,1,ie)*wt(j,1,2,1,ie)
            enddo
            do i=2,nx-1
               u(i, 1,1,ie)=u(i, 1,1,ie)*wt(i,1,1,2,ie)
               u(i,ny,1,ie)=u(i,ny,1,ie)*wt(i,1,2,2,ie)
            enddo
         enddo
      else
         do ie=1,nelv
            do k=1,nz
            do j=1,ny
               u( 1,j,k,ie)=u( 1,j,k,ie)*wt(j,k,1,1,ie)
               u(nx,j,k,ie)=u(nx,j,k,ie)*wt(j,k,2,1,ie)
            enddo
            enddo
            do k=1,nz
            do i=2,nx-1
               u(i, 1,k,ie)=u(i, 1,k,ie)*wt(i,k,1,2,ie)
               u(i,ny,k,ie)=u(i,ny,k,ie)*wt(i,k,2,2,ie)
            enddo
            enddo
            do j=2,ny-1
            do i=2,nx-1
               u(i,j, 1,ie)=u(i,j, 1,ie)*wt(i,j,1,3,ie)
               u(i,j,nz,ie)=u(i,j,nz,ie)*wt(i,j,2,3,ie)
            enddo
            enddo
         enddo
      endif
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_rstr_wt(wt,nx,ny,nz,l,w)
      include 'SIZE'
      include 'INPUT'
      integer nx,ny,nz,l
      real w(nx,ny,nz,nelv)
      real wt(nx,nz,2,ldim,nelv)
      
      integer ie
      !init border nodes to 1
      call rzero(w,nx*ny*nz*nelv)
c     print *, 'Setup rstr wt: ',nx,ny,nz,nelv
      if (.not.if3d) then
         do ie=1,nelv
            do i=1,nx
               w(i,1,1,ie)=1.0
               w(i,ny,1,ie)=1.0
            enddo
            do j=1,ny
               w(1,j,1,ie)=1.0
               w(nx,j,1,ie)=1.0
            enddo
         enddo
      else
         do ie=1,nelv
            do j=1,ny
            do i=1,nx
               w(i,j,1,ie)=1.0
               w(i,j,nz,ie)=1.0
            enddo
            enddo
            do k=1,nz
            do i=1,nx
               w(i,1,k,ie)=1.0
               w(i,ny,k,ie)=1.0
            enddo
            enddo
            do k=1,nz
            do j=1,ny
               w(1,j,k,ie)=1.0
               w(nx,j,k,ie)=1.0
            enddo
            enddo
         enddo
      endif
      call hsmg_dssum(w,l)
      !invert the count w to get the weight wt
      if (.not. if3d) then
         do ie=1,nelv
            do j=1,ny
               wt(j,1,1,1,ie)=1.0/w(1,j,1,ie)
               wt(j,1,2,1,ie)=1.0/w(nx,j,1,ie)
            enddo
            do i=1,nx
               wt(i,1,1,2,ie)=1.0/w(i,1,1,ie)
               wt(i,1,2,2,ie)=1.0/w(i,ny,1,ie)
            enddo
         enddo
      else
         do ie=1,nelv
            do k=1,nz
            do j=1,ny
               wt(j,k,1,1,ie)=1.0/w(1,j,k,ie)
               wt(j,k,2,1,ie)=1.0/w(nx,j,k,ie)
            enddo
            enddo
            do k=1,nz
            do i=1,nx
               wt(i,k,1,2,ie)=1.0/w(i,1,k,ie)
               wt(i,k,2,2,ie)=1.0/w(i,ny,k,ie)
            enddo
            enddo
            do j=1,ny
            do i=1,nx
               wt(i,j,1,3,ie)=1.0/w(i,j,1,ie)
               wt(i,j,2,3,ie)=1.0/w(i,j,nz,ie)
            enddo
            enddo
         enddo
      endif
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_mask(wt,nx,ny,nz,l,w)
      include 'SIZE'
      include 'INPUT'
      integer nx,ny,nz,l
      real w(nx,ny,nz,nelv)
      real wt(nx,nz,2,ldim,nelv)
      
      integer ie
      integer lbr,rbr,lbs,rbs,lbt,rbt,two
c     init everything to 1

      n = nx*ny*nz*nelv
      call rone(w,n)

c     set dirichlet nodes to zero
      ierr = 0
      two  = 2
      do ie=1,nelv
         call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,ie,two,ierr)
         if (ierr.ne.0) then
            ierr = -1
            call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,ie,two,ierr)
         endif

         if(lbr.eq.1) then
            do k=1,nz
            do j=1,ny
               w(1,j,k,ie)=0.0
            enddo
            enddo
         endif
         if(rbr.eq.1) then
            do k=1,nz
            do j=1,ny
               w(nx,j,k,ie)=0.0
            enddo
            enddo
         endif
         if(lbs.eq.1) then
            do k=1,nz
            do i=1,nx
               w(i,1,k,ie)=0.0
            enddo
            enddo
         endif
         if(rbs.eq.1) then
            do k=1,nz
            do i=1,nx
               w(i,ny,k,ie)=0.0
            enddo
            enddo
         endif
         if(if3d) then
            if(lbt.eq.1) then
               do j=1,ny
               do i=1,nx
                  w(i,j,1,ie)=0.0
               enddo
               enddo
            endif
            if(rbt.eq.1) then
               do j=1,ny
               do i=1,nx
                  w(i,j,nz,ie)=0.0
               enddo
               enddo
            endif
         endif
      enddo
c     do direct stiffness multiply

      call hsmg_dsprod(w,l)


c     store weight
      if (.not. if3d) then
         do ie=1,nelv
            do j=1,ny
               wt(j,1,1,1,ie)=w(1,j,1,ie)
               wt(j,1,2,1,ie)=w(nx,j,1,ie)
            enddo
            do i=1,nx
               wt(i,1,1,2,ie)=w(i,1,1,ie)
               wt(i,1,2,2,ie)=w(i,ny,1,ie)
            enddo
         enddo
      else
         do ie=1,nelv
            do k=1,nz
            do j=1,ny
               wt(j,k,1,1,ie)=w(1,j,k,ie)
               wt(j,k,2,1,ie)=w(nx,j,k,ie)
            enddo
            enddo
            do k=1,nz
            do i=1,nx
               wt(i,k,1,2,ie)=w(i,1,k,ie)
               wt(i,k,2,2,ie)=w(i,ny,k,ie)
            enddo
            enddo
            do k=1,nz
            do j=1,ny
               wt(j,k,1,3,ie)=w(i,j,1,ie)
               wt(j,k,2,3,ie)=w(i,j,nz,ie)
            enddo
            enddo
         enddo
      endif

      ierrmx = iglmax(ierr,1)
      if (ierrmx.gt.0) then
         if (ierr.gt.0) write(6,*) nid,ierr,' BC FAIL'
         call exitti('B INVALID BC FOUND in genfast$',ierrmx)
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_schwarz_wt(ifsqrt)
      logical ifsqrt
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      
      integer l,i,nl,nlz

      i = mg_schwarz_wt_index(mg_lmax,mg_fld-1)
      do l=2,mg_lmax-1
         mg_schwarz_wt_index(l,mg_fld)=i
         nl = mg_nh(l)
         nlz = mg_nh(l)
         if(.not.if3d) nlz=1
         i=i+nl*nlz*4*ldim*nelv
         if(i .gt. lmg_swt*4*ldim*lelv) then
            itmp = i/(4*ldim*lelv)
            write(6,*) 'lmg_swt too small',i,itmp,lmg_swt,l
            call exitt
         endif

         call h1mg_setup_schwarz_wt_1(
     $      mg_schwarz_wt(mg_schwarz_wt_index(l,mg_fld)),l,ifsqrt)

      enddo
      mg_schwarz_wt_index(l,mg_fld)=i

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_schwarz_wt(ifsqrt)
      logical ifsqrt
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      
      integer l,i,nl,nlz

      i = mg_schwarz_wt_index(mg_lmax,mg_fld-1)
      do l=2,mg_lmax

         mg_schwarz_wt_index(l,mg_fld)=i
         nl  = mg_nh(l)
         nlz = mg_nhz(l)
         i   = i+nl*nlz*4*ldim*nelv

         if (i .gt. lmg_swt*4*ldim*lelv) then
            itmp = i/(4*ldim*lelv)
            write(6,*) 'lmg_swt too small',i,itmp,lmg_swt,l
            call exitt
         endif

         call h1mg_setup_schwarz_wt_1(
     $      mg_schwarz_wt(mg_schwarz_wt_index(l,mg_fld)),l,ifsqrt)

      enddo

      mg_schwarz_wt_index(l,mg_fld)=i

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_wt(e,l)
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      
      if(.not.if3d) call hsmg_schwarz_wt2d(
     $    e,mg_schwarz_wt(mg_schwarz_wt_index(l,mg_fld)),mg_nh(l))
      if(if3d) call hsmg_schwarz_wt3d(
     $    e,mg_schwarz_wt(mg_schwarz_wt_index(l,mg_fld)),mg_nh(l))
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_wt2d(e,wt,n)
      include 'SIZE'
      integer n
      real e(n,n,nelv)
      real wt(n,4,2,nelv)
      
      integer ie,i,j
      do ie=1,nelv
         do j=1,n
            e(1  ,j,ie)=e(1  ,j,ie)*wt(j,1,1,ie)
            e(2  ,j,ie)=e(2  ,j,ie)*wt(j,2,1,ie)
            e(n-1,j,ie)=e(n-1,j,ie)*wt(j,3,1,ie)
            e(n  ,j,ie)=e(n  ,j,ie)*wt(j,4,1,ie)
         enddo
         do i=3,n-2
            e(i,1  ,ie)=e(i,1  ,ie)*wt(i,1,2,ie)
            e(i,2  ,ie)=e(i,2  ,ie)*wt(i,2,2,ie)
            e(i,n-1,ie)=e(i,n-1,ie)*wt(i,3,2,ie)
            e(i,n  ,ie)=e(i,n  ,ie)*wt(i,4,2,ie)
         enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_schwarz_wt3d(e,wt,n)
      include 'SIZE'
      integer n
      real e(n,n,n,nelv)
      real wt(n,n,4,3,nelv)
      
      integer ie,i,j,k
      do ie=1,nelv
         do k=1,n
         do j=1,n
            e(1  ,j,k,ie)=e(1  ,j,k,ie)*wt(j,k,1,1,ie)
            e(2  ,j,k,ie)=e(2  ,j,k,ie)*wt(j,k,2,1,ie)
            e(n-1,j,k,ie)=e(n-1,j,k,ie)*wt(j,k,3,1,ie)
            e(n  ,j,k,ie)=e(n  ,j,k,ie)*wt(j,k,4,1,ie)
         enddo
         enddo
         do k=1,n
         do i=3,n-2
            e(i,1  ,k,ie)=e(i,1  ,k,ie)*wt(i,k,1,2,ie)
            e(i,2  ,k,ie)=e(i,2  ,k,ie)*wt(i,k,2,2,ie)
            e(i,n-1,k,ie)=e(i,n-1,k,ie)*wt(i,k,3,2,ie)
            e(i,n  ,k,ie)=e(i,n  ,k,ie)*wt(i,k,4,2,ie)
         enddo
         enddo
         do j=3,n-2
         do i=3,n-2
            e(i,j,1  ,ie)=e(i,j,1  ,ie)*wt(i,j,1,3,ie)
            e(i,j,2  ,ie)=e(i,j,2  ,ie)*wt(i,j,2,3,ie)
            e(i,j,n-1,ie)=e(i,j,n-1,ie)*wt(i,j,3,3,ie)
            e(i,j,n  ,ie)=e(i,j,n  ,ie)*wt(i,j,4,3,ie)
         enddo
         enddo
      enddo
      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_coarse_solve(e,r)
      include 'SIZE'
      include 'DOMAIN'
      include 'ESOLV'
      include 'GEOM'
      include 'SOLN'
      include 'PARALLEL'
      include 'HSMG'
      include 'CTIMER'
      include 'INPUT'
      include 'TSTEP'
      real e(1),r(1)
c
      integer n_crs_tot
      save    n_crs_tot
      data    n_crs_tot /0/
c
      if (icalld.eq.0) then ! timer info
         ncrsl=0
         tcrsl=0.0
      endif
      icalld = 1

      if (ifsync) call nekgsync()

      ncrsl  = ncrsl  + 1
      etime1=dnekclock()

      call fgslib_crs_solve(xxth(ifield),e,r)

      tcrsl=tcrsl+dnekclock()-etime1

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_solve
      include 'SIZE'
      include 'HSMG'
      
      integer l,i,nl,nlz
      i = mg_solve_index(mg_lmax+1,mg_fld-1)
      do l=1,mg_lmax
         mg_solve_index(l,mg_fld)=i
         i=i+mg_nh(l)*mg_nh(l)*mg_nhz(l)*nelv
         if(i .gt. lmg_solve*lelv) then
            itmp = i/lelv
            write(6,*) 'lmg_solve too small',i,itmp,lmg_solve,l
            call exitt
         endif
      enddo
      mg_solve_index(l,mg_fld)=i

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_solve(e,r)
      real e(1),r(1)
      include 'SIZE'
      include 'HSMG'
      include 'GEOM'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
      include 'CTIMER'
      include 'PARALLEL'
      
      common /quick/ ecrs  (2)  ! quick work array
     $             , ecrs2 (2)  ! quick work array
c     common /quick/ ecrs  (lx2*ly2*lz2*lelv)  ! quick work array
c    $             , ecrs2 (lx2*ly2*lz2*lelv)  ! quick work array

      common /scrhi/ h2inv (lx1,ly1,lz1,lelv)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv),
     $               h2    (lx1,ly1,lz1,lelv)
      
      integer ilstep,iter
      save    ilstep,iter
      data    ilstep,iter /0,0/

      real    rhoavg,copt(2),copw(2)
      save    rhoavg,copt1,copt2
      data    rhoavg,copt1,copt2 /3*1./  ! Default copt = 1 for additive

      integer l,nt
      integer*8 ntotg,nxyz2

      logical if_hybrid

      mg_fld = 1
      if (ifield.gt.1) mg_fld = 2

      if (istep.ne.ilstep) then
         ilstep = istep
         ntot1  = lx1*ly1*lz1*nelv
         rhoavg = glsc2(vtrans,bm1,ntot1)/volvm1
      endif

      n = lx2*ly2*lz2*nelv
c     call copy(e,r,n)
c     return
 
      if (icalld.eq.0) then

         tddsl=0.0
         nddsl=0

         icalld = 1
         taaaa = 0
         tbbbb = 0
         tcccc = 0
         tdddd = 0
         teeee = 0
      endif

      nddsl  = nddsl  + 1
      etime1 = dnekclock()

      
c     n = lx2*ly2*lz2*nelv
c     rmax = glmax(r,n)
c     if (nid.eq.0) write(6,*) istep,n,rmax,' rmax1'
       
      iter = iter + 1      

      l = mg_lmax
      nt = mg_nh(l)*mg_nh(l)*mg_nhz(l)*nelv
      ! e := W M        r
      !         Schwarz
      time_0 = dnekclock()
      call local_solves_fdm(e,r)

      time_1 = dnekclock()

c     if (param(41).eq.1) if_hybrid = .true.
      if_hybrid = .false.

      if (if_hybrid) then
         ! w := E e
         rbd1dt = rhoavg*bd(1)/dt ! Assumes constant density!!!
         call cdabdtp(mg_work2,e,h1,h2,h2inv,1)
         call cmult  (mg_work2,rbd1dt,nt)
         time_2 = dnekclock()
         if (istep.eq.1) then
            copt(1)  = vlsc2(r       ,mg_work2,nt)
            copt(2)  = vlsc2(mg_work2,mg_work2,nt)
            call gop(copt,copw,'+  ', 2)
            copt(1)  = copt(1)/copt(2)
            avg2     = 1./iter
            avg1     = 1.-avg2
            copt1    = avg1*copt1 + avg2*copt(1)
            if(nio.eq.0)write(6,1)istep,iter,rbd1dt,copt(1),copt1,'cpt1'
    1       format(2i6,1p3e14.5,2x,a4)
         endif
         ! w := r - w
         do i = 1,nt
            mg_work2(i) = r(i) - copt1*mg_work2(i)
            e       (i) = copt1*e(i)
            ecrs2   (i) = mg_work2(i)
         enddo

      else   ! Additive
         ! w := r - w
         do i = 1,nt
            mg_work2(i) = r(i)
         enddo
         time_2 = dnekclock()
      endif
 
      do l = mg_lmax-1,2,-1

c        rmax = glmax(mg_work2,nt)
c        if (nid.eq.0) write(6,*) l,nt,rmax,' rmax2'

         nt = mg_nh(l)*mg_nh(l)*mg_nhz(l)*nelv
         !          T
         ! r   :=  J w
         !  l         
         call hsmg_rstr(mg_solve_r(mg_solve_index(l,mg_fld)),mg_work2,l)

         ! w  := r
         !        l
         call copy(mg_work2,mg_solve_r(mg_solve_index(l,mg_fld)),nt)
         ! e  := M        w
         !  l     Schwarz  
         call hsmg_schwarz(
     $          mg_solve_e(mg_solve_index(l,mg_fld)),mg_work2,l)

         ! e  := W e
         !  l       l
         call hsmg_schwarz_wt(mg_solve_e(mg_solve_index(l,mg_fld)),l)

c        call exitti('quit in mg$',l)

         ! w  := r  - w
         !        l
         do i = 0,nt-1
            mg_work2(i+1) = mg_solve_r(mg_solve_index(l,mg_fld)+i)
     $         !-alpha*mg_work2(i+1)
         enddo
      enddo

      call hsmg_rstr_no_dssum(
     $   mg_solve_r(mg_solve_index(1,mg_fld)),mg_work2,1)

      nzw = ldim-1

      call hsmg_do_wt(mg_solve_r(mg_solve_index(1,mg_fld)),
     $                mg_mask(mg_mask_index(1,mg_fld)),2,2,nzw)

      !        -1
      ! e  := A   r
      !  1         1
      call hsmg_coarse_solve(mg_solve_e(mg_solve_index(1,mg_fld)),
     $                       mg_solve_r(mg_solve_index(1,mg_fld)))

      call hsmg_do_wt(mg_solve_e(mg_solve_index(1,mg_fld)),
     $                mg_mask(mg_mask_index(1,mg_fld)),2,2,nzw)
      time_3 = dnekclock()
      do l = 2,mg_lmax-1
         nt = mg_nh(l)*mg_nh(l)*mg_nhz(l)*nelv
         ! w   :=  J e
         !            l-1
         call hsmg_intp
     $      (mg_work2,mg_solve_e(mg_solve_index(l-1,mg_fld)),l-1)

         ! e   :=  e  + w
         !  l       l
         do i = 0,nt-1
            mg_solve_e(mg_solve_index(l,mg_fld)+i) =
     $        + mg_solve_e(mg_solve_index(l,mg_fld)+i) + mg_work2(i+1)
         enddo
      enddo
      l = mg_lmax
      nt = mg_nh(l)*mg_nh(l)*mg_nhz(l)*nelv
      ! w   :=  J e
      !            m-1

      call hsmg_intp(mg_work2,
     $   mg_solve_e(mg_solve_index(l-1,mg_fld)),l-1)

      if (if_hybrid.and.istep.eq.1) then
         ! ecrs := E e_c
         call cdabdtp(ecrs,mg_work2,h1,h2,h2inv,1)
         call cmult  (ecrs,rbd1dt,nt)
         copt(1)  = vlsc2(ecrs2,ecrs,nt)
         copt(2)  = vlsc2(ecrs ,ecrs,nt)
         call gop(copt,copw,'+  ', 2)
         copt(1)  = copt(1)/copt(2)
         avg2     = 1./iter
         avg1     = 1.-avg2
         copt2    = avg1*copt2 + avg2*copt(1)
         if(nio.eq.0)write(6,1)istep,iter,rbd1dt,copt(1),copt2,'cpt2'
      endif
      ! e := e + w

      do i = 1,nt
         e(i) = e(i) + copt2*mg_work2(i)
      enddo
      time_4 = dnekclock()
c     print *, 'Did an MG iteration'
c
      taaaa = taaaa + (time_1 - time_0)
      tbbbb = tbbbb + (time_2 - time_1)
      tcccc = tcccc + (time_3 - time_2)
      tdddd = tdddd + (time_4 - time_3)
      teeee = teeee + (time_4 - time_0)
c
c     A typical time breakdown:
c
c  1.3540E+01  5.4390E+01  1.1440E+01  1.2199E+00  8.0590E+01 HSMG time
c
c  ==>  54/80 = 67 % of preconditioner time is in residual evaluation!
c
      call ortho (e)

      tddsl  = tddsl + ( dnekclock()-etime1 )



      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_setup_mg_nx()
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'HSMG'
      include 'SEMHAT'
      real w(lx1+2)
      integer nf,nc,nr
      integer nx,ny,nz

      integer mgn2(10)
      save    mgn2
      data    mgn2 / 1, 2, 2, 2, 2, 3, 3, 5, 5, 5/
c     data    mgn2 / 1, 2, 3, 4, 5, 6, 7, 8, 9, 0

c     if (param(82).eq.0) param(82)=2  ! nek default
c     if (np.eq.1)        param(82)=2  ! single proc. too slow
      p82 = 2                          ! potentially variable nxc
      mg_lmax = 3
c     mg_lmax = 4
      if (lx1.eq.4) mg_lmax = 2
c     if (param(79).ne.0) mg_lmax = param(79)
      mglx1    = p82-1 !1
      mg_nx(1) = mglx1
      mg_ny(1) = mglx1
      mg_nz(1) = mglx1
      if (.not.if3d) mg_nz(1) = 0 

      mglx2 = 2*(lx2/4) + 1
      if (lx1.eq.5)  mglx2 = 3
c     if (lx1.eq.6)  mglx2 = 3
      if (lx1.le.10) mglx2 = mgn2(lx1)
      if (lx1.eq.8)  mglx2 = 4
      if (lx1.eq.8)  mglx2 = 3

c     mglx2 = min(3,mglx2)  
      

      mg_nx(2) = mglx2
      mg_ny(2) = mglx2
      mg_nz(2) = mglx2
      if (.not.if3d) mg_nz(2) = 0 

      mg_nx(3) = mglx2+1
      mg_ny(3) = mglx2+1
      mg_nz(3) = mglx2+1
      if (.not.if3d) mg_nz(3) = 0 

      mg_nx(mg_lmax) = lx1-1
      mg_ny(mg_lmax) = ly1-1
      mg_nz(mg_lmax) = lz1-1

      if (nio.eq.0) write(*,*) 'mg_nx:',(mg_nx(i),i=1,mg_lmax)
      if (nio.eq.0) write(*,*) 'mg_ny:',(mg_ny(i),i=1,mg_lmax)
      if (nio.eq.0) write(*,*) 'mg_nz:',(mg_nz(i),i=1,mg_lmax)

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_index_0 ! initialize index sets
      include 'SIZE'
      include 'HSMG'

      n = lmgn*(lmgs+1)

      call izero( mg_rstr_wt_index      , n )
      call izero( mg_mask_index         , n )
      call izero( mg_solve_index        , n )
      call izero( mg_fast_s_index       , n )
      call izero( mg_fast_d_index       , n )
      call izero( mg_schwarz_wt_index   , n )
      
      return
      end
c----------------------------------------------------------------------
      subroutine outfldn (x,n,txt10,ichk) ! writes into unit=40+ifiled
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(n,n,1,lelt)
      character*10 txt10
c
      integer idum,e
      save    idum
      data    idum /3/
      if (idum.lt.0)   return
      m = 40 + ifield                 ! unit #
c
C
      mtot = n*n*nelv
      if (n.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(m,116) txt10,k,ie,xmin,xmax,ichk,time
            write(m,117) 
            do j=n,1,-1
              if (n.eq.2) write(m,102) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.3) write(m,103) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.4) write(m,104) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.5) write(m,105) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.6) write(m,106) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.7) write(m,107) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.8) write(m,108) ((x(i,j,k,e+l),i=1,n),e=1,ne)
            enddo
         enddo
      enddo

C
  102 FORMAT(4(2f9.5,2x))
  103 FORMAT(4(3f9.5,2x))
  104 FORMAT(4(4f7.3,2x))
  105 FORMAT(5f9.5,10x,5f9.5)
  106 FORMAT(6f9.5,5x,6f9.5)
  107 FORMAT(7f8.4,5x,7f8.4)
  108 FORMAT(8f8.4,4x,8f8.4)
c
  116 FORMAT(  /,5X,'     ^              ',/,
     $    5X,'   Y |              ',/,
     $    5X,'     |              ',A10,/,
     $    5X,'     +---->         ','Plane = ',I2,'/',I2,2x,2e12.4,/,
     $    5X,'       X            ','Step  =',I9,f15.5)
  117 FORMAT(' ')
c
c     if (ichk.eq.1.and.idum.gt.0) call checkit(idum)
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldn0 (x,n,txt10,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(n,n,1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /3/
      if (idum.lt.0) return
c
C
      mtot = n*n*nelv
      if (n.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(6,116) txt10,k,ie,xmin,xmax,ichk,time
            write(6,117) 
            do j=n,1,-1
              if (n.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,n),e=1,ne)
              if (n.eq.8) write(6,108) ((x(i,j,k,e+l),i=1,n),e=1,ne)
            enddo
         enddo
      enddo

C
  102 FORMAT(4(2f9.5,2x))
  103 FORMAT(4(3f9.5,2x))
  104 FORMAT(4(4f7.3,2x))
  105 FORMAT(5f9.5,10x,5f9.5)
  106 FORMAT(6f9.5,5x,6f9.5)
  107 FORMAT(7f8.4,5x,7f8.4)
  108 FORMAT(8f8.4,4x,8f8.4)
c
  116 FORMAT(  /,5X,'     ^              ',/,
     $    5X,'   Y |              ',/,
     $    5X,'     |              ',A10,/,
     $    5X,'     +---->         ','Plane = ',I2,'/',I2,2x,2e12.4,/,
     $    5X,'       X            ','Step  =',I9,f15.5)
  117 FORMAT(' ')
c
c     if (ichk.eq.1.and.idum.gt.0) call checkit(idum)
      return
      end
c-----------------------------------------------------------------------
      subroutine outflda (x,n,txt10,ichk) ! writes into unit=p130+ifiled
      INCLUDE 'SIZE'                      ! or into std. output for p130<9
      INCLUDE 'TSTEP'                     ! truncated below eps=p131
      INCLUDE 'INPUT'                     ! param(130)
      real x(1)
      character*10 txt10                  ! note: n is not used
c     parameter (eps=1.e-7)
C
      p130 = param(130)
      eps  = param(131)
      if (p130.le.0)    return
      m    = 6
      if (p130.gt.9)  m = p130 + ifield

      ntot = lx1*ly1*lz1*nelfld(ifield)

      xmin = glmin(x,ntot)
      xmax = glmax(x,ntot)
      xavg = glsum(x,ntot)/ntot

      if (abs(xavg).lt.eps) xavg = 0.     ! truncation

      if (nid.eq.0) write(m,10) txt10,ichk,ntot,xavg,xmin,xmax

   10 format(3X,a10,2i8,' pts, avg,min,max = ',1p3g14.6)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldan(x,n,txt10,ichk) ! writes x(1:n) into unit=p130+ifiled
      INCLUDE 'SIZE'                      ! or into std. output for 0<p130<9
      INCLUDE 'TSTEP'                     ! truncated below eps=p131
      INCLUDE 'INPUT'
      real x(1)
      character*10 txt10
c     parameter (eps=1.e-7)
C
      p130 = param(130)
      eps  = param(131)
      if (p130.le.0)    return
      m    = 6
      if (p130.gt.9)  m = p130 + ifield

      ntot = n

      xmin = glmin(x,ntot)
      xmax = glmax(x,ntot)
      xavg = glsum(x,ntot)/ntot

      if (abs(xavg).lt.eps) xavg = 0.     ! truncation

      if (nid.eq.0) write(m,10) txt10,ichk,ntot,xavg,xmin,xmax

   10 format(3X,a10,2i8,' pts, avg,min,max = ',1p3g11.3)
c  10 format(3X,a10,2i8,' pts, avg,min,max = ',1p3g14.6)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_solve(z,rhs,if_hybrid)  !  Solve preconditioner: Mz=rhs
      real z(1),rhs(1)

c     Assumes that preprocessing has been completed via h1mg_setup()


      include 'SIZE'
      include 'HSMG'       ! Same array space as HSMG
      include 'GEOM'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
      include 'CTIMER'
      include 'PARALLEL'
      
      common /scrhi/ h2inv (lx1,ly1,lz1,lelv)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv),
     $               h2    (lx1,ly1,lz1,lelv)
      parameter (lt=lx1*ly1*lz1*lelt)
      common /scrmg/ e(2*lt),w(lt),r(lt)
      integer p_msk,p_b
      logical if_hybrid

c     if_hybrid = .true.    ! Control this from gmres, according
c     if_hybrid = .false.   ! to convergence efficiency

      nel   = nelfld(ifield)

      op    =  1.                                     ! Coefficients for h1mg_ax
      om    = -1.
      sigma =  1.
      if (if_hybrid) sigma = 2./3.

      l     = mg_h1_lmax
      n     = mg_h1_n(l,mg_fld)
      is    = 1                                       ! solve index

      call h1mg_schwarz(z,rhs,sigma,l)                ! z := sigma W M       rhs
                                                      !               Schwarz
      call copy(r,rhs,n)                              ! r  := rhs
      if (if_hybrid) call h1mg_axm(r,z,op,om,l,w)     ! r  := rhs - A z
                                                      !  l

      do l = mg_h1_lmax-1,2,-1                        ! DOWNWARD Leg of V-cycle
         is = is + n
         n  = mg_h1_n(l,mg_fld)
                                                      !          T
         call h1mg_rstr(r,l,.true.)                   ! r   :=  J r
                                                      !  l         l+1
!        OVERLAPPING Schwarz exchange and solve:
         call h1mg_schwarz(e(is),r,sigma,l)           ! e := sigma W M       r
                                                      !  l            Schwarz l

         if(if_hybrid)call h1mg_axm(r,e(is),op,om,l,w)! r  := r - A e
                                                      !  l           l
      enddo
      is = is+n
                                                      !         T
      call h1mg_rstr(r,1,.false.)                     ! r  :=  J  r
                                                      !  l         l+1
      p_msk = p_mg_msk(l,mg_fld)
      call h1mg_mask(r,mg_imask(p_msk),nel)           !        -1
      call hsmg_coarse_solve ( e(is) , r )            ! e  := A   r
      call h1mg_mask(e(is),mg_imask(p_msk),nel)       !  1     1   1

c     nx = mg_nh(1)
c     call outnxfld (e(is),nx,nelv,'ecrsb4',is)
c     call h1mg_mask(e(is),mg_imask(p_msk),nel)       !  1     1   1
c     call outnxfld (e(is),nx,nelv,'ecrsaf',is)
c     call exitt

      do l = 2,mg_h1_lmax-1                           ! UNWIND.  No smoothing.
         im = is
         is = is - n
         n  = mg_h1_n(l,mg_fld)
         call hsmg_intp (w,e(im),l-1)                 ! w   :=  J e
         i1=is-1                                      !            l-1
         do i=1,n
            e(i1+i) = e(i1+i) + w(i)                  ! e   :=  e  + w
         enddo                                        !  l       l
      enddo

      l  = mg_h1_lmax
      n  = mg_h1_n(l,mg_fld)
      im = is  ! solve index
      call hsmg_intp(w,e(im),l-1)                     ! w   :=  J e
      do i = 1,n                                      !            l-1
         z(i) = z(i) + w(i)                           ! z := z + w
      enddo

      call dsavg(z) ! Emergency hack --- to ensure continuous z!

      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_axm(w,p,aw,ap,l,wk)
c
c     w  := aw*w + ap*H*p, level l, with mask and dssum
c
c     Hu := div. h1 grad u + h2 u
c
c        ~= h1 A u + h2 B u
c
c     Here, we assume that pointers into g() and h1() and h2() have
c     been established
c
      include 'SIZE'
      include 'HSMG'
      include 'TSTEP'  ! nelfld

      real w(1),p(1),wk(1)

      integer p_h1,p_h2,p_g,p_b,p_msk
      logical ifh2

      p_h1  = p_mg_h1  (l,mg_fld)
      p_h2  = p_mg_h2  (l,mg_fld)
      p_g   = p_mg_g   (l,mg_fld)
      p_b   = p_mg_b   (l,mg_fld)
      p_msk = p_mg_msk (l,mg_fld)

      if (p_h1 .eq.0) call mg_set_h1  (p_h1 ,l)
      if (p_h2 .eq.0) call mg_set_h2  (p_h2 ,l)
      if (p_g  .eq.0) call mg_set_gb  (p_g,p_b,l)
      if (p_msk.eq.0) call mg_set_msk (p_msk,l)

      ifh2 = .true.
      if (p_h2.eq.0) ifh2 = .false.  ! Need a much better mech.
      ifh2 = .false.

      nx = mg_nh(l)
      ny = mg_nh(l)
      nz = mg_nhz(l)
      ng = 3*ldim-3


      call h1mg_axml (wk,p
     $               ,mg_h1(p_h1),mg_h2(p_h2),nx,ny,nz,nelfld(ifield)
     $               ,mg_g (p_g) , ng ,mg_b(p_b), mg_imask(p_msk),ifh2)


      call hsmg_dssum (wk,l)

      n = nx*ny*nz*nelfld(ifield)
      call add2sxy    (w,aw,wk,ap,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_axml
     $  (w,p,h1,h2,nx,ny,nz,nel,g,ng,b,mask,ifh2)
c
c     w  := aw*w + ap*H*p, level l, with mask and dssum
c
c     Hu := div. h1 grad u + h2 u
c
c        ~= h1 A u + h2 B u
c

      include 'SIZE'
      include 'TOTAL'
      include 'HSMG'

      real w (nx*ny*nz,nel), p (nx*ny*nz,nel)
     $   , h1(nx*ny*nz,nel), h2(nx*ny*nz,nel)
     $   , b (nx*ny*nz,nel), g (ng*nx*ny*nz,nel)
      integer mask(1)

      logical ifh2

      parameter (lxyz=lx1*ly1*lz1)
      common /ctmp0/ ur(lxyz),us(lxyz),ut(lxyz)

      integer e

      do e=1,nel

         call axe(w(1,e),p(1,e),h1(1,e),h2(1,e),g(1,e),ng,b(1,e)
     $            ,nx,ny,nz,ur,us,ut,ifh2,ifrzer(e),e)
   
         im = mask(e)
         call mg_mask_e(w,mask(im)) ! Zero out Dirichlet conditions

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_mask(w,mask,nel)
      include 'SIZE'

      real    w   (1)
      integer mask(1)        ! Pointer to Dirichlet BCs
      integer e
      
      do e=1,nel
         im = mask(e)
         call mg_mask_e(w,mask(im)) ! Zero out Dirichlet conditions
      enddo

      return
      end
c----------------------------------------------------------------------
      subroutine mg_mask_e(w,mask) ! Zero out Dirichlet conditions
      include 'SIZE'
      real w(1)
      integer mask(0:1)

      n=mask(0)
      do i=1,n
c        write(6,*) i,mask(i),n,' MG_MASK'
         w(mask(i)) = 0.
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine axe
     $     (w,p,h1,h2,g,ng,b,nx,ny,nz,ur,us,ut,ifh2,ifrz,e)

      include 'SIZE'
      include 'INPUT'   ! if3d
      logical ifh2,ifrz

      real w (nx*ny*nz), p (nx*ny*nz)
     $   , h1(nx*ny*nz), h2(nx*ny*nz)
     $   , b (nx*ny*nz), g (ng,nx*ny*nz)
     $   , ur(nx*ny*nz), us(nx*ny*nz), ut(nx*ny*nz)
      integer e

      nxyz = nx*ny*nz

      call gradl_rst(ur,us,ut,p,nx,if3d)
c     if (e.eq.1) then
c        call outmat(p ,nx,ny,'ur A p',e)
c        call outmat(ur,nx,ny,'ur A r',e)
c        call outmat(us,nx,ny,'ur A s',e)
c     endif

      if (if3d) then
         do i=1,nxyz
            wr = g(1,i)*ur(i) + g(4,i)*us(i) + g(5,i)*ut(i)
            ws = g(4,i)*ur(i) + g(2,i)*us(i) + g(6,i)*ut(i)
            wt = g(5,i)*ur(i) + g(6,i)*us(i) + g(3,i)*ut(i)
            ur(i) = wr*h1(i)
            us(i) = ws*h1(i)
            ut(i) = wt*h1(i)
         enddo
      elseif (ifaxis) then
         call exitti('Blame Paul for no gradl_rst support yet$',nx)
      else
         do i=1,nxyz
            wr = g(1,i)*ur(i) + g(3,i)*us(i)
            ws = g(3,i)*ur(i) + g(2,i)*us(i)
c           write(6,3) i,ur(i),wr,g(1,i)/b(i),b(i)
c 3         format(i5,1p4e12.4,' wrws')
            ur(i) = wr*h1(i)
            us(i) = ws*h1(i)
         enddo
      endif

      call gradl_rst_t(w,ur,us,ut,nx,if3d)

c     if (e.eq.1) then
c        call outmat(w ,nx,ny,'ur B w',e)
c        call outmat(ur,nx,ny,'ur B r',e)
c        call outmat(us,nx,ny,'ur B s',e)
c     endif

      if (ifh2) then
        do i=1,nxyz
          w(i)=w(i)+h2(i)*b(i)*p(i)
        enddo
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_tnsr1(v,nv,nu,A,At)
c
c     v = [A (x) A] u      or
c     v = [A (x) A (x) A] u 
c
      integer nv,nu
      real v(1),A(1),At(1)
      include 'SIZE'
      include 'INPUT'
      if (.not. if3d) then
         call hsmg_tnsr1_2d(v,nv,nu,A,At)
      else
         call hsmg_tnsr1_3d(v,nv,nu,A,At,At)
      endif
      return
      end
c-------------------------------------------------------T--------------
      subroutine hsmg_tnsr1_2d(v,nv,nu,A,Bt) ! u = A u B
      integer nv,nu
      real v(1),A(1),Bt(1)
      include 'SIZE'
      common /hsmgw/ work(lx1*lx1)
      integer e

      nv2 = nv*nv
      nu2 = nu*nu

      if (nv.le.nu) then
         iv=1
         iu=1
         do e=1,nelv
            call mxm(A,nv,v(iu),nu,work,nu)
            call mxm(work,nv,Bt,nu,v(iv),nv)
            iv = iv + nv2
            iu = iu + nu2
         enddo
      else
         do e=nelv,1,-1
            iu=1+nu2*(e-1)
            iv=1+nv2*(e-1)
            call mxm(A,nv,v(iu),nu,work,nu)
            call mxm(work,nv,Bt,nu,v(iv),nv)
         enddo
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine hsmg_tnsr1_3d(v,nv,nu,A,Bt,Ct) ! v = [C (x) B (x) A] u
      integer nv,nu
      real v(1),A(1),Bt(1),Ct(1)
      include 'SIZE'
      parameter (lwk=(lx1+2)*(ly1+2)*(lz1+2))
      common /hsmgw/ work(0:lwk-1),work2(0:lwk-1)
      integer e,e0,ee,es

      e0=1
      es=1
      ee=nelv

      if (nv.gt.nu) then
         e0=nelv
         es=-1
         ee=1
      endif

      nu3 = nu**3
      nv3 = nv**3

      do e=e0,ee,es
         iu = 1 + (e-1)*nu3
         iv = 1 + (e-1)*nv3
         call mxm(A,nv,v(iu),nu,work,nu*nu)
         do i=0,nu-1
            call mxm(work(nv*nu*i),nv,Bt,nu,work2(nv*nv*i),nv)
         enddo
         call mxm(work2,nv*nv,Ct,nu,v(iv),nv)
      enddo

      return
      end
c------------------------------------------   T  -----------------------
      subroutine h1mg_rstr(r,l,ifdssum) ! r =J r,   l is coarse level
      include 'SIZE'
      include 'HSMG'
      logical ifdssum

      real r(1)
      integer l

      call hsmg_do_wt(r,mg_rstr_wt(mg_rstr_wt_index(l+1,mg_fld))
     $                     ,mg_nh(l+1),mg_nh(l+1),mg_nhz(l+1))

      call hsmg_tnsr1(r,mg_nh(l),mg_nh(l+1),mg_jht(1,l),mg_jh(1,l))

      if (ifdssum) call hsmg_dssum(r,l)

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup()
      include 'SIZE'
      include 'TOTAL'
      include 'HSMG'

      common /scrhi/ h2inv (lx1,ly1,lz1,lelt)
      common /scrvh/ h1    (lx1,ly1,lz1,lelt),
     $               h2    (lx1,ly1,lz1,lelt)

      integer p_h1,p_h2,p_g,p_b,p_msk


      param(59) = 1
      call geom_reset(1)  ! Recompute g1m1 etc. with deformed only

      n = lx1*ly1*lz1*nelt
      call rone (h1   ,n)
      call rzero(h2   ,n)
      call rzero(h2inv,n)

      call h1mg_setup_mg_nx
      call h1mg_setup_semhat ! SEM hat matrices for each level
      call hsmg_setup_intp   ! Interpolation operators
      call h1mg_setup_dssum  ! set direct stiffness summation handles
      call h1mg_setup_wtmask ! set restriction weight matrices and bc masks
      call h1mg_setup_fdm    ! set up fast diagonalization method
      call h1mg_setup_schwarz_wt(.false.)
      call hsmg_setup_solve  ! set up the solver

      l=mg_h1_lmax
      call mg_set_h1  (p_h1 ,l)
      call mg_set_h2  (p_h2 ,l)
      call mg_set_gb  (p_g,p_b,l)
      call mg_set_msk (p_msk,l)

      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_setup_mg_nx()
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'HSMG'
      include 'SEMHAT'
      include 'TSTEP'   ! nelfld
      real w(lx1+2)
      integer nf,nc,nr
      integer nx,ny,nz

      integer mgn2(10)
      save    mgn2
      data    mgn2 / 1, 2, 2, 2, 2, 3, 3, 5, 5, 5/
c     data    mgn2 / 1, 2, 3, 4, 5, 6, 7, 8, 9, 0

c     if (param(82).eq.0) param(82)=2  ! nek default
c     if (np.eq.1)        param(82)=2  ! single proc. too slow
      p82 = 2                          ! potentially variable nxc
      mg_h1_lmax = 3
c     mg_h1_lmax = 4
      if (lx1.eq.4) mg_h1_lmax = 2
c     if (param(79).ne.0) mg_h1_lmax = param(79)
      mg_lmax = mg_h1_lmax
      mglx1    = p82-1 !1
      mg_nx(1) = mglx1
      mg_ny(1) = mglx1
      mg_nz(1) = mglx1
      if (.not.if3d) mg_nz(1) = 0 

      mglx2 = 2*(lx2/4) + 1
      if (lx1.eq.5)  mglx2 = 3
c     if (lx1.eq.6)  mglx2 = 3
      if (lx1.le.10) mglx2 = mgn2(lx1)
      if (lx1.eq.8)  mglx2 = 4
      if (lx1.eq.8)  mglx2 = 3

      mglx2 = min(3,mglx2)  ! This choice seems best (9/24/12)

      mg_nx(2) = mglx2
      mg_ny(2) = mglx2
      mg_nz(2) = mglx2
      if (.not.if3d) mg_nz(2) = 0 

      mg_nx(3) = mglx2+1
      mg_ny(3) = mglx2+1
      mg_nz(3) = mglx2+1
      if (.not.if3d) mg_nz(3) = 0 

      mg_nx(mg_h1_lmax) = lx1-1
      mg_ny(mg_h1_lmax) = ly1-1
      mg_nz(mg_h1_lmax) = lz1-1

      if (nio.eq.0) write(*,*) 'h1_mg_nx:',(mg_nx(i),i=1,mg_h1_lmax)
      if (nio.eq.0) write(*,*) 'h1_mg_ny:',(mg_ny(i),i=1,mg_h1_lmax)
      if (nio.eq.0) write(*,*) 'h1_mg_nz:',(mg_nz(i),i=1,mg_h1_lmax)

      do ifld=1,ldimt1
      do l=1,mg_lmax
         mg_h1_n(l,ifld)=(mg_nx(l)+1)
     $                  *(mg_ny(l)+1)
     $                  *(mg_nz(l)+1)*nelfld(ifld)
      enddo
      enddo
      
      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_semhat ! SEM hat matrices for each level
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      include 'SEMHAT'

      do l=1,mg_h1_lmax
         n = mg_nx(l)     ! polynomial order
         call semhat(ah,bh,ch,dh,zh,dph,jph,bgl,zglhat,dgl,jgl,n,wh)
         call copy(mg_ah(1,l),ah,(n+1)*(n+1))
         call copy(mg_bh(1,l),bh,n+1)
         call copy(mg_dh(1,l),dh,(n+1)*(n+1))
         call transpose(mg_dht(1,l),n+1,dh,n+1)
         call copy(mg_zh(1,l),zh,n+1)

         mg_nh(l)=n+1
         mg_nhz(l)=mg_nz(l)+1

      enddo
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_dssum
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'HSMG'
      parameter (lxyz=(lx1+2)*(ly1+2)*(lz1+2))
      common /c_is1/ glo_num(lxyz*lelt)
      common /ivrtx/ vertex ((2**ldim)*lelt)

      integer*8 glo_num
      integer vertex
      integer nx,ny,nz
      integer l
      
      ncrnr = 2**ldim
      call get_vert


      do l=1,mg_lmax  ! set up direct stiffness summation for each level
         nx=mg_nh(l)
         ny=mg_nh(l)
         nz=mg_nhz(l)
         call setupds(mg_gsh_handle(l,mg_fld),nx,ny,nz
     $                ,nelv,nelgv,vertex,glo_num)
         nx=nx+2
         ny=ny+2
         nz=nz+2
         if(.not.if3d) nz=1
         call setupds(mg_gsh_schwarz_handle(l,mg_fld),nx,ny,nz
     $                ,nelv,nelgv,vertex,glo_num)
      enddo

      return
      end
c----------------------------------------------------------------------
      subroutine mg_set_msk(p_msk ,l0)
      include 'SIZE'
      include 'HSMG'
      include 'TSTEP'
      integer p_msk

      l                  = mg_h1_lmax
      p_mg_msk(l,mg_fld) = 0
      n                  = mg_h1_n(l,mg_fld)


      do l=mg_h1_lmax,1,-1
         nx = mg_nh  (l)
         ny = mg_nh  (l)
         nz = mg_nhz (l)

         p_msk = p_mg_msk(l,mg_fld)

         call h1mg_setup_mask
     $     (mg_imask(p_msk),nm,nx,ny,nz,nelfld(ifield),l,mg_work)

         if (l.gt.1) p_mg_msk(l-1,mg_fld)=p_mg_msk(l,mg_fld)+nm

      enddo

      p_msk = p_mg_msk(l0,mg_fld)

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_mask(mask,nm,nx,ny,nz,nel,l,w)
      include 'SIZE'
      include 'INPUT'        ! if3d

      integer mask(1)        ! Pointer to Dirichlet BCs
      integer nx,ny,nz,l
      real w(nx,ny,nz,nel)
      
      integer e,count,ptr
      integer lbr,rbr,lbs,rbs,lbt,rbt,two

      zero = 0
      nxyz = nx*ny*nz
      n    = nx*ny*nz*nel

      call rone(w,n)   ! Init everything to 1

      ierrmx = 0       ! BC verification
      two    = 2
      do e=1,nel       ! Set dirichlet nodes to zero

         call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,e,two,ierr)
c        write(6,6) e,lbr,rbr,lbs,rbs,ierr,nx
c   6    format(i5,2x,4i3,2x,i2,3x,i5,'  lbr,rbr,lbs')

         if (lbr.eq.1) call facev(w,e,4,zero,nx,ny,nz)
         if (rbr.eq.1) call facev(w,e,2,zero,nx,ny,nz)
         if (lbs.eq.1) call facev(w,e,1,zero,nx,ny,nz)
         if (rbs.eq.1) call facev(w,e,3,zero,nx,ny,nz)
         if (if3d) then
            if (lbt.eq.1) call facev(w,e,5,zero,nx,ny,nz)
            if (rbt.eq.1) call facev(w,e,6,zero,nx,ny,nz)
         endif
         ierrmx = max(ierrmx,ierr)
      enddo

      call hsmg_dsprod(w,l)    ! direct stiffness multiply

c
c     Prototypical mask layout, nel=5:
c
c    e=1 ...             10
c      1  2  3  4  5 ... 10 | 11 12 13 14 | 15 | 16 |
c     11 15 16 ...          |  3 p1 p2 p3 |  0 |  0 | ...
c                              ^
c                              |
c                              |_count for e=1
c

      nm  = 1                  ! store mask
      do e=1,nel

         mask(e) = nel+nm
         count   = 0          ! # Dirchlet points on element e
         ptr     = mask(e)

         do i=1,nxyz
            if (w(i,1,1,e).eq.0) then
               nm    = nm   +1
               count = count+1
               ptr   = ptr  +1
               mask(ptr) = i + nxyz*(e-1)   ! where I mask on element e 
            endif
         enddo


         ptr       = mask(e)
         mask(ptr) = count

         nm        = nm+1     ! bump pointer to hold next count

      enddo

      nm = nel + nm-1 ! Return total number of mask pointers/counters

      ierrmx = iglmax(ierrmx,1)
      if (ierrmx.gt.0) then
         if (ierr.gt.0) write(6,*) nid,ierr,' BC FAIL h1'
         call exitti('D INVALID BC FOUND in h1mg_setup_mask$',ierrmx)
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine mg_set_h1  (p_h1 ,l0)
      include 'SIZE'
      include 'HSMG'
      integer pf,pc

c     As a first pass, rely on the cheesy common-block interface to get h1

      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $             , h2    (lx1,ly1,lz1,lelv)
     $             , h2inv (lx1,ly1,lz1,lelv)

      integer p_h1

      l                 = mg_h1_lmax
      p_mg_h1(l,mg_fld) = 0
      n                 = mg_h1_n(l,mg_fld)

      call copy (mg_h1,h1,n)   ! Fine grid is just original h1

      nx = mg_nh(l)
      ny = mg_nh(l)
      nz = mg_nhz(l)

      do l=mg_h1_lmax-1,1,-1

         p_mg_h1(l,mg_fld) = p_mg_h1(l+1,mg_fld) + n
         n                 = mg_h1_n(l  ,mg_fld)

         pf                = p_mg_h1(l+1,mg_fld)
         pc                = p_mg_h1(l  ,mg_fld)

         call hsmg_intp_fc (mg_h1(pc),mg_h1(pf),l)

      enddo

      p_h1 = p_mg_h1(l0,mg_fld)

      return
      end
c-----------------------------------------------------------------------
      subroutine mg_set_h2  (p_h2 ,l0)
      include 'SIZE'
      include 'HSMG'

c     As a first pass, rely on the cheesy common-block interface to get h2

      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $             , h2    (lx1,ly1,lz1,lelv)
     $             , h2inv (lx1,ly1,lz1,lelv)

      integer p_h2,pf,pc

      l                 = mg_h1_lmax
      p_mg_h2(l,mg_fld) = 0
      n                 = mg_h1_n(l,mg_fld)

      call copy (mg_h2,h2,n)   ! Fine grid is just original h2

      nx = mg_nh(l)
      ny = mg_nh(l)
      nz = mg_nhz(l)

      do l=mg_h1_lmax-1,1,-1

         p_mg_h2(l,mg_fld) = p_mg_h2(l+1,mg_fld) + n
         n                 = mg_h1_n(l  ,mg_fld)

         pf                = p_mg_h2(l+1,mg_fld)
         pc                = p_mg_h2(l  ,mg_fld)

         call hsmg_intp_fc (mg_h2(pc),mg_h2(pf),l)

      enddo

      p_h2 = p_mg_h2(l0,mg_fld)

      return
      end
c-----------------------------------------------------------------------
      subroutine hsmg_intp_fc(uc,uf,l) ! l is coarse level

      include 'SIZE'
      include 'HSMG'

      real uc(1),uf(1)


      nc = mg_nh(l)
      nf = mg_nh(l+1)
      call hsmg_tnsr(uc,nc,uf,nf,mg_jhfc(1,l),mg_jhfct(1,l))

      return
      end
c-----------------------------------------------------------------------
      subroutine mg_intp_fc_e(uc,uf,nxc,nyc,nzc,nxf,nyf,nzf,e,l,w)
      include 'SIZE'
      include 'INPUT'      ! if3d
      include 'HSMG'

      real uf(nxf,nyf,nzf),uc(nxc,nyc,nzc),w(1)
      integer e

      if (if3d) then

         n1=nxf*nyf
         n2=nzf
         n3=nzc
         call mxm(uf,n1,mg_jhfct(1,l),n2,w,n3)

         lf=1           ! pointers into work array w()
         lc=1 + n1*n3
         lc0=lc

         n1=nxf
         n2=nyf
         n3=nyc

         do k=1,nzc
            call mxm(w(lf),n1,mg_jhfct(1,l),n2,w(lc),n3)
            lf = lf + n1*n2
            lc = lc + n1*n3
         enddo

         lf=lc0  ! Rewind fine pointer to start of coarse data
         n1=nxc
         n2=nxf
         n3=nyc*nzc
         call mxm(mg_jhfc(1,l),n1,w(lf),n2,uc,n3)

      else ! 2D

         n1=nxf
         n2=nyf
         n3=nyc
         call mxm(uf,n1,mg_jhfct(1,l),n2,w,n3)

         n1=nxc
         n2=nxf
         n3=nyc
         call mxm(mg_jhfc(1,l),n1,w,n2,uc,n3)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mg_intp_gfc_e(gc,gf,ng,nxc,nyc,nzc,nxf,nyf,nzf,e,l,w)
      include 'SIZE'
      include 'INPUT'      ! if3d
      include 'HSMG'

      real gf(ng,nxf,nyf,nzf),gc(ng,nxc,nyc,nzc),w(1)
      integer e


      if (if3d) then

         n1=ng*nxf*nyf
         n2=nzf
         n3=nzc
         call mxm(gf,n1,mg_jhfct(1,l),n2,w,n3)

         lf=1           ! pointers into work array w()
         lc=1 + n1*n3
         lc0=lc

         n1=ng*nxf
         n2=nyf
         n3=nyc

         do k=1,nzc
            call mxm(w(lf),n1,mg_jhfct(1,l),n2,w(lc),n3)
            lf = lf + n1*n2
            lc = lc + n1*n3
         enddo

         lf=lc0  ! Rewind fine pointer to start of coarse data
         n1=ng
         n2=nxf
         n3=nxc

         do k=1,nyc*nzc
            call mxm(w(lf),n1,mg_jhfct(1,l),n2,gc(1,1,k,1),n3)
            lf = lf + n1*n2
         enddo

      else ! 2D

         n1=ng*nxf
         n2=nyf
         n3=nyc
         call mxm(gf,n1,mg_jhfct(1,l),n2,w,n3)

         lf=1           ! pointers into work array w()

         n1=ng
         n2=nxf
         n3=nxc

         do k=1,nyc
            call mxm(w(lf),n1,mg_jhfct(1,l),n2,gc(1,1,k,1),n3)
            lf = lf + n1*n2
         enddo

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mg_scale_mass (b,g,wt,ng,nx,ny,nz,wk,ifinv)
      include 'SIZE'
      include 'INPUT'  ! if3d
      include 'HSMG'

      real b(1),g(ng,1),wt(1),wk(1)
      logical ifinv

      common /ctmp0/ wi(2*lx1+4)

      n = nx*ny*nz

      if (nx.le.2*lx1) then
         if (ifinv) then
            call invers2(wi,wt,nx)
         else
            call copy(wi,wt,nx)
         endif
      else
         call exitti('mg_scale_mass: wi too small$',nx)
      endif

      if (if3d) then
         l=0
         do k=1,nz
         do j=1,ny
            wjk=wi(j)*wi(k)
            do i=1,nx
               l=l+1
               wk(l) = wjk*wi(i)
            enddo
         enddo
         enddo

         do k=1,n
            b(k)   = wk(k)*b(k)
            g(1,k) = wk(k)*g(1,k)
            g(2,k) = wk(k)*g(2,k)
            g(3,k) = wk(k)*g(3,k)
            g(4,k) = wk(k)*g(4,k)
            g(5,k) = wk(k)*g(5,k)
            g(6,k) = wk(k)*g(6,k)
         enddo

      else      ! 2D
         l=0
         do j=1,ny
         do i=1,nx
            l=l+1
            wk(l) = wi(i)*wi(j)
         enddo
         enddo

         do k=1,n
            b(k)   = wk(k)*b(k)
            g(1,k) = wk(k)*g(1,k)
            g(2,k) = wk(k)*g(2,k)
            g(3,k) = wk(k)*g(3,k)
         enddo

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mg_set_gb  (p_g,p_b,l0)
      include 'SIZE'
      include 'HSMG'
      include 'MASS'   ! bm1
      include 'TSTEP'  ! nelfld

      integer p_g,p_b,e
      common /ctmp1/ w(lx1*ly1*lz1*lelt*2)

      l                 = mg_h1_lmax
      p_mg_b (l,mg_fld) = 0
      p_mg_g (l,mg_fld) = 0
      n                 = mg_h1_n(l,mg_fld)


      ng = 3*(ldim-1)  ! 3 or 6 elements to symm dxd tensor

      do l=mg_h1_lmax-1,1,-1

         p_mg_b (l,mg_fld) = p_mg_b (l+1,mg_fld) + n
         p_mg_g (l,mg_fld) = p_mg_g (l+1,mg_fld) + n*ng
         n                 = mg_h1_n(l  ,mg_fld)

      enddo

      do e=1,nelfld(ifield)
       do l=mg_h1_lmax,1,-1

         nx = mg_nh(l)
         ny = mg_nh(l)
         nz = mg_nhz(l)
         nxyz = nx*ny*nz

         p_g = p_mg_g (l,mg_fld) + ng*nx*ny*nz*(e-1)
         p_b = p_mg_b (l,mg_fld) +    nx*ny*nz*(e-1)

         if (l.eq.mg_h1_lmax) then
            call gxfer_e (mg_g(p_g) ,ng,e             ) ! Fine grid=original G
            call copy    (mg_b(p_b) ,bm1(1,1,1,e),nxyz) ! Fine grid=original B
            call mg_scale_mass                          ! Divide out Wghts
     $         (mg_b(p_b),mg_g(p_g),mg_bh(1,l),ng,nx,ny,nz,w,.true.)
         else

c           Generate G and B by interpolating their continous counterparts onto
c           the coarse grid and collocating with coarse-grid quadrature weights

            call mg_intp_gfc_e
     $            (mg_g(p_g),mg_g(l_g),ng,nx,ny,nz,nxl,nyl,nzl,e,l,w)

            call mg_intp_fc_e
     $            (mg_b(p_b),mg_b(l_b)   ,nx,ny,nz,nxl,nyl,nzl,e,l,w)

            call mg_scale_mass                         ! Reinstate weights
     $      (mg_b(l_b),mg_g(l_g),mg_bh(1,l+1),ng,nxl,nyl,nzl,w,.false.)

         endif

         l_b = p_b
         l_g = p_g

         nxl = nx
         nyl = ny
         nzl = nz

       enddo

       call mg_scale_mass                         ! Reinstate weights
     $      (mg_b(l_b),mg_g(l_g),mg_bh(1,1),ng,nxl,nyl,nzl,w,.false.)


      enddo

      p_b  = p_mg_b (l0,mg_fld)
      p_g  = p_mg_g (l0,mg_fld)

      return
      end
c-----------------------------------------------------------------------
      subroutine gxfer_e (g,ng,e) 
      include 'SIZE'
      include 'TOTAL'

      real g(ng,1)
      integer e

      nxyz = lx1*ly1*lz1

c     ifdfrm(e) = .true.  ! TOO LATE

      if (if3d) then
         do i=1,nxyz
            g(1,i) = g1m1(i,1,1,e)
            g(2,i) = g2m1(i,1,1,e)
            g(3,i) = g3m1(i,1,1,e)
            g(4,i) = g4m1(i,1,1,e)
            g(5,i) = g5m1(i,1,1,e)
            g(6,i) = g6m1(i,1,1,e)
         enddo
      else
         do i=1,nxyz
            g(1,i) = g1m1(i,1,1,e)
            g(2,i) = g2m1(i,1,1,e)
            g(3,i) = g4m1(i,1,1,e)
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine chkr(name3,ii)
      include 'SIZE'
      include 'TOTAL'
      include 'HSMG'
      character*3 name3

      write(6,*) mg_h1_lmax,ii,' ',name3,' CHKR'

      return
      end
c-----------------------------------------------------------------------
      subroutine outgmat(a,ng,nx,ny,name6,k,e)

      integer e
      real a(ng,nx,ny)
      common /ctmp0/ w(100000)
      character*6 name6

c     do i=1,ng
      do i=1,1
         sum = 0.
         do ii=1,nx*ny
            w(ii)=a(i,ii,1)
            sum = sum + a(i,ii,1)
         enddo

         write(6,1) name6,i,k,e,nx,ny,ng,sum
    1    format(a6,6i5,f12.5,'  outgmat')

         call outmatz(w,nx,ny,name6,i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine outmatz(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6

      sum=0.
      sua=0.
      do i=1,m*n
         sum=sum+    a(i,1)
         sua=sua+abs(a(i,1))
      enddo
      sum=sum/(m*n)
      sua=sua/(m*n)

      write(6,*) 
      write(6,1) ie,name6,m,n,sum,sua
    1 format(i8,' matrix: ',a6,2i5,1p2e12.4)

      n12 = min(m,12)
      do j=m,1,-1
         write(6,6) ie,name6,(a(i,j),i=1,n12)
      enddo
    6 format(i3,1x,a6,12f9.5)
c     write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine h1mg_setup_schwarz_wt_2(wt,ie,n,work,ifsqrt)
      include 'SIZE'
      real wt(1),work(1)
      logical ifsqrt

      if (ldim.eq.2) call h1mg_setup_schwarz_wt2d_2(wt,ie,n,work,ifsqrt)
      if (ldim.eq.3) call h1mg_setup_schwarz_wt3d_2(wt,ie,n,work,ifsqrt)

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_schwarz_wt2d_2(wt,ie,n,work,ifsqrt)
      include 'SIZE'
      logical ifsqrt
      integer n
      real wt(n,4,2,nelv)
      real work(n,n)
      
      integer ie,i,j
      do j=1,n
         wt(j,1,1,ie)=1.0/work(1,j)
         wt(j,2,1,ie)=1.0/work(2,j)
         wt(j,3,1,ie)=1.0/work(n-1,j)
         wt(j,4,1,ie)=1.0/work(n,j)
      enddo
      do i=1,n
         wt(i,1,2,ie)=1.0/work(i,1)
         wt(i,2,2,ie)=1.0/work(i,2)
         wt(i,3,2,ie)=1.0/work(i,n-1)
         wt(i,4,2,ie)=1.0/work(i,n)
      enddo
      if(ifsqrt) then
         do ii=1,2
         do j=1,4
         do i=1,n
            wt(i,j,ii,ie)=sqrt(wt(i,j,ii,ie))
         enddo
         enddo
         enddo
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_schwarz_wt3d_2(wt,ie,n,work,ifsqrt)
      include 'SIZE'
      logical ifsqrt
      integer n
      real wt(n,n,4,3,nelv)
      real work(n,n,n)
      
      integer ie,i,j,k
      integer lbr,rbr,lbs,rbs,lbt,rbt

      ierr = 0
      do k=1,n
      do j=1,n
         wt(j,k,1,1,ie)=1.0/work(1,j,k)
         wt(j,k,2,1,ie)=1.0/work(2,j,k)
         wt(j,k,3,1,ie)=1.0/work(n-1,j,k)
         wt(j,k,4,1,ie)=1.0/work(n,j,k)
      enddo
      enddo
      do k=1,n
      do i=1,n
         wt(i,k,1,2,ie)=1.0/work(i,1,k)
         wt(i,k,2,2,ie)=1.0/work(i,2,k)
         wt(i,k,3,2,ie)=1.0/work(i,n-1,k)
         wt(i,k,4,2,ie)=1.0/work(i,n,k)
      enddo
      enddo
      do j=1,n
      do i=1,n
         wt(i,j,1,3,ie)=1.0/work(i,j,1)
         wt(i,j,2,3,ie)=1.0/work(i,j,2)
         wt(i,j,3,3,ie)=1.0/work(i,j,n-1)
         wt(i,j,4,3,ie)=1.0/work(i,j,n)
      enddo
      enddo
      if(ifsqrt) then
         do ii=1,3
         do k=1,4
         do j=1,4
         do i=1,n
            wt(i,j,k,ii,ie)=sqrt(wt(i,j,k,ii,ie))
         enddo
         enddo
         enddo
         enddo
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_setup_schwarz_wt_1(wt,l,ifsqrt)
      include 'SIZE'
      include 'INPUT'  ! if3d
      include 'TSTEP'  ! ifield
      include 'HSMG'

      real wt(1),work(1)
      logical ifsqrt

      integer enx,eny,enz,pm

      zero =  0
      one  =  1
      onem = -1

      n  = mg_h1_n (l,mg_fld)
      pm = p_mg_msk(l,mg_fld)

      enx=mg_nh(l)+2
      eny=mg_nh(l)+2
      enz=mg_nh(l)+2
      if(.not.if3d) enz=1
      ns = enx*eny*enz*nelfld(ifield)
      i  = ns+1

      call rone(mg_work(i),ns)
 
c     Sum overlap region (border excluded)
      call hsmg_extrude(mg_work,0,zero,mg_work(i),0,one ,enx,eny,enz)
      call hsmg_schwarz_dssum(mg_work(i),l)
      call hsmg_extrude(mg_work(i),0,one ,mg_work,0,onem,enx,eny,enz)
      call hsmg_extrude(mg_work(i),2,one,mg_work(i),0,one,enx,eny,enz)

      if(.not.if3d) then ! Go back to regular size array
         call hsmg_schwarz_toreg2d(mg_work,mg_work(i),mg_nh(l))
      else
         call hsmg_schwarz_toreg3d(mg_work,mg_work(i),mg_nh(l))
      endif

      call hsmg_dssum(mg_work,l)                           ! sum border nodes


      nx = mg_nh(l)
      ny = mg_nh(l)
      nz = mg_nh(l)
      if (.not.if3d) nz=1
      nxyz = nx*ny*nz
      k    = 1
      do ie=1,nelfld(ifield)
c        call outmat(mg_work(k),nx,ny,'NEW WT',ie)
         call h1mg_setup_schwarz_wt_2(wt,ie,nx,mg_work(k),ifsqrt)
         k = k+nxyz
      enddo
c     stop

      return
      end
c----------------------------------------------------------------------
