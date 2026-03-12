c-----------------------------------------------------------------------
      subroutine uzawa_gmres(res,h1,h2,h2inv,intype,iter)

c     Solve the pressure equation by right-preconditioned 
c     GMRES iteration.
c     intype =  0  (steady)
c     intype =  1  (explicit)
c     intype = -1  (implicit)

      include 'SIZE'
      include 'TOTAL'
      include 'GMRES'
      common  /ctolpr/ divex
      common  /cprint/ ifprint
      logical          ifprint
      real             res  (lx2*ly2*lz2*lelv)
      real             h1   (lx1,ly1,lz1,lelv)
      real             h2   (lx1,ly1,lz1,lelv)
      real             h2inv(lx1,ly1,lz1,lelv)

      common /scrmg/    wp (lx2,ly2,lz2,lelv)

      common /ctmp0/   wk1(lgmres),wk2(lgmres)
      common /cgmres1/ y(lgmres)

      real alpha, l, temp
      integer j,m
c
      logical iflag
      save    iflag
      data    iflag /.false./
      real    norm_fac
      save    norm_fac
c
      real*8 etime1,dnekclock
c
      if(.not.iflag) then
         iflag=.true.
         call uzawa_gmres_split0(ml_gmres,mu_gmres,bm2,bm2inv,
     $                           lx2*ly2*lz2*nelv)
         norm_fac = 1./sqrt(volvm2)
      endif
c
      etime1 = dnekclock()
      etime_p = 0.
      divex = 0.
      iter  = 0
      m = lgmres
c
      call chktcg2(tolps,res,iconv)
      if (param(21).gt.0.and.tolps.gt.abs(param(21))) 
     $   tolps = abs(param(21))
c     if (param(21).lt.0) tolps = abs(param(21))
      if (istep.eq.0) tolps = 1.e-4
      tolpss = tolps
c
      ntot2  = lx2*ly2*lz2*nelv
c
      iconv = 0
      call rzero(x_gmres,ntot2)

      do while(iconv.eq.0.and.iter.lt.100)

         if(iter.eq.0) then
                                                        !      -1
            call col3(r_gmres,ml_gmres,res,ntot2)       ! r = L  res
c           call copy(r_gmres,res,ntot2)
         else
            !update residual
            call copy(r_gmres,res,ntot2)                      ! r = res
            call cdabdtp(w_gmres,x_gmres,h1,h2,h2inv,intype)  ! w = A x
            call add2s2(r_gmres,w_gmres,-1.,ntot2)            ! r = r - w
                                                              !      -1
            call col2(r_gmres,ml_gmres,ntot2)                 ! r = L   r
         endif
                                                            !            ______
         gamma_gmres(1) = sqrt(glsc2(r_gmres,r_gmres,ntot2))! gamma  = \/ (r,r) 
                                                            !      1
         if(iter.eq.0) then
            div0 = gamma_gmres(1)*norm_fac
            if (param(21).lt.0) tolpss=abs(param(21))*div0
         endif

         !check for lucky convergence
         rnorm = 0.
         if(gamma_gmres(1) .eq. 0.) goto 9000
         temp = 1./gamma_gmres(1)
         call cmult2(v_gmres(1,1),r_gmres,temp,ntot2)! v  = r / gamma
                                                     !  1            1
         do j=1,m
            iter = iter+1
                                                           !       -1
            call col3(w_gmres,mu_gmres,v_gmres(1,j),ntot2) ! w  = U   v
                                                           !           j
            
            etime2 = dnekclock()
            if(param(43).eq.1) then
               call uzprec(z_gmres(1,j),w_gmres,h1,h2,intype,wp)
            else                                        !       -1
               call hsmg_solve(z_gmres(1,j),w_gmres)    ! z  = M   w
c              call copy(z_gmres(1,j),w_gmres,ntot2)    ! z  = M   w
            endif     
            etime_p = etime_p + dnekclock()-etime2
     
            call cdabdtp(w_gmres,z_gmres(1,j),    ! w = A z
     $                   h1,h2,h2inv,intype)      !        j
     
                                                  !      -1
            call col2(w_gmres,ml_gmres,ntot2)     ! w = L   w

c           !modified Gram-Schmidt
c           do i=1,j
c              h_gmres(i,j)=glsc2(w_gmres,v_gmres(1,i),ntot2) ! h    = (w,v )
c                                                             !  i,j       i
c              call add2s2(w_gmres,v_gmres(1,i),-h_gmres(i,j),ntot2) ! w = w - h    v
c           enddo                                                    !          i,j  i


c           2-PASS GS, 1st pass:

            do i=1,j
               h_gmres(i,j)=vlsc2(w_gmres,v_gmres(1,i),ntot2) ! h    = (w,v )
            enddo                                             !  i,j       i

            call gop(h_gmres(1,j),wk1,'+  ',j)          ! sum over P procs

            do i=1,j
               call add2s2(w_gmres,v_gmres(1,i),-h_gmres(i,j),ntot2) ! w = w - h    v
            enddo                                                    !          i,j  i


c           2-PASS GS, 2nd pass:
c
c           do i=1,j
c              wk1(i)=vlsc2(w,v_gmres(1,i),ntot2) ! h    = (w,v )
c           enddo                                 !  i,j       i
c                                                 !
c           call gop(wk1,wk2,'+  ',j)             ! sum over P procs
c
c           do i=1,j
c              call add2s2(w,v_gmres(1,i),-wk1(i),ntot2) ! w = w - h    v
c              h(i,j) = h(i,j) + wk1(i)                  !          i,j  i
c           enddo


            !apply Givens rotations to new column
            do i=1,j-1
               temp = h_gmres(i,j)                   
               h_gmres(i  ,j)=  c_gmres(i)*temp 
     $                        + s_gmres(i)*h_gmres(i+1,j)  
               h_gmres(i+1,j)= -s_gmres(i)*temp 
     $                        + c_gmres(i)*h_gmres(i+1,j)
            enddo
                                                              !            ______
            alpha = sqrt(glsc2(w_gmres,w_gmres,ntot2))        ! alpha =  \/ (w,w)
            rnorm = 0.
            if(alpha.eq.0.) goto 900  !converged
            l = sqrt(h_gmres(j,j)*h_gmres(j,j)+alpha*alpha)
            temp = 1./l
            c_gmres(j) = h_gmres(j,j) * temp
            s_gmres(j) = alpha  * temp
            h_gmres(j,j) = l
            gamma_gmres(j+1) = -s_gmres(j) * gamma_gmres(j)
            gamma_gmres(j)   =  c_gmres(j) * gamma_gmres(j)

c            call outmat(h,m,j,' h    ',j)
            
            rnorm = abs(gamma_gmres(j+1))*norm_fac
            ratio = rnorm/div0
            if (ifprint.and.nio.eq.0) 
     $         write (6,66) iter,tolpss,rnorm,div0,ratio,istep
   66       format(i5,1p4e12.5,i8,' Divergence')

#ifndef FIXITER
            if (rnorm .lt. tolpss) goto 900  !converged
#else
            if (iter.gt.param(151)-1) goto 900
#endif
            if (j.eq.m) goto 1000 !not converged, restart

            temp = 1./alpha
            call cmult2(v_gmres(1,j+1),w_gmres,temp,ntot2) ! v    = w / alpha
                                                           !  j+1            
         enddo
  900    iconv = 1
 1000    continue
         !back substitution
         !     -1
         !c = H   gamma
         do k=j,1,-1
            temp = gamma_gmres(k)
            do i=j,k+1,-1
               temp = temp - h_gmres(k,i)*c_gmres(i)
            enddo
            c_gmres(k) = temp/h_gmres(k,k)
         enddo
         !sum up Arnoldi vectors
         do i=1,j
            call add2s2(x_gmres,z_gmres(1,i),c_gmres(i),ntot2) 
                       ! x = x + c  z
                       !          i  i
         enddo
c        if(iconv.eq.1) call dbg_write(x,lx2,ly2,lz2,nelv,'esol',3)
      enddo
 9000 continue
c
      divex = rnorm
c     iter = iter - 1
c
c     DIAGNOSTICS
c      call copy   (w,x,ntot2)
       call ortho  (w_gmres) ! Orthogonalize wrt null space, if present
c      call copy(r,res,ntot2) !r = res
c      call cdabdtp(r,w,h1,h2,h2inv,intype)  ! r = A w
c      do i=1,ntot2
c         r(i) = res(i) - r(i)               ! r = res - r
c      enddo
c      call uzawa_gmres_temp(r,bm2inv,ntot2)
c                                               !            ______
c      gamma(1) = sqrt(glsc2(r,r,ntot2)/volvm2) ! gamma  = \/ (r,r) 
c                                               !      1
c      print *, 'GMRES end resid:',gamma(1)
c     END DIAGNOSTICS
      call copy(res,x_gmres,ntot2)

      call ortho (res)  ! Orthogonalize wrt null space, if present

      etime1 = dnekclock()-etime1
      if (nio.eq.0) write(6,9999) istep,'  U-PRES gmres  ', 
     &                            iter,divex,div0,tolpss,etime_p,etime1
c     call flush_hack
 9999 format(i11,a,I6,1p5e13.4)

      return
      end

c-----------------------------------------------------------------------

      subroutine uzawa_gmres_split0(l,u,b,binv,n)
      integer n
      real l(n),u(n),b(n),binv(n)
      integer i
      do i=1,n
         l(i)=sqrt(binv(i))
         u(i)=sqrt(b(i))
         if(abs(u(i)*l(i)-1.0).gt.1e-13) print *, i, u(i)*l(i)
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine uzawa_gmres_split(l,u,b,binv,n)
      integer n
      real l(n),u(n),b(n),binv(n)
      integer i
      do i=1,n
c        l(i)=sqrt(binv(i))
c        u(i)=sqrt(b(i))

c        u(i)=sqrt(b(i))
c        l(i)=1./u(i)

c        l(i)=sqrt(binv(i))
         l(i)=1.
         u(i)=1./l(i)


c        if(abs(u(i)*l(i)-1.0).gt.1e-13)write(6,*) i,u(i)*l(i),' gmr_sp'
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine uzawa_gmres_temp(a,b,n)
      integer n
      real a(n),b(n)
      integer i
      do i=1,n
         a(i)=sqrt(b(i))*a(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine ax(w,x,h1,h2,n)
      include 'SIZE'
      include 'TOTAL'

c
c     w = A*x for pressure iteration
c

      integer n
      real w(n),x(n),h1(n),h2(n)

      imsh = 1
      isd  = 1
      call axhelm (w,x,h1,h2,imsh,isd)
      call dssum  (w,lx1,ly1,lz1)
      call col2   (w,pmask,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine hmh_gmres(res,h1,h2,wt,iter)

c     Solve the Helmholtz equation by right-preconditioned 
c     GMRES iteration.

     
      include 'SIZE'
      include 'TOTAL'
      include 'FDMH1'
      include 'GMRES'
      common  /ctolpr/ divex
      common  /cprint/ ifprint
      logical          ifprint
      real             res  (lx1*ly1*lz1*lelv)
      real             h1   (lx1,ly1,lz1,lelv)
      real             h2   (lx1,ly1,lz1,lelv)
      real             wt   (lx1,ly1,lz1,lelv)

      common /scrcg/ d(lx1*ly1*lz1*lelv),wk(lx1*ly1*lz1*lelv)

      common /cgmres1/ y(lgmres)
      common /ctmp0/   wk1(lgmres),wk2(lgmres)
      real alpha, l, temp
      integer outer

      logical iflag,if_hyb
      save    iflag,if_hyb
c     data    iflag,if_hyb  /.false. , .true. /
      data    iflag,if_hyb  /.false. , .false. /
      real    norm_fac
      save    norm_fac

      real*8 etime1,dnekclock


      n = lx1*ly1*lz1*nelv

      etime1 = dnekclock()
      etime_p = 0.
      divex = 0.
      maxit = iter
      iter  = 0
      m     = lgmres

      if(.not.iflag) then
         iflag=.true.
         call uzawa_gmres_split(ml_gmres,mu_gmres,bm1,binvm1,
     $                          lx1*ly1*lz1*nelv)
         norm_fac = 1./sqrt(volvm1)
      endif

      if (param(100).ne.2) call set_fdm_prec_h1b(d,h1,h2,nelv)

      call chktcg1(tolps,res,h1,h2,pmask,vmult,1,1)
      if (param(21).gt.0.and.tolps.gt.abs(param(21))) 
     $   tolps = abs(param(21))
      if (istep.eq.0) tolps = 1.e-4
      tolpss = tolps
c
      iconv = 0
      call rzero(x_gmres,n)

      outer = 0
      do while (iconv.eq.0)

         outer = outer+1

         if(iter.eq.0) then                   !      -1
            call col3(r_gmres,ml_gmres,res,n) ! r = L  res
c           call copy(r,res,n)
         else
            !update residual
            call copy  (r_gmres,res,n)           ! r = res
            call ax    (w_gmres,x_gmres,h1,h2,n) ! w = A x
            call add2s2(r_gmres,w_gmres,-1.,n)   ! r = r - w
                                                 !      -1
            call col2(r_gmres,ml_gmres,n)        ! r = L   r
         endif
                                                            !            ______
         gamma_gmres(1) = sqrt(glsc3(r_gmres,r_gmres,wt,n)) ! gamma  = \/ (r,r) 
                                                            !      1
         if(iter.eq.0) then
            div0 = gamma_gmres(1)*norm_fac
            if (param(21).lt.0) tolpss=abs(param(21))*div0
         endif

         !check for lucky convergence
         rnorm = 0.
         if(gamma_gmres(1) .eq. 0.) goto 9000
         temp = 1./gamma_gmres(1)
         call cmult2(v_gmres(1,1),r_gmres,temp,n) ! v  = r / gamma
                                                  !  1            1
         do j=1,m
            iter = iter+1
                                                       !       -1
            call col3(w_gmres,mu_gmres,v_gmres(1,j),n) ! w  = U   v
                                                       !           j

c . . . . . Overlapping Schwarz + coarse-grid . . . . . . .

            etime2 = dnekclock()

c           if (outer.gt.2) if_hyb = .true.       ! Slow outer convergence
            if (ifmgrid) then
               if (param(40).ge.0 .and. param(40).le.2) then
                  call h1mg_solve(z_gmres(1,j),w_gmres,if_hyb) ! z  = M   w
               else if (param(40).eq.3) then
                  call fem_amg_solve(z_gmres(1,j),w_gmres)
               endif
            else                                            !  j
               kfldfdm = ldim+1
               if (param(100).eq.2) then
                   call h1_overlap_2 (z_gmres(1,j),w_gmres,pmask)
               else
                   call fdm_h1
     $               (z_gmres(1,j),w_gmres,d,pmask,vmult,nelv,
     $                ktype(1,1,kfldfdm),wk)
               endif
               call crs_solve_h1 (wk,w_gmres)        ! z  = M   w
               call add2         (z_gmres(1,j),wk,n) !  j        
            endif


            call ortho        (z_gmres(1,j)) ! Orthogonalize wrt null space, if present
            etime_p = etime_p + dnekclock()-etime2
c . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

     
            call ax  (w_gmres,z_gmres(1,j),h1,h2,n) ! w = A z
                                                    !        j
     
                                                    !      -1
            call col2(w_gmres,ml_gmres,n)           ! w = L   w

c           !modified Gram-Schmidt

c           do i=1,j
c              h_gmres(i,j)=glsc3(w_gmres,v_gmres(1,i),wt,n) ! h    = (w,v )
c                                                            !  i,j       i

c              call add2s2(w_gmres,v_gmres(1,i),-h_gmres(i,j),n) ! w = w - h    v
c           enddo                                                !          i,j  i

c           2-PASS GS, 1st pass:

            do i=1,j
               h_gmres(i,j)=vlsc3(w_gmres,v_gmres(1,i),wt,n) ! h    = (w,v )
            enddo                                            !  i,j       i

            call gop(h_gmres(1,j),wk1,'+  ',j)          ! sum over P procs

            do i=1,j
               call add2s2(w_gmres,v_gmres(1,i),-h_gmres(i,j),n) ! w = w - h    v
            enddo                                                !          i,j  i


c           2-PASS GS, 2nd pass:
c
c           do i=1,j
c              wk1(i)=vlsc3(w_gmres,v_gmres(1,i),wt,n) ! h    = (w,v )
c           enddo                                      !  i,j       i
c                                                      !
c           call gop(wk1,wk2,'+  ',j)                  ! sum over P procs
c
c           do i=1,j
c              call add2s2(w_gmres,v_gmres(1,i),-wk1(i),n)    ! w = w - h    v
c              h_gmres(i,j) = h_gmres(i,j) + wk1(i)           !          i,j  i
c           enddo

            !apply Givens rotations to new column
            do i=1,j-1
               temp = h_gmres(i,j)                   
               h_gmres(i  ,j)=  c_gmres(i)*temp 
     $                        + s_gmres(i)*h_gmres(i+1,j)  
               h_gmres(i+1,j)= -s_gmres(i)*temp 
     $                        + c_gmres(i)*h_gmres(i+1,j)
            enddo
                                                      !            ______
            alpha = sqrt(glsc3(w_gmres,w_gmres,wt,n)) ! alpha =  \/ (w,w)
            rnorm = 0.
            if(alpha.eq.0.) goto 900  !converged
            l = sqrt(h_gmres(j,j)*h_gmres(j,j)+alpha*alpha)
            temp = 1./l
            c_gmres(j) = h_gmres(j,j) * temp
            s_gmres(j) = alpha  * temp
            h_gmres(j,j) = l
            gamma_gmres(j+1) = -s_gmres(j) * gamma_gmres(j)
            gamma_gmres(j)   =  c_gmres(j) * gamma_gmres(j)

            rnorm = abs(gamma_gmres(j+1))*norm_fac
            ratio = rnorm/div0
            if (ifprint.and.nio.eq.0) 
     $         write (6,66) iter,tolpss,rnorm,div0,ratio,istep
   66       format(i5,1p4e12.5,i8,' Divergence')

#ifndef FIXITER
            if (iter+1.gt.maxit) goto 900
            if (rnorm .lt. tolpss) goto 900  !converged
#else
            if (iter.gt.param(151)-1) goto 900
#endif
            if (j.eq.m) goto 1000 !not converged, restart

            temp = 1./alpha
            call cmult2(v_gmres(1,j+1),w_gmres,temp,n) ! v    = w / alpha
                                                       !  j+1            
         enddo
  900    iconv = 1
 1000    continue
         !back substitution
         !     -1
         !c = H   gamma
         do k=j,1,-1
            temp = gamma_gmres(k)
            do i=j,k+1,-1
               temp = temp - h_gmres(k,i)*c_gmres(i)
            enddo
            c_gmres(k) = temp/h_gmres(k,k)
         enddo
         !sum up Arnoldi vectors
         do i=1,j
            call add2s2(x_gmres,z_gmres(1,i),c_gmres(i),n) ! x = x + c  z
         enddo                                             !          i  i
c        if(iconv.eq.1) call dbg_write(x,lx1,ly1,lz1,nelv,'esol',3)
      enddo
 9000 continue

      divex = rnorm
      call copy(res,x_gmres,n)

      call ortho   (res) ! Orthogonalize wrt null space, if present

      etime1 = dnekclock()-etime1
      if (nio.eq.0) write(6,9999) istep,iter,divex,div0,tolpss,etime_p,
     &                            etime1,if_hyb
c     call flush_hack
 9999 format(4x,i7,'  PRES gmres ',4x,i5,1p5e13.4,1x,l4)

      if (outer.le.2) if_hyb = .false.

      return
      end
c-----------------------------------------------------------------------
      subroutine set_overlap2
c
c     Sets up the gather scatter and the SEM operators
c
      include 'SIZE'  
      include 'TOTAL' 
 
      common /c_is1/ glo_num(lxs*lys*lzs*lelv)  
      integer*8 glo_num
      common /ivrtx/ vertex ((2**ldim)*lelt)
      common /handle/ gsh_dd
      integer vertex,gsh_dd

      mz = ldim-2
      nx = lx1+2
      ny = ly1+2
      nz = lz1+2*mz
      call setupds_no_crn(gsh_dd,nx,ny,nz,nelv,nelgv,vertex,glo_num)
      call swap_lengths ! Set up Matrices for FDM
      call gen_fast_g

      return
      end
c-----------------------------------------------------------------------
      subroutine h1_overlap_2(u,v,mask)
c
c     Local overlapping Schwarz solves with overlap of 2
c
      include 'SIZE'
      include 'TOTAL'


      common /cwork1/ v1(lxs,lys,lzs,lelt)
      common /handle/ gsh_dd
      integer gsh_dd

      real u(lx1,lx1,lz1,1),v(1),mask(1)
      integer e

      n = lx1*ly1*lz1*nelfld(ifield)

      call dd_swap_vals(v1,v,gsh_dd)

      iz1 = 0
      if (if3d) iz1=1
      do ie=1,nelfld(ifield)
         do iz=1,lz1
         do iy=1,ly1
         do ix=1,lx1
            u(ix,iy,iz,ie) = v1(ix+1,iy+1,iz+iz1,ie)
         enddo
         enddo
         enddo
      enddo

      call dssum (u,lx1,ly1,lz1)
      call col2  (u,mask,n)


      return
      end
c-----------------------------------------------------------------------
      subroutine dd_swap_vals(v1,v0,gsh_dd)

      include 'SIZE'
      include 'TOTAL'

      common /work1/ w1(lxs,lys,lzs)           ! work arrarys for locals
     $              ,w2(lxs,lys,lzs)
      integer gsh_dd

      real v1(lxs,lys,lzs,lelt)
      real v0(lx1,ly1,lz1,lelt)

      integer e

      mz = ldim-2

      nx = lx1+2
      ny = ly1+2
      nz = lz1+2*mz

      n  = lx1*ly1*lz1*nelv
      m  = (lx1+2)*(ly1+2)*(lz1+2*mz)*nelv

      do e=1,nelv
         call rzero_g        (v1,e,nx,ny,nz)
         call fill_interior_g(v1,v0,e,lx1,lz1,mz,nelv)    ! v0      --> v1(int)
         call dface_ext_g    (v1,2,e,nx,nz)               ! v1(int) --> v1(face)
      enddo
c
c                 ~ ~T
c     This is the Q Q  part
c
      call fgslib_gs_op(gsh_dd,v1,1,1,0)  ! 1 ==> +         ! swap v1 & add vals

      do e =1,nelv
         call dface_add1si_g  (v1,-1.,2,e,nx,nz)
         call fastdm1_g       (v1(1,1,1,e),e,w1,w2)
         call s_face_to_int2_g(v1,-1.,2,e,nx,nz)
      enddo
c
c     Exchange/add elemental solutions
c
      call fgslib_gs_op              (gsh_dd,v1,1,1,0)
      do e =1,nelv
         call s_face_to_int2_g(v1,1.,2,e,nx,nz)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_fast_g
c
c     Generate fast diagonalization matrices for each element
c
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'WZ'

      parameter (lxss=lxs*lxs)
      common /fastg/  sr(lxss,2,lelv),ss(lxss,2,lelv),st(lxss,2,lelv)
     $             ,  df(lxs*lys*lzs,lelv)

      common /ctmpf/  lr(2*lx1+4),ls(2*lx1+4),lt(2*lx1+4)
     $              , llr(lelt),lls(lelt),llt(lelt)
     $              , lmr(lelt),lms(lelt),lmt(lelt)
     $              , lrr(lelt),lrs(lelt),lrt(lelt)
      real lr ,ls ,lt
      real llr,lls,llt
      real lmr,lms,lmt
      real lrr,lrs,lrt

      integer lbr,rbr,lbs,rbs,lbt,rbt
     

      call load_semhat_weighted   !   Fills the SEMHAT arrays
     

      ierr = 0
      do ie=1,nelv

         call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,ie,3,ierr)
c
c        Set up matrices for each element.
c
         call set_up_fast_1D_sem_g( sr(1,1,ie),lr,nr ,lbr,rbr
     $                      ,llr(ie),lmr(ie),lrr(ie),ie)
         call set_up_fast_1D_sem_g( ss(1,1,ie),ls,ns ,lbs,rbs
     $                      ,lls(ie),lms(ie),lrs(ie),ie)
         if (if3d) then
            call set_up_fast_1D_sem_g( st(1,1,ie),lt,nt ,lbt,rbt
     $                      ,llt(ie),lmt(ie),lrt(ie),ie)
         endif
c
c        Set up diagonal inverse
c
         if (if3d) then
            eps = 1.e-5 * (vlmax(lr(2),nr-2)
     $                  +  vlmax(ls(2),ns-2) + vlmax(lt(2),nt-2))
            l   = 1
            do k=1,nt
            do j=1,ns
            do i=1,nr
               diag = lr(i) + ls(j) + lt(k)
               if (diag.gt.eps) then
                  df(l,ie) = 1.0/diag
               else
c                 write(6,3) ie,'Reset Eig in gen fast:',i,j,k,l
c    $                         ,eps,diag,lr(i),ls(j),lt(k)
c   3             format(i6,1x,a21,4i5,1p5e12.4)
                  df(l,ie) = 0.0
               endif
               l = l+1
            enddo
            enddo
            enddo
         else
            eps = 1.e-5*(vlmax(lr(2),nr-2) + vlmax(ls(2),ns-2))
            l   = 1
            do j=1,ns
            do i=1,nr
               diag = lr(i) + ls(j)
               
               if (diag.gt.eps) then
                  df(l,ie) = 1.0/diag
                  
               else
c                 write(6,2) ie,'Reset Eig in gen fast:',i,j,l
c    $                         ,eps,diag,lr(i),ls(j)
c   2             format(i6,1x,a21,3i5,1p4e12.4)
                  df(l,ie) = 0.0
               endif
               l = l+1
            enddo
            enddo
         endif
c
c        Next element ....
c
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_sem_g(s,lam,n,lbc,rbc,ll,lm,lr,ie)
      include 'SIZE'
      include 'SEMHAT'

      parameter (lr3=2*lxs*lxs)
      common /fast1dsem/ g(lr3),w(lr3)

      real g,w
      real s(1),lam(1),ll,lm,lr
      integer lbc,rbc

      integer bb0,bb1,eb0,eb1,n,n1
      logical l,r

      n=lx1-1
      !bcs on E are from normal vel component
      if(lbc.eq.2 .or. lbc.eq.3) then !wall,sym  - Neumann
         eb0=0
         bb0=0
      else !outflow,element                      - Dirichlet
         eb0=1
         bb0=1
      endif
      if(rbc.eq.2 .or. rbc.eq.3) then !wall,sym  - Neumann
         eb1=n
         bb1=n
      else !outflow,element                      - Dirichlet
         eb1=n-1
         bb1=n-1
      endif
      l = (lbc.eq.0)
      r = (rbc.eq.0)

c     calculate A tilde operator
      call set_up_fast_1D_sem_op_a(s,eb0,eb1,l,r,ll,lm,lr,ah)
c     calculate B tilde operator
      call set_up_fast_1D_sem_op_b(g,bb0,bb1,l,r,ll,lm,lr,bh)

      n=n+3
c     call outmat   (s,n,n,'  A   ',ie)
c     call outmat   (g,n,n,'  B   ',ie)
      call generalev(s,g,lam,n,w)
      if(.not.l) call row_zero(s,n,n,1)
      if(.not.r) call row_zero(s,n,n,n)
      call transpose(s(n*n+1),n,s,n) ! compute the transpose of s
c     call outmat   (s,n,n,'  S   ',ie)
c     call outmat   (s(n*n+1),n,n,'  St  ',1)
c     call exitt


      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_sem_op_a(g,b0,b1,l
     $                                   ,r,ll,lm,lr,ah)
c            -1 T
c     G = J B  J
c
c     gives the inexact restriction of this matrix to
c     an element plus two node on either side
c
c     g - the output matrix
c     b0, b1 - the range for Bhat indices for the element
c              (enforces boundary conditions)
c     l, r - whether there is a left or right neighbor
c     ll,lm,lr - lengths of left, middle, and right elements
c     ah - hat matrix for A or B
c
c     result is inexact because:
c        neighbor's boundary condition at far end unknown
c        length of neighbor's neighbor unknown
c        (these contribs should be small for large N and
c         elements of nearly equal size)
c
      include 'SIZE'
      real g(0:lx1+1,0:lx1+1)
      real ah(0:lx1-1,0:lx1-1)
      real ll,lm,lr
      integer b0,b1
      logical l,r

      real bl,bm,br
      integer n

      n =lx1-1

c
c     compute the weight of A hat
c
      if (l) bl = 2. /ll
             bm = 2. /lm
      if (r) br = 2. /lr

      call rzero(g,(n+3)*(n+3))
      do j=1,n+1
         do i=1,n+1
            g(i,j) = ah(i-1,j-1)*bm
         enddo
      enddo

      if (l) then
         g(0,0) = g(0,0) + bl*ah(n-1,n-1)
         g(0,1) = g(0,1) + bl*ah(n-1,n  )
         g(1,0) = g(1,0) + bl*ah(n  ,n-1)
         g(1,1) = g(1,1) + bl*ah(n  ,n  )
      elseif (b0.eq.0) then          !Neumann BC
         g(0,0) = 1.
      else                           !Dirichlet BC
         g(0,0)     = 1.
         g(1,1)     = 1.
         do i=2,n+1
            g(i,1)  = 0.
            g(1,i)  = 0.
         enddo
      endif

      if (r) then
         g(n+1,n+1) = g(n+1,n+1) + br*ah(0,0)
         g(n+1,n+2) = g(n+1,n+2) + br*ah(0,1)
         g(n+2,n+1) = g(n+2,n+1) + br*ah(0,1)
         g(n+2,n+2) = g(n+2,n+2) + br*ah(1,1)
      elseif (b1.eq.n) then          !Neumann BC
         g(n+2,n+2)=1.
      else                           !Dirichlet BC
         g(n+2,n+2)   = 1.
         g(n+1,n+1)   = 1.
         do i=1,n
            g(i,n+1)  = 0.
            g(n+1,i)  = 0.
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_sem_op_b(g,b0,b1,l
     $                                   ,r,ll,lm,lr,bh)
c            -1 T
c     G = J B  J
c
c     gives the inexact restriction of this matrix to
c     an element plus two node on either side
c
c     g - the output matrix
c     b0, b1 - the range for Bhat indices for the element
c              (enforces boundary conditions)
c     l, r - whether there is a left or right neighbor
c     ll,lm,lr - lengths of left, middle, and right elements
c     bh - hat matrix for  B
c
c     result is inexact because:
c        neighbor's boundary condition at far end unknown
c        length of neighbor's neighbor unknown
c        (these contribs should be small for large N and
c         elements of nearly equal size)
c
      include 'SIZE'
      real g(0:lx1+1,0:lx1+1)
      real bh(0:lx1-1)
      real ll,lm,lr
      integer b0,b1
      logical l,r

      real bl,bm,br
      integer n

      n =lx1-1
c
c     compute the weight of B hat
c
      if (l) bl = ll / 2.
             bm = lm / 2.
      if (r) br = lr / 2.

      call rzero(g,(n+3)*(n+3))
      do i=1,n+1
            g(i,i) = bh(i-1)*bm
      enddo

      if (l) then
         g(0,0) = g(0,0) + bl*bh(n-1)
         g(1,1) = g(1,1) + bl*bh(n  )
      elseif (b0.eq.0) then          !Neumann BC
         g(0,0) = 1.
      else                           !Dirichlet BC
         g(0,0)     = 1.
         g(1,1)     = 1.
         do i=2,n+1
            g(i,1)  = 0.
            g(1,i)  = 0.
         enddo
      endif

      if (r) then
         g(n+1,n+1) = g(n+1,n+1) + br*bh(0)
         g(n+2,n+2) = g(n+2,n+2) + br*bh(1)
      elseif (b1.eq.n) then          !Neumann BC
         g(n+2,n+2)=1.
      else                           !Dirichlet BC
         g(n+2,n+2)   = 1.
         g(n+1,n+1)   = 1.
         do i=1,n
            g(i,n+1)  = 0.
            g(n+1,i)  = 0.
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine fill_interior_g(v1,v,e,nx,nz,iz1,nel)

      real v1(nx+2,nx+2,nz+2*iz1,nel) ! iz1=ldim-2
      real v (nx  ,nx  ,nz      ,nel)
      integer e

      ny = nx

      do iz=1,nz
      do iy=1,ny
      do ix=1,nx
         v1(ix+1,iy+1,iz+iz1,e) = v(ix,iy,iz,e)
      enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine dface_ext_g(x,t,e,nx,nz)
c     Extend interior to face of element

      include 'SIZE'
      include 'INPUT'
      real x(nx,nx,nz,1)
      integer e,t

      ny = nx

      if (if3d) then
        do iz=2,nz-1
        do ix=2,nx-1
            x(ix,1 ,iz,e) = x(ix, 1+t,iz,e)
            x(ix,ny,iz,e) = x(ix,ny-t,iz,e)
        enddo
        enddo
  
        do iz=2,nz-1
        do iy=2,ny-1
            x(1 ,iy,iz,e) = x( 1+t,iy,iz,e)
            x(nx,iy,iz,e) = x(nx-t,iy,iz,e)
        enddo
        enddo
  
        do iy=2,ny-1
        do ix=2,nx-1
            x(ix,iy,1 ,e) = x(ix,iy, 1+t,e)
            x(ix,iy,nz,e) = x(ix,iy,nz-t,e)
        enddo
        enddo
      else
         do ix=2,nx-1
            x(ix,1 ,1,e) = x(ix, 1+t,1,e)
            x(ix,ny,1,e) = x(ix,ny-t,1,e)
         enddo
         do iy=2,ny-1
            x(1 ,iy,1,e) = x( 1+t,iy,1,e)
            x(nx,iy,1,e) = x(nx-t,iy,1,e)
         enddo
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine dface_add1si_g(x,c,t,e,nx,nz)
c     Scale interior and add to face of element

      include 'SIZE'
      include 'INPUT'
      real x(nx,nx,nz,1)

      integer e,t

      ny = nx

      if (if3d) then

         do iz=2,nz-1
         do ix=2,nx-1
            x(ix,1 ,iz,e) = x(ix,1 ,iz,e) + c*x(ix, 1+t,iz,e)
            x(ix,ny,iz,e) = x(ix,ny,iz,e) + c*x(ix,ny-t,iz,e)
         enddo
         enddo

         do iz=2,nz-1
         do iy=2,ny-1
            x(1 ,iy,iz,e) = x(1 ,iy,iz,e) + c*x( 1+t,iy,iz,e)
            x(nx,iy,iz,e) = x(nx,iy,iz,e) + c*x(nx-t,iy,iz,e)
         enddo
         enddo

         do iy=2,ny-1
         do ix=2,nx-1
            x(ix,iy,1 ,e) = x(ix,iy,1 ,e) + c*x(ix,iy, 1+t,e)
            x(ix,iy,nz,e) = x(ix,iy,nz,e) + c*x(ix,iy,nz-t,e)
         enddo
         enddo

      else  ! 2D

         do ix=2,nx-1
            x(ix,1 ,1,e) = x(ix,1 ,1,e) + c*x(ix, 1+t,1,e)
            x(ix,ny,1,e) = x(ix,ny,1,e) + c*x(ix,ny-t,1,e)
         enddo
         do iy=2,ny-1
            x(1  ,iy,1,e) = x(1 ,iy,1,e) + c*x( 1+t,iy,1,e)
            x(nx ,iy,1,e) = x(nx,iy,1,e) + c*x(nx-t,iy,1,e)
         enddo

      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine fastdm1_g(R,ie,w1,w2)
c
c     Fast diagonalization solver for FEM on mesh 1
c
      include 'SIZE'

      parameter (lxss=lxs*lxs)
      common /fastg/  sr(lxss,2,lelv),ss(lxss,2,lelv),st(lxss,2,lelv)
     $             ,  df(lxs*lys*lzs,lelv)

      parameter (lxyz = lxs*lys*lzs)

      real r(1),w1(1),w2(1)

      nx = lx1+2
c
c      T
c     S  r
      call tensr3 (w1,nx,r ,nx,sr(1,2,ie),ss(1,1,ie),st(1,1,ie),w2)
c
c
c      -1 T
c     D  S  r
c
      call col2   (w1,df(1,ie),lxyz)
c
c
c        -1 T
c     S D  S  r
c
      call tensr3 (r ,nx,w1,nx,sr(1,1,ie),ss(1,2,ie),st(1,2,ie),w2)

      return
      end
c-----------------------------------------------------------------------
      subroutine s_face_to_int2_g(x,c,t,e,nx,nz)
c
c     Scale face and add to interior of element
c
      include 'SIZE'
      include 'INPUT'
      real x(nx,nx,nz,1)
      integer t,e

      ny=nx


      if (if3d) then

         do iz=2,nz-1
         do ix=2,nx-1
            x(ix, 1+t,iz,e) = c*x(ix,1 ,iz,e) + x(ix, 1+t,iz,e)
            x(ix,ny-t,iz,e) = c*x(ix,ny,iz,e) + x(ix,ny-t,iz,e)
         enddo
         enddo

         do iz=2,nz-1
         do iy=2,ny-1
            x( 1+t,iy,iz,e) = c*x(1 ,iy,iz,e) + x( 1+t,iy,iz,e)
            x(nx-t,iy,iz,e) = c*x(nx,iy,iz,e) + x(nx-t,iy,iz,e)
         enddo
         enddo

         do iy=2,ny-1
         do ix=2,nx-1
            x(ix,iy, 1+t,e) = c*x(ix,iy,1 ,e) + x(ix,iy, 1+t,e)
            x(ix,iy,nz-t,e) = c*x(ix,iy,nz,e) + x(ix,iy,nz-t,e)
         enddo
         enddo

      else
c     2D
         do ix=2,nx-1
            x(ix, 1+t,1,e) = c*x(ix,1 ,1,e) + x(ix, 1+t,1,e)
            x(ix,ny-t,1,e) = c*x(ix,ny,1,e) + x(ix,ny-t,1,e)
         enddo
         do iy=2,ny-1
            x( 1+t,iy,1,e) = c*x(1 ,iy,1,e) + x( 1+t,iy,1,e)
            x(nx-t,iy,1,e) = c*x(nx,iy,1,e) + x(nx-t,iy,1,e)
         enddo
      endif
      
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldr_g(x,txt10,nx,nz,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(nx,nx,nz,lelt)
      character*10 txt10

      integer idum,e
      save idum
      data idum /3/

      if (idum.lt.0) return

      ny = nx


      mtot = nx*ny*nz*nelv
      if (nx.gt.8.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)

      nell = nelt
      rnel = nell
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nell-ne+1
      k = 1
      do ie=1,1
         ne = 0
         write(6,116) txt10,k,ie,xmin,xmax,istep,time,nx
         do l=12,0,-4
            write(6,117)
            do j=ny,1,-1
              if (nx.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.8) write(6,118) ((x(i,j,k,e+l),i=1,nx),e=1,4)
            enddo
         enddo
      enddo

  102 FORMAT(4(2f9.5,2x))
  103 FORMAT(4(3f9.5,2x))
  104 FORMAT(4(4f7.3,2x))
  105 FORMAT(5f9.5,10x,5f9.5)
  106 FORMAT(6f7.1,5x,6f7.1,5x,6f7.1,5x,6f7.1)
  107 FORMAT(7f8.4,5x,7f8.4)
  108 FORMAT(8f8.4,4x,8f8.4)
  118 FORMAT(8f12.9)

  116 FORMAT(  /,5X,'     ^              ',/,
     $    5X,'   Y |              ',/,
     $    5X,'     |              ',A10,/,
     $    5X,'     +---->         ','Plane = ',I2,'/',I2,2x,2e12.4,/,
     $    5X,'       X            ','Step  =',I9,f15.5,i2)
  117 FORMAT(' ')

      if (ichk.eq.1.and.idum.gt.0) call checkit(idum)
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldi_g(x,txt10,nx,nz,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      integer x(nx,nx,nz,lelt)
      character*10 txt10

      integer idum,e,xmin,xmax
      save idum
      data idum /3/

      if (idum.lt.0) return

      ny = nx


      mtot = nx*ny*nz*nelv
      if (nx.gt.8.or.nelv.gt.16) return
      xmin = iglmin(x,mtot)
      xmax = iglmax(x,mtot)

      nell = nelt
      rnel = nell
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nell-ne+1
      k = 1
      do ie=1,1
         ne = 0
         write(6,116) txt10,k,ie,xmin,xmax,istep,time,nx
         do l=12,0,-4
            write(6,117)
            do j=ny,1,-1
              if (nx.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,nx),e=1,4)
              if (nx.eq.8) write(6,118) ((x(i,j,k,e+l),i=1,nx),e=1,4)
            enddo
         enddo
      enddo

  102 FORMAT(4(2i9,2x))
  103 FORMAT(4(3i9,2x))
  104 FORMAT(4(4i7,2x))
  105 FORMAT(5i9,10x,5i9)
  106 FORMAT(6i7,5x,6i7,5x,6i7,5x,6i7)
  107 FORMAT(7i8,5x,7i8)
  108 FORMAT(8i8,4x,8i8)
  118 FORMAT(8i12)

  116 FORMAT(  /,5X,'     ^              ',/,
     $    5X,'   Y |              ',/,
     $    5X,'     |              ',A10,/,
     $    5X,'     +---->         ','Plane = ',I2,'/',I2,2x,2i12,/,
     $    5X,'       X            ','Step  =',I9,f15.5,i2)
  117 FORMAT(' ')

      if (ichk.eq.1.and.idum.gt.0) call checkit(idum)
      return
      end
c-----------------------------------------------------------------------
      subroutine setupds_no_crn(gs_h,nx,ny,nz,nel,melg,vertex,glo_num)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'NONCON'
      integer   gs_h,vertex(1),e
      integer*8 ngv,glo_num(nx,ny,nz,nel)
      

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

c     set up the global numbering
      call set_vert(glo_num,ngv,nx,nel,vertex,.false.)

c     zero out corners
      mz1 = max(1,nz-1)
      do e=1,nel
      do k=1,nz,mz1
      do j=1,ny,ny-1
      do i=1,nx,nx-1
         glo_num(i,j,k,e) = 0
      enddo
      enddo
      enddo
      enddo


      ntot = nx*ny*nz*nel

      t0 = dnekclock()
      call fgslib_gs_setup(gs_h,glo_num,ntot,nekcomm,mp) ! initialize gs code
      t1 = dnekclock()

      et = t1-t0

c     call gs_chkr(glo_num)

      if (nio.eq.0) then
         write(6,1) et,nx,nel,ntot,ngv,gs_h
    1    format('   gs_init time',1pe11.4,' seconds ',i3,4i10)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rzero_g(a,e,nx,ny,nz)

      real a(nx,ny,nz,1)
      integer e

      do  i=1,nx
      do  j=1,ny
      do  k=1,nz
          a(i,j,k,e) = 0.0
      enddo
      enddo
      enddo

      return
      END
c-----------------------------------------------------------------------

