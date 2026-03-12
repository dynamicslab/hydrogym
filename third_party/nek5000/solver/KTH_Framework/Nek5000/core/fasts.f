c-----------------------------------------------------------------------
      subroutine local_solves_fdm(u,v)
c
c     Given an input vector v, this returns the additive Schwarz solution
c
c                       -1
c                  u = M   v
c
c                      T     -1  
c     where M = sum ( R_i A_i    R_i )
c                i
c
c     The R_i's are simply index_set restriction operators.
c
c     The local solves are performed with the fast diagonalization method.
c
      include 'SIZE'
      include 'INPUT'
      include 'DOMAIN'
      include 'PARALLEL'
c
      include 'TSTEP'
      include 'CTIMER'
c
      real u(lx2,ly2,lz2,lelv),v(lx2,ly2,lz2,lelv)
      common /scrpre/ v1(lx1,ly1,lz1,lelv)
     $               ,w1(lx1,ly1,lz1),w2(lx1,ly1,lz1)
      common /scrover/ ar(lelv)

      parameter(lxx=lx1*lx1, levb=lelv+lbelv)
      common /fastd/  df(lx1*ly1*lz1,levb)
     $             ,  sr(lxx*2,levb),ss(lxx*2,levb),st(lxx*2,levb)
      integer e,eb,eoff

c
      if (icalld.eq.0) tsolv=0.0
      icalld=icalld+1
      nsolv=icalld
c
      ntot1 = lx1*ly1*lz1*nelv
      ntot2 = lx2*ly2*lz2*nelv
c
c     Fill interiors
      iz1 = 0
      if (if3d) iz1=1
      call rzero(v1,ntot1)
      do e=1,nelv
         do iz=1,lz2
         do iy=1,ly2
         do ix=1,lx2
            v1(ix+1,iy+1,iz+iz1,e) = v(ix,iy,iz,e)
         enddo
         enddo
         enddo
      enddo
      call dface_ext    (v1)
      call dssum        (v1,lx1,ly1,lz1)
      call dface_add1si (v1,-1.)
c
c     Now solve each subdomain problem:
c
      etime1=dnekclock()

      eoff  = 0
      if (ifield.gt.1) eoff  = nelv

      do e = 1,nelv
         eb = e + eoff
         call fastdm1(v1(1,1,1,e),df(1,eb)
     $                           ,sr(1,eb),ss(1,eb),st(1,eb),w1,w2)
      enddo
      tsolv=tsolv+dnekclock()-etime1
c
c     Exchange/add elemental solutions
c
      call s_face_to_int (v1,-1.)
      call dssum         (v1,lx1,ly1,lz1)
      call s_face_to_int (v1, 1.)
      if(param(42).eq.0) call do_weight_op(v1)
c
c     Map back to pressure grid (extract interior values)
c
      do e=1,nelv
         do iz=1,lz2
         do iy=1,ly2
         do ix=1,lx2
            u(ix,iy,iz,e) = v1(ix+1,iy+1,iz+iz1,e)
         enddo
         enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine fastdm1(r,df,sr,ss,st,w1,w2)
c
c     Fast diagonalization solver for FEM on mesh 1
c
      include 'SIZE'
      parameter (lxx=lx1*lx1,lxyz=lx1*ly1*lz1)

      real r(1),df(1),sr(lxx,2),ss(lxx,2),st(lxx,2),w1(1),w2(1)
c
c
c      T
c     S  r
      call tensr3 (w1,lx1,r ,lx1,sr(1,2),ss(1,1),st(1,1),w2)
c
c     
c      -1 T
c     D  S  r
c
      call col2   (w1,df,lxyz)
c
c
c        -1 T
c     S D  S  r
c
      call tensr3 (r ,lx1,w1,lx1,sr(1,1),ss(1,2),st(1,2),w2)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine tensr3(v,nv,u,nu,A,Bt,Ct,w)
C
C     -  Tensor product application of v = (C x B x A) u
C        NOTE -- the transpose of B & C must be input, rather than B & C.
C
C     -  scratch arrays: w(nu*nu*nv)
C
C
      include 'SIZE'
      include 'INPUT'
      real v(nv,nv,nv),u(nu,nu,nu)
      real A(1),Bt(1),Ct(1)
      real w(1)

      if (nu.gt.nv) then
         write(6,*) nid,nu,nv,' ERROR in tensr3. Contact P.Fischer.'
         write(6,*) nid,nu,nv,' Memory problem.'
         call exitt
      endif

      if (if3d) then
         nuv = nu*nv
         nvv = nv*nv
         call mxm(A,nv,u,nu,v,nu*nu)
         k=1
         l=1
         do iz=1,nu
            call mxm(v(k,1,1),nv,Bt,nu,w(l),nv)
            k=k+nuv
            l=l+nvv
         enddo
         call mxm(w,nvv,Ct,nu,v,nv)
      else
         call mxm(A,nv,u,nu,w,nu)
         call mxm(w,nv,Bt,nu,v,nv)
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine s_face_to_int(x,c)
c
c     Scale face and add to interior of element     
c
      include 'SIZE'
      include 'INPUT'
      real x(lx1,ly1,lz1,1)
c
      do ie=1,nelv
c
        if (if3d) then
c
          do iz=2,lz1-1
          do ix=2,lx1-1
            x(ix,2    ,iz,ie) = c*x(ix,1  ,iz,ie) + x(ix,2    ,iz,ie)
            x(ix,ly1-1,iz,ie) = c*x(ix,ly1,iz,ie) + x(ix,ly1-1,iz,ie)
          enddo
          enddo
c
          do iz=2,lz1-1
          do iy=2,ly1-1
            x(2    ,iy,iz,ie) = c*x(1  ,iy,iz,ie) + x(2    ,iy,iz,ie)
            x(lx1-1,iy,iz,ie) = c*x(lx1,iy,iz,ie) + x(lx1-1,iy,iz,ie)
          enddo
          enddo
c
          do iy=2,ly1-1
          do ix=2,lx1-1
            x(ix,iy,2    ,ie) = c*x(ix,iy,1  ,ie) + x(ix,iy,2    ,ie)
            x(ix,iy,lz1-1,ie) = c*x(ix,iy,lz1,ie) + x(ix,iy,lz1-1,ie)
          enddo
          enddo
c
        else
c         2D
          do ix=2,lx1-1
            x(ix,2    ,1,ie) = c*x(ix,1  ,1,ie) + x(ix,2    ,1,ie)
            x(ix,ly1-1,1,ie) = c*x(ix,ly1,1,ie) + x(ix,ly1-1,1,ie)
          enddo
          do iy=2,ly1-1
            x(2    ,iy,1,ie) = c*x(1  ,iy,1,ie) + x(2    ,iy,1,ie)
            x(lx1-1,iy,1,ie) = c*x(lx1,iy,1,ie) + x(lx1-1,iy,1,ie)
          enddo
        endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine dface_ext(x)
c     Extend interior to face of element     
c
      include 'SIZE'
      include 'INPUT'
      real x(lx1,ly1,lz1,1)
c
      do ie=1,nelv
c
         if (if3d) then
c
          do iz=2,lz1-1
          do ix=2,lx1-1
            x(ix,1  ,iz,ie) = x(ix,2    ,iz,ie)
            x(ix,ly1,iz,ie) = x(ix,ly1-1,iz,ie)
          enddo
          enddo
c
          do iz=2,lz1-1
          do iy=2,ly1-1
            x(1  ,iy,iz,ie) = x(2    ,iy,iz,ie)
            x(lx1,iy,iz,ie) = x(lx1-1,iy,iz,ie)
          enddo
          enddo
c
          do iy=2,ly1-1
          do ix=2,lx1-1
            x(ix,iy,1  ,ie) = x(ix,iy,2    ,ie)
            x(ix,iy,lz1,ie) = x(ix,iy,lz1-1,ie)
          enddo
          enddo
c
        else
c
          do ix=2,lx1-1
            x(ix,1  ,1,ie) = x(ix,2    ,1,ie)
            x(ix,ly1,1,ie) = x(ix,ly1-1,1,ie)
          enddo
          do iy=2,ly1-1
            x(1  ,iy,1,ie) = x(2    ,iy,1,ie)
            x(lx1,iy,1,ie) = x(lx1-1,iy,1,ie)
          enddo
c
        endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine dface_add1si(x,c)
c     Scale interior and add to face of element     
c
      include 'SIZE'
      include 'INPUT'
      real x(lx1,ly1,lz1,1)
c
      do ie=1,nelv
c
        if (if3d) then
c
          do iz=2,lz1-1
          do ix=2,lx1-1
            x(ix,1  ,iz,ie) = x(ix,1  ,iz,ie) + c*x(ix,2    ,iz,ie)
            x(ix,ly1,iz,ie) = x(ix,ly1,iz,ie) + c*x(ix,ly1-1,iz,ie)
          enddo
          enddo
c
          do iz=2,lz1-1
          do iy=2,ly1-1
            x(1  ,iy,iz,ie) = x(1  ,iy,iz,ie) + c*x(2    ,iy,iz,ie)
            x(lx1,iy,iz,ie) = x(lx1,iy,iz,ie) + c*x(lx1-1,iy,iz,ie)
          enddo
          enddo
c
          do iy=2,ly1-1
          do ix=2,lx1-1
            x(ix,iy,1  ,ie) = x(ix,iy,1  ,ie) + c*x(ix,iy,2    ,ie)
            x(ix,iy,lz1,ie) = x(ix,iy,lz1,ie) + c*x(ix,iy,lz1-1,ie)
          enddo
          enddo
c
        else
c
c         2D
c
          do ix=2,lx1-1
            x(ix,1  ,1,ie) = x(ix,1  ,1,ie) + c*x(ix,2    ,1,ie)
            x(ix,ly1,1,ie) = x(ix,ly1,1,ie) + c*x(ix,ly1-1,1,ie)
          enddo
          do iy=2,ly1-1
            x(1  ,iy,1,ie) = x(1  ,iy,1,ie) + c*x(2    ,iy,1,ie)
            x(lx1,iy,1,ie) = x(lx1,iy,1,ie) + c*x(lx1-1,iy,1,ie)
          enddo
c
        endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine init_weight_op

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      parameter(levb=lelv+lbelv)
      common /swaplengths/ l(lx1,ly1,lz1,lelv)
      common /weightop/ w(lx2,lz2,2,3,levb)
      real l
      real w
      integer e,e0,eb

      e0  = 0
      if (ifield.gt.1) e0 = nelv

      n=lx2+1
      if (if3d) then
         do e=1,nelv
            call rzero(l(1,1,1,e),lx1*ly1*lz1)
            do k=2,n
               do j=2,n
                  l(2,j,k,e)=1
                  l(n,j,k,e)=1
               enddo
            enddo
            do k=2,n
               do i=2,n
                  l(i,2,k,e)=1
                  l(i,n,k,e)=1
               enddo
            enddo
            do j=2,n
               do i=2,n
                  l(i,j,2,e)=1
                  l(i,j,n,e)=1
               enddo
            enddo
         enddo
      else
         do e=1,nelv
            call rzero(l(1,1,1,e),lx1*ly1*lz1)
            do j=2,n
               l(2,j,1,e)=1
               l(n,j,1,e)=1
            enddo
            do i=2,n
               l(i,2,1,e)=1
               l(i,n,1,e)=1
            enddo
         enddo
      endif
      
      call dface_ext(l)
      call dssum(l,lx1,ly1,lz1)
      call dface_add1si(l,-1.)
      call s_face_to_int(l,1.)
c     l now holds the count matrix C on the outer pressure nodes
      if (if3d) then
         do e=1,nelv
            eb = e0+e
            do k=1,lz2
               do j=1,ly2
                  w(j,k,1,1,eb)=1.0/l(2,j+1,k+1,e)
                  w(j,k,2,1,eb)=1.0/l(n,j+1,k+1,e)
               enddo
            enddo
            do k=1,lz2
               do i=1,lx2
                  w(i,k,1,2,eb)=1.0/l(i+1,2,k+1,e)
                  w(i,k,2,2,eb)=1.0/l(i+1,n,k+1,e)
               enddo
            enddo
            do j=1,ly2
               do i=1,lx2
                  w(i,j,1,3,eb)=1.0/l(i+1,j+1,2,e)
                  w(i,j,2,3,eb)=1.0/l(i+1,j+1,n,e)
               enddo
            enddo
         enddo
      else
         do e=1,nelv
            eb = e0+e
            do j=1,ly2
               w(j,1,1,1,eb)=1.0/l(2,j+1,1,e)
               w(j,1,2,1,eb)=1.0/l(n,j+1,1,e)
            enddo
            do i=1,lx2
               w(i,1,1,2,eb)=1.0/l(i+1,2,1,e)
               w(i,1,2,2,eb)=1.0/l(i+1,n,1,e)
            enddo
         enddo
      endif
      end
c-----------------------------------------------------------------------
      subroutine do_weight_op(x)
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      parameter(levb=lelv+lbelv)
      common /weightop/ w(lx2,lz2,2,3,levb)
      real w

      real x(0:lx1-1,0:ly1-1,0:lz1-1,1)
      integer e,e0,eb

      e0  = 0
      if (ifield.gt.1) e0 = nelv

      if (if3d) then
         do e=1,nelv
            eb = e0 + e
            do k=1,lz2
               do j=1,ly2
                  x(  1,j,k,e)=w(j,k,1,1,eb)*x(  1,j,k,e)
                  x(lx2,j,k,e)=w(j,k,2,1,eb)*x(lx2,j,k,e)
               enddo
            enddo
            do k=1,lz2
               do i=2,lx2-1
                  x(i,  1,k,e)=w(i,k,1,2,eb)*x(i,  1,k,e)
                  x(i,ly2,k,e)=w(i,k,2,2,eb)*x(i,ly2,k,e)
               enddo
            enddo
            do j=2,ly2-1
               do i=2,lx2-1
                  x(i,j,  1,e)=w(i,j,1,3,eb)*x(i,j,  1,e)
                  x(i,j,lz2,e)=w(i,j,2,3,eb)*x(i,j,lz2,e)
               enddo
            enddo
         enddo
      else
         do e=1,nelv
            eb = e0 + e
            do j=1,ly2
               x(  1,j,0,e)=w(j,1,1,1,eb)*x(  1,j,0,e)
               x(lx2,j,0,e)=w(j,1,2,1,eb)*x(lx2,j,0,e)
            enddo
            do i=2,lx2-1
               x(i,  1,0,e)=w(i,1,1,2,eb)*x(i,  1,0,e)
               x(i,ly2,0,e)=w(i,1,2,2,eb)*x(i,ly2,0,e)
            enddo
         enddo
      endif
      end
