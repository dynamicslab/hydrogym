c-----------------------------------------------------------------------
      subroutine gen_fast(df,sr,ss,st,x,y,z)
c
c     Generate fast diagonalization matrices for each element
c
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'WZ'
c
      parameter(lxx=lx1*lx1)
      real df(lx1*ly1*lz1,1),sr(lxx*2,1),ss(lxx*2,1),st(lxx*2,1)
c
      common /ctmpf/  lr(2*lx1+4),ls(2*lx1+4),lt(2*lx1+4)
     $              , llr(lelt),lls(lelt),llt(lelt)
     $              , lmr(lelt),lms(lelt),lmt(lelt)
     $              , lrr(lelt),lrs(lelt),lrt(lelt)
      real lr ,ls ,lt 
      real llr,lls,llt
      real lmr,lms,lmt
      real lrr,lrs,lrt
c
      integer lbr,rbr,lbs,rbs,lbt,rbt,e
c
      real x(lx1,ly1,lz1,nelv)
      real y(lx1,ly1,lz1,nelv)
      real z(lx1,ly1,lz1,nelv)
      real axwt(lx2)

      ierr = 0

      do e=1,nelv
c
         if (param(44).eq.1) then
           call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,e,2,ierr)
         else
           call get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,e,3,ierr)
         endif
c
c        Set up matrices for each element.
c
         if (param(44).eq.1) then
           call set_up_fast_1D_fem( sr(1,e),lr,nr ,lbr,rbr
     $                      ,llr(e),lmr(e),lrr(e),zgm2(1,1),lx2,e)
         else
           call set_up_fast_1D_sem( sr(1,e),lr,nr ,lbr,rbr
     $                      ,llr(e),lmr(e),lrr(e),e)
         endif
         if (ifaxis) then
            xsum = vlsum(wxm2,lx2)
            do i=1,ly2
               yavg = vlsc2(y(1,i,1,e),wxm2,lx2)/xsum
               axwt(i) = yavg
            enddo
            call set_up_fast_1D_fem_ax( ss(1,e),ls,ns ,lbs,rbs
     $                 ,lls(e),lms(e),lrs(e),zgm2(1,2),axwt,ly2,e)
         else
            if (param(44).eq.1) then
               call set_up_fast_1D_fem( ss(1,e),ls,ns ,lbs,rbs
     $                      ,lls(e),lms(e),lrs(e),zgm2(1,2),ly2,e)
            else
               call set_up_fast_1D_sem( ss(1,e),ls,ns ,lbs,rbs
     $                      ,lls(e),lms(e),lrs(e),e)
            endif
         endif
         if (if3d) then
            if (param(44).eq.1) then
               call set_up_fast_1D_fem( st(1,e),lt,nt ,lbt,rbt
     $                      ,llt(e),lmt(e),lrt(e),zgm2(1,3),lz2,e)
            else
               call set_up_fast_1D_sem( st(1,e),lt,nt ,lbt,rbt
     $                      ,llt(e),lmt(e),lrt(e),e)
            endif
         endif
c
c        DIAGNOSTICS
c
c        n12 = min(9,nr)
c        write(6,1) e,'1D lr',llr(e),lmr(e),lrr(e),(lr(k),k=1,n12)
c        write(6,1) e,'1D ls',lls(e),lms(e),lrs(e),(ls(k),k=1,n12)
c        if (if3d) 
c    $   write(6,1) e,'1D lt',llt(e),lmt(e),lrt(e),(lt(k),k=1,n12)
c   1    format(i6,1x,a5,1p12e12.4)
c
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
                  df(l,e) = 1.0/diag
               else
c                 write(6,3) e,'Reset Eig in gen fast:',i,j,k,l
c    $                         ,eps,diag,lr(i),ls(j),lt(k)
c   3             format(i6,1x,a21,4i5,1p5e12.4)
                  df(l,e) = 0.0
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
                  df(l,e) = 1.0/diag
               else
c                 write(6,2) e,'Reset Eig in gen fast:',i,j,l
c    $                         ,eps,diag,lr(i),ls(j)
c   2             format(i6,1x,a21,3i5,1p4e12.4)
                  df(l,e) = 0.0
               endif
               l = l+1
            enddo
            enddo
         endif
c
c        Next element ....
c
      enddo

      ierrmx = iglmax(ierr,1)
      if (ierrmx.gt.0) then
         if (ierr.gt.0) write(6,*) nid,ierr,' BC FAIL'
         call exitti('E INVALID BC FOUND in genfast$',ierrmx)
      endif


      return
      end
c-----------------------------------------------------------------------
      subroutine gen_fast_spacing(x,y,z)
c
c     Generate fast diagonalization matrices for each element
c
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'WZ'
c
      parameter(lxx=lx1*lx1)
c
      common /ctmpf/  lr(2*lx1+4),ls(2*lx1+4),lt(2*lx1+4)
     $              , llr(lelt),lls(lelt),llt(lelt)
     $              , lmr(lelt),lms(lelt),lmt(lelt)
     $              , lrr(lelt),lrs(lelt),lrt(lelt)
      real lr ,ls ,lt 
      real llr,lls,llt
      real lmr,lms,lmt
      real lrr,lrs,lrt
c
      integer lbr,rbr,lbs,rbs,lbt,rbt,e
c
      real x(lx1,ly1,lz1,nelv)
      real y(lx1,ly1,lz1,nelv)
      real z(lx1,ly1,lz1,nelv)
      real axwt(lx2)

      ierr = 0

      if (param(44).eq.1) then
c                                    __ __ __
c        Now, for each element, compute lr,ls,lt between specified planes
c
         n1 = lx2
         n2 = lx2+1
         nz0 = 1
         nzn = 1
         if (if3d) then
            nz0= 0
            nzn=n2
         endif
         eps = 1.e-7
         if (wdsize.eq.8)  eps = 1.e-14
c
c        Find mean spacing between "left-most" planes
         call plane_space2(llr,lls,llt, 0,wxm2,x,y,z,n1,n2,nz0,nzn)
c
c        Find mean spacing between "middle" planes
         call plane_space (lmr,lms,lmt, 1,n1,wxm2,x,y,z,n1,n2,nz0,nzn)
c
c        Find mean spacing between "right-most" planes
         call plane_space2(lrr,lrs,lrt,n2,wxm2,x,y,z,n1,n2,nz0,nzn)
c
      else
         call load_semhat_weighted    !   Fills the SEMHAT arrays
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine plane_space_std(lr,ls,lt,i1,i2,w,x,y,z,nx,nxn,nz0,nzn)
c
c     This routine now replaced by "plane_space()"
c
c     Here, spacing is based on arithmetic mean. 
c     New verision uses harmonic mean.  pff 2/10/07
c
      include 'SIZE'
      include 'INPUT'
c
      real w(1),lr(1),ls(1),lt(1)
      real x(0:nxn,0:nxn,nz0:nzn,1)
      real y(0:nxn,0:nxn,nz0:nzn,1)
      real z(0:nxn,0:nxn,nz0:nzn,1)
      real lr2,ls2,lt2
c                                    __ __ __
c     Now, for each element, compute lr,ls,lt between specified planes
c
      ny = nx
      nz = nx
      j1 = i1
      k1 = i1
      j2 = i2
      k2 = i2
c
      do ie=1,nelv
c
         if (if3d) then
            lr2  = 0.
            wsum = 0.
            do k=1,nz
            do j=1,ny
               weight = w(j)*w(k)
               lr2  = lr2  + ( (x(i2,j,k,ie)-x(i1,j,k,ie))**2
     $                     +   (y(i2,j,k,ie)-y(i1,j,k,ie))**2
     $                     +   (z(i2,j,k,ie)-z(i1,j,k,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            lr2     = lr2/wsum
            lr(ie)  = sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do k=1,nz
            do i=1,nx
               weight = w(i)*w(k)
               ls2  = ls2  + ( (x(i,j2,k,ie)-x(i,j1,k,ie))**2
     $                     +   (y(i,j2,k,ie)-y(i,j1,k,ie))**2
     $                     +   (z(i,j2,k,ie)-z(i,j1,k,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            ls2     = ls2/wsum
            ls(ie)  = sqrt(ls2)
c
            lt2 = 0.
            wsum = 0.
            do j=1,ny
            do i=1,nx
               weight = w(i)*w(j)
               lt2  = lt2  + ( (x(i,j,k2,ie)-x(i,j,k1,ie))**2
     $                     +   (y(i,j,k2,ie)-y(i,j,k1,ie))**2
     $                     +   (z(i,j,k2,ie)-z(i,j,k1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            lt2     = lt2/wsum
            lt(ie)  = sqrt(lt2)
c
         else
            lr2 = 0.
            wsum = 0.
            do j=1,ny
               weight = w(j)
               lr2  = lr2  + ( (x(i2,j,1,ie)-x(i1,j,1,ie))**2
     $                     +   (y(i2,j,1,ie)-y(i1,j,1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            lr2     = lr2/wsum
            lr(ie)  = sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do i=1,nx
               weight = w(i)
               ls2  = ls2  + ( (x(i,j2,1,ie)-x(i,j1,1,ie))**2
     $                     +   (y(i,j2,1,ie)-y(i,j1,1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            ls2     = ls2/wsum
            ls(ie)  = sqrt(ls2)
c           write(6,*) 'lrls',ie,lr(ie),ls(ie)
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine plane_space(lr,ls,lt,i1,i2,w,x,y,z,nx,nxn,nz0,nzn)
c
c     Here, spacing is based on harmonic mean.  pff 2/10/07
c
c
      include 'SIZE'
      include 'INPUT'
c
      real w(1),lr(1),ls(1),lt(1)
      real x(0:nxn,0:nxn,nz0:nzn,1)
      real y(0:nxn,0:nxn,nz0:nzn,1)
      real z(0:nxn,0:nxn,nz0:nzn,1)
      real lr2,ls2,lt2
c                                    __ __ __
c     Now, for each element, compute lr,ls,lt between specified planes
c
      ny = nx
      nz = nx
      j1 = i1
      k1 = i1
      j2 = i2
      k2 = i2
c
      do ie=1,nelv
c
         if (if3d) then
            lr2  = 0.
            wsum = 0.
            do k=1,nz
            do j=1,ny
               weight = w(j)*w(k)
c              lr2  = lr2  + ( (x(i2,j,k,ie)-x(i1,j,k,ie))**2
c    $                     +   (y(i2,j,k,ie)-y(i1,j,k,ie))**2
c    $                     +   (z(i2,j,k,ie)-z(i1,j,k,ie))**2 )
c    $                     *   weight
               lr2  = lr2  +   weight /
     $                       ( (x(i2,j,k,ie)-x(i1,j,k,ie))**2
     $                     +   (y(i2,j,k,ie)-y(i1,j,k,ie))**2
     $                     +   (z(i2,j,k,ie)-z(i1,j,k,ie))**2 )
               wsum = wsum + weight
            enddo
            enddo
            lr2     = lr2/wsum
            lr(ie)  = 1./sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do k=1,nz
            do i=1,nx
               weight = w(i)*w(k)
c              ls2  = ls2  + ( (x(i,j2,k,ie)-x(i,j1,k,ie))**2
c    $                     +   (y(i,j2,k,ie)-y(i,j1,k,ie))**2
c    $                     +   (z(i,j2,k,ie)-z(i,j1,k,ie))**2 )
c    $                     *   weight
               ls2  = ls2  +   weight /
     $                       ( (x(i,j2,k,ie)-x(i,j1,k,ie))**2
     $                     +   (y(i,j2,k,ie)-y(i,j1,k,ie))**2
     $                     +   (z(i,j2,k,ie)-z(i,j1,k,ie))**2 )
               wsum = wsum + weight
            enddo
            enddo
            ls2     = ls2/wsum
            ls(ie)  = 1./sqrt(ls2)
c
            lt2 = 0.
            wsum = 0.
            do j=1,ny
            do i=1,nx
               weight = w(i)*w(j)
c              lt2  = lt2  + ( (x(i,j,k2,ie)-x(i,j,k1,ie))**2
c    $                     +   (y(i,j,k2,ie)-y(i,j,k1,ie))**2
c    $                     +   (z(i,j,k2,ie)-z(i,j,k1,ie))**2 )
c    $                     *   weight
               lt2  = lt2  +   weight /
     $                       ( (x(i,j,k2,ie)-x(i,j,k1,ie))**2
     $                     +   (y(i,j,k2,ie)-y(i,j,k1,ie))**2
     $                     +   (z(i,j,k2,ie)-z(i,j,k1,ie))**2 )
               wsum = wsum + weight
            enddo
            enddo
            lt2     = lt2/wsum
            lt(ie)  = 1./sqrt(lt2)
c
         else              ! 2D
            lr2 = 0.
            wsum = 0.
            do j=1,ny
               weight = w(j)
c              lr2  = lr2  + ( (x(i2,j,1,ie)-x(i1,j,1,ie))**2
c    $                     +   (y(i2,j,1,ie)-y(i1,j,1,ie))**2 )
c    $                     *   weight
               lr2  = lr2  + weight /
     $                       ( (x(i2,j,1,ie)-x(i1,j,1,ie))**2
     $                       + (y(i2,j,1,ie)-y(i1,j,1,ie))**2 )
               wsum = wsum + weight
            enddo
            lr2     = lr2/wsum
            lr(ie)  = 1./sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do i=1,nx
               weight = w(i)
c              ls2  = ls2  + ( (x(i,j2,1,ie)-x(i,j1,1,ie))**2
c    $                     +   (y(i,j2,1,ie)-y(i,j1,1,ie))**2 )
c    $                     *   weight
               ls2  = ls2  + weight /
     $                       ( (x(i,j2,1,ie)-x(i,j1,1,ie))**2
     $                     +   (y(i,j2,1,ie)-y(i,j1,1,ie))**2 )
               wsum = wsum + weight
            enddo
            ls2     = ls2/wsum
            ls(ie)  = 1./sqrt(ls2)
c           write(6,*) 'lrls',ie,lr(ie),ls(ie)
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine plane_space2(lr,ls,lt,i1,w,x,y,z,nx,nxn,nz0,nzn)
c
c     Here, the local spacing is already given in the surface term.
c     This addition made to simplify the periodic bdry treatment.
c
c
      include 'SIZE'
      include 'INPUT'
c
      real w(1),lr(1),ls(1),lt(1)
      real x(0:nxn,0:nxn,nz0:nzn,1)
      real y(0:nxn,0:nxn,nz0:nzn,1)
      real z(0:nxn,0:nxn,nz0:nzn,1)
      real lr2,ls2,lt2
c                                    __ __ __
c     Now, for each element, compute lr,ls,lt between specified planes
c
      ny = nx
      nz = nx
      j1 = i1
      k1 = i1
c
      do ie=1,nelv
c
         if (if3d) then
            lr2  = 0.
            wsum = 0.
            do k=1,nz
            do j=1,ny
               weight = w(j)*w(k)
               lr2  = lr2  + ( (x(i1,j,k,ie))**2
     $                     +   (y(i1,j,k,ie))**2
     $                     +   (z(i1,j,k,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            lr2     = lr2/wsum
            lr(ie)  = sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do k=1,nz
            do i=1,nx
               weight = w(i)*w(k)
               ls2  = ls2  + ( (x(i,j1,k,ie))**2
     $                     +   (y(i,j1,k,ie))**2
     $                     +   (z(i,j1,k,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            ls2     = ls2/wsum
            ls(ie)  = sqrt(ls2)
c
            lt2 = 0.
            wsum = 0.
            do j=1,ny
            do i=1,nx
               weight = w(i)*w(j)
               lt2  = lt2  + ( (x(i,j,k1,ie))**2
     $                     +   (y(i,j,k1,ie))**2
     $                     +   (z(i,j,k1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            enddo
            lt2     = lt2/wsum
            lt(ie)  = sqrt(lt2)
c           write(6,1) 'lrlslt',ie,lr(ie),ls(ie),lt(ie)
    1       format(a6,i5,1p3e12.4)
c
         else
            lr2 = 0.
            wsum = 0.
            do j=1,ny
               weight = w(j)
               lr2  = lr2  + ( (x(i1,j,1,ie))**2
     $                     +   (y(i1,j,1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            lr2     = lr2/wsum
            lr(ie)  = sqrt(lr2)
c
            ls2 = 0.
            wsum = 0.
            do i=1,nx
               weight = w(i)
               ls2  = ls2  + ( (x(i,j1,1,ie))**2
     $                     +   (y(i,j1,1,ie))**2 )
     $                     *   weight
               wsum = wsum + weight
            enddo
            ls2     = ls2/wsum
            ls(ie)  = sqrt(ls2)
c           write(6,*) 'lrls',ie,lr(ie),ls(ie),lt(ie)
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_fem(s,lam,n,lbc,rbc,ll,lm,lr,z,nz,e)
      real s(1),lam(1),ll,lm,lr,z(1)
      integer lbc,rbc,e
c
      parameter (m=100)
      real dx(0:m)
      integer icalld
      save    icalld
      data    icalld/0/
c
      icalld=icalld+1
c
      if (nz.gt.m-3) then
         write(6,*) 'ABORT. Error in set_up_fast_1D_fem. Increase m to'
     $             , nz
         call exitt
      endif
c
c     In the present scheme, each element is viewed as a d-fold
c     tensor of (1+nz+1) arrays, even if funky bc's are applied 
c     on either end of the 1D array.
c
      n = nz+2
c
c     Compute spacing, dx()
c
      call set_up_1D_geom(dx,lbc,rbc,ll,lm,lr,z,nz)
c
      nn1 = n*n + 1
      call gen_eigs_A_fem(s,s(nn1),lam,n,dx,lbc,rbc)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_1D_geom(dx,lbc,rbc,ll,lm,lr,z,nz)
c
      real dx(0:1),ll,lm,lr,z(2)
      integer lbc,rbc
c
c
c     Set up the 1D geometry for the tensor-product based overlapping Schwarz
c
c     Upon return: 
c
c       dx() contains the spacing required to set up the stiffness matrix.  
c
c
c     Input:
c
c       lbc (rbc) is 0 if the left (right) BC is Dirichlet, 1 if Neumann. 
c
c       ll is the space between the left-most Gauss point of the middle
c          element and the right-most Gauss point of the LEFT element
c
c       lm is the space between the left-most Gauss point of the middle
c          element and the right-most Gauss point of the MIDDLE element
c
c       lr is the space between the right-most Gauss point of the middle
c          element and the left-most Gauss point of the RIGHT element
c
c       --- if ll (lr) is very small (0), it indicates that there is no
c           left (right) spacing, and that the left (right) BC is Neumann.
c
c
c       z() is the array of nz Gauss points on the interval ]-1,1[.
c
c     Boundary conditions:
c
c     bc = 0  -- std. Dirichlet bc applied 2 points away from interior
c     bc = 1  -- Dirichlet bc applied 1 point away from interior (outflow)
c     bc = 2  -- Neumann bc applied on interior point (W,v,V,SYM,...)
c
c
c
c     Geometry:
c
c
c        dx0       dx1   dx2    dx3   dx5    dx5        dx6
c
c    bl        |<--ll-->|<------lm------>|<---lr--->|           br
c     0--------x-----|--x---x--------x---x--|-------x-----------0
c
c       left elem.         middle elem.         right elem.
c                   -1                     +1
c
c
c    "bl" = (extrapolated) location of Gauss point lx2-1 in left elem.
c
c    "br" = (extrapolated) location of Gauss point 2 in right elem.
c
c    Overlapping Schwarz applied with homogeneous Dirichlet boundary
c    conditions at "bl" and "br", and with a single d.o.f. extending
c    in to each adjacent domain.
c
      eps = 1.e-5
      call rone(dx,nz+3)
c
c     Middle
      scale = lm/(z(nz)-z(1))
      do i=1,nz-1
         dx(i+1) = (z(i+1)-z(i))*scale
      enddo

c     Left end
      if (lbc.eq.0) then
         dzm0   = z(1) + 1.
         dxm0   = scale*dzm0
         dxln   = ll - dxm0
         scalel = dxln/dzm0
         dx(0)  = scalel*(z(2)-z(1))
         dx(1)  = ll
      elseif (lbc.eq.1) then
         dx(1)  = ll
      endif
c
c     Right end
      if (rbc.eq.0) then
         dzm0      = z(1) + 1.
         dxm0      = scale*dzm0
         dxr0      = lr - dxm0
         scaler    = dxr0/dzm0
         dx(nz+1)  = lr
         dx(nz+2)  = scaler*(z(2)-z(1))
      elseif (rbc.eq.1) then
         dx(nz+1)  = lr
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_eigs_A_fem(sf,sft,atd,n,l,lbc,rbc)
c
c     Set up Fast diagonalization solver for FEM on mesh 2
c
      real sf(n,n),sft(n,n),atd(1),l(0:1)
      integer lbc,rbc
c
      parameter (m=100)
      real atu(m),ad(m),au(m),c(m),bh(m),li(0:m)
c
      if (n.gt.m) then
         write(6,*) 'ABORT. Error in gen_eigs_A_fem. Increase m to',n
         call exitt
      endif
c
c     Get delta x's
c
      do i=0,n
         li(i) = 1.0/l(i)
      enddo
c                          ^   ^
c     Fill initial arrays, A & B:
c
      call rzero(ad,n)
      call rzero(au,n-1)
      call rzero(bh,n)
c
      ie1 = lbc
      ien = n-rbc
      do ie=ie1,ien
c
c        il,ir are the left and right endpts of element ie.
         il = ie
         ir = ie+1
c
         if (ie.gt.0) ad(il) = ad(il) +       li(ie)
         if (ie.lt.n) ad(ir) = ad(ir) +       li(ie)
         if (ie.gt.0) au(il) =        -       li(ie)
         if (ie.gt.0) bh(il) = bh(il) + 0.5 * l(ie)
         if (ie.lt.n) bh(ir) = bh(ir) + 0.5 * l(ie)
      enddo
c
c     Take care of bc's (using blasting)
      bhm = vlmax(bh(2),n-2)/(n-2)
      ahm = vlmax(ad(2),n-2)/(n-2)
c
      if (lbc.gt.0) then
         au(1) = 0.
         ad(1) = ahm
         bh(1) = bhm
      endif
c
      if (rbc.gt.0) then
         au(n-1) = 0.
         ad(n  ) = ahm
         bh(n  ) = bhm
      endif
c
c
      do i=1,n
         c(i) = sqrt(1.0/bh(i))
      enddo
c                                        ~
c     Scale rows and columns of A by C:  A = CAC
c
      do i=1,n
         atd(i) = c(i)*ad(i)*c(i)
      enddo
c
c     Scale upper diagonal
c
      atu(1) = 0.
      do i=1,n-1
         atu(i) = c(i)*au(i)*c(i+1)
      enddo
c                                             ~
c     Compute eigenvalues and eigenvectors of A
c
      call calcz(atd,atu,n,dmax,dmin,sf,ierr)
      if (ierr.eq.1) then
         nid = mynode()
         write(6,6) nid,' czfail:',(l(k),k=0,n)
    6    format(i5,a8,1p16e10.2)
         call exitt
      endif
c
c     Sort eigenvalues and vectors
c
      call sort(atd,atu,n)
      call transpose(sft,n,sf,n)
      do j=1,n
         call swap(sft(1,j),atu,n,au)
      enddo
      call transpose(sf,n,sft,n)
c
c     Make "like" eigenvectors of same sign (for ease of diagnostics)
c
      do j=1,n
         avg = vlsum(sf(1,j),n)
         if (avg.lt.0) call chsign(sf(1,j),n)
      enddo
c
c     Clean up zero eigenvalue
c
      eps = 1.0e-6*dmax
      do i=1,n
c        if (atd(i).lt.eps) 
c    $      write(6,*) 'Reset Eig in gen_Afem:',i,n,atd(i)
         if (atd(i).lt.eps) atd(i) = 0.0
      enddo
c
c     scale eigenvectors by C:
c
      do i=1,n
         do j=1,n
            sf(i,j) = sf(i,j)*c(i)
         enddo
      enddo
c                                        ^
c     Orthonormalize eigenvectors w.r.t. B inner-product
c
      do j=1,n
         alpha = vlsc3(bh,sf(1,j),sf(1,j),n)
         alpha = 1.0/sqrt(alpha)
         call cmult(sf(1,j),alpha,n)
      enddo
c
c     Diagnostics
c
c     do j=1,n
c        do i=1,n
c           sft(i,j) = vlsc3(bh,sf(1,i),sf(1,j),n)
c        enddo
c     enddo
c
c     n8 = min(n,8)
c     do i=1,n
c        write(6,2) (sft(i,j),j=1,n8)
c     enddo
c   2 format('Id:',1p8e12.4)
c 
      call transpose(sft,n,sf,n)
      return
      end
c-----------------------------------------------------------------------
      subroutine get_fast_bc(lbr,rbr,lbs,rbs,lbt,rbt,e,bsym,ierr)
      integer                lbr,rbr,lbs,rbs,lbt,rbt,e,bsym
c
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'TOPOL'
      include 'TSTEP'
c
      integer fbc(6)
c
c     ibc = 0  <==>  Dirichlet
c     ibc = 1  <==>  Dirichlet, outflow (no extension)
c     ibc = 2  <==>  Neumann,   


      do iface=1,2*ldim
         ied = eface(iface)
         ibc = -1

         if (ifmhd) call mhd_bc_dn(ibc,iface,e) ! can be overwritten by 'mvn'

         if (cbc(ied,e,ifield).eq.'   ') ibc = 0
         if (cbc(ied,e,ifield).eq.'E  ') ibc = 0
         if (cbc(ied,e,ifield).eq.'msi') ibc = 0
         if (cbc(ied,e,ifield).eq.'MSI') ibc = 0
         if (cbc(ied,e,ifield).eq.'P  ') ibc = 0
         if (cbc(ied,e,ifield).eq.'p  ') ibc = 0
         if (cbc(ied,e,ifield).eq.'O  ') ibc = 1
         if (cbc(ied,e,ifield).eq.'ON ') ibc = 1
         if (cbc(ied,e,ifield).eq.'o  ') ibc = 1
         if (cbc(ied,e,ifield).eq.'on ') ibc = 1
         if (cbc(ied,e,ifield).eq.'MS ') ibc = 1
         if (cbc(ied,e,ifield).eq.'ms ') ibc = 1
         if (cbc(ied,e,ifield).eq.'MM ') ibc = 1
         if (cbc(ied,e,ifield).eq.'mm ') ibc = 1
         if (cbc(ied,e,ifield).eq.'mv ') ibc = 2
         if (cbc(ied,e,ifield).eq.'mvn') ibc = 2
         if (cbc(ied,e,ifield).eq.'v  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'V  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'W  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'SYM') ibc = bsym
         if (cbc(ied,e,ifield).eq.'SL ') ibc = 2
         if (cbc(ied,e,ifield).eq.'sl ') ibc = 2
         if (cbc(ied,e,ifield).eq.'SHL') ibc = 2
         if (cbc(ied,e,ifield).eq.'shl') ibc = 2
         if (cbc(ied,e,ifield).eq.'A  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'S  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'s  ') ibc = 2
         if (cbc(ied,e,ifield).eq.'J  ') ibc = 0
         if (cbc(ied,e,ifield).eq.'SP ') ibc = 0

         fbc(iface) = ibc

         if (ierr.eq.-1) write(6,1) ibc,ied,e,ifield,cbc(ied,e,ifield)
  1      format(2i3,i8,i3,2x,a3,'  get_fast_bc_error')

      enddo

      if (ierr.eq.-1) call exitti('Error A get_fast_bc$',e)

      lbr = fbc(1)
      rbr = fbc(2)
      lbs = fbc(3)
      rbs = fbc(4)
      lbt = fbc(5)
      rbt = fbc(6)

      ierr = 0 
      if (ibc.lt.0) ierr = lglel(e)

c     write(6,6) e,lbr,rbr,lbs,rbs,(cbc(k,e,ifield),k=1,4)
c   6 format(i5,2x,4i3,3x,4(1x,a3),'  get_fast_bc')

      return
      end
c-----------------------------------------------------------------------
      subroutine outv(x,n,name3)
      character*3 name3
      real x(1)
c
      nn = min (n,10)
      write(6,6) name3,(x(i),i=1,nn)
    6 format(a3,10f12.6)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6
c
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      n12 = min(n,12)
      do i=1,m
         write(6,6) ie,name6,(a(i,j),j=1,n12)
      enddo
    6 format(i3,1x,a6,12f9.5)
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_fem_ax
     $   (s,lam,n,lbc,rbc,ll,lm,lr,z,y,nz,ie)
      real s(1),lam(1),ll,lm,lr,z(1),y(1)
      integer lbc,rbc
c
      parameter (m=100)
      real dx(0:m)
      integer icalld
      save    icalld
      data    icalld/0/
c
      icalld=icalld+1
c
      if (nz.gt.m-3) then
         write(6,*) 'ABORT. Error in set_up_fast_1D_fem. Increase m to'
     $             , nz
         call exitt
      endif
c
c     In the present scheme, each element is viewed as a d-fold
c     tensor of (1+nz+1) arrays, even if funky bc's are applied 
c     on either end of the 1D array.
c
      n = nz+2
c
c     Compute spacing, dx()
c
      call set_up_1D_geom_ax(dx,lbc,rbc,ll,lm,lr,z,y,nz)
c
      nn1 = n*n + 1
      call gen_eigs_A_fem_ax(s,s(nn1),lam,n,dx,y,lbc,rbc)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_1D_geom_ax(dx,lbc,rbc,ll,lm,lr,z,y,nz)
c
      real dx(0:1),ll,lm,lr,z(2),y(1)
      integer lbc,rbc
c
c
c     Set up the 1D geometry for the tensor-product based overlapping Schwarz
c
c     Upon return: 
c
c       dx() contains the spacing required to set up the stiffness matrix.  
c
c
c     Input:
c
c       lbc (rbc) is 0 if the left (right) BC is Dirichlet, 1 if Neumann. 
c
c       ll is the space between the left-most Gauss point of the middle
c          element and the right-most Gauss point of the LEFT element
c
c       lm is the space between the left-most Gauss point of the middle
c          element and the right-most Gauss point of the MIDDLE element
c
c       lr is the space between the right-most Gauss point of the middle
c          element and the left-most Gauss point of the RIGHT element
c
c       --- if ll (lr) is very small (0), it indicates that there is no
c           left (right) spacing, and that the left (right) BC is Neumann.
c
c
c       z() is the array of nz Gauss points on the interval ]-1,1[.
c
c     Boundary conditions:
c
c     bc = 0  -- std. Dirichlet bc applied 2 points away from interior
c     bc = 1  -- Dirichlet bc applied 1 point away from interior (outflow)
c     bc = 2  -- Neumann bc applied on interior point (W,v,V,SYM,...)
c
c
c
c     Geometry:
c
c
c        dx0       dx1   dx2    dx3   dx5    dx5        dx6
c
c    bl        |<--ll-->|<------lm------>|<---lr--->|           br
c     0--------x-----|--x---x--------x---x--|-------x-----------0
c
c       left elem.         middle elem.         right elem.
c                   -1                     +1
c
c
c    "bl" = (extrapolated) location of Gauss point lx2-1 in left elem.
c
c    "br" = (extrapolated) location of Gauss point 2 in right elem.
c
c    Overlapping Schwarz applied with homogeneous Dirichlet boundary
c    conditions at "bl" and "br", and with a single d.o.f. extending
c    in to each adjacent domain.
c
      eps = 1.e-5
      call rone(dx,nz+3)
c
c     Middle
      scale = lm/(z(nz)-z(1))
      do i=1,nz-1
         dx(i+1) = (z(i+1)-z(i))*scale
      enddo

c     Left end
      if (lbc.eq.0) then
         dzm0   = z(1) + 1.
         dxm0   = scale*dzm0
         dxln   = ll - dxm0
         scalel = dxln/dzm0
         dx(0)  = scalel*(z(2)-z(1))
         dx(1)  = ll
      elseif (lbc.eq.1) then
         dx(1)  = ll
      endif
c
c     Right end
      if (rbc.eq.0) then
         dzm0      = z(1) + 1.
         dxm0      = scale*dzm0
         dxr0      = lr - dxm0
         scaler    = dxr0/dzm0
         dx(nz+1)  = lr
         dx(nz+2)  = scaler*(z(2)-z(1))
      elseif (rbc.eq.1) then
         dx(nz+1)  = lr
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_eigs_A_fem_ax(sf,sft,atd,n,l,y,lbc,rbc)
c
c     Set up Fast diagonalization solver for FEM on mesh 2
c
      real sf(n,n),sft(n,n),atd(1),l(0:1),y(1)
      integer lbc,rbc
c
      parameter (m=100)
      real atu(m),ad(m),au(m),c(m),bh(m),li(0:m)
c
      if (n.gt.m) then
         write(6,*) 'ABORT. Error in gen_eigs_A_fem. Increase m to',n
         call exitt
      endif
c
c     Get delta x's
c
      do i=0,n
         li(i) = 1.0/l(i)
      enddo
c                          ^   ^
c     Fill initial arrays, A & B:
c
      call rzero(ad,n)
      call rzero(au,n-1)
      call rzero(bh,n)
c
      ie1 = lbc
      ien = n-rbc
      do ie=ie1,ien
c
c        il,ir are the left and right endpts of element ie.
         il = ie
         ir = ie+1
c
         if (ie.gt.0) ad(il) = ad(il) +       li(ie)
         if (ie.lt.n) ad(ir) = ad(ir) +       li(ie)
         if (ie.gt.0) au(il) =        -       li(ie)
         if (ie.gt.0) bh(il) = bh(il) + 0.5 * l(ie)
         if (ie.lt.n) bh(ir) = bh(ir) + 0.5 * l(ie)
      enddo
c
c     Take care of bc's (using blasting)
      bhm = vlmax(bh(2),n-2)/(n-2)
      ahm = vlmax(ad(2),n-2)/(n-2)
c
      if (lbc.gt.0) then
         au(1) = 0.
         ad(1) = ahm
         bh(1) = bhm
      endif
c
      if (rbc.gt.0) then
         au(n-1) = 0.
         ad(n  ) = ahm
         bh(n  ) = bhm
      endif
c
c
      do i=1,n
         c(i) = sqrt(1.0/bh(i))
      enddo
c                                        ~
c     Scale rows and columns of A by C:  A = CAC
c
      do i=1,n
         atd(i) = c(i)*ad(i)*c(i)
      enddo
c
c     Scale upper diagonal
c
      atu(1) = 0.
      do i=1,n-1
         atu(i) = c(i)*au(i)*c(i+1)
      enddo
c                                             ~
c     Compute eigenvalues and eigenvectors of A
c
      call calcz(atd,atu,n,dmax,dmin,sf,ierr)
      if (ierr.eq.1) then
         nid = mynode()
         write(6,6) nid,' czfail2:',(l(k),k=0,n)
    6    format(i5,a8,1p16e10.2)
         call exitt
      endif
c
c     Sort eigenvalues and vectors
c
      call sort(atd,atu,n)
      call transpose(sft,n,sf,n)
      do j=1,n
         call swap(sft(1,j),atu,n,au)
      enddo
      call transpose(sf,n,sft,n)
c
c     Make "like" eigenvectors of same sign (for ease of diagnostics)
c
      do j=1,n
         avg = vlsum(sf(1,j),n)
         if (avg.lt.0) call chsign(sf(1,j),n)
      enddo
c
c     Clean up zero eigenvalue
c
      eps = 1.0e-6*dmax
      do i=1,n
c        if (atd(i).lt.eps) 
c    $      write(6,*) 'Reset Eig in gen_Afm_ax:',i,n,atd(i)
         if (atd(i).lt.eps) atd(i) = 0.0
      enddo
c
c     scale eigenvectors by C:
c
      do i=1,n
         do j=1,n
            sf(i,j) = sf(i,j)*c(i)
         enddo
      enddo
c                                        ^
c     Orthonormalize eigenvectors w.r.t. B inner-product
c
      do j=1,n
         alpha = vlsc3(bh,sf(1,j),sf(1,j),n)
         alpha = 1.0/sqrt(alpha)
         call cmult(sf(1,j),alpha,n)
      enddo
c
c     Diagnostics
c
c     do j=1,n
c        do i=1,n
c           sft(i,j) = vlsc3(bh,sf(1,i),sf(1,j),n)
c        enddo
c     enddo
c
c     n8 = min(n,8)
c     do i=1,n
c        write(6,2) (sft(i,j),j=1,n8)
c     enddo
c   2 format('Id:',1p8e12.4)
c 
      call transpose(sft,n,sf,n)
      return
      end
c-----------------------------------------------------------------------
      subroutine load_semhat_weighted    !   Fills the SEMHAT arrays
c
c     Note that this routine performs the following matrix multiplies
c     after getting the matrices back from semhat:
c
c     dgl = bgl dgl
c     jgl = bgl jgl
c
      include 'SIZE'
      include 'SEMHAT'
c
      nr = lx1-1
      call semhat(ah,bh,ch,dh,zh,dph,jph,bgl,zglhat,dgl,jgl,nr,wh)
      call do_semhat_weight(jgl,dgl,bgl,nr)
c
      return
      end
c----------------------------------------------------------------------
      subroutine do_semhat_weight(jgl,dgl,bgl,n)
      real bgl(1:n-1),jgl(1:n-1,0:n),dgl(1:n-1,0:n)

      do j=0,n
         do i=1,n-1
            jgl(i,j)=bgl(i)*jgl(i,j)
         enddo
      enddo
      do j=0,n
         do i=1,n-1
            dgl(i,j)=bgl(i)*dgl(i,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine semhat(a,b,c,d,z,dgll,jgll,bgl,zgl,dgl,jgl,n,w)
c
c     Generate matrices for single element, 1D operators:
c
c        a    = Laplacian
c        b    = diagonal mass matrix
c        c    = convection operator b*d
c        d    = derivative matrix
c        dgll = derivative matrix,    mapping from pressure nodes to velocity
c        jgll = interpolation matrix, mapping from pressure nodes to velocity
c        z    = GLL points
c
c        zgl  = GL points
c        bgl  = diagonal mass matrix on GL
c        dgl  = derivative matrix,    mapping from velocity nodes to pressure
c        jgl  = interpolation matrix, mapping from velocity nodes to pressure
c
c        n    = polynomial degree (velocity space)
c        w    = work array of size 2*n+2
c
c     Currently, this is set up for pressure nodes on the interior GLL pts.
c
c
      real a(0:n,0:n),b(0:n),c(0:n,0:n),d(0:n,0:n),z(0:n)
      real dgll(0:n,1:n-1),jgll(0:n,1:n-1)
c
      real bgl(1:n-1),zgl(1:n-1)
      real dgl(1:n-1,0:n),jgl(1:n-1,0:n)
c
      real w(0:2*n+1)
c
      np = n+1
      nm = n-1
      n2 = n-2
c
      call zwgll (z,b,np)
c
      do i=0,n
         call fd_weights_full(z(i),z,n,1,w)
         do j=0,n
            d(i,j) = w(j+np)                   !  Derivative matrix
         enddo
      enddo

      if (n.eq.1) return                       !  No interpolation for n=1

      do i=0,n
         call fd_weights_full(z(i),z(1),n2,1,w(1))
         do j=1,nm
            jgll(i,j) = w(j   )                  !  Interpolation matrix
            dgll(i,j) = w(j+nm)                  !  Derivative    matrix
         enddo
      enddo
c
      call rzero(a,np*np)
      do j=0,n
      do i=0,n
         do k=0,n
            a(i,j) = a(i,j) + d(k,i)*b(k)*d(k,j)
         enddo
         c(i,j) = b(i)*d(i,j)
      enddo
      enddo
c
      call zwgl (zgl,bgl,nm)
c
      do i=1,n-1
         call fd_weights_full(zgl(i),z,n,1,w)
         do j=0,n
            jgl(i,j) = w(j   )                  !  Interpolation matrix
            dgl(i,j) = w(j+np)                  !  Derivative    matrix
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine fd_weights_full(xx,x,n,m,c)
c
c     This routine evaluates the derivative based on all points
c     in the stencils.  It is more memory efficient than "fd_weights"
c
c     This set of routines comes from the appendix of 
c     A Practical Guide to Pseudospectral Methods, B. Fornberg
c     Cambridge Univ. Press, 1996.   (pff)
c
c     Input parameters:
c       xx -- point at wich the approximations are to be accurate
c       x  -- array of x-ordinates:   x(0:n)
c       n  -- polynomial degree of interpolant (# of points := n+1)
c       m  -- highest order of derivative to be approxxmated at xi
c
c     Output:
c       c  -- set of coefficients c(0:n,0:m).
c             c(j,k) is to be applied at x(j) when
c             the kth derivative is approxxmated by a 
c             stencil extending over x(0),x(1),...x(n).
c
c
      real x(0:n),c(0:n,0:m)
c
      c1       = 1.
      c4       = x(0) - xx
c
      do k=0,m
      do j=0,n
         c(j,k) = 0.
      enddo
      enddo
      c(0,0) = 1.
c
      do i=1,n
         mn = min(i,m)
         c2 = 1.
         c5 = c4
         c4 = x(i)-xx
         do j=0,i-1
            c3 = x(i)-x(j)
            c2 = c2*c3
            do k=mn,1,-1
               c(i,k) = c1*(k*c(i-1,k-1)-c5*c(i-1,k))/c2
            enddo
            c(i,0) = -c1*c5*c(i-1,0)/c2
            do k=mn,1,-1
               c(j,k) = (c4*c(j,k)-k*c(j,k-1))/c3
            enddo
            c(j,0) = c4*c(j,0)/c3
         enddo
         c1 = c2
      enddo
c     call outmat(c,n+1,m+1,'fdw',n)
      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_fast_1D_sem(s,lam,n,lbc,rbc,ll,lm,lr,ie)
      include 'SIZE'
      include 'SEMHAT'
c
      common /fast1dsem/ g(lr2),w(lr2)
c
      real g,w
      real s(1),lam(1),ll,lm,lr
      integer lbc,rbc
      
      integer bb0,bb1,eb0,eb1,n,n1
      logical l,r
      
      n=lx1-1
      !bcs on E are from normal vel component
      if(lbc.eq.2 .or. lbc.eq.3) then !wall,sym - dirichlet velocity
         eb0=1
      else !outflow,element - neumann velocity
         eb0=0
      endif
      if(rbc.eq.2 .or. rbc.eq.3) then !wall,sym - dirichlet velocity
         eb1=n-1
      else !outflow,element - neumann velocity
         eb1=n
      endif
      !bcs on B are from tangent vel component
      if(lbc.eq.2) then !wall - dirichlet velocity
         bb0=1
      else !outflow,element,sym - neumann velocity
         bb0=0
      endif
      if(rbc.eq.2) then !wall - dirichlet velocity
         bb1=n-1
      else !outflow,element,sym - neumann velocity
         bb1=n
      endif
c
      l = (lbc.eq.0)
      r = (rbc.eq.0)
c
c     calculate E tilde operator
      call set_up_fast_1D_sem_op(s,eb0,eb1,l,r,ll,lm,lr,bh,dgl,0)
c     call outmat(s,n+1,n+1,'  Et  ',ie)
c     calculate B tilde operator
      call set_up_fast_1D_sem_op(g,bb0,bb1,l,r,ll,lm,lr,bh,jgl,1)
c     call outmat(g,n+1,n+1,'  Bt  ',ie)
      
      n=n+1
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
      subroutine set_up_fast_1D_sem_op(g,b0,b1,l,r,ll,lm,lr,bh,jgl,jscl)
c            -1 T
c     G = J B  J
c
c     gives the inexact restriction of this matrix to
c     an element plus one node on either side
c
c     g - the output matrix
c     b0, b1 - the range for Bhat indices for the element
c              (enforces boundary conditions)
c     l, r - whether there is a left or right neighbor
c     ll,lm,lr - lengths of left, middle, and right elements
c     bh - hat matrix for B
c     jgl - hat matrix for J (should map vel to pressure)
c     jscl - how J scales
c            0: J = Jh
c            1: J = (L/2) Jh
c
c     result is inexact because:
c        neighbor's boundary condition at far end unknown
c        length of neighbor's neighbor unknown
c        (these contribs should be small for large N and
c         elements of nearly equal size)
c
      include 'SIZE'
      real g(0:lx1-1,0:lx1-1)
      real bh(0:lx1-1),jgl(1:lx2,0:lx1-1)
      real ll,lm,lr
      integer b0,b1
      logical l,r
      integer jscl
c
      real bl(0:lx1-1),bm(0:lx1-1),br(0:lx1-1)
      real gl,gm,gr,gll,glm,gmm,gmr,grr
      real fac
      integer n
      n=lx1-1
c
c
c     compute the scale factors for J      
      if (jscl.eq.0) then
         gl=1.
         gm=1.
         gr=1.
      elseif (jscl.eq.1) then
         gl=0.5*ll
         gm=0.5*lm
         gr=0.5*lr
      endif
      gll = gl*gl
      glm = gl*gm
      gmm = gm*gm
      gmr = gm*gr
      grr = gr*gr
c
c     compute the summed inverse mass matrices for
c     the middle, left, and right elements
      do i=1,n-1
         bm(i)=2. /(lm*bh(i))
      enddo
      if (b0.eq.0) then
         bm(0)=0.5*lm*bh(0)
         if(l) bm(0)=bm(0)+0.5*ll*bh(n)
         bm(0)=1. /bm(0)
      endif
      if (b1.eq.n) then
         bm(n)=0.5*lm*bh(n)
         if(r) bm(n)=bm(n)+0.5*lr*bh(0)
         bm(n)=1. /bm(n)
      endif
c     note that in computing bl for the left element,
c     bl(0) is missing the contribution from its left neighbor
      if (l) then
         do i=0,n-1
            bl(i)=2. /(ll*bh(i))
         enddo
         bl(n)=bm(0)
      endif
c     note that in computing br for the right element,
c     br(n) is missing the contribution from its right neighbor
      if (r) then
         do i=1,n
            br(i)=2. /(lr*bh(i))
         enddo
         br(0)=bm(n)
      endif
c      
      call rzero(g,(n+1)*(n+1))
      do j=1,n-1
         do i=1,n-1
            do k=b0,b1
               g(i,j) = g(i,j) + gmm*jgl(i,k)*bm(k)*jgl(j,k)
            enddo
         enddo
      enddo
c      
      if (l) then
         do i=1,n-1
            g(i,0) = glm*jgl(i,0)*bm(0)*jgl(n-1,n)
            g(0,i) = g(i,0)
         enddo
c        the following is inexact
c        the neighbors bc's are ignored, and the contribution
c        from the neighbor's neighbor is left out
c        that is, bl(0) could be off as noted above
c        or maybe i should go from 1 to n
         do i=0,n
            g(0,0) = g(0,0) + gll*jgl(n-1,i)*bl(i)*jgl(n-1,i)
         enddo
      else
         g(0,0)=1.
      endif
c      
      if (r) then
         do i=1,n-1
            g(i,n) = gmr*jgl(i,n)*bm(n)*jgl(1,0)
            g(n,i) = g(i,n)
         enddo
c        the following is inexact
c        the neighbors bc's are ignored, and the contribution
c        from the neighbor's neighbor is left out
c        that is, br(n) could be off as noted above
c        or maybe i should go from 0 to n-1
         do i=0,n
            g(n,n) = g(n,n) + grr*jgl(1,i)*br(i)*jgl(1,i)
         enddo
      else
         g(n,n)=1.
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine swap_lengths

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'WZ'
      common /swaplengths/ l(lx1,ly1,lz1,lelv)
      common /ctmpf/  lr(2*lx1+4),ls(2*lx1+4),lt(2*lx1+4)
     $              , llr(lelt),lls(lelt),llt(lelt)
     $              , lmr(lelt),lms(lelt),lmt(lelt)
     $              , lrr(lelt),lrs(lelt),lrt(lelt)
      real lr ,ls ,lt
      real llr,lls,llt
      real lmr,lms,lmt
      real lrr,lrs,lrt

      real l,l2d
      integer e

      n2 = lx1-1
      nz0 = 1
      nzn = 1
      nx  = lx1-2
      if (if3d) then
         nz0 = 0
         nzn = n2
      endif
      call plane_space(lmr,lms,lmt,0,n2,wxm1,xm1,ym1,zm1,nx,n2,nz0,nzn)

      n=n2+1
      if (if3d) then
         do e=1,nelv
         do j=2,n2
         do k=2,n2
            l(1,k,j,e) = lmr(e)
            l(n,k,j,e) = lmr(e)
            l(k,1,j,e) = lms(e)
            l(k,n,j,e) = lms(e)
            l(k,j,1,e) = lmt(e)
            l(k,j,n,e) = lmt(e)
         enddo
         enddo
         enddo
         call dssum(l,n,n,n)
         do e=1,nelv
            llr(e) = l(1,2,2,e)-lmr(e)
            lrr(e) = l(n,2,2,e)-lmr(e)
            lls(e) = l(2,1,2,e)-lms(e)
            lrs(e) = l(2,n,2,e)-lms(e)
            llt(e) = l(2,2,1,e)-lmt(e)
            lrt(e) = l(2,2,n,e)-lmt(e)
         enddo
      else
         do e=1,nelv
         do j=2,n2
            l(1,j,1,e) = lmr(e)
            l(n,j,1,e) = lmr(e)
            l(j,1,1,e) = lms(e)
            l(j,n,1,e) = lms(e)
c           call outmat(l(1,1,1,e),n,n,' L    ',e)
         enddo
         enddo
c        call outmat(l(1,1,1,25),n,n,' L    ',25)
         call dssum(l,n,n,1)
c        call outmat(l(1,1,1,25),n,n,' L    ',25)
         do e=1,nelv
c           call outmat(l(1,1,1,e),n,n,' L    ',e)
            llr(e) = l(1,2,1,e)-lmr(e)
            lrr(e) = l(n,2,1,e)-lmr(e)
            lls(e) = l(2,1,1,e)-lms(e)
            lrs(e) = l(2,n,1,e)-lms(e)
         enddo
      endif
      return
      end
c----------------------------------------------------------------------
      subroutine row_zero(a,m,n,e)
      integer m,n,e
      real a(m,n)
      do j=1,n
         a(e,j)=0.
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mhd_bc_dn(ibc,face,e)
      integer                  face,e
c
c     sets Neumann BC on pressure (ibc=2) for face and e(lement) except
c     when ifield normal component has (homogeneous) Neumann
c     boundary condition setting ibc=1 (i.e. Direchlet BC on pressure)
c
c     Note: 'SYM' on a plane with r,s,t-normal is 'dnn','ndn','nnd'? bsym?
c
      include 'SIZE'
      include 'TOPOL'
      include 'INPUT'
      include 'TSTEP'

      ied = eface(face)	! symmetric -> preprocessor notation
      nfc = face+1
      nfc = nfc/2	! = 1,2,3 for face 1 & 2,3 & 4,5 & 6

      if (indx1(cbc(ied,e,ifield),'d',1).gt.0)   ibc=2

      if (indx1(cbc(ied,e,ifield),'n',1).gt.nfc) ibc=1 ! 'n' for V_n

      return
      end
c-----------------------------------------------------------------------
