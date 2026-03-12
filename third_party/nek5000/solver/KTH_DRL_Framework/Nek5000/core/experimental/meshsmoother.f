      subroutine meshsmoother
      include 'SIZE'
      include 'TOTAL'

      parameter(ndfsbc=1)         !number of boundary conditions
      character*3 dfsbc(ndfsbc)
      save        dfsbc
      data        dfsbc /'W  '/  !BCs listed here

      idftyp = 0      !distance function - 0 -> exponential, 1-> tanh
      alpha = 15.     !Input for wall distance function 
      beta  = 0.1     !

      nouter = 50      !total loops around laplacian and optimizer smoothing
      nlap = 20        !number of laplacian iterations in each loop
      nopt = 20        !number of optimization iterations in each loop

      mtyp = 1         !metric type

      call smoothmesh(mtyp,nouter,nlap,nopt,ndfsbc,dfsbc,idftyp,alpha,
     $                 beta)
     
      return
      end
c-----------------------------------------------------------------------
      subroutine smoothmesh(mtyp,nouter,nlap,nopt,nbc,dcbc,
     $                      idftyp,alpha,beta)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      parameter (nxyzc=lxc*lyc*lzc,nxyz=lx1*ly1*lz1)

      real xmc(lxc,lyc,lzc,lelv),
     $     ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv),
     $     mltc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      real x8(2**ldim,lelv*(2**ldim)),
     $     y8(2**ldim,lelv*(2**ldim)),
     $     z8(2**ldim,lelv*(2**ldim)),
     $     nodmask(2**ldim,lelv*(2**ldim)),
     $     dis((2**ldim)*lelv*(2**ldim)),
     $     mlt((2**ldim)*lelv*(2**ldim))
      common /coarsemesh8/ x8,y8,z8

      real  dx(nxyzc,lelv),dy(nxyzc,lelv),dz(nxyzc,lelv),
     $      xbp(nxyzc,lelv),ybp(nxyzc,lelv),zbp(nxyzc,lelv)
      common / msmbackup / dx,dy,dz,xbp,ybp,zbp

      real dxm(nxyz,lelv),dym(nxyz,lelv),dzm(nxyz,lelv),
     $     dxn(nxyz,lelv),dyn(nxyz,lelv),dzn(nxyz,lelv)

      integer opt,gshl,gshlc,lapinv,optinv
      real alpha,beta,etstart, etend,f1sav,f1,f1pre
      character*3 dcbc(nbc)
c     -------------------
c     -------------------
ccc   INITIALIZE VARIABLES
      lapinv = 0
      optinv = 0

      if (nid.eq.0.and.loglevel.ge.5) 
     $            write(6,*) 'SMOOTHER-check original mesh'
      call fix_geom
c     Copy original mesh from xm1 to dxm
      call copy(dxm,xm1,lx1*ly1*lz1*nelv)
      call copy(dym,ym1,lx1*ly1*lz1*nelv)
      if (ldim.eq.3) call copy(dzm,zm1,lx1*ly1*lz1*nelv)

ccc   Generate interpolators for going to lx1 = 3
      call gen_int_lx1_to_3

ccc   COPY THE ORIGINAL MESH TO dx,dy,dz vectors
      call copy(dx,xmc,lxc*lyc*lzc*nelv)
      call copy(dy,ymc,lxc*lyc*lzc*nelv)
      if (ldim.eq.3) call copy(dz,zmc,lxc*lyc*lzc*nelv)
ccc   COPY THE ORIGINAL MESH FOR BACKUP IF MESH BECOMES INVERTED
      call copy(xbp,xmc,lxc*lyc*lzc*nelt)
      call copy(ybp,ymc,lxc*lyc*lzc*nelt)
      if (ldim.eq.3) call copy(zbp,zmc,lxc*lyc*lzc*nelt)

      call xmtox8(xmc,x8)
      call xmtox8(ymc,y8)
      if (ldim.eq.3) call xmtox8(zmc,z8)

ccc   CREATE MASK
      call genmask(nodmask,mlt,gshl,mltc,gshlc)

ccc   CONSTRUCT WEIGHT FUNCTION
      call disfun(dis,idftyp,alpha,beta,dcbc,nbc)

      etstart=dnekclock()
ccc   GET INITIAL ENERGY
      call getglobsum(x8,y8,z8,mlt,gshl,2**ldim,1,f1sav)
      if (nid.eq.0) 
     $   write(6,'(A,1p1e13.6)') 'SMOOTHER-initial energy',f1sav
      f1 = f1sav

ccc   START SMOOTHING HERE
      do j=1,nouter
         f1pre = f1
         if (nid.eq.0.and.loglevel.ge.2) 
     $     write(6,'(A,I5)') 'SMOOTHER-iteration',j
         mtyp = 1 !if mtyp = 1, jacobian, 2 = l^2, 3 - scale jacobian
         if (nlap.gt.0) then 
           call fastlap(nlap,nodmask,mlt,gshl,dis,lapinv,mltc,gshlc)
           if (lapinv.eq.1) nlap = 0
           if (lapinv.eq.1) call restbackmesh
         endif
         if (nopt.gt.0) then
           call optCG(nopt,nodmask,mlt,gshl,dis,mtyp,optinv,mltc,gshlc)
           if (optinv.eq.1) call restbackmesh !xbp->xm1 and xbp->x8
           if (optinv.eq.1) nopt = 0
         endif
        
         call getglobsum(x8,y8,z8,mlt,gshl,2**ldim,mtyp,f1)
  
         call xmtox8(xmc,x8)
         call xmtox8(ymc,y8)
         if (ldim.eq.3) call xmtox8(zmc,z8)

         call xmctoxm1
  
         if (nid.eq.0) write(6,'(A,I7,1p1e13.5)') 'SMOOTHER-energy',j,f1
         if (nid.eq.0.and.loglevel.ge.2) write(6,*) 'loop complete',j
         if (f1.ge.f1pre) goto 5001  !mesh didn't improve since last iteration
         if (nopt.eq.0.and.nlap.eq.0) goto 5001 !no iterations left to do
      enddo
 5001 continue
    
      etend=dnekclock()
ccc   RESTORE BOUNDARY LAYER
      call restbndrlayer(dx,dy,dz,dis,mltc,gshlc)  !dx,dy,dz is now actually smooth-coarse
      call fix_geom

ccc   Solve the Laplace's equation to morph the interior mesh to match
ccc   the actual surface mesh
      call opsub3(dxn,dyn,dzn,dxm,dym,dzm,xm1,ym1,zm1)
      call my_mv_mesh(dxn,dyn,dzn)
      call opadd2(xm1,ym1,zm1,dxn,dyn,dzn)

      call getglobsum(x8,y8,z8,mlt,gshl,2**ldim,mtyp,f1)
      if (nid.eq.0) then
        write(6,'(A,1p1e13.6)') 'SMOOTHER-initial energy ',f1sav
        write(6,'(A,1p1e13.6)') 'SMOOTHER-final energy ',f1
        write(6,'(A,f5.2)') 'SMOOTHER-improve % ',100.*(f1sav-f1)/f1sav
        write(6,'(A,1p1e13.6)') 'SMOOTHER-time taken ',etend-etstart
      endif

      call geom_reset(1)    ! recompute Jacobians, etc.
       
ccc   OUTPUT THE MESH
c      call gen_rea_full(1)                   !output the rea for smooth mesh

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_int_lx1_to_3 !interpolate mesh from lx1 to lx1=3
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real      dxmc(3,3),dymc(3,3),dzmc(3,3)
      real      dxtmc(3,3),dytmc(3,3),dztmc(3,3)
      common /coarseders/ dxmc,dymc,dzmc,dxtmc,dytmc,dztmc
      real      ixtmc3(lx1,3),ixmc3(3,lx1)
      real      ixtmf3(3,lx1),ixmf3(lx1,3)
      common /coarseints/ ixmc3,ixtmc3,ixmf3,ixtmf3
      real      zgmc(3),wxmc(3)     !points and weights
      real      zgmf(lx1),wxmf(lx1) !points and weights
      real      w(lx1,lx1)

c     Generate GLL points and weight
      call zwgll(zgmc,wxmc,3)
      call zwgll(zgmf,wxmf,lx1)
c     Generate derivative matrices with transpose operators
      call dgll(dxmc,dxtmc,zgmc,3,3)
      call dgll(dymc,dytmc,zgmc,3,3)
      call rzero(dzmc,3*3)
      call rzero(dztmc,3*3)
      if (ldim.eq.3) call dgll(dzmc,dztmc,zgmc,3,3)
c     Generate interpolation matrices
      call igllm (ixmc3,ixtmc3,zgmf,zgmc,nx1,3,nx1,3)
      call igllm (ixmf3,ixtmf3,zgmc,zgmf,3,nx1,3,nx1)

c     Interpolate mesh to lx1=3
      call xm1toxmc

      return
      end
c-----------------------------------------------------------------------
      subroutine int_fine_to_coarse_2d(u1,u2,n1,n2)
      include 'SIZE'
c     Interpolate u1 to u2 using interpolation matrices in12 and in12^T
c     u1 is size n1xn1 and u2 is size n2xn2
      real ixtmc3(lx1,3),ixmc3(3,lx1)
      real ixtmf3(3,lx1),ixmf3(lx1,3)
      common /coarseints/ ixmc3,ixtmc3,ixmf3,ixtmf3
      real u1(n1,n1),u2(n2,n2)
      real w(20,20)
 
      if (lx1.gt.20) write(6,*) 'SMOOTHER-increase size of work array 
     $                   in int_fine_to_coarse and related routines '
      if (lx1.gt.20) call exitt

      call mxm(ixmc3,n2,u1,n1,w,n1)
      call mxm(w,n2,ixtmc3,n1,u2,n2)
   
      return
      end
c-----------------------------------------------------------------------
      subroutine int_fine_to_coarse_3d(u1,u2,n1,n2)
      include 'SIZE'
c     Interpolate u1 to u2 using interpolation matrices in12 and in12^T
c     u1 is size n1xn1 and u2 is size n2xn2
      real ixtmc3(lx1,3),ixmc3(3,lx1)
      real ixtmf3(3,lx1),ixmf3(lx1,3)
      common /coarseints/ ixmc3,ixtmc3,ixmf3,ixtmf3
      real u1(n1,n1,n1),u2(n2,n2,n2)
      real w(20*20*20)
      real v(20*20*20)
 
      if (lx1.gt.20) write(6,*) 'SMOOTHER-increase size of work array 
     $                   in int_fine_to_coarse and related routines '
      if (lx1.gt.20) call exitt
      
      mm = n1*n1
      mn = n1*n2
      nn = n2*n2

      call mxm(ixmc3,n2,u1,n1,v ,mm)
      iv=1
      iw=1
      do k=1,n1
         call mxm(v(iv),n2,ixtmc3,n1,w(iw),n2)
         iv = iv+mn
         iw = iw+nn
      enddo
      call mxm(w,nn,ixtmc3,n1,u2,n2)
  
      return
      end
c-----------------------------------------------------------------------
      subroutine int_coarse_to_fine_2d(u1,u2,n1,n2)
      include 'SIZE'
c     Interpolate u1 to u2 using interpolation matrices in12 and in12^T
c     u1 is size n1xn1 and u2 is size n2xn2
c     u1 is fine, u2 is coarse
      real ixtmc3(lx1,3),ixmc3(3,lx1)
      real ixtmf3(3,lx1),ixmf3(lx1,3)
      common /coarseints/ ixmc3,ixtmc3,ixmf3,ixtmf3
      real zgmc(3),wxmc(3) !points and weights
      real zgmf(lx1),wxmf(lx1) !points and weights
      common /coarswz/ zgmc,wxmc,zgmf,wxmf
      real u1(n1,n1),u2(n2,n2)
      real w(20,20)

      if (lx1.gt.20) write(6,*) 'SMOOTHER-increase size of work array 
     $                   in int_coarse_to_fine and related routines '
      if (lx1.gt.20) call exitt

      call mxm(ixmf3,n1,u2,n2,w,n2)
      call mxm(w,n1,ixtmf3,n2,u1,n1)

      return
      end
c-----------------------------------------------------------------------
      subroutine int_coarse_to_fine_3d(u1,u2,n1,n2)
      include 'SIZE'
c     Interpolate u1 to u2 using interpolation matrices in12 and in12^T
c     u1 is size n1xn1 and u2 is size n2xn2
      real ixtmc3(lx1,3),ixmc3(3,lx1)
      real ixtmf3(3,lx1),ixmf3(lx1,3)
      common /coarseints/ ixmc3,ixtmc3,ixmf3,ixtmf3
      real u1(n1,n1,n1),u2(n2,n2,n2)
      real w(20*20*20)
      real v(20*20*20)

      if (lx1.gt.20) write(6,*) 'SMOOTHER-increase size of work array 
     $                   in int_coarse_to_fine and related routines '
      if (lx1.gt.20) call exitt

      mm = n2*n2
      mn = n2*n1
      nn = n1*n1

      call mxm(ixmf3,n1,u2,n2,v ,mm)
      iv=1
      iw=1
      do k=1,n2
         call mxm(v(iv),n1,ixtmf3,n2,w(iw),n1)
         iv = iv+mn
         iw = iw+nn
      enddo
      call mxm(w,nn,ixtmf3,n2,u1,n1)

      return
      end
c-----------------------------------------------------------------------
      subroutine genbackupmesh   !xm1->xbp   backup the mesh
      include 'SIZE'
      include 'TOTAL'
! backsup the mesh
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      parameter(lt = lxc*lyc*lzc*lelv)
      real dx(lt),dy(lt),dz(lt),xbp(lt),ybp(lt),zbp(lt)
      common / msmbackup / dx,dy,dz,xbp,ybp,zbp

      call copy(xbp,xmc,lxc*lyc*lzc*nelt)
      call copy(ybp,ymc,lxc*lyc*lzc*nelt)
      if (ldim.eq.3) call copy(zbp,zmc,lxc*lyc*lzc*nelt)

      return
      end
c-----------------------------------------------------------------------
      subroutine restbackmesh  !xbp->xm1 and xbp->x8
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      real x8(2,2,ldim-1,lelv*(2**ldim)),y8(2,2,ldim-1,lelv*(2**ldim))
      real z8(2,2,ldim-1,lelv*(2**ldim))
      common /coarsemesh8/ x8,y8,z8

      parameter(lt = lxc*lyc*lzc*lelv)
      real dx(lt),dy(lt),dz(lt),xbp(lt),ybp(lt),zbp(lt)
      common / msmbackup / dx,dy,dz,xbp,ybp,zbp
     
      call copy(xmc,xbp,lxc*lyc*lzc*nelt)
      call copy(ymc,ybp,lxc*lyc*lzc*nelt)
      if (ldim.eq.3) call copy(zmc,zbp,lxc*lyc*lzc*nelt)
      call xmtox8(xmc,x8)
      call xmtox8(ymc,y8)
      if (ldim.eq.3) call xmtox8(zmc,z8)

      return
      end
c-----------------------------------------------------------------------
      subroutine optCG(itmax,nodmask,mlt,gshl,dis,opt,optinv,mltc,gshlc)
      include 'SIZE'
      include 'TOTAL'

      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      parameter (nxyzc=lxc*lyc*lzc,nxyz=lx1*ly1*lz1)

      real xmc(lxc,lyc,lzc,lelv),
     $     ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv),
     $     mltc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      real x8(2**ldim,lelv*(2**ldim)),
     $     y8(2**ldim,lelv*(2**ldim)),
     $     z8(2**ldim,lelv*(2**ldim)),
     $     nodmask(2**ldim,lelv*(2**ldim)),
     $     dis(2**ldim,lelv*(2**ldim)),
     $     mlt(2**ldim,lelv*(2**ldim)),
     $     dx(2**ldim,lelv*(2**ldim)),
     $     dy(2**ldim,lelv*(2**ldim)),
     $     dz(2**ldim,lelv*(2**ldim)),
     $     dfdx(2**ldim,lelv*(2**ldim),ldim),
     $     dfdxn(2**ldim,lelv*(2**ldim),ldim),
     $     sk(2**ldim,lelv*(2**ldim),ldim)
      common /coarsemesh8/ x8,y8,z8

      integer iter,gshl,siz,eg,opt,optinv,kerr,gshlc
      real f2,lambda,num,den,scale2
      real alpha,bk,num1,den1,wrk1
      real hval(2**ldim,lelv*(2**ldim))

      optinv = 0 !initialize to 0

      if (nid.eq.0.and.loglevel.ge.2) 
     $         write(6,*) 'Optimization loop started'

      if (opt.eq.1) siz = 2**ldim !for jacobian
      if (opt.eq.2) siz = 4+8*(ldim-2) !for len
      n1 = 2**ldim
      n2 = nelv*(2**ldim)*n1

      call get_nodscale(hval,x8,y8,z8,gshl)
c      if (nid.eq.0.and.loglevel.ge.4) 
c     $   write(6,'(A,1p1e13.5)') 'perturbation amount is ',hval

      call gradf(f2,dfdx,x8,y8,z8,mlt,gshl,siz,opt,hval)
      do i=1,ldim
        call copy(sk(1,1,i),dfdx(1,1,i),n2)
        call cmult(sk(1,1,i),-1.,n2)
      enddo

      do iter=1,itmax
c  do line search at this point
         call
     $  dolsalpha(x8,y8,z8,sk,alpha,siz,opt,mlt,gshl,nodmask,iter)

         if (alpha.eq.0) goto 5002

         call col4(dx,sk(1,1,1),nodmask,dis,n2)
         call col4(dy,sk(1,1,2),nodmask,dis,n2)
         if (ldim.eq.3) call col4(dz,sk(1,1,ldim),nodmask,dis,n2)

         call add2s2(x8,dx,alpha,n2)
         call add2s2(y8,dy,alpha,n2)
         if (ldim.eq.3) call add2s2(z8,dz,alpha,n2)

c Gradf at new positions
         call gradf(f2,dfdxn,x8,y8,z8,mlt,gshl,siz,opt,hval)

c  get Bk = dfdxn'*dfdxn / (dfdx'*dfdx)
         num1 = 0.
         den1 = 0.
         do k=1,ldim
         do j=1,nelv*(2**ldim)
         do i=1,2**ldim
          num1 = dfdxn(i,j,k)*mlt(i,j)*dfdxn(i,j,k) + num1
          den1 = dfdx(i,j,k)*mlt(i,j)*dfdx(i,j,k) + den1
         enddo
         enddo
         enddo
         call gop(num1,wrk1,'+  ',1)
         call gop(den1,wrk1,'+  ',1)
         bk = num1/den1
  
         do k=1,ldim
         do j=1,nelv*(2**ldim)
         do i=1,2**ldim
          sk(i,j,k) = -dfdxn(i,j,k) + bk*sk(i,j,k)
          dfdx(i,j,k) = dfdxn(i,j,k)
         enddo
         enddo
         enddo
        
         if (mod(iter,5).eq.0.or.iter.eq.itmax) then
          call x8toxm(xmc,x8)
          call x8toxm(ymc,y8)
          if (ldim.eq.3) call x8toxm(zmc,z8)
          call fixcurs(mltc,gshlc)
          call xmtox8(xmc,x8)
          call xmtox8(ymc,y8)
          if (ldim.eq.3) call xmtox8(zmc,z8)
         endif
      enddo

 5002 continue
      if (alpha.eq.0) then
       optinv = 2
       call x8toxm(xmc,x8)
       call x8toxm(ymc,y8)
       if (ldim.eq.3) call x8toxm(zmc,z8)
       call fixcurs(mltc,gshlc)
       call xmtox8(xmc,x8)
       call xmtox8(ymc,y8)
       if (ldim.eq.3) call xmtox8(zmc,z8)
      endif

      call glmapm1chkinv(kerr)
      if (kerr.gt.0) then
        optinv = 1  !Terminate smoother
        if (nid.eq.0) write(6,*) 'Terminating optimizer'
      else
        call genbackupmesh
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine 
     $   dolsalpha(xe,ye,ze,sk,alpha,siz,opt,mlt,gshl,nodmask,iter)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real     xmc(lxc,lyc,lzc,lelv),
     $         ymc(lxc,lyc,lzc,lelv),
     $         zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc
      real     xe(2**ldim,lelv*(2**ldim)),
     $         ye(2**ldim,lelv*(2**ldim)),
     $         ze(2**ldim,lelv*(2**ldim)),
     $         dx(2**ldim,lelv*(2**ldim)),
     $         dy(2**ldim,lelv*(2**ldim)),
     $         dz(2**ldim,lelv*(2**ldim)),
     $         xs(2**ldim,lelv*(2**ldim)),
     $         ys(2**ldim,lelv*(2**ldim)),
     $         zs(2**ldim,lelv*(2**ldim)),
     $         nodmask(2**ldim,lelv*(2**ldim)),
     $         mlt(2**ldim,lelv*(2**ldim)),
     $         sk((2**ldim),lelv*(2**ldim),ldim),
     $         jac(2**ldim,lelv*(2**ldim))
      real alpha,wrk1,f1,f2
      integer siz,opt,z,gshl,invflag
      integer n1,n2,chk1,chk2,tstop,maxcount,iter

      common /lsinfo / lssteps
      real lssteps(5)
      integer icalld
      save    icalld
      data    icalld /0/
      integer idx
      save    idx   
      data    idx    /0/

      if (icalld.eq.0) then
        call rzero(lssteps,5)
        icalld = 1
        idx = 0
      endif

      if (idx.lt.6) then
        alpha = 1.0
      else
        dumsum = vlsum(lssteps,5)
        alpha = 2.*dumsum
      endif
      alphasav = alpha
      maxcount = 20
      chk1 = 0
      chk2 = 0
      tstop = 0
      n1 = 2**ldim
      n2 = nelv*(2**ldim)*n1

      call copy(xs(1,1),xe(1,1),n2)
      call copy(ys(1,1),ye(1,1),n2)
      if (ldim.eq.3) call copy(zs(1,1),ze(1,1),n2)

      call getglobsum(xs,ys,zs,mlt,gshl,siz,opt,f1)
      z = 0
      do while (tstop.eq.0.and.z.le.maxcount)
        z = z+1
        call col3c(dx,sk(1,1,1),nodmask,alpha,n2)
        call col3c(dy,sk(1,1,2),nodmask,alpha,n2)
        if (ldim.eq.3) call col3c(dz,sk(1,1,ldim),nodmask,alpha,n2)

        call add3(xs,xe,dx,n2)
        call add3(ys,ye,dy,n2)
        if (ldim.eq.3) call add3(zs,ze,dz,n2)

        call getglobsum(xs,ys,zs,mlt,gshl,siz,opt,f2)

        if (f2.lt.f1) then
          chk1 = 1
        endif
        call gop(chk1,wrk1,'m  ',1)

        chk2 = 0
        if (chk1.eq.1) then
          call x8toxm(xmc,xs)
          call x8toxm(ymc,ys)
          if (ldim.eq.3) call x8toxm(zmc,zs)
          call glmapm1chkinv(invflag)
          if (invflag.eq.0) chk2 = 1
        endif
        call gop(chk2,wrk1,'m  ',1)

        if (chk1.eq.1.and.chk2.eq.1.) then
           alphasav = alpha
           tstop = 1.
        else
           chk1 = 0.
           chk2 = 0.
           alpha = alpha/(10.)
        endif
      enddo

      if (idx.lt.6) then
        idx = idx+1
        lssteps(idx) = alphasav
      else
        lssteps(1) = lssteps(2)
        lssteps(2) = lssteps(3)
        lssteps(3) = lssteps(4)
        lssteps(4) = lssteps(5)
        lssteps(5) = alphasav
      endif

      if (tstop.eq.1) then 
        alpha = alphasav
        if (nid.eq.0.and.loglevel.ge.3) write(6,101) iter,f2
  101   format(i5,' glob_phi ',1p1e13.6)
c        write(6,*) z,'number of ls steps'
      else
        alpha = 0.
        if (nid.eq.0.and.loglevel.ge.4) 
     $       write(6,*) 'SMOOTHER-line-search alpha set to 0'
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine getglobsum(xe,ye,ze,mlt,gshl,siz,opt,fl)
      include 'SIZE'
      include 'TOTAL'
      real xe(2**ldim,lelv*(2**ldim))
      real ye(2**ldim,lelv*(2**ldim))
      real ze(2**ldim,lelv*(2**ldim))
      real mlt(2**ldim,lelv*(2**ldim))
      integer gshl,siz,e,opt
      real f1,fl,wrk1

      fl = 0.
      do e=1,nelv*(2**ldim)
        if (opt.eq.1) 
     $     call get_jac(f1,xe(1,e),ye(1,e),ze(1,e),siz,e,1)
        if (opt.eq.2) call get_len(f1,xe(1,e),ye(1,e),ze(1,e),siz)
        fl = fl+f1
      enddo
      call gop(fl,wrk1,'+  ',1)

      return
      end
c-----------------------------------------------------------------------
      subroutine disfun(dis,funtype,alpha,beta,dcbc,nbc)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real dd1(lx1*ly1*lz1*lelv),
     $     dd2(lx1*ly1*lz1*lelv),
     $     ddd(lx1*ly1*lz1*lelv),
     $     ddc(lxc*lyc*lzc,lelv),
     $     dis((2**ldim)*lelv*(2**ldim))
      integer i,nbc,j,funtype
      character*3 dcbc(nbc)
      real dscale,dmax,alpha,beta,dum2

      if (nid.eq.0.and.loglevel.ge.5) 
     $            write(6,*) 'calculate distance function'

      call rone (dd1,lx1*ly1*lz1*nelv)
      call sm_cheap_dist(dd1,1,dcbc(1))  ! Distance function scaling

      if (nbc.ge.2) then
      do i=2,nbc
         call rone (dd2,lx1*ly1*lz1*nelv)
         call sm_cheap_dist(dd2,1,dcbc(i))  ! Distance function scaling
         do j=1,lx1*ly1*lz1*nelv
           dd1(j) = min(dd1(j),dd2(j))
         enddo
      enddo
      endif

      dmax   = glamax(dd1,lx1*ly1*lz1*nelv)
      dscale = 1./dmax
      call cmult(dd1,dscale,lx1*ly1*lz1*nelv) !normalized 0 to 1
c      call outpost(dd1,vy,vz,pr,t,'   ')

      nxyz = lx1*ly1*lz1
      if (ldim.eq.2) then
       do i=1,nelv
         call int_fine_to_coarse_2d(dd1((i-1)*nxyz+1),ddc(1,i),lx1,lxc) 
       enddo
      else
       do i=1,nelv
         call int_fine_to_coarse_3d(dd1((i-1)*nxyz+1),ddc(1,i),lx1,lxc)
       enddo
      endif
      call xmtox8(ddc,dis)
c
      if (funtype.eq.0) then
        do i=1,(2**ldim)*nelv*(2**ldim)
          dis(i) = (1-EXP(-dis(i)/beta))
       enddo
      elseif (funtype.eq.1) then
        do i=1,(2**ldim)*nelv*(2**ldim)
          dis(i) = 0.5*(tanh(alpha*(dis(i)-beta))+1)
        enddo
      else
          call exitti('Please set the funtype to 0 or 1$',funtype)
      endif
        
      if (nid.eq.0.and.loglevel.ge.5) 
     $            write(6,'(1p1e13.4,A)') dmax,'max disfun'

      return
      end
c-----------------------------------------------------------------------
      subroutine restbndrlayer(dx,dy,dz,dis,mltc,gshlc)
      include 'SIZE'
      include 'TOTAL'
c     dis*smoothmesh + (1-dis)*original mesh
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),
     $     ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      real dx(lxc*lyc*lzc,lelv),
     $     dy(lxc*lyc*lzc,lelv),
     $     dz(lxc*lyc*lzc,lelv),
     $     dis((2**ldim)*lelv*(2**ldim)),
     $     dis2(lxc,lyc,lzc,lelv),
     $     mltc(lxc,lyc,lzc,lelv)

      integer gshlc

      call x8toxm(dis2,dis)
      dum2 = glamax(dis2,lxc*lyc*lzc*nelv)
      dum2 = 1./dum2
      call cmult(dis2,dum2,lxc*lyc*lzc*nelv)

      call col2(xmc,dis2,lxc*lyc*lzc*nelv)   !w*xs
      call col2(ymc,dis2,lxc*lyc*lzc*nelv)
      if (ndim.eq.3) call col2(zmc,dis2,lxc*lyc*lzc*nelv)

      call add2(xmc,dx,lxc*lyc*lzc*nelv)     !w*xs+xo
      call add2(ymc,dy,lxc*lyc*lzc*nelv)
      if (ndim.eq.3) call add2(zmc,dz,lxc*lyc*lzc*nelv)

      call col2(dx,dis2,lxc*lyc*lzc*nelv)    !w*xo
      call col2(dy,dis2,lxc*lyc*lzc*nelv)
      if (ndim.eq.3) call col2(dz,dis2,lxc*lyc*lzc*nelv)

      call sub2(xmc,dx,lxc*lyc*lzc*nelv)     !w*xs+xo-w*xo=w*xs+xo(1-w)
      call sub2(ymc,dy,lxc*lyc*lzc*nelv)     !this is now store in xmc..zmc
      if (ndim.eq.3) call sub2(zmc,dz,lxc*lyc*lzc*nelv)

      call sub2(dx,xmc,lxc*lyc*lzc*nelv)     !dx=xo-xmc
      call sub2(dy,ymc,lxc*lyc*lzc*nelv)
      if (ndim.eq.3) call sub2(dz,zmc,lxc*lyc*lzc*nelv)

      call fixcurs(mltc,gshlc)
      call xmctoxm1

      return
      end
c-----------------------------------------------------------------------
      subroutine masklayers(nodmask,nlayers)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real nodmask(2**ldim,lelv*(2**ldim))
      real d(lx1*ly1*lz1,lelv)
      real vcmask(lxc,lyc,lzc,lelv)
      integer e,f,i,j,nlayers,valm,c

      character*3 tcbc(5)
      save        tcbc
      data        tcbc /'W  ','v  ','V  ','mv ','MV ' /

      call x8toxm(v1mask,nodmask)

      call rone(d,nx1*ny1*nz1*nelv)
      do e=1,nelv
      do f=1,2*ldim
      do c=1,5
         if(cbc(f,e,1).eq.tcbc(c)) call facev(d,e,f,zero,nx1,ny1,nz1)
      enddo
      enddo
      enddo
      call dsop(d,'mul',nx1,ny1,nz1)

      do i=1,nlayers
        call dsop(d,'mul',nx1,ny1,nz1)
         do e=1,nelv
          val = glamin(d(1,e),lx1**ldim)
          call cmult(d(1,e),val,lx1**ldim)
         enddo
      enddo

      call dsop(d,'mul',nx1,ny1,nz1)
      call col2(v1mask,d,lx1*ly1*lz1*nelv)
      do i=1,nelv
       call int_fine_to_coarse_2d(v1mask(1,1,1,e),vcmask(1,1,1,e),lx1,3)
      enddo
      call xmtox8(vcmask,nodmask)

      return
      end
c-----------------------------------------------------------------------
      subroutine fixcurs(mltc,gshlc)
      include 'SIZE'
      include 'TOTAL'
c     This routine fixes the curved edges post smoothing
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),
     $     ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      common /ctmp0/ zg(3)
      real w1(lxc*lyc*lzc*lelv),w2(lxc*lyc*lzc*lelv)
      real vcmask(lxc,lyc,lzc,lelv)
      integer f,e,nedge,n,nfaces,gshlc
      integer ke(3,12)
      real    xyz(3,3)
      real    xd(lxc*lyc*lzc),
     $        yd(lxc*lyc*lzc),
     $        zd(lxc*lyc*lzc),
     $        mltc(lxc*lyc*lzc*lelv)
      real nxv,nyv,nzv

      save    ke
      data    ke /  1, 2, 3,    3, 6, 9,    9, 8, 7,    7, 4, 1
     $           , 19,20,21,   21,24,27,   27,26,25,   25,22,19
     $           ,  1,10,19,    3,12,21,    9,18,27,    7,16,25 /
       integer kin(7)

c     START BY LOOPING OVER EACH ELEMENT AND THEN OVER EACH EDGE
      nedge = 4 + 8*(ldim-2)
      nfaces = 2*ldim

      n      = lxc*lyc*lzc*nelv
      call rone  (vcmask,n)
      do e=1,nelv                
      do f=1,nfaces              
       if (cbc(f,e,1).ne.'E  '.and.cbc(f,e,1).ne.'   ') 
     $ call facev (vcmask,e,f,0.0,lxc,lyc,lzc)
      enddo
      enddo
      call fgslib_gs_op(gshlc,vcmask,1,2,0)

      do e=1,nelv
       do j=1,nedge
        call rzero(xyz,9)
        do i=1,3
         xyz(1,i) = xmc(ke(i,j),1,1,e)
         xyz(2,i) = ymc(ke(i,j),1,1,e)
         if (ldim.eq.3) xyz(3,i) = zmc(ke(i,j),1,1,e)
        enddo

        if (vcmask(ke(2,j),1,1,e).gt.0.5) then
         call fixedgs(xyz,nxv,nyv,nzv)
         xmc(ke(2,j),1,1,e) = nxv
         ymc(ke(2,j),1,1,e) = nyv
         if (ldim.eq.3) zmc(ke(2,j),1,1,e) = nzv
        endif
       enddo

c      fix face center and body center
        zg(1) = -1.
        zg(2) = 0.
        zg(3) = 1.
        call copy(xd,xmc(1,1,1,e),lxc*lyc*lzc)
        call copy(yd,ymc(1,1,1,e),lxc*lyc*lzc)
        if (ldim.eq.3) call copy(zd,zmc(1,1,1,e),lxc*lyc*lzc)
        call gh_face_extend(xd,zg,3,2,w1,w2)
        call gh_face_extend(yd,zg,3,2,w1,w2)
        if (ldim.eq.3) call gh_face_extend(zd,zg,3,2,w1,w2)
        kin(1) = 5
        kin(2) = 11
        kin(3) = 13
        kin(4) = 15
        kin(5) = 17
        kin(6) = 23
        kin(7) = 14
        do i=1,(ldim-2)*6+1
         xmc(kin(i),1,1,e) = xd(kin(i))
         ymc(kin(i),1,1,e) = yd(kin(i))
         if (ldim.eq.3) zmc(kin(i),1,1,e) = zd(kin(i))
        enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine fixedgs(xyz,nx,ny,nz)
      real xyz(3,0:2) ! Coordinates
      real x0(3),x1(3),x2(3),v2(3),v1(3),l02,l01,n1i,u1(3),u2(3)
      real nx,ny,nz
      real h,tol,h2

      call copy(x0,xyz(1,0),3)
      call copy(x1,xyz(1,1),3)
      call copy(x2,xyz(1,2),3)
                             !                  ^ x2
      call sub3(v1,x1,x0,3)  ! v1=x1-x0        /  \ v2     p1 = x0+v2*dot
      call sub3(v2,x2,x0,3)  ! v2=x2-x0    x1 .<-- x0
                             !                  v1
      l02 = vlsc2(v2,v2,3)   ! || x2-x0 ||
      if (l02.gt.0) then
         l02 = sqrt(l02)
         scl = 1./l02
         call cmult2(u2,v2,scl,3)  ! Unit tangent
      endif

      l01 = vlsc2(v1,v1,3)   ! || x1-x0 ||
      if (l01.gt.0) then
         l01 = sqrt(l01)
         scl = 1./l01
         call cmult2(u1,v1,scl,3)  ! Unit tangent
      endif

      dot = vlsc2(v1,u2,3)

      if (dot.le.0) then
         write(6,*) 'SMOOTHER-ERROR 1 IN SHIFT - YOU SHOULD ABORT'
         return
      elseif (dot.gt.l02) then
         write(6,*) 'SMOOTHER-ERROR 2 IN SHIFT - YOU SHOULD ABORT'
         return
      endif
         h = 0.
         h2 = 0.
         tol = 1.e-8
      do i=1,3
         p1i = x0(i) + u2(i)*dot   ! Projection of x1 onto [x0,x2]
         n1i = x1(i) - p1i         ! Normal vector
         h = h + n1i**2
         h2 = h2+ (x2(i)-x0(i))**2
         xmi = 0.5*(x0(i)+x2(i))
         x1(i) = xmi + n1i         ! X1 point shifted to be centered at midpoint
      enddo
         if (h.le.h2*tol) then
            x1(1) = 0.5*(x0(1)+x2(1))
            x1(2) = 0.5*(x0(2)+x2(2))
            x1(3) = 0.5*(x0(3)+x2(3))
         endif
         

      nx = x1(1)
      ny = x1(2)
      nz = x1(3)

      return
      end
c-----------------------------------------------------------------------
      subroutine gradf(f2,dfdx,x8,y8,z8,mlt,gshl,siz,opt,h)
      include 'SIZE'
      include 'TOTAL'
      real x8(2**ldim,lelv*(2**ldim)),
     $     y8(2**ldim,lelv*(2**ldim)),
     $     z8(2**ldim,lelv*(2**ldim)),
     $    mlt(2**ldim,lelv*(2**ldim)),
     $   dfdx(2**ldim,lelv*(2**ldim),ldim)
      real xt(2**ldim),
     $     yt(2**ldim),
     $     zt(2**ldim)
      integer siz,opt,vertex,gshl,e,eg,e0,f
      real par(siz)
      real f1,fl,gl,f2,h(2**ldim,lelv*(2**ldim))
 
      f1 = 0
      do e=1,nelv*(2**ldim)
      if (opt.eq.1) 
     $    call get_jac(fl,x8(1,e),y8(1,e),z8(1,e),siz,e,0)
      if (opt.eq.2) call get_len(fl,x8(1,e),y8(1,e),z8(1,e),siz)  
         f1 = f1+fl
         call copy(xt,x8(1,e),2**ldim)
         call copy(yt,y8(1,e),2**ldim)
         call copy(zt,z8(1,e),2**ldim)
         do j=1,2**ldim
            xt(j) = x8(j,e)+h(j,e) 
      if (opt.eq.1) call get_jac(gl,xt,yt,zt,siz,e,1)
      if (opt.eq.2) call get_len(gl,xt,yt,zt,siz)
            xt(j) = x8(j,e)-h(j,e) 
      if (opt.eq.1) call get_jac(fl,xt,yt,zt,siz,e,1)
      if (opt.eq.2) call get_len(fl,xt,yt,zt,siz)
            xt(j) = x8(j,e) 
           dfdx(j,e,1) = (gl-fl)/(2.*h(j,e))

            yt(j) = y8(j,e)+h(j,e) 
      if (opt.eq.1) call get_jac(gl,xt,yt,zt,siz,e,2)
      if (opt.eq.2) call get_len(gl,xt,yt,zt,siz)
            yt(j) = y8(j,e)-h(j,e) 
      if (opt.eq.1) call get_jac(fl,xt,yt,zt,siz,e,2)
      if (opt.eq.2) call get_len(fl,xt,yt,zt,siz)
            yt(j) = y8(j,e) 
           dfdx(j,e,2) = (gl-fl)/(2.*h(j,e))

            if (ldim.eq.3) then
             zt(j) = z8(j,e)+h(j,e) 
      if (opt.eq.1) call get_jac(gl,xt,yt,zt,siz,e,3)
      if (opt.eq.2) call get_len(gl,xt,yt,zt,siz)
             zt(j) = z8(j,e)-h(j,e) 
      if (opt.eq.1) call get_jac(fl,xt,yt,zt,siz,e,3)
      if (opt.eq.2) call get_len(fl,xt,yt,zt,siz)
             zt(j) = z8(j,e) 
           dfdx(j,e,ldim) = (gl-fl)/(2.*h(j,e))
            endif
          enddo 
      enddo 

      call fgslib_gs_op(gshl,dfdx(1,1,1),1,1,0)
      call fgslib_gs_op(gshl,dfdx(1,1,2),1,1,0)
      if (ldim.eq.3) call fgslib_gs_op(gshl,dfdx(1,1,ldim),1,1,0)

      f2 = glsum(f1,1)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_jac(val,x,y,z,siz,el,dire)
      include 'SIZE'
      include 'TOTAL'
c     This routine compiles the jacobian matrix and then does the 
c     frobenius norm of the matrix times that of the inverse
      integer i,j,k,siz,el,dire

      real par(siz),det(siz)
      real jm(ldim,ldim),jin(ldim,ldim),frn(ldim**2)
      real x(2**ldim),y(2**ldim),z(2**ldim),jac(2**ldim)
      real fr1,fr2,sum1,dumc,val

      integer bzindx(24),czindx(24)
      integer penalty
      SAVE bzindx
      DATA bzindx / 2,3,5, 1,4,6, 4,1,7, 3,2,8, 
     $             6,7,1, 5,8,2, 8,5,3, 7,6,4 /
c     bzindx tells which node is node connected to in r,s,t direction
c     example: node 1 is connected to 2,3,5; 2 to 1,4,6 and so on  
      SAVE czindx
      DATA czindx / 1,1,1,  -1,1,1, 1,-1,1, -1,-1,1,
     $              1,1,-1, -1,1,-1, 1,-1,-1, -1,-1,-1 / 
c     this tells which node is to be subtracted.

      if (siz.ne.2**ldim) then
         write(6,*) 'siz value wrong input into get_jac'
         write(6,*) siz,'sizval',2**ldim,'should be this'
         call exitt
      endif

      do i=1,2**ldim
        ind1 = (i-1)*3 !tells where to look in b/czindx
        do j=1,ldim
         ind2 = ind1+j
            jm(1,j) = 0.5*czindx(ind2)*(x(bzindx(ind2))-x(i))
            jm(2,j) = 0.5*czindx(ind2)*(y(bzindx(ind2))-y(i))
            if (ldim.eq.3) 
     $      jm(ldim,j) = 0.5*czindx(ind2)*(z(bzindx(ind2))-z(i))
        enddo
c     jacobian matrix has been calculated at this point. 
c     now calculate determinant
       if (ldim.eq.3) then
         jac(i)=jm(1,1)*(jm(2,2)*jm(ldim,ldim)-jm(2,ldim)*jm(ldim,2))+ 
     $          jm(1,ldim)*(jm(2,1)*jm(ldim,2)-jm(2,2)*jm(ldim,1))   +
     $          jm(1,2)*(jm(2,ldim)*jm(ldim,1)-jm(2,1)*jm(ldim,ldim))
        else
           jac(i)=jm(1,1)*jm(2,2)-jm(2,1)*jm(1,2) 
        endif
c      calculate inverse matrix
       if (ldim.eq.3) then
         jin(1,1) = jm(2,2)*jm(ldim,ldim)-jm(ldim,2)*jm(2,ldim)
         jin(2,1) = jm(2,ldim)*jm(ldim,1)-jm(ldim,ldim)*jm(2,1)
         jin(ldim,1) = jm(2,1)*jm(ldim,2)-jm(ldim,1)*jm(2,2)

         jin(1,2) = jm(1,ldim)*jm(ldim,2)-jm(ldim,ldim)*jm(1,2)
         jin(2,2) = jm(1,1)*jm(ldim,ldim)-jm(ldim,1)*jm(1,ldim)
         jin(ldim,2) = jm(1,2)*jm(ldim,1)-jm(ldim,2)*jm(1,1)
       
         jin(1,ldim) = jm(1,2)*jm(2,ldim)-jm(2,2)*jm(1,ldim)
         jin(2,ldim) = jm(1,ldim)*jm(2,1)-jm(2,ldim)*jm(1,1)
         jin(ldim,ldim) = jm(1,1)*jm(2,2)-jm(2,1)*jm(1,2)
       else
         jin(1,1) = jm(2,2)
         jin(2,1) = -jm(2,1)

         jin(1,2) = -jm(1,2)
         jin(2,2) = jm(1,1)
       endif
       
       dumc = 1/jac(i)
       call cmult(jin,dumc,ldim**2) !scale inverse by inv det
  
       call rzero(frn,ldim**2)
       call col3(frn,jm,jm,ldim**2) !square the entries
       sum1 = vlsum(frn,ldim**2)
       fr1 = SQRT(sum1)           !squareroot
             
       call rzero(frn,ldim**2)
       call col3(frn,jin,jin,ldim**2)
       sum1 = vlsum(frn,ldim**2)
       fr2 = SQRT(sum1)

       par(i) = fr1*fr2
       par(i) = (par(i)/ldim)**2

      enddo

      val = vlsum(par,siz)
      val = val/siz  !normalize to 1 for each element
      val=abs(val)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_len(val,x,y,z,siz)
      include 'SIZE'
      include 'TOTAL'
      integer siz
      real x(2**ldim),y(2**ldim),z(2**ldim)
      real par(siz),xm,ym,zm,val

      integer azindx(24)
      SAVE    azindx
      DATA    azindx  / 1,2 ,2,4, 3,4, 1,3, 2,6, 6,8,
     $                  4,8, 5,6, 7,8, 5,7, 1,5, 3,7 /
c     azindx tells what nodes make up each edge

      do i=1,siz
        ind = (i-1)*2
        xm = x(azindx(ind+1))-x(azindx(ind+2))
        ym = y(azindx(ind+1))-y(azindx(ind+2))
        zm = z(azindx(ind+1))-z(azindx(ind+2))
        par(i) = sqrt(xm*xm+ym*ym+zm*zm)
        par(i) = par(i)**2
      enddo

      val = vlsum(par,siz)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_nodscale(scalek,x8,y8,z8,gshl)
      include 'SIZE'
      include 'TOTAL'
      real x8(2**ldim,lelv*(2**ldim)),
     $     y8(2**ldim,lelv*(2**ldim)),
     $     z8(2**ldim,lelv*(2**ldim))
      real curval,xm,ym,zm,work1(1),dum,wrk1
c      real scalek
      real scalek(2**ldim,lelv*(2**ldim))
      integer bzindx(24),e,gshl
      DATA bzindx / 2,3,5, 1,4,6, 4,1,7, 3,2,8,
     $             6,7,1, 5,8,2, 8,5,3, 7,6,4 /
c     bzindx tells what node is connected to what node

      n1 = nelv*(2**ldim)
      n2 = n1*(2**ldim)
      curval = 1e+10
      call rone(scalek,n2)
      call cmult(scalek,curval,n2)

      do e=1,nelv*(2**ldim)
      do i=1,2**ldim
        curval = 1e+10
        ind = (i-1)*3
      do j=1,ldim
        xm = x8(i,e)-x8(bzindx(ind+j),e)
        ym = y8(i,e)-y8(bzindx(ind+j),e)
        zm = 0.
      if (ldim.eq.3) zm = z8(i,e)-z8(bzindx(ind+j),e)
        dum = sqrt(xm*xm+ym*ym+zm*zm)
        curval = min(dum,curval)
      enddo
        scalek(i,e) = curval
      enddo
      enddo
c
      fac = 1.e-2
      call cmult(scalek,fac,n2)
      call fgslib_gs_op(gshl,scalek,1,3,0)

      return
      end
c-----------------------------------------------------------------------
      subroutine genmask(nodmask,mlt,gshl,mltc,gshlc)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      integer gs_handle
      integer vertex(1)
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /ivrtx/ vertex
      integer*8 glo_num(lxc*lyc*lzc,lelv),ngv
      integer*8 glo_numk2(2**ldim,lelv*(2**ldim))

      real     mlt(2**ldim,lelv*(2**ldim)),
     $     nodmask(2**ldim,lelv*(2**ldim))
      real    mltc(lxc*lyc*lzc*lelv),
     $     wrkmask(lxc*lyc*lzc*lelv)
      integer gshlc,gshl,e,eg,e0,f

      integer zindx(64)
      SAVE zindx
      DATA zindx /  1,  2,  4,  5,  10, 11, 13, 14,
     $              2,  3,  5,  6,  11, 12, 14, 15,
     $              4,  5,  7,  8,  13, 14, 16, 17,
     $              5,  6,  8,  9,  14, 15, 17, 18,
     $              10, 11, 13, 14, 19, 20, 22, 23,
     $              11, 12, 14, 15, 20, 21, 23, 24,
     $              13, 14, 16, 17, 22, 23, 25, 26,
     $              14, 15, 17, 18, 23, 24, 26, 27 /


      call setupds_center(gs_handle,lxc,lyc,lzc,nelv,
     $  nelgv,vertex,glo_num)

c     setup the mask first so that it can be distribute as well
      zero = 0.
      call rone(wrkmask,lxc*lyc*lzc*nelv)
      do e=1,nelv
      do f=1,2*ldim
         if(cbc(f,e,1).ne.'E  '.and.cbc(f,e,1).ne.'   ')then
           call facev(wrkmask,e,f,zero,lxc,lyc,lzc)
         endif
      enddo
      enddo

      n = (lxc*lyc*lzc)*nelv
      call rone(mltc,n)
      call fgslib_gs_setup(gshlc,glo_num,n,nekcomm,np)
      call fgslib_gs_op(gshlc,mltc,1,1,0)

      call fgslib_gs_op(gshlc,wrkmask,1,2,0)
      call xmtox8(wrkmask,nodmask)

      n = 0
      do e=1,nelv     !each element
      do j=1,2**ldim  !broken down into 4 quads/8hexes
        n = n+1
        ind1 = (j-1)*8
        do k=1,2**ldim
          glo_numk2(k,n) = glo_num(zindx(ind1+k),e)
        enddo
      enddo
      enddo

      n = (2**ldim)*(nelv*(2**ldim))
      call fgslib_gs_setup(gshl,glo_numk2,n,nekcomm,np)
      call rone(mlt,n)
      call fgslib_gs_op(gshl,mlt,1,1,0)   ! '+'
      xmlt = glmax(mlt,n)
      call invcol1(mlt,n)

      call fgslib_gs_op(gshl,nodmask,1,2,0)

      return
      end
c-----------------------------------------------------------------------
      subroutine setupds_center(gs_handle,nx,ny,nz,nel,melg,
     $                        vertex,glo_num)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'NONCON'
      integer gs_handle
      integer vertex(1)
      integer*8 glo_num(1),ngv
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      n = nx*ny*nz*nel
      call set_vert(glo_num,ngv,nx,nel,vertex,.true.)
      call fgslib_gs_setup(gs_handle,glo_num,n,nekcomm,mp)
      return
      end
c-----------------------------------------------------------------------
      subroutine fastlap(iter,nodmask,mlt,gshl,dis,lapinv,mltc,gshlc)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),
     $     ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc

      real x8(2,2,ldim-1,lelv*(2**ldim)),
     $     y8(2,2,ldim-1,lelv*(2**ldim)),
     $     z8(2,2,ldim-1,lelv*(2**ldim))
      common /coarsemesh8/ x8,y8,z8

      real dxx(2**ldim,lelv*(2**ldim)),
     $     dyy(2**ldim,lelv*(2**ldim)),
     $     dzz(2**ldim,lelv*(2**ldim)),
     $     dis(2**ldim,lelv*(2**ldim)),
     $     mlt(2**ldim,lelv*(2**ldim)),
     $     nodmask(2**ldim,lelv*(2**ldim)),
     $     mltc(lxc*lyc*lzc,lelv)
      integer z,e,f,gshl,lapinv,kerr,gshlc
      real xbar,ybar,zbar,sfac

      if (nid.eq.0.and.loglevel.ge.2) 
     $    write(6,*) 'laplacian smoothing started'

      n    = 2**ldim
      sfac = 0.01
      do k=1,iter
         do e=1,nelv*(2**ldim)
            xbar = vlsum(x8(1,1,1,e),n)/(n)
            ybar = vlsum(y8(1,1,1,e),n)/(n)
            if (ldim.eq.3) zbar = vlsum(z8(1,1,1,e),n)/(n)
            do i=1,n
               dxx(i,e) = sfac*(xbar-x8(i,1,1,e))
               dyy(i,e) = sfac*(ybar-y8(i,1,1,e))
        if (ldim.eq.3) dzz(i,e) = sfac*(zbar-z8(i,1,1,e))
            enddo
         enddo

         n2 = nelv*(2**ldim)*(n)

         call col2(dxx,nodmask,n2)
         call col2(dxx,dis,n2)
         call dsavg_general(dxx,mlt,gshl)
         call add2(x8,dxx,n2)

         call col2(dyy,nodmask,n2)
         call col2(dyy,dis,n2)
         call dsavg_general(dyy,mlt,gshl)
         call add2(y8,dyy,n2)
        
         if (ldim.eq.3) then
          call col2(dzz,nodmask,n2)
          call col2(dzz,dis,n2)
          call dsavg_general(dzz,mlt,gshl)
          call add2(z8,dzz,n2)
         endif

         dxm =glamax(dxx,n2)
         dym =glamax(dyy,n2)
         if (ldim.eq.3) then
            dzm =glamax(dzz,n2)
         else
            dzm = 0.
         endif
         if (nio.eq.0.and.loglevel.ge.3) write(6,1) k,dxm,dym,dzm
   1     format(i5,1p3e12.3,' dxm')

      enddo

      call x8toxm(xmc,x8)
      call x8toxm(ymc,y8)
      if (ldim.eq.3) call x8toxm(zmc,z8)
      call fixcurs(mltc,gshlc)
      call xmtox8(xmc,x8)
      call xmtox8(ymc,y8)
      if (ldim.eq.3) call xmtox8(zmc,z8)

      call glmapm1chkinv(kerr)
      if (kerr.gt.0.) then
        lapinv = 1  !Terminate Laplacian smoothing
      if (nid.eq.0.and.loglevel.ge.2) write(6,*) 'terminating Laplacian'
      else
       call genbackupmesh
      endif
       
      return
      end
c-----------------------------------------------------------------------
      subroutine dsavg_general(u,mlt,gshl)
      include 'SIZE'
      include 'TOTAL'
      integer gshl
      real u(2**ldim,lelv*(2**ldim))
      real mlt(2**ldim,lelv*(2**ldim))

      call fgslib_gs_op(gshl,u,1,1,0) ! '+'
      call col2(u,mlt,(2**ldim)*nelv*(2**ldim))
      
      return
      end
c-----------------------------------------------------------------------
      subroutine xmtox8(xd,x8)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real x8(2**ldim,lelv*(2**ldim))
      real xd(lxc*lyc*lzc,lelv)
      integer e,k,n,j,ind1
      integer zindx(64)
      SAVE zindx
      DATA zindx /  1,  2,  4,  5,  10, 11, 13, 14,
     $              2,  3,  5,  6,  11, 12, 14, 15,
     $              4,  5,  7,  8,  13, 14, 16, 17,
     $              5,  6,  8,  9,  14, 15, 17, 18,
     $              10, 11, 13, 14, 19, 20, 22, 23,
     $              11, 12, 14, 15, 20, 21, 23, 24,
     $              13, 14, 16, 17, 22, 23, 25, 26,
     $              14, 15, 17, 18, 23, 24, 26, 27 /

      n = 0
      do e=1,nelv     !each element
      do j=1,2**ldim  !broken down into 4 quads/8hexes
        n = n+1
        ind1 = (j-1)*8
        do k=1,2**ldim
          x8(k,n) = xd(zindx(ind1+k),e)
        enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine x8toxm(xd,x8)
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real x8(2**ldim,lelv*(2**ldim))
      real xd(lxc*lyc*lzc,lelv)
      integer zindx((2**3)**2)
      integer e,k,n,j,ind1
      DATA zindx /  1,  2,  4,  5,  10, 11, 13, 14,
     $              2,  3,  5,  6,  11, 12, 14, 15,
     $              4,  5,  7,  8,  13, 14, 16, 17,
     $              5,  6,  8,  9,  14, 15, 17, 18,
     $              10, 11, 13, 14, 19, 20, 22, 23,
     $              11, 12, 14, 15, 20, 21, 23, 24,
     $              13, 14, 16, 17, 22, 23, 25, 26,
     $              14, 15, 17, 18, 23, 24, 26, 27 /


      n = 0
      do e=1,nelv     !each element
      do j=1,2**ldim  !broken down into 4 quads/8hexes
        n = n+1
        ind1 = (j-1)*8
        do k=1,2**ldim
          xd(zindx(ind1+k),e) = x8(k,n)
        enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine col3c(a,b,c,d,n)
      real a(1),b(1),c(1),d

      do i=1,n
         a(i)=b(i)*c(i)*d
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine glmapm1chkinv(kerr)
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
C
C     Note: Subroutines GLMAPM1, GEODAT1, AREA2, SETWGTR and AREA3 
C           share the same array structure in Scratch Common /SCRNS/.
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      parameter (nxc=3,nyc=3,nzc=1+(ldim-2)*(nxc-1))
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc
C
      real           XRMc(LXc,LYc,LZc,LELv)
     $ ,             YRMc(LXc,LYc,LZc,LELv)
     $ ,             XSMc(LXc,LYc,LZc,LELv)
     $ ,             YSMc(LXc,LYc,LZc,LELv)
     $ ,             XTMc(LXc,LYc,LZc,LELv)
     $ ,             YTMc(LXc,LYc,LZc,LELv)
     $ ,             ZRMc(LXc,LYc,LZc,LELv)
      real           ZSMc(LXc,LYc,LZc,LELv)
     $ ,             ZTMc(LXc,LYc,LZc,LELv)
     $ ,            jacmc(lxc,lyc,lzc,lelv)
      real           rxmc(lxc,lyc,lzc,lelv)
     $ ,             rymc(lxc,lyc,lzc,lelv)
     $ ,             rzmc(lxc,lyc,lzc,lelv)
     $ ,             sxmc(lxc,lyc,lzc,lelv)
     $ ,             symc(lxc,lyc,lzc,lelv)
     $ ,             szmc(lxc,lyc,lzc,lelv)
     $ ,             txmc(lxc,lyc,lzc,lelv)
     $ ,             tymc(lxc,lyc,lzc,lelv)
     $ ,             tzmc(lxc,lyc,lzc,lelv)
      integer kerr,ierr
C
      NXYc  = NXc*NYc
      NYZc  = NYc*NZc
      NXYZc = NXc*NYc*NZc
      NTOTc = NXYZc*NELv
C
      CALL XYZRSTc (XRMc,YRMc,ZRMc,XSMc,YSMc,ZSMc,XTMc,YTMc,ZTMc,
     $             IFAXIS)


C
      IF (NDIM.EQ.2) THEN
         CALL RZERO   (JACMc,NTOTc)
         CALL ADDCOL3 (JACMc,XRMc,YSMc,NTOTc)
         CALL SUBCOL3 (JACMc,XSMc,YRMc,NTOTc)
         CALL COPY    (RXMc,YSMc,NTOTc)
         CALL COPY    (RYMc,XSMc,NTOTc)
         CALL CHSIGN  (RYMc,NTOTc)
         CALL COPY    (SXMc,YRMc,NTOTc)
         CALL CHSIGN  (SXMc,NTOTc)
         CALL COPY    (SYMc,XRMc,NTOTc)
         CALL RZERO   (RZMc,NTOTc)
         CALL RZERO   (SZMc,NTOTc)
         CALL RONE    (TZMc,NTOTc)
      ELSE
         CALL RZERO   (JACMc,NTOTc)
         CALL ADDCOL4 (JACMc,XRMc,YSMc,ZTMc,NTOTc)
         CALL ADDCOL4 (JACMc,XTMc,YRMc,ZSMc,NTOTc)
         CALL ADDCOL4 (JACMc,XSMc,YTMc,ZRMc,NTOTc)
         CALL SUBCOL4 (JACMc,XRMc,YTMc,ZSMc,NTOTc)
         CALL SUBCOL4 (JACMc,XSMc,YRMc,ZTMc,NTOTc)
         CALL SUBCOL4 (JACMc,XTMc,YSMc,ZRMc,NTOTc)
         CALL ASCOL5  (RXMc,YSMc,ZTMc,YTMc,ZSMc,NTOTc)
         CALL ASCOL5  (RYMc,XTMc,ZSMc,XSMc,ZTMc,NTOTc)
         CALL ASCOL5  (RZMc,XSMc,YTMc,XTMc,YSMc,NTOTc)
         CALL ASCOL5  (SXMc,YTMc,ZRMc,YRMc,ZTMc,NTOTc)
         CALL ASCOL5  (SYMc,XRMc,ZTMc,XTMc,ZRMc,NTOTc)
         CALL ASCOL5  (SZMc,XTMc,YRMc,XRMc,YTMc,NTOTc)
         CALL ASCOL5  (TXMc,YRMc,ZSMc,YSMc,ZRMc,NTOTc)
         CALL ASCOL5  (TYMc,XSMc,ZRMc,XRMc,ZSMc,NTOTc)
         CALL ASCOL5  (TZMc,XRMc,YSMc,XSMc,YRMc,NTOTc)
      ENDIF
C
      kerr = 0
      DO 500 ie=1,NELv
         CALL CHKJACINV(JACMc(1,1,1,ie),NXYZc,ie,xmc(1,1,1,ie),
     $ ymc(1,1,1,ie),zmc(1,1,1,ie),ndim,ierr)
         if (ierr.ne.0) kerr = kerr+1
  500 CONTINUE
      kerr = iglsum(kerr,1)
      
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine chkjacinv(jac,n,iel,X,Y,Z,ND,IERR)
c
      include 'SIZE'
      include 'PARALLEL'
C
C     Check the array JAC for a change in sign.
C
      REAL JAC(N),x(1),y(1),z(1)
      integer ierr
c
      ierr = 0
      SIGN = JAC(1)
      DO 100 I=2,N
         IF (SIGN*JAC(I).LE.0.0) THEN
            ierr = 1
         ENDIF
  100 CONTINUE

      RETURN
      END
C-----------------------------------------------------------------------
      subroutine xyzrstc(xrmc,yrmc,zrmc,xsmc,ysmc,zsmc,
     $                   XTMc,YTMc,ZTMc,IFAXIS)
C-----------------------------------------------------------------------
C
C     Compute global-to-local derivatives on mesh 1.
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
C
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      DIMENSION XRMc(LXc,LYc,LZc,1),YRMc(LXc,LYc,LZc,1)
     $        , ZRMc(LXc,LYc,LZc,1),XSMc(LXc,LYc,LZc,1)
     $        , YSMc(LXc,LYc,LZc,1),ZSMc(LXc,LYc,LZc,1)
     $        , XTMc(LXc,LYc,LZc,1),YTMc(LXc,LYc,LZc,1)
     $        , ZTMc(LXc,LYc,LZc,1)
      LOGICAL IFAXIS
      real dxmc(3,3),dymc(3,3),dzmc(3,3)
      real dxtmc(3,3),dytmc(3,3),dztmc(3,3)
      common /coarseders/ dxmc,dymc,dzmc,dxtmc,dytmc,dztmc
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc
C
      NXYc=lxc*lyc
      NYZc=lyc*lzc
C
      DO 100 IEL=1,NELv
C
      CALL MXM (DXMc,lxc,XMc(1,1,1,IEL),lxc,XRMc(1,1,1,IEL),NYZc)
      CALL MXM (DXMc,lxc,YMc(1,1,1,IEL),lxc,YRMc(1,1,1,IEL),NYZc)
      CALL MXM (DXMc,lxc,ZMc(1,1,1,IEL),lxc,ZRMc(1,1,1,IEL),NYZc)
C
      DO 10 IZ=1,lzc
      CALL MXM (XMc(1,1,IZ,IEL),lxc,DYTMc,lyc,XSMc(1,1,IZ,IEL),lyc)
      CALL MXM (YMc(1,1,IZ,IEL),lxc,DYTMc,lyc,YSMc(1,1,IZ,IEL),lyc)
      CALL MXM (ZMc(1,1,IZ,IEL),lxc,DYTMc,lyc,ZSMc(1,1,IZ,IEL),lyc)
   10 CONTINUE
C
      IF (ldim.EQ.3) THEN
         CALL MXM (XMc(1,1,1,IEL),NXYc,DZTMc,lzc,XTMc(1,1,1,IEL),lzc)
         CALL MXM (YMc(1,1,1,IEL),NXYc,DZTMc,lzc,YTMc(1,1,1,IEL),lzc)
         CALL MXM (ZMc(1,1,1,IEL),NXYc,DZTMc,lzc,ZTMc(1,1,1,IEL),lzc)
      ELSE
         CALL RZERO (XTMc(1,1,1,IEL),NXYc)
         CALL RZERO (YTMc(1,1,1,IEL),NXYc)
         CALL RONE  (ZTMc(1,1,1,IEL),NXYc)
      ENDIF
C
  100 CONTINUE
C
      RETURN
      END
C-----------------------------------------------------------------------
      subroutine xm1toxmc
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc
      integer e

      if (ldim.eq.2) then
        do e=1,nelv
         call int_fine_to_coarse_2d(xm1(1,1,1,e),xmc(1,1,1,e),lx1,3)
         call int_fine_to_coarse_2d(ym1(1,1,1,e),ymc(1,1,1,e),lx1,3)
        enddo
      else
        do e=1,nelv
         call int_fine_to_coarse_3d(xm1(1,1,1,e),xmc(1,1,1,e),lx1,3)
         call int_fine_to_coarse_3d(ym1(1,1,1,e),ymc(1,1,1,e),lx1,3)
         call int_fine_to_coarse_3d(zm1(1,1,1,e),zmc(1,1,1,e),lx1,3)
        enddo
      endif

      RETURN
      END
C-----------------------------------------------------------------------
      subroutine xmctoxm1
      include 'SIZE'
      include 'TOTAL'
      parameter (lxc=3,lyc=3,lzc=1+(ldim-2)*(lxc-1))
      real xmc(lxc,lyc,lzc,lelv),ymc(lxc,lyc,lzc,lelv),
     $     zmc(lxc,lyc,lzc,lelv)
      common /coarsemesh/ xmc,ymc,zmc
      integer e

      if (ldim.eq.2) then
        do e=1,nelv
         call int_coarse_to_fine_2d(xm1(1,1,1,e),xmc(1,1,1,e),lx1,3)
         call int_coarse_to_fine_2d(ym1(1,1,1,e),ymc(1,1,1,e),lx1,3)
        enddo
      else
        do e=1,nelv
         call int_coarse_to_fine_3d(xm1(1,1,1,e),xmc(1,1,1,e),lx1,3)
         call int_coarse_to_fine_3d(ym1(1,1,1,e),ymc(1,1,1,e),lx1,3)
         call int_coarse_to_fine_3d(zm1(1,1,1,e),zmc(1,1,1,e),lx1,3)
        enddo
      endif

      RETURN
      END
C-----------------------------------------------------------------------
      subroutine my_mv_mesh(dx,dy,dz)
      include 'SIZE'
      include 'TOTAL'

      parameter (lt = lx1*ly1*lz1*lelv)
      common /mrthoi/ napprx(2),nappry(2),napprz(2)
      common /mrthov/ apprx(lt,0:mxprev)
     $              , appry(lt,0:mxprev)
     $              , apprz(lt,0:mxprev)
      common /mstuff/ d(lt),h1(lt),h2(lt),mask(lt)
      real mask
      real mask2(lx1,ly1,lz1,lelv)
      real dx(1),dy(1),dz(1)

      integer e,f

      integer icalld
      save    icalld
      data    icalld /0/

      n = nx1*ny1*nz1*nelv
      nface = 2*ndim

      napprx(1)=0
      nappry(1)=0
      napprz(1)=0

      nxz   = nx1*nz1
      nxyz  = nx1*ny1*nz1

      volbl = 0.   ! Volume of elements in boundary layer
      do e=1,nelv
      do f=1,nface
         if (cbc(f,e,1).eq.'W  ') then
            srfbl = srfbl + vlsum(area(1,1,f,e),nxz )
            volbl = volbl + vlsum(bm1 (1,1,1,e),nxyz)
         endif
      enddo
      enddo
      srfbl = glsum(srfbl,1)  ! Sum over all processors
      volbl = glsum(volbl,1)

      delta = volbl / srfbl   ! Avg thickness of b.l. elements

      call rone (h1,n)
      call rzero(h2,n)
      call sm_cheap_dist(d,1,'W  ')

      deltap = 5*delta  ! Protected b.l. thickness
      do i=1,n
         arg   = -(d(i)/deltap)**2
         h1(i) = h1(i) + 9*exp(arg)
      enddo

      call rone(mask,n)
      do e=1,nelv
      do f=1,nface
        if (cbc(f,e,1).ne.'E  '.and.cbc(f,e,1).ne.'   ') 
     $     call facev(mask,e,f,0.,nx1,ny1,nz1)
      enddo
      enddo
      call dsop(mask,'mul',nx1,ny1,nz1)
      call copy(mask2,mask,n)
      call chsign(mask2,n)
      call cadd(mask2,1.,n)
      call col2(dx,mask2,n)
      call col2(dy,mask2,n)
      call col2(dz,mask2,n)
      tol = -1.e-6
      call laplaceh('mshx',dx,h1,h2,mask,vmult,1,tol,500,apprx,napprx)
      call laplaceh('mshy',dy,h1,h2,mask,vmult,1,tol,500,appry,nappry)
      if (if3d)
     $ call laplaceh('mshz',dz,h1,h2,mask,vmult,1,tol,500,apprz,napprz)

      return
      end
c-----------------------------------------------------------------------
      subroutine laplaceh
     $     (name,u,h1,h2,mask,mult,ifld,tli,maxi,approx,napprox)
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
c
      character*4 name
      real u(1),h1(1),h2(1),mask(1),mult(1),approx (1)
      integer   napprox(1)

      parameter (lt=lx1*ly1*lz1*lelv)
      common /scruz/ r (lt),ub(lt)

      logical ifstdh
      character*4  cname
      character*6  name6

      logical ifwt,ifvec

      call chcopy(cname,name,4)
      call capit (cname,4)

      call blank (name6,6)
      call chcopy(name6,name,4)
      ifwt  = .true.
      ifvec = .false.
      isd   = 1
      imsh  = 1
      nel   = nelfld(ifld)

      n = nx1*ny1*nz1*nel

      call copy (ub,u,n)             ! ub = u on boundary
      call dsavg(ub)                 ! Make certain ub is in H1
                                     !     _
      call axhelm (r,ub,h1,h2,1,1)   ! r = A*ub

      do i=1,n                       !        _
         r(i)=-r(i)*mask(i)          ! r = -M*A*ub
      enddo

      call dssum  (r,nx1,ny1,nz1)    ! dssum rhs

      call project1
     $    (r,n,approx,napprox,h1,h2,mask,mult,ifwt,ifvec,name6)

      tol = -abs(tli)
      if (nel.eq.nelv) then
        call hmhzpf (name,u,r,h1,h2,mask,mult,imsh,tol,maxi,isd,binvm1)
      else
        call hmhzpf (name,u,r,h1,h2,mask,mult,imsh,tol,maxi,isd,bintm1)
      endif

      call project2
     $     (u,n,approx,napprox,h1,h2,mask,mult,ifwt,ifvec,name6)

      call add2(u,ub,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine sm_cheap_dist(d,ifld,b)
c     this is a copy of the routine in navier5.f, modified to be less
c     verbose


      include 'SIZE'
      include 'GEOM'       ! Coordinates
      include 'INPUT'      ! cbc()
      include 'TSTEP'      ! nelfld
      include 'PARALLEL'   ! gather-scatter handle for field "ifld"

      real d(lx1,ly1,lz1,lelt)

      character*3 b  ! Boundary condition of interest

      integer e,eg,f

      nel = nelfld(ifld)
      n = lx1*ly1*lz1*nel

      call domain_size(xmin,xmax,ymin,ymax,zmin,zmax)

      xmn = min(xmin,ymin)
      xmx = max(xmax,ymax)
      if (if3d) xmn = min(xmn ,zmin)
      if (if3d) xmx = max(xmx ,zmax)

      big = 10*(xmx-xmn)
      call cfill(d,big,n)

      nface = 2*ldim
      do e=1,nel     ! Set d=0 on walls
      do f=1,nface
         if (cbc(f,e,ifld).eq.b) call facev(d,e,f,0.,lx1,ly1,lz1)
      enddo
      enddo

      do ipass=1,10000
         dmax    = 0
         nchange = 0
         do e=1,nel
           do k=1,lz1
           do j=1,ly1
           do i=1,lx1
             i0=max(  1,i-1)
             j0=max(  1,j-1)
             k0=max(  1,k-1)
             i1=min(lx1,i+1)
             j1=min(ly1,j+1)
             k1=min(lz1,k+1)
             do kk=k0,k1
             do jj=j0,j1
             do ii=i0,i1

              if (if3d) then
               dtmp = d(ii,jj,kk,e) + dist3d(
     $           xm1(ii,jj,kk,e),ym1(ii,jj,kk,e),zm1(ii,jj,kk,e)
     $          ,xm1(i ,j ,k ,e),ym1(i ,j ,k ,e),zm1(i ,j ,k ,e))
              else
               dtmp = d(ii,jj,kk,e) + dist2d(
     $           xm1(ii,jj,kk,e),ym1(ii,jj,kk,e)
     $          ,xm1(i ,j ,k ,e),ym1(i ,j ,k ,e))
              endif

              if (dtmp.lt.d(i,j,k,e)) then
                d(i,j,k,e) = dtmp
                nchange = nchange+1
                dmax = max(dmax,d(i,j,k,e))
              endif
             enddo
             enddo
             enddo

           enddo
           enddo
           enddo
         enddo
         call fgslib_gs_op(gsh_fld(ifld),d,1,3,0) ! min over all elements
         nchange = iglsum(nchange,1)
         dmax = glmax(dmax,1)
         if (nio.eq.0.and.loglevel.ge.3) write(6,1) ipass,nchange,dmax,b
    1    format(i9,i12,1pe12.4,' max distance b: ',a3)
         if (nchange.eq.0) goto 1000
      enddo
 1000 return
      end
c-----------------------------------------------------------------------
