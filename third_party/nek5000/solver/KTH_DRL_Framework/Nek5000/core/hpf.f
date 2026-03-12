      subroutine MAKE_HPF

      include 'SIZE'
      include 'SOLN'
      include 'INPUT'         ! param(110),(111)
      include 'TSTEP'         ! ifield
      include 'MASS'          ! BM1

      integer nxyz
      parameter (nxyz=lx1*ly1*lz1)
      integer n

      integer lm,lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)
      real hpf_filter(lm2)

      integer hpf_kut
      real hpf_chi
      logical hpf_ifboyd

      integer nel

      integer icalld
      save icalld
      data icalld /0/

      real TA1,TA2,TA3
      COMMON /SCRUZ/ TA1 (LX1,LY1,LZ1,LELV)
     $ ,             TA2 (LX1,LY1,LZ1,LELV)
     $ ,             TA3 (LX1,LY1,LZ1,LELV)

      real hpf_op(lx1,lx1)
      save hpf_op

c---------------------------------------- 
      if(.not. iffilter(ifield)) return

      hpf_kut = int(param(101))+1
      hpf_chi = -1.0*abs(param(103))
c     Boyd transform to preserve element boundary values is 
c     linearly unstable when used as forcing.
      hpf_ifboyd = .false.      

      nel = nelv
      n = nxyz*nel

      if (hpf_chi.eq.0) return
      if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'apply hpf ',
     $                                 ifield, hpf_kut, hpf_chi

      if (icalld.eq.0) then
c       Create the filter transfer function
        call hpf_trns_fcn(hpf_filter,hpf_kut)

c       Build the matrix to apply the filter function
c       to an input field
        call build_hpf_mat(hpf_op,hpf_filter,hpf_ifboyd)

c       Only initialize once    
        icalld=icalld+1 
      endif

      if (ifield.eq.1) then
c       Apply the filter
c       to velocity fields
        if (jp.eq.0) then 
          call build_hpf_fld(ta1,vx,hpf_op,lx1,lz1)
          call build_hpf_fld(ta2,vy,hpf_op,lx1,lz1)
          if (if3d) call build_hpf_fld(ta3,vz,hpf_op,lx1,lz1)

c         Multiply by filter weight (chi)
          call cmult(ta1,hpf_chi,n)    
          call cmult(ta2,hpf_chi,n)    
          if (if3d) call cmult(ta3,hpf_chi,n)    

c         Multiply by Mass matrix 
c         and add to forcing term 
          call opadd2col (bfx,bfy,bfz,ta1,ta2,ta3,bm1)
        else
c         Apply filter on velocity perturbation fields
          call build_hpf_fld(ta1,vxp(1,jp),hpf_op,lx1,lz1)
          call build_hpf_fld(ta2,vyp(1,jp),hpf_op,lx1,lz1)
          if (if3d) call build_hpf_fld(ta3,vzp(1,jp),hpf_op,lx1,lz1)

c         Multiply by filter weight (chi)
          call cmult(ta1,hpf_chi,n)    
          call cmult(ta2,hpf_chi,n)    
          if (if3d) call cmult(ta3,hpf_chi,n)    

c         Multiply by Mass matrix 
c         and add to forcing term 
          call opadd2col (bfxp(1,jp),bfyp(1,jp),bfzp(1,jp),
     $                    ta1,ta2,ta3,bm1)
        endif        

      else

c       Apply filter to temp/passive scalar fields
        if (jp.eq.0) then
          call build_hpf_fld(ta1,t(1,1,1,1,ifield-1),
     $         hpf_op,lx1,lz1)

c         Multiply by filter weight (chi)
          call cmult(ta1,hpf_chi,n)    

c         Multiply by Mass matrix    
c         and add to source term
          call addcol3(bq(1,1,1,1,ifield-1),ta1,bm1,n)
        else
c         Apply filter on scalar perturbation field                
          call build_hpf_fld(ta1,tp(1,ifield-1,jp),hpf_op,lx1,lz1)

c         Multiply by filter weight (chi)
          call cmult(ta1,hpf_chi,n)    

c         Multiply by Mass matrix    
c         and add to source term
          call addcol3(bqp(1,ifield-1,jp),ta1,bm1,n)
        endif  

      endif

      return
      end

c----------------------------------------------------------------------

      subroutine build_hpf_mat(op_mat,f_filter,ifboyd)

c     Builds the operator for high pass filtering
c     Transformation matrix from nodal to modal space.
c     Applies f_filter to the the legendre coefficients
c     Transforms back to nodal space
c     Operation: V * f_filter * V^(-1)
c     Where V is the transformation matrix from modal to nodal space

c      implicit none

      include 'SIZE'

      logical IFBOYD 
      integer n
      parameter (n=lx1*lx1)
      integer lm, lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)

      real f_filter(lm2)
      real op_mat(lx1,lx1)

      real ref_xmap(lm2)
      real wk_xmap(lm2)

      real wk1(lm2),wk2(lm2)
      real indr(lm),ipiv(lm),indc(lm)

      real rmult(lm)
      integer ierr

      integer i,j

      call spec_coeff_init(ref_xmap,ifboyd)
      
      call copy(wk_xmap,ref_xmap,lm2)
      call copy(wk1,wk_XMAP,lm2)

      call gaujordf  (wk1,lx1,lx1,indr,indc,ipiv,ierr,rmult)  ! xmap inverse

      call mxm  (f_filter,lx1,wk1,lx1,wk2,lx1)        !          -1
      call mxm  (wk_xmap,lx1,wk2,lx1,op_mat,lx1)      !     V D V

      return
      end

c---------------------------------------------------------------------- 

      subroutine build_hpf_fld(v,u,f,nx,nz)

c     Appies the operator f to field u
c     using tensor operations
c     v = f*u

c      implicit none

      include 'SIZE'
      include 'TSTEP'         ! ifield

      integer nxyz 
      parameter (nxyz=lx1*ly1*lz1)

      real w1(nxyz),w2(nxyz)              ! work arrays
      real v(nxyz,lelv)                   ! output fld
      real u(nxyz,lelv)                   ! input fld
c
      integer nx,nz

      real f(nx,nx),ft(nx,nx)             ! operator f and its transpose
c
      integer e,i,j,k
      integer nel

      nel = nelv

      call copy(v,u,nxyz*nel)
c
      call transpose(ft,nx,f,nx)
c
      if (ldim.eq.3) then
        do e=1,nel
c         Filter
          call copy(w2,v(1,e),nxyz)
          call mxm(f,nx,w2,nx,w1,nx*nx)
          i=1
          j=1
          do k=1,nx
             call mxm(w1(i),nx,ft,nx,w2(j),nx)
             i = i+nx*nx
             j = j+nx*nx
          enddo
          call mxm (w2,nx*nx,ft,nx,w1,nx)

          call sub3(w2,u(1,e),w1,nxyz)
          call copy(v(1,e),w2,nxyz)
c          call copy(v(1,e),w1,nxyz)  

        enddo
      else
        do e=1,nel
c         Filter
          call copy(w1,v(1,e),nxyz)
          call mxm(f ,nx,w1,nx,w2,nx)
          call mxm(w2,nx,ft,nx,w1,nx)

          call sub3(w2,u(1,e),w1,nxyz)
          call copy(v(1,e),w2,nxyz)
c          call copy(v(1,e),w1,nxyz) 
        enddo
      endif
c
      return
      end

c---------------------------------------------------------------------- 

      subroutine hpf_trns_fcn(diag,kut)

c      implicit none

      include 'SIZE'
      include 'PARALLEL'

      real diag(lx1*lx1)
      integer nx,k0,kut,kk,k

      real amp

c     Set up transfer function
c
      nx = lx1
      call ident   (diag,nx)
c
      kut=kut                                     ! kut=additional modes
      k0 = nx-kut
      do k=k0+1,nx
        kk = k+nx*(k-1)
        amp = ((k-k0)*(k-k0)+0.)/(kut*kut+0.)     ! Normalized amplitude. quadratic growth
        diag(kk) = 1.-amp
      enddo

c     Output normalized transfer function
      k0 = lx1+1
      if (nio.eq.0) then
        write(6,6) 'HPF :',((1.-diag(k)), k=1,lx1*lx1,k0)
   6    format(a8,16f9.6,6(/,8x,16f9.6))
      endif

      return
      end

c---------------------------------------------------------------------- 

      subroutine spec_coeff_init(ref_xmap,ifboyd)
c     Initialise spectral coefficients
c     For legendre transform

c      implicit none

      include 'SIZE'
      include 'WZ'

      integer lm, lm2
      parameter (lm=40)
      parameter (lm2=lm*lm)

c     local variables
      integer i, j, k, n, nx, kj
c     Legendre polynomial
      real plegx(lm)
      real z
      real ref_xmap(lm2)
      real pht(lm2)

c     Change of basis
      logical IFBOYD
c---------------------------------------- 

      nx = LX1
      kj = 0
      n  = nx-1
      do j=1,nx
        z = ZGM1(j,1)
        call legendre_poly(plegx,z,n)
        kj = kj+1
        pht(kj) = plegx(1)
        kj = kj+1
        pht(kj) = plegx(2)

        if (IFBOYD) then        ! change basis to preserve element boundary values
          do k=3,nx
             kj = kj+1
             pht(kj) = plegx(k)-plegx(k-2)
          enddo
        else                    ! legendre basis    
          do k=3,nx
             kj = kj+1
             pht(kj) = plegx(k)
          enddo         
        endif
      enddo

      call transpose (ref_xmap,nx,pht,nx)

      return
      end
