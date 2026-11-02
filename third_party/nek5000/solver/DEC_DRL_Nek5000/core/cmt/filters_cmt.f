c---------------------------------------------------------------------
      subroutine filter_cmtvar(continuous)
      include 'SIZE'
      include 'INPUT'
      include 'WZ'
      include 'DEALIAS'
      include 'CMTDATA'

      common /CMTFILTERS/  intv(lx1,lx1),intt(lx1,lx1)
     $                  ,  intvd(lxd,lxd),inttd(lxd,lxd)
     $                  ,  wk1(lx1,lx1,lz1),wk2(lx1,lx1,lz1)
     $                  ,  wkd1(lxd,lxd,lzd),wkd2(lxd,lxd,lzd)
      real                 intv, intt, intvd, inttd
!     real ::  intv(lx1,lx1),intt(lx1,lx1)

      real zgmd(nxd),wxmd(nxd)
      logical continuous
c     declare ncut and wght are declared in input
c     ncut is param(101) 
      integer icalld
      save    icalld
      data    icalld /0/


c     write(6,*)'Enter filter subroutine'
      if (icalld.eq.0) then
         icalld = 1
         if (continuous) then ! use existing nek5000 code in a
                              ! manner similar to nek5000
            if(nio.eq.0)  
     $         write(6,*)'Build new filter with bubble functions'
            call userfilt_bubble(intv,zgm1,nx1,nio)
c           call build_new_filter(intv,zgm1,nx1,ncut,wght,nio)
c           if (nx1.eq.nxd)then
c              nxyz = nxd*nyd*nzd
c              if(nio.eq.0) write(6,*)'Copy newly built filter built'
c              call copy(intvd,intv,nxyz)
c           else
c              if(nio.eq.0)write(6,*)'build new filter with',ncutd,wghtd
c              call zwgll(zgmd,wxmd,nxd)
c              call build_new_filter(intvd,zgmd,nxd,ncutd,wghtd,nio)
c           endif
c           do i=1,lx1*lx1
c              if(nio.eq.0) write(669,*)'filter',intv(i,1)
c           enddo
         else ! dg filter using Legendre basis instead of Lk-Lkm2
            if(nio.eq.0)  
     $         write(6,*)'Build new filter with Legendre basis'
            call userfilt(intv,zgm1,nx1,nio)
         endif
      endif

      call filtercv(intv,nx1,nz1,wk1,wk2,intt,if3d,u1max)

      return
      end
c---------------------------------------------------------------------

      subroutine filtercv(f,nx,nz,w1,w2,ft,if3d,dmax)
c     taken from filterq in navier5.f
      include 'SIZE'
      include 'TSTEP'
    
      COMMON /solnconsvar/ U(LX1,LY1,LZ1,TOTEQ,lelt) 
      logical if3d
c
      real w1(1),w2(1)
c
      real f(nx,nx),ft(nx,nx)
      integer e,eq

      call transpose(ft,nx,f,nx)
c
      nxyz=nx*nx*nz
      dmax = 0.


c     nel = nelfld(ifield)


      if (if3d) then
         do e=1,nelv
            do eq=1,toteq
c              Filter
               call copy(w2,u(1,1,1,eq,e),nxyz)
               call mxm(f,nx,w2,nx,w1,nx*nx)
               i=1
               j=1
               do k=1,nx
                  call mxm(w1(i),nx,ft,nx,w2(j),nx)
                  i = i+nx*nx
                  j = j+nx*nx
               enddo
               call mxm (w2,nx*nx,ft,nx,w1,nx)
               call sub3(w2,u(1,1,1,eq,e),w1,nxyz)
               call copy(u(1,1,1,eq,e),w1,nxyz)
               smax = vlamax(w2,nxyz)
               dmax = max(dmax,abs(smax))
            enddo
         enddo
c
      else
         do e=1,nelv
            do eq=1,toteq
c              Filter
               call copy(w1,u(1,1,1,eq,e),nxyz)
               call mxm(f,nx,w1,nx,w2,nx)
               call mxm(w2,nx,ft,nx,w1,nx)
c
               call sub3(w2,u(1,1,1,eq,e),w1,nxyz)
               call copy(u(1,1,1,eq,e),w1,nxyz)
               smax = vlamax(w2,nxyz)
               dmax = max(dmax,abs(smax))
            enddo
         enddo
      endif
c
      return
      end

!---------------------------------------------------------------------

      subroutine userfilt_bubble(intv,zpts,nx,nio)

!     This routing builds a 1D filter with a transfer function that
!     will be specified from the .usr file. Variables are projected on
!     to the space defined by bubble functions. 

      real intv(nx,nx),zpts(nx)
c
      parameter (lm=84)
      parameter (lm2=lm*lm)
      real      phi(lm2),pht(lm2),diag(lm2),rmult(lm),Lj(lm)
      integer   indr(lm),indc(lm),ipiv(lm)
c
      if (nx.gt.lm) then
        if(nio.eq.0) write(6,*) 'ABORT in build_new_filter:',nx,lm
         call exitt
      endif
c
      kj = 0
      n  = nx-1
      do j=1,nx
         z = zpts(j)
         call legendre_poly(Lj,z,n)
         kj = kj+1
         pht(kj) = Lj(1)
         kj = kj+1
         pht(kj) = Lj(2)
         do k=3,nx
            kj = kj+1
            pht(kj) = Lj(k)-Lj(k-2)
         enddo
      enddo
      call transpose (phi,nx,pht,nx)
      call copy      (pht,phi,nx*nx)
      call gaujordf  (pht,nx,nx,indr,indc,ipiv,ierr,rmult)
c
c     Set up transfer function

      call cmt_usrflt(rmult)
      call ident   (diag,nx)

      do k=1,nx
         amp=rmult(k)
         if (amp .ne. 1.0) then
            kk = k+nx*(k-1)
            diag(kk) = amp
         endif
      enddo

      call mxm  (diag,nx,pht,nx,intv,nx)      !          -1
      call mxm  (phi ,nx,intv,nx,pht,nx)      !     V D V
      call copy (intv,pht,nx*nx)

      do k=1,nx*nx
         pht(k) = 1.-diag(k)
      enddo
      np1 = nx+1
      if (nio.eq.0) then
         write(6,6) 'filt amp',(pht (k),k=1,nx*nx,np1)
         write(6,6) 'filt trn',(diag(k),k=1,nx*nx,np1)
   6     format(a8,16f7.4,6(/,8x,16f7.4))
      endif

      return
      end
!---------------------------------------------------------------------

      subroutine userfilt(intv,zpts,nx,nio)

!     This routing builds a 1D filter with a transfer function that
!     will be specified from the .usr file. DG ONLY as a Legendre basis
!     means endpoints are affected!

      real intv(nx,nx),zpts(nx)
c
      parameter (lm=84)
      parameter (lm2=lm*lm)
      real      phi(lm2),pht(lm2),diag(lm2),rmult(lm),Lj(lm)
      integer   indr(lm),indc(lm),ipiv(lm)
c
      if (nx.gt.lm) then
        if(nio.eq.0) write(6,*) 'ABORT in build_new_filter:',nx,lm
         call exitt
      endif
c
      kj = 0
      n  = nx-1
      do j=1,nx
         z = zpts(j)
         call legendre_poly(Lj,z,n)
         do k=1,nx
            kj = kj+1
            pht(kj) = Lj(k)
         enddo
      enddo
      call transpose (phi,nx,pht,nx)
      call copy      (pht,phi,nx*nx)
      call gaujordf  (pht,nx,nx,indr,indc,ipiv,ierr,rmult)
c
c     Set up transfer function

      call cmt_usrflt(rmult)
      call ident   (diag,nx)

      do k=1,nx
         amp=rmult(k)
         if (amp .ne. 1.0) then
            kk = k+nx*(k-1)
            diag(kk) = amp
         endif
      enddo

      call mxm  (diag,nx,pht,nx,intv,nx)      !          -1
      call mxm  (phi ,nx,intv,nx,pht,nx)      !     V D V
      call copy (intv,pht,nx*nx)

      do k=1,nx*nx
         pht(k) = 1.-diag(k)
      enddo
      np1 = nx+1
      if (nio.eq.0) then
         write(6,6) 'filt amp',(pht (k),k=1,nx*nx,np1)
         write(6,6) 'filt trn',(diag(k),k=1,nx*nx,np1)
   6     format(a8,16f7.4,6(/,8x,16f7.4))
      endif

      return
      end
