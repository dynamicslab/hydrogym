c-----------------------------------------------------------------------
      program int_tp
c
c     Interpolate from a tensor-product SE geometry to a regular tp array
c
c     NOTE:  COMPILE in real*4 mode only, because of byte_read/write.
c
c
      parameter (maxpts = 19000000)
c     parameter (maxpts = 35000000)

      parameter (nmax   = 4*maxpts)
c     parameter (nmax   = 8*maxpts)
      common /c1/ uin (nmax)
      common /c2/ uout(nmax)
 
      parameter (lmax=3000)             ! max old nel in x,y,z
c     parameter (lmax=10000)            ! max old nel in x,y,z  ! in get*_geo
      common /gold/ xe(0:lmax),ye(0:lmax),ze(0:lmax)
      common /iold/ nelx,nely,nelz,nx,ny,nz
 
      parameter (mmax=1000)
c     parameter (mmax=1200)		! max new nel
      common /gnew/ xm(0:mmax),ym(0:mmax),zm(0:mmax)
 
      character*40 fname
      logical      if_byte_sw,if3d,ifreg_out,ifbyte,iffbin
c
c     Start reading .fld files
c
      open(unit=10,file='file.list',status='old')
      nfiles = 0
 
      write(6,*) 'Regime of output grid: 1 - uniform, 2 - SEM,
     $ 3 - non-uniform ? '
      read (5,*) iuniform
      ifreg_out = .true.
      if (iuniform.eq.2) ifreg_out = .false.
 
      do ifile = 1,100000
 
         call blank(fname,40)
         read(10,10,end=101) fname
         write(6,*) ifile,' ',fname
   10    format(a40)

         write(6,*) 'Input File Format: 0 - ASCII, 1 - binary(fld),
     $   2 - binary(0.f0000?) ? '
         read (5,*) ib
         ifbyte=.false.
         iffbin=.false.
         if(ib.ne.0) ifbyte = .true.
         if(ib.eq.2) iffbin = .true.

         call get_old_data(uin,nx,ny,nz,nel,nfld,if_byte_sw,if3d,fname,
     $                                               iffbin,ifbyte,ierr)
         call xyz_chk(uin,nx,ny,nz,nel,nfld,if3d)
         if (ierr.ne.0) stop
         nfiles = nfiles+1
 
         ntot = nel*nx*ny*nz
         n    = nfld*ntot          ! u,v,w,p.t -- 5 variables
         if (n.gt.nmax) then
            write(6,*) 'recompile with new maxpts=nmax:',nmax,n,nel,nx,
     $                                                   nfld
            call exit
         endif
 
         if (ifile.eq.1) then
c           call get_old_geom (xe,nelx,ye,nely,ze,nelz,lmax)        ! SE geometry
            call get_gen_box  (xe,nelx,ye,nely,ze,nelz,lmax,'old')  ! SE geometry
            call get_new_geom 
     $           (xm,mx,ym,my,zm,mz,nenx,neny,nenz,mmax,if3d,iuniform)
            call set_interpolation_ops(xe,nelx,ye,nely,ze,nelz
     $                                ,nx,ny,nz,xm,mx,ym,my,zm,mz,if3d)
 
            mz1  = mz + 1
            if(.not.if3d) 
     $      mz1  = 1
            nout = (mx+1)*(my+1)*mz1
                 
            if (nout.gt.maxpts) then
               write(6,*) 'Increase maxpts:',maxpts,nout,mx,my,mz
               call exit
            endif
 
         endif
 
         call tp_interp  (uout,mx,my,mz,nfld
     $                   ,uin ,nx,ny,nz,nelx,nely,nelz,if3d)
 
         if (ifreg_out) then
            write(6,50) mx+1,my+1,mz1,nout,ifile
            call tp_out  (uout,nfld,nout,ifile)
            write(6,*) 'done tp.out',ifile
         else
            call sem_out
     $           (uin,uout,nfld,nout,mx,my,mz,nenx,neny,nenz,ifile,if3d)
         endif
 
      enddo
   50 format('Writing ',i12,' x ',i12,' x ',i12,'  =  ',i18,' points
     $ to file ',i6)
      stop
 
  101 continue
      write(6,*) ' Done : ',ifile-1, ' files'
 
      stop
      end
c-----------------------------------------------------------------------
      subroutine add2(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) + y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine copy(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine sub2(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) - y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2sq(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) + y(i)*y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine izero(x,n)
      integer x(1)
      do i=1,n
         x(i) = 0
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine rzero(x,n)
      real x(1)
      do i=1,n
         x(i) = 0.
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cmult(x,c,n)
      real x(1),c
      do i=1,n
         x(i) = x(i) * c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine transpose(a,lda,b,ldb)
      real a(lda,1),b(ldb,1)
 
      do j=1,ldb
         do i=1,lda
            a(i,j) = b(j,i)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine blank(x,n)
      character*1 x(1)
      do i=1,n
         x(i) = ' '
      enddo
      return
      end
c-----------------------------------------------------------------------
      function glmin(x,n)
      real x(1)
 
      s = x(1)
 
      do i=1,n
         s = min(s,x(i))
      enddo
      glmin = s
      return
      end
c-----------------------------------------------------------------------
      function glmax(x,n)
      real x(1)
 
      s = x(1)
 
      do i=1,n
         s = max(s,x(i))
      enddo
      glmax = s
      return
      end
c-----------------------------------------------------------------------
      function ltrunc(string,l)
      character*1 string(l)
      character*1   blnk
      data blnk/' '/
 
      do 100 i=l,1,-1
         l1=i
         if (string(i).ne.blnk) goto 200
  100 continue
      l1=0
  200 continue
      ltrunc=l1
      return
      end
c-----------------------------------------------------------------------
      subroutine chcopy(a,b,n)
      character*1 a(1), b(1)
      do 100 i = 1, n
 100     a(i) = b(i)
      return
      end
c-----------------------------------------------------------------------
      logical function if_byte_swap_test(bytetest)
 
      real*4 bytetest
      real*4 test_pattern
      save   test_pattern
 
      integer icalld
      save    icalld
      data    icalld /0/
 
      test_pattern = 6.54321
      if_byte_swap_test = .false.
      if (bytetest.ne.test_pattern) if_byte_swap_test = .true.
 
      if (icalld.eq.0) write(6,*)'Byte swap:',if_byte_swap_test,bytetest
      icalld = 1
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_old_data(x,nx,ny,nz,nel,nfld,if_byte_sw,if3d,fname,
     $                                               iffbin,ifbyte,ierr)
 
      character*40 fname
      integer fnamei(10)
      character*1 fnamec(40)
      equivalence (fnamec,fnamei)
      real x(1)
   
      logical iffbin,ifbyte

      ierr = 1
c
c     Open file
c
      if(ifbyte) then
        len = ltrunc    (fname,40)
        call izero      (fnamei,10)
        call chcopy     (fnamec,fname,len)
        call byte_open  (fnamec)
        if(iffbin) then 
          call get_bin_data2(x,nx,ny,nz,nel,nfld,if_byte_sw,if3d)
        else
          call get_bin_data (x,nx,ny,nz,nel,nfld,if_byte_sw,if3d)
        endif
        call byte_close()
      else
        open(unit=8,file=fname,status='old')
        call get_data(x,nx,ny,nz,nel,nfld,if3d)
        close(unit=8)
      endif

      ierr = 0

      return
      end
c-----------------------------------------------------------------------
      subroutine get_bin_data(x,nx,ny,nz,nel,nfld,if_byte_sw,if3d)

      real*4 x(1)

      common /c132/  s132
      character*132 s132
      character*1  s1321(132)
      equivalence (s132,s1321)
  
      logical if_byte_sw, if_byte_swap_test, if3d
      common /byte_key/ bytetest

c
c     Get 132 character string 
c
      call blank(s132,132)
      call byte_read(s132,33)
 
      open(unit=22,file='tmp')
      write(22,80) s132
   80 format(a132)
      rewind(22)
      read (22,*,err=100,end=100) nel,nx,ny,nz
      close(unit=22)
 
      if3d = .false.
      if (nz.gt.1) if3d = .true.
 
      call get_nfld(nfld,s132,nel,if3d)
 
      ntot = nel*nx*ny*nz
      write(6,*) 'this is ntot:',nel,nx,ny,nz,nfld
 
c
c     Test header
c
      call byte_read(bytetest,1)
      if_byte_sw = if_byte_swap_test(bytetest)
 
      n = nfld*ntot
      call byte_read(x,n)  ! Should be "word_read"
      if (if_byte_sw) call byte_reverse(x,n)
 
      return
 100  continue
      write(6,*) "Error reading tmp file!  ABORT!!"
      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine get_bin_data2(x,nx,ny,nz,nel,nfld,if_byte_sw,if3d)

      real*4 x(1)
      real*4 tmp(1)

      character*132 hdr
      character*1  hdr1(132)
      equivalence (hdr,hdr1)
  
      logical if_byte_sw, if_byte_swap_test, if3d
      common /byte_key/ bytetest
    
      character*4 dummy
      logical ifx,ifu

c
c     Get 120 character string 
c
      call blank(hdr,132)
      call byte_read(hdr,33)
 
      open(unit=22,file='tmp')
      write(22,132) hdr
  132 format(a132)
      rewind(22)
      read (22,*,err=100,end=100) dummy,iwds,nx,ny,nz,nel
      close(unit=22)
 
      if3d = .false.
      if (nz.gt.1) if3d = .true.
      ndim = 2
      if(if3d) ndim=3
 
      call parse_hdr(nfld,hdr,nel,if3d,ifx,ifu)
 
      write(6,*) 'this is ntot:',nel,nx,ny,nz,nfld
 
c
c     Test header
c
      call byte_read(bytetest,1)
      if_byte_sw = if_byte_swap_test(bytetest)
      
      call read_bin_u(x,nx,ny,nz,nel,nfld,ndim,if_byte_sw,ifx,ifu)

      return

 100  continue
      write(6,*) "Error reading header!  ABORT!!"
      call exit
 
      return
      end
c-----------------------------------------------------------------------
      subroutine read_bin_u(u,nx,ny,nz,nel,nfld,ndim,if_byte_sw,ifx,ifu)

      real*4 u(nx*ny*nz,nfld,nel)
      real*4 tmp(nx*ny*nz*nel*nfld)
      logical if_byte_sw,ifx,ifu

      call byte_read(tmp,nel) !dummy read first nel 

      nxyz = nx*ny*nz
      ntot = nx*ny*nz*nel
      n    = nfld*ntot

      call byte_read(tmp(1),n) 
      if (if_byte_sw) call byte_reverse(tmp,n)

      ifld = 1
      ioff = 1
      if(ifx) then
        do i =1,nel
           ioffs = ioff+(i-1)*ndim*nxyz
           call copy(u(1,ifld,i),tmp(ioffs),nxyz)
           call copy(u(1,ifld+1,i),tmp(ioffs+nxyz),nxyz)
           if(ndim.eq.3) call copy(u(1,ifld+2,i),tmp(ioffs+2*nxyz),nxyz)
        enddo
        ifld = ifld+ndim
        ioff = ioff+ndim*nxyz*nel
      endif

      if(ifu) then
        do i =1,nel
           ioffs = ioff+(i-1)*ndim*nxyz
           call copy(u(1,ifld,i),tmp(ioffs),nxyz)
           call copy(u(1,ifld+1,i),tmp(ioffs+nxyz),nxyz)
           if(ndim.eq.3) call copy(u(1,ifld+2,i),tmp(ioffs+2*nxyz),nxyz)
        enddo
        ifld = ifld+ndim
        ioff = ioff+ndim*nxyz*nel
      endif

      do j=ifld,nfld
        do i =1,nel
           ioffs = ioff+(i-1)*nxyz
           call copy(u(1,j,i),tmp(ioffs),nxyz)
        enddo
        ioff=ioff+nxyz*nel
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine get_data(x,nx,ny,nz,nel,nfld,if3d)

      common /c132/  s132
      character*132 s132
      character*1  s1321(132)
      equivalence (s132,s1321)
  
      logical  if3d
      common /byte_key/ bytetest

      real x(1)
 
c
c     Get 132 character string 
c
      call blank(s132,132)
      read(8,80) s132
  
      open(unit=22,file='tmp')
      write(22,80) s132
   80 format(a132)
      rewind(22)
      read (22,*,err=100,end=100) nel,nx,ny,nz
      close(unit=22)
 
      if3d = .false.
      if (nz.gt.1) if3d = .true.
 
      call get_nfld(nfld,s132,nel,if3d)
 
      ntot = nel*nx*ny*nz
      write(6,*) 'this is ntot:',nel,nx,ny,nz,nfld
 
      call read_u(x,nx,ny,nz,nel,nfld)
   
      return

 100  continue
      write(6,*) "Error reading tmp file!  ABORT!!"
      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine read_u(u,nx,ny,nz,nel,nfld)
      real*4 u(nx*ny*nz,nfld,nel)

c     Dummy read first nel
      read(8,*,end=100) (tmp,i=1,nel)

      nxyz = nx*ny*nz
      do i = 1,nel
         read(8,*,err=100,end=100) 
     $    ((u(ixyz,ifld,i),ifld=1,nfld),ixyz=1,nxyz) 
      enddo
      return

 100  write(6,*) 'ERROR READING DATA! ',nfld,ntot,i
      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine out_data(x,n,if_byte_sw,fname,y)
 
      real*4 x(1),y(1)
      character*40 fname
      common /c132/  s132
      character*80 s132
 
      logical if_byte_sw
      common /byte_key/ bytetest
c
c     Open file
c
      call byte_open(fname)
c
c     Write original header
c
      call byte_write(s132,33)
      call byte_write(bytetest,1)
 
      call copy(y,x,n)
      if (if_byte_sw) call byte_reverse(y,n)
      call byte_write(y,n)
 
      call byte_close()
 
      return
      end
c-----------------------------------------------------------------------
      subroutine stats(x,n,name)
      character*3 name
      real x(n,1)
      real ymax(5),ymin(5)
 
      do k=1,5
         ymax(k) = glmax(x(1,k),n)
         ymin(k) = glmin(x(1,k),n)
      enddo
 
      write(6,*)
      write(6,1) name,' min ',(ymin(k),k=1,5)
      write(6,1) name,' max ',(ymax(k),k=1,5)
 
   1  format(a3,a5,1p6e13.5)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine dnorm (d2,di,x1,x2,n)
      real x1(n),x2(n)
 
      d2 = 0.
      di = 0.
      do i=1,n
         dif = x1(i)-x2(i)
         di  = max(di,abs(dif))
         d2  = d2 + dif*dif
      enddo
      if (n.gt.0.and.d2.gt.0) d2 = sqrt(d2)/n
      return
      end
c-----------------------------------------------------------------------
      subroutine set_interpolation_ops(xe,nelx,ye,nely,ze,nelz
     $                          ,nx,ny,nz
     $                          ,xm,mx,ym,my,zm,mz,if3d)
 
      real xe(0:nelx),ye(0:nely),ze(0:nelz)
      real xm(0:mx),ym(0:my),zm(0:mz)
      logical if3d
 
      parameter (lmax=3000)
      parameter (lx1=20)
      common /jops/ jx(lx1*lx1,lmax),jy(lx1*lx1,lmax),jz(lx1*lx1,lmax)
      real jx,jy,jz
 
      common /iptr/ px(0:lmax),py(0:lmax),pz(0:lmax)
      integer px,py,pz
 
      call set_interpolation_ops_1d(jx,nx,px,xe,nelx,xm,mx,.false.)
      call set_interpolation_ops_1d(jy,ny,py,ye,nely,ym,my,.true.)
      if (if3d) 
     $call set_interpolation_ops_1d(jz,nz,pz,ze,nelz,zm,mz,.true.)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine set_interpolation_ops_1d(jx,nx,pe,xe,nelx,xm,mx
     $                                   ,iftranspose)
c
c     OUTPUT:
c
c     jx(1,e), e = 1...nelx:   set of interpolation operators
c
c     pe(0:e), e = 1...nelx:   pointers for the index set for element e
c
c
      parameter(lx1=20)
      real jx(lx1*lx1,nelx)
      real xe(0:nelx),xm(0:mx)
      real wk(lx1*lx1),z(lx1),r(lx1*lx1)
      logical iftranspose
      integer pe(0:nelx),e
 
 
      call izero(pe,nelx+1)
 
      call zwgll(z,wk,nx)
 
      
      do je=1,nelx-1  ! First, count # in each interval
      do i=0,mx
         if (xe(je-1).le.xm(i).and.xm(i).lt.xe(je)) then
            pe(je) = pe(je) + 1
         endif
      enddo
      enddo
 
      do je=nelx,nelx    ! First, count # in each interval
      do i=0,mx
         if (xe(je-1).le.xm(i).and.xm(i).le.xe(je)) then
            pe(je) = pe(je) + 1
         endif
      enddo
      enddo
 
      i0 = 0
      do i=0,mx    ! Find starting point (1st xm point >= xe(0)
         if (xm(i).ge.xe(0)) then
            i0 = i
            goto 10
         endif
      enddo
   10 continue
 
      pe(0) = i0
      do e=1,nelx                   ! Convert count to pointers
         pe(e) = pe(e-1) + pe(e)
      enddo
c
c     Now, build interpolation operators for all non-empty index sets
c
      do e=1,nelx
         i0 = pe(e-1)
         i1 = pe(e)
         nn = i1-i0
         if (nn.gt.0) then
            x0 = xe(e-1)
            x1 = xe(e)
            do i=1,nn
               j=i0+i-1
               r(i) = 2*(xm(j)-x0)/(x1-x0) - 1.
            enddo
            if (nn*nx.gt.lx1*lx1) then
               write(6,*) nn,nx,lx1,e,' ERROR - increase lx1.'
               call exit
            endif
            if (nn.gt.lx1*lx1) then
               write(6,*) nn,lx1,e,' ERRO2 - increase lx1.'
               call exit
            endif
 
            call igllm (jx(1,e),wk,z,r,nx,nn,nx,nn)
            if (iftranspose) then  ! transpose for y and z
               call copy(wk,jx(1,e),nx*nn)
               call transpose(jx(1,e),nx,wk,nn)
            endif
         endif
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_new_geom 
     $           (xm,mx,ym,my,zm,mz,nenx,neny,nenz,mmax,if3d,iuniform)
 
      real xm(0:mmax),ym(0:mmax),zm(0:mmax)
      logical if3d
 
      if (iuniform.eq.1) then   ! Uniform mesh
       call get_uni_geo     (xm,mx,ym,my,zm,mz,mmax,if3d)
      elseif (iuniform.eq.2) then 
       call get_gen_box_new (xm,mx,ym,my,zm,mz,nenx,neny,nenz,mmax,if3d)
      elseif (iuniform.eq.3) then 
       call get_nonuni_geo  (xm,mx,ym,my,zm,mz,mmax,if3d)
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_uni_geo(xm,mx,ym,my,zm,mz,mmax,if3d)
c     "uniform" mesh
      real xm(0:mmax),ym(0:mmax),zm(0:mmax)
      logical if3d,ifpx,ifpy,ifpz
 
      parameter (lmax=3000)
      common /gold/ xe(0:lmax),ye(0:lmax),ze(0:lmax)
      common /iold/ nelx,nely,nelz,nx,ny,nz

      character*1 opt
 
      write(6,*) 'input number of cells in x,y, and z:'
      read (5,*) mx,my,mz
 
      if (mx.gt.mmax.or.my.gt.mmax.or.mz.gt.mmax) then
         write(6,*) 'increase mmax. quit in get_uni_geom'
         write(6,*) mx,my,mz,mmax
         call exit
      endif
 
      ifpx = .false.   ! Turn on/off periodicity (for FFT data)
      ifpy = .false.
      ifpz = .false.
      write(6,*) "Periodic in X direction? (y/n) "
      read(5,*) opt
      if(opt.eq.'Y'.or.opt.eq.'y') ifpx=.true.

      write(6,*) "Periodic in Y direction? (y/n) "
      read(5,*) opt
      if(opt.eq.'Y'.or.opt.eq.'y') ifpy=.true.
     
      if(if3d) then
        write(6,*) "Periodic in Z direction? (y/n) "
        read(5,*) opt
        if(opt.eq.'Y'.or.opt.eq.'y') ifpz=.true.
      endif
      if(ifpx.or.ifpy.or.ifpz) write(6,*) "NOTE: periodicity will 
     $   result in a shift of (h/2)! "

c
c     UNIFORM MESH 
c
      x0 = xe(0)
      x1 = xe(nelx)
      dx = (x1-x0)/mx
      do i=0,mx-1
         xm(i) = x0 + i*dx
      enddo
      xm(mx) = x1
      if (ifpx) then
         mx = mx-1   ! periodicity
         do i=0,mx-1
            xm(i) = xm(i)+(dx/2)
         enddo
      endif
 
      y0 = ye(0)
      y1 = ye(nely)
      dy = (y1-y0)/my
      do i=0,my-1
         ym(i) = y0 + i*dy
      enddo
      ym(my) = y1
      if (ifpy) then
         my = my-1   ! periodicity
         do i=0,my-1
            ym(i) = ym(i)+(dy/2)
         enddo
      endif
 
      if (if3d) then
         z0 = ze(0)
         z1 = ze(nelz)
         dz = (z1-z0)/mz
         do i=0,mz-1
            zm(i) = z0 + i*dz
         enddo
         zm(mz) = z1
         if (ifpz) then
            mz = mz-1   ! periodicity
            do i=0,mz-1
               zm(i) = zm(i)+(dz/2)
            enddo
         endif
      endif

 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_nonuni_geo(xm,mx,ym,my,zm,mz,mmax,if3d)
 
      logical if3d
      parameter (lmax=3000)
      real xm(0:lmax),ym(0:lmax),zm(0:lmax)
 
      mz = 1.
 
      write(6,*) "Opening file 'new.geom' to read pts "
      open(unit=20,file='new.geom',err=100)
 
      read(20,*) mx
      do i=0,mx
         read(20,*) xm(i)
      enddo
 
      read(20,*) my
      do i=0,my
         read(20,*) ym(i)
      enddo
 
      if (if3d) then
      read(20,*) mz
      do i=0,mz
         read(20,*) zm(i)
      enddo
      endif
      close(unit=20)
 
      if (mx.gt.lmax.or.my.gt.lmax.or.(if3d.and.mz.gt.lmax)) then
         write(6,*) 'increase lmax. quit in get_nonuni_geo'
         write(6,*) mx,my,mz,lmax
         call exit
      endif
   
      return

 100  continue 
      write(6,*) "Error opening new.geom!!  ABORT"
      call exitt
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_gen_box_new
     $                 (xm,mx,ym,my,zm,mz,nelx,nely,nelz,mmax,if3d)
 
      real xm(0:mmax),ym(0:mmax),zm(0:mmax)
      logical if3d
 
      parameter (lmax=3000)
      common /enew/ xe(0:lmax),ye(0:lmax),ze(0:lmax)
 
      call get_gen_box  (xe,nelx,ye,nely,ze,nelz,lmax,'new')  ! SE geometry
 
c     call outmat(xe,1,nelx+1,'x gen3',nelx)
c     call outmat(ye,1,nely+1,'y gen3',nely)
c     call outmat(ze,1,nelz+1,'z gen3',nelz)
 
      write(6,*) 'Input polynomial degree for new mesh:'
      read (5,*) nx
      nx1 = nx+1
      ny1 = nx+1
      nz1 = nx+1
 
      call get_gen_box_new_1d (xm,mx,nx1,xe,nelx,mmax,' X ')
      call get_gen_box_new_1d (ym,my,ny1,ye,nely,mmax,' Y ')
      mz = 1
      if (if3d) call get_gen_box_new_1d (zm,mz,nz1,ze,nelz,mmax,' Z ')
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_gen_box_new_1d (xm,mx,nx1,xe,nel,mmax,name3)
 
      real xm(0:mmax),xe(0:nel)
      character*3 name3
      integer e
 
      parameter (nmax=20)
      common /znew/ zg(nmax),wg(nmax)
 
 
      if (nx1.gt.nmax) then
         write(6,*) 'Error, nx1 too large:',nx1,nmax,name3
         call exitt
      endif
 
      call zwgll(zg,wg,nx1)
 
      k=0
      do e=1,nel
         nxx=nx1-1
         if (e.eq.nel) nxx=nx1
         dz = (xe(e)-xe(e-1))/2.
         do i=1,nxx
            xm(k) = xe(e-1) + dz*(zg(i)+1)
            k     = k+1
         enddo
      enddo
      mx = k-1
 
      if (mx.gt.mmax) then
         write(6,*) 'Error, mx too large:',mx,mmax,name3
         call exitt
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine tp_interp  (uout,mx,my,mz,nfld
     $                      ,uin ,nx,ny,nz,nelx,nely,nelz,if3d)
 
      real uout(0:mx,0:my,0:mz,nfld)
      real uin (nx,ny,nz,nfld,nelx,nely,nelz)
      logical if3d
 
      if (if3d) then
         call tp_interp_3d (uout,mx,my,mz,nfld
     $                     ,uin ,nx,ny,nz,nelx,nely,nelz)
      else
         call tp_interp_2d (uout,mx,my,nfld,uin ,nx,ny,nelx,nely)
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine tp_interp_3d (uout,mx,my,mz,nfld
     $                      ,uin ,nx,ny,nz,nelx,nely,nelz)
 
      real uout(0:mx,0:my,0:mz,nfld)
      real uin (nx,ny,nz,nfld,nelx,nely,nelz)
 
 
      parameter(lx1=20)
      common /jwrk/ w1(lx1*lx1*lx1),w2(lx1*lx1*lx1)
 
      parameter (lmax=3000)
      real jx,jy,jz
      common /jops/ jx(lx1*lx1,lmax),jy(lx1*lx1,lmax),jz(lx1*lx1,lmax)
 
      common /iptr/ px(0:lmax),py(0:lmax),pz(0:lmax)
      integer px,py,pz
 
      integer ex,ey,ez
 
      do ez=1,nelz
      do ey=1,nely
      do ex=1,nelx
 
         iz0 = pz(ez-1)
         iz1 = pz(ez)
         nnz = iz1-iz0
 
         iy0 = py(ey-1)
         iy1 = py(ey)
         nny = iy1-iy0
 
         ix0 = px(ex-1)
         ix1 = px(ex)
         nnx = ix1-ix0

         nnn = nnx*nny*nnz
         if (nnn.gt.0) then
            nyz = ny*nz
            do ifld=1,nfld
 
               call mxm(jx(1,ex),nnx,uin(1,1,1,ifld,ex,ey,ez),nx,w1,nyz)
 
               iw1=1
               iw2=1
               do iz=1,nz
                  call mxm(w1(iw1),nnx,jy(1,ey),ny,w2(iw2),nny)
                  iw1 = iw1+nnx*ny
                  iw2 = iw2+nnx*nny
               enddo
 
               nxy = nnx*nny
               call mxm(w2,nxy,jz(1,ez),nz,w1,nnz)
 
               k = 0
               do iz=iz0,iz1-1
               do iy=iy0,iy1-1
               do ix=ix0,ix1-1
                  k = k+1
                  uout(ix,iy,iz,ifld) = w1(k)
               enddo
               enddo
               enddo
            enddo
         endif
 
      enddo
      enddo
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine tp_interp_2d (uout,mx,my,nfld,uin,nx,ny,nelx,nely)
 
      real uout(0:mx,0:my,nfld)
      real uin (nx,ny,nfld,nelx,nely)
 
 
      parameter(lx1=20)
      common /jwrk/ w1(lx1*lx1*lx1),w2(lx1*lx1*lx1)
 
      parameter (lmax=3000)
      real jx,jy,jz
      common /jops/ jx(lx1*lx1,lmax),jy(lx1*lx1,lmax),jz(lx1*lx1,lmax)
 
      common /iptr/ px(0:lmax),py(0:lmax),pz(0:lmax)
      integer px,py,pz
 
      integer ex,ey,ez
 
      do ey=1,nely
      do ex=1,nelx
 
         iy0 = py(ey-1)
         iy1 = py(ey)
         nny = iy1-iy0
 
         ix0 = px(ex-1)
         ix1 = px(ex)
         nnx = ix1-ix0

         nnn = nnx*nny
         if (nnn.gt.0) then
            do ifld=1,nfld
               call mxm(jx(1,ex),nnx,uin(1,1,ifld,ex,ey),nx,w2,ny)
               call mxm(w2      ,nnx,jy(1,ey)           ,ny,w1,nny)
 
               k = 0
               do iy=iy0,iy1-1
               do ix=ix0,ix1-1
                  k = k+1
                  uout(ix,iy,ifld) = w1(k)
               enddo
               enddo
            enddo
         endif

      enddo
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_nfld(nfld,s132,nel,if3d)
 
      character*132 s132
 
      common /cexcod/ excoder(10)
      character*2     excoder
 
      logical if3d

      if (nel.lt.10 000) then
         read(s132,'(4i4,1x,g13.4,i5,1x,10a2)'
     $   ,err=1500,end=1500)
     $    neltr,nxr,nyr,nzr,rstime,istepr,(excoder(i),i=1,10)
      else
         read(s132,'(i10,3i4,1P1e18.9,i9,1x,15a2)'
     $   ,err=1500,end=1500)
     $    neltr,nxr,nyr,nzr,rstime,istepr,(excoder(i),i=1,10)
      endif
      print*,neltr,nxr,nyr,nzr,rstime,istepr
C
C     Figure out position of data in file "IFILE"
C
      write(6,*) 'excoder:',(excoder(k),k=1,10)
      nfld = 0
      do i=1,10
         if (excoder(i).eq.'X') then
            nfld = nfld + 2
            if (if3d) nfld = nfld + 1
         endif
         if (excoder(i).eq.'U') then
            nfld = nfld + 2
            if (if3d) nfld = nfld + 1
         endif
         if (excoder(i).eq.'P') then
            nfld = nfld + 1
         endif
         if (excoder(i).eq.'T') then
            nfld = nfld + 1
         endif
         if (excoder(i).eq.' 1') then
            nfld = nfld + 1
         endif
         if (excoder(i).eq.' 2') then
            nfld = nfld + 1
         endif
         if (excoder(i).eq.' 3') then
            nfld = nfld + 1
         endif
         if (excoder(i).eq.' 4') then
            nfld = nfld + 1
         endif
      enddo
 
      return
 
 1500 continue
      write(6,*) s132
      write(6,*) 'Error reading s132. Abort.'
      call exitt
 
      return
      end
c-----------------------------------------------------------------------
      subroutine parse_hdr(nfld,hdr,nel,if3d,ifx,ifu)

      character*132 hdr
 
      common /cexcod/ excoder(10)
      character*2     excoder
      character*10 rdcode
      character*1  rdcode1(10)
      equivalence (rdcode,rdcode1)
 
      logical if3d,ifx,ifu
      character*4 dummy

      read(hdr,*,err=1500,end=1500)
     $ dummy,iwds,nxr,nyr,nzr,neltr,nelr,rstime
     $      ,istepr,ifiler,ifiler2,rdcode

      print*,neltr,nxr,nyr,nzr,rstime,istepr
C
C     Figure out position of data in file "IFILE"
C
      write(6,*) 'excoder:',(rdcode1(k),k=1,10)
      nfld = 0
      ifx = .false.
      ifu = .false.
      k   = 1
      do i=1,10
         if (rdcode1(i).eq.'X') then
            nfld = nfld + 2
            if (if3d) nfld = nfld + 1
            ifx = .true.

            excoder(k  ) = 'X'
            excoder(k+1) = 'Y'
            if(if3d)excoder(k+2) = 'Z'
            k=k+2
            if(if3d) k=k+1
         elseif (rdcode1(i).eq.'U') then
            nfld = nfld + 2
            if (if3d) nfld = nfld + 1
            ifu = .true.

            excoder(k  ) = 'U'
            k=k+1
         elseif (rdcode1(i).eq.'P') then
            nfld = nfld + 1
            excoder(k  ) = 'P'
            k=k+1
         elseif (rdcode1(i).eq.'T') then
            nfld = nfld + 1
            excoder(k  ) = 'T'
            k=k+1
         elseif (rdcode1(i).eq.'S') then
            read(rdcode1(i+1),'(I1)')NPS1
            read(rdcode1(i+2),'(I1)')NPS0
            nps = 10*nps1+nps0
            nfld = nfld + nps
            do j=1,nps
               write(excoder(k),'(I2)') j
               k=k+1
            enddo
         else
            if(k.le.10) then
              excoder(k)=' '
              k=k+1
            endif
         endif
      enddo

      return

 1500 continue
      write(6,*) hdr
      write(6,*) 'Error reading hdr. Abort.'
      call exitt

      return
      end
c----------------------------------------------------------------------
      subroutine xyz_chk(u,nx,ny,nz,nel,nfld,if3d)
      real u(nx,ny,nz,nfld,nel)
      integer e

      n = nx*ny*nz
      do k=1,nfld

         xmn = u(1,1,1,k,1)
         xmx = u(1,1,1,k,1)
         do e=1,nel
         do i=1,n
            xmn = min(xmn,u(i,1,1,k,e))
            xmx = max(xmx,u(i,1,1,k,e))
         enddo
         enddo

         write(6,1) k,n,xmn,xmx,' Umax',nx,ny,nz,nel

      enddo

    1 format(i2,i12,1p2e12.4,a5,3i3,i12)

      return
      end
c-----------------------------------------------------------------------
      subroutine outmat(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6
 
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
      subroutine lex2sem_fld(v,nr,ns,nt,nelx,nely,nelz,nfld,if3d,u)
 
      logical if3d
 
      real v(0:nr,0:ns,0:nt,nfld,1)
      real u(nr*nelx+1,ns*nely+1,nt*nelz+1,nfld)
      integer e,ex,ey,ez

      l = 0

      do kfld=1,nfld
       if (if3d) then
         kk=2
         do ez=1,nelz
            kk=kk-1
            do k=0,nt
               jj=2
               do ey=1,nely
                  jj=jj-1
                  do j=0,ns
                     ii=2
                     do ex=1,nelx
                        ii=ii-1
                        do i=0,nr
                           e=ex+nelx*((ey-1)+nely*(ez-1))
                           v(i,j,k,kfld,e) = u(ii,jj,kk,kfld)
                           ii=ii+1
                        enddo
                     enddo
                     jj=jj+1
                  enddo
               enddo
               kk=kk+1
            enddo
         enddo
       else
         jj=2
         do ey=1,nely
            jj = jj-1
            do j=0,ns
               ii=2
               do ex=1,nelx
                  ii = ii-1
                  do i=0,nr
                     e=ex+nelx*(ey-1)
                     v(i,j,0,kfld,e) = u(ii,jj,1,kfld)
                     ii=ii+1
                  enddo
               enddo
               jj=jj+1
            enddo
         enddo
       endif
 
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine sem_out
     $    (uw,uo,nfld,nout,mx,my,mz,nelx,nely,nelz,ifile,if3d)
 
      real uo(nout,nfld)
      real uw(1)
      logical if3d
 
      character*10 fout
      character*1  ans
      save         ans
      data         ans /' '/

      nx = mx/nelx
      ny = my/nely
      nz = 0
      if (if3d) nz = mz/nelz

      nx1 = nx+1
      ny1 = ny+1
      nz1 = nz+1
      write(6,*) 'nx:',nx,ny,nz,nelx,nely,nelz
      write(6,*) 'nx:',nx1,ny1,nz1,nfld,if3d

      call lex2sem_fld(uw,nx,ny,nz,nelx,nely,nelz,nfld,if3d,uo)

      if (ans.eq.' ') then
         write(6,*) 'formatted output (y/n)?'
         read (5,*) ans
      endif

      if (ans.eq.'y'.or.ans.eq.'Y') then
         write(fout,1) ifile
    1    format('fld.',i6.6)
         open(unit=11,file=fout)
         call outfld_fm(uw,nx1,ny1,nz1,nelx,nely,nelz,nfld,if3d)
         close(11)
      else
         call outfld_unf
     $        (uw,nx1,ny1,nz1,nelx,nely,nelz,nfld,if3d,ifile)
      endif

      write(6,*) 'leave sem.out',ifile

      return
      end
c-----------------------------------------------------------------------
      subroutine outfld_fm
     $    (uw,nx1,ny1,nz1,nelx,nely,nelz,nfld,if3d)
 
      real uw(nx1*ny1*nz1,nfld,1)
      integer e
      logical if3d
 
      common /cexcod/ excoder(10)		! different from prepost.f
      character*2     excoder
 
      istep = 0
      time  = 0
 
      nelt = nelx*nely
      if (if3d) nelt = nelx*nely*nelz
      nxyz = nx1*ny1*nz1

      write(6,*) 'nel:',nelt,nelx,nely,nelz
      write(6,*) 'nxy:',nxyz,nx1,ny1,nz1,nfld
 
      if (nelt.lt.10 000) then
         write(11,'(4i4,1x,1PE13.4,i5,1x,15a2,1x,a12)')
     $   nelt,nx1,ny1,nz1,time,istep,(excoder(i),i=1,10)
      else
         write(11,'(i10,3i4,1P1e18.9,i9,1x,15a2)')
     $   nelt,nx1,ny1,nz1,time,istep,(excoder(i),i=1,10)
      endif
 
      cdrror = 0.
      write(11,'(6g11.4)')(cdrror,i=1,nelt)
 
      do e=1,nelt
      do i=1,nxyz
         write(11,2) (uw(i,k,e),k=1,nfld)
      enddo
      enddo
    2 format(1p6e15.7)
c   2 format(1p7e12.4)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine outfld_unf   ! unformated output
     $    (uw,nx1,ny1,nz1,nelx,nely,nelz,nfld,if3d,ifile)
 
      real uw(nx1*ny1*nz1,nfld,1)
      integer e,f
      logical if3d

      common /cexcod/ excoder(10)		! different from prepost.f
      character*2     excoder

      real*4 test_pattern
      save   test_pattern
      data   test_pattern /6.54321/

      character*132 s132

      integer      fnamei(2)
      character*40 fnamec
      equivalence (fnamec,fnamei)
      character*40 fname
c
c     Open file
c
      call blank      (fname,40)
      write           (fname,1) ifile
    1 format          ('fld.',i4.4)

      call izero      (fnamei,2)
      call chcopy     (fnamec,fname,8)
      call byte_open  (fnamec)


      istep = 0
      time  = 0

      nxyz = nx1*ny1*nz1
      nelt = nelx*nely
      if (if3d) nelt = nelx*nely*nelz

      call blank(s132,132)
      if (nelt.lt.10000) then
         write(s132,'(4i4,1x,1PE13.4,i5,1x,15a2,1x,a12)')
     $   nelt,nx1,ny1,nz1,time,istep,(excoder(i),i=1,10)
      else
         write(s132,'(i10,3i4,1P1e18.9,i9,1x,15a2)')
     $   nelt,nx1,ny1,nz1,time,istep,(excoder(i),i=1,10)
      endif
      call byte_write(s132,33)
      call byte_write(test_pattern,1)


      ioe = 100
      if (nelt.gt.10000) ioe = 1000

      do e=1,nelt
         if (mod(e,ioe).eq.0) write(6,*) 'writing: ',e,nxyz,nfld,ifile
         do f=1,nfld
            call byte_write(uw(1,f,e),nxyz)
         enddo
      enddo
      call byte_close

      return
      end
c-----------------------------------------------------------------------
      subroutine tp_out (uout,nfld,nout,ifile)
 
      real uout(nout,nfld)
 
      character*10 fout
      character*1  ans
      save         ans
      data         ans /' '/
 
      write(fout,1) ifile
    1 format('fld.',i6.6)
 
      if (ans.eq.' ') then
         write(6,*) 'formatted output (y/n)?'
         read (5,*) ans
      endif

      if (ans.eq.'n'.or.ans.eq.'N') then
         open(unit=11,file=fout,form='unformatted')

         write(11) uout
      else
         open(unit=11,file=fout)
         do i=1,nout
            write(11,2) (uout(i,k),k=1,nfld)
         enddo
    2    format(1p6e13.5)
      endif

      close(11)
      write(6,*) 'leave tp.out',ifile
 
      return
      end
c-----------------------------------------------------------------------
      subroutine outu(u,nx,ny,nz,nel,nfld) 
      real*4 u(nx*ny*nz,nfld,nel)
  
       nxyz = nx*ny*nz
       do i=1,nel
       do j = 1,nxyz
          write(41,*) u(j,1,i)
          write(42,*) u(j,2,i)
          write(43,*) u(j,3,i)
          write(44,*) u(j,4,i)
          write(45,*) u(j,5,i)
          write(46,*) u(j,6,i)
          write(47,*) u(j,7,i)
          write(48,*) u(j,8,i)
          write(49,*) u(j,9,i)
       enddo
       enddo

       stop

      return
      end
c-----------------------------------------------------------------------
