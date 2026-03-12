c - - SIZE internals
      integer lelv
      parameter(lelv=lelt)

      integer ly1,lz1
      parameter(ly1=lx1)
      parameter(lz1=1 + (ldim-2)*(lx1-1))

      integer lyd,lzd
      parameter(lyd=lxd)
      parameter(lzd=1 + (ldim-2)*(lxd-1))

      integer ly2,lz2
      parameter(ly2=lx2)
      parameter(lz2=1 + (ldim-2)*(lx2-1))

      integer ly1m,lz1m
      parameter(ly1m=lx1m)
      parameter(lz1m=1 + (ldim-2)*(lx1m-1))

      ! Averaging
      integer ax1,ay1,az1
      parameter (ax1=lx1)
      parameter (ay1=ax1)
      parameter (az1=1 + (ldim-2)*(ax1-1))

      integer ax2,ay2,az2
      parameter(ax2=lx2)
      parameter(ay2=ax2)
      parameter(az2=1 + (ldim-2)*(ax2-1))

      ! Adjoint
      integer lpelv
      parameter(lpelv=lpelt)

      integer lpx1,lpy1,lpz1
      parameter(lpx1=lx1)
      parameter(lpy1=lpx1)
      parameter(lpz1=1 + (ldim-2)*(lpx1-1))

      integer lpx2,lpy2,lpz2
      parameter(lpx2=lx2)
      parameter(lpy2=lpx2)
      parameter(lpz2=1 + (ldim-2)*(lpx2-1))

      ! MHD
      integer lbelv
      integer lbx1,lby1,lbz1
      parameter(lbelv=lbelt)

      parameter(lbx1=lx1)
      parameter(lby1=lbx1)
      parameter(lbz1=1 + (ldim-2)*(lbx1-1))

      integer lbx2,lby2,lbz2
      parameter(lbx2=lx2)
      parameter(lby2=lbx2)
      parameter(lbz2=1 + (ldim-2)*(lbx2-1))

      integer lxz
      parameter (lxz=lx1*lz1)

      integer lzl
      parameter (lzl=3 + 2*(ldim-3))

      integer ldimt1,ldimt3
      parameter (ldimt1=ldimt+1)
      parameter (ldimt3=ldimt+3)

      integer lx3,ly3,lz3
      parameter (lx3=lx1)
      parameter (ly3=ly1)
      parameter (lz3=lz1)

      integer lctmp0,lctmp1
      parameter (lctmp0 =2*lx1*ly1*lz1*lelt)
      parameter (lctmp1 =4*lx1*ly1*lz1*lelt)

      integer maxmor
      parameter (maxmor = lelt)

      integer nio
      common/IOFLAG/ nio  ! for logfile verbosity control

      integer lxs,lys,lzs
      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Preconditioner

      integer maxmbr
      parameter (maxmbr=lelt*6)

      ! cvode
      integer lcvx1,lcvy1,lcvz1
      parameter(lcvx1=lx1)
      parameter(lcvy1=lcvx1)
      parameter(lcvz1=1 + (ldim-2)*(lcvx1-1))

      ! nek-nek
      integer nmaxl_nn,nfldmax_nn
      parameter (nmaxl_nn=
     $          min(1+(nsessmax-1)*2*ldim*lxz*lelt,2*ldim*lxz*lelt))
      parameter (nfldmax_nn=
     $          min(1+(nsessmax-1)*(ldim+1+ldimt),ldim+1+ldimt))

      integer loglevel,optlevel
      common /lolevels/ loglevel,optlevel

      integer       nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
      common/dimn/  nelv,nelt,nfield,npert,nid,idsess
     $ ,nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nxd,nyd,nzd,ndim,ldimr
