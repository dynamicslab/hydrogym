c-----------------------------------------------------------------------
      subroutine lb_setqvol(flocal,qvol,in,m,isInactive)
c
c     Compute user specified volumetric source term vector using
c     flocal(real inout(m),m) for all 'active' (isInactive==0) 
c     fluid points
c
      include 'SIZE'
c
      real qvol(lx1,ly1,lz1,lelv,*),in(lx1,ly1,lz1,lelt,*)
      integer isInactive(lx1,ly1,lz1,*)

      integer lb_imap(lx1*ly1*lz1*lelv)
      common /lbr/ buf(ldimt,lx1*ly1*lz1*lelv)
      real buf

      external flocal

      lqvol = lx1*ly1*lz1*lelv
      lin   = lx1*ly1*lz1*lelt
      ntot  = lx1*ly1*lz1*nelv

      ! pack input data
      n = 0
      do i = 1,ntot
        if(isInactive(i,1,1,1).eq.0) then
          n = n + 1              
          lb_imap(n) = i
          do j = 1,m
             k = (j-1)*lin + i
             buf(j,n) = in(k,1,1,1,1)
          enddo 
        endif     
      enddo

      ! distribute input data ->compute output data ->transfer back
      nmax = lin 
      call lb_process_items(n,buf,flocal,m,nmax)

      ! unpack output data 
      do i = 1,n
         do k = 1,m
            j = (k-1)*lqvol + lb_imap(i)
            qvol(j,1,1,1,1) = buf(k,i) 
         enddo  
      enddo  

      return
      end
c-----------------------------------------------------------------------
      subroutine lb_process_items(nin,rdata,flocal,m,nmax)
c
      include 'SIZE'
      common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal

      real rdata(1)
      external flocal    

      integer*8 i8gl_running_sum, i8rsum
      integer*8 n8, np8, nb8

      common /scrns/ vi(2,lx1*ly1*lz1*lelt)
      integer vi

      integer icalld,cr_lb
      data icalld /0/
      save icalld,cr_lb

      parameter(kid = 1) ! column to store local id 
      parameter(kp  = 2) ! column to store rank tag

      real tcomm
      data tcomm /0.0/
      save tcomm

      n  = nin
      n0 = n

      if(icalld.eq.0) then
        call fgslib_crystal_setup(cr_lb,nekcomm,npp)
        icalld = 1
      endif

      ! partition into chunks 
      ! note: simple approach but not of approx. equal size
      n8  = n
      np8 = npp
      ng8 = i8glsum(n8,1)
      nb8 = ng8/np8
      do i = 0,mod(ng8,np8)-1
         if(nid.eq.i) nb8 = nb8 + 1
      enddo
      i8rsum = i8gl_running_sum(n8) - n8 
      do i=1,n
         vi(kid,i) = i
         ig = i8rsum + i
         vi(kp ,i) = (ig-1)/nb8
      enddo

      if (loglevel.gt.2) then
        n0_max = iglmax(n0,1)
        n0_min = iglmin(n0,1)
      endif

      etime = dnekclock_sync()
      call fgslib_crystal_tuple_transfer
     &     (cr_lb,n,nmax,vi,2,vl,0,rdata,m,kp)
      tcomm = tcomm + dnekclock_sync() - etime

      if (loglevel.gt.2) then
        n_max = iglmax(n,1)
        n_min = iglmin(n,1)
      endif

      do j = 1,n
         jj = (j-1)*m + 1
         call flocal(rdata(jj),m)
      enddo

      etime = dnekclock_sync()
      call fgslib_crystal_tuple_transfer
     &     (cr_lb,n,nmax,vi,2,vl,0,rdata,m,kp)
      tcomm = tcomm + dnekclock_sync() - etime

      if (n.gt.nmax) call exitti('lb_process_items nmax too small$',n)
      if (n.ne.n0)   call exitti('lb_process_items unexpected n$',n)

      key = kid ! restore based on local id
      call fgslib_crystal_tuple_sort
     &     (cr_lb,n,vi,2,vl,0,rdata,m,key,1)

      if (loglevel.gt.2 .and. nid.eq.0) then
         write(6,*) 'lb before nmax/nmin:', n0_max, n0_min
         write(6,*) 'lb after  nmax/nmin:', n_max , n_min
         write(6,*) 'lb tcomm           :', tcomm
      endif

      return
      end
