      subroutine lb_process_items(n_,rdata,flocal,m,nmax)
c
      include 'SIZE'
      common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal

      real rdata(1)
      external flocal    

      integer*8 i8glsum, i8gl_running_sum
      integer*8 irsum, nn

      common /scrns/ vi(2,lx1*ly1*lz1*lelt)
      integer vi

      integer icalld,cr_lb
      data icalld /0/
      save icalld,cr_lb

      kp = 2 ! column to store rank tag
      n = n_

      if(icalld.eq.0) then
        call crystal_setup(cr_lb,nekcomm,npp)
        icalld = 1
      endif

      ! create processor mapping
      nn = (i8glsum(nn,1)-1)/npp + 1
      irsum = i8gl_running_sum(n) - n 
      do i=1,n
         vi(1,i) = i              ! local id
         vi(2,i) = (irsum+i-1)/nn ! rank tag 
      enddo

      ! transfer input data to target processor
      n0 = n
      call crystal_tuple_transfer
     &     (cr_lb,n,nmax,vi,2,vl,0,rdata,m,kp)

      ! perform local computation
c      write(6,*) nid, n0, n
      do j = 1,n
         jj = (j-1)*m + 1
         call flocal(rdata(jj),m)
      enddo

      ! transfer output back to original processor
      call crystal_tuple_transfer
     &     (cr_lb,n,nmax,vi,2,vl,0,rdata,m,kp)

      ! restore original order
      key = 1 ! based on local id
      call crystal_tuple_sort
     &     (cr_lb,n,vi,2,vl,0,rdata,m,key,1)

      return
      end
c-----------------------------------------------------------------------
      subroutine lb_setqvol(flocal,out,in,bmask,m)
c
c     Set user specified volumetric forcing function (e.g. heat source)
c     flocal for all 'active' points (bmask is 1) using a simple    
c     load balancing mechanism.
c
      include 'SIZE'
c
      real out(1),in(1)
      integer bmask(1)

      integer lb_imap(lx1*ly1*lz1*lelt)
      common /lbr/ buf(ldimt,lx1*ly1*lz1*lelt)
      real buf

c      common /VPTSOL/  dummy(6*lx1*ly1*lz1*lelv)
c     &                ,buf(ldimt,lx1*ly1*lz1*lelt)
      external flocal

      ntot = nx1*ny1*nz1*nelt
      ltot = lx1*ly1*lz1*lelt

      ! pack input data
      n = 0
      do i = 1,ntot
        if(bmask(i).eq.1) then  ! is this point active?
          n = n + 1              
          lb_imap(n) = i
          do j = 1,m
             k = (j-1)*ltot + i
             buf(j,n) = in(k)
          enddo 
        endif     
      enddo

      ! distribute input data ->compute output data ->transfer back
      call lb_process_items(n,buf,flocal,m,ltot)

      ! unpack output data 
      do i = 1,n
         do k = 1,m
            j = (k-1)*ltot + lb_imap(i)
            out(j) = buf(k,i) 
         enddo  
      enddo  

      return
      end
