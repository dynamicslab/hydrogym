      subroutine nek_flops(flops,mflops)
      real*4 rtime,ptime,mflops
      integer*8 flops

      call getflops_papi(flops,mflops)

      return
      end

      subroutine getflops_papi(flops,mflops)
#ifdef PAPI
      include 'f77papi.h'
      real*4 rtime,ptime,mflops
      integer*8 flops

      call papif_flops(rtime,ptime,flops,mflops,ierr)
      if(ierr.gt.0) then
        flops = -1
        mflops = -1
      endif
#endif
 
      return
      end 
