      function dnekgflpops()
      real*4 rtime,ptime,mflops
      integer*8 flpops

      call getflops_papi(flpops,mflops)
      dnekgflpops = flpops

      return
      end
c-----------------------------------------------------------------------
      function dnekgflops()
      real*4 mflops
      integer*8 flpops

      call getflops_papi(flpops,mflops)
      dnekgflops = mflops/1e3

      return
      end
c-----------------------------------------------------------------------
      subroutine getflops_papi(flpops,mflops)
#ifdef PAPI
      include 'f77papi.h'
      real*4 rtime,ptime,mflops
      integer*8 flpops

      call papif_flops(rtime,ptime,flpops,mflops,ierr)
      if(ierr.ne.0) then
        flpops = -1
        mflops = -1
      endif
#else
      flpops = -1
      mflops = -0
#endif
 
      return
      end 
