      program jbigw_quick
c
      rt = 0.679022969093 ! this is the correct one
      rt = 0.689022969093 
c
      nt = 17
      one= 1.
      pi = 4.*atan(one)
      dt = 2.*pi/nt
      d  = 0
      r0 = 3
      nsteps = 1133
      isteps = 33
      do i=0,nt-1
         th = i*dt
         x  = r0 + rt*cos(th)
         y  =      rt*sin(th)
         z  =      0
         ic = i+2
         write(6,1) x,y,z,d,nsteps,isteps,ic
      enddo
    1 format(3f15.11,f3.0,3i9)
      stop
      end
