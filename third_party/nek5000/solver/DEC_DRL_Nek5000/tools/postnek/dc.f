c-----------------------------------------------------------------------
      subroutine draw_circ1(x3,y3,z3,r)
      include 'devices.inc'
c     draws a circle around XX,YY,ZZ
c
      xm=xiso(x3-r,y3,z3)
      ym=yiso(x3-r,y3,z3)
      xp=xiso(x3+r,y3,z3)
      yp=yiso(x3+r,y3,z3)
      dx = (xp-xm)**2 + (yp-ym)**2
c
      xm=xiso(x3,y3-r,z3)
      ym=yiso(x3,y3-r,z3)
      xp=xiso(x3,y3+r,z3)
      yp=yiso(x3,y3+r,z3)
      dy = (xp-xm)**2 + (yp-ym)**2
      dr = max(dx,dy)
c
      if (if3d) then
         xm=xiso(x3,y3,z3-r)
         ym=yiso(x3,y3,z3-r)
         xp=xiso(x3,y3,z3+r)
         yp=yiso(x3,y3,z3+r)
         dz = (xp-xm)**2 + (yp-ym)**2
         dr = max(dr,dz)
      endif
c
      dr = 0.5*sqrt(dr)
c
      xc=xiso(x3,y3,z3)
      yc=yiso(x3,y3,z3)
c     write(6,1) xc,yc,x3,y3,z3,dr,r
    1 format(8f10.5)
c
      x = xc+dr
      y = yc
      call movec(x,y)
c
      n = 100
      pi2n = 8.*atan(1.0)/n
      do i=1,n
         theta = i*pi2n
         x = xc + dr*cos(theta)
         y = yc + dr*sin(theta)
         call drawc(x,y)
      enddo
      return
      end
c-----------------------------------------------------------------------
