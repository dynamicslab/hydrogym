c-----------------------------------------------------------------------
      subroutine ps_write(x3,y3,z3,s,n)
c     print string s at (x,y,z)
c
      include 'devices.inc'
      logical iftmp
c
      character*1 s(1)
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
      if (iftmp .and. ifhard0 .and. ifposts) then
         xpc = 550-500* yscr(yc)
         ypc =  50+500* xscr(xc)
         rc  =     500*(xscr(dr)-xscr(0))
         write(45,2) xpc,ypc,'(
    2    format('newpath',3f12.4,' 0 360 arc closepath stroke')
c   2    format('newpath',3f12.4,' 0 360 arc fill closepath stroke')
      endif
c
      ifhard=iftmp
      return
      end
c-----------------------------------------------------------------------
