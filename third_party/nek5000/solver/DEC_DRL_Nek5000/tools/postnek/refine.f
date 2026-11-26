      subroutine genadj
      INCLUDE 'basics.inc'
c
      call izero(adje,6*nelm)
      call izero(adjf,6*nelm)
c
      call gencen
      call mkside
c
      do 1200 ie = 1,nel
      do 1100 je = ie+1,nel
c
            DIST1=(XCEN(IE)-XCEN(Je))**2+(YCEN(IE)-YCEN(Je))**2
     $           +(ZCEN(IE)-ZCEN(Je))**2
            DIST2= 1.01*( RCEN(IE) + RCEN(JE) )**2
            IF (DIST2.GE.DIST1) THEN
               epsrr = .01*min(rcen(ie),rcen(je))
               epsrr = epsrr**2
               do 200 is=1,nsides
                  if (adje(ie,is).eq.0) then
                     do 100 js=1,nsides
                        if (adje(je,js).eq.0) then
                           DIST1=(sides(IE,is,1)-sides(JE,js,1))**2
     $                          +(sides(IE,is,2)-sides(JE,js,2))**2
     $                          +(sides(IE,is,3)-sides(JE,js,3))**2
                           if (dist1.le.epsrr) then
                              adje(je,js) = ie
                              adjf(je,js) = is
                              adje(ie,is) = je
                              adjf(ie,is) = js
                           endif
                        endif
  100                continue
                  endif
  200          continue
            endif
 1100    continue
 1200    continue
c
      return
      end
      subroutine zippy(
