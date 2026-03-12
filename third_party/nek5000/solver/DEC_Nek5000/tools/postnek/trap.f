c-----------------------------------------------------------------------
      subroutine gradient(a,b,c,b1,b2)
      include 'basics.inc'
      include 'basicsp.inc'
c
      call prs('Descent alg? (1=GS, 2=S.D., 3=c.g.$')
      call rei(idesct)
      call prs('Input start location x,y:$')
      call rerr(x1,x2)
c
c     if (idesct.eq.1) then
c        do 10 i=1,1000
c           dx1 = 1.0
c           dx1 = 1.0
c     
      return
      end
c-----------------------------------------------------------------------
      subroutine procmap
      include 'basics.inc'
      include 'basicsp.inc'
      character*80 logfile
      return 
      end
c
c-----------------------------------------------------------------------
      subroutine fcnnct 
C 
C     Find connectivity of elements via sort + hash table
C
      include 'basics.inc'
      include 'basicsp.inc'
      dimension cen(nelm),ind(nelm)
c
c     Find "long" direction
c
c     Undo ---- this stuff is only needed if you want to generate
c               the connectivity matrix in latex ; pff 7-1-94
c
      if (nelm.gt.000) return
c
      if (nelm.gt.200) return
      call gencen
      xmax = xbmax(1)
      ymax = ybmax(1)
      zmax = zbmax(1)
      xmin = xbmin(1)
      ymin = ybmin(1)
      zmin = zbmin(1)
      dxmax = 0.0
      dymax = 0.0
      dzmax = 0.0
      do 100 ie = 1,nel
c
c        compute avg element dimensions in each direction
c
         dx = (xbmax(ie)-xbmin(ie))
         dy = (ybmax(ie)-ybmin(ie))
         dz = (zbmax(ie)-zbmin(ie))
c
         dxmax = max(dx,dxmax)
         dymax = max(dy,dymax)
         dzmax = max(dz,dzmax)
c
         xlavg = xlavg + dx
         ylavg = ylavg + dy
         zlavg = zlavg + dz
         xmin = min (xmin,xbmin(ie))
         xmax = max (xmax,xbmax(ie))
         ymin = min (ymin,ybmin(ie))
         ymax = max (ymax,ybmax(ie))
         zmin = min (zmin,zbmin(ie))
         zmax = max (zmax,zbmax(ie))
  100 continue
c
      xlavg = xlavg / float(nel)
      ylavg = ylavg / float(nel)
      zlavg = zlavg / float(nel)
c
      xnavg = (xmax-xmin)/xlavg
      ynavg = (ymax-ymin)/ylavg
      if (if3d) znavg = (zmax-zmin)/zlavg
c
c     Find the (long) direction having the most elements
c
      ilong = 1
      avgnmx = max(xnavg,ynavg)
      if (ynavg.gt.xnavg) then
         ilong = 2
      endif
      if (znavg.gt.avgnmx.and.if3d) then
         ilong = 3
         avgnmx = znavg
      endif
c
c     Sort in long direction
c
      if (ilong.eq.1) then
         call SORT(xcen,ind,nel)
         call swap2(cen,xcen,ind,nel)
         dmax = dxmax
         h2 = (xmax+xmin)/2.0
         nh = (xmax-xmin)/dmax
         h0 = h2 - nh*dmax - dmax
         cmax = xmax
      elseif (ilong.eq.2) then
         call SORT(ycen,ind,nel)
         call swap2(cen,ycen,ind,nel)
         dmax = dymax
         h2 = (ymax+ymin)/2.0
         nh = (xmax-xmin)/dmax
         h0 = h2 - nh*dmax - dmax
         cmax = xmax
      elseif (ilong.eq.3) then
         call SORT(zcen,ind,nel)
         call swap2(cen,zcen,ind,nel)
         dmax = dzmax
         h2 = (zmax+zmin)/2.0
         nh = (zmax-zmin)/dmax
         h0 = h2 - 0.5*nh*dmax - dmax
         cmax = zmax
      endif
      write(6,6)(xcen(ie),ie=1,nel)
      write(6,6)(cen(ie),ie=1,nel)
    6 format(10f8.2)
      write(6,*) 'h2,nh,h0,dmax,ilong'
      write(6,*)  h2,nh,h0,dmax,ilong
c
c     Pseudo hash table
c
c     find all interactions within current window
c
      call putunit(nel)
      i0 = 1
      ncrnr = 2**ndim
      do 1000 ih=1,2*nh
         if (h0.lt.cmax) then
            h2 = h0+2.0*dmax
            i2 = interval(h2,cen,nel)
c
c           Now scan for interactions amongs elements i0 to i2
c
            write(6,*) 'i02:',i0,i2,h0,h2
            do 300 j=i0,i2
               je = ind(j)
               do 200 i=i0,i2
                  ie = ind(i)
c
c                 ...latex file
c                 if (mod(je,50).eq.0) then
c                    call putx(ie,je,nel,':')
c                 elseif (mod(ie,50).eq.0) then
c                    call putx(ie,je,nel,'.')
c                 endif 
c
                  IF (JE.EQ.IE) THEN
c                    ninrow(ie)=ninrow(ie)+1
c                    jerow(ie,ninrow(ie)) = je
c                    ...latex file
                     csize = 0.33
                     call putch(ie,je,nel,csize)
                     GOTO 200
                  endif
                  DIST=(XCEN(IE)-XCEN(JE))**2+(YCEN(IE)-YCEN(JE))**2
     $                +(ZCEN(IE)-ZCEN(JE))**2
                  DST2=(RCEN(IE)+RCEN(JE))**2
c                 write(6,101) ie,je,dst2,dist,rcen(ie),rcen(je)
  101             format(2x,'dist:',2i3,4f12.7)
                  IF (DST2.GT.DIST) THEN
c
c                    Elements are close, check for vertex interactions
c
                     eps = min(rcen(ie),rcen(je))/1000.0
                     do 190 jc=1,ncrnr
                     do 190 ic=1,ncrnr
                        dij = (x(ie,ic)-x(je,jc))**2 
     $                      + (y(ie,ic)-y(je,jc))**2 
     $                      + (z(ie,ic)-z(je,jc))**2 
                        if (dij.lt.eps) THEN
c                          ninrow(ie)=ninrow(ie)+1
c                          jerow(ie,ninrow(ie)) = je
c                          ...latex file
                           csize = 0.33
                           call putch(ie,je,nel,csize)
                           GOTO 200
c
                        endif
  190                continue
                  endif
  200          continue
  300       continue
c
c           translate window
c
            h0 = h0+dmax
            i0 = interval(h0,cen,nel)
         endif
 1000 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine SWAP2(w,a,IND,N)
C
C     Use IND to sort array A   (p 233 Num. Rec.), 5/26/93 pff.
C
      DIMENSION A(1),W(1),IND(1)
C
      DO 20 J=1,N
         w(j)=a(ind(j))
   20 continue
      RETURN
      END
      FUNCTION INTERVAL(X,XI,N1)
C
C     Return the interval, i, for which x_i <= x < x_i+1
C
      dimension xi(0:n1)
      integer khi,klo,icalld
      save    khi,klo,icalld
      data    khi,klo,icalld /0,0,0/
C
      n=n1-1
      IF (ICALLD.EQ.0) THEN
         ICALLD=1
         KLO=0
         KHI=N
      ELSE
         IF (XI(KLO).LE.X.AND.X.LT.XI(KHI)) THEN
            INTERVAL=KLO+1
            RETURN
         ELSE
            KLO=0
            KHI=N
         ENDIF
      ENDIF
C
    1 IF ((KHI-KLO).GT.1) THEN
         K=(KHI+KLO)/2
         IF (Xi(K).GT.X) THEN
            KHI=K
         ELSE
            KLO=K
         ENDIF
         GOTO 1
      ENDIF
      INTERVAL=KLO+1
C
      H=Xi(KHI)-Xi(KLO)
      IF (H.LE.0) THEN
         WRITE(6,*) Xi(KHI),khi,klo, 'Hey - you blew it.'
         interval = max(interval,1)
         interval = min(interval,n1)
         RETURN
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine putunit(n)
c
      uleng = 5.5/float(n)
      write(29,1) uleng
    1 format('{\setlength{\unitlength}{',f9.5,'in}')
c     write(29,2) n,n
c   2 format('\small \','begin{picture}(',i3,',',i3,')(0.0,0.0)')
      write(29,2)
    2 format('\small ')
      write(29,3) n,n
    3 format('\begin{picture}(',i3,',',i3,')(0.0,0.0)')
      return
      end
c-----------------------------------------------------------------------
      subroutine putx(ii,j,n,x)
c
      character*1 x
      i = n-ii
      write(29,1) j,i,x
    1 format(' \put(',i4,',',i4,') {',a1,'}')
      return
      end
c-----------------------------------------------------------------------
      subroutine putch(ii,j,n,x)
c
      i = n-ii
      write(29,1) j,i,x
    1 format(' \put(',i4,',',i4,') {\circle{',f7.4,'}}')
      return
      end
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      FUNCTION TRAP(PRO,YPT,DS,NPTPR)
      include 'basics.inc'
C     Trapeziodal integration scheme
      REAL PRO(0:500),YPT(0:500)
      ONE = 1.0
      TWOPI = 8.0*ATAN(ONE)
      PINGRL = 0.0
      IF (IFAXIS) THEN
         DO 100 IPT=0,NPTPR
            PINGRL = PINGRL+PRO(IPT)*YPT(IPT)
  100    CONTINUE
         PINGRL=PINGRL-0.5*(PRO(0)*YPT(0)+PRO(NPTPR)*YPT(NPTPR))
         PINGRL=PINGRL*DS*TWOPI
      ELSE
         DO 200 IPT=0,NPTPR
            PINGRL = PINGRL+PRO(IPT)
  200    CONTINUE
         PINGRL=PINGRL-0.5*(PRO(0)+PRO(NPTPR))
         PINGRL=PINGRL*DS
      ENDIF
      TRAP = PINGRL
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine fileout
c
c     Give file output options
c     19 Sep 1998   pff
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer iname(10)
      character*40 tname
      character*40 fname
      equivalence (fname,iname)
c
      integer nppcell_last,nppcell
      save    nppcell_last,nppcell
      data    nppcell_last,nppcell  /-1,0/
c
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c

      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld = icalld + 1
c
c        Set vtk_out format
         nppcell = 3
c
c        Set pixel map default
         xpmn = 0.
         ypmn = 0.
         call getwin(xpmx,ypmx)
         ifclip    = .false.
         ifillt    = .false.
         ifdouble  = .false.
         iffmt_vtk = .true.
      endif
c
      nchoic=1
      item(nchoic)                 = 'MAIN MENU'
      nchoic=nchoic+1
      item(nchoic)                 = 'Set vtk sampling size'
      nchoic=nchoic+1
      item(nchoic)                 = 'VIEW vtk S'
      nchoic=nchoic+1
      item(nchoic)                 = 'DUMP vtk XYZ+S'
      nchoic=nchoic+1
      item(nchoic)                 = 'DUMP vtk ALL'
      nchoic=nchoic+1
      item(nchoic)                 = 'DUMP vtk surface'
      nchoic=nchoic+1
      item(nchoic)                 = 'Animate vtk srf'
      nchoic=nchoic+1
      item(nchoic)                 = 'Animate vtk multifield'
      nchoic=nchoic+1
      item(nchoic)                 = 'Animate vtk vol'
      nchoic=nchoic+1
      item(nchoic)                 = 'XYZ dump'
      nchoic=nchoic+1
      item(nchoic)                 = 'XWD dump'
      nchoic=nchoic+1
      item(nchoic)                 = 'pixel dump'
      nchoic=nchoic+1
      item(nchoic)                 = 'set pixel window size'
c
    1 call menu(xmouse,ymouse,button,'output file')
      if (choice.eq.'MAIN MENU') then
         return
      else if(choice.eq.'Set vtk sampling size') then
         call prsis('Input points per cell edge:[$',nppcell,']$')
         call res(line,70)
         if (line.ne.' ') then
             call reader(value,ierr)
             nppcell=value
         endif
         goto 1
c
      elseif (choice.eq.'Animate vtk vol') then
         call mltiplt4
c
      elseif (choice.eq.'Animate vtk srf') then
         call mltiplt3
c
      elseif (choice.eq.'Animate vtk multifield') then
         call mltiplt_all
c
      elseif (choice.eq.'DUMP vtk surface') then
         call vtk_out_surf('W  ',nppcell,.true.)
         close(unit=34)
         goto 1
c
      elseif (choice.eq.'DUMP vtk ALL') then
         call vtk_out_all(nppcell,.true.)
         close(unit=33)
         goto 1
c
      elseif (choice.eq.'DUMP vtk XYZ+S') then
c
         call vtk_out_xyz_sc(nppcell,.true.)
         nppcell_last = nppcell
         close(unit=33)
c
         call vtk_out_surf('W  ',nppcell,.true.)
         close(unit=34)
         goto 1
c
      elseif (choice.eq.'VIEW vtk S') then
         if (nppcell.ne.nppcell_last) then
            call vtk_out_xyz_sc(nppcell,.false.)
            nppcell_last = nppcell
         endif
         call vtk_out_sc(.false.)
         goto 1
c
      elseif (choice.eq.'XYZ dump') then
         call dump_xyz_fld
c
      elseif (choice.eq.'XWD dump') then
         call grab_window_xwd('xwdump\0')
c
      elseif (choice.eq.'pixel dump')then
         iw = xpmx-xpmn
         ih = ypmx-ypmn
         call izero(iname,10)
         call blank(tname,40)
         write(tname,40) iw,ih
   40    format(i4.4,'x',i4.4)
c  40    format(i4.4,'x',i4.4,'\0')
         call chcopy(fname,tname,9)
         call grab_window_raw(xpmn,ypmn,xpmx,ypmx,fname)
c
      elseif (choice.eq.'set pixel window size') then
c
         call get_box(xmse1,ymse1,xmse2,ymse2)
c
         xscr1 = xscr(xmse1)
         yscr1 = yscr(ymse1)
         xscr2 = xscr(xmse2)
         yscr2 = yscr(ymse2)
c
         xpix1 = scoordx(xscr1)
         ypix1 = scoordy(yscr1)
         xpix2 = scoordx(xscr2)
         ypix2 = scoordy(yscr2)
c
         xpmn = min(xpix1,xpix2)
         xpmx = max(xpix1,xpix2)
         ypmn = min(ypix1,ypix2)
         ypmx = max(ypix1,ypix2)
         write(6,*) 'xp:',xpmn,xpmx,ypmn,ypmx
c
         i=xpmn
         xpmn=i
         j=xpmx
         xpmx=j
         k=ypmn
         ypmn=k
         l=ypmx
         ypmx=l
         write(6,*) 'xp:',xpmn,xpmx,ypmn,ypmx

         goto 1
c
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine transpose(a,lda,b,ldb)
      real a(lda,1),b(ldb,1)
c
      do j=1,ldb
         do i=1,lda
            a(i,j) = b(j,i)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine grab_window_xwd0
      return
      end
c-----------------------------------------------------------------------
      subroutine grab_window_raw0
      return
      end
c-----------------------------------------------------------------------
      subroutine get_box(xmse1,ymse1,xmse2,ymse2)
c
c     This routine allows the user to specify a rectangular box
c
c     xmsei,ymsei  are returned in screen coordinates  (0,1)
c
      include 'basics.inc'
      LOGICAL IFTMP
C
      IFTMP =IFGRID
      IFGRID=.FALSE.
      CALL PRS('Input with mouse 2 points on edge of new frame.$')
      CALL PRS('Menu area for keybd input. RIGHT click for auto.$')
c     CALL PRS('Menu area to restore to original size.$')
C
      CALL MOUSE(XMSE1,YMSE1,BUTTON)
      xscrm = xscr(xmse1)
      yscrm = yscr(ymse1)
      ipix  = scoordx(xscrm)
      jpix  = scoordy(yscrm)
c     call prsrr('Physcl coordinates:$',xmse1,ymse1)
c     call prsrr('Screen coordinates:$',xscrm,yscrm)
c     call prsii('Pixel  coordinates:$',ipix,jpix)
   10 CONTINUE
      if (button.eq.'RIGHT') then
c
c        1st click was right, get near screen edges
c
         xscrm1 = 0.02
         yscrm1 = 0.02
         xmse1  = xphy(xscrm1)
         ymse1  = yphy(yscrm1)
         xscrm2 = 0.98
         yscrm2 = 0.98
         xmse2  = xphy(xscrm2)
         ymse2  = yphy(yscrm2)
c
      elseif (button.eq.'RIGHT'.and.nel.lt.0) then
c
c        1st click was right, get "convex hull" (pff 04/12/00)
c
         call quickscan_xy(x1,y1,x2,y2)
c  
c        Got two (x,y) pairs
c
         xmse1 = x1
         ymse1 = y1
         xmse2 = x2
         ymse2 = y2
c
c        Re-order xmsei and ymsei
c
         xmn = min(xmse1,xmse2)
         ymn = min(ymse1,ymse2)
         xmx = max(xmse1,xmse2)
         ymx = max(ymse1,ymse2)
c
         xmse1 = xmn
         ymse1 = ymn
         xmse2 = xmx
         ymse2 = ymx
c
c        draw a box around frame area
c
         call movec(xmn,ymn)
         call drawc(xmx,ymn)
         call drawc(xmx,ymx)
         call drawc(xmn,ymx)
         call drawc(xmn,ymn)
c
      else
c
c        Entering coordinates, either w/ mouse or keyboard
c
         if (xscrm.gt.1) then
c           Keyboard
            CALL PRS('Type in x-y pixel coordinates:$')
            CALL REii(ipix,jpix)
            IF(IFLEARN)WRITE(3,*) ipix,jpix
            pixi  = ipix
            pixj  = jpix
            xscrm = scoordx_inv(pixi)
            yscrm = scoordy_inv(pixj)
            xmse1  = xphy(xscrm)
            ymse1  = yphy(yscrm)
         endif
c
c
         CALL COLOR(1)
         CALL MOVESC(XSCRM-.015,YSCRM)
         CALL DRAWSC(XSCRM+.015,YSCRM)
         CALL MOVESC(XSCRM,YSCRM-.015)
         CALL DRAWSC(XSCRM,YSCRM+.015)
c
         call prsrr('Physcl coordinates:$',xmse1,ymse1)
         call prsrr('Screen coordinates:$',xscrm,yscrm)
         call prsii('Pixel  coordinates:$',ipix,jpix)
c
c
c        Get second location
c
         CALL MOUSE(XMSE2,YMSE2,BUTTON)
         xscrm2 = xscr(xmse2)
         yscrm2 = yscr(ymse2)
         ipix   = scoordx(xscrm2)
         jpix   = scoordy(yscrm2)
c        call prsrr('Physcl coordinates:$',xmse2,ymse2)
c        call prsrr('Screen coordinates:$',xscrm2,yscrm2)
c        call prsii('Pixel  coordinates:$',ipix,jpix)
         if (xscrm2.gt.1) then
            CALL PRS('Type in x, y pixel coordinates:$')
            CALL REii(ipix,jpix)
            IF(IFLEARN)WRITE(3,*) ipix,jpix
            pixi = ipix
            pixj = jpix
            xscrm2 = scoordx_inv(pixi)
            yscrm2 = scoordy_inv(pixj)
            xmse2  = xphy(xscrm2)
            ymse2  = yphy(yscrm2)
         endif
         call prsrr('Physcl coordinates:$',xmse2,ymse2)
         call prsrr('Screen coordinates:$',xscrm2,yscrm2)
         call prsii('Pixel  coordinates:$',ipix,jpix)
c
c
         CALL COLOR(1)
         CALL MOVESC(XSCRM2-.015,YSCRM2)
         CALL DRAWSC(XSCRM2+.015,YSCRM2)
         CALL MOVESC(XSCRM2,YSCRM2-.015)
         CALL DRAWSC(XSCRM2,YSCRM2+.015)
c
c        Re-order xmsei and ymsei
c
         xmn = min(xmse1,xmse2)
         ymn = min(ymse1,ymse2)
         xmx = max(xmse1,xmse2)
         ymx = max(ymse1,ymse2)
c
         xmse1 = xmn
         ymse1 = ymn
         xmse2 = xmx
         ymse2 = ymx
c
c        draw a box around frame area
c
         call movec(xmn,ymn)
         call drawc(xmx,ymn)
         call drawc(xmx,ymx)
         call drawc(xmn,ymx)
         call drawc(xmn,ymn)
c
         CALL PRS('Click in menu area if OK, or re-enter points.$')
         CALL MOUSE(XMSE3,YMSE3,BUTTON)
         IF (XSCR(XMSE3).GE.1.0) THEN
c
c           OK.   
c
            xscr1 = xscr(xmse1)
            yscr1 = yscr(ymse1)
c
            xscr2 = xscr(xmse2)
            yscr2 = yscr(ymse2)
c
         else
c
c           Erase old "+"
c
            CALL COLOR(0)
            CALL MOVESC(XSCR(XMSE1)-.015,YSCR(YMSE1))
            CALL DRAWSC(XSCR(XMSE1)+.015,YSCR(YMSE1))
            CALL MOVESC(XSCR(XMSE1),YSCR(YMSE1)-.015)
            CALL DRAWSC(XSCR(XMSE1),YSCR(YMSE1)+.015)
            CALL MOVESC(XSCR(XMSE2)-.015,YSCR(YMSE2))
            CALL DRAWSC(XSCR(XMSE2)+.015,YSCR(YMSE2))
            CALL MOVESC(XSCR(XMSE2),YSCR(YMSE2)-.015)
            CALL DRAWSC(XSCR(XMSE2),YSCR(YMSE2)+.015)
c
c           Un-draw a box around frame area
c
            call movec(xmn,ymn)
            call drawc(xmx,ymn)
            call drawc(xmx,ymx)
            call drawc(xmn,ymx)
            call drawc(xmn,ymn)
c
            XMSE 1= XMSE3
            YMSE1 = YMSE3
            xscrm = xscr(xmse1)
            yscrm = yscr(ymse1)
            ipix  = scoordx(xscrm)
            jpix  = scoordy(yscrm)
            CALL PRS('Click 2nd point on edge of new frame.$')
            GOTO 10
         ENDIF
      ENDIF
c
      write(line,70) xmse1,ymse1,xmse2,ymse2
   70 format
     $  ('Got (x1,y1)=(',f8.4,',',f8.4,') (x2,y2)=(',f8.4,',',f8.4,')$')
      call prs(line)
c
      IFGRID=iftmp
      return
      end
c-----------------------------------------------------------------------
      subroutine pix_map_out
c
c     Give file output options
c     19 Sep 1998   pff
c
      include 'basics.inc'
c
      integer      iname(10)
      character*40 fname
      equivalence (fname,iname)
c
      iw = xpmx-xpmn
      ih = ypmx-ypmn
c
c
      call blank(line,70)
      write(line,41) iw,ih
   41 format('writing to file ',i4.4,'x',i4.4,'.rgb$')
      call prs(line)
c
      call izero(iname,10)
      write(fname,40) iw,ih
   40 format(i4.4,'x',i4.4)
      call grab_window_raw(xpmn,ypmn,xpmx,ypmx,fname)
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_xyz_sc(mx,ifout)
c
c     Generate an FEM input file based on Nekton .rea file
c
C
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifout
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwki/ loc(lvtk),ind(lvtk)
      common /vtkwka/ w1 (lvtk),wk (lvtk)
      common /vtkwsi/ mx_save,nvtk,nelclip,ncells
      common /vtkwkr/ xcmn,xcmx
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
c
      common /bigvtki/ jglob(lvtk),kcell(8,lvtk)
      common /bigvtkr/ vtkxyz(lvtk,3)
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c
      real xm(1),ym(1),zm(1)
      equivalence (xm,vtkxyz(1,1))
      equivalence (ym,vtkxyz(1,2))
      equivalence (zm,vtkxyz(1,3))
c
      integer ic(8)
c
      logical ifnew
      save    ifnew
c
c
      ifnew = .false.
      if (mx.ne.mx_save) ifnew = .true.
      mx_save = mx
      my = mx
      mz = mx
      if (.not.if3d) mz=1
c
      if (ifnew) then
c
c        Fomatted output?
c
         iffmt_vtk = .true.
         ifdouble  = .false.
         if (ifout) then
c        Query format type
            ibin=1
c           call prs('ASCII (1) or BINARY (0) vtk file? Enter 1 or 0:$')
c           call rei(ibin)
c           if (ibin.eq.0) iffmt_vtk = .false.
c        Query symmetry option
c           call prs('Symmetry?  (1=yes)$')
c           call rei(isym)
            isym=0
            if (isym.eq.1) ifdouble = .true.
         endif
c
C        Clip vtk output?
c
         ifclip = .true.
         call prs('Clip volume in x? (enter xmn,xmx or  0,0 for no)$')
         call rerr(xcmn,xcmx)
         if (xcmn.eq.0.  .and. xcmx.eq.0.) ifclip = .false.
         nx3     = nx*ny*nz
         nelclip = nel
         call find_xin(nelclip,ifclip,ifxin,xp,nx3,nel,xcmn,xcmx)
      endif
      nton=nx*ny*nz*nel
      ntot=mx*my*mz*nelclip
      nvtk=ntot
c
      write(6,*) 'This is iffmt_vtk:',iffmt_vtk
      if (ifclip) call prsrr('Clipping in effect!$',xcmn,xcmx)
      call prsii('nelclip $',nelclip,nel)
      call prsii('ntot    $',nvtk   ,nton)
c
c
c
      call prsii('call mapreg3d mx,nx$',mx,nx)
      call mapreg3d(xm,mx,lvtk,xp,nx,nel,ifclip,ifxin,wk)
      call mapreg3d(ym,mx,lvtk,yp,nx,nel,ifclip,ifxin,wk)
      if (if3d) 
     $call mapreg3d(zm,mx,lvtk,zp,nx,nel,ifclip,ifxin,wk)
c
c     Establish local to global node numbering for mx points
c
      call locglobm(jglob,nglb,mx,nel)
      write(6,*) 'this is nglb',nglb,ntot,mx,nel
c
c     Now, use jglob to sort data
c
      call icopy (loc ,jglob   ,ntot)
      call isort (loc ,ind     ,ntot) ! isort actually sorts now
c     call iswap (loc ,w1  ,ind,ntot)
c
c     Compress and output coordinate data
c
      call prsii('call mapreg3d mx,nx$',mx,nx)
      call mapreg3d(xm,mx,lvtk,xp,nx,nel,ifclip,ifxin,wk)
      call mapreg3d(ym,mx,lvtk,yp,nx,nel,ifclip,ifxin,wk)
      if (if3d) 
     $call mapreg3d(zm,mx,lvtk,zp,nx,nel,ifclip,ifxin,wk)
c
      call compress (ngpt,xm,loc,ind,ntot,w1)
      call compress (ngpt,ym,loc,ind,ntot,w1)
      call compress (ngpt,zm,loc,ind,ntot,w1)
c
      call vtk_out_head('vtkfile',7,ifout,33)
      if (ifnew) call vtk_out_xyz (xm,ym,zm,ngpt,ifout)
c
c     Output cell data
c
      if (ifnew) call vtk_out_hex (jglob,mx,my,mz,nelclip,ifout)
      if (ifnew) call coef2
c
c
c
c     Now, use loc,ind to sort data
c
c     Compress and output scalar work data
c
      m_scalar = 2
      do i=1,m_scalar

c        quick hack:  force quantity

         if (i.eq.1) QUANTY = 'VORTEX'
         if (i.eq.2) QUANTY = 'PRESSURE'
         call setwrk(.false.)

         call mapreg3d (wk,mx,lvtk,work,nx,nel,ifclip,ifxin,w1)
         call compress         (ngpt,wk,loc,ind,ntot,w1)

         call vtk_out_sca      (wk,ngpt,ifout,i)

      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compress (ngpt,x,loc,ind,n,wk)
      real    x(n),wk(n)
      integer loc(n),ind(n)
c
c     Compress array x
c
c
      call swap (x,wk,ind,n)
c
      lmax = 0
      ngpt = 0
      if (loc(1).ne.0) ngpt=1
      do i=2,n
         if (loc(i).ne.loc(i-1) .and. loc(i).ne.0 ) then
            ngpt = ngpt+1
            x(ngpt) = x(i)
         endif
         lmax = max(lmax,loc(i))
      enddo
      write(6,*) 'inside compress ',n,ngpt,lmax
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_hex(icell,mx,my,mz,nel,ifout)
c
      integer icell(mx,my,mz,nel)
      integer jcell(8)
      logical ifout
c
      parameter (lvtk=2000000)  !  8530000)
      common /bigvtki/ jglob(lvtk),kcell(8,lvtk)
      common /vtkwsi/ mx_save,nvtk,nelclip,ncells
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
      if (ifout) then
c
c        count number of valid cells (with non-zero vertex pointers)
c        and load into kcell
c     
         ncells = 0
         do ie=1,nel
         do k=1,mz-1
         do j=1,my-1
         do i=1,mx-1
            l = 0
            jcell_min = 1
            do k0 = 0,1
            do j0 = 0,1
            do i0 = 0,1
               kk = k+k0
               jj = j+j0
               ii = i+i0
               l  = l+1
               jcell(l)  = icell(ii,jj,kk,ie)
               jcell_min = min(icell(ii,jj,kk,ie),jcell_min)
c              write(6,111) jcell_min,ie,ii,jj,kk,jcell(l),l
111            format(7i6,' JCELL')
            enddo
            enddo
            enddo
            if (jcell_min.gt.0) then
c              Save cell data for checking in postnek
               if (ncells.lt.lvtk) then
                  ncells = ncells+1
c
c                 VTK nodes are numbered from 0 to n-1!  (pff 9/29/98)
c                 vtk takes the EB format!  (see p. 605 of VTK 2nd Ed.)
c
                  kcell(1,ncells) = jcell(1)-1
                  kcell(2,ncells) = jcell(2)-1
                  kcell(3,ncells) = jcell(4)-1
                  kcell(4,ncells) = jcell(3)-1
                  kcell(5,ncells) = jcell(5)-1
                  kcell(6,ncells) = jcell(6)-1
                  kcell(7,ncells) = jcell(8)-1
                  kcell(8,ncells) = jcell(7)-1
               else
                  call prsii('kcell too small (trap.f)$',ncells,lvtk)
               endif
            endif
         enddo
         enddo
         enddo
         enddo
c
         nptswr = 9*ncells
c
         if (ifdouble) then
            n2cells = 2*ncells
            n2ptswr = 2*nptswr
            write(6,1)  n2cells,n2ptswr
            write(33,1) n2cells,n2ptswr
         else
            write(6,1)  ncells,nptswr
            write(33,1) ncells,nptswr
         endif
         i8 = 8
         if (iffmt_vtk) then
            do k=1,ncells
               write(33,2) i8,(kcell(l,k),l=1,8)
            enddo
         else
            write(33) (i8,(kcell(l,k),l=1,8),k=1,ncells)
         endif
         if (ifdouble) then
            do k=1,ncells
               jcell(1) = kcell(1,k) + ncells
               jcell(3) = kcell(2,k) + ncells
               jcell(2) = kcell(3,k) + ncells
               jcell(4) = kcell(4,k) + ncells
               jcell(5) = kcell(5,k) + ncells
               jcell(7) = kcell(6,k) + ncells
               jcell(6) = kcell(7,k) + ncells
               jcell(8) = kcell(8,k) + ncells
               write(33,2) i8,(jcell(l),l=1,8)
            enddo
         endif
      endif
c
c     Reload kcell into useful form for postnek contouring
c
      ncells = 0
      do ie=1,nel
      do k=1,mz-1
      do j=1,my-1
      do i=1,mx-1
         l = 0
         jcell_min = 1
         do k0 = 0,1
         do j0 = 0,1
         do i0 = 0,1
            kk = k+k0
            jj = j+j0
            ii = i+i0
            l  = l+1
            jcell(l)  = icell(ii,jj,kk,ie)
            jcell_min = min(jcell(l),jcell_min)
         enddo
         enddo
         enddo
         if (jcell_min.gt.0) then
c           Save cell data for contouring in postnek
            if (ncells.lt.lvtk) then
               ncells = ncells+1
               do kj=1,8
                  kcell(kj,ncells) = jcell(kj)
               enddo
            endif
         endif
      enddo
      enddo
      enddo
      enddo
c
    1 format('CELLS ',2i9)
    2 format(i1,8i8)
c
      i12 = 12
      write(6 ,11) ncells
      if (ifout) then
c
         if (ifdouble) then
            write(33,11) n2cells
            nacells = n2cells
         else
            write(33,11) ncells
            nacells = ncells
         endif
c
         if (iffmt_vtk) then
            do i=1,nacells
               write(33,12) i12
            enddo
         else
            write(33) (i12,k=1,nacells)
         endif
      endif
c
   11 format('CELL_TYPES',i9)
   12 format(i2)
c
      if (ncells.gt.lvtk) then
         call prsii('Not saving vtk cell data.$',ncells,lvtk)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_sca (ww,n,ifout,ivtk_dump)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifout
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
      real ww(1)
c
c
      if (ivtk_dump.eq.1) then
c        We need to specify the number of points
         if (ifout) nold = n
         n2 = n
         if (ifdouble) n2 = 2*n
         if (ifout) write(33,1) n2
         write(6 ,1) n2
    1    format('POINT_DATA ',i9)
      endif
c
      if (ifout) write(33,2) quanty
      write(6 ,2) quanty
    2 format('SCALARS ',a20,' float')
c
      if (ifout) write(33,3)
      write(6 ,3)
    3 format('LOOKUP_TABLE default')
c
      wwmin = glmin(ww,n)
      wwmax = glmax(ww,n)
      call blank(line,70)
      write(line,7) wwmin,wwmax,quanty
    7 format('minmax:',1p2e11.3,2x,a20,'$')
      call prs(line)
c
      if (ifout) then
         do i=1,n
            write(33,10) ww(i)
         enddo
         if (ifdouble) then
            do i=1,n
               write(33,10) ww(i)
            enddo
         endif
      endif
   10 format(1pe13.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine fd_weights_full(xx,x,n,m,c)
c
c     This routine evaluates the derivative based on all points
c     in the stencils.  It is more memory efficient than "fd_weights"
c
c     This set of routines comes from the appendix of 
c     A Practical Guide to Pseudospectral Methods, B. Fornberg
c     Cambridge Univ. Press, 1996.   (pff)
c
c     Input parameters:
c       xx -- point at wich the approximations are to be accurate
c       x  -- array of x-ordinates:   x(0:n)
c       n  -- polynomial degree of interpolant (# of points := n+1)
c       m  -- highest order of derivative to be approxxmated at xi
c
c     Output:
c       c  -- set of coefficients c(0:n,0:m).
c             c(j,k) is to be applied at x(j) when
c             the kth derivative is approxxmated by a 
c             stencil extending over x(0),x(1),...x(n).
c
c
      real x(0:n),c(0:n,0:m)
c
      c1       = 1.
      c4       = x(0) - xx
c
      do k=0,m
      do j=0,n
         c(j,k) = 0.
      enddo
      enddo
      c(0,0) = 1.
c
      do i=1,n
         mn = min(i,m)
         c2 = 1.
         c5 = c4
         c4 = x(i)-xx
         do j=0,i-1
            c3 = x(i)-x(j)
            c2 = c2*c3
            do k=mn,1,-1
               c(i,k) = c1*(k*c(i-1,k-1)-c5*c(i-1,k))/c2
            enddo
            c(i,0) = -c1*c5*c(i-1,0)/c2
            do k=mn,1,-1
               c(j,k) = (c4*c(j,k)-k*c(j,k-1))/c3
            enddo
            c(j,0) = c4*c(j,0)/c3
         enddo
         c1 = c2
      enddo
      s0 = 0.
      do i=0,n
         s0 = s0+c(i,0)
      enddo
c     write(6,2) 'fd wt sum:',s0
    1 format('c',10f8.4)
    2 format(a10,10f8.4)
      return
      end
c-----------------------------------------------------------------------
      subroutine map_tnsr(u2,n2,u1,n1,i,it,w1,w2,if3d)
c
c     tensor-product operator  u1 to u2
c
      real u2(n2,n2,n2) , u1(n1,n1,n1)
      real i(n2,n1),it(n1,n2)
      real w1 (n2,n1,n1)
      real w2 (n2,n2,n1)
c
      logical if3d
c
      if (if3d) then
         call mxm(i,n2,u1,n1,w1,n1*n1)
         do k=1,n1
            call mxm(w1(1,1,k),n2,it,n1,w2(1,1,k),n2)
         enddo
         call mxm(w2,n2*n2,it,n1,u2,n2)
      else
         call mxm(i ,n2,u1,n1,w1,n1)
         call mxm(w1,n2,it,n1,u2,n2)
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine mapreg3d(w_map,mx,ldmap,w_in,nx,ne,ifclip,ifxin,w)
c
      real w_map(mx*mx*mx,ldmap)
      real w_in (nx*nx*nx,ne)
      logical ifclip,ifxin(1)
c
      real w(0:1)
c
c
      parameter (nxm=30)
      real gll_pts(nxm)
      real gll_wts(nxm)
      real i (3*nxm*nxm)
      real it(3*nxm*nxm)
c
c     call copy(w_map,w_in,nx*nx*nx*ne)
c     return
c
      if (nx.gt.nxm .or. nx*mx.gt. 3*nxm*nxm) then
         write(6,*) 'increase array dimensions in mapreg3d',mx,nx,nxm
         call copy(w_map,w_in,nx*nx*nx*ne)
         return
      endif
c
      call legend(gll_pts,gll_wts,nx)
c
c     Use Fornberg's routine to compute interpolation coefficients
c
      nx1 = nx-1
      l  = 1
      do k=1,mx
         z0 = (2.*(k-1))/(mx-1) - 1.
         call fd_weights_full(z0,gll_pts,nx1,0,it(l))
         l  = l+nx
      enddo
c
      call transpose(i,mx,it,nx)
c
      nmx  = max(nx,mx)
      l    = nmx*nmx*nmx
      nc   = 0
      nmap = mx*mx*mx
      do ie = 1,ne
         if (ifxin(ie).or.(.not.ifclip)) then
            nc = nc+1
            call map_tnsr
     $      (w_map(1,nc),mx,w_in(1,ie),nx,i,it,w,w(l),.true.)
            nmap = nmap+mx*mx*mx
            if (nmap.gt.ldmap) then
               call prsii
     $             ('Not enough space to map all the data$',nmap,ldmap)
               call prsii('in mapreg3d. Number of cells,els$',nc,ne)
               return
            endif
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_head(name,lname,ifout,io)
      character*1 name(lname)
      logical ifout
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c
      character*1  stru1(40,3)
      character*40 stru(3)
      equivalence (stru,stru1)
      save         stru
      data         stru /'# vtk DataFile Version 2.0              '
     $                  ,'BINARY                                  '
     $                  ,'DATASET UNSTRUCTURED_GRID               '/
c
      character*1  strf1(40,3)
      character*40 strf(3)
      equivalence (strf,strf1)
      save         strf
      data         strf /'# vtk DataFile Version 2.0              '
     $                  ,'ASCII                                   '
     $                  ,'DATASET UNSTRUCTURED_GRID               '/
c
c
      integer icalld
      save    icalld
      data    icalld /0/
      character*9 fname
c
c
      if (ifout) then
c
         icalld = icalld+1
c
c        write(fname,4) icalld
c   4    format('vtk.',i3.3)
c
         if (io.eq.33) write(fname,5) 
         if (io.eq.34) write(fname,6) 
    5    format('vt1.vtk')
    6    format('obj.vtk')
c
         if (iffmt_vtk) then
            open (unit=io,file=fname,form='formatted')
         else
            open (unit=io,file=fname,form='unformatted')
         endif
      endif
c
      i = 0
      do ii=1,4
         if (ii.ne.2) then
            i = i+1
            if (ifout) then
               len = ltrunc(strf(i),40)
               if (iffmt_vtk) then
                  write(io,1) (strf1(k,i),k=1,len)
                else
                  write(io) (stru1(k,i),k=1,len)
                endif
            endif
         else
            if (ifout) then
               len = ltrunc(name,lname)
               if (iffmt_vtk) then
                  write(io,1) (name(k),k=1,len)
                else
                  write(io) (name(k),k=1,len)
                endif
             endif
             write(6 ,1) (name(k),k=1,lname)
         endif
      enddo
    1 format(40a1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_xyz(x,y,z,n,ifout)
      real x(1),y(1),z(1)
c
      logical ifout
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
      character*80 string
      character*1  string1(80)
      equivalence (string,string1)
c
      n2 = n
      if (ifdouble) n2 = 2*n
      write(6 ,1) n2
c
      if (ifout) then
         if (iffmt_vtk) then
            write(33,1) n2
            do i=1,n
               write(33,2) x(i),y(i),z(i)
            enddo
            if (ifdouble) then
               do i=1,n
                  ym = -y(i)
                  write(33,2) x(i),ym,z(i)
               enddo
            endif
         else
            call blank(string,80)
            write(string,1) n
            len=ltrunc(string,80)
            write(33) (string1(k),k=1,n)
            write(33) (x(i),y(i),z(i),i=1,n)
         endif
      endif
c
    1 format('POINTS ',i9,' float')
    2 format(1p3e13.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_sc(ifout)
c
c     Output scalar data
c
C
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifout
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwki/ loc(lvtk),ind(lvtk)
      common /vtkwka/ w1 (lvtk),wk (lvtk)
      common /vtkwsi/ mx_save,nvtk,nelclip
      common /vtkwkr/ xcmn,xcmx
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
c
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c
c     Compress and output scalar work data
c
      mx = mx_save
      my = mx
      mz = mx
      if (.not.if3d) mz=1
      ntot = nvtk
c
      call mapreg3d    (wk,mx,lvtk,work,nx,nel,ifclip,ifxin,w1)
      call compress    (ngpt,wk,loc,ind,ntot,w1)
c     call vtk_out_sca (wk,ngpt,ifout,1)
      vtkmin = glmin   (wk,ngpt)
      vtkmax = glmax   (wk,ngpt)
c
c
c     Query for verification contour
c
      call prsrr('Input scalar contour level$',vtkmin,vtkmax)
      call prs  ('(0=return)$')
      call rer  (vtkcut)
      if (vtkcut.eq.0.) return
c
c     Get fill option
c
      call prs('Input fill (1) or no fill (0)$')
      call rei  (ifill)
      ifillt = .true.
      if (ifill.ne.1) ifillt = .false.
c
c     Contour scalar surface to check vtk output
c
      call vtk_srf_cnt(vtkcut,wk,ngpt)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_srf_cnt(vtkcut,vtkw,ngpt)
c
c     Contour scalar data
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwsi/  mx_save,nvtk,nelclip,ncells
      common /bigvtki/ jglob(lvtk),kcell(8,lvtk)
      common /bigvtkr/ vtkxyz(lvtk,3)
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c     Count number of polygons
c
      common /dpoly/ npoly
c
      real   vtkw(1)
c
      integer itet(4,5)
      save    itet
      data    itet   / 1,2,3,5
     $               , 2,3,4,8
     $               , 2,5,6,8
     $               , 3,5,7,8
     $               , 2,3,5,8 /
c
      real    w(8),v(3,8),vs(3,4),ws(4)
      integer l(8)
C
      IF (ifillt) THEN
        ICLR=15
        CALL FILLP(-ICLR)
        CALL COLOR(ICLR)
      ENDIF
c
      npoly = 0
c
      do ic=1,ncells
         l(1) = kcell(1,ic)
         w(1) = vtkw(l(1))
         vmin = w(1)
         vmax = w(1)
c        xavg = vtkxyz(l(1),1)
c        yavg = vtkxyz(l(1),2)
c        zavg = vtkxyz(l(1),3)
         do iv=2,8
            l(iv) = kcell(iv,ic)
            w(iv) = vtkw(l(iv))
            vmin  = min(vmin,w(iv))
            vmax  = max(vmax,w(iv))
c           xavg = xavg+vtkxyz(l(iv),1)
c           yavg = yavg+vtkxyz(l(iv),2)
c           zavg = zavg+vtkxyz(l(iv),3)
         enddo
c        xavg = .125*xavg
c        yavg = .125*yavg
c        zavg = .125*zavg
c
         if (vmin.lt.vtkcut.and.vtkcut.lt.vmax) then
c           we have potential countour
c           write(6 ,1) xavg,yavg,zavg,vmin,vtkcut,vmax,ic,ncells
c           write(67,1) xavg,yavg,zavg,vmin,vtkcut,vmax,ic,ncells
    1       format('ct',1p6e12.4,2i9)
            do iv=1,8
               l(iv) = kcell(iv,ic)
               v(1,iv) = vtkxyz(l(iv),1)
               v(2,iv) = vtkxyz(l(iv),2)
               v(3,iv) = vtkxyz(l(iv),3)
            enddo
c
            do ktet = 1,5
               do kv=1,4
                  vs(1,kv) = v(1,itet(kv,ktet))
                  vs(2,kv) = v(2,itet(kv,ktet))
                  vs(3,kv) = v(3,itet(kv,ktet))
                  ws(  kv) = w(  itet(kv,ktet))
               enddo
               call countour_tet(vtkcut,ws,vs,ifillt)
            enddo
         endif
      enddo
c
      call prsii('Number of polygons, cells:$',npoly,ncells)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine countour_tet(w,ws,vs,iffill)
      real w,ws(4),vs(3,4)
      logical iffill
c
      real xyz(3,4)
c
c     Count number of vertices less than w 
c
      nlt = 0
c
      imin = 1
      imax = 1
      do i=1,4
         if (ws(i).lt.w)        nlt = nlt+1
         if (ws(i).lt.ws(imin)) imin = i
         if (ws(i).gt.ws(imax)) imax = i
      enddo
c
      if (nlt.eq.0 .or. nlt.eq.4) return
c
c
      l = 0
      if (nlt.eq.1) then
         do i=1,4
            if (i.ne.imin) then
               l = l+1
               beta = (w-ws(imin))/(ws(i)-ws(imin))
               do j=1,3
                  xyz(j,l) = vs(j,imin) + beta*(vs(j,i)-vs(j,imin))
               enddo
            endif
         enddo
         call draw_poly_fill(xyz,3,ifill)
c
      elseif (nlt.eq.3) then
         do i=1,4
            if (i.ne.imax) then
               l = l+1
               beta = (w-ws(imax))/(ws(i)-ws(imax))
               do j=1,3
                  xyz(j,l) = vs(j,imax) + beta*(vs(j,i)-vs(j,imax))
               enddo
            endif
         enddo
         call draw_poly_fill(xyz,3,ifill)
      else
c        Quadrilateral -- we won't bother to avoid hour-glassing for this hack...
         do i=1,4
c           find less than brances
            if (ws(i).lt.w .and. i.ne.imin) i2 = i
         enddo
c
         do i=1,4
            if (i.ne.i2.and.i.ne.imin) then
c
               l = l+1
               beta = (w-ws(imin))/(ws(i)-ws(imin))
               do j=1,3
                  xyz(j,l) = vs(j,imin) + beta*(vs(j,i)-vs(j,imin))
               enddo
c
               l = l+1
               beta = (w-ws(i2))/(ws(i)-ws(i2))
               do j=1,3
                  xyz(j,l) = vs(j,i2) + beta*(vs(j,i)-vs(j,i2))
               enddo
            endif
         enddo
         call draw_poly_fill(xyz,4,iffill)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine draw_poly_fill(xyz,n,iffill)
      real xyz(3,n)
c
      common /dpoly/ npoly
c
      logical iffill
c
      npoly = npoly+1
c
      if (iffill) call beginpp
      call movec(xiso(xyz(1,n),xyz(2,n),xyz(3,n))
     $          ,yiso(xyz(1,n),xyz(2,n),xyz(3,n)))
c
      do i=1,n
         call drawc(xiso(xyz(1,i),xyz(2,i),xyz(3,i))
     $             ,yiso(xyz(1,i),xyz(2,i),xyz(3,i)))
      enddo
      if (iffill) call endp
c
      return
      end
c-----------------------------------------------------------------------
      subroutine find_xin(nc,ifclip,ifxin,xp,nx3,nel,xcmn,xcmx)
      logical ifclip,ifxin(1)
      real xp(nx3,nel)
c
      xmina =  1.e20
      xmaxa = -1.e20
      do ie=1,nel
         ifxin(ie) = .true.
      enddo
      nc = nel
      if (ifclip) then
         nc = 0
         do ie=1,nel
            xmin = glmin(xp(1,ie),nx3)
            xmax = glmax(xp(1,ie),nx3)
            if (xcmn.le.xmax.and.xmin.le.xcmx) then
               ifxin(ie) = .true.
               xmina = min(xmin,xmina)
               xmaxa = max(xmax,xmaxa)
            else
               ifxin(ie) = .false.
            endif
c           ifxin(ie) = .true.
            if (ifxin(ie)) nc = nc+1
         enddo
      endif
      write(6,*) 'XMinMax:',xmina,xmaxa,nc,nel
c
      return
      end
c-----------------------------------------------------------------------
      subroutine settol3n(tol,nx,nel,x,y,z)
c
      real tol(nx,nx,nx,nel)
      real x  (nx,nx,nx,nel)
      real y  (nx,nx,nx,nel)
      real z  (nx,nx,nx,nel)
c
      tolmin = 1.e20
      tolmax = 0.
c     call prsi('input scale for tol (e.g. .01)$',nx)
c     call rer (scale)
      scale = .002
c     scale = .050
c
      do ie=1,nel
      do k=1,nx-1
      do j=1,nx-1
      do i=1,nx-1
         l = 0
         do k0 = 0,1
         do j0 = 0,1
         do i0 = 0,1
            kk = k+k0
            jj = j+j0
            ii = i+i0
            l  = l+1
            if (l.eq.2) then
               d2 = (x(i,j,k,ie)-x(ii,jj,kk,ie))**2
     $            + (y(i,j,k,ie)-y(ii,jj,kk,ie))**2
     $            + (z(i,j,k,ie)-z(ii,jj,kk,ie))**2
               d2min = d2
            elseif (l.gt.2) then
               d2 = (x(i,j,k,ie)-x(ii,jj,kk,ie))**2
     $            + (y(i,j,k,ie)-y(ii,jj,kk,ie))**2
     $            + (z(i,j,k,ie)-z(ii,jj,kk,ie))**2
               d2min = min(d2,d2min)
            endif
         enddo
         enddo
         enddo
         if (d2min.gt.0) then
            d2min = scale*sqrt(d2min)
         else
            call prsr('Hey!! d2min < 0 ??? in settl3$',d2min)
            write(6,*) 'ie',ie,i,j,k,ii,jj,kk
            write(6,*) 'xy',x(i,j,k,ie),y(i,j,k,ie),z(i,j,k,ie)
            call prs ('Input any key to continue$')
            call res (line,1)
         endif
c        Now map to all 8 entries on this cell (most are later overwritten)
         do k0 = 0,1
         do j0 = 0,1
         do i0 = 0,1
            kk = k+k0
            jj = j+j0
            ii = i+i0
            tol (ii,jj,kk,ie) = d2min
         enddo
         enddo
         enddo
         tolmin = min(tolmin,d2min)
         tolmax = max(tolmax,d2min)
      enddo
      enddo
      enddo
      enddo
c
      write(6,*) ' tolerances',tolmin,tolmax
      return
      end
c-----------------------------------------------------------------------
      subroutine settol3v(tol,ldt,nx,nel,x,y,z)
c
      real tol(ldt,3)
      real x  (ldt)
      real y  (ldt)
      real z  (ldt)
c
      tolmin = 1.e20
      tolmax = 0.
c
c
c     Quick and stupid approach... set to close to machine eps
c
      do i=1,ldt
         tol(i,1) = 5.e-5 * abs ( x(i) ) + 3.e-6
         tol(i,2) = 5.e-5 * abs ( y(i) ) + 3.e-6
         tol(i,3) = 5.e-5 * abs ( z(i) ) + 3.e-6
c
         tolmin = min(tolmin,tol(i,1))
         tolmin = min(tolmin,tol(i,2))
         tolmin = min(tolmin,tol(i,3))
         tolmax = max(tolmax,tol(i,1))
         tolmax = max(tolmax,tol(i,2))
         tolmax = max(tolmax,tol(i,3))
c
      enddo
      write(6,*) ldt,' tolerances',tolmin,tolmax
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_surf1(cbi,mx,ifout)
c
c
c     Generate a surface file highlighting any boundary with type
c     specified by cbi
c
c     This routine dumps the GLL coordinates of surfaces marked with
c     boundary condition "cbv".
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
c
      logical ifout
      character*3 cbi,cbv,cbt
c
      call vtk_out_head_obj('vtkobj',6,ifout,34)
c
      nfout  = 0
      nfaces = 2*ndim
      if (ifout) open(unit=38,file='tmp.vt1')
c
      do ie = 1,nel
         if (ifxin(ie).or.(.not.ifclip)) then
            do iface=1,nfaces
               cbv = cbc(iface,ie,1)
               cbt = cbc(iface,ie,1)
               if (cbt.eq.cbi .or. cbv.eq.cbi ) then
                  if (ifout) call out_face1(nfout,iface,ie,38,xp,yp,zp)
                  nfout = nfout+1
               endif
            enddo
         endif
      enddo
c
c     Now read the data from unit 38 and 39 back in, and write to unit 34
c
      npoints = nx*nx*nfout
c
c     POINT DATA
c
      write(6 ,1) npoints
      if (ifout) then
         rewind(38)
         write(34,1) npoints
         do i=1,npoints
            read (38,*) xx,yy,zz
            write(34,2) xx,yy,zz
         enddo
      endif
c
    1 format('POINTS ',i9,' float')
    2 format(1p3e13.5)
c
c     POLYGON DATA
c
      ncells = (nx-1)*(nx-1)*nfout
      ncpts  = 5*ncells
      write(6 ,3)  ncells,ncpts
    3 format('POLYGONS ',2i9)
      if (ifout) then
         write(34,3) ncells,ncpts
         do k=1,nfout
            do j=1,nx-1
            do i=1,nx-1
               jc1 = i-1 + (j-1)*nx + (k-1)*nx*nz
               jc2 = jc1+1
               jc3 = jc2+nx
               jc4 = jc1+nx
               write(34,4) jc1,jc2,jc3,jc4
    4          format(' 4 ',4i9)
            enddo
            enddo
         enddo
      endif
c
      close (unit=38)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_face1(nfout,ifce,ie,io,xf,yf,zf)
c
c     dump the GLL coordinates of specified face (ifce in prenek format)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real xf(nx,ny,nz,1)
      real yf(nx,ny,nz,1)
      real zf(nx,ny,nz,1)
c
      call dsset(nx,ny,nz)
c
      iface = eface1(ifce)
c
      JS1    = SKPDAT(IFACE,1)
      JF1    = SKPDAT(IFACE,2)
      JSKIP1 = SKPDAT(IFACE,3)
      JS2    = SKPDAT(IFACE,4)
      JF2    = SKPDAT(IFACE,5)
      JSKIP2 = SKPDAT(IFACE,6)
C
      do j2=js2,jf2,jskip2
      do j1=js1,jf1,jskip1
         write(io,*) xf(j1,j2,1,ie),yf(j1,j2,1,ie),zf(j1,j2,1,ie)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_head_obj(name,lname,ifout,io)
      character*1 name(lname)
      logical ifout
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c
      character*1  stru1(40,3)
      character*40 stru(3)
      equivalence (stru,stru1)
      save         stru
      data         stru /'# vtk DataFile Version 2.0              '
     $                  ,'BINARY                                  '
     $                  ,'DATASET POLYDATA                        '/
c
      character*1  strf1(40,3)
      character*40 strf(3)
      equivalence (strf,strf1)
      save         strf
      data         strf /'# vtk DataFile Version 2.0              '
     $                  ,'ASCII                                   '
     $                  ,'DATASET POLYDATA                        '/
c
c
      integer icalld
      save    icalld
      data    icalld /0/
      character*9 fname
c
c
      if (ifout) then
c
         icalld = icalld+1
c
c        write(fname,4) icalld
c   4    format('vtk.',i3.3)
c
         if (io.eq.33) write(fname,5) 
         if (io.eq.34) write(fname,6) 
    5    format('vt1.vtk')
    6    format('obj.vtk')
c
         if (iffmt_vtk) then
            open (unit=io,file=fname,form='formatted')
         else
            open (unit=io,file=fname,form='unformatted')
         endif
      endif
c
      i = 0
      do ii=1,4
         if (ii.ne.2) then
            i = i+1
            if (ifout) then
               len = ltrunc(strf(i),40)
               if (iffmt_vtk) then
                  write(io,1) (strf1(k,i),k=1,len)
                else
                  len = ltrunc(stru(i),40)
                  write(io,1) (stru1(k,i),k=1,len)
                endif
            endif
         else
            if (ifout) write(io,1) (name(k),k=1,lname)
            write(6 ,1) (name(k),k=1,lname)
         endif
      enddo
    1 format(40a1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mapreg3d1(w_map,mx,w_in,nx,w)
c
      real w_map(mx*mx*mx)
      real w_in (nx*nx*nx)
c
      real w(0:1)
c
      integer nx_last,mx_last
      save    nx_last,mx_last
      data    nx_last,mx_last  /-1,-1/
c
      parameter (nxm=30)
      real gll_pts(nxm)
      real gll_wts(nxm)
      real i (3*nxm*nxm)
      real it(3*nxm*nxm)
      save gll_pts,gll_wts,i,it
c
      if (nx.gt.nxm .or. nx*mx.gt. 3*nxm*nxm) then
         write(6,*) 'increase array dimensions in mapreg3d1',mx,nx,nxm
         call copy(w_map,w_in,nx*nx*nx)
         return
      endif
c
      if (nx.ne.nx_last .or. mx.ne.mx_last) then
c
c        Recompute interpolation matrices
c
         call legend(gll_pts,gll_wts,nx)
c        Use Fornberg's routine to compute interpolation coefficients
         nx1 = nx-1
         l  = 1
         do k=1,mx
            z0 = (2.*(k-1))/(mx-1) - 1.
            call fd_weights_full(z0,gll_pts,nx1,0,it(l))
            l  = l+nx
         enddo
         call transpose(i,mx,it,nx)
c
         nx_last = nx
         mx_last = mx
c
      endif
c
c
      nmx  = max(nx,mx)
      l    = nmx*nmx*nmx
      call map_tnsr(w_map,mx,w_in,nx,i,it,w,w(l),.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_surf (cbi,mx,ifout)
c
c
c     Generate a surface file highlighting any boundary with type
c     specified by cbi
c
c     This routine dumps the GLL coordinates of surfaces marked with
c     boundary condition "cbv".
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwka/ w1 (lvtk)
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
      logical ifout
      character*3 cbi,cbv,cbt
c
      call vtk_out_head_obj('vtkobj',6,ifout,34)
c
      nfout  = 0
      nfaces = 2*ndim
      if (ifout) open(unit=38,file='tmp.vt1')
      if (ifout) open(unit=39,file='tmp.vt2')
c
      do ie = 1,nel
         write(6,*) 'ifxin',ie,ifxin(ie),ifclip
         if (ifxin(ie).or.(.not.ifclip)) then
            do iface=1,nfaces
               cbv = cbc(iface,ie,1)
               cbt = cbc(iface,ie,1)
               if (cbt.eq.cbi .or. cbv.eq.cbi) then
                  if (ifout) call out_face
     $             (mx,nfout,iface,ie,xp,yp,zp,38,work,39,w1)
                  nfout = nfout+1
               endif
            enddo
         endif
      enddo
c
c     Now read the data from unit 38 and 39 back in, and write to unit 34
c
      npoints = mx*mx*nfout
c
c     POINT DATA
c
      write(6 ,1) npoints
      if (ifout) then
         rewind(38)
         write(34,1) npoints
         do i=1,npoints
            read (38,*) xx,yy,zz
            write(34,2) xx,yy,zz
         enddo
         close (unit=38)
      endif
c
    1 format('POINTS ',i9,' float')
    2 format(1p3e13.5)
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c     POLYGON DATA
      ncells = (mx-1)*(mx-1)*nfout
      ncpts  = 5*ncells
      write(6 ,3)  ncells,ncpts
    3 format('POLYGONS ',2i9)
      if (ifout) then
         write(34,3) ncells,ncpts
         do k=1,nfout
            do j=1,mx-1
            do i=1,mx-1
               jc1 = i-1 + (j-1)*mx + (k-1)*mx*mx
               jc2 = jc1+1
               jc3 = jc2+mx
               jc4 = jc1+mx
               write(34,4) jc1,jc2,jc3,jc4
    4          format(' 4 ',4i9)
            enddo
            enddo
         enddo
      endif
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c     POINT DATA
      write(6 ,21) npoints
   21 format('POINT_DATA ',i9,/,'SCALARS PRES2 float',/
     $      ,'LOOKUP_TABLE default')
c
      if (ifout) then
         rewind(39)
         write(34,21) npoints
         do i=1,npoints
            read (39,*) ww
            write(34,2) ww
         enddo
         close (unit=39)
      endif
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_face(mx,nfout,ifce,ie,xf,yf,zf,io,wf,iw,ww)
c
c     dump the GLL coordinates of specified face (ifce in prenek format)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real xf(nx,ny,nz,1)
      real yf(nx,ny,nz,1)
      real zf(nx,ny,nz,1)
      real wf(nx,ny,nz,1)
      real ww(mx,mx,mx,5)
c
      call mapreg3d1(ww(1,1,1,1),mx,xf(1,1,1,ie),nx,ww(1,1,1,5))
      call mapreg3d1(ww(1,1,1,2),mx,yf(1,1,1,ie),nx,ww(1,1,1,5))
      call mapreg3d1(ww(1,1,1,3),mx,zf(1,1,1,ie),nx,ww(1,1,1,5))
      call mapreg3d1(ww(1,1,1,4),mx,wf(1,1,1,ie),nx,ww(1,1,1,5))
c
      call dsset     (mx,mx,mx)
c
      iface = eface1(ifce)
c
      JS1    = SKPDAT(IFACE,1)
      JF1    = SKPDAT(IFACE,2)
      JSKIP1 = SKPDAT(IFACE,3)
      JS2    = SKPDAT(IFACE,4)
      JF2    = SKPDAT(IFACE,5)
      JSKIP2 = SKPDAT(IFACE,6)
C
      write(6,*) 'j1',js1,jf1,jskip1,ifce,nx
      write(6,*) 'j2',js2,jf2,jskip2,iface,mx
      do j2=js2,jf2,jskip2
      do j1=js1,jf1,jskip1
         write(io,*) ww(j1,j2,1,1),ww(j1,j2,1,2),ww(j1,j2,1,3)
         write(iw,*) ww(j1,j2,1,4)
      enddo
      enddo
c
      call dsset     (nx,ny,nz)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine quickscan_xy(x1,y1,x2,y2)
      include 'basics.inc'
c
      XMAX = -1.0E20
      YMAX = -1.0E20
      XMIN =  1.0E20
      YMIN =  1.0E20
      nc = 4
      if (if3d) nc = 8
      DO 50 IEL=1,NEL
         DO 50 IC=1,nc
         XMAX = MAX(XMAX,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         XMIN = MIN(XMIN,XISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         YMAX = MAX(YMAX,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
         YMIN = MIN(YMIN,YISO(X(IEL,IC),Y(IEL,IC),Z(IEL,IC)) )
50    CONTINUE
c
      dx = xmax-xmin
      dy = ymax-ymin
      x1 = xmin-.025*dx
      x2 = xmax+.025*dx
      y1 = ymin-.025*dy
      y2 = ymax+.025*dy
c
      return
      end
c-----------------------------------------------------------------------
      subroutine s3to2_regular
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the z-direction and produces a 
c     2d .fld file.    NOTE:  *regular* assumes that data is stacked
c     in identical z-planes.
c
c
      integer nelzz
      save    nelzz
      data    nelzz /0/
c
      if (nelzz.eq.0) then
         call prs('Enter number of elements in z-direction:$')
         call rei(nelzz)
      endif
c
c
      nelxy  = nel/nelzz
      nxy    = nx*ny
      nxyz   = nx*ny*nz
      n2d    = nelxy*nx*ny
      ntot   = nel*nx*ny*nz
c
      dzmax  = glmax(zp,ntot) - glmin(zp,ntot)
c
      call rzero(wkv1,ntot)
      call rzero(wkv2,ntot)
      call rzero(wkv3,ntot)
      call rzero(work,ntot)
c
c
      do iez=1,nelzz
c
         is = 1+n2d*nz*(iez-1)
c
         zlen = (vlmax(zp(is),nxyz) - vlmin(zp(is),nxyz)) / dzmax
c
         call slab_sum
     $      (wkv1,wkv2,wkv3,work,u(is),v(is),p(is),w(is),nxy,nelxy,zlen)
c    $      (wkv1,wkv2,wkv3,work,u(is),v(is),p(is),t(is),nxy,nelxy,zlen)
c
      enddo
c
      call outfld(nx,ny,1,nelxy,2,wkv1,wkv2,wkv3,wkv3,work)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine slab_sum(ua,va,pa,ta,uin,vin,pin,tin,nxy,nelxy,zlen)
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ua(nxy,1),va(nxy,1),pa(nxy,1),ta(nxy,1)
      real uin(nxy,nz,1),vin(nxy,nz,1),pin(nxy,nz,1),tin(nxy,nz,1)
      real zlen
c
c     This routine computes a weighted sum of data in the z-direction 
c     for a single slab of elements
c
      do ie=1,nelxy
         do iz=1,nz
            scale = zlen*0.5*wght(iz)
            do i=1,nxy
               ua(i,ie) = ua(i,ie) + scale*uin(i,iz,ie)
               va(i,ie) = va(i,ie) + scale*vin(i,iz,ie)
               pa(i,ie) = pa(i,ie) + scale*pin(i,iz,ie)
               ta(i,ie) = ta(i,ie) + scale*tin(i,iz,ie)
            enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_uvwpt_regular
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /sfldwrk/ odumpw,odump
      integer odumpw,odump
c
c     This routine averages data in the z-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
c     As of Jan 02, 2002, the geometry may be deformed in the z-plane
c
c     ntot = nx*ny*nz*nel
c     call copy(u,zp,ntot)
c     call prs('copy z to u!!!, avg_uvwpt_regular!!!$')
c
      call avg_u_regular(u)
      call avg_u_regular(v)
      call avg_u_regular(w)
      call avg_u_regular(p)
      call avg_u_regular(t)
c
c     Restore/recompute current work array
c
      call setwrk(.true.)
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_u_regular(uin)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the z-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
      real uin(1)
      real num,den
c
      integer nelzz
      save    nelzz
      data    nelzz /0/
c
      if (.not.if3d) return
      if (nelzz.eq.0) then
         call prs('Enter number of elements in z-direction:$')
         call rei(nelzz)
      endif
c
c
      nelxy  = nel/nelzz
      nxy    = nx*ny
      nxyz   = nx*ny*nz
      n3d1   = nelxy*nx*ny*nz
c
      do ie=1,nelxy
         i  = 1 + nxyz*(ie-1)
         iw = 1 + nxy *(ie-1)
         do ix=1,nxy
            num = 0.
            den = 0.
            do je=1,nelzz
               k = i + n3d1*(je-1)
               do jz=1,nz

c                 num = num+jacm1(k)*uin(k)
c                 den = den+jacm1(k)

                  num = num+jacm1(k)*wght(jz)*uin(k) ! need zwght
                  den = den+jacm1(k)*wght(jz)        ! cuz k doesn't advance

c                 num = num+wght(jz)*uin(k) ! need zwght
c                 den = den+wght(jz)        ! cuz k doesn't advance

                  k   = k  + nxy
               enddo
            enddo
            if (den.eq.0.) then
               write(6,*) 'den=0? avg_u_regular',i,je,nz,num
               work(iw) = 0.
            else
               work(iw) = num/den
            endif
            i  = i +1
            iw = iw+1
         enddo
      enddo
c
      is = 1
      do iez=1,nelzz
         js=1
         do iexy = 1,nelxy
            do iz = 1,nz
               call copy(uin(is),work(js),nxy)
               is = is+nxy
            enddo
            js = js+nxy
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_u_regular_OLD(uin)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the z-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
      real uin(1)
c
      integer nelzz
      save    nelzz
      data    nelzz /0/
c
      if (.not.if3d) return
      if (nelzz.eq.0) then
         call prs('Enter number of elements in z-direction:$')
         call rei(nelzz)
      endif
c
c
      nelxy  = nel/nelzz
      nxy    = nx*ny
      nxyz   = nx*ny*nz
      n2d    = nelxy*nx*ny
      ntot   = nel*nx*ny*nz
c
      dzmax  = glmax(zp,ntot) - glmin(zp,ntot)
c
      call rzero(work,ntot)
c
c
      do iez=1,nelzz
         is = 1+n2d*nz*(iez-1)
         zlen = (vlmax(zp(is),nxyz) - vlmin(zp(is),nxyz)) / dzmax
         call slab_sum1 (work,uin(is),nxy,nelxy,zlen)
      enddo
c
      is = 1
      do iez=1,nelzz
         js=1
         do iexy = 1,nelxy
            do iz = 1,nz
               call copy(uin(is),work(js),nxy)
               is = is+nxy
            enddo
            js = js+nxy
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine slab_sum1(ua,uin,nxy,nelxy,zlen)
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ua(nxy,1)
      real uin(nxy,nz,1)
      real zlen
c
c     This routine computes a weighted sum of data in the z-direction 
c     for a single slab of elements
c
      do ie=1,nelxy
         do iz=1,nz
            scale = zlen*0.5*wght(iz)
            do i=1,nxy
               ua(i,ie) = ua(i,ie) + scale*uin(i,iz,ie)
            enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_tprd(tprd,xrc,yrc,zrc)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     Get theta, phi, r,  dtheta, dphi, dr
c
      real tprd(6,nel)
      real xrc(nx,ny,nz,nel),yrc(nx,ny,nz,nel),zrc(nx,ny,nz,nel)
c
c     find cg of each element in polar coordinates
c
      one = 1.
      pi  = 4.*atan(one)
c
      nz1 = max(1,nz-1)
      npt = 2**ndim
c
      do ie=1,nel
c
         tmin =  1.e20
         pmin =  1.e20
         rmin =  1.e20
c
         tmax = -1.e20
         pmax = -1.e20
         rmax = -1.e20
c
         do iz=1,nz,nz1
         do iy=1,ny,ny-1
         do ix=1,nx,nx-1
            r2   = xrc(ix,iy,iz,ie)**2
     $           + yrc(ix,iy,iz,ie)**2
            r2   = sqrt(r2)
c
            r    = xrc(ix,iy,iz,ie)**2
     $           + yrc(ix,iy,iz,ie)**2
     $           + zrc(ix,iy,iz,ie)**2
            r    = sqrt(r)
c
            tht  = atan2(yrc(ix,iy,iz,ie),xrc(ix,iy,iz,ie))
            phi  = atan2(zrc(ix,iy,iz,ie),r2)
c
            tmin = min(tmin,tht)
            pmin = min(pmin,phi)
            rmin = min(rmin,r)
c
            tmax = max(tmax,tht)
            pmax = max(pmax,phi)
            rmax = max(rmax,r)
         enddo
         enddo
         enddo
c
         dr   = rmax-rmin
         dp   = pmax-pmin
         dt   = tmax-tmin
         if (dt.gt.pi) then
c           we have to re-do this set
            t1 =  1.e20
            t2 = -1.e20
            do iz=1,nz,nz1
            do iy=1,ny,ny-1
            do ix=1,nx,nx-1
               tht  = atan2(yrc(ix,iy,iz,ie),xrc(ix,iy,iz,ie))
               if (tht.lt.0) tht = tht + 2.*pi
               t1 = min(t1,tht)
               t2 = max(t2,tht)
            enddo
            enddo
            enddo
            dt   = t2-t1
c
c           write(6,6) ie,t1,t2,tmin,tmax,dt
c  6        format('bdt',i6,1p5e12.4)
            tmin = t1
            tmax = t2
            dt   = tmax-tmin
         endif
c
         tavg = 0.5*(tmax+tmin)
         pavg = 0.5*(pmax+pmin)
         ravg = 0.5*(rmax+rmin)
c
         tprd(1,ie) = tavg
         tprd(2,ie) = pavg
         tprd(3,ie) = ravg
         tprd(4,ie) = dt
         tprd(5,ie) = dp
         tprd(6,ie) = dr
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_uvwpt_radial
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /sfldwrk/ odumpw,odump
      integer odumpw,odump
c
c     This routine averages data in the R-direction and stuffs it back
c     into u,v,w,p,t
c
      call avg_u_radial(u)
      call avg_u_radial(v)
      call avg_u_radial(w)
      call avg_u_radial(p)
      call avg_u_radial(t)
c
c     Restore/recompute current work array
c
c     call setwrk(.true.)
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_all(mx,ifout)
c
c     Generate an FEM input file based on Nekton .rea file
c
c     Dump all data of interest:
c
c     xyz
c     vortex
c     pressure
c     ux,uy,uz
c     wx,wy,wz
c
C
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifout
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwki/ loc(lvtk),ind(lvtk)
      common /vtkwka/ w1 (lvtk),wk (lvtk)
      common /vtkwsi/ mx_save,nvtk,nelclip,ncells
      common /vtkwkr/ xcmn,xcmx
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
c
      common /bigvtki/ jglob(lvtk),kcell(8,lvtk)
      common /bigvtkr/ vtkxyz(lvtk,3)
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
c
      real xm(1),ym(1),zm(1)
      equivalence (xm,vtkxyz(1,1))
      equivalence (ym,vtkxyz(1,2))
      equivalence (zm,vtkxyz(1,3))
c
      integer ic(8)
c
      logical ifnew
      save    ifnew
c
c
      ifnew = .false.
      if (mx.ne.mx_save) ifnew = .true.
      mx_save = mx
      my = mx
      mz = mx
      if (.not.if3d) mz=1
c
      if (ifnew) then
c
c        Fomatted output?
c
         iffmt_vtk = .true.
         ifdouble  = .false.
         if (ifout) then
c        Query format type
            ibin=1
c           call prs('ASCII (1) or BINARY (0) vtk file? Enter 1 or 0:$')
c           call rei(ibin)
c           if (ibin.eq.0) iffmt_vtk = .false.
c        Query symmetry option
c           call prs('Symmetry?  (1=yes)$')
c           call rei(isym)
            isym=0
            if (isym.eq.1) ifdouble = .true.
         endif
c
C        Clip vtk output?
c
         ifclip = .true.
         call prs('Clip volume in x? (enter xmn,xmx or  0,0 for no)$')
         call rerr(xcmn,xcmx)
         if (xcmn.eq.0.  .and. xcmx.eq.0.) ifclip = .false.
         nx3     = nx*ny*nz
         nelclip = nel
         call find_xin(nelclip,ifclip,ifxin,xp,nx3,nel,xcmn,xcmx)
      endif
      nton=nx*ny*nz*nel
      ntot=mx*my*mz*nelclip
      nvtk=ntot
c
      write(6,*) 'This is iffmt_vtk:',iffmt_vtk
      if (ifclip) call prsrr('Clipping in effect!$',xcmn,xcmx)
      call prsii('nelclip $',nelclip,nel)
      call prsii('ntot    $',nvtk   ,nton)
c
c
c
      call prsii('call mapreg3d mx,nx$',mx,nx)
      call mapreg3d(xm,mx,lvtk,xp,nx,nel,ifclip,ifxin,wk)
      call mapreg3d(ym,mx,lvtk,yp,nx,nel,ifclip,ifxin,wk)
      if (if3d) 
     $call mapreg3d(zm,mx,lvtk,zp,nx,nel,ifclip,ifxin,wk)
c
c     Establish local to global node numbering for GLL points
c
      call locglobm(jglob,nglb,mx,nel)
      write(6,*) 'this is nglb',nglb,ntot,mx,nel
c
c     Now, use jglob to sort data
c
c     call out_glob(jglob,mx,my,mz,nel)
      call icopy (loc ,jglob   ,ntot)
      call isort (loc ,ind     ,ntot) ! isort actually sorts now
c     call iswap (loc ,w1  ,ind,ntot)
c
c     Compress and output coordinate data
c
      call prsii('call mapreg3d mx,nx$',mx,nx)
      call mapreg3d(xm,mx,lvtk,xp,nx,nel,ifclip,ifxin,wk)
      call mapreg3d(ym,mx,lvtk,yp,nx,nel,ifclip,ifxin,wk)
      if (if3d) 
     $call mapreg3d(zm,mx,lvtk,zp,nx,nel,ifclip,ifxin,wk)
c
      call compress (ngpt,xm,loc,ind,ntot,w1)
      call compress (ngpt,ym,loc,ind,ntot,w1)
      call compress (ngpt,zm,loc,ind,ntot,w1)
c
      call vtk_out_head('vtkfile',7,ifout,33)
      if (ifnew) call vtk_out_xyz (xm,ym,zm,ngpt,ifout)
c
c     Output cell data
c
      if (ifnew) call vtk_out_hex (jglob,mx,my,mz,nelclip,ifout)
c
c
c     Reset scalar/vector field data
c
      QUANTY = 'VORTEX'
      call setwrk(.true.)
c
c     Now, use loc,ind to sort data
c
c     Compress and output scalar work data
c
      m_scalar = 2
      do i=1,m_scalar
         call mapreg3d (wk,mx,lvtk,work,nx,nel,ifclip,ifxin,w1)
         call compress         (ngpt,wk,loc,ind,ntot,w1)
c
c
c        quick hack:  force name to be "PRESSURE" on 2nd pass..
         if (i.eq.1) QUANTY = 'VORTEX'
         if (i.eq.2) QUANTY = 'PRESSURE'
         call vtk_out_sca      (wk,ngpt,ifout,i)
c
c        Get 2nd scalar
c
         QUANTY = 'PRESSURE'
         DERIV  = 'NO'
         COMPON = 'M'
         if (i.lt.m_scalar) call setwrk(.false.)
      enddo
c
c
c     Compress and output vector work data
c
ccc         QUANTY = 'VELOCITY'
ccc         DERIV  = 'NO'
ccc         COMPON = 'X'
ccc         call setwrk(.false.)
ccc   c
ccc         m_vector = 2
ccc         do i=1,m_vector
ccc            call mapreg3d(xm,mx,lvtk,wkv1,nx,nel,ifclip,ifxin,wk)
ccc            call mapreg3d(ym,mx,lvtk,wkv2,nx,nel,ifclip,ifxin,wk)
ccc            if (if3d) 
ccc        $   call mapreg3d(zm,mx,lvtk,wkv3,nx,nel,ifclip,ifxin,wk)
ccc   c
ccc            call compress (ngpt,xm,loc,ind,ntot,w1)
ccc            call compress (ngpt,ym,loc,ind,ntot,w1)
ccc            call compress (ngpt,zm,loc,ind,ntot,w1)
ccc   c
ccc            call vtk_out_vec (xm,ym,zm,ngpt,ifout,3)
ccc   c
ccc   c        Get 2nd vector
ccc   c
ccc            QUANTY = 'VORTICITY'
ccc            DERIV  = 'NO'
ccc            COMPON = 'X'
ccc            if (i.lt.m_vector) call setwrk(.false.)
ccc         enddo
ccc   c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_vec (uu,vv,ww,n,ifout,ivtk_dump)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      logical ifout
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
      real uu(1),vv(1),ww(1)
c
c
      if (ivtk_dump.eq.1) then
c        We need to specify the number of points
         if (ifout) nold = n
         n2 = n
         if (ifdouble) n2 = 2*n
         if (ifout) write(33,1) n2
         write(6 ,1) n2
    1    format('POINT_DATA ',i9)
      endif
c
      if (ifout) write(33,2) quanty
      write(6 ,2) quanty
    2 format('VECTORS ',a20,' float')
c
      wwmin = glmin(ww,n)
      wwmax = glmax(ww,n)
      call blank(line,70)
      write(line,7) wwmin,wwmax,quanty
    7 format('minmax:',1p2e11.3,2x,a20,'$')
      call prs(line)
c
      if (ifout) then
         do i=1,n
            write(33,10) uu(i),vv(i),ww(i)
         enddo
         if (ifdouble) then
            do i=1,n
               write(33,10) uu(i),vv(i),ww(i)
            enddo
         endif
      endif
   10 format(1p3e13.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vtk_out_surf0(cbi,mx,ifout)
c
c
c     Generate a surface file highlighting any selected surface
c
c     This routine dumps the GLL coordinates of surfaces marked with
c     boundary condition "cbv".
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (lvtk=2000000)  !  8530000)
      common /vtkwka/ w1 (lvtk)
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble
c
      logical ifout
      character*3 cbi,cbv,cbt
c
      call vtk_out_head_obj('vtkobj',6,ifout,34)
c
      nfout  = 0
      nfaces = 2*ndim
      if (ifout) open(unit=38,file='tmp.vt1')
      if (ifout) open(unit=39,file='tmp.vt2')
c
      do i = 1,nlstp
         call decod(iplane,ipln,ie,idum,list,nx,6,nelm)
c
c        ipln = 1 or 2   ==>  X-face
c        ipln = 3 or 4   ==>  Y-face
c        ipln = 5 or 6   ==>  Z-face
c
         iface = eface(ipln)
         write(6,*) 'ifxin',ie,ifxin(ie),ifclip,iface,ipln
         if (ifxin(ie).or.(.not.ifclip)) then
            if (ifout) call out_face
     $          (mx,nfout,iface,ie,xp,yp,zp,38,work,39,w1)
            nfout = nfout+1
         endif
      enddo
c
c     Now read the data from unit 38 and 39 back in, and write to unit 34
c
      npoints = mx*mx*nfout
c
c     POINT DATA
c
      write(6 ,1) npoints
      if (ifout) then
         rewind(38)
         write(34,1) npoints
         do i=1,npoints
            read (38,*) xx,yy,zz
            write(34,2) xx,yy,zz
         enddo
         close (unit=38)
      endif
c
    1 format('POINTS ',i9,' float')
    2 format(1p3e13.5)
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c     POLYGON DATA
      ncells = (mx-1)*(mx-1)*nfout
      ncpts  = 5*ncells
      write(6 ,3)  ncells,ncpts
    3 format('POLYGONS ',2i9)
      if (ifout) then
         write(34,3) ncells,ncpts
         do k=1,nfout
            do j=1,mx-1
            do i=1,mx-1
               jc1 = i-1 + (j-1)*mx + (k-1)*mx*mx
               jc2 = jc1+1
               jc3 = jc2+mx
               jc4 = jc1+mx
               write(34,4) jc1,jc2,jc3,jc4
    4          format(' 4 ',4i9)
            enddo
            enddo
         enddo
      endif
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c     POINT DATA
      write(6 ,21) npoints
   21 format('POINT_DATA ',i9,/,'SCALARS PRES2 float',/
     $      ,'LOOKUP_TABLE default')
c
      if (ifout) then
         rewind(39)
         write(34,21) npoints
         do i=1,npoints
            read (39,*) ww
            write(34,2) ww
         enddo
         close (unit=39)
      endif
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
      return
      end
c-----------------------------------------------------------------------
      subroutine clrc(str11)
      include 'basics.inc'
      include 'basicsp.inc'
      character*11 str11
c
      call clear
      CALL HARD
      CALL HEJECT
      CALL DRMESH
      CALL DRFRAM
      CALL NOHARD
c
      call prs(str11)
      call prs('hit any key to continue$')
      call res (ans,1)
      return
      end
c-----------------------------------------------------------------------
      subroutine set_cluster(cluster,ncl,nelzz,tprd)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the R-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
      real uin(1)
c
      real    tprd(6,nelm),w6(6)
      integer ind(nelm),ind1(nelm),ninseg(nelm),iw(nelm)
      logical ifseg(nelm)
c
      integer cluster (nelm),ncl
c
      write(6,*) 'this is nelzz',nelzz,nel
      call get_tprd(tprd,xp,yp,zp)
      call iint(ind1,nel)
c
      nseg = 1
      ninseg(nseg) = nel
      do i=1,nel
         ifseg(i)=.false.
      enddo
      ifseg(1)=.true.
c
      write(6,*) 'this is nseg',nseg,ndim
      do idim=1,ndim
         j1 = 1
         do iseg=1,nseg
c           write(6,*) 'this is iseg',iseg,ninseg(iseg),idim
            call tuple_sort(tprd(1,j1),6,ninseg(iseg),idim,1,ind,w6)
            call iswap     (ind1(j1),iw,ind,ninseg(iseg))
            j1 = j1+ninseg(iseg)
         enddo
c
         if (idim.le.2) then
c
c           check for jumps in current coordinate 
c           (for theta & phi, but not r)
c
            do i=2,nel
               aj = 0.1*(min(tprd(idim+3,i),tprd(idim+3,i-1)))
               dj = abs(tprd(idim,i)-tprd(idim,i-1))
               if (dj.gt.aj) ifseg(i) = .true.
            enddo
c
c           Count up number of different segments
            nseg = 0
            do i=1,nel
               if (ifseg(i)) then
                  nseg = nseg+1
                  ninseg(nseg) = 1
               else
                  ninseg(nseg) = ninseg(nseg) + 1
               endif
c              Assign cluster id to original element
               cluster(ind1(i)) = nseg
            enddo
c           Number of clusters = number of segments
            ncl = nseg
         endif
      enddo
c
c     Compute nelzz --- this should equal to ninseg
      nelzz = ninseg(1)
      do i=1,nseg
         if (ninseg(i).ne.nelzz) 
     $   write(6,*) 'Ninseg! :( ',ninseg(i),i,nelzz
      enddo
      write(6,*) 'nelzz,nseg:',nelzz,nseg
c
c     do i=1,nel
c        write(8,6) ifseg(i),i,(tprd(k,i),k=1,6)
c     enddo
c   6 format(l4,i6,1p6e14.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_u_radial(uin)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the R-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
      real uin(1)
c
      integer nelzz
      save    nelzz
      data    nelzz /0/
      real    tprd(6,nelm),rmin,rmax
      save    tprd,rmin,rmax
      integer ind(nelm),ind1(nelm),ninseg(nelm),iw(nelm)
      logical ifseg(nelm)
c
      integer cluster (nelm),ncl
      save    cluster,ncl
c
      write(6,*) 'this is nelzz',nelzz,nel
      if (nelzz.eq.0) then
         call set_cluster(cluster,ncl,nelzz,tprd)
         call get_tprd(tprd,xp,yp,zp)
c
         rmin =  9.e20
         rmax = -9.e20
         do ie=1,nel
            rmin = min(rmin,tprd(3,ie))
            rmax = max(rmax,tprd(3,ie))
            tprd(ie,1) = tprd(6,ie)
         enddo
      endif
c
      call avg_u_radial2(uin,work,rmin,rmax,tprd,cluster,ncl,nelzz)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_u_radial2(uin,wk,rmin,rmax,dr,cluster,ncl,nelzz)
      include 'basics.inc'
      include 'basicsp.inc'
c
c     This routine averages data in the r-direction and stuffs it back
c     into u,v,w,p,t
c
c     NOTE:  *regular* assumes that data is stacked in identical z-planes.
c
      real uin(nx,ny,nz,nel),wk(nx,ny,nel),dr(nel)
      integer cluster(1)
c
c
c
      nelxy  = nel/nelzz
      nxy    = nx*ny
      nxyz   = nx*ny*nz
      n2d    = nelxy*nx*ny
      ntot   = nel*nx*ny*nz
c
      dzmax  = rmax-rmin
      den = 0.5*(rmax*rmax - rmin*rmin)
      if (den.eq.0) then
         call prsrr('Error avg_u_radial, rmax=rmin:$',rmin,rmax)
         return
      endif
c
      call rzero(wk,ntot)
c
c
      i = 0
      do ie=1,nel
         ic=cluster(ie)
c        write(6,*) 'ic:',ic,ie
         do iz=1,nz
         do iy=1,ny
         do ix=1,nx
            i = i+1
            r    = xp(i)**2 + yp(i)**2 + zp(i)**2
            r    = sqrt(r)
            wgt   = dr(ie)*r*wght(iz)/den
            wk(ix,iy,ic) = wk(ix,iy,ic) + uin(ix,iy,iz,ie)*wgt
         enddo
         enddo
         enddo
      enddo
c
c     Map back to original position
c
      do ie=1,nel
         ic=cluster(ie)
         do iy=1,ny
         do ix=1,nx
         do iz=1,nz
            uin(ix,iy,iz,ie) = wk(ix,iy,ic)
         enddo
         enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_u_cylinder(ur,uin,mlx)
c
c     Compute cylinder average of uin -- assumes tensor-product geometry
c
      include 'basics.inc'
      include 'basicsp.inc'
      real ur(1),uin(1)
c
      integer melx,mely,melz
      save    melx,mely,melz
      data    melx,mely,melz / 3*0 /
c
      common /cylavg/ jac1di(nxm*nelm),rad1d(nxm*nelm)
      real jac1di
c
      if (melx.eq.0) then
         melx = abs(param(116))
         mely = abs(param(117))
         melz = 1
         if (if3d) melz = abs(param(118))
         mel = melx*mely*melz
         if (mel.ne.nel) 
     $     call prsii('PROBLEM W/ CYLINDER AVG. (p116-118):$',mel,nel)
         call get_jac1di(jac1di,rad1d,melx,mely,melz)
         write(6,*) 'this is melx:',melx,mely,melz
      endif
      mlx = melx
c
      do ie=1,melx
      do i =1,nx
         i_ie = i + nx*(ie-1)
         ur(i_ie) = 0.
      enddo
      enddo
c
      ixyz = 0
      do ke=1,melz
      do je=1,mely
      do ie=1,melx
      do k =1,nz
      do j =1,ny
      do i =1,nx
         ixyz = ixyz+1
         i_ie = i + nx*(ie-1)
         ur(i_ie) = ur(i_ie)+jac1di(i_ie)*jacm1(ixyz)*uin(ixyz)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_jac1di(jac1di,rad1d,melx,mely,melz)
c
c     Get 1D Jacobian in "r" direction, assuming tensor-product mesh
c
      include 'basics.inc'
      include 'basicsp.inc'
      real jac1di(nx,melx),rad1d(nx,melx)
c
      nxyz = nx*ny*nz
      do ie=1,melx
         i0 = nxyz*(ie-1)
         do i =1,nx
            i0 = i0+1
            rad1d(i,ie) = 0
            rad2        = xp(i0)*xp(i0) + yp(i0)*yp(i0)
            if (rad2.gt.0) rad1d(i,ie) = sqrt(rad2)
         enddo
      enddo
c
      call mxm(dgdr,nx,rad1d,nx,jac1di,melx)
      do i=1,nx*melx
         jac1di(i,1) = 1./jac1di(i,1)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine avg_cylinder
c
c     Compute cylinder average of uin -- assumes tensor-product geometry
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cylav1/ ru(nxm*nelm),rv(nxm*nelm),rw(nxm*nelm)
      common /cylavg/ jac1di(nxm*nelm),rad1d(nxm*nelm)
      real jac1di
c
      call avg_u_cylinder(ru,u,mlx)
      call avg_u_cylinder(rv,v,mlx)
      call avg_u_cylinder(rw,w,mlx)
c
      open(unit=79,file='avgcyl.dat')
      call prsii('This is mlx:$',mlx,nx)
      do i=1,nx*mlx
         write(79,7) time,rad1d(i),ru(i),rv(i),rw(i)
         write(6 ,7) time,rad1d(i),ru(i),rv(i),rw(i)
      enddo
      close(unit=79)
      call prsii('This is mlx:$',mlx,nx)
    7 format(1p5e13.5)
      write(79,*)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_glob(jglob,mx,my,mz,nel)
c
      integer jglob(mx,my,mz,nel)
c
      n = 0
      do l=1,nel
      do k=1,mz
      do j=1,my
      do i=1,mx
         n = n+1
         write(6,1) jglob(i,j,k,l),i,j,k,l,n
      enddo
      enddo
      enddo
      enddo
1     format(6i7,' jglob()')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outfld(mx,my,mz,mel,mdim,uu,vv,ww,pp,tt)
c
c     Take as input unformatted (param(66)=4) .fld file and put it as 
c     output
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      character*40 fname
      character*80 s80
c
      real*4  bytetest
      logical if_byte_sw, if_byte_swap_test, if_byte_swap_test8
c
      common /value_fld_i/ nxr,nyr,nzr,neltr
      common /value_fld_c/ excode_i(10)
      character*2 excode_i
c
c     Set up output file
c
      call blank(fname,40)
      write(fname,1)
    1 format('xyz.fld00')
      call terminate_string(fname,40)
      call byte_open(fname)
      call prs('Writing data to xyz.fld00$')
c
c     Write header info from new file, so we know how many elements, etc.
c
      call blank(excode_i,20)
      excode_i(1) = 'U '
      excode_i(2) = 'P '
      excode_i(3) = 'T '
      idsize = 4
      write(s80,'(4i4,1x,g13.4,i5,1x,10a2,i2)')
     $ mel,mx,my,mz,time,istep,(excode_i(i),i=1,10),idsize
      call byte_write(s80,20)
c
c     write test pattern to determine if byte-swapping is req'd
      test_pattern = 6.54321
      call byte_write(test_pattern,1)
c
c     Dump data, scalar fld by scalar fld
c
      k=1
      nxyzr = mx*my*mz
      do ie=1,nel
c
c        call byte_write(xp(k),nxyzr)
c        call byte_write(yp(k),nxyzr)
c        if (if3d) call byte_write(zp(k),nxyzr)
c
         call byte_write(uu(k),nxyzr)
         call byte_write(vv(k),nxyzr)
         if (mdim.eq.3) call byte_write(ww(k),nxyzr)
         call byte_write(pp(k),nxyzr)
         call byte_write(tt(k),nxyzr)
c
         k=k+nxyzr
      enddo
c
c     Close output file
c
      call byte_close()
c
      return
      end
c-----------------------------------------------------------------------
