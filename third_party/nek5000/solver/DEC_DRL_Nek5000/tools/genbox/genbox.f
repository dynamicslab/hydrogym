C
C     5 Aug 1994 updated  Paul F. Fischer
c     18 Jan 1996  pff
c     3/19 1996    pff
c     12/8 2009    pff - merged genbox / genb6
C
c     This program generates a nekton .rea file which comprises
c     a number of *tensor-product* boxes, i.e., a set of Nelx x Nely x Nelz
c     elements, where the locations of the vertices of the elements are
c     given as one-dimensional arrays of length (Nelx+1), (Nely+1), and (Nelz+1), 
c     respectively.
c
C    .For each box, the User specifies the number of elements in the 
c     x-y-z directions and the corresponding locations of the vertices
c     and the boundary conditions on the box
c
C    .INPUT FILE FORMAT:
c
c     line  0:    name of .rea file (start in col. 1, include .rea suffix)
c     line  1:    ndim              (MUST be same as in above .rea file! )
c     line  2:    Nelx, Nely, Nelz Nfld   for Box 1 
c     line  3:    x_0  x_1 ....  x_Nelx   for Box 1
c     line  4:    y_0  y_1 ....  y_Nely   for Box 1
c     line  5:    z_0  z_1 ....  z_Nelz   for Box 1
c     line  6:    bc1,bc2,bc3,bc4,bc5,bc6 for Box 1, fld 1
c     line  7:    bc1,bc2,bc3,bc4,bc5,bc6 for Box 1, fld 2  (If Nfld > 1)
c     line  8:    Nelx, Nely, Nelz Nfld   for Box 2 
c     line  9:          etc.
c
c
c    .Note that the boundary conditions are read with (6(a3,1x)) format, so
c     those lines must start in column 1 of the input file.  The rest is
c     free formatted, so any standard input format is ok.
c
c    .A typical input file for a 3D case is the following:
c
c>#
c>#========================================================
c>#    comments
c>#
c>#    This is the set of box data for full spherical configuration
c>#
c>#========================================================
c>#
c>ps3.rea
c>3                         NDIM
c>3                         NFLDS
c>box_1                     (Any string with 1st character .ne. "c" or "C")
c>-4   4  -4                (nelx,nely,nelz for Box) 
c>0 1.                      (if nelx<0, length to be equally divided)
c>0 .25 .5 .75 1.0          (if nely>0, nely+1 y-values for start/stop of els)
c>0 1.
c>P  ,P  ,P  ,P  ,P  ,P     (fluid bc's)
c>P  ,P  ,P  ,P  ,P  ,P     (temp  bc's)
c>P  ,P  ,P  ,P  ,P  ,P     (passive scalar bc's)
c>box_2                     (Any string with 1st character .ne. "c" or "C")
c>-4   4  -4                (nelx,nely,nelz for Box) 
c>1 2.                      (if nelx<0, length to be equally divided)
c>0 .25 .5 .75 1.0          (if nely>0, nely+1 y-values for start/stop of els)
c>0 1.
c>P  ,P  ,P  ,P  ,P  ,P     (fluid bc's)
c>P  ,P  ,P  ,P  ,P  ,P     (temp  bc's)
c>P  ,P  ,P  ,P  ,P  ,P     (passive scalar bc's)
c
c
c------------------------------------------------------------------------------
      program genbox
      include 'SIZE'
      character*132 string
      character*1  string1(132)
      equivalence (string,string1)
 
      logical     if3d,  ifevenx,ifeveny,ifevenz
     $           ,ifflow,ifheat, iffo,   if_multi_seg, ifmhd

      character*3 cbc1,   cbc2,   cbc3,   cbc4,   cbc5,   cbc6
      real        rbc1(5),rbc2(5),rbc3(5),rbc4(5),rbc5(5),rbc6(5)
      integer eface(6)
      save    eface
      data    eface  / 4,2,1,3,5,6 /
      integer nelxyz(3)

      character*1  apt(52)
      character*52 apt52
      equivalence (apt52,apt)
      data apt52/'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'/

      integer nlx(mbox),nly(mbox),nlz(mbox)
      real x(0:maxx,mbox),y(0:maxx,mbox),z(0:maxx,mbox)
      real xc(mbox),yc(mbox),zc(mbox)
      real curve(8,maxel)
      character*3 cbc(6,mbox,20)
      character*1 boxcirc(mbox)
      integer e0

      real*8 rbc8(6)
      integer ibc(6)

      common /genbr/ x,y,z,xc,yc,zc,curve
      common /genbc/ cbc

      real*4 buf (60)
      real   buf2(30)
      equivalence (buf,buf2)


      character*80 hdr
      real*4 test
      data   test  / 6.54321 /



      one    = 1.
      pi     = 4.*atan(one)
      pi180  = pi/180.

      wdsize = 4
      eps=1.0e-12
      oneeps = 1.0+eps
      if (oneeps.ne.1.0) then
         wdsize=8
      endif


c     Get the input file, which specifies the base .rea file
      call blank(string,132)
      write(6,*) 'input file name:'
      read (5,132) string
  132 format(a132)
      open (unit=7,file=string,status='old')
c
C     Read in name of previously generated NEKTON data set.
C     (Must have same dimension and number of fields as current run)
C
      call gets(string,132,iend,7)
 
      open (unit=8,file=string,status='old')
      open (unit=9,file='box.rea')
c-----------------------------------------------------------------------
c     here is where the 2d/3d determination is made....
      if3d = .false.
      iffo = .false.
      call geti1(ndim,iend,7)
      if(ndim.gt.0) then  ! default is ascii
        iffo = .true.
      else
        call byte_open('box.re2' // char(0))
      endif
      ndim = abs(ndim)
      if (ndim.eq.3) then 
          if3d = .true.
      elseif(ndim.ne.2) then
          write(6,*) "Original .rea file has invalid NDIM -- ",ndim
          write(6,*) "********Setting new NDIM = 2************* "
          ndim = 2
      endif
      
c-----------------------------------------------------------------------
c     here is where the fluid, fluid+heat, mhd determination is made....
      call getr1(rfld,iend,7)
      nfld = int(rfld)

      ifheat = .false.
      ifmhd  = .false.
      if (nfld.gt.0) then
         ifflow = .true.    
         if (nfld.gt.1)   ifheat = .true. ! fluid and heat
         if (nfld.ne.rfld) then
             ifmhd  = .true.              ! mhd
             nfld   = nfld+1
         endif
      else
         ifflow = .false.                 ! heat only, no mhd
         ifheat = .true.
         nfld   = abs(nfld)
      endif
      write(6,*) ifflow,ifheat,ifmhd,nfld,rfld
c-----------------------------------------------------------------------
      call scanout(string,'PARAMETERS FOLLOW',17,8,9)
      read(string,*) nparam
      call scanparam(string,string1,nparam,nfld,ifflow,ifheat,ifmhd,8,9)

      call scanout(string,'LOGICAL SWITCHES',16,8,9)
      read(string,*) nlogic
      call set_logical(ifflow,ifheat,8,9,nlogic)
 
c     call scanout(string,'MESH DATA',9,8,9)
c-----------------------------------------------------------------------
      
      nel  = 0
      nbox = 0
      do ibox=1,mbox 

         ifevenx = .false.
         ifeveny = .false.
         ifevenz = .false.
 
         call gets(boxcirc(ibox),1,iend,7)
         if (iend.eq.1) goto 99
         if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') 
     $                       call getr2(xc(ibox),yc(ibox),iend,7)
       

         if_multi_seg = .false.
         if (boxcirc(ibox).eq.'m'.or.boxcirc(ibox).eq.'M') 
     $      if_multi_seg = .true.
 
c----------------------------------------------------------------------
c       If cyl, call seperate routines..
         if (boxcirc(ibox).eq.'y'.or.boxcirc(ibox).eq.'Y') then
            if (iffo) then
                call cyl(if3d,ifflow,nfld,string,string1)
            else
                write(6,*) "Currently setup for ascii output only"
            endif
            call exitt
         endif
c----------------------------------------------------------------------
       
         if (if_multi_seg) then
            call get_multi_seg
     $           (nelxyz,x(0,ibox),y(0,ibox),z(0,ibox),maxx,if3d)
            nelx = nelxyz(1)
            nely = nelxyz(2)
            nelz = 1
            if (if3d) nelz = nelxyz(3)
            do ifld=1,nfld
               call getcv(cbc(1,ibox,ifld),3,6,iend,7)
               write(6,*) 'CBC:',(cbc(k,ibox,ifld),k=1,6),ifld,ibox
            enddo
         elseif (if3d) then

            call geti3(nelx,nely,nelz,iend,7)
 
            if (nelx.lt.0) ifevenx = .true.
            if (nely.lt.0) ifeveny = .true.
            if (nelz.lt.0) ifevenz = .true.
            nelx = abs(nelx)
            nely = abs(nely)
            nelz = abs(nelz)

            if(nelx*nely*nelz.gt.maxel) then
              write(6,*)
     $           'ABORT: increase maxel and recompile!',nelx,nely
              call exitt
            elseif (nelx.gt.maxx.or.nely.gt.maxx.or.nelz.gt.maxx) then
               write(6,*) 'ABORT, increase maxx and recompile',
     $                       nelx,nely,nelz,maxx
              call exitt
            endif
 
            nery = nely
            if (boxcirc(ibox).eq.'C') nery = 1
            if (iend.eq.1) goto 99
            ninbox = nelx*nely*nelz
            write(6,6) ninbox,nelx,nely,nelz,ibox,nfld
c----------------------------------------------------------------------
c           Generate xyz data
 
            if (ifevenx) then
               call getr3(x0,x1,ratio,iend,7)
               if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
                   if(x0.le.xc(ibox)) then
                      write(6,*) 'ABORT, x0 is equal to the center!'
                      call exitt
                   endif
               endif
               if(x1.le.x0) then
                  write(6,*) 'ABORT, must have x1 > x0 !'
                  write(6,*) 'x0: ',x0,' x1: ',x1
                  call exitt
               endif
               if (ratio.le.0) ratio=1.
               dx = (x1-x0)/nelx
               x(0,ibox) = 0.
               do k=1,nelx
                  x(k,ibox) = x(k-1,ibox) + dx
                  dx        = dx*ratio
               enddo
               scale = (x1-x0)/x(nelx,ibox)
               do k=0,nelx
                  x(k,ibox) = x0 + scale*x(k,ibox)
               enddo
            else
               call getrv(x(0,ibox),nelx+1,iend,7)
               if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
                   if(x(0,ibox).le.xc(ibox)) then
                      write(6,*) 'ABORT, x0 is equal to the center!'
                      call exitt
                   endif
               endif
            endif
 
            if (ifeveny) then
               call getr3(y0,y1,ratio,iend,7)
               if(y1.le.y0) then
                  write(6,*) 'ABORT, must have y1 > y0 !'
                  write(6,*) 'y0: ',y0,' y1: ',y1
                  call exitt
               endif
               if (ratio.le.0) ratio=1.
               dy = (y1-y0)/nely
               y(0,ibox) = 0.
               do k=1,nely
                  y(k,ibox) = y(k-1,ibox) + dy
                  dy        = dy*ratio
               enddo
               scale = (y1-y0)/y(nely,ibox)
               do k=0,nely
                  y(k,ibox) = y0 + scale*y(k,ibox)
               enddo
            else
               call getrv(y(0,ibox),nely+1,iend,7)
            endif
 
            if (ifevenz) then
               call getr3(z0,z1,ratio,iend,7)
               if(z1.le.z0) then
                  write(6,*) 'ABORT, must have z1 > z0 !'
                  write(6,*) 'z0: ',z0,' z1: ',z1
                  call exitt
               endif
               if (ratio.le.0) ratio=1.
               dz = (z1-z0)/nelz
               z(0,ibox) = 0.
               do k=1,nelz
                  z(k,ibox) = z(k-1,ibox) + dz
                  dz        = dz*ratio
               enddo
               scale = (z1-z0)/z(nelz,ibox)
               do k=0,nelz
                  z(k,ibox) = z0 + scale*z(k,ibox)
               enddo
            else
               call getrv(z(0,ibox),nelz+1,iend,7)
            endif
 
            do ifld=1,nfld
               call getcv(cbc(1,ibox,ifld),3,6,iend,7)
c              write(6,*) 'CBC:',(cbc(k,ibox,ifld),k=1,6),ifld,ibox
            enddo
         else
            nelz = 1
            call geti2(nelx,nely,iend,7)
 
            if (nelx.lt.0) ifevenx = .true.
            if (nely.lt.0) ifeveny = .true.
            nelx = abs(nelx)
            nely = abs(nely)

            if(nelx*nely*nelz.gt.maxel) then
              write(6,*) 'ABORT: increase maxel and recompile!'
              call exitt
            elseif (nelx.gt.maxx.or.nely.gt.maxx.or.nelz.gt.maxx) then
               write(6,*) 'ABORT, increase maxx and recompile',
     $                       nelx,nely,nelz,maxx
              call exitt
            endif
 
            nery = nely
            if (boxcirc(ibox).eq.'C') nery = 1
            if (iend.eq.1) goto 99
            ninbox = nelx*nely*nelz
            write(6,6) ninbox,nelx,nely,nery,ibox,nfld
 
            if (ifevenx) then
               call getr3(x0,x1,ratio,iend,7)
               if(x1.le.x0) then
                  write(6,*) 'ABORT, must have x1 > x0 !'
                  write(6,*) 'x0: ',x0,' x1: ',x1
                  call exitt
               endif
               if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
                   if(x0.le.xc(ibox)) then
                      write(6,*) 'ABORT, x0 is equal to the center!'
                      call exitt
                   endif
               endif
               if (ratio.le.0) ratio=1.
               dx = (x1-x0)/nelx
               x(0,ibox) = 0.
               do k=1,nelx
                  x(k,ibox) = x(k-1,ibox) + dx
                  dx        = dx*ratio
               enddo
               scale = (x1-x0)/x(nelx,ibox)
               do k=0,nelx
                  x(k,ibox) = x0 + scale*x(k,ibox)
               enddo
            else
               call getrv(x(0,ibox),nelx+1,iend,7)
               if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
                   if(x(0,ibox).le.xc(ibox)) then
                      write(6,*) 'ABORT, x0 is equal to the center!'
                      call exitt
                   endif
               endif
            endif
c           write(999,*) (x(i,1),i=0,nelx)

            if (ifeveny) then
               call getr3(y0,y1,ratio,iend,7)
               if(y1.le.y0) then
                  write(6,*) 'ABORT, must have y1 > y0 !'
                  write(6,*) 'y0: ',y0,' y1: ',y1
                  call exitt
               endif
               if (ratio.le.0) ratio=1.
               dy = (y1-y0)/nely
               y(0,ibox) = 0.
               do k=1,nely
                  y(k,ibox) = y(k-1,ibox) + dy
                  dy        = dy*ratio
               enddo
               scale = (y1-y0)/y(nely,ibox)
               do k=0,nely
                  y(k,ibox) = y0 + scale*y(k,ibox)
               enddo
            else
               call getrv(y(0,ibox),nely+1,iend,7)
            endif
c           write(998,*) (y(i,1),i=0,nely) 

 
            do ifld=1,nfld
               call getcv(cbc(1,ibox,ifld),3,4,iend,7)
            enddo
         endif
         nlx(ibox) = nelx
         nly(ibox) = nely
         nlz(ibox) = nelz
         nel = nel + nelx*nely*nelz
         nbox = nbox+1
      enddo
    6 format('Reading',i12,' =',3i9,' elements for box',i4,'.')
   99 continue

      call gets(boxcirc(ibox),1,iend,7)
      if (iend.ne.1) then
         write(6,*) 'Error: number of boxes >',mbox
         write(6,*) 'Increase mbox in genbox.f and'
         write(6,*) 'remake nek5_svn/trunk/tools/genbox'
         call exitt
      endif


      if(nel.gt.maxel) then
        write(6,*)
     $       'ABORT: increase maxel and recompile!',nel,maxel
             call exitt
      elseif (nelx.gt.maxx.or.nely.gt.maxx.or.nelz.gt.maxx) then
        write(6,*) 'ABORT, increase maxx and recompile',
     $       nelx,nely,nelz,maxx
        call exitt
      endif
 
      if(iffo) then
        write(6,*) 'Beginning construction of box.rea'
      else
        write(6,*) 'Beginning construction of box.rea/box.re2'
      endif
      write(6,*) nel,' elements will be created for ',nbox,' boxes.'
      write(6,*) 
c
c     Construct mesh
c
      if(iffo) then
        write(9,10) nel,ndim,nel
   10   format(i12,i3,i12,'           NEL,NDIM,NELV')
      else
        write(9,10) -nel,ndim,nel
        call blank(hdr,80)
        if(wdsize.ne.8) then       !8byte decision!!
          write(hdr,111) nel,ndim,nel
        else
          write(hdr,112) nel,ndim,nel
        endif
  111   format('#v001',i9,i3,i9,' hdr')
  112   format('#v002',i9,i3,i9,' hdr')
        call byte_write(hdr,20)   ! assumes byte_open() already issued
        call byte_write(test,1)   ! write the endian discriminator
      endif
 
      ncurv = 0

      call rzero(curve,8*nel)
 
c----------------------------------------------------------------------
c     OUTPUT mesh data
      if (if3d) then
        ie   = 0
        ilev = 0
        do ibox=1,nbox
          if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
c           CIRCLE (x = r, y = theta) .... specified in std. "box" way...
            do k=1,nlz(ibox)
              z1 = z(k-1,ibox)
              z2 = z(k  ,ibox)
              ia   = 0
c             ilev = ilev+1
              ilev = k
              do j=1,nly(ibox)
                t1 = pi180*y(j-1,ibox)
                t2 = pi180*y(j  ,ibox)
                do i=1,nlx(ibox)
                   r1 = x(i-1,ibox)
                   r2 = x(i  ,ibox)
                   ie = ie+1
                   ia = ia+1
                   ia = mod1(ia,52)
                   x1 = xc(ibox) + r1*cos(t1)
                   x2 = xc(ibox) + r2*cos(t1)
                   x3 = xc(ibox) + r2*cos(t2)
                   x4 = xc(ibox) + r1*cos(t2)
                   y1 = yc(ibox) + r1*sin(t1)
                   y2 = yc(ibox) + r2*sin(t1)
                   y3 = yc(ibox) + r2*sin(t2)
                   y4 = yc(ibox) + r1*sin(t2)
                   curve(2,ie) =  r2
                   curve(4,ie) = -r1
                   curve(6,ie) =  r2
                   curve(8,ie) = -r1
                   ncurv = ncurv+4
c                  write(9,11) ie,ilev,apt(ia),'0'
                   if(iffo) then 
                      write(9,11) ie,ilev,' ','0'
                      write(9,12) x1,x2,x3,x4
                      write(9,12) y1,y2,y3,y4
                      write(9,12) z1,z1,z1,z1
                      write(9,12) x1,x2,x3,x4
                      write(9,12) y1,y2,y3,y4
                      write(9,12) z2,z2,z2,z2
                   elseif(wdsize.eq.4) then
                    igroup = 0
                    call byte_write(igroup, 1)
                    buf(1)  = x1
                    buf(2)  = x2
                    buf(3)  = x3
                    buf(4)  = x4
                    buf(5)  = x1
                    buf(6)  = x2
                    buf(7)  = x3
                    buf(8)  = x4
                    buf(9)  = y1
                    buf(10) = y2
                    buf(11) = y3
                    buf(12) = y4
                    buf(13) = y1
                    buf(14) = y2
                    buf(15) = y3
                    buf(16) = y4
                    buf(17) = z1
                    buf(18) = z1
                    buf(19) = z1
                    buf(20) = z1
                    buf(21) = z2
                    buf(22) = z2
                    buf(23) = z2
                    buf(24) = z2
                    call byte_write(buf,24)
                   else
                    rgroup = 0.0
                    call byte_write(rgroup, 2)
                    buf2(1)  = x1
                    buf2(2)  = x2
                    buf2(3)  = x3
                    buf2(4)  = x4
                    buf2(5)  = x1
                    buf2(6)  = x2
                    buf2(7)  = x3
                    buf2(8)  = x4
                    buf2(9)  = y1
                    buf2(10) = y2
                    buf2(11) = y3
                    buf2(12) = y4
                    buf2(13) = y1
                    buf2(14) = y2
                    buf2(15) = y3
                    buf2(16) = y4
                    buf2(17) = z1
                    buf2(18) = z1
                    buf2(19) = z1
                    buf2(20) = z1
                    buf2(21) = z2
                    buf2(22) = z2
                    buf2(23) = z2
                    buf2(24) = z2
                    call byte_write(buf,48)
                   endif
                enddo
              enddo
            enddo

          else

           do k=1,nlz(ibox)
            z1 = z(k-1,ibox)
            z2 = z(k  ,ibox)
 
            ia   = 0
c           ilev = ilev+1
            ilev = k
            do j=1,nly(ibox)
               y1 = y(j-1,ibox)
               y2 = y(j  ,ibox)
 
               do i=1,nlx(ibox)
                  x1 = x(i-1,ibox)
                  x2 = x(i  ,ibox)
 
                  ie = ie+1
                  ia = ia+1
                  ia = mod1(ia,52)
                  if(iffo) then
                    write(9,11) ie,ilev,apt(ia),'0'
 
                    write(9,12) x1,x2,x2,x1
                    write(9,12) y1,y1,y2,y2
                    write(9,12) z1,z1,z1,z1
 
                    write(9,12) x1,x2,x2,x1
                    write(9,12) y1,y1,y2,y2
                    write(9,12) z2,z2,z2,z2
                  elseif(wdsize.eq.4) then
                    igroup = 0
                    call byte_write(igroup, 1)
                    buf(1)  = x1
                    buf(2)  = x2
                    buf(3)  = x2
                    buf(4)  = x1
                    buf(5)  = x1
                    buf(6)  = x2
                    buf(7)  = x2
                    buf(8)  = x1
                    buf(9)  = y1
                    buf(10) = y1
                    buf(11) = y2
                    buf(12) = y2
                    buf(13) = y1
                    buf(14) = y1
                    buf(15) = y2
                    buf(16) = y2
                    buf(17) = z1
                    buf(18) = z1
                    buf(19) = z1
                    buf(20) = z1
                    buf(21) = z2
                    buf(22) = z2
                    buf(23) = z2
                    buf(24) = z2
                    call byte_write(buf,24)
                  else
                    rgroup = 0.0
                    call byte_write(rgroup, 2)
                    buf2(1)  = x1
                    buf2(2)  = x2
                    buf2(3)  = x2
                    buf2(4)  = x1
                    buf2(5)  = x1
                    buf2(6)  = x2
                    buf2(7)  = x2
                    buf2(8)  = x1
                    buf2(9)  = y1
                    buf2(10) = y1
                    buf2(11) = y2
                    buf2(12) = y2
                    buf2(13) = y1
                    buf2(14) = y1
                    buf2(15) = y2
                    buf2(16) = y2
                    buf2(17) = z1
                    buf2(18) = z1
                    buf2(19) = z1
                    buf2(20) = z1
                    buf2(21) = z2
                    buf2(22) = z2
                    buf2(23) = z2
                    buf2(24) = z2
                    call byte_write(buf,48)
                  endif
               enddo
            enddo
          enddo
          endif
        enddo
   12   format(4g14.6)
   11   format(
     $'            ELEMENT',i12,' [',i5,a1,']  GROUP  ',a1)
      else
        ilev = 1
        ie=0
        ia=0
        one    = 1.
        pi     = 4.*atan(one)
        pi180  = pi/180.
        do ibox=1,nbox
           if (boxcirc(ibox).eq.'c'.or.boxcirc(ibox).eq.'C') then
c             CIRCLE (x = r, y = theta) .... specified in std. "box" way...
              do j=1,nly(ibox)
                 t1 = pi180*y(j-1,ibox)
                 t2 = pi180*y(j  ,ibox)
                 do i=1,nlx(ibox)
                    r1 = x(i-1,ibox)
                    r2 = x(i  ,ibox)
                    ie = ie+1
                    ia = ia+1
                    ia = mod1(ia,52)
                    x1 = xc(ibox) + r1*cos(t1)
                    x2 = xc(ibox) + r2*cos(t1)
                    x3 = xc(ibox) + r2*cos(t2)
                    x4 = xc(ibox) + r1*cos(t2)
                    y1 = yc(ibox) + r1*sin(t1)
                    y2 = yc(ibox) + r2*sin(t1)
                    y3 = yc(ibox) + r2*sin(t2)
                    y4 = yc(ibox) + r1*sin(t2)
                    curve(2,ie) =  r2
                    curve(4,ie) = -r1
                    if(iffo) then
                      write(9,11) ie,ilev,apt(ia),'0'
                      write(9,12) x1,x2,x3,x4
                      write(9,12) y1,y2,y3,y4
                    elseif(wdsize.eq.4) then
                      igroup = 0
                      call byte_write(igroup, 1)
                      buf(1) = x1
                      buf(2) = x2
                      buf(3) = x3
                      buf(4) = x4
                      buf(5) = y1
                      buf(6) = y2
                      buf(7) = y3
                      buf(8) = y4
                      call byte_write(buf,8)
                    else
                      rgroup = 0.0
                      call byte_write(rgroup, 2)
                      buf2(1) = x1
                      buf2(2) = x2
                      buf2(3) = x3
                      buf2(4) = x4
                      buf2(5) = y1
                      buf2(6) = y2
                      buf2(7) = y3
                      buf2(8) = y4
                      call byte_write(buf,16)
                    endif
                    ncurv = ncurv+2
                 enddo
              enddo
           elseif (boxcirc(ibox).eq.'C') then
c             CIRCLE (x = r, y = theta) .... specified in Cool fast way...
              dt = y(1,ibox)
              t2 = y(0,ibox)
              do j=1,nly(ibox)
                 t1 = t2
                 t2 = t1 + dt
                 p1 = pi180*t1
                 p2 = pi180*t2
                 write(6,*) 'p1:',p1,t1,t2
                 do i=1,nlx(ibox)
                    r1 = x(i-1,ibox)
                    r2 = x(i  ,ibox)
                    ie = ie+1
                    ia = ia+1
                    ia = mod1(ia,52)
                    x1 = xc(ibox) + r1*cos(p1)
                    x2 = xc(ibox) + r2*cos(p1)
                    x3 = xc(ibox) + r2*cos(p2)
                    x4 = xc(ibox) + r1*cos(p2)
                    y1 = yc(ibox) + r1*sin(p1)
                    y2 = yc(ibox) + r2*sin(p1)
                    y3 = yc(ibox) + r2*sin(p2)
                    y4 = yc(ibox) + r1*sin(p2)
                    curve(2,ie) =  r2
                    curve(4,ie) = -r1
                    if(iffo) then
                      write(9,11) ie,ilev,apt(ia),'0'
                      write(9,12) x1,x2,x3,x4
                      write(9,12) y1,y2,y3,y4
                    elseif(wdsize.eq.4) then 
                      igroup = 0
                      call byte_write(igroup, 1)
                      buf(1) = x1
                      buf(2) = x2
                      buf(3) = x3
                      buf(4) = x4
                      buf(5) = y1
                      buf(6) = y2
                      buf(7) = y3
                      buf(8) = y4
                      call byte_write(buf,8)
                    else
                      rgroup = 0.0
                      call byte_write(rgroup, 2)
                      buf2(1) = x1
                      buf2(2) = x2
                      buf2(3) = x3
                      buf2(4) = x4
                      buf2(5) = y1
                      buf2(6) = y2
                      buf2(7) = y3
                      buf2(8) = y4
                      call byte_write(buf,16)
                    endif
                    ncurv = ncurv+2
                 enddo
              enddo
           else
c             BOX
              do j=1,nly(ibox)
                 y1 = y(j-1,ibox)
                 y2 = y(j  ,ibox)
                 do i=1,nlx(ibox)
                    x1 = x(i-1,ibox)
                    x2 = x(i  ,ibox)
                    ie = ie+1
                    ia = ia+1
                    ia = mod1(ia,52)
                    if(iffo) then
                      write(9,11) ie,ilev,apt(ia),'0'
                      write(9,12) x1,x2,x2,x1
                      write(9,12) y1,y1,y2,y2
                    elseif(wdsize.eq.4) then
                      igroup = 0
                      call byte_write(igroup, 1)
                      buf(1) = x1
                      buf(2) = x2
                      buf(3) = x2
                      buf(4) = x1
                      buf(5) = y1
                      buf(6) = y1
                      buf(7) = y2
                      buf(8) = y2
                      call byte_write(buf,8)
                    else
                      rgroup = 0.0
                      call byte_write(rgroup, 2)
                      buf2(1) = x1
                      buf2(2) = x2
                      buf2(3) = x2
                      buf2(4) = x1
                      buf2(5) = y1
                      buf2(6) = y1
                      buf2(7) = y2
                      buf2(8) = y2
                      call byte_write(buf,16)
                    endif
                 enddo
              enddo
           endif
        enddo
      endif
 
c----------------------------------------------------------------------
c     output curve stuff and Boundary conditions
      maxedge = 4
      if(if3d) maxedge=8
      zero = 0.
      if(iffo) then
        write(9,28) ncurv
   28   format(
     $   '  ***** CURVED SIDE DATA *****',/,
     $      i12,' Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE')

        if (nel.lt.1000) then
           do ie=1,nel
              do iedge = 2,maxedge,2
                 if (curve(iedge,ie).ne.0) write(9,290) 
     $            iedge,ie,curve(iedge,ie),(zero,k=1,4),'C'
              enddo
           enddo
  290      format(i3,i3,5g14.6,1x,a1)
         elseif(nel.lt.1 000 000) then
           do ie=1,nel
              do iedge = 2,maxedge,2
                 if (curve(iedge,ie).ne.0) write(9,291) 
     $              iedge,ie,curve(iedge,ie),(zero,k=1,4),'C'
              enddo
           enddo
  291      format(i2,i6,5g14.6,1x,a1)
         else
           do ie=1,nel
              do iedge = 2,maxedge,2
                 if (curve(iedge,ie).ne.0) write(9,292) 
     $              iedge,ie,curve(iedge,ie),(zero,k=1,4),'C'
              enddo
           enddo
  292      format(i2,i12,5g14.6,1x,a1)
         endif
      elseif(wdsize.eq.4) then 
         call byte_write(ncurv,1)  
         do ie=1,nel
            do iedge = 2,maxedge,2
               if (curve(iedge,ie).ne.0) then
                  call icopy(buf(1),ie,1)
                  call icopy(buf(2),iedge,1)
                  buf(3) = curve(iedge,ie)
                  buf(4) = zero
                  buf(5) = zero
                  buf(6) = zero
                  buf(7) = zero
                  call chcopy(buf(8),'C',1)
                  call byte_write(buf,8)
               endif
            enddo
         enddo
      else
         rcurv=ncurv
         call byte_write(rcurv,2)  
         do ie=1,nel
            do iedge = 2,maxedge,2
               if (curve(iedge,ie).ne.0) then
                  buf2(1) = ie
                  buf2(2) = iedge
                  buf2(3) = curve(iedge,ie)
                  buf2(4) = zero
                  buf2(5) = zero
                  buf2(6) = zero
                  buf2(7) = zero
                  call chcopy(buf2(8),'C',1)
                  call byte_write(buf,16)
               endif
            enddo
         enddo
      endif

 
c----------------------------------------------------------------------
c     output Boundary conditions
      if(iffo) 
     &   write(9,31) 
   31    format('  ***** BOUNDARY CONDITIONS *****')
 
      icount = 0
      do ifld=1,nfld

         if (iffo) then
          if (ifflow) then
            if2 = ifld-2
            if (ifld.eq.1) write(9,41)
            if (ifld.eq.2) write(9,42)
            if (ifld.gt.2) write(9,43) if2
   41       format('  ***** FLUID   BOUNDARY CONDITIONS *****')
   42       format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   43       format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')
          else
            if2 = ifld-1
            if (ifld.eq.1) write(9,51)
            if (ifld.eq.1) write(9,52)
            if (ifld.gt.1) write(9,53) if2
   51       format('  ***** NO FLUID   BOUNDARY CONDITIONS *****')
   52       format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   53       format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')
          endif
         endif

         nbc = 0
         if (if3d) then
           do ipass=1,2

             ie = 0
             e0 = 0

             if (ipass.eq.2 .and. .not. iffo) then
                if(wdsize.eq.4) then
                   call byte_write(nbc,1)
                else
                   rbc=nbc
                   call byte_write(rbc,2)
                endif
             endif
             do ibx=1,nbox
              do iez=1,nlz(ibx)
              do iey=1,nly(ibx)
              do iex=1,nlx(ibx)

               call rzero(rbc1,5)
               call rzero(rbc2,5)
               call rzero(rbc3,5)
               call rzero(rbc4,5)
               call rzero(rbc5,5)
               call rzero(rbc6,5)
               call rzero8(rbc8,6)

               nelx = nlx(ibx)
               nely = nly(ibx)
               nelz = nlz(ibx)
               ilftz = mod1(iez-1+nelz,nelz)
               irgtz = mod1(iez+1+nelz,nelz)
               ilfty = mod1(iey-1+nely,nely)
               irgty = mod1(iey+1+nely,nely)
               ilftx = mod1(iex-1+nelx,nelx)
               irgtx = mod1(iex+1+nelx,nelx)

               rbc1(1) = nely*nelx*(iez-1)+nelx*(iey-1) + ilftx + e0
               rbc8(1) = nely*nelx*(iez-1)+nelx*(iey-1) + ilftx + e0
               rbc1(2) = eface(2)
               rbc2(1) = nely*nelx*(iez-1)+nelx*(iey-1) + irgtx + e0
               rbc8(2) = nely*nelx*(iez-1)+nelx*(iey-1) + irgtx + e0
               rbc2(2) = eface(1)

               rbc3(1) = nely*nelx*(iez-1)+nelx*(ilfty-1) + iex + e0
               rbc8(3) = nely*nelx*(iez-1)+nelx*(ilfty-1) + iex + e0
               rbc3(2) = eface(4)
               rbc4(1) = nely*nelx*(iez-1)+nelx*(irgty-1) + iex + e0
               rbc8(4) = nely*nelx*(iez-1)+nelx*(irgty-1) + iex + e0
               rbc4(2) = eface(3)

               rbc5(1) = nely*nelx*(ilftz-1)+nelx*(iey-1) + iex + e0
               rbc8(5) = nely*nelx*(ilftz-1)+nelx*(iey-1) + iex + e0
               rbc5(2) = eface(6)
               rbc6(1) = nely*nelx*(irgtz-1)+nelx*(iey-1) + iex + e0
               rbc8(6) = nely*nelx*(irgtz-1)+nelx*(iey-1) + iex + e0
               rbc6(2) = eface(5)

               if (iex.eq.1) then
                  cbc1=cbc(1,ibx,ifld)
                  if (cbc1.ne.'P  ')  then
                      call rzero(rbc1,5)
                      rbc8(1) = 0.0
                  endif
                  if (cbc1.ne.'E  ')  nbc = nbc + 1
               else
                  cbc1='E  '
               endif
 
               if (iex.eq.nelx) then
                  cbc2=cbc(2,ibx,ifld)
                  if (cbc2.ne.'P  ') then 
                      call rzero(rbc2,5)
                      rbc8(2) = 0.0
                  endif
                  if (cbc2.ne.'E  ')  nbc = nbc + 1
               else
                  cbc2='E  '
               endif
 
               if (iey.eq.1) then
                  cbc3=cbc(3,ibx,ifld)
                  if (cbc3.ne.'P  ') then
                      call rzero(rbc3,5)
                      rbc8(3) = 0.0
                  endif
                  if (cbc3.ne.'E  ')  nbc = nbc + 1
               else
                  cbc3='E  '
               endif
 
               if (iey.eq.nely) then
                  cbc4=cbc(4,ibx,ifld)
                  if (cbc4.ne.'P  ') then 
                      call rzero(rbc4,5)
                      rbc8(4) = 0.0
                  endif
                  if (cbc4.ne.'E  ')  nbc = nbc + 1
               else
                  cbc4='E  '
               endif
 
               if (iez.eq.1) then
                  cbc5=cbc(5,ibx,ifld)
                  if (cbc5.ne.'P  ') then 
                      call rzero(rbc5,5)
                      rbc8(5) = 0.0
                  endif
                  if (cbc5.ne.'E  ')  nbc = nbc + 1
               else
                  cbc5='E  '
               endif
 
               if (iez.eq.nelz) then
                  cbc6=cbc(6,ibx,ifld)
                  if (cbc6.ne.'P  ') then
                     call rzero(rbc6,5)
                     rbc8(6) = 0.0
                  endif
                  if (cbc6.ne.'E  ')  nbc = nbc + 1
               else
                  cbc6='E  '
               endif
c
c              output bc's in preproc. notation
c
               ie = ie+1
               if(iffo .and. ipass.eq.1) then
                 if (nel.lt.1 000) then
                    write(9,20) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                    write(9,20) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                    write(9,20) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                    write(9,20) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                    write(9,20) cbc5,ie,eface(5),(rbc5(j),j=1,5)
                    write(9,20) cbc6,ie,eface(6),(rbc6(j),j=1,5)
                 elseif (nel.lt.100 000) then
                    write(9,21) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                    write(9,21) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                    write(9,21) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                    write(9,21) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                    write(9,21) cbc5,ie,eface(5),(rbc5(j),j=1,5)
                    write(9,21) cbc6,ie,eface(6),(rbc6(j),j=1,5)
                 elseif (nel.lt.1 000 000) then
                    write(9,22) cbc3,ie,(rbc3(j),j=1,5)
                    write(9,22) cbc2,ie,(rbc2(j),j=1,5)
                    write(9,22) cbc4,ie,(rbc4(j),j=1,5)
                    write(9,22) cbc1,ie,(rbc1(j),j=1,5)
                    write(9,22) cbc5,ie,(rbc5(j),j=1,5)
                    write(9,22) cbc6,ie,(rbc6(j),j=1,5)
                 else        
                    write(9,23) cbc3,ie,rbc8(3),(rbc3(j),j=2,5)
                    write(9,23) cbc2,ie,rbc8(2),(rbc2(j),j=2,5)
                    write(9,23) cbc4,ie,rbc8(4),(rbc4(j),j=2,5)
                    write(9,23) cbc1,ie,rbc8(1),(rbc1(j),j=2,5)
                    write(9,23) cbc5,ie,rbc8(5),(rbc5(j),j=2,5) 
                    write(9,23) cbc6,ie,rbc8(6),(rbc6(j),j=2,5) 
                 endif

   20            format(1x,a3,2i3,5g14.6)
   21            format(1x,a3,i5,i1,5g14.6)
   22            format(1x,a3,i6,5g14.6)
   23            format(1x,a3,i12,5g18.11)

               elseif (.not. iffo) then

                 if(ipass.eq.2) then
                   do ii = 1,6
                     ibc(ii) = rbc8(ii) 
                   enddo
                   call blank(buf(1),30*4)

                   if(wdsize.eq.4) then
                     call icopy(buf(1),ie,1)
                     if(cbc3.ne.'E  ') then
                       call icopy(buf(2),eface(3),1)
                       call copy48(buf(3),rbc3,5)
                       call chcopy(buf(8),cbc3,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(3),1)
                       call byte_write(buf,8)
                       icount = icount+1
                     endif
                     if(cbc2.ne.'E  ') then 
                       call icopy(buf(2),eface(2),1)
                       call copy48(buf(3),rbc2,5)
                       call chcopy(buf(8),cbc2,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(2),1)
                       call byte_write(buf,8)
                       icount = icount+1
                     endif
                     if(cbc4.ne.'E  ') then 
                       call icopy(buf(2),eface(4),1)
                       call copy48(buf(3),rbc4,5)
                       call chcopy(buf(8),cbc4,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(4),1)
                       call byte_write(buf,8)
                       icount = icount+1
                     endif
                     if(cbc1.ne.'E  ') then 
                       call icopy(buf(2),eface(1),1)
                       call copy48(buf(3),rbc1,5)
                       call chcopy(buf(8),cbc1,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(1),1)
                       call byte_write(buf,8)
                       icount = icount+1
                     endif
                     if(cbc5.ne.'E  ') then 
                      call icopy(buf(2),eface(5),1)
                      call copy48(buf(3),rbc5,5)
                      call chcopy(buf(8),cbc5,3)
                      if(nel.ge.1000000) call icopy(buf(3),ibc(5),1)
                      call byte_write(buf,8)
                      icount = icount+1
                     endif
                     if(cbc6.ne.'E  ') then 
                      call icopy(buf(2),eface(6),1)
                      call copy48(buf(3),rbc6,5)
                      call chcopy(buf(8),cbc6,3)
                      if(nel.ge.1000000) call icopy(buf(3),ibc(6),1)
                      call byte_write(buf,8)
                      icount = icount+1
                     endif
                   else
                     buf2(1)=ie
                     if(cbc3.ne.'E  ') then
                       buf2(2)=eface(3)
                       call copy(buf2(3),rbc3,5)
                       call chcopy(buf2(8),cbc3,3)
                       if(nel.ge.1000000) buf2(3)=ibc(3)
                       call byte_write(buf,16)
                       icount = icount+1
                     endif
                     if(cbc2.ne.'E  ') then 
                        buf2(2)=eface(2)
                        call copy  (buf2(3),rbc2,5)
                        call chcopy(buf2(8),cbc2,3)
                        if(nel.ge.1000000) buf(3)=ibc(2)
                        call byte_write(buf,16)
                        icount = icount+1
                     endif
                     if(cbc4.ne.'E  ') then 
                       buf2(2)=eface(4)
                       call copy  (buf2(3),rbc4,5)
                       call chcopy(buf2(8),cbc4,3)
                       if(nel.ge.1000000) buf2(3)=ibc(4)
                       call byte_write(buf,16)
                       icount = icount+1
                     endif
                     if(cbc1.ne.'E  ') then 
                       buf2(2)=eface(1)
                       call copy  (buf2(3),rbc1,5)
                       call chcopy(buf2(8),cbc1,3)
                       if(nel.ge.1000000) buf2(3)=ibc(1)
                       call byte_write(buf,16)
                       icount = icount+1
                     endif
                     if(cbc5.ne.'E  ') then 
                      buf2(2)=eface(5)
                      call copy  (buf2(3),rbc5,5)
                      call chcopy(buf2(8),cbc5,3)
                      if(nel.ge.1000000) buf2(3)=ibc(5)
                      call byte_write(buf,16)
                      icount = icount+1
                     endif
                     if(cbc6.ne.'E  ') then 
                      buf2(2)=eface(6)
                      call copy  (buf2(3),rbc6,5)
                      call chcopy(buf2(8),cbc6,3)
                      if(nel.ge.1000000) buf2(3)=ibc(6)
                      call byte_write(buf,16)
                      icount = icount+1
                     endif
                   endif
                 endif
               endif
              enddo
              enddo
              enddo
              e0 = e0+nlx(ibx)*nly(ibx)*nlz(ibx)
             enddo
           enddo
         else
           do ipass=1,2

             e0 = 0
             ie = 0
             if(ipass.eq.2 .and. .not. iffo) then
                if(wdsize.eq.4) then
                   call byte_write(nbc,1)
                else
                   rbc=nbc
                   call byte_write(rbc,2)
                endif
             endif
             do ibx=1,nbox
              do iey=1,nly(ibx)
              do iex=1,nlx(ibx)

               call rzero(rbc1,5)
               call rzero(rbc2,5)
               call rzero(rbc3,5)
               call rzero(rbc4,5)
               call rzero8(rbc8,4)

               nelx = nlx(ibx)
               nely = nly(ibx)
               ilfty = mod1(iey-1+nely,nely)
               irgty = mod1(iey+1+nely,nely)
               ilftx = mod1(iex-1+nelx,nelx)
               irgtx = mod1(iex+1+nelx,nelx)

               rbc1(1) = nelx*(iey-1) + ilftx + e0
               rbc8(1) = nelx*(iey-1) + ilftx + e0
               rbc1(2) = eface(2)
               rbc2(1) = nelx*(iey-1) + irgtx + e0
               rbc8(2) = nelx*(iey-1) + irgtx + e0
               rbc2(2) = eface(1)

               rbc3(1) = nelx*(ilfty-1) + iex + e0
               rbc8(3) = nelx*(ilfty-1) + iex + e0
               rbc3(2) = eface(4)
               rbc4(1) = nelx*(irgty-1) + iex + e0
               rbc8(4) = nelx*(irgty-1) + iex + e0
               rbc4(2) = eface(3)

               if (iex.eq.1) then
                  cbc1=cbc(1,ibx,ifld)
                  if (cbc1.ne.'P  ') then 
                     call rzero(rbc1,5)
                     rbc8(1) = 0.0
                  endif
                  if (cbc1.ne.'E  ') nbc = nbc + 1
               else
                  cbc1='E  '
               endif
 
               if (iex.eq.nelx) then
                  cbc2=cbc(2,ibx,ifld)
                  if (cbc2.ne.'P  ') then 
                     call rzero(rbc2,5)
                     rbc8(2) = 0.0
                  endif
                  if (cbc2.ne.'E  ') nbc = nbc + 1
               else
                  cbc2='E  '
               endif
 
               if (iey.eq.1) then
                  cbc3=cbc(3,ibx,ifld)
                  if (cbc3.ne.'P  ') then 
                     call rzero(rbc3,5)
                     rbc8(3) = 0.0
                  endif
                  if (cbc5.ne.'E  ') nbc = nbc + 1
               else
                  cbc3='E  '
               endif
 
               if (iey.eq.nely) then
                  cbc4=cbc(4,ibx,ifld)
                  if (cbc4.ne.'P  ') then 
                     call rzero(rbc4,5)
                     rbc8(4) = 0.0
                  endif
                  if (cbc4.ne.'E  ') nbc = nbc + 1
               else
                  cbc4='E  '
               endif
c
c              output bc's in preproc. notation
c
               ie = ie+1
               if(iffo .and. ipass.eq.1) then
                 if (nel.lt.1 000) then
                    write(9,20) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                    write(9,20) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                    write(9,20) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                    write(9,20) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                 elseif (nel.lt.100 000) then
                    write(9,21) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                    write(9,21) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                    write(9,21) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                    write(9,21) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                 elseif (nel.lt.1 000 000) then
                    write(9,22) cbc3,ie,(rbc3(j),j=1,5)
                    write(9,22) cbc2,ie,(rbc2(j),j=1,5)
                    write(9,22) cbc4,ie,(rbc4(j),j=1,5)
                    write(9,22) cbc1,ie,(rbc1(j),j=1,5)
                 else
                    write(9,23) cbc3,ie,rbc8(3),(rbc3(j),j=2,5)
                    write(9,23) cbc2,ie,rbc8(2),(rbc2(j),j=2,5)
                    write(9,23) cbc4,ie,rbc8(4),(rbc4(j),j=2,5)
                    write(9,23) cbc1,ie,rbc8(1),(rbc1(j),j=2,5)
                 endif
               elseif (.not. iffo) then
                 if(ipass.eq.2) then
                   do ii = 1,6
                      ibc(ii) = rbc8(ii) 
                   enddo
                   call blank(buf(1),30*4)
                   if(wdsize.eq.4) then
                     call icopy(buf(1),ie,1)
 
                     if(cbc3.ne.'E  ') then 
                       call icopy(buf(2),eface(3),1)
                       call copy48(buf(3),rbc3,5)
                       call chcopy(buf(8),cbc3,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(3),1)
                       call byte_write(buf,8)
                     endif
 
                     if(cbc2.ne.'E  ') then 
                       call icopy(buf(2),eface(2),1)
                       call copy48(buf(3),rbc2,5)
                       call chcopy(buf(8),cbc2,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(2),1)
                       call byte_write(buf,8)
                     endif
 
                     if(cbc4.ne.'E  ') then 
                       call icopy(buf(2),eface(4),1)
                       call copy48(buf(3),rbc4,5)
                       call chcopy(buf(8),cbc4,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(4),1)
                       call byte_write(buf,8)
                     endif
 
                     if(cbc1.ne.'E  ') then 
                       call icopy(buf(2),eface(1),1)
                       call copy48(buf(3),rbc1,5)
                       call chcopy(buf(8),cbc1,3)
                       if(nel.ge.1000000) call icopy(buf(3),ibc(1),1)
                       call byte_write(buf,8)
                     endif 
                   else
                     buf2(1)=ie
 
                     if(cbc3.ne.'E  ') then 
                       buf2(2)=eface(3)
                       call copy(buf2(3),rbc3,5)
                       call chcopy(buf2(8),cbc3,3)
                       if(nel.ge.1000000) buf2(3)=ibc(3)
                       call byte_write(buf,16)
                     endif
 
                     if(cbc2.ne.'E  ') then 
                       buf2(2)=eface(2)
                       call copy(buf2(3),rbc2,5)
                       call chcopy(buf2(8),cbc2,3)
                       if(nel.ge.1000000) buf2(3)=ibc(2)
                       call byte_write(buf,16)
                     endif
 
                     if(cbc4.ne.'E  ') then 
                       buf2(2)=eface(4)
                       call copy(buf2(3),rbc4,5)
                       call chcopy(buf2(8),cbc4,3)
                       if(nel.ge.1000000) buf2(3)=ibc(4)
                       call byte_write(buf,16)
                     endif
 
                     if(cbc1.ne.'E  ') then 
                       buf2(2)=eface(1)
                       call copy(buf2(3),rbc1,5)
                       call chcopy(buf2(8),cbc1,3)
                       if(nel.ge.1000000) buf2(3)=ibc(1)
                       call byte_write(buf,16)
                     endif
                   endif
                 endif
               endif
              enddo
              enddo
              e0 = e0+nlx(ibx)*nly(ibx)*nlz(ibx)
             enddo
           enddo
         endif
      enddo
c
c     Scan through .rea file until end of bcs
c
c     if (ifflow) then
c        if (nfld.eq.1) call nekscan(string,'THERMAL BOUNDARY',16,8)
c        if (nfld.ge.2) call nekscan(string,'PRESOLVE',8,8)
c     else !  heat only
c        call nekscan(string,'PRESOLVE',8,8)
c     endif

c     lout = ltrunc(string1,132)
c     write (9,81) (string1(j),j=1,lout)
c  81 format(132a1)

      if(iffo.and.ifflow) then
        if (nfld.eq.1) write(9,*) 
     &     ' ***** NO THERMAL BOUNDARY CONDITIONS *****'
      endif
 
c     Scan through .rea file until end of bcs
 
      call nekscan(string,'RESTART',7,8)
      lout = ltrunc(string1,132)
      write (9,81) (string1(j),j=1,lout)
   81 format(132a1)

      call scanout(string,'xxxx',4,8,9)
      if(.not.iffo)  call byte_close()
c
c     Scan through and output .rea file until end of file
c
      close(8)
      close(9)

      end
c-----------------------------------------------------------------------
      subroutine scanout(string,input,len,infile,outfile)
 
c     scan through infile until input is found and output
c     all lines save that containing input to "outfile"
c
      character*132 string
      character*1 input(1)
      integer infile,outfile,len
 
      character*1 string1(132)
 
      do line=1,10000000
         call blank(string,132)
         read (infile ,132,end=100,err=100) string
         call ccopy(string1,string,132)
         lout = ltrunc(string1,132)
         write (outfile,81) (string1(j),j=1,lout)
         if (indx1(string,input,len).ne.0) return
      enddo
  132 format(a132)
   81 format(132a1)
 
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine nekscan(string,input,len,infile)
c
c     scan through infile until input is found 
c
      character*132 string
      character*1 input(1)
      integer infile,outfile
 
      do line=1,10000000
         call blank(string,132)
         read (infile ,132,end=100,err=100) string
         if (indx1(string,input,len).ne.0) return
      enddo
  132 format(a132)
 
  100 continue
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE BLANK(A,N)
      CHARACTER*1 A(1)
      CHARACTER*1 BLNK
      SAVE        BLNK
      DATA        BLNK /' '/
      DO 10 I=1,N
         A(I)=BLNK
   10 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION LTRUNC(STRING,L)
      CHARACTER*1 STRING(L)
      CHARACTER*1   BLNK
      DATA BLNK/' '/
 
      DO 100 I=L,1,-1
         L1=I
         IF (STRING(I).NE.BLNK) GOTO 200
  100 CONTINUE
      L1=0
  200 CONTINUE
      LTRUNC=L1
      RETURN
      END
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*132 S1,S2
 
      N1=132-L2+1
      INDX1=0
      IF (N1.LT.1) RETURN
 
      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            RETURN
         ENDIF
  100 CONTINUE
 
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CCOPY(A,B,N)
      CHARACTER*1 A(1), B(1)
      DO 100 I = 1, N
 100     A(I) = B(I)
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE RZERO8(A,N)
      real*8  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE RZERO(A,N)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION MOD1(I,N)
C
C     Yields MOD(I,N) with the exception that if I=K*N, result is N.
C
      MOD1=0
      IF (I.EQ.0) RETURN
      IF (N.EQ.0) THEN
         WRITE(6,*)
     $  'WARNING:  Attempt to take MOD(I,0) in FUNCTION MOD1.'
         RETURN
      ENDIF
      II = I+N-1
      MOD1 = MOD(II,N)+1
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine scannocom(iend,infile)
c
c     scan through infile until "no comment" is found
c
      character*132 string
      character*1  string1(132)
      equivalence (string1,string)
 
      character*1 comment
      save        comment
      data        comment /'#'/
 
      iend = 0
      do line=1,10000000
         call blank(string,132)
         read (infile ,132,end=100,err=100) string
 
         if   (indx1(string,comment,1).ne.1) then
c             write(*,*) line
c             write(6,132) string
              open(unit=99,file='box.tmp')
              len = ltrunc(string,132)
              write(99,81) (string1(k),k=1,len)
              write(6,81) (string1(k),k=1,len)
              close(unit=99)
              return
         else
              i1 = indx1(string,comment,1)
              write(6,*) i1,' i1', comment
         endif
 
      enddo
  132 format(a132)
   81 format(132a1)
 
  100 continue
      iend = 1
      return
      end
c-----------------------------------------------------------------------
      subroutine geti1(i,iend,io)
c
c     Get an integer from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) i
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine geti2(i1,i2,iend,io)
c
c     Get an integer from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) i1,i2
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine geti3(i1,i2,i3,iend,io)
c
c     Get an integer from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) i1,i2,i3
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getr3(r1,r2,r3,iend,io)

c     Get three reals from first uncommented line

      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) r1,r2,r3
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getr2(r1,r2,iend,io)

c     Get two reals from first uncommented line

      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) r1,r2
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getr1(r1,iend,io)

c     Get two reals from first uncommented line

      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) r1
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getrv(r,n,iend,io)
      real r(n)
c
c     Get reals from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return

      call cfill(r,1e+30,n)

      istart=1
      do ipass=1,n
         open(unit=99,file='box.tmp')
         read(99,*,end=1,iostat=istat) (r(k),k=istart,n)
   1     close(unit=99)

         icounter = 0  !  check how many points we have already read
         do i=1,n
            if(r(i).ne.1e+30) icounter = icounter + 1
         enddo
 
         if (icounter.lt.n) then
            call scannocom(iend,io)
            istart=icounter+1
         else
            return
         endif
      enddo

      write(6,*) 'error in getrv:', icounter,n
      return

c    1 continue
c      close(unit=99)
c      write(6,*) 'this is k:',k,n
c      read( 7,*,end=2) (r(j),j=k,n)
c      return
c    2 continue
c      iend=1
c      return
      end
c-----------------------------------------------------------------------
      subroutine getcv0(c,m,n,iend,io)
      character*1 c(m,n)
      character*1 adum(132)
c
c     Get character strings, with no seperator, from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,1,end=2) (adum(k),k=1,132)
    1 format(132a1)
    2 continue
 
      i = 0
      do l=1,n
         do k=1,m
            i = i+1
            c(k,l) = adum(i)
         enddo
      enddo
      do j=1,n
         write(6,3) m,n,(c(i,j),i=1,m)
    3    format(2i4,'getcv0:',132a1)
      enddo
 
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getcv(c,m,n,iend,io)
      character*1 c(m,n)
      character*1 adum(132)
c
c     Get character strings, with single seperator, from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,1,end=2) (adum(k),k=1,132)
    1 format(132a1)
    2 continue
 
      i = 0
      do l=1,n
         do k=1,m
            i = i+1
            if(adum(i).eq.',') then
               write(6,*) "ABORT, each BC must be 3 characters long"
               write(6,*) "BC  : ",l
               call exitt
            endif
            c(k,l) = adum(i)
         enddo
 
c        bump pointer for "," in input string
         i = i+1
      enddo
 
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine gets(c,n,iend,io)
      character*1 c(n)
c
c     Get character string from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,1) (c(k),k=1,n)
    1 format(132a1)
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine cfill(a,b,n)
      DIMENSION  A(1)
 
      DO I = 1, N
        A(I) = B
      ENDDO
      
      return
      end
c-----------------------------------------------------------------------
      subroutine icopy(a,b,n)
      integer a(1), b(1)
      do 100 i = 1, n
 100     a(i) = b(i)
      return
      end
c-----------------------------------------------------------------------
      subroutine chcopy(a,b,n)
      character*1 a(1), b(1)
      do 100 i = 1, n
 100     a(i) = b(i)
      return
      end
c-----------------------------------------------------------------------
      subroutine copy(a,b,n)
      real a(1), b(1)
      do 100 i = 1, n
 100     a(i) = b(i)
      return
      end
c-----------------------------------------------------------------------
      subroutine copy48(a,b,n)  !copy a *8 into a *4
      real*4 a(1)
      real*8 b(1)
      do i = 1, n
         a(i) = b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine exitt

      call exit 
 
      return
      end
c-----------------------------------------------------------------------
      subroutine getiv(r,n,iend,io)
      integer r(n)
      parameter (ibig=99999999)
c
c     Get integers from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return

c     serious hack for f90
      call ifill(r,ibig,n)

      open(unit=99,file='box.tmp')
      read(99,*,end=1) (r(k),k=1,n)
      close(unit=99)
      return
    1 continue
      close(unit=99)
      do k=1,n
         if (r(k).eq.ibig) goto 15
      enddo
   15 continue
      write(6,*) 'this is k:',k,n
      read( 7,*,end=2) (r(j),j=k,n)
      return
    2 continue
      iend=1
      return
      end
c-----------------------------------------------------------------------
      subroutine get_multi_seg(nelxyz,x,y,z,m,if3d)
 
      integer nelxyz(3)
      real x(0:m),y(0:m),z(0:m)
      logical if3d
 
      integer nels (1000)
      real    gains(1000)
      real    xs   (0:1000)
c
c     This routine allows the user to input a complex sequence of
c     segments for each of the x, y and z directions, so that
c     genbox will generate the desired tensor product array.
c
c     Input format for each coordinat is
c
c     nsegs
c     nel_1    nel_2 ...  nel_nsegs
c     gain_1  gain_2 ... gain_nsegs
c     x_0, x_1, x_2, ...    x_nsegs
c
c     The output will be:
c
c     x(0)...x(nel_1)...x_(nel_2)...x(nel_segs),
c
c     where the segment between x(e-1) and x(e) is filled with
c     a distribution determined by gain_e.   Uniform spacing
c     corresponds to gain_e = 1, otherwise, a geometric sequence
c     is generated, with dx_i+1 = dx_i * gain_e
c
      ndim = 2
      if (if3d) ndim=3
 
      do id=1,ndim
 
         call geti1(nseg,iend,7)
         call getiv(nels ,nseg  ,iend,7)
         call getrv(xs   ,nseg+1,iend,7)
         call getrv(gains,nseg  ,iend,7)
         write(6,*) 'NSEG: ',id,nseg,(nels(k),k=1,nseg)
 
         nelxyz(id) = ilsum(nels,nseg)
         if (nelxyz(id).gt.m) then
            write(6,*) 'NEL too large:',nelxyz(id),m,id
            write(6,*) 'ABORT in get_multi_seg.'
            call exitt
         endif
 
         k = 0
         do is=1,nseg
            nel = nels(is)
            x0  = xs(is-1)
            x1  = xs(is)
            if (id.eq.1) call geometric_x(x(k),nel,x0,x1,gains(is))
            if (id.eq.2) call geometric_x(y(k),nel,x0,x1,gains(is))
            if (id.eq.3) call geometric_x(z(k),nel,x0,x1,gains(is))
            k = k + nel
         enddo
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine geometric_x(x,n,x0,x1,gain)
 
      real x(0:n)
 
      x(0) = 0.
      dx = (x1-x0)/n
      do i=1,n
         x(i) = x(i-1) + dx
         dx   = dx*gain
      enddo
 
      scale = (x1-x0)/(x(n)-x(0))
      do i=0,n
         x(i) = x0 + scale*x(i)
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine overflow_chk(n_req,n_avail,var,sub)
c
c     Check for buffer overflow
c
      character*3 var
      character*6 sub
 
      if (n_req.gt.n_avail) then
         write(6,9) n_req,n_avail,var,sub
    9    format(' ERROR: requested array space (',i12
     $         ,') exceeds allocated amount (',i12,').'
     $         ,/,' ABORTING.',3x,a3,2x,a6,' overflow_chk')
         call exit
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine cmult(x,a,n)
      real x(n),a
      do i=1,n
         x(i) = a*x(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cadd(x,a,n)
      real x(n),a
      do i=1,n
         x(i) = x(i) + a
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine get_xyz_distribution (x,nelx)
      real x(0:2)
      if (nelx.ge.0) return ! else, uniform or geometric spacing
 
      x0 = x(0)
      x1 = x(1)
      r  = x(2)
      write(6,*) 'x0:',x0,x1,r,nelx
      if(x1.le.x0) then
        write(6,*) 'ABORT, must have x1 > x0 !'
        call exitt
      endif

      nelx = abs(nelx)
      if (r.eq.1.0) then
         dx = (x1-x0)/nelx
         do i=1,nelx
            x(i) = x0 + i*dx
         enddo
         x(nelx) = x1
      else
         dx = (x1-x0)/nelx
         x(0) = 0.
         do i=1,nelx
            x(i) = x(i-1) + dx
            dx = r*dx
         enddo
         scale = (x1-x0)/x(nelx)
         call cmult(x,scale,nelx+1)
         call cadd (x,x0   ,nelx+1)
         x(nelx) = x1
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat6(a,m,n,name6,ilabel)
      real a(m,n)
      character*6 name6
 
      n10=min(n,10)
      do i=1,m
         write(6,6) ilabel,name6,(a(i,j),j=1,n10)
    6    format(i5,1x,a6,1p10e11.4)
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine cyl(if3d,ifflow,nfld,string,string1)

      include 'SIZE'

      character*132 string
      character*1  string1(132)

      integer nlx(mbox),nly(mbox),nlz(mbox)
      real x(0:maxx,mbox),y(0:maxx,mbox),z(0:maxx,mbox)
      real xc(mbox),yc(mbox),zc(mbox)
      real curve(8,maxel)
      character*3 cbc(6,mbox,3)
      character*1 boxcirc(mbox)
 
      common /genbr/ x,y,z,xc,yc,zc,curve
      common /genbc/ cbc
 
      integer nels(3)
      real rad(0:maxx),tht(0:maxx)
      character*1 cob(0:maxx),ccurve(8,maxx)
 
      logical if3d,ifflow
      integer e
 
      call cyl_box(x,y,z,nels,cob,rad,tht,cbc,curve,ccurve
     $                  ,maxx,if3d,nfld)
 
      nelx = abs(nels(1))
      nely = abs(nels(2))
      nelz = abs(nels(3))
      nel  = nelx*nely
      if (if3d) nel  = nelx*nely*nelz
 
      call out_xy_box    (x,y,z,nels,maxx,if3d)
      call out_xy_curves (curve,ccurve,nels,if3d)
      call out_tens_bcs  (cbc,nelx,nely,nelz,1,nfld,if3d,ifflow)
c     call out_tens_bcs  (cbc,nlx,nly,nlz,nbox,nfld,if3d) ! > 1 box, later
 
      call nekscan(string,'RESTART',7,8)
      lout = ltrunc(string1,132)
      write (9,81) (string1(j),j=1,lout)
   81 format(132a1)

      call scanout(string,'xxxx',4,8,9)
c      if(.not.iffo)  call byte_close()
      close(unit=8)
      close(unit=9)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine cyl_box(x,y,z,nels,cob,rad,tht,cbc,curve,ccurve
     $                  ,maxx,if3d,nfld)
c
c     Input is similar to genbox.
c
c     A standard input would be of the form:
c
c CYLBOX
c 2 -8 -2  NELX,Y,Z:  in r, theta, and z ( < 0 ==> regular distribution)
c c  ,o  ,b  inner-most cylinder, next "octagon", last "box" (Cartesian)
c .5 .75 1.0     r0, r1, r2.
c 0  1.0         -- fraction of circular arc (result is multiplied by 2pi)
c 0  2.5  1.5    -- z0, z1, geometric ratio
c bcx,bcx,bcy,bcy,bcz,bcz   Usual bcs
 
      integer nels(3)
      real x(4,maxx*maxx),y(4,maxx*maxx)
      real rad(0:maxx),tht(0:maxx),z(0:maxx),curve(5,8,maxx)
      character*3 cbc(6,nfld)
      character*1 cob(0:maxx),ccurve(8,maxx)
      logical if3d
 
      real xy0(2)
      integer e
 
      integer ipf2e(4)
      save    ipf2e
      data    ipf2e / 1,2,4,3 /
 
      call get_cyl_box(xy0,nels,cob,rad,tht,z,cbc,maxx,if3d,nfld)
 
      nelx = nels(1)
      nely = nels(2)
      nelz = nels(3)
 
      call get_xyz_distribution (rad,nelx) ! Get distribution if nelx < 0, etc.
      call get_xyz_distribution (tht,nely)
      call get_xyz_distribution (z  ,nelz)
 
      nelx = abs(nelx)
      nely = abs(nely)
      nelz = abs(nelz)
 
      one    = 1.
      twopi  = 8.*atan(one)
c     call outmat6(rad,1,nelx+1,'radius',1)
c     call outmat6(tht,1,nely+1,'theta ',1)
c
      neln = nelx*nely
      if (if3d) neln = nelx*nely*nelz
 
      write(6,*) 'Beginning construction of cylinder box'
      write(6,*) neln,' elements will be created for',nbox,' boxes.'
      write(6,*) nelx,nely
      write(6,*) 
c
c     Construct cylinder mesh in the z=0 plane (i.e., no z-dependence)
c
      e = 0
      do iy=1,nely
      do ix=1,nelx
         e = e+1
         call blank(ccurve(1,e),8)
         call rzero(curve (1,1,e),8*5)
         if (cob(ix-1).eq.'c'.or.cob(ix-1).eq.'C') then
            ccurve(4,e)   = 'C'                  ! EB notation
            curve (1,4,e) = -rad(ix-1)
            if (if3d) then
               ccurve(8,e)   = 'C'
               curve (1,8,e) = -rad(ix-1)
            endif
         endif
 
         if (cob(ix).eq.'c'.or.cob(ix).eq.'C') then
            ccurve(2,e)   = 'C'                  ! EB notation
            curve (1,2,e) = rad(ix)
            if (if3d) then
               ccurve(6,e)   = 'C'
               curve (1,6,e) = rad(ix)
            endif
         endif
 
         do i=0,1
            ix1 = ix-1 + i
            if (cob(ix1).eq.'c'.or.cob(ix1).eq.'C' .or. 
     $          cob(ix1).eq.'o'.or.cob(ix1).eq.'O') then ! L2 projection
               do j=0,1
                  ij  = 1 + i + 2*j                          ! h-cube ordering
                  iy1 = iy-1 + j
                  tha = twopi*tht(iy1)
                  x(ij,e) = rad(ix1)*cos(tha)
                  y(ij,e) = rad(ix1)*sin(tha)

                  if(ix1.gt.0.and.
     $               cob(ix1-1).eq.'b'.or.cob(ix1-1).eq.'B'.or.
     $               cob(ix1-1).eq.'p'.or.cob(ix1-1).eq.'P') then
                      if(abs(x(ij,e)).le.abs(x(ij,e-1))) then
                        write(6,*)x(ij,e),x(ij,e-1),e
                        write(6,*)"Curve sides must be greater than box"
                        write(6,*)"Change .box file to proceed"
                        call exitt()                    
                      endif
                  endif

               enddo
            elseif (cob(ix1).eq.'p'.or.cob(ix1).eq.'P') then ! L1 projection
               if(e.eq.1.and.mod(nely,2).ne.0) 
     $           write(6,*) "WARNING!!!!!!!! ",
     $           "Box-like geometry works best with even number of nely"
               do j=0,1
                  ij   = 1 + i + 2*j                         ! h-cube ordering
                  iy1  = iy-1 + j
                  rad2 = 2*rad(ix1)
                  tha  = twopi*tht(iy1)
                  x(ij,e) = rad2*cos(tha)
                  y(ij,e) = rad2*sin(tha)
                  if (abs(x(ij,e)).ge.abs(y(ij,e))) then ! threshold on x
                     ratio   = abs(rad(ix1)/x(ij,e))
                     if (x(ij,e).gt.0) x(ij,e) =  rad(ix1)
                     if (x(ij,e).lt.0) x(ij,e) = -rad(ix1)
                     y(ij,e) = ratio*y(ij,e)
                  else                       ! threshold on y
                     ratio   = abs(rad(ix1)/y(ij,e))
                     x(ij,e) = ratio*x(ij,e)
                     if (y(ij,e).gt.0) y(ij,e) =  rad(ix1)
                     if (y(ij,e).lt.0) y(ij,e) = -rad(ix1)
                  endif
               enddo
            else                               ! Cartesian box, proporionl
               if(e.eq.1.and.mod(nely,2).ne.0) 
     $           write(6,*) "WARNING!!!!!!!! ",
     $           "Box-like geometry works best with even number of nely"
               do j=0,1
                  ij   = 1 + i + 2*j           ! h-cube ordering
                  iy1  = iy-1 + j
                  rad2 = 2.*rad(ix1)
                  posp = 8*tht(iy1)            ! position along perimeter
                  if (-1.le.posp.and.posp.lt.1) then
                     x(ij,e) = rad(ix1)
                     y(ij,e) = rad(ix1)*posp
                  elseif (1.le.posp.and.posp.lt.3) then
                     x(ij,e) = rad(ix1)*(2-posp)
                     y(ij,e) = rad(ix1)
                  elseif (3.le.posp.and.posp.lt.5) then
                     x(ij,e) = -rad(ix1)
                     y(ij,e) =  rad(ix1)*(4-posp)
                  elseif (5.le.posp.and.posp.lt.7) then
                     x(ij,e) =  rad(ix1)*(posp-6)
                     y(ij,e) = -rad(ix1)
                  elseif (7.le.posp.and.posp.lt.9) then
                     x(ij,e) = rad(ix1)
                     y(ij,e) = rad(ix1)*(posp-8)
                  else
                     write(6,*) 'theta > 1. ABORT.',e,i,j,ij,tht(iy1)
                     call exitt
                  endif
               enddo
            endif
         enddo
      enddo
      enddo
 
      n = 4*nelx*nely
      call cadd(x,xy0(1),n)
      call cadd(y,xy0(2),n)
 
      epsm = 1.e-6
      do e=1,nelx*nely
         do ij=1,4
            if (abs(x(ij,e)).lt.epsm) x(ij,e) = 0.
            if (abs(y(ij,e)).lt.epsm) y(ij,e) = 0.
            ie = ipf2e(ij)
            write(11,6) x(ie,e),y(ie,e)
         enddo
         ij = 1
         ie = ipf2e(ij)
         write(11,6) x(ie,e),y(ie,e)
         write(11,6)
      enddo
    6 format(1p2e15.7)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine get_cyl_box(xy0,nels,cob,rad,tht,z,cbc,maxx,if3d,nfld)
c
c     Input is similar to genbox.
c
c     A standard input would be of the form:
c
c
c CYLBOX
c 2 -8 -2  NELX,Y,Z:  in r, theta, and z ( < 0 ==> regular distribution)
c c  ,o  ,b  inner-most cylinder, next "octagon", last "box" (Cartesian)
c .5 .75 1.0     r0, r1, r2.
c 0  1.0         -- fraction of circular arc (result is multiplied by 2pi)
c 0  2.5  1.5    -- z0, z1, geometric ratio
c bcx,bcx,bcy,bcy,bcz,bcz   Usual bcs
c
c
      integer nels(3)
      real    xy0(2),rad(0:maxx),tht(0:maxx),z(0:maxx)
      character*1 cob(0:maxx)
      character*3 cbc(6,nfld)
      logical if3d
 
      if (if3d) then
         call geti3(nelx,nely,nelz,iend,7)
         call overflow_chk(nelx,maxx,'nlx','cylbox')
         call overflow_chk(nely,maxx,'nly','cylbox')
         call overflow_chk(nelz,maxx,'nlz','cylbox')
      else
         call geti2(nelx,nely,iend,7)
         call overflow_chk(nelx,maxx,'nlx','cylbox')
         call overflow_chk(nely,maxx,'nly','cylbox')
         nelz = 0
      endif
      nels(1) = nelx
      nels(2) = nely
      nels(3) = nelz
 
      call getr2 (xy0(1),xy0(2),iend,7)  ! Coordinates of cyl. ctr.
 
      nelx1 = abs(nelx) + 1
      call getcv0(cob,1,nelx1,iend,7)   ! Cylinder type: c,o,b,p, etc.

      if (nelx.gt.0) then
         call getrv(rad,nelx+1,iend,7)
      else
         call getrv(rad,3,iend,7)           ! r0,  r1 , ratio
      endif
 
      if (nely.gt.0) then
         call getrv(tht,nely+1,iend,7)
      else
         call getrv(tht,3,iend,7)           ! th0, th1 , ratio
      endif
 
      if (nelz.gt.0) then
         call getrv(z,nelz+1,iend,7)
      elseif (nelz.lt.0) then
         call getrv(z  ,3,iend,7)           ! z0,  z1 , ratio
      endif
 
      do ifld=1,nfld
         call getcv(cbc(1,ifld),3,6,iend,7)
         write(6,*) 'CBC:',(cbc(k,ifld),k=1,6),ifld,ibox
      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine out_xy_box(x,y,z,nels,maxx,if3d)
c
c     Output a tensor-product (x,y) x z array of elements
c
      integer nels(3)
      real    x(4,1),y(4,1),z(0:maxx)
      logical if3d
 
      integer e
      integer ipf2e(4)
      save    ipf2e
      data    ipf2e / 1,2,4,3 /
 
      character*1  apt(52)
      character*52 apt52
      equivalence (apt52,apt)
      data apt52/'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'/
 
 
      nelx = abs(nels(1))
      nely = abs(nels(2))
      nelz = abs(nels(3))
 
      nel  = nelx*nely
      ndim = 2
      if (if3d) then
         nel  = nelx*nely*nelz
         ndim = 3
      endif
 
      write(9,10) nel,ndim,nel
   10 format(i12,i3,i12,'       NEL,NDIM,NELV')
 
      if (if3d) then
         e = 0
         do ke=1,nelz
            ie = 0
            ia = 0
            do iy=1,nely
            do ix=1,nelx
               e  =  e+1
               ie = ie+1
               ia = ia+1
               ia = mod1(ia,52)
 
               write(9,11) e,ke,apt(ia),'0'
 
               write(9,12) (x(ipf2e(k),ie),k=1,4)!convert to prenek numbering
               write(9,12) (y(ipf2e(k),ie),k=1,4)
               write(9,12) z(ke-1),z(ke-1),z(ke-1),z(ke-1)
 
               write(9,12) (x(ipf2e(k),ie),k=1,4)!convert to prenek numbering
               write(9,12) (y(ipf2e(k),ie),k=1,4)
               write(9,12) z(ke),z(ke),z(ke),z(ke)
 
            enddo
            enddo
         enddo
      else
         e  = 0
         ia = 0
         ke = 1
         do iy=1,nely
         do ix=1,nelx
            e  =  e+1
            ia = ia+1
            ia = mod1(ia,52)
 
            write(9,11) e,ke,apt(ia),'0'
 
            write(9,12) (x(ipf2e(k),e),k=1,4)!convert to prenek numbering
            write(9,12) (y(ipf2e(k),e),k=1,4)
 
         enddo
         enddo
      endif
 
   11 format(
     $'            ELEMENT',i12,' [',i5,a1,']    GROUP     ',a1)
c    $'            ELEMENT',i5,' [',i5,a1,']    GROUP     ',a1)
   12 format(1p4e15.7)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine out_curves(curve,ccurve,nel,if3d)
c
c     output curve side info 
c
      real        curve(5,8,nel)
      character*1 ccurve(8,nel)
      logical if3d
      integer e
c
c     First, count number of nontrivial curves
c
      ncurv = 0
      do e=1,nel
      do i=1,8
         if (ccurve(i,e).ne.' ') ncurv = ncurv+1
      enddo
      enddo
      write(9,28) ncurv
   28 format(
     $ '  ***** CURVED SIDE DATA *****',/,
     $    i12,' Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE')
 
      z=0
      do e=1,nel
      do i=1,8
         if (ccurve(i,e).ne.' ') then
            if (nel.lt.1000) then
               write(9,290) i,e,(curve(k,i,e),k=1,5),ccurve(i,e)
            elseif (nel.le.1 000 000) then
               write(9,291) i,e,(curve(k,i,e),k=1,5),ccurve(i,e)
            else
               write(9,292) i,e,(curve(k,i,e),k=1,5),ccurve(i,e)
            endif
         endif
      enddo
      enddo
  290 format(i3,i3,5g14.6,1x,a1)
  291 format(i2,i6,5g14.6,1x,a1)
  292 format(i2,i12,5g14.6,1x,a1)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine out_xy_curves(curve,ccurve,nels,if3d)
c
c     Output tensor-product [ (x,y) x z ] curve side info.
c
c
      real        curve(5,8,1)
      character*1 ccurve(8,1)
      integer nels(3)
      logical if3d
      integer e
 
      nelx = abs(nels(1))
      nely = abs(nels(2))
      nelz = abs(nels(3))
      nel2 = nelx*nely
      if (.not.if3d) nelz = 1
c
c     First, count number of nontrivial curves
c
      ncurv = 0
      do e=1,nel2
      do i=1,8
         if (ccurve(i,e).ne.' ') ncurv = ncurv+1
      enddo
      enddo
      ncurv = ncurv*nelz
      write(9,28) ncurv
   28 format(
     $ '  ***** CURVED SIDE DATA *****',/,
     $    i12,' Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE')
 
      e = 0
      do ke=1,nelz
      do ie=1,nel2
         e = e+1
         do i=1,8
            if (ccurve(i,ie).ne.' ') then
               if (nel.lt.1000) then
                  write(9,290) i,e,(curve(k,i,ie),k=1,5),ccurve(i,ie)
               elseif (nel.lt.1 000 000) then
                  write(9,291) i,e,(curve(k,i,ie),k=1,5),ccurve(i,ie)
               else
                  write(9,292) i,e,(curve(k,i,ie),k=1,5),ccurve(i,ie)
               endif
            endif
         enddo
      enddo
      enddo
  290 format(i3,i3,5g14.6,1x,a1)
  291 format(i2,i6,5g14.6,1x,a1)
  292 format(i2,i12,5g14.6,1x,a1)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine out_tens_bcs(cbc,nlx,nly,nlz,nbox,nfld,if3d,ifflow)
c
c     output bcs for multi-box tensor-product mesh
c
      character*3 cbc(6,nbox,nfld)
      integer     nlx(nbox),nly(nbox),nlz(nbox)
      logical     if3d,ifflow
 
      character*3 cbc1,   cbc2,   cbc3,   cbc4,   cbc5,   cbc6
      real        rbc1(5),rbc2(5),rbc3(5),rbc4(5),rbc5(5),rbc6(5)
      real*8 rbc8(6)
 
      integer eface(6)
      save    eface
      data    eface  / 4,2,1,3,5,6 /
 
 
      write(9,31) 
   31 format('  ***** BOUNDARY CONDITIONS *****')
 
 
      do ifld=1,nfld
         if (ifflow) then
            if2 = ifld-2
            if (ifld.eq.1) write(9,41)
            if (ifld.eq.2) write(9,42)
            if (ifld.ge.3) write(9,43) if2
   41       format('  ***** FLUID   BOUNDARY CONDITIONS *****')
   42       format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   43       format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')
         else
            if2 = ifld-1
            if (ifld.eq.1) write(9,51)
            if (ifld.eq.1) write(9,52)
            if (ifld.gt.1) write(9,53) if2
   51       format('  ***** NO FLUID   BOUNDARY CONDITIONS *****')
   52       format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   53       format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')
         endif
 
         ie = 0
         if (if3d) then
            do ibx=1,nbox
            do iez=1,nlz(ibx)
            do iey=1,nly(ibx)
            do iex=1,nlx(ibx)
 
               call rzero(rbc1,5)
               call rzero(rbc2,5)
               call rzero(rbc3,5)
               call rzero(rbc4,5)
               call rzero(rbc5,5)
               call rzero(rbc6,5)
               call rzero8(rbc8,6)
 
               nelx = nlx(ibx)
               nely = nly(ibx)
               nelz = nlz(ibx)
               ilftz = mod1(iez-1+nelz,nelz)
               irgtz = mod1(iez+1+nelz,nelz)
               ilfty = mod1(iey-1+nely,nely)
               irgty = mod1(iey+1+nely,nely)
               ilftx = mod1(iex-1+nelx,nelx)
               irgtx = mod1(iex+1+nelx,nelx)
 
               rbc1(1) = nely*nelx*(iez-1)+nelx*(iey-1) + ilftx
               rbc8(1) = nely*nelx*(iez-1)+nelx*(iey-1) + ilftx
               rbc1(2) = eface(2)
               rbc2(1) = nely*nelx*(iez-1)+nelx*(iey-1) + irgtx
               rbc8(2) = nely*nelx*(iez-1)+nelx*(iey-1) + irgtx
               rbc2(2) = eface(1)
 
               rbc3(1) = nely*nelx*(iez-1)+nelx*(ilfty-1) + iex
               rbc8(3) = nely*nelx*(iez-1)+nelx*(ilfty-1) + iex
               rbc3(2) = eface(4)
               rbc4(1) = nely*nelx*(iez-1)+nelx*(irgty-1) + iex
               rbc8(4) = nely*nelx*(iez-1)+nelx*(irgty-1) + iex
               rbc4(2) = eface(3)
 
               rbc5(1) = nely*nelx*(ilftz-1)+nelx*(iey-1) + iex
               rbc8(5) = nely*nelx*(ilftz-1)+nelx*(iey-1) + iex
               rbc5(2) = eface(6)
               rbc6(1) = nely*nelx*(irgtz-1)+nelx*(iey-1) + iex
               rbc8(6) = nely*nelx*(irgtz-1)+nelx*(iey-1) + iex
               rbc6(2) = eface(5)
 
               if (iex.eq.1) then
                  cbc1=cbc(1,ibx,ifld)
                  if (cbc1.ne.'P  '.and.cbc1.ne.'E  ') then 
                    call rzero(rbc1,5)
                    rbc8(1)=0.0
                  endif
               else
                  cbc1='E  '
               endif
 
               if (iex.eq.nelx) then
                  cbc2=cbc(2,ibx,ifld)
                  if (cbc2.ne.'P  '.and.cbc2.ne.'E  ') then
                    call rzero(rbc2,5)
                    rbc8(2)=0.0
                  endif
               else
                  cbc2='E  '
               endif
 
               if (iey.eq.1) then
                  cbc3=cbc(3,ibx,ifld)
                  if (cbc3.ne.'P  '.and.cbc3.ne.'E  ') then 
                    call rzero(rbc3,5)
                    rbc8(3)=0.0
                  endif
               else
                  cbc3='E  '
               endif
 
               if (iey.eq.nely) then
                  cbc4=cbc(4,ibx,ifld)
                  if (cbc4.ne.'P  '.and.cbc4.ne.'E  ') then 
                    call rzero(rbc4,5)
                    rbc8(4)=0.0
                  endif
               else
                  cbc4='E  '
               endif
 
               if (iez.eq.1) then
                  cbc5=cbc(5,ibx,ifld)
                  if (cbc5.ne.'P  '.and.cbc5.ne.'E  ') then 
                    call rzero(rbc5,5)
                    rbc8(5)=0.0
                  endif
               else
                  cbc5='E  '
               endif
 
               if (iez.eq.nelz) then
                  cbc6=cbc(6,ibx,ifld)
                  if (cbc6.ne.'P  '.and.cbc6.ne.'E  ')then 
                    call rzero(rbc6,5)
                    rbc8(6)=0.0
                  endif
               else
                  cbc6='E  '
               endif
c
c              output bc's in preproc. notation
c
               ie = ie+1
               if (nel.lt.1000) then
                  write(9,20) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                  write(9,20) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                  write(9,20) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                  write(9,20) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                  write(9,20) cbc5,ie,eface(5),(rbc5(j),j=1,5)
                  write(9,20) cbc6,ie,eface(6),(rbc6(j),j=1,5)
               elseif (nel.lt.100 000) then
                  write(9,21) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                  write(9,21) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                  write(9,21) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                  write(9,21) cbc1,ie,eface(1),(rbc1(j),j=1,5)
                  write(9,21) cbc5,ie,eface(5),(rbc5(j),j=1,5)
                  write(9,21) cbc6,ie,eface(6),(rbc6(j),j=1,5)
               elseif (nel.lt.1 000 000) then
                  write(9,22) cbc3,ie,(rbc3(j),j=1,5)
                  write(9,22) cbc2,ie,(rbc2(j),j=1,5)
                  write(9,22) cbc4,ie,(rbc4(j),j=1,5)
                  write(9,22) cbc1,ie,(rbc1(j),j=1,5)
                  write(9,22) cbc5,ie,(rbc5(j),j=1,5)
                  write(9,22) cbc6,ie,(rbc6(j),j=1,5)
               else
                  write(9,23) cbc3,ie,rbc8(1),(rbc3(j),j=2,5)
                  write(9,23) cbc2,ie,rbc8(2),(rbc2(j),j=2,5)
                  write(9,23) cbc4,ie,rbc8(3),(rbc4(j),j=2,5)
                  write(9,23) cbc1,ie,rbc8(4),(rbc1(j),j=2,5)
                  write(9,23) cbc5,ie,rbc8(5),(rbc5(j),j=2,5)
                  write(9,23) cbc6,ie,rbc8(6),(rbc6(j),j=2,5)
               endif
   20          format(1x,a3,2i3,5g14.6)
   21          format(1x,a3,i5,i1,5g14.6)
   22          format(1x,a3,i6,5g14.6)
   23          format(1x,a3,i12,5g18.11)
            enddo
            enddo
            enddo
            enddo
         else
            do ibx=1,nbox
            do iey=1,nly(ibx)
            do iex=1,nlx(ibx)
 
               call rzero(rbc1,5)
               call rzero(rbc2,5)
               call rzero(rbc3,5)
               call rzero(rbc4,5)
               call rzero8(rbc8,4)
 
               nelx = nlx(ibx)
               nely = nly(ibx)
               ilfty = mod1(iey-1+nely,nely)
               irgty = mod1(iey+1+nely,nely)
               ilftx = mod1(iex-1+nelx,nelx)
               irgtx = mod1(iex+1+nelx,nelx)
 
               rbc1(1) = nelx*(iey-1) + ilftx
               rbc8(1) = nelx*(iey-1) + ilftx
               rbc1(2) = eface(2)
               rbc2(1) = nelx*(iey-1) + irgtx
               rbc8(2) = nelx*(iey-1) + irgtx
               rbc2(2) = eface(1)
 
               rbc3(1) = nelx*(ilfty-1) + iex
               rbc8(3) = nelx*(ilfty-1) + iex
               rbc3(2) = eface(4)
               rbc4(1) = nelx*(irgty-1) + iex
               rbc8(4) = nelx*(irgty-1) + iex
               rbc4(2) = eface(3)
 
               if (iex.eq.1) then
                  cbc1=cbc(1,ibx,ifld)
                  if (cbc1.ne.'P  '.and.cbc1.ne.'E  ') then 
                     call rzero(rbc1,5) 
                     rbc8(1) = 0.0
                  endif
               else
                  cbc1='E  '
               endif
 
               if (iex.eq.nelx) then
                  cbc2=cbc(2,ibx,ifld)
                  if (cbc2.ne.'P  '.and.cbc2.ne.'E  ') then 
                     call rzero(rbc2,5)
                     rbc8(2) = 0.0
                  endif
               else
                  cbc2='E  '
               endif
 
               if (iey.eq.1) then
                  cbc3=cbc(3,ibx,ifld)
                  if (cbc3.ne.'P  '.and.cbc3.ne.'E  ') then 
                     call rzero(rbc3,5)
                     rbc8(3) = 0.0
                  endif
               else
                  cbc3='E  '
               endif
 
               if (iey.eq.nely) then
                  cbc4=cbc(4,ibx,ifld)
                  if (cbc4.ne.'P  '.and.cbc4.ne.'E  ') then
                     call rzero(rbc4,5)
                     rbc8(4) = 0.0
                  endif
               else
                  cbc4='E  '
               endif
c
c              output bc's in preproc. notation
c
               ie = ie+1
               if (nel.lt.1000) then
                  write(9,20) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                  write(9,20) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                  write(9,20) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                  write(9,20) cbc1,ie,eface(1),(rbc1(j),j=1,5)
               elseif (nel.lt.100 000) then
                  write(9,21) cbc3,ie,eface(3),(rbc3(j),j=1,5)
                  write(9,21) cbc2,ie,eface(2),(rbc2(j),j=1,5)
                  write(9,21) cbc4,ie,eface(4),(rbc4(j),j=1,5)
                  write(9,21) cbc1,ie,eface(1),(rbc1(j),j=1,5)
               elseif (nel.lt.1 000 000) then
                  write(9,22) cbc3,ie,(rbc3(j),j=1,5)
                  write(9,22) cbc2,ie,(rbc2(j),j=1,5)
                  write(9,22) cbc4,ie,(rbc4(j),j=1,5)
                  write(9,22) cbc1,ie,(rbc1(j),j=1,5)
               else
                  write(9,23) cbc3,ie,rbc8(1),(rbc3(j),j=2,5)
                  write(9,23) cbc2,ie,rbc8(2),(rbc2(j),j=2,5)
                  write(9,23) cbc4,ie,rbc8(3),(rbc4(j),j=2,5)
                  write(9,23) cbc1,ie,rbc8(4),(rbc1(j),j=2,5)
               endif
            enddo
            enddo
            enddo
         endif
      enddo
      if(ifflow) then
        if (nfld.eq.1) write(9,*) 
     &     ' ***** NO THERMAL BOUNDARY CONDITIONS *****'
      endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine ifill(x,ic,n)
      integer x(1)
      do i=1,n
         x(i) = ic
      enddo
      return
      end
c-----------------------------------------------------------------------
      function ilsum(x,n)
      integer x(1)
      ilsum = 0
      do i=1,n
         ilsum = ilsum + x(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine set_logical(ifflow,ifheat,inf,outf,nlogic)
 
      logical ifflow, ifheat
      integer inf,outf

      integer line,lout
      character*132 temps
      character*1   temps1(132)
      logical chk_heat,chk_flow

      chk_flow = .false.
      chk_heat = .false.

      do line=1,nlogic
         call blank(temps,132)
         read (inf,132,end=100,err=100) temps
         call ccopy(temps1,temps,132)             
         lout = ltrunc(temps1,132)                !len of temps1/temps
         
         if (indx1(temps,'IFFLOW',6).ne.0) then  
            chk_flow = .true.
            if (ifflow) then
               write(outf,*) ' T     IFFLOW' 
            else
               write(outf,*) ' F     IFFLOW' 
            endif
         elseif (indx1(temps,'IFHEAT',6).ne.0) then
            chk_heat = .true.
            if (ifheat) then 
               write(outf,*) ' T     IFHEAT'
            else
               write(outf,*) ' F     IFHEAT'
            endif
         else
           write (outf,81) (temps1(j),j=1,lout)          
         endif
      enddo

      if (chk_flow.and.chk_heat) then
         goto 10
      elseif(.not.chk_flow.and.ifflow) then
         write(outf,*) ' T     IFFLOW'
         write(6,*) "USER NEEDS TO INCREASE NUMBER OF LOGICAL SWITCHES",
     $    " BY 1 IN .REA FILE!!!!!!!!!!!!!!!!!"
      elseif(.not.chk_heat.and.ifheat) then
         write(outf,*) ' T     IFHEAT'
         write(6,*) "USER NEEDS TO INCREASE NUMBER OF LOGICAL SWITCHES",
     $    " BY 1 IN .REA FILE!!!!!!!!!!!!!!!!!"
      endif

 10   do line = 1,10
         call blank(temps,132)
         read (inf,132,end=100,err=100) temps
         call ccopy(temps1,temps,132)             
         lout = ltrunc(temps1,132)                !len of temps1/temps
         write (outf,81) (temps1(j),j=1,lout)          
         if (indx1(temps,'MESH DATA',9).ne.0) return
      enddo

  132 format(a132)
   81 format(132a1)

  100 continue
      write (6,*) 'In subroutine set_logical'
      return
      end
c-----------------------------------------------------------------------
      subroutine scanparam(str,str1,nparam,nfd,ifflw,ifht,ifmhd
     $                                              ,infil,outfil)

      integer infil,outfil
      character*132 str
      character*1   str1(132)
      logical ifflw, ifht, ifmhd

      nps = nfd - 1
      if (ifflw.and.ifht) nps = nfd-2
      if (ifmhd)          nps = nps-1
      if (nps.lt.0) nps = 0
      do i = 1,nparam
         call blank(str,132)
         read(infil,'(a132)') str
         call ccopy(str1,str,132)

         if(i.eq.23) then 
           write(outfil,90) nps
   90      format(' ',i2,'              p23 NPSCAL')
         else 
           lout = ltrunc(str1,132)
           write (outfil,81) (str1(j),j=1,lout)
   81      format(132a1)
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
