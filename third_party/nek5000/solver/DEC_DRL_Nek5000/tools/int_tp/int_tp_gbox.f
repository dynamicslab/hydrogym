c-----------------------------------------------------------------------
      subroutine get_gen_box(xe,nelx,ye,nely,ze,nelz,lmax,name3)
      real xe(0:1),ye(0:1),ze(0:1)
      logical if3d
 
      character*3  name3
 
      character*80 string
      character*1  string1(80)
      equivalence (string,string1)
 
 
      call blank (string,80)
      write(6,1) name3
    1 format('Input name of ',a3,' genbox geometry file:')
      read (5,*) string
c
c     Read in name of previously generated NEKTON data set.
c     (Must have same dimension and number of fields as current run)
c
      open (unit=7,file=string,status='old')
      call gets(string,80,iend,7)  ! This is a dummy read, for compatibility
c
c     here is where the 2d/3d determination is made....
c
      call geti1(ndim,iend,7)
      call geti1(nfld,iend,7)                ! Determine number of fields
 
      if3d=.false.
      ndim=abs(ndim)
      if (ndim.eq.3) if3d=.true.
      call getbox(xe,nelx,ye,nely,ze,nelz,nfld,if3d)  ! Read in the .box file
 
      if (nelx.gt.lmax) then
         write(6,*) 'nelx too large',nelx,lmax
         call exitt
      elseif (nely.gt.lmax) then
         write(6,*) 'nely too large',nely,lmax
         call exitt
      elseif (nelz.gt.lmax) then
         write(6,*) 'nelz too large',nelz,lmax
         call exitt
      endif
 
      close (unit=7)
 
c     call outmat(xe,1,nelx+1,'x gen2',nelx)
c     call outmat(ye,1,nely+1,'y gen2',nely)
c     call outmat(ze,1,nelz+1,'z gen2',nelz)
 
      return
      end
c-----------------------------------------------------------------------
      subroutine getbox(x,nelx,y,nely,z,nelz,nfld,if3d)
 
      real x(0:1),y(0:1),z(0:1)
 
      logical if3d,ifevenx,ifeveny,ifevenz,if_multi_seg
      character*1 boxcirc
 
      integer nelxyz(3)
 
 
      ifevenx = .false.
      ifeveny = .false.
      ifevenz = .false.
 
      call gets(boxcirc,1,iend,7)
      if (iend.eq.1) goto 99
      if (boxcirc.eq.'c'.or.boxcirc.eq.'C') 
     $                    call getr2(xc,yc,iend,7)
      if_multi_seg = .false.
      if (boxcirc.eq.'m'.or.boxcirc.eq.'M') 
     $      if_multi_seg = .true.
 
      if (boxcirc.eq.'y'.or.boxcirc.eq.'Y') then
         write(6,*) 'cylindrical coords. not supported for t-p interp.'
         call exitt
      endif

      if (if_multi_seg) then
         write(6,*) 'if_multi_seg not supported for t-p interpolation'
         call exitt
      elseif (if3d) then
         call geti3(nelx,nely,nelz,iend,7)
 
         if (nelx.lt.0) ifevenx = .true.
         if (nely.lt.0) ifeveny = .true.
         if (nelz.lt.0) ifevenz = .true.
         nelx = abs(nelx)
         nely = abs(nely)
         nelz = abs(nelz)
c
c        if (nelx.gt.lelx.or.nely.gt.lely.or.nelz.gt.lelz) then
c           write(6,*) 'Abort, increase lelx,..lelz and recompile',
c    $                    nelx,nely,nelz,lelx,lely,lelz
c           call exitt
c        endif
c
         nery = nely
         if (boxcirc.eq.'C') nery = 1
         if (iend.eq.1) goto 99
         ninbox = nelx*nely*nelz
         write(6,6) ninbox,nelx,nely,nelz,nfld
 
         if (ifevenx) then
            call getr3(x0,x1,ratio,iend,7)
            if (boxcirc.eq.'c'.or.boxcirc.eq.'C') then
               if(x0.le.xc) then
                 write(6,*) 'ABORT, x0 is equal to the center!'
                 call exitt
               endif
            endif
            if (ratio.le.0) ratio=1.
            dx = (x1-x0)/nelx
            x(0) = 0.
            do k=1,nelx
               x(k) = x(k-1) + dx
               dx        = dx*ratio
            enddo
            scale = (x1-x0)/x(nelx)
            do k=0,nelx
               x(k) = x0 + scale*x(k)
            enddo
         else
            call getrv(x(0),nelx+1,iend,7)
            if (boxcirc.eq.'c'.or.boxcirc.eq.'C') then
               if(x(0).le.xc) then
                 write(6,*) 'ABORT, x0 is equal to the center!'
                 call exitt
               endif
            endif
         endif
 
         if (ifeveny) then
            call getr3(y0,y1,ratio,iend,7)
            if (ratio.le.0) ratio=1.
            dy = (y1-y0)/nely
            y(0) = 0.
            do k=1,nely
               y(k) = y(k-1) + dy
               dy        = dy*ratio
            enddo
            scale = (y1-y0)/y(nely)
            do k=0,nely
               y(k) = y0 + scale*y(k)
            enddo
         else
            call getrv(y(0),nely+1,iend,7)
         endif
 
         if (ifevenz) then
            call getr3(z0,z1,ratio,iend,7)
            if (ratio.le.0) ratio=1.
            dz = (z1-z0)/nelz
            z(0) = 0.
            do k=1,nelz
               z(k) = z(k-1) + dz
               dz        = dz*ratio
            enddo
            scale = (z1-z0)/z(nelz)
            do k=0,nelz
               z(k) = z0 + scale*z(k)
            enddo
         else
            call getrv(z(0),nelz+1,iend,7)
         endif
 
      else         ! 2D
         nelz = 1
         call geti2(nelx,nely,iend,7)
 
         if (nelx.lt.0) ifevenx = .true.
         if (nely.lt.0) ifeveny = .true.
         nelx = abs(nelx)
         nely = abs(nely)
c        if (nelx.gt.lelx.or.nely.gt.lely) then
c           write(6,*) 'Abort, increase lelx,lely and recompile',
c    $                    nelx,nely,lelx,lely
c           call exitt
c        endif
c
         nery = nely
         if (boxcirc.eq.'C') nery = 1
         if (iend.eq.1) goto 99
         ninbox = nelx*nely*nelz
         write(6,6) ninbox,nelx,nely,nery,nfld
 
         if (ifevenx) then
            call getr3(x0,x1,ratio,iend,7)
            if (boxcirc.eq.'c'.or.boxcirc.eq.'C') then
                if(x0.le.xc) then
                  write(6,*) 'ABORT, x0 is equal to the center!'
                  call exitt
                endif
            endif

            if (ratio.le.0) ratio=1.
            dx = (x1-x0)/nelx
            x(0) = 0.
            do k=1,nelx
               x(k) = x(k-1) + dx
               dx        = dx*ratio
            enddo
            scale = (x1-x0)/x(nelx)
            do k=0,nelx
               x(k) = x0 + scale*x(k)
            enddo
         else
            call getrv(x(0),nelx+1,iend,7)
            if (boxcirc.eq.'c'.or.boxcirc.eq.'C') then
                if(x(0).le.xc) then
                  write(6,*) 'ABORT, x0 is equal to the center!'
                  call exitt
                endif
            endif
         endif
 
         if (ifeveny) then
            call getr3(y0,y1,ratio,iend,7)
            if (ratio.le.0) ratio=1.
            dy = (y1-y0)/nely
            y(0) = 0.
            do k=1,nely
               y(k) = y(k-1) + dy
               dy        = dy*ratio
            enddo
            scale = (y1-y0)/y(nely)
            do k=0,nely
               y(k) = y0 + scale*y(k)
            enddo
         else
            call getrv(y(0),nely+1,iend,7)
         endif
 
      endif
      nel = nelx*nely*nelz
    6 format('Reading',i12,' =',3i6,' elements for box',i3,'.')
   99 continue
 
c     call outmat(x,1,nelx+1,'x genb',nelx)
c     call outmat(y,1,nely+1,'y genb',nely)
c     call outmat(z,1,nelz+1,'z genb',nelz)
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
      subroutine getr2(r1,r2,iend,io)
c
c     Get two reals from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) r1,r2
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getr3(r1,r2,r3,iend,io)
c
c     Get two reals from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,*) r1,r2,r3
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getrv(r,n,iend,io)
      real r(n)
      parameter (big=1.e29)
c
c     Get reals from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
c
c     serious hack for f90
      call cfill(r,big,n)
 
      open(unit=99,file='box.tmp')
      read(99,*,end=1) (r(k),k=1,n)
      close(unit=99)
      return
    1 continue
      close(unit=99)
      do k=1,n
         if (r(k).eq.big) goto 15
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
      subroutine getiv(r,n,iend,io)
      integer r(n)
      parameter (ibig=99999999)
c
c     Get integers from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
c
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
      subroutine getcv0(c,m,n,iend,io)
      character*1 c(m,n)
      character*1 adum(80)
c
c     Get character strings, with no seperator, from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,1,end=2) (adum(k),k=1,80)
    1 format(80a1)
    2 continue
c
      i = 0
      do l=1,n
         do k=1,m
            i = i+1
            c(k,l) = adum(i)
         enddo
      enddo
      do j=1,n
         write(6,3) m,n,(c(i,j),i=1,m)
    3    format(2i4,'getcv0:',80a1)
      enddo
 
      close(unit=99)
      return
      end
c-----------------------------------------------------------------------
      subroutine getcv(c,m,n,iend,io)
      character*1 c(m,n)
      character*1 adum(80)
c
c     Get character strings, with single seperator, from first uncommented line
c
      call scannocom(iend,io)
      if (iend.ne.0) return
      open(unit=99,file='box.tmp')
      read(99,1,end=2) (adum(k),k=1,80)
    1 format(80a1)
    2 continue
 
      i = 0
      do l=1,n
         do k=1,m
            i = i+1
            c(k,l) = adum(i)
         enddo
 
c        bump pointer for "," in input string
         i = i+1
      enddo
      do j=1,n
         write(6,3) m,n,(c(i,j),i=1,m)
    3    format(2i4,'getcv:',80a1)
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
    1 format(80a1)
      close(unit=99)
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
            write(6,*) 'Abort in get_multi_seg.'
            call exit
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
      subroutine get_xyz_distribution (x,nelx)
      real x(0:1)
      if (nelx.ge.0) return ! else, uniform or geometric spacing
 
      x0 = x(0)
      x1 = x(1)
c?    r  = x(2)
      write(6,*) 'x0:',x0,x1,r,nelx
 
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
      subroutine scannocom(iend,infile)
c
c     scan through infile until "no comment" is found
c
      character*80 string
      character*1  string1(80)
      equivalence (string1,string)
 
      character*1 comment
      save        comment
      data        comment /'#'/
 
      iend = 0
      do line=1,100000
         call blank(string,80)
         read (infile ,80,end=100,err=100) string
 
         write(6,80) string
         if   (indx1(string,comment,1).ne.1) then
              open(unit=99,file='box.tmp')
              len = ltrunc(string,80)
              write(99,81) (string1(k),k=1,len)
              close(unit=99)
              return
         else
              i1 = indx1(string,comment,1)
c             write(6,*) i1,' i1', comment
         endif
 
      enddo
   80 format(a80)
   81 format(80a1)
 
  100 continue
      iend = 1
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
      subroutine iint(x,n)
      integer x(1)
      do i=1,n
         x(i) = i
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine exitt
      write(6,*) 'exitt called'
      call exit
      return
      end
c-----------------------------------------------------------------------
      subroutine ifill(x,c,n)
      integer x(1),c
      do i=1,n
         x(i) = c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cfill(x,c,n)
      real x(1),c
      do i=1,n
         x(i) = c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cadd(x,c,n)
      real x(1),c
      do i=1,n
         x(i) = x(i) + c
      enddo
      return
      end
c-----------------------------------------------------------------------
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*80 S1,S2
C
      N1=80-L2+1
      INDX1=0
      IF (N1.LT.1) RETURN
C
      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            RETURN
         ENDIF
  100 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
