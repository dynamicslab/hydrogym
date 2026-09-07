      program fld_avg
c
c     Compute average and rms fluctuation of .fld files.
c
c     NOTE:  COMPILE in real*4 mode only, because of byte_read/write.
c
c
      parameter (nmax=2900000*5)
      common /c1/ xnew(nmax)
      common /c2/ xavg(nmax)
      common /c3/ xdif(nmax)
c
      character*40 fname
      logical      if_byte_sw
c
c
c     Compute average
c
      open(unit=10,file='file.list',status='old')
c
      write(6,*) 'input n:'
      read (5,*) n
c
      if (n.le.0) then
         nelv = 1021
         nx   = 12
         ntot = nx*nx*nx*nelv
         n    = 5*ntot
      endif
c
      if (n.gt.nmax) then
         write(6,*) 'recompile with new nmax:',nmax,n
         call exit
      endif
c
      call rzero(xavg,nmax)
      nfiles = 0
      do ifile = 1,1000
c
         call blank(fname,40)
         read(10,10,end=101) fname
   10    format(a40)
c
         call get_data(xnew,n,if_byte_sw,fname)
c
         nfiles = nfiles+1
         call add2(xavg,xnew,n)
         write(6,12) nfiles,n,fname
   12    format('File',i4,'.',i9,' points avgd.',5x,a40)
c
         call outl(xnew,5,'new',nfiles)
         call outl(xavg,5,'avg',nfiles)
      enddo
c
  101 continue
c
      scale = 1./nfiles
      call cmult(xavg,scale,n)
      call outl(xavg,5,'fvg',nfiles)
      call stat(xavg,ntot,'avg')
c
      call out_data(xavg,n,if_byte_sw,'avg.fld\0',xnew)
c
c     Compute rms difference
c
      call rzero(xdif,nmax)
      rewind(unit=10)
      nfiles = 0
      do ifile = 1,1000
c
         call blank(fname,40)
         read(10,10,end=201) fname
c
         call get_data(xnew,n,if_byte_sw,fname)
c
         nfiles = nfiles+1
         call sub2(xnew,xavg,n)
         call add2sq(xdif,xnew,n)
         call outl(xnew,5,'new',nfiles)
         call outl(xdif,5,'dif',nfiles)
         write(6,22) nfiles,n,fname
   22    format('File',i4,'.',i9,' points difd.',5x,a40)
c
      enddo
c
  201 continue
c
      scale = 1./nfiles
      call cmult(xdif,scale,n)
      call vsqrt(xdif,n)
      call outl(xdif,5,'dff',nfiles)
      call stat(xdif,ntot,'dif')
c
      call out_data(xdif,n,if_byte_sw,'dif.fld\0',xnew)
c
      stop
      end
c-----------------------------------------------------------------------
      subroutine add2(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) + y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine copy(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine sub2(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) - y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2sq(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = x(i) + y(i)*y(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine izero(x,n)
      integer x(1)
      do i=1,n
         x(i) = 0
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine rzero(x,n)
      real x(1)
      do i=1,n
         x(i) = 0.
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine cmult(x,c,n)
      real x(1),c
      do i=1,n
         x(i) = x(i) * c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine vsqrt(x,n)
      real x(1)
      do i=1,n
         if (x(i).gt.0) then
            x(i) = sqrt(x(i))
         else
            write(6,*) i,n,' vsqrt? arg:',x(i)
            x(i) = 0.
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine blank(x,n)
      character*1 x(1)
      do i=1,n
         x(i) = ' '
      enddo
      return
      end
c-----------------------------------------------------------------------
      function glmin(x,n)
      real x(1)
c
      s = x(1)
c
      do i=1,n
         s = min(s,x(i))
      enddo
      glmin = s
      return
      end
c-----------------------------------------------------------------------
      function glmax(x,n)
      real x(1)
c
      s = x(1)
c
      do i=1,n
         s = max(s,x(i))
      enddo
      glmax = s
      return
      end
c-----------------------------------------------------------------------
      FUNCTION LTRUNC(STRING,L)
      CHARACTER*1 STRING(L)
      CHARACTER*1   BLNK
      DATA BLNK/' '/
C
      DO 100 I=L,1,-1
         L1=I
         IF (STRING(I).NE.BLNK) GOTO 200
  100 CONTINUE
      L1=0
  200 CONTINUE
      LTRUNC=L1
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHCOPY(A,B,N)
      CHARACTER*1 A(1), B(1)
      DO 100 I = 1, N
 100     A(I) = B(I)
      RETURN
      END
c-----------------------------------------------------------------------
      logical function if_byte_swap_test(bytetest)
c
      real*4 bytetest
      real*4 test_pattern
      save   test_pattern
c
      test_pattern = 6.54321
      if_byte_swap_test = .false.
      if (bytetest.ne.test_pattern) if_byte_swap_test = .true.
c
      write(6,*) 'Byte swap:',if_byte_swap_test,bytetest
      return
      end
c-----------------------------------------------------------------------
      subroutine get_data(x,n,if_byte_sw,fname)
C
      real*4 x(1)
      character*40 fname
c
      integer fnamei(10)
      character*1 fnamec(40)
      equivalence (fnamec,fnamei)
c
      logical if_byte_sw, if_byte_swap_test
      common /byte_key/ bytetest
c
c
c     Open file
c
      len = ltrunc    (fname,40)
      call izero      (fnamei,10)
      call chcopy     (fnamec,fname,len)
      call byte_open  (fnamec)
c
c     Test header
c
      call byte_read(bytetest,1)
      if_byte_sw = if_byte_swap_test(bytetest)
C
      call byte_read(x,n)
      if (if_byte_sw) call byte_reverse(x,n)
c
      call byte_close()
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_data(x,n,if_byte_sw,fname,y)
C
      real*4 x(1),y(1)
      character*40 fname
c
      logical if_byte_sw
      common /byte_key/ bytetest
c
c
c     Open file
c
      call byte_open(fname)
c
c     Write original header
c
      call byte_write(bytetest,1)
c
      call copy(y,x,n)
      if (if_byte_sw) call byte_reverse(y,n)
      call byte_write(y,n)
c
      call byte_close()
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outl(x,n,name)
      character*3 name
      real x(1)
c
      n1 = min(n,6)
      write(6,1) name,(x(k),k=1,n1)
   1  format(a3,1p6e13.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine stat(x,n,name)
      character*3 name
      real x(n,1)
      real ymax(5),ymin(5)
c
      do k=1,5
         ymax(k) = glmax(x(1,k),n)
         ymin(k) = glmin(x(1,k),n)
      enddo
c
      write(6,1) name,' min ',(ymin(k),k=1,5)
      write(6,1) name,' max ',(ymax(k),k=1,5)
c
   1  format(a3,a5,1p6e13.5)
c
      return
      end
c-----------------------------------------------------------------------
