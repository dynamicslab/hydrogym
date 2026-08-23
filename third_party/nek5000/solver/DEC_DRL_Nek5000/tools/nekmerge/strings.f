c-----------------------------------------------------------------------
      subroutine scanout(string,input,len,infile,outfile)
c
c     scan through infile until input is found and output
c     all lines save that containing input to "outfile"
c
      character*80 string
      character*1 input(1)
      integer infile,outfile
c
      character*1 string1(80)
c
      do line=1,100000
         call blank(string,80)
         read (infile ,80,end=100,err=100) string
c        write(6,*) string
c        if (indx1(input,'end',3).ne.0) write(6,*) string
         if (indx1(string,input,len).ne.0) return
         call chcopy(string1,string,80)
         lout = ltrunc(string1,80)
         if (outfile.gt.0) write (outfile,81) (string1(j),j=1,lout)
      enddo
   80 format(a80)
   81 format(80a1)
c
 100  continue
      return
      end

c-----------------------------------------------------------------------
      subroutine scanoutafter(string,input,len,infile,outfile)
c
c     scan through infile until input is found and output
c     all lines after save that containing input to "outfile"
c
      character*80 string
      character*1 input(1)
      integer infile,outfile
c
      character*1 string1(80)
c
      do line=1,100000
         call blank(string,80)
         read (infile ,80,end=100,err=100) string
         if (indx1(string,input,len).ne.0) return
         call chcopy(string1,string,80)
         lout = ltrunc(string1,80)
         if (outfile.gt.0) write (outfile,81) (string1(j),j=1,lout)
      enddo
   80 format(a80)
   81 format(80a1)
c
 100  continue
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
      subroutine blank(s,n)
      character*1 s(1)
      do i=1,n
        s(i)=' '
      enddo
      return
      end
c-----------------------------------------------------------------------
c     comes from n2to3
      function ltrunc(s,n)
      character*1 s(1)
      ltrunc = 0
      do j=n,1,-1
         if (s(j).ne.' ') then
            ltrunc = j 
            return
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
c     comes from n2to3
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*80 S1,S2
C
      N1=80-L2+1
      INDX1=0
      IF (N1.LT.1) RETURN
C
      DO 300 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            RETURN
         ENDIF
300   CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine cleanr(x,n)
      real x(1)

      eps = 1.e-30

      do i=1,n
         if (abs(x(i)).lt.eps) x(i) = 0.0
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine rzero(x,n)
      real x(1)
      do i=1,n
         x(i) = 0.0
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
      subroutine out_data(y,n,fname)
C
      real*4 x(1),y(1)
      character*40 fname
c
c      logical if_byte_sw
c      common /byte_key/ bytetest
c
c
c     Open file
c
      call byte_open(fname)
c
c     Write original header
c

c      call byte_write(bytetest,1)
c
c      call copy(y,x,n)
c      if (if_byte_sw) call byte_reverse(y,n)
      call byte_write(y,n)
c
      call byte_close()
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lineout(io,s,n)
      character*1 s(n)
      m = ltrunc(s,n)
      write(io,1) (s(k),k=1,m)
    1 format(132a1)
      return
      end
c-----------------------------------------------------------------------
      subroutine fbyte_write(s,n)
      character*4 s(n)

      common /byte_flag/ ifbyte
      logical ifbyte

      if (ifbyte) then
         call byte_write(s,n)
      else
         call lineout(6,s,n)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine strg_write(s,n)
      character*4 s(n)

      write(6,4) (s(k),k=1,n)
    4 format(20a4)

      return
      end
c-----------------------------------------------------------------------
      subroutine  int_write(s,n)
      integer s(n)

      write(6,4) (s(k),k=1,n)
    4 format(10i8)

      return
      end
c-----------------------------------------------------------------------
      subroutine real_write(s,n)
      real s(n)

      write(6,4) (s(k),k=1,n)
    4 format(1p5e13.5)

      return
      end
c-----------------------------------------------------------------------
      subroutine bdry_write(s,n)
      real s(n)

      call  int_write(s(1),2)
      call real_write(s(3),5)
      call strg_write(s(8),1)

      return
      end
c-----------------------------------------------------------------------
