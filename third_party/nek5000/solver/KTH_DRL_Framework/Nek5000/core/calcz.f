c-----------------------------------------------------------------------
      subroutine calcz(d,e,n,dmax,dmin,z,ierr)
c
c     Num. Rec. 2nd Ed., p.473
c
c     Note:  d(1:n) is the diagonal of the sym. tridiagonal matrix
c            e(1:n) is the upper diagonal of the tridiagonal matrix,
c                   WITH e(n) ARBITRARY (a slight shift from Num.Rec.)
c
c            z(n:n) is the packed array of eigenvectors
c
      real  d(n),e(n),z(n,n)
      real  smalln,small
c
      call ident(z,n)
      one = 1.
c
c     Find smallest number  (pff mod to Num. Rec. 2nd Ed., p.473)
c
      small = 0.5
      do i = 1,100
         smalln = small * small
         if (smalln .eq.0) then
            do j=1,1000
               smalln = 0.5*small
               if (smalln .eq.0) goto 10
               small = smalln
            enddo
         endif
         small = smalln
      enddo
   10 continue
      small = 10.*small
      small = max(small,1e-99)
      
c     write(6,*) 'this is small:',small
c
      do 15 l=1,n
         iter = 0
c
    1    do 12 m=l,n-1
            dd = abs( d(m) ) + abs( d(m+1) )
            de = e(m) + dd
            df = abs(dd - de)
c           write(6,112) iter,m,'de:',dd,de,df,small
            if ( df .le. small ) goto 2
   12    continue
  112    format(i3,i9,1x,a3,1p4e16.8)
         m = n
c
    2    if ( m .ne. l ) then
            if ( iter .eq. 600 ) then
               write (6,*) 'too many iterations in calc'
c              n10 = min(n,10)
c              do i=1,n
c                 write(6,9) d(i),(z(i,j),j=1,n10)
c              enddo
c   9          format(1pe12.4,' e:',1p10e12.4)
c              call exitt
               ierr=1
               return
            endif
c
            iter = iter + 1
            g = ( d(l+1) - d(l) ) / ( 2.0 * e(l) )
            r = pythag(g,one)
            g = d(m) - d(l) + e(l)/(g+sign(r,g))
            s = 1.0
            c = 1.0
            p = 0.0
c
            do 14 i = m-1,l,-1
               f = s * e(i)
               b = c * e(i)
               r = pythag(f,g)
               e(i+1) = r
               if ( abs(r) .le. small ) then
                  d(i+1) = d(i+1) - p
                  e(m)   = 0.
                  goto 1
               endif
               s = f/r
               c = g/r
               g = d(i+1) - p
               r = ( d(i)-g )*s + 2.*c*b
               p = s*r
               d(i+1) = g + p
               g = c*r - b
c      ...     find eigenvectors ... (new, 11/19/94, pff, p.363. Num.Rec.I.)
               do 13 k=1,n
                  f = z(k,i+1)
                  z(k,i+1) = s*z(k,i)+c*f
                  z(k,i  ) = c*z(k,i)-s*f
   13          continue
c      ...     end of eigenvector section ... 
   14       continue
c
            d(l) = d(l) - p
            e(l) = g
            e(m) = 0.0
            goto 1
         endif
c
   15 continue
c
c     write (8,8) (d(j),j=1,n)
c   8 format('eig:',8f10.4)
c
      dmin = d(1)
      dmax = d(1)
      do 40 i = 1 , n
        dmin = min( d(i) , dmin )
        dmax = max( d(i) , dmax )
   40 continue
c
c     Output eigenvectors
c
c     n10 = min(n,10)
c     do i=1,n
c        write(6,9) d(i),(z(i,j),j=1,n10)
c     enddo
c   9 format(1pe12.4,' e:',1p10e12.4)
c
      ierr=0
      return
      end
c-----------------------------------------------------------------------
      function pythag(a,b)
c
c     Compute sqrt(a*a + b*b) w/o under- or over-flow.
c
      absa=abs(a) 
      absb=abs(b) 
      if (absa.gt.absb) then
         pythag = absa*sqrt(1. + (absb/absa)**2 )
      else
         if (absb.eq.0.) then
            pythag = 0.
         else
            pythag = absb*sqrt(1. + (absa/absb)**2 )
         endif
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine ident(a,n)
      real  a(n,n)
      call rzero(a,n*n)
      do i=1,n
         a(i,i) = 1.0
      enddo
      return
      end
c-----------------------------------------------------------------------
