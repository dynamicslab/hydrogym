c-----------------------------------------------------------------------
      subroutine semhat(a,b,c,d,dp,jp,z,n,w)
c
c     Generate matrices for single element, 1D operators:
c
c        a  = Laplacian
c        b  = diagonal mass matrix
c        c  = convection operator b*d
c        d  = derivative matrix
c        dp = derivative matrix, mapping from pressure nodes to velocity
c        jp = interpolation matrix, mapping from pressure nodes to velocity
c        z  = GLL points
c        n  = polynomial degree (velocity space)
c        w  = work array of size 2*n+2
c
c     Currently, this is set up for pressure nodes on the interior GLL pts.
c
c
      real a(0:n,0:n),b(0:n),c(0:n,0:n),d(0:n,0:n),z(0:n)
      real dp(0:n,1:n-1),jp(0:n,1:n-1)
      real w(0:1)
c
      np = n+1
      nm = n-1
      n2 = n-2
c
      call zwgll (z,b,np)
c
      do i=0,n
         call fd_weights_full(z(i),z,n,1,w)
         do j=0,n
            d(i,j) = w(j+np)                   !  Derivative matrix
         enddo
      enddo
c
      do i=0,n
         call fd_weights_full(z(i),z(1),n2,1,w(1))
         do j=1,nm
            jp(i,j) = w(j   )                  !  Interpolation matrix
            dp(i,j) = w(j+nm)                  !  Derivative    matrix
         enddo
      enddo
c
      call rzero(a,np*np)
      do j=0,n
      do i=0,n
         do k=0,n
            a(i,j) = a(i,j) + d(k,i)*b(k)*d(k,j)
         enddo
         c(i,j) = b(i)*d(i,j)
      enddo
      enddo
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
c     call outmat(c,n+1,m+1,'fdw',n)
      return
      end
c-----------------------------------------------------------------------
