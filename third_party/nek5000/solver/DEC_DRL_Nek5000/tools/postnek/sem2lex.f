


David,


This is the _serial_ version of this routine -- it assumes
that all of the data can fit in memory.


c-----------------------------------------------------------------------
      subroutine sem2lex(ul,us,nelx,nely,nelz)
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
c
      real ul(nx,nelx,ny,nely,nz,nelz)
      real us(nx,ny,nz,nel)
c     
      le = 0
      do ke=1,nelz
      do je=1,nely
      do ie=1,nelx
         le = le+1
         do iz=1,nz
         do iy=1,ny
         do ix=1,nx
            ul(ix,ie,iy,je,iz,ke) = us(ix,iy,iz,le)
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


A parallel version of this routine would look something like:


c-----------------------------------------------------------------------
      subroutine sem2lex(ul,us,nelx,nely,nelz)
c
      include 'SIZE'
      include 'PARALLEL'  ! need lglel, gllel
c
      real ul(nx1,nelx,ny1,nely,nz1,nelz)
      real us(nx1,ny1,nz1,nelv)
c
      integer eg
c     
      eg = 0
      do ke=1,nelz
      do je=1,nely
      do ie=1,nelx
         eg = eg+1
         if (gllnid(eg).eq.nid) then  ! element eg is on this processor
            le = gllel(ge)            ! le is local element number
            do iz=1,nz1
            do iy=1,ny1
            do ix=1,nx1
               ul(ix,ie,iy,je,iz,ke) = us(ix,iy,iz,le) ! map to global name space
            enddo
            enddo
            enddo
         endif
      enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------


The problem with the above routine is that you generally will not
have enough space on one processor to hold all of ul() because

               nelv << nelx*nely*nelx 

The latter are _global_ index ranges (associated with the _mesh_),
the former is the index range that is local to processor p.

The line that says "map to global name space" is where the 
spectral element format should be written into the globally
indexed file system.

Please let me know if you need further clarification.

Thanks,

Paul








