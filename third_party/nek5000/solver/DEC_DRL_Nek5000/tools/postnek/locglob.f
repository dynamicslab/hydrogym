c-----------------------------------------------------------------------
      subroutine locglob
c
c     Establish local-to-global numbering -
c
c     Algorithm is proceeds as follows.
c
c       1.  First, number SEM vertices using lexicographical sort.
c           (Alternative would be to read the .map file)
c
c       2.  Insert edge/face/volume data, as currently done in nek5000
c
c
c
c     NOTES:  . Periodic boundary conditions NOT supported
c             . Does not currently support message passing parallelism
c             . jacobians/metrics are used for workspace, 
c                    so these are recomputed before return
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (l8=8*nelm)
      common /lglob1/ nvert,vertex (l8)
      common /lglob2/ iwk(0:36*nelm)
      integer vertex
      integer e
c
      logical ifcenter
c
      call  locglob_lexico(vertex,nvert) ! Get vertex numbering
c
      ifcenter = .true.   ! Enumerate spectral element centers
      if (if3d) then
         i = 24*nelm
         call setvert3d(iglob,nglob,nx,nel,vertex,skpdat
     $                 ,ifcenter,iwk,iwk(i),iwk,iwk(i))
      else
         i = 27*nelm
         call setvert2d(iglob,nglob,nx,nel,vertex,ifcenter,iwk,iwk(i))
      endif
      write(6,*) 'THIS is nglob',nglob
c
      return
      end
c-----------------------------------------------------------------------
      subroutine locglobm(jglb,nglb,mx,mel)
c
c     Establish local-to-global numbering -
c
c     Algorithm is proceeds as follows.
c
c       1.  First, number SEM vertices using lexicographical sort.
c           (Alternative would be to read the .map file)
c
c       2.  Insert edge/face/volume data, as currently done in nek5000
c
c
c
c     NOTES:  . Periodic boundary conditions NOT supported
c             . Does not currently support message passing parallelism
c             . jacobians/metrics are used for workspace, 
c                    so these are recomputed before return
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer jglb(1)
      parameter (l8=8*nelm)
      common /lglob1/ nvert,vertex (l8)
      common /lglob2/ iwk(0:36*nelm)
      integer vertex
      integer e
c
      logical ifcenter
c
      call  locglob_lexico(vertex,nvert) ! Get vertex numbering
      write(6,*) 'NVERT LEXICO:',nvert,mx,mel
c
c     l = 0
c     do e=1,nel
c     do i=1,8
c        l = l+1
c        write(6,*) vertex(l),l,i,e,' VERTEX',nvert
c     enddo
c     enddo
c
c
      ifcenter = .true.   ! Enumerate spectral element centers
      if (if3d) then
         i = 24*nelm
         call dsset(mx,mx,mx)
         call setvert3d(jglb,nglb,mx,mel,vertex,skpdat
     $                 ,ifcenter,iwk,iwk(i),iwk,iwk(i))
         call dsset(nx,ny,nz)
      else
         i = 27*nelm
         call setvert2d(jglb,nglb,mx,mel,vertex,ifcenter,iwk,iwk(i))
      endif
      write(6,*) 'THIS is nglb',nglb,mx,mel
c
      mtot = mel*mx**ndim
c     do i=1,mtot
c        write(71,*) i,jglb(i),nglb
c     enddo
c     call exit
c     l = 0
c     do e=1,nel
c     do i=1,8
c        l = l+1
c        write(6,*) vertex(l),l,i,e,' VERTX2',nvert
c     enddo
c     enddo
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine locglob_lexico(locglob,nglb)
C
C     Establish local global numbering   pff   7/5/94
C
C     NOTES:  . Periodic boundary conditions NOT supported
C             . Does not currently support message passing parallelism
C
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      integer locglob(1)
c
      parameter (l8=8*nelm)
      common /ctmp0/ wx (l8),wy (l8),wz (l8),tol (l8)
     $             , loc(l8),ind(l8),ninseg(l8),ifseg(l8)
      logical ifseg
c
      integer e,f,ednum(8)
      save ednum
      data ednum  / 1,2,4,3 , 5,6,8,7 /
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.gt.0) return
      icalld = 1
c
      nxyz   = 2**ndim          !  Only sorting SEM vertices
      ntot   = nxyz*nel
c
      j = 0
      do e=1,nel
      do i=1,nxyz
         j         = j + 1
         ied       = ednum(i)   !  Ed, again.
         loc (j)   = j          !  Establish initial pointers
         wx  (j)   = x(e,ied)   !  Map Ed vertices to sortable list
         wy  (j)   = y(e,ied)
         wz  (j)   = z(e,ied)
         ifseg (j) = .false.
      enddo
      enddo
c
      mx = 2
      call settol(tol,mx,nel,wx,wy,wz,if3d) ! Set geometric tolerances
c
      nseg        = 1
      ifseg(1)    = .true.
      ninseg(1)   = ntot
c
      do ipass = 1,ndim
      do j=1,ndim
c       Sort within each segment
         write(6,*) 'locglob:',j,nseg,ntot
         i=1
         do iseg=1,nseg
c
            if  (j.eq.1) then
               call sort (wx(i),ind,ninseg(iseg))    ! Sort by x
            elseif  (j.eq.2) then
               call sort (wy(i),ind,ninseg(iseg))    !  "   "  y
            else
               call sort (wz(i),ind,ninseg(iseg))    !  "   "  z
            endif
c
            call iswap (loc (i),locglob,ind,ninseg(iseg)) ! Swap corresponding
            call swap  (tol (i),locglob,ind,ninseg(iseg)) !      data
            call swap  (wx  (i),locglob,ind,ninseg(iseg))
            call swap  (wy  (i),locglob,ind,ninseg(iseg)) ! locglob() is a
            if (if3d)                                     ! work array
     $      call swap  (wz  (i),locglob,ind,ninseg(iseg))
c
            i  =   i + ninseg(iseg)
c
         enddo
c
         if (j.eq.1) then      !  Check for jumps in current coordinate 
           do i=2,ntot
           if (abs(wx(i)-wx(i-1)).gt.min(tol(i),tol(i-1)))
     $        ifseg(i)=.true.
           enddo
         elseif (j.eq.2) then
           do i=2,ntot
           if (abs(wy(i)-wy(i-1)).gt.min(tol(i),tol(i-1)))
     $        ifseg(i)=.true.
           enddo
         elseif (if3d) then
           do i=2,ntot
           if (abs(wz(i)-wz(i-1)).gt.min(tol(i),tol(i-1)))
     $        ifseg(i)=.true.
           enddo
         endif
c
         nseg = 0              !  Count up number of different segments
         do i=1,ntot
            if (ifseg(i)) then
               nseg = nseg+1
               ninseg(nseg) = 1
            else
               ninseg(nseg) = ninseg(nseg) + 1
            endif
         enddo
c
      enddo
      enddo
C
C     Assign global node numbers (now sorted lexigraphically!)
C
      ig = 0
      do i=1,ntot
         if (ifseg(i)) ig=ig+1
         locglob(loc(i)) = ig
      enddo
c     do i=1,ntot
c        write(61,*) i,loc(i),locglob(loc(i)),locglob(i)
c     enddo
c
      nglb = ig
c
      write(s,6) nseg,nglb,ntot
    6 format('done locglob_lexico:',3i9,'$')
      call prs(s)
      return
      end
c-----------------------------------------------------------------------
      subroutine settol(tol,nx,nel,x,y,z,if3d)
      real tol(1),x(1),y(1),z(1)
      logical if3d
c
      if (if3d) then
         call settol3(tol,nx,nel,x,y,z)
      else
         call settol2(tol,nx,nel,x,y)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine settol3(tol,nx,nel,x,y,z)
c
      real tol(1)
      real x  (nx,nx,nx,nel)
      real y  (nx,nx,nx,nel)
      real z  (nx,nx,nx,nel)
c
      l = 0
      do ie=1,nel
         do k=1,nx
            kl = max(k-1,1)
            kr = min(k+1,nx)
            do j=1,nx
               jl = max(j-1,1)
               jr = min(j+1,nx)
               do i=1,nx
                  l  = l+1
                  il = max(i-1,1)
                  ir = min(i+1,nx)
                  tol(l) = .001*( abs(x(ir,jr,kr,ie)-x(il,jl,kl,ie))
     $                           +abs(y(ir,jr,kr,ie)-y(il,jl,kl,ie))
     $                           +abs(z(ir,jr,kr,ie)-z(il,jl,kl,ie)) )
               enddo
            enddo
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine settol2(tol,nx,nel,x,y)
c
      real tol(1)
      real x  (nx,nx,nel)
      real y  (nx,nx,nel)
c
      l = 0
      do ie=1,nel
         do j=1,nx
            jl = max(j-1,1)
            jr = min(j+1,nx)
            do i=1,nx
               l  = l+1
               il = max(i-1,1)
               ir = min(i+1,nx)
               tol(l) = .001*( abs(x(ir,jr,ie)-x(il,jl,ie))
     $                        +abs(y(ir,jr,ie)-y(il,jl,ie)) )
            enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dssum(uin,wk)
      include 'basics.inc'
      include 'basicsp.inc'
c
      real uin(1),wk(1)
c
      ntot=nx*ny*nz*nel
      call rzero(wk,ntot)
c
      do i=1,ntot
         wk(iglob(i)) = wk(iglob(i)) + uin(i)
      enddo
c
      do i=1,ntot
         uin(i) = wk(iglob(i))
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dsavg(uin,wk)
      include 'basics.inc'
      include 'basicsp.inc'
c
      real uin(1),wk(1)
c
      ntot=nx*ny*nz*nel
      call rzero(wk,ntot)
c
      do i=1,ntot
         wk(iglob(i)) = wk(iglob(i)) + mult(i)*uin(i)
      enddo
c
      do i=1,ntot
         uin(i) = wk(iglob(i))
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine setvert2d(glo_num,ngv,mx,nel,vertex,ifcenter,s,snum)
c
c     set up gsexch interfaces for direct stiffness summation.  pff 2/3/98
c     hmt revisited 12/10/01
c
c
      integer glo_num(1),vertex(0:1,0:1,nel),ngv,mx
      logical ifcenter
      integer s(3,9,nel),snum(9*nel)
c
c     4 vertices, 4 edges, 1 center
c
c
      integer gvf(6),facet(6),aa(6),key(3),key2(0:3)

      ndim = 2
c
c     do i=1,nel
c        write(6,1) (vertex(k,0,i),k=0,3),i,nel
c  1     format(4i9,2i5,' VERTEX')
c     enddo
c
c     memory check...
c
      my=mx
c
      key(1)=1
      key(2)=2
      key(3)=3
c
c     build big structure list, vertices, edges, faces
c
      call izero(s,27*nel)
      do ie=1,nel
         l=0
         do j=0,1           !VERTICES
         do i=0,1
            l=l+1
            s(1,l,ie) = vertex(i,j,ie) 
            s(2,l,ie) = vertex(i,j,ie)
            s(3,l,ie) = 1   ! vertex
         enddo
         enddo
c
         do j=0,1           !EDGES
            l=l+1
            do i=0,1                                    ! x-edge
               i0 = min(vertex(0,j,ie),vertex(1,j,ie)) 
               i1 = max(vertex(0,j,ie),vertex(1,j,ie))
               s(i+1,l,ie) = i*i1 + (1-i)*i0            ! store w/ min leading
            enddo
            s(3,l,ie) = 2   ! vertex
         enddo
         do j=0,1
            l=l+1
            do i=0,1                                    ! y-edge
               i0 = min(vertex(j,0,ie),vertex(j,1,ie)) 
               i1 = max(vertex(j,0,ie),vertex(j,1,ie))
               s(i+1,l,ie) = i*i1 + (1-i)*i0            ! store w/ min leading
            enddo
            s(3,l,ie) = 2   ! vertex
         enddo
c
         imin = vertex(0,0,ie)
         do j=0,1           !CENTER
         do i=0,1
            if (vertex(i,j,ie).le.imin) then
               i0 = 1-i
               j0 = 1-j
               iopp = vertex(i0,j0,ie)
               imin = vertex(i ,j ,ie)
            endif
         enddo
         enddo
         l=l+1
         s(1,l,ie) = imin     ! store w/ min leading
c        s(2,l,ie) = iopp     ! store w/ min leading
c        s(3,l,ie) = 0        ! center
         s(2,l,ie) = 0        ! store w/ min leading
         s(3,l,ie) = ie       ! center
c
      enddo
c
c     compute running tally of number of dof's 
c
      key2(0) = (mx-2)**2
      key2(1) = 1
      key2(2) = mx-2
c
      nstruct = 9*nel
      call irank_vec_tally(snum,n_tally,s,3,nstruct,key,3,key2,aa)
c
c     Assign vertices.  Assume hypercube ordering.
c
      do ie=1,nel
         ieg = ie ! lglel(ie,node)
         l=0
         do j=0,1
         do i=0,1
            l=l+1
            il  = 1 + (mx-1)*i + mx*(mx-1)*j
            ile = il + mx*mx*(ie-1)
            glo_num(ile)   = s(1,l,ieg)+1
         enddo
         enddo
      enddo
c
      n_on_edge = mx-2
      do ie=1,nel
         ieg = ie ! lglel(ie,node)
         l=2**ndim
c
c        Edges 1-2
         do j=0,1                                            ! x-edge
            l   = l+1
            igv = s(1,l,ieg)
            i0  = mx*(mx-1)*j
            i0e = i0 + mx*mx*(ie-1)
            if (glo_num(i0e+1).gt.glo_num(i0e+mx)) then
               do i=2,mx-1                                   ! std forward case
                  glo_num(i0e+i) = igv + i-1
               enddo
            else
               do i=2,mx-1                                   ! backward case
                  glo_num(i0e+i) = igv + 1 + n_on_edge-(i-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
c
c        Edges 3-4
         do i=0,1
            l   = l+1
            igv = s(1,l,ieg)
            i0  = 1+(mx-1)*i
            i0e = i0 + mx*mx*(ie-1)
            if (glo_num(i0e).gt.glo_num(i0e+mx*(mx-1))) then
               do j=2,mx-1                                   ! std forward case
                  glo_num(i0e+(j-1)*mx) = igv + j-1
               enddo
            else
               do j=2,mx-1                                   ! backward case
                  glo_num(i0e+(j-1)*mx) = igv + 1 + n_on_edge-(j-1)
               enddo
            endif
         enddo
      enddo
c
      if(ifcenter) then 
         do ie=1,nel           ! Centers
            ieg = ie ! lglel(ie,node)
            igv = s(1,9,ieg)
            k   = 0
            do j=2,mx-1
               do i=2,mx-1
                  ii = i+(j-1)*mx + mx*mx*(ie-1)
                  k  = k+1
                  glo_num(ii) = igv + k
               enddo
            enddo
         enddo
      else
         do ie=1,nel           ! Centers
            ieg = ie ! lglel(ie,node)
            igv = s(1,9,ieg)
            k   = 0
            do j=2,mx-1
               do i=2,mx-1
                  ii = i+(j-1)*mx + mx*mx*(ie-1)
                  k  = k+1
                  glo_num(ii) = 0
               enddo
            enddo
         enddo
      endif
c
      ngv   = n_tally
c
c
c     Quick check on maximum #dofs:
      m    = mx*mx*nel
      ngvm = iglmax(glo_num,m)
      if (nid.eq.0) write(6,*) 'NUM Coarse verts:',ngvm,ngv,n_tally
c
      return
      end
c-----------------------------------------------------------------------
      subroutine setvert3d(glo_num,ngv,mx,nel,vertex,skpdat
     $                    ,ifcenter,edge,enum,face,fnum)
c
c     NOTE: edge == face,  enum == fnum
c
c     set up gsexch interfaces for direct stiffness summation.  pff 2/3/98
c     hmt revisited 12/10/01
c
c
      integer glo_num(1),vertex(0:1,0:1,0:1,nel),ngv,mx,skpdat(6,6)
      logical ifcenter
c
      integer icface (4,10)
      save    icface
      data    icface/ 1,3,5,7, 2,4,6,8,    ! 3D
     $                1,2,5,6, 3,4,7,8,    ! 3D
     $                1,2,3,4, 5,6,7,8,    ! 3D
     $                1,3,0,0, 2,4,0,0,    ! 2D
     $                1,2,0,0, 3,4,0,0  /  ! 2D
C
c
c     NOTE: The following variables are assumed to occupy the
c           same memory locations:
c
c           edge == face,  enum == fnum
c
      integer  edge(0:1,0:1,0:1,3,nel),enum(12,nel)
      integer  face(3,6,nel),fnum(6,nel)
c
      integer gvf(4),ind(4),facet(4),aa(3),key(3)
      logical ifij
c
c     memory check...
c
      my   = mx
      mz   = mx
      mxyz = mx*my*mz
c
c     if (nid.eq.0) write(6,*) 'in setvert3d',mx,my,mz
c
      key(1)=1
      key(2)=2
      key(3)=3
c
c     Count number of unique vertices
      ndim = 3
      nlv  = 2**ndim
      ngv  = ivlmax(vertex,nlv*nel)
c     write(6,*) nlv,nel,ngv,(vertex(k,0,0,1),k=0,7),' NGV??'
c
c     Assign hypercube ordering of vertices.
      do ie=1,nel
         ieg = ie ! lglel(ie,node)
         do k=0,1
         do j=0,1
         do i=0,1
c           Local to global node number (vertex)
            il  = 1 + (mx-1)*i + mx*(my-1)*j + mx*my*(mz-1)*k
            ile = il + mx*my*mz*(ie-1)
            glo_num(ile) = vertex(i,j,k,ieg)
c           if (mx.lt.12) write(51,*) glo_num(ile),ile,i,j,k,ieg
         enddo
         enddo
         enddo
      enddo
      call outi6(glo_num,' aaa',ngv)
      if (mx.eq.2) return
c
c     Assign edge labels by bounding vertices.  
      do ie=1,nel
         do k=0,1
         do j=0,1
         do i=0,1
            edge(i,j,k,1,ie) = vertex(i,j,k,ie)  ! r-edge
            edge(j,i,k,2,ie) = vertex(i,j,k,ie)  ! s-edge
            edge(k,i,j,3,ie) = vertex(i,j,k,ie)  ! t-edge
         enddo
         enddo
         enddo
      enddo
c
c     Sort edges by bounding vertices.
      do i=0,12*nel
         if (edge(0,i,0,1,1).gt.edge(1,i,0,1,1)) then
            kswap = edge(0,i,0,1,1)
            edge(0,i,0,1,1) = edge(1,i,0,1,1)
            edge(1,i,0,1,1) = kswap
         endif
      enddo
c     old ...
c     Sort edges by bounding vertices.
c
c      do i=1,12*nel
c         if (edge(0,i,1,1,1).gt.edge(1,i,1,1,1)) then
c            kswap = edge(0,i,1,1,1)
c            edge(0,i,1,1,1) = edge(1,i,1,1,1)
c            edge(1,i,1,1,1) = kswap
c         endif
c      enddo
c     Assign a number (rank) to each unique edge
      if (mx.eq.12) then
        call irank_vec(enum,n_unique_edges,edge,2,12*nel,key,2,aa)
      else
        call irank_vec(enum,n_unique_edges,edge,2,12*nel,key,2,aa)
c       call irank_vecx(enum,n_unique_edges,edge,2,12*nel,key,2,aa)
      endif
      nel12 = 12*nel
      write(6,*) n_unique_edges,nel12,' NEDGE!! '
c     if (mx.lt.12) then
c        do ie=1,nel
c        do k=1,12
c           write(81,*) k,ie,enum(k,ie),n_unique_edges
c        enddo
c        enddo
c        stop
c     endif
c
c= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
c     Assign global vertex numbers to SEM nodes on each edge
      n_on_edge = mx-2
      do ie=1,nel
c
         ieg = ie ! lglel(ie,node)
         iedg_loc = 0
c
c        Edges 1-4
         do k=0,1
         do j=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,ieg)-1)
            i0  = mx*(my-1)*j + mx*my*(mz-1)*k
            i0e = i0 + mxyz*(ie-1)
            if (glo_num(i0e+1).lt.glo_num(i0e+mx)) then
               do i=2,mx-1                                   ! std forward case
                  glo_num(i0e+i) = igv + i-1
               enddo
            else
               do i=2,mx-1                                   ! backward case
                  glo_num(i0e+i) = igv + 1 + n_on_edge-(i-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
c
c        Edges 5-8
         do k=0,1
         do i=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,ieg)-1)
            i0  = 1+(mx-1)*i + mx*my*(mz-1)*k
            i0e = i0 + mxyz*(ie-1)
            if (glo_num(i0e).lt.glo_num(i0e+mx*(mx-1))) then
               do j=2,mx-1                                   ! std forward case
                  glo_num(i0e+(j-1)*mx) = igv + j-1
               enddo
            else
               do j=2,mx-1                                   ! backward case
                  glo_num(i0e+(j-1)*mx) = igv + 1 + n_on_edge-(j-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
c
c        Edges 9-12
         do j=0,1
         do i=0,1
            igv = ngv + n_on_edge*(enum(iedg_loc+1,ieg)-1)
            i0  = 1 + (mx-1)*i + mx*(mx-1)*j
            i0e = i0 + mxyz*(ie-1)
            if (glo_num(i0e).lt.glo_num(i0e+mx*mx*(mx-1))) then
               do k=2,mx-1                                   ! std forward case
                  glo_num(i0e+(k-1)*mx*mx) = igv + k-1
               enddo
            else
               do k=2,mx-1                                   ! backward case
                  glo_num(i0e+(k-1)*mx*mx) = igv + 1 + n_on_edge-(k-1)
               enddo
            endif
            iedg_loc = iedg_loc + 1
         enddo
         enddo
      enddo
c
c     Currently assigned number of vertices
      ngvv  = ngv
      ngv   = ngv + n_unique_edges*n_on_edge
      call outi6(glo_num,' baa',ngv)
c
c
c     Assign faces by 3-tuples 
c
c     (The following variables all take the symmetric 
c     notation of IFACE as arguments:)
c
c     ICFACE(i,IFACE) -   Gives the 4 vertices which reside on face IFACE
c                         as depicted below, e.g. ICFACE(i,2)=2,4,6,8.
c
c                        3+-----+4    ^ Y
c                        /  2  /|     |
c     Edge 1 extends    /     / |     |
c       from vertex   7+-----+8 +2    +----> X
c       1 to 2.        |  4  | /     /
c                      |     |/     /
c                     5+-----+6    Z
c                         3
c
c
      nfaces=ndim*2
      ncrnr =2**(ndim-1)
      do ie=1,nel
         do ifac=1,nfaces
            do icrn=1,ncrnr
               i                  = icface(icrn,ifac)-1
               facet(icrn)        = vertex(i,0,0,ie)
            enddo
            call isort(facet,gvf,ncrnr)
            call icopy(face(1,ifac,ie),facet,ncrnr-1)
         enddo
      enddo
c
c     Assign a number (rank) to each unique face
      call irank_vec(fnum,n_unique_faces,face,3,6*nel,key,3,aa)
c     write(6,*) (fnum(k,1),k=1,n_unique_faces),' FNUM'
c
c     Now assign global node numbers on the interior of each face
c
      call dsset (mx,my,mz)
      do ie=1,nel
       ieg= ie ! lglel(ie,node)
       do iface=1,nfaces
         i0 = skpdat(iface,1)  !  Note, this indexing
         i1 = skpdat(iface,2)  !  is backwards from 
         is = skpdat(iface,3)  !  Nek5000  ( pff, 8/8/04 )
         j0 = skpdat(iface,4)
         j1 = skpdat(iface,5)
         js = skpdat(iface,6)
c
c        On each face, count from minimum global vertex number,
c        towards smallest adjacent vertex number.  e.g., suppose
c        the face is defined by the following global vertex numbers:
c
c
c                    11+--------+81
c                      |c      d|
c                      |        |
c                      |        |
c                      |a      b|
c                    15+--------+62
c                          
c        We would count from c-->a, then towards d.
c
         gvf(1) = glo_num(i0+mx*(j0-1)+mxyz*(ie-1))
         gvf(2) = glo_num(i1+mx*(j0-1)+mxyz*(ie-1))
         gvf(3) = glo_num(i0+mx*(j1-1)+mxyz*(ie-1))
         gvf(4) = glo_num(i1+mx*(j1-1)+mxyz*(ie-1))
c
         call irank(gvf,ind,4)
c
c        ind(1) tells which element of gvf() is smallest.
c
         ifij = .false.
         if (ind(1).eq.1) then
            idir =  1
            jdir =  1
            if (gvf(2).lt.gvf(3)) ifij = .true.
         elseif (ind(1).eq.2) then
            idir = -1
            jdir =  1
            if (gvf(1).lt.gvf(4)) ifij = .true.
         elseif (ind(1).eq.3) then
            idir =  1
            jdir = -1
            if (gvf(4).lt.gvf(1)) ifij = .true.
         elseif (ind(1).eq.4) then
            idir = -1
            jdir = -1
            if (gvf(3).lt.gvf(2)) ifij = .true.
         endif
c
         if (idir.lt.0) then
            it=i0
            i0=i1
            i1=it
            is=-is
         endif
c
         if (jdir.lt.0) then
            jt=j0
            j0=j1
            j1=jt
            js=-js
         endif
c
         mxx = mx*mx
         n_on_face = (mx-2)*(my-2)
         ig0 = ngv + n_on_face*(fnum(iface,ieg)-1)
c        write(6,*) mx,my,n_on_face,ngv,ig0,'  N ON FAC1'
         if (ifij) then
            k=0
            l=0
            do j=j0,j1,js
            do i=i0,i1,is
               k=k+1
c              this is a serious kludge to stay on the face interior
               if (k.gt.mx.and.k.lt.mxx-mx .and.
     $            mod(k,mx).ne.1.and.mod(k,mx).ne.0) then
c                 interior
                  l = l+1
                  glo_num(i+mx*(j-1)+mxyz*(ie-1)) = l + ig0
               endif
            enddo
            enddo
         else
            k=0
            l=0
            do i=i0,i1,is
            do j=j0,j1,js
               k=k+1
c              this is a serious kludge to stay on the face interior
               if (k.gt.mx.and.k.lt.mxx-mx .and.
     $            mod(k,mx).ne.1.and.mod(k,mx).ne.0) then
c                 interior
                  l = l+1
                  glo_num(i+mx*(j-1)+mxyz*(ie-1)) = l + ig0
               endif
            enddo
            enddo
         endif
       enddo
      enddo
c
c     Finally,  number interiors  
c     ngvs := number of global vertices on surface of subdomains
c
      ngve  = ngv
      ngv   = ngv + n_unique_faces*n_on_face
      ngvs  = ngv
c     write(6,*) ngve,ngv,n_unique_faces,n_on_face,'  N ON FACE'
      call outi6(glo_num,' caa',ngv)
c
      n_in_interior = (mx-2)*(my-2)*(mz-2)
      if (ifcenter) then
         do ie=1,nel
            ig0 = ngv + n_in_interior*(ie-1) ! (lglel(ie,node)-1)
            l = 0
            do k=2,mz-1
               do j=2,my-1
                  do i=2,mx-1
                     l = l+1
                     glo_num(i+mx*(j-1)+mx*my*(k-1)+mxyz*(ie-1)) = ig0+l
                  enddo
               enddo
            enddo
         enddo
      else
         do ie=1,nel
            ig0 = ngv + n_in_interior*(ie-1) ! (lglel(ie,node)-1)
            l = 0
            do k=2,mz-1
               do j=2,my-1
                  do i=2,mx-1
                     l = l+1
                     glo_num(i+mx*(j-1)+mx*my*(k-1)+mxyz*(ie-1)) = 0
                  enddo
               enddo
            enddo
         enddo
      endif
c
      ngv = ngv + n_in_interior*nel
      call outi6(glo_num,' daa',ngv)
c
c     Quick check on maximum #dofs:
      m    = mxyz*nel
      ngvm = iglmax(glo_num,m)
      if (nid.eq.0) write(6,*)
      if (nid.eq.0) write(6,1) mx,ngvv,ngve,ngvs,ngv,ngvm
      if (nid.eq.0) write(6,1) m,mxyz,nel,n_in_interior
      if (nid.eq.0) write(6,*)
    1 format('setupds3d:',9i9)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outi6(glo_num,name4,ngv)
c
      character*4 name4
      integer glo_num(1)
c
      return
      write(6,*)
      write(6,*) name4,ngv
      write(6,1) (glo_num(k),k=1,64)
   1  format(16i3)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine irank_vec_tally(ind,nn,a,m,n,key,nkey,key2,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is destroyed
c               a(1,i) tally of preceding structure values
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),key2(0:3),aa(m)
      logical iftuple_ianeb,a_ne_b
c
c
      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)
c     do i=1,n
c        write(6,*) i,' sort:',(a(k,i),k=1,3)
c     enddo
c
c
c     Find unique a's
c
      call icopy(aa,a,m)
      nn=1
      mm=0
c
      a(1,1) = nn
      a(2,1)=ind(1)
      a(3,1)=mm
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
         if (a_ne_b) then              ! new structure
            ms = aa(3)                 ! structure type
            if (aa(2).eq.0) ms = aa(2) ! structure type
            mm = mm+key2(ms)           ! n dofs
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         a(1,i) = nn
         a(2,i) = ind(i)
         a(3,i) = mm
      enddo
      ms = aa(3)
      if (aa(2).eq.0) ms = aa(2) ! structure type
      nn = mm+key2(ms)
c
c     Set ind() to rank
c
      do i=1,n
         iold=a(2,i)
         ind(iold) = a(1,i)
      enddo
c
c     Set a1() to number of preceding dofs
c
      do i=1,n
         iold=a(2,i)
         a(1,iold) = a(3,i)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine irank_vecx(ind,nn,a,m,n,key,nkey,aa)
c
c     Compute rank of each unique entry a(1,i) 
c
c     Output:   ind(i)  i=1,...,n    (global) rank of entry a(*,i)
c               nn  = max(rank)
c               a(j,i) is destroyed
c
c     Input:    a(j,i) j=1,...,m;  i=1,...,n  
c               m      :   leading dim. of v  (ldv must be .ge. m)
c               key    :   sort key
c               nkey   :   
c
c     Although not mandatory, this ranking procedure is probably
c     most effectively employed when the keys are pre-sorted. Thus,
c     the option is provided to sort vi() prior to the ranking.
c
c
      integer ind(n),a(m,n)
      integer key(nkey),aa(m)
      logical iftuple_ianeb,a_ne_b
c
      if (m.eq.1) then
c
         write(6,*) 
     $        'WARNING: For single key, not clear that rank is unique!'
         call irank(a,ind,n)
         return
      endif
c
c
      nk = min(nkey,m)
      call ituple_sort(a,m,n,key,nk,ind,aa)
c     do i=1,n
c        write(92,*) i,ind(i)
c     enddo
c
c     Find unique a's
c
      nn=1
c
      call icopy(aa,a,m)
      a(1,1) = nn
      a(2,1)=ind(1)
c
      do i=2,n
         a_ne_b = iftuple_ianeb(aa,a(1,i),key,nk)
c        write(6,1) nn,i,a_ne_b,aa,(a(k,i),k=1,nk)
c  1     format(2i7,1x,l4,1x,12i6)
         if (a_ne_b) then
            call icopy(aa,a(1,i),m)
            nn = nn+1
         endif
         a(1,i) = nn
         a(2,i) = ind(i)
         i10 = min(i,10)
c        do j=1,i10
c           write(94,*) j,ind(j),a(2,j),a(1,j)
c        enddo
         write(94,*)
      enddo
c     do i=1,n
c        write(93,*) i,ind(i),a(2,i),a(1,i),m
c     enddo
c
c     Set ind() to rank
c
      do i=1,n
         iold=a(2,i)
         ind(iold) = a(1,i)
c        write(91,*) i,iold,a(1,i),ind(i)
      enddo
      call exitt
c
      return
      end
c-----------------------------------------------------------------------
