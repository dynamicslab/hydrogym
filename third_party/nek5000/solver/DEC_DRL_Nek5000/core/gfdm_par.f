c-----------------------------------------------------------------------
      subroutine g25d_init
      include 'SIZE'
c
      write(6,*) nid,' no g25d, EXIT.'
      call exitt
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gfdm_init(nx,ny,nz,ifemati,kwave2)
c
c     This is the driver for the Global Fast Diagonalization Method
c
c     By the time this routine is called, it is assumed that the
c     spectral elements are already mapped as clusters of pencils
c     in the "primary" direction, which is user selectable by setting
c     Nelx, Nely, or Nelz (resp., p116,p117,p118) to a negative number.
c
c     Note that the processor-to-element mapping must be established
c     early, that is, prior to reading the mesh data.   Consequently,
c     it is necessary to establish  nelx, nely, nelz, and ifgfdm=.true.
c     before the mesh read takes place.   Thus, those variables, as
c     well as determination of the primary direction (ip=1,2, or 3,
c     depending on whether x, y, or z is the primary direction) prior
c     to this point in the code.
c
c     The overall structure of this routine is to first establish the
c     mappings for SEM-to-tensor-product form, tensor-product to post-
c     transpose state, etc., then to establish the 1D operators in each
c     of the d directions (d=2 or 3).
c
c
      include 'SIZE'
      include 'PARALLEL'
      include 'ZPER'

      logical ifemati
      real    kwave2

      ifemat = ifemati   ! Flag for Pn-Pn-2 vs Pn-Pn

      call gfdm_check_array_sizes (nx,ny,nz)
      call gfdm_mappings          (nx,ny,nz)
      call gfdm_ops               (nx,ny,nz,kwave2,1)

      return
      end
c-----------------------------------------------------------------------
      subroutine gfdm_check_array_sizes(nx,ny,nz)
      include 'SIZE'
      include 'PARALLEL'
      include 'ZPER'

      if (lelg_sm.lt.lelg.or.ltfdm2.lt.2*nx*ny*nz*nelt) then
         if (nio.eq.0) then
            write(6,*) 'gfdm_array_check fail A:',lelg_sm,ltfdm2,lelg
            write(6,*) 'modify lelg_sm,ltfdm2 in ZPER; makenek clean'
         endif
         nz1 = 1/(nx1-ny1)
         call exitt
      endif

      if (lp_small.lt.np) then
         if (nio.eq.0) then
            write(6,*) 'gfdm_array_check fail B:',lp_small,np
            write(6,*) 'modify lp_small > np in ZPER; makenek clean'
         endif
         call exitt
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine gfdm_mappings(nx,ny,nz) ! Typ., nx2,ny2,nz2
c
c     This routine establishes SEM to FDM mappings for pressure arrays.
c
c  0. The initial mapping is the std. SEM  (nx,ny,nz,nelv)
c
c  1. The first permuation (tpn1) maps the SEM ordering into a standard
c     (global) lexicographical ordering of the points (x by y by z).
c     tpn1() is not used in the execution phase, but simply helps to
c     allow the std. lexicographical ordering to be mapped into subsequent
c     p,s,t (primary, secondary, tertiary) orderings, where p,s,t can be
c     an arbitrary permuation of X,Y,Z.
c
c  2. The second permuation (tpn2) maps the data along "pencils" in the
c     primary direction (ip).  
c
c                   ip=1 ==> X is primary direction.
c                   ip=2 ==> Y is primary direction.
c                   ip=3 ==> Z is primary direction.
c
c     The format after the second permutation is as (m x mp) arrary,
c     where mp is the number of points in the primary direction and
c     m = (ntot/mp).
c
c     The motivation for numbering the primary direction second is
c     that the data is then ready to be partitioned among P processors
c     when the complete exchange is invoked.
c
c  3. The third permuation (tpn3) maps the data along "slabs" in the
c     secondary and tertiary directions, with the primary direction
c     embedded in the middle.  For example, if the dimensions of the
c     lexicographical data in the p,s,t ordering are np,ns,nt, 
c     respectively, then data after applying tpn3() will be stored
c     as an (ns by np~ by nt) array.   Here, np~ is roughly np/P and
c     corresponds to the number of points in the slab on the given
c     processor.    Note that a complete exchange (cex) is required
c     to map the data from pencils in the ip direction so slabs in the
c     (is x it) directions.
c
c     The advantage of having the ip direction buried in the middle
c     is that the application of the 1D transforms in the is and it
c     directions can then be performed as matrix-matrix products without
c     having to (locally) transpose the data.
c
      include 'SIZE'
      include 'PARALLEL'
      include 'ZPER'

c     This next call assigns the std. global tensor-product numbering to
c     each pressure degree-of-freedom, based on an (nelx by nely by nelz)
c     (lexicographical) ordering of the elements.

      call assign_tp_numbering_pres(tpn1,nelx,nely,nelz,nx,ny,nz
     $                                    ,lglel(1),nelv)

      ntot = nelv*nx*ny*nz
      ni   = nelx*nx
      nj   = nely*ny
      nk   = nelz*nz

      ip=pst2lex(1)
      is=pst2lex(2)
      it=pst2lex(3)
c
c     This call re-aligns the global numbering in tpn1 to the
c     "pre-transpose" format required for application of the
c     phys-to-wave transformation along the initial pencils
c     of data.
c
      call reassign_tp_numbering
     $             (tpn3,tpn2,tpn1,mp,ms,mt,ntot,ip,is,it,ni,nj,nk)
c
c
c     To map from SEM to tpn2, use:    swapt_ip(blah,tpn2,ntot),
c                              or:     swapt_2 (utp2,usem,tpn2,ntot)
c
c
c     Swap these orderings in preparation for complete exchange (cex).
c
      call irank   (tpn2,ind23,ntot) !  allow to map from tpn2 to tpn3
      call icopy   (tpn2,ind23,ntot) !  
      call iswap_ip(tpn1,tpn2,ntot) !  allow to map from tpn2 to tpn3
      call iswap_ip(tpn3,tpn2,ntot) !  allow to map from tpn2 to tpn3
c
      m = ntot/mp
      call cex_setup(part_in,mcex,part_out,m,mp,nid,np)
      mc   = ms*mt               ! # points per slab after cex
      mpt  = mcex/mc             ! mcex = # points returned by cex
      call cexi  (ind23,tpn1,m,mp,part_out,part_in,msg_id,isize,nid,np)
      call icopy (tpn1,ind23,mcex)
      call cexi  (ind23,tpn3,m,mp,part_out,part_in,msg_id,isize,nid,np)
      call icopy (tpn3,ind23,mcex)
c
      call irank   (tpn3,ind23,mcex)   ! mapping in transposed state
      call iswap_ip(tpn1,ind23,mcex)   ! original wavenumbers transposed
c
      return
      end
c-----------------------------------------------------------------------
      subroutine assign_tp_numbering_pres(tpn,nex,ney,nez,nx,ny,nz
     $                                    ,lglel,nel)
      integer tpn(nx,ny,nz,nel),lglel(nel)
c
c
      nelxy = nex*ney
      ni    = nex*nx
      nj    = ney*ny
c
      do ie=1,nel
c
c        First, find out where each element is in the global TP array:
c
         ieg = lglel(ie)
         iex = mod1(ieg,nex)
         iez = 1 + (ieg-1)/nelxy
         iey = 1 + (ieg-1)/nex
         iey = mod1(iey,ney)
c        write(6,1) 'El:',ieg,ie,iex,iey,iez
c
c        Next, assign interior (pressure nodes) numbers
c
         i0 = nx*(iex-1)
         j0 = ny*(iey-1)
         k0 = nz*(iez-1)
         do k=1,nz
         do j=1,ny
         do i=1,nx
            ii = i0+i
            jj = j0+j
            kk = k0+k
            ig = ii + ni*(jj-1) + ni*nj*(kk-1)
            tpn(i,j,k,ie) = ig
c           write(6,1) 'tpn',ie,i,j,k,ii,jj,kk,ig
    1       format(a3,2x,15i5)
         enddo
         enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine reassign_tp_numbering
     $             (tpn3,tpn2,tpn1,mp,ms,mt,n,ip,is,it,ni,nj,nk)
      integer tpn3(n),tpn2(n),tpn1(n)
      integer p
c
      include 'SIZE'
c
c
      nij = ni*nj
      njk = nj*nk
      nki = nk*ni
c
c     For tpn2:   (is,it,ip):
c
      if (ip.eq.1) then
         mp      = ni
         istride = njk
         if (is.eq.2) then
            ms      = nj
            mt      = nk
            jstride = 1
            kstride = nj
         else
            ms      = nk
            mt      = nj
            jstride = nk
            kstride = 1
         endif
      elseif (ip.eq.2) then
         mp      = nj
         jstride = nki
         if (is.eq.1) then
            ms      = ni
            mt      = nk
            istride = 1
            kstride = ni
         else
            ms      = nk
            mt      = ni
            istride = nk
            kstride = 1
         endif
      else
         mp      = nk
         kstride = nij
         if (is.eq.2) then
            ms      = ni
            mt      = nj
            jstride = 1
            istride = nj
         else
            ms      = nj
            mt      = ni
            istride = 1
            jstride = ni
         endif
      endif
c
      do p=1,n
         ijk = tpn1(p)
         i   = mod1(ijk,ni)
         k   = 1 + (ijk-1)/nij
         j   = 1 + (ijk-1)/ni
         j   = mod1(j  ,nj)
         ijk_new = 1 + istride*(i-1) + jstride*(j-1) + kstride*(k-1)
         tpn2(p) = ijk_new
c           write(6,1) 'tp2',p,ijk,i,j,k,ijk_new
    1       format(a3,2x,18i5)
      enddo
c
c     For tpn3:   (is,ip,it):
c
      if (it.eq.1) then
         istride = njk
         if (is.eq.2) then
            jstride = 1
            kstride = nj
         else
            kstride = 1
            jstride = nk
         endif
      elseif (it.eq.2) then
         jstride = nki
         if (is.eq.1) then
            istride = 1
            kstride = ni
         else
            kstride = 1
            istride = nk
         endif
      else
         kstride = nij
         if (is.eq.2) then
            jstride = 1
            istride = nj
         else
            istride = 1
            jstride = ni
         endif
      endif
c
      do p=1,n
         ijk = tpn1(p)
         i   = mod1(ijk,ni)
         k   = 1 + (ijk-1)/nij
         j   = 1 + (ijk-1)/ni
         j   = mod1(j  ,nj)
         ijk_new = 1 + istride*(i-1) + jstride*(j-1) + kstride*(k-1)
         tpn3(p) = ijk_new
c           write(6,1) 'tp3',ijk,i,j,k,ijk_new,p,nid
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cex_setup(part_in,nr,part_out,m,n,nid,np)
c
c     This routine sets up a complete exchange of an m x n matrix.
c
c     Here, n>=P is assumed to be the same on each processor.
c     However, m may vary.
c
c     nr is the number of returned values on proc. nid
c
c
c     No re-ordering is performed upon data receipt, as it's assumed
c     that that takes place outside this routine.
c
c     real u(m,n),w(1)
      integer part_in(0:np),part_out(0:np)
c
c     First, determine storage required for result (w)
c
      do i=1,np
         part_in(i) = 0
      enddo
      part_in(nid+1) = m
      call igop(part_in,part_out,'+  ',np+1)
c
c     Next, determine the partition of the incoming data (that is
c     to be transposed).  We assume that u(m,n) is being partitioned
c     along its second axis, such that roughly (n/P) columns will be
c     sent to each processor.
c
c     This numbering scheme is designed so that node 0 is lightly loaded
c
c     This is a stupid dealing algorithm.
c
      do i=0,np
         part_out(i) = 0
      enddo
c
      k = np
      do i=1,n
         part_out(k) = part_out(k)+1
         k=k-1
         if (k.eq.0) k = np
      enddo
c
c
c     Convert counts to pointers  -  note:  part_out simply counts columns
c                                           part_in counts memory locations
c
      ncol_in = part_out(nid+1)
      part_out (0) = 1
      part_in(0) = 1
      do i=1,np
         part_out (i) = part_out (i-1) + part_out (i)
         part_in(i) = part_in(i-1) + part_in(i)*ncol_in
      enddo
c
      nr = part_in(np)-part_in(0)    ! number of returned values
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cexr(w,u,m,n,part_out,part_in,msg_id,wdsize,nid,np)
c
c     Complete exchange of an m x n matrix.   REAL version
c
c     Here, n is assumed to be the same on each processor.
c     However, m may vary.
c
c     No re-ordering is performed upon data receipt, as it's assumed
c     that that takes place outside this routine.
c
c     include 'mpif.h'
c
      real u(m,n),w(1)
      integer part_out(0:np),part_in(0:np)
      integer msg_id(0:np,2)
      integer wdsize
c
c
c     Post receives
c
      do k=0,np-1
         msgtag = 555+k
         j0 = part_in(k  )
         j1 = part_in(k+1)
         len = j1-j0
         if (k.ne.nid) then
            msg_id(k,1) = irecv(msgtag,w(j0),len*wdsize)
         else
            l0 = j0
         endif
      enddo
c
c     Synchronize, so we can use forced message types
c     call nekgsync()
c
c     Call csend, using shift
c
      msgtag = 555+nid
c
      do kk=0,np-1
c        k  = xor(nid,kk)
         k  = nid+kk
         k  = mod(k,np)
         j0 = part_out(k  )
         j1 = part_out(k+1)
         len = m*(j1-j0)
         if (k.ne.nid) then
            msg_id(k,2) = isend(msgtag,u(1,j0),len*wdsize,k,0)
         else
            do l=0,len-1
               w(l0+l) = u(1+l,j0)
            enddo
         endif
      enddo
c
c     Clean up incoming recvs
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,2))
      enddo
c     Clean up outgoing sends
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,1))
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cextr(u,m,n,w,part_out,part_in,msg_id,wdsize,nid,np)
c
c     This is the transpose of cex   REAL version
c
c     Here, n is assumed to be the same on each processor.
c     However, m may vary.
c
c     No re-ordering is performed upon data receipt, as it's assumed
c     that that takes place outside this routine.
c
c     include 'mpif.h'
c
      real u(m,n),w(1)
      integer part_out(0:np),part_in(0:np)
      integer msg_id(0:np,2)
      integer wdsize
c
c
c     Post receives
c
      do k=0,np-1
         msgtag = 555+k
         j0 = part_out(k  )
         j1 = part_out(k+1)
         len = m*(j1-j0)
         if (k.ne.nid) then
            msg_id(k,1) = irecv(msgtag,u(1,j0),len*wdsize)
         else
            l0=j0
         endif
      enddo
c
c     Synchronize, so we can use forced message types
c     call nekgsync()
c
c     Call csend, using xor
c
      msgtag = 555+nid
c
      do kk=0,np-1
c        k  = xor(nid,kk)
         k  = nid+kk
         k  = mod(k,np)
         j0 = part_in(k  )
         j1 = part_in(k+1)
         len = (j1-j0)
         if (k.ne.nid) then
            msg_id(k,2) = isend(msgtag,w(j0),len*wdsize,k,0)
         else
            do l=0,len-1
               u(1+l,l0) = w(j0+l)
            enddo
         endif
      enddo
c
c     Clean up incoming recvs
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,2))
      enddo
c     Clean up outgoing sends
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,1))
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cexi(w,u,m,n,part_out,part_in,msg_id,wdsize,nid,np)
c
c     Complete exchange of an m x n matrix.   INTEGER version
c
c     Here, n is assumed to be the same on each processor.
c     However, m may vary.
c
c     No re-ordering is performed upon data receipt, as it's assumed
c     that that takes place outside this routine.
c
c     include 'mpif.h'
c
      integer u(m,n),w(1)
      integer part_out(0:np),part_in(0:np)
      integer msg_id(0:np,2)
      integer wdsize
c
c
c     Post receives
c
      do k=0,np-1
         msgtag = 555+k
         j0 = part_in(k  )
         j1 = part_in(k+1)
         len = j1-j0
         if (k.ne.nid) then
            msg_id(k,1) = irecv(msgtag,w(j0),len*wdsize)
         else
            l0 = j0
         endif
      enddo
c
c     Synchronize, so we can use forced message types
c     call nekgsync()
c
c     Call csend, using xor
c
      msgtag = 555+nid
c
      do kk=0,np-1
c        k  = xor(nid,kk)
         k  = nid+kk
         k  = mod(k,np)
         j0 = part_out(k  )
         j1 = part_out(k+1)
         len = m*(j1-j0)
         if (k.ne.nid) then
            msg_id(k,2) = isend(msgtag,u(1,j0),len*wdsize,k,0)
         else
            do l=0,len-1
               w(l0+l) = u(1+l,j0)
            enddo
         endif
      enddo
c
c     Clean up incoming recvs
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,2))
      enddo
c     Clean up outgoing sends
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,1))
      enddo
c     call outmati(w,m,n,'cexi  ')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cexti(u,m,n,w,part_out,part_in,msg_id,wdsize,nid,np)
c
c     This is the transpose of cex  INTEGER version
c
c     Here, n is assumed to be the same on each processor.
c     However, m may vary.
c
c     No re-ordering is performed upon data receipt, as it's assumed
c     that that takes place outside this routine.
c
c     include 'mpif.h'
c
      integer u(m,n),w(1)
      integer part_out(0:np),part_in(0:np)
      integer msg_id(0:np,2)
      integer wdsize
c
c
c     Post receives
c
      do k=0,np-1
         msgtag = 555+k
         j0 = part_out(k  )
         j1 = part_out(k+1)
         len = m*(j1-j0)
         if (k.ne.nid) then
            msg_id(k,1) = irecv(msgtag,u(1,j0),len*wdsize)
         else
            l0=j0
         endif
      enddo
c
c     Synchronize, so we can use forced message types
c     call nekgsync()
c
c     Call csend, using xor
c
      msgtag = 555+nid
c
      do kk=0,np-1
c        k  = xor(nid,kk)
         k  = nid+kk
         k  = mod(k,np)
         j0 = part_in(k  )
         j1 = part_in(k+1)
         len = (j1-j0)
         if (k.ne.nid) then
            msg_id(k,2) = isend(msgtag,w(j0),len*wdsize,k,0)
         else
            do l=0,len-1
               u(1+l,l0) = w(j0+l)
            enddo
         endif
      enddo
c
c     Clean up incoming recvs
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,2))
      enddo
c     Clean up outgoing sends
      do k=0,np-1
         if (k.ne.nid) call msgwait(msg_id(k,1))
      enddo
c
      return
      end
c-----------------------------------------------------------------------
