c-----------------------------------------------------------------------
      subroutine out_acsr(acsr,ia,ja,n)
      real acsr(1)
      integer ia(2),ja(1)
      character*1 adum
c
      open (unit=40,file='q0')
      open (unit=41,file='q1')
      open (unit=42,file='q2')
c
      write(6,*) 'this is ia:',ia(1),ia(2),n
      do i=1,n
         nc = ia(i+1)-ia(i)
         j0 = ia(i)-1
         do jc=1,nc
            j = ja  (jc+j0)
            a = acsr(jc+j0)
            write(40,40) a
            write(41,41) i
            write(42,41) j
            write(6,*) i,j,a
         enddo
      enddo
   40 format(1pe20.12)
   41 format(i9)
c
      close (unit=40)
      close (unit=41)
      close (unit=42)
c
c     write(6,*) 'Output in q0,q1,q2. Continue? <cr> '
c     read (5,*,end=1,err=1) adum
    1 continue
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compress_acsr(acsr,ia,ja,n)
c
c     Compress csr formatted a matrix based on drop tolerance, eps.
c
      real acsr(1)
      integer ia(1),ja(1)
c
      eps = 1.e-10
c
      k0 = ia(1)-1
      do i=1,n
         nc = ia(i+1)-ia(i)
         j0 = ia(i)-1

C        row sum as metric
         aa = 0.0
         do jc=1,nc
            aa = aa+abs(acsr(jc+j0))
         enddo
         if (aa.lt.eps) then
            write(6,*) 'inf-norm of row is small',aa
            aa=1.0
         endif
C
         kc = 0
         do jc=1,nc
            ae = abs(acsr(jc+j0)/aa)
            if (ae.gt.eps) then
c              bump column pointer
               kc = kc+1
               acsr(kc+k0) = acsr(jc+j0)
               ja  (kc+k0) = ja  (jc+j0)
            endif
         enddo
         ia (i) = k0+1
         k0     = k0+kc
      enddo
      ia(i) = k0+1
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outbox(xmax,xmin,ymax,ymin,io)
c
      dx = xmax-xmin
      dy = ymax-ymin
      dd = max(dx,dy)/2.
c
      xh = (xmax+xmin)/2.
      yh = (ymax+ymin)/2.
      xmn = xh - dd
      xmx = xh + dd
      ymn = yh - dd
      ymx = yh + dd
c
      write(io,*) 
      write(io,*) ymn,xmn
      write(io,*) ymx,xmn
      write(io,*) ymx,xmx
      write(io,*) ymn,xmx
      write(io,*) ymn,xmn
      write(io,*) 
c
      return
      end
c-----------------------------------------------------------------------
      subroutine imout(x,m,n,name)
      integer x(m,1)
      character*9 name
c
      n15 = min(n,30)
      do i=1,m
         write(6,6) n,name,(x(i,k),k=1,n15)
      enddo
    6 format(i4,1x,a9,':',2x,30i3)
      return
      end
c-----------------------------------------------------------------------
      subroutine out_abd(abd,lda,n,m)
      real    abd(lda,1)
c
      write(6,*) 'abd:',lda,m,n
      do j=1,n
          write(6,6) (abd(i,j),i=lda,1,-1)
      enddo
   6  format(12f8.3)
c.
c.
      open(unit=40,file='b0')
      open(unit=41,file='b1')
      open(unit=42,file='b2')
      mm = lda-1
      do 20 j = 1, n
         i1 = max0(1, j-mm)
         do 10 i = i1, j
            k = i-j+mm+1
            a = abd(k,j)
            write(40,*) a
            write(41,*) i
            write(42,*) j
   10    continue
   20 continue
c
      close(unit=40)
      close(unit=41)
      close(unit=42)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rarr_out(x,name13)
      include 'SIZE'
      include 'INPUT'
c
      real x(lx1,ly1,lz1,lelt)
      character*13 name13
c
      if (nelv.gt.20) return
      write(6,*) 
      write(6,1) name13
    1 format('rarr ',3x,a13)
c
c     3 D
c
      if (if3d) then
         do iz=1,nz1
            write(6,*) 
            do j=ny1,1,-1
               write(6,3) (x(k,j,iz,1),k=1,nx1),(x(k,j,iz,2),k=1,nx1)
            enddo
         enddo
         write(6,*)
         write(6,*)
         return
      endif
c
c     2 D
c
      if (nelv.gt.2) then
         write(6,*) 
         do j=ny1,1,-1
            write(6,6) (x(k,j,1,3),k=1,nx1),(x(k,j,1,4),k=1,nx1)
         enddo
         write(6,*)
         write(6,*)
      endif
c
      do j=ny1,1,-1
         write(6,6) (x(k,j,1,1),k=1,nx1),(x(k,j,1,2),k=1,nx1)
      enddo
      write(6,*)
    3 format(4f6.2,5x,4f6.2)
    6 format(6f8.4,5x,6f8.4)
      return
      end
c-----------------------------------------------------------------------
      subroutine iarr_out(x,name)
      include 'SIZE'
      include 'INPUT'
c
      integer x(lx1,ly1,lz1,lelt)
      character*13 name
c
      if (nelv.gt.20) return
      write(6,*) 
      write(6,1) name
    1 format(a13)
c
c     3 D
c
      if (if3d) then
         do iz=1,nz1
            write(6,*) 
            do j=ny1,1,-1
               write(6,3) (x(k,j,iz,1),k=1,nx1),(x(k,j,iz,2),k=1,nx1)
            enddo
         enddo
         write(6,*)
         write(6,*)
         return
      endif
c
c     2 D
c
      if (nelv.gt.2) then
         write(6,*) 
         do j=ny1,1,-1
            write(6,6) (x(k,j,1,3),k=1,nx1),(x(k,j,1,4),k=1,nx1)
         enddo
         write(6,*)
         write(6,*)
      endif
c
      do j=ny1,1,-1
         write(6,6) (x(k,j,1,1),k=1,nx1),(x(k,j,1,2),k=1,nx1)
      enddo
      write(6,*)
    3 format(4i5,5x,4i5)
    6 format(5i5,5x,5i5)
      return
      end
c-----------------------------------------------------------------------
      subroutine iar2_out(x,name)
      include 'SIZE'
c
      integer x(lx2,ly2,lz2,lelt)
      character*13 name
c
      if (nelv.gt.20) return
      write(6,*) 
      write(6,1) name
    1 format(a13)
      if (nelv.gt.2) then
         write(6,*) 
         do j=ny2,1,-1
            write(6,6) (x(k,j,1,3),k=1,nx2),(x(k,j,1,4),k=1,nx2)
         enddo
         write(6,*)
         write(6,*)
      endif
c
      do j=ny2,1,-1
         write(6,6) (x(k,j,1,1),k=1,nx2),(x(k,j,1,2),k=1,nx2)
      enddo
      write(6,*)
    6 format(3i5,5x,3i5)
      return
      end
c-----------------------------------------------------------------------
      subroutine scsr_permute(bcsr,ib,jb,acsr,ia,ja,n
     $                                  ,icperm,inverse,nonzero,nzero)
c
c     Permutes a Symmetric Compressed Sparse Row formatted matrix
c     to minimize bandwidth
c
c     Reqired space:  NONZERO is an array of size 2*nnz+2,  where nnz
c                     is the number of nonzeros.  NONZERO and jb may
c                     occupy the same space.
c
c     NOTE!!:   "inverse" is used as a scratch array for reals
c               in the swap call below... Therefore, inverse must
c               be aligned on even byte boundaries when running
c               in 64 bit precision.
c
c
      real bcsr(1)
      integer ib(1),jb(1)
c
      real acsr(1)
      integer ia(1),ja(1)
c
      integer icperm(n),nonzero(2,0:1)
      integer inverse(n),nzero(n)
C     HMT HACK ... if we're actualy using smbwr.c then we need to 
C     uncomment the following line!!!
C      integer sm_bandwidth_reduction
c
      integer icalld,max_n 
      save    icalld,max_n 
      data    icalld,max_n /0,0/
C
C     HMT HACK ... if we're actualy using smbwr.c then we need to 
C     delete the following line!!!
      write(6,*) 'HMT HACK in subroutine scsr_permute() ... pls fix!'
      call exitt
C
      icalld=icalld+1
c
      write(6,*) 'scsr_permute',n,acsr(1)
      mnz    = 0
      mbw_in = 0
      do i=1,n
         n1 = ia(i)  +1
         n2 = ia(i+1)-1
         do jj=n1,n2
            mnz = mnz+1
            j   = ja(jj)
            nonzero(1,mnz) = i
            nonzero(2,mnz) = j
C            write (6,*) i,j
            mbw_in = max(mbw_in,abs(i-j))
         enddo
      enddo
      nonzero(1,0) = n
      nonzero(2,0) = mnz
C
      itype = -1
C      itype = -2
      if (n.le.200) itype = -2
      write(6,*) 'this is mnz:',mnz
C      write(6,*)  nonzero(1,0), nonzero(2,0)
c
C     HMT HACK ... if we're actualy using smbwr.c then we need to 
C     uncomment the following line!!!
C      ierr  = sm_bandwidth_reduction(nonzero(1,0),icperm(1),itype)
      ierr = 1
      if (ierr.eq.0) then
         write(6,*) 'scsr_permute :: sm_bandwidth_reduction failed',ierr
         call exitt
      elseif (ierr.lt.0) then
         write(6,*) 'scsr_permute :: graph disconnected?',ierr
         call exitt
      endif
c
c     Setup inverse map
c
      do i=1,n
         inverse(icperm(i))=i
      enddo
c
C      n20 = min(50,n)
C      write(6,20) (icperm (k),k=1,n20)
C      write(6,21) (inverse(k),k=1,n20)
C 20   format('icp:',20i4)
C 21   format('inv:',20i4)
c
c
c     Count *number* of nonzeros on each row in the new row pointer
c
      call izero(nzero,2*mnz)
      do i=1,n
         n1 = ia(i)
         n2 = ia(i+1)-1
         ii = inverse(i)
         do jc=n1,n2
            j=ja(jc)
            jj = inverse(j)
            if (jj.ge.ii) then
               nzero(ii) = nzero(ii)+1
            else
               nzero(jj) = nzero(jj)+1
            endif
         enddo
      enddo
c
c     Convert to pointers
c
      ib(1) = 1
      do i=1,n
         ib(i+1) = ib(i)+nzero(i)
      enddo
c
c     Fill permuted array
c
      call icopy(nzero,ib,n)
      do i=1,n
         n1 = ia(i)
         n2 = ia(i+1)-1
         ii = inverse(i)
         do jc=n1,n2
            j=ja(jc)
            jj = inverse(j)
            if (jj.ge.ii) then
               jb  (nzero(ii)) = jj
               bcsr(nzero(ii)) = acsr(jc)
               nzero(ii) = nzero(ii)+1
            else
               jb  (nzero(jj)) = ii
               bcsr(nzero(jj)) = acsr(jc)
               nzero(jj) = nzero(jj)+1
            endif
         enddo
      enddo
c
c     Sort each row of b in ascending column order -- 
c                             just for subsequent access efficiency
c
      m2 = mod(n2,2)+2
      do i=1,n
         n1 = ib(i)
         n2 = ib(i+1)-ib(i)
         call isort(jb  (n1),nzero,n2)
c        write(6,*) 'call swap:',n,i,m2,n1,n2
         call swap (bcsr(n1),nzero,n2,inverse)
      enddo
c
      if (icalld.le.10.or.mod(icalld,50).eq.0.or.n.gt.max_n) then
         mbw = mbw_csr(ib,jb,n)
         write(6,*) ierr,mbw,mbw_in,n,mnz,'   New bandwidth'
         if (n.gt.max_n) max_n = n
      endif
c
      return
      end
c=======================================================================
      subroutine scsr_to_spb(abd,lda,acsr,ia,ja,n)
c
c     This, and the companion routines map a *symmetrically*
c     stored (upper 1/2 only) Compressed Sparse Row formatted
c     matrix to the linpack banded format.
c
c
      real abd(1),acsr(1)
      integer m,n,ia(1),ja(1)
c
      lda = mbw_csr(ia,ja,n) + 1
      call scsr_to_spbm(abd,lda,acsr,ia,ja,n)
c
      return
      end
c=======================================================================
      subroutine scsr_to_spbm(abd,lda,acsr,ia,ja,n)
c
      real abd(lda,1),acsr(1)
      integer n,ia(1),ja(1)
c
      call rzero(abd,lda*n)
      do i=1,n
         n1 = ia(i)
         n2 = ia(i+1)-1
         do jc=n1,n2
            j=ja(jc)
            klin = lda + (i-j)
            jlin = j
            abd(klin,jlin) = acsr(jc)
         enddo
      enddo
      return
      end
      function mbw_csr(ia,ja,n) 
c
c     Determine the maximum bandwidth for a csr formatted matrix
c
      integer ia(1),ja(1)
c
      m = 0
      do i=1,n
         n1 = ia(i)
         n2 = ia(i+1)-1
         do jc=n1,n2
            j=ja(jc)
            m = max(m,abs(i-j))
         enddo
      enddo
      mbw_csr = m
      return
      end
c=======================================================================
      subroutine out_spbmat(abd,n,lda,name)
      real    abd(lda,1)
c
      character*9 name
      character*6 s(22)
c
      m = lda-1
      write(6,1) name,n,m
    1 format(/,'SPB Mat:',a9,3x,'n =',i3,' m =',i3,/)
c
      n22 = min(n,22)
      do i=1,n
         call blank(s,132)
         ii = lda*i
         do k=0,m
            j = i+k
            a = abd(ii+k*m,1)
            if (a.ne.0..and.j.le.n22) write(s(j),6) a
         enddo
         write(6,22) (s(k),k=1,n22)
      enddo
    6 format(f6.3)
   22 format(22a6)
c
      return
      end
c=======================================================================
      subroutine swap(b,ind,n,temp)
      real B(1),TEMP(1)
      integer IND(1)
C***
C***  SORT ASSOCIATED ELEMENTS BY PUTTING ITEM(JJ)
C***  INTO ITEM(I), WHERE JJ=IND(I).
C***
      DO 20 I=1,N
         JJ=IND(I)
         TEMP(I)=B(JJ)
   20 CONTINUE
      DO 30 I=1,N
   30 B(I)=TEMP(I)
      RETURN
      END
c=======================================================================
      subroutine ipermute(a,icperm,n,b)
      integer a(1),b(1)
      integer icperm(1)
c
      call icopy(b,a,n)
      do i=1,n
         a(i) = b(icperm(i))
      enddo
      return
      end
c=======================================================================
      subroutine out_csrmat(acsr,ia,ja,n,name9)
      real    acsr(1)
      integer ia(1),ja(1)
c
      character*9 name9
      character*6 s(22)
c
      nnz = ia(n+1)-ia(1)
c
      write(6,1) name9,n,nnz
    1 format(/,'CSR Mat:',a9,3x,'n =',i3,3x,'nnz =',i5,/)
c
      n22 = min(n,22)
      n29 = min(n,29)
      do i=1,n29
         call blank(s,132)
         n1 = ia(i)
         n2 = ia(i+1)-1
         do jj=n1,n2
            j = ja  (jj)
            a = acsr(jj)
            if (a.ne.0..and.j.le.n22) write(s(j),6) a
         enddo
         write(6,22) (s(k),k=1,n22)
      enddo
    6 format(f6.2)
   22 format(22a6)
c
      return
      end
c=======================================================================
