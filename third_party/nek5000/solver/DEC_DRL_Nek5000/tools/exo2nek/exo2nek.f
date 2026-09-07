      program exo2nek

      include 'SIZE'

      call read_input_name
      call exodus_read
      call convert
      call gen_re2

      end 
c-----------------------------------------------------------------------
      subroutine read_input_name

      include 'SIZE'

      character*1 re2nam1(80)
      character*1 exonam1(32)

      equivalence(re2name,re2nam1)
      equivalence(exoname,exonam1) 

      call blank (exoname, 32)
      call blank (re2name, 80)

      write(6,*) 'Input (.exo) file name:'
      read (5,'(a32)') exoname
      len = ltrunc(exoname,32)

      call chcopy(re2name        ,exoname,32)
      call chcopy(exonam1(len+1) ,'.exo' , 4)
      call chcopy(re2nam1(len+1) ,'.re2' , 4)

      return 
      end
c-----------------------------------------------------------------------
      subroutine exodus_read
c
c  Subroutine to read an exodusII binary file containing a mesh.
c  It uses exodus fortran binding subroutines, which depend on
c  the netcdf library for low level data access.
c
      include 'exodusII.inc'
      include 'SIZE'

      integer exoid, cpu_ws, io_ws

      character*(MXSTLN) typ, qa_record(4,10)
      character*(MXLNLN) titl
      character*1        cdum

      integer idblk              (max_num_elem_blk)
      integer num_attr           (max_num_elem_blk)  ! not used
      integer num_nodes_per_elem (max_num_elem_blk)
c
c open EXODUS II file
c
      cpu_ws = 8 ! use real*8 to communicate with exodus
      io_ws  = 0
      exoid  = exopen (exoname, EXREAD, cpu_ws, io_ws, vers, ierr)
      if (ierr.lt.0) then
        write(6,'(2a)') "ERROR: cannot open file ", exoname 
        STOP
      endif
      write(6,*)
      write(6,'(2a,f4.2)') exoname," is an EXODUSII file; version ",vers
      write(6,'(a,i2)')            "I/O word size", io_ws
c
c read database parameters
c
      call exgini (exoid, titl, num_dim, num_nodes, num_elem,
     &             num_elem_blk, num_node_sets, num_side_sets, ierr)
      if (ierr.lt.0) then
        write(6,'(a)') "ERROR: cannot read exodusII parameters (exgini)"
        STOP
      endif
      write (6,    '(/"database parameters:"/ /
     &               "title = ", a81 / /
     &               "num_dim = ", i8 /
     &               "num_nodes = ", i8 /
     &               "num_elem = ", i8 /
     &               "num_elem_blk = ", i8 /
     &               "num_side_sets = ", i3)')
     &               titl,num_dim, num_nodes, num_elem,
     &               num_elem_blk, num_side_sets
      write (6,*)
c
c perform some checks
c
      if (num_elem.gt.max_num_elem) then
        write(6,'(a)')
     &    "ERROR: number of elements larger that max_num_elem! "
        write(6,'(a,i8,a)') "Set max_num_elem >= ", num_elem,
     &                      " and recompile exo2nek. "
        STOP
      endif
      if (num_side_sets.gt.max_num_sidesets) then
        write(6,'(a)')
     &    "ERROR: number of sidesets > max_num_sidesets. "
        write(6,'(a,i2,a)') "Set max_num_sidesets >= ",num_side_sets,
     &                      " and recompile exo2nek. "
        STOP
      endif
c
c read element block parameters
c
      call exgebi (exoid, idblk, ierr)
      if (ierr.lt.0) then
        write(6,'(a)') "ERROR: cannot read block ids (exgebi)"
        STOP
      endif

      do i = 1, num_elem_blk
        call exgelb (exoid, idblk(i), typ, num_elem_in_block(i),
     &               num_nodes_per_elem(i), num_attr(i), ierr)
        if (ierr.lt.0) then
          write(6,'(a,i3,a)')
     &    "ERROR: cannot read parameters for block ",i," (exgelb)"
          STOP
        endif
        write (6, '("element block id = ", i2,/
     &              "element type = ", a9,/
     &              "num_elem_in_block = ", i8,/
     &              "num_nodes_per_elem = ", i2)')
     &              idblk(i), typ, num_elem_in_block(i),
     &              num_nodes_per_elem(i)
        if (num_dim.eq.3.and.num_nodes_per_elem(i).ne.27) then
          write(6,'(a)')
     &     "ERROR: Only HEX27 elements are allowed in a 3D mesh!"
          write(6,'(a,i3)') "num_nodes_per_elem= ",num_nodes_per_elem(i)
          STOP
        elseif (num_dim.eq.2.and.num_nodes_per_elem(i).ne.9) then
          write(6,'(a)')
     &      "ERROR: Only QUAD9 elements are allowed in a 2D mesh!"
          write(6,'(a,i3)') "num_nodes_per_elem= ",num_nodes_per_elem(i)
          STOP
        endif
        write (6,*)
      enddo
c
c read nodal coordinates values from database
c
      call exgcor (exoid, x_exo, y_exo, z_exo, ierr)
      if (ierr.lt.0) then
        write(6,'(a)') "ERROR: cannot read nodal coordinates (exgcor)"
        STOP
      endif
c
c read element connectivity
c
      iend = 0
      do 60 i = 1, num_elem_blk
        istart = iend + 1
        call exgelc (exoid, idblk(i), connect(istart), ierr)
        iend = num_nodes_per_elem(i)*num_elem_in_block(i)
        if (ierr.lt.0) then
          write(6,'(a)') "ERROR: cannot read elm. connectivity (exgelc)"
          STOP
        endif
60    continue
c
c read individual side sets
c
      if (num_side_sets .gt. 0) then
        call exgssi (exoid, idss, ierr)
        if (ierr.lt.0) then
          write(6,'(a)') "ERROR: cannot read SideSet ids (exgssi)"
          STOP
        endif

        do i = 1, num_side_sets
          call exgsp (exoid,idss(i),num_sides_in_set(i),idum,ierr)
          if (ierr.lt.0) then
            write(6,'(a,i3,a)')
     &      "ERROR: cannot read parameters for SideSet No.",i," (exgsp)"
            STOP
          endif
          write (6, '("side set ", i2, " num_sides = ", i8)')
     &           idss(i), num_sides_in_set(i)
          call exgss (exoid,idss(i),elem_list(1,i),side_list(1,i),ierr)
          if (ierr.lt.0) then
            write(6,'(a,i3,a)')
     &      "ERROR: cannot read parameters for SideSet No.",i," (exgss)"
            STOP
          endif
        enddo
        write (6,*)
      else
        write(6,'(a)') "WARNING: No SideSets in exodus file!"
      endif
c
c read QA records
c
      call exinq (exoid, EXQA, num_qa_rec, fdum, cdum, ierr)
      if (ierr.lt.0) then
        write(6,'(a)') "ERROR: cannot read QA records (exinq QA) "
        STOP
      elseif (ierr.gt.0) then
        write(6,'(a)') "INFO: file does not contain any QA records"
      else
        call exgqa (exoid, qa_record, ierr)
        if (ierr.lt.0) then
          write(6,'(a,i3)') "WARNING: cannot read QA records (exgqa)"
        else
          write (6, '("QA records = ")')
          if (num_qa_rec.gt.10) then
            write(6,'(2a)')
     &        'WARNING: Cannot handle more than 10 QA records',
     &        'Printing only the first 10...'
          else
            do i = 1, num_qa_rec
              do j = 1, 4
                write (6,'(a)') qa_record(j,i)
              enddo
            enddo
          endif
        endif
      endif

      return
      end
C-----------------------------------------------------------------
      subroutine convert
c
c  Subroutine to convert an already read exodusII mesh to a nek
c  mesh. The idea is to fill each element's node coordinates of
c  size lx1**3 (3D) or lx1**2 (2D) (lx1=3) with the hex27/quad9
c  coordinates.
c
      include 'exodusII.inc'
      include 'SIZE'
c
c node and face conversion (it works at least for cubit):
c
      integer exo_to_nek_vert3D(27)
      data    exo_to_nek_vert3D
     &      / 19,  1,  7, 25, 21,  3,  9, 27, 10                  ! hex27 to nek numbering
     &      ,  4, 16, 22, 20,  2,  8, 26, 12,  6
     &      , 18, 24, 14, 13, 15, 23,  5, 11, 17 /

      integer exo_to_nek_vert2D(9)
      data    exo_to_nek_vert2D   / 1, 3, 9, 7, 2, 6, 8, 4, 5  /  ! quad9 to nek numbering

      integer exo_to_nek_face3D(6)
      data    exo_to_nek_face3D  / 1, 5, 3, 6, 4, 2 /    ! symmetric face numbering

      integer exo_to_nek_face2D(4)
      data    exo_to_nek_face2D  / 1, 2, 3, 4 /          ! symmetric face numbering

      write(6,*)
      write(6,'(A)') 'Converting elements ...'
      write(6,*)

      nvert = 3**num_dim

      do iel = 1, num_elem
        if (mod(iel,100).eq.0.or.iel.eq.num_elem) then
          write(6,'(A,I8)') 'Reading element ',iel
        endif
        do ivert = 1, nvert
          if (num_dim.eq.2) then
            jvert = exo_to_nek_vert2D(ivert)
            xm1(jvert,1,1,iel)=x_exo(connect(nvert*(iel-1)+ivert))
            ym1(jvert,1,1,iel)=y_exo(connect(nvert*(iel-1)+ivert))
          else
            jvert = exo_to_nek_vert3D(ivert)
            xm1(jvert,1,1,iel)=x_exo(connect(nvert*(iel-1)+ivert))
            ym1(jvert,1,1,iel)=y_exo(connect(nvert*(iel-1)+ivert))
            zm1(jvert,1,1,iel)=z_exo(connect(nvert*(iel-1)+ivert))
          endif
        enddo
      enddo
c zero-out bc and curve sides arrays
      call blank   (cbc,3*2*ldim*max_num_elem)
      call rzero   (bc,5*2*ldim*max_num_elem)
      call blank   (ccurve,(4+8*(ldim-2))*max_num_elem)
      call rzero   (curve,2*ldim*12*max_num_elem)
c
c set bc's
c
      if (num_side_sets.eq.0) return   ! no sidesets

      write(6,'(a)') ""
      write(6,'(a)') "Converting SidSets ..."
c the expensive part, improve it...
      do iss=1,num_side_sets   ! loop over ss 
        write(6,'(a)') ""
        write(6,'(a,i2)') "Sideset ",idss(iss)
        do iel=1,num_elem
          if (mod(iel,100).eq.0.or.iel.eq.num_elem) then
            write(6,'(A,I8)') '  Element ',iel
          endif
          do ifc=1,2*num_dim             ! loop over faces
            do i=1,num_sides_in_set(iss) ! loop over sides in ss
              if      ( (iel.eq.elem_list(i,iss))
     &        .and.   (ifc.eq.side_list(i,iss)) ) then
                if (num_dim.eq.2) then
                  jfc = exo_to_nek_face2D(ifc)
                else
                  jfc = exo_to_nek_face3D(ifc)
                endif
                cbc(jfc,iel)   = 'EXO' ! dummy exodus bc 
                bc (5,jfc,iel) = idss(iss)
              endif
            enddo
          enddo
        enddo
      enddo

      return
      end
C--------------------------------------------------------------------
      subroutine gen_re2

      include 'SIZE'

      write(6,*)
      write(6,'(A,A)') 'writing ', re2name

      call open_re2
      call write_xyz
      call write_curve
      call write_bc
      call close_re2

      return
      end
C--------------------------------------------------------------------
      subroutine open_re2

      include 'SIZE'

      character*1 file(80)
      character*80  hdr


      real*4 test
      data   test  / 6.54321 /

      len = ltrunc(re2name,80)
      call chcopy(file,re2name,80)
      call chcopy(file(len+1),char(0),1) 

      call byte_open(file,ierr)
            
c  Write the header
      call blank     (hdr,80)    
      write(hdr,1) num_elem, num_dim, num_elem
    1 format('#v003',i9,i3,i9,' this is the hdr')
      call byte_write(hdr,20,ierr)         
      call byte_write(test,1,ierr)     ! write the endian discriminator

      return
      end
C--------------------------------------------------------------------
      subroutine write_xyz

      include 'SIZE'

      real     xx(8), yy(8), zz(8)
      real*8   rgroup, buf2(30)

      integer isym2pre(8)   ! Symmetric-to-prenek vertex ordering
      data    isym2pre    / 1 , 2 , 4 , 3 , 5 , 6 , 8 , 7 /

      nxs = 3-1  ! always nx1=ny1=nz1=3 here
      nys = 3-1
      nzs = 3-1

      igr    = 0
      rgroup = igr

      do iel=1,num_elem
        l = 0
        if (num_dim.eq.3) then
          do k=0,1
          do j=0,1
          do i=0,1
            l=l+1
            li=isym2pre(l)
            xx(li) = xm1(1+i*nxs,1+j*nys,1+k*nzs,iel)
            yy(li) = ym1(1+i*nxs,1+j*nys,1+k*nzs,iel)
            zz(li) = zm1(1+i*nxs,1+j*nys,1+k*nzs,iel)
          enddo
          enddo
          enddo
        else
          do j=0,1
          do i=0,1
            l=l+1
            li=isym2pre(l)
            xx(li) = xm1(1+i*nxs,1+j*nys,1,iel)
            yy(li) = ym1(1+i*nxs,1+j*nys,1,iel)
          enddo
          enddo
        endif

        call byte_write(rgroup, 2,ierr)

        if (num_dim.eq.3) then
          call copy        (buf2(1) ,xx,8)
          call copy        (buf2(9) ,yy,8)
          call copy        (buf2(17),zz,8)
          call byte_write  (buf2(1) ,16, ierr)
          call byte_write  (buf2(9) ,16, ierr)
          call byte_write  (buf2(17),16, ierr)
        else
          call copy        (buf2(1),xx,4)
          call copy        (buf2(5),yy,4)
          call byte_write  (buf2(1),8, ierr)
          call byte_write  (buf2(5),8, ierr)
        endif
      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine write_curve

      include 'SIZE'

      real*8     buf2(30)
      real*8     rcurve

      character*1 cc

      do iel=1,num_elem
         call gen_rea_midside_e(iel)
      enddo

      nedge = 4 + 8*(num_dim-2)
      ncurv = 0
      do iel=1,num_elem
        do iedge=1,nedge
          if (ccurve(iedge,iel).ne.' ') ncurv = ncurv+1
        enddo
      enddo

      rcurve = ncurv
      call byte_write(rcurve,2, ierr)

      do iel=1,num_elem
        do iedge=1,nedge
          if (ccurve(iedge,iel).ne.' ') then
            if (ccurve(iedge,iel).eq.'C') cc='C'
            if (ccurve(iedge,iel).eq.'s') cc='s'
            if (ccurve(iedge,iel).eq.'m') cc='m'
            buf2(1) = iel
            buf2(2) = iedge
            call copy       (buf2(3),curve(1,iedge,iel),5)
            call blank      (buf2(8),8)
            call chcopy     (buf2(8),cc,1)
            call byte_write (buf2,16, ierr)
          endif
        enddo
      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine write_bc
       
      include 'SIZE'

      real*8  rbc, buf2(30)

      character*3 ch3
      character*1 chdum
      data        chdum /' '/

      if (num_side_sets.eq.0) return

      nface = 2*num_dim
      nbc   = 0

      do iel=1,num_elem
        do ifc=1,nface
          if (cbc(ifc,iel).ne.'   ')  nbc = nbc + 1
        enddo
      enddo

      rbc = nbc
      call byte_write (rbc,2, ierr)

      do iel = 1,num_elem
        do ifc = 1,nface
          ch3 = cbc(ifc,iel)
          if (ch3.eq.'EXO') then
            buf2(1)=iel
            buf2(2)=ifc
            call copy   (buf2(3),bc(1,ifc,iel),5)
            call blank  (buf2(8),8)
            call chcopy (buf2(8),ch3,3)
            if (num_elem.ge.1 000 000) then
              ibc     = bc(1,ifc,iel)
              buf2(3) = ibc
            endif
            call byte_write (buf2,16, ierr)
          endif
        enddo
      enddo

      return
      end 
C-----------------------------------------------------------------------
      subroutine close_re2

      call byte_close (ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_rea_midside_e(e)

      include 'SIZE'

      real        len
      real        x3(27),y3(27),z3(27),xyz(3,3)
      character*1 ccrve(12)
      integer     e,edge

      integer e3(3,12)
      save    e3
      data    e3 /  1, 2, 3,    3, 6, 9,    9, 8, 7,    7, 4, 1
     $           , 19,20,21,   21,24,27,   27,26,25,   25,22,19
     $           ,  1,10,19,    3,12,21,    9,18,27,    7,16,25 /

      call chcopy(ccrve,ccurve(1,e),12)

      call map2reg     (x3,3,xm1(1,1,1,e),1)  ! Map to 3x3x3 array
      call map2reg     (y3,3,ym1(1,1,1,e),1)
      if (num_dim.eq.3) call map2reg    (z3,3,zm1(1,1,1,e),1)

c     Take care of spherical curved face defn
      if (ccurve(5,e).eq.'s') then
         call chcopy (ccrve(1),'ssss',4) ! face 5
         call chcopy (ccrve(5),' ',1)    ! face 5
      endif
      if (ccurve(6,e).eq.'s') then
         call chcopy (ccrve(5),'ssss',4) ! face 6
      endif

      tol   = 1.e-4
      tol2  = tol**2
      nedge = 4 + 8*(num_dim-2)

      do i=1,nedge
         if (ccrve(i).eq.' ') then
            do j=1,3
               xyz(1,j) = x3(e3(j,i))
               xyz(2,j) = y3(e3(j,i))
               xyz(3,j) = z3(e3(j,i))
            enddo
            len = 0.
            h   = 0.
            do j=1,num_dim
               xmid = .5*(xyz(j,1)+xyz(j,3))
               h    = h   + (xyz(j,2)-xmid)**2
               len  = len + (xyz(j,3)-xyz(j,1))**2
            enddo
            if (h.gt.tol2*len) ccurve(i,e) = 'm'
            if (h.gt.tol2*len) call copy (curve(1,i,e),xyz(1,2),num_dim)
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine map2reg(ur,n,u,nel)
c
c     Map scalar field u() to regular n x n x n array ur
c
      include 'SIZE'

      real    ur(1), u(3*3*3,1)
      integer e

      ldr = n**num_dim

      k=1
      do e=1,nel
         if (num_dim.eq.2) call map2reg_2di_e(ur(k),n,u(1,e),3)
         if (num_dim.eq.3) call map2reg_3di_e(ur(k),n,u(1,e),3)
         k = k + ldr
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine map2reg_2di_e(uf,n,uc,m) ! Fine, uniform pt

      real uf(n,n),uc(m,m)

      parameter (l=50)
      common /cmap2d/ j(l*l),jt(l*l),w(l*l),z(l)

      integer mo,no
      save    mo,no
      data    mo,no / 0,0 /

      if (m.ne.mo .or. n.ne.no ) then

          call zwgll (z,w,m)
          call zuni  (w,n)

          call gen_int_gz(j,jt,w,n,z,m)

      endif

      call mxmf2(j,n,uc,m,w ,m)
      call mxmf2(w,n,jt,m,uf,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine map2reg_3di_e(uf,n,uc,m) ! Fine, uniform pt

      real uf(n,n,n),uc(m,m,m)

      parameter (l=50)
      common /cmap3d/ j(l*l),jt(l*l),v(l*l*l),w(l*l*l),z(l)

      integer mo,no
      save    mo,no
      data    mo,no / 0,0 /

      if (m.ne.mo .or. n.ne.no ) then

          call zwgll (z,w,m)
          call zuni  (w,n)

          call gen_int_gz(j,jt,w,n,z,m)

      endif

      mm = m*m
      mn = m*n
      nn = n*n

      call mxmf2(j,n,uc,m,v ,mm)
      iv=1
      iw=1
      do k=1,m
         call mxmf2(v(iv),n,jt,m,w(iw),n)
         iv = iv+mn
         iw = iw+nn
      enddo
      call mxmf2(w,nn,jt,m,uf,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine gen_int_gz(j,jt,g,n,z,m)

c     Generate interpolater from m z points to n g points

c        j   = interpolation matrix, mapping from z to g
c        jt  = transpose of interpolation matrix
c        m   = number of points on z grid
c        n   = number of points on g grid

      real j(n,m),jt(m,n),g(n),z(m)

      mpoly  = m-1
      do i=1,n
         call fd_weights_full(g(i),z,mpoly,0,jt(1,i))
      enddo

      call transpose(j,n,jt,m)

      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE BLANK(A,N)
      CHARACTER*1 A(1)
      CHARACTER*1 BLNK
      SAVE        BLNK
      DATA        BLNK /' '/
C
      DO 10 I=1,N
         A(I)=BLNK
   10 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine exitt

      stop
      return
      end
c-----------------------------------------------------------------------
      subroutine copy(a,b,n)
      real a(1),b(1)

      do i=1,n
         a(i)=b(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine chcopy(a,b,n)
      CHARACTER*1 A(1), B(1)
C
      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      END
c-----------------------------------------------------------------------
      subroutine icopy(a,b,n)
      INTEGER A(1), B(1)
C
      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      END
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

      return
      end
c-----------------------------------------------------------------------
      subroutine transpose(a,lda,b,ldb)
      real a(lda,1),b(ldb,1)
c
      do j=1,ldb
         do i=1,lda
            a(i,j) = b(j,i)
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      function ltrunc(string,l)
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
      return
      END
c-----------------------------------------------------------------------
      subroutine zuni(z,np)
c
c     Generate equaly spaced np points on the interval [-1:1]
c
      real z(1)

      dz = 2./(np-1)
      z(1) = -1.
      do i = 2,np-1
         z(i) = z(i-1) + dz
      enddo
      z(np) = 1.

      return
      end
c-----------------------------------------------------------------------
      subroutine rzero(a,n)
      real A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      return
      END

