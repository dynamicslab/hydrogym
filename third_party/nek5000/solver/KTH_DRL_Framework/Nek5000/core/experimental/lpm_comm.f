c-----------------------------------------------------------------------
      subroutine lpm_comm_setup
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
#     include "LPM"

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      nmsh = lpm_rparam(3)

      call interp_setup(i_fp_hndl,0.0,nmsh,nelt)
      call fgslib_crystal_setup(i_cr_hndl,nekcomm,np)

      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_comm_findpts
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      common /intp_h/ ih_intp(2,1)

      ih_intp2 = ih_intp(2,i_fp_hndl)

      ix = 1
      iy = 2
      iz = 3

      call fgslib_findpts(ih_intp2           !   call fgslib_findpts( ihndl,
     $        , lpm_iprop (1 ,1),LPM_LIP        !   $        rcode,1,
     $        , lpm_iprop (3 ,1),LPM_LIP        !   &        proc,1,
     $        , lpm_iprop (2 ,1),LPM_LIP        !   &        elid,1,
     $        , lpm_rprop2(1 ,1),LPM_LRP2       !   &        rst,ndim,
     $        , lpm_rprop2(4 ,1),LPM_LRP2       !   &        dist,1,
     $        , lpm_y     (ix,1),LPM_LRS        !   &        pts(    1),1,
     $        , lpm_y     (iy,1),LPM_LRS        !   &        pts(  n+1),1,
     $        , lpm_y     (iz,1),LPM_LRS ,LPM_NPART) !   &   pts(2*n+1),1,n)

      do i=1,lpm_npart
         lpm_iprop(4,i) = lpm_iprop(3,i)
      enddo

      ! instead get which bin it is in
      do i=1,lpm_npart
         ! check if particles are greater or less than binb bounds....
         ! add below....

         ii    = floor((lpm_y(ix,i)-lpm_binb(1))/lpm_rdxgp) 
         jj    = floor((lpm_y(iy,i)-lpm_binb(3))/lpm_rdygp) 
         kk    = floor((lpm_y(iz,i)-lpm_binb(5))/lpm_rdzgp) 
         if (.not. if3d) kk = 0
         if (ii .eq. lpm_ndxgp) ii = lpm_ndxgp - 1
         if (jj .eq. lpm_ndygp) jj = lpm_ndygp - 1
         if (kk .eq. lpm_ndzgp) kk = lpm_ndzgp - 1
         ndum  = ii + lpm_ndxgp*jj + lpm_ndxgp*lpm_ndygp*kk
         nrank = modulo(ndum, np)

         lpm_iprop(8,i)  = ii
         lpm_iprop(9,i)  = jj
         lpm_iprop(10,i) = kk
         lpm_iprop(11,i) = ndum

         lpm_iprop(4,i)  = nrank ! where particle is actually moved
      enddo

      n = nid_glcount(lpm_iprop(4,1),LPM_LIP,lpm_npart)
      ierr = 0
      if (n.gt.LPM_LPART) ierr = 1 
      ierr = iglsum(ierr,1)
      if (ierr.gt.0) then
         nmax = iglmax(n,1)
         call exitti('LPM_LPART too small, require >$',nmax)
      endif

      return
      end
c-----------------------------------------------------------------------
      integer function nid_glcount(a,is,n)
c
c     returns global nid count in distributed integer array 
c
      include 'mpif.h'
      include 'SIZE'
      include 'PARALLEL'

      integer a(*)

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal

      integer   disp_unit
      integer*8 wsize, tdisp
      data      tdisp /0/

      integer win, shared_counter
      save    win, shared_counter

      integer one
      parameter (one = 1)

      integer icalld
      data    icalld /0/
      save    icalld

#ifdef MPI
      if (icalld.eq.0) then
         disp_unit = ISIZE
         wsize     = disp_unit
         call MPI_win_create(shared_counter,
     $                       wsize,
     $                       disp_unit,
     $                       MPI_INFO_NULL,
     $                       nekcomm,win,ierr)
         icalld = 1
      endif

      shared_counter = 0

      call MPI_win_fence(MPI_MODE_NOPRECEDE,win,ierr)
      do i = 1,n
         call MPI_accumulate(one,1,MPI_INTEGER,a((i-1)*is+1),
     $                       tdisp,1,MPI_INTEGER,MPI_SUM,win,ierr)
      enddo
      call MPI_win_fence(MPI_MODE_NOSUCCEED,win,ierr)

      nid_glcount = shared_counter 
#else
      nid_glcount = n
#endif
 
      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_comm_crystal
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
#     include "LPM"

      logical partl    
      integer lpm_ipmap1(1,LPM_LPART)
     >       ,lpm_ipmap2(1,LPM_LPART)
     >       ,lpm_ipmap3(1,LPM_LPART)
     >       ,lpm_ipmap4(1,LPM_LPART)

      parameter(lrf = LPM_LRS*4 + LPM_LRP + LPM_LRP2)
      real rwork(lrf,LPM_LPART)

      do i=1,lpm_npart
         ic = 1
         call copy(rwork(ic,i),lpm_y(1,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(rwork(ic,i),lpm_y1((i-1)*LPM_LRS+1),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(rwork(ic,i),lpm_ydot(1,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(rwork(ic,i),lpm_ydotc(1,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(rwork(ic,i),lpm_rprop(1,i),LPM_LRP)
         ic = ic + LPM_LRP
         call copy(rwork(ic,i),lpm_rprop2(1,i),LPM_LRP2)
      enddo

      j0 = 4 ! proc key
      call fgslib_crystal_tuple_transfer(i_cr_hndl,lpm_npart ,LPM_LPART
     $           ,lpm_iprop ,LPM_LIP,partl,0,rwork,lrf ,j0)

      do i=1,lpm_npart
         ic = 1
         call copy(lpm_y(1,i),rwork(ic,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(lpm_y1((i-1)*LPM_LRS+1),rwork(ic,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(lpm_ydot(1,i),rwork(ic,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(lpm_ydotc(1,i),rwork(ic,i),LPM_LRS)
         ic = ic + LPM_LRS
         call copy(lpm_rprop(1,i),rwork(ic,i),LPM_LRP)
         ic = ic + LPM_LRP
         call copy(lpm_rprop2(1,i),rwork(ic,i),LPM_LRP2)
      enddo
        
      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_comm_bin_setup
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      integer  el_face_num(18),el_edge_num(36),el_corner_num(24),
     >                            nfacegp, nedgegp, ncornergp
      integer  ifac(3), icount(3)
      real     d2new(3)
      logical  partl

      real                      lpm_xerange(2,3,lpm_lbmax)
      common /lpm_elementrange/ lpm_xerange

c     face, edge, and corner number, x,y,z are all inline, so stride=3
      el_face_num = (/ -1,0,0, 1,0,0, 0,-1,0, 0,1,0, 0,0,-1, 0,0,1 /)
      el_edge_num = (/ -1,-1,0 , 1,-1,0, 1,1,0 , -1,1,0 ,
     >                  0,-1,-1, 1,0,-1, 0,1,-1, -1,0,-1,
     >                  0,-1,1 , 1,0,1 , 0,1,1 , -1,0,1  /)
      el_corner_num = (/
     >                 -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1,
     >                 -1,-1,1,  1,-1,1,  1,1,1,  -1,1,1 /)

      nfacegp   = 4  ! number of faces
      nedgegp   = 4  ! number of edges
      ncornergp = 0  ! number of corners

      if (if3d) then
         nfacegp   = 6  ! number of faces
         nedgegp   = 12 ! number of edges
         ncornergp = 8  ! number of corners
      endif

      ix = 1
      iy = 2
      iz = 3

      iperiodicx = int(lpm_rparam(8))
      iperiodicy = int(lpm_rparam(9))
      iperiodicz = int(lpm_rparam(10))

      ! compute binb
      xmin = 1E8
      ymin = 1E8
      zmin = 1E8
      xmax = 0.
      ymax = 0.
      zmax = 0.
      do i=1,lpm_npart
         rduml = lpm_y(ix,i) - lpm_d2chk(2)
         rdumr = lpm_y(ix,i) + lpm_d2chk(2)
         if (rduml .lt. xmin) xmin = rduml
         if (rdumr .gt. xmax) xmax = rdumr

         rduml = lpm_y(iy,i) - lpm_d2chk(2)
         rdumr = lpm_y(iy,i) + lpm_d2chk(2)
         if (rduml .lt. ymin) ymin = rduml
         if (rdumr .gt. ymax) ymax = rdumr

         rduml = lpm_y(iz,i) - lpm_d2chk(2)
         rdumr = lpm_y(iz,i) + lpm_d2chk(2)
         if (rduml .lt. zmin) zmin = rduml
         if (rdumr .gt. zmax) zmax = rdumr
      enddo

      lpm_binb(1) = glmin(xmin,1)
      lpm_binb(2) = glmax(xmax,1)
      lpm_binb(3) = glmin(ymin,1)
      lpm_binb(4) = glmax(ymax,1)
      lpm_binb(5) = 0.0
      lpm_binb(6) = 0.0
      if(if3d) lpm_binb(5) = glmin(zmin,1)
      if(if3d) lpm_binb(6) = glmax(zmax,1)

      lpm_binb(1) = max(lpm_binb(1),lpm_xdrange(1,1))
      lpm_binb(2) = min(lpm_binb(2),lpm_xdrange(2,1))
      lpm_binb(3) = max(lpm_binb(3),lpm_xdrange(1,2))
      lpm_binb(4) = min(lpm_binb(4),lpm_xdrange(2,2))
      if(if3d)lpm_binb(5) = max(lpm_binb(5),lpm_xdrange(1,3))
      if(if3d)lpm_binb(6) = min(lpm_binb(6),lpm_xdrange(2,3))

      if (iperiodicx .eq. 1) then
         lpm_binb(1) = lpm_xdrange(1,1)
         lpm_binb(2) = lpm_xdrange(2,1)
      endif
      if (iperiodicy .eq. 1) then
         lpm_binb(3) = lpm_xdrange(1,2)
         lpm_binb(4) = lpm_xdrange(2,2)
      endif
      if (iperiodicz .eq. 1 .and. if3d) then
         lpm_binb(5) = lpm_xdrange(1,3)
         lpm_binb(6) = lpm_xdrange(2,3)
      endif

      ifac(1) = 1
      ifac(2) = 1
      ifac(3) = 1
      icount(1) = 0
      icount(2) = 0
      icount(3) = 0
      d2new(1) = lpm_d2chk(2)
      d2new(2) = lpm_d2chk(2)
      d2new(3) = lpm_d2chk(2)

      lpm_ndxgp = floor( (lpm_binb(2) - lpm_binb(1))/d2new(1))
      lpm_ndygp = floor( (lpm_binb(4) - lpm_binb(3))/d2new(2))
      lpm_ndzgp = 1
      if (if3d) lpm_ndzgp = floor( (lpm_binb(6) - lpm_binb(5))/d2new(3))

c     if (lpm_ndxgp*lpm_ndygp*lpm_ndzgp .gt. np) then
      if (lpm_ndxgp*lpm_ndygp*lpm_ndzgp .gt. np .or. 
     >    int(lpm_rparam(4)) .eq. 1) then
         nmax = 1000
         d2chk_save = lpm_d2chk(2)
         
         do i=1,nmax
         do j=0,ndim-1
            iflg = 0
            ifac(j+1) = 1 + i
            d2new(j+1) = (lpm_binb(2+2*j) - lpm_binb(1+2*j))/ifac(j+1)
            nbb = ifac(1)*ifac(2)*ifac(3)
            if (int(lpm_rparam(4)) .eq. 0) then
               if(d2new(j+1) .lt. d2chk_save .or. nbb .gt. np) iflg = 1
            elseif (int(lpm_rparam(4)) .eq. 1) then
               if( nbb .gt. np ) iflg = 1
            endif
            if (iflg .eq. 1) then
               icount(j+1) = icount(j+1) + 1
               ifac(j+1) = ifac(j+1) - icount(j+1)
               d2new(j+1) = (lpm_binb(2+2*j) -lpm_binb(1+2*j))/ifac(j+1)
            endif
         enddo
            if (icount(1) .gt. 0) then
            if (icount(2) .gt. 0) then
            if (icount(3) .gt. 0) then
               exit
            endif
            endif
            endif
         enddo
      endif

! -------------------------------------------------------
c SETUP 3D BACKGROUND GRID PARAMETERS FOR GHOST PARTICLES
! -------------------------------------------------------
      ! how many spacings in each direction
      lpm_ndxgp = floor( (lpm_binb(2) - lpm_binb(1))/d2new(1))
      lpm_ndygp = floor( (lpm_binb(4) - lpm_binb(3))/d2new(2))
      lpm_ndzgp = 1
      if (if3d) lpm_ndzgp = floor( (lpm_binb(6) - lpm_binb(5))/d2new(3))

      ! grid spacing for that many spacings
      lpm_rdxgp = (lpm_binb(2) - lpm_binb(1))/real(lpm_ndxgp)
      lpm_rdygp = (lpm_binb(4) - lpm_binb(3))/real(lpm_ndygp)
      lpm_rdzgp = 1.
      if (if3d) lpm_rdzgp = (lpm_binb(6) - lpm_binb(5))/real(lpm_ndzgp)

      ninc = 2
      rxlbin = lpm_binb(1)
      rxrbin = lpm_binb(2)
      rylbin = lpm_binb(3)
      ryrbin = lpm_binb(4)
      rzlbin = lpm_binb(5)
      rzrbin = lpm_binb(6)
      if (iperiodicx .ne. 1) then
         rxlbin = rxlbin - ninc/2*lpm_rdxgp
         rxrbin = rxrbin + ninc/2*lpm_rdxgp
         rxlbin = max(rxlbin,lpm_xdrange(1,1))
         rxrbin = min(rxrbin,lpm_xdrange(2,1))
      endif
      if (iperiodicy .ne. 1) then
         rylbin = rylbin - ninc/2*lpm_rdygp
         ryrbin = ryrbin + ninc/2*lpm_rdygp
         rylbin = max(rylbin,lpm_xdrange(1,2))
         ryrbin = min(ryrbin,lpm_xdrange(2,2))
      endif
      if (iperiodicz .ne. 1) then
      if (if3d) then
         rzlbin = rzlbin - ninc/2*lpm_rdzgp
         rzrbin = rzrbin + ninc/2*lpm_rdzgp
         rzlbin = max(rzlbin,lpm_xdrange(1,3))
         rzrbin = min(rzrbin,lpm_xdrange(2,3))
      endif
      endif

      nbin_now = lpm_ndxgp*lpm_ndygp*lpm_ndzgp

      if (int(lpm_rparam(4)) .eq. 1) return ! only for projection

! see which bins are in which elements
      lpm_neltb = 0
      do ie=1,nelt
      do k=1,nz1
      do j=1,ny1
      do i=1,nx1
         rxval = xm1(i,j,k,ie) 
         ryval = ym1(i,j,k,ie) 
         rzval = 0.
         if(if3d) rzval = zm1(i,j,k,ie)

         if (rxval .gt. lpm_binb(2)) goto 1233
         if (rxval .lt. lpm_binb(1)) goto 1233
         if (ryval .gt. lpm_binb(4)) goto 1233
         if (ryval .lt. lpm_binb(3)) goto 1233
         if (if3d .and. rzval .gt. lpm_binb(6)) goto 1233
         if (if3d .and. rzval .lt. lpm_binb(5)) goto 1233

         ii    = floor((rxval-lpm_binb(1))/lpm_rdxgp) 
         jj    = floor((ryval-lpm_binb(3))/lpm_rdygp) 
         kk    = floor((rzval-lpm_binb(5))/lpm_rdzgp) 
         if (.not. if3d) kk = 0
         if (ii .eq. lpm_ndxgp) ii = lpm_ndxgp - 1
         if (jj .eq. lpm_ndygp) jj = lpm_ndygp - 1
         if (kk .eq. lpm_ndzgp) kk = lpm_ndzgp - 1
         ndum  = ii + lpm_ndxgp*jj + lpm_ndxgp*lpm_ndygp*kk
         nrank = modulo(ndum,np)

         lpm_neltb = lpm_neltb + 1
         if(lpm_neltb .gt. lpm_lbmax) then
           write(6,*) 'increase lbmax',nid,lpm_neltb,lpm_lbmax
           call exitt
         endif

         lpm_er_map(1,lpm_neltb) = ie
         lpm_er_map(2,lpm_neltb) = nid
         lpm_er_map(3,lpm_neltb) = ndum
         lpm_er_map(4,lpm_neltb) = nrank
         lpm_er_map(5,lpm_neltb) = nrank
         lpm_er_map(6,lpm_neltb) = nrank

         if (lpm_neltb .gt. 1) then
         do il=1,lpm_neltb-1
            if (lpm_er_map(1,il) .eq. ie) then
            if (lpm_er_map(4,il) .eq. nrank) then
               lpm_neltb = lpm_neltb - 1
               goto 1233
            endif
            endif
         enddo
         endif
 1233 continue
      enddo
      enddo
      enddo
      enddo

      nxyz = lx1*ly1*lz1
      do ie=1,lpm_neltb
         iee = lpm_er_map(1,ie)
         call copy(lpm_xm1b(1,1,1,1,ie), xm1(1,1,1,iee),nxyz)
         call copy(lpm_xm1b(1,1,1,2,ie), ym1(1,1,1,iee),nxyz)
         call copy(lpm_xm1b(1,1,1,3,ie), zm1(1,1,1,iee),nxyz)
      enddo

      lpm_neltbb = lpm_neltb

      do ie=1,lpm_neltbb
         call icopy(lpm_er_maps(1,ie),lpm_er_map(1,ie),LPM_LRMAX)
      enddo


      nl   = 0
      nii  = LPM_LRMAX
      njj  = 6
      nrr  = nxyz*3
      nkey = 3
      call fgslib_crystal_tuple_transfer(i_cr_hndl,lpm_neltb,lpm_lbmax
     >                  , lpm_er_map,nii,partl,nl,lpm_xm1b,nrr,njj)
      call fgslib_crystal_tuple_sort    (i_cr_hndl,lpm_neltb
     $              , lpm_er_map,nii,partl,nl,lpm_xm1b,nrr,nkey,1)


      do ie=1,lpm_neltb
      do k=1,nz1
      do j=1,ny1
      do i=1,nx1
         rxval = lpm_xm1b(i,j,k,1,ie)
         ryval = lpm_xm1b(i,j,k,2,ie)
         rzval = 0.
         if(if3d) rzval = lpm_xm1b(i,j,k,3,ie)
         
         ii    = floor((rxval-lpm_binb(1))/lpm_rdxgp) 
         jj    = floor((ryval-lpm_binb(3))/lpm_rdygp) 
         kk    = floor((rzval-lpm_binb(5))/lpm_rdzgp) 
         if (.not. if3d) kk = 0
         if (ii .eq. lpm_ndxgp) ii = lpm_ndxgp - 1
         if (jj .eq. lpm_ndygp) jj = lpm_ndygp - 1
         if (kk .eq. lpm_ndzgp) kk = lpm_ndzgp - 1
         ndum  = ii + lpm_ndxgp*jj + lpm_ndxgp*lpm_ndygp*kk

         lpm_modgp(i,j,k,ie,1) = ii
         lpm_modgp(i,j,k,ie,2) = jj
         lpm_modgp(i,j,k,ie,3) = kk
         lpm_modgp(i,j,k,ie,4) = ndum
   
      enddo
      enddo
      enddo
      enddo

      do ie=1,lpm_neltb
         lpm_xerange(1,1,ie) = vlmin(lpm_xm1b(1,1,1,1,ie),nxyz)
         lpm_xerange(2,1,ie) = vlmax(lpm_xm1b(1,1,1,1,ie),nxyz)
         lpm_xerange(1,2,ie) = vlmin(lpm_xm1b(1,1,1,2,ie),nxyz)
         lpm_xerange(2,2,ie) = vlmax(lpm_xm1b(1,1,1,2,ie),nxyz)
         lpm_xerange(1,3,ie) = vlmin(lpm_xm1b(1,1,1,3,ie),nxyz)
         lpm_xerange(2,3,ie) = vlmax(lpm_xm1b(1,1,1,3,ie),nxyz)

         ilow  = floor((lpm_xerange(1,1,ie) - lpm_binb(1))/lpm_rdxgp)
         ihigh = floor((lpm_xerange(2,1,ie) - lpm_binb(1))/lpm_rdxgp)
         jlow  = floor((lpm_xerange(1,2,ie) - lpm_binb(3))/lpm_rdygp)
         jhigh = floor((lpm_xerange(2,2,ie) - lpm_binb(3))/lpm_rdygp)
         klow  = floor((lpm_xerange(1,3,ie) - lpm_binb(5))/lpm_rdzgp)
         khigh = floor((lpm_xerange(2,3,ie) - lpm_binb(5))/lpm_rdzgp)
         if (.not. if3d) then
            klow = 0
            khigh = 0
         endif

         lpm_el_map(1,ie) = ilow  + lpm_ndxgp*jlow  
     >                            + lpm_ndxgp*lpm_ndygp*klow
         lpm_el_map(2,ie) = ihigh + lpm_ndxgp*jhigh 
     >                            + lpm_ndxgp*lpm_ndygp*khigh
         lpm_el_map(3,ie) = ilow
         lpm_el_map(4,ie) = ihigh
         lpm_el_map(5,ie) = jlow
         lpm_el_map(6,ie) = jhigh
         lpm_el_map(7,ie) = klow
         lpm_el_map(8,ie) = khigh
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_comm_ghost_create
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      character*132 deathmessage
      real xdlen,ydlen,zdlen,rxdrng(3),rxnew(3)
      integer iadd(3),gpsave(27)
      real map(LPM_LRP_PRO)

      integer  el_face_num(18),el_edge_num(36),el_corner_num(24),
     >                            nfacegp, nedgegp, ncornergp

c     face, edge, and corner number, x,y,z are all inline, so stride=3
      el_face_num = (/ -1,0,0, 1,0,0, 0,-1,0, 0,1,0, 0,0,-1, 0,0,1 /)
      el_edge_num = (/ -1,-1,0 , 1,-1,0, 1,1,0 , -1,1,0 ,
     >                  0,-1,-1, 1,0,-1, 0,1,-1, -1,0,-1,
     >                  0,-1,1 , 1,0,1 , 0,1,1 , -1,0,1  /)
      el_corner_num = (/ -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1,
     >                   -1,-1,1,  1,-1,1,  1,1,1,  -1,1,1 /)

      nfacegp   = 4  ! number of faces
      nedgegp   = 4  ! number of edges
      ncornergp = 0  ! number of corners

      if (if3d) then
         nfacegp   = 6  ! number of faces
         nedgegp   = 12 ! number of edges
         ncornergp = 8  ! number of corners
      endif

      iperiodicx = int(lpm_rparam(8))
      iperiodicy = int(lpm_rparam(9))
      iperiodicz = int(lpm_rparam(10))

! ------------------------
c CREATING GHOST PARTICLES
! ------------------------
      jx    = 1
      jy    = 2
      jz    = 3
      jdp   = int(lpm_rparam(5))

      xdlen = lpm_binb(2) - lpm_binb(1)
      ydlen = lpm_binb(4) - lpm_binb(3)
      zdlen = -1.
      if (if3d) zdlen = lpm_binb(6) - lpm_binb(5)
      if (iperiodicx .ne. 1) xdlen = -1
      if (iperiodicy .ne. 1) ydlen = -1
      if (iperiodicz .ne. 1) zdlen = -1

      rxdrng(1) = xdlen
      rxdrng(2) = ydlen
      rxdrng(3) = zdlen

      lpm_npart_gp = 0

      rfac = 1.0

      do ip=1,lpm_npart

         call lpm_project_map(map,lpm_y(1,ip),lpm_ydot(1,ip)
     >                       ,lpm_ydotc(1,ip),lpm_rprop(1,ip))
         lpm_cp_map(1,ip) = lpm_y(jx,ip)      ! x coord
         lpm_cp_map(2,ip) = lpm_y(jy,ip)      ! y coord
         lpm_cp_map(3,ip) = lpm_y(jz,ip)      ! z coord
         lpm_cp_map(4,ip) = lpm_rprop(jdp,ip) ! filter non-dim scale
         do j=1,LPM_LRP_PRO
            lpm_cp_map(4+j,ip) = map(j)
         enddo

         rxval = lpm_cp_map(1,ip)
         ryval = lpm_cp_map(2,ip)
         rzval = 0.
         if(if3d) rzval = lpm_cp_map(3,ip)

         iip    = lpm_iprop(8,ip)
         jjp    = lpm_iprop(9,ip)
         kkp    = lpm_iprop(10,ip)

         rxl = lpm_binb(1) + lpm_rdxgp*iip
         rxr = rxl + lpm_rdxgp
         ryl = lpm_binb(3) + lpm_rdygp*jjp
         ryr = ryl + lpm_rdygp
         rzl = 0.0
         rzr = 0.0
         if (if3d) then
            rzl = lpm_binb(5) + lpm_rdzgp*kkp
            rzr = rzl + lpm_rdzgp
         endif

         isave = 0

         ! faces
         do ifc=1,nfacegp
            ist = (ifc-1)*3
            ii1 = iip + el_face_num(ist+1) 
            jj1 = jjp + el_face_num(ist+2)
            kk1 = kkp + el_face_num(ist+3)

            iig = ii1
            jjg = jj1
            kkg = kk1

            distchk = 0.0
            dist = 0.0
            if (ii1-iip .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (ii1-iip .lt. 0) dist = dist +(rxval - rxl)**2
               if (ii1-iip .gt. 0) dist = dist +(rxval - rxr)**2
            endif
            if (jj1-jjp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (jj1-jjp .lt. 0) dist = dist +(ryval - ryl)**2
               if (jj1-jjp .gt. 0) dist = dist +(ryval - ryr)**2
            endif
            if (if3d) then
            if (kk1-kkp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (kk1-kkp .lt. 0) dist = dist +(rzval - rzl)**2
               if (kk1-kkp .gt. 0) dist = dist +(rzval - rzr)**2
            endif
            endif
            distchk = sqrt(distchk)
            dist = sqrt(dist)
            if (dist .gt. distchk) cycle

            iflgx = 0
            iflgy = 0
            iflgz = 0
            ! periodic if out of domain - add some ifsss
            if (iig .lt. 0 .or. iig .gt. lpm_ndxgp-1) then
               iflgx = 1
               iig =modulo(iig,lpm_ndxgp)
               if (iperiodicx .ne. 1) cycle
            endif
            if (jjg .lt. 0 .or. jjg .gt. lpm_ndygp-1) then
               iflgy = 1
               jjg =modulo(jjg,lpm_ndygp)
               if (iperiodicy .ne. 1) cycle
            endif
            if (kkg .lt. 0 .or. kkg .gt. lpm_ndzgp-1) then
               iflgz = 1  
               kkg =modulo(kkg,lpm_ndzgp)
               if (iperiodicz .ne. 1) cycle
            endif

            iflgsum = iflgx + iflgy + iflgz
            ndumn = iig + lpm_ndxgp*jjg + lpm_ndxgp*lpm_ndygp*kkg
            nrank = modulo(ndumn,np)

            if (nrank .eq. nid .and. iflgsum .eq. 0) cycle

            do i=1,isave
               if (gpsave(i) .eq. nrank .and. iflgsum .eq.0) goto 111
            enddo
            isave = isave + 1
            gpsave(isave) = nrank

            ibctype = iflgx+iflgy+iflgz
                 
            rxnew(1) = rxval
            rxnew(2) = ryval
            rxnew(3) = rzval
       
            iadd(1) = ii1
            iadd(2) = jj1
            iadd(3) = kk1

            call lpm_comm_check_periodic_gp(rxnew,rxdrng,iadd)
                 
            lpm_npart_gp = lpm_npart_gp + 1
            lpm_iprop_gp(1,lpm_npart_gp) = nrank
            lpm_iprop_gp(2,lpm_npart_gp) = iig
            lpm_iprop_gp(3,lpm_npart_gp) = jjg
            lpm_iprop_gp(4,lpm_npart_gp) = kkg
            lpm_iprop_gp(5,lpm_npart_gp) = ndumn

            lpm_rprop_gp(1,lpm_npart_gp) = rxnew(1)
            lpm_rprop_gp(2,lpm_npart_gp) = rxnew(2)
            lpm_rprop_gp(3,lpm_npart_gp) = rxnew(3)
            do k=4,LPM_LRP_GP
               lpm_rprop_gp(k,lpm_npart_gp) = lpm_cp_map(k,ip)
            enddo
  111 continue
         enddo

         ! edges
         do ifc=1,nedgegp
            ist = (ifc-1)*3
            ii1 = iip + el_edge_num(ist+1) 
            jj1 = jjp + el_edge_num(ist+2)
            kk1 = kkp + el_edge_num(ist+3)

            iig = ii1
            jjg = jj1
            kkg = kk1

            distchk = 0.0
            dist = 0.0
            if (ii1-iip .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (ii1-iip .lt. 0) dist = dist +(rxval - rxl)**2
               if (ii1-iip .gt. 0) dist = dist +(rxval - rxr)**2
            endif
            if (jj1-jjp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (jj1-jjp .lt. 0) dist = dist +(ryval - ryl)**2
               if (jj1-jjp .gt. 0) dist = dist +(ryval - ryr)**2
            endif
            if (if3d) then
            if (kk1-kkp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (kk1-kkp .lt. 0) dist = dist +(rzval - rzl)**2
               if (kk1-kkp .gt. 0) dist = dist +(rzval - rzr)**2
            endif
            endif
            distchk = sqrt(distchk)
            dist = sqrt(dist)
            if (dist .gt. distchk) cycle

            iflgx = 0
            iflgy = 0
            iflgz = 0
            ! periodic if out of domain - add some ifsss
            if (iig .lt. 0 .or. iig .gt. lpm_ndxgp-1) then
               iflgx = 1
               iig =modulo(iig,lpm_ndxgp)
               if (iperiodicx .ne. 1) cycle
            endif
            if (jjg .lt. 0 .or. jjg .gt. lpm_ndygp-1) then
               iflgy = 1
               jjg =modulo(jjg,lpm_ndygp)
               if (iperiodicy .ne. 1) cycle
            endif
            if (kkg .lt. 0 .or. kkg .gt. lpm_ndzgp-1) then
               iflgz = 1  
               kkg =modulo(kkg,lpm_ndzgp)
               if (iperiodicz .ne. 1) cycle
            endif

            iflgsum = iflgx + iflgy + iflgz
            ndumn = iig + lpm_ndxgp*jjg + lpm_ndxgp*lpm_ndygp*kkg
            nrank = modulo(ndumn,np)

            if (nrank .eq. nid .and. iflgsum .eq. 0) cycle

            do i=1,isave
               if (gpsave(i) .eq. nrank .and. iflgsum .eq.0) goto 222
            enddo
            isave = isave + 1
            gpsave(isave) = nrank

            ibctype = iflgx+iflgy+iflgz
                 
            rxnew(1) = rxval
            rxnew(2) = ryval
            rxnew(3) = rzval
       
            iadd(1) = ii1
            iadd(2) = jj1
            iadd(3) = kk1

            call lpm_comm_check_periodic_gp(rxnew,rxdrng,iadd)
                 
            lpm_npart_gp = lpm_npart_gp + 1
            lpm_iprop_gp(1,lpm_npart_gp) = nrank
            lpm_iprop_gp(2,lpm_npart_gp) = iig
            lpm_iprop_gp(3,lpm_npart_gp) = jjg
            lpm_iprop_gp(4,lpm_npart_gp) = kkg
            lpm_iprop_gp(5,lpm_npart_gp) = ndumn

            lpm_rprop_gp(1,lpm_npart_gp) = rxnew(1)
            lpm_rprop_gp(2,lpm_npart_gp) = rxnew(2)
            lpm_rprop_gp(3,lpm_npart_gp) = rxnew(3)
            do k=4,LPM_LRP_GP
               lpm_rprop_gp(k,lpm_npart_gp) = lpm_cp_map(k,ip)
            enddo
  222 continue
         enddo

         ! corners
         do ifc=1,ncornergp
            ist = (ifc-1)*3
            ii1 = iip + el_corner_num(ist+1) 
            jj1 = jjp + el_corner_num(ist+2)
            kk1 = kkp + el_corner_num(ist+3)

            iig = ii1
            jjg = jj1
            kkg = kk1

            distchk = 0.0
            dist = 0.0
            if (ii1-iip .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (ii1-iip .lt. 0) dist = dist +(rxval - rxl)**2
               if (ii1-iip .gt. 0) dist = dist +(rxval - rxr)**2
            endif
            if (jj1-jjp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (jj1-jjp .lt. 0) dist = dist +(ryval - ryl)**2
               if (jj1-jjp .gt. 0) dist = dist +(ryval - ryr)**2
            endif
            if (if3d) then
            if (kk1-kkp .ne. 0) then
               distchk = distchk + (rfac*lpm_d2chk(2))**2
               if (kk1-kkp .lt. 0) dist = dist +(rzval - rzl)**2
               if (kk1-kkp .gt. 0) dist = dist +(rzval - rzr)**2
            endif
            endif
            distchk = sqrt(distchk)
            dist = sqrt(dist)
            if (dist .gt. distchk) cycle

            iflgx = 0
            iflgy = 0
            iflgz = 0
            ! periodic if out of domain - add some ifsss
            if (iig .lt. 0 .or. iig .gt. lpm_ndxgp-1) then
               iflgx = 1
               iig =modulo(iig,lpm_ndxgp)
               if (iperiodicx .ne. 1) cycle
            endif
            if (jjg .lt. 0 .or. jjg .gt. lpm_ndygp-1) then
               iflgy = 1
               jjg =modulo(jjg,lpm_ndygp)
               if (iperiodicy .ne. 1) cycle
            endif
            if (kkg .lt. 0 .or. kkg .gt. lpm_ndzgp-1) then
               iflgz = 1  
               kkg =modulo(kkg,lpm_ndzgp)
               if (iperiodicz .ne. 1) cycle
            endif

            iflgsum = iflgx + iflgy + iflgz
            ndumn = iig + lpm_ndxgp*jjg + lpm_ndxgp*lpm_ndygp*kkg
            nrank = modulo(ndumn,np)

            if (nrank .eq. nid .and. iflgsum .eq. 0) cycle

            do i=1,isave
               if (gpsave(i) .eq. nrank .and. iflgsum .eq.0) goto 333
            enddo
            isave = isave + 1
            gpsave(isave) = nrank

            ibctype = iflgx+iflgy+iflgz
                 
            rxnew(1) = rxval
            rxnew(2) = ryval
            rxnew(3) = rzval
       
            iadd(1) = ii1
            iadd(2) = jj1
            iadd(3) = kk1

            call lpm_comm_check_periodic_gp(rxnew,rxdrng,iadd)
                 
            lpm_npart_gp = lpm_npart_gp + 1
            lpm_iprop_gp(1,lpm_npart_gp) = nrank
            lpm_iprop_gp(2,lpm_npart_gp) = iig
            lpm_iprop_gp(3,lpm_npart_gp) = jjg
            lpm_iprop_gp(4,lpm_npart_gp) = kkg
            lpm_iprop_gp(5,lpm_npart_gp) = ndumn

            lpm_rprop_gp(1,lpm_npart_gp) = rxnew(1)
            lpm_rprop_gp(2,lpm_npart_gp) = rxnew(2)
            lpm_rprop_gp(3,lpm_npart_gp) = rxnew(3)
            do k=4,LPM_LRP_GP
               lpm_rprop_gp(k,lpm_npart_gp) = lpm_cp_map(k,ip)
            enddo
  333 continue
         enddo

      enddo

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_comm_check_periodic_gp(rxnew,rxdrng,iadd)
      include 'SIZE'
      include 'TOTAL'
#     include "LPM"

      real rxnew(3), rxdrng(3)
      integer iadd(3), irett(3), ntype, ntypel(7)

      xloc = rxnew(1)
      yloc = rxnew(2)
      zloc = rxnew(3)

      xdlen = rxdrng(1)
      ydlen = rxdrng(2)
      zdlen = rxdrng(3)

      ii = iadd(1)
      jj = iadd(2)
      kk = iadd(3)

      irett(1) = 0
      irett(2) = 0
      irett(3) = 0

      if (xdlen .gt. 0 ) then
      if (ii .ge. lpm_ndxgp) then
         xloc = xloc - xdlen
         irett(1) = 1
         goto 123
      endif
      endif
      if (xdlen .gt. 0 ) then
      if (ii .lt. 0) then
         xloc = xloc + xdlen
         irett(1) = 1
         goto 123
      endif
      endif

  123 continue    
      if (ydlen .gt. 0 ) then
      if (jj .ge. lpm_ndygp) then
         yloc = yloc - ydlen
         irett(2) = 1
         goto 124
      endif
      endif
      if (ydlen .gt. 0 ) then
      if (jj .lt. 0) then
         yloc = yloc + ydlen
         irett(2) = 1
         goto 124
      endif
      endif
  124 continue

      if (if3d) then
         if (zdlen .gt. 0 ) then
         if (kk .ge. lpm_ndzgp) then
            zloc = zloc - zdlen
            irett(3) = 1
            goto 125
         endif
         endif
         if (zdlen .gt. 0 ) then
         if (kk .lt. 0) then
            zloc = zloc + zdlen
            irett(3) = 1
            goto 125
         endif
         endif
      endif
  125 continue

      rxnew(1) = xloc
      rxnew(2) = yloc
      rxnew(3) = zloc

      return
      end
c----------------------------------------------------------------------
      subroutine lpm_comm_ghost_send
      include 'SIZE'
#     include "LPM"

      logical partl         

      ! send ghost particles
      call fgslib_crystal_tuple_transfer(i_cr_hndl
     >                                  ,lpm_npart_gp,LPM_LPART_GP
     >                                  ,lpm_iprop_gp,LPM_LIP_GP
     >                                  ,partl,0
     >                                  ,lpm_rprop_gp,LPM_LRP_GP
     >                                  ,1)

      return
      end
c----------------------------------------------------------------------
