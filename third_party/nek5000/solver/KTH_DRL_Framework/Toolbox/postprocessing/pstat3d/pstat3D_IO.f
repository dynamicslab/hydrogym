!> @file pstat3D_IO.f
!! @ingroup pstat3d
!! @brief Post processing I/O routines for statistics module
!! @author Adam Peplinski
!! @date Mar 13, 2019
!=======================================================================
!> @brief Write field data data to the file
!! @ingroup pstat3d
      subroutine pstat3d_mfo
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
!      include 'RESTART'
!      include 'PARALLEL'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! local variables
      integer il
!-----------------------------------------------------------------------
      ! save all fields
      ifxyo = .true.
      ifvo = .true.
      ifpo = .false.
      ifto = .true.
      do il=1, npscal
          ifpsco(il)= .false.
      enddo

      call outpost(pstat_ruavg(1,1,1),pstat_ruavg(1,1,2), ! U,V,W,uu
     $     pstat_ruavg(1,1,3),pr,pstat_ruavg(1,1,5),'a01')
      call outpost(pstat_ruavg(1,1,6),pstat_ruavg(1,1,7), ! vv,ww,uv,uw
     $     pstat_ruavg(1,1,9),pr,pstat_ruavg(1,1,11),'a02')
      call outpost(pstat_ruavg(1,1,10),pstat_ruavg(1,1,4), ! vw,P,pp,ppp
     $     pstat_ruavg(1,1,8),pr,pstat_ruavg(1,1,27),'a03')
      call outpost(pstat_ruavg(1,1,38),pstat_ruavg(1,1,24), ! pppp,uuu,vvv,www
     $     pstat_ruavg(1,1,25),pr,pstat_ruavg(1,1,26),'a04')
      call outpost(pstat_ruavg(1,1,28),pstat_ruavg(1,1,29), ! uuv,uuw,uvv,vvw
     $     pstat_ruavg(1,1,30),pr,pstat_ruavg(1,1,31),'a05')
      call outpost(pstat_ruavg(1,1,32),pstat_ruavg(1,1,33), ! uww,vww,uvw,Pxx
     $     pstat_ruavg(1,1,34),pr,pstat_rutmp(1,1,1),'a06')
      call outpost(pstat_rutmp(1,1,2),pstat_rutmp(1,1,3), ! Pyy,Pzz,Pxy,Pxz
     $     pstat_rutmp(1,1,4),pr,pstat_rutmp(1,1,5),'a07')
      call outpost(pstat_rutmp(1,1,6),pstat_ruavg(1,1,39), ! Pyz,Dxx,Dyy,Dzz
     $     pstat_ruavg(1,1,40),pr,pstat_ruavg(1,1,41),'a08')
      call outpost(pstat_ruavg(1,1,42),pstat_ruavg(1,1,43), ! Dxy,Dxz,Dyz,Txx
     $     pstat_ruavg(1,1,44),pr,pstat_runew(1,1,22),'a09')
      call outpost(pstat_runew(1,1,23),pstat_runew(1,1,24), ! Tyy,Tzz,Txy,Txz
     $     pstat_runew(1,1,25),pr,pstat_runew(1,1,27),'a10')
      call outpost(pstat_runew(1,1,26),pstat_runew(1,1,16), ! Tyz,VDxx,VDyy,VDzz
     $     pstat_runew(1,1,17),pr,pstat_runew(1,1,18),'a11')
      call outpost(pstat_runew(1,1,19),pstat_runew(1,1,21), ! VDxy,VDxz,VDyz,Pixx
     $     pstat_runew(1,1,20),pr,pstat_rutmp(1,1,7),'a12')
      call outpost(pstat_rutmp(1,1,8),pstat_rutmp(1,1,9), ! Piyy,Pizz,Pixy,Pixz
     $     pstat_rutmp(1,1,10),pr,pstat_rutmp(1,1,11),'a13')
      call outpost(pstat_rutmp(1,1,12),pstat_runew(1,1,10), ! Piyz,Cxx,Cyy,Czz
     $     pstat_runew(1,1,11),pr,pstat_runew(1,1,12),'a14')
      call outpost(pstat_runew(1,1,13),pstat_runew(1,1,15), ! Cxy,Cxz,Cyz,Pk
     $     pstat_runew(1,1,14),pr,pstat_rutmp(1,1,13),'a15')
      call outpost(pstat_rutmp(1,1,14),pstat_rutmp(1,1,15), ! Dk,Tk,VDk,Pik
     $     pstat_rutmp(1,1,16),pr,pstat_rutmp(1,1,17),'a16')
      call outpost(pstat_rutmp(1,1,18),pstat_rutmp(1,1,19), ! Ck,Resk,PTxx,PTyy
     $     pstat_ruavg(1,1,12),pr,pstat_ruavg(1,1,13),'a17')
      call outpost(pstat_ruavg(1,1,14),pstat_ruavg(1,1,15), ! PTzz,PTxy,PTxz,PTyz
     $     pstat_ruavg(1,1,16),pr,pstat_ruavg(1,1,17),'a18')
      call outpost(pstat_ruavg(1,1,18),pstat_ruavg(1,1,19), ! PSxx,PSyy,PSzz,PSxy
     $     pstat_ruavg(1,1,20),pr,pstat_ruavg(1,1,21),'a19')
      call outpost(pstat_ruavg(1,1,22),pstat_ruavg(1,1,23), ! PSxz,PTyz,dUdx,dUdy
     $     pstat_runew(1,1,1),pr,pstat_runew(1,1,2),'a20')
      call outpost(pstat_runew(1,1,3),pstat_runew(1,1,4), ! dUdz,dVdx,dVdy,dVdz
     $     pstat_runew(1,1,5),pr,pstat_runew(1,1,6),'a21')
      call outpost(pstat_runew(1,1,7),pstat_runew(1,1,8), ! dWdx,dWdy,dWdz,Tk
     $     pstat_runew(1,1,9),pr,pstat_rutmp(1,1,15),'a22')

      return
      end subroutine
!=======================================================================
!> @brief Read interpolation points position and redistribute them
!! @ingroup pstat3d
      subroutine pstat3d_mfi_interp
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'PSTAT3D'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer il, jl            ! loop index
      integer ierr              ! error flag
      integer ldiml             ! dimesion of interpolation file
      integer nptsr             ! number of points in the file
      integer npass             ! number of messages to send
      real rtmp_pts(ldim,lhis)
      real*4 rbuffl(2*ldim*lhis)
      real rtmp1, rtmp2
      character*132 fname       ! file name
      integer hdrl
      parameter (hdrl=32)
      character*32 hdr         ! file header
      character*4 dummy
      real*4 bytetest

      ! functions
      logical if_byte_swap_test

!#define DEBUG
#ifdef DEBUG
      character*3 str1, str2
      integer iunit
      ! call number
      integer icalld
      save icalld
      data icalld /0/
#endif
!-----------------------------------------------------------------------
      ! master opens files and gets point number
      ierr = 0
      if (nid.eq.pid00) then
         !open the file
         fname='DATA/int_pos'
         call byte_open(fname,ierr)

         ! read header
         call blank     (hdr,hdrl)
         call byte_read (hdr,hdrl/4,ierr)
         if (ierr.ne.0) goto 101

         ! big/little endian test
         call byte_read (bytetest,1,ierr)
         if(ierr.ne.0) goto 101
         if_byte_sw = if_byte_swap_test(bytetest,ierr)
         if(ierr.ne.0) goto 101

         ! extract header information
         read(hdr,*,iostat=ierr) dummy, wdsizr, ldiml, nptsr
      endif

 101  continue

      call mntr_check_abort(pstat_id,ierr,
     $       'pstat_mfi_interp: Error opening point files')

      ! broadcast header data
      call bcast(wdsizr,isize)
      call bcast(ldiml,isize)
      call bcast(nptsr,isize)
      call bcast(if_byte_sw,lsize)

      ! check dimension consistency
      if (ldim.ne.ldiml) call mntr_check_abort(pstat_id,
     $       'pstat_mfi_interp: Inconsisten dimension.')

      ! calculate point distribution; I assume it is post-processing
      ! done on small number of cores, so I assume nptsr >> mp
      pstat_nptot = nptsr
      pstat_npt = nptsr/mp
      if (pstat_npt.gt.0) then
         pstat_npt1 = mod(pstat_nptot,mp)
      else
         pstat_npt1 = pstat_nptot
      endif
      if (nid.lt.pstat_npt1) pstat_npt = pstat_npt +1

      ! stamp logs
      call mntr_logi(pstat_id,lp_prd,
     $          'Interpolation point number :', pstat_nptot)

      ierr = 0
      if (pstat_npt.gt.lhis) ierr = 1
      call mntr_check_abort(pstat_id,ierr,
     $       'pstat_mfi_interp: lhis too small')

      ! read and redistribute points
      ! this part is not optimised, but it is post-processing
      ! done locally, so I don't care
      if (nid.eq.pid00) then
         if (pstat_nptot.gt.0) then
            ! read points for the master rank
            ldiml = ldim*pstat_npt*wdsizr/4
            call byte_read (rbuffl,ldiml,ierr)

            ! get byte shift
            if (if_byte_sw) then
               if(wdsizr.eq.8) then
                  call byte_reverse8(rbuffl,ldiml,ierr)
               else
                  call byte_reverse(rbuffl,ldiml,ierr)
               endif
            endif

            ! copy data
            ldiml = ldim*pstat_npt
            if (wdsizr.eq.4) then
               call copy4r(pstat_int_pts,rbuffl,ldiml)
            else
               call copy(pstat_int_pts,rbuffl,ldiml)
            endif

            ! redistribute rest of points
            npass = min(mp,pstat_nptot)
            do il = 1,npass-1
               nptsr = pstat_npt
               if (pstat_npt1.gt.0.and.il.ge.pstat_npt1) then
                  nptsr = pstat_npt -1
               endif
               ! read points for the slave rank
               ldiml = ldim*nptsr*wdsizr/4
               call byte_read (rbuffl,ldiml,ierr)

               ! get byte shift
               if (if_byte_sw) then
                  if(wdsizr.eq.8) then
                     call byte_reverse8(rbuffl,ldiml,ierr)
                  else
                     call byte_reverse(rbuffl,ldiml,ierr)
                  endif
               endif

               ! copy data
               ldiml = ldim*nptsr
               if (wdsizr.eq.4) then
                  call copy4r(rtmp_pts,rbuffl,ldiml)
               else
                  call copy(rtmp_pts,rbuffl,ldiml)
               endif

               ! send data
               ldiml = ldiml*wdsizr
               call csend(il,rtmp_pts,ldiml,il,jl)
            enddo
         endif
      else
         if (pstat_npt.gt.0) then
            call crecv2(nid,pstat_int_pts,ldim*pstat_npt*wdsize,0)
         endif
      endif

      ! master closes files
      if (nid.eq.pid00) then
        call byte_close(ierr)
      endif

#ifdef DEBUG
      ! for testing
      ! to output refinement
      icalld = icalld+1
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='INTpos.txt'//str1//'i'//str2)

      write(iunit,*) pstat_nptot, pstat_npt
      do il=1, pstat_npt
         write(iunit,*) il, (pstat_int_pts(jl,il),jl=1,ldim)
      enddo

      close(iunit)
#endif
#undef DEBUG

      return
      end subroutine
!=======================================================================
!> @brief Geather data and write it down
!! @ingroup pstat3d
      subroutine pstat3d_mfo_interp
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'PARALLEL'
      include 'GEOM'
      include 'PSTAT3D'

      ! local variables
      integer il
      integer ierr         ! error flag
      character*132 fname       ! file name
      character*500 head   ! file header
      character*500 ftm    ! header format
      real*4 test
      parameter (test=6.54321)

      real  rtmp

      real lx,ly,lz        ! box dimensions
      integer nlx,nly,nlz  ! for tensor product meshes
      integer iavfr
      integer int_nvar     ! number of interpolated varibales

      integer wdsl, isl    ! double and integer sizes

      ! functions
      real glmin, glmax
!-----------------------------------------------------------------------
      ! double and integer sizes
      wdsl = wdsize/4
      isl = isize/4

      ! gether information for file header
      il = lx1*ly1*lz1*nelt
      lx = glmax(xm1,il) - glmin(xm1,il)
      ly = glmax(ym1,il) - glmin(ym1,il)
      if (if3D) then
         lz = glmax(zm1,il) - glmin(zm1,il)
      else
         lz = 0.0     ! this should be changed
      endif
      ! for tensor product meshes; element count
      nlx = nelgv
      nly = 1
      nlz = 1
      ! frequency of averaging in steps
      iavfr = pstat_nstep
      ! stat averagign time
      rtmp = pstat_etime-pstat_stime
      ! currently I interpolate and save 87 variables
      int_nvar = 87
      ! this is far from optimal, but for post-processing I do not care
      ! master opens files and writes header
      ierr = 0
      if (nid.eq.pid00) then
         !open the file
         fname='int_fld'
         call byte_open(fname,ierr)

         if (ierr.ne.0) goto 20

         ! write file's header
         ftm="('#iv1',1x,i1,1x,"//
     $   "1p,'(Re =',e17.9,') (Lx, Ly, Lz =',3e17.9,"//
     $   "') (nelx, nely, nelz =',3i9,') (Polynomial order =',3i9,"//
     $   "') (Nstat =',i9,') (start time =',e17.9,"//
     $   "') (end time =',e17.9,') (effective average time =',e17.9,"//
     $   "') (time step =',e17.9,') (nrec =',i9"//
     $   "') (time interval =',e17.9,') (npoints =',i9,')')"
         write(head,ftm) wdsize,
     $    1.0/param(2),lx,ly,lz,nlx,nly,nlz,lx1,ly1,lz1,
     $    int_nvar,pstat_stime,pstat_etime,rtmp/iavfr,
     $    rtmp/pstat_istepr,pstat_istepr/iavfr,rtmp,pstat_nptot
         call byte_write(head,115,ierr)

         ! write big/little endian test
         call byte_write(test,1,ierr)

         if (ierr.ne.0) goto 20

         ! write parameter set with all the digits
         call byte_write(1.0/param(2),wdsl,ierr)
         call byte_write(lx,wdsl,ierr)
         call byte_write(ly,wdsl,ierr)
         call byte_write(lz,wdsl,ierr)
         call byte_write(nlx,isl,ierr)
         call byte_write(nly,isl,ierr)
         call byte_write(nlz,isl,ierr)
         call byte_write(lx1,isl,ierr)
         call byte_write(ly1,isl,ierr)
         call byte_write(lz1,isl,ierr)
         call byte_write(int_nvar,isl,ierr)
         call byte_write(pstat_stime,wdsl,ierr)
         call byte_write(pstat_etime,wdsl,ierr)
         call byte_write(rtmp/iavfr,wdsl,ierr)
         call byte_write(rtmp/pstat_istepr,wdsl,ierr)
         call byte_write(pstat_istepr/iavfr,isl,ierr)
         call byte_write(rtmp,wdsl,ierr)
         call byte_write(pstat_nptot,isl,ierr)
      endif

 20   continue
      call mntr_check_abort(pstat_id,ierr,
     $     'Error opening interpolation file in pstat_mfo_interp.')

      ! write down point coordinates
      call pstat3d_field_out(pstat_int_pts,ldim,ierr)
      call mntr_check_abort(pstat_id,ierr,
     $    'Error writing coordinates in pstat_mfo_interp.')

      ! geather single field data and write it down to the file
      ! this is kind of strange, but I have to keep variables order from a** files
      call pstat3d_field_out(pstat_int_avg(1,1),1,ierr) ! U
      call mntr_check_abort(pstat_id,ierr,'Error writing U interp.')
      call pstat3d_field_out(pstat_int_avg(1,2),1,ierr) ! V
      call mntr_check_abort(pstat_id,ierr,'Error writing V interp.')
      call pstat3d_field_out(pstat_int_avg(1,3),1,ierr) ! W
      call mntr_check_abort(pstat_id,ierr,'Error writing W interp.')
      call pstat3d_field_out(pstat_int_avg(1,5),1,ierr) ! uu
      call mntr_check_abort(pstat_id,ierr,'Error writing uu interp.')
      call pstat3d_field_out(pstat_int_avg(1,6),1,ierr) ! vv
      call mntr_check_abort(pstat_id,ierr,'Error writing vv interp.')
      call pstat3d_field_out(pstat_int_avg(1,7),1,ierr) ! ww
      call mntr_check_abort(pstat_id,ierr,'Error writing ww interp.')
      call pstat3d_field_out(pstat_int_avg(1,9),1,ierr) ! uv
      call mntr_check_abort(pstat_id,ierr,'Error writing uv interp.')
      call pstat3d_field_out(pstat_int_avg(1,11),1,ierr) ! uw
      call mntr_check_abort(pstat_id,ierr,'Error writing uw interp.')
      call pstat3d_field_out(pstat_int_avg(1,10),1,ierr) ! vw
      call mntr_check_abort(pstat_id,ierr,'Error writing vw interp.')
      call pstat3d_field_out(pstat_int_avg(1,4),1,ierr) ! P
      call mntr_check_abort(pstat_id,ierr,'Error writing P interp.')
      call pstat3d_field_out(pstat_int_avg(1,8),1,ierr) ! pp
      call mntr_check_abort(pstat_id,ierr,'Error writing pp interp.')
      call pstat3d_field_out(pstat_int_avg(1,27),1,ierr) ! ppp
      call mntr_check_abort(pstat_id,ierr,'Error writing ppp interp.')
      call pstat3d_field_out(pstat_int_avg(1,38),1,ierr) ! pppp
      call mntr_check_abort(pstat_id,ierr,'Error writing pppp interp.')
      call pstat3d_field_out(pstat_int_avg(1,24),1,ierr) ! uuu
      call mntr_check_abort(pstat_id,ierr,'Error writing uuu interp.')
      call pstat3d_field_out(pstat_int_avg(1,25),1,ierr) ! vvv
      call mntr_check_abort(pstat_id,ierr,'Error writing vvv interp.')
      call pstat3d_field_out(pstat_int_avg(1,26),1,ierr) ! www
      call mntr_check_abort(pstat_id,ierr,'Error writing www interp.')
      call pstat3d_field_out(pstat_int_avg(1,28),1,ierr) ! uuv
      call mntr_check_abort(pstat_id,ierr,'Error writing uuv interp.')
      call pstat3d_field_out(pstat_int_avg(1,29),1,ierr) ! uuw
      call mntr_check_abort(pstat_id,ierr,'Error writing uuw interp.')
      call pstat3d_field_out(pstat_int_avg(1,30),1,ierr) ! uvv
      call mntr_check_abort(pstat_id,ierr,'Error writing uvv interp.')
      call pstat3d_field_out(pstat_int_avg(1,31),1,ierr) ! vvw
      call mntr_check_abort(pstat_id,ierr,'Error writing vvw interp.')
      call pstat3d_field_out(pstat_int_avg(1,32),1,ierr) ! uww
      call mntr_check_abort(pstat_id,ierr,'Error writing uww interp.')
      call pstat3d_field_out(pstat_int_avg(1,33),1,ierr) ! vww
      call mntr_check_abort(pstat_id,ierr,'Error writing vww interp.')
      call pstat3d_field_out(pstat_int_avg(1,34),1,ierr) ! uvw
      call mntr_check_abort(pstat_id,ierr,'Error writing uvw interp.')
      call pstat3d_field_out(pstat_int_tmp(1,1),1,ierr) ! Pxx
      call mntr_check_abort(pstat_id,ierr,'Error writing Pxx interp.')
      call pstat3d_field_out(pstat_int_tmp(1,2),1,ierr) ! Pyy
      call mntr_check_abort(pstat_id,ierr,'Error writing Pyy interp.')
      call pstat3d_field_out(pstat_int_tmp(1,3),1,ierr) ! Pzz
      call mntr_check_abort(pstat_id,ierr,'Error writing Pzz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,4),1,ierr) ! Pxy
      call mntr_check_abort(pstat_id,ierr,'Error writing Pxy interp.')
      call pstat3d_field_out(pstat_int_tmp(1,5),1,ierr) ! Pxz
      call mntr_check_abort(pstat_id,ierr,'Error writing Pxz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,6),1,ierr) ! Pyz
      call mntr_check_abort(pstat_id,ierr,'Error writing Pyz interp.')
      call pstat3d_field_out(pstat_int_avg(1,39),1,ierr) ! Dxx
      call mntr_check_abort(pstat_id,ierr,'Error writing Dxx interp.')
      call pstat3d_field_out(pstat_int_avg(1,40),1,ierr) ! Dyy
      call mntr_check_abort(pstat_id,ierr,'Error writing Dyy interp.')
      call pstat3d_field_out(pstat_int_avg(1,41),1,ierr) ! Dzz
      call mntr_check_abort(pstat_id,ierr,'Error writing Dzz interp.')
      call pstat3d_field_out(pstat_int_avg(1,42),1,ierr) ! Dxy
      call mntr_check_abort(pstat_id,ierr,'Error writing Dxy interp.')
      call pstat3d_field_out(pstat_int_avg(1,43),1,ierr) ! Dxz
      call mntr_check_abort(pstat_id,ierr,'Error writing Dxz interp.')
      call pstat3d_field_out(pstat_int_avg(1,44),1,ierr) ! Dyz
      call mntr_check_abort(pstat_id,ierr,'Error writing Dyz interp.')
      call pstat3d_field_out(pstat_int_new(1,22),1,ierr) ! Txx
      call mntr_check_abort(pstat_id,ierr,'Error writing Txx interp.')
      call pstat3d_field_out(pstat_int_new(1,23),1,ierr) ! Tyy
      call mntr_check_abort(pstat_id,ierr,'Error writing Tyy interp.')
      call pstat3d_field_out(pstat_int_new(1,24),1,ierr) ! Tzz
      call mntr_check_abort(pstat_id,ierr,'Error writing Tzz interp.')
      call pstat3d_field_out(pstat_int_new(1,25),1,ierr) ! Txy
      call mntr_check_abort(pstat_id,ierr,'Error writing Txy interp.')
      call pstat3d_field_out(pstat_int_new(1,27),1,ierr) ! Txz
      call mntr_check_abort(pstat_id,ierr,'Error writing Txz interp.')
      call pstat3d_field_out(pstat_int_new(1,26),1,ierr) ! Tyz
      call mntr_check_abort(pstat_id,ierr,'Error writing Tyz interp.')
      call pstat3d_field_out(pstat_int_new(1,16),1,ierr) ! VDxx
      call mntr_check_abort(pstat_id,ierr,'Error writing VDxx interp.')
      call pstat3d_field_out(pstat_int_new(1,17),1,ierr) ! VDyy
      call mntr_check_abort(pstat_id,ierr,'Error writing VDyy interp.')
      call pstat3d_field_out(pstat_int_new(1,18),1,ierr) ! VDzz
      call mntr_check_abort(pstat_id,ierr,'Error writing VDzz interp.')
      call pstat3d_field_out(pstat_int_new(1,19),1,ierr) ! VDxy
      call mntr_check_abort(pstat_id,ierr,'Error writing VDxy interp.')
      call pstat3d_field_out(pstat_int_new(1,21),1,ierr) ! VDxz
      call mntr_check_abort(pstat_id,ierr,'Error writing VDxz interp.')
      call pstat3d_field_out(pstat_int_new(1,20),1,ierr) ! VDyz
      call mntr_check_abort(pstat_id,ierr,'Error writing VDyz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,7),1,ierr) ! Pixx
      call mntr_check_abort(pstat_id,ierr,'Error writing Pixx interp.')
      call pstat3d_field_out(pstat_int_tmp(1,8),1,ierr) ! Piyy
      call mntr_check_abort(pstat_id,ierr,'Error writing Piyy interp.')
      call pstat3d_field_out(pstat_int_tmp(1,9),1,ierr) ! Pizz
      call mntr_check_abort(pstat_id,ierr,'Error writing Pizz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,10),1,ierr) ! Pixy
      call mntr_check_abort(pstat_id,ierr,'Error writing Pixy interp.')
      call pstat3d_field_out(pstat_int_tmp(1,11),1,ierr) ! Pixz
      call mntr_check_abort(pstat_id,ierr,'Error writing Pixz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,12),1,ierr) ! Piyz
      call mntr_check_abort(pstat_id,ierr,'Error writing Piyz interp.')
      call pstat3d_field_out(pstat_int_new(1,10),1,ierr) ! Cxx
      call mntr_check_abort(pstat_id,ierr,'Error writing Cxx interp.')
      call pstat3d_field_out(pstat_int_new(1,11),1,ierr) ! Cyy
      call mntr_check_abort(pstat_id,ierr,'Error writing Cyy interp.')
      call pstat3d_field_out(pstat_int_new(1,12),1,ierr) ! Czz
      call mntr_check_abort(pstat_id,ierr,'Error writing Czz interp.')
      call pstat3d_field_out(pstat_int_new(1,13),1,ierr) ! Cxy
      call mntr_check_abort(pstat_id,ierr,'Error writing Cxy interp.')
      call pstat3d_field_out(pstat_int_new(1,15),1,ierr) ! Cxz
      call mntr_check_abort(pstat_id,ierr,'Error writing Cxz interp.')
      call pstat3d_field_out(pstat_int_new(1,14),1,ierr) ! Cyz
      call mntr_check_abort(pstat_id,ierr,'Error writing Cyz interp.')
      call pstat3d_field_out(pstat_int_tmp(1,13),1,ierr) ! Pk
      call mntr_check_abort(pstat_id,ierr,'Error writing Pk interp.')
      call pstat3d_field_out(pstat_int_tmp(1,14),1,ierr) ! Dk
      call mntr_check_abort(pstat_id,ierr,'Error writing Dk interp.')
      call pstat3d_field_out(pstat_int_tmp(1,15),1,ierr) ! Tk
      call mntr_check_abort(pstat_id,ierr,'Error writing Tk interp.')
      call pstat3d_field_out(pstat_int_tmp(1,16),1,ierr) ! VDk
      call mntr_check_abort(pstat_id,ierr,'Error writing VDk interp.')
      call pstat3d_field_out(pstat_int_tmp(1,17),1,ierr) ! Pik
      call mntr_check_abort(pstat_id,ierr,'Error writing Pik interp.')
      call pstat3d_field_out(pstat_int_tmp(1,18),1,ierr) ! Ck
      call mntr_check_abort(pstat_id,ierr,'Error writing Ck interp.')
      call pstat3d_field_out(pstat_int_tmp(1,19),1,ierr) ! Resk
      call mntr_check_abort(pstat_id,ierr,'Error writing Resk interp.')
      call pstat3d_field_out(pstat_int_avg(1,12),1,ierr) ! PTxx
      call mntr_check_abort(pstat_id,ierr,'Error writing PTxx interp.')
      call pstat3d_field_out(pstat_int_avg(1,13),1,ierr) ! PTyy
      call mntr_check_abort(pstat_id,ierr,'Error writing PTyy interp.')
      call pstat3d_field_out(pstat_int_avg(1,14),1,ierr) ! PTzz
      call mntr_check_abort(pstat_id,ierr,'Error writing PTzz interp.')
      call pstat3d_field_out(pstat_int_avg(1,15),1,ierr) ! PTxy
      call mntr_check_abort(pstat_id,ierr,'Error writing PTxy interp.')
      call pstat3d_field_out(pstat_int_avg(1,16),1,ierr) ! PTxz
      call mntr_check_abort(pstat_id,ierr,'Error writing PTxz interp.')
      call pstat3d_field_out(pstat_int_avg(1,17),1,ierr) ! PTyz
      call mntr_check_abort(pstat_id,ierr,'Error writing PTyz interp.')
      call pstat3d_field_out(pstat_int_avg(1,18),1,ierr) ! PSxx
      call mntr_check_abort(pstat_id,ierr,'Error writing PSxx interp.')
      call pstat3d_field_out(pstat_int_avg(1,19),1,ierr) ! PSyy
      call mntr_check_abort(pstat_id,ierr,'Error writing PSyy interp.')
      call pstat3d_field_out(pstat_int_avg(1,20),1,ierr) ! PSzz
      call mntr_check_abort(pstat_id,ierr,'Error writing PSzz interp.')
      call pstat3d_field_out(pstat_int_avg(1,21),1,ierr) ! PSxy
      call mntr_check_abort(pstat_id,ierr,'Error writing PSxy interp.')
      call pstat3d_field_out(pstat_int_avg(1,22),1,ierr) ! PSxz
      call mntr_check_abort(pstat_id,ierr,'Error writing PSxz interp.')
      call pstat3d_field_out(pstat_int_avg(1,23),1,ierr) ! PSyz
      call mntr_check_abort(pstat_id,ierr,'Error writing PSyz interp.')
      call pstat3d_field_out(pstat_int_new(1,1),1,ierr) ! dUdx
      call mntr_check_abort(pstat_id,ierr,'Error writing dUdx interp.')
      call pstat3d_field_out(pstat_int_new(1,2),1,ierr) ! dUdy
      call mntr_check_abort(pstat_id,ierr,'Error writing dUdy interp.')
      call pstat3d_field_out(pstat_int_new(1,3),1,ierr) ! dUdz
      call mntr_check_abort(pstat_id,ierr,'Error writing dUdz interp.')
      call pstat3d_field_out(pstat_int_new(1,4),1,ierr) ! dVdx
      call mntr_check_abort(pstat_id,ierr,'Error writing dVdx interp.')
      call pstat3d_field_out(pstat_int_new(1,5),1,ierr) ! dVdy
      call mntr_check_abort(pstat_id,ierr,'Error writing dVdy interp.')
      call pstat3d_field_out(pstat_int_new(1,6),1,ierr) ! dVdz
      call mntr_check_abort(pstat_id,ierr,'Error writing dVdz interp.')
      call pstat3d_field_out(pstat_int_new(1,7),1,ierr) ! dWdx
      call mntr_check_abort(pstat_id,ierr,'Error writing dWdx interp.')
      call pstat3d_field_out(pstat_int_new(1,8),1,ierr) ! dWdy
      call mntr_check_abort(pstat_id,ierr,'Error writing dWdy interp.')
      call pstat3d_field_out(pstat_int_new(1,9),1,ierr) ! dWdz
      call mntr_check_abort(pstat_id,ierr,'Error writing dWdz interp.')

      ! master closes the file
      if (nid.eq.pid00) then
         call byte_close(ierr)
      endif

      call mntr_check_abort(pstat_id,ierr,
     $     'Error closing interpolation file in pstat_mfo_interp.')


      return
      end subroutine
!=======================================================================
!> @brief Geather single field data and write it down
!! @ingroup pstat3d
!! @param[in]   int_field     interpolated field
!! @param[in]   fldim         field dimension
!! @param[out]  ierr          error flag
      subroutine pstat3d_field_out(int_field,fldim,ierr)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'PSTAT3D'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      !argument list
      integer fldim, ierr
      real int_field(fldim*lhis)

      ! local variables
      integer jl, kl   ! loop index
      integer npass        ! number of messages to send for single field
      integer npts         ! number of points for transfer
      integer itmp         ! temporary variables
      real rtmpv(lhis*ldim), rtmpv1(lhis*ldim), rtmp
      real*4 rtmpv2(2*lhis*ldim)
      equivalence (rtmpv1,rtmpv2)

      integer wdsl              ! double size
!-----------------------------------------------------------------------
      ierr = 0
      wdsl = wdsize/4
      
      if (nid.eq.0) then
         ! first master writes its own data
         if (wdsl.eq.2) then
            call copy(rtmpv1,int_field,pstat_npt*fldim)
            call byte_write(rtmpv2,pstat_npt*fldim*wdsl,ierr)
         else
            call copyX4(rtmpv2,int_field,pstat_npt*fldim)
            call byte_write(rtmpv2,pstat_npt*fldim,ierr)
         endif

         ! geather data from slaves
         npass = min(mp,pstat_nptot)
         do jl = 1,npass-1
            npts = pstat_npt
            if (pstat_npt1.gt.0.and.jl.ge.pstat_npt1) then
               npts = pstat_npt -1
            endif
            call csend(jl,itmp,isize,jl,kl) ! hand shaiking
            call crecv2(jl,rtmpv,npts*fldim*wdsize,jl)

            ! write data
            if (wdsl.eq.2) then
               call copy(rtmpv1,rtmpv,npts*fldim)
               call byte_write(rtmpv2,npts*fldim*wdsl,ierr)
            else
               call copyX4(rtmpv2,rtmpv,npts*fldim)
               call byte_write(rtmpv2,npts*fldim,ierr)
            endif
         enddo
      else
         ! slaves send their data
         if (pstat_npt.gt.0) then
            call crecv2(nid,itmp,isize,0) ! hand shaiking
            call csend(nid,int_field,pstat_npt*fldim*wdsize,0,itmp)
         endif
      endif

      return
      end subroutine
!=======================================================================
