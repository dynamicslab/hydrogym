!> @file io_tools.f
!! @ingroup io_tools
!! @brief Set of I/O related tools for KTH modules
!! @author Adam Peplinski
!! @date Mar 7, 2016
!=======================================================================
!> @brief Register io tool module
!! @ingroup io_tools
!! @note This routine should be called in frame_usr_register
      subroutine io_register()
      implicit none

      include 'FRAMELP'
      include 'IOTOOLD'

      ! local variables
      integer lpmid
!-----------------------------------------------------------------------
      ! check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,io_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(io_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'Parent module ['//'FRAME'//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(io_id,lpmid,io_name,'I/O TOOLS')

      return
      end subroutine
!=======================================================================
!> @brief Get free file unit number and store max unit value
!! @ingroup io_tools
!! @param[out] iunit     file unit
!! @param[out] ierr      error mark
!! @see io_file_close
      subroutine io_file_freeid(iunit, ierr)
      implicit none

      include 'FRAMELP'
      include 'IOTOOLD'

      ! argument list
      integer iunit
      integer ierr

      ! local variables
      logical ifcnnd            ! is unit connected
!-----------------------------------------------------------------------
      ! initialise variables
      ierr=0
      iunit = io_iunit_min

      do
         inquire(unit=iunit,opened=ifcnnd,iostat=ierr)
         if(ifcnnd) then
            iunit = iunit +1
         else
            exit
         endif
      enddo

      if (iunit.gt.io_iunit_max) io_iunit_max = iunit

      return
      end subroutine
!=======================================================================
!> @brief Close all opened files up to sotred max unit numer
!! @ingroup io_tools
!! @see io_file_freeid
      subroutine io_file_close()
      implicit none

      include 'FRAMELP'
      include 'IOTOOLD'

      ! local variables
      integer iunit, ierr
      logical ifcnnd            ! is unit connected
!-----------------------------------------------------------------------
      do iunit = io_iunit_min, io_iunit_max
         inquire(unit=iunit,opened=ifcnnd,iostat=ierr)
         if(ifcnnd) close(iunit)
      enddo
      io_iunit_max = io_iunit_min

      return
      end subroutine
!=======================================================================
!> @brief Generate file name according to nek rulles without opening the file
!! @details It is a modified version of @ref mfo_open_files from prepost.f but
!! without equivalence and file opening part. I split file name generation
!! and file opening as different tools can require this.
!! @ingroup io_tools
!! @param[out]  fname     file name
!! @param[in]   bname     base name
!! @param[in]   prefix    prefix
!! @param[out]  ierr      error mark
      subroutine io_mfo_fname(fname,bname,prefix,ierr)
      implicit none

      include 'SIZE'
      include 'INPUT'           ! IFREGUO, IFMPIIO
      include 'RESTART'         ! NFILEO
      include 'FRAMELP'
      include 'IOTOOLD'


      ! argument list
      character*132  fname, bname
      character*3 prefix
      integer ierr

      ! local variables
      integer ndigit, itmp
      real rfileo

      character*(*)  six
      parameter(six='??????')
!-----------------------------------------------------------------------
      ! initialise variables
      ierr = 0
      fname = ''

      ! numbe or IO nodes
      if (IFMPIIO) then
        rfileo = 1
      else
        rfileo = NFILEO
      endif
      ndigit = log10(rfileo) + 1

      ! Add directory
      if (ifdiro) fname = 'A'//six(1:ndigit)//'/'

      ! Add prefix
      if (prefix(1:1).ne.' '.and.prefix(2:2).ne.' '
     $    .and.prefix(3:3).ne.' ')
     $     fname = trim(fname)//trim(adjustl(prefix))

      ! Add SESSION
      fname = trim(fname)//trim(adjustl(bname))

      if (IFREGUO) fname = trim(fname)//'_reg'

      ! test string length
      itmp = len_trim(fname)
      if (itmp.eq.0) then
         call mntr_error(io_id,'io_mfo_fname; zero lenght fname.')
         ierr = 1
         return
      elseif ((itmp+ndigit+2+5).gt.132) then
         call mntr_error(io_id,'io_mfo_fname; fname too long.')
         ierr = 2
         return
      endif

      ! Add file-id holder and .f appendix
      fname = trim(fname)//six(1:ndigit)//'.f'

      return
      end subroutine
!=======================================================================
!> @brief Open field file
!! @details This routine opens the file (serial or parallel depending on
!!    parmeter set by @ref io_init) using Nek5000 C routines. I need it
!!    for a number of tools writing restart files that do not directly
!!    stick to the numbering scheme used in @ref mfo_open_files.
!! @ingroup io_tools
!! @param[in]   hname      file name
!! @param[out]  ierr       error mark
      subroutine io_mbyte_open(hname,ierr)
      implicit none

      include 'SIZE'            ! NID
      include 'INPUT'           ! ifmpiio
      include 'RESTART'         ! fid0, pid0, ifh_mbyte
      include 'FRAMELP'
      include 'IOTOOLD'

      ! argumnt list
      integer fid, ierr
      character*132 hname

      ! local variables
      character*132 fname
      integer itmp
!-----------------------------------------------------------------------
      ! initialise variables
      ierr = 0
      ! work on local copy
      fname = trim(adjustl(hname))

      ! test string length
      itmp = len_trim(fname)
      if (itmp.eq.0) then
         call mntr_error(io_id,'io_mbyte_open; zero lenght fname.')
         ierr = 1
         return
      endif

      ! add file number
      call addfid(fname,fid0)

      call mntr_log(io_id,lp_ess,'Opening file: '//trim(fname))
      if(ifmpiio) then
        call byte_open_mpi(fname,ifh_mbyte,.false.,ierr)
      else
        ! add ending character; required by C
        fname = trim(fname)//CHAR(0)
        call byte_open(fname,ierr)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Close field file
!! @details This routine closes the file (serial or parallel depending on
!!    parmeter set by @ref io_init) using Nek5000 C routines.
!! @ingroup io_tools
!! @param[out]  ierr       error mark
      subroutine io_mbyte_close(ierr)
      implicit none

      include 'SIZE'            ! NID
      include 'INPUT'           ! ifmpiio
      include 'RESTART'         ! pid0, ifh_mbyte

      ! argumnt list
      integer ierr

      ! local variables
      character*132 fname
      integer itmp
!-----------------------------------------------------------------------
      ! initialise variables
      ierr = 0

      ! close the file
      if (nid.eq.pid0) then
         if(ifmpiio) then
           call byte_close_mpi(ifh_mbyte,ierr)
         else
           call byte_close(ierr)
         endif
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write single vector to the file
!! @details This routine is based on @ref mfo_outfld but can be used for
!!    writing 2D sections of 3D simulation.
!! @ingroup io_tools
!! @param[inout] offs               offset of global vector beginning
!! @param[in]    lvx,lvy,lvz        vector to write
!! @param[in]    lnx,lny,lnz        element dimensions
!! @param[in]    lnel               local number of filed elements
!! @param[in]    lnelg              global number of filed elements
!! @param[in]    lndim              written domain dimension
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine io_mfov(offs,lvx,lvy,lvz,lnx,lny,lnz,
     $           lnel,lnelg,lndim)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'FRAMELP'
      include 'IOTOOLD'

      ! argumnt list
      integer*8 offs
      integer lnx,lny,lnz,lnel,lnelg,lndim
      real lvx(lnx,lny,lnz,lnel), lvy(lnx,lny,lnz,lnel),
     $     lvz(lnx,lny,lnz,lnel)

      ! local variables
      integer*8 loffs
      integer il, ik, itmp

      real rvx(lxo*lxo*(1 + (ldim-2)*(lxo-1))*lelt),
     $     rvy(lxo*lxo*(1 + (ldim-2)*(lxo-1))*lelt),
     $     rvz(lxo*lxo*(1 + (ldim-2)*(lxo-1))*lelt)
      common /SCRUZ/ rvx, rvy, rvz
!-----------------------------------------------------------------------
      if (ifreguo) then
         ! check size of mapping space
         if (nrg.gt.lxo) then
            call mntr_warn(io_id,
     $          'io_mfov; nrg too large, reset to lxo!')
            nrg = lxo
         endif

         ! map to regular mesh
         ! this code works with square element only
         itmp = nrg**lndim
         if (lndim.eq.2) then
            ik=1
            do il=1,lnel
               call map2reg_2di_e(rvx(ik),nrg,lvx(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
            ik=1
            do il=1,lnel
               call map2reg_2di_e(rvy(ik),nrg,lvy(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
         else
            ik = 1
            do il=1,lnel
               call map2reg_3di_e(rvx(ik),nrg,lvx(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
            ik = 1
            do il=1,lnel
               call map2reg_3di_e(rvy(ik),nrg,lvy(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
            ik = 1
            do il=1,lnel
               call map2reg_3di_e(rvz(ik),nrg,lvz(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
         endif

         ! shift offset taking onto account elements on processes with smaller id
         itmp = 1 + (lndim-2)*(nrg-1)
         ! to ensure proper integer prolongation
         loffs = offs + int(nelB,8)*int(lndim*wdsizo*nrg*nrg*itmp,8)
         call byte_set_view(loffs,ifh_mbyte)

         ! write vector
         call mfo_outv(rvx,rvy,rvz,lnel,nrg,nrg,itmp)

         ! update offset
         offs = offs + int(lnelg,8)*int(lndim*wdsizo*nrg*nrg*itmp,8)
      else
         ! shift offset taking onto account elements on processes with smaller id
         ! to ensure proper integer prolongation
         loffs = offs + int(nelB,8)*int(lndim*wdsizo*lnx*lny*lnz,8)
         call byte_set_view(loffs,ifh_mbyte)

         ! write vector
         call mfo_outv(lvx,lvy,lvz,lnel,lnx,lny,lnz)

         ! update offset
         offs = offs + int(lnelg,8)*int(lndim*wdsizo*lnx*lny*lnz,8)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Write single scalar to the file
!! @details This routine is based on @ref mfo_outfld but can be used for
!!    writing 2D sections of 3D simulation.
!! @ingroup io_tools
!! @param[inout] offs               offset of global vector beginning
!! @param[in]    lvs                scalar to write
!! @param[in]    lnx,lny,lnz        element dimensions
!! @param[in]    lnel               local number of field elements
!! @param[in]    lnelg              global number of filed elements
!! @param[in]    lndim              written domain dimension
!! @remark This routine uses global scratch space \a SCRUZ.
      subroutine io_mfos(offs,lvs,lnx,lny,lnz,lnel,lnelg,lndim)
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'FRAMELP'
      include 'IOTOOLD'

      ! argumnt list
      integer*8 offs
      integer lnx,lny,lnz,lnel,lnelg,lndim
      real lvs(lnx,lny,lnz,lnel)

      ! local variables
      integer*8 loffs
      integer il, ik, itmp

      real rvs(lxo*lxo*(1 + (ldim-2)*(lxo-1))*lelt)
      common /SCRUZ/ rvs
!-----------------------------------------------------------------------
      if (ifreguo) then
         ! check size of mapping space
         if (nrg.gt.lxo) then
            call mntr_warn(io_id,
     $          'io_mfos; nrg too large, reset to lxo!')
            nrg = lxo
         endif

         ! map to regular mesh
         ! this code works with square element only
         itmp = nrg**lndim
         if (lndim.eq.2) then
            ik=1
            do il=1,lnel
               call map2reg_2di_e(rvs(ik),nrg,lvs(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
         else
            ik = 1
            do il=1,lnel
               call map2reg_3di_e(rvs(ik),nrg,lvs(1,1,1,il),lnx)
               ik = ik + itmp
            enddo
         endif

         ! shift offset taking onto account elements on processes with smaller id
         itmp = 1 + (lndim-2)*(nrg-1)
         ! to ensure proper integer prolongation
         loffs = offs + int(nelB,8)*int(wdsizo*nrg*nrg*itmp,8)
         call byte_set_view(loffs,ifh_mbyte)

         ! write vector
         call mfo_outs(rvs,lnel,nrg,nrg,itmp)

         ! update offset
         offs = offs + int(lnelg,8)*int(wdsizo*nrg*nrg*itmp,8)
      else
         ! shift offset taking onto account elements on processes with smaller id
         ! to ensure proper integer prolongation
         loffs = offs + int(nelB,8)*int(wdsizo*lnx*lny*lnz,8)
         call byte_set_view(loffs,ifh_mbyte)

         ! write vector
         call mfo_outs(lvs,lnel,lnx,lny,lnz)

         ! update offset
         offs = offs + int(lnelg,8)*int(wdsizo*lnx*lny*lnz,8)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Read vector filed from the file
!! @details This is version of @ref mfi_getv that does not perform
!!    interpolation and allows to specify element size.
!! @param[inout] offs         offset of global vector beginning
!! @param[out]   uf, vf, wf   vector field compinents
!! @param[in]    lnx,lny,lnz  element size
!! @param[in]    lnel         number of elements
!! @param[in]    ifskip       reading flag (for non-mpi formats)
!! @remarks This routine uses global scratch space \a VRTHOV and \a SCRNS
      subroutine io_mfiv(offs,uf,vf,wf,lnx,lny,lnz,lnel,ifskip)
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'FRAMELP'
      include 'IOTOOLD'

      ! argument list
      integer*8 offs
      integer lnx,lny,lnz,lnel
      real uf(lnx*lny*lnz,lnel),vf(lnx*lny*lnz,lnel),
     $     wf(lnx*lny*lnz,lnel)
      logical ifskip

      ! local variables
      integer lndim, nxyzr, nxyzw, nxyzv, mlen
      integer num_recv, num_avail, nread, nelrr
      integer el, il, kl, ll, ierr
      integer ei, eg, jnid, jeln
      integer msg_id(lelt)
      integer*8 i8tmp

      ! read buffer
      integer lrbs
      parameter(lrbs=20*lx1*ly1*lz1*lelt)
      real*4 w2(lrbs)
      common /vrthov/ w2

      ! communication buffer
      integer lwk
      parameter (lwk = 14*lx1*ly1*lz1*lelt)
      real*4 wk(lwk)
      common /SCRNS/ wk

      ! functions
      integer irecv, iglmax
!-----------------------------------------------------------------------
      call nekgsync() ! clear outstanding message queues.

      ! check element size
      if ((nxr.ne.lnx).or.(nyr.ne.lny).or.(nzr.ne.lnz)) then
         call mntr_abort(io_id,'io_mfiv, wrong element size')
      endif

      if (lnz.gt.1) then
         lndim = 3
      else
         lndim = 2
      endif

      nxyzr  = lndim*lnx*lny*lnz   ! element size
      mlen   = nxyzr*wdsizr  ! message length
      if (wdsizr.eq.8) then
         nxyzw = 2*nxyzr
      else
         nxyzw = nxyzr
      endif

      ! shift offset
      i8tmp = offs + int(nelBr,8)*int(mlen,8)
      call byte_set_view(i8tmp,ifh_mbyte)

      ! check message buffer
      num_recv  = mlen
      num_avail = lwk*4
      call lim_chk(num_recv,num_avail,'     ','     ','io_mfiv a')

      ! setup read buffer
      if (nid.eq.pid0r) then
         i8tmp = int(nxyzw,8)*int(nelr,8)
         nread = i8tmp/int(lrbs,8)
         if (mod(i8tmp,int(lrbs,8)).ne.0) nread = nread + 1
         if(ifmpiio) nread = iglmax(nread,1) ! needed because of collective read
         nelrr = nelr/nread
      endif
      call bcast(nelrr,4)
      call lim_chk(nxyzw*nelrr,lrbs,'     ','     ','io_mfiv b')

      ! reset error flag
      ierr = 0

      if (ifskip) then ! just read deata and do not use it (avoid communication)

         if (nid.eq.pid0r) then ! only i/o nodes will read
            ! read blocks of size nelrr
            kl = 0
            do il = 1,nread
               if (il.eq.nread) then ! clean-up
                  nelrr = nelr - (nread-1)*nelrr
                  if (nelrr.lt.0) nelrr = 0
               endif

               if (ierr.eq.0) then
                  if (ifmpiio) then
                    call byte_read_mpi(w2,nxyzw*nelrr,-1,ifh_mbyte,ierr)
                  else
                    call byte_read (w2,nxyzw*nelrr,ierr)
                  endif
               endif
            enddo
         endif

         call nekgsync()

      else   ! read and use data
         ! pre-post recieves
         if (np.gt.1) then
            ll = 1
            do el=1,nelt
               msg_id(el) = irecv(el,wk(ll),mlen)
               ll = ll+nxyzw
            enddo
         endif

         if (nid.eq.pid0r.and.np.gt.1) then ! only i/o nodes will read
            ! read blocks of size nelrr
            kl = 0
            do il = 1,nread
               if (il.eq.nread) then ! clean-up
                  nelrr = nelr - (nread-1)*nelrr
                  if (nelrr.lt.0) nelrr = 0
               endif

               if (ierr.eq.0) then
                  if (ifmpiio) then
                    call byte_read_mpi(w2,nxyzw*nelrr,-1,ifh_mbyte,ierr)
                  else
                    call byte_read (w2,nxyzw*nelrr,ierr)
                  endif
               endif

               ! distribute data across target processors
               ll = 1
               do el = kl+1,kl+nelrr
                  jnid = gllnid(er(el))                ! where is er(e) now?
                  jeln = gllel(er(el))
                  if(ierr.ne.0) call rzero(w2(ll),mlen)
                  call csend(jeln,w2(ll),mlen,jnid,0)  ! blocking send
                  ll = ll+nxyzw
               enddo
               kl = kl + nelrr
            enddo
         elseif (np.eq.1) then
            if (ifmpiio) then
               call byte_read_mpi(wk,nxyzw*nelr,-1,ifh_mbyte,ierr)
            else
               call byte_read(wk,nxyzw*nelr,ierr)
            endif
         endif

         ! distinguish between vector lengths
         nxyzv = lnx*lny*lnz
         if (wdsizr.eq.8) then
            nxyzw = 2*nxyzv
         else
            nxyzw = nxyzv
         endif

         ll = 1
         do el=1,nelt
            if (np.gt.1) then
               call msgwait(msg_id(el))
               ei = el
            elseif(np.eq.1) then
              ei = er(el)
            endif
            if (if_byte_sw) then
               if (wdsizr.eq.8) then
                  call byte_reverse8(wk(ll),nxyzr*2,ierr)
               else
                  call byte_reverse(wk(ll),nxyzr,ierr)
               endif
            endif
            ! copy data
            if (wdsizr.eq.4) then
               call copy4r(uf(1,ei),wk(ll),nxyzv)
               call copy4r(vf(1,ei),wk(ll + nxyzw),nxyzv)
               if (lndim.eq.3)
     $             call copy4r(wf(1,ei),wk(ll + 2*nxyzw),nxyzv)
            else
               call copy(uf(1,ei),wk(ll),nxyzv)
               call copy(vf(1,ei),wk(ll + nxyzw),nxyzv)
               if (lndim.eq.3)
     $             call copy(wf(1,ei),wk(ll + 2*nxyzw),nxyzv)
            endif
            ll = ll+ldim*nxyzw
         enddo

      endif

      ! update offset
      offs = offs + int(nelgr,8)*int(mlen,8)

      call err_chk(ierr,'Error reading restart data,in io_mfiv.$')
      return
      end subroutine
!=======================================================================
!> @brief Read scalar filed from the file
!! @details This is version of @ref mfi_gets that does not perform
!!    interpolation and allows to specify element size.
!! @ingroup io_tools
!! @param[inout] offs         offset of global vector beginning
!! @param[out]   uf           scalar field
!! @param[in]    lnx,lny,lnz  element size
!! @param[in]    lnel         number of elements
!! @param[in]    ifskip       reading flag (for non-mpi formats)
!! @remarks This routine uses global scratch space \a VRTHOV and \a SCRNS
      subroutine io_mfis(offs,uf,lnx,lny,lnz,lnel,ifskip)
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'FRAMELP'
      include 'IOTOOLD'

      ! argument list
      integer*8 offs
      integer lnx,lny,lnz,lnel
      real uf(lnx*lny*lnz,lnel)
      logical ifskip

      ! local variables
      integer nxyzr, nxyzw, mlen
      integer num_recv, num_avail, nread, nelrr
      integer el, il, kl, ll, ierr
      integer ei, eg, jnid, jeln
      integer msg_id(lelt)
      integer*8 i8tmp

      ! read buffer
      integer lrbs
      parameter(lrbs=20*lx1*ly1*lz1*lelt)
      real*4 w2(lrbs)
      common /vrthov/ w2

      ! communication buffer
      integer lwk
      parameter (lwk = 14*lx1*ly1*lz1*lelt)
      real*4 wk(lwk)
      common /SCRNS/ wk

      ! functions
      integer irecv, iglmax
!-----------------------------------------------------------------------
      call nekgsync() ! clear outstanding message queues.

      ! check element size
      if ((nxr.ne.lnx).or.(nyr.ne.lny).or.(nzr.ne.lnz)) then
         call mntr_abort(io_id,'io_mfis, wrong element size')
      endif

      nxyzr  = lnx*lny*lnz   ! element size
      mlen   = nxyzr*wdsizr  ! message length
      if (wdsizr.eq.8) then
         nxyzw = 2*nxyzr
      else
         nxyzw = nxyzr
      endif

      ! shift offset
      i8tmp = offs + int(nelBr,8)*int(mlen,8)
      call byte_set_view(i8tmp,ifh_mbyte)

      ! check message buffer
      num_recv  = mlen
      num_avail = lwk*4
      call lim_chk(num_recv,num_avail,'     ','     ','io_mfis a')

      ! setup read buffer
      if (nid.eq.pid0r) then
         i8tmp = int(nxyzw,8)*int(nelr,8)
         nread = i8tmp/int(lrbs,8)
         if (mod(i8tmp,int(lrbs,8)).ne.0) nread = nread + 1
         if(ifmpiio) nread = iglmax(nread,1) ! needed because of collective read
         nelrr = nelr/nread
      endif
      call bcast(nelrr,4)
      call lim_chk(nxyzw*nelrr,lrbs,'     ','     ','io_mfis b')

      ! reset error flag
      ierr = 0

      if (ifskip) then ! just read deata and do not use it (avoid communication)

         if (nid.eq.pid0r) then ! only i/o nodes will read
            ! read blocks of size nelrr
            kl = 0
            do il = 1,nread
               if (il.eq.nread) then ! clean-up
                  nelrr = nelr - (nread-1)*nelrr
                  if (nelrr.lt.0) nelrr = 0
               endif

               if (ierr.eq.0) then
                  if (ifmpiio) then
                    call byte_read_mpi(w2,nxyzw*nelrr,-1,ifh_mbyte,ierr)
                  else
                    call byte_read (w2,nxyzw*nelrr,ierr)
                  endif
               endif
            enddo
         endif

         call nekgsync()

      else   ! read and use data
         ! pre-post recieves
         if (np.gt.1) then
            ll = 1
            do el=1,nelt
               msg_id(el) = irecv(el,wk(ll),mlen)
               ll = ll+nxyzw
            enddo
         endif

         if (nid.eq.pid0r.and.np.gt.1) then ! only i/o nodes will read
            ! read blocks of size nelrr
            kl = 0
            do il = 1,nread
               if (il.eq.nread) then ! clean-up
                  nelrr = nelr - (nread-1)*nelrr
                  if (nelrr.lt.0) nelrr = 0
               endif

               if (ierr.eq.0) then
                  if (ifmpiio) then
                    call byte_read_mpi(w2,nxyzw*nelrr,-1,ifh_mbyte,ierr)
                  else
                    call byte_read (w2,nxyzw*nelrr,ierr)
                  endif
               endif

               ! distribute data across target processors
               ll = 1
               do el = kl+1,kl+nelrr
                  jnid = gllnid(er(el))                ! where is er(e) now?
                  jeln = gllel(er(el))
                  if(ierr.ne.0) call rzero(w2(ll),mlen)
                  call csend(jeln,w2(ll),mlen,jnid,0)  ! blocking send
                  ll = ll+nxyzw
               enddo
               kl = kl + nelrr
            enddo
         elseif (np.eq.1) then
            if (ifmpiio) then
               call byte_read_mpi(wk,nxyzw*nelr,-1,ifh_mbyte,ierr)
            else
               call byte_read(wk,nxyzw*nelr,ierr)
            endif
         endif

         ll = 1
         do el=1,nelt
            if (np.gt.1) then
               call msgwait(msg_id(el))
               ei = el
            elseif(np.eq.1) then
               ei = er(el)
            endif
            if (if_byte_sw) then
               if (wdsizr.eq.8) then
                  call byte_reverse8(wk(ll),nxyzw,ierr)
               else
                  call byte_reverse(wk(ll),nxyzw,ierr)
               endif
            endif
            ! copy data
            if (wdsizr.eq.4) then
               call copy4r(uf(1,ei),wk(ll),nxyzr)
            else
               call copy  (uf(1,ei),wk(ll),nxyzr)
            endif
            ll = ll+nxyzw
         enddo

      endif

      ! update offset
      offs = offs + int(nelgr,8)*int(mlen,8)

      call err_chk(ierr,'Error reading restart data,in io_mfis.$')
      return
      end subroutine
!=======================================================================
