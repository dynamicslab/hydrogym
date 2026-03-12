!> @file pstat2D.f
!! @ingroup pstat2d
!! @brief Post processing for statistics module
!! @author Adam Peplinski
!! @date Mar 13, 2019
!=======================================================================
!> @brief Register post processing statistics module
!! @ingroup pstat2d
!! @note This routine should be called in frame_usr_register
      subroutine pstat2d_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer lpmid, il
      real ltim
      character*2 str

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! timing
      ltim = dnekclock()

      !     check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,pstat_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(pstat_name)//'] already registered')
         return
      endif

      ! find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'parent module ['//'FRAME'//'] not registered')
      endif

      ! register module
      call mntr_mod_reg(pstat_id,lpmid,pstat_name,
     $      'Post processing for statistics')

      ! register timers
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      ! total time
      call mntr_tmr_reg(pstat_tmr_tot_id,lpmid,pstat_id,
     $     'PSTAT_TOT','Pstat total time',.false.)
      lpmid = pstat_tmr_tot_id
      ! initialisation
      call mntr_tmr_reg(pstat_tmr_ini_id,lpmid,pstat_id,
     $     'PSTAT_INI','Pstat initialisation time',.true.)
      ! averaging
      call mntr_tmr_reg(pstat_tmr_avg_id,lpmid,pstat_id,
     $     'PSTAT_AVG','Pstat averaging time',.true.)
      ! derivative calculation
      call mntr_tmr_reg(pstat_tmr_der_id,lpmid,pstat_id,
     $     'PSTAT_DER','Pstat derivative calculation time',.true.)
      ! interpolation
      call mntr_tmr_reg(pstat_tmr_int_id,lpmid,pstat_id,
     $     'PSTAT_INT','Pstat interpolation time',.true.)

      ! register and set active section
      call rprm_sec_reg(pstat_sec_id,pstat_id,'_'//adjustl(pstat_name),
     $     'Runtime paramere section for pstat module')
      call rprm_sec_set_act(.true.,pstat_sec_id)

      ! register parameters
      call rprm_rp_reg(pstat_crd_fnr_id,pstat_sec_id,'C2D_FNUM',
     $     'c2D file number',rpar_int,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_amr_irnr_id,pstat_sec_id,'AMR_NREF',
     $ 'Nr. of initial refinemnt (AMR only)',rpar_int,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_nfile_id,pstat_sec_id,'STS_NFILE',
     $ 'Number of stat files',rpar_int,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_stime_id,pstat_sec_id,'STS_STIME',
     $ 'Statistics starting time',rpar_real,1,0.0,.false.,' ')
      call rprm_rp_reg(pstat_nstep_id,pstat_sec_id,'STS_NSTEP',
     $ 'Number of steps between averaging (in sts file)',
     $  rpar_int,10,0.0,.false.,' ')

      ! set initialisation flag
      pstat_ifinit=.false.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_tot_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise pstat module
!! @ingroup pstat2d
!! @note This routine should be called in frame_usr_init
      subroutine pstat2d_init()
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer itmp, il
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
!     check if the module was already initialised
      if (pstat_ifinit) then
         call mntr_warn(pstat_id,
     $        'module ['//trim(pstat_name)//'] already initiaised.')
         return
      endif

      ! timing
      ltim = dnekclock()

      ! get runtime parameters
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_crd_fnr_id,rpar_int)
      pstat_crd_fnr = abs(itmp)
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_amr_irnr_id,rpar_int)
      pstat_amr_irnr = abs(itmp)
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_nfile_id,rpar_int)
      pstat_nfile = abs(itmp)
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_stime_id,rpar_real)
      pstat_stime = rtmp
      call rprm_rp_get(itmp,rtmp,ltmp,ctmp,pstat_nstep_id,rpar_int)
      pstat_nstep = abs(itmp)

      ! set field swapping array
      do il = 1,26
         pstat_swfield(il) = il
      enddo
      do il = 27,32
         pstat_swfield(il) = il+1
      enddo
      pstat_swfield(33) = 27
      pstat_swfield(34) = 38
      do il = 35,38
         pstat_swfield(il) = il-1
      enddo
      do il = 39,44
         pstat_swfield(il) = il
      enddo

      ! everything is initialised
      pstat_ifinit=.true.

      ! timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_ini_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Check if module was initialised
!! @ingroup pstat2d
!! @return pstat2d_is_initialised
      logical function pstat2d_is_initialised()
      implicit none

      include 'SIZE'
      include 'PSTAT2D'
!-----------------------------------------------------------------------
      pstat2d_is_initialised = pstat_ifinit

      return
      end function
!=======================================================================
!> @brief Main interface of pstat module
!! @ingroup pstat2d
      subroutine pstat2d_main
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer il

      ! simple timing
      real ltim

      ! functions
      real dnekclock
!-----------------------------------------------------------------------
      ! read element centre data
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Updating 2D mesh structure')
      call pstat2d_mfi_crd2D
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_ini_id,1,ltim)

      ! perform element ordering to get 2D mesh corresponing to 3D projected one
      ! in case of AMR this performs refinemnt of 2D mesh as well
      call pstat2d_mesh_manipulate

      ! read and average fields
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Field averaging')
      call pstat2d_sts_avg
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_avg_id,1,ltim)

      ! calculate derivatives
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Derivative calculation')
      call pstat2d_deriv
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_der_id,1,ltim)

      ! interpolate into the set of points
      ltim = dnekclock()
      call mntr_log(pstat_id,lp_inf,'Point interpolation')
      call pstat2d_interp
      ltim = dnekclock() - ltim
      call mntr_tmr_add(pstat_tmr_int_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Manipulate mesh to find proper element ordering (in case of AMR refine)
!! @ingroup pstat2d
      subroutine pstat2d_mesh_manipulate
      implicit none

      include 'SIZE'
      include 'GEOM'
      include 'PARALLEL'
      include 'INPUT'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer ierr
      integer il, jl, kl, inf_cnt
      integer ifpts       ! findpts flag
      real tol            ! interpolation tolerance
      integer nt
      integer npt_max     ! max communication set
      integer nxf, nyf, nzf  ! fine mesh for bb-test
      real bb_t           ! relative size to expand bounding boxes by
      real toldist
      parameter (toldist = 5e-6)

      integer rcode(lelt), proc(lelt),elid(lelt)
      real dist(lelt), rst(ldim*lelt)

      integer nfail, npass

      ! for sorting
      integer idim, isdim
      parameter (idim=4, isdim = 4*lelt)
      integer isort(idim,isdim), ind(isdim), iwork(idim)
      integer ipass, iseg, nseg, gnseg
      integer ninseg(lelt)    ! elements in segment
      logical ifseg(lelt)     ! segment borders

      integer ibuf(2)

      ! functions
      integer iglsum

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
#ifdef AMR
      ! reset amr level falg
      do il = 1, lelt
         pstat_refl(il) = 0
      enddo

      ! initial refinement
      do il=1,pstat_amr_irnr
         call amr_refinement()
      enddo
#endif

      ! find the mapping of the 3D projected elements on the 2D mesh and refine
      tol     = 5e-13
      nt       = lx1*ly1*lz1*lelt
      npt_max = 256
      nxf     = 2*lx1
      nyf     = 2*ly1
      nzf     = 2*lz1
      bb_t    = 0.01

      ! infinit loop
      inf_cnt = 0
      do
         ! start interpolation tool on given mesh
         call fgslib_findpts_setup(ifpts,nekcomm,mp,ldim,
     &                            xm1,ym1,zm1,lx1,ly1,lz1,
     &                            nelt,nxf,nyf,nzf,bb_t,nt,nt,
     &                            npt_max,tol)
         ! identify elements
         call fgslib_findpts(ifpts,rcode,1,
     &                      proc,1,
     &                      elid,1,
     &                      rst,ldim,
     &                      dist,1,
     &                      pstat_cnt(1,1),ldim,
     &                      pstat_cnt(2,1),ldim,
     &                      pstat_cnt(ldim,1),ldim,pstat_nel)
         ! close interpolation tool
         call fgslib_findpts_free(ifpts)

         ! find problems with interpolation
         nfail = 0
         do il = 1,pstat_nel
            ! check return code
            if (rcode(il).eq.1) then
               if (sqrt(dist(il)).gt.toldist) nfail = nfail + 1
            elseif(rcode(il).eq.2) then
               nfail = nfail + 1
            endif
         enddo
         nfail = iglsum(nfail,1)

#ifdef DEBUG
         ! for testing
         ! to output refinement
         icalld = icalld+1
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='CRDfpts.txt'//str1//'i'//str2)

         write(iunit,*) pstat_nelg, pstat_nel, nfail
         do il=1,pstat_nel
            write(iunit,*) il, pstat_gnel(il), proc(il), elid(il)+1,
     &       rcode(il), dist(il), (rst(jl+(il-1)*ldim),jl=1,ldim)
         enddo

         close(iunit)
#endif

         if (nfail.gt.0) call mntr_abort(pstat_id,
     $     'Elements not identified in pstat_mesh_manipulate')

         ! sort data
         ! pack it first
         do il=1,pstat_nel
            isort(1,il) = proc(il)
            isort(2,il) = elid(il) + 1 ! go from c to fortran count
            isort(3,il) = pstat_lev(il)
            isort(4,il) = pstat_gnel(il)
         enddo

         !tupple sort
         ! mark no section boundaries
         do il=1,pstat_nel
            ifseg(il) = .FALSE.
         enddo
         ! perform local sorting to identify unique set sorting by directions
         ! first run => whole set is one segment
         nseg        = 1
         ifseg(1)    = .TRUE.
         ninseg(1)   = pstat_nel
         ! Multiple passes eliminates false positives
         do ipass=1,2
            do jl=1,2          ! Sort within each segment (proc, element nr)

               il=1
               do iseg=1,nseg
                  call ituple_sort(isort(1,il),idim,ninseg(iseg),jl,1,
     $              ind,iwork)     ! key = jl
                  il = il + ninseg(iseg)
               enddo

               do il=2,pstat_nel
                  ! find segments borders
                  if (isort(jl,il).ne.isort(jl,il-1)) ifseg(il)=.TRUE.
               enddo

               ! Count up number of different segments
               nseg = 0
               do il=1,pstat_nel
               if (ifseg(il)) then
                  nseg = nseg+1
                  ninseg(nseg) = 1
               else
                  ninseg(nseg) = ninseg(nseg) + 1
               endif
               enddo
            enddo                  ! jl=1,2
         enddo                     ! ipass=1,2
         ! sorting end

         ! check global number of segments
         gnseg = iglsum(nseg,1)

         ! remove multiplicities
         jl = 1
         do iseg=1,nseg
            do kl=1,idim
               isort(kl,iseg) = isort(kl,jl)
            enddo
            do il=jl+1,jl + ninseg(iseg)
               if(isort(3,iseg).lt.isort(3,jl)) then
                  do kl=1,idim
                     isort(kl,iseg) = isort(kl,il)
                  enddo
               endif
            enddo
            jl = jl + ninseg(iseg)
         enddo

#ifdef DEBUG
         ! for testing
         ! to output refinement
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='CRDsort.txt'//str1//'i'//str2)

         write(iunit,*) nseg

         do il=1,nseg
            write(iunit,*) il, ninseg(il), (isort(jl,il),jl=1,idim)
         enddo

         close(iunit)
#endif

         ! transfer and sort data
         call fgslib_crystal_ituple_transfer
     &      (cr_h,isort,idim,nseg,isdim,1)
         il = 2
         call fgslib_crystal_ituple_sort
     &      (cr_h,isort,idim,nseg,il,1)

         ! remove duplicates
         iseg = 1
         do il = 2, nseg
            if (isort(2,iseg).eq.isort(2,il)) then
               if (isort(3,iseg).lt.isort(3,il)) then
                  do kl=1,idim
                     isort(kl,iseg) = isort(kl,il)
                  enddo
               endif
            else
               iseg = iseg + 1
               if (iseg.ne.il) then
                  do kl=1,idim
                     isort(kl,iseg) = isort(kl,il)
                  enddo
               endif
            endif
         enddo
         nseg = iseg

#ifdef DEBUG
         ! for testing
         ! to output refinement
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='CRDtrans.txt'//str1//'i'//str2)

         write(iunit,*) nseg, nelv

         do il=1,nseg
            write(iunit,*) il, (isort(jl,il),jl=1,idim)
         enddo

         close(iunit)
#endif

         ! check consistency
         if (nseg.ne.nelt) then
            ierr = 1
         else
            ierr = 0
         endif
         call mntr_check_abort(pstat_id,ierr,
     $     'Inconsistent segment number in pstat_mesh_manipulate')

#ifdef AMR
         ! transfer data for refinement mark
         do il = 1, nseg
            pstat_refl(isort(2,il)) = isort(3,il)
         enddo

         ! perform refinement
         call amr_refinement()

         inf_cnt = inf_cnt + 1
         if (inf_cnt.gt.99) call mntr_abort(pstat_id,
     $     'Infinit loop too long; possible problem with mark check.')

         pstat_elmod = iglsum(pstat_elmod,1)
         if (gnseg.eq.pstat_nelg.and.pstat_elmod.eq.0) then
            call mntr_logi(pstat_id,lp_prd,
     $          'Finished refinement; cycles number :', inf_cnt)
            exit
         endif
#else
         exit
#endif
      enddo

      ! set global element number for transfer
      do il = 1, nelt
         pstat_gnel(isort(2,il)) = isort(4,il)
      enddo

      ! notify element owners
      do il = 1, nelt
         isort(1,il) = gllnid(pstat_gnel(il))
         isort(2,il) = gllel(pstat_gnel(il))
         isort(3,il) = il
         isort(4,il) = lglel(il)
      enddo

      ! transfer and sort data
      iseg = nelt
      call fgslib_crystal_ituple_transfer(cr_h,isort,idim,iseg,isdim,1)
      ! sanity check
      ierr = 0
      if (iseg.ne.nelt) ierr = 1
      call mntr_check_abort(pstat_id,ierr,
     $     'Inconsistent element number in pstat_mesh_manipulate')
      il = 2
      call fgslib_crystal_ituple_sort(cr_h,isort,idim,iseg,il,1)

#ifdef DEBUG
      ! sanity check
      ierr = 0
      do il=1,iseg
         if (isort(3,il).ne.gllel(isort(4,il))) ierr = 1
      enddo
      call mntr_check_abort(pstat_id,ierr,
     $  'Inconsistent element transfer number in pstat_mesh_manipulat')
      ierr = 0
      do il=1,iseg
         if (isort(1,il).ne.gllnid(isort(4,il))) ierr = 1
      enddo
      call mntr_check_abort(pstat_id,ierr,
     $  'Inconsistent process transfer number in pstat_mesh_manipulat')
#endif

      ! save transfer data
      do il=1,nelt
         pstat_gnel(il) = isort(4,il)
      enddo

#ifdef DEBUG
      ! for testing
      ! to output refinement
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalld
      open(unit=iunit,file='CRDmap.txt'//str1//'i'//str2)

      write(iunit,*) nelgt,nelt
      do il=1,nelt
         write(iunit,*) il, pstat_gnel(il)
      enddo

      close(iunit)
#endif

#undef DEBUG
      return
      end subroutine
!=======================================================================
!> @brief Reshuffle elements between sts and current ordering
!! @ingroup pstat2d
      subroutine pstat2d_transfer()
      implicit none

      include 'SIZE'
      include 'GEOM'
      include 'SOLN'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer il, jl
      integer lnelt, itmp
      integer isw
      parameter (isw=2)

      ! transfer arrays
      integer vi(isw,LELT)
      integer*8 vl(1)           ! required by crystal rauter

      integer key(1)            ! required by crystal rauter; for sorting
!-----------------------------------------------------------------------
      ! element size
      itmp = lx1*ly1*lz1

      ! coordinates
      ! number of local elements
      lnelt = nelt

      ! pack transfer data
      do il=1,lnelt
         vi(1,il) = gllel(pstat_gnel(il))
         vi(2,il) = gllnid(pstat_gnel(il))
      enddo

      ! transfer arrays
      call fgslib_crystal_tuple_transfer
     $     (cr_h,lnelt,LELT,vi,isw,vl,0,xm1,itmp,2)

      ! test local element number
      if (lnelt.ne.NELT) then
         call mntr_abort('Error: pstat_transfer; lnelt /= nelt')
      endif

      ! sort elements acording to their global number
      key(1) = 1
      call fgslib_crystal_tuple_sort
     $     (cr_h,lnelt,vi,isw,vl,0,xm1,itmp,key,1)

      ! number of local elements
      lnelt = nelt

      ! pack transfer data
      do il=1,lnelt
         vi(1,il) = gllel(pstat_gnel(il))
         vi(2,il) = gllnid(pstat_gnel(il))
      enddo

      ! transfer arrays
      call fgslib_crystal_tuple_transfer
     $     (cr_h,lnelt,LELT,vi,isw,vl,0,ym1,itmp,2)

      ! test local element number
      if (lnelt.ne.NELT) then
         call mntr_abort('Error: pstat_transfer; lnelt /= nelt')
      endif

      ! sort elements acording to their global number
      key(1) = 1
      call fgslib_crystal_tuple_sort
     $     (cr_h,lnelt,vi,isw,vl,0,ym1,itmp,key,1)

      ! for each variable
      do il=1, pstat_svar
         ! number of local elements
         lnelt = nelt

         ! pack transfer data
         do jl=1,lnelt
            vi(1,jl) = gllel(pstat_gnel(jl))
            vi(2,jl) = gllnid(pstat_gnel(jl))
         enddo

         ! transfer array
         call fgslib_crystal_tuple_transfer
     $        (cr_h,lnelt,LELT,vi,isw,vl,0,t(1,1,1,1,il+1),itmp,2)

         ! test local element number
         if (lnelt.ne.NELT) then
            call mntr_abort('Error: pstat_transfer; lnelt /= nelt')
         endif

         ! sort elements acording to their global number
         key(1) = 1
         call fgslib_crystal_tuple_sort
     $        (cr_h,lnelt,vi,isw,vl,0,t(1,1,1,1,il+1),itmp,key,1)
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Read in fields and average them
!! @ingroup pstat2d
      subroutine pstat2d_sts_avg
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'RESTART'
      include 'TSTEP'
      include 'SOLN'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer il, jl            ! loop index
      integer ierr              ! error mark
      integer nvec              ! single field length
      character*3 prefix        ! file prefix
      character*132 fname       ! file name
      character*132 bname       ! base name
      character*6  str          ! file number
      integer nps1,nps0,npsr    ! number of passive scalars in the file
      integer istepr            ! number of time step in restart files
      real ltime, dtime         ! simulation time and time update
      real rtmp

!-----------------------------------------------------------------------
      ! no regular mesh; important for file name generation
      ifreguo = .false.

      call io_init

      ierr=0
      ! open files on i/o nodes
      prefix='sts'
      ! get base name (SESSION)
      bname = trim(adjustl(session))

      ! mark variables to be read
      ifgetx=.true.
      ifgetz=.false.
      ifgetu=.false.
      ifgetw=.false.
      ifgetp=.false.
      ifgett=.false.
      do il=1,pstat_svar
          ifgtps(il)=.TRUE.
      enddo

      ! initial time and step count
      ltime = pstat_stime
      istepr = 0

      ! initilise vectors
      call rzero(pstat_ruavg,lx1**ldim*lelt*pstat_svar)
      nvec = lx1*ly1*nelt

      ! loop over stat files
      do il = 1,pstat_nfile

         call io_mfo_fname(fname,bname,prefix,ierr)
         write(str,'(i5.5)') il
         fname = trim(fname)//trim(str)

         fid0 = 0
         call addfid(fname,fid0)

         ! add directory name
         fname = 'DATA/'//trim(fname)

         !call load_fld(fname)
         call mfi(fname,il)

         ! extract number of passive scalars in the file and check consistency
         do jl=1,10
            if (rdcode1(jl).eq.'S') then
               read(rdcode1(jl+1),'(i1)') nps1
               read(rdcode1(jl+2),'(i1)') nps0
               npsr = 10*nps1+nps0
            endif
         enddo
         if (npsr.ne.pstat_svar) call mntr_abort(pstat_id,
     $        'Inconsistent number of fielsd in sts file.')

         ! calculate interval and update time
         dtime = timer - ltime
         ltime = timer

         ! sum number of time steps
         istepr = istepr + istpr

         ! reshuffle lelements
         call pstat2d_transfer()

         ! accumulate fileds
         do jl = 1,pstat_svar
            call add2s2(pstat_ruavg(1,1,pstat_swfield(jl)),
     $           t(1,1,1,1,jl+1),dtime,nvec)
         enddo

      enddo

      ! save data for file header
      pstat_etime = ltime
      pstat_istepr = istepr

      ! divide by time span
      if (ltime.ne.pstat_stime) then
         rtmp = 1.0/(ltime-pstat_stime)
      else
         rtmp = 1.0
      endif

      do il = 1,pstat_svar
         call cmult(pstat_ruavg(1,1,il),rtmp,nvec)
      enddo

      ! save all averaged fields
      ifvo = .false.
      ifpo = .false.
      ifto = .false.
      do il=1,pstat_svar
          ifpsco(il)=.TRUE.
      enddo
      ! put variables back to temperature
      do il=1,pstat_svar
         call copy(t(1,1,1,1,il+1),pstat_ruavg(1,1,il),nvec)
      enddo
      call outpost2(vx,vy,vz,pr,t,pstat_svar+1,'st1')

      return
      end subroutine
!=======================================================================
!> @brief Calculate derivatives
!! @ingroup pstat2d
      subroutine pstat2d_deriv
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! local variables
      integer il                    ! loop index
      integer nvec                  ! single field length
      real dudx(lx1*ly1*lz1,lelt,3) ! field derivatives
!-----------------------------------------------------------------------
      ! update derivative arrays
      call geom_reset(1)

      ! initilise vectors
      call rzero(pstat_ruder,lx1**ldim*lelt*pstat_dvar)
      nvec = lx1*ly1*nelt

      ! Tensor 1. Compute 2D derivative tensor with U
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,1))
      call copy(pstat_ruder(1,1,1),dudx(1,1,1),nvec)        ! dU/dx
      call copy(pstat_ruder(1,1,2),dudx(1,1,2),nvec)        ! dU/dy

      ! Tensor 1. Compute 2D derivative tensor with V
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,2))
      call copy(pstat_ruder(1,1,3),dudx(1,1,1),nvec)        ! dV/dx
      call copy(pstat_ruder(1,1,4),dudx(1,1,2),nvec)        ! dV/dy

      ! Tensor 2. Compute 2D derivative tensor with W
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,3))
      call copy(pstat_ruder(1,1,5),dudx(1,1,1),nvec)        ! dW/dx
      call copy(pstat_ruder(1,1,6),dudx(1,1,2),nvec)        ! dW/dy

      ! Tensor 2. Compute 2D derivative tensor with P
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,4))
      call copy(pstat_ruder(1,1,7),dudx(1,1,1),nvec)        ! dP/dx
      call copy(pstat_ruder(1,1,8),dudx(1,1,2),nvec)        ! dP/dy

      ! Tensor 3. Compute 2D derivative tensor with <uu>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,5))
      call copy(pstat_ruder(1,1,9),dudx(1,1,1),nvec)        ! d<uu>/dx
      call copy(pstat_ruder(1,1,10),dudx(1,1,2),nvec)       ! d<uu>/dy

      ! Tensor 3. Compute 2D derivative tensor with <vv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,6))
      call copy(pstat_ruder(1,1,11),dudx(1,1,1),nvec)       ! d<vv>/dx
      call copy(pstat_ruder(1,1,12),dudx(1,1,2),nvec)       ! d<vv>/dy

      ! Tensor 4. Compute 2D derivative tensor with <ww>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,7))
      call copy(pstat_ruder(1,1,13),dudx(1,1,1),nvec)       ! d<ww>/dx
      call copy(pstat_ruder(1,1,14),dudx(1,1,2),nvec)       ! d<ww>/dy

      ! Tensor 4. Compute 2D derivative tensor with <pp>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,8))
      call copy(pstat_ruder(1,1,15),dudx(1,1,1),nvec)       ! d<pp>/dx
      call copy(pstat_ruder(1,1,16),dudx(1,1,2),nvec)       ! d<pp>/dy

      ! Tensor 5. Compute 2D derivative tensor with <uv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,9))
      call copy(pstat_ruder(1,1,17),dudx(1,1,1),nvec)       ! d<uv>/dx
      call copy(pstat_ruder(1,1,18),dudx(1,1,2),nvec)       ! d<uv>/dy

      ! Tensor 5. Compute 2D derivative tensor with <vw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,10))
      call copy(pstat_ruder(1,1,19),dudx(1,1,1),nvec)       ! d<vw>/dx
      call copy(pstat_ruder(1,1,20),dudx(1,1,2),nvec)       ! d<vw>/dy

      ! Tensor 6. Compute 2D derivative tensor with <uw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,11))
      call copy(pstat_ruder(1,1,21),dudx(1,1,1),nvec)       ! d<uw>/dx
      call copy(pstat_ruder(1,1,22),dudx(1,1,2),nvec)       ! d<uw>/dy

      ! Tensor 6. Compute 2D derivative tensor with <uuu>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,24))
      call copy(pstat_ruder(1,1,23),dudx(1,1,1),nvec)       ! d<uuu>/dx
      call copy(pstat_ruder(1,1,24),dudx(1,1,2),nvec)       ! d<uuu>/dy

      ! Tensor 7. Compute 2D derivative tensor with <vvv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,25))
      call copy(pstat_ruder(1,1,25),dudx(1,1,1),nvec)       ! d<vvv>/dx
      call copy(pstat_ruder(1,1,26),dudx(1,1,2),nvec)       ! d<vvv>/dy

      ! Tensor 7. Compute 2D derivative tensor with <www>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,26))
      call copy(pstat_ruder(1,1,27),dudx(1,1,1),nvec)       ! d<www>/dx
      call copy(pstat_ruder(1,1,28),dudx(1,1,2),nvec)       ! d<www>/dy

      ! Tensor 8. Compute 2D derivative tensor with <ppp>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,27))
      call copy(pstat_ruder(1,1,29),dudx(1,1,1),nvec)       ! d<ppp>/dx
      call copy(pstat_ruder(1,1,30),dudx(1,1,2),nvec)       ! d<ppp>/dy

      ! Tensor 8. Compute 2D derivative tensor with <uuv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,28))
      call copy(pstat_ruder(1,1,31),dudx(1,1,1),nvec)       ! d<uuv>/dx
      call copy(pstat_ruder(1,1,32),dudx(1,1,2),nvec)       ! d<uuv>/dy

      ! Tensor 9. Compute 2D derivative tensor with <uuw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,29))
      call copy(pstat_ruder(1,1,33),dudx(1,1,1),nvec)       ! d<uuw>/dx
      call copy(pstat_ruder(1,1,34),dudx(1,1,2),nvec)       ! d<uuw>/dy

      ! Tensor 9. Compute 2D derivative tensor with <vvu>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,30))
      call copy(pstat_ruder(1,1,35),dudx(1,1,1),nvec)       ! d<vvu>/dx
      call copy(pstat_ruder(1,1,36),dudx(1,1,2),nvec)       ! d<vvu>/dy

      ! Tensor 10. Compute 2D derivative tensor with <vvw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,31))
      call copy(pstat_ruder(1,1,37),dudx(1,1,1),nvec)       ! d<vvw>/dx
      call copy(pstat_ruder(1,1,38),dudx(1,1,2),nvec)       ! d<vvw>/dy

      ! Tensor 10. Compute 2D derivative tensor with <wwu>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,32))
      call copy(pstat_ruder(1,1,39),dudx(1,1,1),nvec)       ! d<wwu>/dx
      call copy(pstat_ruder(1,1,40),dudx(1,1,2),nvec)       ! d<wwu>/dy

      ! Tensor 11. Compute 2D derivative tensor with <wwv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,33))
      call copy(pstat_ruder(1,1,41),dudx(1,1,1),nvec)       ! d<wwu>/dx
      call copy(pstat_ruder(1,1,42),dudx(1,1,2),nvec)       ! d<wwu>/dy

      ! Tensor 11. Compute 2D derivative tensor with <uvw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,34))
      call copy(pstat_ruder(1,1,43),dudx(1,1,1),nvec)       ! d<uvw>/dx
      call copy(pstat_ruder(1,1,44),dudx(1,1,2),nvec)       ! d<uvw>/dy

      ! Tensor 12. Compute 2D derivative tensor with dU/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,1))
      call copy(pstat_ruder(1,1,45),dudx(1,1,1),nvec)       ! d2U/dx2

      ! Tensor 12. Compute 2D derivative tensor with dU/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,2))
      call copy(pstat_ruder(1,1,46),dudx(1,1,2),nvec)       ! d2U/dy2

      ! Tensor 13. Compute 2D derivative tensor with dV/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,3))
      call copy(pstat_ruder(1,1,47),dudx(1,1,1),nvec)       ! d2V/dx2

      ! Tensor 13. Compute 2D derivative tensor with dV/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,4))
      call copy(pstat_ruder(1,1,48),dudx(1,1,2),nvec)       ! d2V/dy2

      ! Tensor 14. Compute 2D derivative tensor with dW/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,5))
      call copy(pstat_ruder(1,1,49),dudx(1,1,1),nvec)       ! d2W/dx2

      ! Tensor 14. Compute 2D derivative tensor with dW/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,6))
      call copy(pstat_ruder(1,1,50),dudx(1,1,2),nvec)       ! d2W/dy2

      ! Tensor 15. Compute 2D derivative tensor with d<uu>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,9))
      call copy(pstat_ruder(1,1,51),dudx(1,1,1),nvec)       ! d2<uu>/dx2

      ! Tensor 15. Compute 2D derivative tensor with d<uu>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,10))
      call copy(pstat_ruder(1,1,52),dudx(1,1,2),nvec)       ! d2<uu>/dy2

      ! Tensor 16. Compute 2D derivative tensor with d<vv>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,11))
      call copy(pstat_ruder(1,1,53),dudx(1,1,1),nvec)       ! d2<vv>/dx2

      ! Tensor 16. Compute 2D derivative tensor with d<vv>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,12))
      call copy(pstat_ruder(1,1,54),dudx(1,1,2),nvec)       ! d2<vv>/dy2

      ! Tensor 17. Compute 2D derivative tensor with d<ww>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,13))
      call copy(pstat_ruder(1,1,55),dudx(1,1,1),nvec)       ! d2<ww>/dx2

      ! Tensor 17. Compute 2D derivative tensor with d<ww>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,14))
      call copy(pstat_ruder(1,1,56),dudx(1,1,2),nvec)       ! d2<ww>/dy2

      ! Tensor 18. Compute 2D derivative tensor with d<uv>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,17))
      call copy(pstat_ruder(1,1,57),dudx(1,1,1),nvec)       ! d2<uv>/dx2

      ! Tensor 18. Compute 2D derivative tensor with d<uv>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,18))
      call copy(pstat_ruder(1,1,58),dudx(1,1,2),nvec)       ! d2<uv>/dy2

      ! Tensor 19. Compute 2D derivative tensor with d<uw>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,21))
      call copy(pstat_ruder(1,1,59),dudx(1,1,1),nvec)       ! d2<uw>/dx2

      ! Tensor 19. Compute 2D derivative tensor with d<uw>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,22))
      call copy(pstat_ruder(1,1,60),dudx(1,1,2),nvec)       ! d2<uw>/dy2

      ! Tensor 20. Compute 2D derivative tensor with d<vw>/dx
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,19))
      call copy(pstat_ruder(1,1,61),dudx(1,1,1),nvec)       ! d2<vw>/dx2

      ! Tensor 20. Compute 2D derivative tensor with d<vw>/dy
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruder(1,1,20))
      call copy(pstat_ruder(1,1,62),dudx(1,1,2),nvec)       ! d2<vw>/dy2

      ! Tensor 21. Compute 2D derivative tensor with <pu>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,12))
      call copy(pstat_ruder(1,1,63),dudx(1,1,1),nvec)       ! d<pu>/dx
      call copy(pstat_ruder(1,1,64),dudx(1,1,2),nvec)       ! d<pu>/dy

      ! Tensor 21. Compute 2D derivative tensor with <pv>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,13))
      call copy(pstat_ruder(1,1,65),dudx(1,1,1),nvec)       ! d<pv>/dx
      call copy(pstat_ruder(1,1,66),dudx(1,1,2),nvec)       ! d<pv>/dy

      ! Tensor 22. Compute 2D derivative tensor with <pw>
      call gradm1(dudx(1,1,1),dudx(1,1,2),dudx(1,1,3),
     $      pstat_ruavg(1,1,14))
      call copy(pstat_ruder(1,1,67),dudx(1,1,1),nvec)       ! d<pw>/dx
      call copy(pstat_ruder(1,1,68),dudx(1,1,2),nvec)       ! d<pw>/dy

      ! save all derivatives
      ifvo = .false.
      ifpo = .false.
      ifto = .false.
      do il=1,pstat_dvar
          ifpsco(il)=.TRUE.
      enddo
      ! put variables back to temperature
      do il=1,pstat_dvar
         call copy(t(1,1,1,1,il+1),pstat_ruder(1,1,il),nvec)
      enddo
      call outpost2(vx,vy,vz,pr,t,pstat_dvar+1,'st2')

      return
      end subroutine
!=======================================================================
!> @brief Interpolate int the set of points
!! @ingroup pstat2d
      subroutine pstat2d_interp
      implicit none

      include 'SIZE'
      include 'GEOM'
      include 'FRAMELP'
      include 'PSTAT2D'

      ! global data structures
      integer mid,mp,nekcomm,nekgroup,nekreal
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      ! local variables
      integer ifpts       ! findpts flag
      real tol            ! interpolation tolerance
      integer nt
      integer npt_max     ! max communication set
      integer nxf, nyf, nzf  ! fine mesh for bb-test
      real bb_t           ! relative size to expand bounding boxes by
      real toldist
      parameter (toldist = 5e-6)

      integer rcode(lhis), proc(lhis),elid(lhis)
      real dist(lhis), rst(ldim*lhis)

      integer nfail, npass

      integer il                    ! loop index
      !integer nvec                  ! single field length

      ! functions
      integer iglsum

!#define DEBUG
#ifdef DEBUG
      character*3 str1, str2
      integer iunit, ierr, jl
      ! call number
      integer icalld
      save icalld
      data icalld /0/
#endif
!-----------------------------------------------------------------------
      ! read point position
      call pstat2d_mfi_interp

      ! initialise interpolation tool
      tol     = 5e-13
      nt       = lx1*ly1*lz1*lelt
      npt_max = 256
      nxf     = 2*lx1
      nyf     = 2*ly1
      nzf     = 2*lz1
      bb_t    = 0.01

      ! start interpolation tool on given mesh
      call fgslib_findpts_setup(ifpts,nekcomm,mp,ldim,xm1,ym1,zm1,
     &     lx1,ly1,lz1,nelt,nxf,nyf,nzf,bb_t,nt,nt,npt_max,tol)

      ! identify points
      call fgslib_findpts(ifpts,rcode,1,proc,1,elid,1,rst,ldim,dist,1,
     &     pstat_int_pts(1,1),ldim,pstat_int_pts(2,1),ldim,
     &     pstat_int_pts(ldim,1),ldim,pstat_npt)

      ! find problems with interpolation
      nfail = 0
      do il = 1,pstat_npt
         ! check return code
         if (rcode(il).eq.1) then
            if (sqrt(dist(il)).gt.toldist) nfail = nfail + 1
         elseif(rcode(il).eq.2) then
            nfail = nfail + 1
         endif
      enddo
      nfail = iglsum(nfail,1)

#ifdef DEBUG
         ! for testing
         ! to output refinement
         icalld = icalld+1
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='INTfpts.txt'//str1//'i'//str2)

         write(iunit,*) pstat_nptot, pstat_npt, nfail
         do il=1,pstat_npt
            write(iunit,*) il, proc(il), elid(il), rcode(il), dist(il),
     $       (rst(jl+(il-1)*ldim),jl=1,ldim)
         enddo

         close(iunit)
#endif

      if (nfail.gt.0) call mntr_abort(pstat_id,
     $     'pstat_interp: Points not mapped')

      ! Interpolate averaged fields
      do il=1,pstat_svar
         call fgslib_findpts_eval(ifpts,pstat_int_avg (1,il),1,
     &        rcode,1,proc,1,elid,1,rst,ndim,pstat_npt,
     &        pstat_ruavg(1,1,il))
      enddo

#ifdef DEBUG
         ! for testing
         ! to output refinement
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='INTavg.txt'//str1//'i'//str2)

         write(iunit,*) pstat_nptot, pstat_npt
         do il=1,pstat_npt
            write(iunit,*) il, (pstat_int_avg(il,jl),jl=1,4)
         enddo

         close(iunit)
#endif

      ! Interpolate fields derivatives
      do il=1,pstat_dvar
         call fgslib_findpts_eval(ifpts,pstat_int_der (1,il),1,
     &        rcode,1,proc,1,elid,1,rst,ndim,pstat_npt,
     &        pstat_ruder(1,1,il))
      enddo

#ifdef DEBUG
         ! for testing
         ! to output refinement
         call io_file_freeid(iunit, ierr)
         write(str1,'(i3.3)') NID
         write(str2,'(i3.3)') icalld
         open(unit=iunit,file='INTder.txt'//str1//'i'//str2)

         write(iunit,*) pstat_nptot, pstat_npt
         do il=1,pstat_npt
            write(iunit,*) il, (pstat_int_der(il,jl),jl=1,4)
         enddo

         close(iunit)
#endif

      ! finalise interpolation tool
      call fgslib_findpts_free(ifpts)

      ! write down interpolated values
      call pstat2d_mfo_interp

#undef DEBUG
      return
      end subroutine
!=======================================================================
