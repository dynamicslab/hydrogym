!> @file map2D.f
!! @ingroup map2d
!! @brief 3D to 2D element mapping routines
!! @details Partially based on the old statistic code
!! @note This code works for extruded meshes only
!! @author Adam Peplinski
!! @date May 29, 2018
!=======================================================================
!> @brief Register 2D mapping routines
!! @ingroup map2d
!! @note This routine should be called in frame_usr_register
      subroutine map2d_register()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'FRAMELP'
      include 'MAP2D'

!     local variables
      integer lpmid, il
      real ltim
      character*2 str

!     functions
      real dnekclock
!-----------------------------------------------------------------------
!     timing
      ltim = dnekclock()

!     check if the current module was already registered
      call mntr_mod_is_name_reg(lpmid,map2d_name)
      if (lpmid.gt.0) then
         call mntr_warn(lpmid,
     $        'module ['//trim(map2d_name)//'] already registered')
         return
      endif

!     find parent module
      call mntr_mod_is_name_reg(lpmid,'FRAME')
      if (lpmid.le.0) then
         lpmid = 1
         call mntr_abort(lpmid,
     $        'parent module ['//'FRAME'//'] not registered')
      endif
      
!     register module
      call mntr_mod_reg(map2d_id,lpmid,map2d_name,
     $      'Mapping 3D mesh to 2D section')

!     register timer
      call mntr_tmr_is_name_reg(lpmid,'FRM_TOT')
      call mntr_tmr_reg(map2d_tmr_id,lpmid,map2d_id,
     $     'MAP2D_TOT','2D mapping total time',.false.)

!     set initialisation flag
      map2d_ifinit=.false.
      
!     timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(map2d_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Initilise map2d module
!! @ingroup map2d
!! @note This routine should be called in frame_usr_init
      subroutine map2d_init()
      implicit none

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'FRAMELP' 
      include 'MAP2D'

!     local variables
      integer itmp
      real rtmp, ltim
      logical ltmp
      character*20 ctmp

      integer il, jl

!     functions
      real dnekclock
!-----------------------------------------------------------------------
!     timing
      ltim = dnekclock()      

!     check if the module was already initialised
      if (map2d_ifinit) then
         call mntr_warn(map2d_id,
     $        'module ['//trim(map2d_name)//'] already initiaised.')
         return
      endif

      call map2d_get

!     reshuffle coordinate arrays
      call mntr_log(map2d_id,lp_vrb,'Creating 2D mesh')
      call map2d_init_coord

!     everything is initialised
      map2d_ifinit=.true.
      
!     timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(map2d_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Get 3D to 2D element mapping
!! @ingroup map2d
!! @remark This routine uses global scratch space \a SCRNS , \a SCRVH
!!   and \a SCRUZ
      subroutine map2d_get()
      implicit none

      include 'SIZE'
      include 'FRAMELP' 
      include 'MAP2D'           ! 2D mapping speciffic variables

!     work arrays
      integer lctrs1 ,lctrs2    ! array sizes
      parameter (lctrs1=3,lctrs2=2*LX1*LY1*LZ1*LELT)
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      integer owner(lctrs2)     ! mark node with smallest id
      logical ifseg(lctrs2)     ! segment borders
      common /SCRNS/ ctrs
      common /SCRVH/ ifseg
      common /SCRUZ/ cell, ninseg, ind, owner

      integer nelsort           ! number of local 3D elements to sort
      integer nseg              ! segments number
      integer il, jl, iseg      ! loop index
      integer ierr              ! error flag

      real ltol                 ! tolerance for detection of section borders
      parameter (ltol = 1.0e-4)
      
!     simple timing
      real ltim

!     functions
      integer iglsum, iglmin, iglmax
      real dnekclock

!#define DEBUG
#ifdef DEBUG
!     for testing
      character*3 str1, str2
      integer iunit
      ! call number
      integer icalldl
      save icalldl
      data icalldl /0/
#endif
!-----------------------------------------------------------------------
!     simple timing
      ltim = dnekclock()
      
!     stamp logs
      call mntr_log(map2d_id,lp_inf,'3D=>2D mapping begin')

      
      call mntr_log(map2d_id,lp_vrb,'Local 3D=>2D mapping')

!     fill in arrays using user interface
!     We can sort only part of the domain, so first mark and copy
!     all elements in the region you are interested in
!     set uniform direction, cell centres and diagonals
      call user_map2d_get(map2d_idir,ctrs,cell,lctrs1,lctrs2,nelsort,
     $     map2d_xm1,map2d_ym1,ierr)

      call mntr_check_abort(map2d_id,ierr,'Wrong element shape')

      if (map2d_idir.gt.ldim) call mntr_abort(map2d_id,
     $        'Wrong mapping direction')
      
!     check array sizes vs number of elements for sorting
      if (lctrs2.lt.nelsort) then
         ierr = 1
      else
         ierr = 0
      endif
      call mntr_check_abort(map2d_id,ierr,'Too many element to sort')

      if (NELT.lt.nelsort) then
         ierr = 1
      else
         ierr = 0
      endif
      call mntr_check_abort(map2d_id,ierr,
     $     'More element to sort than local elements')
      
!     local sort to get unique 2D elements
      call map2d_get_local(ctrs,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg,nelsort,ltol)

!     generate local 3D => 2D mapping
!     local number of unique 2D elements
      MAP2D_LNUM = nseg

!     mark all elements as unwanted
      call ifill(MAP2D_LMAP,-1,NELT)
      
!     for all segments count 3D elements
      jl=1
      do iseg=1,nseg
!     within segment
         do il=1,ninseg(iseg)
            MAP2D_LMAP(cell(jl)) = iseg
            jl=jl+1
         enddo
      enddo

#ifdef DEBUG
!     testing
      icalldl = icalldl+1
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalldl
      open(unit=iunit,file='map2d_loc.txt'//str1//'i'//str2)
      
      write(iunit,*) nseg, NELV
      write(iunit,*) 'Coordinates'
      do il=1,nseg
         write(iunit,*) il, ctrs(:,il)
      enddo
      write(iunit,*) 'Mapping'
      do il=1,NELV
         write(iunit,*) il, MAP2D_LMAP(il)
      enddo
      close(iunit)
#endif
      
      call mntr_log(map2d_id,lp_vrb,'Global 3D=>2D mapping')

!     reset ownership and local => global map
!     this routine will produce the simplest ownership without 
!     taking into account work ballancing
      call ifill(MAP2D_GMAP,-1,nseg)
      call ifill(MAP2D_OWN,-1,nseg)
      
      call map2d_get_global(ctrs,owner,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg,ltol)

!     find number of of elements owned
      MAP2D_LOWN = 0
      do il=1,MAP2D_LNUM
         if (MAP2D_OWN(il).eq.NID) MAP2D_LOWN = MAP2D_LOWN + 1
      enddo
!     global number of unique 2D elements
      MAP2D_GNUM = iglsum(MAP2D_LOWN,1)
!     imbalance
      il = iglmin(MAP2D_LOWN,1)
      jl = iglmax(MAP2D_LOWN,1)
      
!     stamp logs
      call mntr_logi(map2d_id,lp_inf,
     $     'Global number of unique 2D elements: ',MAP2D_GNUM)
      call mntr_log(map2d_id,lp_vrb,'Owned 2D element imbalance:')
      call mntr_logi(map2d_id,lp_vrb,'   min: ',il)
      call mntr_logi(map2d_id,lp_vrb,'   max: ',jl)

#ifdef DEBUG
!     testing
      call io_file_freeid(iunit, ierr)
      write(str1,'(i3.3)') NID
      write(str2,'(i3.3)') icalldl
      open(unit=iunit,file='map2d_glob.txt'//str1//'i'//str2)
      
      write(iunit,*) MAP2D_GNUM, MAP2D_LOWN
      write(iunit,*) 'Mapping'
      do il=1,MAP2D_LNUM
         write(iunit,*) il, MAP2D_GMAP(il),MAP2D_OWN(il)
      enddo
      close(iunit)
#endif
#undef DEBUG

!     stamp logs
      call mntr_log(map2d_id,lp_inf,'3D=>2D mapping end')

!     timing
      ltim = dnekclock() - ltim
      call mntr_tmr_add(map2d_tmr_id,1,ltim)

      return
      end subroutine
!=======================================================================
!> @brief Get local 3D=>2D mapping
!! @ingroup map2d
!! @param[out]  ctrs              2D element centres
!! @param[out]  cell              local element numberring
!! @param[out]  ninseg            elements in segmen (work array)
!! @param[out]  ind               sorting index (work array)
!! @param[out]  ifseg             segment borders (work array)
!! @param[in]   lctrs1,lctrs2     array sizes
!! @param[out]  nseg              segments number
!! @param[in]   nelsort           number of local 3D elements to sort
!! @param[in]   tol               tolerance to find segment borders
      subroutine map2d_get_local(ctrs,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg,nelsort,tol)
      implicit none

      include 'SIZE'

!     argument list
      integer lctrs1,lctrs2     ! array sizes
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer nseg              ! segments number
      integer nelsort           ! number of local 3D elements to sort
      real tol                  ! tolerance to find segment borders
!     work arrays
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      logical ifseg(lctrs2)     ! segment borders

!     local variables
      integer el, il, jl        ! loop indexes
      integer ierr              ! error flag

!     local sorting
      integer key               ! sorting key
      integer ipass, iseg       ! loop index
      real aa(lctrs1)           ! dummy array
!-----------------------------------------------------------------------
!     for every element
      do el=1,nelsort
!     reset segments borders
         ifseg(el) = .FALSE.
      enddo

!     perform local sorting to identify unique set sorting by directions
!     first run => whole set is one segment
      nseg        = 1
      ifseg(1)    = .TRUE.
      ninseg(1)   = nelsort

!     Multiple passes eliminates false positives
      do ipass=1,2
         do jl=1,ldim-1          ! Sort within each segment (dimention)

            il=1
            do iseg=1,nseg
               call tuple_sort(ctrs(1,il),lctrs1,ninseg(iseg),jl,1,
     $              ind,aa)     ! key = jl
               call iswap_ip(cell(il),ind,ninseg(iseg)) ! Swap position
               il = il + ninseg(iseg)
            enddo
 
            do il=2,nelsort
!     find segments borders
               if (abs(ctrs(jl,il)-ctrs(jl,il-1)).gt.
     $              tol*min(ctrs(3,il),ctrs(3,il-1)))
     $              ifseg(il)=.TRUE.
            enddo

!  Count up number of different segments
            nseg = 0
            do il=1,nelsort
               if (ifseg(il)) then
                  nseg = nseg+1
                  ninseg(nseg) = 1
               else
                  ninseg(nseg) = ninseg(nseg) + 1
               endif
            enddo
         enddo                  ! jl=1,2
      enddo                     ! ipass=1,2
!     sorting end

!     contract coordinate set
!     for all segments
!     count 3D elements
      jl=ninseg(1) +1
      do iseg=2,nseg
         do il = 1,lctrs1
            ctrs(il,iseg) = ctrs(il,jl)
         enddo
         jl = jl + ninseg(iseg)
      enddo

      return
      end subroutine
!=======================================================================
!> @brief Get global 3D=>2D mapping
!! @ingroup map2d
!! @param[inout]  ctrs            2D element centres
!! @param[inout]  owner           global element ownership (work array)
!! @param[inout]  cell            local element numberring (work array)
!! @param[inout]  ninseg          elements in segmen (work array)
!! @param[inout]  ind             sorting index (work array)
!! @param[inout]  ifseg           segment borders (work array)
!! @param[in]     lctrs1,lctrs2   array sizes
!! @param[inout]  nseg            segments number
!! @param[in]     tol             tolerance to find segment borders
      subroutine map2d_get_global(ctrs,owner,cell,ninseg,ind,ifseg,
     $     lctrs1,lctrs2,nseg,tol)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'FRAMELP'
      include 'MAP2D'

!     argument list
      integer lctrs1,lctrs2     ! array sizes
      real ctrs(lctrs1,lctrs2)  ! 2D element centres for sorting
      integer nseg              ! segments number
      real tol                  ! tolerance to find segment borders
!     work arrays
      integer owner(lctrs2)     ! mark node with smallest id
      integer cell(lctrs2)      ! local element numberring
      integer ninseg(lctrs2)    ! elements in segment
      integer ind(lctrs2)       ! sorting index
      logical ifseg(lctrs2)     ! segment borders

!     local variables
      integer lnseg             ! initia number of segments
      integer csteps            ! numer of steps in the cycle
      integer lwork             ! working array size
      integer umrkgl            ! global number of unmarked zones
!     local coppies
      real lctrs(lctrs1,LELT)   ! local copy 2D element centres
      integer nsort             ! number of elements to sort
      integer nsorted           ! number of sorted elements

      integer igpass            ! numer of executed cycles
      integer igpass_max        ! max numer of cycles
      parameter (igpass_max = 100)

      integer icstep            ! loop index

!     communication
      integer msg_id1, msg_id2  ! message id for non-blocking receive
      integer srcid, dstid      ! source and destination node id
      integer len               ! buffer size
      integer cnsort            ! number of elements to receive

!     local sorting
      integer key               ! sorting key
      integer ipass, iseg, il, jl, kl ! loop index
      real aa(lctrs1)           ! dummy array

!     error mark
      integer ierror

      character*3 str

!     functions
      integer iglsum, irecv, iglmin, iglmax
!-----------------------------------------------------------------------
!     make a local copy of initial set
      lnseg = nseg
      il = lctrs1*nseg
      call copy(lctrs,ctrs,il)

!     get number of steps to exchange all the data in the ring
      csteps=int(log(NP+0.0)/log(2.0))
      if(NP.gt.2**csteps) csteps=csteps+1
      
!     free array size
      lwork = lctrs2/2
      
!     get global number of unmarked zones
      umrkgl = iglsum(nseg,1)
!     initial number of elements to sort
      nsort = min(nseg,lwork)
!     initial number of sorted elements
      nsorted = 0

!     fill initial cell and ownership arrays
      do il=1,nsort
         cell(il) = il
         owner(il) = NID
      enddo
      
!     following loop has to be executed as long as unmarked zones 
!     exists
!     count global passes
      igpass = 1
      do

!     stamp log
         write(str,'(i3.3)')  igpass
         call mntr_logi(map2d_id,lp_vrb,
     $        'Cycle '//str//'; globally unmarked = ',umrkgl)

!     collect information within the ring
         do icstep=1,csteps

!     exchange information between processors
!     source and destination
            il = 2**(icstep-1)
            srcid = NID - il
            dstid = NID + il
            if (srcid.lt.0) srcid = srcid + NP
            if (dstid.ge.NP) dstid = dstid - NP

!     set buffer for the number of elements to receive
            len = ISIZE
            msg_id1 = irecv(0,cnsort,len)

!     send local size of the buffer
            call csend(0,nsort,len,dstid,0)

!     finish communication
            call msgwait(msg_id1)

!     exchange coordinates and ownership
!     receive
            len = WDSIZE*lctrs1*cnsort
            msg_id1 = irecv(1,ctrs(1,nsort+1),len)

            len = ISIZE*cnsort
            msg_id2 = irecv(2,owner(nsort+1),len)

!     send
            len = WDSIZE*lctrs1*nsort
            call csend(1,ctrs,len,dstid,0)

            len = ISIZE*nsort
            call csend(2,owner,len,dstid,0)

!     reset cell for the received elements to -1
!     this way all the non-local elements are marked
            do il=nsort + 1,nsort + cnsort
               cell(il) = -1
            enddo

!     update number of elements to sort
            nsort = nsort + cnsort

!     perform local sorting to identify unique set
!     sorting by directions
!     reset section boudarry mark
            do il=1,nsort
               ifseg(il) = .FALSE.
            enddo
!     first run => whole set is one segment
            nseg        = 1
            ifseg(1)    = .TRUE.
            ninseg(1)   = nsort

!     finish communication
            call msgwait(msg_id1)
            call msgwait(msg_id2)
            
! Multiple passes eliminates false positives
            do ipass=1,2
               do jl=1,ldim-1    ! Sort within each segment (dimension)

                  il =1
                  do iseg=1,nseg
                     call tuple_sort(ctrs(1,il),lctrs1,ninseg(iseg),
     $                    jl,1, ind,aa) ! key = jl
!     Swap position
                     call iswap_ip(cell(il),ind,ninseg(iseg)) 
                     call iswap_ip(owner(il),ind,ninseg(iseg))
                     il = il + ninseg(iseg)
                  enddo
 
                  do il=2,nsort
!     find segments borders
                     if (abs(ctrs(jl,il)-ctrs(jl,il-1)).gt.
     $                    tol*min(ctrs(3,il),ctrs(3,il-1)))
     $                    ifseg(il)=.TRUE.
                  enddo

!     Count up number of different segments
                  nseg = 0      
                  do il=1,nsort
                     if (ifseg(il)) then
                        nseg = nseg+1
                        ninseg(nseg) = 1
                     else
                        ninseg(nseg) = ninseg(nseg) + 1
                     endif
                  enddo
               enddo            ! jl=1,2
            enddo               ! ipass=1,2
!     local sorting end

!     contract coordinate set
!     for all segments
            jl=ninseg(1) +1
            do iseg=2,nseg
               do il = 1,lctrs1
                  ctrs(il,iseg) = ctrs(il,jl)
               enddo
               jl = jl + ninseg(iseg)
            enddo
!     contract ownership
!     for all segments
            jl=1
            do iseg=1,nseg
               owner(iseg) = owner(jl)
               jl = jl + 1
!     within segment
               do il=2,ninseg(iseg)
                  if (owner(iseg).gt.owner(jl)) owner(iseg) = owner(jl)
                  jl = jl + 1
               enddo
            enddo
!     contract cell
            ierror = 0
!     for all segments
            jl=1
            do iseg=1,nseg
               cell(iseg) = cell(jl)
               jl = jl + 1
!     within segment
!     for checking consistency
!     in every section can be only 1 non negative cell entrance
               kl = 0
               if (cell(iseg).ne.-1) kl = kl+1
               do il=2,ninseg(iseg)
                  if (cell(iseg).lt.cell(jl)) cell(iseg) = cell(jl)
                  if (cell(jl).ne.-1) kl = kl+1
                  jl=jl + 1
               enddo
               if (kl.gt.1) ierror = ierror +1
            enddo

!     check consistency
            call mntr_check_abort(map2d_id,ierror,
     $           'Too many local elements in section')

!     update number of elements to sort
            nsort = min(nseg,lwork)

         enddo                  ! icstep
!     global exchange and sort end

         ierror = 0
!     mark elements that can be mapped
         do il=1,nsort
            if (cell(il).ne.-1) then
!     check consistency; was this cell mapped previously
               if(MAP2D_GMAP(cell(il)).ne.-1) then
                  ierror = ierror +1
               endif
               MAP2D_GMAP(cell(il)) = nsorted + il
               MAP2D_OWN(cell(il))  = owner(il)
            endif
         enddo

!     check consistency
         call mntr_check_abort(map2d_id,ierror,
     $           'Element already assigned')

!     update number of sorted elements
         nsorted = nsorted + nsort

!     count local unmarked zones
         nseg = 0
         do il=1,lnseg
            if(MAP2D_GMAP(il).eq.-1) then
               nseg = nseg + 1
!     fill in coordinates and mark initial ownership
               if (nseg.le.lwork) then
                  call copy(ctrs(1,nseg),lctrs(1,il),lctrs1)
                  owner(nseg) = NID
                  cell(nseg) = il
               endif
            endif
         enddo

!     get global number of unmarked zones
         umrkgl = iglsum(nseg,1)

         if (umrkgl.eq.0) exit

!     update number of elements to sort
         nsort = min(nseg,lwork)

!     count global passes
         igpass = igpass +1

!     is igpass too big; something is wrong exit
         if (igpass.gt.igpass_max) then
            call mntr_abort(map2d_id,'Max iterations exceeded')
         endif

      enddo                     ! infinite loop

      return
      end subroutine
!=======================================================================
!> @brief Generate 2D mesh out of 3D one.
!! @ingroup map2d
      subroutine map2d_init_coord
      implicit none

      include 'SIZE'
      include 'MAP2D'

!     local variables
      integer len               ! buffer size
      integer il                ! loop index
      integer el                ! destination element
      integer imark(lelt)       ! element mark
      real rtmp(lx1,lz1,lelt,2) ! dummy arrays
      common /ctmp0/ rtmp

!#define DEBUG
#ifdef DEBUG
!     for testing
      character*2 str
      integer iunit
#endif
!-----------------------------------------------------------------------
      len = lx1*lz1
      call ifill(imark, -1,nelv)
      do il=1,nelv
         el = map2d_lmap(il)
         if (el.gt.0) then
            if (imark(el).eq.-1) then
               imark(el) = 1
               call copy(rtmp(1,1,el,1),map2d_xm1(1,1,il),len)
               call copy(rtmp(1,1,el,2),map2d_ym1(1,1,il),len)
            endif
         endif
      enddo

!     copy arrays back
      len = len*map2d_lnum
      call copy(map2d_xm1,rtmp(1,1,1,1),len)
      call copy(map2d_ym1,rtmp(1,1,1,2),len)

#ifdef DEBUG
!     testing
      write(str,'(i2.2)') nid
      call io_file_freeid(iunit, ierr)
      open(unit=iunit,file='map2d_init_coord.txt'//str)
      write(iunit,*) nid, nelv, map2d_idir, map2d_lnum
      do el=1,nelv
         write(iunit,*) 'Element ', el
         do jl=1,nz1
            do il=1,nx1
               write(iunit,*) il,jl,map2d_xm1(il,jl,el),
     $              map2d_ym1(il,jl,el)
            enddo
         enddo
      enddo
      close(iunit)
#endif
#undef DEBUG
      return
      end subroutine
!======================================================================
