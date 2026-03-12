!=======================================================================
c     Adam Peplinski; 2015.10.20
c     VERSION FOR DNS AND LINEAR SIMULATIONS
c     This version supports only proper restart for DNS and for
c     perturbation mode with single(!!!!!) NPERT=1
c     perturbation and not advected basefield IFBASE=F. For DNS only
c     'rs8' files are written. For perturbation mode  files 'rs8'
c     include perturbation data and the basefield is saved in 'rb8'.
c-----------------------------------------------------------------------
c
c     full-restart routines, called from userchk
c
c     saving and reading necessary files
c
c     Parameters used by this set of subroutines:
!     CHKPOINT:  
!     IFCHKPTRST - if restart
!     CHKPTSTEP - checkpiont dump frequency (number of time steps)
c
c     PARAM(66) - write format
c     PARAM(67) - read format
c
c     In the case of multistep time-integration method one needs data 
c     from NBDINP timestep. I use standard full_restart and 
c     full_restart_save subroutines. There are four 'rs8...' reatart 
c     files saved in double precission. Only .f format is supported. 
c     In any case two sets of restart files (1-4;5-8) are created.
c
c     NOTICE!!!!
c     To make this varsion to work correctly
c     mod(total_number_of_steps,IOSTEPs).ge.NBDINP
c     Otherwise last checkpoint will not be complete.
c
!=======================================================================
!***********************************************************************
!     read parameters checkpoint
      subroutine chkpt_param_in(fid)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            !
cc MA:      include 'PARALLEL_DEF' 
      include 'PARALLEL'        ! ISIZE, WDSIZE, LSIZE,CSIZE
      include 'CHKPOINT'

!     argument list
      integer fid               ! file id

!     local variables
      integer ierr

!     namelists
      namelist /CHKPOINT/ CHKPTSTEP, IFCHKPTRST
!-----------------------------------------------------------------------
!     default values
! ! !       CHKPTSTEP = 100
! ! !       IFCHKPTRST = .FALSE.
!     read the file
      ierr=0
! ! !       if (NID.eq.0) then
! ! !          read(unit=fid,nml=CHKPOINT,iostat=ierr)
! ! !       endif
! ! !       call err_chk(ierr,'Error reading CHKPOINT parameters.$')

!     broadcast data
! ! !       call bcast(CHKPTSTEP,ISIZE)
! ! !       call bcast(IFCHKPTRST,LSIZE)

      return
      end
!***********************************************************************
!     write parameters checkpoint
      subroutine chkpt_param_out(fid)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            !
      include 'CHKPOINT'

!     argument list
      integer fid               ! file id

!     local variables
      integer ierr

!     namelists
      namelist /CHKPOINT/ CHKPTSTEP, IFCHKPTRST
!-----------------------------------------------------------------------
      ierr=0
      if (NID.eq.0) then
         write(unit=fid,nml=CHKPOINT,iostat=ierr)
      endif
      call err_chk(ierr,'Error writing CHKPOINT parameters.$')

      return
      end
!***********************************************************************
!     main checkpoint interface
      subroutine checkpoint
      implicit none
!-----------------------------------------------------------------------

      call checkpoint_init

      call checkpoint_IO

      return
      end
!***********************************************************************
!     initialise checkpoint
      subroutine checkpoint_init
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            ! NID, NPERT
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'           ! IOSTEP, ISTEP
cc MA:      include 'INPUT_DEF'
      include 'INPUT'           ! SESSION, IFPERT, IFBASE
!      include 'RESTART_DEF'
!      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'        ! ISIZE
      include 'CHKPOINT'        ! CHKPTSTEP, IFCHKPTRST, CHKPTSET_O, CHKPTSET_I, CHKPTNRSF, CHKPTNFILE, 

!     local variables
      integer len, k, i, ierr

      character*132 fname, bname

      character*3 prefix

      character*6  str

      character*17 kst
      save         kst
      data         kst / '0123456789abcdefx' /

      integer icalld
      save    icalld
      data    icalld  /0/

      integer iunit

!     functions
      integer ltrunc
!-----------------------------------------------------------------------
c     this is done only once
      if (icalld.eq.0) then
         icalld = icalld + 1

!     check checkpoint frequency
         if (CHKPTSTEP.le.0) then
            if (NIO.eq.0) write (*,*) 'Warning; checkpoint_init: ',
     $           ' CHKPTSTEP = 0; resetting to ',NSTEPS-CHKPTNRSF+1
            CHKPTSTEP = NSTEPS-CHKPTNRSF+1
         elseif(mod(NSTEPS,CHKPTSTEP).ne.(CHKPTNRSF-1)) then
            if (NIO.eq.0) write (*,*) 'Warning; checkpoint_init: ',
     $           ' CHKPTSTEP and NSTEPS not optimal'
         endif

!     set negetive value of chkptset_i to mark that restart was not 
!     initialised yet
         CHKPTSET_I = -1

c     check perturbation parameters
         if (IFPERT) then
            if (IFBASE.or.NPERT.gt.1.or.IFMHD) then
               if(NIO.eq.0)
     $              write(6,*) 'CHECKPOINT: not supported mode'
               call exitt
            endif
         else                   ! IFPERT
            if (IFMHD) then
               if(NIO.eq.0)
     $              write(6,*) 'CHECKPOINT: not supported mode'
               call exitt
            endif
         endif                  ! IFPERT

c     create chkptrstf name (SESSION.restart)
         call blank(fname,132)
         call blank(bname,132)
         call blank(chkptrstf,80)

         k = 1
         len = ltrunc(SESSION,132) !  Add SESSION
         fname(k:k+len-1)=SESSION(1:len)
         k = k+len
         fname(k:k+7)='.restart'
         k = k+8


         call chcopy(chkptrstf,fname,k)
         

c     create names acording to mfo_open_files
         if (IFCHKPTRST) then 

!     create names of restart files
!     get set number from the file SESSION.restart
            ierr=0
            if(NID.eq.0) then
!     find free unit
               call IO_freeid(iunit, ierr)
!     get file name and open file
               if(ierr.eq.0) then
                  open (unit=iunit,file=chkptrstf,status='old',
     $                 action='read',iostat=ierr)
                  if(ierr.eq.0) read(unit=iunit,fmt=*,iostat=ierr) len
                  close(unit=iunit,iostat=ierr)
               endif
               if(ierr.eq.0.and.len.ne.0.and.len.ne.1) ierr=1

            endif
            call err_chk(ierr,'Error reading .restart file.$')

            call bcast(len ,ISIZE)
            CHKPTSET_I = len
            
c     create file names
            len = ltrunc(SESSION,132) !  Add SESSION
            
            
            call chcopy(bname,SESSION,len)           

c     fill chkptfname array with 'rb8' and 'rs' file names
            if (IFPERT) then
c     assumes only single perturbation
               CHKPTNFILE = 2
               

c     prefix and name for basefield
               prefix(1:2)='rb'
               len=min(17,2*CHKPTNRSF)
               len= len+1
               prefix(3:3)=kst(len:len)

               call IO_mfo_fname(prefix,fname,bname,k)

               
c     is fname too long?
               if ((k+5).gt.80) then
                  if(NIO.eq.0) write(6,*)
     $                 'ERROR; checkpoint: too long file name'
                  call exitt
               endif

               do i=1,CHKPTNRSF
                  call blank(CHKPTFNAME(i,1),80)
c     this assume IFBASE=F
                  write(str,54) 1
c                     write(str,54) CHKPTNRSF*CHKPTSET_I+i
 54               format(i5.5)
                  fname(k:k+4)=str(1:5)
                  call chcopy(CHKPTFNAME(i,1),fname,k+5)
c                     if (NID.eq.0) write(6,*) CHKPTFNAME(i,1)
               enddo

c     create prefix and name for perturbation
               prefix(1:2)='rs'
               len=min(17,2*CHKPTNRSF)
               len= len+1
               prefix(3:3)=kst(len:len)
               call IO_mfo_fname(prefix,fname,bname,k)

c     is fname too long?
               if ((k+5).gt.80) then
                  if(NID.eq.0) write(6,*)
     $                 'checkpoint: too long file name'
                  call exitt
               endif

               do i=1,CHKPTNRSF
                  call blank(CHKPTFNAME(i,2),80)
                  write(str,54) CHKPTNRSF*CHKPTSET_I+i
                  fname(k:k+4)=str(1:5)
                  call chcopy(CHKPTFNAME(i,2),fname,k+5)
c                     if (NID.eq.0) write(6,*) CHKPTFNAME(i,2)
               enddo
            else                ! DNS
               CHKPTNFILE = 1

c     create prefix and name for DNS
               prefix(1:2)='rs'
               len=min(17,2*CHKPTNRSF)
               len= len+1
               prefix(3:3)=kst(len:len)
               call IO_mfo_fname(prefix,fname,bname,k)

c     is fname too long?
               if ((k+5).gt.80) then
                  if(NIO.eq.0) write(6,*)
     $                 'checkpoint: too long file name'
                  call exitt
               endif

               do i=1,CHKPTNRSF
                  call blank(CHKPTFNAME(i,1),80)
                  write(str,54) CHKPTNRSF*CHKPTSET_I+i
                  fname(k:k+4)=str(1:5)
                  call chcopy(CHKPTFNAME(i,1),fname,k+5)
c                     if (NIO.eq.0) write(6,*) CHKPTFNAME(i,1)
               enddo
            endif

         endif                  ! IFCHKPTRST

!     set initial file set number; it is different from chkptset_i 
!     because nek5000 always starts from 1
         CHKPTSET_O = 1

      endif                     ! icalld

      return
      end
c***********************************************************************
!     write and read from disc
      subroutine checkpoint_IO()
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            ! NID, NPERT
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'           ! ISTEP
cc MA:      include 'INPUT_DEF'
      include 'INPUT'           ! IFREGUO
      include 'CHKPOINT'        ! CHKPTSTEP, IFCHKPTRST, CHKPTSET_O, CHKPTSET_I, CHKPTNRSF, CHKPTNFILE

!     local variables
      integer ierr, iotest, iunit

      logical lifreguo
!-----------------------------------------------------------------------
c     set some parameters
      lifreguo= IFREGUO
      IFREGUO = .false.

      if (IFCHKPTRST.and.(ISTEP.lt.CHKPTNRSF))
     $     call checkpoint_pert()

c      CHKPTSTEP = IOSTEP          ! Trigger save based on iostep
      call checkpoint_save_pert(CHKPTSTEP)

c     save file set in SESSION.restart
      ierr=0
      if(NID.eq.0) then
         iotest = 0
         if (ISTEP.gt.CHKPTSTEP/2.and.
     $     mod(ISTEP+CHKPTSTEP-iotest,CHKPTSTEP).eq.(CHKPTNRSF-1)) then

            if(CHKPTSET_O.eq.0) then
               CHKPTSET_O=1
            else
               CHKPTSET_O=0
            endif

!     find free unit
            call IO_freeid(iunit, ierr)
!     get file name and open file
            if(ierr.eq.0) then
               open (unit=iunit,file=CHKPTRSTF,
     $              action='write',iostat=ierr)
               if(ierr.eq.0)
     $              write(unit=iunit,fmt=*,iostat=ierr) CHKPTSET_O
               close(unit=iunit,iostat=ierr)
            endif
         endif                  ! ISTEP
      endif                     ! NID
      call err_chk(ierr,'Error writing to .restart file.$')

c     put parameters back
      IFREGUO = lifreguo

      return
      end
c***********************************************************************
c     VERSION FOR PERTURBATION MODE
c     following two subroutines are modiffications of 
c     full_restart_save
c     restart_save
c     from prepost.f.
c     This version supports only proper restart for single(!!!!!)
c     perturbation with not advected basefield IFBASE=F. In this case
c     files rs8 include perturbation and the basefield is saved in rb8.
c     The more general case (npert/=1 and IFBASE/=F) requeres increse
c     of file numbers to be written and e.g. hanges in restart_nfld.
      subroutine checkpoint_save_pert(iosave)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            ! NPERT, NID
cc MA:      include 'INPUT_DEF'
      include 'INPUT'           ! IFPERT, IFBASE

!     rgument list
      integer iosave

!     local variables
      integer save_size,nfld_save
!-----------------------------------------------------------------------
c     check perturbation parameters
      if ((IFPERT.and.(IFBASE.or.NPERT.gt.1)).or.IFMHD) then
         if(NID.eq.0) write(6,*) 'CHECKPOINT: not supported mode'
         call exitt
      endif

c     there is some problem with nfld_save; sometimes it would be good
c     to increase its value, but restart_nfld doesn't allow for that
      nfld_save=4  ! For checkpoint
      save_size=8  ! For checkpoint

      call restart_save_pert(iosave,save_size,nfld_save)

      return
      end
c***********************************************************************
c     Save current fields for later restart.
c
c     Input arguments:
c
c       .iosave plays the usual triggering role, like iostep
c
c       .save_size = 8 ==> dbl. precision output
c
c       .nfldi is the number of rs files to save before overwriting
      subroutine restart_save_pert(iosave,save_size,nfldi)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'

!     argument list
      integer iosave,save_size,nfldi

!     local variables
      character*3 prefix, bprefix

      character*17 kst
      save         kst
      data         kst / '0123456789abcdefx' /

      logical if_full_pres_tmp, ifxyo_tmp

      integer icalld
      save    icalld
      data    icalld  /0/

      integer i2, iosav, iotest, iwdsizo, m1, mfld, mt
      integer nfld, nfld2, npscal1
      real p66
!-----------------------------------------------------------------------
      iosav = iosave

      if (iosav.eq.0) iosav = iostep
      if (iosav.eq.0) return

      iotest = 0
c     if (iosav.eq.iostep) iotest = 1  ! currently spoiled because of 
c                                      ! incompatible format of .fld
c                                      ! and multi-file i/o;  the latter
c                                      ! is the only form used for restart

      nfld  = nfldi*2
      nfld2 = nfld/2
      mfld  = min(17,nfld)
c     this is not supported; only second order time integration for MHD?
c     Why only 2 files per set not 3?
c      if (ifmhd) nfld2 = nfld/4

      i2 = iosav/2
      m1 = istep+iosav-iotest
      mt = mod(istep+iosav-iotest,iosav)
      prefix = '   '
      bprefix = '   '

      if (istep.gt.iosav/2  .and.
     $   mod(istep+iosav-iotest,iosav).lt.nfld2) then ! save
         prefix(1:2)='rs'
         prefix(3:3)=kst(mfld+1:mfld+1)
         bprefix(1:2)='rb'
         bprefix(3:3)=kst(mfld+1:mfld+1)

         iwdsizo = wdsizo
         wdsizo  = save_size
         p66 = param(66)
         param(66) = 6          ! force multi-file out
         ifxyo_tmp = IFXYO
         IFXYO = .TRUE.         ! force writing coordinates

         npscal1 = npscal+1
         if (.not.ifheat) npscal1 = 0

         if_full_pres_tmp = if_full_pres
         if (save_size.eq.8) if_full_pres = .true. !Preserve mesh 2 pressure

         if (IFPERT) then

c     save basefiled
c     do this only once
            if (icalld.eq.0) then
               icalld=1
               call outpost2(vx,vy,vz,pr,t,npscal1,bprefix)
            endif

c     save perturbation
            call outpost2(vxp(1,1),vyp(1,1),vzp(1,1),prp(1,1),tp(1,1,1)
     $           ,npscal1,prefix) ! perturbation
         else                   ! DNS
            call outpost2(vx,vy,vz,pr,t,npscal1,prefix) ! basefield
         endif

         wdsizo    = iwdsizo  ! Restore output parameters
         param(66) = p66
         IFXYO = ifxyo_tmp
         if_full_pres = if_full_pres_tmp

      endif

c     if (nid.eq.0) write(6,8) istep,prefix,nfld,nfld2,i2,m1,mt
c  8  format(i8,' prefix ',a3,5i5)

      if_full_pres = .false.
      return
      end
c***********************************************************************
c     VERSION FOR PERTURBATION MODE
c     following two subroutines are modiffications of 
c     full_restart
c     restart
c     mfi
c     map_pm1_to_pr
c     from ic.f.
c***********************************************************************
      subroutine checkpoint_pert()
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'CHKPOINT'        ! CHKPTFNAME,CHKPTNRSF,CHKPTNFILE

!     local variables
      integer ifile, i
      real p67
!-----------------------------------------------------------------------
      ifile = istep+1  ! istep=0,1,...

      if (ifile.le.CHKPTNRSF) then
         p67 = param(67)
         param(67) = 6.00
         do i=1,CHKPTNFILE
            call chcopy (initc(i),CHKPTFNAME(ifile,i),80)
         enddo
         call bcast  (initc,132*CHKPTNFILE)

         if (IFPERT) then       ! perturbation
c     for IFBASE=F basefield loaded during istep=0 only
c     perturbation from 'rs' files in all steps
            if (IFBASE) then
               call restart_pert(CHKPTNFILE,1)
            elseif (ifile.eq.1) then
               call restart_pert(CHKPTNFILE,1)
            else
               call restart_pert(CHKPTNFILE,2)
            endif
         elseif (IFMHD) then    ! MHD
            call restart(2)
         else                   ! DNS
            call restart(1)
         endif                  ! IFPERT

         param(67)=p67
      endif
   
      return
      end
c***********************************************************************
c     this version supports .f (param(67).eq.6.0) format only
c***********************************************************************
C
C     (1) Open restart file(s)
C     (2) Check previous spatial discretization 
C     (3) Map (K1,N1) => (K2,N2) if necessary
C
C     nfiles > 1 has several implications:
C
C     i.   For std. run, data is taken from last file in list, unless
C          explicitly specified in argument list of filename
C
C     ii.  For MHD and perturbation cases, 1st file is for U,P,T;
C          subsequent files are for B-field or perturbation fields
C
C
C     Adam Peplinski
C     There is additional input parameter specifying starting number
C     for file loop. It can be useful for perturbation mode with not
C     advected baseflow.

      subroutine restart_pert(nfiles,nfilstart)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'            ! NID
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'           ! TIME
cc MA:      include 'INPUT_DEF'
      include 'INPUT'           ! PARAM
cc MA:      include 'RESTART_DEF'
      include 'RESTART'

!     argument list
      integer nfiles, nfilstart

!     local variables
      logical ifbasefl

      integer ifile,ndumps

      character*132 fname
!     functions
      real glmax
C-----------------------------------------------------------------------
      if(nfiles.lt.1) return
      if(NID.eq.0) write(6,*) 'Reading checkpoint data'

c use new reader (only binary support)
      if (PARAM(67).eq.6.0) then
         do ifile=nfilstart,nfiles
            call sioflag(ndumps,fname,initc(ifile))
            call mfi_pert(fname,ifile)
         enddo
         call setup_convect(3)
         if (nid.ne.0) time=0
         time = glmax(time,1) ! Sync time across processors
         return
      else
         if (NIO.eq.0) write(6,*) 'RESTART_PERT supporst .f only'
         call exitt
      endif

      return
      end
c***********************************************************************
c
c     (1) Open restart file(s)
c     (2) Check previous spatial discretization 
c     (3) Map (K1,N1) => (K2,N2) if necessary
c
c     nfiles > 1 has several implications:
c
c     i.   For std. run, data is taken from last file in list, unless
c          explicitly specified in argument list of filename
c
c     ii.  For MHD and perturbation cases, 1st file is for U,P,T;
c          subsequent files are for B-field or perturbation fields
      subroutine mfi_pert(fname,ifile)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'
cc MA:      include 'GEOM_DEF'
      include 'GEOM'

!     ragiment list
      character*132  fname
      integer ifile

!     local variables
      character*132 hdr
      logical if_full_pres_tmp

      integer lwk
      parameter (lwk = 7*lx1*ly1*lz1*lelt)
      real wk(lwk), pm1(lx1*ly1*lz1,lelv)
      common /scrns/ wk
      common /scrcg/ pm1
      integer e, j, k, ierr

      integer*8 offs0,offs,nbyte,stride,strideB,nxyzr8
      integer iofldsr
      real tiostart, tio, dnbyte
!     functions
      real dnekclock, glsum
c-----------------------------------------------------------------------
      tiostart=dnekclock()

      call mfi_prepare(fname)       ! determine reader nodes +
                                    ! read hdr + element mapping 

      offs0   = iHeadersize + 4 + isize*nelgr
      nxyzr8  = nxr*nyr*nzr
      strideB = nelBr* nxyzr8*wdsizr
      stride  = nelgr* nxyzr8*wdsizr

      if_full_pres_tmp = if_full_pres
      if (wdsizr.eq.8) if_full_pres = .true. !Preserve mesh 2 pressure

      iofldsr = 0
      if (ifgetxr) then      ! if available
         offs = offs0 + ndim*strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifgetx) then
c            if(nid.eq.0) write(6,*) 'Reading mesh'
            call mfi_getv(xm1,ym1,zm1,wk,lwk,.false.)
         else                ! skip the data
            call mfi_getv(xm1,ym1,zm1,wk,lwk,.true.)
         endif
         iofldsr = iofldsr + ndim
      endif

      if (ifgetur) then
         offs = offs0 + iofldsr*stride + ndim*strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifgetu) then
c     MHD
            if (ifmhd.and.ifile.eq.2) then
c               if(nid.eq.0) write(6,*) 'Reading B field'
               call mfi_getv(bx,by,bz,wk,lwk,.false.)
c     perturbation mode
c     importatn assumption; there are no MHD perturbation
            elseif (ifpert.and.ifile.ge.2) then
               j=ifile-1  ! pointer to perturbation field
               call mfi_getv(vxp(1,j),vyp(1,j),vzp(1,j),wk,lwk,.false.)
            else
c               if(nid.eq.0) write(6,*) 'Reading velocity field'
               call mfi_getv(vx,vy,vz,wk,lwk,.false.)
            endif
         else
            call mfi_getv(vx,vy,vz,wk,lwk,.true.)
         endif
         iofldsr = iofldsr + ndim
      endif

      if (ifgetpr) then
         offs = offs0 + iofldsr*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifgetp) then
c           if(nid.eq.0) write(6,*) 'Reading pressure field'
            call mfi_gets(pm1,wk,lwk,.false.)
         else
            call mfi_gets(pm1,wk,lwk,.true.)
         endif
         iofldsr = iofldsr + 1
      endif

      if (ifgettr) then
         offs = offs0 + iofldsr*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifgett) then
c     perturbation mode
            if (ifpert.and.ifile.ge.2) then
               j=ifile-1  ! pointer to perturbation field
               call mfi_gets(tp(1,1,j),wk,lwk,.false.)
            else
c               if(nid.eq.0) write(6,*) 'Reading temperature field'
               call mfi_gets(t,wk,lwk,.false.)
            endif
         else
            call mfi_gets(t,wk,lwk,.true.)
         endif
         iofldsr = iofldsr + 1
      endif

      do k=1,ldimt-1
         if (ifgtpsr(k)) then
            offs = offs0 + iofldsr*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            if (ifgtps(k)) then
c               if(nid.eq.0) write(6,'(A,I2,A)') ' Reading ps',k,' field'
c     perturbation mode
               if (ifpert.and.ifile.ge.2) then
                  j=ifile-1     ! pointer to perturbation field
                  call mfi_gets(tp(1,k+1,j),wk,lwk,.false.)
               else
                  call mfi_gets(t(1,1,1,1,k+1),wk,lwk,.false.)
               endif
            else
               call mfi_gets(t(1,1,1,1,k+1),wk,lwk,.true.)
            endif
            iofldsr = iofldsr + 1
         endif
      enddo
      nbyte = 0
      if(nid.eq.pid0r) nbyte = iofldsr*nelr*wdsizr*nxr*nyr*nzr

      if (ifgtim) time = timer

      ierr = 0

#ifdef MPIIO
      if (nid.eq.pid0r) call byte_close_mpi(ifh_mbyte,ierr)
#else
      if (nid.eq.pid0r) call byte_close(ierr)
#endif
      call err_chk(ierr,'Error closing restart file, in mfi_pert.$')
      tio = dnekclock()-tiostart

      dnbyte = nbyte
      nbyte = glsum(dnbyte,1)
      nbyte = nbyte + iHeaderSize + 4 + isize*nelgr

      if(NIO.eq.0) write(6,7) istep,time,
     &             nbyte/tio/1024/1024/10,
     &             nfiler
    7 format(/,i9,1pe12.4,' done :: Read checkpoint data',/,
     &       30X,'avg data-throughput = ',f7.1,'MBps',/,
     &       30X,'io-nodes = ',i5,/)

c     Adam Peplinski;  axis_interp_ic not modified
      if (ifaxis) call axis_interp_ic(pm1)      ! Interpolate to axi mesh
      if (ifgetp) call map_pm1_to_pr_pert(pm1,ifile) ! Interpolate pressure

      if_full_pres = if_full_pres_tmp

      return
      end
c***********************************************************************
      subroutine map_pm1_to_pr_pert(pm1,ifile)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'

!     argument list
      real pm1(lx1*ly1*lz1,lelv)
      integer ifile

!     local variables
      logical if_full_pres_tmp

      integer e, nxyz2, j, ie2
!-----------------------------------------------------------------------
      nxyz2 = nx2*ny2*nz2

      if (ifmhd.and.ifile.eq.2) then
         do e=1,nelv
            if (if_full_pres) then
               call copy  (pm(1,1,1,e),pm1(1,e),nxyz2)
            else
               call map12 (pm(1,1,1,e),pm1(1,e),e)
            endif
         enddo
      elseif (ifpert.and.ifile.ge.2) then
         j=ifile-1     ! pointer to perturbation field
         if (ifsplit) then
            call copy (prp(1,j),pm1,nx1*ny1*nz1*nelv)
         else
            do e=1,nelv
               ie2 = (e-1)*nxyz2+1
               if (if_full_pres) then
                  call copy(prp(ie2,j),pm1(1,e),nxyz2)
               else
                  call map12(prp(ie2,j),pm1(1,e),e)
               endif
            enddo
         endif
      elseif (ifsplit) then
         call copy (pr,pm1,nx1*ny1*nz1*nelv)
      else
         do e=1,nelv
            if (if_full_pres) then
               call copy  (pr(1,1,1,e),pm1(1,e),nxyz2)
            else
               call map12 (pr(1,1,1,e),pm1(1,e),e)
            endif
         enddo
      endif
   
      return
      end
c***********************************************************************
