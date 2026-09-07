c-----------------------------------------------------------------------
      subroutine readat
C
C     Read in data from preprocessor input file (.rea)
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'ZPER'
 
      logical ifbswap,ifre2,parfound
      character*132 string
      integer idum(3*numsts+3)

      ierr = 0
      call flush_io

      ! check if new rea file version exists
      if(nid.eq.0) inquire(file=parfle, exist=parfound)
      call bcast(parfound,lsize)
      if (parfound) then
         if(nio.eq.0) write(6,'(A,A)') ' Reading ', parfle
         call readat_new
         return
      endif  

      if(nid.eq.0) then
        write(6,'(A,A)') ' Reading ', reafle
        open (unit=9,file=reafle,status='old', iostat=ierr)
      endif

      call bcast(ierr,isize)
      if (ierr .gt. 0) call exitti('Cannot open .rea file!$',1)

C     Read parameters and logical flags
      call rdparam

C     Read Mesh Info 
      if(nid.eq.0) then
        read(9,*)    ! xfac,yfac,xzero,yzero
        read(9,*)    ! dummy
        if (ifmoab) then
           read(9,*) h5mfle
           ! read fluid/solid material set ids
           read(9,*) numflu, numoth
           if (numflu+numoth .gt. numsts) then
              write(6,'(A)') 
     $             'Number of fluid+other material sets too large.'
              write(6, '(A)') 
     $             'Need to increase NUMSTS in file INPUT.'
              call exitt
           else if (numoth .gt. 0 .and. .not. ifheat) then 
              call exitt(
     $       'Error: no. of other sets is non-zero but ifheat = false.')
           endif
           read(9,*) (matids(i), i = 1, numflu+numoth)
           do i = numflu+numoth+1, numsts
              matids(i) = -1
           enddo
           read(9,*) (matindx(i), i = 1, numflu+numoth)
           do i = numflu+numoth+1, numsts
              matindx(i) = -1
           enddo
           do i = 1, lelt
              imatie(i) = -1
           enddo
           read(9,*) numbcs
           if (numbcs .gt. numsts) then
              write(6,'(A)') 
     $             'Number of BC sets too large.'
              write(6, '(A)') 
     $             'Need to increase NUMSTS in file INPUT.'
              call exitti
           endif
           do iset = 1, numbcs
              read(9,'(2I5,A3)') ibcsts(iset), bcf(iset), bctyps(iset)

           enddo
           nelgs = 0
           do iset = numbcs+1, numsts
              bcf(iset) = -1
              bctyps(iset) = 'E  '
              ibcsts(iset) = -1
           enddo
        else
           read(9,*)  nelgs,ndim,nelgv
           nelgt = abs(nelgs)
        endif
      endif
      call bcast(nelgs,ISIZE)
      call bcast(ndim ,ISIZE)
      call bcast(nelgv,ISIZE)
      call bcast(nelgt,ISIZE)
      call bcast(h5mfle,132)
      if (ifmoab) then
c pack into long int array and bcast as that
         if (nid .eq. 0) then
            idum(1) = numflu
            idum(2) = numoth
            idum(3) = numbcs
            do iset = 1, numsts
               idum(3+iset) = matids(iset)
            enddo
            do iset = 1, numsts
               idum(3+numflu+numoth+iset) = ibcsts(iset)
            enddo
            do iset = 1, numsts
               idum(3+numflu+numoth+numbcs+iset) = matindx(iset)
            enddo
         endif
         call bcast(idum, ISIZE*(3+3*numsts))
         call bcast(bctyps, 3*numsts)
         call bcast(bcf, ISIZE*numsts)

         if (nid .ne. 0) then
            numflu = idum(1)
            numoth = idum(2)
            numbcs = idum(3)
            do iset = 1, numsts
               matids(iset) = idum(3+iset)
            enddo
            do iset = 1, numsts
               ibcsts(iset) = idum(3+numflu+numoth+iset)
            enddo
            do iset = 1, numsts
               matindx(iset) = idum(3+numflu+numoth+numbcs+iset)
            enddo
         endif
      endif
      ifre2 = .false.
      if(nelgs.lt.0) ifre2 = .true.     ! use new .re2 reader

      ifgtp = .false.
      if (ndim.lt.0) ifgtp = .true.     ! domain is a global tensor product

      if (ifmoab) then
#ifdef MOAB
         call nekMOAB_import
#endif
      else
        if (ifre2) call open_bin_file(ifbswap) ! rank0 will open and read
        call chk_nel  ! make certain sufficient array sizes

        if (.not.ifgtp) call mapelpr  ! read .map file, est. gllnid, etc.
        if (ifre2) then
          call bin_rd1(ifbswap) ! rank0 will read mesh data + distribute
        else
          maxrd = 32               ! max # procs to read at once
          mread = (np-1)/maxrd+1   ! mod param
          iread = 0                ! mod param
          x     = 0
          do i=0,np-1,maxrd
             call nekgsync()
             if (mod(nid,mread).eq.iread) then
                if (nid.ne.0) then
                  open(UNIT=9,FILE=REAFLE,STATUS='OLD')
                  call cscan(string,'MESH DATA',9)
                  read(9,*) string
                endif 
                if (ifgtp) then
                   call genbox
                else
                   call rdmesh
                   call rdcurve !  Curved side data
                   call rdbdry  !  Boundary Conditions
                endif
                if (nid.ne.0) close(unit=9)
             endif
             iread = iread + 1
          enddo
        endif
      endif

C     Read Restart options / Initial Conditions / Drive Force
      CALL RDICDF
C     Read materials property data
      CALL RDMATP
C     Read history data
      CALL RDHIST
C     Read output specs
      CALL RDOUT
C     Read objects
      CALL RDOBJ

      call nekgsync()

C     End of input data, close read file.
      if(nid.eq.0) then
        close(unit=9)
        call echopar
      endif

c     This is not an excellent place for this check, but will
c     suffice for now.   5/6/10
      if (ifchar.and.(nelgv.ne.nelgt)) call exitti(
     $ 'ABORT: IFCHAR curr. not supported w/ conj. ht transfer$',nelgv)

      return
      END
c-----------------------------------------------------------------------
      subroutine rdparam
C
C     .Read in parameters supplied by preprocessor and
C      (eventually) echo check.
C
C     .Broadcast run parameters to all processors
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
      INCLUDE 'ZPER'

      character*132 string(100)

      VNEKTON = 3 ! dummy not really used anymore

      optlevel = 1! fixed for now
      loglevel = 1! fixed for now
      
      IF(NID.EQ.0) THEN
        READ(9,*,ERR=400)
        READ(9,*,ERR=400)
        READ(9,*,ERR=400) NDIM
        READ(9,*,ERR=400) NPARAM
        DO 20 I=1,NPARAM
           READ(9,*,ERR=400) PARAM(I)
   20   CONTINUE
      ENDIF
      call bcast(NDIM  ,ISIZE)
      call bcast(NPARAM,ISIZE)
      call bcast(PARAM ,200*WDSIZE)

      NPSCAL=INT(PARAM(23))
      NPSCL1=NPSCAL+1
      NPSCL2=NPSCAL+2

      IF (NPSCL1.GT.LDIMT) THEN
         if(nid.eq.0) then
           WRITE(6,21) LDIMT,NPSCL1
   21      FORMAT(//,2X,'Error: This NEKTON Solver has been compiled'
     $             /,2X,'       for',I4,' passive scalars.  This run'
     $             /,2X,'       requires that LDIMT be set to',I4,'.')
         endif
         call exitt
      ENDIF
   
c   
c     Read in the passive scalar conduct and rhocp's:
c
c     fluid
c                 .viscosity is PARAM(2)
c                 .if it is negative, it indicates that Re has been input
c                 .therefore, redefine PARAM(2) = -1.0/PARAM(2)
c
      if(param(2) .lt.0.0) param(2)  = -1.0/param(2)
      if(param(8) .lt.0.0) param(8)  = -1.0/param(8)
      if(param(29).lt.0.0) param(29) = -1.0/param(29)
C
      CPFLD(1,1)=PARAM(2)
      CPFLD(1,2)=PARAM(1)
C     temperature
      CPFLD(2,1)=PARAM(8)
      CPFLD(2,2)=PARAM(7)
      CPFLD(2,3)=PARAM(9)
c
c     passive scalars
c
      IF(NID.EQ.0) THEN
        READ(9,*,ERR=400) NSKIP
        IF (NSKIP.GT.0 .AND. NPSCAL.GT.0) THEN
           READ(9,*,ERR=400)(CPFLD(I,1),I=3,NPSCL2)
           IF(NPSCL2.LT.9)READ(9,*)
           READ(9,*,ERR=400)(CPFLD(I,2),I=3,NPSCL2)
           IF(NPSCL2.LT.9)READ(9,*)
           do i=3,npscl2
              if (cpfld(i,1).lt.0) cpfld(i,1) = -1./cpfld(i,1)
              if (cpfld(i,2).lt.0) cpfld(i,2) = -1./cpfld(i,2)
           enddo
        ELSE
           DO 25 I=1,NSKIP
              READ(9,*,ERR=500)
   25         CONTINUE
        ENDIF
      ENDIF
      call bcast(cpfld,WDSIZE*LDIMT1*3)

C
C     Read logical equation type descriptors....
C
      IFTMSH(0) = .false.
      do i=1,NPSCL2
         IFTMSH(i) = .false.
         IFADVC(i) = .false. 
         IFDIFF(i) = .true. 
      enddo      

      do i=1,NPSCL1
         IDPSS(i) = 0 ! use Helmholtz for passive scalars 
      enddo

      IFFLOW    = .false.
      IFHEAT    = .false.
      IFTRAN    = .false.
      IFAXIS    = .false.
      IFAZIV    = .false.
      IFSTRS    = .false.
      IFLOMACH  = .false.
      IFMODEL   = .false.
      IFKEPS    = .false.
      IFMVBD    = .false.
      IFCHAR    = .false.
      IFDG      = .false.
      IFANLS    = .false.
      IFMOAB    = .false.
      IFCOUP    = .false.
      IFVCOUP   = .false.
      IFMHD     = .false.
      IFESSR    = .false.
      IFTMSH(0) = .false.
      IFUSERVP  = .false.
      IFCONS    = .false.    ! Use conservation form?
      IFUSERMV  = .false.
      IFCYCLIC  = .false.
      IFSYNC    = .false.
      IFEXPLVIS = .false.
      IFSCHCLOB = .false.
c     IFSPLIT   = .false.

      ifbase = .true.
      ifpert = .false.


      IF(NID.EQ.0) READ(9,*,ERR=500) NLOGIC
      call bcast(NLOGIC,ISIZE)
      IF(NLOGIC.GT.100) THEN
          if(nid.eq.0)
     $       write(6,*) 'ABORT: Too many logical switches', NLOGIC
          call exitt
      ENDIF

      if(nid.eq.0) READ(9,'(A132)',ERR=500) (string(i),i=1,NLOGIC)
      call bcast(string,100*132*CSIZE)

      do i = 1,NLOGIC
         call capit(string(i),132)
         if (indx1(string(i),'IFTMSH' ,6).gt.0) then 
             read(string(i),*,ERR=490) (IFTMSH(II),II=0,NPSCL2)
         elseif (indx1(string(i),'IFNAV'  ,5).gt.0 .and.
     &           indx1(string(i),'IFADVC' ,6).gt.0) then 
              read(string(i),*,ERR=490) (IFADVC(II),II=1,NPSCL2)
         elseif (indx1(string(i),'IFADVC' ,6).gt.0) then
              read(string(i),*,ERR=490) (IFADVC(II),II=1,NPSCL2)
         elseif (indx1(string(i),'IFFLOW' ,6).gt.0) then
              read(string(i),*) IFFLOW
         elseif (indx1(string(i),'IFHEAT' ,6).gt.0) then 
              read(string(i),*) IFHEAT
         elseif (indx1(string(i),'IFTRAN' ,6).gt.0) then 
              read(string(i),*) IFTRAN
         elseif (indx1(string(i),'IFAXIS' ,6).gt.0) then 
              read(string(i),*) IFAXIS
         elseif (indx1(string(i),'IFAZIV' ,6).gt.0) then 
              read(string(i),*) IFAZIV
         elseif (indx1(string(i),'IFSTRS' ,6).gt.0) then 
              read(string(i),*) IFSTRS
         elseif (indx1(string(i),'IFLO'   ,4).gt.0) then 
              read(string(i),*) IFLOMACH
         elseif (indx1(string(i),'IFMGRID',7).gt.0) then 
c             read(string(i),*) IFMGRID
         elseif (indx1(string(i),'IFKEPS' ,6).gt.0) then 
              read(string(i),*) IFKEPS
         elseif (indx1(string(i),'IFMODEL',7).gt.0) then 
              read(string(i),*) IFMODEL
         elseif (indx1(string(i),'IFMVBD' ,6).gt.0) then 
              read(string(i),*) IFMVBD
         elseif (indx1(string(i),'IFCHAR' ,6).gt.0) then 
              read(string(i),*) IFCHAR
         elseif (indx1(string(i),'IFDG'   ,4).gt.0) then 
              read(string(i),*) IFDG
         elseif (indx1(string(i),'IFANLS' ,6).gt.0) then 
              read(string(i),*) IFANLS
         elseif (indx1(string(i),'IFMOAB' ,6).gt.0) then 
              read(string(i),*) IFMOAB
         elseif (indx1(string(i),'IFCOUP' ,6).gt.0) then 
              read(string(i),*) IFCOUP
         elseif (indx1(string(i),'IFVCOUP' ,7).gt.0) then 
              read(string(i),*) IFVCOUP
         elseif (indx1(string(i),'IFMHD'  ,5).gt.0) then 
              read(string(i),*) IFMHD
         elseif (indx1(string(i),'IFCONS' ,6).gt.0) then 
              read(string(i),*) IFCONS
         elseif (indx1(string(i),'IFUSERVP',8).gt.0) then 
              read(string(i),*) IFUSERVP
         elseif (indx1(string(i),'IFUSERMV',8).gt.0) then 
              read(string(i),*) IFUSERMV
         elseif (indx1(string(i),'IFCYCLIC',8).gt.0) then 
              read(string(i),*) IFCYCLIC
         elseif (indx1(string(i),'IFPERT'  ,6).gt.0) then 
              read(string(i),*) IFPERT
         elseif (indx1(string(i),'IFBASE'  ,6).gt.0) then 
              read(string(i),*) IFBASE
         elseif (indx1(string(i),'IFSYNC'  ,6).gt.0) then 
              read(string(i),*) IFSYNC
         elseif (indx1(string(i),'IFEXPLVIS',9).gt.0) then 
              read(string(i),*) IFEXPLVIS
         elseif (indx1(string(i),'IFSCHCLOB',9).gt.0) then 
              read(string(i),*) IFSCHCLOB
         elseif (indx1(string(i),'IFSPLIT' ,7).gt.0) then 
c              read(string,*) IFSPLIT
         else
              if(nid.eq.0) then
                write(6,'(1X,2A)') 'ABORT: Unknown logical flag', string
                write(6,'(30(A,/))') 
     &           ' Available logical flags:',
     &           '   IFTMSH'   ,
     &           '   IFADVC'   ,  
     &           '   IFFLOW'   ,
     &           '   IFHEAT'   ,
     &           '   IFTRAN'   ,
     &           '   IFAXIS'   ,
     &           '   IFCYCLIC' ,
     &           '   IFSTRS'   ,
     &           '   IFLOMACH' ,
     &           '   IFMGRID'  ,
     &           '   IFKEPS'   ,
     &           '   IFMVBD'   ,
     &           '   IFCHAR'   ,
     &           '   IFDG'     ,
     &           '   IFANLS'   ,
     &           '   IFUSERVP' ,
     &           '   IFUSERMV' ,
     &           '   IFSYNC'   ,
     &           '   IFCYCLIC' ,
     &           '   IFSPLIT'  ,
     &           '   IFEXPLVIS',
     &           '   IFCONS'   ,
     &           '   IFMOAB'   ,
     &           '   IFCOUP'   ,
     &           '   IFVCOUP'
              endif
              call exitt
         endif
 490  continue
      enddo

      ifmgrid   = .false.
      if (ifsplit) ifmgrid   = .true.

      if (ifaxis.and..not.ifsplit) then ! Use standard Schwarz/PCG solver
         ifmgrid   = .false.
         param(42) = 1.00000  !  p042 0=gmres/1=pcg
         param(43) = 1.00000  !  p043 0=semg/1=schwarz
         param(44) = 1.00000  !  p044 0=E-based/1=A-based prec.
      endif

      if (param(29).ne.0.) ifmhd  = .true.
      if (ifmhd)           ifessr = .true.
      if (ifmhd)           npscl1 = npscl1 + 1
      if (param(30).gt.0)  ifuservp = .true.
      if (param(31).ne.0.) ifpert = .true.
      if (param(31).lt.0.) ifbase = .false.   ! don't time adv base flow
      npert = abs(param(31)) 

      IF (NPSCL1.GT.LDIMT .AND. IFMHD) THEN
         if(nid.eq.0) then
           WRITE(6,22) LDIMT,NPSCL1
   22      FORMAT(/s,2X,'Error: This NEKTON Solver has been compiled'
     $             /,2X,'       for',I4,' passive scalars.  A MHD run'
     $             /,2X,'       requires that LDIMT be set to',I4,'.')
         endif
         call exitt
      ENDIF

      if (ifmvbd) then
         if (lx1.ne.lx1m.or.ly1.ne.ly1m.or.lz1.ne.lz1m) 
     $      call exitti('Need lx1m=lx1 etc. in SIZE . $',lx1m)
      endif

      ifldmhd = npscal + 3
      if (ifmhd) then
         cpfld(ifldmhd,1) = param(29)  ! magnetic viscosity
         cpfld(ifldmhd,2) = param( 1)  ! magnetic rho same as for fluid
      endif
C
C     Set up default time dependent coefficients - NSTEPS,DT.
C
      if (.not.iftran) then
         if (ifflow.and.ifsplit) then
            iftran=.true.
         else
            param(11) = 1.0
            param(12) = 1.0
            param(19) = 0.0
         endif
      endif
c
c     Check here for global fast diagonalization method or z-homogeneity.
c     This is here because it influence the mesh read, which follows.
      nelx   = abs(param(116))   ! check for global tensor-product structure
      nely   = abs(param(117))
      nelz   = abs(param(118))
      n_o    = 0

      if (n_o.eq.0) then
         ifzper=.false.
         ifgfdm=.false.
         if (nelz.gt.0) ifzper=.true.
         if (nelx.gt.0) ifgfdm=.true.
         if (nelx.gt.0) ifzper=.false.
      endif


C
C     Do some checks
C
      IF(NDIM.NE.LDIM)THEN
         IF(NID.EQ.0) THEN
           WRITE(6,10) LDIM,NDIM
   10      FORMAT(//,2X,'ERROR: This NEKTON Solver has been compiled'
     $             /,2X,'       for spatial dimension equal to',I2,'.'
     $             /,2X,'       The data file has dimension',I2,'.')
         ENDIF
         call exitt
      ENDIF
      IF (NDIM.EQ.3) IF3D=.TRUE.
      IF (NDIM.NE.3) IF3D=.FALSE.

      if (if3d) then
         if (ly1.ne.lx1.or.lz1.ne.lx1) then
            if (nid.eq.0) write(6,13) lx1,ly1,lz1
   13       format('ERROR: lx1,ly1,lz1:',3i5,' must be equal for 3D')
            call exitt
         endif
         if (ly2.ne.lx2.or.lz2.ne.lx2) then
            if (nid.eq.0) write(6,14) lx2,ly2,lz2
   14       format('ERROR: lx2,ly2,lz2:',3i5,' must be equal for 3D')
            call exitt
         endif
      else
         if (ly1.ne.lx1.or.lz1.ne.1) then
            if (nid.eq.0) write(6,12) lx1,ly1,lz1
   12       format('ERROR: ',3i5,' must have lx1=ly1; lz1=1, for 2D')
            call exitt
         endif
         if (ly2.ne.lx2.or.lz2.ne.1) then
            if (nid.eq.0) write(6,11) lx2,ly2,lz2
   11       format('ERROR: ',3i5,' must have lx2=ly2; lz2=1, for 2D')
            call exitt
         endif
      endif

      if (lgmres.lt.5 .and. param(42).eq.0) then
         if(nid.eq.0) write(6,*)
     $   'WARNING: lgmres might be too low!'
      endif


      if (ifsplit) then
         if (lx1.ne.lx2) then
            if (nid.eq.0) write(6,43) lx1,lx2
   43    format('ERROR: lx1,lx2:',2i4,' must be equal for IFSPLIT=T')
            call exitt
         endif
      else
         if (lx2.lt.lx1-2) then
            if (nid.eq.0) write(6,44) lx1,lx2
   44    format('ERROR: lx1,lx2:',2i4,' lx2 must be lx-2 for IFSPLIT=F')
           call exitt
         endif
      endif

      if (ifcvode) then 
         if(nid.eq.0) write(6,*) 
     $   'ABORT: Using CVODE requires .par file!'
         call exitt
      endif

      if (ifmoab .and..not. ifsplit) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: MOAB in Pn-Pn-2 is not supported'
         call exitt
      endif

      ktest = (lx1-lx1m) + (ly1-ly1m) + (lz1-lz1m)
      if (ifstrs.and.ktest.ne.0) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: Stress formulation requires lx1m=lx1, etc. in SIZE'
         call exitt
      endif

      if (ifgfdm.and.ifsplit) call exitti
     $  ('ERROR: FDM (p116>0) requires lx2=lx1-2 in SIZE$',lx2)

      if (ifgfdm.and.lfdm.eq.0) call exitti
     $  ('ERROR: FDM requires lfdm=1 in SIZE$',lfdm)

c      if (ifsplit .and. ifstrs) then
c         if(nid.eq.0) write(6,*) 
c     $   'ABORT: Stress formulation in Pn-Pn is not supported'
c         call exitt
c      endif

      if (ifsplit .and. ifmhd) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: MHD in Pn-Pn is not supported'
         call exitt
      endif

      if (ifmhd .and. lbx1.ne.lx1) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For MHD, need lbx1=lx1, etc.; Change SIZE '
         call exitt
      endif

      if (ifpert .and. lpx1.ne.lx1) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For Lyapunov, need lpx1=lx1, etc.; Change SIZE '
      endif

      if (if3d) ifaxis = .false.

      if (iflomach .and. .not.ifsplit) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For lowMach, need lx2=lx1, etc.; Change SIZE '
         call exitt
      endif

      if (iflomach .and. .not.ifheat) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT For lowMach, need ifheat=true; Change IFHEAT'
         call exitt
      endif

c     if (ifsplit .and. param(55).ne.0) then
c        if(nid.eq.0) write(6,*) 
c    $   'ABORT: Fixed mass flux not supported for Pn-Pn'
c        call exitt
c     endif


      if (ifmhd)           ifchar = .false.   ! For now, at least.

c     set dealiasing handling
      if (param(99).lt.0) then
         param(99) = -1       ! No  dealiasing
      else
         param(99) = 4        ! default
         if (ifaxis) param(99) = 3             ! For now, at least.
         if (ifmvbd) param(99) = 3             ! For now, at least.
      endif

      if (ifchar .and. param(99).lt.0) then
        if (nid.eq.0) write(6,*) 
     &     'ABORT: Characteristic scheme needs dealiasing!'
        call exitt
      endif

      if (param(99).gt.-1 .and. (lxd.lt.lx1 .or. lyd.lt.ly1 .or.
     &   lzd.lt.lz1)) then
         if(nid.eq.0) write(6,*)
     &   'ABORT: Dealiasing space too small; Check lxd,lyd,lzd in SIZE '
         call exitt
      endif

c     set I/O format handling
c     if (param(67).lt.0) then
c        param(67) = 0        ! ASCII
c     else ! elseif (param(67).ne.4) then
c        param(67) = 6        ! binary is default
c     endif

c     if (param(66).lt.0) then
c        param(66) = 0        ! ASCII
c     else ! elseif (param(66).ne.4) then
c        param(66) = 6        ! binary is default
c     endif

c     SET DEFAULT TO 6, ADJUSTED IN USR FILE ONLY
      param(66) = 6
      param(67) = 6

      if (ifdg) param(59)=1  ! No fast operator eval for DG

#ifndef MOAB
      if (ifmoab) then
         print *,"ABORT: ifmoab = .true. in input but this ",
     $ "version of nek not compiled with MOAB."
         call exitti
      endif
#endif

      return

C
C     Error handling:
C
  400 CONTINUE
      if(nid.eq.0) WRITE(6,401)
  401 FORMAT(2X,'ERROR READING PARAMETER DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDPARAM.')
      call exitt
C
  500 CONTINUE
      if(nid.eq.0) WRITE(6,501)
  501 FORMAT(2X,'ERROR READING LOGICAL DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDPARAM.')
      call exitt
C
      return
      END
c-----------------------------------------------------------------------
      subroutine rdmesh
C
C     .Read number of elements
C
C     .Construct sequential element-processor partition according
C      to number of elements and processors
C
C     .Selectively read mesh (defined by element vertices, and group numbers)
C      on each processor
C
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      character*1 adum
      real    dum(4)


c     Read elemental mesh data, formatted
      iffmtin = .true.

      NSIDES=NDIM*2
      DO 40 IEG=1,NELGT
         IF (GLLNID(IEG).EQ.NID) THEN
            IEL=GLLEL(IEG)

            igroup(iel) = 0
            read(9,30,err=31,end=600) igroup(iel)
   30       format(43x,i5)
c           read(9,*,err=31,end=600) adum
   31       continue

C           Read Corner data
            IF(NDIM.EQ.2)THEN
               READ(9,*,ERR=500,END=600) (XC(IC,IEL),IC=1,4)
               READ(9,*,ERR=500,END=600) (YC(IC,IEL),IC=1,4)
                              call rzero (zc(1 ,iel)     ,4)
            ELSE IF(NDIM.EQ.3)THEN
               READ(9,*,ERR=500,END=600) (XC(IC,IEL),IC=1,4)
               READ(9,*,ERR=500,END=600) (YC(IC,IEL),IC=1,4)
               READ(9,*,ERR=500,END=600) (ZC(IC,IEL),IC=1,4)
               READ(9,*,ERR=500,END=600) (XC(IC,IEL),IC=5,8)
               READ(9,*,ERR=500,END=600) (YC(IC,IEL),IC=5,8)
               READ(9,*,ERR=500,END=600) (ZC(IC,IEL),IC=5,8)
            ENDIF
         ELSE
C           Skip over this data for element NOT on this processor
            READ(9,41,ERR=500,END=600) ADUM
C           Read Corner data
            IF(NDIM.EQ.2)THEN
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
            ELSE IF(NDIM.EQ.3)THEN
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
            ENDIF
         ENDIF
   40 CONTINUE
   41 FORMAT(A1)
C
C     End of mesh read.
C
      return
C
C     Error handling:
C
  400 CONTINUE
      if(nid.eq.0) WRITE(6,401) 
  401 FORMAT(2X,'ERROR READING SCALE FACTORS, CHECK READ FILE'
     $    ,/,2X,'ABORTING IN ROUTINE RDMESH.')
      call exitt

  500 CONTINUE
      if(nid.eq.0) WRITE(6,501) IEG
  501 FORMAT(2X,'ERROR READING MESH DATA NEAR ELEMENT',I12
     $    ,/,2X,'ABORTING IN ROUTINE RDMESH.')
      call exitt

  600 CONTINUE
      if(nid.eq.0) WRITE(6,601) IEG
  601 FORMAT(2X,'ERROR 2 READING MESH DATA NEAR ELEMENT',I12
     $    ,/,2X,'ABORTING IN ROUTINE RDMESH.')
      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine rdcurve
C
C     .Read curve side data
C
C     .Disperse curve side data to all processors according 
C      to sequential partition scheme
C
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      CHARACTER*1 ANS
C
C
C
      IF (IFFMTIN) THEN
C
C     Read formatted curve side data 
C
      READ(9,*)
      READ(9,*)NCURVE
      CALL RZERO(CURVE ,72*LELT)
      CALL BLANK(CCURVE,12*LELT)
      IF (NCURVE.GT.0) THEN
         DO 50 ICURVE=1,NCURVE
            IF (NELGT.LT.1000) THEN
               READ(9,60,ERR=500,END=500) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ELSEIF (NELGT.LT.1 000 000) THEN
               READ(9,61,ERR=500,END=500) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ELSE
               READ(9,62,ERR=500,END=500) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ENDIF
   60       FORMAT(I3,I3 ,5G14.6,1X,A1)
   61       FORMAT(I2,I6 ,5G14.6,1X,A1)
   62       FORMAT(I2,I12,5G14.6,1X,A1)

            IF (GLLNID(IEG).EQ.NID) THEN
               IEL=GLLEL(IEG)
               CURVE (1,IEDG,IEL)=R1
               CURVE (2,IEDG,IEL)=R2
               CURVE (3,IEDG,IEL)=R3
               CURVE (4,IEDG,IEL)=R4
               CURVE (5,IEDG,IEL)=R5
               CCURVE(  IEDG,IEL)=ANS
            ENDIF
   50    CONTINUE
      ENDIF
      return
C
C     Error handling:
C
  500 CONTINUE
      if(nid.eq.0) WRITE(6,501)
  501 FORMAT(2X,'ERROR READING CURVE SIDE DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDCURVE.')
      call exitt
      return
C
      ELSE
C
C     Read unformatted curve side data 
C
      READ(8) NCURVE
      CALL RZERO(CURVE ,72*LELT)
      CALL BLANK(CCURVE,12*LELT)
      IF (NCURVE.GT.0) THEN
         DO 1050 ICURVE=1,NCURVE
            READ(8,ERR=1500,END=1500) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            IF (GLLNID(IEG).EQ.NID) THEN
               IEL=GLLEL(IEG)
               CURVE (1,IEDG,IEL)=R1
               CURVE (2,IEDG,IEL)=R2
               CURVE (3,IEDG,IEL)=R3
               CURVE (4,IEDG,IEL)=R4
               CURVE (5,IEDG,IEL)=R5
               CCURVE(  IEDG,IEL)=ANS
            ENDIF
 1050    CONTINUE
      ENDIF
      return
C
C     Error handling:
C
 1500 CONTINUE
      if(nid.eq.0) WRITE(6,1501)
 1501 FORMAT(2X,'ERROR READING unformatted CURVE SIDE DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDCURVE.')
      call exitt
C
      return
      ENDIF
      END
c-----------------------------------------------------------------------
      subroutine rdbdry
C
C     .Read Boundary Conditions (and connectivity data)
C
C     .Disperse boundary condition data to all processors 
C      according to sequential partition scheme
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'SCRCT'
      CHARACTER CBC1*1,CBC3*3,CHTEMP*1,CHTMP3*3
      EQUIVALENCE (CHTEMP,CHTMP3)
      character*132 string
C
C     Set up TEMPORARY value for NFIELD - NFLDT
C
      NFLDT = 1
      IF (IFHEAT) NFLDT=2+NPSCAL
      if (ifmhd ) nfldt=2+npscal+1
      NBCS      = NFLDT
      IBCS      = 2
      IF (IFFLOW) IBCS = 1
      NSIDES    = 2*NDIM
C
C     Read boundary conditions for all fields
C
      LCBC=18*LELT*(LDIMT1 + 1)
      LRBC=30*LELT*(LDIMT1 + 1)
      CALL RZERO(BC ,LRBC)
      CALL BLANK(CBC,LCBC)
C
C-----------------------------------------------------------------
C  Formatted Reads
C-----------------------------------------------------------------
C
      IF (IFFMTIN) THEN
C
      READ(9,*,ERR=500,END=500)  !   ***** BOUNDARY CONDITIONS *****
      ibcnew = 1
      DO 100 IFIELD=ibcnew,NBCS  !     DO 100 IFIELD=IBCS,NBCS
        NEL=NELGT
        if (.not.iftmsh(ifield)) nel = nelgv
C       Fluid and/or thermal
        read(9,81) string        !  ***** FLUID   BOUNDARY CONDITIONS *****
        call capit(string,132)

c       write(6,*) 'reading BC:',ifield,ibcs,nbcs
c       write(6,81) string
c       if1 = indx1(string,'NO ',3)
c       write(6,*) if1,' if NO.  quit.',ifield,ibcs,nbcs
c       write(6,*) ifield,iftmsh(ifield),nel,' iftmsh'
c       call exitt


        if (indx1(string,'NO ',3).eq.0) then ! we have acitve bc info
C
         IF(VNEKTON .LE. 2.52) NBCREA = 3
         IF(VNEKTON .GE. 2.55) NBCREA = 5
C
         DO 80 IEG=1,NEL
         DO 80 ISIDE=1,NSIDES
            IF (GLLNID(IEG).EQ.NID) THEN
               IEL=GLLEL(IEG)
               IF (NELGT.LT.1000) THEN
                  READ(9,50,ERR=500,END=500)    
     $            CHTEMP,
     $            CBC(ISIDE,IEL,IFIELD),ID1,ID2,
     $            (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
c                 write(6,50)
c    $            CHTEMP,
c    $            CBC(ISIDE,IEL,IFIELD),ID1,ID2,
c    $            (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
   50             FORMAT(A1,A3,2I3,5G14.6)
               ELSEIF (NELGT.LT.100 000) THEN
                  READ(9,51,ERR=500,END=500)    
     $            CHTEMP,
     $            CBC(ISIDE,IEL,IFIELD),ID1,ID2,
     $            (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
   51             FORMAT(A1,A3,I5,I1,5G14.6)
               ELSEIF (NELGT.LT.1 000 000) THEN
                  READ(9,52,ERR=500,END=500)    
     $            CHTEMP,
     $            CBC(ISIDE,IEL,IFIELD),ID1,
     $            (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
   52             FORMAT(A1,A3,I6,5G14.6)
               ELSE
                  READ(9,53,ERR=500,END=500)    
     $            CHTEMP,
     $            CBC(ISIDE,IEL,IFIELD),ID1,
     $            (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
   53             FORMAT(A1,A3,I12,5G18.11)
               ENDIF
C              Mesh B.C.'s in 1st column of 1st field
               IF (CHTEMP.NE.' ') CBC(ISIDE,IEL,0)(1:1)= CHTEMP
C              check for fortran function as denoted by lower case bc's:
               CBC1=CBC(ISIDE,IEL,IFIELD)
               CBC3=CBC(ISIDE,IEL,IFIELD)
               ICBC1=ICHAR(CBC1)
c              IF (ICBC1.GE.97.AND.ICBC1.LE.122) THEN
c                 IF(CBC3(3:3).NE.'i')NLINES=BC(1,ISIDE,IEL,IFIELD)
c                 IF(CBC3(3:3).EQ.'i')NLINES=BC(4,ISIDE,IEL,IFIELD)
c                 DO 60 I=1,NLINES
c  60             READ(9,*,ERR=500,END=500)
c              ENDIF
            ELSE
               READ(9,*,ERR=500,END=500)   cbc1  ! dummy read, pff 4/28/05
            ENDIF
   80    CONTINUE
        endif
   81   format(a132)
  100 CONTINUE
C
C     END OF BC READ
C
C     Check for dummy line:  "NO THERMAL B.C.'S"
      IF (NFLDT.EQ.1) READ(9,*,ERR=500,END=500)
C
      return
C
C     Error handling:
C
  500 CONTINUE
      if(nid.eq.0) WRITE(6,501) IFIELD,IEG
  501 FORMAT(2X,'ERROR READING BOUNDARY CONDITIONS FOR FIELD',I4,I12
     $    ,/,2X,'ABORTING IN ROUTINE RDBDRY.')
      call exitt
      return
C
C
      ELSE
C
C-----------------------------------------------------------------
C  UNformatted Reads
C-----------------------------------------------------------------
C
c     READ(8,ERR=500,END=500)
      DO 1100 IFIELD=IBCS,NBCS
         NEL=NELGT
C        Fluid and/or thermal
         NBCREA = 5
C
         DO 1080 IEG=1,NEL
         DO 1080 ISIDE=1,NSIDES
            IF (GLLNID(IEG).EQ.NID) THEN
               IEL=GLLEL(IEG)
               READ(8,ERR=1500,END=1500)    
     $         CHTMP3,
     $         CBC(ISIDE,IEL,IFIELD),ID1,ID2,
     $         (BC(II,ISIDE,IEL,IFIELD),II=1,NBCREA)
C
C              Mesh B.C.'s in 1st column of 1st field
               IF (CHTEMP.NE.' ') CBC(ISIDE,IEL,0)(1:1)= CHTEMP
C              check for fortran function as denoted by lower case bc's:
            ELSE
               IEL=1
               READ(8,ERR=1500,END=1500) CHTMP3,
     $         CBCS(ISIDE,IEL),ID1,ID2,(BCS(II,ISIDE,IEL),II=1,NBCREA)
C              check for fortran function as denoted by lower case bcs:
            ENDIF
 1080    CONTINUE
 1100 CONTINUE
C
C     END OF BC READ
C
      return
C
C     Error handling:
C
 1500 CONTINUE
      if(nid.eq.0) WRITE(6,1501) IFIELD,IEG
 1501 FORMAT(2X,'ERROR READING BOUNDARY CONDITIONS FOR FIELD',I4,I12
     $    ,/,2X,'(unformatted) ABORTING IN ROUTINE RDBDRY.')
      call exitt
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine rdicdf
C
C     .Read Initial Conditions / Drive Force
C
C     .Broadcast ICFILE to all processors
C
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'

      character*132 line
      logical      ifgtil

      ierr = 0

      if (nid.eq.0) then   !  Read names of restart files

        call blank(initc,15*132)
        read (9,80,err=200,end=200) line
        call capit(line,132)
        if (indx1(line,'RESTART',7).ne.0) then
           if (.not.ifgtil(nskip,line)) goto 200
C          read(line,*,err=200,end=200) nskip
           do 50 i=1,nskip
              read(9,80,err=200,end=200) initc(i)
   50      continue
           read(9,80,err=200,end=200) line
        endif
   80   format(a132)

        if (.not.ifgtil(nskip,line)) goto 200

C       Read initial conditions
        do 100 i=1,nskip
           read(9,80,err=200,end=200) line
  100   continue

C       Read drive force data
        read(9,*,err=200,end=200)
        read(9,*,err=200,end=200) nskip
        do 110 i=1,nskip
          read(9,80,err=200,end=200) line
  110   continue
      endif

      ierr = iglmax(ierr,1)
      if (ierr.eq.0) then
         call bcast(initc,15*132*csize)
         return
      else
         goto 210
      endif
c
c     Error handling:
c
  200 ierr = 1
      ierr = iglmax(ierr,1)
      
  210 continue
      if (nid.eq.0) write(6,300)
  300 format(2x,'Error reading initial condition/drive force data'
     $    ,/,2x,'aborting in routine rdicdf.')
      call exitti('rdicdf error$',ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine rdmatp
C
C     .Read materials property data
C
C     .Disperse material properties to all processors according 
C      to sequential partition scheme
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'

      CHARACTER*132 LINE
C
      CALL IZERO(MATYPE,16*LDIMT1)
      CALL RZERO(CPGRP ,48*LDIMT1)
C
C     Read material property data
C
      IF(NID.EQ.0) THEN
        READ(9,*,ERR=200,END=200)
        READ(9,*,ERR=200,END=200) NSKIP
        READ(9,*,ERR=200,END=200) NPACKS
        DO 100 IIG=1,NPACKS
           IFVPS=.TRUE.
           READ(9,*)IGRP,IFLD,ITYPE
           MATYPE(IGRP,IFLD)=ITYPE
           DO 100 IPROP=1,3
              IF(ITYPE.EQ.1) READ(9,* ) CPGRP(IGRP,IFLD,IPROP)
              IF(ITYPE.EQ.2) READ(9,80) LINE
   80   FORMAT(A132)
  100   CONTINUE
      ENDIF

      CALL BCAST(MATYPE,16*LDIMT1*ISIZE)
      CALL BCAST(CPGRP ,48*LDIMT1*WDSIZE)

      return
C
C     Error handling:
C
  200 CONTINUE
      if(nid.eq.0) WRITE(6,201)
  201 FORMAT(2X,'ERROR READING MATERIAL PROPERTIES DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDMATP.')
      call exitt
C
      return
      END
c-----------------------------------------------------------------------
      subroutine rdhist
C
C     .Read history data
C
C     .Broadcast to all processors
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
C
      CALL BLANK (HCODE ,11*lhis)
      CALL IZERO (LOCHIS, 4*lhis)

      ierr=0
      IF(NID.EQ.0) THEN
C       Read history data
        READ (9,*)
        READ (9,*,ERR=200,END=200) NHIS
        if (nhis.gt.lhis) then
           write(6,*) nid,' Too many history pts. RESET LHIS.',nhis,lhis
           ierr=1
        endif
 
        if(ierr.eq.0) then
C       HCODE(10) IS WHETHER IT IS HISTORY, STREAKLINE, PARTICLE, ETC.
        if (nhis.gt.0) then
           do i=1,nhis
              if (nelgt.lt.100000) then
                 read(9,130,err=200,end=200)
     $           (hcode(ii,i),ii=1,11),(lochis(i2,i),i2=1,4)
  130            format(1x,11a1,1x,4i5)
              else
                 read(9,131,err=200,end=200)
     $           (hcode(ii,i),ii=1,11),(lochis(i2,i),i2=1,4)
  131            format(1x,11a1,1x,3i5,i10)
              endif
c
c           threshold lochis locations to allow easy specification of "NX,NY,NZ"
c           pff 1/7/97
c
              if (hcode(10,i).eq.'H') then
                 lochis(1,i) = min(lochis(1,i),nx1)
                 lochis(2,i) = min(lochis(2,i),ny1)
                 lochis(3,i) = min(lochis(3,i),nz1)
c
c              if lochis_k = -1, set it to nxk/2   pff 8/21/03
c
                 if (lochis(1,i).eq.-1) lochis(1,i) = (nx1+1)/2
                 if (lochis(2,i).eq.-1) lochis(2,i) = (ny1+1)/2
                 if (lochis(3,i).eq.-1) lochis(3,i) = (nz1+1)/2
              endif
           enddo
        endif
        endif
      ENDIF
      call err_chk(ierr,' Too many histroy pts. RESET LHIS$')

      call bcast(NHIS  ,ISIZE)
      call bcast(HCODE ,11*LHIS*CSIZE)
      call bcast(LOCHIS,4*LHIS*ISIZE)

      return
C
C     Error handling:
C
  200 CONTINUE
      if(nid.eq.0) WRITE(6,201)
  201 FORMAT(2X,'ERROR READING HISTORY DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDHIST.')
      call exitt
C
      return
      END
c-----------------------------------------------------------------------
      subroutine rdout
C
C     .Read output specs
C
C     .Broadcast to all processors
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'

      logical lbuf(5+ldimt1)  

      call lfalse(lbuf,5+ldimt1)
      iflag = 0                           ! Check for valid ipsco read

      IF(NID.EQ.0) THEN                   ! Read output specs

        READ(9,*,ERR=200,END=200)
        READ(9,*,ERR=200,END=200) NOUTS
        READ(9,*,ERR=200,END=200) IFXYO
        READ(9,*,ERR=200,END=200) IFVO
        READ(9,*,ERR=200,END=200) IFPO
        READ(9,*,ERR=200,END=200) IFTO
        READ(9,*,ERR=200,END=200) IFBO   !  IFTGO

        lbuf(1) = IFXYO
        lbuf(2) = IFVO
        lbuf(3) = IFPO
        lbuf(4) = IFTO
        lbuf(5) = IFBO

        k = 5

        call lfalse(ifpsco,ldimt1)
        read(9,*,err=200,end=200) ipsco
        if (ipsco.gt.0) then
           if (ipsco.gt.ldimt1) then    ! Invalid ifpsco read
              iflag = 1
           else
              do i=1,ipsco
                 read(9,*,err=200,end=200) ifpsco(i)
                 k = k+1
                 lbuf(k) = ifpsco(i)
              enddo
           endif
        endif

      endif


      iflag = iglmax(iflag,1)                       ! Check for valid ipsco read
      if (iflag.gt.0) call exitti                   ! Invalid ifpsco read
     $   ('Error in rdout.  Increase ldimt1 in SIZE to$',ipsco)

      k = 5+ldimt1
      call bcast(lbuf ,LSIZE*k)
      call bcast(IPSCO,ISIZE  )

      ifxyo = lbuf(1)  
      ifvo  = lbuf(2) 
      ifpo  = lbuf(3) 
      ifto  = lbuf(4) 
      ifbo  = lbuf(5) 

      k = 5
      do i=1,ipsco
         k = k+1
         ifpsco(i) = lbuf(k)
      enddo

      return

C
C     Error handling:
C
  200 CONTINUE
      WRITE(6,201)
  201 FORMAT(2X,'ERROR READING OUTPUT SPECIFICATION DATA'
     $    ,/,2X,'ABORTING IN ROUTINE RDOUT.')
      call exitt
C
      return
      END
c-----------------------------------------------------------------------
      subroutine rdobj
C
C     .Read objects
C
C     .Broadcast to all processors
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
C
C     Default if no data is read No Objects
C
      ierr=0
      IF(NID.EQ.0) THEN
        NOBJ=0
        READ(9,*,ERR=200,END=200)
        READ(9,*,ERR=200,END=200) NOBJ
 
        IF(NOBJ.GT.MAXOBJ) ierr=1
 
        if(ierr.eq.0) then
         DO 10 IOBJ = 1,NOBJ
            READ(9,*,ERR=200,END=200) NMEMBER(IOBJ)
            IF(NMEMBER(IOBJ).GT.MAXMBR)THEN
               PRINT*,'ERROR: Too many members in object ',IOBJ
               ierr=2
            ENDIF
            if(ierr.eq.0) then
             DO 5 MEMBER=1,NMEMBER(IOBJ)
                READ(9,*,ERR=200,END=200) OBJECT(IOBJ,MEMBER,1),
     $                                    OBJECT(IOBJ,MEMBER,2)
    5        CONTINUE
            endif
   10    CONTINUE
         write(6,*) nobj,' objects found'
     $            ,(nmember(k),k=1,nobj)
        endif
      endif
      call err_chk(ierr,'ERROR, too many objects:$')
 
      call bcast(NOBJ   ,ISIZE)
      call bcast(NMEMBER,MAXOBJ*ISIZE)
      call bcast(OBJECT ,MAXOBJ*MAXMBR*2*ISIZE)

 
      return
C
C     Error handling:  For old versions, default to no objects
C
  200 CONTINUE
      NOBJ=0
 
      return
      END
c-----------------------------------------------------------------------
      subroutine vrdsmsh
C
C=====================================================================
C     Verify that mesh and dssum are properly defined by performing
C        a direct stiffness operation on the X,Y and Z coordinates.
C     Note that periodic faces are not checked here.
C=====================================================================
C
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      COMMON /SCRNS/ TA(LX1,LY1,LZ1,LELT),TB(LX1,LY1,LZ1,LELT)
     $           ,QMASK(LX1,LY1,LZ1,LELT),tmp(2)
      CHARACTER*3 CB

c      call  vrdsmshx  ! verify mesh topology

      if(nio.eq.0) write(*,*) 'verify mesh topology'

      IERR      = 0
      EPS       = 1.0e-04
      EPS       = 1.0e-03
      IFIELD    = 1
      IF (IFHEAT) IFIELD = 2
      NXYZ1     = NX1*NY1*NZ1
      NTOT      = NX1*NY1*NZ1*NELT
      NFACES    = 2*NDIM

      xmx = glmax(xm1,ntot)
      xmn = glmin(xm1,ntot)
      ymx = glmax(ym1,ntot)
      ymn = glmin(ym1,ntot)
      zmx = glmax(zm1,ntot)
      zmn = glmin(zm1,ntot)
      if (nio.eq.0) write(6,*) xmn,xmx,' Xrange'
      if (nio.eq.0) write(6,*) ymn,ymx,' Yrange'
      if (nio.eq.0) write(6,*) zmn,zmx,' Zrange'
c     return
C
C     First check - use 1/Multiplicity
C
      IF (IFHEAT) THEN 
         CALL COPY(TA,TMULT,NTOT)
      ELSE
         CALL COPY(TA,VMULT,NTOT)
      ENDIF
c
c     write(6,1) 
c    $(nid,'tab4',lglel(ie),(ta(k,1,1,ie),k=1,nx1*ny1),ie=1,nelt)
c   1 format(i3,a4,i3,16f5.2)
c
      CALL DSSUM(TA,NX1,NY1,NZ1)
c
c     write(6,1) 
c    $(nid,'taaf',lglel(ie),(ta(k,1,1,ie),k=1,nx1*ny1),ie=1,nelt)
c
      CALL RONE (TB,NTOT)
      CALL SUB2 (TB,TA,NTOT)
      DO 1000 IE=1,NELT
      ieg=lglel(ie)
      DO 1000 IZ=1,NZ1
      DO 1000 IY=1,NY1
      DO 1000 IX=1,NX1
         IF (ABS(TB(IX,IY,IZ,IE)).GT.EPS ) THEN
            WRITE(6,1005) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TA(IX,IY,IZ,IE),eps
c           WRITE(7,1005) IX,IY,IZ,IEG
c    $      ,XM1(IX,IY,IZ,IE),TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE)
c    $      ,QMASK(IX,IY,IZ,IE)
 1005       FORMAT(2X,'WARNING: DSSUM problem at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,2X,'Near X =',3G16.8,', d:',2G16.8)
            IERR=4
         ENDIF
 1000 CONTINUE
C
C     Set up QMASK quickly to annihilate checks on periodic bc's
C
      CALL RONE(QMASK,NTOT)
      DO 100 IEL=1,NELT
      DO 100 IFACE=1,NFACES
         CB =CBC(IFACE,IEL,IFIELD)
         IF (CB.EQ.'P  '.or.cb.eq.'p  ') 
     $         CALL FACEV(QMASK,IEL,IFACE,0.0,NX1,NY1,NZ1)
  100 CONTINUE
      CALL DSOP(QMASK,'MUL',NX1,NY1,NZ1)

c      xxmin = glmin(xm1,ntot)
c      yymin = glmin(ym1,ntot)
c      zzmin = glmin(zm1,ntot)
c      xxmax = glmax(xm1,ntot)
c      yymax = glmax(ym1,ntot)
c      zzmax = glmax(zm1,ntot)
c      if (nio.eq.0) write(6,7) xxmin,yymin,zzmin,xxmax,yymax,zzmax
c    7 format('xyz minmx2:',6g13.5)



C
C     X-component
C
      CALL COPY(TA,XM1,NTOT)
      CALL COPY(TB,XM1,NTOT)
      CALL DSOP(TA,'MIN',NX1,NY1,NZ1)
      CALL DSOP(TB,'MAX',NX1,NY1,NZ1)
      CALL SUB2(TA,XM1,NTOT)
      CALL SUB2(TB,XM1,NTOT)
      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)
      DO 1100 IE=1,NELT
         XSCMAX = VLMAX(XM1(1,1,1,IE),NXYZ1)
         XSCMIN = VLMIN(XM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(XSCMAX-XSCMIN)
         SCAL2=ABS(XSCMAX)
         SCAL3=ABS(XSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         XSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1100 IZ=1,NZ1
         DO 1100 IY=1,NY1
         DO 1100 IX=1,NX1
         if (abs(ta(ix,iy,iz,ie)*xscale).gt.eps .or.
     $       abs(tb(ix,iy,iz,ie)*xscale).gt.eps ) then
            write(6,1105) ix,iy,iz,ieg
     $      ,xm1(ix,iy,iz,ie),ym1(ix,iy,iz,ie),zm1(ix,iy,iz,ie)
     $      ,tb(ix,iy,iz,ie),ta(ix,iy,iz,ie),XSCALE
 1105       format(1x,'WARNING1 Element mesh mismatch at:',/
     $            ,1x,'i,j,k,ie:',3i5,I12,/
     $            ,1X,'Near X =',3G16.8,', d:',3G16.8)
            ierr=1
         endif
 1100 CONTINUE
C
C     Y-component
C
      CALL COPY(TA,YM1,NTOT)
      CALL COPY(TB,YM1,NTOT)
      CALL DSOP(TA,'MIN',NX1,NY1,NZ1)
      CALL DSOP(TB,'MAX',NX1,NY1,NZ1)
      CALL SUB2(TA,YM1,NTOT)
      CALL SUB2(TB,YM1,NTOT)
      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)
      DO 1200 IE=1,NELT
         YSCMAX = VLMAX(YM1(1,1,1,IE),NXYZ1)
         YSCMIN = VLMIN(YM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(YSCMAX-YSCMIN)
         SCAL2=ABS(YSCMAX)
         SCAL3=ABS(YSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         YSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1200 IZ=1,NZ1
         DO 1200 IY=1,NY1
         DO 1200 IX=1,NX1
         IF (ABS(TA(IX,IY,IZ,IE)*YSCALE).GT.EPS .OR.
     $       ABS(TB(IX,IY,IZ,IE)*YSCALE).GT.EPS ) THEN
            WRITE(6,1205) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE),yscale
 1205       FORMAT(1X,'WARNING2 Element mesh mismatch at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,1X,'Near Y =',3G16.8,', d:',3G16.8)
            IERR=2
         ENDIF
 1200 CONTINUE
C
C     Z-component
C
      IF (IF3D) THEN
       CALL COPY(TA,ZM1,NTOT)
       CALL COPY(TB,ZM1,NTOT)
       CALL DSOP(TA,'MIN',NX1,NY1,NZ1)
       CALL DSOP(TB,'MAX',NX1,NY1,NZ1)
       CALL SUB2(TA,ZM1,NTOT)
       CALL SUB2(TB,ZM1,NTOT)
       CALL COL2(TA,QMASK,NTOT)
       CALL COL2(TB,QMASK,NTOT)
       DO 1300 IE=1,NELT
          ZSCMAX = VLMAX(ZM1(1,1,1,IE),NXYZ1)
          ZSCMIN = VLMIN(ZM1(1,1,1,IE),NXYZ1)
          SCAL1=ABS(ZSCMAX-ZSCMIN)
          SCAL2=ABS(ZSCMAX)
          SCAL3=ABS(ZSCMIN)
          SCAL1=MAX(SCAL1,SCAL2)
          SCAL1=MAX(SCAL1,SCAL3)
          ZSCALE = 1./SCAL1
          ieg=lglel(ie)
          DO 1300 IZ=1,NZ1
          DO 1300 IY=1,NY1
          DO 1300 IX=1,NX1
          IF (ABS(TA(IX,IY,IZ,IE)*ZSCALE).GT.EPS .OR.
     $        ABS(TB(IX,IY,IZ,IE)*ZSCALE).GT.EPS ) THEN
           WRITE(6,1305) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE),zscale
 1305       FORMAT(1X,'WARNING3 Element mesh mismatch at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,1X,'Near Z =',3G16.8,', d:',3G16.8)
            IERR=3
          ENDIF
 1300  CONTINUE
      ENDIF
 
      ierr = iglsum(ierr,1)
      IF (IERR.gt.0) THEN
         if(nid.eq.0) WRITE(6,1400) 
 1400    FORMAT
     $   (' Mesh consistency check failed.  EXITING in VRDSMSH.')
            call exitt
      ENDIF
 
      tmp(1)=ierr
      CALL GOP(tmp,tmp(2),'M  ',1)
      IF (tmp(1).ge.4.0) THEN
         WRITE(6,1400) 
     $   (' Mesh consistency check failed.  EXITING in VRDSMSH.')
         call exitt
      ENDIF
 
      if(nio.eq.0) then
        write(6,*) 'done :: verify mesh topology'
        write(6,*) ' '
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine vrdsmshx  ! verify mesh topology
C
C=====================================================================
C     Verify that mesh and dssum are properly defined by performing
C        a direct stiffness operation on the X,Y and Z coordinates.
C     Note that periodic faces are not checked here.
C=====================================================================
C
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      common /scrns/ tc(lx1,ly1,lz1,lelt),td(lx1,ly1,lz1,lelt)
     $             , ta(lx1,ly1,lz1,lelt),tb(lx1,ly1,lz1,lelt)
     $             , qmask(lx1,ly1,lz1,lelt)
      CHARACTER*3 CB
C
      IERR      = 0
      EPS       = 1.0e-04
      EPS       = 1.0e-03
      IFIELD    = 1
      IF (IFHEAT) IFIELD = 2
      NXYZ1     = NX1*NY1*NZ1
      NTOT      = NX1*NY1*NZ1*NELT
      NFACES    = 2*NDIM

      xmx = glmax(xm1,ntot)
      xmn = glmin(xm1,ntot)
      ymx = glmax(ym1,ntot)
      ymn = glmin(ym1,ntot)
      zmx = glmax(zm1,ntot)
      zmn = glmin(zm1,ntot)
      if (nio.eq.0) write(6,*) xmn,xmx,' Xrange'
      if (nio.eq.0) write(6,*) ymn,ymx,' Yrange'
      if (nio.eq.0) write(6,*) zmn,zmx,' Zrange'
c     return
C
C     First check - use 1/Multiplicity
C
      IF (IFHEAT) THEN 
         CALL COPY(TA,TMULT,NTOT)
      ELSE
         CALL COPY(TA,VMULT,NTOT)
      ENDIF
c
c     write(6,1) 
c    $(nid,'tab4',lglel(ie),(ta(k,1,1,ie),k=1,nx1*ny1),ie=1,nelt)
c   1 format(i3,a4,i3,16f5.2)
c
      CALL DSSUM(TA,NX1,NY1,NZ1)
c
c     write(6,1) 
c    $(nid,'taaf',lglel(ie),(ta(k,1,1,ie),k=1,nx1*ny1),ie=1,nelt)
c
      CALL RONE (TB,NTOT)
      CALL SUB2 (TB,TA,NTOT)
      DO 1000 IE=1,NELT
      ieg=lglel(ie)
      DO 1000 IZ=1,NZ1
      DO 1000 IY=1,NY1
      DO 1000 IX=1,NX1
         IF (ABS(TB(IX,IY,IZ,IE)).GT.EPS ) THEN
            WRITE(6,1005) IX,IY,IZ,IEG
     $      ,XM1(IX,IY,IZ,IE),YM1(IX,IY,IZ,IE),ZM1(IX,IY,IZ,IE)
     $      ,TA(IX,IY,IZ,IE),eps
c           WRITE(7,1005) IX,IY,IZ,IEG
c    $      ,XM1(IX,IY,IZ,IE),TB(IX,IY,IZ,IE),TA(IX,IY,IZ,IE)
c    $      ,QMASK(IX,IY,IZ,IE)
 1005       FORMAT(2X,'WARNING: DSSUM problem at:',/
     $            ,1X,'I,J,K,IE:',3I5,i12,/
     $            ,2X,'Near X =',3G16.8,', d:',2G16.8)
            IERR=4
         ENDIF
 1000 CONTINUE
C
C     Set up QMASK quickly to annihilate checks on periodic bc's
C
      CALL RONE(QMASK,NTOT)
      DO 100 IEL=1,NELT
      DO 100 IFACE=1,NFACES
         CB =CBC(IFACE,IEL,IFIELD)
         IF (CB.EQ.'P  '.or.cb.eq.'p  ') 
     $         CALL FACEV(QMASK,IEL,IFACE,0.0,NX1,NY1,NZ1)
  100 CONTINUE
      call dsop(QMASK,'MUL',NX1,NY1,NZ1)

      xxmin = glmin(xm1,ntot)
      yymin = glmin(ym1,ntot)
      zzmin = glmin(zm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ym1,ntot)
      zzmax = glmax(zm1,ntot)
      if (nio.eq.0) write(6,7) xxmin,yymin,zzmin,xxmax,yymax,zzmax
    7 format('xyz minmx2:',6g13.5)


C
C     X-component
C
      call copy(ta,xm1,ntot)
      call copy(tb,xm1,ntot)
      call dsop(ta,'min',nx1,ny1,nz1)
      call dsop(tb,'max',nx1,ny1,nz1)

      call copy(tc,xm1,ntot)
      call copy(td,xm1,ntot)
      call dsop(tc,'min',nx1,ny1,nz1)
      call dsop(td,'max',nx1,ny1,nz1)

      xxmin = glmin(xm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,9) xxmin,yymin,zzmin,xxmax,yymax,zzmax
    9 format('xyz minmx3:',6g13.5)

      CALL SUB2(TA,XM1,NTOT)
      CALL SUB2(TB,XM1,NTOT)

      xxmin = glmin(qmask,ntot)
      xxmax = glmax(qmask,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,19) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   19 format('xyz minmx4:',6g13.5)

      CALL COL2(TA,QMASK,NTOT)
      CALL COL2(TB,QMASK,NTOT)

      xxmin = glmin(qmask,ntot)
      xxmax = glmax(qmask,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,29) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   29 format('xyz minmx5:',6g13.5)

      DO 1100 IE=1,NELT
         XSCMAX = VLMAX(XM1(1,1,1,IE),NXYZ1)
         XSCMIN = VLMIN(XM1(1,1,1,IE),NXYZ1)
         SCAL1=ABS(XSCMAX-XSCMIN)
         SCAL2=ABS(XSCMAX)
         SCAL3=ABS(XSCMIN)
         SCAL1=MAX(SCAL1,SCAL2)
         SCAL1=MAX(SCAL1,SCAL3)
         XSCALE = 1./SCAL1
         ieg=lglel(ie)
         DO 1100 IZ=1,NZ1
         DO 1100 IY=1,NY1
         DO 1100 IX=1,NX1
         if (abs(ta(ix,iy,iz,ie)*xscale).gt.eps .or.
     $       abs(tb(ix,iy,iz,ie)*xscale).gt.eps ) then
            write(6,1105) nid,ix,iy,iz,ie,ieg
     $      ,xm1(ix,iy,iz,ie),tc(ix,iy,iz,ie),td(ix,iy,iz,ie)
     $      ,ym1(ix,iy,iz,ie),zm1(ix,iy,iz,ie)
     $      ,ta(ix,iy,iz,ie),tb(ix,iy,iz,ie),xscale
     $      ,qmask(ix,iy,iz,ie)
 1105       format(i4.4,1x,'ie:',3i3,i10,i10,1p9e11.3)
c1105       format(i4.4,1x,'ie:',3i3,i6,1p9e11.3)
            ierr=1
            goto 1101
         endif
 1100 CONTINUE
 1101 CONTINUE

      xxmin = glmin(xm1,ntot)
      xxmax = glmax(xm1,ntot)
      yymax = glmax(ta ,ntot)
      yymin = glmin(ta ,ntot)
      zzmin = glmin(tb ,ntot)
      zzmax = glmax(tb ,ntot)
      if (nio.eq.0) write(6,39) xxmin,yymin,zzmin,xxmax,yymax,zzmax
   39 format('xyz minmx5:',6g13.5)

c     ifvo = .true.
c     ifpo = .false.
c     ifto = .true.
c     call outpost(xm1,ta,tb,pr,qmask,'   ')
c     call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine rotat2(xyz,angle,npts)
C
C     Rotate NPTS through ANGLE (in two directions IF3D).
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZ(3,1)
      COMMON /CTMP0/ RMTRX(3,3),RX(3,3),RZ(3,3),XYZN(3,10)
C
      SINA=SIN(ANGLE)
      COSA=COS(ANGLE)
      CALL RZERO(RX,9)
      CALL RZERO(RZ,9)
      RX(1,1)=COSA
      RX(2,2)=COSA
      RX(1,2)=SINA
      RX(2,1)=-SINA
      RX(3,3)=1.0
      IF (IF3D) THEN
         RZ(1,1)=COSA
         RZ(3,3)=COSA
         RZ(1,3)=SINA
         RZ(3,1)=-SINA
         RZ(2,2)=1.0
      ELSE
         RZ(1,1)=1.0
         RZ(2,2)=1.0
         RZ(3,3)=1.0
      ENDIF
      CALL MXM(RX,3,RZ,3,RMTRX,3)
C
C     Strip mine mxms in chunks of 10:
      DO 100 I=1,NPTS-10,10
         CALL MXM(RMTRX,3,XYZ(1,I),3,XYZN,10)
         CALL COPY(XYZ(1,I),XYZN,30)
  100 CONTINUE
      N10=MOD1(NPTS,10)
      I=NPTS-N10+1
      CALL RZERO(XYZN,30)
      IF (N10.GT.0) THEN
         CALL MXM(RMTRX,3,XYZ(1,I),3,XYZN,N10)
         CALL COPY(XYZ(1,I),XYZN,3*N10)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine scale(xyzl,nl)
C
C     Rescale XYZL such that the mean value of IXX=IYY=IZZ for each element.
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZL(3,8,LELT)
      COMMON /CTMP0/ VO(LELT),XYZI(3,LELT),CG(3,LELT)
     $              ,TI(6),WORK(6)
C
C     Compute volumes -
C
      CALL VOLUME2(VO,XYZL,NL)
      VTOT=GLSUM (VO,NL)
C
C     Compute (weighted) average inertia for each element.
C
      NCRNR=2**NDIM
      CALL RZERO(TI,6)
      DO 100 IL=1,NL
         VO0 = VO(IL)/VTOT
         CALL INRTIA(XYZI(1,IL),CG(1,IL),XYZL(1,1,IL),NCRNR,1)
         TI(1)=TI(1)+XYZI(1,IL)*VO0
         TI(2)=TI(2)+XYZI(2,IL)*VO0
         TI(3)=TI(3)+XYZI(3,IL)*VO0
         TI(4)=TI(4)+CG(1,IL)  *VO0
         TI(5)=TI(5)+CG(2,IL)  *VO0
         TI(6)=TI(6)+CG(3,IL)  *VO0
  100 CONTINUE
      CALL GOP(TI,WORK,'+  ',6)
      XI  =SQRT(TI(1))
      YI  =SQRT(TI(2))
      ZI  =1.0
      IF (IF3D) ZI=SQRT(TI(3))
C
C     Rescale ( & shift to a nearly mean zero )
C
      DO 200 IL=1,NL
      DO 200 IC=1,NCRNR
         XYZL(1,IC,IL)=(XYZL(1,IC,IL)-TI(4))/XI
         XYZL(2,IC,IL)=(XYZL(2,IC,IL)-TI(5))/YI
         XYZL(3,IC,IL)=(XYZL(3,IC,IL)-TI(6))/ZI
  200 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine inrtia(xyzi,cg,xyzl,n,itype)
C
C     Compute cg and inertia for a collection of unit point masses.
C     This is a global (multiprocessor) operation, only IF itype=2.
C
      DIMENSION XYZI(3),CG(3),XYZL(3,1)
      DIMENSION TI(4),WORK(4)
C
      TI(1)=0.0
      TI(2)=0.0
      TI(3)=0.0
      TI(4)=N
      DO 100 I=1,N
         TI(1)=TI(1)+XYZL(1,I)
         TI(2)=TI(2)+XYZL(2,I)
         TI(3)=TI(3)+XYZL(3,I)
  100 CONTINUE
      IF (ITYPE.EQ.2) CALL GOP(TI,WORK,'+  ',4)
      IF (TI(4).EQ.0.0) TI(4)=1.0
      CG(1)=TI(1)/TI(4)
      CG(2)=TI(2)/TI(4)
      CG(3)=TI(3)/TI(4)
C
      TI(1)=0.0
      TI(2)=0.0
      TI(3)=0.0
      DO 200 I=1,N
         TI(1)=TI(1)+( XYZL(1,I)-CG(1) )**2
         TI(2)=TI(2)+( XYZL(2,I)-CG(2) )**2
         TI(3)=TI(3)+( XYZL(3,I)-CG(3) )**2
  200 CONTINUE
      IF (ITYPE.EQ.2) CALL GOP(TI,WORK,'+  ',3)
      TI(1)=TI(1)/TI(4)
      TI(2)=TI(2)/TI(4)
      TI(3)=TI(3)/TI(4)
      IF (ITYPE.EQ.2) THEN
C        std. def'n of inertia.
         XYZI(1)=TI(2)+TI(3)
         XYZI(2)=TI(3)+TI(1)
         XYZI(3)=TI(1)+TI(2)
      ELSE
         XYZI(1)=TI(1)
         XYZI(2)=TI(2)
         XYZI(3)=TI(3)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine volume2(vol,xyz,n)
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      DIMENSION XYZ(3,2,2,2,1)
      DIMENSION VOL(1)
C
      DO 1000 IE=1,N
         VOL(IE)=0.0
         IF (IF3D) THEN        
           DO 20 K=1,2
           DO 20 J=1,2
           DO 20 I=1,2
              VOL1 = (XYZ(1,2,J,K,IE)-XYZ(1,1,J,K,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,I,J,2,IE)-XYZ(3,I,J,1,IE))
              VOL2 = (XYZ(1,2,J,K,IE)-XYZ(1,1,J,K,IE))
     $             * (XYZ(2,I,J,2,IE)-XYZ(2,I,J,1,IE))
     $             * (XYZ(3,I,2,K,IE)-XYZ(3,I,1,K,IE))
              VOL3 = (XYZ(1,I,2,K,IE)-XYZ(1,I,1,K,IE))
     $             * (XYZ(2,2,J,K,IE)-XYZ(2,1,J,K,IE))
     $             * (XYZ(3,I,J,2,IE)-XYZ(3,I,J,1,IE))
              VOL4 = (XYZ(1,I,J,2,IE)-XYZ(1,I,J,1,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,I,2,K,IE)-XYZ(3,I,1,K,IE))
              VOL5 = (XYZ(1,I,2,K,IE)-XYZ(1,I,1,K,IE))
     $             * (XYZ(2,I,J,2,IE)-XYZ(2,I,J,1,IE))
     $             * (XYZ(3,2,J,K,IE)-XYZ(3,1,J,K,IE))
              VOL6 = (XYZ(1,I,J,2,IE)-XYZ(1,I,J,1,IE))
     $             * (XYZ(2,I,2,K,IE)-XYZ(2,I,1,K,IE))
     $             * (XYZ(3,2,J,K,IE)-XYZ(3,1,J,K,IE))
              VOL(IE) = VOL(IE)+VOL1+VOL2+VOL3+VOL4+VOL5+VOL6
   20      CONTINUE
           VOL(IE)=VOL(IE)/8.0
         ELSE
C     2-D:
            DO 40 J=1,2
            DO 40 I=1,2
              VOL1 = (XYZ(1,2,J,1,IE)-XYZ(1,1,J,1,IE))
     $             * (XYZ(2,I,2,1,IE)-XYZ(2,I,1,1,IE))
              VOL3 = (XYZ(1,I,2,1,IE)-XYZ(1,I,1,1,IE))
     $             * (XYZ(2,2,J,1,IE)-XYZ(2,1,J,1,IE))
              VOL(IE)=VOL(IE)+VOL1+VOL3
   40      CONTINUE
           VOL(IE)=VOL(IE)/4.0
         ENDIF
         VOL(IE)=ABS(VOL(IE))
 1000 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine findcg(cg,xyz,n)
C
C     Compute cg for N elements.
C
      INCLUDE 'SIZE'
      DIMENSION CG(3,1),XYZ(3,8,1)
C
      NCRNR=2**NDIM
      CALL RZERO(CG,3*N)
      DO 100 I =1,N
      DO 100 IC=1,NCRNR
         CG(1,I)=CG(1,I)+XYZ(1,IC,I)
         CG(2,I)=CG(2,I)+XYZ(2,IC,I)
         CG(3,I)=CG(3,I)+XYZ(3,IC,I)
  100 CONTINUE
      TMP=1.0/(NCRNR)
      CALL CMULT(CG,TMP,3*N)
      return
      END
c-----------------------------------------------------------------------
      subroutine divide(list1,list2,nl1,nl2,ifok,list,nl,xyzi,cg,WGT)
C
C     Divide the elements associated with this subdomain according to
C     the direction having the smallest moment of inertia (the "long"
C     direction).
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'TSTEP'
C
      DIMENSION LIST(LELT),LIST1(LELT),LIST2(LELT)
      DIMENSION XYZI(3),CG(3,LELT),wgt(1)
      COMMON /CTMP0/ XCG(LELT),YCG(LELT),ZCG(LELT)
      REAL IXX,IYY,IZZ
      INTEGER WORK(2),WRK2(2)
      LOGICAL IFOK
C
C     Choose "long" direction:
C
      IXX=XYZI(1)
      IYY=XYZI(2)
      IZZ=XYZI(3)
      IF (IF3D) THEN
         IF (IXX.LE.IYY.AND.IXX.LE.IZZ) THEN
            DO 104 IE=1,NL
               XCG(IE)=CG(1,IE)
               YCG(IE)=CG(2,IE)
               ZCG(IE)=CG(3,IE)
  104       CONTINUE
         ELSEIF (IYY.LE.IXX.AND.IYY.LE.IZZ) THEN
            DO 106 IE=1,NL
               XCG(IE)=CG(2,IE)
               YCG(IE)=CG(3,IE)
               ZCG(IE)=CG(1,IE)
  106       CONTINUE
         ELSEIF (IZZ.LE.IXX.AND.IZZ.LE.IYY) THEN
            DO 108 IE=1,NL
               XCG(IE)=CG(3,IE)
               YCG(IE)=CG(1,IE)
               ZCG(IE)=CG(2,IE)
  108       CONTINUE
         ENDIF
      ELSE
C     2-D:
         IF (IXX.LE.IYY) THEN
            DO 114 IE=1,NL
               XCG(IE)=CG(1,IE)
               YCG(IE)=CG(2,IE)
  114       CONTINUE
         ELSE
            DO 116 IE=1,NL
               XCG(IE)=CG(2,IE)
               YCG(IE)=CG(1,IE)
  116       CONTINUE
         ENDIF
      ENDIF
      call col2(xcg,wgt,nl)
      call col2(ycg,wgt,nl)
      call col2(zcg,wgt,nl)
C
C     Find median value of CG to determine dividing point:
C
      XM=FMDIAN(XCG,NL,IFOK)
      YM=FMDIAN(YCG,NL,IFOK)
      ZM=0.0
      IF (IF3D) ZM=FMDIAN(ZCG,NL,IFOK)
C
C     Diagnostics
C
      IF (.NOT.IFOK) THEN
         WRITE(6,130) NID,NL,XM,YM,ZM
         DO 120 IL=1,NL
            WRITE(6,135) NID,IL,XCG(IL),YCG(IL),ZCG(IL)
  120    CONTINUE
  130    FORMAT(I10,'DIVIDE: NL,XM,YM,ZM',I3,3F12.5)
  135    FORMAT(I10,'DIVIDE: NID,IL,XC,YC,ZCG',I4,3F12.5)
      ENDIF
C
C=============================================================
C     Divide LIST into LIST1 (XCG < XM) and LIST2 (XCG>XM).
C=============================================================
C
      NL1=0
      NL2=0
      DO 200 IE=1,NL
         IF (XCG(IE).LT.XM) THEN
            NL1=NL1+1
            LIST1(NL1)=LIST(IE)
         ENDIF
         IF (XCG(IE).GT.XM) THEN
            NL2=NL2+1
            LIST2(NL2)=LIST(IE)
         ENDIF
         IF (XCG(IE).EQ.XM) THEN
C
C           We have to look at the other directions to arrive at
C           a unique subdivision algortithm.
C
C
C           More Diagnostics
C
            IF (.NOT.IFOK) WRITE(6,201) NID,IE,XCG(IE),XM
  201    FORMAT(I10,'DIVIDE: IE,XCG,XM:',I4,3F12.5)
C
            IF (YCG(IE).LT.YM) THEN
               NL1=NL1+1
               LIST1(NL1)=LIST(IE)
            ENDIF
            IF (YCG(IE).GT.YM) THEN
               NL2=NL2+1
               LIST2(NL2)=LIST(IE)
            ENDIF
            IF (YCG(IE).EQ.YM) THEN
C              look at 3rd direction.  
               IF (IF3D .AND. ZCG(IE).LT.ZM) THEN
                  NL1=NL1+1
                  LIST1(NL1)=LIST(IE)
               ELSE IF (IF3D .AND. ZCG(IE).GT.ZM) THEN
                  NL2=NL2+1
                  LIST2(NL2)=LIST(IE)
               ELSE 
C                 for 2- or 3-D intdeterminate case:
                  NL1=NL1+1
                  LIST1(NL1)=LIST(IE)
               ENDIF
            ENDIF
C
         ENDIF
  200 CONTINUE
C
C     Check for an even distribution (i.e. - not different by
C     more than 1):
C
      IFOK=.TRUE.  
      WORK(1)=NL1 
      WORK(2)=NL2
      CALL IGOP(WORK,WRK2,'+  ',2)
      IF (ABS(WORK(1)-WORK(2)).GT.1) IFOK=.FALSE.
C
      return
      END
c-----------------------------------------------------------------------
      subroutine bin_rd1(ifbswap)  ! read mesh, curve, and bc info

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      logical ifbswap

  
      etime1 = dnekclock()

                  ibc = 2
      if (ifflow) ibc = 1

                  nfldt = 1
      if (ifheat) nfldt = 2+npscal
      if (ifmhd ) nfldt = 2+npscal+1

c
c     If p32 = 0.1, there will be no bcs read in
c
      if (param(32).gt.0) nfldt = ibc + param(32)-1

      lcbc=18*lelt*(ldimt1 + 1)
      call blank(cbc,lcbc)

      if (nio.eq.0) write(6,*)    '  reading mesh '
      call bin_rd1_mesh  (ifbswap)   ! version 1 of binary reader
      if (nio.eq.0) write(6,*) '  reading curved sides '
      call bin_rd1_curve (ifbswap)

      do ifield = ibc,nfldt
         if (nio.eq.0) write(6,*) '  reading bc for ifld',ifield
         call bin_rd1_bc (cbc(1,1,ifield),bc(1,1,1,ifield),ifbswap)
      enddo

      call nekgsync
      ierr=0
      if(nid.eq.0) then
        call byte_close(ierr)
        write(6,*) 'done :: read .re2 file'
        write(6,*) ' '
      endif
      call err_chk(ierr,'Error closing re2 file. Abort $')

      return
      end
c-----------------------------------------------------------------------
      subroutine buf_to_xyz(buf,e,ifbswap,ierr)! version 1 of binary reader

      include 'SIZE'
      include 'TOTAL'
      logical ifbswap

      integer e,eg,buf(0:49)

      nwds = (1 + ndim*(2**ndim))*(wdsizi/4) ! group + 2x4 for 2d, 3x8 for 3d

      if     (ifbswap.and.ierr.eq.0.and.wdsizi.eq.8) then
          call byte_reverse8(buf,nwds,ierr)
      elseif (ifbswap.and.ierr.eq.0.and.wdsizi.eq.4) then
          call byte_reverse (buf,nwds,ierr)
      endif
      if(ierr.ne.0) return

      if(wdsizi.eq.8) then
         call copyi4(igroup(e),buf(0),1) !0-1
         if (ndim.eq.3) then
            call copy  (xc(1,e),buf( 2),8) !2 --17
            call copy  (yc(1,e),buf(18),8) !18--33
            call copy  (zc(1,e),buf(34),8) !34--49
         else
            call copy  (xc(1,e),buf( 2),4) !2 --9
            call copy  (yc(1,e),buf(10),4) !10--17
          endif
      else
         igroup(e) = buf(0)
         if (if3d) then
            call copy4r(xc(1,e),buf( 1),8)
            call copy4r(yc(1,e),buf( 9),8)
            call copy4r(zc(1,e),buf(17),8)
         else
            call copy4r(xc(1,e),buf( 1),4)
            call copy4r(yc(1,e),buf( 5),4)
         endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine buf_to_curve(buf)    ! version 1 of binary reader

      include 'SIZE'
      include 'TOTAL'

      integer e,eg,f,buf(30)

      if(wdsizi.eq.8) then
        call copyi4(eg,buf(1),1) !1-2
        e  = gllel(eg)

        call copyi4(f,buf(3),1) !3-4

        call copy  ( curve(1,f,e),buf(5) ,5) !5--14
        call chcopy(ccurve(  f,e),buf(15),1)!15
      else
        eg = buf(1)
        e  = gllel(eg)
        f  = buf(2)

        call copy4r( curve(1,f,e),buf(3),5)
        call chcopy(ccurve(f,e)  ,buf(8),1)
      endif

c     write(6,1) eg,e,f,(curve(k,f,e),k=1,5),ccurve(f,e)
c   1 format(2i7,i3,5f10.3,1x,a1,'ccurve')

      return
      end
c-----------------------------------------------------------------------
      subroutine buf_to_bc(cbl,bl,buf)    ! version 1 of binary reader

      include 'SIZE'
      include 'TOTAL'

      character*3 cbl(6,lelt)
      real         bl(5,6,lelt)

      integer e,eg,f,buf(30)

      if(wdsizi.eq.8) then
        call copyi4(eg,buf(1),1) !1-2
        e  = gllel(eg)

        call copyi4(f,buf(3),1) !3-4

        call copy  (bl(1,f,e),buf(5),5) !5--14
        call chcopy(cbl( f,e),buf(15),3)!15-16

        if(nelt.ge.1000000.and.cbl(f,e).eq.'P  ')
     $   call copyi4(bl(1,f,e),buf(5),1) !Integer assign connecting P element

      else
        eg = buf(1)
        e  = gllel(eg)
        f  = buf(2)

        call copy4r ( bl(1,f,e),buf(3),5)
        call chcopy (cbl(  f,e),buf(8),3)

        if (nelgt.ge.1 000 000.and.cbl(f,e).eq.'P  ') 
     $     bl(1,f,e) = buf(3) ! Integer assign of connecting periodic element
      endif



c     write(6,1) eg,e,f,cbl(f,e),' CBC',nid
c  1  format(2i8,i4,2x,a3,a4,i8)

      return
      end
c-----------------------------------------------------------------------
      subroutine bin_rd1_mesh(ifbswap)    ! version 1 of binary reader

      include 'SIZE'
      include 'TOTAL'
      logical ifbswap

      integer e,eg,buf(55)

      nwds = (1 + ndim*(2**ndim))*(wdsizi/4) ! group + 2x4 for 2d, 3x8 for 3d
      len  = 4*nwds                          ! 4 bytes / wd

      if (nwds.gt.55.or.isize.gt.4) then
         write(6,*) nid,' Error in bin_rd1_mesh: buf size',nwds,isize
         call exitt
      endif

      call nekgsync()

      niop = 10
      do k=1,8
         if (nelgt/niop .lt. 100) goto 10
         niop = niop*10
      enddo
   10 continue

      ierr  = 0
      ierr2 = 0
      len1  = 4
      do eg=1,nelgt             ! sync NOT needed here

         mid = gllnid(eg)
         e   = gllel (eg)
!     tag for sending and receiving changed from global (eg) to local (e) element number
!     to avoid problems with MPI_TAG_UB on CRAY
#ifdef DEBUG
         if (nio.eq.0.and.mod(eg,niop).eq.0) write(6,*) eg,' mesh read'
#endif
         if (mid.ne.nid.and.nid.eq.0) then              ! read & send

            if(ierr.eq.0) then
              call byte_read  (buf,nwds,ierr)
              call csend(e,ierr,len1,mid,0)
              if(ierr.eq.0) call csend(e,buf,len,mid,0)
            else
              call csend(e,ierr,len1,mid,0)
            endif

         elseif (mid.eq.nid.and.nid.ne.0) then          ! recv & process

            call crecv      (e,ierr,len1)
            if(ierr.eq.0) then
              call crecv      (e,buf,len)
              call buf_to_xyz (buf,e,ifbswap,ierr2)
            endif
 
         elseif (mid.eq.nid.and.nid.eq.0) then          ! read & process

            if(ierr.eq.0) then
              call byte_read  (buf,nwds,ierr)
              call buf_to_xyz (buf,e,ifbswap,ierr2)
            endif
         endif

      enddo
      ierr = ierr + ierr2
      call err_chk(ierr,'Error reading .re2 mesh. Abort. $')

      return
      end
c-----------------------------------------------------------------------
      subroutine bin_rd1_curve (ifbswap) ! v. 1 of curve side reader

      include 'SIZE'
      include 'TOTAL'
      logical ifbswap

      integer e,eg,buf(55)
      real rcurve

      nwds = (2 + 1 + 5)*(wdsizi/4) !eg+iside+ccurve+curve(6,:,:) !only 5 in rea
      len  = 4*nwds      ! 4 bytes / wd

      if (nwds.gt.55.or.isize.gt.4) then
         write(6,*)nid,' Error in bin_rd1_curve: buf size',nwds,isize
         call exitt
      endif

      call nekgsync()

      ierr = 0
      len1 = 4
      if (nid.eq.0) then  ! read & send/process

         if(wdsizi.eq.8) then
           call byte_read(rcurve,2,ierr)
           if (ifbswap) call byte_reverse8(rcurve,2,ierr)
           ncurve = rcurve
         else
           call byte_read(ncurve,1,ierr)
           if (ifbswap) call byte_reverse(ncurve,1,ierr)
         endif

         do k=1,ncurve
           if(ierr.eq.0) then
              call byte_read(buf,nwds,ierr)
              if(wdsizi.eq.8) then
                if(ifbswap) call byte_reverse8(buf,nwds-2,ierr)
                call copyi4(eg,buf(1),1)  !1,2
              else
                if (ifbswap) call byte_reverse(buf,nwds-1,ierr) ! last is char
                eg  = buf(1)
              endif

              mid = gllnid(eg)
              if (mid.eq.0.and.ierr.eq.0) then
                 call buf_to_curve(buf)
              else
                 if(ierr.eq.0) then
                   call csend(mid,buf,len,mid,0)
                 else
                   goto 98
                 endif
              endif
           else
              goto 98
           endif
         enddo
  98     call buf_close_out  ! notify all procs: no more data

      else               ! wait for data from node 0

         ncurve_mx = 12*nelt
         do k=1,ncurve_mx+1   ! +1 to make certain we receive the close-out

            call crecv(nid,buf,len)
            if(wdsizi.eq.8) then 
               call copyi4(ichk,buf(1),1)
               if(ichk.eq.0) goto 99
               call buf_to_curve(buf)
            elseif (buf(1).eq.0) then
               goto 99
            else
               call buf_to_curve(buf)
            endif
            
         enddo
   99    call buf_close_out

      endif
      call err_chk(ierr,'Error reading .re2 curved data. Abort.$')


      return
      end
c-----------------------------------------------------------------------
      subroutine bin_rd1_bc (cbl,bl,ifbswap) ! v. 1 of bc reader

      include 'SIZE'
      include 'TOTAL'
      logical ifbswap

      character*3 cbl(6,lelt)
      real         bl(5,6,lelt)

      integer e,eg,buf(55)
      real rbc_max

      nwds = (2 + 1 + 5)*(wdsizi/4)   ! eg + iside + cbc + bc(5,:,:)
      len  = 4*nwds      ! 4 bytes / wd

      if (nwds.gt.55.or.isize.gt.4) then
         write(6,*) nid,' Error in bin_rd1_bc: buf size',nwds,isize
         call exitt
      endif

      do e=1,nelt   ! fill up cbc w/ default
      do k=1,6
         cbl(k,e) = 'E  '
      enddo
      enddo

      call nekgsync()
      ierr=0
      len1=4
      if (nid.eq.0) then  ! read & send/process
  
         if(wdsizi.eq.8) then
           call byte_read(rbc_max,2,ierr)
           if (ifbswap) call byte_reverse8(rbc_max,2,ierr) ! last is char
           nbc_max = rbc_max
         else
           call byte_read(nbc_max,1,ierr)
           if (ifbswap) call byte_reverse(nbc_max,1,ierr) ! last is char
         endif

         do k=1,nbc_max
c           write(6,*) k,' dobc1 ',nbc_max
            if(ierr.eq.0) then
               call byte_read(buf,nwds,ierr)
               if(wdsizi.eq.8) then
                 if (ifbswap) call byte_reverse8(buf,nwds-2,ierr)
                 call copyi4(eg,buf(1),1) !1&2 of buf
               else
                 if (ifbswap) call byte_reverse(buf,nwds-1,ierr) ! last is char
                 eg  = buf(1)
               endif
               mid = gllnid(eg)
c              write(6,*) k,' dobc3 ',eg,mid

               if (mid.eq.0.and.ierr.eq.0) then
                   call buf_to_bc(cbl,bl,buf)
               else
c                  write(6,*) mid,' sendbc1 ',eg
                   if(ierr.eq.0) then
                     call csend(mid,buf,len,mid,0)
                   else
                     goto 98
                   endif
c                  write(6,*) mid,' sendbc2 ',eg
               endif
c              write(6,*) k,' dobc2 ',nbc_max,eg
            else
               goto 98
            endif
         enddo
c        write(6,*) mid,' bclose ',eg,nbc_max
  98     call buf_close_outv ! notify all procs: no more data

      else               ! wait for data from node 0

         nbc_max = 2*ndim*nelt
         do k=1,nbc_max+1  ! Need one extra !

c           write(6,*) nid,' recvbc1',k
            call crecv(nid,buf,len)
c           write(6,*) nid,' recvbc2',k,buf(1)

            if(wdsizi.eq.8) then 
               call copyi4(ichk,buf(1),1)
               if(ichk.eq.0) goto 99
               call buf_to_bc(cbl,bl,buf)
            elseif (buf(1).eq.0) then
                goto 99
            else
                call buf_to_bc(cbl,bl,buf)
            endif
            
         enddo
   99    call buf_close_outv

      endif

      call err_chk(ierr,'Error reading boundary data for re2. Abort.$')

      return
      end
c-----------------------------------------------------------------------
      subroutine bufchk(buf,n)
      integer n
      real buf(n)
      do i=1,n
         write(6,*) buf(i), ' whhhh'
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine buf_close_outv  ! this is the stupid O(P) formulation

      include 'SIZE'
      include 'PARALLEL'
      integer*4 zero
      real      rzero

      len   = wdsizi
      rzero = 0
      zero  = 0
c     write(6,*) nid,' bufclose'
      if (nid.eq.0) then
         do mid=1,np-1
            if(wdsizi.eq.8)call csend(mid,rzero,len,mid,0)
            if(wdsizi.eq.4)call csend(mid, zero,len,mid,0)
c           write(6,*) mid,' sendclose'
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine buf_close_out  ! this is the stupid O(P) formulation

      include 'SIZE'
      include 'PARALLEL'
      integer*4 zero
      real      rzero

c     len  = 4
      len   = wdsizi
      zero = 0
      rzero = 0
      if (nid.eq.0) then
         do mid=1,np-1
            if(wdsizi.eq.8)call csend(mid,rzero,len,mid,0)
            if(wdsizi.eq.4)call csend(mid, zero,len,mid,0)
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine open_bin_file(ifbswap) ! open file & chk for byteswap

      include 'SIZE'
      include 'TOTAL'

      logical ifbswap,if_byte_swap_test

      integer fnami (33)
      character*132 fname
      equivalence (fname,fnami)

      character*132 hdr
      character*5 version
      real*4      test

      ierr=0
      if (nid.eq.0) then
         write(6,'(A,A)') ' Reading ', re2fle
         call izero(fnami,33)
         m = indx2(re2fle,132,' ',1)-1
         call chcopy(fname,re2fle,m)
   
         call byte_open(fname,ierr)
         if(ierr.ne.0) goto 100
         call byte_read(hdr,20,ierr)
         if(ierr.ne.0) goto 100

         read (hdr,1) version,nelgt,ndim,nelgv
    1    format(a5,i9,i3,i9)
 
         wdsizi = 4
         if(version.eq.'#v002') wdsizi = 8
         if(version.eq.'#v003') then
           wdsizi = 8
           param(32) = 1
         endif

         call byte_read(test,1,ierr)
         if(ierr.ne.0) goto 100
         ifbswap = if_byte_swap_test(test,ierr)
         if(ierr.ne.0) goto 100

      endif
 
 100  call err_chk(ierr,'Error opening or reading .re2 header. Abort.$')

      call bcast(wdsizi, ISIZE)
      call bcast(ifbswap,LSIZE)
      call bcast(nelgv  ,ISIZE)
      call bcast(ndim   ,ISIZE)
      call bcast(nelgt  ,ISIZE)
      call bcast(param(32),WDSIZE)

      if(wdsize.eq.4.and.wdsizi.eq.8) 
     $   call exitti('wdsize=4 & wdsizi(re2)=8 not compatible$',wdsizi)

      return
      end
c-----------------------------------------------------------------------
      subroutine chk_xyz
      include 'SIZE'
      include 'TOTAL'
      integer e,f,eg

      do e=1,nelt
         eg = lglel(e)
         write(6,1) nid,eg,e,(cbc(f,e,1),f=1,6)
      enddo
    1 format(3i12,6(1x,a3),'  cbc')

      return
      end
c-----------------------------------------------------------------------
      subroutine chk_nel
      include 'SIZE'
      include 'TOTAL'

      neltmx=np*lelt
      nelvmx=np*lelv

      neltmx=min(neltmx,lelg)
      nelvmx=min(nelvmx,lelg)

      nelgt = iglmax(nelgt,1)
      nelgv = iglmax(nelgv,1)

c     write(6,*) nid,' inside chk_nel',nelgt,neltmx,nelvmx

      if (nelgt.gt.neltmx.or.nelgv.gt.nelvmx) then
         if (nid.eq.0) then
          lelt_needed = nelgt/np
          if (mod(nelgt,np).ne.0) lelt_needed = lelt_needed + 1 
          write(6,12) lelt,lelg,lelt_needed,np,nelgt
 12         format(//,2X,'ABORT: Problem size too large!'
     $         ,/,2X
     $         ,/,2X,'This solver has been compiled for:'
     $         ,/,2X,'   number of elements/proc  (lelt):',i12
     $         ,/,2X,'   total number of elements (lelg):',i12
     $         ,/,2X
     $         ,/,2X,'Recompile with the following SIZE  parameters:'
     $         ,/,2X,'   lelt >= ',i12,'  for np = ',i12
     $         ,/,2X,'   lelg >= ',i12,/)
c           write(6,*)'help:',lp,np,nelvmx,nelgv,neltmx,nelgt
c           write(6,*)'help:',lelt,lelv,lelgv
         endif
         call exitt
      endif

      if(nelgt.gt.nelgt_max) then
        if(nid.eq.0) write(6,*)
     $               'ABORT: Total number of elements too large!',
     $               '       nel_max = ', nelgt_max 
        call exitt
      endif

      if (nelt.gt.lelt) then
        write(6,'(A,3I12)') 'ABORT: nelt>lelt!', nid, nelt, lelt
        call exitt
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cscan(sout,key,nk)

      character*132 sout,key
      character*132 string
      character*1  string1(132)
      equivalence (string1,string)
c
      do i=1,100000000
         call blank(string,132)
         read (nk,80,end=100,err=100) string
         call chcopy(sout,string,132)
c        write (6,*) string
         if (indx1(string,key,nk).ne.0) return
      enddo
  100 continue
c
   80 format(a132)
      return

      end
c-----------------------------------------------------------------------
