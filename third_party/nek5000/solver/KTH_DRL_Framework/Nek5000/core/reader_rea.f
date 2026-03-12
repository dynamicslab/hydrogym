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

      character*132 string(100)

      VNEKTON = 3 ! dummy not really used anymore

      optlevel = 1! fixed for now
      loglevel = 1! fixed for now
      
      IF(NID.EQ.0) THEN
        READ(9,*,ERR=400)
        READ(9,*,ERR=400)
        READ(9,*,ERR=400) ldimr
        READ(9,*,ERR=400) NPARAM
        DO 20 I=1,NPARAM
           READ(9,*,ERR=400) PARAM(I)
   20   CONTINUE
      ENDIF
      call bcast(ldimr,ISIZE)
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

c     Use same tolerances for all fields 
      restol(0) = param(22) ! mesh
      restol(1) = param(22)
      do i=1,ldimt
         restol(1+i) = param(22)
      enddo
      call bcast(restol, (ldimt1+1)*wdsize)
   
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
      IFTMSH(0)    = .false.
      IFPROJFLD(0) = .false.
      IFDGFLD(0)   = .false.
      do i=1,NPSCL2
         IFTMSH(i)    = .false.
         IFADVC(i)    = .false. 
         IFDGFLD(i)   = .false.
         IFFILTER(i)  = .false.
         IFDIFF(i)    = .true.
         IFDEAL(i)    = .true. ! still depends on param(99)
         IFPROJFLD(i) = .false. 
         if (param(94).gt.0) IFPROJFLD(i) = .true. 
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

      ifreguo = .false.   ! by default we dump the data based on the GLL mesh

      ifrich = .false.

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
C
C     Do some checks
C
      IF(ldimr.NE.LDIM)THEN
         IF(NID.EQ.0) THEN
           WRITE(6,10) LDIM,ldimr
   10      FORMAT(//,2X,'ERROR: This NEKTON Solver has been compiled'
     $             /,2X,'       for spatial dimension equal to',I2,'.'
     $             /,2X,'       The data file has dimension',I2,'.')
         ENDIF
         call exitt
      ENDIF
      IF (ldim.EQ.3) IF3D=.TRUE.
      IF (ldim.NE.3) IF3D=.FALSE.

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

      if (param(40).eq.3 .and. .not.ifsplit) then
         call exitti
     $    ('ERROR: Selected preconditioner requires lx2=lx1$',lx2)
      endif

      if (ifcvode) then 
         if(nid.eq.0) write(6,*) 
     $   'ABORT: Using CVODE requires .par file!'
         call exitt
      endif

      if (ifsplit .and. ifuservp .and. .not.ifstrs) then
         if(nid.eq.0) write(6,*) 
     $   'Enable stress formulation to support PN/PN and IFUSERVP=T'    
         ifstrs = .true.
      endif

      if (ifcyclic .and. .not.ifstrs) then
         if(nid.eq.0) write(6,*) 
     $   'Enable stress formulation to support cyclic BC'               
         ifstrs = .true.
      endif

      ktest = (lx1-lx1m) + (ly1-ly1m) + (lz1-lz1m)
      if (ifstrs.and.ktest.ne.0) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: Stress formulation requires lx1m=lx1, etc. in SIZE'
         call exitt
      endif

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

      if (ifneknekc.and.(nelgv.ne.nelgt)) call exitti(
     $ 'ABORT: nek-nek not supported w/ conj. ht transfer$',1)

      if (ifchar.and.(nelgv.ne.nelgt)) call exitti(
     $ 'ABORT: IFCHAR curr. not supported w/ conj. ht transfer$',nelgv)

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

      if (.not.ifsplit .and. ifaxis .and. ifstrs) then
        if (nid.eq.0) write(6,*)
     $    'ABORT: Axisymetric and stress formulation not supported ' //
     $    'for PN/PN-2$'
        call exitt
      endif

      if (param(99).gt.-1 .and. (lxd.lt.lx1 .or. lyd.lt.ly1 .or.
     &   lzd.lt.lz1)) then
         if(nid.eq.0) write(6,*)
     &   'ABORT: Dealiasing space too small; Check lxd,lyd,lzd in SIZE '
         call exitt
      endif

c     SET PRESSURE SOLVER DEFAULTS, ADJUSTED IN USR FILE ONLY
      param(41) = 0 ! use additive SEMG
                ! 1 use hybrid SEMG (not yet working... but coming soon!)
      param(42) = 0 ! use GMRES for iterative solver w/ nonsymmetric weighting
                ! 1 use PCG for iterative solver, do not use weighting
      param(43) = 0 ! use additive multilevel scheme (requires param(42).eq.0)
                ! 1 use original 2 level scheme
      param(44) = 0 ! base top-level additive Schwarz on restrictions of E
                ! 1 base top-level additive Schwarz on restrictions of A

c     SET DEFAULT TO 6, ADJUSTED IN USR FILE ONLY
      param(66) = 6
      param(67) = 6

      param(59) = 1 ! No fast operator eval, ADJUSTED IN USR FILE ONLY
      param(33) = 0

      fem_amg_param(1) = 0
      crs_param(1) = 0

      filterType = 0
      if (param(103).gt.0) then 
         filterType = 1
         call ltrue(iffilter,size(iffilter)) 
      endif

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

      NSIDES=ldim*2
      DO 40 IEG=1,NELGT
         IF (GLLNID(IEG).EQ.NID) THEN
            IEL=GLLEL(IEG)

            igroup(iel) = 0
c            read(9,30,err=31,end=600) igroup(iel)
c   30       format(43x,i5)
            read(9,*,err=31,end=600) adum
   31       continue

C           Read Corner data
            IF(ldim.EQ.2)THEN
               READ(9,*,ERR=500,END=600) (XC(IC,IEL),IC=1,4)
               READ(9,*,ERR=500,END=600) (YC(IC,IEL),IC=1,4)
                              call rzero (zc(1 ,iel)     ,4)
            ELSE IF(ldim.EQ.3)THEN
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
            IF(ldim.EQ.2)THEN
               READ(9,41,ERR=500,END=600) ADUM
               READ(9,41,ERR=500,END=600) ADUM
            ELSE IF(ldim.EQ.3)THEN
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
      NSIDES    = 2*ldim
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
      ierr=0
      if(nid.eq.0) then
c       read history data
        read (9,*)
        read (9,*,err=200,end=200) nhis
        do i = 1,nhis   
           read (9,*)
        enddo
      endif

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

