c-----------------------------------------------------------------------
      subroutine setics
C-----------------------------------------------------------------------
C
C     Set initial conditions.
C
C-----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'DEALIAS'
      INCLUDE 'INPUT'
      INCLUDE 'IXYZ'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'MASS'
      INCLUDE 'MVGEOM'
      INCLUDE 'PARALLEL'
      INCLUDE 'TSTEP'
 
      logical  iffort(  ldimt1,0:lpert)
     $       , ifrest(0:ldimt1,0:lpert)
     $       , ifprsl(  ldimt1,0:lpert)
 
      LOGICAL  IFANYP
      common /rdump/ ntdump
      common /inelr/ nelrr
      common /ctmp1/ work(lx1,ly1,lz1,lelv)
     $ ,             ta1 (lx2,ly1,lz1)
     $ ,             ta2 (lx2,ly2,lz1)
      integer*8 ntotg,nn

      real psmax(ldimt)

      if(nio.eq.0) write(6,*) 'set initial conditions'


      nxyz2=nx2*ny2*nz2       ! Initialize all fields:
      ntot2=nxyz2*nelv
      nxyz1=nx1*ny1*nz1
      ntott=nelt*nxyz1
      ntotv=nelv*nxyz1
      ltott=lelt*nxyz1

      call rzero(vx,ntott)
      call rzero(vy,ntott)
      call rzero(vz,ntott)
      call rzero(pr,nxyz2*nelt)
      do 10 ifld=1,ldimt
         call rzero(t(1,1,1,1,ifld),ntott)
   10 continue

      jp = 0                  ! Set counter for perturbation analysis

      irst = param(46)        ! for lee's restart (rarely used)
      if (irst.gt.0)  call setup_convect(2)

c     If moving geometry then add a perturbation to the
c     mesh coordinates (see Subroutine INIGEOM)

      if (ifmvbd) call ptbgeom

C     Find out what type of i.c. is requested
C     Current options: 
C
C     (1) - User specified fortran function (default is zero i.c.)
C     (2) - Restart from file(s)
C     (3) - Activate pre-solver => steady diffusion / steady Stokes
C
C     If option (2) is requested, also return with the name of the
C     restart file(s) together with the associated dump number

      call slogic (iffort,ifrest,ifprsl,nfiles)

C     Set up proper initial values for turbulence model arrays
      IF (IFMODEL) CALL PRETMIC

C      ***** TEMPERATURE AND PASSIVE SCALARS ******
C
C     Check if any pre-solv necessary for temperature/passive scalars

      IFANYP = .FALSE.
      DO 100 IFIELD=2,NFIELD
         IF (IFPRSL(IFIELD,jp)) THEN
            IF (NIO.EQ.0) WRITE(6,101) IFIELD
            IFANYP = .TRUE.
         ENDIF
  100 CONTINUE
  101 FORMAT(2X,'Using PRESOLVE option for field',I2,'.')

C     Fortran function initial conditions for temp/pass. scalars.
      maxfld = nfield
      if (ifmodel.and.ifkeps) maxfld = nfield-2
      if (ifmhd) maxfld = npscal+3

c     Always call nekuic (pff, 12/7/11)
      do ifield=1,maxfld
         if (nio.eq.0) write(6,*) 'nekuic (1) for ifld ', ifield
         call nekuic
      enddo

C     If any pre-solv, do pre-solv for all temperatur/passive scalar fields
      if (ifanyp) call prsolvt

      jp = 0 ! jp=0 --> base field, not perturbation field
      do 200 ifield=2,maxfld
         if (iffort(ifield,jp)) then
            if (nio.eq.0) write(6,*) 'call nekuic for ifld ', ifield
            call nekuic
         endif
 200  continue

      if (ifpert) then
         ifield=2
         do jp=1,npert
         if (nio.eq.0) write(6,*) 'nekuicP',ifield,jp,iffort(ifield,jp)
            if (iffort(ifield,jp)) call nekuic
         enddo
      endif
      jp = 0
     

      call nekgsync()
      call restart(nfiles) !  Check restart files
      call nekgsync()


C      ***** VELOCITY ******
C     (If restarting for V, we're done,
C     ...else, do pre-solv for fluid if requested.)

      ifield = 1
      if (ifprsl(ifield,jp)) call prsolvv


C     Fortran function initial conditions for velocity.
      ifield = 1
      if (iffort(ifield,jp)) then
         if (nio.eq.0) write(6,*) 'call nekuic for vel  '
         call nekuic
      endif
c
      if (ifpert) then
         ifield=1
         do jp=1,npert
            if (iffort(ifield,jp)) call nekuic
            if (nio.eq.0) write(6,*) 'ic vel pert:',iffort(1,jp),jp
         enddo
      endif
      jp = 0

      ntotv = nx1*ny1*nz1*nelv

C     Fortran function initial conditions for turbulence k-e model
      if (ifmodel .and. ifkeps) then
         mfldt = nfield - 1
         do 300 ifield=mfldt,nfield
            if (iffort(ifield,jp)) call nekuic
 300     continue
      endif

C     Initial mesh velocities
      if (ifmvbd) call opcopy (wx,wy,wz,vx,vy,vz)
      if (ifmvbd.and..not.ifrest(0,jp)) call meshv (2)

C     Compute additional initial values for turbulence model arrays
C     based on I.C.
      if (ifmodel) call postmic

C     If convection-diffusion of a passive scalar with a fixed velocity field,
C     make sure to fill up lagged arrays since this will not be done in
C     the time-stepping procedure (no flow calculation) (01/18/91 -EMR).

      if (.not.ifflow.and.ifheat) then
         ITEST=0
         DO 400 IFIELD=2,NFIELD
            IF (IFADVC(IFIELD)) ITEST=1
 400     CONTINUE
         IF (ITEST.EQ.1) THEN
            NBDMAX = 3
            NBDSAV = NBDINP
            NBDINP = NBDMAX
            DO 500 I=1,NBDMAX
               CALL LAGVEL
 500        CONTINUE
            NBDINP = NBDSAV
         ENDIF
      ENDIF
     
C     Ensure that all processors have the same time as node 0.
      if (nid.ne.0) time=0.0
      time=glsum(time,1)
      ntdump=0
      if (timeio.ne.0.0) ntdump = int( time/timeio )

C     Ensure that initial field is continuous!

      nxyz1=nx1*ny1*nz1
      ntott=nelt*nxyz1
      ntotv=nelv*nxyz1
      nn = nxyz1
      ntotg=nelgv*nn

      if (.not.ifdg) then
         ifield = 2
         if (ifflow) ifield = 1
         call rone(work,ntotv)
         ifield = 1
         call dssum  (work,nx1,ny1,nz1)
         call col2   (work,vmult,ntotv)
         rdif  = glsum(work,ntotv)
         rtotg = ntotg
         rdif  = (rdif-rtotg)/rtotg
         if (abs(rdif).gt.1e-14) then
            if(nid.eq.0)write(*,*)'ERROR: dssum test has failed!',rdif
            call exitt
         endif
      endif

      vxmax = glamax(vx,ntotv)
      vymax = glamax(vy,ntotv)
      vzmax = glamax(vz,ntotv)
      prmax = glamax(pr,ntot2)

      ntot = nxyz1*nelfld(2)
      ttmax = glamax(t ,ntot)

      do i=1,NPSCAL
         ntot = nx1*ny1*nz1*nelfld(i+2)
         psmax(i) = glamax(T(1,1,1,1,i+1),ntot)
      enddo

c     small=1.0E-20      ! THIS WAS ONCE IMPORTANT ON CERTAIN ARCHITECTURES.
c     ifldsave = ifield  ! IT AVOIDS DENORMALIZED ARITHMETIC.
c     if (vxmax.eq.0.0) call perturb(vx,1,small)
c     if (vymax.eq.0.0) call perturb(vy,1,small)
c     if (vzmax.eq.0.0) call perturb(vz,1,small)
c     if (prmax.eq.0.0.and.ifsplit) call perturb(pr,1,small)
c     if (ttmax.eq.0.0) call perturb(t ,2,small)
c     do i=1,npscal
c        ntot = nxyz1*nelfld(i+2)
c        if(psmax(i).eq.0) call perturb(t(1,1,1,1,1+i),i+2,small)
c     enddo
c     ifield = ifldsave
    
      if (ifflow.and..not.ifdg)  then  ! Current dg is for scalars only
         ifield = 1
         call opdssum(vx,vy,vz)
         call opcolv (vx,vy,vz,vmult)
         if (ifsplit) call dsavg(pr)  ! continuous pressure
         if (ifvcor)  call ortho(pr)  ! remove any mean
      endif

c     if (ifmhd.and..not.ifdg) then   ! Current dg is for scalars only
      if (ifmhd) then
         ifield = ifldmhd
         call opdssum(bx,by,bz)
         call opcolv (bx,by,bz,vmult)
      endif

      if (ifheat.and..not.ifdg) then  ! Don't project if using DG
         ifield = 2
         call dssum(t ,nx1,ny1,nz1)
         call col2 (t ,tmult,ntott)
         do ifield=3,nfield
            call dssum(t(1,1,1,1,i-1),nx1,ny1,nz1)
            if(iftmsh(ifield)) then
              call col2 (t(1,1,1,1,i-1),tmult,ntott)
            else
              call col2 (t(1,1,1,1,i-1),vmult,ntotv)
            endif
         enddo
      endif
c
c     if (ifpert.and..not.ifdg) then ! Still not DG
      if (ifpert) then
         do jp=1,npert
            ifield = 1
            call opdssum(vxp(1,jp),vyp(1,jp),vzp(1,jp))
            call opcolv (vxp(1,jp),vyp(1,jp),vzp(1,jp),vmult)
            ifield = 2

c           note... must be updated for addl pass. scal's. pff 4/26/04
            if (.not.ifdg) call dssum(tp(1,1,jp),nx1,ny1,nz1)
            if (.not.ifdg) call col2 (tp(1,1,jp),tmult,ntotv)

            vxmax = glamax(vxp(1,jp),ntotv)
            vymax = glamax(vyp(1,jp),ntotv)
            if (nio.eq.0) write(6,111) jp,vxmax,vymax
  111       format(i5,1p2e12.4,' max pert vel')
         enddo
      endif
      jp = 0

C print min values
      xxmax = glmin(xm1,ntott)
      yymax = glmin(ym1,ntott)
      zzmax = glmin(zm1,ntott)

      vxmax = glmin(vx,ntotv)
      vymax = glmin(vy,ntotv)
      vzmax = glmin(vz,ntotv)
      prmax = glmin(pr,ntot2)

      ntot = nxyz1*nelfld(2)
      ttmax = glmin(t ,ntott)

      do i=1,ldimt-1
         ntot = nxyz1*nelfld(i+2)
         psmax(i) = glmin(T(1,1,1,1,i+1),ntot)
      enddo

      if (nio.eq.0) then
         write(6,19) xxmax,yymax,zzmax
   19    format(' xyz min  ',5g13.5)
      endif
      if (nio.eq.0) then
         write(6,20) vxmax,vymax,vzmax,prmax,ttmax
   20    format(' uvwpt min',5g13.5)
      endif
      if (ldimt-1.gt.0) then
         if (nio.eq.0) write(6,21) (psmax(i),i=1,LDIMT-1)
   21    format(' PS min   ',50g13.5)
      endif

c print max values
      xxmax = glmax(xm1,ntott)
      yymax = glmax(ym1,ntott)
      zzmax = glmax(zm1,ntott)

      vxmax = glmax(vx,ntotv)
      vymax = glmax(vy,ntotv)
      vzmax = glmax(vz,ntotv)
      prmax = glmax(pr,ntot2)

      ntot = nxyz1*nelfld(2)
      ttmax = glmax(t ,ntott)

      do i=1,ldimt-1
         ntot = nxyz1*nelfld(i+2)
         psmax(i) = glmax(T(1,1,1,1,i+1),ntot)
      enddo

      if (nio.eq.0) then
         write(6,16) xxmax,yymax,zzmax
   16    format(' xyz max  ',5g13.5)
      endif

      if (nio.eq.0) then
         write(6,17) vxmax,vymax,vzmax,prmax,ttmax
   17    format(' uvwpt max',5g13.5)
      endif

      if (ldimt-1.gt.0) then
         if (nio.eq.0)  then
            write(6,18) (psmax(i),i=1,ldimt-1)
   18       format(' PS max   ',50g13.5)
         endif
      endif

      if (iflomach .and. ifdp0dt) then
        if (p0th.le.0) call exitti('Invalid thermodynamic pressure!$',1)
        if (gamma0.lt.0) call exitti('Invalid gamma0!$',1)
      endif

      if (ifrest(0,jp)) then !  mesh has been read in.
         if (nio.eq.0) write(6,*) 'Restart: recompute geom. factors.'
         call geom_reset(1)  !  recompute geometric factors
      endif

c     call outpost(vx,vy,vz,pr,t,'   ')
c     call exitti('setic exit$',nelv)

      if(nio.eq.0) then
        write(6,*) 'done :: set initial conditions'
        write(6,*) ' '
      endif

      return
      end
C            
c-----------------------------------------------------------------------
      subroutine slogic (iffort,ifrest,ifprsl,nfiles)
C---------------------------------------------------------------------
C
C     Set up logicals for initial conditions.
C
C---------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'RESTART'
c
      logical  iffort(  ldimt1,0:lpert)
     $       , ifrest(0:ldimt1,0:lpert)
     $       , ifprsl(  ldimt1,0:lpert)
c
      character*132 line,fname,cdum
      character*2  s2
      character*1  line1(132)
      equivalence (line1,line)
C
C     Default is user specified fortran function (=0 if not specified)
C
      nfldt = nfield
      if (ifmhd) nfldt = nfield+1

      do jp=0,npert
         ifrest(0,jp) = .false.
         do ifld=1,nfldt
            iffort(ifld,jp) = .true.
            ifrest(ifld,jp) = .false.
            ifprsl(ifld,jp) = .false.
         enddo
      enddo

      jp = 0
      nfiles=0

c     Check for Presolve options     

      DO 1000 ILINE=1,15 
         LINE=INITC(ILINE)
         CALL CAPIT(LINE,132)
         IF (INDX1(LINE,'PRESOLV',7).NE.0) THEN
C           found a presolve request
            CALL BLANK(INITC(ILINE),132)
            CALL LJUST(LINE)
            CALL CSPLIT(CDUM,LINE,' ',1)
C
            IF (LTRUNC(LINE,132).EQ.0) THEN
               IF (NIO.EQ.0) WRITE(6,700)
  700          FORMAT(/,2X,'Presolve options: ALL')
C              default - all fields are presolved.
               DO 800 IFIELD=1,nfldt
                  ifprsl(ifield,jp) = .true.
                  iffort(ifield,jp) = .false.
  800          CONTINUE
            ELSE
C           check line for arguments
C
               LL=LTRUNC(LINE,132)
               IF (NIO.EQ.0) WRITE(6,810) (LINE1(L),L=1,LL)
  810          FORMAT(/,2X,'Presolve options: ',132A1)
C
               IF (INDX_CUT(LINE,'U',1).NE.0) THEN
                  ifprsl(1,jp) = .true.
                  iffort(1,jp) = .false.
               ENDIF
C
               IF (INDX_CUT(LINE,'T',1).NE.0) THEN
                  ifprsl(2,jp) = .true.
                  iffort(2,jp) = .false.
               ENDIF
C
               DO 900 IFIELD=3,NPSCAL+2
                  IP=IFIELD-2
                  WRITE(S2,901) IP
                  IF (INDX_CUT(LINE,S2,2).NE.0) THEN
                     ifprsl(ifield,jp) = .true.
                     iffort(ifield,jp) = .false.
                  ENDIF
  900          CONTINUE
  901          FORMAT('P',I1)
            ENDIF
         ENDIF
 1000    CONTINUE
C
C     Check for restart options
C
      jp = 0
      DO 2000 ILINE=1,15
         if (ifpert) jp=iline-1
         LINE=INITC(ILINE)
         IF (LTRUNC(LINE,132).NE.0) THEN
C           found a filename
            NFILES=NFILES+1
            INITC(NFILES)=LINE
C
C            IF (NIO.EQ.0.AND.NFILES.EQ.1) WRITE(6,1010) LINE
            IF (NIO.EQ.0.) WRITE(6,1010) LINE
 1010       FORMAT(1X,'Checking restart options: ',A132)
c            IF (NID.EQ.0) WRITE(6,'(A132)') LINE
C
C           Parse restart options
 
            call sioflag(ndumps,fname,line)

            if (ifgetx) then
               ifrest(0,jp) = .true.
            endif
            if (ifgetu) then
               iffort(1,jp) = .false.
               ifprsl(1,jp) = .false.
               ifrest(1,jp) = .true.
            endif
            if (ifgett) then
               iffort(2,jp) = .false.
               ifprsl(2,jp) = .false.
               ifrest(2,jp) = .true.
            endif
            do 1900 ifield=3,nfldt
c              write(6,*) 'ifgetps:',(ifgtps(k),k=1,ldimt-1)
               if (ifgtps(ifield-2)) then
                  iffort(ifield,jp) = .false.
                  ifprsl(ifield,jp) = .false.
                  ifrest(ifield,jp) = .true.
               endif
 1900       continue
         endif
 2000 continue

      return
      end
c-----------------------------------------------------------------------
      subroutine restart(nfiles)
C----------------------------------------------------------------------
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
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      INCLUDE 'RESTART'

      common /inelr/ nelrr

      PARAMETER (LXR=LX1+6)
      PARAMETER (LYR=LY1+6)
      PARAMETER (LZR=LZ1+6)
      PARAMETER (LXYZR=LXR*LYR*LZR)
      PARAMETER (LXYZT=LX1*LY1*LZ1*LELT)
      PARAMETER (LPSC9=LDIMT+9)

      common /scrcg/ pm1(lx1*ly1*lz1,lelv)
      COMMON /SCRNS/ SDUMP(LXYZT,7)
      integer mesg(40)

C     note, this usage of CTMP1 will be less than elsewhere if NELT ~> 9.
      COMMON /CTMP1/ TDUMP(LXYZR,LPSC9)
      real*4         tdump
c
      REAL SDMP2(LXYZT,LDIMT)

c     cdump comes in via PARALLEL (->TOTAL)

      character*30 excoder
      character*1  excoder1(30)
      equivalence (excoder,excoder1)


      character*132 fname
      character*1  fname1(132)
      equivalence (fname1,fname)

      integer       hnami (30)
      character*132 hname
      character*1   hname1(132)
      equivalence  (hname,hname1)
      equivalence  (hname,hnami )

      CHARACTER*132 header

C     Local logical flags to determine whether to copy data or not.
      logical ifok,iffmat
      integer iposx,iposz,iposu,iposw,iposp,ipost,ipsps(ldimt1)

      logical ifbytsw, if_byte_swap_test
      real*4   bytetest

c
      ifok=.false.
      ifbytsw = .false.

      if(nfiles.lt.1) return
      if(nio.eq.0) write(6,*) 'Reading checkpoint data'

c use new reader (only binary support)
      p67 = abs(param(67))
      if (p67.eq.6.0) then
         do ifile=1,nfiles
            call sioflag(ndumps,fname,initc(ifile))
            call mfi(fname,ifile)
         enddo
         if (nid.ne.0) time=0
         time = glmax(time,1) ! Sync time across processors
         return
      endif

c use old reader (for ASCII + old binary support)
      
      if (param(67).lt.1.0) then  ! zero only. should be abs.
         iffmat=.true.  ! ascii
      else
         iffmat=.false. ! binary
      endif

      do 6000 ifile=1,nfiles
        call sioflag(ndumps,fname,initc(ifile))
        ierr = 0
        if (nid.eq.0) then

          if (iffmat) then
            open (unit=91,file=fname,status='old',err=500)
          else
            len= ltrunc(fname,79)
            call izero (hnami,20)
            call chcopy(hname1,fname,len)
c           test for presence of file
            open (unit=91,file=hname
     $           ,form='unformatted',status='old',err=500)
            close(unit=91)
            call byte_open(hname,ierr)
            if(ierr.ne.0) goto 500
          ENDIF
          ifok = .true.
        endif

  500   continue
        call lbcast(ifok)
        if (.not.ifok) goto 5000
         
         ndumps = 1
C
C        Only NODE 0 reads from the disk.
C
         DO 1000 IDUMP=1,NDUMPS

            IF (NID.EQ.0) THEN
                ! read header
               if (iffmat) then
                 ierr = 2
                 if(mod(param(67),1.0).eq.0) then ! old header format
                   if(nelgt.lt.10000) then
                     read(91,91,err=10,end=10)
     $              neltr,nxr,nyr,nzr,rstime,istepr,(excoder1(i),i=1,30)
   91                format(4i4,1x,g13.4,i5,1x,30a1)
                     ierr=0
                   else
                     read(91,92,err=10,end=10)
     $              neltr,nxr,nyr,nzr,rstime,istepr,(excoder1(i),i=1,30)
   92                format(i10,3i4,1P1e18.9,i9,1x,30a1)
                     ierr=0
                   endif
                 else                          ! new head format
                   read(91,'(A132)',err=10,end=10) header
                   read(header,*)
     &                  neltr,nxr,nyr,nzr,rstime,istepr,excoder
                   ierr=0
                 endif
               else
                 if(mod(param(67),1.0).eq.0) then  ! old header format
                   call byte_read(hnami,20,ierr)
                   if(ierr.ne.0) goto 10
                   icase = 2
                   if (nelgt.lt.10000) icase = 1
                   ipass = 0
   93              continue  ! test each possible case  UGLY (7/31/07)
                   if(ipass.lt.2) then
                     ipass = ipass+1
                     if(icase.eq.1) then
                       read(hname,'(4i4,1x,g13.4,i5,1x,30a1)',
     $                     err=94,end=94) 
     $                     neltr,nxr,nyr,nzr,rstime,istepr,
     $                     (excoder1(i),i=1,30)
                       goto 95
                     else
                       read(hname,'(i10,3i4,1P1e18.9,i9,1x,30a1)',
     $                     err=94,end=94)
     $                     neltr,nxr,nyr,nzr,rstime,istepr,
     $                     (excoder1(i),i=1,30)
                       goto 95
                     endif
   94                icase = 3-icase  !  toggle: 2-->1  1-->2
                     goto 93
                   else
                     ierr=2
                     goto 10
                   endif
   95              continue
                 else                         ! new head format
                   call byte_read(header,20,ierr)
                   if(ierr.ne.0) goto 10
                   read(header,*)
     &             neltr,nxr,nyr,nzr,rstime,istepr,excoder
                 endif
                 call byte_read(bytetest,1,ierr)
c                call byte_read2(bytetest,1,ierr)
                 if(ierr.ne.0) goto 10
                 ifbytsw = if_byte_swap_test(bytetest,ierr)
                 if(ierr.ne.0) goto 10
               endif
               mesg(1) = neltr
               mesg(2) = nxr
               mesg(3) = nyr
               mesg(4) = nzr
               write(6,*)  'Read mode: ', param(67)
               write(6,333)'neltr,nxr,nyr,nzr: ', neltr,nxr,nyr,nzr
  333          format(A,i9,3i4)
               call chcopy(mesg(5),excoder1,20)
               len  = 14*isize
            endif
   10       call err_chk(ierr,'Error reading restart header. Abort.$')

            IF (IDUMP.EQ.1) THEN
               len  = 14*isize
               call bcast(mesg,len)
               neltr = mesg(1)
               nxr   = mesg(2)
               nyr   = mesg(3)
               nzr   = mesg(4)
               call   chcopy(excoder1,mesg(5),20)

               call lbcast(ifbytsw)

C              Bounds checking on mapped data.
               IF (NXR.GT.LXR) THEN
                  WRITE(6,20) NXR,NX1
   20             FORMAT(//,2X,
     $            'ABORT:  Attempt to map from',I3,
     $            ' to N=',I3,'.',/,2X,
     $            'NEK5000 currently supports mapping from N+6 or less.'
     $            ,/,2X,'Increase N or LXR in IC.FOR.')
                  CALL EXITT
               ENDIF
C
C              Figure out position of data in file "IFILE"
C
               NOUTS=0
               IPOSX=0
               IPOSY=0
               IPOSZ=0
               IPOSU=0
               IPOSV=0
               IPOSW=0
               IPOSP=0
               IPOST=0
               DO 40 I=1,NPSCAL
                  IPSPS(I)=0
   40          CONTINUE

               IPS = 0
               NPS = 0
               DO 50 I=1, 30
                  IF (excoder1(i).EQ.'X') THEN
                     NOUTS=NOUTS + 1
                     IPOSX=NOUTS
                     NOUTS=NOUTS+1
                     IPOSY=NOUTS
                     IF (IF3D) THEN
                        NOUTS=NOUTS + 1
                        IPOSZ=NOUTS
                     ENDIF
                  ENDIF
                  IF (excoder1(i).EQ.'U') THEN
                     NOUTS=NOUTS + 1
                     IPOSU=NOUTS
                     NOUTS=NOUTS+1
                     IPOSV=NOUTS
                     IF (IF3D) THEN
                        NOUTS=NOUTS + 1
                        IPOSW=NOUTS
                     ENDIF
                  ENDIF
                  IF (excoder1(i).EQ.'P') THEN
                     NOUTS=NOUTS + 1
                     IPOSP=NOUTS
                  ENDIF
                  IF (excoder1(i).EQ.'T') THEN
                     NOUTS=NOUTS + 1
                     IPOST=NOUTS
                  ENDIF
                  IF(mod(param(67),1.0).eq.0.0) THEN
                    i1 = i1_from_char(excoder1(i))
                    if (0.lt.i1.and.i1.lt.10) then
                       if (i1.le.ldimt1) then
                          nouts=nouts + 1
                          ipsps(i1)=nouts
                       else
                          if (nid.eq.0) write(6,2) i1,i,excoder1(i)
   2                      format(2i4,a1,' PROBLEM W/ RESTART DATA')
                       endif
                    endif
                  ELSE
                    IF(excoder1(i).EQ.'S') THEN
                       READ(excoder1(i+1),'(I1)') NPS1
                       READ(excoder1(i+2),'(I1)') NPS0
                       NPS=10*NPS1 + NPS0 
                       DO IS = 1, NPS
                         NOUTS=NOUTS + 1
                         IPSPS(IS)=NOUTS
                       ENDDO
                       GOTO 50
                    ENDIF
                  ENDIF
   50          CONTINUE
 
               IF (NPS.GT.(LDIMT-1)) THEN
                  IF (NID.EQ.0) THEN 
                    WRITE(*,'(A)') 
     &               'ERROR: restart file has a NSPCAL > LDIMT'
                    WRITE(*,'(A,I2)') 
     &               'Change LDIMT in SIZE'
                  ENDIF
                  CALL EXITT
               ENDIF

               lname=ltrunc(fname,132)
               if (nio.eq.0) write(6,61) (fname1(i),i=1,lname)
               if (nio.eq.0) write(6,62) 
     $             iposu,iposv,iposw,iposp,ipost,nps,nouts
   61          FORMAT(/,2X,'Restarting from file ',132A1)
   62          FORMAT(2X,'Columns for restart data U,V,W,P,T,S,N: ',7I4)

C              Make sure the requested data is present in this file....
               if (iposx.eq.0) ifgetx=.false.
               if (iposy.eq.0) ifgetx=.false.
               if (iposz.eq.0) ifgetz=.false.
               if (iposu.eq.0) ifgetu=.false.
               if (iposv.eq.0) ifgetu=.false.
               if (iposw.eq.0) ifgetw=.false.
               if (iposp.eq.0) ifgetp=.false.
               if (ipost.eq.0) ifgett=.false.
               do 65 i=1,npscal
                  if (ipsps(i).eq.0) ifgtps(i)=.false.
   65          continue

C              End of restart file header evaluation.

            ENDIF
C
C           Read the error estimators
C           not supported at the moment => just do dummy reading
C
            ifok = .false.
            IF(NID.EQ.0)THEN
               if (iffmat)
     &             READ(91,'(6G11.4)',END=15)(CDUMP,I=1,NELTR)
               ifok = .true.
            ENDIF
  15        continue
            call lbcast(ifok)
            if(.not.ifok) goto 1600
C
C           Read the current dump, double buffer so that we can
C           fit the data on a distributed memory machine,
C           and so we won't have to read the restart file twice
C           in case of an incomplete data file.
C
            NXYZR = NXR*NYR*NZR
C
C           Read the data
C
            nelrr = min(nelgt,neltr) ! # of elements to _really_read
                                     ! why not just neltr?
            do 200 ieg=1,nelrr
               ifok = .false.
               IF (NID.EQ.0) THEN
                 IF (MOD(IEG,10000).EQ.1) WRITE(6,*) 'Reading',IEG
                 IF (iffmat) THEN
                    READ(91,*,ERR=70,END=70)
     $              ((tdump(IXYZ,II),II=1,NOUTS),IXYZ=1,NXYZR)
                 ELSE
                    do ii=1,nouts
                       call byte_read(tdump(1,II),nxyzr,ierr)
                       if(ierr.ne.0) then
                          write(6,*) "Error reading xyz restart data"
                          goto 70
                       endif
                    enddo
                 ENDIF
                 IFOK=.TRUE.
               ENDIF
C
C              Notify other processors that we've read the data OK.
C
  70           continue
               call lbcast(ifok)
               IF (.NOT.IFOK) GOTO 1600
C
C              MAPDMP maps data from NXR to NX1
C              (and sends data to the appropriate processor.)
C
C              The buffer SDUMP is used so that if an incomplete dump
C              file is found (e.g. due to UNIX io buffering!), then
C              the previous read data stored in VX,VY,.., is not corrupted.
C
               IF (IFGETX) CALL MAPDMP
     $         (SDUMP(1,1),TDUMP(1,IPOSX),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETX) CALL MAPDMP
     $         (SDUMP(1,2),TDUMP(1,IPOSY),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETZ) CALL MAPDMP
     $         (SDUMP(1,3),TDUMP(1,IPOSZ),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETU) CALL MAPDMP
     $         (SDUMP(1,4),TDUMP(1,IPOSU),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETU) CALL MAPDMP
     $         (SDUMP(1,5),TDUMP(1,IPOSV),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETW) CALL MAPDMP
     $         (SDUMP(1,6),TDUMP(1,IPOSW),IEG,NXR,NYR,NZR,ifbytsw)
               IF (IFGETP) CALL MAPDMP
     $         (SDUMP(1,7),TDUMP(1,IPOSP),IEG,NXR,NYR,NZR,ifbytsw)
               if (ifgett) call mapdmp
     $         (SDMP2(1,1),TDUMP(1,IPOST),IEG,NXR,NYR,NZR,ifbytsw)

C              passive scalars
               do 100 ips=1,npscal
                  if (ifgtps(ips)) call mapdmp(sdmp2(1,ips+1)
     $               ,tdump(1,ipsps(ips)),ieg,nxr,nyr,nzr,IFBYTSW)
  100          continue
 
  200       continue
C
C           Successfully read a complete field, store it.
C
            nerr = 0              ! Count number of elements rec'd by nid
            do ieg=1,nelrr
               mid = gllnid(ieg)
               if (mid.eq.nid) nerr = nerr+1
            enddo

            nxyz2=nx2*ny2*nz2
            nxyz1=nx1*ny1*nz1
            ntott=nerr*nxyz1
            ntotv=nerr*nxyz1   ! Problem for differing Vel. and Temp. counts!
                               ! for now we read nelt dataset

            if (ifmhd.and.ifile.eq.2) then
               if (ifgetu) call copy(bx,sdump(1,4),ntott)
               if (ifgetu) call copy(by,sdump(1,5),ntott)
               if (ifgetw) call copy(bz,sdump(1,6),ntott)
               if (ifgetp) then
                 if (nio.eq.0) write(6,*) 'getting restart pressure'
                 if (ifsplit) then
                    call copy( pm,sdump(1,7),ntotv)
                 else
                  do iel=1,nelv
                    iiel = (iel-1)*nxyz1+1
                    call map12 (pm(1,1,1,iel),sdump(iiel,7),iel)
                  enddo
                 endif
               endif
               if (ifaxis.and.ifgett) 
     $            call copy(t(1,1,1,1,2),sdmp2(1,1),ntott)
            elseif (ifpert.and.ifile.ge.2) then
               j=ifile-1  ! pointer to perturbation field
               if (ifgetu) call copy(vxp(1,j),sdump(1,4),ntotv)
               if (ifgetu) call copy(vyp(1,j),sdump(1,5),ntotv)
               if (ifgetw) call copy(vzp(1,j),sdump(1,6),ntotv)
               if (ifgetp) then
                  if (nio.eq.0) write(6,*) 'getting restart pressure'
                  if (ifsplit) then
                     call copy(prp(1,j),sdump(1,7),ntotv)
                  else
                     do ie=1,nelv
                        ie1 = (ie-1)*nxyz1+1
                        ie2 = (ie-1)*nxyz2+1
                        call map12 (prp(ie2,j),sdump(ie1,7),ie)
                     enddo
                  endif
               endif
               if (ifgett) call copy(tp(1,1,j),sdmp2(1,1),ntott)
C              passive scalars
               do ips=1,NPSCAL
                  if (ifgtps(ips))
     $            call copy(tp(1,ips+1,j),sdmp2(1,ips+1),ntott)
               enddo

            else  ! Std. Case
               if (ifgetx) call copy(xm1,sdump(1,1),ntott)
               if (ifgetx) call copy(ym1,sdump(1,2),ntott)
               if (ifgetz) call copy(zm1,sdump(1,3),ntott)
               if (ifgetu) call copy(vx ,sdump(1,4),ntotv)
               if (ifgetu) call copy(vy ,sdump(1,5),ntotv)
               if (ifgetw) call copy(vz ,sdump(1,6),ntotv)
               if (ifgetp) call copy(pm1,sdump(1,7),ntotv)
               if (ifgett) call copy(t,sdmp2(1,1),ntott)
C              passive scalars
               do i=1,NPSCAL
                  if (ifgtps(i))
     $            call copy(t(1,1,1,1,i+1),sdmp2(1,i+1),ntott)
               enddo

               if (ifaxis) call axis_interp_ic(pm1)      ! Interpolate to axi mesh
               if (ifgetp) call map_pm1_to_pr(pm1,ifile) ! Interpolate pressure

               if (ifgtim) time=rstime
            endif

 1000    CONTINUE
         GOTO 1600
 
 1600    CONTINUE
C
         IF (IDUMP.EQ.1.AND.NID.EQ.0) THEN
            write(6,1700) fname
            write(6,1701) ieg,ixyz
            write(6,1702) 
     $            ((tdump(jxyz,ii),ii=1,nouts),jxyz=ixyz-1,ixyz)
 1700       FORMAT(5X,'WARNING:  No data read in for file ',A132)
 1701       FORMAT(5X,'Failed on  element',I4,',  point',I5,'.')
 1702       FORMAT(5X,'Last read dump:',/,5G15.7)
            write(6,*) nid,'call exitt 1702a',idump
            call exitt
         ELSEIF (IDUMP.EQ.1) THEN
            write(6,*) nid,'call exitt 1702b',idump
            call exitt
         ELSE
            IDUMP=IDUMP-1
            IF (NIO.EQ.0) WRITE(6,1800) IDUMP
 1800       FORMAT(2X,'Successfully read data from dump number',I3,'.')
         ENDIF
         if (iffmat) then
            if (nid.eq.0) close(unit=91)
         else
            ierr = 0
            if (nid.eq.0) call byte_close(ierr)
            call err_chk(ierr,'Error closing restart file in restart$')
         endif
         GOTO 6000

C        Can't open file...
 5000    CONTINUE
         if (nid.eq.0) write(6,5001) fname 
 5001    FORMAT(2X,'   *******   ERROR   *******    '
     $       ,/,2X,'   *******   ERROR   *******    '
     $       ,/,2X,'   Could not open restart file:'
     $       ,/,A132
     $      ,//,2X,'Quitting in routine RESTART.')
         CLOSE(UNIT=91)
         call exitt
 5002    CONTINUE
         IF (NIO.EQ.0) WRITE(6,5001) HNAME 
         call exitt
C
C
C     End of IFILE loop
 6000 CONTINUE
C
      return
      end
C
c-----------------------------------------------------------------------
      subroutine sioflag(ndumps,fname,rsopts)
C
C     Set IO flags according to Restart Options File, RSOPTS
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'RESTART'
      INCLUDE 'TSTEP'

      character*132 rsopts,fname
      character*2  s2
      logical ifgtrl

C     Scratch variables..
      logical ifdeft,ifanyc
      CHARACTER*132 RSOPT     ,LINE
      CHARACTER*1  RSOPT1(132),LINE1(132)
      EQUIVALENCE (RSOPT1,RSOPT)
      EQUIVALENCE (LINE1,LINE)
C
C     Parse filename
C
C        CSPLIT splits S1 into two parts, delimited by S2.  
C        S1 returns with 2nd part of S1.  CSPLIT returns 1st part.
C
      rsopt=rsopts
      call ljust(rsopt)
      call csplit(fname,rsopt,' ',1)
C     check fname for user supplied extension.
      if (indx1(fname,'.',1).eq.0) then
         len=ltrunc(fname,132)
         len1=len+1
         len4=len+4
         fname(len1:len4)='.fld'
      endif
C
C     Parse restart options
C
C     set default flags
C
      ifgetx=.false.
      ifgetz=.false.
      ifgetu=.false.
      ifgetw=.false.
      ifgetp=.false.
      ifgett=.false.
      do 100 i=1,ldimt-1
         ifgtps(i)=.false.
  100 continue
      ifgtim=.true.
      ndumps=0
C
C     Check for default case - just a filename given, no i/o options specified
C
      ifdeft=.true.
C
C     Parse file for i/o options and/or dump number
C
      CALL CAPIT(RSOPT,132)

      IF (LTRUNC(RSOPT,132).NE.0) THEN
C
C        Check for explicit specification of restart TIME.
C
         ITO=INDX_CUT(RSOPT,'TIME',4)
         IFGTIM=.TRUE.
         IF (ITO.NE.0) THEN
C           user has specified the time explicitly.
            IT1=INDX_CUT(RSOPT,'=',1)
            IT8=132-IT1
            CALL BLANK(LINE,132)
            CALL CHCOPY(LINE,RSOPT1(IT1),IT8)
            IF (IFGTRL(TTIME,LINE)) THEN
               IFGTIM=.FALSE.
               TIME=TTIME
            ENDIF
C           remove the user specified time from the RS options line.
            ITA=132-ITO+1
            CALL BLANK(RSOPT1(ITO),ITA)
            CALL LJUST(LINE)
            IT1=INDX1(LINE,' ',1)
            ITB=132-IT1+1
            CALL CHCOPY(RSOPT1(ITO),LINE1(IT1),ITB)
         ENDIF

C        Parse field specifications.

         IXO=INDX_CUT(RSOPT,'X',1)
         IF (IXO.NE.0) THEN
            ifdeft=.false.
            IFGETX=.TRUE.
            IF (IF3D) IFGETZ=.TRUE.
         ENDIF

         IVO=INDX_CUT(RSOPT,'U',1)
         IF (IVO.NE.0) THEN
            ifdeft=.false.
            IFGETU=.TRUE.
            IF (IF3D) IFGETW=.TRUE.
         ENDIF

         IPO=INDX_CUT(RSOPT,'P',1)
         IF (IPO.NE.0) THEN
            ifdeft=.false.
            IFGETP=.TRUE.
         ENDIF

         ITO=INDX_CUT(RSOPT,'T',1)
         IF (ITO.NE.0) THEN
            ifdeft=.false.
            ifgett=.true.
         ENDIF

         do 300 i=1,ldimt-1
            write (s2,301) i
            ipo=indx_cut(rsopt,s2,2)
            if (ipo.ne.0) then
               ifdeft=.false.
               ifgtps(i)=.true.
            endif
  300    continue
  301    format('P',i1)

C        Get number of dumps from remainder of user supplied line.
         if (ifgtrl(tdumps,rsopt)) ndumps=int(tdumps)
      endif

C     If no fields were explicitly specified, assume getting all fields. 
      if (ifdeft) then
         IFGETX=.TRUE.
         IF (IF3D) IFGETZ=.TRUE.
         IFANYC=.FALSE.
         DO 400 I=1,NFIELD
            IF (IFADVC(I)) IFANYC=.TRUE.
  400    CONTINUE
         IF (IFFLOW.OR.IFANYC) THEN
            IFGETU=.TRUE.
            IF (IF3D) IFGETW=.TRUE.
         ENDIF
         if (ifflow) ifgetp=.true.
         if (ifheat) ifgett=.true.
#ifdef CMTNEK
         ifgett=.true. ! CMT-nek still not compatible with IFHEAT
#endif
         do 410 i=1,ldimt-1
            ifgtps(i)=.TRUE.
  410    continue
      endif




      return
      END
c-----------------------------------------------------------------------
      subroutine mapdmp(sdump,tdump,ieg,nxr,nyr,nzr,if_byte_sw)
C----------------------------------------------------------------------
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'PARALLEL'
C
      PARAMETER (LXYZ1=LX1*LY1*LZ1)
      PARAMETER (LXR=LX1+6)
      PARAMETER (LYR=LY1+6)
      PARAMETER (LZR=LZ1+6)
      PARAMETER (LXYZR=LXR*LYR*LZR)
C
      REAL   SDUMP(LXYZ1,LELT)
      REAL*4 TDUMP(LXYZR)
c
      logical if_byte_sw
c
      NXYZ=NX1*NY1*NZ1
      NXYR=NXR*NYR*NZR
      ierr=0
C
C     Serial processor code:
C
      IF (NP.EQ.1) THEN

         IF (if_byte_sw) call byte_reverse(TDUMP,NXYR,ierr)
         if(ierr.ne.0) call exitti("Error in mapdmp")
         IF (NXR.EQ.NX1.AND.NYR.EQ.NY1.AND.NZR.EQ.NZ1) THEN
            CALL COPY4r(SDUMP(1,IEG),TDUMP,NXYZ)
         ELSE
C           do the map    (assumes that NX=NY=NZ, or NX=NY, NZ=1)
            call mapab4r(sdump(1,ieg),tdump,nxr,1)
         ENDIF

      ELSE
C
C     Parallel code - send data to appropriate processor and map.
C
         JNID=GLLNID(IEG)
c     tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
         MTYPE=3333+GLLEL(IEG)
         LEN=4*NXYR
         LE1=4
         IF (NID.EQ.0.AND.JNID.NE.0) THEN
c           hand-shake
            CALL CSEND(MTYPE,TDUMP,LE1,JNID,NULLPID)
            CALL CRECV2(MTYPE,dummy,LE1,JNID)
            CALL CSEND(MTYPE,TDUMP,LEN,JNID,NULLPID)
         ELSEIF (NID.NE.0.AND.JNID.EQ.NID) THEN
C           Receive data from node 0
            CALL CRECV2(MTYPE,dummy,LE1,0)
            CALL CSEND(MTYPE,TDUMP,LE1,0,NULLPID)
            CALL CRECV2(MTYPE,TDUMP,LEN,0)
         ENDIF
C
C        If the data is targeted for this processor, then map 
C        to appropriate element.
C
         IF (JNID.EQ.NID) THEN
            IE=GLLEL(IEG)
            IF (if_byte_sw) call byte_reverse(TDUMP,NXYR,ierr)
            IF (NXR.EQ.NX1.AND.NYR.EQ.NY1.AND.NZR.EQ.NZ1) THEN
               CALL COPY4r(SDUMP(1,IE),TDUMP,NXYZ)
            ELSE
               call mapab4r(sdump(1,ie),tdump,nxr,1)
            ENDIF
         ENDIF
         call err_chk(ierr,'Error using byte_reverse in mapdmp.$')
C
C        End of parallel distribution/map routine.
C
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine mapab(x,y,nxr,nel)
C---------------------------------------------------------------
C
C     Interpolate Y(NXR,NYR,NZR,NEL) to X(NX1,NY1,NZ1,NEL)
C     (assumes that NXR=NYR=NZR, or NXR=NYR, NZR=1)
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'IXYZ'
      INCLUDE 'WZ'
C
      PARAMETER (LXR=LX1+6)
      PARAMETER (LYR=LY1+6)
      PARAMETER (LZR=LZ1+6)
      PARAMETER (LXYZR=LXR*LYR*LZR)
      PARAMETER (LXYZ1=LX1*LY1*LZ1)
      DIMENSION X(NX1,NY1,NZ1,NEL)
      DIMENSION Y(NXR,NXR,NXR,NEL)

      common /ctmp0/  xa(lxyzr)      ,xb(lx1,ly1,lzr) ,xc(lxyzr)
     $              , zgmr(lxr)      ,wgtr(lxr)
      common /ctmpab/ ires(lxr,lxr)  ,itres(lxr,lxr)
      real ires,itres

      INTEGER NOLD
      SAVE    NOLD
      DATA    NOLD /0/
C
      NZR = NXR
      IF(NZ1.EQ.1) NZR=1
      NYZR = NXR*NZR
      NXY1 = NX1*NY1
C
      IF (NXR.NE.NOLD) THEN
         NOLD=NXR
         CALL ZWGLL   (ZGMR,WGTR,NXR)
         CALL IGLLM   (IRES,ITRES,ZGMR,ZGM1,NXR,NX1,NXR,NX1)      
         IF (NIO.EQ.0) WRITE(6,10) NXR,NX1
   10       FORMAT(2X,'Mapping restart data from Nold=',I2
     $               ,' to Nnew=',I2,'.')
      ENDIF
C
      DO 1000 IE=1,NEL
         CALL MXM (IRES,NX1,Y(1,1,1,IE),NXR,XA,NYZR)
         DO 100 IZ=1,NZR
            IZOFF = 1 + (IZ-1)*NX1*NXR
            CALL MXM (XA(IZOFF),NX1,ITRES,NXR,XB(1,1,IZ),NY1)
  100    CONTINUE
         IF (NDIM.EQ.3) THEN
            CALL MXM (XB,NXY1,ITRES,NZR,X(1,1,1,IE),NZ1)
         ELSE
            CALL COPY(X(1,1,1,IE),XB,NXY1)
         ENDIF
 1000 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine mapab4R(x,y,nxr,nel)
C---------------------------------------------------------------
C
C     Interpolate Y(NXR,NYR,NZR,NEL) to X(NX1,NY1,NZ1,NEL)
C     (assumes that NXR=NYR=NZR, or NXR=NYR, NZR=1)
c
c     Input:  real*4,  Output:  default precision
c
C---------------------------------------------------------------
C
      INCLUDE 'SIZE'
      INCLUDE 'IXYZ'
      INCLUDE 'WZ'
C
      PARAMETER (LXR=LX1+6)
      PARAMETER (LYR=LY1+6)
      PARAMETER (LZR=LZ1+6)
      PARAMETER (LXYZR=LXR*LYR*LZR)
      PARAMETER (LXYZ1=LX1*LY1*LZ1)
      REAL*4 X(NX1,NY1,NZ1,NEL)
      REAL   Y(NXR,NXR,NXR,NEL)

      common /ctmp0/  xa(lxyzr)      ,xb(lx1,ly1,lzr) ,xc(lxyzr)
     $              , zgmr(lxr)      ,wgtr(lxr)
      common /ctmpa4/ ires(lxr,lxr)  ,itres(lxr,lxr)
      real ires,itres

      INTEGER NOLD
      SAVE    NOLD
      DATA    NOLD /0/

      NZR = NXR
      IF(NZ1.EQ.1) NZR=1
      NYZR = NXR*NZR
      NXY1 = NX1*NY1
      nxyzr = nxr*nxr*nzr
C
      IF (NXR.NE.NOLD) THEN
         NOLD=NXR
         CALL ZWGLL   (ZGMR,WGTR,NXR)
         CALL IGLLM   (IRES,ITRES,ZGMR,ZGM1,NXR,NX1,NXR,NX1)      
         IF (NIO.EQ.0) WRITE(6,10) NXR,NX1
   10       FORMAT(2X,'Mapping restart data from Nold=',I2
     $               ,' to Nnew=',I2,'.')
      ENDIF
C
      DO 1000 IE=1,NEL
         call copy4r(xc,y(1,1,1,ie),nxyzr)
         CALL MXM (IRES,NX1,xc,NXR,XA,NYZR)
         DO 100 IZ=1,NZR
            IZOFF = 1 + (IZ-1)*NX1*NXR
            CALL MXM (XA(IZOFF),NX1,ITRES,NXR,XB(1,1,IZ),NY1)
  100    CONTINUE
         IF (NDIM.EQ.3) THEN
            CALL MXM (XB,NXY1,ITRES,NZR,X(1,1,1,IE),NZ1)
         ELSE
            CALL COPY(X(1,1,1,IE),XB,NXY1)
         ENDIF
 1000 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      function i1_from_char(s1)
      character*1 s1

      character*10 n10
      save         n10
      data         n10 / '0123456789' /

      i1_from_char = indx2(n10,10,s1,1)-1

      return
      end
c-----------------------------------------------------------------------
      integer function indx2(s1,l1,s2,l2)
      character*132 s1,s2

      n1=l1-l2+1
      indx2=0
      if (n1.lt.1) return

      do i=1,n1
         i2=i+l2-1
         if (s1(i:i2).eq.s2(1:l2)) then
            indx2=i
            return
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*132 S1,S2
C
      N1=132-L2+1
      INDX1=0
      IF (N1.LT.1) return
C
      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            return
         ENDIF
  100 CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      INTEGER FUNCTION INDX_CUT(S1,S2,L2)
C
C     INDX_CUT is returned with the location of S2 in S1 (0 if not found)
C     S1     is returned with 1st occurance of S2 removed.
C
      CHARACTER*1 S1(132),S2(132)
C
      I1=INDX1(S1,S2,L2)
C
      IF (I1.NE.0) THEN
C
         N1=132-L2
         DO 100 I=I1,N1
            I2=I+L2
C           remove the 1st occurance of S2 from S1.
            S1(I)=S1(I2)
  100    CONTINUE
         N2=N1+1
         DO 200 I=N2,132
            S1(I)=' '
  200    CONTINUE
      ENDIF
C
      INDX_CUT=I1
      return
      END
c-----------------------------------------------------------------------
      subroutine csplit(s0,s1,s2,l0)
      CHARACTER*132 S0,S1,S2
C     split string S1 into two parts, delimited by S2.
C
      I2=INDX_CUT(S1,S2,L0)
      IF (I2.EQ.0) return
C
      I1=I2-1
      CALL BLANK(S0,132)
      S0(1:I1)=S1(1:I1)
      CALL LSHFT(S1,I2)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine lshft(string,ipt)
C     shift string from IPT to the left
C     INPUT : "abcde......    test    "
C     OUTPUT: "e......    test        "     if ipt.eq.5
      CHARACTER*1 STRING(132)
C
      DO 20 J=1,133-IPT
         IJ=IPT+J-1
         STRING(J)=STRING(IJ)
   20 CONTINUE
      DO 30 J=134-IPT,132
         STRING(J)=' '
   30 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine ljust(string)
C     left justify string
      CHARACTER*1 STRING(132)
C
      IF (STRING(1).NE.' ') return
C
      DO 100 I=2,132
C
         IF (STRING(I).NE.' ') THEN
            DO 20 J=1,133-I
               IJ=I+J-1
               STRING(J)=STRING(IJ)
   20       CONTINUE
            DO 30 J=134-I,132
               STRING(J)=' '
   30       CONTINUE
            return
         ENDIF
C
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine chknorm (ifzero)
C----------------------------------------------------------------------
C
C     Check if trivial user specified initial conditions
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      LOGICAL IFZERO
C
      IFZERO = .TRUE.
C
      IF (IFFLOW) THEN
         IFIELD = 1
         IMESH  = 1
         CALL UNORM
         IF (VNRML8.GT.0.) IFZERO = .FALSE.
      ENDIF
      IF (IFHEAT) THEN
         DO 100 IFIELD=2,NFIELD
            IMESH = 1
            IF (IFTMSH(IFIELD)) IMESH = 2
            CALL UNORM
            IF (TNRML8(IFIELD).GT.0.) IFZERO = .FALSE.
 100     CONTINUE
      ENDIF
c
      return
      END
C
c-----------------------------------------------------------------------
      subroutine prsolvt
C----------------------------------------------------------------------
C
C     Use steady state solution as initial condition 
C     for temperatur/passive scalar
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      LOGICAL  IFSAV1,IFSAV2(LDIMT1)
C
      IF (NIO.EQ.0) WRITE(6,*) ' '
      IF (NIO.EQ.0) WRITE(6,*) 'Conduction pre-solver activated'
C
C     Set logical IFTRAN to false (steady state)
C     Save logicals for convection
C     Turn convection off
C
      IFSAV1 = IFTRAN
      IFTRAN = .FALSE.
      DO 100 IFIELD=2,NFIELD
         IFSAV2(IFIELD) = IFADVC(IFIELD)
         IFADVC(IFIELD) = .FALSE.
 100  CONTINUE
C
      CALL SETPROP
      CALL SETSOLV
C
      IF(NIO.EQ.0)WRITE(6,*)'Steady conduction/passive scalar problem'
C
      DO 200 IGEOM=1,2
         CALL HEAT (IGEOM)
 200  CONTINUE
C
C     Set IFTRAN to true again
C     Turn convection on again
C
      IFTRAN = IFSAV1
      DO 300 IFIELD=2,NFIELD
         IFADVC(IFIELD) = IFSAV2(IFIELD)
 300  CONTINUE
C
      return
      END
C
c-----------------------------------------------------------------------
      subroutine prsolvv
C----------------------------------------------------------------------
C
C     Use steady Stokes solution as initial condition 
C     for flow problem
C
C----------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      LOGICAL  IFSAV1,IFSAV2
C
      IF (NIO.EQ.0) WRITE(6,*) ' '
      IF (NIO.EQ.0) WRITE(6,*) 'Velocity pre-solver activated'
C
C     Initialize velocity to some non-trivial RHS to avoid FP trap in i860.
C
      IF (PARAM(60).NE.0.0) THEN
         SMALL=10.0E-10
         CALL CFILL(VX,SMALL,NTOTV)
         CALL CFILL(VY,SMALL,NTOTV)
         CALL CFILL(VZ,SMALL,NTOTV)
      ENDIF
C
C     Set logical IFTRAN to false (steady state)
C     Save logicals for convection
C     Turn convection off
C
      IF (IFSPLIT) THEN
        WRITE(6,10)
   10   FORMAT(
     $ /,2X,'ERROR: Steady Stokes Flow initial condition cannot'
     $,/,2X,'       be computed using the splitting formulation.'
     $,/,2X,'       Either compute using UZAWA, or remove PRESOLVE'
     $,/,2X,'       request for velocity.'
     $,/,2X
     $,/,2X,'       ABORTING IN PRSOLVV.')
        CALL EXITT
      ENDIF
C
      IFSAV1 = IFTRAN
      IFSAV2 = IFADVC(IFIELD)
      IFTRAN = .FALSE.
      IFADVC(IFIELD) = .FALSE.
C
      CALL SETPROP
      CALL SETSOLV
      IF (IFNATC) GTHETA = GTHETA+10.
C
      IF (NIO.EQ.0) WRITE (6,*) 'Steady Stokes problem'
      DO 100 IGEOM=1,2
         IF (.NOT.IFSPLIT) CALL FLUID (IGEOM)
 100  CONTINUE
      IF (IFNATC) GTHETA = GTHETA-10.
C
C     Set IFTRAN to true again
C     Turn convection on again
C
      IFTRAN = IFSAV1
      IFADVC(IFIELD) = IFSAV2
C
      return
      END
C
c-----------------------------------------------------------------------
      subroutine nekuic ! user specified fortran function (=0 if not specified)

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      include 'TURBO'
      include 'PARALLEL'
      include 'NEKUSE'

      integer e,eg

      nel   = nelfld(ifield)

      if (ifmodel .and. ifkeps .and. ifield.eq.ifldk) then

         do e=1,nel
            eg = lglel(e)
            do 100 k=1,nz1
            do 100 j=1,ny1
            do 100 i=1,nx1
               call nekasgn (i,j,k,e)
               call useric  (i,j,k,eg)
               t(i,j,k,e,ifield-1) = turbk
 100        continue
         enddo

      elseif (ifmodel .and. ifkeps .and. ifield.eq.iflde) then

         do e=1,nel
            eg = lglel(e)
            do 200 k=1,nz1
            do 200 j=1,ny1
            do 200 i=1,nx1
               call nekasgn (i,j,k,e)
               call useric  (i,j,k,eg)
               t(i,j,k,e,ifield-1) = turbe
 200        continue
         enddo
C
      else
         do e=1,nel
            eg = lglel(e)
            do 300 k=1,nz1
            do 300 j=1,ny1
            do 300 i=1,nx1
              call nekasgn (i,j,k,e)
              call useric  (i,j,k,eg)
              if (jp.eq.0) then
                if (ifield.eq.1) then
                  vx(i,j,k,e) = ux
                  vy(i,j,k,e) = uy
                  vz(i,j,k,e) = uz
                elseif (ifield.eq.ifldmhd) then
                  bx(i,j,k,e) = ux
                  by(i,j,k,e) = uy
                  bz(i,j,k,e) = uz
                else
                  t(i,j,k,e,ifield-1) = temp
                endif
              else
                ijke = i+nx1*((j-1)+ny1*((k-1) + nz1*(e-1)))
                if (ifield.eq.1) then
                  vxp(ijke,JP) = ux
                  vyp(ijke,JP) = uy
                  vzp(ijke,JP) = uz
                else
                  tp(ijke,ifield-1,JP) = temp
                endif
              endif

 300        continue
         enddo

      endif


      return
      END
c-----------------------------------------------------------------------
      subroutine capit(lettrs,n)
C     Capitalizes string of length n
      CHARACTER LETTRS(N)
C
      DO 5 I=1,N
         INT=ICHAR(LETTRS(I))
         IF(INT.GE.97 .AND. INT.LE.122) THEN
            INT=INT-32
            LETTRS(I)=CHAR(INT)
         ENDIF
5     CONTINUE
      return
      END
c-----------------------------------------------------------------------
      LOGICAL FUNCTION IFGTRL(VALUE,LINE)
C
C     Read VALUE from LINE and set IFGTRL to .TRUE. if successful,
C                                  IFGTRL to .FALSE. otherwise.
C
C     This complicated function is necessary thanks to the Ardent,
C     which won't allow free formatted reads (*) from internal strings!
C
      CHARACTER*132 LINE
      CHARACTER*132 WORK
      CHARACTER*8  FMAT
C
C     Note that the format Fn.0 is appropriate for fields of type:
C          34   34.0  34.0e+00
C     The only difficulty would be with '34' but since we identify
C     the field width exactly, there is no problem.
C
      IFGTRL=.FALSE.
      VALUE=0.0
C
      WORK=LINE
      CALL LJUST(WORK)
      IFLDW=INDX1(WORK,' ',1)-1
C
      IF (IFLDW.GT.0) THEN
         WRITE(FMAT,10) IFLDW
   10    FORMAT('(F',I3.3,'.0)')
         READ(WORK,FMAT,ERR=100,END=100) TVAL
         VALUE=TVAL
         IFGTRL=.TRUE.
         return
      ENDIF
C
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      LOGICAL FUNCTION IFGTIL(IVALUE,LINE)
C
C     Read IVALUE from LINE and set IFGTIL to .TRUE. if successful,
C                                   IFGTIL to .FALSE. otherwise.
C
C     This complicated function is necessary thanks to the Ardent,
C     which won't allow free formatted reads (*) from internal strings!
C
      CHARACTER*132 LINE
      CHARACTER*132 WORK
      CHARACTER*8  FMAT
C
      IFGTIL=.FALSE.
      IVALUE=0
C
      WORK=LINE
      CALL LJUST(WORK)
      IFLDW=INDX1(WORK,' ',1)-1
C
      IF (IFLDW.GT.0) THEN
         WRITE(FMAT,10) IFLDW
   10    FORMAT('(F',I3.3,'.0)')
         READ(WORK,FMAT,ERR=100,END=100) TVAL
         IVALUE=INT(TVAL)
         IFGTIL=.TRUE.
         return
      ENDIF
C
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine perturb(tt,ifld,eps)
      include 'SIZE'
      include 'TOTAL'
c
      real tt(1)
      integer ifld

      ifield = ifld

      n = nx1*ny1*nz1*nelfld(ifield)
      call vcospf(tt,bm1,n)
      call cmult(tt,eps,n)
      call dssum(tt,nx1,ny1,nz1)

      return
      end
c-----------------------------------------------------------------------
      subroutine vcospf(x,y,n)
      real x(1),y(1)
      do i=1,n
         x(i) = cos(1000.*y(i))
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine vbyte_swap(x,n)
      character*1 x(0:3,1),tmp0,tmp1
      character*1 in (0:3), out(0:3)
      real*4      in4     , out4
      equivalence (in ,in4 )
      equivalence (out,out4)
c
      do i=1,n
         do j=0,3
            in (j) = x(j,i)
         enddo
         tmp0   = x(0,i)
         tmp1   = x(1,i)
         x(0,i) = x(3,i)
         x(1,i) = x(2,i)
         x(2,i) = tmp1
         x(3,i) = tmp0
         do j=0,3
            out(j) = x(j,i)
         enddo
         write(6,*) 'swap:',i,in4,out4
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      logical function if_byte_swap_test(bytetest,ierr)
      include 'SIZE'
 
      real*4 bytetest,test2
      real*4 test_pattern
      save   test_pattern
 
      test_pattern = 6.54321
      eps          = 0.00020
      etest        = abs(test_pattern-bytetest)
      if_byte_swap_test = .true.
      if (etest.le.eps) if_byte_swap_test = .false.
 
      test2 = bytetest
      call byte_reverse(test2,1,ierr)
      if (nid.eq.0) 
     $   write(6,*) 'byte swap:',if_byte_swap_test,bytetest,test2
      return
      end
c-----------------------------------------------------------------------
      subroutine geom_reset(icall)
C
C     Generate geometry data
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      include 'WZ'
c
      COMMON /scruz/ XM3 (LX1,LY1,LZ1,LELT)
     $ ,             YM3 (LX1,LY1,LZ1,LELT)
     $ ,             ZM3 (LX1,LY1,LZ1,LELT)
C
c
      if(nio.eq.0) write(6,*) 'regenerate geometry data',icall

      ntot = nx1*ny1*nz1*nelt
c
      if (lx3.eq.lx1) then
         call copy(xm3,xm1,ntot)
         call copy(ym3,ym1,ntot)
         call copy(zm3,zm1,ntot)
      else
         call map13_all(xm3,xm1)
         call map13_all(ym3,ym1)
         if (if3d) call map13_all(zm3,zm1)
      endif
c
      CALL GEOM1 (XM3,YM3,ZM3)
      CALL GEOM2
      CALL UPDMSYS (1)
      CALL VOLUME
      CALL SETINVM
      CALL SETDEF
      CALL SFASTAX
c
      if(nio.eq.0) then
        write(6,*) 'done :: regenerate geometry data',icall
        write(6,*) ' '
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine dsavg(u)
c
c
      include 'SIZE'
      include 'TOTAL'
      real u(lx1,ly1,lz1,lelt)
c
c
c     Take direct stiffness avg of u
c
c
      ifieldo = ifield
      if (ifflow) then
         ifield = 1
         ntot = nx1*ny1*nz1*nelv
         call dssum(u,nx1,ny1,nz1)
         call col2 (u,vmult,ntot)
      else
         ifield = 2
         ntot = nx1*ny1*nz1*nelt
         call dssum(u,nx1,ny1,nz1)
         call col2 (u,tmult,ntot)
      endif
      ifield = ifieldo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine map13_all(x3,x1)
c
      include 'SIZE'
      include 'TOTAL'
c
      real x3(lx3,ly3,lz3,lelt)
      real x1(lx1,ly1,lz1,lelt)
c
      integer e
c
      do e=1,nelt
         call map13 (x3(1,1,1,e),x1(1,1,1,e),e)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mfi_gets(u,wk,lwk,iskip)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'


      real u(lx1*ly1*lz1,1)

      real*4 wk(lwk) ! message buffer
      parameter(lrbs=20*lx1*ly1*lz1*lelt)
      common /vrthov/ w2(lrbs) ! read buffer
      real*4 w2

      integer e,ei,eg,msg_id(lelt)
      logical iskip

      call nekgsync() ! clear outstanding message queues.

      nxyzr = nxr*nyr*nzr   
      len   = nxyzr*wdsizr  ! message length
      if (wdsizr.eq.8) nxyzr = 2*nxyzr

      ! check message buffer
      num_recv  = len
      num_avail = lwk*wdsize
      call lim_chk(num_recv,num_avail,'     ','     ','mfi_gets a')

      ! setup read buffer
      if (nid.eq.pid0r) then
         nread = nxyzr*nelr/lrbs
         if(mod(nxyzr*nelr,lrbs).ne.0) nread = nread + 1
#ifdef MPIIO 
         nread = iglmax(nread,1) ! needed because of collective read
#endif
         nelrr = nelr/nread
      endif
      call bcast(nelrr,4)
      call lim_chk(nxyzr*nelrr,lrbs,'     ','     ','mfi_gets b')

      ! pre-post recieves
      if (np.gt.1) then
         l = 1
         do e=1,nelt
c     Tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
            msg_id(e) = irecv(e,wk(l),len)
            l = l+nxyzr
         enddo
      endif

      ierr = 0
      if (nid.eq.pid0r.and.np.gt.1) then ! only i/o nodes will read
         ! read blocks of size nelrr
         k = 0
         do i = 1,nread
            if(i.eq.nread) then ! clean-up 
              nelrr = nelr - (nread-1)*nelrr
              if(nelrr.lt.0) nelrr = 0
            endif
            
            if(ierr.eq.0) then
#ifdef MPIIO 
              call byte_read_mpi(w2,nxyzr*nelrr,-1,ifh_mbyte,ierr)
#else
              call byte_read (w2,nxyzr*nelrr,ierr)
#endif
            endif

            ! distribute data across target processors
            l = 1
            do e = k+1,k+nelrr
               jnid = gllnid(er(e))                ! where is er(e) now?
c     Tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
               jeln = gllel(er(e))
               if(ierr.ne.0) call rzero(w2(l),len)
               call csend(jeln,w2(l),len,jnid,0)  ! blocking send
               l = l+nxyzr
            enddo
            k  = k + nelrr
         enddo
      elseif (np.eq.1) then
#ifdef MPIIO 
         call byte_read_mpi(wk,nxyzr*nelr,-1,ifh_mbyte,ierr)
#else
         call byte_read(wk,nxyzr*nelr,ierr)
#endif
      endif


c     if (if_byte_sw.and.wdsizr.eq.8) then
c        if(nid.eq.0) 
c    &     write(6,*) 'ABORT: byteswap for 8byte restart data ', 
c    &                'not supported'
c        call exitt
c     endif

      if (iskip) then
         call nekgsync() ! clear outstanding message queues.
         goto 100     ! don't use the data
      endif

      nxyzr = nxr*nyr*nzr
      nxyzv = nxr*nyr*nzr
      nxyzw = nxr*nyr*nzr
      if (wdsizr.eq.8) nxyzw = 2*nxyzw

      l = 1
      do e=1,nelt
         if (np.gt.1) then
            call msgwait(msg_id(e))
            ei = e
         elseif(np.eq.1) then
            ei = er(e)
         endif
         if (if_byte_sw) then
            if(wdsizr.eq.8) then
              call byte_reverse8(wk(l),nxyzv*2,ierr)
            else
              call byte_reverse(wk(l),nxyzv,ierr)
            endif
         endif
         if (nxr.eq.nx1.and.nyr.eq.ny1.and.nzr.eq.nz1) then
            if (wdsizr.eq.4) then         ! COPY
               call copy4r(u(1,ei),wk(l        ),nxyzr)
            else
               call copy  (u(1,ei),wk(l        ),nxyzr)
            endif
         else                             ! INTERPOLATE
            if (wdsizr.eq.4) then
               call mapab4r(u(1,ei),wk(l        ),nxr,1)
            else
               call mapab  (u(1,ei),wk(l        ),nxr,1)
            endif
         endif
         l = l+nxyzw
      enddo


 100  call err_chk(ierr,'Error reading restart data,in gets.$')
      return
      end
c-----------------------------------------------------------------------
      subroutine mfi_getv(u,v,w,wk,lwk,iskip)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      real u(lx1*ly1*lz1,1),v(lx1*ly1*lz1,1),w(lx1*ly1*lz1,1)
      logical iskip

      real*4 wk(lwk) ! message buffer
      parameter(lrbs=20*lx1*ly1*lz1*lelt)
      common /vrthov/ w2(lrbs) ! read buffer
      real*4 w2

      integer e,ei,eg,msg_id(lelt)

      call nekgsync() ! clear outstanding message queues.

      nxyzr = ndim*nxr*nyr*nzr
      len   = nxyzr*wdsizr             ! message length in bytes
      if (wdsizr.eq.8) nxyzr = 2*nxyzr

      ! check message buffer
      num_recv  = len
      num_avail = lwk*wdsize
      call lim_chk(num_recv,num_avail,'     ','     ','mfi_getv a')

      ! setup read buffer
      if(nid.eq.pid0r) then
         nread = nxyzr*nelr/lrbs
         if(mod(nxyzr*nelr,lrbs).ne.0) nread = nread + 1
#ifdef MPIIO 
         nread = iglmax(nread,1) ! needed because of collective read
#endif
         nelrr = nelr/nread
      endif
      call bcast(nelrr,4)
      call lim_chk(nxyzr*nelrr,lrbs,'     ','     ','mfi_getv b')

      ! pre-post recieves (one mesg per element)
      ! this assumes we never pre post more messages than supported
      if (np.gt.1) then
         l = 1
         do e=1,nelt
c     tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
            msg_id(e) = irecv(e,wk(l),len)
            l = l+nxyzr
         enddo
      endif

      ierr = 0
      if (nid.eq.pid0r .and. np.gt.1) then ! only i/o nodes
         k = 0
         do i = 1,nread
            if(i.eq.nread) then ! clean-up 
              nelrr = nelr - (nread-1)*nelrr
              if(nelrr.lt.0) nelrr = 0
            endif

            if(ierr.eq.0) then
#ifdef MPIIO 
              call byte_read_mpi(w2,nxyzr*nelrr,-1,ifh_mbyte,ierr)
#else
              call byte_read (w2,nxyzr*nelrr,ierr)
#endif
            endif

            ! redistribute data based on the current el-proc map
            l = 1
            do e = k+1,k+nelrr
               jnid = gllnid(er(e))                ! where is er(e) now?
c     tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
               jeln = gllel(er(e))
               if(ierr.ne.0) call rzero(w2(l),len)
               call csend(jeln,w2(l),len,jnid,0)  ! blocking send
               l = l+nxyzr
            enddo
            k  = k + nelrr
         enddo
      elseif (np.eq.1) then
#ifdef MPIIO 
         call byte_read_mpi(wk,nxyzr*nelr,-1,ifh_mbyte,ierr)
#else
         call byte_read(wk,nxyzr*nelr,ierr)
#endif
      endif

c     if (if_byte_sw.and.wdsizr.eq.8) then
c        if(nid.eq.0) 
c    &     write(6,*) 'ABORT: byteswap for 8byte restart data ', 
c    &                'not supported'
c        call exitt
c     endif

      if (iskip) then
         call nekgsync() ! clear outstanding message queues.
         goto 100     ! don't assign the data we just read
      endif

      nxyzr = nxr*nyr*nzr
      nxyzv = ndim*nxr*nyr*nzr
      nxyzw = nxr*nyr*nzr
      if (wdsizr.eq.8) nxyzw = 2*nxyzw

      l = 1
      do e=1,nelt
         if (np.gt.1) then
            call msgwait(msg_id(e))
            ei = e
         else if(np.eq.1) then
            ei = er(e) 
         endif
         if (if_byte_sw) then
            if(wdsizr.eq.8) then
               call byte_reverse8(wk(l),nxyzv*2,ierr)
            else
               call byte_reverse(wk(l),nxyzv,ierr)
            endif
         endif
         if (nxr.eq.nx1.and.nyr.eq.ny1.and.nzr.eq.nz1) then
            if (wdsizr.eq.4) then         ! COPY
               call copy4r(u(1,ei),wk(l        ),nxyzr)
               call copy4r(v(1,ei),wk(l+  nxyzw),nxyzr)
               if (if3d) 
     $         call copy4r(w(1,ei),wk(l+2*nxyzw),nxyzr)
            else
               call copy  (u(1,ei),wk(l        ),nxyzr)
               call copy  (v(1,ei),wk(l+  nxyzw),nxyzr)
               if (if3d) 
     $         call copy  (w(1,ei),wk(l+2*nxyzw),nxyzr)
            endif
         else                             ! INTERPOLATE
            if (wdsizr.eq.4) then
               call mapab4r(u(1,ei),wk(l        ),nxr,1)
               call mapab4r(v(1,ei),wk(l+  nxyzw),nxr,1)
               if (if3d) 
     $         call mapab4r(w(1,e),wk(l+2*nxyzw),nxr,1)
            else
               call mapab  (u(1,ei),wk(l        ),nxr,1)
               call mapab  (v(1,ei),wk(l+  nxyzw),nxr,1)
               if (if3d) 
     $         call mapab  (w(1,ei),wk(l+2*nxyzw),nxr,1)
            endif
         endif
         l = l+ndim*nxyzw
      enddo

 100  call err_chk(ierr,'Error reading restart data, in getv.$')
      return
      end
c-----------------------------------------------------------------------
      subroutine mfi_parse_hdr(hdr,ierr)
      include 'SIZE'

      character*132 hdr

      if (indx2(hdr,132,'#std',4).eq.1) then
          call parse_std_hdr(hdr)
      else
         if (nio.eq.0) write(6,80) hdr
         if (nio.eq.0) write(6,80) 'NONSTD HDR, parse_hdr, abort.'
  80     format(a132)
         ierr = 1
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine parse_std_hdr(hdr)
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'PARALLEL'
      include 'RESTART'

      character*132 hdr
      character*4 dummy

      read(hdr,*,iostat=ierr) dummy
     $         ,  wdsizr,nxr,nyr,nzr,nelr,nelgr,timer,istpr
     $         ,  ifiler,nfiler
     $         ,  rdcode      ! 74+20=94
     $         ,  p0thr

      if (ierr.gt.0) then ! try again without mean pressure
        read(hdr,*,err=99) dummy
     $         ,  wdsizr,nxr,nyr,nzr,nelr,nelgr,timer,istpr
     $         ,  ifiler,nfiler
     $         ,  rdcode      ! 74+20=94
      endif

#ifdef MPIIO
      if ((nelr/np + np).gt.lelr) then
        write(6,'(A,I6)') 'ABORT: nelr>lelr on rank',nid
        call exitt
      endif
#else
      if (nelr.gt.lelr) then
        write(6,'(A,I6)') 'ABORT: nelr>lelr on rank',nid
        call exitt
      endif
#endif

      ifgtim  = .true.  ! always get time
      ifgetxr = .false.
      ifgetur = .false.
      ifgetpr = .false.
      ifgettr = .false.
      do k=1,ldimt-1
         ifgtpsr(k) = .false.
      enddo

      NPSR = 0
      do i=1,10 
         if (rdcode1(i).eq.'X') ifgetxr = .true.
         if (rdcode1(i).eq.'U') ifgetur = .true.
         if (rdcode1(i).eq.'P') ifgetpr = .true.
         if (rdcode1(i).eq.'T') ifgettr = .true.
         if (rdcode1(i).eq.'S') then
            read(rdcode1(i+1),'(I1)') NPS1
            read(rdcode1(i+2),'(I1)') NPS0
            NPSR = 10*NPS1+NPS0
            NPS  = NPSR
            if(NPSR.gt.ldimt-1) NPS=ldimt-1
            do k=1,NPS
               ifgtpsr(k) = .true.
            enddo
            ! nothing will follow
            GOTO 50
         endif
      enddo

  50  if (NPS.lt.NPSR) then
         if (nid.eq.0) then 
           write(*,'(A,/,A)') 
     &      'WARNING: restart file has a NSPCAL > LDIMT',
     &      'read only part of the fld-data!'
         endif
      endif

      if (NPS.lt.NPSCAL) then
         if (nid.eq.0) then 
           write(*,'(A,/,A)') 
     &      'WARNING: NPSCAL read from restart file differs from ',
     &      'currently used NPSCAL!'
         endif
      endif

      p0th = p0thr

      return

   99 continue   !  If we got here, then the May 2008 variant of std hdr
                 !  failed and we may have an older input file.

      call parse_std_hdr_2006(hdr,rdcode)  ! try the original header format

      return
      end
c-----------------------------------------------------------------------
      subroutine parse_std_hdr_2006(hdr,rlcode)
      include 'SIZE'
      include 'INPUT'
      include 'RESTART'

      character*132 hdr
      character*1 rlcode(20)

c                4  7  10  13   23    33    53    62     68     74
      read(hdr,1) wdsizr,nxr,nyr,nzr,nelr,nelgr,timer,istpr
     $         , ifiler,nfiler
     $         , (rlcode(k),k=1,20)                   ! 74+20=94
    1 format(4x,i2,3i3,2i10,e20.13,i9,2i6,20a1)

      if (nid.eq.0) write(6,*) 'WARNING: reading depreacted header!'

      if (nelr.gt.lelr) then
        write(6,*)nid,nelr,lelr,'parse_std_hdr06: inc. lelr in RESTART'
        call exitt
      endif

c     Assign read conditions, according to rdcode
c     NOTE: In the old hdr format: what you see in file is what you get.
      ifgtim  = .true.  ! always get time
      ifgetxr = .false.
      ifgetur = .false.
      ifgetpr = .false.
      ifgettr = .false.
      do k=1,npscal
         ifgtpsr(k) = .false.
      enddo

      if (rlcode(1).eq.'X') ifgetxr = .true.
      if (rlcode(2).eq.'U') ifgetur = .true.
      if (rlcode(3).eq.'P') ifgetpr = .true.
      if (rlcode(4).eq.'T') ifgettr = .true.
      do k=1,npscal
         if (rlcode(4+k).ne.' ') ifgtpsr(k) = .true.
      enddo


      return
      end
c-----------------------------------------------------------------------
      subroutine mfi(fname,ifile)
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
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      character*132 hdr
      character*132  fname

      logical if_full_pres_tmp

      parameter (lwk = 7*lx1*ly1*lz1*lelt)
      common /scrns/ wk(lwk)
      common /scrcg/ pm1(lx1*ly1*lz1,lelv)
      integer e

      integer*8 offs0,offs,nbyte,stride,strideB,nxyzr8

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
            if (ifmhd.and.ifile.eq.2) then
c               if(nid.eq.0) write(6,*) 'Reading B field'
               call mfi_getv(bx,by,bz,wk,lwk,.false.)
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
c            if(nid.eq.0) write(6,*) 'Reading temperature field'
            call mfi_gets(t,wk,lwk,.false.)
         else
            call mfi_gets(t,wk,lwk,.true.)
         endif
         iofldsr = iofldsr + 1
      endif

      ierr = 0
      do k=1,ldimt-1
         if (ifgtpsr(k)) then
            offs = offs0 + iofldsr*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            if (ifgtps(k)) then
c               if(nid.eq.0) write(6,'(A,I2,A)') ' Reading ps',k,' field'
               call mfi_gets(t(1,1,1,1,k+1),wk,lwk,.false.)
            else
               call mfi_gets(t(1,1,1,1,k+1),wk,lwk,.true.)
            endif
            iofldsr = iofldsr + 1
         endif
      enddo
      nbyte = 0
      if(nid.eq.pid0r) nbyte = iofldsr*nelr*wdsizr*nxr*nyr*nzr

      if (ifgtim) time = timer

#ifdef MPIIO
      if (nid.eq.pid0r) call byte_close_mpi(ifh_mbyte,ierr)
#else
      if (nid.eq.pid0r) call byte_close(ierr)
#endif
      call err_chk(ierr,'Error closing restart file, in mfi.$')
      tio = dnekclock()-tiostart

      dnbyte = nbyte
      nbyte = glsum(dnbyte,1)
      nbyte = nbyte + iHeaderSize + 4 + isize*nelgr

      if (tio.eq.0) tio=1 ! Avoid division by zero, pff, 11/29/15
      if (nio.eq.0) write(6,7) istep,time,
     &             nbyte/tio/1024/1024/10,
     &             nfiler
    7 format(/,i9,1pe12.4,' done :: Read checkpoint data',/,
     &       30X,'avg data-throughput = ',f7.1,'MBps',/,
     &       30X,'io-nodes = ',i5,/)


      if (ifaxis) call axis_interp_ic(pm1)      ! Interpolate to axi mesh
      if (ifgetp) call map_pm1_to_pr(pm1,ifile) ! Interpolate pressure

      if_full_pres = if_full_pres_tmp

      return
      end
c-----------------------------------------------------------------------
      subroutine mbyte_open(hname,fid,ierr) ! open  blah000.fldnn
      include 'SIZE'
      include 'TSTEP'
      include 'RESTART'
 
      integer fid
      character*132 hname

      character*8  eight,fmt,s8
      save         eight
      data         eight / "????????" /

      character*132 fname
      character*1  fname1(132)
      equivalence (fname1,fname)

      integer      iname(33)
      equivalence (iname,fname)

      call izero  (iname,33)
      len = ltrunc(hname,132)
      call chcopy (fname,hname,len)

      do ipass=1,2      ! 2nd pass, in case 1 file/directory
         do k=8,1,-1
            i1 = indx1(fname,eight,k)
            if (i1.ne.0) then ! found k??? string
               write(fmt,1) k,k
    1          format('(i',i1,'.',i1,')')
               write(s8,fmt) fid
               call chcopy(fname1(i1),s8,k)
               goto 10
            endif
         enddo
   10    continue
      enddo
      
#ifdef MPIIO
      call byte_open_mpi(fname,ifh_mbyte,ierr)
      if(nio.eq.0) write(6,6) istep,(fname1(k),k=1,len)
    6 format(1i8,' OPEN: ',132a1)
#else
      call byte_open(fname,ierr)
      if(nio.eq.0)write(6,6) nid,istep,(fname1(k),k=1,len)
    6 format(2i8,' OPEN: ',132a1)
#endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mfi_prepare(hname)  ! determine which nodes are readers
      character*132 hname

      include 'SIZE'
      include 'PARALLEL'
      include 'RESTART'

      integer stride
      character*132 hdr
      logical if_byte_swap_test
      real*4 bytetest

      integer*8 offs0,offs

      integer sum

      ierr = 0
#ifndef MPIIO
      ! rank0 (i/o master) will do a pre-read to get some infos 
      ! we need to have in advance
      if (nid.eq.0) then
         call mbyte_open(hname,0,ierr) ! open  blah000.fldnn
         if(ierr.ne.0) goto 101
         call blank     (hdr,iHeaderSize)
         call byte_read (hdr, iHeaderSize/4,ierr)
         if(ierr.ne.0) goto 101
         call byte_read (bytetest,1,ierr)
         if(ierr.ne.0) goto 101
         if_byte_sw = if_byte_swap_test(bytetest,ierr) ! determine endianess
         if(ierr.ne.0) goto 101
         call mfi_parse_hdr(hdr,ierr)
      endif

 101  continue
      call err_chk(ierr,'Error reading restart header in mfi_prepare$')

      call bcast(if_byte_sw,lsize) 
      call bcast(hdr,iHeaderSize)  
      if(nid.ne.0) call mfi_parse_hdr(hdr,ierr)

      stride = np / nfiler
      if (stride.lt.1) then
         write(6,*) nfiler,np,'  TOO MANY FILES, mfi_prepare'
         call exitt
      endif

      if (mod(nid,stride).eq.0) then ! i/o clients
         pid0r = nid
         pid1r = nid + stride
         fid0r = nid / stride
         if (nid.ne.0) then ! don't do it again for rank0
            call blank     (hdr,iHeaderSize)
            call mbyte_open(hname,fid0r,ierr) ! open  blah000.fldnn
            if(ierr.ne.0) goto 102
            call byte_read (hdr, iHeaderSize/4,ierr)  
            if(ierr.ne.0) goto 102
            call byte_read (bytetest,1,ierr) 
            if(ierr.ne.0) goto 102
            call mfi_parse_hdr (hdr,ierr)  ! replace hdr with correct one 
         endif
         call byte_read (er,nelr,ierr)     ! get element mapping
         if (if_byte_sw) call byte_reverse(er,nelr,ierr)
      endif
#else
      pid0r = nid
      pid1r = nid
      offs0 = iHeaderSize + 4
      call mbyte_open(hname,0,ierr)
      ierr=iglmax(ierr,1)
      if(ierr.ne.0) goto 103

      call byte_read_mpi(hdr,iHeaderSize/4,pid00,ifh_mbyte,ierr)
      ierr=iglmax(ierr,1)
      if(ierr.ne.0) goto 103

      call byte_read_mpi(bytetest,1,pid00,ifh_mbyte,ierr)

 103  continue 
      call err_chk(ierr,'Error reading header/element map.$')
      
      call bcast(hdr,iHeaderSize) 
      call bcast(bytetest,4) 

      if_byte_sw = if_byte_swap_test(bytetest,ierr) ! determine endianess
      call mfi_parse_hdr(hdr,ierr)
      if(nfiler.ne.1) then
        if(nid.eq.0) write(6,*) 'ABORT: too many restart files!'
        call exitt
      endif
      nfiler = np

      ! number of elements to read 
      nelr = nelgr/np
      do i = 0,mod(nelgr,np)-1
         if(i.eq.nid) nelr = nelr + 1
      enddo
      nelBr = igl_running_sum(nelr) - nelr 
      offs = offs0 + nelBr*isize

      call byte_set_view(offs,ifh_mbyte)
      call byte_read_mpi(er,nelr,-1,ifh_mbyte,ierr)
      if (if_byte_sw) call byte_reverse(er,nelr,ierr)
#endif
 102  continue
      call err_chk(ierr,'Error reading header/element map.$')

      return
      end
c-----------------------------------------------------------------------
      subroutine axis_interp_ic(pm1)

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'

      real pm1(lx1,ly1,lz1,lelv)

      common /ctmp0/ axism1 (lx1,ly1)
      integer e

      if (.not.ifaxis) return

      do e=1,nelv
         if (ifrzer(e)) then
           if (ifgetx) then
             call mxm   (xm1(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy  (xm1(1,1,1,e),axism1,nx1*ny1)
             call mxm   (ym1(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy  (ym1(1,1,1,e),axism1,nx1*ny1)
           endif
           if (ifgetu) then
             call mxm    (vx(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy   (vx(1,1,1,e),axism1,nx1*ny1)
             call mxm    (vy(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy   (vy(1,1,1,e),axism1,nx1*ny1)
           endif
           if (ifgetw) then
             call mxm    (vz(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy   (vz(1,1,1,e),axism1,nx1*ny1)
           endif
           if (ifgetp) then
             call mxm    (pm1(1,1,1,e),nx1,iatlj1,ny1,axism1,ny1)
             call copy   (pm1(1,1,1,e),axism1,nx1*ny1)
           endif
           if (ifgett) then
             call mxm  (t (1,1,1,e,1),nx1,iatlj1,ny1,axism1,ny1)
             call copy (t (1,1,1,e,1),axism1,nx1*ny1)
           endif
           do ips=1,npscal
            is1 = ips + 1
            if (ifgtps(ips)) then
             call mxm (t(1,1,1,e,is1),nx1,iatlj1,ny1,axism1,ny1)
             call copy(t(1,1,1,e,is1),axism1,nx1*ny1)
            endif
           enddo
         endif
      enddo
   
      return
      end
c-----------------------------------------------------------------------
      subroutine map_pm1_to_pr(pm1,ifile)

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'

      logical if_full_pres_tmp

      real pm1(lx1*ly1*lz1,lelv)
      integer e

      nxyz2 = nx2*ny2*nz2

      if (ifmhd.and.ifile.eq.2) then
         do e=1,nelv
            if (if_full_pres) then
               call copy  (pm(1,1,1,e),pm1(1,e),nxyz2)
            else
               call map12 (pm(1,1,1,e),pm1(1,e),e)
            endif
         enddo
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
c-----------------------------------------------------------------------
      subroutine full_restart(s80,n_restart)
      include 'SIZE'
      include 'TOTAL'

      character*80 s80(n_restart)

      ifile = istep+1  ! istep=0,1,...

      if (ifile.le.n_restart) then
         p67 = param(67)
         param(67) = 6.00
         call chcopy (initc,s80(ifile),80)
         call bcast  (initc,80)
         call restart       (1)
         param(67)=p67
      endif
   
      return
      end
c-----------------------------------------------------------------------
