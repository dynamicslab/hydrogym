c-----------------------------------------------------------------------
      subroutine set_outfld

c     Check if we are going to checkpoint at this timestep
c     and set ifoutfld accordingly

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common /rdump/ ntdump

      ifoutfld = .false.

      if (iostep.le.0 .and. timeio.le.0) return

c      if (istep.ge.nsteps) lastep=1

      if (iostep.gt.0) then
         if(mod(istep,iostep).eq.0) ifoutfld=.true.
      else if (timeioe.ne.0.0) then
         if (dnekclock_sync()-etimes .ge. (ntdump + 1)*timeio) then
            ntdump=ntdump+1
            ifoutfld=.true.
         endif 
      else if (timeio.ne.0.0) then
         if (time.ge.(ntdump + 1)*timeio) then
            ntdump=ntdump+1
            ifoutfld=.true.
         endif
      endif

      if (ioinfodmp.ne.0 .or. lastep.eq.1) ifoutfld=.true. 

      return
      end
c-----------------------------------------------------------------------
      subroutine check_ioinfo

c     Check for io request in file 'ioinfo'

      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'

      parameter (lxyz=lx1*ly1*lz1)
      parameter (lpsc9=ldimt1+9)
      common /ctmp1/ tdump(lxyz,lpsc9)
      real*4         tdump
      real           tdmp(4)
      equivalence   (tdump,tdmp)

      integer maxstep
      save    maxstep
      data    maxstep /999999999/

      character*132 fname
      character*1   fname1(132)
      equivalence  (fname,fname1)

      ioinfodmp=0
      if (nid.eq.0 .and. (mod(istep,10).eq.0 .or. istep.lt.200)) then
         call blank(fname1,size(fname1))
         len = ltrunc(path,132)
         call chcopy(fname1,path,len)
         call chcopy(fname1(len+1),'ioinfo',6)
         open(unit=87,file=fname,status='old',err=88)
         read(87,*,end=87,err=87) idummy
         if (ioinfodmp.eq.0) ioinfodmp=idummy
         if (idummy.ne.0) then  ! overwrite last i/o request
            rewind(87)
            write(87,86)
   86       format(' 0')
         endif
   87    continue
         close(unit=87)
   88    continue
         if (ioinfodmp.ne.0) write(6,*) 'Output:',ioinfodmp
      endif

      tdmp(1)=ioinfodmp
      call gop(tdmp,tdmp(3),'+  ',1)
      ioinfodmp=tdmp(1)
      if (ioinfodmp.lt.0) maxstep=abs(ioinfodmp)
      if (istep.ge.maxstep.or.ioinfodmp.eq.-2) lastep=1

      return
      end
c-----------------------------------------------------------------------
      subroutine prepost(ifdoin,prefin)

c     Store results for later postprocessing
c
c     Recent updates:
c
c     p65 now indicates the number of parallel i/o files; iff p66 >= 6
c
c     we now check whether we are going to checkpoint in set_outfld
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common /scrcg/ pm1 (lx1,ly1,lz1,lelv)

      character*3    prefin,prefix

      logical  ifdoin

      if (ioinfodmp.eq.-2) return

#ifdef TIMER
      etime1=dnekclock_sync()
#endif

      prefix = prefin
      if (prefix.eq.'his') prefix = '   '

      if (ifdoin) then
         icalld=icalld+1
         nprep=icalld

         call prepost_map(0) ! map pr and axisymm. arrays
         call outfld(prefix)
         call prepost_map(1) ! map back axisymm. arrays

#ifdef TIMER
         tprep=tprep+dnekclock_sync()-etime1
#endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine prepost_map(isave) ! isave=0-->fwd, isave=1-->bkwd

c     Store results for later postprocessing

      include 'SIZE'
      include 'TOTAL'
C
C     Work arrays and temporary arrays
C
      common /scruz/ vxax   (lx1,ly1,lelv)
     $             , vyax   (lx1,ly1,lelv)
     $             , prax   (lx2,ly2,lelv)
     $             , yax    (lx1,ly1,lelt)
      common /scrmg/ tax    (lx1,ly1,lelt,ldimt)
      common /scrcg/ pm1    (lx1,ly1,lz1,lelv)
C
c
      common /prepst/ pa(lx1,ly2,lz2),pb(lx1,ly1,lz2)
      integer e

      if (isave.eq.0) then ! map to GLL grid

         if (ifaxis) then
            ntotm1 = lx1*ly1*nelt
            call copy (yax,ym1,ntotm1)
            do 5 e=1,nelt
               if (ifrzer(e)) then
                  call mxm  (ym1(1,1,1,e),lx1,iatjl1,ly1,pb,ly1)
                  call copy (ym1(1,1,1,e),pb,lx1*ly1)
               endif
    5       continue
            if (ifflow) then
               ntotm1 = lx1*ly1*nelt
               ntotm2 = lx2*ly2*nelt
               call copy (vxax,vx,ntotm1)
               call copy (vyax,vy,ntotm1)
               call copy (prax,pr,ntotm2)
               do 10 e=1,nelt
                  if (ifrzer(e)) then
                     call mxm  (vx(1,1,1,e),lx1,iatjl1,ly1,pb,ly1)
                     call copy (vx(1,1,1,e),pb,lx1*ly1)
                     call mxm  (vy(1,1,1,e),lx1,iatjl1,ly1,pb,ly1)
                     call copy (vy(1,1,1,e),pb,lx1*ly1)
                     call mxm  (pr(1,1,1,e),lx2,iatjl2,ly2,pb,ly2)
                     call copy (pr(1,1,1,e),pb,lx2*ly2)
                  endif
 10            continue
            endif
            if (ifheat) then
               ntotm1 = lx1*ly1*nelt
               do 15 ifldt=1,npscal+1
                  call copy (tax(1,1,1,ifldt),t(1,1,1,1,ifldt),ntotm1)
 15            continue
               do 30 e=1,nelt
                  if (ifrzer(e)) then
                    do 25 ifldt=1,npscal+1
                      call mxm  (t(1,1,1,e,ifldt),lx1,iatjl1,ly1,
     $                                                  pb,ly1)
                      call copy (t(1,1,1,e,ifldt),pb,lx1*ly1)
 25                 continue
                  endif
 30            continue
            endif
         endif
C        Map the pressure onto the velocity mesh
C
         ntott = lx1*ly1*lz1*nelt
         ntot1 = lx1*ly1*lz1*nelt
         nyz2  = ly2*lz2
         nxy1  = lx1*ly1
         nxyz  = lx1*ly1*lz1
         nxyz2 = lx2*ly2*lz2
C
         
         call rzero(pm1,ntott)
         if (ifsplit) then
            call copy(pm1,pr,ntot1)
         elseif (if_full_pres) then
            call rzero(pm1,ntot1)
            do e=1,nelt
               call copy(pm1(1,1,1,e),pr(1,1,1,e),nxyz2)
            enddo
         else
            do 1000 e=1,nelt
               call mxm (ixm21,lx1,pr(1,1,1,e),lx2,pa(1,1,1),nyz2)        
               do 100 iz=1,lz2
                  call mxm (pa(1,1,iz),lx1,iytm21,ly2,pb(1,1,iz),ly1)
  100          continue
               call mxm (pb(1,1,1),nxy1,iztm21,lz2,pm1(1,1,1,e),lz1)
 1000       continue
         endif

      else       ! map back

         if (ifaxis) then
            ntot1 = lx1*ly1*nelt
            call copy (ym1,yax,ntot1)
            if (ifflow) then
               ntot1 = lx1*ly1*nelt
               ntot2 = lx2*ly2*nelt
               call copy (vx,vxax,ntot1)
               call copy (vy,vyax,ntot1)
               call copy (pr,prax,ntot2)
            endif
            if (ifheat) then
               ntot1 = lx1*ly1*nelt
               do 3000 ifldt=1,npscal+1
                  call copy (t(1,1,1,1,ifldt),tax(1,1,1,ifldt),ntot1)
 3000          continue
            endif
         endif

      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine outfld(prefix)

c     output .fld file 

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
C
C     Work arrays and temporary arrays
C
      common /scrcg/ pm1 (lx1,ly1,lz1,lelv)
c
c     note, this usage of CTMP1 will be less than elsewhere if NELT ~> 3.
      parameter (lxyz=lx1*ly1*lz1)
      parameter (lpsc9=ldimt1+9)
      common /ctmp1/ tdump(lxyz,lpsc9)
      real*4         tdump
      real           tdmp(4)
      equivalence   (tdump,tdmp)

      real*4         test_pattern

      character*3    prefix
      character*1    fhdfle1(132)
      character*132   fhdfle
      equivalence   (fhdfle,fhdfle1)

      character*1 excode(30)
      character*10 frmat

      common /nopenf/ nopen(99)

      common /rdump/ ntdump
      data ndumps / 0 /

      logical ifxyo_s

      if(nio.eq.0) then 
        WRITE(6,1001) istep,time
 1001   FORMAT(/,i9,1pe12.4,' Write checkpoint')
      endif
      call nekgsync()      

      p66 = param(66)
      if (abs(p66).eq.6) then
         call mfo_outfld(prefix)
         return
      endif

      ifxyo_s = ifxyo              ! Save ifxyo

      iprefix = i_find_prefix(prefix,99)

      ierr = 0
      if (nid.eq.0) then

c       Open new file for each dump on /cfs
        nopen(iprefix)=nopen(iprefix)+1

        if (prefix.eq.'   '.and.nopen(iprefix).eq.1) ifxyo = .true. ! 1st file

        if (prefix.eq.'rst'.and.max_rst.gt.0) 
     $         nopen(iprefix) = mod1(nopen(iprefix),max_rst) ! restart

        call file2(nopen(iprefix),prefix)
        if (p66.lt.1.0) then
           open(unit=24,file=fldfle,form='formatted',status='unknown')
        else
           call byte_open (fldfle,ierr)
c          write header as character string
           call blank(fhdfle,132)
        endif
      endif
      call bcast(ifxyo,lsize)
      if(p66.ge.1.0)
     $   call err_chk(ierr,'Error opening file in outfld. Abort. $')

C     Figure out what goes in EXCODE
      CALL BLANK(EXCODE,30)
      NDUMPS=NDUMPS+1
      i=1
      if (mod(p66,1.0).eq.0.0) then !old header format
         IF(IFXYO) then
            EXCODE(1)='X'
            EXCODE(2)=' '
            EXCODE(3)='Y'
            EXCODE(4)=' '
            i = 5
            IF(IF3D) THEN
              EXCODE(i)  ='Z'
              EXCODE(i+1)=' '
              i = i + 2
            ENDIF
         ENDIF
         IF(IFVO) then
            EXCODE(i)  ='U'
            EXCODE(i+1)=' '
            i = i + 2
         ENDIF
         IF(IFPO) THEN
           EXCODE(i)='P'
           EXCODE(i+1)=' '
           i = i + 2
         ENDIF
         IF(IFTO) THEN
           EXCODE(i)='T '
           EXCODE(i+1)=' '
           i = i + 1
         ENDIF
         do iip=1,ldimt1
            if (ifpsco(iip)) then
              write(excode(iip+I)  ,'(i1)') iip
              write(excode(iip+I+1),'(a1)') ' '
              i = i + 1 
            endif
         enddo
      else
         !new header format
         IF (IFXYO) THEN
            EXCODE(i)='X'
            i = i + 1
         ENDIF
         IF (IFVO) THEN
            EXCODE(i)='U'
            i = i + 1
         ENDIF
         IF (IFPO) THEN
            EXCODE(i)='P'
            i = i + 1
         ENDIF
         IF (IFTO) THEN
            EXCODE(i)='T'
            i = i + 1
         ENDIF
         IF (LDIMT.GT.1) THEN
            NPSCALO = 0
            do k = 1,ldimt-1
              if(ifpsco(k)) NPSCALO = NPSCALO + 1
            enddo
            IF (NPSCALO.GT.0) THEN
               EXCODE(i) = 'S'
               WRITE(EXCODE(i+1),'(I1)') NPSCALO/10
               WRITE(EXCODE(i+2),'(I1)') NPSCALO-(NPSCALO/10)*10
            ENDIF
         ENDIF
      endif
     

C     Dump header
      ierr = 0
      if (nid.eq.0) call dump_header(excode,p66,ierr)
      call err_chk(ierr,'Error dumping header in outfld. Abort. $')

      call get_id(id)

      nxyz  = lx1*ly1*lz1

      ierr = 0
      do ieg=1,nelgt

         jnid = gllnid(ieg)
         ie   = gllel (ieg)

         if (nid.eq.0) then
            if (jnid.eq.0) then
               call fill_tmp(tdump,id,ie)
            else
               mtype=2000+ie
               len=4*id*nxyz
               dum1=0.
               call csend (mtype,dum1,wdsize,jnid,nullpid)
               call crecv2 (mtype,tdump,len,jnid)
            endif
            if(ierr.eq.0) call out_tmp(id,p66,ierr)
         elseif (nid.eq.jnid) then
            call fill_tmp(tdump,id,ie)
            dum1=0.
            mtype=2000+ie
            len=4*id*nxyz
            call crecv2 (mtype,dum1,wdsize,node0)
            call csend (mtype,tdump,len,node0,nullpid)
         endif
      enddo
      call err_chk(ierr,'Error writing file in outfld. Abort. $')

      ifxyo = ifxyo_s           ! restore ifxyo

      if (nid.eq.0) call close_fld(p66,ierr)
      call err_chk(ierr,'Error closing file in outfld. Abort. $')

      return
      end
c-----------------------------------------------------------------------
      subroutine file2(nopen,PREFIX)
C----------------------------------------------------------------------
C
C     Defines machine specific input and output file names.
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'PARALLEL'
C
      CHARACTER*132 NAME
      CHARACTER*1   SESS1(132),PATH1(132),NAM1(132)
      EQUIVALENCE  (SESSION,SESS1)
      EQUIVALENCE  (PATH,PATH1)
      EQUIVALENCE  (NAME,NAM1)
      CHARACTER*1  DMP(4),FLD(4),REA(4),HIS(4),SCH(4) ,ORE(4), NRE(4)
      CHARACTER*4  DMP4  ,FLD4  ,REA4  ,HIS4  ,SCH4   ,ORE4  , NRE4
      EQUIVALENCE (DMP,DMP4), (FLD,FLD4), (REA,REA4), (HIS,HIS4)
     $          , (SCH,SCH4), (ORE,ORE4), (NRE,NRE4)
      CHARACTER*1  NUMRL(0:9)
      DATA DMP4,FLD4,REA4 /'.dmp','.fld','.rea'/
      DATA HIS4,SCH4      /'.his','.sch'/
      DATA ORE4,NRE4      /'.ore','.nre'/
      DATA NUMRL          /'0','1','2','3','4','5','6','7','8','9'/
      CHARACTER*78  STRING
c
      character*1    prefix(3)
C
      call blank(name  ,132)
      call blank(fldfle,132)
C
      LS=LTRUNC(SESSION,132)
      LPP=LTRUNC(PATH,132)
      LSP=LS+LPP
      l = 0

c     Construct file names containing full path to host:
c      DO 100 I=1,LPP
c         l = l+1
c         NAM1(l)=PATH1(I)
c  100 CONTINUE
C
      if (prefix(1).ne.' '.and.prefix(2).ne.' '.and.
     $     prefix(3).ne.' ') then
         do i=1,3
            l = l+1
            NAM1(l)=prefix(i)
         enddo
      endif
C
      DO 200 I=1,LS
         l = l+1
         NAM1(l)=SESS1(I)
  200 CONTINUE
C
C .fld file
      DO 300 I=1,4
         l = l+1
         NAM1(l)=FLD(I)
  300 CONTINUE
      if (nopen.lt.100) then
C        less than 100 dumps....
         ITEN=NOPEN/10
         l = l+1
         NAM1(l)=NUMRL(ITEN)
         IONE=MOD(NOPEN,10)
         l = l+1
         NAM1(l)=NUMRL(IONE)
      elseif (nopen.lt.1000) then
C        less than 1000 dumps....
         IHUN=NOPEN/100
         l = l+1
         NAM1(l)=NUMRL(IHUN)
         ITEN=MOD(NOPEN,100)/10
         l = l+1
         NAM1(l)=NUMRL(ITEN)
         IONE=MOD(NOPEN,10)
         l = l+1
         NAM1(l)=NUMRL(IONE)
      elseif (nopen.lt.10000) then
C        less than 10000 dumps....
         ITHO=NOPEN/1000
         l = l+1
         NAM1(l)=NUMRL(ITHO)
         IHUN=MOD(NOPEN,1000)/100
         l = l+1
         NAM1(l)=NUMRL(IHUN)
         ITEN=MOD(NOPEN,100)/10
         l = l+1
         NAM1(l)=NUMRL(ITEN)
         IONE=MOD(NOPEN,10)
         l = l+1
         NAM1(l)=NUMRL(IONE)
      endif
      FLDFLE=NAME
C
C     Write the name of the .fld file to the logfile.
C
      if (nio.eq.0) then
         call chcopy(string,fldfle,78)
         write(6,1000) istep,time,string
 1000    format(/,i9,1pe12.4,' OPEN: ',a78)
      endif
 
      return
      end
c=======================================================================
      subroutine rzero4(a,n)
      real*4 A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      return
      end
c=======================================================================
      subroutine copyX4(a,b,n)
      REAL*4 A(1)
      REAL   B(1)
      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      end
c=======================================================================
      subroutine copy4r(a,b,n)
      real   a(1)
      real*4 b(1)
      do i = 1, n
         a(i) = b(i)
      enddo
      return
      end
c=======================================================================
      function i_find_prefix(prefix,imax)
c
      character*3 prefix
      character*3 prefixes(99)
      save        prefixes
      data        prefixes /99*'...'/
c
      integer nprefix
      save    nprefix
      data    nprefix /0/
c
c     Scan existing list of prefixes for a match to "prefix"
c
      do i=1,nprefix
         if (prefix.eq.prefixes(i)) then
            i_find_prefix = i
            return
         endif
      enddo
c
c     If we're here, we didn't find a match.. bump list and return
c
      nprefix                = nprefix + 1
      prefixes(nprefix)      = prefix
      i_find_prefix          = nprefix
c
c     Array bounds check on prefix list
c
      if (nprefix.gt.99.or.nprefix.gt.imax) then
         write(6,*) 'Hey! nprefix too big! ABORT in i_find_prefix'
     $      ,nprefix,imax
         call exitt
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dump_header(excodein,p66,ierr)

      include 'SIZE'
      include 'TOTAL'

      character*30  excodein

      character*30 excode
      character*1  excode1(30)
      equivalence (excode,excode1) 

      real*4         test_pattern

      character*1 fhdfle1(132)
      character*132 fhdfle
      equivalence (fhdfle,fhdfle1)

      write(excode,'(A30)') excodein

      ikstep = istep
      do ik=1,10
         if (ikstep.gt.9999) ikstep = ikstep/10
      enddo

      call blank(fhdfle,132)

c       write(6,111)               !       print on screen
c     $     nelgt,lx1,ly1,lz1,time,istep,excode
c
      if (mod(p66,1.0).eq.0.0) then !       old header format
         if (p66.lt.1.0) then       !ASCII
           if(nelgt.lt.10000) then
            WRITE(24,'(4i4,1pe14.7,I5,1X,30A1,1X,A12)')
     $           NELGT,lx1,ly1,lz1,TIME,ikstep,(EXCODE1(I),I=1,30),
     $           'NELT,NX,NY,N'
           else
            WRITE(24,'(i10,3i4,1pe18.9,I9,1X,30A1,1X,A12)')
     $           NELGT,lx1,ly1,lz1,TIME,ikstep,(EXCODE1(I),I=1,30),
     $           'NELT,NX,NY,N'
           endif
         else                       !Binary
            if (nelgt.lt.10000) then
               WRITE(fhdfle,'(4I4,1pe14.7,I5,1X,30A1,1X,A12)')
     $              NELGT,lx1,ly1,lz1,TIME,ikstep,(EXCODE1(I),I=1,30),
     $              ' 4 NELT,NX,NY,N'
            else
               write(fhdfle,'(i10,3i4,1P1e18.9,i9,1x,30a1)')
     $         nelgt,lx1,ly1,lz1,time,istep,(excode1(i),i=1,30)
            endif
            call byte_write(fhdfle,20,ierr)
         endif
      else                        !       new header format
         if (p66.eq.0.1) then
            write(24,111)
     $           nelgt,lx1,ly1,lz1,time,istep,excode
        else       
             write(fhdfle,111)
     $            nelgt,lx1,ly1,lz1,time,istep,excode
             call byte_write(fhdfle,20,ierr)
        endif
 111    FORMAT(i10,1x,i2,1x,i2,1x,i2,1x,1P1e18.9,1x,i9,1x,a)
      endif

      if(ierr.ne.0) return

      CDRROR=0.0
      if (p66.LT.1.0) then       !       formatted i/o
         WRITE(24,'(6G11.4)')(CDRROR,I=1,NELGT)   ! dummy 
      else
C       write byte-ordering test pattern to byte file...
        test_pattern = 6.54321
        call byte_write(test_pattern,1,ierr)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine fill_tmp(tdump,id,ie)
C
      include 'SIZE'
      include 'TOTAL'
c
      common /scrcg/ pm1 (lx1,ly1,lz1,lelv)
C
C     Fill work array
C
      parameter (lxyz=lx1*ly1*lz1)
      parameter (lpsc9=ldimt1+9)
      real*4 tdump(lxyz,lpsc9)
C
      nxyz = lx1*ly1*lz1
c
      ID=0
      IF(IFXYO)then
         ID=ID+1
         CALL COPYx4(TDUMP(1,ID),XM1(1,1,1,IE),NXYZ)
         ID=ID+1
         CALL COPYx4(TDUMP(1,ID),YM1(1,1,1,IE),NXYZ)
         IF(IF3D) then
            ID=ID+1
            CALL COPYx4(TDUMP(1,ID),ZM1(1,1,1,IE),NXYZ)
         ENDIF
      ENDIF
c
      IF(IFVO)then
         IF (IE.LE.NELT) then
            ID=ID+1
            CALL COPYx4(TDUMP(1,ID),VX(1,1,1,IE),NXYZ)
            ID=ID+1
            CALL COPYx4(TDUMP(1,ID),VY(1,1,1,IE),NXYZ)
            IF(IF3D)then
               ID=ID+1
               CALL COPYx4(TDUMP(1,ID),VZ(1,1,1,IE),NXYZ)
            ENDIF
         ELSE
            ID=ID+1
            CALL RZERO4(TDUMP(1,ID),NXYZ)
            ID=ID+1
            CALL RZERO4(TDUMP(1,ID),NXYZ)
            IF(IF3D)then
               ID=ID+1
               CALL RZERO4(TDUMP(1,ID),NXYZ)
            ENDIF
         ENDIF
      ENDIF
      IF(IFPO)then
         IF (IE.LE.NELT) then
            ID=ID+1
            CALL COPYx4(TDUMP(1,ID),PM1(1,1,1,IE),NXYZ)
         ELSE
            ID=ID+1
            CALL RZERO4(TDUMP(1,ID),NXYZ)
         ENDIF
      ENDIF
      IF(IFTO)then
         ID=ID+1
         CALL COPYx4(TDUMP(1,ID),T(1,1,1,IE,1),NXYZ)
      ENDIF
C     PASSIVE SCALARS
      do iip=1,ldimt1
         if (ifpsco(iip)) then
            id=id+1
            call copyX4(tdump(1,id),t(1,1,1,ie,iip+1),nxyz)
        endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_id(id) !  Count amount of data to be shipped

      include 'SIZE'
      include 'TOTAL'

      id=0

      if (ifxyo) id=id+ldim
      if (ifvo)  id=id+ldim
      if (ifpo)  id=id+1
      if (ifto)  id=id+1

      do iip=1,ldimt1
         if (ifpsco(iip)) id=id+1     !     Passive scalars
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine close_fld(p66,ierr)

      include 'SIZE'
      include 'TOTAL'

      if (nid.eq.0) then
         if (p66.lt.1) then
            close(unit=24)
         else
            call byte_close(ierr)
         endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine out_tmp(id,p66,ierr)

      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)
      parameter (lpsc9=ldimt1+9)

      common /ctmp1/ tdump(lxyz,lpsc9)
      real*4         tdump

      character*11 frmat

      nxyz = lx1*ly1*lz1

      call blank(frmat,11)
      if (id.le.9) then
         WRITE(FRMAT,1801) ID
 1801    FORMAT('(1p',I1,'e14.6)')
      else
         WRITE(FRMAT,1802) ID
 1802    FORMAT('(1p',I2,'e14.6)')
      endif

      if (p66.lt.1.0) then
C       formatted i/o
        WRITE(24,FRMAT)
     $      ((TDUMP(I,II),II=1,ID),I=1,NXYZ)
      else
C        C binary i/o
         do ii=1,id
            call byte_write(tdump(1,ii),nxyz,ierr)
            if(ierr.ne.0) goto 101
         enddo
      endif
 101  continue

      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_outfld(prefix)  ! muti-file output

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      common /scrcg/ pm1 (lx1,ly1,lz1,lelv)  ! mapped pressure

      integer*8 offs0,offs,nbyte,stride,strideB,nxyzo8
      character*3 prefix
      logical ifxyo_s
 
      common /SCRUZ/  ur1(lxo*lxo*lxo*lelt)
     &              , ur2(lxo*lxo*lxo*lelt)
     &              , ur3(lxo*lxo*lxo*lelt)

      tiostart=dnekclock_sync()

      call io_init

      ifxyo_s = ifxyo 
      ifxyo_  = ifxyo
      nout = nelt
      nxo  = lx1
      nyo  = ly1
      nzo  = lz1
      if (ifreguo) then ! dump on regular (uniform) mesh
         if (nrg.gt.lxo) then
            if (nid.eq.0) write(6,*) 
     &         'WARNING: nrg too large, reset to lxo!'
            nrg = lxo
         endif
         nxo  = nrg
         nyo  = nrg
         nzo  = 1
         if(if3d) nzo = nrg
      endif
      offs0 = iHeaderSize + 4 + isize*nelgt

      ierr=0
      if (nid.eq.pid0) then
         call mfo_open_files(prefix,ierr)         ! open files on i/o nodes
      endif
      call err_chk(ierr,'Error opening file in mfo_open_files. $')
      call bcast(ifxyo_,lsize)
      ifxyo = ifxyo_
      call mfo_write_hdr                     ! create element mapping +

c     call exitti('this is wdsizo A:$',wdsizo)
                                             ! write hdr
      nxyzo8  = nxo*nyo*nzo
      strideB = nelB * nxyzo8*wdsizo
      stride  = nelgt* nxyzo8*wdsizo

      ioflds = 0
      ! dump all fields based on the t-mesh to avoid different
      ! topologies in the post-processor
      if (ifxyo) then
         offs = offs0 + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifreguo) then
            call map2reg(ur1,nrg,xm1,nout)
            call map2reg(ur2,nrg,ym1,nout)
            if (if3d) call map2reg(ur3,nrg,zm1,nout)
            call mfo_outv(ur1,ur2,ur3,nout,nxo,nyo,nzo)
         else
            call mfo_outv(xm1,ym1,zm1,nout,nxo,nyo,nzo)
         endif
         ioflds = ioflds + ldim
      endif
      if (ifvo ) then
         offs = offs0 + ioflds*stride + ldim*strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifreguo) then
             call map2reg(ur1,nrg,vx,nout)
             call map2reg(ur2,nrg,vy,nout)
             if (if3d) call map2reg(ur3,nrg,vz,nout)
             call mfo_outv(ur1,ur2,ur3,nout,nxo,nyo,nzo) 
         else
            call mfo_outv(vx,vy,vz,nout,nxo,nyo,nzo)  ! B-field handled thru outpost
         endif
         ioflds = ioflds + ldim
      endif
      if (ifpo ) then
         offs = offs0 + ioflds*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifreguo) then
            call map2reg(ur1,nrg,pm1,nout)
            call mfo_outs(ur1,nout,nxo,nyo,nzo)
         else
            call mfo_outs(pm1,nout,nxo,nyo,nzo)
         endif
         ioflds = ioflds + 1
      endif
      if (ifto ) then
         offs = offs0 + ioflds*stride + strideB
         call byte_set_view(offs,ifh_mbyte)
         if (ifreguo) then
            call map2reg(ur1,nrg,t,nout)
            call mfo_outs(ur1,nout,nxo,nyo,nzo)
         else
            call mfo_outs(t,nout,nxo,nyo,nzo)
         endif
         ioflds = ioflds + 1
      endif
      do k=1,ldimt-1
         if(ifpsco(k)) then
           offs = offs0 + ioflds*stride + strideB
           call byte_set_view(offs,ifh_mbyte)
           if (ifreguo) then
              call map2reg(ur1,nrg,t(1,1,1,1,k+1),nout)
              call mfo_outs(ur1,nout,nxo,nyo,nzo)
           else
              call mfo_outs(t(1,1,1,1,k+1),nout,nxo,nyo,nzo)
           endif
           ioflds = ioflds + 1
         endif
      enddo
      dnbyte = 1.*ioflds*nout*wdsizo*nxo*nyo*nzo

      if (if3d) then
         offs0   = offs0 + ioflds*stride
         strideB = nelB *2*4   ! min/max single precision
         stride  = nelgt*2*4
         ioflds  = 0
         ! add meta data to the end of the file
         if (ifxyo) then
            offs = offs0 + ldim*strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatav(xm1,ym1,zm1,nout)
            ioflds = ioflds + ldim
         endif
         if (ifvo ) then
            offs = offs0 + ioflds*stride + ldim*strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatav(vx,vy,vz,nout)
            ioflds = ioflds + ldim
         endif
         if (ifpo ) then
            offs = offs0 + ioflds*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatas(pm1,nout)
            ioflds = ioflds + 1
         endif
         if (ifto ) then
            offs = offs0 + ioflds*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            call mfo_mdatas(t,nout)
            ioflds = ioflds + 1
         endif
         do k=1,ldimt-1
            offs = offs0 + ioflds*stride + strideB
            call byte_set_view(offs,ifh_mbyte)
            if(ifpsco(k)) call mfo_mdatas(t(1,1,1,1,k+1),nout)
            ioflds = ioflds + 1
         enddo
         dnbyte = dnbyte + 2.*ioflds*nout*wdsizo
      endif

      ierr = 0
      if (nid.eq.pid0) then 
         if(ifmpiio) then
           call byte_close_mpi(ifh_mbyte,ierr)
         else
           call byte_close(ierr)
         endif
      endif
      call err_chk(ierr,'Error closing file in mfo_outfld. Abort. $')

      tio = dnekclock_sync()-tiostart
      if (tio.le.0) tio=1.

      dnbyte = glsum(dnbyte,1)
      dnbyte = dnbyte + iHeaderSize + 4. + isize*nelgt
      dnbyte = dnbyte/1024/1024
      if(nio.eq.0) write(6,7) istep,time,dnbyte,dnbyte/tio,
     &             nfileo
    7 format(/,i9,1pe12.4,' done :: Write checkpoint',/,
     &       30X,'file size = ',3pG12.2,'MB',/,
     &       30X,'avg data-throughput = ',0pf7.1,'MB/s',/,
     &       30X,'io-nodes = ',i5,/)

      ifxyo = ifxyo_s ! restore old value

      return
      end
c-----------------------------------------------------------------------
      subroutine io_init ! determine which nodes will output
      character*132 hname

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      ifdiro = .false.

      ifmpiio = .false.
      if(abs(param(65)).eq.1 .and. abs(param(66)).eq.6) ifmpiio=.true.
#ifdef NOMPIIO
      ifmpiio = .false.
#endif

      wdsizo = 4
      if (param(63).gt.0) wdsizo = 8 ! 64-bit .fld file
      nrg = lxo

      if(ifmpiio) then
        nfileo  = np
        nproc_o = 1
        fid0    = 0
        pid0    = nid
        pid1    = 0
      else
        if(param(65).lt.0) ifdiro = .true. !  p65 < 0 --> multi subdirectories
        nfileo  = abs(param(65))
        if(nfileo.eq.0) nfileo = 1
        if(np.lt.nfileo) nfileo=np   
        nproc_o = np / nfileo              !  # processors pointing to pid0
        fid0    = nid/nproc_o              !  file id
        pid0    = nproc_o*fid0             !  my parent i/o node
        pid1    = min(np-1,pid0+nproc_o-1) !  range of sending procs
      endif

      ! how many elements are present up to rank nid
      nn = nelt
      nelB = igl_running_sum(nn)
      nelB = nelB - nelt
     
      pid00 = glmin(pid0,1)

      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_open_files(prefix,ierr) ! open files

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      character*1 prefix(3)
      character*3 prefx

      character*132  fname
      character*1    fnam1(132)
      equivalence   (fnam1,fname)

      character*6  six,str
      save         six
      data         six / "??????" /


      character*1 slash,dot
      save        slash,dot
      data        slash,dot  / '/' , '.' /

      integer nopen(99,2)
      save    nopen
      data    nopen  / 198*0 /

      call blank(fname,132)      !  zero out for byte_open()

      iprefix        = i_find_prefix(prefix,99)
      if (ifreguo) then
         nopen(iprefix,2) = nopen(iprefix,2)+1
         nfld             = nopen(iprefix,2)
      else
         nopen(iprefix,1) = nopen(iprefix,1)+1
         nfld             = nopen(iprefix,1)
      endif

      call chcopy(prefx,prefix,3)        ! check for full-restart request
      if (prefx.eq.'rst'.and.max_rst.gt.0) nfld = mod1(nfld,max_rst)

      call restart_nfld( nfld, prefix ) ! Check for Restart option.
      if (prefx.eq.'   '.and.nfld.eq.1) ifxyo_ = .true. ! 1st file

      if(ifmpiio) then
        rfileo = 1
      else
        rfileo = nfileo
      endif
      ndigit = log10(rfileo) + 1

      lenp = ltrunc(path,132)
      call chcopy(fnam1(1),path,lenp)    
      k = 1 + lenp 
 
      if (ifdiro) then                                  !  Add directory
         call chcopy(fnam1(k),'A',1)
         k = k + 1
         call chcopy(fnam1(k),six,ndigit)  ! put ???? in string
         k = k + ndigit
         call chcopy(fnam1(k),slash,1)
         k = k + 1
      endif

      if (prefix(1).ne.' '.and.prefix(2).ne.' '.and.    !  Add prefix
     $    prefix(3).ne.' ') then
         call chcopy(fnam1(k),prefix,3)
         k = k + 3
      endif

      len=ltrunc(session,132)                           !  Add SESSION
      call chcopy(fnam1(k),session,len)
      k = k+len
     
      if (ifreguo) then
         len=4
         call chcopy(fnam1(k),'_reg',len)
         k = k+len
      endif

      call chcopy(fnam1(k),six,ndigit)                  !  Add file-id holder
      k = k + ndigit

      call chcopy(fnam1(k  ),dot,1)                     !  Add .f appendix
      call chcopy(fnam1(k+1),'f',1)
      k = k + 2

      write(str,4) nfld                                 !  Add nfld number
    4 format(i5.5)
      call chcopy(fnam1(k),str,5)
      k = k + 5

      call addfid(fname,fid0)

      if(ifmpiio) then
        if(nio.eq.0)    write(6,*) '      FILE:',fname 
        call byte_open_mpi(fname,ifh_mbyte,.false.,ierr) 
      else
        if(nid.eq.pid0) write(6,*) '      FILE:',fname 
        call byte_open(fname,ierr)
      endif
 
      return
      end
c-----------------------------------------------------------------------

      subroutine restart_nfld( nfld, prefix ) 
      include 'SIZE' ! For nio
      character*3 prefix
c
c     Check for Restart option and return proper nfld value.
c     Also, convenient spot to explain restart strategy.
c
c
c     The approach is as follows:
c
c         Prefix rs4 would indicate 4 files in the restart cycle.
c         
c         This would be normal usage for velocity only, with
c         checkpoints taking place in synch with standard io.
c
c         The resultant restart sequence might look like:
c
c         blah.fld09           Step 0
c         rs4blah.fld01             1
c         rs4blah.fld02             2
c
c         which implies that fld09 would be used as the i.c.
c         in the restart, rs4blah.fld01 would overwrite the
c         solution at Step 1, and rs4blah.fld02 would overwrite
c         Step 2.   Net result is that Steps 0-2 of the restart
c         session have solutions identical to those computed in
c         the prior run.   (It's important that both runs use
c         the same dt in this case.)
c
c
c         Another equally possible restart sequence would be:
c
c
c         blah.fld10           Step 0
c         rs4blah.fld03             1
c         rs4blah.fld04             2
c
c         Why the 3 & 4 ?   If one were to use only 1 & 2, there
c         is a risk that the system crashes while writing, say,
c         rs4blah.fld01, in which case the restart is compromised --
c         very frustrating at the end of a run that has been queued
c         for a week.  By providing a toggled sequence in pairs such as
c
c         (1,2),   (3,4),  (1,2), ...
c
c         ensures that one always has at least one complete restart
c         sequence.   In the example above, the following files would
c         be written, in order:
c
c         :
c         :
c         blah.fld09
c         rs4blah.fld01
c         rs4blah.fld02
c         blah.fld10
c         rs4blah.fld03
c         rs4blah.fld04
c         blah.fld11
c         rs4blah.fld01       (overwriting existing rs4blah.fld01)
c         rs4blah.fld02       (    "           "        "  .fld02)
c         blah.fld12
c         rs4blah.fld03       (   etc.  )
c         rs4blah.fld04
c         :
c         :
c
c
c         Other strategies are possible, according to taste.
c
c         Here is a data-intensive one:
c
c         MHD + double-precision restart, but single-precision std files
c
c         In this case, single-precision files are kept as the running
c         file sequence (i.e., for later post-processing) but dbl-prec.
c         is required for restart.  A total of 12 temporary restart files
c         must be saved:  (3 for velocity, 3 for B-field) x 2 for redundancy.
c
c         This is expressed, using hexadecimal notation (123456789abc...),
c         as prefix='rsc'.
c         
c         
      character*16 kst
      save         kst
      data         kst / '0123456789abcdef' /
      character*1  ks1(0:15),kin
      equivalence (ks1,kst)

c
c
      if (indx1(prefix,'rs',2).eq.1) then
         read(prefix,3) kin
    3    format(2x,a1)
         do kfld=1,15
            if (ks1(kfld).eq.kin) goto 10
         enddo
   10    if (kfld.eq.16) kfld=4 ! std. default
         nfln = mod1(nfld,kfld) ! Restart A (1,2) and B (3,4)
         if (nio.eq.0) write(6,*) nfln,nfld,kfld,' kfld'
         nfld = nfln
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine full_restart_save(iosave)

      integer iosave,save_size,nfld_save
      logical if_full_pres_tmp

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'

      if (PARAM(27).lt. 0) then
          nfld_save=abs(PARAM(27))  ! For full restart
      else 
          nfld_save=3
      endif
      save_size=8  ! For full restart

      dtmp = param(63)
      if_full_pres_tmp = if_full_pres     

      param(63) = 1 ! Enforce 64-bit output
      if_full_pres = .true. !Preserve mesh 2 pressure

      if (lastep.ne.1) call restart_save(iosave,nfld_save)

      param(63) = dtmp
      if_full_pres = if_full_pres_tmp 

      return
      end
c-----------------------------------------------------------------------
      subroutine restart_save(iosave,nfldi)

      integer iosave,nfldi


c     Save current fields for later restart.
c
c     Input arguments:
c
c       .iosave plays the usual triggering role, like iostep
c
c       .nfldi is the number of rs files to save before overwriting
c

      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'

      character*3 prefix

      character*17 kst
      save         kst
      data         kst / '0123456789abcdefx' /
      character*1  ks1(0:16)
      equivalence (ks1,kst)

      logical if_full_pres_tmp

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
      if (ifmhd) nfld2 = nfld/4

      i2 = iosav/2
      m1 = istep+iosav-iotest
      mt = mod(istep+iosav-iotest,iosav)
      prefix = '   '

      if (istep.gt.iosav/2  .and.
     $   mod(istep+iosav-iotest,iosav).lt.nfld2) then ! save
         write(prefix,'(A)') 'rs_'
c         write(prefix,3) ks1(mfld)
c    3    format('rs',a1)

         p66 = param(66)
         param(66) = 6
         if (ifmhd) call outpost2(bx,by,bz,pm,t,0,prefix)  ! first B
         call prepost (.true.,prefix)
         param(66) = p66

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine outpost(v1,v2,v3,vp,vt,name3)

      include 'SIZE'
      include 'INPUT'

      real v1(1),v2(1),v3(1),vp(1),vt(1)
      character*3 name3


      itmp=0
      if (ifto) itmp=1
      call outpost2(v1,v2,v3,vp,vt,itmp,name3)

      return
      end
c-----------------------------------------------------------------------
      subroutine outpost2(v1,v2,v3,vp,vt,nfldt,name3)

      include 'SIZE'
      include 'SOLN'
      include 'INPUT'

      parameter(ltot1=lx1*ly1*lz1*lelt)
      parameter(ltot2=lx2*ly2*lz2*lelv)
      common /outtmp/  w1(ltot1),w2(ltot1),w3(ltot1),wp(ltot2)
     &                ,wt(ltot1,ldimt)
c
      real v1(1),v2(1),v3(1),vp(1),vt(ltot1,1)
      character*3 name3
      logical if_save(ldimt)
c
      ntot1  = lx1*ly1*lz1*nelt
      ntot1t = lx1*ly1*lz1*nelt
      ntot2  = lx2*ly2*lz2*nelt

      if(nfldt.gt.ldimt) then
        write(6,*) 'ABORT: outpost data too large (nfldt>ldimt)!'
        call exitt
      endif

c store solution
      call copy(w1,vx,ntot1)
      call copy(w2,vy,ntot1)
      call copy(w3,vz,ntot1)
      call copy(wp,pr,ntot2)
      do i = 1,nfldt
         call copy(wt(1,i),t(1,1,1,1,i),ntot1t)
      enddo

c swap with data to dump
      call copy(vx,v1,ntot1)
      call copy(vy,v2,ntot1)
      call copy(vz,v3,ntot1)
      call copy(pr,vp,ntot2)
      do i = 1,nfldt
         call copy(t(1,1,1,1,i),vt(1,i),ntot1t)
      enddo

c dump data
      if_save(1) = ifto
      ifto = .false.
      if(nfldt.gt.0) ifto = .true. 
      do i = 1,ldimt-1
         if_save(i+1) = ifpsco(i)
         ifpsco(i) = .false.   
         if(i+1.le.nfldt) ifpsco(i) = .true.
      enddo

      call prepost(.true.,name3)

      ifto = if_save(1)
      do i = 1,ldimt-1
         ifpsco(i) = if_save(i+1) 
      enddo

c restore solution data
      call copy(vx,w1,ntot1)
      call copy(vy,w2,ntot1)
      call copy(vz,w3,ntot1)
      call copy(pr,wp,ntot2)
      do i = 1,nfldt
         call copy(t(1,1,1,1,i),wt(1,i),ntot1t)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_mdatav(u,v,w,nel)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      real u(lx1*ly1*lz1,1),v(lx1*ly1*lz1,1),w(lx1*ly1*lz1,1)

      real*4 buffer(1+6*lelt)

      integer e

      call nekgsync() ! clear outstanding message queues.

      nxyz = lx1*ly1*lz1
      n    = 2*ldim
      len  = 4 + 4*(n*lelt)   ! recv buffer size
      leo  = 4 + 4*(n*nelt) 
      ierr = 0

      ! Am I an I/O node?
      if (nid.eq.pid0) then
         j = 1
         do e=1,nel
            buffer(j+0) = vlmin(u(1,e),nxyz) 
            buffer(j+1) = vlmax(u(1,e),nxyz)
            buffer(j+2) = vlmin(v(1,e),nxyz) 
            buffer(j+3) = vlmax(v(1,e),nxyz)
            j = j + 4
            if(if3d) then
              buffer(j+0) = vlmin(w(1,e),nxyz) 
              buffer(j+1) = vlmax(w(1,e),nxyz)
              j = j + 2
            endif
         enddo

         ! write out my data
         nout = n*nel
         if(ierr.eq.0) then
           if(ifmpiio) then
             call byte_write_mpi(buffer,nout,-1,ifh_mbyte,ierr)
           else
             call byte_write(buffer,nout,ierr)
           endif
         endif

         ! write out the data of my childs
         idum  = 1
         do k=pid0+1,pid1
            mtype = k
            call csend(mtype,idum,4,k,0)           ! handshake
            call crecv(mtype,buffer,len)
            inelp = buffer(1)
            nout  = n*inelp
            if(ierr.eq.0) then 
              if(ifmpiio) then 
                call byte_write_mpi(buffer(2),nout,-1,ifh_mbyte,ierr)
              else
                call byte_write(buffer(2),nout,ierr)
              endif
            endif
         enddo
      else
         j = 1
         buffer(j) = nel
         j = j + 1
         do e=1,nel
            buffer(j+0) = vlmin(u(1,e),nxyz) 
            buffer(j+1) = vlmax(u(1,e),nxyz)
            buffer(j+2) = vlmin(v(1,e),nxyz) 
            buffer(j+3) = vlmax(v(1,e),nxyz)
            j = j + 4
            if(n.eq.6) then
              buffer(j+0) = vlmin(w(1,e),nxyz) 
              buffer(j+1) = vlmax(w(1,e),nxyz)
              j = j + 2
            endif
         enddo

         ! send my data to my pararent I/O node
         mtype = nid
         call crecv(mtype,idum,4)                ! hand-shake
         call csend(mtype,buffer,leo,pid0,0)     ! u4 :=: u8
      endif

      call err_chk(ierr,'Error writing data to .f00 in mfo_mdatav. $')

      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_mdatas(u,nel)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      real u(lx1*ly1*lz1,1)

      real*4 buffer(1+2*lelt)

      integer e

      call nekgsync() ! clear outstanding message queues.

      nxyz = lx1*ly1*lz1
      n    = 2
      len  = 4 + 4*(n*lelt)    ! recv buffer size
      leo  = 4 + 4*(n*nelt)
      ierr = 0

      ! Am I an I/O node?
      if (nid.eq.pid0) then
         j = 1
         do e=1,nel
            buffer(j+0) = vlmin(u(1,e),nxyz) 
            buffer(j+1) = vlmax(u(1,e),nxyz)
            j = j + 2
         enddo

         ! write out my data
         nout = n*nel
         if(ierr.eq.0) then 
           if(ifmpiio) then
             call byte_write_mpi(buffer,nout,-1,ifh_mbyte,ierr)
           else
             call byte_write(buffer,nout,ierr)
           endif
         endif

         ! write out the data of my childs
         idum  = 1
         do k=pid0+1,pid1
            mtype = k
            call csend(mtype,idum,4,k,0)           ! handshake
            call crecv(mtype,buffer,len)
            inelp = buffer(1)
            nout  = n*inelp
            if(ierr.eq.0) then 
              if(ifmpiio) then
                call byte_write_mpi(buffer(2),nout,-1,ifh_mbyte,ierr)
              else
                call byte_write(buffer(2),nout,ierr)
              endif
            endif
         enddo
      else
         j = 1
         buffer(j) = nel
         j = j + 1
         do e=1,nel
            buffer(j+0) = vlmin(u(1,e),nxyz) 
            buffer(j+1) = vlmax(u(1,e),nxyz)
            j = j + 2
         enddo

         ! send my data to my pararent I/O node
         mtype = nid
         call crecv(mtype,idum,4)                ! hand-shake
         call csend(mtype,buffer,leo,pid0,0)     ! u4 :=: u8
      endif

      call err_chk(ierr,'Error writing data to .f00 in mfo_mdatas. $')

      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_outs(u,nel,mx,my,mz)   ! output a scalar field

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      real u(mx,my,mz,1)

      common /SCRNS/ u4(2+lxo*lxo*lxo*2*lelt)
      real*4         u4
      real*8         u8(1+lxo*lxo*lxo*1*lelt)
      equivalence    (u4,u8)

      integer e

      umax = glmax(u,nel*mx*my*mz)
      umin = glmin(u,nel*mx*my*mz)
      if(nid.eq.0) write(6,'(A,2g13.5)') ' min/max:', umin,umax

      call nekgsync() ! clear outstanding message queues.
      if(mx.gt.lxo .or. my.gt.lxo .or. mz.gt.lxo) then
        if(nid.eq.0) write(6,*) 'ABORT: lxo too small'
        call exitt
      endif

      nxyz = mx*my*mz
      len  = 8 + 8*(lelt*nxyz)  ! recv buffer size
      leo  = 8 + wdsizo*(nel*nxyz)
      ntot = nxyz*nel

      idum = 1
      ierr = 0

      if (nid.eq.pid0) then

         if (wdsizo.eq.4) then             ! 32-bit output
             call copyx4 (u4,u,ntot)
         else
             call copy   (u8,u,ntot)
         endif
         nout = wdsizo/4 * ntot
         if(ierr.eq.0) then 
           if(ifmpiio) then
             call byte_write_mpi(u4,nout,-1,ifh_mbyte,ierr)
           else
             call byte_write(u4,nout,ierr)          ! u4 :=: u8
           endif
         endif

         ! write out the data of my childs
         idum  = 1
         do k=pid0+1,pid1
            mtype = k
            call csend(mtype,idum,4,k,0)       ! handshake
            call crecv(mtype,u4,len)
            nout  = wdsizo/4 * nxyz * u8(1)
            if (wdsizo.eq.4.and.ierr.eq.0) then
               if(ifmpiio) then
                 call byte_write_mpi(u4(3),nout,-1,ifh_mbyte,ierr)
               else
                 call byte_write(u4(3),nout,ierr)
               endif
            elseif(ierr.eq.0) then
               if(ifmpiio) then
                 call byte_write_mpi(u8(2),nout,-1,ifh_mbyte,ierr)
               else
                 call byte_write(u8(2),nout,ierr)
               endif
            endif
         enddo

      else

         u8(1)= nel
         if (wdsizo.eq.4) then             ! 32-bit output
             call copyx4 (u4(3),u,ntot)
         else
             call copy   (u8(2),u,ntot)
         endif

         mtype = nid
         call crecv(mtype,idum,4)            ! hand-shake
         call csend(mtype,u4,leo,pid0,0)     ! u4 :=: u8

      endif

      call err_chk(ierr,'Error writing data to .f00 in mfo_outs. $')

      return
      end
c-----------------------------------------------------------------------

      subroutine mfo_outv(u,v,w,nel,mx,my,mz)   ! output a vector field

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'

      real u(mx*my*mz,1),v(mx*my*mz,1),w(mx*my*mz,1)

      common /SCRNS/ u4(2+lxo*lxo*lxo*6*lelt)
      real*4         u4
      real*8         u8(1+lxo*lxo*lxo*3*lelt)
      equivalence    (u4,u8)

      integer e

      umax = glmax(u,nel*mx*my*mz)
      vmax = glmax(v,nel*mx*my*mz)
      wmax = glmax(w,nel*mx*my*mz)
      umin = glmin(u,nel*mx*my*mz)
      vmin = glmin(v,nel*mx*my*mz)
      wmin = glmin(w,nel*mx*my*mz)
      if(nid.eq.0) write(6,'(A,6g13.5)') ' min/max:', 
     $             umin,umax, vmin,vmax, wmin,wmax

      call nekgsync() ! clear outstanding message queues.
      if(mx.gt.lxo .or. my.gt.lxo .or. mz.gt.lxo) then
        if(nid.eq.0) write(6,*) 'ABORT: lxo too small'
        call exitt
      endif

      nxyz = mx*my*mz
      len  = 8 + 8*(lelt*nxyz*ldim)   ! recv buffer size (u4)
      leo  = 8 + wdsizo*(nel*nxyz*ldim)
      idum = 1
      ierr = 0

      if (nid.eq.pid0) then
         j = 0 
         if (wdsizo.eq.4) then             ! 32-bit output
             do iel = 1,nel
                call copyx4   (u4(j+1),u(1,iel),nxyz)
                j = j + nxyz
                call copyx4   (u4(j+1),v(1,iel),nxyz)
                j = j + nxyz
                if(if3d) then
                  call copyx4 (u4(j+1),w(1,iel),nxyz)
                  j = j + nxyz
                endif
             enddo
         else
             do iel = 1,nel
                call copy     (u8(j+1),u(1,iel),nxyz)
                j = j + nxyz
                call copy     (u8(j+1),v(1,iel),nxyz)
                j = j + nxyz
                if(if3d) then
                  call copy   (u8(j+1),w(1,iel),nxyz)
                  j = j + nxyz
                endif
             enddo
         endif
         nout = wdsizo/4 * ldim*nel * nxyz
         if(ierr.eq.0) then 
           if(ifmpiio) then
             call byte_write_mpi(u4,nout,-1,ifh_mbyte,ierr)
           else
             call byte_write(u4,nout,ierr)          ! u4 :=: u8
           endif
         endif

         ! write out the data of my childs
         do k=pid0+1,pid1
            mtype = k
            call csend(mtype,idum,4,k,0)           ! handshake
            call crecv(mtype,u4,len)
            nout  = wdsizo/4 * ldim*nxyz * u8(1)

            if (wdsizo.eq.4.and.ierr.eq.0) then
               if(ifmpiio) then
                 call byte_write_mpi(u4(3),nout,-1,ifh_mbyte,ierr)
               else
                 call byte_write(u4(3),nout,ierr)
               endif
            elseif(ierr.eq.0) then
               if(ifmpiio) then
                 call byte_write_mpi(u8(2),nout,-1,ifh_mbyte,ierr)
               else
                 call byte_write(u8(2),nout,ierr)
               endif
            endif
         enddo
      else

         u8(1) = nel
         if (wdsizo.eq.4) then             ! 32-bit output
             j = 2
             do iel = 1,nel
                call copyx4   (u4(j+1),u(1,iel),nxyz)
                j = j + nxyz
                call copyx4   (u4(j+1),v(1,iel),nxyz)
                j = j + nxyz
                if(if3d) then
                  call copyx4 (u4(j+1),w(1,iel),nxyz)
                  j = j + nxyz
                endif
             enddo
         else
             j = 1
             do iel = 1,nel
                call copy     (u8(j+1),u(1,iel),nxyz)
                j = j + nxyz
                call copy     (u8(j+1),v(1,iel),nxyz)
                j = j + nxyz
                if(if3d) then
                  call copy   (u8(j+1),w(1,iel),nxyz)
                  j = j + nxyz
                endif
             enddo
         endif

         mtype = nid
         call crecv(mtype,idum,4)            ! hand-shake
         call csend(mtype,u4,leo,pid0,0)     ! u4 :=: u8

      endif

      call err_chk(ierr,'Error writing data to .f00 in mfo_outv. $')
      return
      end
c-----------------------------------------------------------------------
      subroutine mfo_write_hdr          ! write hdr, byte key, els.

      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'PARALLEL'
      include 'RESTART'
      include 'TSTEP'
      real*4 test_pattern
      common /ctmp0/ lglist(0:lelt)

      character*132 hdr
      integer*8 ioff
      logical if_press_mesh

      call nekgsync()
      idum = 1

      if(ifmpiio) then
        nfileoo = 1   ! all data into one file
        nelo = nelgt
      else
        nfileoo = nfileo
        if(nid.eq.pid0) then                ! how many elements to dump
          nelo = nelt
          do j = pid0+1,pid1
             mtype = j
             call csend(mtype,idum,4,j,0)   ! handshake
             call crecv(mtype,inelp,4)
             nelo = nelo + inelp
          enddo
        else
          mtype = nid
          call crecv(mtype,idum,4)          ! hand-shake
          call csend(mtype,nelt,4,pid0,0)   ! u4 :=: u8
        endif 
      endif

      ierr = 0
      if(nid.eq.pid0) then

      call blank(hdr,132)              ! write header
      call blank(rdcode1,10)
      i = 1
      IF (IFXYO) THEN
         rdcode1(i)='X'
         i = i + 1
      ENDIF
      IF (IFVO) THEN
         rdcode1(i)='U'
         i = i + 1
      ENDIF
      IF (IFPO) THEN
         rdcode1(i)='P'
         i = i + 1
      ENDIF
      IF (IFTO) THEN
         rdcode1(i)='T'
         i = i + 1
      ENDIF
      IF (LDIMT.GT.1) THEN
         NPSCALO = 0
         do k = 1,ldimt-1
           if(ifpsco(k)) NPSCALO = NPSCALO + 1
         enddo
         IF (NPSCALO.GT.0) THEN
            rdcode1(i) = 'S'
            WRITE(rdcode1(i+1),'(I1)') NPSCALO/10
            WRITE(rdcode1(i+2),'(I1)') NPSCALO-(NPSCALO/10)*10
         ENDIF
      ENDIF

c     check pressure format
      if_press_mesh = .false.
      if (.not.ifsplit.and.if_full_pres) if_press_mesh = .true.
 
      write(hdr,1) wdsizo,nxo,nyo,nzo,nelo,nelgt,time,istep,fid0,nfileoo
     $            ,(rdcode1(i),i=1,10),p0th,if_press_mesh
    1 format('#std',1x,i1,1x,i2,1x,i2,1x,i2,1x,i10,1x,i10,1x,e20.13,
     &       1x,i9,1x,i6,1x,i6,1x,10a,1pe15.7,1x,l1)

      test_pattern = 6.54321           ! write test pattern for byte swap

      if(ifmpiio) then
        ! only rank0 (pid00) will write hdr + test_pattern
        call byte_write_mpi(hdr,iHeaderSize/4,pid00,ifh_mbyte,ierr)
        call byte_write_mpi(test_pattern,1,pid00,ifh_mbyte,ierr)
      else
        call byte_write(hdr,iHeaderSize/4,ierr)
        call byte_write(test_pattern,1,ierr)
      endif

      endif

      call err_chk(ierr,'Error writing header in mfo_write_hdr. $')

      ! write global element numbering for this group
      if(nid.eq.pid0) then
        if(ifmpiio) then
          ioff = iHeaderSize + 4 + nelB*isize
          call byte_set_view (ioff,ifh_mbyte)
          call byte_write_mpi(lglel,nelt,-1,ifh_mbyte,ierr)
        else
          call byte_write(lglel,nelt,ierr)
        endif

        do j = pid0+1,pid1
           mtype = j
           call csend(mtype,idum,4,j,0)   ! handshake
           len = 4*(lelt+1)
           call crecv(mtype,lglist,len)
           if(ierr.eq.0) then
             if(ifmpiio) then
              call byte_write_mpi(lglist(1),lglist(0),-1,ifh_mbyte,ierr)
             else
              call byte_write(lglist(1),lglist(0),ierr)
             endif
           endif
        enddo
      else
        mtype = nid
        call crecv(mtype,idum,4)          ! hand-shake
        
        lglist(0) = nelt
        call icopy(lglist(1),lglel,nelt)

        len = 4*(nelt+1)
        call csend(mtype,lglist,len,pid0,0)  
      endif 

      call err_chk(ierr,'Error writing global nums in mfo_write_hdr$')
      return
      end
c-----------------------------------------------------------------------
