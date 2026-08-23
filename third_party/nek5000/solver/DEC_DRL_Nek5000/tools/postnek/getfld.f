c----------------------------------------------------------------------
      subroutine getfld(newdump,jerr,if_call_setw,if_call_coef)
C----------------------------------------------------------------------
C
C     Modified from routine RESTART, 5-2-24 pff
C     (1) Open restart file(s)
C     (2) Check previous spatial discretization 
C     (3) Map (K1,N1) => (K2,N2) if necessary
C
C----------------------------------------------------------------------
      include 'basics.inc'
      include 'basicsp.inc'
 
      common /regularize/ regdir
      character*1  regdir
 
      parameter (lxr=nxm)
      parameter (lyr=nym)
      parameter (lzr=nzm)
      parameter (lxyzr=lxr*lyr*lzr)
C     note, this usage of CTMP1 will be less than elsewhere if nelt ~> 9.
      parameter (lpsc9=maxfld+3)
      common /ctmp1/ xdump(lxyzr,lpsc9)
      character*4    cdump(lxyzr,lpsc9)
      equivalence   (cdump,xdump)
      character*2   excoder(15)
      character*10  rdcode 
      character*1   rdcode1(10) 
      equivalence (rdcode,rdcode1)
      character*80 fname
      character*1  fname1(80)
      equivalence (fname1,fname)
 
      logical if_call_setw
      logical if_call_coef
 
      character*132 hname
      character*1   hname1(132)
      equivalence (hname1,hname)
 
      real*4 bytetest
      real*8 bytetest8
 
      character*1  fname_c(80)
      integer      fname_i(20)
      equivalence (fname_i,fname_c)
      character*2 s2
      character*3 s3
      character*4 s4
 
      integer idump
 
      integer icalld,nelhdr
      save    icalld,nelhdr
      data    icalld /0/
 
c     for "setwrk":                  pff 2/28/98
      common /sfldwrk/ odumpw,odump
      integer odumpw,odump
 
      logical if_byte_sw, if_byte_swap_test, if_byte_swap_test8
c
c     local logical flags to determine whether to copy data or not.
c
      logical ifgetx,ifgetz,ifgetu,ifgetw,ifgetp,ifgett,ifgtps(maxfld)
     $       ,ifgtim,ifok,iffmat
      integer iposx,iposz,iposu,iposw,iposp,ipost,ipsps(maxfld)
      integer rdsize
 
      common /fformwrk/ wk
      real wk(3*maxpts)
      logical iffform   ! .f00000 format flag
      character*4 dummy ! .f00000 hdr read 
      integer er(nelm)  ! .f00000 element mapping 
      integer e
      real p66          ! abs value param66

      jerr = 0
      close(unit=24,err=1)
    1 continue
      close(unit=27,err=11)
   11 continue
 
      if (icalld.eq.0) then
         nelhdr = nel
         odump  = -99
      endif
      icalld = icalld+1

c     if (newdump.eq.odump) return
      odump  = newdump
      odumpw = newdump
      newdumps=newdump
      nid=0
      nelv =nel
      nelt =nel
      nelgt=nel
      ifok   =.false.
      iffmat =.true.
      iffform=.false.
      p66 = abs(param(66))
      if (p66.ge.1.0) iffmat =.false.
      if (p66.eq.6.0) then
         iffform =.true.
      endif

      len = ltrunc(session_name,80)
      call blank(fname,80)
      call chcopy(fname,session_name,len)

      if (iffform) then
         if (newdump.ne.0) then
            if (newdump.le.99) then
               call chcopy(fname1(len+1),'0.f000',6)
               len = ltrunc(fname,80)
               write(s2,82) newdump
   82          format(i2.2)
               call chcopy(fname1(len+1),s2,2)
            elseif (newdump.le.999) then
               call chcopy(fname1(len+1),'0.f00',5)
               len = ltrunc(fname,80)
               write(s3,83) newdump
   83          format(i3.3)
               call chcopy(fname1(len+1),s3,3)
            else
               call chcopy(fname1(len+1),'0.f0',4)
               len = ltrunc(fname,80)
               write(s4,84) newdump
   84          format(i4.4)
               call chcopy(fname1(len+1),s4,4)
            endif
         endif
      else
         call chcopy(fname1(len+1),'.fld',4)

         call blank(hname,132)
         call chcopy(hname,session_name,len)
         call chcopy(hname1(len+1),'.fhd',4)
c
c     Append numerical suffix
c
         len = ltrunc(fname,80)
         if (newdump.ne.0) then
            if (newdump.le.99) then
               write(s2,92) newdump
   92          format(i2.2)
               call chcopy(fname1(len+1),s2,2)
               call chcopy(hname1(len+1),s2,2)
            elseif (newdump.le.999) then
               write(s3,93) newdump
   93          format(i3.3)
               call chcopy(fname1(len+1),s3,3)
               call chcopy(hname1(len+1),s3,3)
            else
               write(s4,94) newdump
   94          format(i4.4)
               call chcopy(fname1(len+1),s4,4)
               call chcopy(hname1(len+1),s4,4)
            endif
         endif
      endif
      len = ltrunc(fname,80)
c
c     prepare file name for "byte_open" call 
c
      call izero(fname_i,10)
      call chcopy(fname_c,fname,len)


      ifgetx=.true.
      ifgetz=.true.
      ifgetu=.true.
      ifgetw=.true.
      ifgetp=.true.
      ifgett=.true.
      do 8 i=1,maxfld-4
         ifgtps(i)=.true.
    8 continue

C     Standard Case:   Requested inputs are in the file.

      write(6,*) 
      write(6,*) 'this is iffmat:',iffmat, param(66)

      call chcopy(fld_name,fname,80)
      write(6,80) session_name
      write(6,80) fname
   80 format(a80)
      write(6,*) 
 
      if (iffmat) then
         write(6,*) 'opening file:',len,' ',(fname1(k),k=1,len)
         write(6,*) 'opening file:',len,' ',fname
         open (unit=24,file=fname,err=9,status='old')
         write(6,*) 'opened  file:',len,' ',fname
      elseif (iffform) then
         write(6,*) 'opening file:',len,' ',(fname1(k),k=1,len)
         open (unit=24,file=fname,err=9,status='old')
         close(unit=24)
         call byte_open(fname_c)
      elseif (p66.ge.1.0) then
         write(6,*) 'opening file:',len,' ',(fname1(k),k=1,len)
         open (unit=24,file=fname,err=9,status='old')
         close(unit=24)
         call byte_open(fname_c)
      endif
      goto 10
c
c     Check file
c
    9 continue
c
c        Error condition
c
         write(6,*) 'this is param(66):',param(66)
         len = ltrunc(fname,80)
         fname1(len+1) = '$'
         call prs('WARNING:  Could not open file.$')
         call prs(fname)
         jerr = 1
         return
   10 continue
 
 
      rdsize = 4
      ndumps_read = newdumps
      if (param(65).ne.0) ndumps_read = 1
      ndumps_read = 1                        !   11/04/02
 
      do 1000 jdump=1,ndumps_read
 
         if (nid.eq.0.or.jdump.eq.1) then
           call blank(hname,132)
           if (iffmat) then
            read(24,80,err=1500,end=1500) hname
           elseif(iffform) then
            call byte_read(hname,33)
           elseif(p66.ge.1.0) then
            call byte_read(hname,20)
           endif
c          write(6,*) ' got hname',nelgt,neltr,nel,nelhdr
c          write(6,('a132')) hname
 
           nxr=nx
           nyr=ny
           nzr=nz
           neltr=nel
           if (indx1(hname,'*',1).ne.0) then
c              hack to get around overflow in file header...
               read(hname,'(16x,1X,G13.4,5x,1X,10A2)',ERR=1501,END=1501)
     $         rstime,(excoder(i),i=1,10)
           elseif(iffform) then
               read(hname,*,ERR=1509) dummy
     $         ,  rdsize,nxr,nyr,nzr,neltr,nelgr,rstime,istepr
     $         ,  ifiler,nfiler
     $         ,  rdcode      ! copied from ic.f:parse_std_hdr
               do i =1,10
                  excoder(i)=rdcode1(i)
               enddo
           else
            if (nelhdr.le.9999) then
            READ(hname,'(4I4,1X,G13.4,I5,1X,10A2,i2)',ERR=1502,END=1502)
     $        neltr,nxr,nyr,nzr,rstime,istepr,(excoder(i),i=1,10),rdsize
            else
              read
     $        (hname,'(i10,3i4,1P1e18.9,i9,1x,10a2)',ERR=1503,END=1503)
     $        neltr,nxr,nyr,nzr,rstime,istepr,(excoder(i),i=1,10)
              rdsize = 4
            endif
           endif
           if (param(68).eq.2.0) 
     $      WRITE(29,'(4I4,1X,G13.4,I5,1X,10A2)')
     $      neltr,nxr,nyr,nzr,rstime,istepr,(excoder(i),i=1,10)
           write(6,*) ' done read from hname',nxr,nyr,nzr
 
           time=rstime
           if (rdsize.eq.0) rdsize=4
           write(6,*) 'this is rdsize:',rdsize
         endif
c
c        Reset spatial dimension according to contents of .fld file
c
         param(20) = nxr
         nx = nxr
         ny = nxr
         nz = 1
         if (nx.gt.nxm.or.ny.gt.nym.or.nz.gt.nzm) then
            write(6,*) 'ERROR - nxm,nym,nzm too small.',nxm,nym,nzm
            write(6,*) 'Increase in basics.inc and recompile:',nx,ny,nz
            call exitt
         endif
 
         if (ndim.eq.3) nz = nxr
         if (.not. ifsubset) then
            neltm=maxpts/(nx*ny*nz)
            if(neltr.ge.neltm) 
     $        write(6,*) "NELTM<NELTR. Some elements will not be read.",
     $                   " Increase maxpts in basicsp.inc ",neltm,neltr
            neltr=min(neltr,neltm)
            nel = min(neltr,nel)
         endif
         write(6,*) neltm,maxpts,nxr,nel,neltr,' NELTM '
         if (jdump.eq.1) then
C
C           Find out the position of the data via the header info.
C
C           Assumptions about position of data:
C
C           .If X is found then Y (Z) are implied
C           .If U is found then V (W) are implied
C           .Column data is space or , delimited.
C
C
C           Figure out position of data in file "IFILE"
C
            nouts=0
            iposx=0
            iposy=0
            iposz=0
            iposu=0
            iposv=0
            iposw=0
            iposp=0
            ipost=0
            do 40 i=1,npscal
               ipsps(i)=0
   40       continue
 
c           if (idump.eq.1) write(6,41) (excoder(j),j=1,10),idump
            write(6,41) (excoder(j),j=1,10),idump
   41       format(' excoder:',2x,10a2,i4)
            do 50 i=1,10
               if (excoder(i).eq.'X') then
                  nouts=nouts + 1
                  iposx=nouts
                  ifgetx = .true.
                  if(iffform) then
                    nouts=nouts +1
                    iposy=nouts
c                   ifgety=.true.
                    if(if3d) then
                      nouts=nouts + 1
                      iposz=nouts
                      ifgetz = .true.
                    endif
                  endif
               elseif (excoder(i).eq.'Y') then
                  nouts=nouts+1
                  iposy=nouts
                  ifgetx = .true.
               elseif (excoder(i).eq.'Z') then
                  nouts=nouts + 1
                  iposz=nouts
                  ifgetz = .true.
               elseif (excoder(i).eq.'U') then
                  nouts=nouts + 1
                  iposu=nouts
                  nouts=nouts+1
                  iposv=nouts
                  ifgetu = .true.
                  if (if3d) then
                     nouts=nouts + 1
                     iposw=nouts
                     ifgetw = .true.
                  endif
                  ifflow=.true.
               elseif (excoder(i).eq.'P') then
                  nouts=nouts + 1
                  iposp=nouts
                  ifgetp=.true.
               elseif (excoder(i).eq.'T') then
                  nouts=nouts + 1
                  ipost=nouts
                  ifgett=.true.
                  ifheat=.true.
               elseif (excoder(i).eq.'V') then
                  ifflow=.true.
               elseif (excoder(i).eq.'S') then
                  read(excoder(i+1),'(i1)') nps1
                  read(excoder(i+2),'(i1)') nps0
                  npsr= 10*nps1+nps0
                  nps  = npsr
                  if(npsr.gt.ndim-1) nps=ndim-1
                  do k=1,nps
                     nouts=nouts+1
                     ipsps(k)=nouts
                     ifgtps(k)=.true.
                     ifpsco(k)=.true.
                  enddo
               elseif (excoder(i).ne.' ') then
c                 write(6,*) 'Excoder:', i,excoder(i),nouts
                  read(excoder(i),111) ii
  111             format(i1)
 
                  nouts=nouts + 1
                  ipsps(ii) = nouts
                  ifgtps(ii)=.true.
                  ifpsco(ii)=.true.
               endif
         write(6,*) 
     $  'IFGNGM,IFGETX bbb',ifgngm,ifgetx,i,excoder(i),iposx,iposy
   50       continue
 
            lname=ltrunc(fname,80)
            if (nid.eq.0) write(6,61) (fname1(i),i=1,lname)
            if (nid.eq.0) write(6,62) iposu,iposv,iposw,ipost,nouts
   61       FORMAT(/,2X,'Reading from file ',80A1)
   62       FORMAT(2X,'Columns for data, U,V,W,T,N: ',5I4)
            write(6,*) 'IFGNGM,IFGETX ccc',ifgngm,ifgetx,iposx,iposy
 
C           Make sure the requested data is present in this file....
            if (iposx.eq.0) ifgetx=.false.
            if (iposy.eq.0) ifgetx=.false.
            if (iposz.eq.0) ifgetz=.false.
            if (iposu.eq.0) ifgetu=.false.
            if (iposw.eq.0) ifgetw=.false.
            if (iposp.eq.0) ifgetp=.false.
            if (ipost.eq.0) ifgett=.false.
                            ifgtim=.true.
            do 65 i=2,npscal
               if (ipsps(i).eq.0) ifgtps(i)=.false.
   65       continue
            write(6,*) 'IFGNGM,IFGETX ddd',ifgngm,ifgetx,iposx,iposy
            write(6,*) 'IFGNGM,IFGETX 333',rdsize,' rdsize'
C           End of restart file header evaluation.
         endif
C
C        Read the error estimators
C
         if_byte_sw = .false.
         if(nid.eq.0)then
            if (iffmat) then
             read(24,'(6g11.4)',end=1504)(cerror(1,i),i=1,neltr)
             if (param(68).eq.2.0) 
     $       write(29,'(6g11.4)')(cerror(1,i),i=1,neltr)
            elseif(rdsize.eq.4) then
c            read test pattern to determine if byte-swapping is req'd
             call byte_read(bytetest,1)
             if_byte_sw = if_byte_swap_test(bytetest)
            elseif(rdsize.eq.8) then
             call byte_read(bytetest8,2)
             if_byte_sw = if_byte_swap_test8(bytetest8)
            endif
         endif
         write(6,*) 'IFGNGM,IFGETX 1dd',ifgngm,ifgetx,iposx,iposy
C
C        Read the current dump, double buffer so that we can
C        fit the data on a distributed memory machine,
C        and so we won't have to read the restart file twice
C        in case of an incomplete data file.
C
         nxyzr = nxr*nyr*nzr
C
C        Read the data
C
         write(6,*) 'reading data',idump,nouts,sdump(1,1),rdsize

         i100 = 10
         if (neltr/i100.gt.100) i100 = i100*10
         if (neltr/i100.gt.100) i100 = i100*10
         if (neltr/i100.gt.100) i100 = i100*10
         if (neltr/i100.gt.100) i100 = i100*10
         if (neltr/i100.gt.100) i100 = i100*10
         if (neltr/i100.gt.100) i100 = i100*10

         if(iffform) then
           if(rdsize.eq.8) then
              call prs('8 byte reads not yet supported in getfld')
              call exitt
           endif
             
           call byte_read(er,neltr)
           if (if_byte_sw) call byte_reverse(er,neltr)

           if (ifgetx) then
              call byte_read(wk,nxyzr*neltr*ndim)
              l=1
              do e=1,neltr
                ieg = er(e)
                if (if_byte_sw) call byte_reverse(wk(l),ndim*nxyzr)
                call copy4r(xdump(1,iposx),wk(l      ),nxyzr)
                call copy4r(xdump(1,iposy),wk(l+nxyzr),nxyzr)
                if(ndim.eq.3)
     $            call copy4r(xdump(1,iposz),wk(l+2*nxyzr),nxyzr)
                l=l+ndim*nxyzr

                call mapdmp
     $            (sdump(1,1),xdump(1,iposx),ieg,nxr,nyr,nzr,.false.)
                call mapdmp
     $            (sdump(1,2),xdump(1,iposy),ieg,nxr,nyr,nzr,.false.)
                if (ndim.eq.3) call mapdmp
     $            (sdump(1,3),xdump(1,iposz),ieg,nxr,nyr,nzr,.false.)
              enddo
           endif
           if (ifgetu) then
              call byte_read(wk,nxyzr*neltr*ndim)
              l=1
              do e=1,neltr
                ieg = er(e)
                if (if_byte_sw) call byte_reverse(wk(l),ndim*nxyzr)
                call copy4r(xdump(1,iposu),wk(l),nxyzr)
                call copy4r(xdump(1,iposv),wk(l+nxyzr),nxyzr)
                if(ndim.eq.3)
     $            call copy4r(xdump(1,iposw),wk(l+2*nxyzr),nxyzr)
                l=l+ndim*nxyzr

                call mapdmp
     $            (sdump(1,4),xdump(1,iposu),ieg,nxr,nyr,nzr,.false.)
                call mapdmp
     $            (sdump(1,5),xdump(1,iposv),ieg,nxr,nyr,nzr,.false.)
                if (ndim.eq.3) call mapdmp
     $            (sdump(1,6),xdump(1,iposw),ieg,nxr,nyr,nzr,.false.)
                if (mod(e,1000).eq.0) then
                   umin = vlmin(xdump(1,iposu),nxyzr)
                   umax = vlmax(xdump(1,iposu),nxyzr)
                   write(6,*) e,er(e),umin,umax,' reading: umax'
                endif
              enddo
           endif
           if (ifgetp) then
              call byte_read(wk,nxyzr*neltr)
              l=1
              do e=1,neltr
                 ieg = er(e)
                 if (if_byte_sw) call byte_reverse(wk(l),nxyzr)
                 call copy4r(xdump(1,iposp),wk(l),nxyzr)
                 l=l+nxyzr

                call mapdmp
     $            (sdump(1,7),xdump(1,iposp),ieg,nxr,nyr,nzr,.false.)
              enddo
           endif
          if (ifgett) then
              call byte_read(wk,nxyzr*neltr)
              l=1
              do e=1,neltr
                 ieg = er(e)
                 if (if_byte_sw) call byte_reverse(wk(l),nxyzr)
                 call copy4r(xdump(1,ipost),wk(l),nxyzr)
                 l=l+nxyzr

                 call mapdmp
     $            (sdump(1,8),xdump(1,ipost),ieg,nxr,nyr,nzr,.false.)
              enddo
           endif
           do k=1,npscal 
              if (ifgtps(k)) then
               call byte_read(wk,nxyzr*neltr)
               l=1
               do e=1,neltr
                  ieg = er(e)
                  if (if_byte_sw) call byte_reverse(wk(l),nxyzr)
                  k8 = k+8
                  call copy4r(xdump(1,ipsps(k)),wk(l),nxyzr)
                  l=l+nxyzr
                  call mapdmp
     $              (sdump(1,k8),xdump(1,ipsps(k))
     $              ,ieg,nxr,nyr,nzr,.false.)
               enddo
              endif
           enddo
           np4=min(npscal,4)
           np4=npscal-4
           np4=min(np4,4)
           call byte_close()
         goto 199
         endif
          
         do 200 ier=1,neltr
            ieg = ier
            if (ifsubset) ieg = isubset(ier) ! point to last element if not in
            if (nid.eq.0) then
 
              if (mod(ieg,i100).eq.0 .or. ieg.eq.1 ) 
     $        WRITE(6,*) 'Reading',IEG,nouts,nxyzr,neltr
 
              if (iffmat) then
                 READ(24,*,ERR=1506,END=1506)
     $           ((xdump(ixyz,ii),ii=1,nouts),ixyz=1,nxyzr)
              elseif(rdsize.eq.8) then
                 call prs('8 byte reads not yet supported in getfld')
c                do ii=1,nouts
c                   call byte_read(xdump(1,ii),nxyzr)
c                enddo
              else
                 do ii=1,nouts
                    call byte_read(xdump(1,ii),nxyzr)
                 enddo
              endif
              ifok=.true.
            endif
C
C           Notify other processors that we've read the data OK.
C
c           CALL LBCAST(IFOK)
            if (.not.ifok) goto 1600
C
C           MAPDMP maps data from NXR to NX
C           (and sends data to the appropriate processor.)
C
C           The buffer SDUMP is used so that if an incomplete dump
C           file is found (e.g. due to UNIX io buffering!), then
C           the previous read data stored in VX,VY,.., is not corrupted.
C
c     write(6,*) ieg,ifgetx,ifgetz,iposx,iposz
c    $        ,xdump(1,iposx),xdump(1,iposy),xdump(1,iposz)
            if (ieg.le.nelt) then
              if (ifgetx) call mapdmp
     $        (sdump(1,1),xdump(1,iposx),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetx) call mapdmp
     $        (sdump(1,2),xdump(1,iposy),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetz) call mapdmp
     $        (sdump(1,3),xdump(1,iposz),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetu) call mapdmp
     $        (sdump(1,4),xdump(1,iposu),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetu) call mapdmp
     $        (sdump(1,5),xdump(1,iposv),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetw) call mapdmp
     $        (sdump(1,6),xdump(1,iposw),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgetp) call mapdmp
     $        (sdump(1,7),xdump(1,iposp),ieg,nxr,nyr,nzr,if_byte_sw)
              if (ifgett) call mapdmp
     $        (sdump(1,8),xdump(1,ipost),ieg,nxr,nyr,nzr,if_byte_sw)
              do i=1,npscal
                 i8 = i+8
                 if (ifgtps(i)) call mapdmp
     $              (sdump(1,i8),xdump(1,ipsps(i))
     $              ,ieg,nxr,nyr,nzr,if_byte_sw)
              enddo
C             passive scalars
              np4=min(npscal,4)
              np4=npscal-4
              np4=min(np4,4)
            endif
  200      continue               
            write(6,*) 'IFGNGM,IFGETX 2dd',ifgngm,ifgetx,iposx,iposy
C
C        Successfully read a complete field, store it.
C
C        General geometry?
         if (ifgetx) ifgngm=.true.
         write(6,*) 'IFGNGM,IFGETX aaa',ifgngm,ifgetx
C        passive scalars
         np4=min(4,npscal)
         np4=npscal-4
         np4=min(np4,4)
         time=rstime
         istep = istepr
  199 CONTINUE
 1000 continue
      goto 1600
 
C     Else, end of file found - notify other processors.
 1500    write(s,1591) jdump,ieg,'A'
         goto 1599
 1501    write(s,1591) jdump,ieg,'B'
         goto 1599
 1502    write(s,1591) jdump,ieg,'C'
         goto 1599
 1503    write(s,1591) jdump,ieg,'D'
         goto 1599
 1504    write(s,1591) jdump,ieg,'E'
         goto 1599
 1505    write(s,1591) jdump,ieg,'F'
         goto 1599
 1506    write(s,1591) jdump,ieg,'G'
         goto 1599
 1507    write(s,1591) jdump,ieg,'H'
         goto 1599
 1508    write(s,1591) jdump,ieg,'I'
         goto 1599
 1509    write(s,1591) jdump,ieg,'J'
         goto 1599
 1591    format
     $('Unable to read dump',I4,' near element',I8,1x,a1,'$')

 1599    ifok=.false.
         call prs(s)

 1600 continue
      write(6,*) 'IFGNGM,IFGETX 3dd',ifgngm,ifgetx,iposx,iposy
      write(6,*) 'IFGNGM,IFGETX 3de',nel,neltr
      if (jdump.eq.1.and.nid.eq.0) then
         write(6,1700) fname
         write(6,1701) ieg,ixyz
         write(6,1702) 
     $         ((xdump(jxyz,ii),ii=1,nouts),jxyz=ixyz-1,ixyz)
 1700    FORMAT(5X,'WARNING:  No data read in for file ',A80)
 1701    FORMAT(5X,'Failed on  element',I4,',  point',I5,'.')
 1702    FORMAT(5X,'Last read dump:',/,5G15.7)
      else
         idump=jdump-1
         ndumps=idump
         if (newdump.ne.1) idump=newdump
 
         if (nid.eq.0) write(6,1800) idump
 1800    FORMAT(2X,'Successfully read data from dump number',I3,'.')
         time=rstime
         istep = istepr
         idstep(idump)=istep
         dmptim(idump)=time
      endif
 
      if (p66.ge.1.0.and.p66.ne.6.0) then
         call byte_close()
      else
         close(unit=24)
      endif
            write(6,*) 'IFGNGM,IFGETX 4dd',ifgngm,ifgetx,iposx,iposy
 
      goto 6000
 
C     Can't open file...
 5000 continue
      if (nid.eq.0) write(6,5001) fname 
 5001 FORMAT(2X,'   *******   WARNING   *******    '
     $    ,/,2X,'   *******   WARNING   *******    '
     $    ,/,2X,'   Could not open fld file:'
     $    ,/,A80
     $   ,//,2X,'ASSUMING DEFAULT INITIAL CONDITIONS.')
c     call exitt

      if (p66.ge.1.0.and.p66.ne.6.0) then
         call byte_close()
      else
         close(unit=24,err=5009)
      endif
 
 5009 CONTINUE
 
 
C     End of IFILE loop
 6000 CONTINUE
 
      if (ifregz) then
         call prs('regularize$')
         write(6,*) 'call mapz: ',regdir
         call mapz(u,regdir)
         call mapz(v,regdir)
         call mapz(w,regdir)
         call mapz(p,regdir)
         call mapz(t,regdir)
      endif
 
      write(6,*) 'IFGNGM,IFGETX 5dd',ifgngm,ifgetx,iposx,iposy
      write(6,66) 
     $  ifavgupt,if_call_setw,if_call_coef,ifgetx,ifgngm,ifregz
   66 format('ifavgupt,call_setw,call_coef,ifgetx,ifgngm',7l4)
 
      if (ifavgupt)                      call avg_uvwpt_regular
            write(6,*) 'IFGNGM,IFGETX 6dd',ifgngm,ifgetx,iposx,iposy
      if (ifgetx)                        call reset_xc
            write(6,*) 'IFGNGM,IFGETX 7dd',ifgngm,ifgetx,iposx,iposy
      if (ifgetx.and.if_call_coef)       call coef
            write(6,*) 'IFGNGM,IFGETX 8dd',ifgngm,ifgetx,iposx,iposy
c     if (ifgngm)                        call coef
            write(6,*) 'IFGNGM,IFGETX 9dd',ifgngm,ifgetx,iposx,iposy
      if (if_call_setw)                  call setwrk(.true.)
            write(6,*) 'IFGNGM,IFGETX 0dd',ifgngm,ifgetx,iposx,iposy
c
c     call prs('CALLING avg_cylinder.$')
c     call avg_cylinder
C
            write(6,*) 'IFGNGM,IFGETX add',ifgngm,ifgetx,iposx,iposy
      return
      end
C
c-----------------------------------------------------------------------
      subroutine mapdmp(yd,xd,ieg,nxr,nyr,nzr,IF_BYTE_SW)
C----------------------------------------------------------------------
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      PARAMETER (LXYZ1=NXM*NYM*NZM)
      PARAMETER (LXR=NXM)
      PARAMETER (LYR=NYM)
      PARAMETER (LZR=NZM)
      PARAMETER (LXYZR=LXR*LYR*LZR)
c
      logical if_byte_sw
C
      dimension yd(nxr,nyr,nzr,nelm)
      dimension xd(lxyzr)
      NXYZ=NX*NY*NZ
      NXYR=NXR*NYR*NZR
C
C
      CALL COPY(YD(1,1,1,IEG),XD,NXYZ)
      if (if_byte_sw) call byte_reverse(yd(1,1,1,ieg),nxyz)
c
      return
      END
c-----------------------------------------------------------------------
      subroutine vrnvertQ(a,n)
      REAL*4 A(1)
      ioldexp=-2
      A(1)=REVERTq(A(1),ioldexp,izero)
      DO 100 I=2,N
         i1=i-1
         A(I)=REVERTq(A(I),inewexp,izero)
         if (izero.ne.0.or.mod(i1,1000).eq.0) then
            a(i1)=a(i1)*10.0**(inewexp-ioldexp)
         endif
         ioldexp=inewexp
  100 CONTINUE
      return
      END
      FUNCTION REVERTq(T,ioldexp,izero)
C
      REAL*4 REVERT1
      CHARACTER*4 T
      CHARACTER*12 VC
      CHARACTER*1  V1(12)
      EQUIVALENCE (V1,VC)
C
      INTEGER IC
      CHARACTER*1  C(4)
      EQUIVALENCE (IC,C)
C
      CHARACTER*4  WC
      CHARACTER*1  W1(4)
      EQUIVALENCE (W1,WC)
C
      CHARACTER*1 ALPSRT(4,0:63)
      CHARACTER*4 ALPSR4(0:63)
      SAVE        ALPH64,ALPSRT
      INTEGER INDEX(0:63),INTALP(0:63),LOG64
      INTEGER IWk  (0:63)
      SAVE    INDEX,INTALP,LOG64
      EQUIVALENCE (INTALP,ALPSRT)
      EQUIVALENCE (INTALP,ALPSR4)
C
      CHARACTER*1 ALPH64(0:63)
      INTEGER ICALLD
      SAVE    ICALLD
      DATA    ICALLD /0/
      DATA        ALPH64 
     $  /'1','2','3','4','5','6','7','8','9','0'
     $   ,'a','b','c','d','e','f','g','h','i','j'
     $   ,'k','l','m','n','o','p','q','r','s','t'
     $   ,'u','v','w','x','y','z'
     $   ,'A','B','C','D','E','F','G','H','I','J'
     $   ,'K','L','M','N','O','P','Q','R','S','T'
     $   ,'U','V','W','X','Y','Z','+','-'/
C
C     On first call, set up sorted list of Alpha values so
C     that we can use binary chop to evaluate the integer
C     values.
C
      IF (ICALLD.EQ.0) THEN
         ICALLD=1
         CALL BLANK(ALPSRT,256)
         DO 10 I=0,63
            ALPSRT(4,I)=ALPH64(I)
   10    CONTINUE
         CALL ISORT(INTALP,INDEX,64)      ! isort is now a sort, not a rank
c        CALL ISWAP(INTALP,Iwk,INDEX,64)  ! pff 12/2/04
         DO 20 I=0,63
   20    INDEX(I)=INDEX(I)-1
         LOG64=7
      ENDIF
C
C     Copy T to W
C
      WC=T
      CALL BLANK(C,4)
C
C     Begin divide and conquer search
C
      ILO=0
      IHI=63
      C(4)=W1(1)
      DO 100 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IHUN=INDEX(I)
c           write(6,103) ihun,i,ilog,ilo,ihi
  103 format(' ihun',6i9)
            GOTO 101
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  100 CONTINUE
  101 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(2)
      DO 200 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            ITEN=INDEX(I)
c           write(6,203) iten,i,ilog,ilo,ihi
  203 format(' iten',6i9)
            GOTO 201
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  200 CONTINUE
  201 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(3)
      DO 300 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IONE=INDEX(I)
c           write(6,303) ione,i,ilog,ilo,ihi
  303 format(' ione',6i9)
            GOTO 301
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  300 CONTINUE
  301 CONTINUE
C
      ILO=0
      IHI=63
      C(4)=W1(4)
      DO 400 ILOG=0,LOG64
         I=(ILO+IHI)/2
         IF (IC.EQ.INTALP(I)) THEN
            IEXP=INDEX(I)
c           write(6,403) iexp,i,ilog,ilo,ihi
  403 format(' iexp',6i9)
            GOTO 401
         ELSEIF (IC.LT.INTALP(I)) THEN
            IHI=I-1
         ELSE
            ILO=I+1
         ENDIF
  400 CONTINUE
  401 CONTINUE
C
c     MANTIS=4096*IHUN+64*ITEN+IONE
      MANTIS=4096*IHUN+64*ITEN+IONE-131072
      IEXP  =IEXP-31
      IF (MANTIS.LT.0) THEN
         MANTIS=ABS(MANTIS)
         WRITE(VC,1001) MANTIS,IEXP
      ELSE
         WRITE(VC,1002) MANTIS,IEXP
      ENDIF
 1001 FORMAT('-0.',I5.5,'E',I3.2)
 1002 FORMAT(' 0.',I5.5,'E',I3.2)
C
      READ(VC,2001) W
 2001 FORMAT(E12.5)
 2002 FORMAT(2X,A12)
C
      izero=1
      if (w.eq.0.0) izero=0
      ioldexp=iexp
      REVERTq=W
      return
      END
c-----------------------------------------------------------------------
      subroutine vrnvert(a,n)
      REAL*4 A(1)
      DO 100 I=1,N
         A(I)=REVERT(A(I))
c        write(6,*) a(i),i
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine copy_byte_sw(y,x,n)
c
      character*1 y(0:3,1),x(0:3,1)
c
      do i=1,n
         do j=0,3
            y(3-j,i) = x(j,i)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      logical function if_byte_swap_test8(bytetest)
c
      real*8 bytetest
      real*8 test_pattern
      save   test_pattern
c
      test_pattern = 6.54321
      if_byte_swap_test8 = .false.
      if (bytetest.ne.test_pattern) if_byte_swap_test8 = .true.
c
      write(6,*) 'Byte swap8:',if_byte_swap_test8,bytetest
      return
      end
c-----------------------------------------------------------------------
      logical function if_byte_swap_test(bytetest)
c
      real*4 bytetest,zzz
      real*4 test_pattern
      save   test_pattern
c
      test_pattern = 6.54321
      eps          = 0.0005
      if_byte_swap_test = .true.
      if (abs(bytetest-test_pattern).lt.eps) if_byte_swap_test = .false.

      zzz = bytetest
      call byte_reverse(zzz,1)


      write(6,*) 'Byte swap:',if_byte_swap_test,bytetest,zzz
   
      xxx = bytetest-test_pattern
      yyy = zzz     -test_pattern
      if(xxx.gt.eps.and.yyy.gt.eps) then
         write(6,*)  'ERROR!! COULD NOT FIND BYTESWAP PATTERN'
         write(6,*)  'CHECK PARAM(66) IN .REA FILE:'
         write(6,*)  '   PARAM(66) = 0   --> ASCII  fld'
         write(6,*)  '   PARAM(66) = >=1 --> BINARY fld'
         write(6,*)  '   PARAM(66) = 6   --> F00000'
         call exitt
      endif
 

c     if_byte_swap_test = .false.

      return
      end
c-----------------------------------------------------------------------
      subroutine openfld(iunit,file,SUFFCH,SUFFI,status,iffmat,ierr)
      CHARACTER*(*) FILE,STATUS
      CHARACTER*30 NEWFILE,LFILE
      CHARACTER*80 zfile
      character*1  lfile1(30)
      character*2  s2
      character*3  s3
      equivalence (lfile1,lfile)
      character*4 suffch
      integer     suffi
      logical iffmat
C
C     NULL out local filename character strings
      call blank(lfile  ,30)
      call blank(newfile,30)
C
      LFILE = FILE
      IF (STATUS.EQ.'NEW' .OR. STATUS.EQ.'new') THEN
         NEWFILE=FILE
         do 10 i=30,1,-1
            if(ichar(newfile(I:I)).ne.0.and.newfile(I:I).ne.' ') then
               newfile(I+1:I+1) = '~'
               GO TO 20
            endif
   10    continue
   20    continue
         print*,'renaming ',file,' ',newfile
         MVERR=RENAME(FILE,NEWFILE)
      ENDIF
      if (suffi.ne.0) then
c
c        Append suffix to file name
c
c        len1=ltrunc(lfile,30)+1
         len1=indx1(lfile,suffch,4)+4
         if (suffi.lt.100) then
            write(s2,32) suffi
            call chcopy(lfile1(len1),s2,2)
   32       format(i2.2)
         else
            write(s3,33) suffi
            call chcopy(lfile1(len1),s3,3)
   33       format(i3.3)
         endif
      endif
c
      len=ltrunc(lfile,30)
      call izero(zfile,20)
      call chcopy(zfile,lfile,len)
c
      if (iffmat) then
         write(6,*) 'Trying to open fld file:',zfile,status,iffmat
         write(6,*) 'Trying to open fld file:',lfile,status,iffmat
c        OPEN(UNIT=IUNIT,FILE=zfile,STATUS=STATUS
         OPEN(UNIT=IUNIT,FILE=lfile,STATUS=STATUS
     $       ,FORM='FORMATTED',ERR=1)
         IERR=0
c        IFFMAT=.TRUE.
         write(6,*) 'Successfully opened file: '
     $             ,lfile,' Format:',iffmat
         return
      endif
c
    1 continue
      if (.not.iffmat) then
         write(6,*) 'Trying to open unfmt fld file:',zfile,status
         OPEN(UNIT=IUNIT,FILE=zfile,STATUS=STATUS
     $      ,FORM='UNFORMATTED',ERR=2)
         IERR=0
c        IFFMAT=.FALSE.
         write(6,*) 'Successfully opened file: '
     $             ,lfile,' Format:',iffmat
         return
      endif
c
    2 continue
      IERR=1
c     IFFMAT=.TRUE.
      return
      END
c-----------------------------------------------------------------------
      subroutine reset_xc
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      integer ied(8)
      save    ied
      data    ied / 1,2,4,3 , 5,6,8,7 /
c
      do ie=1,nel
         l=0
         if (if3d) then
            do k=1,nz,nz-1
            do j=1,ny,ny-1
            do i=1,nx,nx-1
               l=l+1
               m=i+nx*(j-1) + nx*ny*(k-1) + nx*ny*nz*(ie-1)
               x(ie,ied(l)) = xp(m)
               y(ie,ied(l)) = yp(m)
               z(ie,ied(l)) = zp(m)
c     write(6,*)ie,ied(l),m,x(ie,ied(l)),y(ie,ied(l)),z(ie,ied(l)),'xyz'
            enddo
            enddo
            enddo
         else
            do j=1,ny,ny-1
            do i=1,nx,nx-1
               l=l+1
               m=i+nx*(j-1) + nx*ny*(ie-1)
c
c              The following is to foil the stupid optimizer
c
               if (j.gt.90) write(6,*) 'm:',m,i,j,nx,ny,ie,l
               x(ie,ied(l)) = xp(m)
               y(ie,ied(l)) = yp(m)
            enddo
            enddo
         endif
      enddo
c
      call gencen
C
      return
      end
c-----------------------------------------------------------------------
      subroutine copy4r(a,b,n)
      real   a(1)
      real*4 b(1)
      do i = 1, n
         a(i) = b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine copy8r(a,b,n)
      real   a(1)
      real*8 b(1)
      do i = 1, n
         a(i) = b(i)
      enddo
      return
      end
C--------------------------------------------------------------------------
