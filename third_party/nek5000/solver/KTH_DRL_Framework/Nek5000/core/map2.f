c-----------------------------------------------------------------------
      subroutine mapelpr()
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SCRCT'
      include 'SOLN'
      include 'TSTEP'
      include 'CTIMER'
c
      logical ifverbm
c
      if (nio.eq.0) then
         write(6,12) 'nelgt/nelgv/lelt:',nelgt,nelgv,lelt
         write(6,12) 'lx1/lx2/lx3/lxd:',lx1,lx2,lx3,lxd
 12      format(1X,A,4I12,/,/)
         write(6,*)
      endif

      etime0 = dnekclock_sync()
      if(nio.eq.0) write(6,'(A)') ' partioning elements to MPI ranks'

      MFIELD=2
      IF (IFFLOW) MFIELD=1
      IF (IFMVBD) MFIELD=0

c     Set up TEMPORARY value for NFIELD - NFLDT
      NFLDT = 1
      IF (IFHEAT) NFLDT = 2 + NPSCAL
c
c     Distributed memory processor mapping
      IF (NP.GT.NELGT) THEN
         IF(NID.EQ.0) THEN
           WRITE(6,1000) NP,NELGT
 1000      FORMAT(2X,'ABORT: Too many processors (',I8
     $          ,') for to few elements (',I12,').'
     $          ,/,2X,'ABORTING IN MAPELPR.')
         ENDIF
         call exitt
      ENDIF

      call set_proc_map()

      if (nelt.gt.lelt) then
         call exitti('nelt > lelt, increase lelt!$',nelt)
      endif

      DO 1200 IFIELD=MFIELD,NFLDT
         IF (IFTMSH(IFIELD)) THEN
            NELG(IFIELD)      = NELGT
         ELSE
            NELG(IFIELD)      = NELGV
         ENDIF
 1200 CONTINUE

C     Output the processor-element map:
      ifverbm=.false.
      if (loglevel .gt. 2) ifverbm=.true.

      if(ifverbm) then
        idum = 1
        if(nid.eq.0) then
           N8 = min(8,nelt)
           write(6 ,1310) node-1,(lglel(ie),ie=1,n8)
           if (NELT.GT.8) write(6 ,1315) (lglel(ie),ie=9,NELT)
           DO inid=1,NP-1
              mtype = inid
              call csend(mtype,idum,4,inid,0)         ! handshake
              call crecv(mtype,inelt,4)               ! nelt of other cpus
              N8 = min(8,inelt)
           ENDDO
 1310      FORMAT(' RANK',I6,' IEG',8I8)
 1315      FORMAT('     ',6X,'    ',8I8)
        else
           mtype = nid
           call crecv(mtype,idum,4)                ! hand-shake
           call csend(mtype,nelt,4,0,0)            ! nelt
           if (loglevel .gt. 2) then
              N8 = min(8,nelt)
              write(6 ,1310) node-1,(lglel(ie),ie=1,n8)
              if (NELT.GT.8) write(6 ,1315) (lglel(ie),ie=9,NELT)
           endif
        endif
      endif

      dtmp = dnekclock_sync() - etime0
      if(nio.eq.0) then
        write(6,*) ' '
        write(6,'(A,g13.5,A,/)')  ' done :: partioning ',dtmp,' sec'
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine outmati(u,m,n,name6)
      integer u(m,n)
      character*6 name6
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c
c     Print out copies of a global matrix
c
      do mid=0,np-1
        call nekgsync
        if (mid.eq.nid) then
         n20 = min(n,20)
         write(6,1) nid,m,n,name6
   1     format(//,3i6,'  Matrix:',2x,a6,/)
         do i=1,m
            write(6,2) nid,name6,(u(i,j),j=1,n20)
         enddo
   2     format(i3,1x,a6,20i6)
        endif
        call nekgsync
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine get_map

      call get_vert

      return
      end
c-----------------------------------------------------------------------
      subroutine get_vert
c
c     Distribute and assign partitions using the .map file
c
      include 'SIZE'
      include 'TOTAL'

      parameter(mdw=2+2**ldim)
      parameter(ndw=7*lx1*ly1*lz1*lelv/mdw)
      common /scrns/ wk(mdw,ndw)
      integer wk

      common /ivrtx/ vertex ((2**ldim),lelt)
      integer vertex

      integer icalld
      save    icalld
      data    icalld  /0/

      if(icalld.gt.0) return
      icalld = 1

      nv = 2**ldim
      call get_vert_map(vertex,nv,wk,mdw,ndw)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_vert_map(vertex,nlv,wk,mdw,ndw)

      include 'SIZE'
      include 'TOTAL'

      integer vertex(nlv,1)
      integer wk(mdw*ndw)

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      integer ibuf(2)
      integer hrsb

      integer*8 eid8(lelt), vtx8(lelt*2**ldim)
      integer iwork(lelt)
      common /SCRCG/ xyz(ldim*lelt*2**ldim)
      common /ctmp0/ eid8, vtx8, iwork

      integer tt,cnt,nrank,ierr

      integer opt_parrsb(3), opt_parmetis(10)
      logical ifbswap

#if defined(PARRSB) || defined(PARMETIS)
      ! read vertex coordinates
      call read_re2_hdr(ifbswap, .false.)
      nelt = nelgt/np
      do i = 1,mod(nelgt,np)
        if (np-i.eq.nid) nelt = nelt + 1
      enddo
      call byte_open_mpi(re2fle,fh_re2,.true.,ierr)
      call readp_re2_mesh(ifbswap, .false.)
      call byte_close_mpi(fh_re2,ierr)

      call read_con(wk,size(wk),neli,nvi,nelgti,nelgvi)
      if (nvi .ne. nlv)
     $   call exitti('Number of vertices do not match!$',nv)
      if (nelgti .ne. nelgt)
     $   call exitti('nelgt for mesh/con differs!$',0)
      if (nelgvi .ne. nelgv)
     $   call exitti('nelgt for mesh/con differs!$',0)
      if (neli .gt. lelt)
     $   call exitti('neli > lelt!$',neli)

c fluid elements
      j  = 0
      ii = 0
      cnt= 0
      do i = 1,neli
         if (wk(ii+1) .le. nelgv) then
            j = j + 1
            eid8(j) = wk(ii+1)
            call icopy48(vtx8((j-1)*nlv+1),wk(ii+2),nlv)

            do tt=1,nlv
              xyz(cnt+1)=xc(tt,i)
              xyz(cnt+2)=yc(tt,i)
              if(ldim.eq.3) then
                xyz(cnt+3)=zc(tt,i)
                cnt=cnt+3
              else
                cnt=cnt+2
              endif
            enddo
         endif
         ii = ii + (nlv+1)
      enddo
      neliv = j

      nel = neliv
      call fpartMesh(eid8,vtx8,xyz,lelt,nel,nlv,nekcomm,
     $  meshPartitioner,ierr)
      call err_chk(ierr,'partMesh fluid failed!$')

      nelv = nel
      nelt = nelv
      ierr = 0 
      if (nelv .gt. lelv) ierr = 1
      call err_chk(ierr,'nelv > lelv!$')
 
      do i = 1,nelv
         lglel(i) = eid8(i)
      enddo
      call isort(lglel,iwork,nelv)
      do i = 1,nelv
         call icopy84(vertex(1,i),vtx8((iwork(i)-1)*nlv+1),nlv)
      enddo

      cnt=0
c solid elements
      if (nelgt.ne.nelgv) then
         j  = 0
         ii = 0
         do i = 1,neli
            if (wk(ii+1) .gt. nelgv) then
               j = j + 1
               eid8(j) = wk(ii+1)
               call icopy48(vtx8((j-1)*nlv+1),wk(ii+2),nlv)

               do tt=1,nlv
                 xyz(cnt+1)=xc(tt,i)
                 xyz(cnt+2)=yc(tt,i)
                 if(ldim.eq.3) then
                   xyz(cnt+3)=zc(tt,i)
                   cnt=cnt+3
                 else
                   cnt=cnt+2
                 endif
               enddo
            endif
            ii = ii + (nlv+1)
         enddo
         nelit = j

         nel = nelit
         call fpartMesh(eid8,vtx8,xyz,lelt,nel,nlv,nekcomm,
     $                  meshPartitioner,ierr)
         call err_chk(ierr,'partMesh solid failed!$')

         nelt = nelv + nel
         ierr = 0 
         if (nelt .gt. lelt) ierr = 1
         call err_chk(ierr,'nelt > lelt!$')
    
         do i = 1,nel
            lglel(nelv+i) = eid8(i)
         enddo
         call isort(lglel(nelv+1),iwork,nel) ! sort locally by global element id
         do i = 1,nel
            call icopy84(vertex(1,nelv+i),vtx8((iwork(i)-1)*nlv+1),nlv)
         enddo
      endif

#ifdef DPROCMAP
      do i = 1,nelt
         ieg = lglel(i)
         if (ieg.lt.1 .or. ieg.gt.nelgt) 
     $      call exitti('invalid ieg!$',ieg)
         ibuf(1) = i
         ibuf(2) = nid
         call dProcmapPut(ibuf,2,0,ieg)
      enddo
#else
      call izero(gllnid,nelgt)
      do i = 1,nelt
         ieg = lglel(i)
         gllnid(ieg) = nid
      enddo
      npass = 1 + nelgt/lelt
      k=1
      do ipass = 1,npass
         m = nelgt - k + 1
         m = min(m,lelt)
         if (m.gt.0) call igop(gllnid(k),iwork,'+  ',m)
         k = k+m
      enddo
#endif 


#else


#ifdef DPROCMAP
      call exitti('DPROCMAP requires PARRSB or PARMETIS!$',0)
#else
      call read_map(vertex,nlv,wk,mdw,ndw)
#endif


#endif

      if(nid.eq.0) write(6,*) ''
      call icopy48(vtx8,vertex,nelt*nlv)
      call printPartStat(vtx8,nelt,nlv,nekcomm)

      return
      end
c-----------------------------------------------------------------------
      subroutine read_con(wk,nwk,nelr,nv,nelgti,nelgvi)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'

      integer wk(nwk)

      logical ifbswap,if_byte_swap_test
      logical ifco2, ifcon

      character*132 confle
      character*1   confle1(132)
      equivalence  (confle,confle1)

      character*132 hdr
      character*5   version
      real*4        test

      integer*8 offs, offs0

      ierr = 0
      ifco2 = .false.
      ifmpiio = .true.
#ifdef NOMPIIO
      ifmpiio = .false.
#endif

      if (nid.eq.0) then
         lfname = ltrunc(reafle,132) - 4
         call blank (confle,132)
         call chcopy(confle,reafle,lfname)
         call chcopy(confle1(lfname+1),'.con',4)
         inquire(file=confle, exist=ifcon)

         if (.not.ifcon) then
            call chcopy(confle1(lfname+1),'.co2',4)
            inquire(file=confle, exist=ifco2)
         endif

        if(.not.ifcon .and. .not.ifco2) ierr = 1
      endif
      call bcast(confle,sizeof(confle))
      if(nid.eq.0) write(6,'(A,A)') ' reading ', confle
      call err_chk(ierr,' Cannot find con file!$')
      call bcast(ifco2,lsize)
      ierr = 0

      ! read header
      if (nid.eq.0) then
         if (ifco2) then
            call byte_open(confle,ierr)
            if(ierr.ne.0) goto 100

            call blank(hdr,sizeof(hdr))
            call byte_read(hdr,sizeof(hdr)/4,ierr)
            if(ierr.ne.0) goto 100

            read (hdr,*) version,nelgti,nelgvi,nv
c    1       format(a5,2i12,i2)

            call byte_read(test,1,ierr)
            if(ierr.ne.0) goto 100
            ifbswap = if_byte_swap_test(test,ierr)
            if(ierr.ne.0) goto 100
         endif
      endif
      call bcast(nelgti,sizeof(nelgti))
      call bcast(nelgvi,sizeof(nelgvi))
      call bcast(nv,sizeof(nv))
      call bcast(ifbswap,sizeof(ifbswap))

      if (ifco2 .and. ifmpiio) then
         if (nid.eq.0) call byte_close(ierr)
         call byte_open_mpi(confle,ifh,.true.,ierr)
         offs0 = sizeof(hdr) + sizeof(test)

         nelr = nelgti/np
         do i = 1,mod(nelgti,np)
            if (np-i.eq.nid) nelr = nelr + 1
         enddo
         call lim_chk(nelr*(nv+1),nwk,'nelr ','nwk   ','read_con  ')

         nelBr = igl_running_sum(nelr) - nelr
         offs  = offs0 + int(nelBr,8)*(nv+1)*ISIZE

         call byte_set_view(offs,ifh)
         call byte_read_mpi(wk,(nv+1)*nelr,-1,ifh,ierr)
         call err_chk(ierr,' Error while reading con file!$')
         call byte_close_mpi(ifh,ierr)
         if (ifbswap) call byte_reverse(wk,(nv+1)*nelr,ierr)
      else
         call exitti('reader only support co2 for now$',0)
      endif

      return

 100  continue
      call err_chk(ierr,'Error opening or reading con header$')

      return
      end
c-----------------------------------------------------------------------

#ifndef DPROCMAP

c-----------------------------------------------------------------------
      subroutine set_proc_map()
C
C     Compute element to processor distribution according to (weighted) 
C     physical distribution in an attempt to minimize exposed number of
C     element interfaces.
C
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'SCRCT'
      include 'TSTEP'
      common /ctmp0/ iwork(lelt)

      REAL*8 dnekclock,t0

      t0 = dnekclock()
      call get_map

c     compute global to local map (no processor info)
      IEL=0
      CALL IZERO(GLLEL,NELGT)
      DO IEG=1,NELGT
         IF (GLLNID(IEG).EQ.NID) THEN
            IEL = IEL + 1
            GLLEL(IEG)=IEL
            NELT = IEL
            if (ieg.le.nelgv) NELV = IEL
         ENDIF
c        write(6,*) 'map2 ieg:',ieg,nelv,nelt,nelgv,nelgt
      ENDDO

c     dist. global to local map to all processors
      npass = 1 + nelgt/lelt
      k=1
      do ipass = 1,npass
         m = nelgt - k + 1
         m = min(m,lelt)
         if (m.gt.0) call igop(gllel(k),iwork,'+  ',m)
         k = k+m
      enddo

c     compute local to global map
c     (i.e. returns global element number given local index and proc id)
      do ieg=1,nelgt
         mid  =gllnid(ieg)
         ie   =gllel (ieg)
         if (mid.eq.nid) lglel(ie)=ieg
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine read_map(vertex,nlv,wk,mdw,ndw)

      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'

      integer vertex(nlv,1)
      integer wk(mdw,ndw)

      logical ifbswap,if_byte_swap_test

      character*132 mapfle
      character*1   mapfle1(132)
      equivalence  (mapfle,mapfle1)

      character*132 hdr
      character*5   version
      real*4        test

      logical ifma2,ifmap
      integer e,eg,eg0,eg1
      integer itmp20(20)

      ierr = 0
      ifma2 = .false.

      if (nid.eq.0) then
         lfname = ltrunc(reafle,132) - 4
         call blank (mapfle,132)
         call chcopy(mapfle,reafle,lfname)
         call chcopy(mapfle1(lfname+1),'.map',4)
         inquire(file=mapfle, exist=ifmap)

         if (.not.ifmap) then
            call chcopy(mapfle1(lfname+1),'.ma2',4)
            inquire(file=mapfle, exist=ifma2)
         endif

        if(.not.ifmap .and. .not.ifma2) ierr = 1 
      endif
      if(nid.eq.0) write(6,'(A,A)') ' Reading ', mapfle
      call err_chk(ierr,' Cannot find map file!$')
      call bcast(ifma2,lsize)
      ierr = 0

      if (nid.eq.0) then
         if (ifma2) then         
            call byte_open(mapfle,ierr)
            if(ierr.ne.0) goto 100

            call blank(hdr,132)
            call byte_read(hdr,132/4,ierr)
            if(ierr.ne.0) goto 100

            read (hdr,1) version,neli,nnzi
    1       format(a5,2i12)

            call byte_read(test,1,ierr)
            if(ierr.ne.0) goto 100
            ifbswap = if_byte_swap_test(test,ierr)
            if(ierr.ne.0) goto 100
         else
            open(unit=80,file=mapfle,status='old',err=100)
            read(80,*,err=100) neli,nnzi
         endif
      endif
 
      call bcast(neli, ISIZE)

      npass = 1 + (neli/ndw)
      if (npass.gt.np) then
         if (nid.eq.0) write(6,*) npass,np,neli,ndw,'Error get_vert_map'
         call exitt
      endif 

      len = 4*mdw*ndw
      if (nid.gt.0.and.nid.lt.npass) msg_id=irecv(nid,wk,len)
      call nekgsync

      if (nid.eq.0) then
         eg0 = 0
         do ipass=1,npass
            eg1 = min(eg0+ndw,neli)

            if (ifma2) then
               nwds = (eg1 - eg0)*(mdw-1)
               call byte_read(wk,nwds,ierr)
               if (ierr.ne.0) goto 200
               if (ifbswap) call byte_reverse(wk,nwds,ierr)

               m = eg1 - eg0
               do eg=eg1,eg0+1,-1 ! reshuffle array
                  jj = (m-1)*(mdw-1) + 1
                  call icopy(itmp20,wk(jj,1),mdw-1)
                  call icopy(wk(1,m),itmp20 ,mdw-1)
                  m = m - 1
               enddo
            else
               m = 0
               do eg=eg0+1,eg1
                  m = m + 1
                  read(80,*,err=200) (wk(k,m),k=1,mdw-1)
               enddo
            endif
            
            m = 0
            do eg=eg0+1,eg1
               m = m + 1
               gllnid(eg) = wk(1,m)  ! must still be divided
               wk(mdw,m) = eg
            enddo
    
            if (ipass.lt.npass) call csend(ipass,wk,len,ipass,0) !send to ipass
            eg0 = eg1
         enddo

         ntuple = m

         if (ifma2) then
            call byte_close(ierr)
         else
            close(80)
         endif
      elseif (nid.lt.npass) then
         call msgwait(msg_id)
         ntuple = ndw
      else
         ntuple = 0
      endif

      lng = isize*neli
      call bcast(gllnid,lng)
      call assign_gllnid(gllnid,gllel,nelgt,nelgv,np) ! gllel is used as scratch

      nelt=0 !     Count number of elements on this processor
      nelv=0
      do eg=1,neli
         if (gllnid(eg).eq.nid) then
            if (eg.le.nelgv) nelv=nelv+1
            if (eg.le.nelgt) nelt=nelt+1
         endif
      enddo

c      if (np.le.64) write(6,*) nid,nelv,nelt,nelgv,nelgt,' NELV'

c     NOW: crystal route vertex by processor id

      ntuple_sum = iglsum(ntuple,1)
      if (ntuple_sum .ne. nelgt) then
         if (nid.eq.0) write(6,*) 'Error invalid tuple sum!'
         call exitt
      endif 

      do i=1,ntuple
         eg=wk(mdw,i)
         wk(1,i)=gllnid(eg)        ! processor id for element eg
      enddo

      key = 1  ! processor id is in wk(1,:)
      call fgslib_crystal_ituple_transfer(cr_h,wk,mdw,ntuple,ndw,key)

      key = mdw  ! Sort tuple list by eg
      nkey = 1
      call fgslib_crystal_ituple_sort(cr_h,wk,mdw,nelt,key,nkey)

      iflag = 0
      if (ntuple.ne.nelt) then
         write(6,*) nid,ntuple,nelv,nelt,nelgt,' NELT FAIL'
         write(6,*) 'Check that .map file and .rea file agree'
         iflag=1
      else
         do e=1,nelt
            call icopy(vertex(1,e),wk(2,e),nlv)
         enddo
      endif

      iflag = iglmax(iflag,1)
      if (iflag.gt.0) then
         do mid=0,np-1
            call nekgsync
            if (mid.eq.nid)
     $      write(6,*) nid,ntuple,nelv,nelt,nelgt,' NELT FB'
            call nekgsync
         enddo
         call nekgsync
         call exitt
      endif

      return

 100  continue
      call err_chk(ierr,'Error opening or reading map header$')

 200  continue
      call err_chk(ierr,'Error while reading map file$')

      return
      end
c-----------------------------------------------------------------------
      subroutine assign_gllnid(gllnid,iunsort,nelgt,nelgv,np)
c
      integer gllnid(1),iunsort(1),nelgt,np 
      integer e,eg


      log2p = log2(np)
      np2   = 2**log2p
      if (np2.eq.np.and.nelgv.eq.nelgt) then   ! std power of 2 case

         npstar = ivlmax(gllnid,nelgt)+1
         nnpstr = npstar/np
         do eg=1,nelgt
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

         return

      elseif (np2.eq.np) then   ! std power of 2 case, conjugate heat xfer

c        Assign fluid elements
         npstar = max(np,ivlmax(gllnid,nelgv)+1)
         nnpstr = npstar/np
         do eg=1,nelgv
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

c        Assign solid elements
         nelgs  = nelgt-nelgv  ! number of solid elements
         npstar = max(np,ivlmax(gllnid(nelgv+1),nelgs)+1)
         nnpstr = npstar/np
         do eg=nelgv+1,nelgt
            gllnid(eg) = gllnid(eg)/nnpstr
         enddo

         return

      elseif (nelgv.ne.nelgt) then
         call exitti
     $       ('Conjugate heat transfer requires P=power of 2.$',np)
      endif


c  Below is the code for P a non-power of two:

c  Split the sorted gllnid array (read from .map file) 
c  into np contiguous partitions. 

c  To load balance the partitions in case of mod(nelgt,np)>0 
c  add 1 contiguous entry out of the sorted list to NODE_i 
c  where i = np-mod(nelgt,np) ... np


      nel   = nelgt/np       ! number of elements per processor
      nmod  = mod(nelgt,np)  ! bounded between 1 ... np-1
      npp   = np - nmod      ! how many paritions of size nel 
 
      ! sort gllnid  
      call isort(gllnid,iunsort,nelgt)

      ! setup partitions of size nel 
      k   = 0
      do ip = 0,npp-1
         do e = 1,nel  
            k = k + 1 
            gllnid(k) = ip
         enddo
      enddo
      ! setup partitions of size nel+1
      if(nmod.gt.0) then 
        do ip = npp,np-1
           do e = 1,nel+1  
              k = k + 1 
              gllnid(k) = ip
           enddo
        enddo 
      endif

      ! unddo sorting to restore initial ordering by
      ! global element number
      call iswapt_ip(gllnid,iunsort,nelgt)

      return
      end
c-----------------------------------------------------------------------

#else

c-----------------------------------------------------------------------
      subroutine set_proc_map()
C
C     Compute element to processor distribution according to (weighted) 
C     physical distribution in an attempt to minimize exposed number of
C     element interfaces.
C
      include 'SIZE'
      include 'INPUT'

      call dProcmapInit()  
      call get_map() 

      return
      end
c-----------------------------------------------------------------------

#endif
