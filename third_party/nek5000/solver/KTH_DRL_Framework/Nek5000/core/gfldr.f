#ifndef NOMPIIO

      subroutine gfldr(sourcefld)
c
c     generic field file reader
c     reads sourcefld and interpolates all avaiable fields
c     onto current mesh
c
c     memory requirement: 
c     nelgs*nxs**ldim < np*(4*lelt*lx1**ldim)
c
      include 'SIZE'
      include 'TOTAL'
      include 'RESTART'
      include 'GFLDR'

      character sourcefld*(*)

      common /scrcg/  pm1(lx1*ly1*lz1,lelv)
      common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal

      character*1   hdr(iHeaderSize)

      integer*8 dtmp8

      logical if_byte_swap_test
      real*4 bytetest

      etime_t = dnekclock_sync()
      if(nio.eq.0) write(6,*) 'call gfldr ',trim(sourcefld) 

      ! open source field file
      ierr = 0
      if(nid.eq.0) then
        open (90,file=sourcefld,status='old',err=100)
        close(90)
        goto 101
 100    ierr = 1
 101  endif
      call err_chk(ierr,' Cannot open source fld file!$')
      call byte_open_mpi(sourcefld,fldh_gfldr,.true.,ierr)

      ! read and parse header
      call byte_read_mpi(hdr,iHeaderSize/4,0,fldh_gfldr,ierr)
      call byte_read_mpi(bytetest,1,0,fldh_gfldr,ierr)

      call mfi_parse_hdr(hdr,ierr)
      call err_chk(ierr,' Invalid header!$')
      ifbswp = if_byte_swap_test(bytetest,ierr)
      call err_chk(ierr,' Invalid endian tag!$')

      nelgs   = nelgr
      nxs     = nxr
      nys     = nyr
      nzs     = nzr
      if(nzs.gt.1) then 
        ldims = 3
      else
        ldims = 2
      endif
      if (ifgtim) time = timer

      ! distribute elements across all ranks
      nels = nelgs/np
      do i = 0,mod(nelgs,np)-1
         if(i.eq.nid) nels = nels + 1
      enddo
      nxyzs      = nxs*nys*nzs
      dtmp8      = nels
      ntots_b    = dtmp8*nxyzs*wdsizr
      rankoff_b  = igl_running_sum(nels) - dtmp8
      rankoff_b  = rankoff_b*nxyzs*wdsizr  
      dtmp8      = nelgs
      nSizeFld_b = dtmp8*nxyzs*wdsizr
      noff0_b    = iHeaderSize + iSize + iSize*dtmp8

      ! do some checks
      if(ldims.ne.ldim) 
     $ call exitti('ldim of source does not match target!$',0)
      if(ntots_b/wdsize .gt. ltots) then
        dtmp8 = nelgs
        lelt_req = dtmp8*nxs*nys*nzs / (np*ltots/lelt)
        lelt_req = lelt_req + 1
        if(nio.eq.0) write(6,*)
     $   'ABORT: buffer too small, increase lelt > ', lelt_req
        call exitt
      endif

      ifldpos = 0
      if(ifgetxr) then
        ! read source mesh coordinates
        call gfldr_getxyz(xm1s,ym1s,zm1s)
        ifldpos = ldim
      else
        call exitti('source does not contain a mesh!$',0)
      endif

      if(if_full_pres) then
        call exitti('no support for if_full_pres!$',0)
      endif

      ! initialize interpolation tool using source mesh
      nxf   = 2*nxs
      nyf   = 2*nys
      nzf   = 2*nzs
      nhash = nels*nxs*nys*nzs 
      nmax  = 128

      call fgslib_findpts_setup(inth_gfldr,nekcomm,np,ldim,
     &                          xm1s,ym1s,zm1s,nxs,nys,nzs,
     &                          nels,nxf,nyf,nzf,bb_t,
     &                          nhash,nhash,nmax,tol)

      ! read source fields and interpolate
      if(ifgetur) then
        if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'reading vel'
        ntot = nx1*ny1*nz1*nelv 
        call gfldr_getfld(vx,vy,vz,ntot,ldim,ifldpos+1)
        ifldpos = ifldpos + ldim
      endif
      if(ifgetpr) then
        if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'reading pr'
        ntot = nx1*ny1*nz1*nelv 
        call gfldr_getfld(pm1,dum,dum,ntot,1,ifldpos+1)
        ifldpos = ifldpos + 1
        if (ifaxis) call axis_interp_ic(pm1)
        call map_pm1_to_pr(pm1,1)
      endif
      if(ifgettr .and. ifheat) then
        if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'reading temp'
        ntot = nx1*ny1*nz1*nelfld(2) 
        call gfldr_getfld(t(1,1,1,1,1),dum,dum,ntot,1,ifldpos+1)
        ifldpos = ifldpos + 1
      endif
      do i = 1,ldimt-1
         if(ifgtpsr(i)) then
           if(nid.eq.0 .and. loglevel.gt.2) 
     $       write(6,*) 'reading scalar',i
           ntot = nx1*ny1*nz1*nelfld(i+2) 
           call gfldr_getfld(t(1,1,1,1,i+1),dum,dum,ntot,1,ifldpos+1) 
           ifldpos = ifldpos + 1
         endif
      enddo

      call byte_close_mpi(fldh_gfldr,ierr)
      etime_t = dnekclock_sync() - etime_t
      call fgslib_findpts_free(inth_gfldr)
      if(nio.eq.0) write(6,'(A,1(1g9.2),A)')
     &                   ' done :: gfldr  ', etime_t, ' sec'

      return
      end
c-----------------------------------------------------------------------
      subroutine gfldr_getxyz(xout,yout,zout)

      include 'SIZE'
      include 'GFLDR'
      include 'RESTART'

      real xout(*)
      real yout(*)
      real zout(*)

      integer*8 ioff_b

 
      ioff_b = noff0_b + ldim*rankoff_b
      call byte_set_view(ioff_b,fldh_gfldr)

      nread = ldim*ntots_b/4
      call byte_read_mpi(bufr,nread,-1,fldh_gfldr,ierr)
      if(ifbswp) then
        if(wdsizr.eq.4) call byte_reverse (bufr,nread,ierr)
        if(wdsizr.eq.8) call byte_reverse8(bufr,nread,ierr)
      endif

      call gfldr_buf2vi (xout,1,bufr,ldim,wdsizr,nels,nxyzs)
      call gfldr_buf2vi (yout,2,bufr,ldim,wdsizr,nels,nxyzs)
      if(ldim.eq.3)
     $ call gfldr_buf2vi(zout,3,bufr,ldim,wdsizr,nels,nxyzs)

      return
      end
c-----------------------------------------------------------------------
      subroutine gfldr_getfld(out1,out2,out3,nout,nldim,ifldpos)

      include 'SIZE'
      include 'GEOM'
      include 'GFLDR'
      include 'RESTART'

      real out1(*)
      real out2(*)
      real out3(*)

      integer*8 ioff_b

      logical ifpts

      integer icalld
      save    icalld
      data    icalld /0/


      ifpts = .false.
      if(icalld.eq.0) then
        ifpts = .true. ! find points
        icalld = 1
      endif

      ! read field data from source fld file
      ioff_b = noff0_b + (ifldpos-1)*nSizeFld_b
      ioff_b = ioff_b  + nldim*rankoff_b
      call byte_set_view(ioff_b,fldh_gfldr)
      nread = nldim*ntots_b/4
      call byte_read_mpi(bufr,nread,-1,fldh_gfldr,ierr)
      if(ifbswp) then
        if(wdsizr.eq.4) call byte_reverse (bufr,nread,ierr)
        if(wdsizr.eq.8) call byte_reverse8(bufr,nread,ierr)
      endif

      ! interpolate onto current mesh
      call gfldr_buf2vi  (buffld,1,bufr,nldim,wdsizr,nels,nxyzs)
      call gfldr_intp    (out1,nout,buffld,ifpts)
      if(nldim.eq.1) return

      call gfldr_buf2vi  (buffld,2,bufr,nldim,wdsizr,nels,nxyzs)
      call gfldr_intp    (out2,nout,buffld,.false.)
      if(nldim.eq.2) return

      if(nldim.eq.3) then
        call gfldr_buf2vi(buffld,3,bufr,nldim,wdsizr,nels,nxyzs)
        call gfldr_intp  (out3,nout,buffld,.false.)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine gfldr_buf2vi(vi,index,buf,ldim,wds,nel,nxyz)

      real    vi(*)
      real*4  buf(*)
      integer wds


      do iel = 1,nel
         j = (iel-1)*nxyz
         k = (iel-1)*ldim*nxyz

         if(index.eq.2) k = k+nxyz
         if(index.eq.3) k = k+2*nxyz

         if(wds.eq.4) call copy4r(vi(j+1),buf(k+1)  ,nxyz)
         if(wds.eq.8) call copy  (vi(j+1),buf(2*k+1),nxyz)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine gfldr_intp(fieldout,nout,fieldin,iffpts)

      include 'SIZE'
      include 'RESTART'
      include 'GEOM'
      include 'GFLDR'

      real    fieldout(nout)
      real    fieldin (*)
      logical iffpts

      integer*8 i8glsum,nfail,nfail_sum

      if(iffpts) then ! locate points (iel,iproc,r,s,t)
        nfail = 0
        toldist = 5e-6
        if(wdsizr.eq.8) toldist = 5e-14

        ntot  = lx1*ly1*lz1*nelt
        call fgslib_findpts(inth_gfldr,
     &                      grcode,1,
     &                      gproc,1,
     &                      gelid,1,
     &                      grst,ldim,
     &                      gdist,1,
     &                      xm1,1,
     &                      ym1,1,
     &                      zm1,1,ntot)

        do i=1,ntot
           if(grcode(i).eq.1 .and. sqrt(gdist(i)).gt.toldist)
     &       nfail = nfail + 1
           if(grcode(i).eq.2) nfail = nfail + 1
        enddo

        nfail_sum = i8glsum(nfail,1)
        if(nfail_sum.gt.0) then
          if(nio.eq.0) write(6,*)
     &      ' WARNING: Unable to find all mesh points in source fld ',
     &      nfail_sum
        endif
      endif

      ! evaluate inut field at given points
      npt = nout
      call fgslib_findpts_eval(inth_gfldr,
     &                         fieldout,1,
     &                         grcode,1,
     &                         gproc,1,
     &                         gelid,1,
     &                         grst,ldim,npt,
     &                         fieldin)

      return
      end

#endif
