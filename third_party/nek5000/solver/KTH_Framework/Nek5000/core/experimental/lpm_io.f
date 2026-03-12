!-----------------------------------------------------------------------
      subroutine lpm_io_read(filein1,npart)
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'MASS'
      include 'GEOM'
      include 'CTIMER'
      include 'TSTEP'
      include 'PARALLEL'
#     include "LPM"

      real*4  rout_pos(3      *LPM_LPART) 
     >       ,rout_sln(LPM_LRS*LPM_LPART)
     >       ,rout_lrp(LPM_LRP*LPM_LPART)
     >       ,rout_lip(3      *LPM_LPART)

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      real                  tcoef(3,3),dt_cmt,time_cmt
      common /timestepcoef/ tcoef,dt_cmt,time_cmt

      character*5 sprop1
      character*9 rprop1

      character (len = *)  filein1
      character*3 filein
      character*12 vtufile
      character*13 vtufile1
      character*50 dumstr
      character*1 dum_read

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      logical partl         
      integer vtu,vtu1,prevs(2,np)
      integer*4 iint
      integer*8 idisp_pos,idisp_sln,idisp_lrp,idisp_lip,disp
      integer*8 stride_len
      real*4 rnptot

      icalld1 = icalld1+1

      LPM_RESTART = .true.

      nnp   = np
      nxx   = LPM_NPART
      npt_total = iglsum(nxx,1)

      jx    = 1
      jy    = 2
      jz    = 3

! --------------------------------------------------
! COPY PARTICLES TO OUTPUT ARRAY
! --------------------------------------------------

      iadd = 0
      if_pos = 3      *isize*npt_total
      if_sln = LPM_LRS*isize*npt_total
      if_lrp = LPM_LRP*isize*npt_total
      if_lip = 3      *isize*npt_total

      if (nid .eq. 0) then

      vtu=867+nid
      open(unit=vtu,file=filein1,access='stream',form="unformatted")

      ivtu_size = -1
      ifound = 0
      do i=1,1000000
      read(vtu) dum_read
      if (dum_read == '_') ifound = ifound + 1
      if (ifound .eq. 2) then
         ivtu_size = i
         exit
      endif
      enddo
      read(vtu) npt_total
      close(vtu)

      npt_total = npt_total/isize/3

      endif

      call bcast(npt_total,isize)

      idisp_pos = ivtu_size + isize*(1)
      idisp_sln = ivtu_size + isize*(3   *npt_total + 2)
      idisp_lrp = ivtu_size + isize*(3   *npt_total + LPM_LRS*npt_total 
     >                               + 3)
      idisp_lip = ivtu_size + isize*(3   *npt_total + LPM_LRS*npt_total 
     >                               + LPM_LRP*npt_total + 4)

      npmax = min(npt_total/LPM_LPART+1,mp)
      stride_len = 0
      if (nid .le. npmax-1 .and. nid. ne. 0) stride_len = nid*LPM_LPART

      npart = LPM_LPART
      if (nid .gt. npmax-1) npart = 0

      ndiff = npt_total - (npmax-1)*LPM_LPART
      if (nid .eq. npmax-1) npart = ndiff

      idisp_pos = idisp_pos + isize*3      *stride_len
      idisp_sln = idisp_sln + isize*LPM_LRS*stride_len
      idisp_lrp = idisp_lrp + isize*LPM_LRP*stride_len
      idisp_lip = idisp_lip + isize*3      *stride_len

      icount_pos = npart*3   
      icount_sln = npart*LPM_LRS
      icount_lrp = npart*LPM_LRP
      icount_lip = npart*3

      iorank = -1

      call byte_open_mpi(filein1,pth,.true.,ierr)


      call byte_set_view(idisp_pos,pth)
      call byte_read_mpi(rout_pos,icount_pos,iorank,pth,ierr)
      call byte_set_view(idisp_sln,pth)
      call byte_read_mpi(rout_sln,icount_sln,iorank,pth,ierr)
      call byte_set_view(idisp_lrp,pth)
      call byte_read_mpi(rout_lrp,icount_lrp,iorank,pth,ierr)
      call byte_set_view(idisp_lip,pth)
      call byte_read_mpi(rout_lip,icount_lip,iorank,pth,ierr)


      call byte_close_mpi(pth,ierr)

      ic_pos = 0
      ic_sln = 0
      ic_lrp = 0
      ic_lip = 0
      do i=1,npart

         ic_pos = ic_pos + 1
         lpm_y(jx,i) = rout_pos(ic_pos)
         ic_pos = ic_pos + 1
         lpm_y(jy,i) = rout_pos(ic_pos)
         if (if3d) then
            ic_pos = ic_pos + 1
            lpm_y(jz,i) = rout_pos(ic_pos)
         else
            ic_pos = ic_pos + 1
            lpm_y(jz,i) = 0.0
         endif

         do j=1,LPM_LRS
            ic_sln = ic_sln + 1
            lpm_y(j,i) = rout_sln(ic_sln)
         enddo

         do j=1,LPM_LRP
            ic_lrp = ic_lrp + 1
            lpm_rprop(j,i) = rout_lrp(ic_lrp)
         enddo

         ic_lip = ic_lip + 1
         lpm_iprop(5,i) = int(rout_lip(ic_lip))
         ic_lip = ic_lip + 1
         lpm_iprop(6,i) = int(rout_lip(ic_lip))
         ic_lip = ic_lip + 1
         lpm_iprop(7,i) = int(rout_lip(ic_lip))

      enddo
         
      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_io_write(filein1,iobig)
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'MASS'
      include 'GEOM'
      include 'CTIMER'
      include 'TSTEP'
      include 'PARALLEL'
#     include "LPM"

      real*4  rout_pos(3      *LPM_LPART) 
     >       ,rout_sln(LPM_LRS*LPM_LPART)
     >       ,rout_lrp(LPM_LRP*LPM_LPART)
     >       ,rout_lip(3      *LPM_LPART)

      real                  tcoef(3,3),dt_cmt,time_cmt
      common /timestepcoef/ tcoef,dt_cmt,time_cmt

      character*5 sprop1
      character*9 rprop1

      character filein1*(*)
      character*3 filein
      character*12 vtufile
      character*13 vtufile1
      character*50 dumstr

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      logical partl         
      integer vtu,vtu1,prevs(2,np)
      integer*4 iint
      integer*8 idisp_pos,idisp_sln,idisp_lrp,idisp_lip
      integer*8 stride_len

      icalld1 = icalld1+1

      nnp   = np
      nxx   = LPM_NPART
      npt_total = iglsum(nxx,1)

      jx    = 1
      jy    = 2
      jz    = 3

      if_sz = len(filein1)
      if (if_sz .lt. 3) then
         filein = 'par'
      else 
         write(filein,'(A3)') filein1
      endif

! --------------------------------------------------
! COPY PARTICLES TO OUTPUT ARRAY
! --------------------------------------------------

      iadd = 0
      if_pos = 3      *isize*npt_total
      if_sln = LPM_LRS*isize*npt_total
      if_lrp = LPM_LRP*isize*npt_total
      if_lip = 3      *isize*npt_total

      ic_pos = iadd
      ic_sln = iadd
      ic_lrp = iadd
      ic_lip = iadd
      do i=1,nxx

         ic_pos = ic_pos + 1
         rout_pos(ic_pos) = lpm_y(jx,i)
         ic_pos = ic_pos + 1
         rout_pos(ic_pos) = lpm_y(jy,i)
         if (if3d) then
            ic_pos = ic_pos + 1
            rout_pos(ic_pos) = lpm_y(jz,i)
         else
            ic_pos = ic_pos + 1
            rout_pos(ic_pos) = 0.0
         endif

         do j=1,LPM_LRS
            ic_sln = ic_sln + 1
            rout_sln(ic_sln) = lpm_y(j,i)
         enddo

         do j=1,LPM_LRP
            ic_lrp = ic_lrp + 1
            rout_lrp(ic_lrp) = lpm_rprop(j,i)
         enddo

         ic_lip = ic_lip + 1
         rout_lip(ic_lip) = lpm_iprop(5,i)
         ic_lip = ic_lip + 1
         rout_lip(ic_lip) = lpm_iprop(6,i)
         ic_lip = ic_lip + 1
         rout_lip(ic_lip) = lpm_iprop(7,i)

      enddo

! --------------------------------------------------
! FIRST GET HOW MANY PARTICLES WERE BEFORE THIS RANK
! --------------------------------------------------
      do i=1,nnp
         prevs(1,i) = i-1
         prevs(2,i) = nxx
      enddo

      nps   = 1 ! index of new proc for doing stuff
      nglob = 1 ! unique key to sort by
      nkey  = 1 ! number of keys (just 1 here)
      ndum = 2
      call fgslib_crystal_ituple_transfer(i_cr_hndl,prevs,
     >                 ndum,nnp,nnp,nps)
      call fgslib_crystal_ituple_sort(i_cr_hndl,prevs,
     >                 ndum,nnp,nglob,nkey)

      stride_len = 0
      if (nid .ne. 0) then
      do i=1,nid
         stride_len = stride_len + prevs(2,i)
      enddo
      endif

! ----------------------------------------------------
! WRITE EACH INDIVIDUAL COMPONENT OF A BINARY VTU FILE
! ----------------------------------------------------
      write(vtufile,'(A3,I5.5,A4)') filein,icalld1,'.vtu'

      if (nid .eq. 0) then

      vtu=867+nid
      open(unit=vtu,file=vtufile,status='replace')

! ------------
! FRONT MATTER
! ------------
      write(vtu,'(A)',advance='no') '<VTKFile '
      write(vtu,'(A)',advance='no') 'type="UnstructuredGrid" '
      write(vtu,'(A)',advance='no') 'version="1.0" '
      if (iobig .eq. 0) then
         write(vtu,'(A)',advance='yes') 'byte_order="LittleEndian">'
      elseif (iobig .eq. 1) then
         write(vtu,'(A)',advance='yes') 'byte_order="BigEndian">'
      endif

      write(vtu,'(A)',advance='yes') ' <UnstructuredGrid>'

      write(vtu,'(A)',advance='yes') '  <FieldData>' 
      write(vtu,'(A)',advance='no')  '   <DataArray '  ! time
      write(vtu,'(A)',advance='no') 'type="Float32" '
      write(vtu,'(A)',advance='no') 'Name="TIME" '
      write(vtu,'(A)',advance='no') 'NumberOfTuples="1" '
      write(vtu,'(A)',advance='no') 'format="ascii"> '
      write(vtu,'(E14.7)',advance='no') time
      write(vtu,'(A)',advance='yes') ' </DataArray> '
      write(vtu,'(A)',advance='no') '   <DataArray '  ! cycle
      write(vtu,'(A)',advance='no') 'type="Int32" '
      write(vtu,'(A)',advance='no') 'Name="CYCLE" '
      write(vtu,'(A)',advance='no') 'NumberOfTuples="1" '
      write(vtu,'(A)',advance='no') 'format="ascii"> '
      write(vtu,'(I0)',advance='no') istep
      write(vtu,'(A)',advance='yes') ' </DataArray> '

      write(vtu,'(A)',advance='yes') '  </FieldData>'
      write(vtu,'(A)',advance='no') '  <Piece '
      write(vtu,'(A)',advance='no') 'NumberOfPoints="'
      write(vtu,'(I0)',advance='no') npt_total
      write(vtu,'(A)',advance='yes') '" NumberOfCells="0"> '

! -----------
! COORDINATES 
! -----------
      iint = 0
      write(vtu,'(A)',advance='yes') '   <Points>'
      call lpm_io_vtu_data(vtu,"Position",3   ,iint)
      iint = iint + 3   *isize*npt_total + isize
      write(vtu,'(A)',advance='yes') '   </Points>'

! ----
! DATA 
! ----
      write(vtu,'(A)',advance='yes') '   <PointData>'

      call lpm_io_vtu_data(vtu,'lpm-y',LPM_LRS,iint)
      iint = iint + LPM_LRS*isize*npt_total + isize

      call lpm_io_vtu_data(vtu,'lpm-rprop',LPM_LRP,iint)
      iint = iint + LPM_LRP*isize*npt_total + isize

      call lpm_io_vtu_data(vtu,'lpm-iprop',3,iint)
      iint = iint + 3*isize*npt_total + isize

      write(vtu,'(A)',advance='yes') '   </PointData> '

! ----------
! END MATTER
! ----------
      write(vtu,'(A)',advance='yes') '   <Cells> '
      write(vtu,'(A)',advance='no')  '    <DataArray '
      write(vtu,'(A)',advance='no') 'type="Int32" '
      write(vtu,'(A)',advance='no') 'Name="connectivity" '
      write(vtu,'(A)',advance='yes') 'format="ascii"/> '
      write(vtu,'(A)',advance='no') '    <DataArray '
      write(vtu,'(A)',advance='no') 'type="Int32" '
      write(vtu,'(A)',advance='no') 'Name="offsets" '
      write(vtu,'(A)',advance='yes') 'format="ascii"/> '
      write(vtu,'(A)',advance='no') '    <DataArray '
      write(vtu,'(A)',advance='no') 'type="Int32" '
      write(vtu,'(A)',advance='no') 'Name="types" '
      write(vtu,'(A)',advance='yes') 'format="ascii"/> '
      write(vtu,'(A)',advance='yes') '   </Cells> '
      write(vtu,'(A)',advance='yes') '  </Piece> '
      write(vtu,'(A)',advance='yes') ' </UnstructuredGrid> '

! -----------
! APPEND DATA  
! -----------
      write(vtu,'(A)',advance='no') ' <AppendedData encoding="raw">'
      close(vtu)

      open(unit=vtu,file=vtufile,access='stream',form="unformatted"
     >    ,position='append')
      write(vtu) '_'
      close(vtu)

      inquire(file=vtufile,size=ivtu_size)
      endif

      call bcast(ivtu_size, isize)

      ! byte-displacements
      idisp_pos = ivtu_size + isize*(3   *stride_len + 1)
      idisp_sln = ivtu_size + isize*(3   *npt_total + LPM_LRS*stride_len
     >                      + 2)
      idisp_lrp = ivtu_size + isize*(3   *npt_total  + LPM_LRS*npt_total
     >                      + LPM_LRP*stride_len + 3)
      idisp_lip = ivtu_size + isize*(3   *npt_total  + LPM_LRS*npt_total
     >                      + LPM_LRP*npt_total + 3*stride_len + 4 )

      ! how much to write
      icount_pos = 3      *nxx
      icount_sln = LPM_LRS*nxx
      icount_lrp = LPM_LRP*nxx
      icount_lip = 3      *nxx

      iorank = -1

      ! integer write
      if (nid .eq. 0) then
        open(unit=vtu,file=vtufile,access='stream',form="unformatted"
     >      ,position='append')
        write(vtu) if_pos
        close(vtu)
      endif

      rdum = dnekclock_sync()

      ! write
      call byte_open_mpi(vtufile,pth,.false.,ierr)
      call byte_set_view(idisp_pos,pth)
      call byte_write_mpi(rout_pos,icount_pos,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

      rdum = dnekclock_sync()

      ! integer write
      if (nid .eq. 0) then
        open(unit=vtu,file=vtufile,access='stream',form="unformatted"
     >      ,position='append')
        write(vtu) if_sln
        close(vtu)
      endif

      rdum = dnekclock_sync()

      ! write
      call byte_open_mpi(vtufile,pth,.false.,ierr)
      call byte_set_view(idisp_sln,pth)
      call byte_write_mpi(rout_sln,icount_sln,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

      ! integer write
      if (nid .eq. 0) then
        open(unit=vtu,file=vtufile,access='stream',form="unformatted"
     >      ,position='append')
        write(vtu) if_lrp
        close(vtu)
      endif

      rdum = dnekclock_sync()

      ! write
      call byte_open_mpi(vtufile,pth,.false.,ierr)
      call byte_set_view(idisp_lrp,pth)
      call byte_write_mpi(rout_lrp,icount_lrp,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

      ! integer write
      if (nid .eq. 0) then
        open(unit=vtu,file=vtufile,access='stream',form="unformatted"
     >      ,position='append')
        write(vtu) if_lip
        close(vtu)
      endif

      ! write
      call byte_open_mpi(vtufile,pth,.false.,ierr)
      call byte_set_view(idisp_lip,pth)
      call byte_write_mpi(rout_lip,icount_lip,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

      if (nid .eq. 0) then
      vtu=867+nid
      open(unit=vtu,file=vtufile,status='old',position='append')

      write(vtu,'(A)',advance='yes') '</AppendedData>'
      write(vtu,'(A)',advance='yes') '</VTKFile>'

      close(vtu)
      endif

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_io_vtu_data(vtu,dataname,ncomp,idist)

      integer vtu,ncomp
      integer*4 idist
      character (len = *) dataname
      character*50 dumstr

      write(vtu,'(A)',advance='no') '    <DataArray '
      write(vtu,'(A)',advance='no') 'type="Float32" '
      write(vtu,'(A)',advance='no') 'Name="'
      write(vtu,'(A)',advance='no') dataname
      write(vtu,'(A)',advance='no') '" NumberOfComponents="'
      write(vtu,'(I0)',advance='no') ncomp
      write(vtu,'(A)',advance='no') '" format="append" '
      write(vtu,'(A)',advance='no') 'offset="'
      write(vtu,'(I0)',advance='no') idist
      write(vtu,'(A)',advance='yes') '"/>'

      return
      end
!-----------------------------------------------------------------------
      subroutine lpm_io_read_restart(filename)
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'MASS'
      include 'GEOM'
      include 'TSTEP'
      include 'PARALLEL'
#     include "LPM"

      PARAMETER(LPM_LRF=3+LPM_LRP+LPM_LRS)
      COMMON /LPM_RFPTS_C/ LPM_RFPTS
      REAL*4 LPM_RFPTS(LPM_LRF,LPM_LPART)

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      character*12 filename

      logical partl         ! This is a dummy placeholder, used in cr()

      integer vtu,vtu1,prevs(2,mp)
      integer*4 iint
      integer*8 stride_len

      integer*8 disp
      integer*4 nptot
      integer pth, nread
      real*4 rnptot

! ------------------------------
! LOAD TOTAL NUMBER OF PARTICLES
! ------------------------------
      pth = 185

      write(6,*) filename

      disp = 0
      icount = 0
      if (nid .eq. 0) icount = 1
      iorank = -1
      call byte_open_mpi(trim(filename),pth,.true.,ierr)
      call byte_set_view(disp,pth)
      call byte_read_mpi(rnptot,icount,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

      nptot = int(rnptot)
      call bcast(nptot, isize)

! --------------------------------------------------
! FIRST GET HOW MANY PARTICLES WERE BEFORE THIS RANK
! --------------------------------------------------
      npmax = min(nptot/LPM_LPART+1,mp)
      stride_len = 0
      if (nid .le. npmax-1 .and. nid. ne. 0) stride_len = nid*LPM_LPART

      nread = LPM_LPART
      if (nid .gt. npmax-1) nread = 0

      ndiff = nptot - (npmax-1)*LPM_LPART
      if (nid .eq. npmax-1) nread = ndiff

! -------------------------
! Parallel MPI file read in
! -------------------------
      disp   = isize + stride_len*LPM_LRF*isize ! is there real*4 var?
      icount = nread*LPM_LRF
      iorank = -1
      call byte_open_mpi(filename,pth,.true.,ierr)
      call byte_set_view(disp,pth)
      call byte_read_mpi(lpm_rfpts,icount,iorank,pth,ierr)
      call byte_close_mpi(pth,ierr)

! ----------------------------------------
! Assign values read in to rpart and ipart
! ----------------------------------------
      i = lpm_npart
      do ii = 1,nread
         i = lpm_npart + ii

         ic = 1
         lpm_iprop(5,i) = int(lpm_rfpts(ic,ii))
         ic = ic + 1
         lpm_iprop(6,i) = int(lpm_rfpts(ic,ii))
         ic = ic + 1
         lpm_iprop(7,i) = int(lpm_rfpts(ic,ii))

         do j=1,LPM_LRS
            ic = ic + 1
            lpm_y(j,i) = lpm_rfpts(ic,ii)
         enddo
         do j=1,LPM_LRP
            ic = ic + 1
            lpm_rprop(j,i) = lpm_rfpts(ic,ii)
         enddo
      enddo
      lpm_npart = i

      call lpm_comm_findpts
      call lpm_comm_crystal

      return
      end
c----------------------------------------------------------------------
