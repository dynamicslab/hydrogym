      subroutine byte_sync_mpi(mpi_fh)

#ifdef MPIIO
      include 'mpif.h'
      call MPI_file_sync(mpi_fh,ierr)
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_open_mpi(fname,mpi_fh,ierr)

      include 'SIZE'
      include 'RESTART'

#ifdef MPIIO
      include 'mpif.h'

      character*132 fname

      if(nid.eq.pid0 .or. nid.eq.pid0r) then
c        write(*,*) nid, 'call MPI_file_open',fname
        call MPI_file_open(nekcomm_io,fname,
     &                     MPI_MODE_RDWR+MPI_MODE_CREATE,
     &                     MPI_INFO_NULL,mpi_fh,ierr)
        if(ierr.ne.0) then
          write(6,*) 'ABORT: Error in byte_open_mpi ', ierr
          return
        endif
      endif
#else
      write(6,*) 'byte_open_mpi: No MPI-IO support!'
      ierr=1
      return
#endif
      ierr=0
      return
      end
C--------------------------------------------------------------------------
      subroutine byte_read_mpi(buf,icount,iorank,mpi_fh,ierr)

      include 'SIZE'
      include 'RESTART'

#ifdef MPIIO
      include 'mpif.h'

      real*4 buf(1)          ! buffer

      if(nid.eq.pid0 .or. nid.eq.pid0r) then
        iout = 4*icount ! icount is in 4-byte words
        if(iorank.ge.0 .and. nid.ne.iorank) iout = 0
c        write(*,*) 'byte_read_mpi', nid, iout/4
#ifdef MPIIO_NOCOL
        call MPI_file_read(mpi_fh,buf,iout,MPI_BYTE,
     &                     MPI_STATUS_IGNORE,ierr)
#else
        call MPI_file_read_all(mpi_fh,buf,iout,MPI_BYTE,
     &                         MPI_STATUS_IGNORE,ierr)
#endif
        if(ierr.ne.0) then
          write(6,*) 'ABORT: Error in byte_read_mpi ', ierr
          return
        endif
      endif
#else
      write(6,*) 'byte_read_mpi: No MPI-IO support!'
      ierr=1
      return
#endif
     
      ierr=0

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_write_mpi(buf,icount,iorank,mpi_fh,ierr)

      include 'SIZE'
      include 'RESTART'

#ifdef MPIIO
      include 'mpif.h'

      real*4 buf(1)          ! buffer

      if(nid.eq.pid0 .or. nid.eq.pid0r) then
        iout = 4*icount ! icount is in 4-byte words
        if(iorank.ge.0 .and. nid.ne.iorank) iout = 0
c        write(*,*) 'byte_write', nid, iout/4
#ifdef MPIIO_NOCOL
        call MPI_file_write(mpi_fh,buf,iout,MPI_BYTE,
     &                      MPI_STATUS_IGNORE,ierr)
#else
        call MPI_file_write_all(mpi_fh,buf,iout,MPI_BYTE,
     &                          MPI_STATUS_IGNORE,ierr)
#endif
        if(ierr.ne.0) then
          write(6,*) 'ABORT: Error in byte_write_mpi ', ierr
          return
        endif
      endif
#else
      write(6,*) 'byte_write_mpi: No MPI-IO support!'
      ierr=1
      return
#endif
      ierr=0
      return
      end
C--------------------------------------------------------------------------
      subroutine byte_close_mpi(mpi_fh,ierr)

      include 'SIZE'
      include 'RESTART'

#ifdef MPIIO
      include 'mpif.h'
      if(nid.eq.pid0 .or. nid.eq.pid0r) then
        call MPI_file_close(mpi_fh,ierr)
      endif
      if(ierr.ne.0) then
         write(6,*) 'ABORT: Error in byte_close_mpi ', ierr
         return
      endif
#else
      if(nio.eq.0) write(6,*) 'byte_close_mpi: No MPI-IO support!'
      ierr=1
      return
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_set_view(ioff_in,mpi_fh)

      include 'SIZE'
      include 'RESTART'

#ifdef MPIIO
      include 'mpif.h'
      integer*8 ioff_in
    
      if(nid.eq.pid0 .or. nid.eq.pid0r) then
         if(ioff_in.lt.0) then
           write(6,*) 'byte_set_view: offset<0!'
           call exitt
         endif
c         write(*,*) 'dataoffset', nid, ioff_in
         call MPI_file_set_view(mpi_fh,ioff_in,MPI_BYTE,MPI_BYTE,
     &                          'native',MPI_INFO_NULL,ierr)
         if(ierr.ne.0) then
           write(6,*) 'ABORT: Error in byte_set_view ', ierr
           call exitt
         endif
      endif
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine nek_comm_io(nn)

      include 'SIZE'
      include 'RESTART'
      include 'PARALLEL'

#ifdef MPIIO
      include 'mpif.h'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /scrns/  irank_io(0:lp-1)

#ifdef MPIIO_NOCOL
      if(nid.eq.0) then
        j = 0
        if(nid.eq.pid0 .or. nid.eq.pid0r) then
          irank_io(j) = nid
          j = j + 1
        endif
        do ir = 1,np-1
          call csend(ir,idum,4,ir,0)           ! handshake
          call crecv(ir,ibuf,4)
          if(ibuf.gt.0) then 
            irank_io(j) = ibuf
            j = j + 1
          endif 
        enddo
      else
         mtype = nid
         ibuf = -1
         if(nid.eq.pid0) then
           ibuf = nid
         endif
         call crecv(mtype,idum,4)                ! hand-shake
         call csend(mtype,ibuf,4,0,0)            ! u4 :=: u8
      endif

      call bcast(irank_io,isize*nn)

c      write(6,*) 'nid', nid, (irank_io(i),i=0,nn-1)

      call mpi_comm_group (nekcomm,nekgroup,ierr)
      if(ierr.gt.0) call exitt
      call mpi_group_incl (nekgroup,nn,irank_io,nekgroup_io,ierr)
      if(ierr.gt.0) call exitt
      call mpi_comm_create(nekcomm,nekgroup_io,nekcomm_io,ierr)
      if(ierr.gt.0) call exitt
      call mpi_group_free (nekgroup_io,ierr)
      if(ierr.gt.0) call exitt
      call mpi_group_free (nekgroup,ierr)
      if(ierr.gt.0) call exitt
#else
      nekcomm_io = nekcomm
      return    
#endif

#endif

      return
      end
