      subroutine byte_sync_mpi(mpi_fh)

      include 'mpif.h'
#ifndef NOMPIIO
      call MPI_file_sync(mpi_fh,ierr)
#else
      call exitti('MPI_file_sync unsupported!$',0)
#endif
      return
      end
C--------------------------------------------------------------------------
      subroutine byte_open_mpi(fnamei,mpi_fh,ifro,ierr)

      include 'mpif.h'

      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal

      character fnamei*(*)
      logical ifro
 
      character*132 fname
      character*1   fname1(132)
      equivalence  (fname1,fname)

      l = ltrunc(fnamei,len(fnamei))
      if(l+1.gt.len(fname))
     $ call exitti('invalid string length$',l)
 
      call chcopy(fname1     ,fnamei ,l) 
      call chcopy(fname1(l+1),char(0),1)

      imode = MPI_MODE_WRONLY+MPI_MODE_CREATE
      if(ifro) then
        imode = MPI_MODE_RDONLY 
      endif

#ifndef NOMPIIO
      call MPI_file_open(nekcomm,fname,imode,
     &                   MPI_INFO_NULL,mpi_fh,ierr)
#else
      call exitti('MPI_file_open unsupported!$',0)
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_read_mpi(buf,icount,iorank,mpi_fh,ierr)

      include 'mpif.h'

      real*4 buf(1)          ! buffer

      iout = icount ! icount is in 4-byte words
#ifndef NOMPIIO
      call MPI_file_read_all(mpi_fh,buf,iout,MPI_REAL,
     &                       MPI_STATUS_IGNORE,ierr)
#else
      call exitti('MPI_file_read_all unsupported!$',0)
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_write_mpi(buf,icount,iorank,mpi_fh,ierr)

      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal

      real*4 buf(1)          ! buffer

      iout = icount ! icount is in 4-byte words
      if(iorank.ge.0 .and. nid.ne.iorank) iout = 0
#ifndef NOMPIIO
      call MPI_file_write_all(mpi_fh,buf,iout,MPI_REAL,
     &                        MPI_STATUS_IGNORE,ierr)
#else
      call exitti('MPI_file_write_all unsupported!$',0)
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_close_mpi(mpi_fh,ierr)

      include 'mpif.h'

#ifndef NOMPIIO
      call MPI_file_close(mpi_fh,ierr)
#else
      call exitti('MPI_file_close unsupported!$',0)
#endif

      return
      end
C--------------------------------------------------------------------------
      subroutine byte_set_view(ioff_in,mpi_fh)

      include 'mpif.h'
      integer*8 ioff_in
    
      if(ioff_in.lt.0) 
     & call exitti('Invalid index in MPI_file_set_view!$',ioff_in)
#ifndef NOMPIIO
      call MPI_file_set_view(mpi_fh,ioff_in,MPI_BYTE,MPI_BYTE,
     &                       'native',MPI_INFO_NULL,ierr)
#endif

      return
      end
