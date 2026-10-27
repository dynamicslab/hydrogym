c*********************************************************************72
      subroutine mpi_scan(data1, data2, n, datatype,
     &  operation, comm, ierror )

      implicit none

      include "mpi_dummy.h"

      integer n

      integer comm
      integer data1(n)
      integer data2(n)
      integer datatype
      integer ierror
      integer operation    ! currently hardwired only for sum only

      ierror = MPI_SUCCESS

      if ( datatype .eq. mpi_double_precision ) then

        call copy  (  data2, data1, n )

      else if ( datatype .eq. mpi_integer ) then

        call icopy (  data2, data1, n )

      else if ( datatype .eq. mpi_integer8 ) then

        call i8copy ( data2, data1, n )

      else if ( datatype .eq. mpi_real ) then

        call rrcopy ( data2, data1, n )

      else

        ierror = MPI_FAILURE

      end if

      return
      end

c*********************************************************************72
      subroutine mpi_abort ( comm, errorcode, ierror )

c*********************************************************************72
c
cc MPI_ABORT shuts down the processes in a given communicator.
c
      implicit none

      integer comm
      integer errorcode
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_ABORT:'
      write ( *, '(a,i12)' ) 
     &  '  Shut down with error code = ', errorcode

      stop
      end
      subroutine mpi_allgather ( data1, nsend, sendtype, data2, 
     &  nrecv, recvtype, comm, ierror )

c*********************************************************************72
c
cc MPI_ALLGATHER gathers data from all the processes in a communicator.
c
      implicit none

      include "mpi_dummy.h"

      integer nsend

      integer comm
      integer data1(nsend)
      integer data2(nsend)
      integer ierror
      integer nrecv
      integer recvtype
      integer sendtype

      ierror = MPI_SUCCESS

      if ( sendtype .eq. mpi_double_precision ) then
        call mpi_copy_double_precision ( data1, data2, nsend, ierror )
      else if ( sendtype .eq. mpi_integer ) then
        call mpi_copy_integer ( data1, data2, nsend, ierror )
      else if ( sendtype .eq. mpi_real ) then
        call mpi_copy_real ( data1, data2, nsend, ierror )
      else
        ierror = MPI_FAILURE
      end if

      return
      end
      subroutine mpi_allgatherv ( data1, nsend, sendtype,
     &  data2, nrecv, ndispls, recvtype, comm, ierror )

c*********************************************************************72
c
cc MPI_ALLGATHERV gathers data from all the processes in a communicator.
c
      implicit none

      include "mpi_dummy.h"

      integer nsend

      integer comm
      integer data1(nsend)
      integer data2(nsend)
      integer ierror
      integer ndispls
      integer nrecv
      integer recvtype
      integer sendtype

      ierror = MPI_SUCCESS

      if ( sendtype .eq. mpi_double_precision ) then
        call mpi_copy_double_precision ( data1, data2, nsend, ierror )
      else if ( sendtype .eq. mpi_integer ) then
        call mpi_copy_integer ( data1, data2, nsend, ierror )
      else if ( sendtype .eq. mpi_real ) then
        call mpi_copy_real ( data1, data2, nsend, ierror )
      else
        ierror = MPI_FAILURE
      end if

      return
      end
      subroutine mpi_allreduce ( data1, data2, n, datatype,
     &  operation, comm, ierror )

c*********************************************************************72
c
cc MPI_ALLREDUCE carries out a reduction operation.
c
      implicit none

      include "mpi_dummy.h"

      integer n

      integer comm
      integer data1(n)
      integer data2(n)
      integer datatype
      integer ierror
      integer operation

      ierror = MPI_SUCCESS

      if ( datatype .eq. mpi_double_precision ) then

        call mpi_reduce_double_precision ( 
     &    data1, data2, n, operation, ierror )

      else if ( datatype .eq. mpi_integer ) then

        call mpi_reduce_integer ( 
     &    data1, data2, n, operation, ierror )

      else if ( datatype .eq. mpi_integer8 ) then

        call mpi_reduce_integer8( 
     &    data1, data2, n, operation, ierror )

      else if ( datatype .eq. mpi_real ) then

        call mpi_reduce_real ( 
     &    data1, data2, n, operation, ierror )

      else

        ierror = MPI_FAILURE

      end if

      return
      end

      subroutine mpi_barrier ( comm, ierror )

c*********************************************************************72
c
cc MPI_BARRIER forces processes within a communicator to wait together.
c
      implicit none

      integer comm
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      return
      end
      subroutine mpi_bcast ( data, n, datatype, node, comm, ierror )

c*********************************************************************72
c
cc MPI_BCAST broadcasts data from one process to all others.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )
      integer node

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_bsend ( data, n, datatype, iproc, itag,
     &  comm, ierror )

c*********************************************************************72
c
cc MPI_BSEND sends data from one process to another, using buffering.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_BSEND - Error!'
      write ( *, '(a)' )  '  Should not send message to self.'

      return
      end
      subroutine mpi_cart_create ( comm, ndims, dims, periods,
     &  reorder, comm_cart, ierror )

c*********************************************************************72
c
cc MPI_CART_CREATE creates a communicator for a Cartesian topology.
c
      implicit none

      integer ndims

      integer comm
      integer comm_cart
      integer dims(*)
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )
      logical periods(*)
      logical reorder

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_cart_get ( comm, ndims, dims, periods,
     &  coords, ierror )

c*********************************************************************72
c
cc MPI_CART_GET returns the "Cartesian coordinates" of the calling process.
c
      implicit none

      integer ndims

      integer comm
      integer coords(*)
      integer dims(*)
      integer i
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )
      logical periods(*)

      ierror = MPI_SUCCESS

      do i = 1, ndims
        coords(i) = 0
      end do

      return
      end
      subroutine mpi_cart_shift ( comm, idir, idisp, isource, 
     &  idest, ierror )

c*********************************************************************72
c
cc MPI_CART_SHIFT finds the destination and source for Cartesian shifts.
c
      implicit none

      integer comm
      integer idest
      integer idir
      integer idisp
      integer ierror
      integer isource
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS
      isource = 0
      idest = 0

      return
      end
      subroutine mpi_comm_dup ( comm, comm_out, ierror )

c*********************************************************************72
c
cc MPI_COMM_DUP duplicates a communicator.
c
      implicit none

      integer comm
      integer comm_out
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS
      comm_out = comm

      return
      end
      subroutine mpi_comm_free ( comm, ierror )

c*********************************************************************72
c
cc MPI_COMM_FREE "frees" a communicator.
c
      implicit none

      integer comm
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_comm_rank ( comm, me, ierror )

c*********************************************************************72
c
cc MPI_COMM_RANK reports the rank of the calling process.
c
      implicit none

      integer comm
      integer ierror
      integer me
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS
      me = 0

      return
      end
      subroutine mpi_comm_size ( comm, nprocs, ierror )

c*********************************************************************72
c
cc MPI_COMM_SIZE reports the number of processes in a communicator.
c
      implicit none

      integer comm
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )
      integer nprocs

      ierror = MPI_SUCCESS
      nprocs = 1

      return
      end
      subroutine mpi_comm_split ( comm, icolor, ikey, comm_new,
     &  ierror )

c*********************************************************************72
c
cc MPI_COMM_SPLIT splits up a communicator based on a key.
c
      implicit none

      integer comm
      integer comm_new
      integer icolor
      integer ierror
      integer ikey
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_copy_double_precision ( data1, data2, n, ierror )

c*********************************************************************72
c
cc MPI_COPY_DOUBLE copies a real*8 vector.
c
      implicit none

      integer n

      real*8 data1(n)
      real*8 data2(n)
      integer i
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      do i = 1, n
        data2(i) = data1(i)
      end do

      return
      end
      subroutine mpi_copy_integer ( data1, data2, n, ierror )

c*********************************************************************72
c
cc MPI_COPY_INTEGER copies an integer vector.
c
      implicit none

      integer n

      integer data1(n)
      integer data2(n)
      integer i
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      do i = 1, n
        data2(i) = data1(i)
      end do

      return
      end
      subroutine mpi_copy_real ( data1, data2, n, ierror )

c*********************************************************************72
c
      implicit none

      integer n

      real data1(n)
      real data2(n)
      integer i
      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      do i = 1, n
        data2(i) = data1(i)
      end do

      return
      end
      subroutine mpi_finalize ( ierror )

c*********************************************************************72
c
cc MPI_FINALIZE shuts down the MPI library.
c
      implicit none

      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_get_count ( istatus, datatype, icount, ierror )

c*********************************************************************72
c
cc MPI_GET_COUNT reports the actual number of items transmitted.
c
      implicit none

      integer datatype
      integer icount
      integer ierror
      integer istatus
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_GET_COUNT - Error!'
      write ( *, '(a)' ) '  Should not query message from self.'

      return
      end
      subroutine mpi_init ( ierror )

c*********************************************************************72
c
cc MPI_INIT initializes the MPI library.
c
      implicit none

      integer ierror
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_SUCCESS

      return
      end
      subroutine mpi_irecv ( data, n, datatype, iproc, itag,
     &  comm, irequest, ierror )

c*********************************************************************72
c
cc MPI_IRECV receives data from another process.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer irequest
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_IRECV - Error!'
      write ( *, '(a)' ) '  Should not recv message from self.'

      return
      end
      subroutine mpi_isend ( data, n, datatype, iproc, itag,
     &  comm, request, ierror )

c*********************************************************************72
c
cc MPI_ISEND sends data from one process to another using nonblocking transmission.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )
      integer request

      request = 0
      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_ISEND - Error!'
      write ( *, '(a)' )  '  Should not send message to self.'

      return
      end
      subroutine mpi_recv ( data, n, datatype, iproc, itag,
     &  comm, istatus, ierror )

c*********************************************************************72
c
cc MPI_RECV receives data from another process within a communicator.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer istatus
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_RECV - Error!'
      write ( *, '(a)' ) '  Should not recv message from self.'

      return
      end
      subroutine mpi_reduce ( data1, data2, n, datatype, operation,
     &  receiver, comm, ierror )

c*********************************************************************72
c
cc MPI_REDUCE carries out a reduction operation.
c
      implicit none

      include "mpi_dummy.h"

      integer n

      integer comm
      integer data1(n)
      integer data2
      integer datatype
      integer ierror
      integer operation
      integer receiver

      ierror = MPI_SUCCESS

      if ( datatype .eq. mpi_double_precision ) then

        call mpi_reduce_double_precision ( 
     &    data1, data2, n, operation, ierror )

      else if ( datatype .eq. mpi_integer ) then

        call mpi_reduce_integer ( 
     &    data1, data2, n, operation, ierror )

      else if ( datatype .eq. mpi_real ) then

        call mpi_reduce_real ( 
     &    data1, data2, n, operation, ierror )

      else

        ierror = MPI_FAILURE

      end if

      return
      end
      subroutine mpi_reduce_double_precision ( 
     &  data1, data2, n, operation, ierror )

c*********************************************************************72
c
cc MPI_REDUCE_DOUBLE_PRECISION carries out a reduction operation on real*8 values.
c
      implicit none

      include "mpi_dummy.h"

      integer n

      real*8 data1(n)
      real*8 data2(n)
      integer i
      integer ierror
      integer operation


      ierror = MPI_SUCCESS

      do i = 1, n
        data2(i) = data1(i)
      end do

      return
      end

      subroutine mpi_reduce_integer8 ( 
     &  data1, data2, n, operation, ierror )

c*********************************************************************72
c
      implicit none

      include "mpi_dummy.h"

      integer n

      integer*8 data1(n)
      integer*8 data2(n)
      integer i
      integer ierror
      integer operation

      ierror = MPI_SUCCESS

      do i = 1, n
         data2(i) = data1(i)
      end do

      ierror = MPI_FAILURE

      return
      end
 
      subroutine mpi_reduce_integer ( 
     &  data1, data2, n, operation, ierror )

c*********************************************************************72
c
      implicit none

      include "mpi_dummy.h"

      integer n

      integer data1(n)
      integer data2(n)
      integer i
      integer ierror
      integer operation

      ierror = MPI_SUCCESS

      do i = 1, n
         data2(i) = data1(i)
      end do

      ierror = MPI_FAILURE

      return
      end

      subroutine mpi_reduce_real ( 
     &  data1, data2, n, operation, ierror )

c*********************************************************************72
c
cc MPI_REDUCE_REAL carries out a reduction operation on reals.
c
c  Discussion:
c
      implicit none

      include "mpi_dummy.h"

      integer n

      real data1(n)
      real data2(n)
      integer i
      integer ierror
      integer operation

      ierror = MPI_SUCCESS

        do i = 1, n
          data2(i) = data1(i)
        end do

      return
      end
      subroutine mpi_reduce_scatter ( data1, data2, n, datatype,
     &  operation, comm, ierror )

c*********************************************************************72
c
cc MPI_REDUCE_SCATTER collects a message of the same length from each process.
c
      implicit none

      include "mpi_dummy.h"

      integer n

      integer comm
      integer data1(n)
      integer data2(n)
      integer datatype
      integer ierror
      integer operation

      ierror = MPI_SUCCESS

      if ( datatype .eq. mpi_double_precision ) then
        call mpi_copy_double_precision ( data1, data2, n, ierror )
      else if ( datatype .eq. mpi_integer ) then
        call mpi_copy_integer ( data1, data2, n, ierror )
      else if ( datatype .eq. mpi_real ) then
        call mpi_copy_real ( data1, data2, n, ierror )
      else
        ierror = MPI_FAILURE
      end if

      return
      end
      subroutine mpi_rsend ( data, n, datatype, iproc, itag,
     &  comm, ierror )

c*********************************************************************72
c
cc MPI_RSEND "ready sends" data from one process to another.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_RSEND - Error!'
      write ( *, '(a)' ) '  Should not send message to self.'

      return
      end
      subroutine mpi_send ( data, n, datatype, iproc, itag,
     &  comm, ierror )

c*********************************************************************72
c
cc MPI_SEND sends data from one process to another.
c
      implicit none

      integer n

      integer comm
      integer data(n)
      integer datatype
      integer ierror
      integer iproc
      integer itag
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_SEND - Error!'
      write ( *, '(a)' )  '  Should not send message to self.'

      return
      end
      subroutine mpi_wait ( irequest, istatus, ierror )

c*********************************************************************72
c
cc MPI_WAIT waits for an I/O request to complete.
c
      implicit none

      integer ierror
      integer irequest
      integer istatus
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_WAIT - Error!'
      write ( *, '(a)' ) '  Should not wait on message from self.'

      return
      end
      subroutine mpi_waitall ( icount, irequest, istatus, ierror )

c*********************************************************************72
c
cc MPI_WAITALL waits until all I/O requests have completed.
c
      implicit none

      integer icount
      integer ierror
      integer irequest
      integer istatus
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_WAITALL - Error!'
      write ( *, '(a)' ) '  Should not wait on message from self.'

      return
      end
      subroutine mpi_waitany ( icount, array_of_requests, index, 
     &  istatus, ierror )

c*********************************************************************72
c
cc MPI_WAITANY waits until one I/O requests has completed.
c
      implicit none

      integer array_of_requests(*)
      integer icount
      integer ierror
      integer index
      integer istatus
      integer MPI_FAILURE
      parameter ( MPI_FAILURE = 1 )
      integer MPI_SUCCESS
      parameter ( MPI_SUCCESS = 0 )

      ierror = MPI_FAILURE

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'MPI_WAITANY - Error!'
      write ( *, '(a)' ) '  Should not wait on message from self.'

      return
      end
      function mpi_wtick ( )

c*********************************************************************72
c
cc MPI_WTICK returns the time between clock ticks.
c
      implicit none
      
      real*8 mpi_wtick
      
      mpi_wtick = 1.0D+00
      
      return
      end
      function mpi_wtime ( )

c*********************************************************************72
c
cc MPI_WTIME returns the elapsed wall clock time.
c
      implicit none

      real*8 mpi_wtime
      real*4 a(2),etime
      a(1)=0.0
      a(2)=0.0
      mpi_wtime = etime(a)

      return
      end

      subroutine mpi_initialized(mpi_is_initialized, ierr)

      mpi_is_initialized = 0 
      ierr = 0

      return
      end

      subroutine mpi_comm_create(icomm,igroup,icommd,ierr)

      icommd = 1

      return
      end

      subroutine mpi_comm_group(icomm,igroup,ierr)

      igroup = 1
      ierr = 0

      return
      end

      subroutine mpi_group_free

      return
      end

      subroutine mpi_attr_get(icomm,ikey,ival,iflag,ierr)
 
      logical iflag

      ival =  9 999 999  ! dummy
 
      return
      end
c-----------------------------------------------------------------------
