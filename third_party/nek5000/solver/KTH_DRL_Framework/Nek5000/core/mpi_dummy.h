c
c  Dummy parameters for MPI F77 stubs
c
      integer mpi_comm_world
      parameter ( mpi_comm_world = 0 )
c
c  Return values.
c
      integer mpi_failure
      parameter ( mpi_failure = 1 )
      integer mpi_success
      parameter ( mpi_success = 0 )
c
c  recv message status
c
      integer mpi_status_size
      parameter ( mpi_status_size = 3 )
      integer mpi_source
      parameter ( mpi_source = 1 )
      integer mpi_tag
      parameter ( mpi_tag = 2 )
      integer mpi_count
      parameter ( mpi_count = 3 )
      integer mpi_status_ignore(mpi_status_size)
      integer mpi_info_null
      parameter (mpi_info_null=0)
c
c  recv flags
c
      integer mpi_any_source
      parameter ( mpi_any_source = -1 )
      integer mpi_any_tag
      parameter ( mpi_any_tag = -1 )
c
c  data types and sizes
c
      integer mpi_integer
      parameter ( mpi_integer = 1 )
      integer mpi_integer8
      parameter ( mpi_integer8 = 6 )
      integer mpi_real
      parameter ( mpi_real = 2 )
      integer mpi_double_precision
      parameter ( mpi_double_precision = 3 )
      integer mpi_logical
      parameter ( mpi_logical = 4 )
      integer mpi_character
      parameter ( mpi_character = 5 )
      integer mpi_byte
      parameter (mpi_byte=1)
c
c  allreduce operations
c
      integer mpi_sum
      parameter ( mpi_sum = 1 )
      integer mpi_max
      parameter ( mpi_max = 2 )
      integer mpi_min
      parameter ( mpi_min = 3 )
      integer mpi_product
      parameter ( mpi_product = 4 )
c
c  timer
c
      double precision mpi_wtime
c
c  I/O
c
      integer mpi_mode_create
      parameter (mpi_mode_create=1)
      integer mpi_mode_rdonly
      parameter (mpi_mode_rdonly=2)
      integer mpi_mode_wronly
      parameter (mpi_mode_wronly=4)
