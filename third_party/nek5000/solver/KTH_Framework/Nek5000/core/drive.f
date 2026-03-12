      program NEKTON

      include 'mpif.h'
      integer comm
      comm = MPI_COMM_WORLD

      call nek_init(comm)
      call nek_solve()
      call nek_end()

      end
