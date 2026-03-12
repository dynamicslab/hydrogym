      program NEKTON
      
      include 'mpif.h'

cc This is a self-defined main Nek5000 program to adopt the MPI_SPLIT
c---------------------------------------------------------
      INTEGER :: intracomm, local_comm, inter_comm
      INTEGER :: rank, size, color, ierr
      INTEGER :: task, result, MASTER, restart_signal
      character*5 request
      logical if_quit_i

      !======================
      ! Initialization
      !======================
      CALL MPI_INIT(ierr)
      CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
      CALL MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
      if_quit_i = .FALSE.
      !======================
      ! Grouping
      !======================
      ! Step 1: Split communicators
      IF (rank .EQ. 0) THEN
          color = 0  ! Master (handled in Python)
      ELSE
          color = 1  ! Workers (NEK5000 runs here)
      ENDIF

      !======================
      ! Split
      !======================
      ! Split the local comm, being consistent with python
      CALL MPI_COMM_SPLIT(MPI_COMM_WORLD, color, rank, local_comm, ierr)
      
      if (rank.eq.1) then
      print *, "[NEK] COMM_SPLIT DONE! RANK, Color",rank,color
      endif 

      IF (color .EQ. 1) THEN
      MASTER = 0 
      intera_comm = local_comm
      
      !======================
      ! Main NEK5000
      !======================
      CALL nek_init(intera_comm)
      CALL nek_solve
      !=======================
      CALL nek_end()
      ENDIF ! if(color.eq.1)
      !======================
      ! Finalization
      !======================
      CALL MPI_COMM_FREE(intera_comm, ierr)
      CALL MPI_FINALIZE(ierr)

      CALL exitt()
      
      END ! End program
      
