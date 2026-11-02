c==============================================
c DRL subroutines for capusling everying 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================


c----------------------------------------------
        subroutine DRL_main
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "INPUT"
        include "TSTEP"
        include 'DRL'
        include 'PARALLEL' ! Communications
        include 'mpif.h'
        logical, save :: evolving = .false.
        integer, save :: i_evolv
        integer, save :: drl_step
        integer ndrl,nst,it
        character*5 request 
        integer parent_comm,ierr,my_nid
        integer MASTER,ierror
        logical drl_check_d
c=============================================
c       Function
c=============================================
        
        ! Reset the restart state
        call drl_if_restart(drl_check_d,2) ! ==> False

        !----------------------
        ! SET-UP DRL_COMM (MUST)
        !----------------------
        if (ISTEP.eq.0) then 
        
        drl_step = UPARAM(1)
        if (NID.eq.0) print *, "[NEK] DRL STEP",drl_step

        MASTER=0        ! We always set Python RANK=0
        CALL MPI_INTERCOMM_CREATE(iglobalcomm, 0, MPI_COMM_WORLD, 
     $                          MASTER, 99, DRL_COMM, IERROR)
        if (NID.eq.0) print *, "[NEK] INTERCOMM ESTABLISHED!"
        
        !----------------------
        ! SET-UP DRL Information (MUST)
        !----------------------
        call drl_init
        
        else ! We will let Nek run 1 step without control 
        !----------------------
        ! INIFITY  MPI LOOP 
        !----------------------
        mpi_loop: do
          if (NID.eq.0) then 
            print *, "======== INTO MPI LOOP ============"
          endif 
          ! FORWARD : EVOLV
          if ((evolving).and.(i_evolv.ne.drl_step)) then
  !---------------------------
            i_evolv = i_evolv+1
  !---------------------------
          ! print *, "ISTEP",ISTEP

            if (NID.eq.0) then
              print *, '[NEK] IEVLOV:',i_evolv,'drl_step:',drl_step
            end if
            
            exit mpi_loop
  !---------------------------
          else
            ! If meets the update limit, stop evolving and refresh the counter
            evolving = .false.
            i_evolv = 1
          endif 
  
          !---------------------------
          ! Broadcast the request FROM STB3
          !---------------------------
          if(NID.eq.0) then
          call MPI_RECV(request,5,MPI_CHARACTER,0,22,
     &              DRL_COMM,MPI_STATUS_IGNORE,ierr)
            print *, "============================="
            print *, "[NEK] RECV REQUEST: ",request
            print *, "============================="

            call MPI_BCAST(request,5,MPI_CHARACTER,
     &              0,iglobalcomm,ierr)
          
          else
            call MPI_BCAST(request,5,MPI_CHARACTER,
     &              0,iglobalcomm,ierr)
          endif ! NID.eq.0

!--------------------------------
          !---------------------------
          ! Execute based on Request
          !---------------------------
          select case (request)
            case ('STATE')
                  call drl_state
                  call nekgsync()
            case ('CNTRL')
                  call drl_action
                  call nekgsync()
            case ('TERMN')
                  call stop_simulation
                  call nekgsync()
            case ('RSETS')
                  call drl_if_restart(drl_check_d,1) ! 1==> .TRUE.
                  call nekgsync()
                  if (NID.eq.0) print *, "[NEK] RESTART Time Evolution!"
                  exit mpi_loop
            case ('EVOLV')
                  evolving = .true.
                  exit mpi_loop
            case ('INTAL')
                  call nekgsync()
                  call drl_info_out
          end select
          
          !----------------------------------------------------
        enddo mpi_loop

        call nekgsync()
        if (NID.eq.0) then 
          print *, "========OUT MPI LOOP!========"
        endif
!-----------------------------
        if (evolving) then 
        if (NID.eq.0) then 
          print *, "========INTO EVLOV!========="
        endif
        !! NOTE: In Simson implementation, it requires keeping calling the action subroutines to modify the B.C
        !! HOWEVER, in NEK, as we are using common block to store the actions, this is not required anymore 
        !! See More details in:/scratch/guastoni/PhD/024-DRL_Channel3D/simson/bla/mpi_drl3d.f90
        ! call drl_action
        
        call drl_reward(i_evolv)
        
        ! call moving_smooth_action(i_evolv,drl_step)

        endif 
!------------------------------
        endif ! If ISTEP.eq.0


1999    return 
        end subroutine DRL_main
c----------------------------------------------


c----------------------------------------------
        subroutine stop_simulation 
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "INPUT"
        include "TSTEP"
        include 'PARALLEL'
        include 'DRL'
        include 'mpif.h'
        character*5 termn, request
        integer parent_comm,ierr,my_nid
        parameter(termn="TERMN")
        logical is_exist
c=============================================
c       Function
c=============================================
        ! call mpi_comm_rank(MPI_COMM_WORLD,my_nid,ierr)
        ! call MPI_COMM_GET_PARENT(parent_comm,ierr)
  
        if(NID.eq.0) print *, "--------------STOP-----------------"
        if(NID.eq.0) print *, "[TERMN] !!!!! LAST STEP !!!!!"
        if(NID.eq.0) print *, "[TERMN] CKPT SAVED!"
        if(NID.eq.0) print *, "--------------STOP-----------------"
        
        ! THIS IS THE WAY TO Go! 
        if(NID.eq.0) print *, "[TERMN] DISCONNECT!"        
        call exitt0        
        end subroutine stop_simulation
c----------------------------------------------


c--------------------------------------------
      subroutine drl_chkpt_reset
c: Just Reset the checkpoints file and time steps when resuming the Simulation 
      include 'SIZE'
      include 'TOTAL'
      include 'mpif.h'
      include 'DRL'

      istep = 0
      drl_rwrt = .TRUE.
      if (NID.eq.0) then  
      print *, "============================"
      print *, "[NEK] Reset TIME LOOP and CHKPT!"
      print *, "============================"
      endif 
      call chkpt_main

      return
      end
c----------------------------------------------

c--------------------------------------------
      subroutine drl_if_restart(drl_check_i,status)
c: Just Reset the checkpoints file and time steps when resuming the Simulation 
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      logical, save :: drl_check 
      integer status
      logical drl_check_i

      if (status.eq.0) then 
      goto 1001
      elseif (status.eq.1) then 
      drl_check = .TRUE. 
      elseif (status.eq.2) then 
      drl_check = .FALSE.
      endif 

      
1001  drl_check_i = drl_check
      if (NID.eq.0) print *, "DRL_STATE",drl_check,drl_check_i

      return
      end
c----------------------------------------------