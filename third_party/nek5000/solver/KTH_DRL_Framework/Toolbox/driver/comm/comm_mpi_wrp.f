!> @file comm_mpi_wrp.f
!! @ingroup comm_mpi
!! @brief Set of MPI wrappers for toolbox communication.
!!
!! @author Adam Peplinski
!! @date May 31, 2016
!=======================================================================
!> @brief Global MPI scan for integer array.
!! @details This routine is simillar to @ref igl_running_sum and
!!  @ref i8gl_running_sum from comm_mpi.f
!! @ingroup comm_mpi
!! @param[inout]  out    output array
!! @param[in]      in     input array
!! @param[in]      nl     buffer length
!! @todo Error mark should be exported
      subroutine ivgl_running_sum(out,in,nl)
      implicit none

      include 'mpif.h'

      integer nid, np, nekcomm,nekgroup,nekreal
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal

      ! argument list
      integer nl
      integer out(nl),in(nl)

      ! local varaibles
      integer ierr
!-----------------------------------------------------------------------
      call mpi_scan(in,out,nl,mpi_integer,mpi_sum,nekcomm,ierr)

      return
      end subroutine
!=======================================================================
!> @brief Broadcast integer array from specified process
!! @details This routine is simillar to @ref lbcast and @ref bcast
!!  from comm_mpi.f
!! @ingroup comm_mpi
!! @param[inout]   buf    array to be broadcased
!! @param[in]      nl     buffer length
!! @param[in]      sid    broadcasting process id
!! @todo Error mark should be exported
      subroutine ibcastn(buf,nl,sid)
      implicit none

      include 'mpif.h'

      integer nid, np, nekcomm,nekgroup,nekreal
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal

      ! argument list
      integer nl,sid
      integer buf(nl)

      ! local varaibles
      integer ierr
!-----------------------------------------------------------------------
      call mpi_bcast (buf,nl,mpi_integer,sid,nekcomm,ierr)

      return
      end subroutine
!=======================================================================
