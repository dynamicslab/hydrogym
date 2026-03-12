!> @file frame.f
!! @ingroup frame
!! @brief Set of routines for framework operation
!! @author Adam Peplinski
!! @date 13 Oct 2017
!=======================================================================
!> @brief Start framework
!! @ingroup frame
!! @note This routine requires \a frame_usr_register and
!!    \a frame_usr_init to be defined in \a setup.usr
      subroutine frame_start
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'FRAMELP'

      ! local variables
      integer log_thr, slen, ierr
      character*18 log_cval
!-----------------------------------------------------------------------
      ! check logging threshold
      if (nid.eq.0) then
         call getenv('FRAMELOGL',log_cval)
         slen = len_trim(log_cval)
         if (slen.gt.0) then
            read(log_cval,'(I2)',iostat=ierr) log_thr
            if (ierr.gt.0) log_thr = lp_inf
         else
            log_thr = lp_inf
         endif
      endif

      call bcast(log_thr,isize)

      ! register backbone modules na runtime parameters
      call mntr_register_mod(log_thr)
      call rprm_register
      call mntr_register_par

      ! regisred user specified moduels
      call frame_usr_register

      ! get runtime parameters from Nek5000 dictionary
      call rprm_dict_get
      call mntr_init
      call rprm_init

      ! initialise user specified modules
      call frame_usr_init

      return
      end subroutine
!=======================================================================
!> @brief Simulataion monitoring
!! @ingroup frame
      subroutine frame_monitor
      implicit none

      include 'SIZE'
      include 'FRAMELP'

!-----------------------------------------------------------------------
      ! monitor simulation wall clock
      call mntr_wclock

      ! place for other monitoring operations

      return
      end subroutine
!=======================================================================
!> @brief Finalise framework
!! @ingroup frame
      subroutine frame_end
      implicit none

      include 'SIZE'
      include 'FRAMELP'

!-----------------------------------------------------------------------
      ! finalilse user specified modules
      call frame_usr_end

      ! close all opened files
      call io_file_close

      ! place for timers summary
      call mntr_tmr_summary_print()

      return
      end subroutine
!=======================================================================
!> @brief Specify master node id
!! @ingroup frame
!! @return frame_get_master
!! @note This routine requires \a frame_usr_end to be defined in
!!    \a setup.usr
      integer function frame_get_master()
      implicit none
!-----------------------------------------------------------------------
      frame_get_master = 0

      return
      end function
!=======================================================================


