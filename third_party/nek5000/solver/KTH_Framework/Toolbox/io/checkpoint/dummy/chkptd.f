!> @file chkptd.f
!! @ingroup chkptdummy
!! @brief Dummy routines for checkpointing.
!! @author Adam Peplinski
!! @date Mar 7, 2016
!=======================================================================
!> @brief Dummy replacement for checkpoint registration.
!! @ingroup chkptdummy
      subroutine chkpts_register
      implicit none
!-----------------------------------------------------------------------
      return
      end subroutine
!=======================================================================
!> @brief Dummy replacement for checkpoint initialisation.
!! @ingroup chkptdummy
      subroutine chkpts_init
      implicit none
!-----------------------------------------------------------------------
      return
      end subroutine
!=======================================================================
!> @brief Dummy replacement for check of module initialisation
!! @ingroup chkptdummy
!! @return chkpts_is_initialised
      logical function chkpts_is_initialised()
      implicit none
!-----------------------------------------------------------------------
      chkpts_is_initialised = .true.

      return
      end function
!=======================================================================
!> @brief Dummy replacement for checkpoint reader.
!! @ingroup chkptdummy
      subroutine chkpts_read
      implicit none
!-----------------------------------------------------------------------
      return
      end subroutine
!=======================================================================
!> @brief Dummy replacement for checkpoint writer.
!! @ingroup chkptdummy
      subroutine chkpts_write
      implicit none
!-----------------------------------------------------------------------
      return
      end subroutine
!=======================================================================
