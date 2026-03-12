!> @file io_tools_block.f
!! @ingroup io_tools
!! @brief Block data to initialise common block for I/O routines
!! @details Following Nek5000 standard I keep block data in seaprate file.
!! The minimal unit id has been choosen to not interact with file ids
!! hard-coded in Nek5000.
!! @author Adam Peplinski
!! @date Mar 7, 2016
!=======================================================================
      block data io_common_init
      include 'IOTOOLD'

      data io_iunit_min/200/
      data io_iunit_max/200/
      end
