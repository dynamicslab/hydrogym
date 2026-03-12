!> @file mntrlog_block.f
!! @ingroup monitor
!! @brief Block data to initialise common blocks in MNTRLOGD
!! @details Following Nek5000 standard I keep block data in seaprate file.
!! @author Adam Peplinski
!! @date Sep 28, 2017
!=======================================================================
      block data mntr_log_common_init
      include 'MNTRLOGD'

      data mntr_ifinit /.false./
      data mntr_stdl /1/
      data mntr_ifconv /.false./
      data mntr_pid0 /0/
      data mntr_mod_num /0/
      data mntr_mod_mpos /0/
      data mntr_mod_id /mntr_id_max*-1/
      data mntr_mod_name /mntr_id_max*mntr_blname/

      end
