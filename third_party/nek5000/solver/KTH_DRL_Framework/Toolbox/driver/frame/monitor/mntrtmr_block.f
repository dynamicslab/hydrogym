!> @file mntrtmr_block.f
!! @ingroup monitor
!! @brief Block data to initialise common blocks in MNTRTMRD
!! @details Following Nek5000 standard I keep block data in seaprate file.
!! @author Adam Peplinski
!! @date Oct 13, 2017
!=======================================================================
      block data mntr_tmr_common_init
      include 'MNTRLOGD'
      include 'MNTRTMRD'

      data mntr_tmr_num /0/
      data mntr_tmr_mpos /0/
      data mntr_tmr_id /mntr_tmr_id_size*-1/
      data mntr_tmr_sum /mntr_tmr_id_max*.false./
      data mntr_tmr_name /mntr_tmr_id_max*mntr_blname/
      data mntr_tmrv_timer /mntr_tmr_id_size*0.0/

      end
