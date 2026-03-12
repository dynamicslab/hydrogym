!> @file rprm_block.f
!! @ingroup runparam
!! @brief Block data to initialise common block for runtime parameter module
!! @details Following Nek5000 standard I keep block data in seaprate file.
!! @author Adam Peplinski
!! @date Sep 28, 2017
!=======================================================================
      block data rprm_common_init
      include 'RPRMD'

      data rprm_ifinit /.false./
      data rprm_pid0 /0/
      data rprm_sec_num /0/
      data rprm_sec_mpos /0/
      data rprm_sec_id /rprm_sec_id_max*-1/
      data rprm_sec_act /rprm_sec_id_max*.false./
      data rprm_sec_name /rprm_sec_id_max*rprm_blname/
      data rprm_par_num /0/
      data rprm_par_mpos /0/
      data rprm_par_id /rprm_par_id_size*-1/
      data rprm_par_name /rprm_par_id_max*rprm_blname/
      data rprm_parv_int /rprm_par_id_max*0/
      data rprm_parv_real /rprm_par_id_max*0.0/
      data rprm_parv_log /rprm_par_id_max*.false./
      data rprm_parv_str /rprm_par_id_max*rprm_blname/

      end
