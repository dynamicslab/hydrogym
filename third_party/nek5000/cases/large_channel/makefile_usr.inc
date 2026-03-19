INC_SRC := ./inc_src

$(OBJDIR)/frame.o           :$(FRAME_SRC)/frame.f;                   $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/mntrlog_block.o   :$(MONITOR_SRC)/mntrlog_block.f;         $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/mntrlog.o         :$(MONITOR_SRC)/mntrlog.f;               $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/mntrtmr_block.o   :$(MONITOR_SRC)/mntrtmr_block.f;         $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/mntrtmr.o         :$(MONITOR_SRC)/mntrtmr.f;               $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/rprm_block.o      :$(RUNPARAM_SRC)/rprm_block.f;           $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/rprm.o            :$(RUNPARAM_SRC)/rprm.f;                 $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/io_tools_block.o  :$(IO_TOOLS_SRC)/io_tools_block.f;       $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/io_tools.o        :$(IO_TOOLS_SRC)/io_tools.f;             $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/chkpoint.o        :$(CHKPT_SRC)/chkpoint.f;                $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/chkpt_mstp.o      :$(CHKPTMS_SRC)/chkpt_mstp.f;            $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/math_tools.o      :$(MATH_SRC)/math_tools.f;               $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/map2D.o           :${MAP2D_SRC}/map2D.f;                   $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/stat.o            :${STAT_SRC}/stat.f ./STATD;             $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/stat_IO.o         :${STAT_SRC}/stat_IO.f ./STATD;          $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/misc.o            :$ misc.f;                               $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_state.o       :$ drl/drl_state.f;                      $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_reward.o       :$ drl/drl_reward.f;                    $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_IO.o          :$ drl/drl_IO.f;                         $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_action.o          :$ drl/drl_action.f;                 $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_main.o          :$ drl/drl_main.f;                     $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/drl_init.o          :$ drl/drl_init.f;                     $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/tsrs.o            :${TSRS_SRC}/tsrs.f ./TSRSD ;            $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@
$(OBJDIR)/tsrs_IO.o         :${TSRS_SRC}/tsrs_IO.f ./TSRSD;          $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@ 
$(OBJDIR)/pts_redistribute.o:${TSRS_SRC}/pts_redistribute.f ./TSRSD; $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@ 
$(OBJDIR)/bdforce.o         :$ bdforce.f;                     $(F77) -c $(FL2) -I$(INC_SRC) $< -o $@