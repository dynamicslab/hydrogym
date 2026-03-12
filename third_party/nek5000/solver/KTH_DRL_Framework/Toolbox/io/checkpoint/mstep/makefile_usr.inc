$(OBJDIR)/chkpt_mstp.o      :chkpt_mstp.f CHKPTMSTPD CHKPOINTD;      $(F77) -c $(FL2) -I./ $< -o $@
