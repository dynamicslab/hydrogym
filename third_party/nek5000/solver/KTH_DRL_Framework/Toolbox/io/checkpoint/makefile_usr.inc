$(OBJDIR)/chkpoint.o        :chkpoint.f CHKPOINTD;                   $(F77) -c $(FL2) -I./ $< -o $@
