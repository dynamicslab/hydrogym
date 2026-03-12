$(OBJDIR)/conht_tools.o     :conht_tools.f CONHTD;                   $(F77) -c $(FL2) -I./ $< -o $@
