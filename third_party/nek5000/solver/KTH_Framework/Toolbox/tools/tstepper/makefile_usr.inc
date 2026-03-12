$(OBJDIR)/tstepper.o        :tstepper.f TSTEPPERD;                   $(F77) -c $(FL2) -I./ $< -o $@
