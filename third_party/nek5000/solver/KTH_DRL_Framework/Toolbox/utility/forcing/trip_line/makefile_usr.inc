$(OBJDIR)/trip.o            :trip.f TRIPD FRAMELP;                   $(F77) -c $(FL2) -I./ $< -o $@
